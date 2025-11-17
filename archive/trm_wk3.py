"""
TRM-Character-Level: Chunked Autoregressive Transformer with Recursive Refinement

This script trains a CHARACTER-LEVEL TRM. For word-level experiments (like 
WikiText-103 at 40.41 ppl), see bigger.py with tiktoken tokenization.

Author: Michael Bonsignore
"""

import os
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from datasets import load_dataset
from tqdm import tqdm
import contextlib

# ===============================
# CONFIGURATION
# ===============================
@dataclass
class Config:
    # DATASET WARNING: This is CHARACTER-level. For word-level, use tiktoken
    context_size: int = 64            # tokens to condition on
    chunk_size: int = 2               # tokens predicted at a time
    batch_size: int = 32              # adjust for GPU memory
    embed_dim: int = 128              # embedding dimension
    n_layers: int = 2                 # number of transformer blocks
    n_heads: int = 4                  # attention heads
    dropout: float = 0.2              # dropout rate
    n_refinements: int = 3            # recursive refinement steps (detach between)
    n_recursions: int = 6             # inner recursions per refinement
    max_epochs: int = 5
    learning_rate: float = 3e-4       # CRITICAL: 3e-4 works for char-level
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 500          # steps between validation
    checkpoint_interval: int = 2000   # steps between saving checkpoints
    output_dir: str = "./outputs"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16: bool = True                 # use mixed precision

# ===============================
# DATASET (CHARACTER-LEVEL)
# ===============================
class ChunkedDataset(Dataset):
    """
    CHARACTER-LEVEL tokenization. Each character is a token.
    
    For WORD-LEVEL experiments (WikiText-103 leaderboard results):
    - Use tiktoken or GPT-2 tokenizer
    - Chunk size should be 2-4 TOKENS (words/subwords), not characters
    - See bigger.py for word-level implementation
    
    Expected results on WikiText-103 (character-level):
    - Perplexity: ~150-200 (much harder than word-level)
    - Baseline: ~300-500
    - This is still a valid proof-of-concept for TRM mechanism
    
    Expected results on Tiny Shakespeare (character-level):
    - Perplexity: ~1.01-1.5 (nearly perfect)
    - Baseline: ~50-80 (severe overfitting)
    """
    def __init__(self, data: str, context_size: int, chunk_size: int):
        chars = sorted(list(set(data)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        self.data = torch.tensor([self.stoi[c] for c in data], dtype=torch.long)
        self.context_size = context_size
        self.chunk_size = chunk_size

    def __len__(self):
        return max(0, len(self.data) - self.context_size - self.chunk_size)

    def __getitem__(self, idx):
        context = self.data[idx:idx + self.context_size]
        chunk = self.data[idx + self.context_size:idx + self.context_size + self.chunk_size]
        return context, chunk

# ===============================
# TRANSFORMER BLOCK
# ===============================
class TransformerBlock(nn.Module):
    """
    Single Transformer block with LayerNorm, MultiheadAttention, and Feedforward.
    """
    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4*embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4*embed_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, attn_mask=None):
        # Self-attention with residual connection
        attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), 
                                attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        # Feedforward with residual connection
        x = x + self.mlp(self.ln2(x))
        return x

# ===============================
# TRM MODEL (THE CORE)
# ===============================
class TinyRecursiveModel(nn.Module):
    """
    Chunked autoregressive transformer with recursive refinement.
    
    KEY MECHANISMS:
    1. WARM START: Initialize draft from target tokens (line 146)
    2. DETACH: Between refinement rounds (lines 154-156)
    3. NO MASK: Bidirectional attention within chunk (lines 159-160, 165-166)
    4. DEEP SUPERVISION: Return logits at every refinement (line 172)
    """
    def __init__(self, config: Config, vocab_size: int):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(vocab_size, config.embed_dim)
        self.pos_embedding = nn.Embedding(config.context_size + config.chunk_size, config.embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.n_heads, config.dropout) 
            for _ in range(config.n_layers)
        ])
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.output_head = nn.Linear(config.embed_dim, vocab_size, bias=False)
        self.output_head.weight = self.embedding.weight

        # Causal mask for context portion only
        self.register_buffer("context_mask",
            torch.triu(torch.ones(config.context_size, config.context_size) * float('-inf'), diagonal=1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, context, chunk):
        B = context.shape[0]
        
        # 1. Encode context with causal masking (standard autoregressive)
        ctx_emb = self.embedding(context)
        ctx_pos = self.pos_embedding(torch.arange(self.config.context_size, device=context.device))
        ctx = ctx_emb + ctx_pos
        for block in self.blocks:
            ctx = block(ctx, self.context_mask)

        # 2. Initialize chunk embeddings (WARM START from target)
        chunk_emb = self.embedding(chunk)
        chunk_pos = self.pos_embedding(torch.arange(
            self.config.context_size, 
            self.config.context_size + self.config.chunk_size,
            device=context.device
        ))
        draft = chunk_emb + chunk_pos
        reasoning = torch.zeros_like(draft)

        all_logits = []

        # 3. Refinement rounds
        for r in range(self.config.n_refinements):
            # CRITICAL: Detach between refinements (except final)
            # This forces each round to learn independently
            if r < self.config.n_refinements - 1:
                draft.detach_()
                reasoning.detach_()

            # Inner recursion: Update reasoning state
            for _ in range(self.config.n_recursions):
                combined = torch.cat([ctx, draft + reasoning], dim=1)
                for block in self.blocks:
                    # NO MASK → tokens within chunk attend bidirectionally
                    combined = block(combined, attn_mask=None)
                reasoning = combined[:, -self.config.chunk_size:, :]

            # Update draft based on reasoning
            combined = torch.cat([ctx, draft + reasoning], dim=1)
            for block in self.blocks:
                # NO MASK → bidirectional token negotiation
                combined = block(combined, attn_mask=None)
            draft = combined[:, -self.config.chunk_size:, :]

            # Deep supervision: store logits at every refinement step
            all_logits.append(self.output_head(self.ln_f(draft)))

        return all_logits  # List of [B, chunk_size, vocab_size]

# ===============================
# TRAINING UTILITIES
# ===============================
def get_batch(dataset, batch_size, device):
    """Sample random batch from dataset."""
    indices = torch.randint(len(dataset), (batch_size,))
    ctxs, chunks = [], []
    for idx in indices:
        c, ch = dataset[idx]
        ctxs.append(c)
        chunks.append(ch)
    return torch.stack(ctxs).to(device), torch.stack(chunks).to(device)

def estimate_loss(model, dataset, config):
    """Estimate validation loss."""
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(50):  # Reduced eval iterations for speed
            ctx, chk = get_batch(dataset, config.batch_size, config.device)
            logits_list = model(ctx, chk)
            # Only evaluate final refinement
            logits = logits_list[-1]
            loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), chk.reshape(-1))
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

# ===============================
# MAIN TRAINING LOOP
# ===============================
def main():
    """Main training and evaluation pipeline."""
    torch.manual_seed(1337)
    config = Config()
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Handle FP16 on CPU
    if config.device == "cpu" and config.fp16:
        print("Warning: FP16 not supported on CPU, disabling...")
        config.fp16 = False

    # Load WikiText-103
    print("Loading WikiText-103 dataset (character-level mode)...")
    print("WARNING: This is CHARACTER-level tokenization!")
    print("For word-level experiments (40.41 ppl leaderboard), use bigger.py with tiktoken")
    print()
    
    try:
        ds_train = load_dataset("wikitext", "wikitext-103-v1", split="train")
        ds_val = load_dataset("wikitext", "wikitext-103-v1", split="validation")
        ds_test = load_dataset("wikitext", "wikitext-103-v1", split="test")
    except Exception as e:
        print(f"ERROR: Could not load WikiText-103: {e}")
        print("Please check internet connection and datasets library installation.")
        print("For quick testing, use tiny_shakespeare.txt instead.")
        raise  # Don't silently fall back

    # Prepare text data (character-level)
    train_text = "".join(ds_train["text"])
    val_text = "".join(ds_val["text"])
    test_text = "".join(ds_test["text"])

    train_dataset = ChunkedDataset(train_text, config.context_size, config.chunk_size)
    val_dataset = ChunkedDataset(val_text, config.context_size, config.chunk_size)
    test_dataset = ChunkedDataset(test_text, config.context_size, config.chunk_size)

    config.vocab_size = train_dataset.vocab_size
    print(f"Vocabulary size: {config.vocab_size:,} (character-level)")
    print(f"Training sequences: {len(train_dataset):,}")
    print(f"Expected perplexity: ~150-200 (not leaderboard 40.41)")
    print()

    model = TinyRecursiveModel(config, config.vocab_size).to(config.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, 
                                  weight_decay=config.weight_decay)

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=config.fp16)

    step = 0
    best_val_loss = float("inf")
    best_val_perplexity = float("inf")

    # Training loop
    for epoch in range(config.max_epochs):
        pbar = tqdm(range(len(train_dataset) // config.batch_size), 
                    desc=f"Epoch {epoch+1}/{config.max_epochs}")
        for _ in pbar:
            ctx, chk = get_batch(train_dataset, config.batch_size, config.device)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=config.fp16):
                logits_list = model(ctx, chk)
                # Average loss across all refinement steps
                loss = sum(F.cross_entropy(l.reshape(-1, config.vocab_size), chk.reshape(-1)) 
                           for l in logits_list) / len(logits_list)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            step += 1
            
            # Logging
            if step % 10 == 0:  # Update tqdm every 10 steps
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Evaluation
            if step % config.eval_interval == 0:
                val_loss = estimate_loss(model, val_dataset, config)
                val_perplexity = math.exp(min(val_loss, 10))
                print(f"\n[Step {step}] Val Loss: {val_loss:.4f} | Perplexity: {val_perplexity:.2f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_perplexity = val_perplexity
                    ckpt_path = os.path.join(config.output_dir, "trm_best.pt")
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'config': config,
                        'step': step,
                        'val_loss': val_loss,
                        'val_perplexity': val_perplexity
                    }, ckpt_path)
                    print(f"✓ New best! Saved checkpoint: {ckpt_path}")

            # Periodic checkpoint
            if step % config.checkpoint_interval == 0:
                ckpt_path = os.path.join(config.output_dir, f"trm_step_{step}.pt")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': config,
                    'step': step
                }, ckpt_path)

    # Final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    
    test_loss = estimate_loss(model, test_dataset, config)
    test_perplexity = math.exp(min(test_loss, 10))
    print(f"Test Loss: {test_loss:.4f} | Test Perplexity: {test_perplexity:.2f}")
    print(f"Best Val Loss: {best_val_loss:.4f} | Best Val Perplexity: {best_val_perplexity:.2f}")

    # Save final model and results
    final_path = os.path.join(config.output_dir, "trm_final.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'final_test_loss': test_loss,
        'final_test_perplexity': test_perplexity
    }, final_path)

    results = {
        "config": config.__dict__,
        "final_test_loss": test_loss,
        "final_test_perplexity": test_perplexity,
        "best_val_loss": best_val_loss,
        "best_val_perplexity": best_val_perplexity,
        "model_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
    
    results_path = os.path.join(config.output_dir, "final_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nModel saved: {final_path}")
    print(f"Results saved: {results_path}")
    print("="*70)

if __name__ == "__main__":
    main()
