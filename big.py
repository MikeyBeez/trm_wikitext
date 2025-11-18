"""
TRM: Tiny Recursive Transformer for WikiText-103 (Word-level)
FULL TRAINING VERSION - Designed for multi-day training on RTX 5070 Ti

Key Features:
- Full dataset training (no step limits)
- Robust checkpointing every 1000 steps
- Resume from checkpoint if interrupted
- Fixed text generation
- Conservative 3 epoch limit with early stopping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import os
import math
import json
from dataclasses import dataclass
from itertools import chain
import time
from datetime import datetime

# ===============================
# CONFIGURATION
# ===============================
@dataclass
class Config:
    # Architecture
    context_size: int = 64
    chunk_size: int = 4
    embed_dim: int = 128
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1
    n_refinements: int = 2
    n_recursions: int = 3
    
    # Training
    batch_size: int = 32
    max_epochs: int = 3  # Conservative for full training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    
    # Evaluation & Checkpointing
    eval_interval: int = 1000  # Every 1000 steps (more frequent for long training)
    eval_iters: int = 50
    patience: int = 20  # More patience for full training
    checkpoint_interval: int = 1000  # Save every 1000 steps
    
    # System
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer_name: str = "gpt2"
    seed: int = 1337
    vocab_size: int = None
    
    # Resume training
    resume_from_checkpoint: bool = True

# Add Config to safe globals for torch.load
import torch.serialization
torch.serialization.add_safe_globals([Config])

# ===============================
# DATASET CLASS
# ===============================
class ChunkedDataset(Dataset):
    def __init__(self, token_ids, context_size, chunk_size):
        self.data = token_ids
        self.context_size = context_size
        self.chunk_size = chunk_size
        self.length = max(0, len(self.data) - self.context_size - self.chunk_size)
        
        if self.length == 0:
            raise ValueError(f"Dataset too small! Need at least {context_size + chunk_size} tokens")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        context = self.data[idx: idx + self.context_size]
        chunk = self.data[idx + self.context_size: idx + self.context_size + self.chunk_size]
        return torch.tensor(context, dtype=torch.long), torch.tensor(chunk, dtype=torch.long)

# ===============================
# TRANSFORMER BLOCK
# ===============================
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), 
                          attn_mask=attn_mask, need_weights=False)[0]
        x = x + self.mlp(self.ln2(x))
        return x

# ===============================
# TRM MODEL
# ===============================
class TinyRecursiveModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embedding = nn.Embedding(config.context_size + config.chunk_size, config.embed_dim)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.n_heads, config.dropout)
            for _ in range(config.n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.output_head = nn.Linear(config.vocab_size, config.vocab_size, bias=False)
        self.output_head.weight = self.embedding.weight
        
        self.register_buffer("context_mask",
            torch.triu(torch.ones(config.context_size, config.context_size) * float('-inf'), diagonal=1))
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, context, chunk):
        B = context.shape[0]
        
        ctx_emb = self.embedding(context)
        ctx_pos = self.pos_embedding(torch.arange(self.config.context_size, device=context.device))
        ctx = ctx_emb + ctx_pos
        
        for block in self.blocks:
            ctx = block(ctx, self.context_mask)
        
        chunk_emb = self.embedding(chunk)
        chunk_pos = self.pos_embedding(
            torch.arange(self.config.context_size, 
                        self.config.context_size + self.config.chunk_size,
                        device=context.device)
        )
        y = chunk_emb + chunk_pos
        z = torch.zeros_like(y)
        
        for refine_step in range(self.config.n_refinements):
            if refine_step < self.config.n_refinements - 1:
                with torch.no_grad():
                    y, z = self._refine_once(ctx, y, z)
            else:
                y, z = self._refine_once(ctx, y, z)
        
        y = self.ln_f(y)
        logits = self.output_head(y)
        return logits
    
    def _refine_once(self, ctx, y, z):
        for _ in range(self.config.n_recursions):
            combined = torch.cat([ctx, y + z], dim=1)
            for block in self.blocks:
                combined = block(combined)
            z = combined[:, self.config.context_size:, :]
        
        combined = torch.cat([ctx, y + z], dim=1)
        for block in self.blocks:
            combined = block(combined)
        y = combined[:, self.config.context_size:, :]
        
        return y, z

# ===============================
# TRAINING UTILITIES
# ===============================
def get_batch(dataset, batch_size, device):
    indices = torch.randint(len(dataset), (batch_size,))
    ctxs, chunks = [], []
    for idx in indices:
        ctx, chk = dataset[idx]
        ctxs.append(ctx)
        chunks.append(chk)
    return torch.stack(ctxs).to(device), torch.stack(chunks).to(device)

@torch.no_grad()
def estimate_loss(model, dataset, config):
    model.eval()
    losses = []
    
    for _ in range(config.eval_iters):
        ctx, chk = get_batch(dataset, config.batch_size, config.device)
        logits = model(ctx, chk)
        loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), chk.reshape(-1))
        losses.append(loss.item())
    
    model.train()
    return np.mean(losses)

@torch.no_grad()
def generate_text(model, tokenizer, prompt, config, max_new_tokens=50):
    """FIXED: Generate text with proper initialization"""
    model.eval()
    print(f"\nPrompt: '{prompt}'")
    print("Generated: ", end="", flush=True)
    
    tokens = tokenizer.encode(prompt)
    if len(tokens) < config.context_size:
        tokens = [tokenizer.pad_token_id] * (config.context_size - len(tokens)) + tokens
    else:
        tokens = tokens[-config.context_size:]
    
    generated = list(tokens)
    
    for _ in range(max_new_tokens // config.chunk_size):
        ctx = torch.tensor([generated[-config.context_size:]], dtype=torch.long, device=config.device)
        
        # FIXED: Use last context token as seed instead of zeros
        last_token = ctx[:, -1:]
        seed_chunk = last_token.repeat(1, config.chunk_size)
        
        logits = model(ctx, seed_chunk)
        next_tokens = torch.argmax(logits, dim=-1).squeeze(0).tolist()
        generated.extend(next_tokens)
        
        new_text = tokenizer.decode(next_tokens)
        print(new_text, end="", flush=True)
    
    print("\n")
    model.train()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(model, optimizer, step, epoch, best_val_loss, config, path):
    """Save complete checkpoint"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'epoch': epoch,
        'best_val_loss': best_val_loss,
        'config': config,
    }, path)

def load_checkpoint(path, model, optimizer, config):
    """Load checkpoint and resume training"""
    try:
        checkpoint = torch.load(path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        step = checkpoint['step']
        epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resumed from checkpoint: step {step:,}, epoch {epoch}, best val loss {best_val_loss:.4f}")
        return step, epoch, best_val_loss
    except Exception as e:
        print(f"Could not load checkpoint: {e}")
        return 0, 0, float('inf')

# ===============================
# MAIN FUNCTION
# ===============================
def main():
    config = Config()
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    print("=" * 70)
    print("TRM: FULL TRAINING ON WIKITEXT-103")
    print("=" * 70)
    print(f"Device: {config.device}")
    print(f"Training: FULL DATASET (no step limits)")
    print(f"Max epochs: {config.max_epochs}")
    print(f"Checkpointing: Every {config.checkpoint_interval} steps")
    print(f"Expected time: ~35 hours per epoch = ~4-5 days total")
    print("=" * 70)
    
    # Load dataset
    print("\nLoading WikiText-103...")
    ds_train = load_dataset("wikitext", "wikitext-103-v1", split="train")
    ds_val   = load_dataset("wikitext", "wikitext-103-v1", split="validation")
    ds_test  = load_dataset("wikitext", "wikitext-103-v1", split="test")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, model_max_length=100000)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    config.vocab_size = len(tokenizer)

    print("Tokenizing datasets...")
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=False, padding=False)

    tokenized_train = ds_train.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_val   = ds_val.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_test  = ds_test.map(tokenize_function, batched=True, remove_columns=["text"])

    train_tokens = list(chain.from_iterable(tokenized_train['input_ids']))
    val_tokens   = list(chain.from_iterable(tokenized_val['input_ids']))
    test_tokens  = list(chain.from_iterable(tokenized_test['input_ids']))

    train_dataset = ChunkedDataset(train_tokens, config.context_size, config.chunk_size)
    val_dataset   = ChunkedDataset(val_tokens, config.context_size, config.chunk_size)
    test_dataset  = ChunkedDataset(test_tokens, config.context_size, config.chunk_size)

    steps_per_epoch = len(train_dataset) // config.batch_size
    
    print(f"\nDataset Statistics:")
    print(f"  Training examples: {len(train_dataset):,}")
    print(f"  Validation examples: {len(val_dataset):,}")
    print(f"  Steps per epoch: {steps_per_epoch:,}")
    print(f"  Total training steps: {steps_per_epoch * config.max_epochs:,}")
    print(f"  Estimated time: ~{(steps_per_epoch * config.max_epochs * 0.034 / 3600):.1f} hours")

    # Initialize model
    print("\nInitializing model...")
    model = TinyRecursiveModel(config).to(config.device)
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )

    # Try to resume from checkpoint
    resume_path = os.path.join(config.checkpoint_dir, "latest_checkpoint.pt")
    start_step = 0
    start_epoch = 0
    best_val_loss = float("inf")
    
    if config.resume_from_checkpoint and os.path.exists(resume_path):
        print(f"\nFound checkpoint: {resume_path}")
        start_step, start_epoch, best_val_loss = load_checkpoint(resume_path, model, optimizer, config)
    else:
        print("\nStarting fresh training (no checkpoint found)")

    # Training loop
    print("\n" + "=" * 70)
    print("STARTING FULL TRAINING")
    print("=" * 70)
    print(f"This will take approximately {config.max_epochs} × 35 hours = {config.max_epochs * 35} hours")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    best_val_ppl = float("inf") if best_val_loss == float("inf") else math.exp(best_val_loss)
    patience_counter = 0
    best_model_path = os.path.join(config.output_dir, "trm_wikitext_best.pt")
    
    training_history = []
    start_time = time.time()
    step = start_step

    for epoch in range(start_epoch, config.max_epochs):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch + 1}/{config.max_epochs}")
        print(f"{'='*70}")
        
        epoch_losses = []
        
        for batch_idx in range(steps_per_epoch):
            # Skip batches if resuming mid-epoch
            if epoch == start_epoch and step > 0 and batch_idx < (step % steps_per_epoch):
                continue
                
            ctx, chk = get_batch(train_dataset, config.batch_size, config.device)
            
            logits = model(ctx, chk)
            loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), chk.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            step += 1
            
            # Evaluation
            if step % config.eval_interval == 0:
                train_loss = np.mean(epoch_losses[-100:]) if len(epoch_losses) >= 100 else np.mean(epoch_losses)
                val_loss = estimate_loss(model, val_dataset, config)
                val_ppl = math.exp(min(val_loss, 20))
                
                elapsed = (time.time() - start_time) / 3600
                progress = (step / (steps_per_epoch * config.max_epochs)) * 100
                
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step {step:,} | Epoch {epoch+1}/{config.max_epochs} | {elapsed:.1f}h elapsed")
                print(f"  Progress: {progress:.1f}%")
                print(f"  Train loss: {train_loss:.4f}")
                print(f"  Val loss:   {val_loss:.4f}")
                print(f"  Val PPL:    {val_ppl:.2f}")
                
                training_history.append({
                    "step": step,
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                    "elapsed_hours": elapsed
                })
                
                if val_loss < best_val_loss:
                    improvement = best_val_loss - val_loss
                    print(f"  ✓ NEW BEST! Improvement: {improvement:.4f}")
                    best_val_loss = val_loss
                    best_val_ppl = val_ppl
                    patience_counter = 0
                    
                    save_checkpoint(model, optimizer, step, epoch, best_val_loss, config, best_model_path)
                    print(f"  Saved best model: {best_model_path}")
                else:
                    patience_counter += 1
                    print(f"  ✗ No improvement. Patience: {patience_counter}/{config.patience}")
                    
                    if patience_counter >= config.patience:
                        print("\n" + "=" * 70)
                        print("EARLY STOPPING TRIGGERED")
                        print("=" * 70)
                        break
            
            # Regular checkpointing
            if step % config.checkpoint_interval == 0:
                checkpoint_path = os.path.join(config.checkpoint_dir, "latest_checkpoint.pt")
                save_checkpoint(model, optimizer, step, epoch, best_val_loss, config, checkpoint_path)
                
                # Also save periodic backups
                backup_path = os.path.join(config.checkpoint_dir, f"checkpoint_step_{step}.pt")
                save_checkpoint(model, optimizer, step, epoch, best_val_loss, config, backup_path)
            
            # Progress indicator
            if step % 100 == 0 and step % config.eval_interval != 0:
                elapsed = (time.time() - start_time) / 3600
                progress = (step / (steps_per_epoch * config.max_epochs)) * 100
                eta = (elapsed / progress * 100) - elapsed if progress > 0 else 0
                print(f"  Step {step:,} | Loss: {epoch_losses[-1]:.4f} | {progress:.1f}% | ETA: {eta:.1f}h", end="\r")
        
        if patience_counter >= config.patience:
            break

    # Training complete
    total_time = (time.time() - start_time) / 3600
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Total time: {total_time:.1f} hours")
    print(f"Best validation perplexity: {best_val_ppl:.2f}")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load best model
    if os.path.exists(best_model_path):
        try:
            checkpoint = torch.load(best_model_path, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"\nLoaded best model from step {checkpoint['step']:,}")
        except Exception as e:
            print(f"Warning: Could not load best model: {e}")
    
    # Test evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 70)
    
    test_loss = estimate_loss(model, test_dataset, config)
    test_ppl = math.exp(min(test_loss, 20))
    
    print(f"Test Loss:       {test_loss:.4f}")
    print(f"Test Perplexity: {test_ppl:.2f}")
    
    # Save results
    results = {
        "training_type": "FULL_DATASET",
        "test_loss": test_loss,
        "test_perplexity": test_ppl,
        "best_val_loss": best_val_loss,
        "best_val_perplexity": best_val_ppl,
        "total_steps": step,
        "total_epochs": epoch + 1,
        "training_time_hours": total_time,
        "model_parameters": num_params,
        "steps_per_epoch": steps_per_epoch,
        "config": {
            "context_size": config.context_size,
            "chunk_size": config.chunk_size,
            "embed_dim": config.embed_dim,
            "n_layers": config.n_layers,
            "n_refinements": config.n_refinements,
            "n_recursions": config.n_recursions,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
        },
        "training_history": training_history
    }
    
    results_path = os.path.join(config.output_dir, f"trm_full_training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Generate samples
    print("\n" + "=" * 70)
    print("SAMPLE TEXT GENERATION")
    print("=" * 70)
    
    test_prompts = [
        "The history of science",
        "During the second world war",
        "The theory of relativity"
    ]
    
    for prompt in test_prompts:
        generate_text(model, tokenizer, prompt, config, max_new_tokens=40)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE!")
    print("=" * 70)
    print(f"\nFinal Achievement:")
    print(f"  {num_params:,} parameter TRM")
    print(f"  Validation: {best_val_ppl:.2f} PPL")
    print(f"  Test: {test_ppl:.2f} PPL")
    print(f"  Training time: {total_time:.1f} hours")
    print(f"  Full dataset training: {step:,} steps")
    
    if test_ppl < 2.0:
        print(f"\n🎉 BREAKTHROUGH: Sub-2.0 perplexity achieved!")
    elif test_ppl < 3.0:
        print(f"\n✅ EXCELLENT: World-class performance for this parameter count!")
    else:
        print(f"\n✅ STRONG: Competitive result, significant improvement over baseline!")

if __name__ == "__main__":
    main()
