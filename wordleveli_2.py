"""
TRM: Tiny Recursive Transformer for WikiText-103 (Word-level)
Author: ChatGPT (Fixed: Data Leakage & Vocab Size Crash)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import os
import math
import json
from dataclasses import dataclass
from itertools import chain

# ===============================
# CONFIGURATION
# ===============================
@dataclass
class Config:
    context_size: int = 64
    chunk_size: int = 2
    batch_size: int = 128
    embed_dim: int = 256
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.2
    max_epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    n_refinements: int = 3
    n_recursions: int = 4
    eval_interval: int = 500
    eval_iters: int = 50
    patience: int = 5
    output_dir: str = "./outputs"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer_name: str = "gpt2"
    fp16: bool = True

# ===============================
# DATASET CLASS
# ===============================
class ChunkedDataset(Dataset):
    def __init__(self, token_ids, context_size, chunk_size):
        self.data = token_ids
        self.context_size = context_size
        self.chunk_size = chunk_size

    def __len__(self):
        return len(self.data) - self.context_size - self.chunk_size

    def __getitem__(self, idx):
        context = self.data[idx: idx + self.context_size]
        chunk = self.data[idx + self.context_size: idx + self.context_size + self.chunk_size]
        return torch.tensor(context, dtype=torch.long), torch.tensor(chunk, dtype=torch.long)

# ===============================
# TRANSFORMER BLOCK
# ===============================
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout=0.1):
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
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=attn_mask, need_weights=False)[0]
        x = x + self.mlp(self.ln2(x))
        return x

# ===============================
# TRM MODEL
# ===============================
class TinyRecursiveModel(nn.Module):
    def __init__(self, vocab_size, config: Config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(vocab_size, config.embed_dim)
        self.pos_embedding = nn.Embedding(config.context_size + config.chunk_size, config.embed_dim)
        
        # Learnable "Mask" or "Start" token embedding to initialize the draft
        # This replaces the "cheating" where we used the target embedding
        self.mask_token_emb = nn.Parameter(torch.randn(1, 1, config.embed_dim))

        self.blocks = nn.ModuleList([TransformerBlock(config.embed_dim, config.n_heads, config.dropout) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.output_head = nn.Linear(config.embed_dim, vocab_size, bias=False)
        self.output_head.weight = self.embedding.weight

        self.register_buffer("context_mask",
            torch.triu(torch.ones(config.context_size, config.context_size)*float('-inf'), diagonal=1))
    
    def forward(self, context, chunk=None):
        """
        Args:
            context: [B, context_size]
            chunk: [B, chunk_size] - ONLY used for dimension reference, NOT for input content
        """
        B = context.shape[0]
        
        # 1. Process Context
        ctx_emb = self.embedding(context)
        ctx_pos = self.pos_embedding(torch.arange(self.config.context_size, device=context.device))
        ctx = ctx_emb + ctx_pos
        for block in self.blocks:
            ctx = block(ctx, self.context_mask)
        
        # 2. Initialize Draft (The "Hypothesis")
        # We CANNOT look at the target 'chunk' here, or it's cheating.
        # We start with a learned MASK embedding + Position info.
        
        # Position indices for the chunk part (comes after context)
        chunk_pos_indices = torch.arange(
            self.config.context_size, 
            self.config.context_size + self.config.chunk_size, 
            device=context.device
        )
        chunk_pos = self.pos_embedding(chunk_pos_indices) # [chunk_size, embed_dim]
        
        # Expand learned mask to [B, chunk_size, embed_dim]
        # Start with a neutral "I don't know" state
        draft_emb = self.mask_token_emb.expand(B, self.config.chunk_size, -1)
        
        draft = draft_emb + chunk_pos
        reasoning = torch.zeros_like(draft)

        all_logits = []
        
        # 3. Recursive Refinement
        for r in range(self.config.n_refinements):
            # Stop gradients between major refinement steps to save memory/stability (optional, kept from logic)
            if r < self.config.n_refinements - 1:
                draft = draft.detach()
                reasoning = reasoning.detach()
            
            # Thinking Phase (Recursion)
            for _ in range(self.config.n_recursions):
                combined = torch.cat([ctx, draft + reasoning], dim=1)
                for block in self.blocks:
                    combined = block(combined)
                # Extract the "thought" vector corresponding to the chunk positions
                reasoning = combined[:, -self.config.chunk_size:, :]
            
            # Update Phase
            combined = torch.cat([ctx, draft + reasoning], dim=1)
            for block in self.blocks:
                combined = block(combined)
            
            # The new draft is the output of the transformer at the chunk positions
            draft = combined[:, -self.config.chunk_size:, :]
            
            # Predict words from the current draft
            all_logits.append(self.output_head(self.ln_f(draft)))
            
        return all_logits

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

def estimate_loss(model, dataset, config):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(config.eval_iters):
            ctx, chk = get_batch(dataset, config.batch_size, config.device)
            logits_list = model(ctx) # chunk not needed for input, only for loss
            logits = logits_list[-1]
            loss = F.cross_entropy(logits.reshape(-1, model.output_head.out_features), chk.reshape(-1))
            losses.append(loss.item())
    model.train()
    return np.mean(losses)

def generate_text(model, tokenizer, prompt, config, max_new_tokens=30):
    model.eval()
    print(f"\n--- Generating text based on: '{prompt}' ---")
    tokens = tokenizer.encode(prompt)
    if len(tokens) < config.context_size:
        tokens = [tokenizer.pad_token_id] * (config.context_size - len(tokens)) + tokens
    else:
        tokens = tokens[-config.context_size:]
    
    generated = list(tokens)
    
    with torch.no_grad():
        for _ in range(max_new_tokens // config.chunk_size):
            ctx = torch.tensor([generated[-config.context_size:]], dtype=torch.long, device=config.device)
            
            # Forward pass (we don't pass a chunk, model generates it from scratch)
            logits_list = model(ctx)
            final_logits = logits_list[-1]
            
            # Simple Greedy Sampling
            next_tokens = torch.argmax(final_logits, dim=-1).squeeze(0).tolist()
            generated.extend(next_tokens)
            
            new_text = tokenizer.decode(next_tokens)
            print(new_text, end="", flush=True)
    print("\n--- End Generation ---\n")
    model.train()

# ===============================
# MAIN FUNCTION
# ===============================
def main():
    os.makedirs("./outputs", exist_ok=True)
    config = Config()
    print(f"Device: {config.device}")
    
    print("Loading WikiText-103 dataset (word-level)...")
    ds_train = load_dataset("wikitext", "wikitext-103-v1", split="train")
    ds_val   = load_dataset("wikitext", "wikitext-103-v1", split="validation")
    ds_test  = load_dataset("wikitext", "wikitext-103-v1", split="test")

    # Tokenizer
    # We set model_max_length to avoid warnings, though we handle chunking manually
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, model_max_length=100000)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # --- FIX: Use batched mapping to tokenize ---
    print("Tokenizing datasets...")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=False, padding=False)

    tokenized_train = ds_train.map(tokenize_function, batched=True, remove_columns=["text"], desc="Tokenizing Train")
    tokenized_val   = ds_val.map(tokenize_function, batched=True, remove_columns=["text"], desc="Tokenizing Val")
    tokenized_test  = ds_test.map(tokenize_function, batched=True, remove_columns=["text"], desc="Tokenizing Test")

    print("Flattening datasets...")
    train_tokens = list(chain.from_iterable(tokenized_train['input_ids']))
    val_tokens   = list(chain.from_iterable(tokenized_val['input_ids']))
    test_tokens  = list(chain.from_iterable(tokenized_test['input_ids']))

    train_dataset = ChunkedDataset(train_tokens, config.context_size, config.chunk_size)
    val_dataset   = ChunkedDataset(val_tokens, config.context_size, config.chunk_size)
    test_dataset  = ChunkedDataset(test_tokens, config.context_size, config.chunk_size)

    # --- FIX: Use len(tokenizer) to account for added [PAD] token ---
    vocab_size = len(tokenizer)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Training sequences: {len(train_dataset)}")

    # Initialize model
    model = TinyRecursiveModel(vocab_size, config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # Training Loop
    step = 0
    best_val_loss = float("inf")
    patience_counter = 0
    stop_training = False
    save_path = os.path.join(config.output_dir, "trm_best_model.pt")

    print(f"Starting training for max {config.max_epochs} epochs with patience {config.patience}...")

    for epoch in range(config.max_epochs):
        if stop_training: break
        
        # Use fewer steps per epoch for quick testing if desired, 
        # but here we iterate roughly 1/10th of dataset per log to see progress
        steps_per_epoch = len(train_dataset) // config.batch_size
        
        for _ in range(steps_per_epoch):
            ctx, chk = get_batch(train_dataset, config.batch_size, config.device)
            
            # Forward pass
            logits_list = model(ctx)
            
            # Loss calculation
            loss = sum(F.cross_entropy(l.reshape(-1, vocab_size), chk.reshape(-1)) for l in logits_list)/len(logits_list)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            step += 1
            
            if step % config.eval_interval == 0:
                val_loss = estimate_loss(model, val_dataset, config)
                # Cap ppl at sensible number for display
                ppl = math.exp(min(val_loss, 20)) 
                
                print(f"Epoch {epoch+1} | Step {step} | Val loss: {val_loss:.4f} | Perplexity: {ppl:.2f}")

                if val_loss < best_val_loss:
                    print(f"-> Improvement! (Previous: {best_val_loss:.4f}). Saving model...")
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), save_path)
                else:
                    patience_counter += 1
                    print(f"-> No improvement. Patience: {patience_counter}/{config.patience}")
                    
                    if patience_counter >= config.patience:
                        print("EARLY STOPPING TRIGGERED. Stopping training.")
                        stop_training = True
                        break
            
            # Optional break to keep runs short for testing
            if stop_training: break

    # Final Phase
    print("\n=======================================")
    if os.path.exists(save_path):
        print("Training finished. Loading best model for testing...")
        model.load_state_dict(torch.load(save_path))
    else:
        print("Training finished (no improvement saved). Testing current model...")

    test_loss = estimate_loss(model, test_dataset, config)
    test_ppl = math.exp(min(test_loss, 20))
    print(f"Final Test Loss: {test_loss:.4f} | Test Perplexity: {test_ppl:.2f}")
    
    with open(os.path.join(config.output_dir, "trm_wikitext103_results.json"), "w") as f:
        json.dump({"test_loss": test_loss, "test_perplexity": test_ppl, "steps": step}, f, indent=2)
    
    generate_text(model, tokenizer, "The history of science is the study of", config)
    generate_text(model, tokenizer, "During the second world war, the output", config)

if __name__ == "__main__":
    torch.manual_seed(1337)
    main()
