"""
TRM: Chunked Autoregressive Transformer with Refinement (Publication Ready)

This script implements:
- Tiny Recursive Model (TRM) that predicts multiple tokens in parallel
- Chunked context + bidirectional refinement
- Warm start, deep supervision, and detach strategy
- Evaluation on WikiText-2 (train/val/test)
- Logging of perplexity

Author: [Your Name]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import math
import os
import json
from dataclasses import dataclass

# ===============================
# CONFIGURATION
# ===============================
@dataclass
class Config:
    context_size: int = 64
    chunk_size: int = 2
    batch_size: int = 128
    embed_dim: int = 64
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.2
    max_epochs: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    n_refinements: int = 3
    n_recursions: int = 6
    eval_interval: int = 250
    eval_iters: int = 100
    output_dir: str = "./outputs"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# DATASET CLASS
# ===============================
class ChunkedDataset(Dataset):
    """
    Custom dataset that creates sequences of context + target chunk.
    Each item is:
    - context: [context_size] tokens
    - chunk: [chunk_size] tokens to predict
    """
    def __init__(self, data: str, context_size: int, chunk_size: int):
        chars = sorted(list(set(data)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        self.data = torch.tensor([self.stoi[ch] for ch in data], dtype=torch.long)
        self.context_size = context_size
        self.chunk_size = chunk_size
        
    def __len__(self):
        return len(self.data) - self.context_size - self.chunk_size
    
    def __getitem__(self, idx):
        context = self.data[idx:idx + self.context_size]
        chunk = self.data[idx + self.context_size:idx + self.context_size + self.chunk_size]
        return context, chunk

# ===============================
# TRANSFORMER BLOCK
# ===============================
class TransformerBlock(nn.Module):
    """
    Single transformer block: Multihead self-attention + feedforward
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
        # Self-attention
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x),
                          attn_mask=attn_mask, need_weights=False)[0]
        # Feedforward
        x = x + self.mlp(self.ln2(x))
        return x

# ===============================
# TRM MODEL
# ===============================
class TinyRecursiveModel(nn.Module):
    """
    Chunked Autoregressive TRM:
    - Predict multiple tokens simultaneously
    - Recursive refinement
    - Warm start from target
    - Detach between refinements
    - Causal mask on context, bidirectional chunk
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embedding = nn.Embedding(config.context_size + config.chunk_size, config.embed_dim)
        self.blocks = nn.ModuleList([TransformerBlock(config.embed_dim, config.n_heads, config.dropout) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.output_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.output_head.weight = self.embedding.weight
        
        # causal mask for context
        self.register_buffer("context_mask",
            torch.triu(torch.ones(config.context_size, config.context_size)*float('-inf'), diagonal=1))
        
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
        ctx_emb = self.embedding(context)
        ctx_pos = self.pos_embedding(torch.arange(self.config.context_size, device=context.device))
        ctx = ctx_emb + ctx_pos
        
        # encode context with causal mask
        for block in self.blocks:
            ctx = block(ctx, self.context_mask)
        
        # warm start chunk
        chunk_emb = self.embedding(chunk)
        chunk_pos = self.pos_embedding(torch.arange(self.config.context_size, 
                                                    self.config.context_size + self.config.chunk_size,
                                                    device=context.device))
        draft = chunk_emb + chunk_pos
        reasoning = torch.zeros_like(draft)
        
        all_logits = []
        
        for r in range(self.config.n_refinements):
            # detach except for final refinement (deep supervision)
            if r < self.config.n_refinements - 1:
                draft = draft.detach()
                reasoning = reasoning.detach()
            
            # recursive refinement
            for _ in range(self.config.n_recursions):
                combined = torch.cat([ctx, draft + reasoning], dim=1)
                for block in self.blocks:
                    combined = block(combined)
                reasoning = combined[:, -self.config.chunk_size:, :]
            
            # update draft
            combined = torch.cat([ctx, draft + reasoning], dim=1)
            for block in self.blocks:
                combined = block(combined)
            draft = combined[:, -self.config.chunk_size:, :]
            
            # deep supervision: store logits for this refinement
            all_logits.append(self.output_head(self.ln_f(draft)))
        
        return all_logits  # list of [B, chunk_size, vocab_size] per refinement

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
            logits_list = model(ctx, chk)
            # final refinement only
            logits = logits_list[-1]
            loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), chk.reshape(-1))
            losses.append(loss.item())
    model.train()
    return np.mean(losses)

# ===============================
# MAIN FUNCTION
# ===============================
def main():
    os.makedirs("./outputs", exist_ok=True)
    config = Config()
    
    # Load WikiText-2
    from datasets import load_dataset
    ds_train = load_dataset("wikitext", "wikitext-2-v1", split="train")
    ds_val   = load_dataset("wikitext", "wikitext-2-v1", split="validation")
    ds_test  = load_dataset("wikitext", "wikitext-2-v1", split="test")

    # Combine text
    train_text = "".join(ds_train["text"])
    val_text   = "".join(ds_val["text"])
    test_text  = "".join(ds_test["text"])
    
    # datasets
    train_dataset = ChunkedDataset(train_text, config.context_size, config.chunk_size)
    val_dataset   = ChunkedDataset(val_text, config.context_size, config.chunk_size)
    test_dataset  = ChunkedDataset(test_text, config.context_size, config.chunk_size)
    config.vocab_size = train_dataset.vocab_size
    
    print(f"Vocabulary size: {config.vocab_size}")
    
    # initialize model
    model = TinyRecursiveModel(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # training loop
    step = 0
    best_val_loss = float("inf")
    for epoch in range(config.max_epochs):
        for _ in range(len(train_dataset)//config.batch_size):
            ctx, chk = get_batch(train_dataset, config.batch_size, config.device)
            logits_list = model(ctx, chk)
            
            # loss: sum over all refinements (deep supervision)
            loss = 0
            for l in logits_list:
                loss += F.cross_entropy(l.reshape(-1, config.vocab_size), chk.reshape(-1))
            loss /= len(logits_list)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            
            step += 1
            if step % config.eval_interval == 0:
                val_loss = estimate_loss(model, val_dataset, config)
                print(f"Step {step} | Val loss: {val_loss:.4f} | Perplexity: {math.exp(min(val_loss,10)):.2f}")
                best_val_loss = min(best_val_loss, val_loss)
    
    # final evaluation
    test_loss = estimate_loss(model, test_dataset, config)
    test_ppl = math.exp(min(test_loss, 10))
    print(f"Final Test Loss: {test_loss:.4f} | Test Perplexity: {test_ppl:.2f}")

    # save results
    result_path = os.path.join(config.output_dir, f"trm_results.json")
    with open(result_path, "w") as f:
        json.dump({"test_loss": test_loss, "test_perplexity": test_ppl}, f, indent=2)
    print(f"Results saved to {result_path}")

if __name__ == "__main__":
    torch.manual_seed(1337)
    main()

