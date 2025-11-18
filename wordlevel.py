"""
TRM: Tiny Recursive Transformer for WikiText-103 (Word-level)
Author: ChatGPT
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
    max_epochs: int = 3
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    n_refinements: int = 3
    n_recursions: int = 4
    eval_interval: int = 500
    eval_iters: int = 50
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
        self.blocks = nn.ModuleList([TransformerBlock(config.embed_dim, config.n_heads, config.dropout) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.output_head = nn.Linear(config.embed_dim, vocab_size, bias=False)
        self.output_head.weight = self.embedding.weight

        # causal mask for context
        self.register_buffer("context_mask",
            torch.triu(torch.ones(config.context_size, config.context_size)*float('-inf'), diagonal=1))
    
    def forward(self, context, chunk):
        B = context.shape[0]
        ctx_emb = self.embedding(context)
        ctx_pos = self.pos_embedding(torch.arange(self.config.context_size, device=context.device))
        ctx = ctx_emb + ctx_pos
        for block in self.blocks:
            ctx = block(ctx, self.context_mask)
        
        chunk_emb = self.embedding(chunk)
        chunk_pos = self.pos_embedding(torch.arange(self.config.context_size, self.config.context_size + self.config.chunk_size, device=context.device))
        draft = chunk_emb + chunk_pos
        reasoning = torch.zeros_like(draft)

        all_logits = []
        for r in range(self.config.n_refinements):
            if r < self.config.n_refinements - 1:
                draft = draft.detach()
                reasoning = reasoning.detach()
            for _ in range(self.config.n_recursions):
                combined = torch.cat([ctx, draft + reasoning], dim=1)
                for block in self.blocks:
                    combined = block(combined)
                reasoning = combined[:, -self.config.chunk_size:, :]
            combined = torch.cat([ctx, draft + reasoning], dim=1)
            for block in self.blocks:
                combined = block(combined)
            draft = combined[:, -self.config.chunk_size:, :]
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
            logits_list = model(ctx, chk)
            logits = logits_list[-1]
            loss = F.cross_entropy(logits.reshape(-1, model.output_head.out_features), chk.reshape(-1))
            losses.append(loss.item())
    model.train()
    return np.mean(losses)

# ===============================
# MAIN FUNCTION
# ===============================
def main():
    os.makedirs("./outputs", exist_ok=True)
    config = Config()
    print("Loading WikiText-103 dataset (word-level)...")
    ds_train = load_dataset("wikitext", "wikitext-103-v1", split="train")
    ds_val   = load_dataset("wikitext", "wikitext-103-v1", split="validation")
    ds_test  = load_dataset("wikitext", "wikitext-103-v1", split="test")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Encode dataset
    train_tokens = tokenizer(ds_train['text'], truncation=False, padding=False)['input_ids']
    val_tokens   = tokenizer(ds_val['text'], truncation=False, padding=False)['input_ids']
    test_tokens  = tokenizer(ds_test['text'], truncation=False, padding=False)['input_ids']

    # Flatten lists
    train_tokens = [tok for sublist in train_tokens for tok in sublist]
    val_tokens = [tok for sublist in val_tokens for tok in sublist]
    test_tokens = [tok for sublist in test_tokens for tok in sublist]

    train_dataset = ChunkedDataset(train_tokens, config.context_size, config.chunk_size)
    val_dataset = ChunkedDataset(val_tokens, config.context_size, config.chunk_size)
    test_dataset = ChunkedDataset(test_tokens, config.context_size, config.chunk_size)

    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")
    print(f"Training sequences: {len(train_dataset)}")

    # Initialize model
    model = TinyRecursiveModel(vocab_size, config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    step = 0
    best_val_loss = float("inf")
    for epoch in range(config.max_epochs):
        for _ in range(len(train_dataset)//config.batch_size):
            ctx, chk = get_batch(train_dataset, config.batch_size, config.device)
            logits_list = model(ctx, chk)
            loss = sum(F.cross_entropy(l.reshape(-1, vocab_size), chk.reshape(-1)) for l in logits_list)/len(logits_list)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            step += 1
            if step % config.eval_interval == 0:
                val_loss = estimate_loss(model, val_dataset, config)
                print(f"Step {step} | Val loss: {val_loss:.4f} | Perplexity: {math.exp(min(val_loss,50)):.2f}")
                best_val_loss = min(best_val_loss, val_loss)

    # Save model
    torch.save(model.state_dict(), os.path.join(config.output_dir, "trm_wikitext103_wordlevel.pt"))

    # Final evaluation
    test_loss = estimate_loss(model, test_dataset, config)
    test_ppl = math.exp(min(test_loss, 50))
    print(f"Final Test Loss: {test_loss:.4f} | Test Perplexity: {test_ppl:.2f}")
    with open(os.path.join(config.output_dir, "trm_wikitext103_results.json"), "w") as f:
        json.dump({"test_loss": test_loss, "test_perplexity": test_ppl}, f, indent=2)
    print("Model and results saved in ./outputs")

if __name__ == "__main__":
    torch.manual_seed(1337)
    main()

