"""
TRM: Chunked Autoregressive Transformer with Refinement (WikiText-103)

Character-level TRM:
- Chunked autoregressive model with recursive refinement
- Uses AdamW optimizer
- Supports mixed-precision training
- Saves model and test results
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import os
import json
from dataclasses import dataclass
from datasets import load_dataset
from tqdm import trange, tqdm

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
    fp16: bool = True  # mixed precision

# ===============================
# DATASET CLASS
# ===============================
class ChunkedDataset(Dataset):
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
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=attn_mask, need_weights=False)[0]
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
        self.blocks = nn.ModuleList([TransformerBlock(config.embed_dim, config.n_heads, config.dropout)
                                     for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.output_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.output_head.weight = self.embedding.weight
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
        ctx_emb = self.embedding(context)
        ctx_pos = self.pos_embedding(torch.arange(self.config.context_size, device=context.device))
        ctx = ctx_emb + ctx_pos
        for block in self.blocks:
            ctx = block(ctx, self.context_mask)

        chunk_emb = self.embedding(chunk)
        chunk_pos = self.pos_embedding(torch.arange(self.config.context_size,
                                                    self.config.context_size + self.config.chunk_size,
                                                    device=context.device))
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
# UTILITIES
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
            loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), chk.reshape(-1))
            losses.append(loss.item())
    model.train()
    return np.mean(losses)

# ===============================
# MAIN
# ===============================
def main():
    os.makedirs("./outputs", exist_ok=True)
    config = Config()
    print("Loading WikiText-103 dataset (character-level mode)...")
    ds_train = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    ds_val   = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    ds_test  = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")

    train_text = "".join(ds_train["text"])
    val_text   = "".join(ds_val["text"])
    test_text  = "".join(ds_test["text"])

    train_dataset = ChunkedDataset(train_text, config.context_size, config.chunk_size)
    val_dataset   = ChunkedDataset(val_text, config.context_size, config.chunk_size)
    test_dataset  = ChunkedDataset(test_text, config.context_size, config.chunk_size)
    config.vocab_size = train_dataset.vocab_size

    print(f"Vocabulary size: {config.vocab_size}")
    print(f"Training sequences: {len(train_dataset)}")
    model = TinyRecursiveModel(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler(enabled=config.fp16)

    step = 0
    best_val_loss = float("inf")
    for epoch in range(config.max_epochs):
        for _ in range(len(train_dataset)//config.batch_size):
            ctx, chk = get_batch(train_dataset, config.batch_size, config.device)
            with torch.amp.autocast(device_type=config.device, enabled=config.fp16):
                logits_list = model(ctx, chk)
                loss = sum(F.cross_entropy(l.reshape(-1, config.vocab_size), chk.reshape(-1)) for l in logits_list) / len(logits_list)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            step += 1

            if step % config.eval_interval == 0:
                val_loss = estimate_loss(model, val_dataset, config)
                print(f"Step {step} | Val loss: {val_loss:.4f} | Perplexity: {math.exp(min(val_loss, 10)):.2f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), os.path.join(config.output_dir, "best_model.pth"))

    # Test evaluation
    model.load_state_dict(torch.load(os.path.join(config.output_dir, "best_model.pth")))
    test_loss = estimate_loss(model, test_dataset, config)
    test_ppl = math.exp(min(test_loss, 10))
    print(f"Final Test Loss: {test_loss:.4f} | Test Perplexity: {test_ppl:.2f}")

    # Save results
    result_path = os.path.join(config.output_dir, "trm_results.json")
    with open(result_path, "w") as f:
        json.dump({"test_loss": test_loss, "test_perplexity": test_ppl}, f, indent=2)
    print(f"Results saved to {result_path}")

if __name__ == "__main__":
    torch.manual_seed(1337)
    main()

