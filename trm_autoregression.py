"""
TRM Autoregressive: ACTUALLY HARD VERSION

Fixes to prevent 1.0 PPL:
1. Context = 8 (very short)
2. Chunk = 4 (predict more words)
3. Vocab = 10,000 (more diversity)
4. Dropout in embeddings (prevent memorization)
5. Random context offsets (no fixed patterns)

Expected: PPL 80-200 range
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from datasets import load_dataset
import numpy as np
import json
from dataclasses import dataclass
import time
from datetime import datetime

@dataclass
class Config:
    # MUCH HARDER TASK
    context_size: int = 8   # Was 16 - now VERY short
    chunk_size: int = 4     # Was 2 - now predict MORE words
    
    embed_dim: int = 128
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.3    # Was 0.2 - more dropout to prevent memorization
    n_refinements: int = 3
    
    batch_size: int = 32
    max_steps: int = 20000
    eval_interval: int = 250
    eval_iters: int = 50
    patience: int = 10
    
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    
    vocab_size: int = 10000  # Was 5000 - LARGER vocabulary
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    
    # NEW: Prevent memorization
    embedding_dropout: float = 0.2  # Dropout on embeddings
    random_context_offset: bool = True  # Random starting positions

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, dropout: float):
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

class AutoregressiveLM(nn.Module):
    """Baseline with anti-memorization features"""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embedding = nn.Embedding(config.context_size + config.chunk_size, config.embed_dim)
        
        # NEW: Dropout on embeddings to prevent memorization
        self.embed_dropout = nn.Dropout(config.embedding_dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.n_heads, config.dropout)
            for _ in range(config.n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.output_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        
        total = config.context_size + config.chunk_size
        self.register_buffer("causal_mask",
            torch.triu(torch.full((total, total), -1e9), diagonal=1))
    
    def forward(self, context, chunk):
        B = context.shape[0]
        
        # Embed with dropout
        ctx_emb = self.embedding(context)
        ctx_emb = self.embed_dropout(ctx_emb)
        ctx_pos = self.pos_embedding(torch.arange(self.config.context_size, device=context.device))
        ctx = ctx_emb + ctx_pos
        
        chunk_emb = self.embedding(chunk)
        chunk_emb = self.embed_dropout(chunk_emb)
        chunk_pos = self.pos_embedding(
            torch.arange(self.config.context_size,
                        self.config.context_size + self.config.chunk_size,
                        device=context.device)
        )
        chk = chunk_emb + chunk_pos
        
        full_seq = torch.cat([ctx, chk], dim=1)
        
        for block in self.blocks:
            full_seq = block(full_seq, self.causal_mask)
        
        chunk_out = full_seq[:, self.config.context_size:, :]
        chunk_out = self.ln_f(chunk_out)
        logits = self.output_head(chunk_out)
        
        return logits

class TRM_AutoregressiveLM(nn.Module):
    """TRM with anti-memorization features"""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embedding = nn.Embedding(config.context_size + config.chunk_size, config.embed_dim)
        
        # NEW: Dropout on embeddings
        self.embed_dropout = nn.Dropout(config.embedding_dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.n_heads, config.dropout)
            for _ in range(config.n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.output_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        
        total = config.context_size + config.chunk_size
        self.register_buffer("causal_mask",
            torch.triu(torch.full((total, total), -1e9), diagonal=1))
    
    def forward(self, context, chunk):
        B = context.shape[0]
        
        # Embed with dropout
        ctx_emb = self.embedding(context)
        ctx_emb = self.embed_dropout(ctx_emb)
        ctx_pos = self.pos_embedding(torch.arange(self.config.context_size, device=context.device))
        ctx = ctx_emb + ctx_pos
        
        chunk_emb = self.embedding(chunk)
        chunk_emb = self.embed_dropout(chunk_emb)
        chunk_pos = self.pos_embedding(
            torch.arange(self.config.context_size,
                        self.config.context_size + self.config.chunk_size,
                        device=context.device)
        )
        chk = chunk_emb + chunk_pos
        
        # Refinement loop
        for refinement in range(self.config.n_refinements):
            full_seq = torch.cat([ctx, chk], dim=1)
            
            for block in self.blocks:
                full_seq = block(full_seq, self.causal_mask)
            
            chk = full_seq[:, self.config.context_size:, :]
        
        chk = self.ln_f(chk)
        logits = self.output_head(chk)
        
        return logits

# ============================================================================
# DATA - WITH ANTI-MEMORIZATION
# ============================================================================

def build_word_vocab(tokens, vocab_size=10000):
    """Larger vocabulary"""
    from collections import Counter
    word_counts = Counter(tokens)
    most_common = word_counts.most_common(vocab_size - 3)
    
    vocab = {'<PAD>': 0, '<MASK>': 1, '<UNK>': 2}
    for word, _ in most_common:
        vocab[word] = len(vocab)
    
    return vocab

def load_word_level_data(config):
    print("Loading WikiText-2...")
    ds_train = load_dataset("wikitext", "wikitext-2-v1", split="train")
    ds_val = load_dataset("wikitext", "wikitext-2-v1", split="validation")
    
    def simple_word_tokenize(text):
        return text.lower().split()
    
    train_words = []
    for example in ds_train:
        words = simple_word_tokenize(example['text'])
        train_words.extend(words)
    
    val_words = []
    for example in ds_val:
        words = simple_word_tokenize(example['text'])
        val_words.extend(words)
    
    print(f"Building vocabulary from {len(train_words):,} words...")
    vocab = build_word_vocab(train_words, config.vocab_size)
    print(f"Vocabulary size: {len(vocab)}")
    
    def words_to_indices(words, vocab):
        return [vocab.get(w, vocab['<UNK>']) for w in words]
    
    train_indices = words_to_indices(train_words, vocab)
    val_indices = words_to_indices(val_words, vocab)
    
    return train_indices, val_indices, vocab

class AutoregressiveDataset(Dataset):
    """Dataset with random offsets to prevent memorization"""
    def __init__(self, token_ids, context_size, chunk_size, random_offset=True):
        self.data = token_ids
        self.context_size = context_size
        self.chunk_size = chunk_size
        self.random_offset = random_offset
        self.length = len(self.data) - context_size - chunk_size - 10  # Extra buffer for offsets
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # NEW: Add random offset to prevent memorization of fixed positions
        if self.random_offset and self.training:
            offset = np.random.randint(0, 5)  # Random 0-4 word offset
            idx = idx + offset
        
        context = self.data[idx:idx + self.context_size]
        chunk = self.data[idx + self.context_size:idx + self.context_size + self.chunk_size]
        return (torch.tensor(context, dtype=torch.long),
                torch.tensor(chunk, dtype=torch.long))

def get_autoregressive_batch(dataset, batch_size, device):
    indices = torch.randint(len(dataset), (batch_size,))
    contexts, chunks = [], []
    for idx in indices:
        ctx, chk = dataset[idx]
        contexts.append(ctx)
        chunks.append(chk)
    return torch.stack(contexts).to(device), torch.stack(chunks).to(device)

# ============================================================================
# TRAINING
# ============================================================================

def train_autoregressive_lm(model, dataset, config, name="Model"):
    """Train with early stopping and anti-memorization"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    model.train()
    
    best_val_loss = float('inf')
    best_val_ppl = float('inf')
    patience_counter = 0
    results = []
    
    print(f"\nTraining {name}...")
    print(f"Context: {config.context_size} words | Chunk: {config.chunk_size} words")
    print(f"Vocab: {config.vocab_size:,} | Early stopping: patience = {config.patience}")
    print(f"Expected: PPL 80-200 (NOT 1.0!)")
    start_time = time.time()
    
    step = 0
    while step < config.max_steps:
        context, chunk = get_autoregressive_batch(dataset['train'], config.batch_size, config.device)
        logits = model(context, chunk)
        loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), chunk.reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        step += 1
        
        if step % config.eval_interval == 0:
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for _ in range(config.eval_iters):
                    context, chunk = get_autoregressive_batch(dataset['val'], config.batch_size, config.device)
                    logits = model(context, chunk)
                    loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), chunk.reshape(-1))
                    val_losses.append(loss.item())
            
            val_loss = np.mean(val_losses)
            val_ppl = np.exp(min(val_loss, 20))
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_ppl = val_ppl
                patience_counter = 0
                marker = "✓"
            else:
                patience_counter += 1
                marker = f"({patience_counter}/{config.patience})"
            
            print(f"Step {step:5d} | Loss: {val_loss:.4f} | PPL: {val_ppl:6.2f} {marker}")
            
            results.append({
                'step': step,
                'val_loss': val_loss,
                'val_ppl': val_ppl
            })
            
            model.train()
            
            if patience_counter >= config.patience:
                print(f"Early stopping at step {step}")
                break
    
    elapsed = (time.time() - start_time) / 60
    print(f"Training complete: {elapsed:.1f} minutes")
    print(f"Best val PPL: {best_val_ppl:.2f}")
    
    if best_val_ppl < 5.0:
        print(f"\n⚠️  WARNING: PPL = {best_val_ppl:.2f} is still too low!")
        print("Model might still be memorizing. Consider:")
        print("  - Even shorter context")
        print("  - More dropout")
        print("  - Larger vocabulary")
    
    return best_val_ppl, results

# ============================================================================
# MAIN
# ============================================================================

def main():
    config = Config()
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    print("=" * 70)
    print("TRM AUTOREGRESSIVE - ACTUALLY HARD VERSION")
    print("=" * 70)
    print(f"Device: {config.device}")
    print(f"\n🔥 AGGRESSIVE ANTI-MEMORIZATION:")
    print(f"  Context: {config.context_size} words (VERY SHORT)")
    print(f"  Chunk: {config.chunk_size} words (MORE TO PREDICT)")
    print(f"  Vocab: {config.vocab_size:,} (2x LARGER)")
    print(f"  Dropout: {config.dropout} + {config.embedding_dropout} on embeddings")
    print(f"  Random offsets: {config.random_context_offset}")
    print(f"\n✅ Expected PPL: 80-200 (realistic)")
    print(f"❌ If we get PPL < 10: Still memorizing!")
    print("=" * 70)
    
    train_tokens, val_tokens, vocab = load_word_level_data(config)
    
    ar_datasets = {
        'train': AutoregressiveDataset(train_tokens, config.context_size, config.chunk_size, 
                                       random_offset=config.random_context_offset),
        'val': AutoregressiveDataset(val_tokens, config.context_size, config.chunk_size,
                                     random_offset=False)  # No offset for val
    }
    
    # Mark training dataset
    ar_datasets['train'].training = True
    ar_datasets['val'].training = False
    
    print("\n" + "=" * 70)
    print("BASELINE: STANDARD AUTOREGRESSIVE")
    print("=" * 70)
    
    baseline_ar = AutoregressiveLM(config).to(config.device)
    print(f"Parameters: {sum(p.numel() for p in baseline_ar.parameters()):,}")
    baseline_ar_ppl, baseline_ar_history = train_autoregressive_lm(
        baseline_ar, ar_datasets, config, "Baseline"
    )
    
    print("\n" + "=" * 70)
    print("TRM: WITH THINKING TIME")
    print("=" * 70)
    
    trm_ar = TRM_AutoregressiveLM(config).to(config.device)
    print(f"Parameters: {sum(p.numel() for p in trm_ar.parameters()):,}")
    trm_ar_ppl, trm_ar_history = train_autoregressive_lm(
        trm_ar, ar_datasets, config, "TRM"
    )
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS: AUTOREGRESSIVE")
    print("=" * 70)
    print(f"Baseline:  PPL = {baseline_ar_ppl:6.2f}")
    print(f"TRM:       PPL = {trm_ar_ppl:6.2f}")
    
    if baseline_ar_ppl < 10 or trm_ar_ppl < 10:
        print("\n⚠️  STILL MEMORIZING! Both models got PPL < 10")
        print("Need even more aggressive anti-memorization:")
        print("  - Context = 4 or 6 (even shorter)")
        print("  - Chunk = 6 or 8 (even more to predict)")
        print("  - Vocab = 15,000 (even larger)")
    else:
        print("\n✅ REALISTIC PERPLEXITY! No memorization.")
        
        if trm_ar_ppl < baseline_ar_ppl:
            improvement = (baseline_ar_ppl - trm_ar_ppl) / baseline_ar_ppl * 100
            print(f"\n🎉 TRM WINS by {improvement:.1f}%!")
            print("Conclusion: 'Thinking time' helps in autoregressive LM!")
        else:
            degradation = (trm_ar_ppl - baseline_ar_ppl) / baseline_ar_ppl * 100
            print(f"\n❌ Baseline wins by {degradation:.1f}%")
            print("Conclusion: Sequential information advantage > thinking time")
    
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print("=" * 70)
    print(f"Task difficulty: Context={config.context_size}, Chunk={config.chunk_size}, Vocab={config.vocab_size:,}")
    print(f"Baseline: {baseline_ar_ppl:.2f} PPL")
    print(f"TRM:      {trm_ar_ppl:.2f} PPL")
    
    if baseline_ar_ppl > 10 and trm_ar_ppl > 10:
        print("\n✅ This is a fair comparison - both models working on hard task")
    
    results = {
        'baseline_ppl': baseline_ar_ppl,
        'trm_ppl': trm_ar_ppl,
        'config': {
            'context_size': config.context_size,
            'chunk_size': config.chunk_size,
            'vocab_size': config.vocab_size,
            'dropout': config.dropout,
            'embedding_dropout': config.embedding_dropout
        },
        'baseline_history': baseline_ar_history,
        'trm_history': trm_ar_history
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"trm_autoregressive_hard_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()
