"""
TRM Language Experiments: FIXED VERSION

Key fixes:
1. Masked LM: Fixed refinement to actually help consistency
2. Autoregressive: Made task harder (shorter context, no memorization)
3. Early stopping: Train until model stops learning
4. Realistic PPL: Should see 50-150 range, not 1.0!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from datasets import load_dataset
import numpy as np
import json
from dataclasses import dataclass
from itertools import chain
import time
from datetime import datetime

@dataclass
class Config:
    # FIXED: Shorter context to prevent memorization
    context_size: int = 16  # Was 32 - too easy to memorize
    chunk_size: int = 2
    embed_dim: int = 128
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.2
    n_refinements: int = 3
    
    batch_size: int = 32
    
    # FIXED: Train until convergence with early stopping
    max_steps: int = 20000  # Maximum if no early stop
    eval_interval: int = 250
    eval_iters: int = 50
    patience: int = 10  # Stop after 10 evals without improvement
    
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    
    vocab_size: int = 5000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

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

# ============================================================================
# EXPERIMENT 1: MASKED LANGUAGE MODEL (FIXED)
# ============================================================================

class MaskedLM(nn.Module):
    """Baseline: Predict each mask independently"""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embedding = nn.Embedding(config.context_size, config.embed_dim)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.n_heads, config.dropout)
            for _ in range(config.n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.output_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
    
    def forward(self, tokens):
        """Bidirectional encoding (can see all context)"""
        B, T = tokens.shape
        
        tok_emb = self.embedding(tokens)
        pos_emb = self.pos_embedding(torch.arange(T, device=tokens.device))
        x = tok_emb + pos_emb
        
        for block in self.blocks:
            x = block(x, None)  # No mask = bidirectional
        
        x = self.ln_f(x)
        logits = self.output_head(x)
        return logits

class TRM_MaskedLM(nn.Module):
    """FIXED: TRM that refines both mask predictions together"""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embedding = nn.Embedding(config.context_size, config.embed_dim)
        
        # Context encoder
        self.context_blocks = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.n_heads, config.dropout)
            for _ in range(config.n_layers)
        ])
        
        # FIXED: Refinement blocks (separate from context encoding)
        self.refine_blocks = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.n_heads, config.dropout)
            for _ in range(config.n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.output_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
    
    def forward(self, tokens, mask_positions):
        """
        FIXED: Refine both mask representations together to ensure consistency
        """
        B, T = tokens.shape
        
        # Encode full context (bidirectional)
        tok_emb = self.embedding(tokens)
        pos_emb = self.pos_embedding(torch.arange(T, device=tokens.device))
        x = tok_emb + pos_emb
        
        for block in self.context_blocks:
            x = block(x, None)
        
        # Extract initial mask representations
        mask_reps = []
        for b in range(B):
            mask_reps.append(torch.stack([
                x[b, mask_positions[b, 0]],
                x[b, mask_positions[b, 1]]
            ]))
        mask_reps = torch.stack(mask_reps)  # [B, 2, embed_dim]
        
        # FIXED: Refine both masks together (they can influence each other)
        for refinement in range(self.config.n_refinements):
            # Let both masks attend to each other
            for block in self.refine_blocks:
                mask_reps = block(mask_reps, None)  # No mask = they see each other
        
        # Final predictions
        mask_reps = self.ln_f(mask_reps)
        logits = self.output_head(mask_reps)  # [B, 2, vocab]
        
        return logits

# ============================================================================
# EXPERIMENT 2: AUTOREGRESSIVE (FIXED - HARDER TASK)
# ============================================================================

class AutoregressiveLM(nn.Module):
    """Baseline: Standard causal LM"""
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
        self.output_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        
        total = config.context_size + config.chunk_size
        self.register_buffer("causal_mask",
            torch.triu(torch.full((total, total), -1e9), diagonal=1))
    
    def forward(self, context, chunk):
        B = context.shape[0]
        
        ctx_emb = self.embedding(context)
        ctx_pos = self.pos_embedding(torch.arange(self.config.context_size, device=context.device))
        ctx = ctx_emb + ctx_pos
        
        chunk_emb = self.embedding(chunk)
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
    """TRM with thinking time"""
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
        self.output_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        
        total = config.context_size + config.chunk_size
        self.register_buffer("causal_mask",
            torch.triu(torch.full((total, total), -1e9), diagonal=1))
    
    def forward(self, context, chunk):
        B = context.shape[0]
        
        ctx_emb = self.embedding(context)
        ctx_pos = self.pos_embedding(torch.arange(self.config.context_size, device=context.device))
        ctx = ctx_emb + ctx_pos
        
        chunk_emb = self.embedding(chunk)
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
# DATA
# ============================================================================

def build_word_vocab(tokens, vocab_size=5000):
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

class MaskedDataset(Dataset):
    def __init__(self, token_ids, context_size, mask_token_id):
        self.data = token_ids
        self.context_size = context_size
        self.mask_token_id = mask_token_id
        self.length = len(self.data) - self.context_size
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        seq = self.data[idx:idx + self.context_size]
        
        mask_positions = np.random.choice(range(1, self.context_size - 1), size=2, replace=False)
        mask_positions = sorted(mask_positions)
        
        masked_seq = seq.copy()
        targets = []
        for pos in mask_positions:
            targets.append(seq[pos])
            masked_seq[pos] = self.mask_token_id
        
        return (torch.tensor(masked_seq, dtype=torch.long),
                torch.tensor(mask_positions, dtype=torch.long),
                torch.tensor(targets, dtype=torch.long))

class AutoregressiveDataset(Dataset):
    def __init__(self, token_ids, context_size, chunk_size):
        self.data = token_ids
        self.context_size = context_size
        self.chunk_size = chunk_size
        self.length = len(self.data) - context_size - chunk_size
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        context = self.data[idx:idx + self.context_size]
        chunk = self.data[idx + self.context_size:idx + self.context_size + self.chunk_size]
        return (torch.tensor(context, dtype=torch.long),
                torch.tensor(chunk, dtype=torch.long))

def get_masked_batch(dataset, batch_size, device):
    indices = torch.randint(len(dataset), (batch_size,))
    seqs, positions, targets = [], [], []
    for idx in indices:
        seq, pos, tgt = dataset[idx]
        seqs.append(seq)
        positions.append(pos)
        targets.append(tgt)
    return (torch.stack(seqs).to(device),
            torch.stack(positions).to(device),
            torch.stack(targets).to(device))

def get_autoregressive_batch(dataset, batch_size, device):
    indices = torch.randint(len(dataset), (batch_size,))
    contexts, chunks = [], []
    for idx in indices:
        ctx, chk = dataset[idx]
        contexts.append(ctx)
        chunks.append(chk)
    return torch.stack(contexts).to(device), torch.stack(chunks).to(device)

# ============================================================================
# TRAINING (WITH EARLY STOPPING)
# ============================================================================

def train_masked_lm(model, dataset, config, name="Model"):
    """Train with early stopping"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    model.train()
    
    best_val_loss = float('inf')
    best_val_ppl = float('inf')
    patience_counter = 0
    results = []
    
    print(f"\nTraining {name}...")
    print(f"Early stopping: patience = {config.patience}")
    start_time = time.time()
    
    step = 0
    while step < config.max_steps:
        # Training step
        seqs, positions, targets = get_masked_batch(dataset['train'], config.batch_size, config.device)
        
        if isinstance(model, TRM_MaskedLM):
            logits = model(seqs, positions)
            loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), targets.reshape(-1))
        else:
            logits = model(seqs)
            pred_logits = torch.stack([
                logits[b, positions[b]] for b in range(seqs.shape[0])
            ])
            loss = F.cross_entropy(pred_logits.reshape(-1, config.vocab_size), targets.reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        step += 1
        
        # Evaluation
        if step % config.eval_interval == 0:
            model.eval()
            val_losses = []
            val_accs = []
            
            with torch.no_grad():
                for _ in range(config.eval_iters):
                    seqs, positions, targets = get_masked_batch(dataset['val'], config.batch_size, config.device)
                    
                    if isinstance(model, TRM_MaskedLM):
                        logits = model(seqs, positions)
                    else:
                        logits = model(seqs)
                        pred_logits = torch.stack([
                            logits[b, positions[b]] for b in range(seqs.shape[0])
                        ])
                        logits = pred_logits
                    
                    loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), targets.reshape(-1))
                    val_losses.append(loss.item())
                    
                    preds = torch.argmax(logits, dim=-1)
                    acc = (preds == targets).float().mean().item()
                    val_accs.append(acc)
            
            val_loss = np.mean(val_losses)
            val_ppl = np.exp(min(val_loss, 20))
            val_acc = np.mean(val_accs)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_ppl = val_ppl
                patience_counter = 0
                marker = "✓"
            else:
                patience_counter += 1
                marker = f"({patience_counter}/{config.patience})"
            
            print(f"Step {step:5d} | Loss: {val_loss:.4f} | PPL: {val_ppl:6.2f} | Acc: {val_acc:.3f} {marker}")
            
            results.append({
                'step': step,
                'val_loss': val_loss,
                'val_ppl': val_ppl,
                'val_acc': val_acc
            })
            
            model.train()
            
            # Early stopping
            if patience_counter >= config.patience:
                print(f"Early stopping at step {step}")
                break
    
    elapsed = (time.time() - start_time) / 60
    print(f"Training complete: {elapsed:.1f} minutes")
    print(f"Best val PPL: {best_val_ppl:.2f}")
    
    return best_val_ppl, val_acc, results

def train_autoregressive_lm(model, dataset, config, name="Model"):
    """Train with early stopping"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    model.train()
    
    best_val_loss = float('inf')
    best_val_ppl = float('inf')
    patience_counter = 0
    results = []
    
    print(f"\nTraining {name}...")
    print(f"Early stopping: patience = {config.patience}")
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
    print("TRM LANGUAGE EXPERIMENTS - FIXED VERSION")
    print("=" * 70)
    print(f"Device: {config.device}")
    print(f"Context: {config.context_size} words (FIXED: shorter to prevent memorization)")
    print(f"Chunk: {config.chunk_size} words")
    print(f"Refinements: {config.n_refinements}")
    print(f"Early stopping: patience = {config.patience}")
    print(f"Expected PPL range: 50-150 (not 1.0!)")
    print("=" * 70)
    
    train_tokens, val_tokens, vocab = load_word_level_data(config)
    mask_token_id = vocab['<MASK>']
    
    all_results = {}
    
    # ========================================================================
    # EXPERIMENT 1: MASKED LANGUAGE MODELING
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: MASKED LANGUAGE MODELING (FIXED)")
    print("=" * 70)
    print("Task: Predict 2 masked words simultaneously")
    print("Fix: Masks can now attend to each other during refinement")
    print("Hypothesis: TRM should WIN (refinement helps consistency)")
    print("=" * 70)
    
    masked_datasets = {
        'train': MaskedDataset(train_tokens, config.context_size, mask_token_id),
        'val': MaskedDataset(val_tokens, config.context_size, mask_token_id)
    }
    
    print("\n--- Baseline: Independent Prediction ---")
    baseline_masked = MaskedLM(config).to(config.device)
    print(f"Parameters: {sum(p.numel() for p in baseline_masked.parameters()):,}")
    baseline_masked_ppl, baseline_masked_acc, baseline_masked_history = train_masked_lm(
        baseline_masked, masked_datasets, config, "Baseline Masked LM"
    )
    
    print("\n--- TRM: Joint Refinement ---")
    trm_masked = TRM_MaskedLM(config).to(config.device)
    print(f"Parameters: {sum(p.numel() for p in trm_masked.parameters()):,}")
    trm_masked_ppl, trm_masked_acc, trm_masked_history = train_masked_lm(
        trm_masked, masked_datasets, config, "TRM Masked LM"
    )
    
    print("\n" + "=" * 70)
    print("EXPERIMENT 1 RESULTS")
    print("=" * 70)
    print(f"Baseline:  PPL = {baseline_masked_ppl:6.2f} | Acc = {baseline_masked_acc:.3f}")
    print(f"TRM:       PPL = {trm_masked_ppl:6.2f} | Acc = {trm_masked_acc:.3f}")
    
    if trm_masked_ppl < baseline_masked_ppl:
        improvement = (baseline_masked_ppl - trm_masked_ppl) / baseline_masked_ppl * 100
        print(f"\n✅ TRM WINS by {improvement:.1f}%!")
        print("Conclusion: Joint refinement helps with consistency")
    else:
        degradation = (trm_masked_ppl - baseline_masked_ppl) / baseline_masked_ppl * 100
        print(f"\n❌ Baseline wins by {degradation:.1f}%")
        print("Conclusion: Refinement still not helping (needs investigation)")
    
    all_results['masked_lm'] = {
        'baseline_ppl': baseline_masked_ppl,
        'trm_ppl': trm_masked_ppl,
        'baseline_history': baseline_masked_history,
        'trm_history': trm_masked_history
    }
    
    # ========================================================================
    # EXPERIMENT 2: AUTOREGRESSIVE
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: AUTOREGRESSIVE (FIXED - HARDER TASK)")
    print("=" * 70)
    print("Task: Predict next 2 words")
    print("Fix: Shorter context (16 not 32) to prevent memorization")
    print("Expected: PPL 50-150 (not 1.0!)")
    print("=" * 70)
    
    ar_datasets = {
        'train': AutoregressiveDataset(train_tokens, config.context_size, config.chunk_size),
        'val': AutoregressiveDataset(val_tokens, config.context_size, config.chunk_size)
    }
    
    print("\n--- Baseline: Standard Autoregressive ---")
    baseline_ar = AutoregressiveLM(config).to(config.device)
    print(f"Parameters: {sum(p.numel() for p in baseline_ar.parameters()):,}")
    baseline_ar_ppl, baseline_ar_history = train_autoregressive_lm(
        baseline_ar, ar_datasets, config, "Baseline Autoregressive"
    )
    
    print("\n--- TRM: With Thinking Time ---")
    trm_ar = TRM_AutoregressiveLM(config).to(config.device)
    print(f"Parameters: {sum(p.numel() for p in trm_ar.parameters()):,}")
    trm_ar_ppl, trm_ar_history = train_autoregressive_lm(
        trm_ar, ar_datasets, config, "TRM Autoregressive"
    )
    
    print("\n" + "=" * 70)
    print("EXPERIMENT 2 RESULTS")
    print("=" * 70)
    print(f"Baseline:  PPL = {baseline_ar_ppl:6.2f}")
    print(f"TRM:       PPL = {trm_ar_ppl:6.2f}")
    
    if trm_ar_ppl < baseline_ar_ppl:
        improvement = (baseline_ar_ppl - trm_ar_ppl) / baseline_ar_ppl * 100
        print(f"\n✅ TRM WINS by {improvement:.1f}%!")
        print("Conclusion: 'Thinking time' helps!")
    else:
        degradation = (trm_ar_ppl - baseline_ar_ppl) / baseline_ar_ppl * 100
        print(f"\n❌ Baseline wins by {degradation:.1f}%")
        print("Conclusion: Sequential info advantage > thinking time")
    
    all_results['autoregressive'] = {
        'baseline_ppl': baseline_ar_ppl,
        'trm_ppl': trm_ar_ppl,
        'baseline_history': baseline_ar_history,
        'trm_history': trm_ar_history
    }
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    print("\nExperiment 1: Masked LM")
    print(f"  Baseline: {baseline_masked_ppl:6.2f} PPL")
    print(f"  TRM:      {trm_masked_ppl:6.2f} PPL")
    if trm_masked_ppl < baseline_masked_ppl:
        print(f"  ✅ TRM wins - refinement helps!")
    else:
        print(f"  ❌ Baseline wins")
    
    print("\nExperiment 2: Autoregressive")
    print(f"  Baseline: {baseline_ar_ppl:6.2f} PPL")
    print(f"  TRM:      {trm_ar_ppl:6.2f} PPL")
    if trm_ar_ppl < baseline_ar_ppl:
        print(f"  ✅ TRM wins - thinking helps!")
    else:
        print(f"  ❌ Baseline wins")
    
    print("\n" + "=" * 70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"trm_experiments_fixed_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved: {results_file}")

if __name__ == "__main__":
    main()
