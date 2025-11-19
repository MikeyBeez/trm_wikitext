"""
3-Mask, 2-Pass Experiment: Testing if fewer passes help with more masks

Question: Do 3 masks work better with FEWER refinement passes?

Current: 3 masks, 3 passes = 1.1% improvement
Test: 3 masks, 2 passes = ???% improvement

Hypothesis: More masks = more work per pass = fewer passes needed
Pattern: N masks needs (4-N) passes?
  - 2 masks: 3 passes optimal (confirmed)
  - 3 masks: 2 passes optimal (?)
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
    context_size: int = 16
    num_masks: int = 3  # Testing 3 masks
    
    embed_dim: int = 128
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.2
    n_refinements: int = 2  # KEY: Testing 2 passes (was 3)
    
    batch_size: int = 32
    max_steps: int = 20000
    eval_interval: int = 250
    eval_iters: int = 50
    patience: int = 10
    
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    
    vocab_size: int = 10000
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
        B, T = tokens.shape
        
        tok_emb = self.embedding(tokens)
        pos_emb = self.pos_embedding(torch.arange(T, device=tokens.device))
        x = tok_emb + pos_emb
        
        for block in self.blocks:
            x = block(x, None)
        
        x = self.ln_f(x)
        logits = self.output_head(x)
        return logits

class TRM_MaskedLM(nn.Module):
    """TRM: Refine all masks together with 2 passes"""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embedding = nn.Embedding(config.context_size, config.embed_dim)
        
        self.context_blocks = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.n_heads, config.dropout)
            for _ in range(config.n_layers)
        ])
        
        self.refine_blocks = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.n_heads, config.dropout)
            for _ in range(config.n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.output_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
    
    def forward(self, tokens, mask_positions):
        B, T = tokens.shape
        
        tok_emb = self.embedding(tokens)
        pos_emb = self.pos_embedding(torch.arange(T, device=tokens.device))
        x = tok_emb + pos_emb
        
        for block in self.context_blocks:
            x = block(x, None)
        
        mask_reps = []
        for b in range(B):
            batch_masks = []
            for m in range(self.config.num_masks):
                batch_masks.append(x[b, mask_positions[b, m]])
            mask_reps.append(torch.stack(batch_masks))
        mask_reps = torch.stack(mask_reps)
        
        # KEY: Only 2 refinement passes
        for refinement in range(self.config.n_refinements):
            for block in self.refine_blocks:
                mask_reps = block(mask_reps, None)
        
        mask_reps = self.ln_f(mask_reps)
        logits = self.output_head(mask_reps)
        
        return logits

def build_word_vocab(tokens, vocab_size=10000):
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
    def __init__(self, token_ids, context_size, mask_token_id, num_masks=3):
        self.data = token_ids
        self.context_size = context_size
        self.mask_token_id = mask_token_id
        self.num_masks = num_masks
        self.length = len(self.data) - self.context_size
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        seq = self.data[idx:idx + self.context_size]
        
        mask_positions = np.random.choice(
            range(1, self.context_size - 1), 
            size=self.num_masks, 
            replace=False
        )
        mask_positions = sorted(mask_positions)
        
        masked_seq = seq.copy()
        targets = []
        for pos in mask_positions:
            targets.append(seq[pos])
            masked_seq[pos] = self.mask_token_id
        
        return (torch.tensor(masked_seq, dtype=torch.long),
                torch.tensor(mask_positions, dtype=torch.long),
                torch.tensor(targets, dtype=torch.long))

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

def train_masked_lm(model, dataset, config, name="Model"):
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
        seqs, positions, targets = get_masked_batch(dataset['train'], config.batch_size, config.device)
        
        if isinstance(model, TRM_MaskedLM):
            logits = model(seqs, positions)
            loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), targets.reshape(-1))
        else:
            logits = model(seqs)
            pred_logits = []
            for b in range(seqs.shape[0]):
                pred_logits.append(torch.stack([
                    logits[b, positions[b, m]] for m in range(config.num_masks)
                ]))
            pred_logits = torch.stack(pred_logits)
            loss = F.cross_entropy(pred_logits.reshape(-1, config.vocab_size), targets.reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        step += 1
        
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
                        pred_logits = []
                        for b in range(seqs.shape[0]):
                            pred_logits.append(torch.stack([
                                logits[b, positions[b, m]] for m in range(config.num_masks)
                            ]))
                        logits = torch.stack(pred_logits)
                    
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
            
            if patience_counter >= config.patience:
                print(f"Early stopping at step {step}")
                break
    
    elapsed = (time.time() - start_time) / 60
    print(f"Training complete: {elapsed:.1f} minutes")
    print(f"Best val PPL: {best_val_ppl:.2f}")
    
    return best_val_ppl, val_acc, results

def main():
    config = Config()
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    print("=" * 70)
    print("3-MASK, 2-PASS EXPERIMENT: FEWER PASSES FOR MORE MASKS?")
    print("=" * 70)
    print(f"Device: {config.device}")
    print(f"Masks: {config.num_masks}")
    print(f"Refinements: {config.n_refinements} (testing if fewer passes help)")
    print(f"\nPrevious results:")
    print(f"  2 masks, 3 passes: +16.1% improvement (OPTIMAL)")
    print(f"  2 masks, 5 passes: -13.4% degradation (OVER-REFINED)")
    print(f"  3 masks, 3 passes:  +1.1% improvement (TOO MANY PASSES?)")
    print(f"\nHypothesis: 3 masks might work better with FEWER passes")
    print(f"Pattern theory: N masks needs (4-N) passes?")
    print("=" * 70)
    
    train_tokens, val_tokens, vocab = load_word_level_data(config)
    mask_token_id = vocab['<MASK>']
    
    masked_datasets = {
        'train': MaskedDataset(train_tokens, config.context_size, mask_token_id, config.num_masks),
        'val': MaskedDataset(val_tokens, config.context_size, mask_token_id, config.num_masks)
    }
    
    print("\n" + "=" * 70)
    print("BASELINE: INDEPENDENT PREDICTIONS")
    print("=" * 70)
    
    baseline = MaskedLM(config).to(config.device)
    print(f"Parameters: {sum(p.numel() for p in baseline.parameters()):,}")
    baseline_ppl, baseline_acc, baseline_history = train_masked_lm(
        baseline, masked_datasets, config, "Baseline"
    )
    
    print("\n" + "=" * 70)
    print("TRM: JOINT REFINEMENT (2 PASSES)")
    print("=" * 70)
    
    trm = TRM_MaskedLM(config).to(config.device)
    print(f"Parameters: {sum(p.numel() for p in trm.parameters()):,}")
    trm_ppl, trm_acc, trm_history = train_masked_lm(
        trm, masked_datasets, config, "TRM (2 passes)"
    )
    
    print("\n" + "=" * 70)
    print("RESULTS: 3-MASK, 2-PASS EXPERIMENT")
    print("=" * 70)
    print(f"Baseline:  PPL = {baseline_ppl:6.2f} | Acc = {baseline_acc:.3f}")
    print(f"TRM:       PPL = {trm_ppl:6.2f} | Acc = {trm_acc:.3f}")
    
    if trm_ppl < baseline_ppl:
        improvement = (baseline_ppl - trm_ppl) / baseline_ppl * 100
        print(f"\n✅ TRM WINS by {improvement:.1f}%!")
        
        three_pass_improvement = 1.1
        print(f"\n📊 PASS DEPTH COMPARISON (3 masks):")
        print(f"  3 passes: {three_pass_improvement:.1f}% improvement")
        print(f"  2 passes: {improvement:.1f}% improvement")
        
        if improvement > three_pass_improvement * 2:
            print(f"\n🎉 MAJOR IMPROVEMENT! +{improvement - three_pass_improvement:.1f} points")
            print(f"  Conclusion: 3 masks works MUCH better with 2 passes!")
            print(f"  Pattern confirmed: More masks need fewer passes")
        elif improvement > three_pass_improvement * 1.3:
            print(f"\n✅ SIGNIFICANT IMPROVEMENT: +{improvement - three_pass_improvement:.1f} points")
            print(f"  Conclusion: 2 passes is better for 3 masks")
            print(f"  Pattern emerging: N masks ~ (4-N) passes")
        else:
            print(f"\n➡️  MODEST IMPROVEMENT: +{improvement - three_pass_improvement:.1f} points")
            print(f"  Conclusion: 2 passes slightly better, but 3 masks still harder")
    else:
        degradation = (trm_ppl - baseline_ppl) / baseline_ppl * 100
        print(f"\n❌ Baseline wins by {degradation:.1f}%")
        print(f"  Even 2 passes doesn't help 3 masks much")
    
    print("\n" + "=" * 70)
    print("SUMMARY: COMPLETE REFINEMENT DEPTH ANALYSIS")
    print("=" * 70)
    print(f"\nConfiguration: {config.num_masks} masks, {config.n_refinements} passes")
    print(f"Baseline: {baseline_ppl:.2f} PPL")
    print(f"TRM:      {trm_ppl:.2f} PPL")
    
    if trm_ppl < baseline_ppl:
        improvement = (baseline_ppl - trm_ppl) / baseline_ppl * 100
        print(f"Result:   {improvement:.1f}% improvement")
        
        print(f"\n💡 COMPLETE PICTURE:")
        print(f"   2 masks, 3 passes: 16.1% improvement ✅ (OPTIMAL)")
        print(f"   2 masks, 5 passes: -13.4% degradation ❌ (TOO MANY)")
        print(f"   3 masks, 3 passes:  1.1% improvement ⚠️ (SUBOPTIMAL)")
        print(f"   3 masks, 2 passes: {improvement:.1f}% improvement {'✅' if improvement > 1.1 else '⚠️'}")
        
        if improvement > 5:
            print(f"\n🎯 DESIGN PATTERN DISCOVERED:")
            print(f"   Optimal passes ≈ (4 - num_masks)")
            print(f"   2 masks: 3 passes")
            print(f"   3 masks: 2 passes")
            print(f"   Prediction: 4 masks would need 1 pass")
    
    results = {
        'num_masks': config.num_masks,
        'n_refinements': config.n_refinements,
        'baseline_ppl': baseline_ppl,
        'baseline_acc': baseline_acc,
        'trm_ppl': trm_ppl,
        'trm_acc': trm_acc,
        'improvement_percent': (baseline_ppl - trm_ppl) / baseline_ppl * 100 if trm_ppl < baseline_ppl else -(trm_ppl - baseline_ppl) / baseline_ppl * 100,
        'baseline_history': baseline_history,
        'trm_history': trm_history,
        'comparison': {
            '3_masks_3_passes': 1.1,
            '3_masks_2_passes': (baseline_ppl - trm_ppl) / baseline_ppl * 100 if trm_ppl < baseline_ppl else 0
        }
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"3masks_2passes_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()
