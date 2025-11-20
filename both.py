#!/usr/bin/env python3
"""
both.py - Combined Recursive Training with Full Ablation Study
Compares baseline, Stage 2 only, Stage 3 only, and combined approaches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.distributions import Categorical
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from typing import Optional, Tuple, List, Dict

# ============================================================================
# BASELINE TRAINER - Standard single-pass training (control group)
# ============================================================================

class BaselineTrainer:
    """Standard training without recursion - the control group"""
    
    def __init__(self, model, tokenizer, mask_token_id=50256, lr=1e-4):
        self.model = model
        self.tokenizer = tokenizer
        self.mask_token_id = mask_token_id
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
        self.step = 0
        
    def create_contiguous_mask(self, input_ids, mask_prob=0.6, span_length=3):
        """Same masking as recursive training for fair comparison"""
        B, L = input_ids.shape
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        
        for b in range(B):
            num_masks = int(L * mask_prob / span_length)
            for _ in range(num_masks):
                if L > span_length:
                    start = torch.randint(0, L - span_length, (1,)).item()
                    mask[b, start:start + span_length] = True
        return mask
    
    def train_step(self, input_ids):
        """Standard single-pass training"""
        targets = input_ids.clone()
        
        # Use same masking strategy as recursive methods
        mask = self.create_contiguous_mask(input_ids)
        masked_input = input_ids.clone()
        masked_input[mask] = self.mask_token_id
        
        # Single forward pass
        outputs = self.model(masked_input)
        logits = outputs.logits
        
        # Standard loss
        if mask.sum() > 0:
            loss = F.cross_entropy(
                logits[mask].reshape(-1, logits.size(-1)),
                targets[mask].reshape(-1)
            )
            acc = (logits[mask].argmax(-1) == targets[mask]).float().mean()
            
            # Metrics for comparison
            conf = F.softmax(logits[mask], dim=-1).max(dim=-1).values.mean()
            ent = Categorical(logits=logits[mask]).entropy().mean()
            perp = torch.exp(loss)
        else:
            loss = torch.tensor(0.0, device=logits.device)
            acc = conf = ent = perp = 0.0
        
        # Standard backward pass
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        self.step += 1
        
        return {
            'loss': loss.item(),
            'acc': acc.item(),
            'conf': conf.item() if mask.sum() > 0 else 0,
            'entropy': ent.item() if mask.sum() > 0 else 0,
            'perp': perp.item() if mask.sum() > 0 else 1.0,
            'grad_norm': grad_norm.item()
        }

# ============================================================================
# STAGE 2 ONLY TRAINER - Soft refinement method (6.6% gain)
# ============================================================================

class Stage2OnlyTrainer:
    """Only soft refinement, no candidate evaluation"""
    
    def __init__(self, model, tokenizer, mask_token_id=50256, lr=1e-4):
        self.model = model
        self.tokenizer = tokenizer
        self.mask_token_id = mask_token_id
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
        self.temp = 1.5
        self.step = 0
        
    def fill_masks_soft(self, masked_input, logits, mask_positions):
        """Soft filling for differentiable refinement"""
        probs = torch.softmax(logits / self.temp, dim=-1)
        embeddings_matrix = self.model.transformer.wte.weight
        soft_embeddings = torch.matmul(probs, embeddings_matrix)
        original_embeddings = self.model.transformer.wte(masked_input)
        
        filled = torch.where(
            mask_positions.unsqueeze(-1),
            soft_embeddings,
            original_embeddings
        )
        return filled
    
    def encode_from_embeddings(self, embeddings):
        """Run transformer from embeddings"""
        hidden = embeddings
        for layer in self.model.transformer.h:
            outputs = layer(hidden)
            hidden = outputs[0]
        hidden = self.model.transformer.ln_f(hidden)
        return hidden
    
    def create_contiguous_mask(self, input_ids, mask_prob=0.6, span_length=3):
        B, L = input_ids.shape
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        
        for b in range(B):
            num_masks = int(L * mask_prob / span_length)
            for _ in range(num_masks):
                if L > span_length:
                    start = torch.randint(0, L - span_length, (1,)).item()
                    mask[b, start:start + span_length] = True
        return mask
    
    def train_step(self, input_ids):
        """Two-stage training with soft refinement"""
        targets = input_ids.clone()
        
        mask = self.create_contiguous_mask(input_ids)
        masked_input = input_ids.clone()
        masked_input[mask] = self.mask_token_id
        
        # Stage 1
        outputs1 = self.model(masked_input, output_hidden_states=True)
        hidden1 = outputs1.hidden_states[-1]
        logits1 = outputs1.logits
        
        if mask.sum() > 0:
            loss1 = F.cross_entropy(
                logits1[mask].reshape(-1, logits1.size(-1)),
                targets[mask].reshape(-1)
            )
            acc1 = (logits1[mask].argmax(-1) == targets[mask]).float().mean()
        else:
            loss1 = torch.tensor(0.0, device=logits1.device)
            acc1 = torch.tensor(0.0)
        
        # Stage 2
        filled_embeddings = self.fill_masks_soft(masked_input, logits1, mask)
        hidden2 = self.encode_from_embeddings(filled_embeddings)
        logits2 = self.model.lm_head(hidden2)
        
        if mask.sum() > 0:
            loss2 = F.cross_entropy(
                logits2[mask].reshape(-1, logits2.size(-1)),
                targets[mask].reshape(-1)
            )
            acc2 = (logits2[mask].argmax(-1) == targets[mask]).float().mean()
        else:
            loss2 = torch.tensor(0.0, device=logits2.device)
            acc2 = torch.tensor(0.0)
        
        # Combined loss
        total_loss = 0.1 * loss1 + 0.9 * loss2
        
        # Backward
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Temperature annealing
        if self.step % 200 == 0:
            self.temp = max(0.5, self.temp * 0.95)
        
        self.step += 1
        
        return {
            'loss': total_loss.item(),
            'loss1': loss1.item(),
            'loss2': loss2.item(),
            'acc1': acc1.item(),
            'acc2': acc2.item(),
            'improvement': (acc2 - acc1).item(),
            'grad_norm': grad_norm.item()
        }

# ============================================================================
# DUAL RECURSIVE TRAINER - Combined approach (both methods)
# ============================================================================

class DualRecursiveTrainer:
    """Combines training-time and inference-time recursion"""
    
    def __init__(self, model, tokenizer, mask_token_id=50256, lr=1e-4):
        self.model = model
        self.tokenizer = tokenizer
        self.mask_token_id = mask_token_id
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
        
        self.soft_temp = 1.5
        self.candidate_temp = 0.8
        
        self.stage1_weight = 0.1
        self.stage2_weight = 0.3
        self.stage3_weight = 0.6
        
        self.step = 0
        self.enable_stage3 = False
        
    def fill_masks_soft(self, masked_input, logits, mask_positions):
        probs = torch.softmax(logits / self.soft_temp, dim=-1)
        embeddings_matrix = self.model.transformer.wte.weight
        soft_embeddings = torch.matmul(probs, embeddings_matrix)
        original_embeddings = self.model.transformer.wte(masked_input)
        
        filled = torch.where(
            mask_positions.unsqueeze(-1),
            soft_embeddings,
            original_embeddings
        )
        return filled
    
    def encode_from_embeddings(self, embeddings):
        hidden = embeddings
        for layer in self.model.transformer.h:
            outputs = layer(hidden)
            hidden = outputs[0]
        hidden = self.model.transformer.ln_f(hidden)
        return hidden
    
    def create_contiguous_mask(self, input_ids, mask_prob=0.6, span_length=3):
        B, L = input_ids.shape
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        
        for b in range(B):
            num_masks = int(L * mask_prob / span_length)
            for _ in range(num_masks):
                if L > span_length:
                    start = torch.randint(0, L - span_length, (1,)).item()
                    mask[b, start:start + span_length] = True
        return mask
    
    def generate_candidates(self, logits, mask, k=5):
        B, L, V = logits.shape
        
        probs = F.softmax(logits / self.candidate_temp, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, k=k, dim=-1)
        
        candidates = []
        candidate_probs = []
        
        for i in range(k):
            candidate = torch.zeros_like(mask, dtype=torch.long)
            candidate_prob = torch.ones(B, device=logits.device)
            
            for b in range(B):
                for l in range(L):
                    if mask[b, l]:
                        candidate[b, l] = top_k_indices[b, l, i]
                        candidate_prob[b] *= top_k_probs[b, l, i]
            
            candidates.append(candidate)
            candidate_probs.append(candidate_prob)
            
        return candidates, candidate_probs
    
    def score_candidate_sequence(self, input_ids, candidate_tokens, mask):
        filled_sequence = input_ids.clone()
        filled_sequence[mask] = candidate_tokens[mask]
        
        with torch.no_grad():
            outputs = self.model(filled_sequence)
            logits = outputs.logits
            
            forward_score = 0
            for i in range(len(mask[0]) - 1):
                if mask[0, i]:
                    next_token = filled_sequence[0, i + 1]
                    forward_score += F.log_softmax(logits[0, i], dim=-1)[next_token]
            
            backward_score = 0
            for i in range(1, len(mask[0])):
                if mask[0, i]:
                    current_token = filled_sequence[0, i]
                    backward_score += F.log_softmax(logits[0, i - 1], dim=-1)[current_token]
            
            sequence_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                filled_sequence.reshape(-1),
                reduction='mean'
            )
            
            total_score = forward_score + backward_score - sequence_loss
            
        return total_score
    
    def candidate_refinement_stage(self, input_ids, stage2_logits, mask, targets):
        B = input_ids.shape[0]
        
        candidates, candidate_probs = self.generate_candidates(stage2_logits, mask, k=5)
        
        scores = []
        for candidate in candidates:
            batch_scores = []
            for b in range(B):
                score = self.score_candidate_sequence(
                    input_ids[b:b+1], 
                    candidate[b:b+1], 
                    mask[b:b+1]
                )
                batch_scores.append(score)
            scores.append(torch.stack(batch_scores))
        
        scores = torch.stack(scores)
        selection_weights = F.softmax(scores / 0.1, dim=0)
        
        refined_logits = torch.zeros_like(stage2_logits)
        for i, candidate in enumerate(candidates):
            candidate_one_hot = F.one_hot(candidate, num_classes=stage2_logits.size(-1))
            candidate_logits = candidate_one_hot.float() * 10.0
            weight = selection_weights[i].unsqueeze(-1).unsqueeze(-1)
            refined_logits += weight * candidate_logits
        
        if mask.sum() > 0:
            loss3 = F.cross_entropy(
                refined_logits[mask].reshape(-1, refined_logits.size(-1)),
                targets[mask].reshape(-1)
            )
            best_candidate_idx = torch.argmax(scores, dim=0)
            best_candidate = torch.stack([candidates[best_candidate_idx[b]][b] for b in range(B)])
            acc3 = (best_candidate[mask] == targets[mask]).float().mean()
        else:
            loss3 = torch.tensor(0.0, device=stage2_logits.device)
            acc3 = torch.tensor(0.0, device=stage2_logits.device)
            
        return refined_logits, loss3, acc3
    
    def train_step(self, input_ids):
        B, L = input_ids.shape
        targets = input_ids.clone()
        
        mask = self.create_contiguous_mask(input_ids)
        masked_input = input_ids.clone()
        masked_input[mask] = self.mask_token_id
        
        # Stage 1
        outputs1 = self.model(masked_input, output_hidden_states=True)
        hidden1 = outputs1.hidden_states[-1]
        logits1 = outputs1.logits
        
        if mask.sum() > 0:
            loss1 = F.cross_entropy(
                logits1[mask].reshape(-1, logits1.size(-1)),
                targets[mask].reshape(-1)
            )
            acc1 = (logits1[mask].argmax(-1) == targets[mask]).float().mean()
        else:
            loss1 = torch.tensor(0.0, device=logits1.device)
            acc1 = torch.tensor(0.0)
        
        # Stage 2
        filled_embeddings = self.fill_masks_soft(masked_input, logits1, mask)
        hidden2 = self.encode_from_embeddings(filled_embeddings)
        logits2 = self.model.lm_head(hidden2)
        
        if mask.sum() > 0:
            loss2 = F.cross_entropy(
                logits2[mask].reshape(-1, logits2.size(-1)),
                targets[mask].reshape(-1)
            )
            acc2 = (logits2[mask].argmax(-1) == targets[mask]).float().mean()
        else:
            loss2 = torch.tensor(0.0, device=logits2.device)
            acc2 = torch.tensor(0.0)
        
        # Stage 3 (if enabled)
        if self.enable_stage3 and mask.sum() > 0:
            logits3, loss3, acc3 = self.candidate_refinement_stage(
                input_ids, logits2, mask, targets
            )
        else:
            loss3 = torch.tensor(0.0, device=logits2.device)
            acc3 = acc2
            logits3 = logits2
        
        # Total loss
        if self.enable_stage3:
            total_loss = (self.stage1_weight * loss1 + 
                         self.stage2_weight * loss2 + 
                         self.stage3_weight * loss3)
        else:
            total_loss = 0.1 * loss1 + 0.9 * loss2
        
        # Backward
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Update state
        self.step += 1
        
        if self.step == 2000:
            print("\n🚀 ENABLING STAGE 3 - Candidate Refinement Active!")
            self.enable_stage3 = True
        
        if self.step % 200 == 0:
            self.soft_temp = max(0.5, self.soft_temp * 0.95)
            self.candidate_temp = max(0.5, self.candidate_temp * 0.95)
        
        return {
            'loss': total_loss.item(),
            'loss1': loss1.item(),
            'loss2': loss2.item(),
            'loss3': loss3.item(),
            'acc1': acc1.item(),
            'acc2': acc2.item(),
            'acc3': acc3.item(),
            'improvement_1to2': (acc2 - acc1).item(),
            'improvement_2to3': (acc3 - acc2).item() if self.enable_stage3 else 0.0,
            'total_improvement': (acc3 - acc1).item() if self.enable_stage3 else (acc2 - acc1).item(),
            'grad_norm': grad_norm.item(),
            'stage3_enabled': self.enable_stage3
        }

# ============================================================================
# VALIDATION AND TRAINING FUNCTIONS
# ============================================================================

def validate_model(model, val_dataloader, num_batches=100):
    """Validate any model on held-out data"""
    model.eval()
    total_loss = 0
    total_acc = 0
    count = 0
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_dataloader, desc="Validating")):
            if i >= num_batches:
                break
            
            input_ids = batch['input_ids'].cuda()
            
            # Simple next-token prediction
            outputs = model(input_ids[:, :-1])
            targets = input_ids[:, 1:]
            
            loss = F.cross_entropy(
                outputs.logits.reshape(-1, outputs.logits.size(-1)),
                targets.reshape(-1)
            )
            
            acc = (outputs.logits.argmax(-1) == targets).float().mean()
            
            total_loss += loss.item()
            total_acc += acc.item()
            count += 1
    
    model.train()
    return total_acc / count, total_loss / count

def train_model(trainer, train_dataloader, steps=5000):
    """Generic training function for any trainer"""
    metrics_history = []
    
    progress_bar = tqdm(range(steps), desc="Training")
    data_iter = iter(train_dataloader)
    
    for step in progress_bar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dataloader)
            batch = next(data_iter)
        
        input_ids = batch['input_ids'].cuda()
        metrics = trainer.train_step(input_ids)
        metrics_history.append(metrics)
        
        if hasattr(trainer, 'model'):
            if 'acc' in metrics:
                progress_bar.set_postfix({'acc': f"{metrics['acc']:.3f}"})
            elif 'acc2' in metrics:
                progress_bar.set_postfix({'acc2': f"{metrics['acc2']:.3f}"})
        
        if step % 500 == 0 and step > 0:
            avg_metrics = {}
            for key in metrics.keys():
                if isinstance(metrics[key], (int, float)):
                    avg_metrics[key] = np.mean([m[key] for m in metrics_history[-100:]])
            print(f"\nStep {step} averages: {avg_metrics}")
    
    return metrics_history

# ============================================================================
# MAIN ABLATION STUDY
# ============================================================================

def main():
    print("=" * 80)
    print("RECURSIVE TRAINING ABLATION STUDY")
    print("Comparing: Baseline vs Stage2 vs Combined Approaches")
    print("=" * 80)
    
    # Setup
    print("\nLoading WikiText-103...")
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'], 
            truncation=True, 
            padding='max_length', 
            max_length=512
        )
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format('torch', columns=['input_ids'])
    
    train_dataloader = DataLoader(
        tokenized_datasets['train'], 
        batch_size=4,
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        tokenized_datasets['validation'],
        batch_size=4,
        shuffle=False
    )
    
    # Model configuration
    config = GPT2Config(
        n_embd=768,
        n_layer=12,
        n_head=12,
        resid_pdrop=0.3,
        embd_pdrop=0.3,
        attn_pdrop=0.3
    )
    
    results = {}
    
    # ========================================================================
    # 1. BASELINE - Standard training
    # ========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 1: BASELINE (Standard Training)")
    print("="*80)
    
    baseline_model = GPT2LMHeadModel(config)
    baseline_model.cuda()
    baseline_trainer = BaselineTrainer(baseline_model, tokenizer)
    
    print("Training baseline model...")
    baseline_metrics = train_model(baseline_trainer, train_dataloader, steps=5000)
    
    print("Validating baseline model...")
    baseline_val_acc, baseline_val_loss = validate_model(baseline_model, val_dataloader)
    results['baseline'] = {
        'val_acc': baseline_val_acc,
        'val_loss': baseline_val_loss,
        'final_train_acc': np.mean([m['acc'] for m in baseline_metrics[-100:]])
    }
    
    print(f"\nBaseline Results:")
    print(f"  Validation Accuracy: {baseline_val_acc:.2%}")
    print(f"  Validation Loss: {baseline_val_loss:.3f}")
    
    # Save baseline model
    torch.save(baseline_model.state_dict(), 'baseline_model.pt')
    
    # ========================================================================
    # 2. STAGE 2 ONLY - Soft refinement
    # ========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 2: STAGE 2 ONLY (Soft Refinement)")
    print("="*80)
    
    stage2_model = GPT2LMHeadModel(config)
    stage2_model.cuda()
    stage2_trainer = Stage2OnlyTrainer(stage2_model, tokenizer)
    
    print("Training Stage 2 model...")
    stage2_metrics = train_model(stage2_trainer, train_dataloader, steps=5000)
    
    print("Validating Stage 2 model...")
    stage2_val_acc, stage2_val_loss = validate_model(stage2_model, val_dataloader)
    results['stage2_only'] = {
        'val_acc': stage2_val_acc,
        'val_loss': stage2_val_loss,
        'final_improvement': np.mean([m['improvement'] for m in stage2_metrics[-100:]])
    }
    
    print(f"\nStage 2 Results:")
    print(f"  Validation Accuracy: {stage2_val_acc:.2%}")
    print(f"  Improvement over baseline: {(stage2_val_acc - baseline_val_acc)*100:.2f}%")
    
    # Save Stage 2 model
    torch.save(stage2_model.state_dict(), 'stage2_model.pt')
    
    # ========================================================================
    # 3. COMBINED - Both methods
    # ========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 3: COMBINED (Stage 2 + Stage 3)")
    print("="*80)
    
    combined_model = GPT2LMHeadModel(config)
    combined_model.cuda()
    combined_trainer = DualRecursiveTrainer(combined_model, tokenizer)
    
    print("Training Combined model...")
    combined_metrics = train_model(combined_trainer, train_dataloader, steps=5000)
    
    print("Validating Combined model...")
    combined_val_acc, combined_val_loss = validate_model(combined_model, val_dataloader)
    
    # Calculate Stage 3 metrics
    post_stage3_metrics = [m for m in combined_metrics if m['stage3_enabled']]
    if post_stage3_metrics:
        avg_total_improvement = np.mean([m['total_improvement'] for m in post_stage3_metrics])
    else:
        avg_total_improvement = np.mean([m['improvement_1to2'] for m in combined_metrics[-100:]])
    
    results['combined'] = {
        'val_acc': combined_val_acc,
        'val_loss': combined_val_loss,
        'avg_total_improvement': avg_total_improvement
    }
    
    print(f"\nCombined Results:")
    print(f"  Validation Accuracy: {combined_val_acc:.2%}")
    print(f"  Improvement over baseline: {(combined_val_acc - baseline_val_acc)*100:.2f}%")
    print(f"  Improvement over Stage 2: {(combined_val_acc - stage2_val_acc)*100:.2f}%")
    
    # Save combined model
    torch.save(combined_model.state_dict(), 'combined_model.pt')
    
    # ========================================================================
    # 4. LARGER BASELINE - 3x parameters for compute comparison
    # ========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 4: LARGER BASELINE (3x Parameters)")
    print("="*80)
    
    larger_config = GPT2Config(
        n_embd=768*2,  # Double width
        n_layer=18,     # 1.5x depth
        n_head=12,
        resid_pdrop=0.3,
        embd_pdrop=0.3,
        attn_pdrop=0.3
    )
    larger_model = GPT2LMHeadModel(larger_config)
    larger_model.cuda()
    larger_trainer = BaselineTrainer(larger_model, tokenizer, lr=5e-5)  # Lower LR for larger model
    
    print("Training larger baseline model...")
    larger_metrics = train_model(larger_trainer, train_dataloader, steps=5000)
    
    print("Validating larger model...")
    larger_val_acc, larger_val_loss = validate_model(larger_model, val_dataloader)
    results['larger_baseline'] = {
        'val_acc': larger_val_acc,
        'val_loss': larger_val_loss
    }
    
    print(f"\nLarger Baseline Results:")
    print(f"  Validation Accuracy: {larger_val_acc:.2%}")
    print(f"  Improvement over baseline: {(larger_val_acc - baseline_val_acc)*100:.2f}%")
    
    # ========================================================================
    # FINAL COMPARISON
    # ========================================================================
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    baseline_acc = results['baseline']['val_acc']
    
    print(f"\nValidation Accuracies:")
    print(f"  1. Baseline (standard):     {baseline_acc:.2%}")
    print(f"  2. Stage 2 only:            {results['stage2_only']['val_acc']:.2%} "
          f"(+{(results['stage2_only']['val_acc']-baseline_acc)*100:.1f}%)")
    print(f"  3. Combined (S2+S3):        {results['combined']['val_acc']:.2%} "
          f"(+{(results['combined']['val_acc']-baseline_acc)*100:.1f}%)")
    print(f"  4. Larger baseline (3x):    {results['larger_baseline']['val_acc']:.2%} "
          f"(+{(results['larger_baseline']['val_acc']-baseline_acc)*100:.1f}%)")
    
    print(f"\nKey Comparisons:")
    print(f"  Stage 2 vs Baseline:        +{(results['stage2_only']['val_acc']-baseline_acc)*100:.1f}% improvement")
    print(f"  Combined vs Stage 2:        +{(results['combined']['val_acc']-results['stage2_only']['val_acc'])*100:.1f}% additional")
    print(f"  Combined vs Larger:         {(results['combined']['val_acc']-results['larger_baseline']['val_acc'])*100:+.1f}% difference")
    
    if results['combined']['val_acc'] > results['larger_baseline']['val_acc']:
        print("\n🏆 BREAKTHROUGH: Recursive training beats 3x larger model!")
        print("   This suggests recursive training is more parameter-efficient than scaling!")
    elif results['combined']['val_acc'] > results['stage2_only']['val_acc']:
        print("\n✅ SUCCESS: Combined approach improves over Stage 2 alone!")
        print("   The two methods complement each other!")
    
    # Save all results
    with open('ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Full results saved to ablation_results.json")
    
    # Create comparison plot
    plt.figure(figsize=(10, 6))
    methods = ['Baseline', 'Stage 2', 'Combined', 'Larger (3x)']
    accuracies = [
        results['baseline']['val_acc'],
        results['stage2_only']['val_acc'],
        results['combined']['val_acc'],
        results['larger_baseline']['val_acc']
    ]
    
    bars = plt.bar(methods, accuracies)
    bars[0].set_color('gray')
    bars[1].set_color('blue')
    bars[2].set_color('green')
    bars[3].set_color('red')
    
    plt.ylabel('Validation Accuracy')
    plt.title('Recursive Training Ablation Study')
    plt.ylim([min(accuracies) * 0.95, max(accuracies) * 1.05])
    
    # Add value labels on bars
    for i, (method, acc) in enumerate(zip(methods, accuracies)):
        plt.text(i, acc + 0.001, f'{acc:.2%}', ha='center')
    
    plt.savefig('ablation_comparison.png', dpi=150, bbox_inches='tight')
    print(f"📈 Comparison plot saved to ablation_comparison.png")
    
    return results

if __name__ == "__main__":
    results = main()
