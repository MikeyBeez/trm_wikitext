#!/usr/bin/env python3
"""
Validation Analysis for Recursive Training
Tests if the improvement generalizes to unseen data
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
import seaborn as sns
import json

class RecursiveValidator:
    def __init__(self, checkpoint_path, tokenizer):
        """Load trained model from checkpoint"""
        self.tokenizer = tokenizer
        self.mask_token_id = tokenizer.eos_token_id  # Using EOS as mask
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        
        # Recreate model architecture
        config = GPT2Config(
            n_embd=768,
            n_layer=12,
            n_head=12,
            resid_pdrop=0.3,
            embd_pdrop=0.3,
            attn_pdrop=0.3
        )
        self.model = GPT2LMHeadModel(config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.cuda()
        self.model.eval()
        
        self.temperature = 0.5  # Use final training temperature
        
    def fill_masks_soft(self, masked_input, logits, mask_positions):
        """Soft filling for differentiable refinement"""
        probs = torch.softmax(logits / self.temperature, dim=-1)
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
        """Create contiguous span masks matching training"""
        B, L = input_ids.shape
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        
        for b in range(B):
            num_masks = int(L * mask_prob / span_length)
            for _ in range(num_masks):
                if L > span_length:
                    start = torch.randint(0, L - span_length, (1,)).item()
                    mask[b, start:start + span_length] = True
        return mask
    
    def analyze_batch(self, input_ids):
        """Analyze one batch for refinement patterns"""
        with torch.no_grad():
            # Create masks
            mask = self.create_contiguous_mask(input_ids)
            masked_input = input_ids.clone()
            masked_input[mask] = self.mask_token_id
            
            # Stage 1
            outputs1 = self.model(masked_input, output_hidden_states=True)
            hidden1 = outputs1.hidden_states[-1]
            logits1 = outputs1.logits
            
            # Stage 2
            filled_embeddings = self.fill_masks_soft(masked_input, logits1, mask)
            hidden2 = self.encode_from_embeddings(filled_embeddings)
            logits2 = self.model.lm_head(hidden2)
            
            # Compute metrics
            if mask.sum() > 0:
                # Accuracy
                pred1 = logits1[mask].argmax(-1)
                pred2 = logits2[mask].argmax(-1)
                targets = input_ids[mask]
                acc1 = (pred1 == targets).float().mean()
                acc2 = (pred2 == targets).float().mean()
                
                # Confidence
                conf1 = F.softmax(logits1[mask], dim=-1).max(dim=-1).values
                conf2 = F.softmax(logits2[mask], dim=-1).max(dim=-1).values
                
                # Entropy
                ent1 = Categorical(logits=logits1[mask]).entropy()
                ent2 = Categorical(logits=logits2[mask]).entropy()
                
                # Perplexity
                loss1 = F.cross_entropy(logits1[mask], targets, reduction='mean')
                loss2 = F.cross_entropy(logits2[mask], targets, reduction='mean')
                perp1 = torch.exp(loss1)
                perp2 = torch.exp(loss2)
                
                # Cosine drift
                h1_flat = hidden1[mask].reshape(-1, hidden1.size(-1))
                h2_flat = hidden2[mask].reshape(-1, hidden2.size(-1))
                drift = 1 - F.cosine_similarity(h1_flat, h2_flat, dim=-1).mean()
                
                # Identify where refinement helped
                improvement_mask = (conf2 - conf1) > 0.05
                
                # Triggers for improvement
                low_conf_trigger = ((conf1 < 0.7) & improvement_mask).float().sum() / max(improvement_mask.sum(), 1)
                high_ent_trigger = ((ent1 > 2.0) & improvement_mask).float().sum() / max(improvement_mask.sum(), 1)
                
                return {
                    'acc1': acc1.item(),
                    'acc2': acc2.item(),
                    'improvement': (acc2 - acc1).item(),
                    'conf1': conf1.mean().item(),
                    'conf2': conf2.mean().item(),
                    'conf_gain': (conf2 - conf1).mean().item(),
                    'ent1': ent1.mean().item(),
                    'ent2': ent2.mean().item(),
                    'perp1': perp1.item(),
                    'perp2': perp2.item(),
                    'drift': drift.item(),
                    'low_conf_trigger': low_conf_trigger.item(),
                    'high_ent_trigger': high_ent_trigger.item(),
                    'improvement_rate': improvement_mask.float().mean().item()
                }
            else:
                return None
    
    def validate_on_dataset(self, dataloader, num_batches=100):
        """Run validation analysis"""
        all_metrics = []
        
        print(f"Running validation on {num_batches} batches...")
        for i, batch in enumerate(tqdm(dataloader)):
            if i >= num_batches:
                break
                
            input_ids = batch['input_ids'].cuda()
            metrics = self.analyze_batch(input_ids)
            if metrics:
                all_metrics.append(metrics)
        
        # Aggregate results
        aggregated = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            aggregated[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        return aggregated, all_metrics
    
    def analyze_uncertainty_patterns(self, all_metrics):
        """Analyze what triggers successful refinement"""
        improvements = [m['improvement'] for m in all_metrics]
        conf_gains = [m['conf_gain'] for m in all_metrics]
        
        # Split into successful vs unsuccessful refinements
        successful = [m for m in all_metrics if m['improvement'] > 0.01]
        unsuccessful = [m for m in all_metrics if m['improvement'] <= 0.01]
        
        analysis = {
            'successful_refinement_rate': len(successful) / len(all_metrics),
            'avg_improvement_when_successful': np.mean([m['improvement'] for m in successful]) if successful else 0,
            'avg_conf1_successful': np.mean([m['conf1'] for m in successful]) if successful else 0,
            'avg_conf1_unsuccessful': np.mean([m['conf1'] for m in unsuccessful]) if unsuccessful else 0,
            'avg_ent1_successful': np.mean([m['ent1'] for m in successful]) if successful else 0,
            'avg_ent1_unsuccessful': np.mean([m['ent1'] for m in unsuccessful]) if unsuccessful else 0,
        }
        
        return analysis
    
    def plot_results(self, all_metrics, save_path='validation_analysis.png'):
        """Create visualization of validation results"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot 1: Improvement distribution
        improvements = [m['improvement'] * 100 for m in all_metrics]
        axes[0, 0].hist(improvements, bins=30, edgecolor='black')
        axes[0, 0].axvline(0, color='red', linestyle='--', label='No improvement')
        axes[0, 0].set_xlabel('Improvement (%)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Stage 2 Improvement Distribution')
        axes[0, 0].legend()
        
        # Plot 2: Confidence vs Improvement
        conf1_vals = [m['conf1'] for m in all_metrics]
        axes[0, 1].scatter(conf1_vals, improvements, alpha=0.5)
        axes[0, 1].set_xlabel('Stage 1 Confidence')
        axes[0, 1].set_ylabel('Improvement (%)')
        axes[0, 1].set_title('Low Confidence → More Improvement?')
        axes[0, 1].axhline(0, color='red', linestyle='--', alpha=0.5)
        
        # Plot 3: Entropy vs Improvement
        ent1_vals = [m['ent1'] for m in all_metrics]
        axes[0, 2].scatter(ent1_vals, improvements, alpha=0.5)
        axes[0, 2].set_xlabel('Stage 1 Entropy')
        axes[0, 2].set_ylabel('Improvement (%)')
        axes[0, 2].set_title('High Entropy → More Improvement?')
        axes[0, 2].axhline(0, color='red', linestyle='--', alpha=0.5)
        
        # Plot 4: Perplexity reduction
        perp_reductions = [(m['perp1'] - m['perp2']) / m['perp1'] * 100 for m in all_metrics]
        axes[1, 0].hist(perp_reductions, bins=30, edgecolor='black')
        axes[1, 0].set_xlabel('Perplexity Reduction (%)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Perplexity Improvement')
        
        # Plot 5: Drift vs Improvement
        drifts = [m['drift'] for m in all_metrics]
        axes[1, 1].scatter(drifts, improvements, alpha=0.5)
        axes[1, 1].set_xlabel('Cosine Drift')
        axes[1, 1].set_ylabel('Improvement (%)')
        axes[1, 1].set_title('Representation Change vs Improvement')
        
        # Plot 6: Success rate over time
        window = 10
        success_rate = []
        for i in range(0, len(improvements) - window):
            window_success = sum(1 for x in improvements[i:i+window] if x > 1) / window
            success_rate.append(window_success * 100)
        axes[1, 2].plot(success_rate)
        axes[1, 2].set_xlabel('Batch Number')
        axes[1, 2].set_ylabel('Success Rate (%)')
        axes[1, 2].set_title(f'Rolling Success Rate (>{1}% improvement)')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        return fig

def main():
    # Setup
    print("=" * 60)
    print("RECURSIVE TRAINING VALIDATION ANALYSIS")
    print("=" * 60)
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load validation dataset
    print("\nLoading WikiText-103 validation set...")
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='validation')
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'], 
            truncation=True, 
            padding='max_length', 
            max_length=512
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format('torch', columns=['input_ids'])
    
    val_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=4,
        shuffle=False
    )
    
    # Initialize validator
    validator = RecursiveValidator('recursive_v0_step_5000.pt', tokenizer)
    
    # Run validation
    aggregated, all_metrics = validator.validate_on_dataset(val_dataloader, num_batches=200)
    
    # Analyze patterns
    patterns = validator.analyze_uncertainty_patterns(all_metrics)
    
    # Print results
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    
    print("\n📊 Core Metrics:")
    print(f"  Accuracy Stage 1:  {aggregated['acc1']['mean']:.1%} ± {aggregated['acc1']['std']:.1%}")
    print(f"  Accuracy Stage 2:  {aggregated['acc2']['mean']:.1%} ± {aggregated['acc2']['std']:.1%}")
    print(f"  Improvement:       {aggregated['improvement']['mean']:.2%} ± {aggregated['improvement']['std']:.2%}")
    print(f"  Max Improvement:   {aggregated['improvement']['max']:.2%}")
    
    print("\n🎯 Refinement Triggers:")
    print(f"  Low confidence trigger rate: {aggregated['low_conf_trigger']['mean']:.1%}")
    print(f"  High entropy trigger rate:   {aggregated['high_ent_trigger']['mean']:.1%}")
    print(f"  Overall improvement rate:    {aggregated['improvement_rate']['mean']:.1%}")
    
    print("\n🔬 Uncertainty Analysis:")
    print(f"  Successful refinement rate:        {patterns['successful_refinement_rate']:.1%}")
    print(f"  Avg improvement when successful:   {patterns['avg_improvement_when_successful']:.2%}")
    print(f"  Conf (successful):   {patterns['avg_conf1_successful']:.3f}")
    print(f"  Conf (unsuccessful): {patterns['avg_conf1_unsuccessful']:.3f}")
    print(f"  Entropy (successful):   {patterns['avg_ent1_successful']:.2f}")
    print(f"  Entropy (unsuccessful): {patterns['avg_ent1_unsuccessful']:.2f}")
    
    print("\n📈 Other Metrics:")
    print(f"  Cosine Drift:      {aggregated['drift']['mean']:.3f} ± {aggregated['drift']['std']:.3f}")
    print(f"  Perplexity Stage 1: {aggregated['perp1']['mean']:.1f}")
    print(f"  Perplexity Stage 2: {aggregated['perp2']['mean']:.1f}")
    print(f"  Perp Reduction:     {(aggregated['perp1']['mean'] - aggregated['perp2']['mean']) / aggregated['perp1']['mean'] * 100:.1f}%")
    
    # Critical interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    
    if aggregated['improvement']['mean'] > 0.01:
        print("✅ SUCCESS: Validation shows consistent improvement!")
        print(f"   Average {aggregated['improvement']['mean']:.2%} improvement generalizes to unseen data")
    elif aggregated['improvement']['mean'] > 0.005:
        print("⚠️  PARTIAL SUCCESS: Small but positive improvement on validation")
        print("   Consider testing on harder tasks where refinement matters more")
    else:
        print("❌ CONCERN: Limited improvement on validation set")
        print("   May need different masking strategy or task distribution")
    
    if patterns['avg_conf1_successful'] < patterns['avg_conf1_unsuccessful']:
        print("\n🎯 KEY FINDING: Model learns uncertainty-aware refinement!")
        print("   Refinement helps more on low-confidence predictions")
    
    if patterns['avg_ent1_successful'] > patterns['avg_ent1_unsuccessful']:
        print("\n🎯 KEY FINDING: High entropy triggers successful refinement!")
        print("   Model identifies ambiguous contexts for extra computation")
    
    # Create visualizations
    print("\n📊 Generating plots...")
    fig = validator.plot_results(all_metrics)
    
    # Save detailed results
    with open('validation_results.json', 'w') as f:
        json.dump({
            'aggregated': aggregated,
            'patterns': patterns,
            'sample_metrics': all_metrics[:10]  # Save sample for inspection
        }, f, indent=2)
    
    print("\n✅ Analysis complete! Results saved to:")
    print("   - validation_analysis.png (plots)")
    print("   - validation_results.json (detailed metrics)")
    
    return aggregated, patterns, all_metrics

if __name__ == "__main__":
    results = main()
