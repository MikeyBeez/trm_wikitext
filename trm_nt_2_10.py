"""
Recursive Refinement Experiment: Comparing Next-Token Prediction Strategies
===========================================================================
Testing whether comparing multiple candidate tokens with recursive refinement
improves language modeling perplexity on WikiText-2.

Three approaches:
1. Baseline: Standard single-token prediction
2. Binary Refinement: Compare top 2 candidates
3. Multi Refinement: Compare top 10 candidates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import numpy as np
import time
import math
from dataclasses import dataclass
from typing import Tuple, Optional
import json

# ===============================
# CONFIGURATION
# ===============================
@dataclass
class Config:
    # Model architecture
    vocabulary_size: int = 10000
    embedding_dimension: int = 256
    hidden_dimension: int = 256
    number_of_layers: int = 4
    number_of_heads: int = 8
    dropout_rate: float = 0.1
    context_length: int = 128
    
    # Refinement settings
    number_of_refinements: int = 3
    
    # Training settings
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    maximum_steps: int = 10000
    warmup_steps: int = 500
    
    # Evaluation settings
    evaluation_interval: int = 250
    evaluation_batches: int = 50
    patience_limit: int = 10
    
    # System settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    random_seed: int = 42
    
# ===============================
# DATASET
# ===============================
class WikiTextDataset(Dataset):
    """Clean WikiText-2 dataset handler"""
    
    def __init__(self, tokens, context_length):
        self.tokens = tokens
        self.context_length = context_length
        
    def __len__(self):
        return len(self.tokens) - self.context_length - 1
    
    def __getitem__(self, idx):
        context = self.tokens[idx:idx + self.context_length]
        target = self.tokens[idx + 1:idx + self.context_length + 1]
        return (
            torch.tensor(context, dtype=torch.long),
            torch.tensor(target, dtype=torch.long)
        )

# ===============================
# MODEL COMPONENTS
# ===============================
class TransformerBlock(nn.Module):
    """Single transformer block with self-attention and feedforward"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config.embedding_dimension,
            config.number_of_heads,
            dropout=config.dropout_rate,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(config.embedding_dimension)
        self.norm2 = nn.LayerNorm(config.embedding_dimension)
        
        self.feedforward = nn.Sequential(
            nn.Linear(config.embedding_dimension, config.hidden_dimension * 4),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dimension * 4, config.embedding_dimension),
            nn.Dropout(config.dropout_rate)
        )
        
    def forward(self, x, attention_mask=None):
        # Self-attention with residual
        normalized = self.norm1(x)
        attention_output, _ = self.attention(
            normalized, normalized, normalized,
            attn_mask=attention_mask,
            need_weights=False
        )
        x = x + attention_output
        
        # Feedforward with residual
        x = x + self.feedforward(self.norm2(x))
        return x

class BaselineModel(nn.Module):
    """Standard autoregressive transformer - no refinement"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embedding = nn.Embedding(config.vocabulary_size, config.embedding_dimension)
        self.position_embedding = nn.Embedding(config.context_length, config.embedding_dimension)
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.number_of_layers)
        ])
        
        # Output projection
        self.norm_final = nn.LayerNorm(config.embedding_dimension)
        self.output_projection = nn.Linear(config.embedding_dimension, config.vocabulary_size)
        
        # Causal mask
        self.register_buffer("causal_mask", 
            torch.triu(torch.ones(config.context_length, config.context_length) * float('-inf'), 
                      diagonal=1))
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_tokens):
        batch_size, sequence_length = input_tokens.shape
        
        # Embeddings
        token_embeddings = self.token_embedding(input_tokens)
        positions = torch.arange(sequence_length, device=input_tokens.device)
        position_embeddings = self.position_embedding(positions)
        
        x = self.dropout(token_embeddings + position_embeddings)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, self.causal_mask[:sequence_length, :sequence_length])
        
        # Output
        x = self.norm_final(x)
        logits = self.output_projection(x)
        
        return logits

class RefinementModel(nn.Module):
    """Transformer with candidate refinement"""
    
    def __init__(self, config: Config, number_of_candidates: int):
        super().__init__()
        self.config = config
        self.number_of_candidates = number_of_candidates
        
        # Base model components (shared with baseline)
        self.token_embedding = nn.Embedding(config.vocabulary_size, config.embedding_dimension)
        self.position_embedding = nn.Embedding(config.context_length, config.embedding_dimension)
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Context encoding blocks
        self.context_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.number_of_layers)
        ])
        
        # Refinement blocks for comparing candidates
        self.refinement_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.number_of_refinements)
        ])
        
        # Output projections
        self.norm_final = nn.LayerNorm(config.embedding_dimension)
        self.output_projection = nn.Linear(config.embedding_dimension, config.vocabulary_size)
        self.candidate_scorer = nn.Linear(config.embedding_dimension, 1)
        
        # Causal mask for context
        self.register_buffer("causal_mask",
            torch.triu(torch.ones(config.context_length, config.context_length) * float('-inf'),
                      diagonal=1))
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_tokens):
        batch_size, sequence_length = input_tokens.shape
        
        # Encode context (same as baseline)
        token_embeddings = self.token_embedding(input_tokens)
        positions = torch.arange(sequence_length, device=input_tokens.device)
        position_embeddings = self.position_embedding(positions)
        
        context = self.dropout(token_embeddings + position_embeddings)
        
        for block in self.context_blocks:
            context = block(context, self.causal_mask[:sequence_length, :sequence_length])
        
        context = self.norm_final(context)
        
        # Get initial logits
        logits = self.output_projection(context)
        
        # For each position, refine top-k candidates
        refined_logits = torch.zeros_like(logits)
        
        for position in range(sequence_length):
            position_logits = logits[:, position, :]  # [batch_size, vocabulary_size]
            
            # Get top-k candidates
            top_values, top_indices = torch.topk(position_logits, self.number_of_candidates, dim=-1)
            
            # Embed candidates
            candidate_embeddings = self.token_embedding(top_indices)  # [batch_size, k, embedding_dim]
            
            # Add positional information
            position_embed = self.position_embedding(
                torch.tensor([position], device=input_tokens.device)
            ).unsqueeze(0).expand(batch_size, self.number_of_candidates, -1)
            
            candidates = candidate_embeddings + position_embed
            
            # Refine candidates by letting them attend to each other
            for refinement_block in self.refinement_blocks:
                # No mask - candidates can fully attend to each other
                candidates = refinement_block(candidates, attention_mask=None)
            
            # Score refined candidates
            refined_scores = self.candidate_scorer(candidates).squeeze(-1)  # [batch_size, k]
            
            # Combine with initial scores
            combined_scores = top_values + refined_scores
            
            # Put refined scores back into full vocabulary tensor
            refined_logits[:, position, :].scatter_(
                1, top_indices, combined_scores
            )
            
            # Copy over non-candidate logits
            mask = torch.ones_like(position_logits, dtype=torch.bool)
            mask.scatter_(1, top_indices, False)
            refined_logits[:, position, :][mask] = position_logits[mask]
        
        return refined_logits

# ===============================
# TRAINING AND EVALUATION
# ===============================
def create_datasets(config: Config):
    """Load and prepare WikiText-2 datasets"""
    
    print("\n" + "="*80)
    print("LOADING WIKITEXT-2 DATASET")
    print("="*80)
    
    # Load dataset
    dataset = load_dataset('wikitext', 'wikitext-2-v1')
    
    # Simple vocabulary (top 10k words)
    all_text = ' '.join(dataset['train']['text'])
    words = all_text.split()
    
    from collections import Counter
    word_counts = Counter(words)
    vocab = ['<pad>', '<unk>'] + [word for word, _ in word_counts.most_common(config.vocabulary_size - 2)]
    word_to_id = {word: i for i, word in enumerate(vocab)}
    
    def tokenize(text):
        return [word_to_id.get(word, word_to_id['<unk>']) for word in text.split()]
    
    # Tokenize datasets
    train_tokens = []
    for text in dataset['train']['text']:
        if text:  # Skip empty lines
            train_tokens.extend(tokenize(text))
    
    val_tokens = []
    for text in dataset['validation']['text']:
        if text:
            val_tokens.extend(tokenize(text))
    
    test_tokens = []
    for text in dataset['test']['text']:
        if text:
            test_tokens.extend(tokenize(text))
    
    # Create datasets
    train_dataset = WikiTextDataset(train_tokens, config.context_length)
    val_dataset = WikiTextDataset(val_tokens, config.context_length)
    test_dataset = WikiTextDataset(test_tokens, config.context_length)
    
    print(f"Training samples:   {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")
    print(f"Test samples:       {len(test_dataset):,}")
    
    return train_dataset, val_dataset, test_dataset

@torch.no_grad()
def evaluate(model, dataset, config: Config):
    """Evaluate model on dataset"""
    model.eval()
    losses = []
    
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    
    for i, (context, target) in enumerate(loader):
        if i >= config.evaluation_batches:
            break
            
        context = context.to(config.device)
        target = target.to(config.device)
        
        logits = model(context)
        loss = F.cross_entropy(
            logits.reshape(-1, config.vocabulary_size),
            target.reshape(-1)
        )
        losses.append(loss.item())
    
    model.train()
    return np.mean(losses)

def train_model(model, train_dataset, val_dataset, config: Config, model_name: str):
    """Train a model and return results"""
    
    print("\n" + "="*80)
    print(f"TRAINING {model_name}")
    print("="*80)
    
    # Count parameters
    total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_parameters:,}")
    
    # Setup training
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Learning rate schedule
    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / config.warmup_steps
        return 1.0
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        shuffle=True
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    step = 0
    training_losses = []
    
    start_time = time.time()
    
    for epoch in range(100):  # Max epochs
        epoch_losses = []
        
        for context, target in train_loader:
            if step >= config.maximum_steps:
                break
                
            context = context.to(config.device)
            target = target.to(config.device)
            
            # Forward pass
            logits = model(context)
            loss = F.cross_entropy(
                logits.reshape(-1, config.vocabulary_size),
                target.reshape(-1)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            scheduler.step()
            
            epoch_losses.append(loss.item())
            step += 1
            
            # Evaluation
            if step % config.evaluation_interval == 0:
                train_loss = np.mean(epoch_losses[-100:]) if len(epoch_losses) >= 100 else np.mean(epoch_losses)
                val_loss = evaluate(model, val_dataset, config)
                val_perplexity = math.exp(min(val_loss, 10))
                
                elapsed_minutes = (time.time() - start_time) / 60
                
                print(f"\nStep {step:,} | Time: {elapsed_minutes:.1f} minutes")
                print(f"  Training Loss:      {train_loss:.4f}")
                print(f"  Validation Loss:    {val_loss:.4f}")
                print(f"  Validation Perplexity: {val_perplexity:.2f}")
                
                training_losses.append({
                    'step': step,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_perplexity': val_perplexity
                })
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    print(f"  ✓ New best validation loss!")
                else:
                    patience_counter += 1
                    print(f"  Patience: {patience_counter}/{config.patience_limit}")
                    
                    if patience_counter >= config.patience_limit:
                        print("\nEarly stopping triggered!")
                        break
        
        if patience_counter >= config.patience_limit or step >= config.maximum_steps:
            break
    
    # Final results
    final_val_loss = evaluate(model, val_dataset, config)
    final_val_perplexity = math.exp(min(final_val_loss, 10))
    
    total_time = (time.time() - start_time) / 60
    
    return {
        'model_name': model_name,
        'parameters': total_parameters,
        'best_validation_loss': best_val_loss,
        'best_validation_perplexity': math.exp(min(best_val_loss, 10)),
        'final_validation_loss': final_val_loss,
        'final_validation_perplexity': final_val_perplexity,
        'total_steps': step,
        'training_time_minutes': total_time,
        'training_history': training_losses
    }

def main():
    """Run the complete experiment"""
    
    # Configuration
    config = Config()
    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)
    
    print("\n" + "="*80)
    print("RECURSIVE REFINEMENT EXPERIMENT")
    print("="*80)
    print(f"Device: {config.device}")
    print(f"Context length: {config.context_length} tokens")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(config)
    
    # Train baseline model
    baseline_model = BaselineModel(config).to(config.device)
    baseline_results = train_model(
        baseline_model, 
        train_dataset, 
        val_dataset, 
        config,
        "BASELINE MODEL (No Refinement)"
    )
    
    # Train 2-candidate refinement model
    refinement_2_model = RefinementModel(config, number_of_candidates=2).to(config.device)
    refinement_2_results = train_model(
        refinement_2_model,
        train_dataset,
        val_dataset,
        config,
        "REFINEMENT MODEL (2 Candidates)"
    )
    
    # Train 10-candidate refinement model
    refinement_10_model = RefinementModel(config, number_of_candidates=10).to(config.device)
    refinement_10_results = train_model(
        refinement_10_model,
        train_dataset,
        val_dataset,
        config,
        "REFINEMENT MODEL (10 Candidates)"
    )
    
    # Test evaluation
    print("\n" + "="*80)
    print("TEST SET EVALUATION")
    print("="*80)
    
    baseline_test_loss = evaluate(baseline_model, test_dataset, config)
    baseline_test_perplexity = math.exp(min(baseline_test_loss, 10))
    
    refinement_2_test_loss = evaluate(refinement_2_model, test_dataset, config)
    refinement_2_test_perplexity = math.exp(min(refinement_2_test_loss, 10))
    
    refinement_10_test_loss = evaluate(refinement_10_model, test_dataset, config)
    refinement_10_test_perplexity = math.exp(min(refinement_10_test_loss, 10))
    
    # Pretty print final results
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│                        VALIDATION RESULTS                       │")
    print("├─────────────────────────────────────────────────────────────────┤")
    print(f"│ Baseline (No Refinement):                                      │")
    print(f"│   Best Loss:       {baseline_results['best_validation_loss']:.4f}                                         │")
    print(f"│   Best Perplexity: {baseline_results['best_validation_perplexity']:.2f}                                         │")
    print(f"│   Parameters:      {baseline_results['parameters']:,}                                     │")
    print(f"│                                                                 │")
    print(f"│ Refinement with 2 Candidates:                                  │")
    print(f"│   Best Loss:       {refinement_2_results['best_validation_loss']:.4f}                                         │")
    print(f"│   Best Perplexity: {refinement_2_results['best_validation_perplexity']:.2f}                                         │")
    print(f"│   Parameters:      {refinement_2_results['parameters']:,}                                     │")
    print(f"│   Improvement:     {((baseline_results['best_validation_perplexity'] - refinement_2_results['best_validation_perplexity']) / baseline_results['best_validation_perplexity'] * 100):.1f}%                                          │")
    print(f"│                                                                 │")
    print(f"│ Refinement with 10 Candidates:                                 │")
    print(f"│   Best Loss:       {refinement_10_results['best_validation_loss']:.4f}                                         │")
    print(f"│   Best Perplexity: {refinement_10_results['best_validation_perplexity']:.2f}                                         │")
    print(f"│   Parameters:      {refinement_10_results['parameters']:,}                                     │")
    print(f"│   Improvement:     {((baseline_results['best_validation_perplexity'] - refinement_10_results['best_validation_perplexity']) / baseline_results['best_validation_perplexity'] * 100):.1f}%                                          │")
    print("└─────────────────────────────────────────────────────────────────┘")
    
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│                          TEST RESULTS                           │")
    print("├─────────────────────────────────────────────────────────────────┤")
    print(f"│ Baseline (No Refinement):                                      │")
    print(f"│   Test Loss:       {baseline_test_loss:.4f}                                         │")
    print(f"│   Test Perplexity: {baseline_test_perplexity:.2f}                                          │")
    print(f"│                                                                 │")
    print(f"│ Refinement with 2 Candidates:                                  │")
    print(f"│   Test Loss:       {refinement_2_test_loss:.4f}                                         │")
    print(f"│   Test Perplexity: {refinement_2_test_perplexity:.2f}                                          │")
    print(f"│   Improvement:     {((baseline_test_perplexity - refinement_2_test_perplexity) / baseline_test_perplexity * 100):.1f}%                                          │")
    print(f"│                                                                 │")
    print(f"│ Refinement with 10 Candidates:                                 │")
    print(f"│   Test Loss:       {refinement_10_test_loss:.4f}                                         │")
    print(f"│   Test Perplexity: {refinement_10_test_perplexity:.2f}                                          │")
    print(f"│   Improvement:     {((baseline_test_perplexity - refinement_10_test_perplexity) / baseline_test_perplexity * 100):.1f}%                                          │")
    print("└─────────────────────────────────────────────────────────────────┘")
    
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│                         CONCLUSIONS                             │")
    print("├─────────────────────────────────────────────────────────────────┤")
    
    if refinement_2_test_perplexity < baseline_test_perplexity and refinement_10_test_perplexity < baseline_test_perplexity:
        print("│ ✓ Refinement helps! Both variants beat baseline.               │")
        if refinement_10_test_perplexity < refinement_2_test_perplexity:
            print("│ ✓ More candidates (10) provides better results than fewer (2). │")
        else:
            print("│ ✓ Fewer candidates (2) is more efficient than more (10).      │")
    elif refinement_2_test_perplexity < baseline_test_perplexity or refinement_10_test_perplexity < baseline_test_perplexity:
        print("│ ◐ Mixed results: Some refinement helps, but not consistently.  │")
    else:
        print("│ ✗ Refinement did not improve results in this experiment.       │")
    
    print("└─────────────────────────────────────────────────────────────────┘")
    
    # Save detailed results
    all_results = {
        'baseline': baseline_results,
        'refinement_2': refinement_2_results,
        'refinement_10': refinement_10_results,
        'test_results': {
            'baseline_test_loss': baseline_test_loss,
            'baseline_test_perplexity': baseline_test_perplexity,
            'refinement_2_test_loss': refinement_2_test_loss,
            'refinement_2_test_perplexity': refinement_2_test_perplexity,
            'refinement_10_test_loss': refinement_10_test_loss,
            'refinement_10_test_perplexity': refinement_10_test_perplexity
        }
    }
    
    with open('refinement_experiment_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print("Results saved to: refinement_experiment_results.json")

if __name__ == "__main__":
    main()
