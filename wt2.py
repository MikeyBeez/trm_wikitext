"""
Tiny Recursive Model for WikiText-2 Word-Level Language Modeling
A transformer-based architecture with iterative refinement for word prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from dataclasses import dataclass
import math
import json
import os
import time
from datetime import datetime
from collections import Counter
from torch.utils.checkpoint import checkpoint
from typing import List, Tuple, Optional

from datasets import load_dataset
import numpy as np

@dataclass
class Configuration:
    """Model and training configuration"""
    context_size: int = 32          # Number of context words
    chunk_size: int = 2             # Number of words to predict per step
    embed_dim: int = 128            # Embedding dimension
    n_context_layers: int = 2       # Transformer layers for context encoding
    n_refine_layers: int = 2        # Transformer layers for refinement
    n_heads: int = 4                # Number of attention heads
    dropout: float = 0.1            # Dropout rate
    n_refinements: int = 2          # Refinement iterations per forward pass
    n_recursions: int = 3           # Recursive passes within each refinement
    
    batch_size: int = 32            # Training batch size
    max_epochs: int = 50            # Maximum training epochs
    learning_rate: float = 1e-4     # Learning rate
    weight_decay: float = 0.01      # Weight decay for regularization
    grad_clip: float = 1.0          # Gradient clipping threshold
    
    eval_interval: int = 200        # Evaluation frequency (steps)
    eval_iterations: int = 50       # Number of batches for evaluation
    patience: int = 20              # Early stopping patience
    
    output_directory: str = "./outputs_wikitext_word"
    checkpoint_directory: str = "./checkpoints_wikitext_word"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    random_seed: int = 42
    maximum_vocabulary_size: int = 30000
    resume_from_checkpoint: bool = True
    gradient_checkpointing: bool = True

class WordTokenizer:
    """Simple word-level tokenizer that splits on whitespace"""
    
    def __init__(self, max_vocab_size: int = 30000):
        self.max_vocab_size = max_vocab_size
        self.word_to_id = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
        self.id_to_word = {0: "<pad>", 1: "<unk>", 2: "<eos>"}
        self.vocabulary_size = 3
        
    def build_vocabulary(self, texts: List[str]):
        """Build vocabulary from training texts"""
        print("Building word vocabulary from training data...")
        
        # Count word frequencies
        word_counts = Counter()
        for text in texts:
            if text and text.strip():
                words = text.split()
                word_counts.update(words)
        
        # Add most common words to vocabulary
        for word, count in word_counts.most_common(self.max_vocab_size - 3):
            if word not in self.word_to_id:
                self.word_to_id[word] = self.vocabulary_size
                self.id_to_word[self.vocabulary_size] = word
                self.vocabulary_size += 1
        
        print(f"Vocabulary size: {self.vocabulary_size:,} unique words")
    
    def encode(self, text: str) -> List[int]:
        """Convert text to list of token IDs"""
        if not text or not text.strip():
            return []
        return [self.word_to_id.get(word, self.word_to_id["<unk>"]) 
                for word in text.split()]
    
    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs to text"""
        words = [self.id_to_word.get(int(tok), "<unk>") for tok in token_ids]
        return " ".join(words)

class ChunkDataset(Dataset):
    """Dataset that provides context windows and target chunks"""
    
    def __init__(self, token_ids: List[int], context_size: int, chunk_size: int):
        self.data = token_ids
        self.context_size = context_size
        self.chunk_size = chunk_size
        self.length = max(0, len(self.data) - self.context_size - self.chunk_size)
        
        if self.length == 0:
            raise ValueError(f"Dataset too small. Required: {context_size + chunk_size + 1} tokens")

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        context = self.data[idx: idx + self.context_size]
        chunk = self.data[idx + self.context_size: idx + self.context_size + self.chunk_size]
        return torch.tensor(context, dtype=torch.long), torch.tensor(chunk, dtype=torch.long)

class TransformerBlock(nn.Module):
    """Transformer block with causal masking"""
    
    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention with residual connection
        attended = self.attention(self.layer_norm1(x), self.layer_norm1(x), self.layer_norm1(x), 
                                 attn_mask=attention_mask, need_weights=False)[0]
        x = x + attended
        
        # Feed-forward with residual connection
        x = x + self.feed_forward(self.layer_norm2(x))
        return x

class TinyRecursiveModel(nn.Module):
    """Recurrently-refined transformer for word-level language modeling"""
    
    def __init__(self, config: Configuration):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocabulary_size, config.embed_dim)
        self.position_embedding = nn.Embedding(config.context_size + config.chunk_size, config.embed_dim)
        
        self.context_blocks = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.n_heads, config.dropout)
            for _ in range(config.n_context_layers)
        ])
        
        self.refinement_blocks = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.n_heads, config.dropout)
            for _ in range(config.n_refine_layers)
        ])
        
        self.layer_norm_final = nn.LayerNorm(config.embed_dim)
        self.output_projection = nn.Linear(config.embed_dim, config.vocabulary_size, bias=False)
        self.output_projection.weight = self.embedding.weight
        
        # Causal masks
        self.register_buffer("context_mask",
            torch.triu(torch.full((config.context_size, config.context_size), -1e9), diagonal=1))
        
        total_length = config.context_size + config.chunk_size
        combined_mask = torch.triu(torch.full((total_length, total_length), -1e9), diagonal=1)
        combined_mask[config.context_size:, :config.context_size] = 0
        self.register_buffer("combined_mask", combined_mask)
        
        self.apply(self._initialize_weights)
        
    def _initialize_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, context: torch.Tensor, chunk: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with proper teacher forcing
        
        Args:
            context: [batch, context_size] - Context words
            chunk: [batch, chunk_size] - Ground truth chunk (training) or None (generation)
        """
        batch_size = context.shape[0]
        
        # Process context
        context_embeddings = self.embedding(context)
        context_positions = self.position_embedding(torch.arange(self.config.context_size, device=context.device))
        context_hidden = context_embeddings + context_positions
        
        for block in self.context_blocks:
            context_hidden = block(context_hidden, self.context_mask)
        
        # Initialize chunk representation
        if chunk is not None:
            # Training: Use ground truth for teacher forcing
            chunk_embeddings = self.embedding(chunk)
        else:
            # Generation: Start with <eos> token
            eos_token_id = 2
            start_tokens = torch.full((batch_size, self.config.chunk_size), eos_token_id, device=context.device)
            chunk_embeddings = self.embedding(start_tokens)
        
        chunk_positions = self.position_embedding(
            torch.arange(self.config.context_size, 
                        self.config.context_size + self.config.chunk_size,
                        device=context.device)
        )
        
        y = chunk_embeddings + chunk_positions
        z = torch.zeros_like(y)
        
        # Recursive refinement
        for refinement_step in range(self.config.n_refinements):
            if refinement_step < self.config.n_refinements - 1:
                with torch.no_grad():
                    y, z = self._refine_once(context_hidden, y, z)
            else:
                y, z = self._refine_once(context_hidden, y, z)
        
        y = self.layer_norm_final(y)
        logits = self.output_projection(y)
        return logits
    
    def _refine_once(self, context_hidden: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single refinement iteration"""
        for _ in range(self.config.n_recursions):
            combined = torch.cat([context_hidden, y + z], dim=1)
            
            for block in self.refinement_blocks:
                if self.config.gradient_checkpointing and self.training:
                    combined = checkpoint(block, combined, self.combined_mask, use_reentrant=False)
                else:
                    combined = block(combined, self.combined_mask)
            
            z = combined[:, self.config.context_size:, :]
        
        combined = torch.cat([context_hidden, y + z], dim=1)
        for block in self.refinement_blocks:
            if self.config.gradient_checkpointing and self.training:
                combined = checkpoint(block, combined, self.combined_mask, use_reentrant=False)
            else:
                combined = block(combined, self.combined_mask)
        y = combined[:, self.config.context_size:, :]
        
        return y, z

def get_batch(dataset: Dataset, batch_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch from the dataset"""
    indices = torch.randint(len(dataset), (batch_size,))
    contexts, chunks = [], []
    for idx in indices:
        context, chunk = dataset[idx]
        contexts.append(context)
        chunks.append(chunk)
    return torch.stack(contexts).to(device), torch.stack(chunks).to(device)

@torch.no_grad()
def estimate_loss(model: nn.Module, dataset: Dataset, config: Configuration) -> float:
    """Estimate loss over multiple batches"""
    model.eval()
    losses = []
    
    for _ in range(config.eval_iterations):
        context, target_chunk = get_batch(dataset, config.batch_size, config.device)
        logits = model(context, chunk=None)
        loss = F.cross_entropy(logits.reshape(-1, config.vocabulary_size), target_chunk.reshape(-1))
        losses.append(loss.item())
    
    model.train()
    return np.mean(losses)

@torch.no_grad()
def generate_text(model: nn.Module, tokenizer: WordTokenizer, 
                  prompt: str, config: Configuration, 
                  max_new_words: int = 20, temperature: float = 0.8, 
                  top_k: int = 40) -> None:
    """Generate text word-by-word"""
    model.eval()
    print(f"\nGeneration prompt: '{prompt}'")
    print("Generated text: ", end="", flush=True)
    
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    if len(tokens) > config.context_size:
        tokens = tokens[-config.context_size:]
    
    generated = list(tokens)
    previous_chunk = None
    
    num_chunks = max_new_words // config.chunk_size
    for _ in range(num_chunks):
        context_tensor = torch.tensor([generated[-config.context_size:]], 
                                    dtype=torch.long, device=config.device)
        
        logits = model(context_tensor, chunk=previous_chunk)
        logits = logits / temperature
        
        predicted_tokens = []
        for position in range(config.chunk_size):
            position_logits = logits[0, position]
            
            if top_k > 0:
                indices_to_remove = position_logits < torch.topk(position_logits, top_k)[0][..., -1, None]
                position_logits[indices_to_remove] = -float('inf')
            
            probabilities = F.softmax(position_logits, dim=-1)
            next_token = torch.multinomial(probabilities, 1)
            predicted_tokens.append(next_token.item())
        
        previous_chunk = torch.tensor([predicted_tokens], dtype=torch.long, device=config.device)
        generated.extend(predicted_tokens)
        
        new_text = tokenizer.decode(predicted_tokens)
        print(new_text, end="", flush=True)
    
    print("\n")
    model.train()

def count_model_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_model_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                         step: int, epoch: int, best_validation_loss: float,
                         config: Configuration, path: str) -> None:
    """Save training checkpoint"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': int(step),
        'epoch': int(epoch),
        'best_validation_loss': float(best_validation_loss),
        'configuration': {k: v if not isinstance(v, (np.ndarray, np.generic)) else v.item() 
                         for k, v in vars(config).items()},
    }, path)

def load_model_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer,
                         config: Configuration) -> Tuple[int, int, float]:
    """Load training checkpoint"""
    try:
        import numpy as np
        torch.serialization.add_safe_globals([np._NoValue])
        
        checkpoint = torch.load(path, map_location=config.device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        step = checkpoint['step']
        epoch = checkpoint['epoch']
        best_validation_loss = checkpoint['best_validation_loss']
        print(f"Resumed from step {step:,}, epoch {epoch}, best validation loss: {best_validation_loss:.4f}")
        return step, epoch, best_validation_loss
    except Exception as error:
        print(f"Failed to load checkpoint: {error}")
        return 0, 0, float('inf')

def main():
    configuration = Configuration()
    torch.manual_seed(configuration.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(configuration.random_seed)
    
    os.makedirs(configuration.output_directory, exist_ok=True)
    os.makedirs(configuration.checkpoint_directory, exist_ok=True)
    
    print("=" * 80)
    print("Tiny Recursive Model: WikiText-2 Word-Level Language Modeling")
    print("=" * 80)
    print(f"Computing device: {configuration.device}")
    print(f"Model architecture:")
    print(f"  Context encoding layers: {configuration.n_context_layers}")
    print(f"  Refinement layers: {configuration.n_refine_layers}")
    print(f"  Attention heads: {configuration.n_heads}")
    print(f"  Refinement iterations: {configuration.n_refinements}")
    print(f"  Recursive passes per iteration: {configuration.n_recursions}")
    print("=" * 80)
    
    # Load datasets
    print("\nLoading WikiText-2 dataset...")
    train_data = load_dataset("wikitext", "wikitext-2-v1", split="train")
    validation_data = load_dataset("wikitext", "wikitext-2-v1", split="validation")
    test_data = load_dataset("wikitext", "wikitext-2-v1", split="test")

    # Create word-level tokenizer
    tokenizer = WordTokenizer(max_vocab_size=configuration.maximum_vocabulary_size)
    
    # Extract non-empty texts
    train_texts = [text for text in train_data["text"] if text and text.strip()]
    validation_texts = [text for text in validation_data["text"] if text and text.strip()]
    test_texts = [text for text in test_data["text"] if text and text.strip()]
    
    # Build vocabulary from training data
    tokenizer.build_vocabulary(train_texts)
    configuration.vocabulary_size = tokenizer.vocabulary_size

    # Tokenize all texts with document boundaries
    def tokenize_texts(texts: List[str]) -> List[int]:
        tokenized = []
        for text in texts:
            if text and text.strip():
                tokens = tokenizer.encode(text)
                tokenized.extend(tokens + [tokenizer.word_to_id["<eos>"]])
        return tokenized

    train_tokens = tokenize_texts(train_texts)
    validation_tokens = tokenize_texts(validation_texts)
    test_tokens = tokenize_texts(test_texts)

    # Create datasets
    train_dataset = ChunkDataset(train_tokens, configuration.context_size, configuration.chunk_size)
    validation_dataset = ChunkDataset(validation_tokens, configuration.context_size, configuration.chunk_size)
    test_dataset = ChunkDataset(test_tokens, configuration.context_size, configuration.chunk_size)

    steps_per_epoch = len(train_dataset) // configuration.batch_size
    print(f"Training tokens: {len(train_tokens):,} | Steps per epoch: {steps_per_epoch:,}")

    # Initialize model
    model = TinyRecursiveModel(configuration).to(configuration.device)
    total_parameters = count_model_parameters(model)
    print(f"Model parameters: {total_parameters:,}")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=configuration.learning_rate,
        weight_decay=configuration.weight_decay
    )

    # Resume from checkpoint if available
    checkpoint_path = os.path.join(configuration.checkpoint_directory, "latest_checkpoint.pt")
    start_step, start_epoch, best_validation_loss = 0, 0, float('inf')
    
    if configuration.resume_from_checkpoint and os.path.exists(checkpoint_path):
        start_step, start_epoch, best_validation_loss = load_model_checkpoint(
            checkpoint_path, model, optimizer, configuration
        )

    # Training loop
    print("\nStarting training...")
    
    best_validation_perplexity = float("inf") if best_validation_loss == float('inf') else math.exp(best_validation_loss)
    patience_counter = 0
    best_model_path = os.path.join(configuration.output_directory, "best_model.pt")
    
    training_history = []
    training_start_time = time.time()
    current_step = start_step
    early_stop_triggered = False

    for epoch in range(start_epoch, configuration.max_epochs):
        print(f"\nEpoch {epoch + 1}/{configuration.max_epochs}")
        epoch_losses = []
        
        for batch_index in range(steps_per_epoch):
            # Skip already completed steps when resuming
            if epoch == start_epoch and current_step > 0 and batch_index < (current_step % steps_per_epoch):
                continue
                
            context_batch, chunk_batch = get_batch(train_dataset, configuration.batch_size, configuration.device)
            
            # Forward pass
            logits = model(context_batch, chunk_batch)
            loss = F.cross_entropy(logits.reshape(-1, configuration.vocabulary_size), chunk_batch.reshape(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), configuration.grad_clip)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            current_step += 1
            
            # Periodic evaluation
            if current_step % configuration.eval_interval == 0:
                recent_train_loss = np.mean(epoch_losses[-100:]) if len(epoch_losses) >= 100 else np.mean(epoch_losses)
                validation_loss = estimate_loss(model, validation_dataset, configuration)
                validation_perplexity = math.exp(min(validation_loss, 20))
                
                elapsed_hours = (time.time() - training_start_time) / 3600
                
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step {current_step:,} | {elapsed_hours:.2f} hours")
                print(f"  Training loss: {recent_train_loss:.4f}")
                print(f"  Validation loss: {validation_loss:.4f}")
                print(f"  Validation perplexity: {validation_perplexity:.2f}")
                
                training_history.append({
                    "step": current_step,
                    "training_loss": recent_train_loss,
                    "validation_loss": validation_loss,
                    "validation_perplexity": validation_perplexity,
                })
                
                # Save best model
                if validation_loss < best_validation_loss:
                    print("  New best validation loss!")
                    best_validation_loss = validation_loss
                    best_validation_perplexity = validation_perplexity
                    patience_counter = 0
                    save_model_checkpoint(
                        model, optimizer, current_step, epoch,
                        best_validation_loss, configuration, best_model_path
                    )
                else:
                    patience_counter += 1
                    print(f"  No improvement. Patience: {patience_counter}/{configuration.patience}")
                    
                    if patience_counter >= configuration.patience:
                        print("\nEarly stopping triggered.")
                        early_stop_triggered = True
                        break
            
            # Progress indicator
            if current_step % 50 == 0 and current_step % configuration.eval_interval != 0:
                print(f"  Step {current_step:,} | Loss: {epoch_losses[-1]:.4f}", end="\r")
        
        if early_stop_triggered:
            break

    total_training_time = (time.time() - training_start_time) / 3600
    print(f"\nTraining complete: {total_training_time:.2f} hours")
    print(f"Best validation perplexity: {best_validation_perplexity:.2f}")
    
    # Load best model for final evaluation
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=configuration.device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final test evaluation
    test_loss = estimate_loss(model, test_dataset, configuration)
    test_perplexity = math.exp(min(test_loss, 20))
    
    print("\nFinal Test Results:")
    print(f"  Test loss: {test_loss:.4f}")
    print(f"  Test perplexity: {test_perplexity:.2f}")
    print(f"  Model parameters: {total_parameters:,}")
    print(f"  Training time: {total_training_time:.2f} hours")
    
    # Text generation examples
    print("\nText Generation Examples (temperature=0.8, top_k=40):")
    generation_prompts = ["The history of", "In the early", "Scientists have discovered"]
    for prompt in generation_prompts:
        generate_text(model, tokenizer, prompt, configuration, max_new_words=20)
    
    # Save final results
    final_results = {
        "test_perplexity": test_perplexity,
        "best_validation_perplexity": best_validation_perplexity,
        "total_parameters": total_parameters,
        "training_time_hours": total_training_time,
        "training_history": training_history,
        "configuration": vars(configuration)
    }
    
    results_file = os.path.join(
        configuration.output_directory,
        f"final_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(results_file, "w") as file:
        json.dump(final_results, file, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()
