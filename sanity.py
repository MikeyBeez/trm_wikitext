"""
TRM: Sanity Check - Overfit a Single Sentence
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from dataclasses import dataclass
import time

# ===============================
# CONFIG (Same as before, but smaller)
# ===============================
@dataclass
class Config:
    context_size: int = 16  # Small context
    chunk_size: int = 2
    embed_dim: int = 128    # Small dim for speed
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.0    # No dropout for overfitting!
    n_refinements: int = 2
    n_recursions: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer_name: str = "gpt2"

# ===============================
# PASTE YOUR MODEL CLASSES HERE
# (TransformerBlock, TinyRecursiveModel)
# I will omit them to save space, assuming they are in 'wordleveli_2.py'
# You can import them or paste them back in.
# ===============================
from wordleveli_2 import TinyRecursiveModel, TransformerBlock # Assuming file is named this

def main():
    config = Config()
    print(f"Sanity Check Device: {config.device}")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # THE TEST DATA: One repeated sentence
    text = "The quick brown fox jumps over the lazy dog. " * 50
    tokens = tokenizer.encode(text)
    
    # Create simple batches
    X, Y = [], []
    for i in range(0, len(tokens) - config.context_size - config.chunk_size, config.chunk_size):
        ctx = tokens[i:i+config.context_size]
        target = tokens[i+config.context_size:i+config.context_size+config.chunk_size]
        X.append(torch.tensor(ctx))
        Y.append(torch.tensor(target))
    
    X = torch.stack(X).to(config.device)
    Y = torch.stack(Y).to(config.device)
    
    print(f"Overfitting on {len(X)} samples...")
    
    model = TinyRecursiveModel(len(tokenizer), config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3) # High LR to force memorization

    # Train loop
    for step in range(501):
        optimizer.zero_grad()
        
        # Random batch
        idx = torch.randint(0, len(X), (32,))
        ctx_batch = X[idx]
        tgt_batch = Y[idx]
        
        logits_list = model(ctx_batch)
        
        # Calculate loss on the FINAL refinement only to force convergence
        final_logits = logits_list[-1]
        loss = F.cross_entropy(final_logits.reshape(-1, len(tokenizer)), tgt_batch.reshape(-1))
        
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f}")
            
            # TEST GENERATION
            if step % 100 == 0:
                model.eval()
                with torch.no_grad():
                    # Prompt with the start of the sentence
                    prompt = "The quick brown"
                    input_ids = tokenizer.encode(prompt)
                    input_tensor = torch.tensor([input_ids], device=config.device)
                    
                    # Pad if necessary
                    if input_tensor.shape[1] < config.context_size:
                        padding = torch.full((1, config.context_size - input_tensor.shape[1]), tokenizer.pad_token_id, device=config.device)
                        input_tensor = torch.cat([padding, input_tensor], dim=1)
                    else:
                        input_tensor = input_tensor[:, -config.context_size:]

                    # Predict next chunk
                    logits = model(input_tensor)[-1]
                    pred_ids = torch.argmax(logits, dim=-1).squeeze().tolist()
                    print(f"  Prompt: '{prompt}' -> Predicted: '{tokenizer.decode(pred_ids)}'")
                model.train()

if __name__ == "__main__":
    main()
