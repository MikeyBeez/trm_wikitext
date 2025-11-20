import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

class MinimalRecursiveTrainer:
    def __init__(self, model, tokenizer, mask_token_id=50256, lr=1e-4):
        self.model = model
        self.tokenizer = tokenizer
        self.mask_token_id = mask_token_id
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
        self.temp = 1.5  # Lowered from 2.0 for more stability
        self.step = 0
        
    def fill_masks_soft(self, masked_input, logits, mask_positions):
        """Maintain full gradient path through soft filling"""
        # Softmax with temperature (no Gumbel yet, keep simple)
        probs = torch.softmax(logits / self.temp, dim=-1)
        
        # Differentiable embedding lookup
        embeddings_matrix = self.model.transformer.wte.weight
        soft_embeddings = torch.matmul(probs, embeddings_matrix)
        
        # Get original embeddings
        original_embeddings = self.model.transformer.wte(masked_input)
        
        # Interpolate at mask positions only
        filled = torch.where(
            mask_positions.unsqueeze(-1),
            soft_embeddings,
            original_embeddings
        )
        return filled
    
    def encode_from_embeddings(self, embeddings):
        """Run transformer from embeddings instead of input_ids"""
        hidden = embeddings
        
        for layer in self.model.transformer.h:
            outputs = layer(hidden)
            hidden = outputs[0]
            
        hidden = self.model.transformer.ln_f(hidden)
        return hidden
    
    def create_contiguous_mask(self, input_ids, mask_prob=0.6, span_length=3):
        """Mask contiguous spans instead of random tokens"""
        B, L = input_ids.shape
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        
        for b in range(B):
            num_masks = int(L * mask_prob / span_length)
            for _ in range(num_masks):
                start = torch.randint(0, max(1, L - span_length), (1,)).item()
                mask[b, start:start + span_length] = True
                
        return mask
    
    def train_step(self, input_ids):
        B, L = input_ids.shape
        targets = input_ids.clone()  # Next token prediction
        
        # Create contiguous span masks (harder than random)
        mask = self.create_contiguous_mask(input_ids, mask_prob=0.6)  # Keep at 60%
        masked_input = input_ids.clone()
        masked_input[mask] = self.mask_token_id
        
        # Stage 1: Forward with masks
        outputs1 = self.model(masked_input, output_hidden_states=True)
        hidden1 = outputs1.hidden_states[-1]
        logits1 = outputs1.logits
        
        # Loss only on masked positions
        if mask.sum() > 0:
            loss1 = F.cross_entropy(
                logits1[mask].reshape(-1, logits1.size(-1)),
                targets[mask].reshape(-1)
            )
        else:
            loss1 = torch.tensor(0.0, device=logits1.device)
        
        # Stage 2: Soft fill and second forward
        filled_embeddings = self.fill_masks_soft(masked_input, logits1, mask)
        hidden2 = self.encode_from_embeddings(filled_embeddings)
        logits2 = self.model.lm_head(hidden2)
        
        # Loss on same masked positions
        if mask.sum() > 0:
            loss2 = F.cross_entropy(
                logits2[mask].reshape(-1, logits2.size(-1)),
                targets[mask].reshape(-1)
            )
        else:
            loss2 = torch.tensor(0.0, device=logits2.device)
        
        # More aggressive weighting - Stage 2 should dominate
        total_loss = 0.1 * loss1 + 0.9 * loss2  # Changed from 0.3/0.7
        
        # Add consistency regularization after warmup
        if self.step > 100 and mask.sum() > 0:
            consistency = F.kl_div(
                F.log_softmax(logits2[mask], dim=-1),
                F.softmax(logits1[mask].detach(), dim=-1),
                reduction='batchmean'
            )
            total_loss = total_loss + 0.05 * consistency  # Small consistency penalty
        
        # Backward pass
        total_loss.backward()
        
        # Gradient norm check
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Critical diagnostics
        with torch.no_grad():
            if mask.sum() > 0:
                pred1 = logits1[mask].argmax(-1)
                pred2 = logits2[mask].argmax(-1)
                true_targets = targets[mask]
                
                acc1 = (pred1 == true_targets).float().mean()
                acc2 = (pred2 == true_targets).float().mean()
                
                # Cosine drift between hidden states
                h1_masked = hidden1[mask].reshape(-1, hidden1.size(-1))
                h2_masked = hidden2[mask].reshape(-1, hidden2.size(-1))
                drift = 1 - F.cosine_similarity(h1_masked, h2_masked, dim=-1).mean()
                
                # Perplexity as difficulty measure
                perp1 = torch.exp(loss1)
                perp2 = torch.exp(loss2)
            else:
                acc1 = acc2 = drift = perp1 = perp2 = 0
        
        # Anneal temperature more gradually
        if self.step % 200 == 0 and self.step > 0:  # Changed from 100
            self.temp = max(0.5, self.temp * 0.95)  # Slower annealing
        
        self.step += 1
        
        return {
            'loss': total_loss.item(),
            'loss1': loss1.item(),
            'loss2': loss2.item(),
            'acc1': acc1.item(),
            'acc2': acc2.item(),
            'improvement': (acc2 - acc1).item(),
            'cosine_drift': drift.item(),
            'perp1': perp1.item(),
            'perp2': perp2.item(),
            'grad_norm': grad_norm.item(),
            'temperature': self.temp,
            'mask_ratio': mask.float().mean().item()
        }

# Setup
def main():
    # Optional wandb - comment out these 3 lines if not using
    # import wandb
    # wandb.init(project="recursive-training-v0", name="wikitext103-minimal")
    
    # Load WikiText-103
    print("Loading WikiText-103...")
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples['text'], 
            truncation=True, 
            padding='max_length', 
            max_length=512
        )
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format('torch', columns=['input_ids'])
    
    # Create dataloader
    train_dataloader = DataLoader(
        tokenized_datasets['train'], 
        batch_size=4,  # Small batch for memory
        shuffle=True
    )
    
    # Initialize model
    config = GPT2Config(
        n_embd=768,
        n_layer=12,
        n_head=12,
        resid_pdrop=0.3,  # Aggressive dropout
        embd_pdrop=0.3,
        attn_pdrop=0.3
    )
    model = GPT2LMHeadModel(config)
    model.cuda()
    
    # Create trainer
    trainer = MinimalRecursiveTrainer(model, tokenizer)
    
    # Training loop
    print("Starting training...")
    for epoch in range(1):
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
        
        for step, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].cuda()
            
            metrics = trainer.train_step(input_ids)
            
            # Log to wandb if using
            # wandb.log(metrics, step=trainer.step)
            
            # Update progress bar
            progress_bar.set_postfix({
                'acc1': f"{metrics['acc1']:.3f}",
                'acc2': f"{metrics['acc2']:.3f}",
                'drift': f"{metrics['cosine_drift']:.3f}",
                'improve': f"{metrics['improvement']:.3f}"
            })
            
            # Critical checks every 100 steps
            if step % 100 == 0 and step > 0:
                print(f"\n[Step {step}] CRITICAL METRICS:")
                print(f"  Improvement: {metrics['improvement']:.4f} (TARGET > 0.01)")
                print(f"  Cosine Drift: {metrics['cosine_drift']:.4f} (MUST BE > 0.01)")
                print(f"  Grad Norm: {metrics['grad_norm']:.2f} (WATCH FOR > 1000)")
                print(f"  Temperature: {metrics['temperature']:.2f}")
                print(f"  Perp Reduction: {(metrics['perp1'] - metrics['perp2']) / max(metrics['perp1'], 1e-8):.3f}")
                
                # Extended threshold - give it more time
                if step >= 1000:  # Changed from 500
                    if metrics['improvement'] < 0.01:
                        print("⚠️ WARNING: Stage 2 improvement still weak")
                        # Don't kill yet, let it run further
                    if metrics['cosine_drift'] < 0.01:
                        print("❌ KILLING: No meaningful refinement happening")
                        break
                        
                # Success signals
                if metrics['improvement'] > 0.02 and metrics['cosine_drift'] > 0.05:
                    print("✅ STRONG SIGNAL: Recursive training is working!")
                elif metrics['improvement'] > 0.01:
                    print("📈 POSITIVE: Seeing consistent improvement")
            
            # Save checkpoint at key milestones
            if step in [500, 1000, 2000, 5000]:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'step': step,
                    'metrics': metrics
                }, f'recursive_v0_step_{step}.pt')
                print(f"Saved checkpoint at step {step}")
                
            # Hard stop at 5000 for initial experiment
            if step >= 5000:
                print("Reached 5000 steps, stopping initial experiment")
                break

if __name__ == "__main__":
    main()
