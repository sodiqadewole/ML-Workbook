import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# -----------------------------------------------------------------------------
# Configuration Class
class GPTConfig:
    def __init__(self, vocab_size=50257, context_length=1024, emb_dim=768, n_layers=12, n_heads=12):
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.n_heads = n_heads

# -----------------------------------------------------------------------------
# Causal Multi-Head Self-Attention
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.emb_dim % config.n_heads == 0
        # Key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.emb_dim, 3 * config.emb_dim)
        # Output projection
        self.c_proj = nn.Linear(config.emb_dim, config.emb_dim)
        # Regularization
        self.n_heads = config.n_heads
        self.emb_dim = config.emb_dim
        # Causal mask (lower-triangular)
        self.register_buffer("bias", torch.tril(torch.ones(config.context_length, config.context_length))
                                     .view(1, 1, config.context_length, config.context_length))

    def forward(self, x):
        B, T, C = x.size() # Batch size, sequence length, embedding dimensionality (emb_dim)

        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.emb_dim, dim=2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)

        # Causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # Re-assemble all head outputs side by side

        # Output projection
        y = self.c_proj(y)
        return y

# -----------------------------------------------------------------------------
# Multilayer Perceptron (MLP) / FeedForward Block
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.emb_dim, 4 * config.emb_dim)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.emb_dim, config.emb_dim)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

# -----------------------------------------------------------------------------
# Transformer Block
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.emb_dim)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.emb_dim)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# -----------------------------------------------------------------------------
# Full GPT-2 Architecture
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.emb_dim),
            wpe = nn.Embedding(config.context_length, config.emb_dim),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = nn.LayerNorm(config.emb_dim),
        ))
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

        # Weight tying (GPT-2 shares weights between token embeddings and language model head)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.context_length, f"Cannot forward sequence of length {t}, block size is {self.config.context_length}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # Shape (t)

        # Forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # Token embeddings of shape (b, t, emb_dim)
        pos_emb = self.transformer.wpe(pos) # Position embeddings of shape (t, emb_dim)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # If we are given targets, calculate the training loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # Note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        for _ in range(max_new_tokens):
            # Crop idx if it exceeds context length
            idx_cond = idx if idx.size(1) <= self.config.context_length else idx[:, -self.config.context_length:]
            # Forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # Pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # Optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # Apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

# -----------------------------------------------------------------------------
# Toy Mock Data Loader & Training Loop Setup
if __name__ == "__main__":
    # For demonstration, we'll use tiny hyperparameters so it runs instantly from scratch
    # Replace these with standard GPT-2 Small parameters (vocab_size=50257, emb_dim=768, n_layers=12, n_heads=12) for real workloads
    config = GPTConfig(vocab_size=1000, context_length=64, emb_dim=128, n_layers=4, n_heads=4)
    model = GPT(config)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Using device: {device}")

    # Generate synthetic random token indices for training illustration
    # Shape: (batch_size=4, sequence_length=64)
    x_train = torch.randint(0, config.vocab_size, (4, config.context_length), device=device)
    y_train = torch.randint(0, config.vocab_size, (4, config.context_length), device=device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print("Starting toy training iteration...")
    model.train()
    
    # 1. Forward pass
    logits, loss = model(x_train, y_train)
    
    # 2. Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # 3. Step
    optimizer.step()
    
    print(f"Iteration completed successfully! Cross Entropy Loss: {loss.item():.4f}")
    
    # Text Generation Demonstration
    print("\nGenerating text tokens from a start prompt...")
    model.eval()
    start_prompt = torch.randint(0, config.vocab_size, (1, 5), device=device) # Prompt with 5 tokens
    generated_tokens = model.generate(start_prompt, max_new_tokens=10, temperature=0.8, top_k=10)
    print(f"Generated sequence token IDs: {generated_tokens.tolist()[0]}")
