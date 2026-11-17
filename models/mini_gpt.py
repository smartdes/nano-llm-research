import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config['d_model'] % config['n_heads'] == 0
        self.key = nn.Linear(config['d_model'], config['d_model'])
        self.query = nn.Linear(config['d_model'], config['d_model'])
        self.value = nn.Linear(config['d_model'], config['d_model'])
        self.proj = nn.Linear(config['d_model'], config['d_model'])
        self.n_heads = config['n_heads']
        self.register_buffer("bias", torch.tril(torch.ones(config['block_size'], config['block_size']))
                                     .view(1, 1, config['block_size'], config['block_size']))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config['d_model'])
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config['d_model'])
        self.mlp = nn.Sequential(
            nn.Linear(config['d_model'], 4 * config['d_model']),
            nn.GELU(),
            nn.Linear(4 * config['d_model'], config['d_model']),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        self.position_embedding = nn.Embedding(config['block_size'], config['d_model'])
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config['n_layers'])])
        self.ln_f = nn.LayerNorm(config['d_model'])
        self.head = nn.Linear(config['d_model'], config['vocab_size'], bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (B, T)) and complete
        the sequence max_new_tokens times, predicting autoregressively.
        """
        for _ in range(max_new_tokens):
            # Crop to the context window if necessary
            idx_cond = idx if idx.size(1) <= self.config['block_size'] else idx[:, -self.config['block_size']:]
            # Forward the model to obtain logits for the current sequence
            logits, _ = self(idx_cond)
            # Focus on the last time step and scale by temperature
            logits = logits[:, -1, :] / temperature
            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # Softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample next token from probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append next token to current sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx