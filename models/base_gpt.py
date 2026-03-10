"""
Base GPT model (no adaptive hypernetwork).

Purpose:
- Reference implementation of our GPT architecture without dynamic LoRA/hyper updates.
- Serves as the plain baseline in comparisons.

Config (default in this file):
- block_size=1024
- vocab_size=50257
- n_layer=4
- n_head=4
- n_embd=256

Notes:
- Uses untied lm_head (separate output projection), so param count is larger than HF GPT2-small.
"""

from dataclasses import dataclass  
import torch
import torch.nn as nn
from torch.nn import functional as F


def build_rope_cache(seq_len: int, head_dim: int, device, base: int = 10000):
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head_dim.")
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base ** (freqs / head_dim))
    angles = torch.outer(positions, inv_freq)
    cos = angles.cos()[None, None, :, :]
    sin = angles.sin()[None, None, :, :]
    return cos, sin


def apply_rotary_emb(x, cos, sin):
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    x_rot = torch.stack(
        (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1
    )
    return x_rot.flatten(-2)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        # check if the number of heads is divisible by the embedding dimension
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # we here take the head dimension
        self.head_dim = config.n_embd // config.n_head

    def forward(self, x, cos, sin):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x        


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln_1(x), cos, sin)
        x = x + self.mlp(self.ln_2(x))
        return x
        

@dataclass
class GPT_Base_Config:
    block_size: int = 256
    vocab_size: int = 50257
    n_head: int = 4
    n_layer: int = 4
    n_embd: int = 256

class GPT_Base(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        head_dim = config.n_embd // config.n_head
        cos, sin = build_rope_cache(config.block_size, head_dim, torch.device("cpu"))
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        # the final classification head, and gpt-2 uses no bias
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        x = self.transformer.wte(idx)
        cos = self.rope_cos[:, :, :T, :].to(device=idx.device, dtype=x.dtype)
        sin = self.rope_sin[:, :, :T, :].to(device=idx.device, dtype=x.dtype)
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x, cos, sin)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
