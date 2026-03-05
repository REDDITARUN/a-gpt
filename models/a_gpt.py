"""
Adaptive GPT model (with dynamic hypernetwork).

Purpose:
- Extends base GPT attention with per-head dynamic low-rank updates to V.
- TinyHeadTransformer generates A,B factors from residual slices, then applies:
    delta_v = v @ A @ B
    v <- v + scale * delta_v

Variants:
- hyper_config=None  -> behaves like base custom GPT (no adaptive updates)
- hyper_config=HyperConfig(...) -> adaptive mode enabled

Default config:
- block_size=1024
- vocab_size=50257
- n_layer=4
- n_head=4
- n_embd=256
"""

from dataclasses import dataclass  
import torch
import torch.nn as nn
from torch.nn import functional as F


# HyperConfig and TinyHeadTransformer
@dataclass
class HyperConfig:
    rank: int = 8           # bottleneck rank (the "r" in LoRA)
    n_heads: int = 2        # heads inside the tiny transformer
    n_layers: int = 2       # how deep the tiny transformer is
    alpha: float = 1.0      # LoRA alpha scaling

class TinyHeadTransformer(nn.Module):
    def __init__(self, head_dim, hyper_config):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(head_dim, hyper_config.n_heads, batch_first=True)
            for _ in range(hyper_config.n_layers)
        ])
        self.rank = hyper_config.rank
        self.scale = hyper_config.alpha / max(1, hyper_config.rank)

        # dynamic LoRA factor generators
        self.to_A = nn.Linear(head_dim, head_dim * self.rank, bias=False)
        self.to_B = nn.Linear(head_dim, self.rank * head_dim, bias=False)

        # LoRA-style stable init: B starts at zero
        nn.init.zeros_(self.to_B.weight)

    def forward(self, x_head):
        # x_head: (B, T, D)
        B, T, D = x_head.shape
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x_head.device)

        out = x_head
        for attn_layer in self.layers:
            out, _ = attn_layer(out, out, out, attn_mask=causal_mask)

        # pool sequence -> conditioning vector
        ctx = out.mean(dim=1)  # (B, D)

        A = self.to_A(ctx).view(B, D, self.rank)   # (B, D, r)
        Bm = self.to_B(ctx).view(B, self.rank, D)  # (B, r, D)
        return A, Bm



# Custom Model

class CausalSelfAttention(nn.Module):

    def __init__(self, config, hyper_config=None):
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
        # we here ta
        self.head_dim = config.n_embd // config.n_head

        # one TinyTransformer per head (only for V)
        self.hyper_v = None
        if hyper_config is not None:
            self.hyper_v = nn.ModuleList([
                TinyHeadTransformer(self.head_dim, hyper_config)
                for _ in range(self.n_head)
            ])

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        x_heads = x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if self.hyper_v is not None:
            # dynamic LoRA on V: generate A,B from residual stream slice, then apply to V
            v_new = []
            for h in range(self.n_head):
                v_h = v[:, h, :, :]                  # (B, T, D)
                x_h = x_heads[:, h, :, :]            # (B, T, D)
                A, Bm = self.hyper_v[h](x_h)         # A:(B, D, r), B:(B, r, D)
                delta_v = torch.einsum("bti,bir,bro->bto", v_h, A, Bm)
                v_new.append(v_h + self.hyper_v[h].scale * delta_v)
            v = torch.stack(v_new, dim=1)            # (B, n_head, T, D)

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

    def __init__(self, config, hyper_config=None):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, hyper_config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
        

@dataclass
class GPT_Custom_Config:
    block_size: int = 1024
    vocab_size: int = 50257
    n_head: int = 4
    n_layer: int = 4
    n_embd: int = 256

class GPT_Custom(nn.Module):

    def __init__(self, config, hyper_config=None):
        super().__init__()
        self.config = config

        # input embedding table
        self.transformer = nn.ModuleDict(dict
        (
            # nn.Embedding(vocab_size: int, n_embd: int)
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),

            # transformer stack
            # each layer has a self-attention head and a feed-forward network
            # nn.ModuleList allows us to store a list of modules
            # it got n_layer number of blocks
            # we need layer norm at the end of the transformer
            h = nn.ModuleList([Block(config, hyper_config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        # the final classification head, and gpt-2 uses no bias
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
