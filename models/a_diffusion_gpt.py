"""
Adaptive diffusion-style GPT model (block denoising + dynamic LoRA on V).

Purpose:
- Starts from diffusion_gpt.py architecture (bidirectional denoising).
- Adds per-head dynamic low-rank updates to V using a tiny conditioning transformer.
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


def rms_norm(x):
    return F.rms_norm(x, (x.size(-1),))


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


@dataclass
class HyperConfig:
    rank: int = 8
    n_heads: int = 2
    n_layers: int = 2
    alpha: float = 1.0


class TinyHeadTransformer(nn.Module):
    def __init__(self, head_dim, hyper_config):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.MultiheadAttention(head_dim, hyper_config.n_heads, batch_first=True)
                for _ in range(hyper_config.n_layers)
            ]
        )
        self.rank = hyper_config.rank
        self.scale = hyper_config.alpha / max(1, hyper_config.rank)
        self.to_A = nn.Linear(head_dim, head_dim * self.rank, bias=False)
        self.to_B = nn.Linear(head_dim, self.rank * head_dim, bias=False)
        nn.init.zeros_(self.to_B.weight)

    def forward(self, x_head):
        # Diffusion backbone is bidirectional, so conditioning is also bidirectional.
        out = x_head
        for attn_layer in self.layers:
            out, _ = attn_layer(out, out, out)
        ctx = out.mean(dim=1)
        B, D = ctx.shape
        A = self.to_A(ctx).view(B, D, self.rank)
        Bm = self.to_B(ctx).view(B, self.rank, D)
        return A, Bm


class BidirectionalSelfAttention(nn.Module):
    def __init__(self, config, hyper_config=None):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_k = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_v = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.hyper_v = None
        if hyper_config is not None:
            self.hyper_v = nn.ModuleList(
                [TinyHeadTransformer(self.head_dim, hyper_config) for _ in range(self.n_head)]
            )

    def forward(self, x, cos, sin):
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        x_heads = x.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = rms_norm(q)
        k = rms_norm(k)

        if self.hyper_v is not None:
            v_new = []
            for h in range(self.n_head):
                v_h = v[:, h, :, :]
                x_h = x_heads[:, h, :, :]
                A, Bm = self.hyper_v[h](x_h)
                delta_v = torch.einsum("bti,bir,bro->bto", v_h, A, Bm)
                v_new.append(v_h + self.hyper_v[h].scale * delta_v)
            v = torch.stack(v_new, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, hyper_config=None):
        super().__init__()
        self.attn = BidirectionalSelfAttention(config, hyper_config=hyper_config)
        self.mlp = MLP(config)

    def forward(self, x, cos, sin):
        x = x + self.attn(rms_norm(x), cos, sin)
        x = x + self.mlp(rms_norm(x))
        return x


@dataclass
class GPT_Adaptive_Diffusion_Config:
    block_size: int = 256
    vocab_size: int = 50257
    n_head: int = 4
    n_layer: int = 4
    n_embd: int = 256
    num_diffusion_steps: int = 32


class GPT_Adaptive_Diffusion(nn.Module):
    def __init__(self, config, hyper_config=None):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.time_emb = nn.Embedding(config.num_diffusion_steps + 1, config.n_embd)
        self.blocks = nn.ModuleList(
            [Block(config, hyper_config=hyper_config) for _ in range(config.n_layer)]
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        head_dim = config.n_embd // config.n_head
        cos, sin = build_rope_cache(config.block_size, head_dim, torch.device("cpu"))
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, mask=None, t=None):
        B, T = idx.size()
        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        x = self.token_emb(idx)
        if t is None:
            t = torch.zeros(B, dtype=torch.long, device=idx.device)
        t = t.clamp(min=0, max=self.config.num_diffusion_steps)
        x = x + self.time_emb(t)[:, None, :]
        x = rms_norm(x)
        cos = self.rope_cos[:, :, :T, :].to(device=idx.device, dtype=x.dtype)
        sin = self.rope_sin[:, :, :T, :].to(device=idx.device, dtype=x.dtype)

        for block in self.blocks:
            x = block(x, cos, sin)
        x = rms_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            if mask is not None:
                mask_flat = mask.view(-1).float()
                per_tok = F.cross_entropy(logits_flat, targets_flat, reduction="none")
                denom = mask_flat.sum().clamp(min=1.0)
                loss = (per_tok * mask_flat).sum() / denom
            else:
                loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss
