"""
Diffusion-style GPT model (block denoising, non-causal attention).

Purpose:
- Model-only diffusion baseline in the same structure as other models in this repo.
- Predicts clean tokens from partially masked/noised token blocks.
- Uses bidirectional attention (is_causal=False) and optional masked-only loss.
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


def rms_norm(x):
    # Functional RMSNorm to keep behavior close to the prior diffusion prototype.
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


class BidirectionalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_k = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_v = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

    def forward(self, x, cos, sin):
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = rms_norm(q)
        k = rms_norm(k)

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
        # Squared-ReLU is kept intentionally to preserve prior behavior.
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = BidirectionalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x, cos, sin):
        x = x + self.attn(rms_norm(x), cos, sin)
        x = x + self.mlp(rms_norm(x))
        return x


@dataclass
class GPT_Diffusion_Config:
    block_size: int = 256
    vocab_size: int = 50257
    n_head: int = 4
    n_layer: int = 4
    n_embd: int = 256
    num_diffusion_steps: int = 32


class GPT_Diffusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.time_emb = nn.Embedding(config.num_diffusion_steps + 1, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
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