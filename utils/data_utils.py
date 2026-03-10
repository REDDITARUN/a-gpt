from pathlib import Path

import tiktoken
import torch


def load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def tokenize_text(text: str, tokenizer_name: str = "gpt2") -> torch.Tensor:
    enc = tiktoken.get_encoding(tokenizer_name)
    token_ids = enc.encode(text)
    return torch.tensor(token_ids, dtype=torch.long)


def split_train_val(tokens: torch.Tensor, val_ratio: float = 0.1):
    n = tokens.numel()
    n_val = int(n * val_ratio)
    if n_val <= 0 or n_val >= n:
        raise ValueError("val_ratio creates an empty split; choose a value in (0, 1).")
    return tokens[:-n_val], tokens[-n_val:]


def get_batch(tokens: torch.Tensor, block_size: int, batch_size: int, device: str):
    if tokens.numel() <= block_size + 1:
        raise ValueError("Token sequence too short for requested block_size.")
    ix = torch.randint(0, tokens.numel() - block_size - 1, (batch_size,))
    x = torch.stack([tokens[i : i + block_size] for i in ix])
    y = torch.stack([tokens[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def get_diffusion_batch(
    tokens: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: str,
    mask_token_id: int,
    num_diffusion_steps: int,
    min_mask_ratio: float = 0.05,
):
    if tokens.numel() <= block_size:
        raise ValueError("Token sequence too short for requested block_size.")
    if num_diffusion_steps <= 0:
        raise ValueError("num_diffusion_steps must be > 0.")

    ix = torch.randint(0, tokens.numel() - block_size, (batch_size,))
    clean = torch.stack([tokens[i : i + block_size] for i in ix]).to(device)

    # Sample diffusion step per sequence and scale mask ratio with timestep.
    t = torch.randint(1, num_diffusion_steps + 1, (batch_size,), device=device)
    mask_ratio = min_mask_ratio + (1.0 - min_mask_ratio) * (t.float() / num_diffusion_steps)
    rand = torch.rand(batch_size, block_size, device=device)
    mask = rand < mask_ratio[:, None]

    # Ensure each sample has at least one masked token for stable masked loss.
    no_mask_rows = torch.where(mask.sum(dim=1) == 0)[0]
    if no_mask_rows.numel() > 0:
        forced_cols = torch.randint(0, block_size, (no_mask_rows.numel(),), device=device)
        mask[no_mask_rows, forced_cols] = True

    x_noisy = clean.clone()
    x_noisy[mask] = mask_token_id
    return x_noisy, clean, mask, t

