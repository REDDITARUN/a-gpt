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

