import math
from pathlib import Path

import torch


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def perplexity_from_loss(loss: float) -> float:
    return math.exp(loss)


def get_cosine_lr(step: int, max_steps: int, warmup_steps: int, base_lr: float, min_lr: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if max_steps <= warmup_steps:
        return base_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    progress = max(0.0, min(1.0, progress))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model, get_batch_fn, eval_iters: int = 20):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(eval_iters):
            xb, yb = get_batch_fn(split)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def save_checkpoint(path: str, model, optimizer, step: int, run_config: dict, best_val_loss: float):
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": run_config,
            "best_val_loss": best_val_loss,
        },
        str(path_obj),
    )


def tokens_per_step(batch_size: int, block_size: int, grad_accum_steps: int) -> int:
    return batch_size * block_size * grad_accum_steps


def steps_per_epoch(num_train_tokens: int, batch_size: int, block_size: int, grad_accum_steps: int) -> float:
    return num_train_tokens / tokens_per_step(batch_size, block_size, grad_accum_steps)

