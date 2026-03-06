import argparse
import importlib
import os
from dataclasses import asdict
from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm.auto import tqdm

from models.a_gpt import GPT_Custom, GPT_Custom_Config, HyperConfig
from models.base_gpt import GPT_Base, GPT_Base_Config
from models.base_gpt_matched import GPT_Base_Matched, GPT_Base_Matched_Config
from train.config import TrainConfig
from utils.data_utils import get_batch, load_text, split_train_val, tokenize_text
from utils.train_utils import (
    count_params,
    get_cosine_lr,
    get_device,
    perplexity_from_loss,
    save_checkpoint,
    steps_per_epoch,
    tokens_per_step,
)


def build_model(cfg: TrainConfig):
    if cfg.model_name == "base":
        model_cfg = GPT_Base_Config()
        return GPT_Base(model_cfg)
    if cfg.model_name == "matched":
        model_cfg = GPT_Base_Matched_Config()
        return GPT_Base_Matched(model_cfg)
    if cfg.model_name == "adaptive":
        model_cfg = GPT_Custom_Config()
        hyper_cfg = HyperConfig()
        return GPT_Custom(model_cfg, hyper_config=hyper_cfg)
    raise ValueError(f"Unknown model_name: {cfg.model_name}")


def maybe_init_wandb(cfg: TrainConfig):
    if not cfg.use_wandb:
        return None
    # On macOS, legacy service mode avoids noisy wandb-core malloc warnings.
    os.environ.setdefault("WANDB_REQUIRE_LEGACY_SERVICE", "true")
    try:
        wandb = importlib.import_module("wandb")
    except ModuleNotFoundError:
        print("wandb not installed; continuing without wandb logging.")
        return None

    run = wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name or None,
        config=asdict(cfg),
    )
    return run


def train(cfg: TrainConfig):
    device = get_device()
    print(f"device: {device}")

    tiny_text = load_text(cfg.tiny_data_path)
    cosmo_text = load_text(cfg.cosmo_data_path)
    wiki_text = load_text(cfg.wiki_data_path)
    code_text = load_text(cfg.code_data_path)
    tiny_tokens = tokenize_text(tiny_text, cfg.tokenizer_name)
    cosmo_tokens = tokenize_text(cosmo_text, cfg.tokenizer_name)
    wiki_tokens = tokenize_text(wiki_text, cfg.tokenizer_name)
    code_tokens = tokenize_text(code_text, cfg.tokenizer_name)
    tiny_train_tokens, tiny_val_tokens = split_train_val(tiny_tokens, cfg.val_ratio)
    cosmo_train_tokens, cosmo_val_tokens = split_train_val(cosmo_tokens, cfg.val_ratio)
    wiki_train_tokens, wiki_val_tokens = split_train_val(wiki_tokens, cfg.val_ratio)
    code_train_tokens, code_val_tokens = split_train_val(code_tokens, cfg.val_ratio)
    print(f"tiny total tokens: {tiny_tokens.numel():,}")
    print(f"tiny train tokens: {tiny_train_tokens.numel():,}")
    print(f"tiny val tokens: {tiny_val_tokens.numel():,}")
    print(f"cosmo total tokens: {cosmo_tokens.numel():,}")
    print(f"cosmo train tokens: {cosmo_train_tokens.numel():,}")
    print(f"cosmo val tokens: {cosmo_val_tokens.numel():,}")
    print(f"wiki total tokens: {wiki_tokens.numel():,}")
    print(f"wiki train tokens: {wiki_train_tokens.numel():,}")
    print(f"wiki val tokens: {wiki_val_tokens.numel():,}")
    print(f"code total tokens: {code_tokens.numel():,}")
    print(f"code train tokens: {code_train_tokens.numel():,}")
    print(f"code val tokens: {code_val_tokens.numel():,}")

    model = build_model(cfg).to(device)
    total_params, trainable_params = count_params(model)
    print(f"model: {cfg.model_name}")
    print(f"total params: {total_params:,}")
    print(f"trainable params: {trainable_params:,}")
    model_block_size = model.config.block_size
    print(f"model block_size: {model_block_size}")

    source_names = ["tiny", "cosmo", "wiki", "code"]
    source_weights = {
        "tiny": cfg.tiny_weight,
        "cosmo": cfg.cosmo_weight,
        "wiki": cfg.wiki_weight,
        "code": cfg.code_weight,
    }
    weight_sum = sum(source_weights.values())
    if weight_sum <= 0:
        raise ValueError("Sum of source weights must be > 0.")
    source_weights = {k: v / weight_sum for k, v in source_weights.items()}
    print(
        "sampling weights:"
        f" tiny={source_weights['tiny']:.3f},"
        f" cosmo={source_weights['cosmo']:.3f},"
        f" wiki={source_weights['wiki']:.3f},"
        f" code={source_weights['code']:.3f}"
    )
    train_sample_probs = torch.tensor([source_weights[n] for n in source_names], dtype=torch.float)

    tps = tokens_per_step(cfg.batch_size, model_block_size, cfg.grad_accum_steps)
    mixed_train_tokens = (
        tiny_train_tokens.numel()
        + cosmo_train_tokens.numel()
        + wiki_train_tokens.numel()
        + code_train_tokens.numel()
    )
    spe = steps_per_epoch(mixed_train_tokens, cfg.batch_size, model_block_size, cfg.grad_accum_steps)
    print(f"tokens/step: {tps:,}")
    print(f"steps/epoch (approx): {spe:.2f}")

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.base_lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )

    split_tokens = {
        "tiny": {"train": tiny_train_tokens, "val": tiny_val_tokens},
        "cosmo": {"train": cosmo_train_tokens, "val": cosmo_val_tokens},
        "wiki": {"train": wiki_train_tokens, "val": wiki_val_tokens},
        "code": {"train": code_train_tokens, "val": code_val_tokens},
    }

    def get_batch_fn(split, dataset):
        if dataset not in split_tokens:
            raise ValueError(f"Unknown dataset: {dataset}")
        source = split_tokens[dataset][split]
        return get_batch(source, model_block_size, cfg.batch_size, device)

    @torch.no_grad()
    def estimate_mixed_loss():
        model.eval()
        losses_by_source = {name: [] for name in source_names}
        for _ in range(cfg.eval_iters):
            for name in source_names:
                xb, yb = get_batch_fn("val", name)
                _, loss = model(xb, yb)
                losses_by_source[name].append(loss.item())
        model.train()
        means = {name: sum(vals) / len(vals) for name, vals in losses_by_source.items()}
        mixed = sum(source_weights[name] * means[name] for name in source_names)
        return means, mixed

    wandb_run = maybe_init_wandb(cfg)
    best_val = float("inf")

    model.train()
    pbar = tqdm(range(cfg.max_steps), desc=f"train/{cfg.model_name}", unit="step")
    for step in pbar:
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for _ in range(cfg.grad_accum_steps):
            # Weighted pretraining: sample each micro-batch from one of four datasets.
            source_idx = int(torch.multinomial(train_sample_probs, num_samples=1).item())
            dataset_name = source_names[source_idx]
            xb, yb = get_batch_fn("train", dataset_name)
            _, loss = model(xb, yb)
            loss = loss / cfg.grad_accum_steps
            loss.backward()
            loss_accum += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)

        lr = get_cosine_lr(step, cfg.max_steps, cfg.warmup_steps, cfg.base_lr, cfg.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        optimizer.step()
        pbar.set_postfix(train_loss=f"{loss_accum:.4f}", lr=f"{lr:.2e}")

        if wandb_run is not None:
            wandb_run.log(
                {
                    "step": step,
                    "lr": lr,
                    "tokens_seen": (step + 1) * tps,
                    "train/loss": loss_accum,
                }
            )

        if step % cfg.eval_interval == 0 or step == cfg.max_steps - 1:
            val_means, mixed_val_loss = estimate_mixed_loss()
            tiny_val_loss = val_means["tiny"]
            cosmo_val_loss = val_means["cosmo"]
            wiki_val_loss = val_means["wiki"]
            code_val_loss = val_means["code"]
            tiny_val_ppl = perplexity_from_loss(tiny_val_loss)
            cosmo_val_ppl = perplexity_from_loss(cosmo_val_loss)
            wiki_val_ppl = perplexity_from_loss(wiki_val_loss)
            code_val_ppl = perplexity_from_loss(code_val_loss)
            mixed_val_ppl = perplexity_from_loss(mixed_val_loss)
            tokens_seen = (step + 1) * tps

            pbar.write(
                f"step {step:5d} | lr {lr:.2e} | "
                f"tiny val {tiny_val_loss:.4f} (ppl {tiny_val_ppl:.2f}) | "
                f"cosmo val {cosmo_val_loss:.4f} (ppl {cosmo_val_ppl:.2f}) | "
                f"wiki val {wiki_val_loss:.4f} (ppl {wiki_val_ppl:.2f}) | "
                f"code val {code_val_loss:.4f} (ppl {code_val_ppl:.2f}) | "
                f"mixed val {mixed_val_loss:.4f} (ppl {mixed_val_ppl:.2f})"
            )

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "step": step,
                        "lr": lr,
                        "tokens_seen": tokens_seen,
                        "val/tiny_loss": tiny_val_loss,
                        "val/tiny_ppl": tiny_val_ppl,
                        "val/cosmo_loss": cosmo_val_loss,
                        "val/cosmo_ppl": cosmo_val_ppl,
                        "val/wiki_loss": wiki_val_loss,
                        "val/wiki_ppl": wiki_val_ppl,
                        "val/code_loss": code_val_loss,
                        "val/code_ppl": code_val_ppl,
                        "val/mixed_loss": mixed_val_loss,
                        "val/mixed_ppl": mixed_val_ppl,
                    }
                )

            if mixed_val_loss < best_val:
                best_val = mixed_val_loss
                ckpt_name = f"{cfg.model_name}_{Path(cfg.checkpoint_name).name}"
                ckpt_path = str(Path(cfg.checkpoint_dir) / ckpt_name)
                save_checkpoint(ckpt_path, model, optimizer, step, asdict(cfg), best_val)
                pbar.write(f"saved best checkpoint: {ckpt_path}")

    if wandb_run is not None:
        wandb_run.finish()


def parse_args():
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Weighted pretraining on tiny/cosmo/wiki/code sources.")
    parser.add_argument("--model", choices=["base", "adaptive", "matched"], default=defaults.model_name)
    parser.add_argument("--tiny_data", default=defaults.tiny_data_path)
    parser.add_argument("--cosmo_data", default=defaults.cosmo_data_path)
    parser.add_argument("--wiki_data", default=defaults.wiki_data_path)
    parser.add_argument("--code_data", default=defaults.code_data_path)
    parser.add_argument("--tiny_weight", type=float, default=defaults.tiny_weight)
    parser.add_argument("--cosmo_weight", type=float, default=defaults.cosmo_weight)
    parser.add_argument("--wiki_weight", type=float, default=defaults.wiki_weight)
    parser.add_argument("--code_weight", type=float, default=defaults.code_weight)
    parser.add_argument("--max_steps", type=int, default=defaults.max_steps)
    parser.add_argument("--batch_size", type=int, default=defaults.batch_size)
    parser.add_argument("--grad_accum_steps", type=int, default=defaults.grad_accum_steps)
    parser.add_argument("--eval_interval", type=int, default=defaults.eval_interval)
    parser.add_argument("--eval_iters", type=int, default=defaults.eval_iters)
    parser.add_argument("--warmup_steps", type=int, default=defaults.warmup_steps)
    parser.add_argument("--base_lr", type=float, default=defaults.base_lr)
    parser.add_argument("--min_lr", type=float, default=defaults.min_lr)
    parser.add_argument("--val_ratio", type=float, default=defaults.val_ratio)
    parser.add_argument("--checkpoint_dir", default=defaults.checkpoint_dir)
    parser.add_argument("--checkpoint_name", default=defaults.checkpoint_name)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default=defaults.wandb_project)
    parser.add_argument("--wandb_run_name", default=defaults.wandb_run_name)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = TrainConfig(
        model_name=args.model,
        tiny_data_path=args.tiny_data,
        cosmo_data_path=args.cosmo_data,
        wiki_data_path=args.wiki_data,
        code_data_path=args.code_data,
        tiny_weight=args.tiny_weight,
        cosmo_weight=args.cosmo_weight,
        wiki_weight=args.wiki_weight,
        code_weight=args.code_weight,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        warmup_steps=args.warmup_steps,
        base_lr=args.base_lr,
        min_lr=args.min_lr,
        val_ratio=args.val_ratio,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.checkpoint_name,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )
    train(cfg)

