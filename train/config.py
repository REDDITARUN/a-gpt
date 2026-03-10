from dataclasses import dataclass


@dataclass
class TrainConfig:
    model_name: str = "base"  # base | adaptive | matched | diffusion | adaptive_diffusion

    # Data sources
    tiny_data_path: str = "data/tiny_stories.txt"
    cosmo_data_path: str = "data/cosmopedia_wikihow_text_trunc.txt"
    wiki_data_path: str = "data/wikitext2_text_trunc.txt"
    code_data_path: str = "data/python_code_text_trunc.txt"
    tokenizer_name: str = "gpt2"

    # Weighted source sampling (normalized in runner)
    tiny_weight: float = 0.10
    cosmo_weight: float = 0.40
    wiki_weight: float = 0.20
    code_weight: float = 0.30

    # Optimization
    max_steps: int = 1000
    warmup_steps: int = 100
    eval_interval: int = 100
    eval_iters: int = 20
    grad_accum_steps: int = 4
    batch_size: int = 8
    base_lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    clip_grad: float = 1.0

    # Data split
    val_ratio: float = 0.1

    # Diffusion denoising options
    diffusion_mask_token_id: int = 50256
    num_diffusion_steps: int = 32
    min_mask_ratio: float = 0.05

    # Logging/checkpoints
    checkpoint_dir: str = "checkpoints"
    checkpoint_name: str = "best.pt"
    use_wandb: bool = False
    wandb_project: str = "gpt-pretrain"
    wandb_run_name: str = ""
