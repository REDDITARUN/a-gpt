import torch
from models.a_gpt import GPT_Custom, GPT_Custom_Config, HyperConfig
from models.base_gpt import GPT_Base, GPT_Base_Config
from models.base_gpt_matched import GPT_Base_Matched, GPT_Base_Matched_Config

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# 0) Base GPT
base_gpt = GPT_Base(GPT_Base_Config()).to(device)
t, tr = count_params(base_gpt)
print(f"Base GPT: total={t:,}, trainable={tr:,}")

# 1) Adaptive GPT
adaptive_gpt = GPT_Custom(GPT_Custom_Config(), hyper_config=HyperConfig()).to(device)
t, tr = count_params(adaptive_gpt)
print(f"Adaptive GPT: total={t:,}, trainable={tr:,}")

# 2) Base GPT Matched
base_gpt_matched = GPT_Base_Matched(GPT_Base_Matched_Config()).to(device)
t, tr = count_params(base_gpt_matched)
print(f"Base GPT Matched: total={t:,}, trainable={tr:,}")
