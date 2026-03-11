# Adaptive GPT Mixed Pretraining

Want the full details on the architecture changes and results? Read the complete write-up [here](LINK_GOES_HERE).

Adaptive GPT applies per-head, input-dependent low-rank updates to attention values. Dynamic factors are generated from the ongoing hidden-state stream and used to modulate \(V\) before attention, enabling context-adaptive computation with limited additional parameters.

This repo trains and compares five GPT-style variants on mixed-text pretraining:

- `base`: plain causal GPT baseline
- `matched`: causal GPT with parameter budget matched to adaptive
- `adaptive`: causal GPT + dynamic LoRA-style hypernetwork on value path
- `diffusion`: bidirectional denoising GPT (diffusion-style masked training)
- `adaptive_diffusion`: diffusion GPT + dynamic LoRA-style hypernetwork on value path

---

## Project structure

- `models/`: model definitions (`base_gpt.py`, `base_gpt_matched.py`, `a_gpt.py`, `diffusion_gpt.py`, `a_diffusion_gpt.py`)
- `data/`: data download + raw text files
- `utils/`: data helpers, training helpers, parameter comparison
- `train/`: training config + main runner


---

##  Setup

```bash
# clone and ...
cd a-gpt

python3 -m venv venv
source venv/bin/activate

python -m pip install -U pip
python -m pip install torch tiktoken datasets tqdm wandb huggingface_hub matplotlib pillow ipykernel
```


## Prepare data

```bash
python data/get_data.py
python utils/data_edit.py
```

This generates/updates processed text files like:
- `data/tiny_stories.txt`
- `data/cosmopedia_wikihow_text_trunc.txt`
- `data/wikitext2_text_trunc.txt`
- `data/python_code_text_trunc.txt`


## Compare parameter counts

```bash
python -m utils.model_comparision
```


## Train models (CLI examples)

> Replace `--max_steps 12` with larger values for real runs.

### Base GPT
```bash
python -u -m train.run \
  --model base \
  --max_steps 12 \
  --wandb_project adaptive-gpt \
  --wandb_run_name base-gpt \
  --use_wandb
```

### Matched GPT
```bash
python -u -m train.run \
  --model matched \
  --max_steps 12 \
  --wandb_project adaptive-gpt \
  --wandb_run_name matched-gpt \
  --use_wandb
```

### Adaptive GPT
```bash
python -u -m train.run \
  --model adaptive \
  --max_steps 12 \
  --wandb_project adaptive-gpt \
  --wandb_run_name adaptive-gpt \
  --use_wandb
```

### Diffusion GPT
```bash
python -u -m train.run \
  --model diffusion \
  --max_steps 12 \
  --num_diffusion_steps 32 \
  --diffusion_mask_token_id 50256 \
  --min_mask_ratio 0.05 \
  --wandb_project adaptive-gpt \
  --wandb_run_name diffusion-gpt \
  --use_wandb
```

### Adaptive Diffusion GPT
```bash
python -u -m train.run \
  --model adaptive_diffusion \
  --max_steps 12 \
  --num_diffusion_steps 32 \
  --diffusion_mask_token_id 50256 \
  --min_mask_ratio 0.05 \
  --wandb_project adaptive-gpt \
  --wandb_run_name adaptive_diffusion-gpt \
  --use_wandb
```


## Key diffusion arguments

- `--num_diffusion_steps`: timestep count for denoising training
- `--diffusion_mask_token_id`: mask token id (default currently `50256`)
- `--min_mask_ratio`: lower bound on corruption ratio per sample


## Checkpoints

```bash
ls -lh checkpoints
```

Expected files:
- `base_best.pt`
- `matched_best.pt`
- `adaptive_best.pt`
- `diffusion_best.pt`
- `adaptive_diffusion_best.pt`


## References

- Nathan Barry, **tiny-diffusion** (character-level language diffusion baseline, diffusion-vs-GPT comparison, visualization tooling): [https://github.com/nathan-barry/tiny-diffusion](https://github.com/nathan-barry/tiny-diffusion)
- Andrej Karpathy, **nanoGPT** (minimal GPT training/finetuning reference implementation): [https://github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)