# Adaptive GPT Mixed Pretraining

This repo trains and compares three GPT variants on mixed text pretraining:

- `base`: plain GPT baseline
- `adaptive`: GPT + dynamic LoRA-style hypernetwork on value path
- `matched`: plain GPT with parameter budget matched to adaptive

Training uses a mixed dataset:

- `data/tiny_stories.txt`
- `data/cosmopedia_wikihow_text_trunc.txt`

## Project structure

- `models/`: model definitions (`a_gpt.py`, `base_gpt.py`, `base_gpt_matched.py`)
- `data/`: raw data scripts and generated text files
- `utils/`: helpers for data, training, and parameter comparison
- `train/`: training config and main runner
- `colab_train.ipynb`: Colab notebook for end-to-end training

## Local setup

```bash
python3 -m venv env
env/bin/pip install --upgrade pip
env/bin/pip install torch tiktoken datasets tqdm wandb huggingface_hub
```

## Prepare data

```bash
cd data && ../env/bin/python get_data.py && cd ..
env/bin/python utils/data_edit.py
```

This creates:

- `data/tiny_stories.txt`
- `data/cosmopedia_wikihow_text.txt`
- `data/cosmopedia_wikihow_text_trunc.txt`

## Train models

Train one model:

```bash
env/bin/python -m train.run --model adaptive --max_steps 1000
```

Train all three sequentially:

```bash
./run_all.sh
```

## W&B logging

Login once:

```bash
env/bin/wandb login
```

Enable logging with:

```bash
env/bin/python -m train.run \
  --model adaptive \
  --use_wandb \
  --wandb_project gpt-pretrain \
  --wandb_run_name adaptive-mixed-v1
```

## Parameter comparison

```bash
env/bin/python -m utils.model_comparision
```

## Colab

Use `colab_train.ipynb` for a notebook workflow on Colab GPU. It includes:

1. dependency install
2. data creation/truncation
3. optional W&B login
4. sequential training for adaptive/matched/base
5. checkpoint inspection

