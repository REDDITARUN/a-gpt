#!/usr/bin/env bash
set -euo pipefail

echo "==> Starting full pipeline"

# 0) Create venv if missing
if [ ! -d "env" ]; then
  echo "==> Creating virtualenv at ./env"
  python3 -m venv env
fi

# 1) Use venv python/pip
PYTHON="env/bin/python"
PIP="env/bin/pip"

echo "==> Upgrading pip"
"$PIP" install --upgrade pip

echo "==> Installing requirements"
"$PIP" install torch tiktoken datasets tqdm wandb huggingface_hub

# 2) Optional: login checks (uncomment if needed first time)
env/bin/wandb login
env/bin/huggingface-cli login

# 3) Data creation (run inside data/ so files land in data/)
echo "==> Downloading raw datasets"
(
  cd data
  ../env/bin/python get_data.py
)

# 4) Truncate cosmopedia to tiny_stories char length
echo "==> Truncating cosmopedia"
"$PYTHON" utils/data_edit.py

# 5) Quick sanity check files exist
echo "==> Verifying dataset files"
test -f data/tiny_stories.txt
test -f data/cosmopedia_wikihow_text.txt
test -f data/cosmopedia_wikihow_text_trunc.txt
echo "    OK: all dataset files found"

# 6) Train adaptive
echo "==> Training ADAPTIVE"
"$PYTHON" -m train.run \
  --model adaptive \
  --use_wandb \
  --wandb_project gpt-pretrain \
  --wandb_run_name adaptive-mixed-v1

# 7) Train matched
echo "==> Training MATCHED"
"$PYTHON" -m train.run \
  --model matched \
  --use_wandb \
  --wandb_project gpt-pretrain \
  --wandb_run_name matched-mixed-v1

# 8) Train base
echo "==> Training BASE"
"$PYTHON" -m train.run \
  --model base \
  --use_wandb \
  --wandb_project gpt-pretrain \
  --wandb_run_name base-mixed-v1

echo "==> Done. Check checkpoints/ and W&B project."