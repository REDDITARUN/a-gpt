from pathlib import Path
import urllib.request

# Always save into this repository's data directory.
data_dir = Path(__file__).resolve().parent
tiny_out = data_dir / "tiny_stories.txt"
cosmo_out = data_dir / "cosmopedia_wikihow_text.txt"

urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    str(tiny_out),
)
print(f"Data saved to {tiny_out}")


# HuggingFaceTB/cosmopedia, subset - wikihow
# https://huggingface.co/datasets/HuggingFaceTB/cosmopedia

# pip install -U datasets
from datasets import load_dataset

ds = load_dataset("HuggingFaceTB/cosmopedia", "wikihow", split="train")
text_only = ds.remove_columns([c for c in ds.column_names if c != "text"])


with cosmo_out.open("w", encoding="utf-8") as f:
    for row in text_only:
        f.write(row["text"].replace("\n", " ") + "\n")
print(f"Data saved to {cosmo_out}")

print("-"*50)
print(f"Sample data from {cosmo_out.name}")
# verify the data
with cosmo_out.open("r", encoding="utf-8") as f:
    data = f.read()
print(data[:500])
print("length of the data: ", len(data))

print("-"*50)
print("-"*50)

print(f"Sample data from {tiny_out.name}")
with tiny_out.open("r", encoding="utf-8") as f:
    data = f.read()
print(data[:500])
print("length of the data: ", len(data))
