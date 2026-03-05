# tiny stories
import urllib.request

urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "tiny_stories.txt"
)
print("Data saved to data/tinyshakespeare/tiny_stories.txt")


# HuggingFaceTB/cosmopedia, subset - wikihow
# https://huggingface.co/datasets/HuggingFaceTB/cosmopedia

# pip install -U datasets
from datasets import load_dataset

ds = load_dataset("HuggingFaceTB/cosmopedia", "wikihow", split="train")
text_only = ds.remove_columns([c for c in ds.column_names if c != "text"])


with open("cosmopedia_wikihow_text.txt", "w", encoding="utf-8") as f:
    for row in text_only:
        f.write(row["text"].replace("\n", " ") + "\n")
print("Data saved to cosmopedia_wikihow_text.txt")

print("-"*50)
print("Sample data from cosmopedia_wikihow_text.txt")
# verify the data
with open("cosmopedia_wikihow_text.txt", "r", encoding="utf-8") as f:
    data = f.read()
print(data[:500])
print("length of the data: ", len(data))

print("-"*50)
print("-"*50)

print("Sample data from tiny_stories.txt")
with open("tiny_stories.txt", "r", encoding="utf-8") as f:
    data = f.read()
print(data[:500])
print("length of the data: ", len(data))
