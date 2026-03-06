from pathlib import Path
import urllib.request

from datasets import load_dataset


def write_text_rows(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for text in rows:
            cleaned = text.replace("\n", " ").strip()
            if cleaned:
                f.write(cleaned + "\n")


def preview(path: Path, n_chars: int = 500):
    text = path.read_text(encoding="utf-8")
    print("-" * 50)
    print(f"Sample data from {path.name}")
    print(text[:n_chars])
    print("length of the data:", len(text))


def file_ready(path: Path, min_bytes: int = 1024) -> bool:
    return path.exists() and path.stat().st_size >= min_bytes


def main():
    # Always save into this repository's data directory.
    data_dir = Path(__file__).resolve().parent
    tiny_out = data_dir / "tiny_stories.txt"
    cosmo_out = data_dir / "cosmopedia_wikihow_text.txt"
    wiki_out = data_dir / "wikitext2_text.txt"
    code_out = data_dir / "python_code_text.txt"

    # 1) Tiny Shakespeare baseline text
    if file_ready(tiny_out):
        print(f"Using existing file: {tiny_out}")
    else:
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            str(tiny_out),
        )
        print(f"Data saved to {tiny_out}")

    # 2) Cosmopedia wikihow subset
    if file_ready(cosmo_out, min_bytes=1024 * 1024):
        print(f"Using existing file: {cosmo_out}")
    else:
        ds_cosmo = load_dataset("HuggingFaceTB/cosmopedia", "wikihow", split="train")
        write_text_rows(cosmo_out, (row["text"] for row in ds_cosmo))
        print(f"Data saved to {cosmo_out}")

    # 3) WikiText-2 (factual/news style)
    if file_ready(wiki_out, min_bytes=1024 * 1024):
        print(f"Using existing file: {wiki_out}")
    else:
        ds_wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        write_text_rows(wiki_out, (row["text"] for row in ds_wiki))
        print(f"Data saved to {wiki_out}")

    # 4) CodeSearchNet Python subset (code style)
    # Keep only raw function/code bodies for a code-heavy pretraining mix.
    if file_ready(code_out, min_bytes=1024 * 1024):
        print(f"Using existing file: {code_out}")
    else:
        ds_code = load_dataset("code_search_net", "python", split="train")
        write_text_rows(code_out, (row["whole_func_string"] for row in ds_code))
        print(f"Data saved to {code_out}")

    # Quick previews
    preview(cosmo_out)
    preview(tiny_out)
    preview(wiki_out)
    preview(code_out)


if __name__ == "__main__":
    main()
