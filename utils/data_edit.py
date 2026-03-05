from pathlib import Path

root = Path(__file__).resolve().parents[1]  # if script is in utils/
data_dir = root / "data"

tiny_path = data_dir / "tiny_stories.txt"
cosmo_path = data_dir / "cosmopedia_wikihow_text.txt"
out_path = data_dir / "cosmopedia_wikihow_text_trunc.txt"

tiny_text = tiny_path.read_text(encoding="utf-8")
cosmo_text = cosmo_path.read_text(encoding="utf-8")

target_chars = len(tiny_text)
trunc_text = cosmo_text[:target_chars]

out_path.write_text(trunc_text, encoding="utf-8")
print("tiny chars:", len(tiny_text))
print("cosmo chars:", len(cosmo_text))
print("trunc chars:", len(trunc_text))
print("saved:", out_path)