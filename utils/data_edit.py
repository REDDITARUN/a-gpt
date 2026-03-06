from pathlib import Path

import tiktoken


TARGET_TOTAL_TOKENS = 300_000_000

root = Path(__file__).resolve().parents[1]
data_dir = root / "data"

tiny_path = data_dir / "tiny_stories.txt"
cosmo_path = data_dir / "cosmopedia_wikihow_text.txt"
wiki_path = data_dir / "wikitext2_text.txt"
code_path = data_dir / "python_code_text.txt"

cosmo_out = data_dir / "cosmopedia_wikihow_text_trunc.txt"
wiki_out = data_dir / "wikitext2_text_trunc.txt"
code_out = data_dir / "python_code_text_trunc.txt"
mixed_out = data_dir / "mixed_300m_tokens.txt"

enc = tiktoken.get_encoding("gpt2")

tiny_text = tiny_path.read_text(encoding="utf-8")
cosmo_text = cosmo_path.read_text(encoding="utf-8")
wiki_text = wiki_path.read_text(encoding="utf-8")
code_text = code_path.read_text(encoding="utf-8")

tiny_tokens = enc.encode(tiny_text)
cosmo_tokens = enc.encode(cosmo_text)
wiki_tokens = enc.encode(wiki_text)
code_tokens = enc.encode(code_text)

tiny_n = len(tiny_tokens)
if tiny_n >= TARGET_TOTAL_TOKENS:
    raise ValueError(
        f"TinyStories already has {tiny_n:,} tokens >= target {TARGET_TOTAL_TOKENS:,}. "
        "Increase target or change policy."
    )

remaining = TARGET_TOTAL_TOKENS - tiny_n
per_source = remaining // 3

if len(cosmo_tokens) < per_source or len(wiki_tokens) < per_source or len(code_tokens) < per_source:
    raise ValueError(
        "One of cosmo/wiki/code has fewer tokens than required equal share. "
        f"required each={per_source:,}, got cosmo={len(cosmo_tokens):,}, "
        f"wiki={len(wiki_tokens):,}, code={len(code_tokens):,}"
    )

# Put any remainder into cosmopedia to hit exact target token count.
cosmo_take = per_source + (remaining - per_source * 3)
wiki_take = per_source
code_take = per_source

cosmo_text_trunc = enc.decode(cosmo_tokens[:cosmo_take])
wiki_text_trunc = enc.decode(wiki_tokens[:wiki_take])
code_text_trunc = enc.decode(code_tokens[:code_take])

cosmo_out.write_text(cosmo_text_trunc, encoding="utf-8")
wiki_out.write_text(wiki_text_trunc, encoding="utf-8")
code_out.write_text(code_text_trunc, encoding="utf-8")

mixed_text = "\n\n".join([tiny_text, cosmo_text_trunc, wiki_text_trunc, code_text_trunc])
mixed_out.write_text(mixed_text, encoding="utf-8")

final_tokens = len(enc.encode(mixed_text))

print(f"target total tokens: {TARGET_TOTAL_TOKENS:,}")
print(f"tiny tokens (kept full): {tiny_n:,}")
print(f"remaining tokens budget: {remaining:,}")
print(f"cosmo tokens used: {cosmo_take:,}")
print(f"wiki tokens used: {wiki_take:,}")
print(f"code tokens used: {code_take:,}")
print(f"final mixed tokens (re-tokenized): {final_tokens:,}")
print("saved:", cosmo_out)
print("saved:", wiki_out)
print("saved:", code_out)
print("saved:", mixed_out)