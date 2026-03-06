from pathlib import Path

import tiktoken


TARGET_TOTAL_TOKENS = 300_000_000
NON_TINY_WEIGHTS = {
    "cosmo": 0.40,
    "wiki": 0.20,
    "code": 0.40,
}

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

def allocate_with_caps(total_budget: int, available: dict, weights: dict):
    allocated = {k: 0 for k in available}
    active = set(available.keys())
    remaining = total_budget

    while remaining > 0 and active:
        active_weight_sum = sum(weights[k] for k in active)
        if active_weight_sum <= 0:
            # fallback: split evenly among active sources
            for k in active:
                weights[k] = 1.0
            active_weight_sum = float(len(active))

        progressed = False
        for key in list(active):
            share = int(remaining * (weights[key] / active_weight_sum))
            if share <= 0:
                share = 1
            cap_left = available[key] - allocated[key]
            take = min(share, cap_left, remaining)
            if take > 0:
                allocated[key] += take
                remaining -= take
                progressed = True
            if allocated[key] >= available[key]:
                active.remove(key)
            if remaining <= 0:
                break

        if not progressed:
            break

    return allocated, remaining


def main():
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

    remaining_budget = TARGET_TOTAL_TOKENS - tiny_n
    available = {
        "cosmo": len(cosmo_tokens),
        "wiki": len(wiki_tokens),
        "code": len(code_tokens),
    }
    allocated, unfilled = allocate_with_caps(remaining_budget, available, NON_TINY_WEIGHTS.copy())
    used_non_tiny = sum(allocated.values())

    cosmo_take = allocated["cosmo"]
    wiki_take = allocated["wiki"]
    code_take = allocated["code"]

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
    print(f"remaining budget after tiny: {remaining_budget:,}")
    print(
        f"available non-tiny tokens: cosmo={available['cosmo']:,}, "
        f"wiki={available['wiki']:,}, code={available['code']:,}"
    )
    print(f"non-tiny tokens used: {used_non_tiny:,}")
    print(f"unfilled non-tiny budget: {unfilled:,}")
    print(f"cosmo tokens used: {cosmo_take:,}")
    print(f"wiki tokens used: {wiki_take:,}")
    print(f"code tokens used: {code_take:,}")
    print(f"final mixed tokens (re-tokenized): {final_tokens:,}")
    print("saved:", cosmo_out)
    print("saved:", wiki_out)
    print("saved:", code_out)
    print("saved:", mixed_out)


if __name__ == "__main__":
    main()