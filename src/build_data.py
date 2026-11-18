"""Build fixed data splits (seed 0) and pull held-out Wikipedia paragraphs.
Prompts/sentences are synthetic (written with AI help, disclosed in AI_USAGE.md)."""
import json, random, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "data"

def lines(p):
    return [l.strip() for l in (D / p).read_text(encoding="utf-8").splitlines() if l.strip()]

def split_prompts():
    rng = random.Random(0)
    sv = lines("prompts_sv.txt"); rng.shuffle(sv); sv = sv[:160]
    en = lines("prompts_en.txt"); rng.shuffle(en); en = en[:80]
    out = {"sv_dev": sv[:50], "sv_test": sv[50:], "en_dev": en[:20], "en_test": en[20:]}
    pairs = [l.split("\t") for l in lines("parallel_en_sv.tsv")][:64]
    out["contrast_en"] = [p[0] for p in pairs]; out["contrast_sv"] = [p[1] for p in pairs]
    (D / "splits.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print({k: len(v) for k, v in out.items()})

def wiki(lang, n=150, seed=0):
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=2000)
    got = []
    for ex in ds:
        paras = [p.strip() for p in ex["text"].split("\n") if 400 <= len(p.strip()) <= 1500]
        if not paras: continue
        got.append({"title": ex["title"], "url": ex["url"], "text": paras[0], "lang": lang})
        if len(got) >= n: break
    return got

if __name__ == "__main__":
    split_prompts()
    rows = wiki("sv") + wiki("en")
    with open(D / "wiki_heldout.jsonl", "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wiki paragraphs", len(rows))
