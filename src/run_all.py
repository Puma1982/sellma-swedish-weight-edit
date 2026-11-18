"""Main experiment. Resumable: every generation batch is cached in results/gen/*.jsonl.
Usage: python run_all.py --seed 0 [--budget-seconds N]   (exit code 2 = budget hit, re-run to continue)"""
import argparse, json, sys, time, pathlib
from common import *
import analysis

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--model", default="qwen")
ap.add_argument("--budget-seconds", type=float, default=1e9)
ap.add_argument("--dev-sv", type=int, default=24)     # dev screening subset (CPU budget)
ap.add_argument("--dev-en", type=int, default=10)
ap.add_argument("--dev-tokens", type=int, default=32)
ap.add_argument("--sampled-n", type=int, default=55)  # sv_test prompts used in the sampled-decoding robustness run
args = ap.parse_args()
T0 = time.time()
RES.mkdir(parents=True, exist_ok=True); GEN = RES / "gen"; GEN.mkdir(exist_ok=True)
tok, model = load_model(args.model)
sp = splits()
seed_all(args.seed)
ALPHAS = [0.5, 1.0, 2.0]
LAYERS = layer_grid(model)
hid = model.config.hidden_size

def cached(name, prompts, ctx=None, system=None, sample=False, seed=0, max_new=None, bs=32):
    f = GEN / f"{name}.jsonl"
    rows = [json.loads(l) for l in open(f, encoding="utf-8")] if f.exists() else []
    while len(rows) < len(prompts):
        if time.time() - T0 > args.budget_seconds:
            print(f"[budget] stopping before {name} ({len(rows)}/{len(prompts)})", flush=True); sys.exit(2)
        i = len(rows); chunk = prompts[i:i + bs]
        c = ctx() if ctx else contextlib.nullcontext()
        with c:
            outs, hit = generate(tok, model, chunk, system=system, sample=sample, seed=seed * 1000 + i, bs=bs, max_new=max_new)
        with open(f, "a", encoding="utf-8") as fh:
            for p, o, h in zip(chunk, outs, hit):
                fh.write(json.dumps({"prompt": p, "output": o, "limit_hit": h}, ensure_ascii=False) + "\n")
        rows += [{"prompt": p, "output": o, "limit_hit": h} for p, o, h in zip(chunk, outs, hit)]
        print(f"[{time.time()-T0:6.0f}s] {name} {len(rows)}/{len(prompts)}", flush=True)
    return rows

def sv_rate(rows): return sum(lang_of(r["output"]) == "sv" for r in rows) / len(rows)
def en_rate(rows): return sum(lang_of(r["output"]) == "en" for r in rows) / len(rows)

# ---- directions (contrast sentences only) ----
dirs = {}
for L in LAYERS:
    d, raw = direction(tok, model, sp["contrast_en"], sp["contrast_sv"], L)
    dirs[L] = (d, raw)
json.dump({str(L): {"raw_norm": dirs[L][1]} for L in LAYERS}, open(RES / "directions.json", "w"))

def edit_ctx(L, a, d=None):
    return lambda: weight_edit(model, L, dirs[L][0] if d is None else d, a)
def steer_ctx(L, a):
    return lambda: steering(model, L, a * dirs[L][1] * dirs[L][0])

# ---- DEV: selection (dev prompts only; test untouched) ----
dsv, den = sp["sv_dev"][:args.dev_sv], sp["en_dev"][:args.dev_en]
dev = {}
b_sv = cached("dev_base_sv", dsv, max_new=args.dev_tokens); b_en = cached("dev_base_en", den, max_new=args.dev_tokens)
dev["baseline"] = {"sv": sv_rate(b_sv), "en": en_rate(b_en)}
for meth, mk in (("edit", edit_ctx), ("steer", steer_ctx)):
    for L in LAYERS:
        for a in ALPHAS:
            r_sv = cached(f"dev_{meth}_L{L}_a{a}_sv", dsv, mk(L, a), max_new=args.dev_tokens)
            r_en = cached(f"dev_{meth}_L{L}_a{a}_en", den, mk(L, a), max_new=args.dev_tokens)
            sv, en = sv_rate(r_sv), en_rate(r_en)
            dev[f"{meth}_L{L}_a{a}"] = {"sv": sv, "en": en, "score": sv - max(0.0, dev["baseline"]["en"] - en)}
def best(meth):
    c = [(v["score"], -float(k.split("_a")[1]), -int(k.split("_L")[1].split("_")[0]), k) for k, v in dev.items() if k.startswith(meth + "_")]
    return max(c)[3]
sel = {}
for meth in ("edit", "steer"):
    k = best(meth); sel[meth] = {"L": int(k.split("_L")[1].split("_")[0]), "alpha": float(k.split("_a")[1]), "dev": dev[k]}
json.dump({"dev_grid": dev, "selected": sel, "layers": LAYERS, "alphas": ALPHAS,
           "dev_subset": {"sv": args.dev_sv, "en": args.dev_en, "max_new_tokens": args.dev_tokens}},
          open(RES / "dev_selection.json", "w"), indent=1)
print("selected", sel, flush=True)
Le, ae = sel["edit"]["L"], sel["edit"]["alpha"]
Ls, as_ = sel["steer"]["L"], sel["steer"]["alpha"]

# ---- TEST (greedy) ----
tsv, ten = sp["sv_test"], sp["en_test"]
conds = {
    "baseline": dict(ctx=None, system=None),
    "edited": dict(ctx=edit_ctx(Le, ae), system=None),
    "prompting": dict(ctx=None, system=SYS_SV),
    "steering": dict(ctx=steer_ctx(Ls, as_), system=None),
    "edit_plus_prompting": dict(ctx=edit_ctx(Le, ae), system=SYS_SV),
}
for name, c in conds.items():
    cached(f"test_{name}_sv", tsv, **c); cached(f"test_{name}_en", ten, **c)
for s in range(5):
    rd = random_unit(hid, s)
    cached(f"test_random{s}_sv", tsv, edit_ctx(Le, ae, rd))
cached("test_random0_en", ten, edit_ctx(Le, ae, random_unit(hid, 0)))

# ---- sampled robustness (T=0.7, seeds 0-2) ----
ssv = tsv[:args.sampled_n]
for s in range(3):
    for name in ("baseline", "edited", "prompting"):
        cached(f"sampled_{name}_s{s}_sv", ssv, sample=True, seed=s, **conds[name])

# ---- perplexity on held-out Wikipedia (CC BY-SA) ----
pf = RES / "perplexity.json"
ppl = json.load(open(pf)) if pf.exists() else {}
wiki = [json.loads(l) for l in open(ROOT / "data" / "wiki_heldout.jsonl", encoding="utf-8")]
if SMOKE: wiki = [w for lg in ("sv", "en") for w in [x for x in wiki if x["lang"] == lg][:SMOKE]]
tx = {lg: [w["text"] for w in wiki if w["lang"] == lg] for lg in ("sv", "en")}
for name, ctx in (("baseline", None), ("edited", edit_ctx(Le, ae)), ("random0", edit_ctx(Le, ae, random_unit(hid, 0))),
                  ("random1", edit_ctx(Le, ae, random_unit(hid, 1))), ("steering", steer_ctx(Ls, as_))):
    if name in ppl: continue
    if time.time() - T0 > args.budget_seconds: print("[budget] ppl", flush=True); sys.exit(2)
    with (ctx() if ctx else contextlib.nullcontext()):
        ppl[name] = {lg: perplexity(tok, model, tx[lg]) for lg in tx}
    json.dump(ppl, open(pf, "w"), indent=1); print("ppl", name, ppl[name], flush=True)

# ---- config + analysis ----
cfg = {"model": MODELS[args.model], "seed": args.seed, "max_new_tokens": MAX_NEW, "layers": LAYERS, "alphas": ALPHAS,
       "selected": sel, "edit_stats": edit_stats(model, Le), "system_prompt": SYS_SV,
       "sampled": {"temperature": 0.7, "seeds": [0, 1, 2], "n_prompts": args.sampled_n},
       "random_seeds": [0, 1, 2, 3, 4], "ppl_max_tokens": 96, "args": vars(args),
       "versions": {"torch": torch.__version__, "transformers": __import__("transformers").__version__}}
json.dump(cfg, open(RES / "config.json", "w"), indent=1)
analysis.run()
print("DONE", time.time() - T0)
