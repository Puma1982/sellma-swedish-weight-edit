"""Compute metrics, paired tests, figures and readable samples from cached generations."""
import json, random, statistics as st
from common import *

SCAND = {"sv", "no", "da", "nn", "nb"}
MAIN = ["baseline", "edited", "prompting", "steering", "edit_plus_prompting"]

def load(name):
    f = RES / "gen" / f"{name}.jsonl"
    return [json.loads(l) for l in open(f, encoding="utf-8")] if f.exists() else None

def ci(k, n): lo, hi = wilson(k, n); return [round(lo, 4), round(hi, 4)]

def metrics(rows, target):
    langs = [lang_of(r["output"]) for r in rows]
    n = len(rows); k = sum(l == target for l in langs)
    m = {"n": n, "rate": k / n, "ci95": ci(k, n), "lang_counts": {l: langs.count(l) for l in sorted(set(langs))},
         "mean_words": sum(len(r["output"].split()) for r in rows) / n,
         "limit_hit_share": sum(r["limit_hit"] for r in rows) / n}
    if target == "sv":
        ks = sum(l in SCAND for l in langs); m["lenient_scandinavian_rate"] = ks / n
        nw = [nonword_rate(r["output"]) for r, l in zip(rows, langs) if l == "sv"]
        tot = sum(c for _, c in nw); bad = sum((x or 0) * c for x, c in nw)
        m["nonword_rate_in_sv_answers"] = bad / tot if tot else None
    return m, langs

def run():
    out = {"greedy_test": {}, "paired_tests": {}, "random": {}, "sampled": {}, "failure_removal": {}}
    L = {}
    for c in MAIN:
        sv, en = load(f"test_{c}_sv"), load(f"test_{c}_en")
        if sv is None: continue
        m, l = metrics(sv, "sv"); L[c] = l
        out["greedy_test"][c] = {"sv_prompts": m}
        if en: out["greedy_test"][c]["en_prompts_english_rate"] = metrics(en, "en")[0]
    # random controls
    rl, rr = [], []
    for s in range(5):
        r = load(f"test_random{s}_sv")
        if r is None: continue
        m, l = metrics(r, "sv"); rl.append(l); rr.append(m)
    if rr:
        out["random"] = {"per_seed_rate": [m["rate"] for m in rr], "mean_rate": st.mean(m["rate"] for m in rr),
                         "sd_rate": st.pstdev(m["rate"] for m in rr) if len(rr) > 1 else 0.0,
                         "mean_nonword": st.mean(m["nonword_rate_in_sv_answers"] or 0 for m in rr),
                         "mean_words": st.mean(m["mean_words"] for m in rr),
                         "pooled_ci95": ci(sum(round(m["rate"] * m["n"]) for m in rr), sum(m["n"] for m in rr))}
        r0 = load("test_random0_en")
        if r0: out["random"]["seed0_en_prompts_english_rate"] = metrics(r0, "en")[0]
    # paired tests (Swedish answer yes/no per prompt)
    def pair(a, b):
        n01, n10, p = mcnemar(a, b); return {"only_second": n01, "only_first": n10, "p": p}
    isv = lambda ls: [x == "sv" for x in ls]
    if "edited" in L:
        for o in ("baseline", "prompting", "steering"):
            if o in L: out["paired_tests"][f"edited_vs_{o}"] = pair(isv(L["edited"]), isv(L[o]))
        for s, l in enumerate(rl): out["paired_tests"][f"edited_vs_random{s}"] = pair(isv(L["edited"]), isv(l))
    if "steering" in L and "baseline" in L: out["paired_tests"]["steering_vs_baseline"] = pair(isv(L["steering"]), isv(L["baseline"]))
    if "prompting" in L and "baseline" in L: out["paired_tests"]["prompting_vs_baseline"] = pair(isv(L["prompting"]), isv(L["baseline"]))
    # failure removal: among baseline non-Swedish prompts, how many does each intervention fix?
    if "baseline" in L:
        fail = [i for i, x in enumerate(L["baseline"]) if x != "sv"]
        fixed = {c: {i for i in fail if L[c][i] == "sv"} for c in L if c != "baseline"}
        broke = {c: sum(1 for i, x in enumerate(L["baseline"]) if x == "sv" and L[c][i] != "sv") for c in L if c != "baseline"}
        out["failure_removal"] = {"baseline_failures": len(fail), "fixed": {c: len(v) for c, v in fixed.items()},
                                  "newly_broken": broke}
        if "edited" in fixed and "prompting" in fixed:
            out["failure_removal"]["fixed_by_edit_and_prompting"] = len(fixed["edited"] & fixed["prompting"])
    # sampled robustness
    for c in ("baseline", "edited", "prompting"):
        rates = []
        for s in range(3):
            r = load(f"sampled_{c}_s{s}_sv")
            if r: rates.append(metrics(r, "sv")[0]["rate"])
        if rates: out["sampled"][c] = {"per_seed_rate": rates, "mean": st.mean(rates)}
    pf = RES / "perplexity.json"
    if pf.exists(): out["perplexity"] = json.load(open(pf))
    json.dump(out, open(RES / "metrics.json", "w"), indent=1)
    tables(out); readable(); figures(out, rr)

def tables(out):
    P = out.get("perplexity", {})
    lines = ["| condition | Swedish ppl | English ppl |", "|---|---|---|"]
    for k, v in P.items(): lines.append(f"| {k} | {v['sv']:.2f} | {v['en']:.2f} |")
    (RES / "perplexity_table.md").write_text("\n".join(lines) + "\n")
    g = out["greedy_test"]
    L2 = ["| condition | Swedish-answer rate [95% CI] | lenient (sv/no/da) | English rate on EN prompts | hunspell non-word % | mean words | hit 64-token limit |", "|---|---|---|---|---|---|---|"]
    for c, v in g.items():
        s = v["sv_prompts"]; e = v.get("en_prompts_english_rate", {})
        L2.append(f"| {c} | {100*s['rate']:.1f}% [{100*s['ci95'][0]:.1f}, {100*s['ci95'][1]:.1f}] | {100*s['lenient_scandinavian_rate']:.1f}% | "
                  f"{100*e.get('rate', float('nan')):.1f}% | {100*(s['nonword_rate_in_sv_answers'] or 0):.1f} | {s['mean_words']:.1f} | {100*s['limit_hit_share']:.0f}% |")
    r = out.get("random")
    if r:
        L2.append(f"| random edit (5 seeds, mean±sd) | {100*r['mean_rate']:.1f}% ± {100*r['sd_rate']:.1f} | – | "
                  f"{100*r.get('seed0_en_prompts_english_rate', {}).get('rate', float('nan')):.1f}% (seed 0) | {100*r['mean_nonword']:.1f} | {r['mean_words']:.1f} | – |")
    (RES / "results_table.md").write_text("\n".join(L2) + "\n")

def readable(n=20):
    idx = sorted(random.Random(0).sample(range(len(splits()["sv_test"])), min(n, len(splits()["sv_test"]))))
    parts = ["# Readable samples (Swedish test prompts, same 20 prompts in every condition; langid label in brackets)\n"]
    for c in MAIN + ["random0"]:
        rows = load(f"test_{c}_sv")
        if not rows: continue
        parts.append(f"\n## {c}\n")
        for i in idx:
            if i < len(rows):
                o = rows[i]["output"].replace("\n", " ")
                parts.append(f"- **{rows[i]['prompt']}**  [{lang_of(rows[i]['output'])}] {o}")
    (RES / "readable_samples.md").write_text("\n".join(parts) + "\n", encoding="utf-8")

def figures(out, rr):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    (ROOT / "figures").mkdir(exist_ok=True)
    names, vals, lo, hi = [], [], [], []
    for c, v in out["greedy_test"].items():
        s = v["sv_prompts"]; names.append(c.replace("_", "\n")); vals.append(100 * s["rate"]); lo.append(100 * (s["rate"] - s["ci95"][0])); hi.append(100 * (s["ci95"][1] - s["rate"]))
        if c == "edited" and out.get("random"):
            r = out["random"]; names.append("random edit\n(5 seeds, pooled)"); vals.append(100 * r["mean_rate"])
            lo.append(100 * (r["mean_rate"] - r["pooled_ci95"][0])); hi.append(100 * (r["pooled_ci95"][1] - r["mean_rate"]))
    fig, ax = plt.subplots(figsize=(8, 4))
    cols = ["#7f7f7f" if n.startswith("baseline") else "#2a6fbb" if "random" not in n else "#c9a227" for n in names]
    ax.bar(range(len(vals)), vals, yerr=[lo, hi], capsize=4, color=cols)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Swedish-answer rate on Swedish test prompts (%)"); ax.set_ylim(0, 105)
    for i, v in enumerate(vals): ax.text(i, v + 3 + hi[i] * 0.0, f"{v:.0f}", ha="center", fontsize=8)
    ax.set_title("Greedy decoding, n=110 test prompts, Wilson 95% CI", fontsize=10)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    plt.tight_layout(); plt.savefig(ROOT / "figures" / "swedish_rate_by_condition.png", dpi=160); plt.close()
    P = out.get("perplexity")
    if P:
        fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
        for ax, lg in zip(axs, ("sv", "en")):
            ks = list(P); ax.bar(ks, [P[k][lg] for k in ks], color="#2a6fbb")
            ax.set_title(f"{'Swedish' if lg=='sv' else 'English'} held-out perplexity", fontsize=10); ax.tick_params(axis="x", labelsize=7, rotation=30)
            for s in ("top", "right"): ax.spines[s].set_visible(False)
        plt.tight_layout(); plt.savefig(ROOT / "figures" / "perplexity.png", dpi=160); plt.close()

if __name__ == "__main__": run()
