"""Stage 3 pilot (dev only): baseline Swedish-answer rate. Switch to fallback model if >=95%."""
import json, sys, time
from common import *
res = {}
for key in ["qwen", "smol"]:
    t = time.time()
    tok, model = load_model(key)
    sp = splits()
    outs, hit = generate(tok, model, sp["sv_dev"])
    langs = [lang_of(o) for o in outs]
    rate = sum(l == "sv" for l in langs) / len(langs)
    res[key] = {"model": MODELS[key], "sv_dev_swedish_rate": rate, "limit_hit_share": sum(hit) / len(hit),
                "lang_counts": {l: langs.count(l) for l in set(langs)}, "secs": time.time() - t,
                "samples": outs[:5]}
    print(key, rate, res[key]["lang_counts"], res[key]["secs"], flush=True)
    if rate < 0.95: break
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "pilot.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
