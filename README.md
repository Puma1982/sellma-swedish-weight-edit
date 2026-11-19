# Targeted weight edit vs. controls for Swedish answer-language (SE.LLMA candidate assignment)

Does a rank-1 weight edit that projects an "English − Swedish" direction out of one MLP output matrix make a small
instruction model answer in Swedish more often, compared with (a) doing nothing, (b) a random rank-1 edit,
(c) a one-line system prompt, and (d) inference-time activation steering? And does it cost anything (English
answers, held-out perplexity, Swedish spelling)?

**Start with `report/research_note.md`** (1–2 pages), then `report/report.pdf` (longer). Everything is own work with AI help (see `AI_USAGE.md`).

## Reproduce
From the repo root (Python 3.10):
```bash
python3.10 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt && python run_all.py --seed 0
```
This downloads `Qwen/Qwen2.5-0.5B-Instruct` at the pinned revision, then runs direction extraction, dev-set selection,
test-set generation for all conditions (baseline, weight edit, random edits, prompting, steering, edit+prompting),
the sampled-decoding check, held-out perplexity and the metrics, and writes everything to `results/`.
The script is resumable: every generation batch is cached in `results/gen/` (the committed cache means a re-run
reuses it; delete `results/gen/` to recompute from scratch). `--budget-seconds N` stops after N seconds; re-run to continue.

Optional: `python src/build_data.py` rebuilds the data (already committed); `python src/pilot.py` the pilot;
`python src/build_report.py` rebuilds `report/report.pdf` (needs Chrome/Chromium).

- **Runtime:** about 3 hours from scratch (10,622 s in the recorded run, `results/run_log_clean.txt`).
- **Hardware used:** Intel Mac, 4 cores, CPU only, fp32, shared with other apps. No GPU needed.
- **Model:** `Qwen/Qwen2.5-0.5B-Instruct` @ `7ae557604adf67be50417f59c2c2f167def9a775` (Apache-2.0). Seed 0; greedy decoding; 64 new tokens.
- Dev-grid screening used a reduced dev subset and 32 tokens, and the sampled-decoding check used 55 test prompts, to fit the CPU budget (`results/config.json`).

## Where the results are
`results/metrics.json`, `results/results_table.md`, `results/perplexity_table.md` (tables) · `results/gen/*.jsonl` (every generation) ·
`results/config.json` (model revision, seed, selected layer/alpha) · `figures/` · `results/readable_samples.md` (examples) ·
`report/research_note.md` (read first) and `report/report.pdf`.

## Layout
`src/` code · `data/` prompts, splits, Wikipedia paragraphs, dictionary (+ `data/README.md` with licences) ·
`results/` generations per condition (`gen/*.jsonl`), `metrics.json`, `config.json`, `dev_selection.json`, tables, `readable_samples.md` ·
`figures/` · `report/` · `AI_USAGE.md`.

Licence: MIT for code; Wikipedia text CC BY-SA 4.0; dictionary LGPL-3.0.
