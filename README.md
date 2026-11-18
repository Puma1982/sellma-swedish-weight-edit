# Targeted weight edit vs. controls for Swedish answer-language (SE.LLMA candidate assignment)

Does a rank-1 weight edit that projects an "English − Swedish" direction out of one MLP output matrix make a small
instruction model answer in Swedish more often, compared with (a) doing nothing, (b) a random rank-1 edit,
(c) a one-line system prompt, and (d) inference-time activation steering? And does it cost anything (English
answers, held-out perplexity, Swedish spelling)?

**Start with `report/research_note.md`** (1–2 pages), then `report/report.pdf` (longer). Everything is own work with AI help (see `AI_USAGE.md`).

## Reproduce
```bash
python3.10 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt && python src/build_data.py   # data is already committed; this step is optional
cd src && python run_all.py --seed 0                          # pilot results are in results/pilot.json (python pilot.py)
python build_report.py                                         # report/report.pdf (needs Chrome)
```
Equivalent one-liner from the repo root: `pip install -r requirements.txt && python src/run_all.py --seed 0`
(run from `src/` or with `PYTHONPATH=src`; the script is resumable: it caches every generation batch in `results/gen/`
and can be stopped/restarted with `--budget-seconds`).

- **Hardware used:** Intel Mac, 4 cores, CPU only, fp32, heavily shared with other apps. Runtime of the full run: see the research note (several hours).
- **Model:** `Qwen/Qwen2.5-0.5B-Instruct` @ `7ae557604adf67be50417f59c2c2f167def9a775` (Apache-2.0). Seed 0; greedy decoding; 64 new tokens.
- Dev-grid screening used a reduced dev subset and 32 tokens, and the sampled-decoding check used 55 test prompts, to fit the CPU budget (`results/config.json`).

## Layout
`src/` code · `data/` prompts, splits, Wikipedia paragraphs, dictionary (+ `data/README.md` with licences) ·
`results/` generations per condition (`gen/*.jsonl`), `metrics.json`, `config.json`, `dev_selection.json`, tables, `readable_samples.md` ·
`figures/` · `report/` · `AI_USAGE.md`.

Licence: MIT for code; Wikipedia text CC BY-SA 4.0; dictionary LGPL-3.0.
