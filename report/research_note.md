# Can a rank-1 weight edit make a small model answer in Swedish? A controlled negative result

Haithem Grissander · SE.LLMA candidate assignment · model `Qwen/Qwen2.5-0.5B-Instruct` @ `7ae55760…` · seed 0 · CPU only

## 1. Critique of "Sparks of Weight-Space Programming" (Takehi et al., MIT CSAIL 2026) — *DRAFT — Haithem to edit*
*(Written from the points in my work order and the assignment text; the article should be re-read before this is relied on.)*

**Strongest evidence.** Kinetix: 1,024/1,024 environments fixed with a one-shot edit, and MinAtar policies produced from scratch. These are clean, countable outcomes where "it worked" is not a judgement call.
**Main limits.**
(i) *Feedback-optimised, not read from the weights.* The Qwen headline used 50 passes of model feedback, so the edit is found by search against the behaviour, not derived by reasoning about weights. That supports "an editor with a feedback loop can find edits", not "weights are interpretable/programmable".
(ii) *Failure removal vs. new capability.* About 93 % of baseline outputs hit the token limit, so the gain is largely a failure (runaway generation) being removed, not a skill being added.
(iii) *Missing baselines:* no decoding changes, steering or fine-tuning comparison, so "weight edit" is never compared with cheaper ways to change the same behaviour.
(iv) *Mixed signals:* non-monotonic IFEval, a null result on Gemma 270M, sycophancy needing an extra attention tool.
(v) *Scale and rigour:* tiny networks, one closed editor model, no variance across runs.
**Justified:** the edit *interface* can change behaviour. **Not justified:** that this is "programming" by understanding weights, or that it generalises to useful capabilities in language models.

## 2. Hypotheses (fixed before the test run)
Question: can one rank-1 edit of one MLP output matrix make a small instruction model **answer Swedish prompts in Swedish** (language-switching failure), and is that a new capability or removal of a failure? It matters for SE.LLMA because small models often answer in English, and a cheap, auditable fix would be valuable.
- **H1 (efficacy).** The edit raises the Swedish-answer rate on held-out Swedish prompts above baseline *and* above a random same-norm edit (exact McNemar p<0.05, paired). *Challenged if* edit ≈ baseline or ≈ random, or if prompting/steering do at least as well.
- **H2 (specificity / failure removal).** Any gain comes without change in English-answer rate, held-out perplexity or Swedish non-word rate, and concentrates on prompts where the model already *can* answer in Swedish (fixed by prompting too). *Challenged if* those metrics degrade, or the gain is not overlapping with prompting-fixable prompts.

## 3. Experiment
Direction d = normalised mean residual (English − Swedish) at the output of layer L over 64 parallel sentence pairs (separate from all prompts). Edit: `W_down(L) ← W_down(L) − α·d·(dᵀW_down(L))`, i.e. project d out of the MLP output. Layer 6 changes 4.36 M weights (0.88 % of 494 M; 896 effective degrees of freedom; relative Frobenius change 7.8 %). (L, α) ∈ {6,9,12,15,18} × {0.5,1,2} chosen **on dev prompts only** by Swedish rate minus English-retention loss. **This is feedback-optimised selection, not reading the weights.** Because of the CPU budget (≈5 s per generation), dev screening used 24 Swedish + 10 English dev prompts and 32 new tokens. Selected: **L=6, α=2**.
Test (110 Swedish, 60 English prompts; greedy, 64 new tokens): (a) baseline, (b) edit, (c) random unit direction, same L/α, 5 seeds, (d) system prompt "Svara alltid på svenska.", (e) inference-time steering `−α·‖μ_en−μ_sv‖·d` at layer L (own dev-tuned: L=15, α=0.5), (f) edit + prompting. Language by `langid` (strict `sv`; a lenient sv/no/da rate is also reported). Regressions: English answer rate, Swedish/English perplexity on 150+150 Wikipedia paragraphs, hunspell sv_SE non-word rate, length. Wilson 95 % CIs; exact McNemar on paired prompts. Robustness: T=0.7, 3 seeds, 55 test prompts. Total runtime ≈ 3 h on a shared 4-core Intel Mac.

## 4. Results (Swedish test prompts, n=110, greedy)
| condition | Swedish-answer rate [95 % CI] | English answers on EN prompts | non-word % | ppl sv / en |
|---|---|---|---|---|
| baseline | 92.7 % [86.3, 96.3] | 100 % | 4.5 | 71.5 / 20.1 |
| **rank-1 edit (L6, α2)** | **92.7 % [86.3, 96.3]** | 100 % | 4.2 | 71.8 / 20.2 |
| random edit (5 seeds) | 92.9 % ± 0.4 | 100 % (seed 0) | 4.4 | 71.8 / 20.1 |
| prompting | 98.2 % [93.6, 99.5] | **93.3 %** | 3.0 | – |
| steering | 10.0 % [5.7, 17.0] | 88.3 % | 18.5 | 1881 / 255 |
| edit + prompting | 99.1 % [95.0, 99.8] | 93.3 % | 2.9 | – |

- **H1 is not supported.** Edited and baseline have the *same* language label on all 110 prompts (0 discordant pairs, p=1.0), although the generated text changes for 99 of 110 prompts (random edits change fewer). Edit vs random: p≥1.0 for every seed. Prompting vs edit: 7 vs 1 discordant, p=0.07 (not significant at n=110, but the effect is in prompting's favour). Sampled decoding agrees: 85.5 % baseline, 85.5 % edit, 93.9 % prompting (3 seeds).
- **Failure removal, not capability (H2, partly supported).** Only 8 baseline prompts fail (3 English, 5 no/nb). The English ones are refusals: *"Hur lär jag barn att cykla?" → "I'm sorry, but I can't assist with that."* in baseline **and** edit; prompting answers in Swedish (7 of 8 failures fixed, 1 newly broken). The Swedish itself is poor in every condition (see `results/readable_samples.md`) and the non-word rate (≈3–4.5 %) barely moves: the intervention that works switches *language selection*, it does not improve Swedish.
- **Costs.** The edit is harmless on perplexity (+0.3 %) and English retention — but also does nothing. Prompting costs 6.7 pts of English answering on English prompts (it forces Swedish everywhere). Steering with the raw difference vector wrecks the model at every tested strength.

## 5. Interpretation, limitations, recommendation — *DRAFT — Haithem to edit*
**Interpretation.** The English−Swedish contrast direction, projected out of one MLP output, alters text but not which language the model chooses; the observed failure here (English refusals) is plausibly driven by a refusal/alignment behaviour that a language-contrast direction does not capture. The only intervention with a visible gain was the cheapest one (a system prompt), which supports the "failure removal" reading and mirrors the article's gap: no inference-time baseline.
**Limitations.** Ceiling: baseline already 92.7 %, only 8 failing prompts, so the test can only detect large effects (CI ±5 pts); one 0.5 B model, one layer-wise direction type, one edit family; dev subset small (the choice of α=2 rested on 1 prompt, i.e. selection was essentially noise, which if anything favours the edit); steering scale was my choice (raw norm × α, smallest α 0.5) and a finer, smaller grid might work; `langid` confuses Swedish/Norwegian (5 labels `no/nb`); synthetic prompts, answer quality not human-rated; this is a feedback-selected edit, not weight-reasoned.
**Recommendation.** Do **not** invest in this specific edit for Swedish language-switching; use prompting/data fixes first. Weight-space editing stays interesting only if it beats those baselines on a harder target.
**Most informative next experiment.** A target with real headroom (a base/small model with ≥30 % English drift, or Swedish *quality* such as a Swedish-specific refusal/format failure), comparing: system prompt, a finer-scaled steering sweep, LoRA fine-tuning on a few hundred Swedish examples, and the edit — with ≥300 test prompts, direction learned from refusal-vs-answer contrast rather than language contrast, and multiple seeds/models.
