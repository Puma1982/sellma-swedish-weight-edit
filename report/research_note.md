# Can a rank-1 weight edit make a small model answer in Swedish? A controlled negative result

Haithem Grissander · SE.LLMA candidate assignment · model `Qwen/Qwen2.5-0.5B-Instruct` (revision pinned in `results/`) · seed 0 · CPU only

## 1. Critical assessment of *Sparks of Weight-Space Programming*

The article argues that a strong language model can read, change and even build neural-network behaviour directly in weight space, without gradient-based training. The most convincing results come from small, well-defined systems with clear success measures:
- A failing Kinetix control policy (171,664 weights) was repaired with a single edit. After the edit it succeeded in 1,024 of 1,024 test episodes; before, it had failed all 256 evaluation episodes.
- MinAtar policies were written from zero-initialised weights, without play, and beat a 100M-step PPO baseline in three games.
- Small networks were identified from their weights alone.

These results support a narrow conclusion: useful weight-level edits are possible in small networks.

The headline language-model result is weaker, for four reasons:
1. **It relies on feedback, not reading.** The Qwen3.5-0.8B result (MATH-500 from 7.9 % to 48.1 %) needed about 50 rounds of feedback from running the model; after 10 and 25 rounds the score was only 12–14 %. This is closer to search guided by behaviour than to reasoning about weights.
2. **It removes a failure rather than adding a capability.** Before the edit, 93 % of answers hit the token limit because the model got stuck in loops; after it, 33 %. The model could already solve many problems once it stopped looping.
3. **It is not compared with simpler methods.** There are no comparisons with prompting, decoding settings, activation steering, LoRA or fine-tuning, so it is unclear whether weight editing has a real advantage.
4. **The evidence is limited.** Most networks are small, one closed editor model is used, and run-to-run variance is not reported. Some results are inconsistent: IFEval goes 49.0 → 39.9 → 62.3, and a Gemma 3 270M follow-up showed no clear gain. Independent replication is limited.

The evidence therefore justifies three claims: a strong LLM can sometimes find useful structure in weights, edits can suppress a specific failure mode, and small networks can be changed without retraining. It does not yet justify the claims that LLMs can read the behaviour of large language models from weights alone, that weight editing creates new capabilities, that it can replace data and training at foundation-model scale, or that the gains come from understanding rather than from feedback-guided search.

## 2. Hypotheses (fixed before the test run)
**Question.** Can one rank-1 edit of one MLP output matrix make a small instruction model answer Swedish prompts in Swedish (a language-switching failure)? And is that a new capability or the removal of a failure? It matters for SE.LLMA because small models often answer in English, and a cheap, auditable fix would be valuable.
- **H1 (efficacy).** The edit raises the Swedish-answer rate on held-out Swedish prompts above both the baseline and a random edit of the same norm (exact McNemar p < 0.05, paired). *Challenged if* the edit ≈ baseline or ≈ random, or if prompting or steering does at least as well.
- **H2 (specificity / failure removal).** Any gain comes without change in the English-answer rate, held-out perplexity or Swedish non-word rate, and it concentrates on prompts where the model already *can* answer in Swedish (i.e. those also fixed by prompting). *Challenged if* those metrics degrade, or if the gain does not overlap with the prompting-fixable prompts.

## 3. Experiment
**The edit.**
- Direction d = the normalised mean residual (English − Swedish) at the output of layer L, over 64 parallel sentence pairs that are kept separate from all prompts.
- Edit: `W_down(L) ← W_down(L) − α·d·(dᵀW_down(L))`, i.e. d is projected out of the MLP output.
- At layer 6 this changes 4.36 M weights (0.88 % of 494 M; 896 effective degrees of freedom; relative Frobenius change 7.8 %).

**Selection.**
- (L, α) ∈ {6, 9, 12, 15, 18} × {0.5, 1, 2}, chosen **on dev prompts only** by Swedish rate minus English-retention loss.
- This is feedback-optimised selection, not reading the weights.
- Because of the CPU budget (≈ 5 s per generation), dev screening used 24 Swedish and 10 English dev prompts with 32 new tokens.
- Selected: **L = 6, α = 2**.

**Test.** 110 Swedish and 60 English prompts, greedy decoding, 64 new tokens. Six conditions:
- (a) baseline;
- (b) the edit;
- (c) a random unit direction with the same L and α, 5 seeds;
- (d) the system prompt "Svara alltid på svenska.";
- (e) inference-time steering `−α·‖μ_en−μ_sv‖·d` at layer L (separately dev-tuned: L = 15, α = 0.5);
- (f) the edit plus the prompt.

**Measurement.**
- Language is identified with `langid` (strict `sv`; a lenient sv/no/da rate is also reported).
- Regression checks: English-answer rate, Swedish and English perplexity on 150 + 150 Wikipedia paragraphs, hunspell sv_SE non-word rate, and answer length.
- Statistics: Wilson 95 % CIs and exact McNemar tests on paired prompts.
- Robustness: T = 0.7, 3 seeds, 55 test prompts.
- Total runtime ≈ 3 h on a shared 4-core Intel Mac.

## 4. Results (Swedish test prompts, n = 110, greedy)
| condition | Swedish-answer rate [95 % CI] | English answers on EN prompts | non-word % | ppl sv / en |
|---|---|---|---|---|
| baseline | 92.7 % [86.3, 96.3] | 100 % | 4.5 | 71.5 / 20.1 |
| **rank-1 edit (L6, α2)** | **92.7 % [86.3, 96.3]** | 100 % | 4.2 | 71.8 / 20.2 |
| random edit (5 seeds) | 92.9 % ± 0.4 | 100 % (seed 0) | 4.4 | 71.8 / 20.1 |
| prompting | 98.2 % [93.6, 99.5] | **93.3 %** | 3.0 | – |
| steering | 10.0 % [5.7, 17.0] | 88.3 % | 18.5 | 1881 / 255 |
| edit + prompting | 99.1 % [95.0, 99.8] | 93.3 % | 2.9 | – |

**H1 is not supported.**
- The edited and baseline models give the *same* language label on all 110 prompts (0 discordant pairs, p = 1.0), although the generated text changes for 99 of the 110 prompts. Random edits change fewer.
- Edit vs random: p = 1.0 for every seed.
- Prompting vs edit: 7 vs 1 discordant pairs, p = 0.07. This is not significant at n = 110, but it favours prompting.
- Sampled decoding agrees: 85.5 % baseline, 85.5 % edit, 93.9 % prompting (3 seeds).

**H2 is partly supported: failure removal, not capability.**
- Only 8 baseline prompts fail: 3 in English and 5 labelled no/nb.
- The English failures are refusals, for example *"Hur lär jag barn att cykla?" → "I'm sorry, but I can't assist with that."*, in both the baseline and the edited model.
- Prompting answers in Swedish: it fixes 7 of the 8 failures and newly breaks 1.
- The Swedish itself is poor in every condition (see `results/readable_samples.md`), and the non-word rate (≈ 3–4.5 %) barely moves. The intervention that works switches *language selection*; it does not improve the Swedish.

**Costs.**
- The edit is harmless on perplexity (+0.3 %) and on English retention, but it also does nothing.
- Prompting costs 6.7 points of English answers on English prompts, because it forces Swedish everywhere.
- Steering with the raw difference vector wrecks the model at every tested strength.

## 5. Interpretation, limitations and recommendation

**Interpretation.** The rank-1 edit changed the wording of 99 of 110 answers but never the answer language. Swedish-answer rates were identical with and without the edit (92.7 %), and random edits of the same size gave the same result. The edit changed the text, not the mechanism that chooses the language.

The failures explain why. The few remaining errors were mainly English refusals ("I'm sorry, but I can't assist with that") and answers labelled as Norwegian, not a lack of Swedish ability. A Swedish-versus-English direction is unlikely to affect refusals, so the edit targeted the wrong mechanism. This mirrors the article: the Qwen fix worked because the failure (looping) was diagnosed correctly first. Choosing the right target mattered more than the editing method.

The only intervention that helped was a one-line prompt, "Svara alltid på svenska." It raised the Swedish-answer rate from 92.7 % to 98.2 % and fixed 7 of the 8 failures, although the difference was not statistically significant at this sample size (p = 0.07). It also lowered English answers to English prompts from 100 % to 93.3 %, as expected from an instruction to always answer in Swedish. This suggests that the Swedish capability was already present and that the failures came from behaviour selection, not missing knowledge.

**Limitations.**
- The baseline was already high (92.7 %, only 8 failures), so only large effects could be detected.
- The layer and edit strength were chosen on a very small development set.
- One model size, one edit type and only a coarse steering search were tested.
- Automatic language identification sometimes confused Swedish and Norwegian.
- The prompts were synthetic and were not systematically reviewed by native speakers.

**Evidence from practice.** I develop a bilingual Swedish/English engineering assistant and have met the same kinds of failures: answers in the wrong language, invented Swedish words and wrong facts. Internally we call our approach "Dracula". The first version, learning from a large commercial model's outputs, was rejected because providers' terms forbid using outputs to build competing models. The current version, Dracula 2.0, keeps the model small and builds checks around it:
- the reply language is chosen from the user's own words;
- Swedish output is checked against a dictionary and grammar rules;
- facts come from versioned records with source and licence;
- numbers come from executed calculations, not from the model;
- every release must pass held-out tests.

These simple, auditable controls fixed exactly the kind of failures the weight edit did not touch. Using licensed, traceable sources also fits SE.LLMA's commitment to lawful data.

**Recommendation.** SE.LLMA should not prioritise weight-space editing as a way to improve Swedish-language capability: the tested edit had no effect, while a simple prompt helped more at almost no cost. The idea may still be useful as a diagnostic tool, to tell apart refusals, instruction-following failures, language-routing failures and real gaps in Swedish knowledge.

**Most informative next experiment.** Choose a Swedish task with real headroom, ideally at least 30 % baseline failures, and diagnose the cause first. Then compare four methods on the same held-out set of at least 300 prompts, with several random seeds and more than one model size:
1. a conditional prompt ("answer in the language of the question");
2. a weight edit aimed at the diagnosed failure;
3. finely tuned activation steering;
4. a small LoRA adaptation.

Track Swedish quality, factual accuracy and English retention as regression checks.
