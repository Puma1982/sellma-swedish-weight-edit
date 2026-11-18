# AI usage

> DRAFT — Haithem to edit and confirm before submitting. The "decisions" section in particular must reflect what Haithem actually did.

**Tools**
- **Claude (chat)**: first critique of the article and a first design draft of the experiment (hypotheses, controls, metrics).
- **Claude Code (Claude Sonnet 5.5, run autonomously from a written work order)**: wrote all code in `src/`, built the
  synthetic prompt sets and contrast sentences, ran the pilot and the experiment on a CPU-only laptop, produced the figures,
  and drafted the README, the research note and the PDF report.
- No paid services or API keys were used. Models and data are open (see `data/README.md`).

**What the AI did not do / what remains the author's**
- Haithem set the task framing, the hypotheses and the decision rules (pilot threshold of 95 %, dev-only selection,
  controls, metrics), and is responsible for the final text. Sections of the research note marked DRAFT are
  AI-written first drafts that Haithem is to edit.
- The Swedish prompts and the parallel sentences are AI-written (synthetic). They have not been reviewed by a native
  speaker in full; the sampled generations in `results/readable_samples.md` are provided so that a Swedish reader can check
  answer quality directly. Language labels come from `langid`, not from human judgement.

**Known weaknesses of the AI-assisted process**
- The critique of the article relies on the points listed in the work order and on the assignment text. The article itself
  should be re-read by the author before the critique is relied upon.
