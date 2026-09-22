NLP Project 1 — Interpretability of Model Collapse
==================================================

A small transformer learns the Singh et al. (2024) in-context-learning symbol
task via an induction head, and is then placed in a model-collapse pipeline.
Two research angles are built in:

  (A) Induction head as a probe on collapse: track its copy-fidelity (induction
      attention strength) every generation.
  (C) Dose-response: sweep the fraction of REAL data mixed into each generation
      and look for a critical fraction at which the circuit / diversity breaks.

The target headline figure overlays the diversity-collapse curve and the
induction-strength curve across the real_fraction sweep.

------------------------------------------------------------------
Setup
------------------------------------------------------------------
python -m venv .venv && source .venv/bin/activate      (optional)
pip install -r requirements.txt
Run every command from the repository ROOT (so `import src...` resolves).

------------------------------------------------------------------
Milestone 1 — verify an induction head forms
------------------------------------------------------------------
python -m experiments.run_single --mode base

With the default config this reaches query accuracy 1.0 and an induction-head
strength ~0.95 within ~750 steps. Expected console tail:
    step 750 | query_acc 1.000 | ppl 1.00 | induction=0.96
    previous-token head: layer 0 head 0  score~0.98
    induction head:      layer 1 head 2  score~0.95

--mode extended is harder (it also predicts a free next symbol) and needs more
steps to converge.

------------------------------------------------------------------
Milestone 2 — the collapse sweep (Options A + C)
------------------------------------------------------------------
python -m experiments.run_collapse \
    --mode extended --generations 6 \
    --fractions 0.0 0.1 0.25 0.5 1.0 --seeds 0 1 2 --steps 4000

Results stream to results/collapse.json (checkpointed after every run). Run the
same sweep with --mode base as the control (expected to stay stable).

------------------------------------------------------------------
What makes the induction circuit form (important design findings)
------------------------------------------------------------------
Getting a CLEAN, content-based induction head (not a positional or direct-match
shortcut) required all of the following; each was verified empirically:
  * Study->query task with independent random orders, so the match offset varies
    and a positional shortcut cannot solve it (src/data.py).
  * Rotary positional embeddings (use_rope): lets the previous-token head learn
    "attend exactly one back". With absolute positions it plateaus at ~0.3.
  * Tied embed/unembed (tie_embeddings): aligns the copy operation.
  * A DECAYING learning rate (lr_decay): a constant LR destabilises the model
    back out of the induction solution once found.
  * A small model + small vocab: difficulty rises steeply with n_pairs and
    n_symbols. Defaults (K=4, 16 symbols) form induction in <1k CPU steps.

SCALING NOTE: on a GPU/MPS, raise n_pairs / n_symbols and --steps for a richer
task. Expect a longer plateau then a phase transition; watch the induction score.
Each generation trains a FRESH model, so give enough steps per generation that
even real_fraction=1.0 reconverges to ~1.0 — otherwise gen-to-gen wobble is
training noise, not collapse.

------------------------------------------------------------------
Layout
------------------------------------------------------------------
src/config.py     hyper-parameters, token scheme, sequence-length logic
src/data.py       the ICL symbol task; study/query construction; base & extended
src/model.py      2-layer attention-only transformer (RoPE, tying); exposes attn
src/train.py      single-generation training loop + metrics
src/collapse.py   generational pipeline; real/synthetic mixing; data generation
src/probes.py     previous-token & induction head scores (Option A)
src/metrics.py    accuracy, perplexity, diversity/entropy (collapse signal)
experiments/      runnable entry points

------------------------------------------------------------------
Modelling choices to defend / vary in the write-up
------------------------------------------------------------------
- In the collapse loop the induction CONTEXT stays real each generation; only the
  query/generative slots are model-generated. The generative slot (sq2, extended
  mode) is the degree of freedom that drives symbol-distribution collapse.
- Diversity is the entropy of the model's generated next-symbol distribution;
  it is normalised by log(n_symbols) and can slightly exceed 1 when a model emits
  label-range tokens in that slot (itself an early collapse signal).
