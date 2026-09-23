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
steps to converge (verified: see "Extended-mode convergence" below).

------------------------------------------------------------------
Milestone 1b — figures for the write-up (training curves, attention maps,
embeddings)
------------------------------------------------------------------
python -m experiments.analyze_single --mode base

Trains one generation and saves everything the report needs into
results/single_<mode>/:
  history.json         raw per-step metrics + which heads were identified
  training_curves.png  train loss / validation query accuracy / induction score
  attention_maps.png   every head's attention pattern on one example sequence
  embeddings.png        symbol cosine-similarity matrix + PCA of token embeddings
  model.pt              trained weights (so figures can be regenerated without
                         retraining)

------------------------------------------------------------------
Milestone 2 — the collapse sweep (Options A + C)
------------------------------------------------------------------
python -m experiments.run_collapse \
    --mode extended --generations 6 --gen_queries all \
    --fractions 0.0 0.1 0.25 0.5 1.0 --seeds 0 1 2 --steps 6000 \
    --out results/collapse_extended_all.json
(repeat with --mode base for the control, and with --gen_queries first for the
original partially-synthetic design). Re-running with the same --out resumes.

--gen_queries first  only the FIRST query label is model-generated; the other K-1
                     keep ground-truth targets, so even at real_fraction=0 the
                     induction task is re-taught from real labels every
                     generation (75% of base / 50% of extended supervised
                     targets stay real). This protects the circuit by design.
--gen_queries all    every query label is model-generated (fully recursive).

Safeguards added (so an optimisation failure is never read as collapse):
  * each generation's pool is split 90/10 train/validation; test = fresh real data;
  * convergence gate: a generation must reach 0.9 x (fraction of correct pool
    labels) validation accuracy, else it is retrained from a new init (up to
    --max_retries 3). Retries, the gate outcome and gate_informative are logged;
  * collision-free seeding: init seed = seed*1000 + 10*gen + attempt.

Headline figures:
python -m experiments.plot_collapse --extended results/collapse_extended_all.json \
    --base results/collapse_base_all.json --out results/figures
  fig_dose_response.png  final-generation value vs real_fraction (mean +/- 1 sd)
  fig_trajectories.png   per-generation curves, one per real_fraction
  summary.csv            the numbers behind both

------------------------------------------------------------------
Extended-mode convergence and its loss floor (verified empirically)
------------------------------------------------------------------
Default config, 1 seed at --steps 12000 plus 5 seeds at --steps 6000 and 3 at 4000:
  base:     transition within ~250-500 steps; query_acc 1.000, induction ~0.95.
  extended: transition is SEED- AND SCHEDULE-DEPENDENT: steps 900-2900 across
            seeds (the LR is cosine over --steps, so the same seed transitions
            at a different step under a different budget). Seed 2 NEVER
            transitioned at 4000 or 6000 steps (acc ~0.46-0.48): more steps
            alone do not guarantee convergence, hence the convergence gate +
            retries in the collapse loop. Loss plateaus at ~0.23, not ~0.

Why the loss floor is real, not a bug: in extended mode one supervised target
(sq2, the "free next symbol") is sampled independently of context
(rng.integers in src/data.py) -- it is genuinely unpredictable, so even a
perfect model cannot drive its loss below log(n_pairs). With n_pairs=4 that is
log(4)=1.386, diluted across the 6 supervised positions per sequence
(4 query labels + sq2 + lq2) gives an expected floor of ln(K)/(K+2) = 0.231.
Verified per position on 4096 fresh sequences (experiments/verify_circuit.py):
CE at sq2 = 1.397 (ln 4 = 1.386), ~1e-4 at every other supervised slot; the
model puts 99.4% of sq2 mass on the K in-context symbols, near-uniformly.
The floor tracks K (experiments/check_extended.py, 1 seed each):
    K=2: 0.1747 (theory 0.1733)   K=3: 0.2216 (0.2197)
    K=4: 0.2329 (0.2310)          K=5: 0.2316 (0.2299)
    K=4 with n_symbols=12: 0.2326 -> independent of vocabulary size.
Cite this floor rather than comparing extended-mode loss to base-mode loss.

Practical implication: use >= 6000 steps per extended generation AND the
convergence gate; a gen that never transitions must be retried, not counted.

------------------------------------------------------------------
What the circuit actually looks like (experiments/verify_circuit.py)
------------------------------------------------------------------
Accuracy alone is not proof of the circuit, so every claim below comes from
attention probes over all K queries plus zero-ablation of heads.
  * Base: REDUNDANT. All 4 L0 heads are previous-token heads (0.86-0.97) and 3
    L1 heads are induction heads (0.88/0.94/0.95). No single-head ablation drops
    accuracy below 0.96; ablating the L1 induction set -> 0.14, all of L0 -> 0.31.
    It is not a two-head circuit.
  * Extended (seed 0): a single previous-token head carries it (ablate L0H1 ->
    0.44) plus a dominant induction head (ablate L1H1 -> 0.69).
  * No direct-match (off-by-one) heads: attention to the matching SYMBOL ~0.
  * SHIFTED induction variant (extended, seed 3): accuracy 1.000 but the canonical
    induction score is only 0.25. L1H3 instead attends to the study symbol AFTER
    the matched pair (0.72 of its mass): layer 0 writes that pair's symbol (t-2
    head) and label (t-1 heads) into the next token, and L1H3 matches on the
    former and reads out the latter. Ablating it -> 0.42. The robust probe
    (src/probes.circuit_probe) scores both variants and reports a causal drop.

------------------------------------------------------------------
Results of the A+C sweep (verified; regenerate with the commands below)
------------------------------------------------------------------
Setup: --gen_queries all, 6 generations, 3 seeds, pool 20k (+10% val),
extended 6000 steps/gen, base 4000 steps/gen, T=1.0.
  python -m experiments.run_collapse --mode extended --gen_queries all \
      --fractions 0.0 1.0 0.25 0.1 0.5 --seeds 0 1 2 --steps 6000 --save_models \
      --out results/sweep/extended_all.json
  python -m experiments.run_collapse --mode base --gen_queries all \
      --fractions 0.0 1.0 0.25 0.1 0.5 --seeds 0 1 2 --steps 4000 --save_models \
      --out results/sweep/base_all.json
  python -m experiments.plot_collapse --extended results/sweep/extended_all*.json \
      --base results/sweep/base_all.json --out results/figures
  python -m experiments.summarise_sweep --files results/sweep/extended_all*.json
  python -m experiments.reprobe_saved --glob "results/sweep/extended_all*_gen*.pt" \
      --mode extended --out results/figures/reprobe_extended.csv
(The rf=0.5 extended runs were executed as a separate process into
extended_all_rf05.json; the plotting/summary scripts merge files.)

Extended, final generation (mean +/- sd over 3 seeds); gen 0: cond. div 0.994,
off-context 0.012:
  real_fraction  cond. diversity  off-context mass  induction  query acc
  0.0            0.819 +/- 0.022  0.052 +/- 0.005   0.959      1.000
  0.1            0.858 +/- 0.018  0.044 +/- 0.003   0.965      1.000
  0.25           0.861 +/- 0.026  0.031 +/- 0.003   0.969      1.000
  0.5            0.898 +/- 0.032  0.022 +/- 0.003   0.955      1.000
  1.0 (control)  0.897 +/- 0.018  0.012 +/- 0.001   0.938      1.000
Marginal symbol diversity stays 1.000-1.009 at EVERY fraction (blind).
Base control: query acc >= 0.997 and >= 98% of synthetic labels correct at all
fractions; induction 0.84-0.96 with no trend in real_fraction.

Findings:
  * Graded, monotone dose-response; no sharp critical fraction. Off-context mass
    (probability on symbols NOT in the context) accumulates roughly linearly per
    generation on self-generated data and is flat at real_fraction=1.
  * Finite-data confound: even real_fraction=1 drops conditional diversity from
    0.994 (gen 0, unlimited fresh data) to ~0.90 (finite 20k pool, ~77 epochs).
    The recursive effect is the gap to that control, not the drop from gen 0.
    (--gen0_pool removes the confound in future runs.)
  * Dissociation: the induction circuit is intact at every fraction. Re-probing
    all 30 saved final-generation models: all canonical, accuracy >= 0.999, and
    ablating layer 1 costs 0.74-1.00 (extended) / 0.61-0.93 (base) accuracy.
  * Circuit multiplicity: fresh models pick among THREE algorithms:
    canonical (copy the matching label), shifted (copy it via the next token),
    and ELIMINATION (layer-1 heads attend AWAY from the match and suppress the
    non-matching labels, OV label-copy score -15.8; ablate all L1 -> 0.40).
    10/90 extended and 6/90 base generations used a non-canonical variant, and
    the legacy single-head probe reported a false 'circuit collapse' in 9 and 6
    of them (the extended gen-1 dip in fig_trajectories is seed 2's init, which
    is shared across fractions and lands on the elimination variant).
  * Convergence gate: 6 (extended) and 9 (base) retries over 90 generations
    each, 0 final gate failures.

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
src/probes.py     head scores + robust circuit probe with ablation (Option A)
src/metrics.py    accuracy, perplexity, marginal + CONDITIONAL diversity
experiments/verify_circuit.py  per-head probes/ablations for a saved model
experiments/check_extended.py  convergence timing + loss-floor check
experiments/plot_collapse.py   headline A+C figures
experiments/plot_circuit.py    circuit figure (L0 routes, L1 attention, ablations)
experiments/summarise_sweep.py numbers for the write-up, from the sweep JSONs
experiments/reprobe_saved.py   re-probe saved gen-0/final models
experiments/tune_hparams.py    validation-based hyper-parameter selection + test
experiments/      runnable entry points

------------------------------------------------------------------
Modelling choices to defend / vary in the write-up
------------------------------------------------------------------
- In the collapse loop the induction CONTEXT stays real each generation; only the
  query/generative slots are model-generated. The generative slot (sq2, extended
  mode) is the degree of freedom that drives symbol-distribution collapse.
- Diversity: the MARGINAL entropy of generated sq2 over all 16 symbol ids
  (normalised by log(n_symbols); can exceed 1 when label tokens are emitted) is
  blind to the collapse that matters. sq2 is always one of the K in-context
  symbols, which are a uniform draw, so the marginal stays ~uniform even if the
  model always copies, say, the first study pair. The primary signal is the
  CONDITIONAL entropy of p(sq2 | context) over the K in-context symbols / ln K
  (cond_diversity), plus slot_entropy (which study slot is copied) and
  off_context mass. The marginal is still logged.
