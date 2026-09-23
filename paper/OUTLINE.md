# Extended abstract — outline (2 pages, firm)

Working title options:
1. *Collapse without forgetting: the induction circuit survives recursive training while the generative choice it serves degenerates*
2. *An induction head as a probe on model collapse*

Status key: **[V]** verified in code/runs (numbers below are measured) · **[P]** pending the full sweep · **[TODO]** not done yet.

Page budget (≈ 2 × 2 columns of NeurIPS-style text incl. figures):
| Section | Rubric weight | Space |
|---|---|---|
| Intro + background | 10% + 10% (negative marking) | ~0.35 page |
| Method | 30% | ~0.6 page |
| Results (2 figures, 1 small table) | 30% | ~0.7 page |
| Discussion + limitations | 20% | ~0.35 page |
References, contribution statement, NeurIPS 2024 checklist, Faculty AI ethics statement → extra pages. Supplementary material allowed after the checklist: move detail there.

---

## 1. Introduction (≈ 1 paragraph)
- ICL via induction heads (Olsson 2022; Singh 2024) + model collapse under recursive training (Shumailov 2023/2024).
- Question: **when a model is recursively trained on its own outputs, does collapse happen *in* the ICL circuit, or *around* it?**
- Contributions (3 bullets):
  1. A mechanistic probe for collapse (angle A): track the induction circuit's attention *and its causal role* every generation.
  2. A real-data dose-response (angle C) over the fraction of real data mixed into each generation.
  3. Methodological findings: (i) the standard marginal diversity metric is blind to the collapse that occurs here; (ii) seeds find different, equally valid circuits, so a fixed attention-pattern probe gives false "circuit collapse" readings.

## 2. Background (≈ 1 short paragraph, cite only what is used)
- Induction circuit = previous-token head (L0) → induction head (L1), K-composition (Olsson 2022; Singh 2024 Fig. 1).
- Collapse = tails lost, distribution contracts towards modes (Shumailov 2024).
- Circuit identification via attention patterns + ablation (Conmy 2023 as the principled extension; we use targeted zero-ablation).

## 3. Method
**Task [V].** Study→query ICL task: K=4 distinct symbols (of 16), each with a per-sequence random label (of 6); study block lists the pairs in one order, query block the same symbols in an independent order → match offset varies, so only content-based induction solves it. Base predicts each query label; Extended additionally predicts a free next symbol sq2 (uniform over the K in-context symbols) and its label lq2.
- Design justification = the failure story (one sentence each, detail in supplement) **[V]**: sparse-signal trap (stuck ~0.29); positional shortcut (copy acc 1.0, zero content matching) → independent query order; direct-match off-by-one head (plateau ~0.5) → caught by attention maps; RoPE + tied embeddings + cosine LR decay all required.

**Model [V].** 2-layer attention-only transformer, d=64, 4 heads, RoPE, tied embed/unembed, ~35.6k params. AdamW, lr 1e-3, warmup 200, cosine decay.
- **[TODO] Hyper-parameter tuning is required by the brief** (train/val/test split used to *select* hyper-parameters). Plan: small grid on gen 0 (lr ∈ {3e-4, 1e-3, 3e-3} × warmup ∈ {100, 200} × d_model ∈ {32, 64}), select by validation loss, report in one line + supplement table.

**Collapse pipeline [V].** Gen 0 trains on fresh real data; each gen g ≥ 1 trains a *fresh* model on a pool of `real_fraction` real + rest synthetic from gen g−1 (T=1 ancestral sampling). Context stays real; answer slots are generated. Two regimes: `first` (only the first query label synthetic; the original design) and `all` (all K). Pool split 90/10 train/val; test = fresh real data.
- Convergence gate: a gen must reach 0.9 × (fraction of correct pool labels) on its val split, else retrained from a new init (≤3 retries, logged). Needed because extended mode fails to transition on some seeds (seed 2 never transitions at 4k or 6k steps) — an optimisation failure must not be read as collapse.

**Metrics.**
- Collapse: **conditional diversity** = mean over contexts of H(p(sq2 | context) over the K in-context symbols) / ln K (1 = uniform); slot entropies (study order, query order); off-context mass. Marginal symbol entropy logged only as a foil.
- Circuit (A): induction copy-mass over all K queries, counting the canonical (→ matching label) and the *shifted* (→ symbol after the matched pair) patterns; **causal drop** = accuracy lost when the induction heads are zero-ablated.
- Loss floor sanity check: extended loss is bounded below by ln K / (K+2) (sq2 irreducible, all else learnable).

## 4. Results
**Fig. 1 (circuit, gen 0) [V]** — left: attention of the L0 previous-token and L1 induction heads on one sequence; right: ablation bar chart. Numbers:
- Base: redundant — all 4 L0 heads previous-token (0.86–0.97), 3 L1 induction heads (0.88/0.94/0.95); no single-head ablation below 0.96 acc; ablate the L1 induction set → 0.14, all of L0 → 0.31.
- Extended (seed 0): one previous-token head (ablate → 0.44), dominant induction head (ablate → 0.69).
- Shifted variant (extended seed 3): acc 1.000 but canonical score 0.25; L1H3 puts 0.72 of its mass on the symbol after the matched pair; ablate → 0.42.
- Loss floor ln K/(K+2): K=2: 0.1747 vs 0.1733; K=3: 0.2216 vs 0.2197; K=4: 0.2329 vs 0.2310; K=5: 0.2316 vs 0.2299; n_symbols=12 control: 0.2326 (one line or a tiny table).

**Fig. 2 (headline, A+C) [P]** — `results/figures/fig_dose_response.png`: final-generation conditional diversity + induction strength vs real_fraction, mean ± 1 sd over 3 seeds; base as control. Pair with `fig_trajectories.png` in the supplement or as panel (b).
- Pilot (real_fraction = 0, extended, 1 seed, `all`) **[V]**: conditional diversity 0.994 → 0.79 over 5 generations (p10: 0.99 → 0.58); marginal diversity ≈ 1.00 throughout; induction 0.94 → 0.97, real-data accuracy 1.000 every gen; off-context mass 0.012 → 0.046; `first` ≈ `all`.
- **[P]** Does conditional diversity recover monotonically with real_fraction? Is there a critical fraction? Does induction stay flat at every fraction (→ dissociation), or break with diversity (→ coupled)?

**Table 1 (optional, tiny) [P]** — gen-5 values at real_fraction ∈ {0, 1}: cond. diversity, marginal diversity, induction, causal drop, retries.

## 5. Discussion (the 20% — interpretation, not recap)
- **Dissociation (if the sweep holds):** collapse lives in the *free generative choice*, not in the ICL algorithm. The query labels are determined by context, so the circuit is re-validated by every sequence; sq2 is under-determined, so the model's own sampling noise feeds back into its prior. Links to Shumailov: tails lost = low-probability in-context options; here "tails" are per-context, not global.
- **Why marginal metrics miss it:** in-context symbols are a uniform draw, so per-context sharpening averages out globally. Implication: collapse audits of ICL-capable models need *conditional* diagnostics.
- **Probe fragility as a finding:** attention-pattern probes are circuit-specific; causal (ablation) measures are needed before claiming a circuit degrades.
- Base vs extended circuit redundancy (3 induction heads vs 1 dominant) → the more generative task is served by a less redundant circuit; hypothesis for fragility.
- **Limitations:** tiny model/vocab; 6 generations; 3 seeds; T=1 only; fresh model per generation (no fine-tuning chain); zero-ablation (not mean/resample); the collapse direction (what the sharpening is keyed on) not yet identified — check query-order slot stats.
- Optional add-on (security framing): seed a bias at gen 0 and watch amplification = data-poisoning through the recursive loop.

## Supplementary (after the checklist)
Failure-mode table with numbers; hyper-parameter grid; per-seed convergence table (transition steps 900–2900, seed 2 never); full per-head probe/ablation tables; trajectories figure; `first` vs `all`.

## Checklist before submission
- [ ] Moodle template, 2 pages exactly (figures legible at print size)
- [ ] NeurIPS 2024 checklist filled + Faculty AI ethics statement
- [ ] Contribution statement
- [ ] Every number in the text traceable to a file in `results/` (no hallucinated values — heavily penalised)
- [ ] README + requirements.txt reproduce every figure
