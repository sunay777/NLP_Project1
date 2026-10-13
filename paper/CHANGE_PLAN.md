# Change plan: robustness fixes and new headline

Due **27 Oct 2026, 17:00**. Rubric weights: Method 30, Results 30, Discussion 20, Structure 10, Background 10 (negative marking).

This document lists what will be **changed** in the existing code, results and paper, and what will be **added**. Numbers quoted under "Motivation" come from an independent review of the saved JSONs. Every number that ends up in the paper will be re-derived by a script and saved under `results/`.

## Ground rules

- `results/sweep/` is frozen. All new runs go to `results/v2/<experiment>/`.
- Every number in the paper is produced by a script and saved to JSON or CSV. No numbers are typed by hand.
- A paper outside the brief is cited only after its arXiv abstract has been checked.
- Commits are small and logical. Long runs are wrapped in `caffeinate -i` and logged to file.
- Code style stays the same: dataclass config, `experiments/*.py` argparse CLIs, resumable `--out`.

---

## 1. Motivation: problems found in the current version

| # | Problem | Consequence |
|---|---------|-------------|
| P1 | Figures use the legacy induction probe, which reports false "circuit collapse" on non-canonical circuit variants. | The dips in `fig_dose_response.png` (a) at base rf=0.5 and in `fig_trajectories.png` panel 2 at gen 1 are probe artefacts. |
| P2 | The claim of a "graded, monotone dose-response" in conditional diversity is not supported. After gen 1, every fraction (including rf=1.0) sits at about 0.86–0.92. rf=0.5 (0.898) and rf=1.0 (0.897) are the same, with seed SD of about 0.02–0.03 and n=3. | Most of the drop is the finite-pool effect. Only off-context mass shows a clean dose-response. |
| P3 | Circuit survival ("dissociation") is partly by construction. The context stays real and only the answer slots are synthetic. | This must be stated explicitly and framed as the **prompt-real, completion-synthetic** regime. |
| P4 | The convergence gate (0.9) is lax. Base rf=0.5 s0 gen 4 passed with acc 0.958, and its gen-5 pool had 11.7 % wrong labels. Base rf=0.25 s2 gen 1 produced 3.1 % wrong labels. | Some downstream generations were trained on corrupted pools. |
| P5 | No validation loss is logged, and perplexity covers only the first query. | Brief metrics are missing. |
| P6 | Only zero-ablation is used. | Mean-ablation is the standard, less off-distribution baseline. |

---

## 2. Changes to existing files

| File | Change | Fixes |
|------|--------|-------|
| `src/train.py` | Log **validation loss** (CE on supervised positions of a held-out real batch) next to the train loss and val accuracy. | P5 |
| `src/metrics.py` | Compute **NLL and perplexity over all K queries**, not only the first. | P5 |
| `src/probes.py` | Add **mean-ablation** (each head's output replaced by its mean over a reference batch) beside zero-ablation. Paper figures use mean-ablation; zero-ablation goes to the supplement, with any differences noted. | P6 |
| `experiments/run_collapse.py` / `src/collapse.py` | **Strict gate**: `--gate 0.98` plus a second criterion (`causal_drop_L1 ≥ 0.5` or query NLL below a threshold). Used for all new runs. | P4 |
| `experiments/run_collapse.py` / `src/collapse.py` | **Poison pulse**: `--pulse_gen`, `--pulse_label_noise` and optionally `--pulse_offcontext`, which corrupt a fraction of synthetic labels in one generation's pool only. | T2.2 |
| `experiments/plot_collapse.py` | Replace the legacy induction curve with a variant-agnostic one (`causal_drop`, or accuracy under all-L1 ablation), with `induction_best` as secondary. Add a **query-NLL panel on log-y**. Overlay the fitted recursion on the off-context panel (solid = fitted fractions, dashed = held-out predictions). Use y-ranges that show the signal. | P1, P2 |
| `experiments/summarise_sweep.py` | Report a **late-generation mean (gens 3–5)** as well as the final generation. Report **paired per-seed differences vs λ = 1** with a sign count, because init seeds are shared across fractions (common random numbers). | P2 |
| `experiments/analyze_single.py` | Regenerate training curves for base and extended: train loss, **val loss**, val accuracy and induction score vs step. | P5 |
| `paper/OUTLINE.md` | Reframe the story (Section 5). Remove or soften the conditional-diversity dose-response claim, state the prompt-real regime in the Method, and report the gate audit. | P2, P3, P4 |
| `requirements.txt` | Pin `torch`, `numpy` and `matplotlib` versions. | Hygiene |
| `README.txt` | One command per figure or table, with expected runtime. State that `.pt` files are not in git and how to regenerate them. | Hygiene |

---

## 3. New files and experiments

### Tier 1: must do (mostly analysis of saved results)

**T1.1 `experiments/fit_recursion.py`: inheritance law.**
For each metric (off-context mass, wrong-label rate of the synthetic share, query NLL, conditional-entropy deficit), fit

  m_{g+1} = a · (1 − λ) · m_g + ε     (λ = real fraction)

- Pooled fit, per-seed fits, and bootstrap CIs for a and ε (2000 draws over seeds × fractions).
- Held-out test: fit on λ ∈ {0, 0.25, 1}, predict λ ∈ {0.1, 0.5}, and report RMSE and coverage.
- Stationary level m* = ε / (1 − a(1−λ)) and perturbation half-life ln 2 / −ln(a(1−λ)).
- Gate-lax models are excluded or flagged separately.
- Output: `results/v2/recursion_fit.json` plus a table CSV.

Preliminary review numbers for off-context mass: a ≈ 0.908, ε ≈ 0.0112, R² ≈ 0.981, and held-out predictions within one seed-SD.

**T1.2 Headline figure.** One composite figure for a 2-column, 2-page abstract: at most 3.3 in tall, fonts at least 7 pt, colour-blind-safe palette.

**T1.5 Gate audit.** `results/v2/gate_audit.csv` lists every model in the existing sweep that fails the strict gate, together with its downstream effect.

**T1.6 Remove the finite-data confound (about 2.5 h).**
```
run_collapse --mode extended --gen_queries all --gen0_pool --fractions 0.0 1.0 \
  --seeds 0 1 2 --steps 6000 --gate 0.98 --save_models \
  --out results/v2/gen0pool/extended.json
```
Gen 0 sees the same 20k-sample budget as later generations. The recursive effect is reported relative to this baseline.

### Tier 2: creative headline (overnight runs)

**T2.1 Pre-registered temperature test of the critical point (about 7–8 h).**
The theory predicts a critical real-data fraction only if a > 1, at λ* = 1 − 1/a. Temperature T > 1 should raise a and ε.
1. Calibration: λ = 0, T ∈ {0.7, 1.3, 1.6}, seed 0, 4 generations. Fit a(T) and ε(T).
2. Pre-registration: write `paper/PREDICTION.md` with a(T), ε(T), λ*(T) and predicted trajectories, and **commit it before the confirmatory sweep**.
3. Confirmatory sweep: λ ∈ {0, 0.25, 0.5} (plus λ* ± 0.1 if a(T) > 1), 3 seeds, 8 generations, strict gate → `results/v2/temperature/`.
4. Figure: phase diagram over (T, λ) coloured by final off-context mass, with the predicted boundary a(T)(1−λ) = 1 overlaid, plus a conditional-entropy panel showing hallucination traded for homogenisation.

**T2.2 Poison-pulse impulse response (about 3.5 h).**
Inject label noise (5 %) into the gen-1 pool only. λ ∈ {0, 0.25, 0.5}, 3 seeds, 6 generations, base and extended. Compare the measured decay ratio and half-life with the prediction (1 − λ)·a, using a from T1.1. Framed as data poisoning in the Discussion and ethics sections.

**T2.3 Mechanism of the hallucination (minutes; uses saved `.pt` files).**
Gen-0 vs gen-5 models at λ = 0:
- Direct logit attribution at the sq2 position: which heads or paths write the off-context logits.
- Whether off-context mass goes to symbols whose embeddings are similar to in-context ones (fuzzy copy) or not (learned prior).
- Induction-head attention leak vs OV copy-score magnitude, to locate the NLL erosion in QK or OV.
- Output: `results/v2/mechanism.json` plus one small figure.

### Tier 3: stretch (only if Tiers 1–2 are done by about 20 Oct)

- `--sq2_zipf α`: Zipfian free choice, to test whether tails are lost faster than ε predicts.
- Pool-size sweep at λ = 1 (N ∈ {5k, 20k, 80k}), to isolate the statistical approximation error.
- Fully synthetic contexts, to test whether circuit survival holds once prompts are synthetic too. If this isn't done, it becomes a stated limitation.

---

## 4. New result locations

| Path | Contents |
|------|----------|
| `results/v2/recursion_fit.json`, `.csv` | Inheritance-law fits, CIs, held-out test |
| `results/v2/gate_audit.csv` | Models failing the strict gate |
| `results/v2/gen0pool/` | Finite-data confound run |
| `results/v2/temperature/` | Calibration and confirmatory temperature sweep |
| `results/v2/pulse/` | Poison-pulse runs |
| `results/v2/mechanism.json` | Logit attribution and QK/OV analysis |
| `paper/PREDICTION.md` | Pre-registered predictions (committed before the confirmatory runs) |

---

## 5. Paper changes

- **Story:**
  1. The circuit forms, and the three failure modes it defeats justify the method.
  2. In the prompt-real regime, collapse leaves the circuit's *decisions* intact (causal, mean-ablation) while its errors compound.
  3. A two-parameter inheritance law fits the data, predicts held-out fractions, explains the absence of a critical point, and is tested by temperature (pre-registered) and by a poison pulse.
  4. Methodological lessons: marginal diversity is blind, attention probes give false collapse readings under circuit multiplicity, and accuracy hides calibration erosion.
- **Working title** (only once T2 results exist): *Recursive training as a linear error filter: induction circuits survive model collapse while their errors compound.*
- **Background** (each to be verified before citing): Bertrand et al. (ICLR 2024), Gerstgrasser et al. (2024), Shumailov et al. (2024), Chan et al. (NeurIPS 2022; only if T3 is done), and Conmy et al. for the ablation methodology. Chen et al. (2026) is mentioned as future work.
- **Page budget:** one composite headline figure, one compact circuit figure and one small table. Everything else goes to the supplement.
- **Submission:** NeurIPS 2024 checklist and Faculty AI ethics statement appended after the references. Final pass: check every number in the paper against a file in `results/`.

---

## 6. Order of work

1. T1.4 logging (validation loss, all-query NLL, mean-ablation)
2. T1.5 strict gate and gate audit
3. T1.1 recursion fit
4. T1.2 figures and T1.3 statistics
5. Overnight: T1.6 and the T2.1 calibration
6. T2.1 pre-registration commit
7. Overnight: T2.1 confirmatory sweep and T2.2 poison pulse
8. T2.3 mechanism
9. Paper

Progress is reported after each tier. Any result that contradicts the inheritance law is treated as a finding to discuss, not something to hide.
