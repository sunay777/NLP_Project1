# Pre-registered prediction: temperature and the inheritance gain

Written 2026-10-21T19:04 by `experiments/preregister.py` from the calibration runs only (seed [100], λ ∈ {0, 0.5}, 5 generations). The confirmatory sweep (`experiments/jobs_v2.py`, jobs `temp_T*`, seeds 0–4, λ ∈ {0, 0.25, 0.5}, 10 generations, strict gate) had **not been run** when this file was committed. The git commit timestamp is the evidence.

**Law** (off-context mass m of p(sq2 | context), λ = real-data fraction):

m_{g+1} = a(T) · (1 − λ) · m_g + ε(T)

Fits use only transitions with m_g < 0.25, the linear regime. This rule was fixed before the calibration data were fitted. 
A critical real-data fraction exists only if a(T) > 1, at λ* = 1 − 1/a(T). Predictions start from m₀ = 0.0106, the gen-0 mean. Gen 0 does not depend on T because T only affects sampling.

## Calibrated parameters

| T | a(T) [95% CI] | ε(T) [95% CI] | P(a > 1) | λ* | R² | n |
|---|---|---|---|---|---|---|
| 0.7 | -0.080 [-0.308, 0.143] | 0.0128 [0.0107, 0.0149] | 0.00 | none (a < 1) | 0.058 | 8 |
| 1 | 0.858 [0.818, 0.890] | 0.0121 [0.0113, 0.0128] | 0.00 | none (a < 1) | 0.996 | 8 |
| 1.3 | 1.439 [1.325, 1.570] | 0.0358 [0.0246, 0.0485] | 1.00 | 0.305 | 0.985 | 8 |
| 1.6 | 2.076 [0.862, 3.199] | 0.0381 [-0.0231, 0.0913] | 0.95 | 0.518 | 0.663 | 6 |

## Predicted off-context mass at generation 9 (point [95% band])

| T | λ = 0 | λ = 0.25 | λ = 0.5 |
|---|---|---|---|
| 0.7 | 0.0119 [0.0111, 0.0129] | 0.0121 [0.0115, 0.0127] | 0.0123 [0.0114, 0.0131] |
| 1 | 0.0666 [0.0600, 0.0713] | 0.0336 [0.0321, 0.0345] | 0.0212 [0.0203, 0.0218] |
| 1.3 | 2.3528 [1.6576, 3.6943] | 0.4663 [0.3555, 0.6517] | 0.1216 [0.0920, 0.1519] |
| 1.6 | 32.9186 [-15.3301, 297.0225] | 4.1801 [-9.5060, 26.8767] | 0.4146 [-1.3006, 1.4699] |

Full per-generation trajectories and bands are in `paper/PREDICTION.json`.

## Tests, fixed before the confirmatory data exist

Each test uses the confirmatory runs only (seeds 0–4), with T = 1 taken from the main sweep (`main_ext`, seeds 0–4, first 10 generations).

1. **Gain rises with temperature.** The fit of a(T) on the confirmatory runs increases monotonically over T ∈ {0.7, 1.0, 1.3, 1.6}.
2. **Trajectories match the prediction.** At each (T, λ), the seed-mean off-context mass lies inside the 95% band at ≥ 80% of those generations 1–9 where the point prediction is ≤ 0.5. This is the linear regime: a mass cannot exceed 1, so the law has to saturate. The test passes overall if it holds for ≥ 7 of the 9 (T, λ) cells.
3. **Contracting cells decelerate.** For cells predicted to contract (a(T)(1−λ) < 1), the late slope (gens 6–9) is smaller than the early slope (gens 1–4) in ≥ 4/5 seeds. Supercritical cells are scored by test 4 instead, because a mass bounded by 1 must slow down as it saturates whatever the gain.
4. **Runaway vs bounded.** Cells predicted supercritical (a(T)(1−λ) ≥ 1) exceed an off-context mass of 0.5 by generation 9 in ≥ 4/5 seeds. Cells predicted to contract stay below the upper end of their predicted gen-9 band in ≥ 4/5 seeds. The extra λ* ± 0.1 cells added by `jobs_v2.py` are scored the same way, which tests the location of the critical point itself.
5. **Temperature trades hallucination for homogenisation.** At generation 9 with λ = 0, off-context mass increases with T and conditional entropy of sq2 decreases as T decreases (both monotone over the 4 temperatures).

Any failed test is reported as a failure of the law, not explained away. If a(T) stays below 1 at every T, the prediction is that collapse in this regime is always graded.
