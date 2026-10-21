"""Turn the temperature-calibration runs into a PRE-REGISTERED prediction.

Calibration (results/v2/temperature_calib/): seed 100 only, lambda in {0, 0.5},
T in {0.7, 1.0, 1.3, 1.6}, 5 generations, strict gate, gen-0 real pool. The
confirmatory sweep (experiments/jobs_v2.py, temp_T*) uses DIFFERENT seeds
(0-4), so it is an independent test.

Per temperature this fits the inheritance law for off-context mass,
    m_{g+1} = a(T) (1 - lambda) m_g + eps(T),
with a residual bootstrap (2000 draws) for 95% CIs, and writes:
  paper/PREDICTION.json  machine-readable (read by jobs_v2.py: lambda* +- 0.1
                         grid points are added for any T with a(T) > 1);
  paper/PREDICTION.md    the human-readable prediction and the pass/fail tests,
                         to be committed BEFORE the confirmatory sweep starts.

Run:  python -m experiments.preregister --files "results/v2/temperature_calib/*.json"
"""
import argparse
import datetime
import glob
import json
import math

import numpy as np

from experiments.fit_recursion import load_runs, transitions, fit, iterate

CONF_LAMBDAS = [0.0, 0.25, 0.5]
CONF_GENS = 10
MAX_M_FIT = 0.25     # fit only transitions starting below this mass (linear regime;
                     # a probability mass cannot grow past 1, so the law must saturate)
MAX_M_TEST = 0.5     # trajectory-band test only where the point prediction is <= this


def residual_bootstrap(pairs, n=2000, seed=0):
    f = fit(pairs)
    x = np.array([p[0] for p in pairs])
    lam = np.array([p[2] for p in pairs])
    y = np.array([p[1] for p in pairs])
    pred = f["a"] * (1 - lam) * x + f["eps"]
    res = y - pred
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        yb = pred + rng.choice(res, size=len(res), replace=True)
        b = fit([(xi, yi, li) for xi, yi, li in zip(x, yb, lam)])
        out.append((b["a"], b["eps"]))
    return f, np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--m0", type=float, default=None,
                    help="gen-0 off-context mass to start predictions from (default: "
                         "calibration gen-0 mean; gen 0 does not depend on T)")
    ap.add_argument("--json", default="paper/PREDICTION.json")
    ap.add_argument("--md", default="paper/PREDICTION.md")
    a = ap.parse_args()
    runs = load_runs(sorted({f for p in a.files for f in glob.glob(p)}))
    by_T = {}
    for r in runs:
        by_T.setdefault(float(r["meta"].get("temperature", 1.0)), []).append(r)
    m0s = [r["records"][0]["cond_diversity"]["off_context"] for r in runs]
    m0 = a.m0 if a.m0 is not None else float(np.mean(m0s))
    pred = {"created": datetime.datetime.now().isoformat(timespec="minutes"),
            "calibration_files": sorted({r["file"] for r in runs}),
            "calibration_seed": sorted({r["seed"] for r in runs}),
            "law": "m_{g+1} = a(T) (1 - lambda) m_g + eps(T), m = off-context mass of p(sq2|context)",
            "m0": m0, "max_m_fit": MAX_M_FIT, "max_m_test": MAX_M_TEST, "confirmatory": {"lambdas": CONF_LAMBDAS, "generations": CONF_GENS,
                                       "seeds": [0, 1, 2, 3, 4]},
            "temperatures": {}}
    for T in sorted(by_T):
        pairs = []
        gate_flags = 0
        for r in by_T[T]:
            t, _ = transitions(r, "off_context", exclude_lax=False)
            pairs += [p for p in t if p[0] < MAX_M_FIT]
            gate_flags += sum(rec.get("converged") is False for rec in r["records"])
        f, boot = residual_bootstrap(pairs)
        a_ci = np.percentile(boot[:, 0], [2.5, 97.5]).tolist()
        e_ci = np.percentile(boot[:, 1], [2.5, 97.5]).tolist()
        ent = {"a": f["a"], "eps": f["eps"], "r2": f["r2"], "n_transitions": f["n"],
               "a_ci95": a_ci, "eps_ci95": e_ci, "gate_failures_in_calibration": gate_flags,
               "lambda_star": (1 - 1 / f["a"]) if f["a"] > 1 else None,
               "p_a_gt_1": float((boot[:, 0] > 1).mean()), "trajectories": {}}
        extra = []
        if ent["lambda_star"] is not None:       # same grid as experiments/jobs_v2.prediction_extras
            extra = sorted({round(min(0.95, max(0.0, ent["lambda_star"] + d)), 2) for d in (-0.1, 0.0, 0.1)})
        ent["extra_lambdas"] = extra
        for lam in sorted(set(CONF_LAMBDAS) | set(extra)):
            traj = np.array([iterate(ai, ei, lam, m0, CONF_GENS) for ai, ei in boot])
            k = f["a"] * (1 - lam)
            ent["trajectories"][f"{lam:g}"] = {
                "point": iterate(f["a"], f["eps"], lam, m0, CONF_GENS),
                "band95": np.percentile(traj, [2.5, 97.5], axis=0).tolist(),
                "gain": k,
                "m_star": f["eps"] / (1 - k) if k < 1 else None,
                "half_life": math.log(2) / -math.log(k) if 0 < k < 1 else None}
        pred["temperatures"][f"{T:g}"] = ent
    Ts = sorted(by_T)
    a_by_T = [pred["temperatures"][f"{T:g}"]["a"] for T in Ts]
    pred["calibration_a_monotone_in_T"] = bool(all(np.diff(a_by_T) > 0))
    json.dump(pred, open(a.json, "w"), indent=1)

    L = []
    L.append("# Pre-registered prediction: temperature and the inheritance gain\n")
    L.append(f"Written {pred['created']} by `experiments/preregister.py` from the calibration runs only "
             f"(seed {pred['calibration_seed']}, λ ∈ {{0, 0.5}}, 5 generations). The confirmatory sweep "
             "(`experiments/jobs_v2.py`, jobs `temp_T*`, seeds 0–4, λ ∈ {0, 0.25, 0.5}, 10 generations, "
             "strict gate) had **not been run** when this file was committed. The git commit timestamp "
             "is the evidence.\n")
    L.append("**Law** (off-context mass m of p(sq2 | context), λ = real-data fraction):\n\n"
             "m_{g+1} = a(T) · (1 − λ) · m_g + ε(T)\n")
    L.append(f"Fits use only transitions with m_g < {MAX_M_FIT}, the linear regime. This rule was fixed before "
             "the calibration data were fitted. ")
    L.append("A critical real-data fraction exists only if a(T) > 1, at λ* = 1 − 1/a(T). "
             f"Predictions start from m₀ = {m0:.4f}, the gen-0 mean. Gen 0 does not depend on T "
             "because T only affects sampling.\n")
    L.append("## Calibrated parameters\n")
    L.append("| T | a(T) [95% CI] | ε(T) [95% CI] | P(a > 1) | λ* | R² | n |")
    L.append("|---|---|---|---|---|---|---|")
    for T in Ts:
        e = pred["temperatures"][f"{T:g}"]
        ls = "none (a < 1)" if e["lambda_star"] is None else f"{e['lambda_star']:.3f}"
        L.append(f"| {T:g} | {e['a']:.3f} [{e['a_ci95'][0]:.3f}, {e['a_ci95'][1]:.3f}] | "
                 f"{e['eps']:.4f} [{e['eps_ci95'][0]:.4f}, {e['eps_ci95'][1]:.4f}] | "
                 f"{e['p_a_gt_1']:.2f} | {ls} | {e['r2']:.3f} | {e['n_transitions']} |")
    L.append("\n## Predicted off-context mass at generation 9 (point [95% band])\n")
    L.append("| T | " + " | ".join(f"λ = {l:g}" for l in CONF_LAMBDAS) + " |")
    L.append("|---|" + "---|" * len(CONF_LAMBDAS))
    for T in Ts:
        e = pred["temperatures"][f"{T:g}"]["trajectories"]
        L.append(f"| {T:g} | " + " | ".join(
            f"{e[f'{l:g}']['point'][-1]:.4f} [{e[f'{l:g}']['band95'][0][-1]:.4f}, "
            f"{e[f'{l:g}']['band95'][1][-1]:.4f}]" for l in CONF_LAMBDAS) + " |")
    L.append("\nFull per-generation trajectories and bands are in `paper/PREDICTION.json`.\n")
    L.append("## Tests, fixed before the confirmatory data exist\n")
    L.append("Each test uses the confirmatory runs only (seeds 0–4), with T = 1 taken from the "
             "main sweep (`main_ext`, seeds 0–4, first 10 generations).\n")
    L.append("1. **Gain rises with temperature.** The fit of a(T) on the confirmatory runs increases "
             "monotonically over T ∈ {0.7, 1.0, 1.3, 1.6}.")
    L.append(f"2. **Trajectories match the prediction.** At each (T, λ), the seed-mean off-context mass lies "
             f"inside the 95% band at ≥ 80% of those generations 1–9 where the point prediction is ≤ {MAX_M_TEST}. "
             "This is the linear regime: a mass cannot exceed 1, so the law has to saturate. The test passes overall "
             "if it holds for ≥ 7 of the 9 (T, λ) cells.")
    L.append("3. **Contracting cells decelerate.** For cells predicted to contract (a(T)(1−λ) < 1), the "
             "late slope (gens 6–9) is smaller than the early slope (gens 1–4) in ≥ 4/5 seeds. "
             "Supercritical cells are scored by test 4 instead, because a mass bounded by 1 must slow down "
             "as it saturates whatever the gain.")
    L.append(f"4. **Runaway vs bounded.** Cells predicted supercritical (a(T)(1−λ) ≥ 1) exceed an off-context "
             f"mass of {MAX_M_TEST} by generation 9 in ≥ 4/5 seeds. Cells predicted to contract stay below the "
             "upper end of their predicted gen-9 band in ≥ 4/5 seeds. The extra λ* ± 0.1 cells added by "
             "`jobs_v2.py` are scored the same way, which tests the location of the critical point itself.")
    L.append("5. **Temperature trades hallucination for homogenisation.** At generation 9 with λ = 0, "
             "off-context mass increases with T and conditional entropy of sq2 decreases as T decreases "
             "(both monotone over the 4 temperatures).\n")
    L.append("Any failed test is reported as a failure of the law, not explained away. If a(T) stays below 1 "
             "at every T, the prediction is that collapse in this regime is always graded.")
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
