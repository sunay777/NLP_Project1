"""The v2 experiment queue: every run behind the v2 results, one job per
(condition, real_fraction, seed) trajectory, so a crash or restart loses at most
one trajectory per worker. experiments/run_queue.py executes these.

Common settings (v2): extended 6000 / base 4000 steps per generation, all K
query labels model-generated, gen 0 trained on a fixed real pool of the same
size as later pools (--gen0_pool: removes the finite-data confound), strict
gate (0.98 x pool label correctness AND causal_drop_L1 >= 0.5), every
generation's weights saved, CPU with 1 torch thread per job (deterministic, so
runs that share a seed share their pools up to the point where they differ:
pulse, temperature and scrub runs are PAIRED with the main sweep).

Cost unit = one extended generation of the default model at 6000 steps.
Stages run in order; within a stage, the longest jobs start first.

Run:  python -m experiments.jobs_v2          (prints the table and total cost)
"""
import json
import os

ROOT = "results/v2"
COMMON = ["--gen_queries", "all", "--gen0_pool", "--gate", "0.98", "--gate_causal", "0.5",
          "--max_retries", "3", "--save_models", "all"]
EXT = ["--mode", "extended", "--steps", "6000"]
BASE = ["--mode", "base", "--steps", "4000"]
FRACS = [0.0, 0.1, 0.25, 0.5, 1.0]
BASE_COST = 0.55      # base generation relative to extended (shorter sequence, 4000 steps)


def _job(stage, name, frac, seed, gens, args, cost_per_gen=1.0):
    out = f"{ROOT}/{name}/rf{frac:g}_s{seed}.json"
    return {"stage": stage, "name": name, "real_fraction": frac, "seed": seed,
            "generations": gens, "out": out, "est_units": round(gens * cost_per_gen, 2),
            "args": COMMON + args + ["--generations", str(gens), "--fractions", f"{frac:g}",
                                     "--seeds", str(seed)]}


def prediction_extras():
    """Extra lambda values near a pre-registered critical point (paper/PREDICTION.json):
    lambda* - 0.1, lambda*, lambda* + 0.1 for every temperature with a(T) > 1."""
    p = "paper/PREDICTION.json"
    if not os.path.exists(p):
        return {}
    pred = json.load(open(p))
    out = {}
    for T, v in pred.get("temperatures", {}).items():
        ls = v.get("lambda_star")
        if ls is not None:
            out[float(T)] = sorted({round(min(0.95, max(0.0, ls + d)), 2) for d in (-0.1, 0.0, 0.1)})
    return out


def jobs():
    J = []
    # ---------------- stage 1: core (headline numbers) ----------------
    # main sweep, 8 seeds; lambda=0 runs 20 generations to test the predicted plateau
    # m* = eps / (1 - a): its half-life ~7 generations, the others' <= 3.5
    for f in FRACS:
        for s in range(8):
            g = 20 if f == 0.0 else 12
            extra = ["--save_pools"] if (f == 0.0 and s < 3) else []   # for data attribution
            J.append(_job(1, "main_ext", f, s, g, EXT + extra))
    for f in FRACS:
        for s in range(5):
            J.append(_job(1, "main_base", f, s, 12, BASE, BASE_COST))
    # pre-registered temperature test (T = 1 is the main sweep); paired with main seeds
    extras = prediction_extras()
    for T in (0.7, 1.3, 1.6):
        for f in sorted({0.0, 0.25, 0.5, *extras.get(T, [])}):
            for s in range(5):
                J.append(_job(1, f"temp_T{T:g}", f, s, 10, EXT + ["--temperature", f"{T:g}"]))
    # poison pulse at gen 1 (5 % of synthetic query labels and of sq2), paired with main
    for f in (0.0, 0.25, 0.5):
        for s in range(5):
            J.append(_job(1, "pulse_ext", f, s, 8, EXT + [
                "--pulse_gen", "1", "--pulse_label_noise", "0.05", "--pulse_offcontext", "0.05"]))
            J.append(_job(1, "pulse_base", f, s, 8, BASE + [
                "--pulse_gen", "1", "--pulse_label_noise", "0.05"], BASE_COST))

    # ---------------- stage 2: mechanism of inheritance + Shumailov links -------------
    # counterfactual attribution: scrub the inherited errors from every synthetic pool
    for sc in ("offctx", "labels"):
        for s in range(5):
            J.append(_job(2, f"scrub_{sc}", 0.0, s, 12, EXT + ["--scrub", sc]))
    # accumulate instead of replace (Gerstgrasser et al., 2024)
    for f in (0.0, 0.25):
        for s in range(5):
            J.append(_job(2, "accumulate", f, s, 12, EXT + ["--accumulate"]))
    # Zipfian free choice: a distribution with tails
    for al in (1.0, 2.0):
        for f in (0.0, 1.0):
            for s in range(5):
                J.append(_job(2, f"zipf{al:g}", f, s, 10, EXT + ["--sq2_zipf", f"{al:g}"]))
    # pool size (statistical approximation error), 20k is the main sweep
    for n in (5000, 80000):
        for f in (0.0, 1.0):
            for s in range(5):
                J.append(_job(2, f"pool{n // 1000}k", f, s, 10, EXT + ["--pool_size", str(n)],
                              1.0 if n <= 20000 else 1.15))
    # fully synthetic contexts: full next-token loss; prompt-real control vs model-written
    for f in (0.0, 1.0):
        for s in range(5):
            J.append(_job(2, "fulllm_promptreal", f, s, 10, EXT + ["--full_lm"], 1.05))
    for f in (0.0, 0.25):
        for s in range(5):
            J.append(_job(2, "fulllm_synthctx", f, s, 10, EXT + ["--full_lm", "--gen_context"], 1.2))

    # ---------------- stage 3: generality + extra power ----------------
    archs = {"L3": (["--n_layers", "3"], 1.5), "L4": (["--n_layers", "4"], 2.0),
             "L2mlp": (["--use_mlp"], 1.6), "L2d128": (["--d_model", "128"], 1.8)}
    for an, (aa, c) in archs.items():
        for f in (0.0, 1.0):
            for s in range(4):
                J.append(_job(3, f"arch_{an}", f, s, 10, EXT + aa, c))
    for s in range(3):     # 1 layer cannot compose a previous-token head: expected to fail
        J.append(_job(3, "arch_L1", 0.0, s, 1, EXT + ["--n_layers", "1", "--max_retries", "1"], 0.6))
    tasks = {"K6": (["--n_pairs", "6"], 1.4), "S32": (["--n_symbols", "32"], 1.0)}
    for tn, (ta, c) in tasks.items():
        for f in (0.0, 1.0):
            for s in range(4):
                J.append(_job(3, f"task_{tn}", f, s, 10, EXT + ta, c))
    for f in FRACS:        # seeds 8-9 bring the main sweep to 10 seeds
        for s in (8, 9):
            J.append(_job(3, "main_ext", f, s, 20 if f == 0.0 else 12, EXT))
    return J


def main():
    J = jobs()
    tot = {}
    for j in J:
        k = (j["stage"], j["name"])
        n, u = tot.get(k, (0, 0.0))
        tot[k] = (n + 1, u + j["est_units"])
    print(f"{'stage':>5} {'experiment':22s} {'jobs':>5} {'units':>8}")
    for (st, nm), (n, u) in sorted(tot.items()):
        print(f"{st:>5} {nm:22s} {n:>5} {u:>8.1f}")
    for st in sorted({j['stage'] for j in J}):
        print(f"stage {st}: {sum(j['est_units'] for j in J if j['stage'] == st):.0f} units")
    print(f"total: {len(J)} jobs, {sum(j['est_units'] for j in J):.0f} units "
          f"(1 unit = one 6000-step extended generation)")


if __name__ == "__main__":
    main()
