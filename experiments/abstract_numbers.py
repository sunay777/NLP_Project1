"""Every derived number quoted in the extended abstract, from results/v2 only.

Run:  python -m experiments.abstract_numbers  ->  results/v2/analysis/abstract_numbers.json
"""
import glob
import json
from collections import Counter

import numpy as np

from experiments.fit_recursion import load_runs

R = "results/v2"
A = f"{R}/analysis"


def runs(name):
    return [{"rf": r["real_fraction"], "seed": r["seed"], "records": r["records"]}
            for r in load_runs(sorted(glob.glob(f"{R}/{name}/*.json")))]


def ms(xs):
    xs = np.asarray(xs, dtype=float)
    return {"mean": float(np.mean(xs)), "sd": float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0,
            "n": int(len(xs))}


def main():
    main_ext = runs("main_ext")
    fracs = sorted({r["rf"] for r in main_ext})
    out = {"n_runs_main_ext": len(main_ext)}

    recs = [rec for r in main_ext for rec in r["records"]]
    out["query_acc_all_min"] = float(min(rec["query_acc_all"] for rec in recs))
    types = Counter(rec["circuit"]["circuit_type"] for rec in recs)
    out["circuit_types"] = dict(types)
    out["non_canonical_share"] = 1 - types.get("canonical", 0) / sum(types.values())

    per = {}
    for f in fracs:
        rs = [r for r in main_ext if r["rf"] == f]
        last = [r["records"][-1] for r in rs]
        first = [r["records"][0] for r in rs]
        per[f"{f:g}"] = {
            "final_generation": len(rs[0]["records"]) - 1,
            "off_context_final": ms([x["cond_diversity"]["off_context"] for x in last]),
            "causal_L1_mean_abl_gen0": ms([x["circuit"]["causal_drop_L1_mean"] for x in first]),
            "causal_L1_mean_abl_final": ms([x["circuit"]["causal_drop_L1_mean"] for x in last]),
            "query_nll_gen0": ms([x["query_nll_all"] for x in first]),
            "query_nll_final": ms([x["query_nll_all"] for x in last]),
            "query_acc_all_final": ms([x["query_acc_all"] for x in last]),
        }
    out["main_ext_by_fraction"] = per

    fits = json.load(open(f"{R}/analysis/main.json"))["main_ext"]
    oc = fits["off_context"]
    out["law_off_context"] = {"pooled": oc["pooled"], "bootstrap": oc["bootstrap"],
                              "heldout_fit": oc["heldout"]["fit"]}
    h = json.load(open(f"{R}/recursion_fit_v2.json"))["groups"]["all"]["off_context"]["heldout"]
    out["law_off_context"]["heldout_test"] = {k: h[k] for k in ("train_fracs", "rmse", "coverage_1sd",
                                                                "coverage_band95")}
    att_rows = json.load(open(f"{R}/attribution.json"))["rows"]
    out["attribution_top1pct_enrichment"] = {
        k: [min(r["enrichment"][k]["top1pct_enrichment"] for r in att_rows),
            max(r["enrichment"][k]["top1pct_enrichment"] for r in att_rows)]
        for k in ("synthetic_offctx_sq2", "synthetic_wrong_label")}
    out["plateau_lambda0"] = fits["plateau_lambda0"]
    out["gate"] = fits["gate"]
    out["law_query_nll"] = fits["query_nll"]["pooled"] | {"a_ci95": fits["query_nll"]["bootstrap"]["a_ci95"]}

    t = json.load(open(f"{R}/analysis/temperature.json"))["tests"]
    out["prereg_tests"] = {k: bool(v["pass"]) for k, v in t.items()}
    out["prereg_test5"] = {"T": t["5_tradeoff"]["T"], "off_context_gen9": t["5_tradeoff"]["off_context_gen9"],
                           "cond_entropy_gen9": t["5_tradeoff"]["cond_entropy_gen9"]}
    out["prereg_test1_a"] = dict(zip(map(str, t["1_gain_monotone_in_T"]["T"]), t["1_gain_monotone_in_T"]["a"]))
    out["prereg_test2_cells"] = f"{t['2_trajectories_in_band']['n_pass']}/{t['2_trajectories_in_band']['n_cells']}"
    out["prereg_test4_cells"] = {k: v["seeds_consistent"] + (" runaway predicted" if v["kind"] == "supercritical" else "")
                                 for k, v in t["4_runaway_vs_bounded"]["cells"].items()}

    p = json.load(open(f"{R}/analysis/pulse.json"))["pulse_ext:extended: wrong-label rate"]
    out["pulse_wrong_label"] = {lam: {k: p[lam][k] for k in ("decay_ratio_observed", "decay_ratio_predicted",
                                                              "half_life_observed", "half_life_predicted")}
                                for lam in p}

    s = json.load(open(f"{R}/analysis/scrub.json"))
    out["scrub_a"] = {"offctx_removed": s["offctx"]["law_fit_scrubbed"]["a"],
                      "wrong_labels_removed": s["labels"]["law_fit_scrubbed"]["a"],
                      "main": oc["pooled"]["a"]}
    out["scrub_r2"] = {"offctx_removed": s["offctx"]["law_fit_scrubbed"]["r2"],
                       "wrong_labels_removed": s["labels"]["law_fit_scrubbed"]["r2"]}
    out["accumulate_a"] =json.load(open(f"{R}/analysis/accumulate.json"))["law_fit_accumulate"]["a"]
    ps = json.load(open(f"{R}/analysis/poolsize.json"))
    out["poolsize_a"] = {k: ps[k]["a"] for k in ps}
    sc = json.load(open(f"{R}/analysis/scale.json"))
    out["scale"] = {k: ({"a": v["law_fit"]["a"], "r2": v["law_fit"]["r2"]} if "law_fit" in v else
                        {"not_converged": v.get("not_converged"), "gen0_acc": v.get("gen0_acc")})
                    for k, v in sc.items()}

    att = json.load(open(f"{R}/attribution.json"))
    out["attribution"] = att.get("summary", att) if isinstance(att, dict) else att

    sx = json.load(open(f"{R}/analysis/synthctx.json"))
    out["full_lm_lambda0_gen9"] = {c: {k: sx[c][f"{k}_l0"]["mean"][-1] for k in
                                       ("acc_all", "causal_L1_mean", "off_context", "query_nll")}
                                   for c in sx}

    by_lam = {}
    for r in main_ext:
        by_lam.setdefault(r["rf"], []).append(r["records"])
    gen_means = [float(np.mean([rs[g]["circuit"]["causal_drop_L1_mean"] for rs in rl]))
                 for rl in by_lam.values() for g in range(len(rl[0]))]
    out["causal_L1_per_gen_mean_range"] = [min(gen_means), max(gen_means)]
    first_all = [r["records"][0] for r in main_ext]
    out["gen0_main_ext"] = {
        "query_acc_all": ms([x["query_acc_all"] for x in first_all]),
        "prev_token_score": ms([x["circuit"]["prev_token_best"] for x in first_all]),
        "induction_score": ms([x["circuit"]["induction_best"] for x in first_all]),
        "off_context": ms([x["cond_diversity"]["off_context"] for x in first_all]),
    }
    out["full_lm_gen0_causal_L1_mean"] = {c: sx[c]["causal_L1_mean_l0"]["mean"][0] for c in sx}

    base = runs("main_base")
    b0 = [r["records"] for r in base if r["rf"] == 0.0]
    bfit = json.load(open(f"{A}/main.json"))["main_base"]["syn_wrong_label"]["pooled"]
    out["base_lambda0"] = {
        "n_runs_main_base": len(base),
        "final_generation": len(b0[0]) - 1,
        "query_acc_all_gen0": ms([rs[0]["query_acc_all"] for rs in b0]),
        "query_acc_all_final": ms([rs[-1]["query_acc_all"] for rs in b0]),
        "query_nll_gen0": ms([rs[0]["query_nll_all"] for rs in b0]),
        "query_nll_final": ms([rs[-1]["query_nll_all"] for rs in b0]),
        "pool_syn_wrong_label_gen1": ms([rs[1]["pool_syn_wrong_label"] for rs in b0]),
        "pool_syn_wrong_label_final": ms([rs[-1]["pool_syn_wrong_label"] for rs in b0]),
        "law_syn_wrong_label": {"a": bfit["a"], "r2": bfit["r2"], "n": bfit["n"]},
    }

    out["runaway_lambda0"] = {}
    for T in ("1.3", "1.6"):
        rl = [r["records"] for r in runs(f"temp_T{T}") if r["rf"] == 0.0]
        G = min(len(x) for x in rl)
        def per_gen(f):
            return [float(np.mean([f(x[g]) for x in rl])) for g in range(G)]
        out["runaway_lambda0"][T] = {
            "n": len(rl),
            "query_acc_all": per_gen(lambda x: x["query_acc_all"]),
            "causal_L1_mean": per_gen(lambda x: x["circuit"]["causal_drop_L1_mean"]),
            "off_context": per_gen(lambda x: x["cond_diversity"]["off_context"]),
            "pool_label_correct": per_gen(lambda x: x["pool_label_correct"]),
        }

    pred = json.load(open("paper/PREDICTION.json"))["temperatures"]
    out["prereg_calibration"] = {T: {"a": v["a"], "lambda_star": v["lambda_star"]} for T, v in pred.items()}

    from experiments.analyse_v2 import law_fit, runs_of
    def gain_ci(rs, fracs):
        return law_fit(rs, train_fracs=fracs, test_fracs=[])["bootstrap"]["a_ci95"]
    def pool_runs(name):
        rs = [r for r in runs_of(name) if r["real_fraction"] in (0.0, 1.0)]
        if name == "main_ext":
            rs = [dict(r, records=r["records"][:10]) for r in rs if r["seed"] < 5]
        return rs
    out["intervention_a_ci95"] = {
        "offctx_removed": gain_ci(runs_of("scrub_offctx"), [0.0]),
        "wrong_labels_removed": gain_ci(runs_of("scrub_labels"), [0.0]),
        "accumulate": gain_ci(runs_of("accumulate"), [0.0, 0.25]),
        "pool5k": gain_ci(pool_runs("pool5k"), [0.0, 1.0]),
        "main_ext_matched": gain_ci(pool_runs("main_ext"), [0.0, 1.0]),
        "pool80k": gain_ci(pool_runs("pool80k"), [0.0, 1.0]),
    }

    out["poolsize_off_context_lambda0_final"] = {
        name: ms([r["records"][-1]["cond_diversity"]["off_context"] for r in runs(name) if r["rf"] == 0.0])
        for name in ("pool5k", "pool80k")}

    json.dump(out, open(f"{R}/analysis/abstract_numbers.json", "w"), indent=1, default=float)
    print(f"-> {R}/analysis/abstract_numbers.json")


if __name__ == "__main__":
    main()
