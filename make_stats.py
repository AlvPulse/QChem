"""Effect sizes, bootstrap CIs and multiplicity control for the report (Table III.2).

  * Holm-Bonferroni adjustment across the 6 probe cells (published Wilcoxon p in report_data).
  * Matched-pairs rank-biserial effect size + bootstrap 95% CI on the median per-task ΔAUC,
    for the cells where we have the 12 per-task arrays (results/figdata/*.npz) -- the gate-K4 and
    Level-8 K6 reproductions. Clearly a reproduction-based effect size (the headline p-values are the
    higher-fidelity runs); both directionally match.
Outputs results/stats_summary.json and prints a markdown-ready table.
"""
import json, numpy as np
import report_data as R

CELLS = [("gate", 4), ("gate", 6), ("gate", 8), ("levelG", 4), ("levelG", 6), ("levelG", 8),
         ("meas_only", 4)]


def holm(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)            # enforce monotonicity
        adj[idx] = min(1.0, running)
    return adj


def rank_biserial(d):
    """Matched-pairs rank-biserial r = (W+ - W-)/(W+ + W-) from signed ranks of |d|."""
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.abs(d))) + 1.0
    wp = ranks[d > 0].sum(); wn = ranks[d < 0].sum()
    return float((wp - wn) / (wp + wn))


def boot_ci(d, B=5000, seed=0):
    rng = np.random.default_rng(seed)
    meds = [np.median(rng.choice(d, len(d), replace=True)) for _ in range(B)]
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def main():
    pub_p = np.array([R.DECOMP[c]["wil_p"] for c in CELLS])
    adj = holm(pub_p)
    rows = []
    for (c, k), p, pa in zip(CELLS, pub_p, adj):
        d = R.DECOMP[(c, k)]
        rows.append(dict(cell=f"{c} K{k}", median_dauc=d["median_dauc"], npos=d["npos"],
                         wilcoxon_p=float(p), holm_p=float(pa)))

    # reproduction effect sizes where we have per-task arrays
    repro = {}
    for kind, k in [("gate", 4), ("levelG", 6)]:
        try:
            z = np.load(f"results/figdata/{kind}_K{k}.npz")
        except FileNotFoundError:
            continue
        d = (z["auc_s"] - z["auc_c"])
        d = d[~np.isnan(d)]
        lo, hi = boot_ci(d)
        repro[f"{kind}_K{k}"] = dict(median_dauc=float(np.median(d)),
                                     rank_biserial=rank_biserial(d), ci=[lo, hi],
                                     npos=int((d > 0).sum()), n=int(len(d)))

    out = dict(holm=rows, reproduction_effect_sizes=repro,
               randomsplit_published_ci=list(R.RANDSPLIT["ci95"]))
    with open("results/stats_summary.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n== Holm-adjusted across the {len(CELLS)} probe cells ==")
    print(f"{'cell':<12} {'med_dAUC':>9} {'pos':>5} {'Wilcoxon':>9} {'Holm':>9}")
    for r in rows:
        print(f"{r['cell']:<12} {r['median_dauc']:>+9.4f} {r['npos']:>3}/12 "
              f"{r['wilcoxon_p']:>9.4g} {r['holm_p']:>9.4g}")
    print("\n== Reproduction effect sizes (per-task, 12 Tox21 tasks) ==")
    for name, v in repro.items():
        print(f"{name:<12} med_dAUC {v['median_dauc']:+.4f}  rank-biserial {v['rank_biserial']:+.3f}  "
              f"95% CI [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]  ({v['npos']}/{v['n']} pos)")
    print(f"\nrandom-split published 95% CI (gate K4): "
          f"[{R.RANDSPLIT['ci95'][0]:+.4f}, {R.RANDSPLIT['ci95'][1]:+.4f}]")


if __name__ == "__main__":
    main()
