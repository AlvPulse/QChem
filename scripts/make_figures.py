"""Publication-quality figures for docs/06_results_benchmarking.md.

Two tiers:
  AGGREGATE  figures depend only on report_data.py (the full-run published numbers) -> always
             produced, no compute. These carry every quantitative/significance claim.
  FINEGRAIN  figures consume results/figdata/*.npz (the reduced-fidelity corroboration harness,
             make_figdata.py). Produced only if the npz exist; labelled as reduced-fidelity so
             they are never confused with the headline numbers.

All figures share one house style and are written to docs/figures/.
"""
import os, glob, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import report_data as R

OUT = "docs/figures"
os.makedirs(OUT, exist_ok=True)

# ----------------------------- house style -----------------------------
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 11, "legend.fontsize": 9.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.6, "font.family": "DejaVu Sans",
})
C = {  # consistent colours
    "gate": "#4C72B0", "levelG": "#C44E52", "meas_only": "#8C8C8C",
    "classicalGNN": "#55A868",
    "structured": "#C44E52", "scrambled": "#4C72B0", "separable": "#55A868",
    "classical": "#8172B3", "zero": "#444444", "sig": "#C44E52", "ns": "#B0B0B0",
}
SIG = lambda p: p == p and p < 0.05


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p)
    plt.close(fig)
    print("wrote", p, flush=True)


# ================================ AGGREGATE ================================
def fig_bias_vs_qubits():
    """Headline scaling figure: the graph-topology bias is SUBSTRATE-INDEPENDENT (quantum AND
    classical) and GROWS with the number of coarse-graph nodes K; n.s. at K=4 for all, robust by
    K=8. Marker labels carry the per-cell seed count (n.s. at K=4 corrects an earlier 2-seed
    artifact)."""
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    off = {"gate": -16, "levelG": 11, "classicalGNN": -16}
    for cfg in ("gate", "levelG", "classicalGNN"):
        ks = sorted([k for (c, k) in R.DECOMP if c == cfg])
        xs = [k for k in ks]
        ys = [R.DECOMP[(cfg, k)]["median_dauc"] for k in ks]
        ps = [R.DECOMP[(cfg, k)]["wil_p"] for k in ks]
        ss = [R.DECOMP[(cfg, k)].get("seeds", "?") for k in ks]
        ax.plot(xs, ys, "-", color=C[cfg], lw=2.2, zorder=2,
                label=R.CONFIG_LABEL[cfg].split(" (")[0])
        for x, y, p, s in zip(xs, ys, ps, ss):
            ax.scatter([x], [y], s=130 if SIG(p) else 80,
                       facecolor=C[cfg] if SIG(p) else "white",
                       edgecolor=C[cfg], linewidth=2, zorder=3)
            ax.annotate(("p=%.3f" % p) + (" *" if SIG(p) else "") + f"\n[{s}s]", (x, y),
                        textcoords="offset points", xytext=(0, off[cfg]),
                        ha="center", fontsize=7.6, color=C[cfg])
    ax.axhline(0, color=C["zero"], lw=1, ls="--", alpha=0.7)
    ax.set_xticks([4, 6, 8])
    ax.set_ylim(-0.0055, 0.0175)
    ax.set_xlabel("Qubits K  (=  coarse graph nodes)")
    ax.set_ylabel("Median per-task ΔAUC  (structured − scrambled)")
    ax.set_title("Graph-topology bias vs circuit size (quantum & classical)")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.text(0.99, 0.01, "filled = Wilcoxon p<0.05 (scaffold CV, 12 Tox21 tasks); [Ns] = seeds",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color="#666")
    save(fig, "fig01_bias_vs_qubits.png")


def fig_decomposition():
    """Place-then-harvest: K=4 bias for gate / levelG / meas_only."""
    order = ["gate", "levelG", "meas_only"]
    vals = [R.DECOMP[(c, 4)]["median_dauc"] for c in order]
    ps = [R.DECOMP[(c, 4)]["wil_p"] for c in order]
    labels = ["Gate-gated\n(single-qubit\nreadout)", "Level 8\n(+ bond-correlator\nreadout)",
              "Measurement-only\n(fixed ring\nentangler)"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    bars = ax.bar(range(3), vals, width=0.6,
                  color=[C["gate"], C["levelG"], C["meas_only"]],
                  edgecolor="black", linewidth=0.6)
    ax.axhline(0, color=C["zero"], lw=1)
    for i, (v, p) in enumerate(zip(vals, ps)):
        star = " *" if SIG(p) else ""
        ax.annotate(f"{v:+.4f}\np={p:.3g}{star}", (i, v),
                    textcoords="offset points", xytext=(0, 8 if v >= 0 else -22),
                    ha="center", fontsize=9)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Median per-task ΔAUC (structured − scrambled)")
    ax.set_title("Level-8 decomposition at K = 4: where the bias lives")
    ax.text(0.5, -0.30, "Readout amplifies the gate bias (+0.0044→+0.0078) but only WITH graph-gated\n"
                        "entanglement; on a fixed ring (meas_only) bond-pooling the true graph HURTS.",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.3, color="#555")
    save(fig, "fig02_decomposition.png")


def fig_absolute_ordering():
    """Absolute pooled ROC-AUC ordering per K: separable < scrambled < structured << classical."""
    ks = [4, 6, 8]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    w = 0.2
    x = np.arange(len(ks))
    sep = [R.CONTEXT[k]["separable"] for k in ks]
    scr = [R.DECOMP[("gate", k)]["scram"] for k in ks]
    st = [R.DECOMP[("gate", k)]["struct"] for k in ks]
    cls = [R.CONTEXT[k]["classical"] for k in ks]
    ax.bar(x - 1.5 * w, sep, w, label="separable (no entangler)", color=C["separable"])
    ax.bar(x - 0.5 * w, scr, w, label="scrambled (random graph)", color=C["scrambled"])
    ax.bar(x + 0.5 * w, st, w, label="structured (true graph)", color=C["structured"])
    ax.bar(x + 1.5 * w, cls, w, label="classical MLP (unconstrained)", color=C["classical"])
    ax.set_xticks(x); ax.set_xticklabels([f"K={k}" for k in ks])
    ax.set_ylim(0.55, 0.74)
    ax.set_ylabel("Pooled-CV ROC-AUC (12 Tox21 tasks)")
    ax.set_title("Absolute performance ordering (gate-gated family)")
    ax.legend(loc="upper center", ncol=2, frameon=False, fontsize=8.5)
    save(fig, "fig03_absolute_ordering.png")


def fig_forest():
    """Forest plot: median ΔAUC per (config,K) with run-level min/max whiskers."""
    rows = [("gate", 4), ("gate", 6), ("gate", 8), ("levelG", 4), ("levelG", 6), ("levelG", 8),
            ("meas_only", 4)]
    labels, meds, los, his, cols = [], [], [], [], []
    for c, k in rows:
        d = R.DECOMP[(c, k)]
        rd = d["run_deltas"]
        labels.append(f"{R.CONFIG_LABEL[c].split(' (')[0]}  K={k}")
        meds.append(d["median_dauc"]); los.append(min(rd)); his.append(max(rd))
        cols.append(C["sig"] if SIG(d["wil_p"]) else C["ns"])
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for yi, m, lo, hi, col in zip(y, meds, los, his, cols):
        ax.plot([lo, hi], [yi, yi], color=col, lw=2, alpha=0.6, zorder=1)
        ax.scatter([m], [yi], s=90, color=col, zorder=2, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color=C["zero"], ls="--", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("ΔAUC (structured − scrambled);  dot = pooled per-task median, bar = run-level range")
    ax.set_title("Inductive-bias effect sizes with run-level spread")
    ax.legend(handles=[Patch(color=C["sig"], label="Wilcoxon p<0.05"),
                       Patch(color=C["ns"], label="not significant")],
              loc="lower right", frameon=False)
    save(fig, "fig04_forest.png")


def fig_absorbability():
    """Bit-exact absorbability residual per benchmark level -> which controls are valid."""
    levels = [l for l in R.ABSORB if isinstance(l, int)]
    levels.sort()
    res = [R.ABSORB[l][0] for l in levels]
    status = [R.ABSORB[l][1] for l in levels]
    cmap = {"no control": "#cccccc", "VACUOUS": "#C44E52", "partial": "#DD8452", "genuine": "#55A868"}
    plotted = [(1e-4 if (r is None or r == 0) else r) for r in res]  # floor for log axis
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    bars = ax.bar(range(len(levels)), plotted, color=[cmap[s] for s in status],
                  edgecolor="black", linewidth=0.6)
    ax.set_yscale("log")
    ax.set_ylim(5e-5, 5)
    for i, (r, s) in enumerate(zip(res, status)):
        txt = "0 (bit-exact)" if r == 0 else ("n/a" if r is None else f"{r:.2g}")
        ax.annotate(txt, (i, plotted[i]), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=8)
    ax.set_xticks(range(len(levels))); ax.set_xticklabels([f"L{l}" for l in levels])
    ax.set_xlabel("run_benchmark.py circuit level")
    ax.set_ylabel("residual  max|structured − scrambled(permuted)|  (log)")
    ax.set_title("Is the benchmark's scramble control valid?  (lower = absorbable = vacuous)")
    ax.legend(handles=[Patch(color=cmap[k], label=k) for k in ["VACUOUS", "partial", "genuine", "no control"]],
              loc="upper left", frameon=False, ncol=2)
    save(fig, "fig05_absorbability.png")


def fig_run_consistency():
    """Per-seed run-level dAUC dots: levelG is consistently positive; gate flips sign at K>=6."""
    rows = [("gate", 4), ("levelG", 4), ("gate", 6), ("levelG", 6)]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for i, (c, k) in enumerate(rows):
        rd = R.DECOMP[(c, k)]["run_deltas"]
        xs = np.full(len(rd), i) + np.linspace(-0.08, 0.08, len(rd))
        ax.scatter(xs, rd, s=70, color=C[c], edgecolor="black", linewidth=0.5, zorder=3)
        ax.scatter([i], [np.mean(rd)], marker="_", s=600, color=C[c], zorder=2)
    ax.axhline(0, color=C["zero"], ls="--", lw=1)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"{c}\nK={k}" for c, k in rows], fontsize=9)
    ax.set_ylabel("Run-level ΔAUC per seed (− = scrambled won that seed)")
    ax.set_title("Seed-level consistency of the bias")
    save(fig, "fig06_run_consistency.png")


def fig_classical_gap():
    ks = [4, 6, 8]
    st = [R.CONTEXT[k]["structured"] for k in ks]
    cls = [R.CONTEXT[k]["classical"] for k in ks]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(ks, cls, "-o", color=C["classical"], lw=2.2, label="classical MLP")
    ax.plot(ks, st, "-o", color=C["structured"], lw=2.2, label="structured quantum")
    for k, a, b in zip(ks, st, cls):
        ax.annotate(f"−{(b-a)*100:.1f} pts", (k, (a + b) / 2), ha="center", fontsize=8.5, color="#555")
    ax.set_xticks(ks); ax.set_xlabel("Qubits K"); ax.set_ylabel("Pooled-CV ROC-AUC")
    ax.set_title("Honest context: classical leads at every scale")
    ax.legend(frameon=False)
    save(fig, "fig07_classical_gap.png")


def fig_radar():
    """Qualitative scorecard (0–1, rubric in docs/06). gate vs Level 8 vs classical."""
    axes = ["Absolute\nAUC", "Bias\nmagnitude", "Bias scales\nwith K",
            "Statistical\nsignificance", "Non-\nabsorbable", "Hardware\nfrugality"]
    # rubric (documented in chapter): normalized 0..1, higher=better
    scores = {
        "gate":     [0.50, 0.40, 0.20, 0.50, 1.00, 0.85],
        "levelG":   [0.55, 0.85, 0.95, 0.90, 1.00, 0.70],
        "classical":[1.00, 0.00, 0.00, 0.00, 1.00, 1.00],
    }
    N = len(axes)
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    ang += ang[:1]
    fig, ax = plt.subplots(figsize=(6.2, 6.2), subplot_kw=dict(polar=True))
    for key, col, lab in [("classical", C["classical"], "classical MLP"),
                          ("gate", C["gate"], "gate-gated quantum"),
                          ("levelG", C["levelG"], "Level 8 quantum")]:
        v = scores[key] + scores[key][:1]
        ax.plot(ang, v, color=col, lw=2, label=lab)
        ax.fill(ang, v, color=col, alpha=0.12)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(axes, fontsize=9)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["", "", "", ""])
    ax.set_ylim(0, 1)
    ax.set_title("Scorecard: bias mechanisms vs classical", pad=24, y=1.06)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False, fontsize=9)
    save(fig, "fig08_radar.png")


# ================================ FINEGRAIN ================================
def _load(kind, k):
    p = f"results/figdata/{kind}_K{k}.npz"
    return np.load(p) if os.path.exists(p) else None


# The illustrative diagnostics are pinned to the Level-8 (levelG) K=6 run -- the HEADLINE scaling
# cell, whose reduced-fidelity reproduction strongly CORROBORATES the published result (median
# +0.0078, 11/12 tasks, Wilcoxon p=0.0007 here vs the full-run +0.0108, p=0.011). The chapter's
# §III.b prose names this exact run, so figure selection is deterministic, not "best available".
# (The levelG K=4 reduced run is deliberately NOT used: its 2 seeds disagreed sharply,
# [+0.0123, -0.0066], and would misrepresent the small-K cell.)
def _illustrative():
    return _load("levelG", 6) or _load("gate", 4)


def fig_training_curves():
    found = False
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    z = _illustrative()
    if z is not None and z["curve_s"].size:
        found = True
        col = C["levelG"]
        ep = np.arange(1, len(z["curve_s"]) + 1)
        ax.plot(ep, z["curve_s"], "-", color=col, lw=2, label="Level 8 (K=6) structured")
        ax.plot(ep, z["curve_c"], "--", color=col, lw=1.6, alpha=0.7, label="Level 8 (K=6) scrambled")
    if not found:
        plt.close(fig); return
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation ROC-AUC (12 Tox21 tasks)")
    ax.set_title("Learning curves  (reduced-fidelity reproduction)")
    ax.legend(frameon=False, fontsize=8.5)
    ax.text(0.98, 0.03, "illustrative: 3 folds × 2 seeds × 18 ep; headline numbers in Tables 2–3",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color="#777")
    save(fig, "fig09_training_curves.png")


def fig_roc_pr_cal():
    z = _illustrative()
    if z is None:
        return
    from sklearn.metrics import roc_curve, precision_recall_curve
    Y, ps, pc = z["Y"], z["pool_s"], z["pool_c"]
    # micro-average across tasks (flatten valid entries)
    def flat(P):
        m = ~np.isnan(Y) & ~np.isnan(P)
        return Y[m].ravel(), P[m].ravel()
    yt_s, yp_s = flat(ps); yt_c, yp_c = flat(pc)
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.2))
    # ROC
    for yt, yp, col, lab in [(yt_s, yp_s, C["structured"], "structured"), (yt_c, yp_c, C["scrambled"], "scrambled")]:
        fpr, tpr, _ = roc_curve(yt, yp)
        axs[0].plot(fpr, tpr, color=col, lw=2, label=lab)
    axs[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    axs[0].set_xlabel("FPR"); axs[0].set_ylabel("TPR"); axs[0].set_title("ROC (micro-avg)")
    axs[0].legend(frameon=False)
    # PR
    for yt, yp, col, lab in [(yt_s, yp_s, C["structured"], "structured"), (yt_c, yp_c, C["scrambled"], "scrambled")]:
        pr, rc, _ = precision_recall_curve(yt, yp)
        axs[1].plot(rc, pr, color=col, lw=2, label=lab)
    axs[1].set_xlabel("Recall"); axs[1].set_ylabel("Precision"); axs[1].set_title("Precision–Recall")
    axs[1].legend(frameon=False)
    # calibration
    for yt, yp, col, lab in [(yt_s, yp_s, C["structured"], "structured"), (yt_c, yp_c, C["scrambled"], "scrambled")]:
        bins = np.linspace(0, 1, 11); idx = np.digitize(yp, bins) - 1
        xs, ys = [], []
        for b in range(10):
            m = idx == b
            if m.sum() > 30:
                xs.append(yp[m].mean()); ys.append(yt[m].mean())
        axs[2].plot(xs, ys, "-o", color=col, lw=2, ms=4, label=lab)
    axs[2].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    axs[2].set_xlabel("Mean predicted prob"); axs[2].set_ylabel("Observed frequency")
    axs[2].set_title("Calibration"); axs[2].legend(frameon=False)
    fig.suptitle("Pooled-CV diagnostics, structured vs scrambled (reduced-fidelity reproduction)",
                 y=1.02, fontsize=11)
    save(fig, "fig10_roc_pr_cal.png")


def fig_per_task_forest():
    z = _illustrative()
    if z is None:
        return
    s, c = z["auc_s"], z["auc_c"]
    d = s - c
    order = np.argsort(d)
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    y = np.arange(len(d))
    cols = [C["structured"] if v > 0 else C["scrambled"] for v in d[order]]
    ax.barh(y, d[order], color=cols, edgecolor="black", linewidth=0.4)
    ax.axvline(0, color=C["zero"], lw=1)
    ax.set_yticks(y); ax.set_yticklabels([f"task {t}" for t in order], fontsize=8)
    ax.set_xlabel("Per-task ΔAUC (structured − scrambled)")
    ax.set_title("Per-task bias (reduced-fidelity reproduction)")
    ax.text(0.98, 0.03, "illustrative; pooled headline median in Table 3",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color="#777")
    save(fig, "fig11_per_task.png")


if __name__ == "__main__":
    print("=== aggregate (from report_data) ===", flush=True)
    fig_bias_vs_qubits(); fig_decomposition(); fig_absolute_ordering(); fig_forest()
    fig_absorbability(); fig_run_consistency(); fig_classical_gap(); fig_radar()
    print("=== finegrain (from results/figdata, if present) ===", flush=True)
    fig_training_curves(); fig_roc_pr_cal(); fig_per_task_forest()
    print("FIGURES DONE", flush=True)
