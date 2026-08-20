"""Device-noise robustness: does the Level-8 bias survive gate/readout decoherence?

Complements the finite-shot analysis (make_shots.py). We model Pauli-twirled local depolarizing
noise of per-qubit strength p: a weight-w Pauli expectation is attenuated by (1-p)^w. This is exact
for a global depolarizing channel and the standard approximation for local Pauli noise; crucially the
TWO-qubit bond-correlators <Z_iZ_j>,<X_iX_j> (Level 8's signal) decay as (1-p)^2 — FASTER than the
single-qubit terms — so this stress-tests the measurement readout the hardest. We optionally add
readout bit-flip error e (Pauli-Z expectations -> (1-2e)*<.>).

Train Level-8 (K=6) clean, then evaluate the trained model with noise-attenuated observables and
sweep p; report the structured-scrambled bias vs noise strength. Reuses make_shots' training.
Outputs results/noise_K6.npz and docs/figures/fig16_noise.png.
"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_bias_probe import featurize, scaffold_folds, standardize, per_task_auc, N_TASKS
from make_shots import train_one, raw_obs, logits_from_obs, K

PS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]      # per-qubit depolarizing strength
READOUT_E = 0.02                               # fixed readout bit-flip error on top
plt.rcParams.update({"savefig.dpi": 200, "savefig.bbox": "tight", "font.size": 11,
                     "axes.titlesize": 12, "axes.titleweight": "bold", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                     "font.family": "DejaVu Sans"})
C = {"struct": "#C44E52", "scram": "#4C72B0", "delta": "#55A868", "base": "#8C8C8C"}


def attenuate(single, zz, xx, p, e=0.0):
    """Weight-w Pauli -> (1-p)^w; single-qubit are weight 1, correlators weight 2. Optional
    readout bit-flip e shrinks every expectation by (1-2e)."""
    r = (1 - 2 * e)
    g = (1 - p)
    return single * g * r, zz * g * g * r, xx * g * g * r


def main():
    QF0, AT, AR, Y, SCAF = featurize(K, ["Tox21", "ToxCast"])
    folds = list(scaffold_folds(SCAF, 3))
    N = len(Y)
    store = {"structured": [], "scrambled": []}
    for fi, (tr, va, te) in enumerate(folds):
        QF = standardize(QF0, tr)
        for variant in ("structured", "scrambled"):
            model, adj = train_one(variant, 0, tr, va, te, QF, AT, AR, Y)
            single, zz, xx = raw_obs(model, QF[te], adj[te])
            store[variant].append(dict(te=te, model=model, adj_te=adj[te],
                                       single=single, zz=zz, xx=xx))
        print(f"  fold {fi+1}/3 trained", flush=True)

    def pooled_auc(variant, p, e):
        probs = np.full((N, N_TASKS), np.nan)
        for blk in store[variant]:
            s, zz, xx = attenuate(blk["single"], blk["zz"], blk["xx"], p, e)
            lg = logits_from_obs(blk["model"], s, zz, xx, blk["adj_te"])
            probs[blk["te"]] = 1 / (1 + np.exp(-lg))
        return float(np.nanmean(per_task_auc(Y, probs)))

    rows = []
    for p in PS:
        s = pooled_auc("structured", p, READOUT_E if p > 0 else 0.0)
        c = pooled_auc("scrambled", p, READOUT_E if p > 0 else 0.0)
        rows.append(dict(p=p, s=s, c=c, d=s - c))
        print(f"  p={p:.2f} (readout e={READOUT_E if p>0 else 0}): struct {s:.4f} scram {c:.4f} "
              f"dAUC {s-c:+.4f}", flush=True)

    os.makedirs("results", exist_ok=True)
    np.savez("results/noise_K6.npz", p=np.array(PS), s=np.array([r["s"] for r in rows]),
             c=np.array([r["c"] for r in rows]), d=np.array([r["d"] for r in rows]),
             readout_e=READOUT_E)

    xs = np.array(PS) * 100
    dd = np.array([r["d"] for r in rows]); ss = np.array([r["s"] for r in rows])
    cc = np.array([r["c"] for r in rows])
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.5))
    axs[0].axhline(0, color="#444", ls=":", lw=1)
    axs[0].plot(xs, dd, "-o", color=C["delta"], lw=2.2)
    axs[0].set_xlabel("Per-qubit depolarizing strength p (%)")
    axs[0].set_ylabel("Topology bias ΔAUC (structured − scrambled)")
    axs[0].set_title(f"Bias vs device noise (Level 8, K=6)\n(+{int(READOUT_E*100)}% readout error for p>0)")
    for x, y in zip(xs, dd):
        axs[0].annotate(f"{y:+.4f}", (x, y), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=8, color=C["delta"])
    axs[1].plot(xs, ss, "-o", color=C["struct"], lw=2, label="structured")
    axs[1].plot(xs, cc, "-o", color=C["scram"], lw=2, label="scrambled")
    axs[1].set_xlabel("Per-qubit depolarizing strength p (%)")
    axs[1].set_ylabel("Pooled-CV ROC-AUC (12 Tox21 tasks)")
    axs[1].set_title("Absolute performance vs device noise")
    axs[1].legend(frameon=False)
    fig.suptitle("Device-noise robustness: Pauli-twirled depolarizing — 2-local readout decays as "
                 "(1−p)² (worst case)", y=1.02, fontsize=11)
    fig.savefig("docs/figures/fig16_noise.png")
    print("wrote docs/figures/fig16_noise.png", flush=True)


if __name__ == "__main__":
    main()
