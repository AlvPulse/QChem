"""Representation probing: do the STRUCTURED quantum features encode the true graph topology?

Interpretability evidence beyond the place-then-harvest correlation figure (fig14). We freeze the
Level-8 (K=6) readout features `[<X,Y,Z>_i ; bond-pooled <Z_iZ_j>, <X_iX_j>]` (5K per molecule) and
train simple linear probes (ridge, 5-fold CV R²) to predict properties of the molecule, for the
structured circuit (true adjacency) vs the scrambled circuit (random adjacency, same weight
multiset).

Two probe families:
  * TOPOLOGY targets (depend on which pairs are bonded -> differ between true and shuffled graph):
      largest adjacency eigenvalue lambda_max(A_true), algebraic connectivity (Fiedler value).
    Prediction: structured features predict these BETTER (they entangle along the true bonds).
  * NODE-FEATURE controls (independent of topology, identical single-qubit encoding in both):
      mean |Gasteiger charge|, aromatic fraction.
    Prediction: structured ~= scrambled (no topology advantage expected).

A structured>scrambled gap on topology targets but parity on node controls is direct evidence that
the inductive bias lives in a topology-aware representation. Outputs results/probe_K6.npz and
docs/figures/fig17_probe.png.
"""
import os, numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from run_bias_probe import featurize, scaffold_folds, standardize
from make_shots import train_one, raw_obs, K

plt.rcParams.update({"savefig.dpi": 200, "savefig.bbox": "tight", "font.size": 11,
                     "axes.titlesize": 12, "axes.titleweight": "bold", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                     "font.family": "DejaVu Sans"})
C = {"struct": "#C44E52", "scram": "#4C72B0"}


def features(model, QF_te, adj_te):
    single, zz, xx = raw_obs(model, QF_te, adj_te)
    with torch.no_grad():
        bzz = model._bond_pool(torch.tensor(zz), torch.tensor(adj_te)).numpy()
        bxx = model._bond_pool(torch.tensor(xx), torch.tensor(adj_te)).numpy()
    return np.concatenate([single, bzz, bxx], axis=1)


def graph_targets(A, QF_raw):
    """Per-molecule probe targets from the TRUE coarse graph A (B,K,K) and raw features QF (B,K,5)."""
    B = A.shape[0]
    lam = np.zeros(B); fied = np.zeros(B)
    for m in range(B):
        Am = A[m]
        w = np.linalg.eigvalsh(Am)
        lam[m] = w[-1]                                   # largest adjacency eigenvalue (topology)
        d = Am.sum(1); L = np.diag(d) - Am               # graph Laplacian
        lw = np.sort(np.linalg.eigvalsh(L))
        fied[m] = lw[1] if len(lw) > 1 else 0.0          # algebraic connectivity (topology)
    charge = np.abs(QF_raw[:, :, 1]).mean(1)             # node control
    aromatic = QF_raw[:, :, 3].mean(1)                   # node control
    return {"lambda_max(A)  [topology]": lam, "Fiedler conn.  [topology]": fied,
            "mean |charge|  [node]": charge, "aromatic frac  [node]": aromatic}


def probe_r2(F, y):
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    return float(np.mean(cross_val_score(pipe, F, y, cv=5, scoring="r2")))


def main():
    QF0, AT, AR, Y, SCAF = featurize(K, ["Tox21", "ToxCast"])
    tr, va, te = next(scaffold_folds(SCAF, 3))
    QF = standardize(QF0, tr)
    m_s, adj_s = train_one("structured", 0, tr, va, te, QF, AT, AR, Y)
    m_c, adj_c = train_one("scrambled", 0, tr, va, te, QF, AT, AR, Y)
    print("  trained structured + scrambled", flush=True)
    F_s = features(m_s, QF[te], adj_s[te])               # structured features (true-A circuit)
    F_c = features(m_c, QF[te], adj_c[te])               # scrambled features (random-A circuit)
    targets = graph_targets(AT[te], QF0[te])             # targets always from the TRUE graph

    rows = []
    for name, y in targets.items():
        r_s, r_c = probe_r2(F_s, y), probe_r2(F_c, y)
        rows.append((name, r_s, r_c))
        print(f"  {name:<24} structured R2 {r_s:+.3f}  scrambled R2 {r_c:+.3f}  diff {r_s-r_c:+.3f}",
              flush=True)

    os.makedirs("results", exist_ok=True)
    np.savez("results/probe_K6.npz", names=[r[0] for r in rows],
             r_struct=[r[1] for r in rows], r_scram=[r[2] for r in rows])

    names = [r[0] for r in rows]; rs = [r[1] for r in rows]; rc = [r[2] for r in rows]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    h = 0.38
    ax.barh(y + h / 2, rs, h, color=C["struct"], label="structured (true-graph circuit)")
    ax.barh(y - h / 2, rc, h, color=C["scram"], label="scrambled (random-graph circuit)")
    ax.axvline(0, color="#444", lw=1)
    for yi, (a, b) in enumerate(zip(rs, rc)):
        ax.annotate(f"{a:+.3f}", (max(a, 0), yi + h / 2), textcoords="offset points",
                    xytext=(4, 0), va="center", fontsize=7.5, color=C["struct"])
        ax.annotate(f"{b:+.3f}", (max(b, 0), yi - h / 2), textcoords="offset points",
                    xytext=(4, 0), va="center", fontsize=7.5, color=C["scram"])
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Linear-probe R² (5-fold CV) predicting the property from frozen quantum features")
    ax.set_title("What do the quantum features encode?\n"
                 "Structured features carry MORE true-graph topology; node features tie")
    ax.legend(frameon=False, loc="center right", fontsize=9)
    fig.savefig("docs/figures/fig17_probe.png")
    print("wrote docs/figures/fig17_probe.png", flush=True)


if __name__ == "__main__":
    main()
