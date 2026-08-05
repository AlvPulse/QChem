"""Mechanistic evidence for the 'place-then-harvest' inductive bias (Level 8).

Claim to substantiate, directly (not via a p-value):
  PLACE   - the graph-gated IsingXX(A[i,j]*theta) entangler concentrates two-qubit quantum
            correlation on the molecule's TRUE bonds.
  HARVEST - bond-pooling with the TRUE adjacency therefore captures more of that correlation
            mass than pooling with a random adjacency (the scrambled control).

We train a representative Level-8 (K=6) circuit briefly on one scaffold fold, then on the held-out
fold measure, per molecule and per qubit pair (i,j):
  connected correlator   C_ij = <Z_i Z_j> - <Z_i><Z_j>
and classify each pair as bonded (A_true[i,j] > 0) vs non-bonded. Outputs:
  * mean |C| on bonded vs non-bonded pairs  (PLACE)
  * on-bond correlation-mass fraction vs the uniform baseline #bonds/#pairs  (PLACE, normalized)
  * harvested mass  sum_ij A[i,j]|C_ij|  with A = true vs random  (HARVEST)
Saves results/mechanism_K6.npz and docs/figures/fig14_mechanism.png.
"""
import os, numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_bias_probe import (featurize, scaffold_folds, standardize, masked_bce,
                            pos_weight, pairs_of, N_TASKS)
from run_levelG_probe import GraphG, CONFIGS

K = 6
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight",
                     "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.family": "DejaVu Sans"})
C = {"struct": "#C44E52", "scram": "#4C72B0", "base": "#8C8C8C"}


def raw_correlators(model, QFt, At):
    """Return single-qubit <Z> (B,K) and pair <Z_iZ_j> (B,P) from the Level-8 circuit."""
    with torch.no_grad():
        a = torch.atan(model.feat(QFt))
        out = model.circ(a[:, :, 0], a[:, :, 1], At, model.theta, model.ringp, model.pairp, model.enc)
        out = [o.float() for o in out]
    k, P = model.k, model.P
    z = torch.stack(out[2 * k:3 * k], -1).numpy()          # <Z_i>  (B,K)
    zz = torch.stack(out[3 * k:3 * k + P], -1).numpy()     # <Z_iZ_j> (B,P)
    return z, zz


def main():
    QF0, AT, AR, Y, SCAF = featurize(K, ["Tox21", "ToxCast"])
    tr, va, te = next(scaffold_folds(SCAF, 3))
    QF = standardize(QF0, tr)
    QFt, ATt, ARt, Yt = (torch.tensor(QF), torch.tensor(AT), torch.tensor(AR), torch.tensor(Y))
    PAIRS = pairs_of(K)
    pi = np.array([i for i, j in PAIRS]); pj = np.array([j for i, j in PAIRS])

    # Train a representative structured Level-8 model briefly (theta must be meaningful).
    torch.manual_seed(0)
    model = GraphG(K, **CONFIGS["levelG"])
    pw = pos_weight(Y, tr)
    qkeys = ("theta", "ringp", "pairp", "enc")
    opt = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if any(q in n for q in qkeys)], "lr": 1e-2},
        {"params": [p for n, p in model.named_parameters() if not any(q in n for q in qkeys)], "lr": 1e-3},
    ], weight_decay=1e-4)
    tr_t = torch.as_tensor(tr)
    for ep in range(12):
        model.train(); o = torch.randperm(len(tr))
        for s in range(0, len(tr), 128):
            bi = tr_t[o[s:s + 128]]
            loss = masked_bce(model(QFt[bi], ATt[bi]), Yt[bi], pw)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"  ep{ep+1} loss {loss.item():.4f}", flush=True)
    model.eval()

    te_t = torch.as_tensor(te)
    z, zz = raw_correlators(model, QFt[te_t], ATt[te_t])     # measured on TRUE-A circuit
    # connected correlator C_ij = <Z_iZ_j> - <Z_i><Z_j>
    Cconn = np.abs(zz - z[:, pi] * z[:, pj])                 # (B,P)
    A_te = AT[te]                                            # (B,K,K)
    A_pair = A_te[:, pi, pj]                                 # (B,P) true bond weight per pair
    AR_pair = AR[te][:, pi, pj]
    bonded = A_pair > 0                                      # (B,P) boolean

    # PLACE: mean |C| on bonded vs non-bonded pairs (per molecule, then averaged)
    on = np.array([Cconn[m, bonded[m]].mean() if bonded[m].any() else np.nan for m in range(len(te))])
    off = np.array([Cconn[m, ~bonded[m]].mean() if (~bonded[m]).any() else np.nan for m in range(len(te))])
    on_m, off_m = np.nanmean(on), np.nanmean(off)

    # PLACE normalized: on-bond mass fraction vs uniform baseline (#bonds/#pairs)
    mass_total = Cconn.sum(1)
    mass_on = (Cconn * bonded).sum(1)
    onfrac = np.nanmean(mass_on / (mass_total + 1e-9))
    base = np.nanmean(bonded.sum(1) / Cconn.shape[1])

    # HARVEST: normalized harvested mass with TRUE vs RANDOM adjacency pooling
    def harvest(Aw):
        num = (Aw * Cconn).sum(1)
        den = (Aw.sum(1) * (Cconn.sum(1) / Cconn.shape[1]) + 1e-9)  # vs uniform expectation
        return np.nanmean(num / den)
    h_true = harvest(A_pair); h_rand = harvest(AR_pair)

    os.makedirs("results", exist_ok=True)
    np.savez("results/mechanism_K6.npz", on=on, off=off, on_m=on_m, off_m=off_m,
             onfrac=onfrac, base=base, h_true=h_true, h_rand=h_rand)
    print(f"PLACE   |C| bonded {on_m:.4f} vs non-bonded {off_m:.4f}  (ratio {on_m/off_m:.2f}x)", flush=True)
    print(f"PLACE   on-bond mass frac {onfrac:.3f} vs uniform baseline {base:.3f}", flush=True)
    print(f"HARVEST true-A {h_true:.3f} vs random-A {h_rand:.3f} (x uniform)", flush=True)

    make_figure(on, off, on_m, off_m, onfrac, base, h_true, h_rand)


def make_figure(on, off, on_m, off_m, onfrac, base, h_true, h_rand):
    os.makedirs("docs/figures", exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.3))
    # P1: |C| bonded vs non-bonded (box)
    d_on = on[~np.isnan(on)]; d_off = off[~np.isnan(off)]
    axs[0].boxplot([d_on, d_off], tick_labels=["bonded\npairs", "non-bonded\npairs"],
                   showfliers=False, patch_artist=True, boxprops=dict(facecolor="#EAEAF2"))
    axs[0].scatter(np.full(len(d_on), 1), d_on, s=4, alpha=0.1, color=C["struct"])
    axs[0].scatter(np.full(len(d_off), 2), d_off, s=4, alpha=0.1, color=C["base"])
    axs[0].set_ylabel("|connected correlator|  |⟨Z_iZ_j⟩−⟨Z_i⟩⟨Z_j⟩|")
    axs[0].set_title(f"PLACE: correlation concentrates on bonds\n(bonded {on_m:.3f} vs {off_m:.3f}, "
                     f"{on_m/off_m:.1f}× )")
    # P2: on-bond mass fraction vs uniform
    axs[1].bar(["measured\non-bond frac", "uniform\nbaseline"], [onfrac, base],
               color=[C["struct"], C["base"]], edgecolor="black", linewidth=0.6)
    axs[1].set_ylim(0, max(onfrac, base) * 1.3)
    axs[1].set_ylabel("fraction of correlation mass on true bonds")
    axs[1].set_title("PLACE (normalized): above chance")
    for i, v in enumerate([onfrac, base]):
        axs[1].annotate(f"{v:.2f}", (i, v), textcoords="offset points", xytext=(0, 4), ha="center")
    # P3: harvested mass true vs random
    axs[2].bar(["true A\n(structured)", "random A\n(scrambled)"], [h_true, h_rand],
               color=[C["struct"], C["scram"]], edgecolor="black", linewidth=0.6)
    axs[2].axhline(1.0, color=C["base"], ls="--", lw=1)
    axs[2].set_ylabel("harvested correlation mass (× uniform)")
    axs[2].set_title(f"HARVEST: true-A pooling wins\n({h_true:.2f}× vs {h_rand:.2f}×)")
    for i, v in enumerate([h_true, h_rand]):
        axs[2].annotate(f"{v:.2f}×", (i, v), textcoords="offset points", xytext=(0, 4), ha="center")
    fig.suptitle("Mechanism: graph-gated entanglement PLACES correlation on bonds; "
                 "bond-pooling HARVESTS it (Level 8, K=6)", y=1.03, fontsize=11.5)
    fig.savefig("docs/figures/fig14_mechanism.png")
    print("wrote docs/figures/fig14_mechanism.png", flush=True)


if __name__ == "__main__":
    # Fast path: if the (slow) measurement pass already ran, just (re)draw from the cache.
    if os.path.exists("results/mechanism_K6.npz"):
        d = np.load("results/mechanism_K6.npz")
        make_figure(d["on"], d["off"], float(d["on_m"]), float(d["off_m"]),
                    float(d["onfrac"]), float(d["base"]), float(d["h_true"]), float(d["h_rand"]))
    else:
        main()
