"""Learning-curve figure: does the inductive-bias gap widen as training data gets scarce?

Inductive bias is fundamentally about sample efficiency, so a chemistry-shaped prior should help
MOST when data is limited. We plot the structured-minus-scrambled gap (ΔAUC) against training-set
size for the gate-gated and Level-8 (measurement) circuits. A gap that grows toward small N is the
on-thesis signature of a real inductive bias.

Reads results/lc_K4.json (written by run_levelG_probe.py --train_fracs ...); writes
docs/figures/fig12_learning_curve.png.
"""
import os, json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = sys.argv[1] if len(sys.argv) > 1 else "results/lc_K4.json"
OUT = "docs/figures"
os.makedirs(OUT, exist_ok=True)
C = {"gate": "#4C72B0", "levelG": "#C44E52"}
LBL = {"gate": "Gate-gated", "levelG": "Level 8 (measurement)"}
SIG = lambda p: (p == p) and p < 0.05

rows = json.load(open(SRC))
K = rows[0]["k"]
configs = ["gate", "levelG"]

fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))

# ---- Panel A: bias (ΔAUC) vs training size -------------------------------
for cfg in configs:
    rs = sorted([r for r in rows if r["name"] == cfg], key=lambda r: r["n_train"])
    xs = [r["n_train"] for r in rs]
    ys = [r["median"] for r in rs]
    ps = [r["wil_p"] for r in rs]
    ax[0].plot(xs, ys, "-", color=C[cfg], lw=2.2, label=LBL[cfg], zorder=2)
    for x, y, p in zip(xs, ys, ps):
        ax[0].scatter([x], [y], s=130 if SIG(p) else 80,
                      facecolor=C[cfg] if SIG(p) else "white",
                      edgecolor=C[cfg], linewidth=2, zorder=3)
        ax[0].annotate(("p=%.3f" % p) + (" *" if SIG(p) else ""), (x, y),
                       textcoords="offset points", xytext=(0, 10 if cfg == "levelG" else -15),
                       ha="center", fontsize=8, color=C[cfg])
ax[0].axhline(0, color="#444", ls="--", lw=1, alpha=0.7)
ax[0].set_xscale("log")
ax[0].set_xlabel("Training molecules per fold (log scale)")
ax[0].set_ylabel("Topology bias  ΔAUC  (structured − scrambled)")
ax[0].set_title(f"Inductive-bias gap vs training-set size (K={K})")
ax[0].legend(frameon=False, loc="best")
ax[0].grid(alpha=0.25)

# ---- Panel B: absolute structured ROC vs size (shows the model is learning) ----
for cfg in configs:
    rs = sorted([r for r in rows if r["name"] == cfg], key=lambda r: r["n_train"])
    xs = [r["n_train"] for r in rs]
    ax[1].plot(xs, [r["struct"] for r in rs], "-o", color=C[cfg], lw=2, label=f"{LBL[cfg]} structured")
    ax[1].plot(xs, [r["scram"] for r in rs], "--", color=C[cfg], lw=1.4, alpha=0.6,
               label=f"{LBL[cfg]} scrambled")
ax[1].axhline(0.5, color="#444", ls=":", lw=1, alpha=0.6)
ax[1].set_xscale("log")
ax[1].set_xlabel("Training molecules per fold (log scale)")
ax[1].set_ylabel("Pooled-CV ROC-AUC (12 Tox21 tasks)")
ax[1].set_title("Absolute performance vs training-set size")
ax[1].legend(frameon=False, fontsize=8, loc="best")
ax[1].grid(alpha=0.25)

plt.tight_layout()
p = os.path.join(OUT, "fig12_learning_curve.png")
fig.savefig(p, dpi=200, bbox_inches="tight")
print("wrote", p)

# quick textual summary
print("\nlevelG - gate bias gap by training size:")
for r in sorted([r for r in rows if r["name"] == "levelG"], key=lambda r: r["n_train"]):
    g = next(x for x in rows if x["name"] == "gate" and x["n_train"] == r["n_train"])
    print(f"  n_train~{r['n_train']:>5}: levelG dAUC {r['median']:+.4f} (p={r['wil_p']:.3f})  "
          f"gate {g['median']:+.4f} (p={g['wil_p']:.3f})  diff {r['median']-g['median']:+.4f}")
