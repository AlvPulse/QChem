"""Quantum-vs-classical readout baseline: does the quantum two-qubit correlator carry topology
signal a classical edge feature does not?

Compares the structured-minus-scrambled bias (ΔAUC) for three models that share the SAME coarse
graph, the SAME true-vs-random adjacency control, and a permutation-invariant graph readout:
  gate          quantum, single-qubit readout
  levelG        quantum, bond-correlator readout  (Level 8)
  classicalGNN  classical message-passing, A-weighted bond-pooled pairwise products

If levelG's gap >> classicalGNN's gap -> the quantum correlator adds signal beyond a classical
edge feature (the headline). If they match -> the benefit is the graph-pooling architecture, not
quantumness (reframe).

Reads results/baseline_K{4,6}.json; writes docs/figures/fig13_quantum_vs_classical.png.
"""
import os, json, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "docs/figures"
os.makedirs(OUT, exist_ok=True)
ORDER = ["gate", "levelG", "classicalGNN"]
LBL = {"gate": "Quantum\ngate-only", "levelG": "Quantum\nLevel 8", "classicalGNN": "Classical\nGNN readout"}
COL = {"gate": "#4C72B0", "levelG": "#C44E52", "classicalGNN": "#55A868"}
SIG = lambda p: (p == p) and p < 0.05

srcs = sys.argv[1:] or sorted(glob.glob("results/baseline_K*.json"))
rows = []
for s in srcs:
    rows += json.load(open(s))
if not rows:
    print("no baseline json found"); sys.exit(0)
Ks = sorted({r["k"] for r in rows})

fig, axes = plt.subplots(1, len(Ks), figsize=(5.6 * len(Ks), 4.6), squeeze=False)
for ax, K in zip(axes[0], Ks):
    present = [c for c in ORDER if any(r["name"] == c and r["k"] == K for r in rows)]
    vals, ps, cols = [], [], []
    for c in present:
        r = next(x for x in rows if x["name"] == c and x["k"] == K)
        vals.append(r["median"]); ps.append(r["wil_p"]); cols.append(COL[c])
    xs = np.arange(len(present))
    ax.bar(xs, vals, width=0.6, color=cols, edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="#444", lw=1)
    for x, v, p in zip(xs, vals, ps):
        star = " *" if SIG(p) else ""
        ax.annotate(f"{v:+.4f}\np={p:.3g}{star}", (x, v),
                    textcoords="offset points", xytext=(0, 8 if v >= 0 else -22),
                    ha="center", fontsize=9, fontweight="bold" if SIG(p) else "normal")
    ax.set_xticks(xs); ax.set_xticklabels([LBL[c] for c in present], fontsize=9)
    ax.set_ylabel("Topology bias  ΔAUC  (structured − scrambled)")
    ax.set_title(f"K = {K}")
    ax.grid(axis="y", alpha=0.25)
fig.suptitle("Does the quantum correlator beat a classical edge feature? (same graph, same control)",
             fontweight="bold")
plt.tight_layout()
p = os.path.join(OUT, "fig13_quantum_vs_classical.png")
fig.savefig(p, dpi=200, bbox_inches="tight")
print("wrote", p)

print("\n--- verdict ---")
for K in Ks:
    def g(c):
        r = [x for x in rows if x["name"] == c and x["k"] == K]
        return r[0] if r else None
    lg, cl = g("levelG"), g("classicalGNN")
    if lg and cl:
        d = lg["median"] - cl["median"]
        verdict = ("quantum > classical (headline)" if d > 0.002 else
                   "quantum ~= classical (reframe)" if abs(d) <= 0.002 else
                   "classical > quantum (reframe hard)")
        print(f"K={K}: levelG dAUC {lg['median']:+.4f} (p={lg['wil_p']:.3f}) vs "
              f"classicalGNN {cl['median']:+.4f} (p={cl['wil_p']:.3f})  ->  {verdict}")
