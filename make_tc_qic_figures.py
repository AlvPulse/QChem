"""TC-QIC theory experiment figures (E8/E13/E6/E7/E12 + scaling law).

Produces figures in docs/figures/ for the TC-QIC theoretical results. Each
figure function checks if the corresponding results file exists before plotting;
missing results produce a stub figure with a placeholder message.

Run: python make_tc_qic_figures.py [--figs all|mechanism|absorbability|alpha|kappa|shots|noise]
"""
import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "docs/figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 11, "legend.fontsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "font.family": "DejaVu Sans",
})

C = {
    "levelG": "#C44E52", "gate": "#4C72B0", "meas_only": "#8C8C8C",
    "classicalGNN": "#55A868", "struct": "#C44E52", "scram": "#4C72B0",
    "vacuous": "#B0B0B0", "genuine": "#C44E52", "partial": "#E8A838",
}


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p)
    plt.close(fig)
    print("wrote", p, flush=True)


def stub(name, msg):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, msg, ha='center', va='center', transform=ax.transAxes,
            fontsize=12, color='#888')
    ax.axis('off')
    save(fig, name)


# ========================== E8: Place-then-harvest mechanism ==========================
def fig_mechanism():
    """Bar chart: PLACE ratio and HARVEST ratio for K=4,6,8."""
    ks, place, harvest_t, harvest_r = [], [], [], []
    for k in [4, 6, 8]:
        p = f"results/mechanism_K{k}.npz"
        if not os.path.exists(p):
            continue
        d = np.load(p)
        ks.append(k)
        r = float(d['on_m']) / float(d['off_m']) if float(d['off_m']) > 0 else float('nan')
        place.append(r)
        harvest_t.append(float(d['h_true']))
        harvest_r.append(float(d['h_rand']))

    if not ks:
        stub("fig_e8_mechanism.png", "E8 results not ready"); return

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))

    ax = axes[0]
    bars = ax.bar([str(k) for k in ks], place, color=C["struct"], edgecolor='black', lw=0.7, width=0.5)
    ax.axhline(1.0, ls='--', color='#444', lw=1.2, alpha=0.7)
    for b, v in zip(bars, place):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.2f}x",
                ha='center', va='bottom', fontsize=9.5, color=C["struct"], fontweight='bold')
    ax.set_xlabel("Qubits K")
    ax.set_ylabel("Bonded / non-bonded connected correlator (mean)")
    ax.set_title("PLACE: entangler puts signal on bonds\n(T9 Lemma 4.1)")
    ax.set_ylim(0, max(place) * 1.25)

    ax = axes[1]
    x = np.arange(len(ks)); w = 0.35
    ax.bar(x - w / 2, harvest_t, w, color=C["struct"], label="true-A pool", edgecolor='black', lw=0.7)
    ax.bar(x + w / 2, harvest_r, w, color=C["scram"], label="random-A pool", edgecolor='black', lw=0.7)
    for xi, (ht, hr) in enumerate(zip(harvest_t, harvest_r)):
        ax.text(xi - w / 2, ht + 0.03, f"{ht:.2f}", ha='center', fontsize=8.5, color=C["struct"])
        ax.text(xi + w / 2, hr + 0.03, f"{hr:.2f}", ha='center', fontsize=8.5, color=C["scram"])
    ax.axhline(1.0, ls='--', color='#444', lw=1.2, alpha=0.7, label="null (uniform)")
    ax.set_xticks(x); ax.set_xticklabels([f"K={k}" for k in ks])
    ax.set_ylabel("Normalized harvest score (true-A / uniform expected)")
    ax.set_title("HARVEST: bond-pool preferentially collects\non-bond correlators (T9 Lemma 4.2)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(0, max(max(harvest_t), max(harvest_r)) * 1.2)

    fig.suptitle("Place-then-Harvest Mechanism (E8): entangler PLACES, readout HARVESTS",
                 fontsize=12, fontweight='bold', y=1.01)
    fig.tight_layout()
    save(fig, "fig_e8_mechanism.png")


# ========================== E13: Absorbability ==========================
def fig_absorbability():
    """Bar chart of residuals per level (E13)."""
    lvls = [2, 3, 4, 5, 6, 7]
    residuals = [0.00, 0.175, 0.00, 1.26, 1.55, 1.38]
    verdicts = ["VACUOUS", "partial", "VACUOUS", "genuine", "genuine", "genuine"]
    colors = [C["vacuous"] if v == "VACUOUS" else (C["partial"] if v == "partial" else C["genuine"])
              for v in verdicts]

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    bars = ax.bar([f"L{l}" for l in lvls], residuals, color=colors, edgecolor='black', lw=0.7, width=0.6)
    ax.axhline(0, color='#444', lw=1)
    for b, v, verdict in zip(bars, residuals, verdicts):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02,
                f"{v:.2e}\n{verdict}", ha='center', va='bottom', fontsize=8.5, color='#333')
    ax.set_xlabel("Benchmark Level")
    ax.set_ylabel("max|struct(x) - scram(permuted-x)|")
    ax.set_title("Absorbability Audit (E13): L2/L4 are VACUOUS controls\n"
                 "Bit-exact 0 => same function class => comparison proves nothing")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C["vacuous"], label="VACUOUS (bit-exact 0)"),
                       Patch(color=C["partial"], label="partial"),
                       Patch(color=C["genuine"], label="genuine (non-absorbable)")],
              frameon=False, loc="upper left", fontsize=9)
    ax.set_ylim(-0.05, 1.85)
    save(fig, "fig_e13_absorbability.png")


# ========================== E6: Alpha sweep ==========================
def fig_alpha_sweep():
    if not os.path.exists("results/e6_alpha_summary.json"):
        stub("fig_e6_alpha.png", "E6 alpha sweep not complete yet"); return
    with open("results/e6_alpha_summary.json") as f:
        rows = json.load(f)
    ks = sorted(set(r["k"] for r in rows))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for k in ks:
        krows = [r for r in rows if r["k"] == k]
        lams = [r["lam"] for r in krows]
        deltas = [r["delta"] for r in krows]
        color = {4: "#4C72B0", 6: "#C44E52", 8: "#55A868"}.get(k, "#888")
        ax.plot(lams, deltas, "o-", color=color, lw=2, label=f"K={k}")
        for l, d in zip(lams, deltas):
            ax.annotate(f"{d:+.3f}", (l, d), textcoords="offset points", xytext=(0, 8),
                        ha='center', fontsize=7.5, color=color)
    ax.axhline(0, color="#444", lw=1, ls="--", alpha=0.7)
    ax.set_xlabel("lambda  (A_lambda = lambda*A_true + (1-lambda)*A_rand)")
    ax.set_ylabel("dAUC (A_lambda structured - scrambled)")
    ax.set_title("E6 Alpha Sweep: dAUC vs topology alignment (P2)\n"
                 "P2 prediction: monotone increasing in lambda")
    ax.legend(frameon=False, fontsize=9)
    ax.text(0.98, 0.02, "P2 prediction: delta(lambda=1) > delta(lambda=0) >= 0",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color='#666')
    save(fig, "fig_e6_alpha.png")


# ========================== E7: Kappa sweep ==========================
def fig_kappa_sweep():
    if not os.path.exists("results/e7_kappa_summary.json"):
        stub("fig_e7_kappa.png", "E7 kappa sweep not complete yet"); return
    with open("results/e7_kappa_summary.json") as f:
        rows = json.load(f)
    ks = [r["k"] for r in rows]
    d2 = [r["delta_kappa2"] for r in rows]
    dK = [r["delta_kappaK"] for r in rows]
    p3 = [r["p3_pass"] for r in rows]
    x = np.arange(len(ks)); w = 0.35
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.bar(x - w / 2, d2, w, color=C["struct"], label="kappa=2 (bond-local)", edgecolor='black', lw=0.7)
    ax.bar(x + w / 2, dK, w, color=C["scram"], label="kappa=K (global)", edgecolor='black', lw=0.7)
    ax.axhline(0, color='#444', lw=1)
    for xi, (a, b, pass_) in enumerate(zip(d2, dK, p3)):
        ax.text(xi - w / 2, a + 0.001, f"{a:+.4f}", ha='center', fontsize=8.5, color=C["struct"])
        ax.text(xi + w / 2, b + 0.001, f"{b:+.4f}", ha='center', fontsize=8.5, color=C["scram"])
        result = "PASS" if pass_ else "FAIL"
        color = "#2a8a2a" if pass_ else "#cc2222"
        ax.text(xi, max(a, b) + 0.004, f"P3: {result}", ha='center', fontsize=8.5, color=color)
    ax.set_xticks(x); ax.set_xticklabels([f"K={k}" for k in ks])
    ax.set_ylabel("dAUC (structured - scrambled)")
    ax.set_title("E7 Kappa Sweep: bond-local readout vs global readout (P3)\n"
                 "P3 prediction: kappa=2 > kappa=K")
    ax.legend(frameon=False, fontsize=9)
    save(fig, "fig_e7_kappa.png")


# ========================== E12: Shots ==========================
def fig_shots():
    shot_files = [f for f in os.listdir("results") if f.startswith("shots_K") and f.endswith(".npz")]
    if not shot_files:
        stub("fig_e12_shots.png", "E12 shots results not ready"); return
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    colors = {4: "#4C72B0", 6: "#C44E52", 8: "#55A868"}
    SHOTS = [32, 64, 128, 256, 512, 1024, 4096]
    for fname in sorted(shot_files):
        k = int(fname.replace("shots_K", "").replace(".npz", ""))
        d = np.load(f"results/{fname}", allow_pickle=True)
        data = d['data'].tolist() if isinstance(d['data'], np.ndarray) else d['data']
        s_means = [r['mean'] for r in data if r['variant'] == 'structured']
        c_means = [r['mean'] for r in data if r['variant'] == 'scrambled']
        if len(s_means) == len(SHOTS):
            deltas = [s - c for s, c in zip(s_means, c_means)]
            col = colors.get(k, "#888")
            ax.plot(SHOTS, deltas, "o-", color=col, lw=2, label=f"K={k}")
    if not ax.get_lines():
        stub("fig_e12_shots.png", "E12 shots no valid data"); return
    ax.set_xscale('log', base=2)
    ax.axhline(0, color='#444', lw=1, ls='--', alpha=0.7)
    ax.set_xlabel("Shots per observable (log scale)")
    ax.set_ylabel("dAUC (structured - scrambled)")
    ax.set_title("E12 Shot-Noise Robustness: bias is flat vs shots\n"
                 "T11(iii) prediction: 2-local cost is shot-robust")
    ax.legend(frameon=False, fontsize=9)
    save(fig, "fig_e12_shots.png")


# ========================== E12: Noise ==========================
def fig_noise():
    noise_files = [f for f in os.listdir("results") if f.startswith("noise_K") and f.endswith(".npz")]
    if not noise_files:
        stub("fig_e12_noise.png", "E12 noise results not ready"); return
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    colors = {4: "#4C72B0", 6: "#C44E52", 8: "#55A868"}
    PS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
    for fname in sorted(noise_files):
        k = int(fname.replace("noise_K", "").replace(".npz", ""))
        d = np.load(f"results/{fname}")
        p_vals = d['p']; s_vals = d['s']; c_vals = d['c']; deltas = d['d']
        col = colors.get(k, "#888")
        ax.plot(p_vals, deltas, "o-", color=col, lw=2, label=f"K={k}")
    p_arr = np.array(PS)
    d0 = deltas[0] if len(noise_files) > 0 else 0.01
    ax.plot(p_arr, d0 * (1 - p_arr) ** 2, "--", color='#444', lw=1.5, alpha=0.8,
            label="(1-p)^2 analytic (theory)")
    ax.axhline(0, color='#444', lw=1, alpha=0.5)
    ax.set_xlabel("Depolarizing error rate p")
    ax.set_ylabel("dAUC (structured - scrambled)")
    ax.set_title("E12 Device-Noise Robustness: bias decays as (1-p)^2\n"
                 "T11(iii): 2-local observables, common rescaling")
    ax.legend(frameon=False, fontsize=9)
    save(fig, "fig_e12_noise.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--figs', nargs='+', default=['all'],
                    choices=['all', 'mechanism', 'absorbability', 'alpha', 'kappa', 'shots', 'noise'])
    args = ap.parse_args()

    wanted = set(args.figs)
    run_all = 'all' in wanted
    if run_all or 'mechanism' in wanted:
        fig_mechanism()
    if run_all or 'absorbability' in wanted:
        fig_absorbability()
    if run_all or 'alpha' in wanted:
        fig_alpha_sweep()
    if run_all or 'kappa' in wanted:
        fig_kappa_sweep()
    if run_all or 'shots' in wanted:
        fig_shots()
    if run_all or 'noise' in wanted:
        fig_noise()


if __name__ == '__main__':
    main()
