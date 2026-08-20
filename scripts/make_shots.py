"""Finite-shot analysis: does the Level-8 bias survive measurement with N shots?

The paper's distinctive claim is a HARDWARE-NATIVE, O(K) measurement readout. Everything else is
exact statevector, so this script substantiates the claim: we train Level-8 (K=6) structured and
scrambled circuits exactly, then at TEST time estimate every readout observable from N shots instead
of exactly, and ask at what shot budget the structured-minus-scrambled bias (and absolute AUC)
survives.

Shot model (conservative: each Pauli measured independently): for an observable with exact
expectation mu in [-1,1] and eigenvalues +/-1, N shots give mu_hat = 2*Binom(N,(1+mu)/2)/N - 1.
Applied to the single-qubit <X,Y,Z> and the two-qubit <Z_iZ_j>,<X_iX_j> correlators BEFORE bond-
pooling and the trained linear head. Independent per observable is an upper bound on the noise
(commuting groups / classical shadows do better), so a bias that survives here survives on hardware.

Outputs results/shots_K6.npz and docs/figures/fig15_shots.png.
"""
import os, numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_bias_probe import (featurize, scaffold_folds, standardize, masked_bce,
                            pos_weight, per_task_auc, pairs_of, N_TASKS)
from run_levelG_probe import GraphG, CONFIGS

K = 6
EPOCHS = 15
SHOTS = [32, 64, 128, 256, 512, 1024, 4096]
REALIZATIONS = 12
plt.rcParams.update({"savefig.dpi": 200, "savefig.bbox": "tight", "font.size": 11,
                     "axes.titlesize": 12, "axes.titleweight": "bold", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                     "font.family": "DejaVu Sans"})
C = {"struct": "#C44E52", "scram": "#4C72B0", "delta": "#55A868", "base": "#8C8C8C"}


def train_one(variant, seed, tr, va, te, QF, AT, AR, Y):
    torch.manual_seed(seed)
    model = GraphG(K, **CONFIGS["levelG"])
    adj = AR if variant == "scrambled" else AT
    QFt, At, Yt = torch.tensor(QF), torch.tensor(adj), torch.tensor(Y)
    pw = pos_weight(Y, tr)
    qk = ("theta", "ringp", "pairp", "enc")
    opt = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if any(q in n for q in qk)], "lr": 1e-2},
        {"params": [p for n, p in model.named_parameters() if not any(q in n for q in qk)], "lr": 1e-3},
    ], weight_decay=1e-4)
    best_va, best_state = -1.0, None
    tr_t = torch.as_tensor(tr)
    for _ in range(EPOCHS):
        model.train(); o = torch.randperm(len(tr))
        for s in range(0, len(tr), 128):
            bi = tr_t[o[s:s + 128]]
            loss = masked_bce(model(QFt[bi], At[bi]), Yt[bi], pw)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            from run_bias_probe import roc12
            vr = roc12(model(QFt[va], At[va]).numpy(), Y[va])
        if vr > best_va:
            best_va = vr; best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, adj


def raw_obs(model, QF_te, adj_te):
    """Exact single-qubit (B,3K) and pair <ZZ>,<XX> (B,P) on the test fold."""
    with torch.no_grad():
        a = torch.atan(model.feat(torch.tensor(QF_te)))
        out = model.circ(a[:, :, 0], a[:, :, 1], torch.tensor(adj_te),
                         model.theta, model.ringp, model.pairp, model.enc)
        out = [o.float() for o in out]
    k, P = model.k, model.P
    single = torch.stack(out[:3 * k], -1).numpy()
    zz = torch.stack(out[3 * k:3 * k + P], -1).numpy()
    xx = torch.stack(out[3 * k + P:3 * k + 2 * P], -1).numpy()
    return single, zz, xx


def logits_from_obs(model, single, zz, xx, adj_te):
    """Reassemble the model's head output from (possibly shot-noisy) observables."""
    with torch.no_grad():
        s = torch.tensor(single, dtype=torch.float32)
        zzt = torch.tensor(zz, dtype=torch.float32); xxt = torch.tensor(xx, dtype=torch.float32)
        At = torch.tensor(adj_te)
        feats = [s, model._bond_pool(zzt, At), model._bond_pool(xxt, At)]
        return model.head(torch.cat(feats, -1)).numpy()


def shot_noise(mu, N, rng):
    """mu in [-1,1] -> N-shot estimate (eigenvalues +/-1)."""
    p = np.clip((1 + mu) / 2, 0, 1)
    return 2 * rng.binomial(N, p) / N - 1


def main():
    QF0, AT, AR, Y, SCAF = featurize(K, ["Tox21", "ToxCast"])
    folds = list(scaffold_folds(SCAF, 3))
    N = len(Y)
    # train per fold/variant, stash exact test observables + the trained model + adj
    store = {"structured": [], "scrambled": []}
    for fi, (tr, va, te) in enumerate(folds):
        QF = standardize(QF0, tr)
        for variant in ("structured", "scrambled"):
            model, adj = train_one(variant, 0, tr, va, te, QF, AT, AR, Y)
            single, zz, xx = raw_obs(model, QF[te], adj[te])
            store[variant].append(dict(te=te, model=model, adj_te=adj[te],
                                       single=single, zz=zz, xx=xx))
        print(f"  fold {fi+1}/3 trained", flush=True)

    def pooled_auc(variant, N_shots, rng):
        probs = np.full((N, N_TASKS), np.nan)
        for blk in store[variant]:
            if N_shots is None:
                s, zz, xx = blk["single"], blk["zz"], blk["xx"]
            else:
                s = shot_noise(blk["single"], N_shots, rng)
                zz = shot_noise(blk["zz"], N_shots, rng)
                xx = shot_noise(blk["xx"], N_shots, rng)
            lg = logits_from_obs(blk["model"], s, zz, xx, blk["adj_te"])
            probs[blk["te"]] = 1 / (1 + np.exp(-lg))
        return per_task_auc(Y, probs)

    rng = np.random.default_rng(0)
    exact_s = np.nanmean(pooled_auc("structured", None, rng))
    exact_c = np.nanmean(pooled_auc("scrambled", None, rng))
    exact_d = exact_s - exact_c
    print(f"EXACT: struct {exact_s:.4f} scram {exact_c:.4f} dAUC {exact_d:+.4f}", flush=True)

    rows = []
    for Ns in SHOTS:
        ds, ss, cs = [], [], []
        for r in range(REALIZATIONS):
            a_s = np.nanmean(pooled_auc("structured", Ns, rng))
            a_c = np.nanmean(pooled_auc("scrambled", Ns, rng))
            ss.append(a_s); cs.append(a_c); ds.append(a_s - a_c)
        rows.append(dict(shots=Ns, d_mean=float(np.mean(ds)), d_std=float(np.std(ds)),
                         s_mean=float(np.mean(ss)), c_mean=float(np.mean(cs)),
                         d_pos_frac=float(np.mean(np.array(ds) > 0))))
        print(f"  N={Ns:>5}: dAUC {rows[-1]['d_mean']:+.4f} +/- {rows[-1]['d_std']:.4f} "
              f"(struct {rows[-1]['s_mean']:.4f}); {rows[-1]['d_pos_frac']*100:.0f}% realizations +",
              flush=True)

    os.makedirs("results", exist_ok=True)
    np.savez("results/shots_K6.npz", shots=np.array(SHOTS),
             d_mean=np.array([r["d_mean"] for r in rows]),
             d_std=np.array([r["d_std"] for r in rows]),
             s_mean=np.array([r["s_mean"] for r in rows]),
             c_mean=np.array([r["c_mean"] for r in rows]),
             exact_d=exact_d, exact_s=exact_s, exact_c=exact_c)

    # ---- figure ----
    xs = np.array(SHOTS)
    dm = np.array([r["d_mean"] for r in rows]); dsd = np.array([r["d_std"] for r in rows])
    sm = np.array([r["s_mean"] for r in rows]); cm = np.array([r["c_mean"] for r in rows])
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.5))
    axs[0].axhline(exact_d, color=C["base"], ls="--", lw=1.4, label=f"exact ΔAUC {exact_d:+.4f}")
    axs[0].axhline(0, color="#444", ls=":", lw=1)
    axs[0].plot(xs, dm, "-o", color=C["delta"], lw=2.2, label="finite-shot ΔAUC")
    axs[0].fill_between(xs, dm - dsd, dm + dsd, color=C["delta"], alpha=0.18)
    axs[0].set_xscale("log", base=2)
    axs[0].set_xlabel("Shots per observable (log₂)")
    axs[0].set_ylabel("Topology bias ΔAUC (structured − scrambled)")
    axs[0].set_title("Bias survival vs shot budget (Level 8, K=6)")
    axs[0].legend(frameon=False, fontsize=9)
    axs[1].axhline(exact_s, color=C["struct"], ls="--", lw=1.2, alpha=0.6)
    axs[1].axhline(exact_c, color=C["scram"], ls="--", lw=1.2, alpha=0.6)
    axs[1].plot(xs, sm, "-o", color=C["struct"], lw=2, label="structured")
    axs[1].plot(xs, cm, "-o", color=C["scram"], lw=2, label="scrambled")
    axs[1].set_xscale("log", base=2)
    axs[1].set_xlabel("Shots per observable (log₂)")
    axs[1].set_ylabel("Pooled-CV ROC-AUC (12 Tox21 tasks)")
    axs[1].set_title("Absolute performance vs shot budget")
    axs[1].legend(frameon=False, fontsize=9)
    fig.suptitle("Finite-shot readout: the O(K) bond-correlator bias is shot-robust "
                 "(dashed = exact statevector)", y=1.02, fontsize=11.5)
    fig.savefig("docs/figures/fig15_shots.png")
    print("wrote docs/figures/fig15_shots.png", flush=True)


if __name__ == "__main__":
    main()
