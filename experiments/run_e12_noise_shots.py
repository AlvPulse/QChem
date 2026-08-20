"""E12: Shot & noise robustness across K=4, 6, 8 -- locality signature.

Extends make_shots.py and make_noise.py (both K=6 only) to K=4, 6, 8.
Tests: does the structured-scrambled bias survive shots and depolarizing noise?
T11(iii) prediction: bias is robust because the cost is 2-local (bond-pooled readout).

Run: python run_e12_noise_shots.py --qubits 4 6 8 [--shots-only] [--noise-only]
Outputs: results/shots_K{k}.npz and results/noise_K{k}.npz for each K.
"""
import os, argparse, numpy as np, torch, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_bias_probe import (featurize, scaffold_folds, standardize, masked_bce,
                             pos_weight, per_task_auc, roc12, pairs_of, N_TASKS)
from run_levelG_probe import GraphG, CONFIGS

EPOCHS = 15
SHOTS = [32, 64, 128, 256, 512, 1024, 4096]
REALIZATIONS = 8
PS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
READOUT_E = 0.02


def train_one(k, variant, seed, tr, va, te, QF, AT, AR, Y):
    torch.manual_seed(seed)
    model = GraphG(k, **CONFIGS["levelG"])
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
            vr = roc12(model(QFt[va], At[va]).numpy(), Y[va])
        if vr > best_va:
            best_va = vr
            best_state = {kk: v.clone() for kk, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, adj


def raw_obs(model, QF_te, adj_te):
    k, P = model.k, model.P
    with torch.no_grad():
        a = torch.atan(model.feat(torch.tensor(QF_te)))
        out = model.circ(a[:, :, 0], a[:, :, 1], torch.tensor(adj_te),
                         model.theta, model.ringp, model.pairp, model.enc)
        out = [o.float() for o in out]
    single = torch.stack(out[:3 * k], -1).numpy()
    zz = torch.stack(out[3 * k:3 * k + P], -1).numpy()
    xx = torch.stack(out[3 * k + P:3 * k + 2 * P], -1).numpy()
    return single, zz, xx


def logits_from_obs(model, single, zz, xx, adj_te):
    with torch.no_grad():
        s = torch.tensor(single, dtype=torch.float32)
        zzt = torch.tensor(zz, dtype=torch.float32)
        xxt = torch.tensor(xx, dtype=torch.float32)
        At = torch.tensor(adj_te)
        return model.head(torch.cat([s, model._bond_pool(zzt, At), model._bond_pool(xxt, At)], -1)).numpy()


def run_shots_k(k, QF0, AT, AR, Y, SCAF):
    print(f"\n[K={k}] Shot-noise robustness", flush=True)
    folds = list(scaffold_folds(SCAF, 3))
    N = len(Y)
    store = {"structured": [], "scrambled": []}
    for fi, (tr, va, te) in enumerate(folds):
        QF = standardize(QF0, tr)
        for variant in ("structured", "scrambled"):
            model, adj = train_one(k, variant, 0, tr, va, te, QF, AT, AR, Y)
            single, zz, xx = raw_obs(model, QF[te], adj[te])
            store[variant].append(dict(te=te, model=model, adj_te=adj[te],
                                       single=single, zz=zz, xx=xx))
        print(f"  K={k} fold {fi+1}/3 trained", flush=True)

    rng = np.random.default_rng(0)

    def shot_noise(mu, N_shots):
        p = np.clip((1 + mu) / 2, 0, 1)
        return 2 * rng.binomial(N_shots, p) / N_shots - 1

    shot_rows = []
    for N_shots in SHOTS:
        for variant in ("structured", "scrambled"):
            aucs = []
            for _ in range(REALIZATIONS):
                probs = np.full((N, N_TASKS), np.nan)
                for blk in store[variant]:
                    s = shot_noise(blk["single"], N_shots)
                    zz = shot_noise(blk["zz"], N_shots)
                    xx = shot_noise(blk["xx"], N_shots)
                    lg = logits_from_obs(blk["model"], s, zz, xx, blk["adj_te"])
                    probs[blk["te"]] = 1 / (1 + np.exp(-lg))
                aucs.append(float(np.nanmean(per_task_auc(Y, probs))))
            shot_rows.append(dict(k=k, shots=N_shots, variant=variant,
                                  mean=float(np.mean(aucs)), std=float(np.std(aucs))))

    out = f"results/shots_K{k}.npz"
    np.savez(out, shots=SHOTS, data=shot_rows)
    print(f"  K={k} shots summary:")
    for v in ("structured", "scrambled"):
        row_s = [r for r in shot_rows if r['variant'] == v]
        row32 = next(r for r in row_s if r['shots'] == 32)
        row4k = next(r for r in row_s if r['shots'] == 4096)
        print(f"    {v}: N=32 AUC={row32['mean']:.4f}  N=4096 AUC={row4k['mean']:.4f}")
    return shot_rows


def run_noise_k(k, QF0, AT, AR, Y, SCAF):
    print(f"\n[K={k}] Device-noise robustness", flush=True)
    folds = list(scaffold_folds(SCAF, 3))
    N = len(Y)
    store = {"structured": [], "scrambled": []}
    for fi, (tr, va, te) in enumerate(folds):
        QF = standardize(QF0, tr)
        for variant in ("structured", "scrambled"):
            model, adj = train_one(k, variant, 0, tr, va, te, QF, AT, AR, Y)
            single, zz, xx = raw_obs(model, QF[te], adj[te])
            store[variant].append(dict(te=te, model=model, adj_te=adj[te],
                                       single=single, zz=zz, xx=xx))

    def attenuate(single, zz, xx, p, e=0.0):
        r = (1 - 2 * e); g = (1 - p)
        return single * g * r, zz * g * g * r, xx * g * g * r

    def pooled_auc(variant, p, e):
        probs = np.full((N, N_TASKS), np.nan)
        for blk in store[variant]:
            s, zz, xx = attenuate(blk["single"], blk["zz"], blk["xx"], p, e)
            lg = logits_from_obs(blk["model"], s, zz, xx, blk["adj_te"])
            probs[blk["te"]] = 1 / (1 + np.exp(-lg))
        return float(np.nanmean(per_task_auc(Y, probs)))

    noise_rows = []
    for p in PS:
        e = READOUT_E if p > 0 else 0.0
        s = pooled_auc("structured", p, e)
        c = pooled_auc("scrambled", p, e)
        noise_rows.append(dict(k=k, p=p, struct=s, scram=c, delta=s - c))
        print(f"  K={k} p={p:.2f}: struct {s:.4f} scram {c:.4f} dAUC {s-c:+.4f}  "
              f"theory (1-p)^2={((1-p)**2):.3f}x", flush=True)

    np.savez(f"results/noise_K{k}.npz",
             p=np.array(PS),
             s=np.array([r["struct"] for r in noise_rows]),
             c=np.array([r["scram"] for r in noise_rows]),
             d=np.array([r["delta"] for r in noise_rows]))
    return noise_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qubits', type=int, nargs='+', default=[4, 6, 8])
    ap.add_argument('--shots-only', action='store_true')
    ap.add_argument('--noise-only', action='store_true')
    ap.add_argument('--datasets', type=str, nargs='+', default=['Tox21', 'ToxCast'])
    args = ap.parse_args()

    all_results = {}
    for k in args.qubits:
        QF0, AT, AR, Y, SCAF = featurize(k, args.datasets)
        if not args.noise_only:
            all_results.setdefault('shots', []).extend(run_shots_k(k, QF0, AT, AR, Y, SCAF))
        if not args.shots_only:
            all_results.setdefault('noise', []).extend(run_noise_k(k, QF0, AT, AR, Y, SCAF))

    with open('results/e12_summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved -> results/e12_summary.json")

    # Print noise summary across K
    if 'noise' in all_results:
        print("\nE12 NOISE SUMMARY: dAUC at p=0.10 (10% depolarizing)")
        print(f"{'K':>4} | {'dAUC(p=0)':>10} | {'dAUC(p=0.10)':>12} | {'ratio':>7}")
        print("-" * 42)
        for k in args.qubits:
            rows = [r for r in all_results['noise'] if r['k'] == k]
            d0 = next(r['delta'] for r in rows if r['p'] == 0.0)
            d10 = next(r['delta'] for r in rows if r['p'] == 0.10)
            ratio = d10 / d0 if d0 != 0 else float('nan')
            pred = (1 - 0.10) ** 2  # (1-p)^2 analytic prediction
            print(f"{k:>4} | {d0:>10.4f} | {d10:>12.4f} | {ratio:>7.3f} (pred {pred:.3f})")


if __name__ == '__main__':
    main()
