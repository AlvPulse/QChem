"""E6: Alpha interpolation sweep -- Delta(lambda) = alpha(lambda)^2 * Delta_ideal.

Tests T9 Lemma 4.2 (SNR proportional to alignment squared) and P2 (monotone
increasing Delta with lambda). Interpolates A_lambda = lambda*AT + (1-lambda)*AR
for lambda in [0, 0.25, 0.5, 0.75, 1.0] at K=6.

P2 prediction: Delta(lambda) is monotone increasing in lambda (alpha >= 0 always
since we interpolate toward true graph; anti-alignment regime requires lambda < 0).

Run: python run_e6_alpha_sweep.py [--qubits 4 6 8] [--seeds 0 1 2] [--folds 3]
Output: results/e6_alpha_K{k}.npz + results/e6_alpha_summary.json
"""
import argparse, json, os
import numpy as np
import torch

from run_bias_probe import (featurize, scaffold_folds, standardize, masked_bce,
                             pos_weight, per_task_auc, roc12, N_TASKS)
from run_levelG_probe import GraphG, CONFIGS

LAMBDAS = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]
EPOCHS = 20


def train_with_adj(k, adj, seed, tr, va, te, QF, Y):
    torch.manual_seed(seed)
    model = GraphG(k, **CONFIGS["levelG"])
    QFt = torch.tensor(QF)
    adjt = torch.tensor(adj)
    Yt = torch.tensor(Y)
    pw = pos_weight(Y, tr)
    qkeys = ("theta", "ringp", "pairp", "enc")
    opt = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if any(q in n for q in qkeys)], "lr": 1e-2},
        {"params": [p for n, p in model.named_parameters() if not any(q in n for q in qkeys)], "lr": 1e-3},
    ], weight_decay=1e-4)
    tr_t = torch.as_tensor(tr)
    best_va, best_state = -1.0, None
    for _ in range(EPOCHS):
        model.train(); o = torch.randperm(len(tr))
        for s in range(0, len(tr), 128):
            bi = tr_t[o[s:s + 128]]
            loss = masked_bce(model(QFt[bi], adjt[bi]), Yt[bi], pw)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vr = roc12(model(QFt[va], adjt[va]).numpy(), Y[va])
        if vr > best_va:
            best_va = vr
            best_state = {kk: v.clone() for kk, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs_te = torch.sigmoid(model(QFt[te], adjt[te])).numpy()
    probs_full = np.full((len(Y), N_TASKS), np.nan, dtype=np.float32)
    probs_full[te] = probs_te
    return float(np.nanmean(per_task_auc(Y, probs_full)))


def run_alpha_k(k, QF0, AT, AR, Y, SCAF, seeds, n_folds):
    print(f"\n[K={k}] Alpha-sweep (lambda={LAMBDAS})", flush=True)
    folds = list(scaffold_folds(SCAF, n_folds))
    N = len(Y)
    rows = []
    for lam in LAMBDAS:
        A_lam = lam * AT + (1 - lam) * AR
        lam_aucs = []
        scram_aucs = []
        for fi, (tr, va, te) in enumerate(folds):
            QF = standardize(QF0, tr)
            for seed in seeds:
                a_lam = train_with_adj(k, A_lam, seed, tr, va, te, QF, Y)
                a_scr = train_with_adj(k, AR, seed, tr, va, te, QF, Y)
                lam_aucs.append(a_lam)
                scram_aucs.append(a_scr)
        mu_lam = float(np.mean(lam_aucs))
        mu_scr = float(np.mean(scram_aucs))
        delta = mu_lam - mu_scr
        rows.append(dict(k=k, lam=lam, auc_lam=mu_lam, auc_scram=mu_scr, delta=delta))
        print(f"  K={k} lam={lam:.2f}: A_lam={mu_lam:.4f} scram={mu_scr:.4f} dAUC={delta:+.4f}",
              flush=True)

    np.savez(f"results/e6_alpha_K{k}.npz",
             lam=np.array(LAMBDAS),
             auc_lam=np.array([r["auc_lam"] for r in rows]),
             auc_scram=np.array([r["auc_scram"] for r in rows]),
             delta=np.array([r["delta"] for r in rows]))

    # P2 check: is Delta monotone increasing with lambda?
    deltas = [r["delta"] for r in rows]
    n_mono = sum(deltas[i] <= deltas[i + 1] for i in range(len(deltas) - 1))
    print(f"  K={k} monotonicity: {n_mono}/{len(deltas)-1} steps increasing", flush=True)
    print(f"  K={k} lam=0 dAUC={deltas[0]:+.4f}, lam=1 dAUC={deltas[-1]:+.4f}  "
          f"(theory: delta_1 > delta_0 > 0)", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qubits', type=int, nargs='+', default=[6])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--folds', type=int, default=3)
    ap.add_argument('--datasets', type=str, nargs='+', default=['Tox21', 'ToxCast'])
    args = ap.parse_args()

    all_rows = []
    for k in args.qubits:
        QF0, AT, AR, Y, SCAF = featurize(k, args.datasets)
        rows = run_alpha_k(k, QF0, AT, AR, Y, SCAF, args.seeds, args.folds)
        all_rows.extend(rows)

    os.makedirs("results", exist_ok=True)
    with open("results/e6_alpha_summary.json", "w") as f:
        json.dump(all_rows, f, indent=2)
    print("\nSaved -> results/e6_alpha_summary.json")

    print("\nE6 ALPHA SUMMARY -- P2 (monotone alignment increase)")
    print(f"{'K':>4} | {'lam=0.00':>10} | {'lam=0.50':>10} | {'lam=1.00':>10} | monotone?")
    print("-" * 60)
    for k in args.qubits:
        krows = [r for r in all_rows if r["k"] == k]
        d0 = next(r["delta"] for r in krows if r["lam"] == 0.0)
        d5 = next(r["delta"] for r in krows if r["lam"] == 0.5)
        d1 = next(r["delta"] for r in krows if r["lam"] == 1.0)
        deltas = [r["delta"] for r in krows]
        mono = all(deltas[i] <= deltas[i + 1] for i in range(len(deltas) - 1))
        print(f"{k:>4} | {d0:>10.4f} | {d5:>10.4f} | {d1:>10.4f} | {'PASS' if mono else 'FAIL'}")


if __name__ == '__main__':
    main()
