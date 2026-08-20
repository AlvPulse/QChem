"""E7: Kappa locality sweep -- Delta(kappa=2) > Delta(kappa=K)?

Tests T11 clause (iii) + P3: bond-local readout (kappa=2, bond-pooled ZZ/XX) should
produce a larger struct-scram gap than global readout (kappa=K, uniform all-pairs pool).
Motivation: the Cerezo 2021 local-cost theorem predicts polynomial gradient variance
for 2-local observables; global (K-local) costs collapse 2^{-K}.

Three kappa levels at K=6:
  kappa=0: single-qubit only (existing 'gate' readout) -- no pair correlators
  kappa=2: bond-local (existing 'levelG' readout) -- bond-pooled ZZ/XX
  kappa=K: global uniform (new 'all' readout) -- all-pairs ZZ/XX, uniform weight

P3 prediction: Delta(kappa=2) > Delta(kappa=K) > Delta(kappa=0).
Equivalently: bond-local outperforms global, which outperforms single-qubit-only.

Run: python run_e7_kappa_sweep.py [--qubits 4 6 8] [--seeds 0 1 2] [--folds 3]
Output: results/e7_kappa_K{k}.npz + results/e7_kappa_summary.json
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

from run_bias_probe import (featurize, scaffold_folds, standardize, masked_bce,
                             pos_weight, per_task_auc, roc12, N_TASKS, FDIM, pairs_of)
from run_levelG_probe import CONFIGS

EPOCHS = 20


def make_global_adj(k, B):
    """Uniform all-pairs adjacency: 1/(K-1) for all i!=j."""
    A = (torch.ones(B, k, k) - torch.eye(k).unsqueeze(0)) / (k - 1)
    return A.numpy().astype(np.float32)


class GraphGKappa(nn.Module):
    """GraphG with configurable readout locality kappa."""
    def __init__(self, k, kappa, n_layers=2, out_dim=N_TASKS):
        super().__init__()
        self.k = k; self.kappa = kappa
        PAIRS = pairs_of(k); P = len(PAIRS)
        self.pi = torch.tensor([i for i, j in PAIRS])
        self.pj = torch.tensor([j for i, j in PAIRS])
        dev = qml.device('default.qubit', wires=k)

        @qml.qnode(dev, interface='torch')
        def circ(ry, rz, adj, theta, ringp, pairp, enc):
            for l in range(n_layers):
                for i in range(k):
                    qml.RY(enc[0] * ry[:, i], wires=i); qml.RZ(enc[1] * rz[:, i], wires=i)
                for pidx, (i, j) in enumerate(PAIRS):
                    qml.IsingXX(adj[:, i, j] * pairp[l, pidx], wires=[i, j])
                for i in range(k):
                    qml.RY(theta[l, i, 0], wires=i); qml.RZ(theta[l, i, 1], wires=i)
                for i in range(k):
                    qml.CRZ(ringp[l, i], wires=[i, (i + 1) % k])
            obs = ([qml.expval(qml.PauliX(i)) for i in range(k)] +
                   [qml.expval(qml.PauliY(i)) for i in range(k)] +
                   [qml.expval(qml.PauliZ(i)) for i in range(k)])
            if kappa >= 2:
                obs += [qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)) for i, j in PAIRS]
                obs += [qml.expval(qml.PauliX(i) @ qml.PauliX(j)) for i, j in PAIRS]
            return obs
        self.circ = circ; self.P = P
        self.feat = nn.Linear(FDIM, 2)
        self.theta = nn.Parameter(torch.randn(n_layers, k, 2) * 0.1)
        self.ringp = nn.Parameter(torch.randn(n_layers, k) * 0.1)
        self.pairp = nn.Parameter(torch.randn(n_layers, P) * 0.1)
        self.enc = nn.Parameter(torch.ones(2))
        head_in = 3 * k + (2 * k if kappa >= 2 else 0)
        self.head = nn.Linear(head_in, out_dim)

    def _pool(self, corr, pool_adj):
        """corr (B,P) -> (B,K): weighted pool. pool_adj is the weight matrix."""
        B = corr.size(0)
        w = pool_adj[:, self.pi, self.pj] * corr
        b = torch.zeros(B, self.k, device=corr.device)
        b = b.index_add(1, self.pi.to(corr.device), w)
        b = b.index_add(1, self.pj.to(corr.device), w)
        return b

    def forward(self, qf, adj, pool_adj=None):
        """pool_adj: the adjacency used for pooling. None = use adj (bond-local)."""
        if pool_adj is None:
            pool_adj = adj
        a = torch.atan(self.feat(qf))
        out = self.circ(a[:, :, 0], a[:, :, 1], adj, self.theta, self.ringp, self.pairp, self.enc)
        out = [o.float() for o in out]
        k, P = self.k, self.P
        feats = [torch.stack(out[:3 * k], -1)]
        if self.kappa >= 2:
            zz = torch.stack(out[3 * k:3 * k + P], -1)
            xx = torch.stack(out[3 * k + P:3 * k + 2 * P], -1)
            feats += [self._pool(zz, pool_adj), self._pool(xx, pool_adj)]
        return self.head(torch.cat(feats, -1))


def train_kappa(k, kappa, adj, pool_adj, seed, tr, va, te, QF, Y):
    torch.manual_seed(seed)
    model = GraphGKappa(k, kappa)
    QFt = torch.tensor(QF)
    adjt = torch.tensor(adj)
    padj = torch.tensor(pool_adj)
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
            loss = masked_bce(model(QFt[bi], adjt[bi], padj[bi]), Yt[bi], pw)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vr = roc12(model(QFt[va], adjt[va], padj[va]).numpy(), Y[va])
        if vr > best_va:
            best_va = vr
            best_state = {kk: v.clone() for kk, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs_te = torch.sigmoid(model(QFt[te], adjt[te], padj[te])).numpy()
    probs_full = np.full((len(Y), N_TASKS), np.nan, dtype=np.float32)
    probs_full[te] = probs_te
    return float(np.nanmean(per_task_auc(Y, probs_full)))


def run_kappa_k(k, QF0, AT, AR, Y, SCAF, seeds, n_folds):
    print(f"\n[K={k}] Kappa locality sweep", flush=True)
    folds = list(scaffold_folds(SCAF, n_folds))
    N = len(Y)
    AG = make_global_adj(k, N)  # all-pairs uniform

    configs = [
        (0,  AT, AT,  "kappa=0 (single-qubit, gate-only)"),
        (2,  AT, AT,  "kappa=2 (bond-local, structured)"),
        (2,  AR, AR,  "kappa=2 (bond-local, scrambled)"),
        (k,  AT, AG,  "kappa=K (global uniform, struct-entangle)"),
        (k,  AR, AG,  "kappa=K (global uniform, scram-entangle)"),
    ]

    rows = []
    for kap, adj, pool_adj, label in configs:
        aucs = []
        for fi, (tr, va, te) in enumerate(folds):
            QF = standardize(QF0, tr)
            for seed in seeds:
                auc = train_kappa(k, kap, adj, pool_adj, seed, tr, va, te, QF, Y)
                aucs.append(auc)
        mu = float(np.mean(aucs))
        rows.append(dict(k=k, kappa=kap, label=label, auc=mu))
        print(f"  K={k} {label}: AUC={mu:.4f}", flush=True)

    # Compute delta for each kappa (struct - scrambled)
    def delta(kap):
        s = next(r["auc"] for r in rows if r["kappa"] == kap and "struct" in r["label"] or
                 (kap == 0 and r["kappa"] == 0))
        if kap == 0:
            return 0.0  # no scram variant for kappa=0 single-qubit
        c = next(r["auc"] for r in rows if r["kappa"] == kap and "scram" in r["label"])
        return s - c

    d2 = next(r["auc"] for r in rows if r["kappa"] == 2 and "struct" in r["label"]) - \
         next(r["auc"] for r in rows if r["kappa"] == 2 and "scram" in r["label"])
    dK = next(r["auc"] for r in rows if r["kappa"] == k and "struct" in r["label"]) - \
         next(r["auc"] for r in rows if r["kappa"] == k and "scram" in r["label"])

    print(f"\n  K={k} P3 CHECK: dAUC(kappa=2) {d2:+.4f} vs dAUC(kappa=K) {dK:+.4f}  "
          f"{'PASS' if d2 > dK else 'FAIL'} (prediction: kappa=2 > kappa=K)", flush=True)

    np.savez(f"results/e7_kappa_K{k}.npz",
             kappas=np.array([r["kappa"] for r in rows]),
             aucs=np.array([r["auc"] for r in rows]),
             labels=np.array([r["label"] for r in rows]),
             delta_2=d2, delta_K=dK)
    return rows, d2, dK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qubits', type=int, nargs='+', default=[6])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--folds', type=int, default=3)
    ap.add_argument('--datasets', type=str, nargs='+', default=['Tox21', 'ToxCast'])
    args = ap.parse_args()

    summary = []
    for k in args.qubits:
        QF0, AT, AR, Y, SCAF = featurize(k, args.datasets)
        rows, d2, dK = run_kappa_k(k, QF0, AT, AR, Y, SCAF, args.seeds, args.folds)
        summary.append(dict(k=k, delta_kappa2=d2, delta_kappaK=dK,
                            p3_pass=bool(d2 > dK)))

    os.makedirs("results", exist_ok=True)
    with open("results/e7_kappa_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved -> results/e7_kappa_summary.json")

    print("\nE7 KAPPA SUMMARY -- P3 (bond-local > global readout)")
    print(f"{'K':>4} | {'dAUC(kappa=2)':>14} | {'dAUC(kappa=K)':>14} | {'P3 pass?':>10}")
    print("-" * 50)
    for r in summary:
        print(f"{r['k']:>4} | {r['delta_kappa2']:>14.4f} | {r['delta_kappaK']:>14.4f} | "
              f"{'PASS' if r['p3_pass'] else 'FAIL':>10}")


if __name__ == '__main__':
    main()
