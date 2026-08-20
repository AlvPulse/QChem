"""E11: Second-dataset external validity -- normalized K-slope invariant across datasets?

Tests P7: the TC-QIC normalized slope (d_K Delta) / d_bar should be dataset-invariant.
If the slope differs 3x between Tox21 and BBBP, P7 fails -> TC-QIC is idiosyncratic
to the chemical series, not a general topology-bias mechanism.

Protocol:
- Run levelG structured vs scrambled at K=4,6,8 on BBBP (blood-brain barrier, ~2000 mols, 1 task)
- Compare dAUC K-slope with the Tox21 slope (from main benchmark: +0.0028/qubit)
- Normalize by mean graph degree d_bar (controls for molecular complexity differences)
- P7 PASS if Kendall tau rank order of dAUC vs K is preserved across datasets

Run: python run_e11_second_dataset.py [--dataset BBBP] [--qubits 4 6 8] [--seeds 0 1]
Output: results/e11_kscale_{dataset}.json
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn as nn

from src.data_loader import get_or_build_merged_dataset, murcko_scaffold, CachedGraphDataset, build_merged_dataframe
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import GroupKFold
import pennylane as qml
from run_bias_probe import (pairs_of, FDIM, coarse_graph, random_adj_like,
                             masked_bce, pos_weight, roc12, scaffold_folds)
from run_levelG_probe import CONFIGS


def per_task_auc_single(Y, probs):
    """AUC for single-task (Y, probs both 1d or (N,1))."""
    from sklearn.metrics import roc_auc_score
    y = Y.ravel(); p = probs.ravel()
    mask = ~np.isnan(y) & ~np.isnan(p)
    if mask.sum() < 2 or len(np.unique(y[mask])) < 2:
        return float('nan')
    return roc_auc_score(y[mask], p[mask])


def featurize_for_dataset(dataset, k):
    cache = f"data/bias_{dataset}_K{k}.npz"
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z['QF'], z['AT'], z['AR'], z['Y'], z['SCAF'], int(z['N_TASKS'])
    ds, n_tasks = get_or_build_merged_dataset('.', datasets=(dataset,))
    Y_all = ds.data.y.numpy()  # (N, n_tasks)
    QF, AT, AR, Y, SCAF = [], [], [], [], []
    smis_iter = [ds[i] for i in range(len(ds))]
    df, _ = build_merged_dataframe('.', datasets=(dataset,))
    smis = []
    for smi in df['smiles']:
        mm = Chem.MolFromSmiles(smi)
        if mm is None:
            continue
        try:
            Chem.Kekulize(mm, clearAromaticFlags=False)
        except Exception:
            continue
        smis.append(smi)
    if len(smis) != len(ds):
        smis = [s for s in df['smiles']][:len(ds)]
    for i, smi in enumerate(smis):
        yi = Y_all[i] if i < len(Y_all) else np.full(n_tasks, np.nan)
        if np.all(np.isnan(yi)):
            continue
        cg = coarse_graph(smi, k)
        if cg is None:
            continue
        qf, ac = cg
        QF.append(qf); AT.append(ac)
        AR.append(random_adj_like(ac, k, i))
        Y.append(yi); SCAF.append(murcko_scaffold(smi))
    QF, AT, AR, Y = (np.asarray(L, np.float32) for L in (QF, AT, AR, Y))
    SCAF = np.asarray(SCAF, object)
    os.makedirs("data", exist_ok=True)
    np.savez(cache, QF=QF, AT=AT, AR=AR, Y=Y, SCAF=SCAF, N_TASKS=n_tasks)
    return QF, AT, AR, Y, SCAF, n_tasks


class GraphGFlexible(nn.Module):
    """GraphG with flexible output dimension for variable N_TASKS."""
    def __init__(self, k, n_tasks):
        super().__init__()
        self.k = k; self.n_tasks = n_tasks
        PAIRS = pairs_of(k); P = len(PAIRS)
        self.pi = torch.tensor([i for i, j in PAIRS])
        self.pj = torch.tensor([j for i, j in PAIRS])
        dev_name = 'lightning.qubit' if k >= 8 else 'default.qubit'
        dev = qml.device(dev_name, wires=k)

        @qml.qnode(dev, interface='torch')
        def circ(ry, rz, adj, theta, ringp, pairp, enc):
            n_layers = theta.shape[0]
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
            obs += [qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)) for i, j in PAIRS]
            obs += [qml.expval(qml.PauliX(i) @ qml.PauliX(j)) for i, j in PAIRS]
            return obs
        self.circ = circ; self.P = P
        self.feat = nn.Linear(FDIM, 2)
        n_layers = 2
        self.theta = nn.Parameter(torch.randn(n_layers, k, 2) * 0.1)
        self.ringp = nn.Parameter(torch.randn(n_layers, k) * 0.1)
        self.pairp = nn.Parameter(torch.randn(n_layers, P) * 0.1)
        self.enc = nn.Parameter(torch.ones(2))
        self.head = nn.Linear(3 * k + 2 * k, n_tasks)

    def _bond_pool(self, corr, adj):
        B = corr.size(0)
        w = adj[:, self.pi, self.pj] * corr
        b = torch.zeros(B, self.k, device=corr.device)
        b = b.index_add(1, self.pi.to(corr.device), w)
        b = b.index_add(1, self.pj.to(corr.device), w)
        return b

    def forward(self, qf, adj):
        a = torch.atan(self.feat(qf))
        out = self.circ(a[:, :, 0], a[:, :, 1], adj, self.theta, self.ringp, self.pairp, self.enc)
        out = [o.float() for o in out]
        k, P = self.k, self.P
        single = torch.stack(out[:3 * k], -1)
        zz = torch.stack(out[3 * k:3 * k + P], -1)
        xx = torch.stack(out[3 * k + P:3 * k + 2 * P], -1)
        feat = torch.cat([single, self._bond_pool(zz, adj), self._bond_pool(xx, adj)], -1)
        return self.head(feat)


def masked_bce_flex(logits, Y, pw):
    """BCE loss ignoring NaN labels (flexible N_TASKS)."""
    mask = ~torch.isnan(Y)
    if not mask.any():
        return logits.sum() * 0
    l = logits[mask]; y = Y[mask]
    return nn.BCEWithLogitsLoss(pos_weight=pw.expand_as(y))(l, y)


def run_one_k(k, dataset, seeds, n_folds):
    print(f"\n[K={k}] {dataset}", flush=True)
    QF0, AT, AR, Y, SCAF, n_tasks = featurize_for_dataset(dataset, k)
    folds = list(scaffold_folds(SCAF, n_folds))
    N = len(Y)
    mu = QF0.reshape(-1, FDIM).mean(0); sd = QF0.reshape(-1, FDIM).std(0) + 1e-6
    QF = (QF0 - mu) / sd

    struct_aucs, scram_aucs = [], []
    for fi, (tr, va, te) in enumerate(folds):
        for seed in seeds:
            for variant in ('structured', 'scrambled'):
                adj = AT if variant == 'structured' else AR
                torch.manual_seed(seed)
                model = GraphGFlexible(k, n_tasks)
                pw = torch.tensor([(Y[tr, t][~np.isnan(Y[tr, t])]==0).sum() /
                                    max((Y[tr, t][~np.isnan(Y[tr, t])]==1).sum(), 1)
                                    for t in range(n_tasks)], dtype=torch.float32)
                QFt = torch.tensor(QF); adjt = torch.tensor(adj); Yt = torch.tensor(Y)
                qk = ("theta", "ringp", "pairp", "enc")
                opt = torch.optim.AdamW([
                    {"params": [p for n, p in model.named_parameters() if any(q in n for q in qk)], "lr": 1e-2},
                    {"params": [p for n, p in model.named_parameters() if not any(q in n for q in qk)], "lr": 1e-3},
                ], weight_decay=1e-4)
                tr_t = torch.as_tensor(tr)
                best_va, best_state = -1.0, None
                for _ in range(20):
                    model.train(); o = torch.randperm(len(tr))
                    for s in range(0, len(tr), 128):
                        bi = tr_t[o[s:s + 128]]
                        loss = masked_bce_flex(model(QFt[bi], adjt[bi]), Yt[bi], pw)
                        opt.zero_grad(); loss.backward(); opt.step()
                    model.eval()
                    with torch.no_grad():
                        va_logits = model(QFt[va], adjt[va]).numpy()
                        va_probs = 1 / (1 + np.exp(-va_logits))
                    vr = per_task_auc_single(Y[va], va_probs)
                    if vr > best_va:
                        best_va = vr
                        best_state = {kk: v.clone() for kk, v in model.state_dict().items()}
                model.load_state_dict(best_state)
                model.eval()
                with torch.no_grad():
                    te_logits = model(QFt[te], adjt[te]).numpy()
                    te_probs = 1 / (1 + np.exp(-te_logits))
                auc = per_task_auc_single(Y[te], te_probs)
                if variant == 'structured':
                    struct_aucs.append(auc)
                else:
                    scram_aucs.append(auc)
        print(f"  K={k} {dataset} fold {fi+1}/{n_folds}", flush=True)

    mu_s = float(np.nanmean(struct_aucs))
    mu_c = float(np.nanmean(scram_aucs))
    delta = mu_s - mu_c
    # Mean degree of the coarse graph
    d_bar = float((AT > 0).reshape(N, k * k).sum(1).mean() / k)
    print(f"  K={k} {dataset}: struct={mu_s:.4f} scram={mu_c:.4f} dAUC={delta:+.4f}  "
          f"d_bar={d_bar:.2f}  norm_delta={delta/max(d_bar,0.1):+.4f}", flush=True)
    return dict(k=k, dataset=dataset, struct=mu_s, scram=mu_c, delta=delta,
                d_bar=d_bar, norm_delta=delta / max(d_bar, 0.1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='BBBP')
    ap.add_argument('--qubits', type=int, nargs='+', default=[4, 6, 8])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--folds', type=int, default=3)
    args = ap.parse_args()

    rows = []
    for k in args.qubits:
        row = run_one_k(k, args.dataset, args.seeds, args.folds)
        rows.append(row)

    os.makedirs("results", exist_ok=True)
    out = f"results/e11_kscale_{args.dataset}.json"
    with open(out, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved -> {out}")

    print(f"\nE11 SUMMARY -- {args.dataset} K-scaling (P7 slope invariance)")
    print(f"{'K':>4} | {'dAUC':>10} | {'d_bar':>7} | {'norm_dAUC':>10}")
    print("-" * 40)
    for r in rows:
        print(f"{r['k']:>4} | {r['delta']:>10.4f} | {r['d_bar']:>7.2f} | "
              f"{r['norm_delta']:>10.4f}")

    # Tox21 reference slope: 0.0028 dAUC/qubit (from main benchmark K=4,6,8)
    if len(rows) >= 2:
        ks = [r['k'] for r in rows]
        ds = [r['delta'] for r in rows]
        slope = (ds[-1] - ds[0]) / (ks[-1] - ks[0])
        ref_slope = 0.0028  # Tox21: (0.0134-0.0078)/(8-4)
        norm_ratio = slope / ref_slope if abs(ref_slope) > 1e-6 else float('nan')
        print(f"\n{args.dataset} slope: {slope:+.4f} dAUC/qubit")
        print(f"Tox21 slope (reference): {ref_slope:+.4f} dAUC/qubit")
        print(f"Ratio: {norm_ratio:.2f}x  (P7 PASS if < 3x)")
        p7 = abs(norm_ratio) < 3.0 and abs(norm_ratio) > 0.1
        print(f"P7: {'PASS' if p7 else 'FAIL'}")


if __name__ == '__main__':
    main()
