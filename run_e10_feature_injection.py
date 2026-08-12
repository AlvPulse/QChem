"""E10: Differential feature injection -- within-cluster variance vs cluster mean.

Tests T2 tightness (P5): the operator-geometry bottleneck means quantum models cannot
benefit from high-frequency within-cluster features that classical models can exploit.

Design: compare AUC gain from AUGMENTED features (cluster mean + within-cluster std/max)
vs ORIGINAL features (cluster mean only) for quantum GraphG and classical ClassicalGNN.

P5 prediction: AUC_classical(augmented) > AUC_classical(original), while
AUC_quantum(augmented) ~= AUC_quantum(original) (bottleneck limits frequency access).

Run: python run_e10_feature_injection.py [--qubits 4 6 8] [--seeds 0 1] [--folds 3]
Output: results/e10_feature_K{k}.npz + results/e10_summary.json
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.cluster import SpectralClustering

from run_bias_probe import (featurize, scaffold_folds,
                             masked_bce, pos_weight, per_task_auc, roc12, N_TASKS,
                             FDIM, coarse_graph, random_adj_like)
from src.data_loader import build_merged_dataframe, murcko_scaffold, CachedGraphDataset
from run_levelG_probe import GraphG, ClassicalGNN, CONFIGS

FDIM_AUG = 3  # additional features: std_charge, max_atomic_num, std_degree
EPOCHS = 20


def augment_coarse_graph(smiles, k):
    """Like coarse_graph but adds within-cluster variance/max features."""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    n = m.GetNumAtoms()
    try:
        AllChem.ComputeGasteigerCharges(m)
    except Exception:
        pass
    feats = []
    for a in m.GetAtoms():
        try:
            q = float(a.GetProp('_GasteigerCharge'))
        except Exception:
            q = 0.0
        if not np.isfinite(q):
            q = 0.0
        feats.append([a.GetAtomicNum(), q, a.GetDegree(),
                      int(a.GetIsAromatic()), int(a.IsInRing())])
    feats = np.asarray(feats, float)
    A = np.zeros((n, n))
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx(); w = b.GetBondTypeAsDouble()
        A[i, j] = A[j, i] = w
    if n <= k:
        labels = np.arange(n) % k
    else:
        try:
            labels = SpectralClustering(n_clusters=k, affinity='precomputed',
                                        assign_labels='discretize', random_state=0).fit_predict(A + 1e-6)
        except Exception:
            labels = np.arange(n) % k

    qf_orig = np.zeros((k, FDIM))
    qf_aug = np.zeros((k, FDIM_AUG))  # std_charge, max_atomic_num, std_degree
    for c in range(k):
        msk = labels == c
        if msk.any():
            qf_orig[c] = feats[msk].mean(0)
            qf_aug[c, 0] = feats[msk, 1].std() if msk.sum() > 1 else 0.0  # std_charge
            qf_aug[c, 1] = feats[msk, 0].max()                              # max_atomic_num
            qf_aug[c, 2] = feats[msk, 2].std() if msk.sum() > 1 else 0.0  # std_degree
    qf_full = np.concatenate([qf_orig, qf_aug], axis=1)  # (K, FDIM+FDIM_AUG)
    Ac = np.zeros((k, k))
    src, dst = np.nonzero(A)
    for i, j in zip(src, dst):
        if labels[i] != labels[j]:
            Ac[labels[i], labels[j]] = max(Ac[labels[i], labels[j]], A[i, j])
            Ac[labels[j], labels[i]] = max(Ac[labels[j], labels[i]], A[i, j])
    Ac /= max(Ac.max(), 1e-9)
    return qf_full, Ac


def featurize_augmented(k, datasets, cache_dir='data'):
    import torch
    cache = os.path.join(cache_dir, f'bias_augmented_K{k}.npz')
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z['QF'], z['AT'], z['AR'], z['Y'], z['SCAF']
    pt = os.path.join(cache_dir, f"featurized_{'_'.join(datasets)}.pt")
    payload = torch.load(pt, weights_only=False)
    from run_bias_probe import CachedGraphDataset
    ds = CachedGraphDataset(payload['data'], payload['slices'])
    y12 = ds.data.y[:, :N_TASKS].numpy()
    df, _ = build_merged_dataframe('.', datasets=tuple(datasets))
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
    assert len(smis) == len(ds), (len(smis), len(ds))
    QF, AT, AR, Y, SCAF = [], [], [], [], []
    for i, smi in enumerate(smis):
        yi = y12[i]
        if np.all(np.isnan(yi)):
            continue
        cg = augment_coarse_graph(smi, k)
        if cg is None:
            continue
        qf, ac = cg
        QF.append(qf); AT.append(ac); AR.append(random_adj_like(ac, k, i))
        Y.append(yi); SCAF.append(murcko_scaffold(smi))
    QF, AT, AR, Y = (np.asarray(L, np.float32) for L in (QF, AT, AR, Y))
    SCAF = np.asarray(SCAF, object)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(cache, QF=QF, AT=AT, AR=AR, Y=Y, SCAF=SCAF)
    return QF, AT, AR, Y, SCAF


def train_auc(model_class, k, fdim, QF0, AT, seed, tr, va, te, Y, epochs):
    torch.manual_seed(seed)
    model = model_class(k)
    # Monkey-patch the input dimension for augmented features
    if hasattr(model, 'feat') and fdim != FDIM:
        model.feat = nn.Linear(fdim, 2)
    QFt = torch.tensor(standardize(QF0, tr))
    At = torch.tensor(AT)
    Yt = torch.tensor(Y)
    pw = pos_weight(Y, tr)
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=1e-2, weight_decay=1e-4)
    tr_t = torch.as_tensor(tr)
    best_va, best_state = -1.0, None
    for _ in range(epochs):
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
    model.eval()
    with torch.no_grad():
        probs_te = torch.sigmoid(model(QFt[te], At[te])).numpy()
    probs_full = np.full((len(Y), N_TASKS), np.nan, dtype=np.float32)
    probs_full[te] = probs_te
    return float(np.nanmean(per_task_auc(Y, probs_full)))


def standardize(QF, tr):
    mu = QF[tr].reshape(-1, QF.shape[-1]).mean(0)
    sd = QF[tr].reshape(-1, QF.shape[-1]).std(0) + 1e-6
    return (QF - mu) / sd


def run_e10_k(k, datasets, seeds, n_folds):
    print(f"\n[K={k}] E10 Feature injection", flush=True)
    QF_orig0, AT, AR, Y, SCAF = featurize(k, datasets)
    QF_aug0, AT2, AR2, Y2, SCAF2 = featurize_augmented(k, datasets)
    assert len(QF_orig0) == len(QF_aug0), "size mismatch"
    folds = list(scaffold_folds(SCAF, n_folds))

    class GraphGAug(GraphG):
        def __init__(self, k):
            super().__init__(k, **CONFIGS["levelG"])
            self.feat = nn.Linear(FDIM + FDIM_AUG, 2)

    class ClassicalGNNAug(ClassicalGNN):
        def __init__(self, k):
            super().__init__(k)
            d = 16
            self.node = nn.Sequential(nn.Linear(FDIM + FDIM_AUG, d), nn.ReLU(), nn.Linear(d, d))

    results = {m: {aug: [] for aug in ('original', 'augmented')}
               for m in ('quantum', 'classical')}

    for fi, (tr, va, te) in enumerate(folds):
        for seed in seeds:
            results['quantum']['original'].append(
                train_auc(GraphG, k, FDIM, QF_orig0, AT, seed, tr, va, te, Y, EPOCHS))
            results['quantum']['augmented'].append(
                train_auc(GraphGAug, k, FDIM + FDIM_AUG, QF_aug0, AT2, seed, tr, va, te, Y, EPOCHS))
            results['classical']['original'].append(
                train_auc(ClassicalGNN, k, FDIM, QF_orig0, AT, seed, tr, va, te, Y, EPOCHS))
            results['classical']['augmented'].append(
                train_auc(ClassicalGNNAug, k, FDIM + FDIM_AUG, QF_aug0, AT2, seed, tr, va, te, Y, EPOCHS))
        print(f"  K={k} fold {fi+1}/{n_folds} done", flush=True)

    rows = {}
    for m in ('quantum', 'classical'):
        orig = float(np.mean(results[m]['original']))
        aug = float(np.mean(results[m]['augmented']))
        gain = aug - orig
        rows[m] = dict(original=orig, augmented=aug, gain=gain)
        print(f"  K={k} {m:10s}: orig={orig:.4f} aug={aug:.4f} gain={gain:+.4f}", flush=True)
    p5_pass = rows['classical']['gain'] > rows['quantum']['gain']
    print(f"  K={k} P5: {'PASS' if p5_pass else 'FAIL'} (classical gain {rows['classical']['gain']:+.4f} "
          f"vs quantum gain {rows['quantum']['gain']:+.4f})", flush=True)
    np.savez(f"results/e10_feature_K{k}.npz",
             quantum_orig=results['quantum']['original'],
             quantum_aug=results['quantum']['augmented'],
             classical_orig=results['classical']['original'],
             classical_aug=results['classical']['augmented'])
    rows['k'] = k; rows['p5_pass'] = bool(p5_pass)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qubits', type=int, nargs='+', default=[6])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--folds', type=int, default=3)
    ap.add_argument('--datasets', type=str, nargs='+', default=['Tox21', 'ToxCast'])
    args = ap.parse_args()

    summary = []
    for k in args.qubits:
        row = run_e10_k(k, args.datasets, args.seeds, args.folds)
        summary.append(row)

    os.makedirs("results", exist_ok=True)
    with open("results/e10_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved -> results/e10_summary.json")
    print("\nE10 SUMMARY -- P5 (classical gains more from high-freq features)")
    print(f"{'K':>4} | {'cls_gain':>10} | {'qml_gain':>10} | {'P5?':>6}")
    print("-" * 38)
    for r in summary:
        print(f"{r['k']:>4} | {r['classical']['gain']:>10.4f} | "
              f"{r['quantum']['gain']:>10.4f} | {'PASS' if r['p5_pass'] else 'FAIL':>6}")


if __name__ == '__main__':
    main()
