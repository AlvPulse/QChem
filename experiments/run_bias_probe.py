"""Quantum graph-topology inductive-bias probe (hardened).

Question: does gating IsingXX entanglement with a molecule's TRUE coarse bond-adjacency beat
gating it with a RANDOM adjacency of equal edge density? The adjacency enters the circuit as
raw per-molecule DATA (no learnable layer in front of it), so -- unlike run_benchmark.py's
permutation scramble, which a free nn.Linear re-absorbs (see docs/03_benchmarking.md, and
_verify_absorb.py: Level 2 is bit-exact identical) -- the topology genuinely cannot be undone
by training. structured > scrambled here therefore isolates a real inductive bias.

Variants (single-qubit encoding identical across all; only the entangler differs):
  structured : IsingXX(true_adj[i,j] * theta)      -- the proposed bias
  scrambled  : IsingXX(rand_adj[i,j] * theta)       -- same density, shuffled topology
  separable  : no IsingXX, no CRZ ring              -- entanglement removed
  classical  : MLP on [coarse feats || adjacency]   -- capacity-unconstrained context

Hardening vs the _alt_b_* scratch:
  * scaffold-grouped CV (Bemis-Murcko) -> structurally novel test folds (OOD), not a random split
  * epoch selected on a scaffold-disjoint VALIDATION ROC, test read once (no test peeking)
  * qubit sweep K in {4,6,8} -> does the bias scale with entanglement capacity?
  * paired structured-vs-scrambled per (fold, seed); sign test + Wilcoxon over the pooled deltas

Usage:
  python run_bias_probe.py --qubits 4 6 8 --folds 5 --seeds 0 1 2 --epochs 30
  python run_bias_probe.py --calibrate          # 2-epoch timing probe per K, then exit
"""
import os, time, argparse, numpy as np, torch, torch.nn as nn
import pennylane as qml
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon, binomtest

from src.data_loader import build_merged_dataframe, murcko_scaffold, CachedGraphDataset

FDIM = 5            # coarse atom features: [atomic_num, gasteiger_q, degree, aromatic, in_ring]
N_TASKS = 12        # Tox21 block


# ---------------- coarse-graph featurization (cheap; from SMILES, no 3D) ----------------
def coarse_graph(smiles, k):
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
        feats.append([a.GetAtomicNum(), q, a.GetDegree(), int(a.GetIsAromatic()), int(a.IsInRing())])
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
    qf = np.zeros((k, FDIM))
    for c in range(k):
        msk = labels == c
        if msk.any():
            qf[c] = feats[msk].mean(0)
    Ac = np.zeros((k, k))
    src, dst = np.nonzero(A)
    for i, j in zip(src, dst):
        if labels[i] != labels[j]:
            Ac[labels[i], labels[j]] += A[i, j]
    Ac /= (Ac.max() + 1e-9)
    return qf.astype(np.float32), Ac.astype(np.float32)


def random_adj_like(A, k, seed):
    """Same edge-weight multiset as A, shuffled onto a random topology (density preserved)."""
    rng = np.random.default_rng(seed)
    iu = np.triu_indices(k, 1); vals = A[iu].copy(); rng.shuffle(vals)
    R = np.zeros((k, k), np.float32); R[iu] = vals; return R + R.T


def featurize(k, datasets, cache_dir='data'):
    cache = os.path.join(cache_dir, f'bias_coarse_K{k}.npz')
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z['QF'], z['AT'], z['AR'], z['Y'], z['SCAF']
    # Align coarse graphs to the featurized dataset order (its y[:, :12] is the Tox21 block).
    pt = os.path.join(cache_dir, f"featurized_{'_'.join(datasets)}.pt")
    payload = torch.load(pt, weights_only=False)
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
            continue                       # keep Tox21-labelled molecules only
        cg = coarse_graph(smi, k)
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


# ---------------- models ----------------
def pairs_of(k):
    return [(i, j) for i in range(k) for j in range(i + 1, k)]


class GraphQ(nn.Module):
    def __init__(self, k, variant='structured', n_layers=2, out_dim=N_TASKS):
        super().__init__()
        self.k = k; self.variant = variant
        entangle = (variant != 'separable')
        PAIRS = pairs_of(k)
        dev = qml.device('default.qubit', wires=k)

        @qml.qnode(dev, interface='torch')
        def circ(ry, rz, adj, theta, ringp, pairp, enc):
            for l in range(n_layers):
                for i in range(k):
                    qml.RY(enc[0] * ry[:, i], wires=i); qml.RZ(enc[1] * rz[:, i], wires=i)
                if entangle:
                    for pidx, (i, j) in enumerate(PAIRS):          # GRAPH-GATED entanglement
                        qml.IsingXX(adj[:, i, j] * pairp[l, pidx], wires=[i, j])
                for i in range(k):
                    qml.RY(theta[l, i, 0], wires=i); qml.RZ(theta[l, i, 1], wires=i)
                if entangle:
                    for i in range(k):
                        qml.CRZ(ringp[l, i], wires=[i, (i + 1) % k])
            return ([qml.expval(qml.PauliX(i)) for i in range(k)] +
                    [qml.expval(qml.PauliY(i)) for i in range(k)] +
                    [qml.expval(qml.PauliZ(i)) for i in range(k)])
        self.circ = circ
        self.feat = nn.Linear(FDIM, 2)
        self.theta = nn.Parameter(torch.randn(n_layers, k, 2) * 0.1)
        self.ringp = nn.Parameter(torch.randn(n_layers, k) * 0.1)
        self.pairp = nn.Parameter(torch.randn(n_layers, len(PAIRS)) * 0.1)
        self.enc = nn.Parameter(torch.ones(2))
        self.head = nn.Linear(3 * k, out_dim)

    def forward(self, qf, adj):
        a = torch.atan(self.feat(qf))
        out = self.circ(a[:, :, 0], a[:, :, 1], adj, self.theta, self.ringp, self.pairp, self.enc)
        return self.head(torch.stack(out, -1).float())


class ClassicalRef(nn.Module):
    def __init__(self, k, out_dim=N_TASKS, h=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(k * FDIM + k * k, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, out_dim))

    def forward(self, qf, adj):
        return self.net(torch.cat([qf.reshape(qf.size(0), -1), adj.reshape(adj.size(0), -1)], -1))


# ---------------- training ----------------
def masked_bce(logits, y, pw):
    m = ~torch.isnan(y); yt = torch.where(m, y, torch.zeros_like(y))
    l = torch.nn.functional.binary_cross_entropy_with_logits(logits, yt, reduction='none', pos_weight=pw)
    return (l * m.float()).sum() / m.sum().clamp(min=1)


def roc12(logits, y):
    p = 1 / (1 + np.exp(-logits)); a = []
    for t in range(N_TASKS):
        v = ~np.isnan(y[:, t])
        if len(np.unique(y[v, t])) > 1:
            a.append(roc_auc_score(y[v, t], p[v, t]))
    return float(np.mean(a)) if a else float('nan')


def pos_weight(Y, tr):
    pw = []
    for t in range(N_TASKS):
        yt = Y[tr][:, t]; v = ~np.isnan(yt); pos = np.nansum(yt[v]); neg = v.sum() - pos
        pw.append(min(neg / (pos + 1e-5), 20.0))
    return torch.tensor(pw, dtype=torch.float32)


def per_task_auc(Y, probs):
    """Per-task ROC over (N,12) probs/labels; NaN where a task has one class."""
    a = np.full(N_TASKS, np.nan)
    for t in range(N_TASKS):
        v = ~np.isnan(Y[:, t]) & ~np.isnan(probs[:, t])
        if len(np.unique(Y[v, t])) > 1:
            a[t] = roc_auc_score(Y[v, t], probs[v, t])
    return a


def train_eval(variant, k, seed, tr, va, te, QF, AT, AR, Y, epochs, batch=128):
    """Select the epoch by VALIDATION roc (no test peeking); return the held-out test-fold
    probabilities (len(te), 12) at that epoch so the caller can pool across folds."""
    torch.manual_seed(seed)
    model = ClassicalRef(k) if variant == 'classical' else GraphQ(k, variant)
    adj = AR if variant == 'scrambled' else AT
    QFt, At, Yt = torch.tensor(QF), torch.tensor(adj), torch.tensor(Y)
    pw = pos_weight(Y, tr)
    qkeys = ('theta', 'ringp', 'pairp', 'enc')
    opt = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if any(q in n for q in qkeys)], 'lr': 1e-2},
        {'params': [p for n, p in model.named_parameters() if not any(q in n for q in qkeys)], 'lr': 1e-3},
    ], weight_decay=1e-4)
    best_va, best_probs = -1.0, np.full((len(te), N_TASKS), np.nan)
    tr_t = torch.as_tensor(tr)
    for _ in range(epochs):
        model.train(); o = torch.randperm(len(tr))
        for s in range(0, len(tr), batch):
            bi = tr_t[o[s:s + batch]]
            loss = masked_bce(model(QFt[bi], At[bi]), Yt[bi], pw)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            va_roc = roc12(model(QFt[va], At[va]).numpy(), Y[va])
            if va_roc > best_va:
                best_va = va_roc
                best_probs = torch.sigmoid(model(QFt[te], At[te])).numpy()
    return best_probs


def scaffold_folds(SCAF, n_splits, val_frac=0.15):
    """Yield (train_idx, val_idx, test_idx) with scaffold-disjoint test AND val."""
    uniq = {s: i for i, s in enumerate(sorted(set(SCAF.tolist())))}
    groups = np.array([uniq[s] for s in SCAF])
    idx = np.arange(len(SCAF))
    gkf = GroupKFold(n_splits=n_splits)
    for tr_all, te in gkf.split(idx, groups=groups):
        g_tr = np.unique(groups[tr_all])
        rng = np.random.default_rng(0); rng.shuffle(g_tr)
        n_val = max(1, int(val_frac * len(g_tr)))
        val_set = set(g_tr[:n_val].tolist())
        is_val = np.array([g in val_set for g in groups[tr_all]])
        yield tr_all[~is_val], tr_all[is_val], te


def standardize(QF, tr):
    mu = QF[tr].reshape(-1, FDIM).mean(0); sd = QF[tr].reshape(-1, FDIM).std(0) + 1e-6
    return (QF - mu) / sd


def run_k(k, datasets, folds, seeds, epochs, context=True):
    QF0, AT, AR, Y, SCAF = featurize(k, datasets)
    fold_list = list(scaffold_folds(SCAF, folds))
    N = len(Y)
    print(f"\n=== K={k} | {N} molecules, {len(set(SCAF.tolist()))} scaffolds, "
          f"{folds} folds x {len(seeds)} seeds | adj nnz/mol {(AT>0).reshape(N,-1).sum(1).mean():.2f} ===",
          flush=True)
    run_deltas, seed_auc_s, seed_auc_c = [], [], []
    ctx = {}
    for seed in seeds:
        pool_s = np.full((N, N_TASKS), np.nan); pool_c = np.full((N, N_TASKS), np.nan)
        for fi, (tr, va, te) in enumerate(fold_list):
            QF = standardize(QF0, tr)
            pool_s[te] = train_eval('structured', k, seed, tr, va, te, QF, AT, AR, Y, epochs)
            pool_c[te] = train_eval('scrambled', k, seed, tr, va, te, QF, AT, AR, Y, epochs)
            # Context (entanglement value + classical gap): one fold, first seed only.
            if context and not ctx and fi == 0 and seed == seeds[0]:
                sep = train_eval('separable', k, seed, tr, va, te, QF, AT, AR, Y, epochs)
                cl = train_eval('classical', k, seed, tr, va, te, QF, AT, AR, Y, epochs)
                ctx = {'separable': float(np.nanmean(per_task_auc(Y[te], sep))),
                       'classical': float(np.nanmean(per_task_auc(Y[te], cl)))}
        auc_s = per_task_auc(Y, pool_s); auc_c = per_task_auc(Y, pool_c)
        seed_auc_s.append(auc_s); seed_auc_c.append(auc_c)
        rd = float(np.nanmean(auc_s) - np.nanmean(auc_c)); run_deltas.append(rd)
        print(f"  K{k} seed{seed}: pooled ROC structured {np.nanmean(auc_s):.4f}  "
              f"scrambled {np.nanmean(auc_c):.4f}  run-BIAS {rd:+.4f}", flush=True)
    # Average per-task ROC across seeds -> 12 stable paired observations.
    A_s = np.nanmean(np.vstack(seed_auc_s), axis=0)
    A_c = np.nanmean(np.vstack(seed_auc_c), axis=0)
    return dict(k=k, run_deltas=np.array(run_deltas), auc_s=A_s, auc_c=A_c, ctx=ctx)


def report(res):
    k = res['k']
    auc_s, auc_c = res['auc_s'], res['auc_c']
    m = ~np.isnan(auc_s) & ~np.isnan(auc_c)
    d = auc_s[m] - auc_c[m]                       # per-task deltas (primary unit)
    n = len(d); npos = int((d > 0).sum())
    sign_p = binomtest(npos, n, 0.5, alternative='greater').pvalue if n else float('nan')
    try:
        w_p = wilcoxon(d, alternative='greater').pvalue
    except ValueError:
        w_p = float('nan')
    rd = res['run_deltas']
    print(f"\n[K={k}] PER-TASK BIAS (structured-scrambled), pooled CV, {n} Tox21 tasks: "
          f"median dAUC {np.median(d):+.4f}, mean {d.mean():+.4f}", flush=True)
    print(f"        {npos}/{n} tasks positive | sign p={sign_p:.4g} | Wilcoxon p={w_p:.4g}", flush=True)
    print(f"        run-level dAUC over {len(rd)} seed(s): {np.round(rd,4).tolist()} "
          f"(mean {rd.mean():+.4f})", flush=True)
    if res['ctx']:
        print(f"        context (fold0): separable {res['ctx']['separable']:.4f}  "
              f"classical {res['ctx']['classical']:.4f}  | structured pooled "
              f"{np.nanmean(auc_s):.4f}", flush=True)
    return dict(k=k, n_tasks=n, median_dauc=float(np.median(d)), mean_dauc=float(d.mean()),
                npos=npos, sign_p=float(sign_p), wilcoxon_p=float(w_p),
                run_mean=float(rd.mean()), ctx=res['ctx'])


def calibrate(datasets, qubits):
    for k in qubits:
        QF0, AT, AR, Y, SCAF = featurize(k, datasets)
        tr, va, te = next(scaffold_folds(SCAF, 5))
        QF = standardize(QF0, tr)
        t0 = time.time()
        train_eval('structured', k, 0, tr, va, te, QF, AT, AR, Y, epochs=2)
        dt = (time.time() - t0) / 2.0
        print(f"K={k}: {dt:.1f}s/epoch  -> ~{dt*30/60:.1f} min per 30-epoch training", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qubits', type=int, nargs='+', default=[4, 6, 8])
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--datasets', type=str, nargs='+', default=['Tox21', 'ToxCast'])
    ap.add_argument('--calibrate', action='store_true')
    ap.add_argument('--no_context', action='store_true')
    args = ap.parse_args()
    if args.calibrate:
        calibrate(args.datasets, args.qubits)
        return
    summary = []
    for k in args.qubits:
        res = run_k(k, args.datasets, args.folds, args.seeds, args.epochs, context=not args.no_context)
        summary.append(report(res))
    print("\n==== BIAS vs QUBITS (per-task paired, pooled scaffold CV) ====", flush=True)
    for s in summary:
        print(f"  K={s['k']:>2}: median dAUC {s['median_dauc']:+.4f}  {s['npos']}/{s['n_tasks']} tasks pos  "
              f"sign p={s['sign_p']:.4g}  Wilcoxon p={s['wilcoxon_p']:.4g}  "
              f"run-level {s['run_mean']:+.4f}", flush=True)


if __name__ == '__main__':
    main()
