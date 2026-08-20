"""Level 8 (a.k.a. "Level G"): a MEASUREMENT-BASED, scalable inductive bias.

This is the project's canonical non-absorbable, scalable inductive-bias level. It does NOT live in
run_benchmark.py's 7-level system: those levels treat qubits as abstract feature slots projected
from a molecule vector, with every input behind a trainable map, so they cannot host a
non-absorbable graph-topology bias (see docs/04_inductive_bias_probe.md sec. 5). Level 8 instead
uses the qubits-as-graph-nodes featurization of run_bias_probe.py (coarse molecular graph -> K
qubits with a real K-by-K bond adjacency), which is what makes the control genuinely non-absorbable.

The 7 existing levels put the bias in gate ROUTING (absorbable; dense O(K^2) gates, 2^K state).
Level 8 instead puts it in WHICH OBSERVABLES are read, selected by the molecular graph:
  * read 2-qubit correlators <Z_i Z_j>, <X_i X_j> for every pair, then
  * bond-pool them per qubit weighted by the coarse adjacency:  b[i] = sum_j A[i,j] * corr(i,j)
The adjacency multiplies the correlator BEFORE the head, so no free Linear can re-route which
physical correlation it harvests -> non-absorbable. Pooling is permutation-invariant and the
readout is O(K) (sparse molecular graphs) -> scalable and hardware-native (2-local Paulis).

We decompose the bias by comparing three configs, each as structured (A=true adj) vs scrambled
(A=random adj, equal density), paired under scaffold CV with the pooled per-task Wilcoxon:
  gate     : graph-gated entangler + single-qubit readout      (bias from GATES only; ~= probe)
  levelG   : graph-gated entangler + bond-correlator readout    (bias from GATES + MEASUREMENT)
  meas_only: FIXED (graph-independent) entangler + bond readout  (bias from MEASUREMENT only)

If meas_only shows a clean structured>scrambled, the measurement mechanism carries bias on its
own; if levelG > gate, the readout adds on top of the gate-gating.
"""
import os, argparse, numpy as np, torch, torch.nn as nn
import pennylane as qml
from scipy.stats import wilcoxon, binomtest

from run_bias_probe import (
    featurize, scaffold_folds, standardize, masked_bce, roc12, pos_weight,
    per_task_auc, pairs_of, FDIM, N_TASKS,
)


class GraphG(nn.Module):
    def __init__(self, k, entangler='graph', readout='graph', n_layers=2, out_dim=N_TASKS, normalize_readout=True, device=None):
        super().__init__()
        self.k = k; self.readout = readout; self.entangler = entangler; self.normalize_readout = normalize_readout
        PAIRS = pairs_of(k); P = len(PAIRS)
        self.pi = torch.tensor([i for i, j in PAIRS]); self.pj = torch.tensor([j for i, j in PAIRS])
        dev_name = device if device else ('lightning.qubit' if k >= 8 else 'default.qubit')
        dev = qml.device(dev_name, wires=k)

        @qml.qnode(dev, interface='torch')
        def circ(ry, rz, adj, theta, ringp, pairp, enc):
            for l in range(n_layers):
                for i in range(k):
                    qml.RY(enc[0] * ry[:, i], wires=i); qml.RZ(enc[1] * rz[:, i], wires=i)
                for pidx, (i, j) in enumerate(PAIRS):
                    if entangler == 'graph':
                        qml.IsingXX(adj[:, i, j] * pairp[l, pidx], wires=[i, j])
                    elif entangler == 'fixed' and j == (i + 1) % k:  # fixed ring, graph-independent
                        qml.IsingXX(pairp[l, pidx], wires=[i, j])
                for i in range(k):
                    qml.RY(theta[l, i, 0], wires=i); qml.RZ(theta[l, i, 1], wires=i)
                for i in range(k):
                    qml.CRZ(ringp[l, i], wires=[i, (i + 1) % k])
            obs = ([qml.expval(qml.PauliX(i)) for i in range(k)] +
                   [qml.expval(qml.PauliY(i)) for i in range(k)] +
                   [qml.expval(qml.PauliZ(i)) for i in range(k)])
            if readout == 'graph':
                obs += [qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)) for i, j in PAIRS]
                obs += [qml.expval(qml.PauliX(i) @ qml.PauliX(j)) for i, j in PAIRS]
                obs += [qml.expval(qml.PauliY(i) @ qml.PauliY(j)) for i, j in PAIRS]
                obs += [qml.expval(qml.PauliX(i) @ qml.PauliZ(j)) for i, j in PAIRS]
                obs += [qml.expval(qml.PauliY(i) @ qml.PauliZ(j)) for i, j in PAIRS]
            return obs
        self.circ = circ; self.P = P
        self.feat = nn.Linear(FDIM, 2)
        self.theta = nn.Parameter(torch.randn(n_layers, k, 2) * 0.1)
        self.ringp = nn.Parameter(torch.randn(n_layers, k) * 0.1)
        self.pairp = nn.Parameter(torch.randn(n_layers, P) * 0.1)
        self.enc = nn.Parameter(torch.ones(2))
        head_in = 3 * k + (5 * k if readout == 'graph' else 0)
        self.head = nn.Linear(head_in, out_dim)

    def _bond_pool(self, corr, adj):
        """corr (B,P) pair correlators -> (B,k): b[i] = sum_j A[i,j] corr(i,j), symmetric."""
        B = corr.size(0)
        w = adj[:, self.pi, self.pj] * corr                  # (B,P) bond-weighted correlator
        b = torch.zeros(B, self.k, device=corr.device)
        b = b.index_add(1, self.pi.to(corr.device), w)
        b = b.index_add(1, self.pj.to(corr.device), w)
        if self.normalize_readout:
            deg = adj.sum(dim=2)  # (B, K) weighted degree
            b = b / (deg + 1e-8)
        return b

    def forward(self, qf, adj):
        a = torch.atan(self.feat(qf))
        out = self.circ(a[:, :, 0], a[:, :, 1], adj, self.theta, self.ringp, self.pairp, self.enc)
        out = [o.float() for o in out]
        k, P = self.k, self.P
        feats = [torch.stack(out[:3 * k], -1)]               # (B,3k) single-qubit X,Y,Z
        if self.readout == 'graph':
            zz = torch.stack(out[3 * k:3 * k + P], -1)        # (B,P)
            xx = torch.stack(out[3 * k + P:3 * k + 2 * P], -1)
            yy = torch.stack(out[3 * k + 2 * P:3 * k + 3 * P], -1)
            xz = torch.stack(out[3 * k + 3 * P:3 * k + 4 * P], -1)
            yz = torch.stack(out[3 * k + 4 * P:3 * k + 5 * P], -1)
            feats += [
                self._bond_pool(zz, adj),
                self._bond_pool(xx, adj),
                self._bond_pool(yy, adj),
                self._bond_pool(xz, adj),
                self._bond_pool(yz, adj)
            ]  # (B,k) each
        return self.head(torch.cat(feats, -1))


class ClassicalGNN(nn.Module):
    """Classical analogue of Level 8's measurement readout, to test whether the *quantum*
    correlator carries topology signal a classical edge feature does not.

    Mirrors Level 8 structurally: a per-node embedding (the analogue of single-qubit observables)
    plus an A-weighted bond-pooled pairwise PRODUCT of node embeddings (the classical counterpart
    of the bond-pooled two-qubit correlator b[i]=sum_j A[i,j]<Z_iZ_j>). The same `structured`
    (true A) vs `scrambled` (random A) control applies, so the structured-scrambled gap measures
    exactly the same thing as Level 8's, but with a classical message-passing readout."""
    def __init__(self, k, d=16, out_dim=N_TASKS):
        super().__init__()
        self.k = k
        self.node = nn.Sequential(nn.Linear(FDIM, d), nn.ReLU(), nn.Linear(d, d))
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, out_dim))

    def forward(self, qf, adj):
        h = self.node(qf)                              # (B,K,d) node embeddings
        # bond-pooled pairwise product: b_i = sum_j A[i,j] (h_i (.) h_j) = h_i (.) sum_j A[i,j] h_j
        agg = torch.einsum('bij,bjd->bid', adj, h)     # (B,K,d) neighbour aggregation
        b = h * agg                                    # (B,K,d) element-wise interaction (~correlator)
        graph = torch.cat([h.mean(1), b.mean(1)], -1)  # (B,2d) permutation-invariant pooling
        return self.head(graph)


def train_eval(cfg, variant, k, seed, tr, va, te, QF, AT, AR, Y, epochs, batch=128):
    torch.manual_seed(seed)
    if cfg.get('kind') == 'classical':
        d = cfg.get('d_by_k', {}).get(k, cfg.get('d', 16))
        model = ClassicalGNN(k, d=d)
    else:
        model = GraphG(k, entangler=cfg['entangler'], readout=cfg['readout'],
                       normalize_readout=cfg.get('normalize_readout', False))
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


def run_cfg(name, cfg, k, datasets, folds, seeds, epochs, max_mols=0, train_frac=1.0):
    QF0, AT, AR, Y, SCAF = featurize(k, datasets)
    if max_mols and max_mols < len(Y):
        # Fixed subsample so a reduced (lower-power) point fits a single foreground call when
        # background execution is unavailable. The structured-vs-scrambled comparison stays paired.
        sel = np.random.default_rng(12345).choice(len(Y), max_mols, replace=False)
        QF0, AT, AR, Y, SCAF = QF0[sel], AT[sel], AR[sel], Y[sel], SCAF[sel]
    fold_list = list(scaffold_folds(SCAF, folds))
    N = len(Y); seed_s, seed_c, run_deltas = [], [], []
    n_train_used = []
    for seed in seeds:
        ps = np.full((N, N_TASKS), np.nan); pc = np.full((N, N_TASKS), np.nan)
        for fi, (tr, va, te) in enumerate(fold_list):
            if train_frac < 1.0:
                # Learning-curve: subsample only the TRAIN set (val/test held full so the per-task
                # ROC is computed on identical molecules across train sizes). Inductive bias should
                # help most when training data is scarce -> the struct-scram gap should widen.
                rng = np.random.default_rng(7000 * seed + fi)
                n_keep = max(4 * N_TASKS, int(round(train_frac * len(tr))))
                tr = rng.choice(tr, min(n_keep, len(tr)), replace=False)
            n_train_used.append(len(tr))
            QF = standardize(QF0, tr)
            ps[te] = train_eval(cfg, 'structured', k, seed, tr, va, te, QF, AT, AR, Y, epochs)
            pc[te] = train_eval(cfg, 'scrambled', k, seed, tr, va, te, QF, AT, AR, Y, epochs)
        a_s, a_c = per_task_auc(Y, ps), per_task_auc(Y, pc)
        seed_s.append(a_s); seed_c.append(a_c)
        run_deltas.append(float(np.nanmean(a_s) - np.nanmean(a_c)))
        print(f"  [{name}] K={k} f={train_frac:g} seed{seed}: struct {np.nanmean(a_s):.4f}  "
              f"scram {np.nanmean(a_c):.4f}  run-dAUC {run_deltas[-1]:+.4f}", flush=True)
    A_s = np.nanmean(np.vstack(seed_s), 0); A_c = np.nanmean(np.vstack(seed_c), 0)
    m = ~np.isnan(A_s) & ~np.isnan(A_c); d = A_s[m] - A_c[m]
    npos = int((d > 0).sum()); n = len(d)
    sgn = binomtest(npos, n, 0.5, alternative='greater').pvalue
    try:
        wp = wilcoxon(d, alternative='greater').pvalue
    except ValueError:
        wp = float('nan')
    ntr = int(np.mean(n_train_used)) if n_train_used else 0
    print(f"\n[{name}] K={k} frac={train_frac:g} (~{ntr} train/fold)  "
          f"structured {np.nanmean(A_s):.4f}  scrambled {np.nanmean(A_c):.4f}", flush=True)
    print(f"        per-task median dAUC {np.median(d):+.4f}  {npos}/{n} pos  "
          f"sign p={sgn:.4g}  Wilcoxon p={wp:.4g}  | run-level {np.mean(run_deltas):+.4f} "
          f"{np.round(run_deltas,4).tolist()}", flush=True)
    return dict(name=name, k=k, train_frac=train_frac, n_train=ntr,
                struct=float(np.nanmean(A_s)), scram=float(np.nanmean(A_c)),
                median=float(np.median(d)), npos=npos, n=n, sign_p=float(sgn), wil_p=float(wp),
                run_deltas=[float(x) for x in run_deltas])


CONFIGS = {
    'gate':         dict(entangler='graph', readout='single'),
    'levelG':       dict(entangler='graph', readout='graph'),
    'meas_only':    dict(entangler='fixed', readout='graph'),
    'classicalGNN': dict(kind='classical'),         # d=16 (~2.6k params, unconstrained context)
    # exact per-K param match to quantum Level 8 (302/452/610): d=7/9/11 -> ~299/435/595 params
    'classicalGNN_pm': dict(kind='classical', d_by_k={4: 7, 6: 9, 8: 11}),
    'levelG_norm':    dict(entangler='graph', readout='graph', normalize_readout=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qubits', type=int, nargs='+', default=[4])
    ap.add_argument('--folds', type=int, default=3)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--configs', type=str, nargs='+', default=list(CONFIGS))
    ap.add_argument('--datasets', type=str, nargs='+', default=['Tox21', 'ToxCast'])
    ap.add_argument('--max_mols', type=int, default=0,
                    help='Subsample to this many molecules (0=all). Lets a reduced point fit a '
                         'single foreground call when background execution is unavailable.')
    ap.add_argument('--train_fracs', type=float, nargs='+', default=[1.0],
                    help='Learning-curve: fractions of the TRAIN set to use (val/test held full). '
                         'Tests whether the inductive-bias gap widens as data gets scarce.')
    ap.add_argument('--out', type=str, default='', help='Optional JSON path to save all rows.')
    args = ap.parse_args()
    rows = []
    for k in args.qubits:
        for frac in args.train_fracs:
            for name in args.configs:
                rows.append(run_cfg(name, CONFIGS[name], k, args.datasets, args.folds, args.seeds,
                                    args.epochs, args.max_mols, frac))
    lc = len(args.train_fracs) > 1
    print("\n==== " + ("LEARNING CURVE (bias vs train fraction)" if lc
                       else "LEVEL G DECOMPOSITION (structured - scrambled bias)") + " ====", flush=True)
    for r in rows:
        tag = f"frac={r['train_frac']:g} (~{r['n_train']} tr)" if lc else ""
        print(f"  {r['name']:>9} K={r['k']} {tag}: median dAUC {r['median']:+.4f}  "
              f"{r['npos']}/{r['n']} pos  sign p={r['sign_p']:.4g}  Wilcoxon p={r['wil_p']:.4g}  "
              f"(struct {r['struct']:.4f} / scram {r['scram']:.4f})", flush=True)
    if args.out:
        import json
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(rows, f, indent=2)
        print(f"\nsaved {len(rows)} rows -> {args.out}", flush=True)


if __name__ == '__main__':
    main()
