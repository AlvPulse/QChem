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
import argparse, numpy as np, torch, torch.nn as nn
import pennylane as qml
from scipy.stats import wilcoxon, binomtest

from run_bias_probe import (
    featurize, scaffold_folds, standardize, masked_bce, roc12, pos_weight,
    per_task_auc, pairs_of, FDIM, N_TASKS,
)


class GraphG(nn.Module):
    def __init__(self, k, entangler='graph', readout='graph', n_layers=2, out_dim=N_TASKS):
        super().__init__()
        self.k = k; self.readout = readout; self.entangler = entangler
        PAIRS = pairs_of(k); P = len(PAIRS)
        self.pi = torch.tensor([i for i, j in PAIRS]); self.pj = torch.tensor([j for i, j in PAIRS])
        dev = qml.device('default.qubit', wires=k)

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
            return obs
        self.circ = circ; self.P = P
        self.feat = nn.Linear(FDIM, 2)
        self.theta = nn.Parameter(torch.randn(n_layers, k, 2) * 0.1)
        self.ringp = nn.Parameter(torch.randn(n_layers, k) * 0.1)
        self.pairp = nn.Parameter(torch.randn(n_layers, P) * 0.1)
        self.enc = nn.Parameter(torch.ones(2))
        head_in = 3 * k + (2 * k if readout == 'graph' else 0)
        self.head = nn.Linear(head_in, out_dim)

    def _bond_pool(self, corr, adj):
        """corr (B,P) pair correlators -> (B,k): b[i] = sum_j A[i,j] corr(i,j), symmetric."""
        B = corr.size(0)
        w = adj[:, self.pi, self.pj] * corr                  # (B,P) bond-weighted correlator
        b = torch.zeros(B, self.k, device=corr.device)
        b = b.index_add(1, self.pi.to(corr.device), w)
        b = b.index_add(1, self.pj.to(corr.device), w)
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
            feats += [self._bond_pool(zz, adj), self._bond_pool(xx, adj)]  # (B,k) each
        return self.head(torch.cat(feats, -1))


def train_eval(cfg, variant, k, seed, tr, va, te, QF, AT, AR, Y, epochs, batch=128):
    torch.manual_seed(seed)
    model = GraphG(k, entangler=cfg['entangler'], readout=cfg['readout'])
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


def run_cfg(name, cfg, k, datasets, folds, seeds, epochs, max_mols=0):
    QF0, AT, AR, Y, SCAF = featurize(k, datasets)
    if max_mols and max_mols < len(Y):
        # Fixed subsample so a reduced (lower-power) point fits a single foreground call when
        # background execution is unavailable. The structured-vs-scrambled comparison stays paired.
        sel = np.random.default_rng(12345).choice(len(Y), max_mols, replace=False)
        QF0, AT, AR, Y, SCAF = QF0[sel], AT[sel], AR[sel], Y[sel], SCAF[sel]
    fold_list = list(scaffold_folds(SCAF, folds))
    N = len(Y); seed_s, seed_c, run_deltas = [], [], []
    for seed in seeds:
        ps = np.full((N, N_TASKS), np.nan); pc = np.full((N, N_TASKS), np.nan)
        for tr, va, te in fold_list:
            QF = standardize(QF0, tr)
            ps[te] = train_eval(cfg, 'structured', k, seed, tr, va, te, QF, AT, AR, Y, epochs)
            pc[te] = train_eval(cfg, 'scrambled', k, seed, tr, va, te, QF, AT, AR, Y, epochs)
        a_s, a_c = per_task_auc(Y, ps), per_task_auc(Y, pc)
        seed_s.append(a_s); seed_c.append(a_c)
        run_deltas.append(float(np.nanmean(a_s) - np.nanmean(a_c)))
        print(f"  [{name}] K={k} seed{seed}: struct {np.nanmean(a_s):.4f}  "
              f"scram {np.nanmean(a_c):.4f}  run-dAUC {run_deltas[-1]:+.4f}", flush=True)
    A_s = np.nanmean(np.vstack(seed_s), 0); A_c = np.nanmean(np.vstack(seed_c), 0)
    m = ~np.isnan(A_s) & ~np.isnan(A_c); d = A_s[m] - A_c[m]
    npos = int((d > 0).sum()); n = len(d)
    sgn = binomtest(npos, n, 0.5, alternative='greater').pvalue
    try:
        wp = wilcoxon(d, alternative='greater').pvalue
    except ValueError:
        wp = float('nan')
    print(f"\n[{name}] K={k}  structured {np.nanmean(A_s):.4f}  scrambled {np.nanmean(A_c):.4f}",
          flush=True)
    print(f"        per-task median dAUC {np.median(d):+.4f}  {npos}/{n} pos  "
          f"sign p={sgn:.4g}  Wilcoxon p={wp:.4g}  | run-level {np.mean(run_deltas):+.4f} "
          f"{np.round(run_deltas,4).tolist()}", flush=True)
    return dict(name=name, k=k, struct=float(np.nanmean(A_s)), scram=float(np.nanmean(A_c)),
                median=float(np.median(d)), npos=npos, n=n, sign_p=float(sgn), wil_p=float(wp))


CONFIGS = {
    'gate':      dict(entangler='graph', readout='single'),
    'levelG':    dict(entangler='graph', readout='graph'),
    'meas_only': dict(entangler='fixed', readout='graph'),
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
    args = ap.parse_args()
    rows = []
    for k in args.qubits:
        for name in args.configs:
            rows.append(run_cfg(name, CONFIGS[name], k, args.datasets, args.folds, args.seeds,
                                args.epochs, args.max_mols))
    print("\n==== LEVEL G DECOMPOSITION (structured - scrambled bias) ====", flush=True)
    for r in rows:
        print(f"  {r['name']:>9} K={r['k']}: median dAUC {r['median']:+.4f}  {r['npos']}/{r['n']} pos  "
              f"sign p={r['sign_p']:.4g}  Wilcoxon p={r['wil_p']:.4g}  "
              f"(struct {r['struct']:.4f} / scram {r['scram']:.4f})", flush=True)


if __name__ == '__main__':
    main()
