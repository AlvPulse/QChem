"""RQ1: does decomposing molecular information into chemically-meaningful profiles
(motif / cycle / spectral) improve learning vs uniform processing of all features?

Both models share an IDENTICAL GINE message-passing backbone and differ ONLY in the readout:
  decomposed : 3 semantic attention-pools (motif = h, cycle = h+aromatic, spectral = h+degree)
               -> 3 downstream sub-nets -> attention-aggregated  (the project's Level-1 design)
  uniform    : 1 attention-pool over a single MLP of the backbone -> 1 downstream sub-net
Parameter counts are matched (uniform sub-net widened) so the contrast isolates the *decomposition*,
not capacity. Tested in BOTH substrates (classical MLP, quantum circuit) on scaffold-grouped CV;
the metric is per-task ROC-AUC, paired (decomposed - uniform) over the 12 Tox21 tasks.

  python run_rq1.py --substrate classical --folds 3 --seeds 0 1 2
  python run_rq1.py --substrate quantum   --folds 3 --seeds 0 1
"""
import os, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import pennylane as qml
from torch_geometric.nn import GINEConv, AttentionalAggregation
from torch_geometric.loader import DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon, binomtest
from src.data_loader import get_or_build_merged_dataset

NODE_VOCAB = (120, 10, 7, 5, 2); EMB = (64, 16, 8, 8, 4)
BOND_VOCAB = (8, 2, 2); BOND_EMB = (16, 4, 4)
H = 64; N_TASKS = 12


# ----------------------------- shared backbone -----------------------------
class Backbone(nn.Module):
    def __init__(self, n_mp=3, dropout=0.2):
        super().__init__()
        self.node_embs = nn.ModuleList([nn.Embedding(v, d) for v, d in zip(NODE_VOCAB, EMB)])
        self.proj = nn.Linear(sum(EMB), H); self.dropout = nn.Dropout(dropout)
        self.bond_embs = nn.ModuleList([nn.Embedding(v, d) for v, d in zip(BOND_VOCAB, BOND_EMB)])
        self.edge_enc = nn.Linear(sum(BOND_EMB) + 1, H)
        self.convs = nn.ModuleList(); self.norms = nn.ModuleList()
        for _ in range(n_mp):
            mlp = nn.Sequential(nn.Linear(H, H), nn.ReLU(), nn.Linear(H, H))
            self.convs.append(GINEConv(mlp, train_eps=True)); self.norms.append(nn.LayerNorm(H))

    def forward(self, data):
        xc = [emb(data.x[:, i].clamp(0, emb.num_embeddings - 1)) for i, emb in enumerate(self.node_embs)]
        h = self.dropout(self.proj(torch.cat(xc, -1)))
        ea = data.edge_attr
        bc = torch.cat([emb(ea[:, k].clamp(0, emb.num_embeddings - 1)) for k, emb in enumerate(self.bond_embs)], -1)
        eac = getattr(data, 'edge_attr_cont', None)
        if eac is None:
            eac = torch.zeros(ea.size(0), 1, device=h.device)
        ee = self.edge_enc(torch.cat([bc, eac], -1))
        for c, n in zip(self.convs, self.norms):
            h = n(h + F.relu(c(h, data.edge_index, ee)))
        batch = getattr(data, 'batch', torch.zeros(h.size(0), dtype=torch.long, device=h.device))
        return h, xc, batch


class DecompHead(nn.Module):
    """Three chemically-meaningful semantic pools (the project's motif/cycle/spectral design)."""
    def __init__(self):
        super().__init__()
        self.motif = nn.Sequential(nn.Linear(H, H), nn.ReLU(), nn.Linear(H, H))
        self.cycle = nn.Sequential(nn.Linear(H + EMB[4], H), nn.ReLU(), nn.Linear(H, H))
        self.spec = nn.Sequential(nn.Linear(H + EMB[1], H), nn.ReLU(), nn.Linear(H, H))
        self.mp = AttentionalAggregation(nn.Linear(H, 1))
        self.cp = AttentionalAggregation(nn.Linear(H, 1))
        self.sp = AttentionalAggregation(nn.Linear(H, 1))

    def forward(self, h, xc, batch):
        m = self.mp(self.motif(h), batch)
        c = self.cp(self.cycle(torch.cat([h, xc[4]], -1)), batch)
        s = self.sp(self.spec(torch.cat([h, xc[1]], -1)), batch)
        return [m, c, s]                              # 3 x (B, H)


class UniformHead(nn.Module):
    """One uniform pool: the SAME backbone information, processed without semantic separation.
    Widened (3 parallel MLPs summed) so its parameter count ~ DecompHead's three streams."""
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(H, 4 * H), nn.ReLU(), nn.Linear(4 * H, H))
        self.pool = AttentionalAggregation(nn.Linear(H, 1))

    def forward(self, h, xc, batch):
        return [self.pool(self.mlp(h), batch)]        # 1 x (B, H)


# ----------------------------- downstream substrates -----------------------------
class ClassicalStream(nn.Module):
    def __init__(self, n_in=H, inner=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, inner), nn.ReLU(), nn.Linear(inner, N_TASKS))

    def forward(self, x):
        return self.net(x)


class QuantumStream(nn.Module):
    """Small trainable circuit per stream (Level-1 style): encode -> RY/RZ + ring CRZ -> <X,Y,Z>."""
    def __init__(self, n_in=H, n_qubits=4, n_layers=2):
        super().__init__()
        self.k = n_qubits
        self.proj = nn.Linear(n_in, n_qubits)
        dev = qml.device('default.qubit', wires=n_qubits)

        @qml.qnode(dev, interface='torch')
        def circ(x, theta, ent):
            for l in range(n_layers):
                for i in range(n_qubits):
                    qml.RY(x[:, i] + theta[l, i, 0], wires=i); qml.RZ(theta[l, i, 1], wires=i)
                for i in range(n_qubits):
                    qml.CRZ(ent[l, i], wires=[i, (i + 1) % n_qubits])
            return ([qml.expval(qml.PauliX(i)) for i in range(n_qubits)] +
                    [qml.expval(qml.PauliY(i)) for i in range(n_qubits)] +
                    [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)])
        self.circ = circ
        self.theta = nn.Parameter(torch.randn(n_layers, n_qubits, 2) * 0.1)
        self.ent = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
        self.head = nn.Linear(3 * n_qubits, N_TASKS)

    def forward(self, x):
        a = torch.atan(self.proj(x))
        out = self.circ(a, self.theta, self.ent)
        return self.head(torch.stack([o.float() for o in out], -1))


class Model(nn.Module):
    def __init__(self, mode, substrate, inner=32):
        super().__init__()
        self.mode = mode
        self.backbone = Backbone()
        self.head = DecompHead() if mode == 'decomposed' else UniformHead()
        n_streams = 3 if mode == 'decomposed' else 1
        n_in = H if mode == 'decomposed' else H
        def stream():
            return ClassicalStream(n_in, inner) if substrate == 'classical' else QuantumStream(n_in)
        self.streams = nn.ModuleList([stream() for _ in range(n_streams)])
        if n_streams > 1:
            self.attn = nn.Sequential(nn.Linear(N_TASKS, 1), nn.Softmax(dim=1))

    def forward(self, data):
        h, xc, batch = self.backbone(data)
        reps = self.head(h, xc, batch)
        logits = [s(r) for s, r in zip(self.streams, reps)]
        if len(logits) == 1:
            return logits[0]
        stk = torch.stack(logits, 1)                  # (B, 3, T)
        w = self.attn(stk)
        return (stk * w).sum(1)


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ----------------------------- train / eval -----------------------------
def masked_bce(logits, y, pw):
    m = ~torch.isnan(y); yt = torch.where(m, y, torch.zeros_like(y))
    l = F.binary_cross_entropy_with_logits(logits, yt, reduction='none', pos_weight=pw)
    return (l * m.float()).sum() / m.sum().clamp(min=1)


def per_task_auc(y, p):
    a = np.full(N_TASKS, np.nan)
    for t in range(N_TASKS):
        v = ~np.isnan(y[:, t]) & ~np.isnan(p[:, t])
        if len(np.unique(y[v, t])) > 1:
            a[t] = roc_auc_score(y[v, t], p[v, t])
    return a


def pos_weight(ds, idx):
    yl = torch.stack([ds[i].y[0, :N_TASKS] for i in idx])
    pw = []
    for t in range(N_TASKS):
        v = ~torch.isnan(yl[:, t]); pos = yl[v, t].sum().item(); neg = v.sum().item() - pos
        pw.append(min(neg / (pos + 1e-5), 20.0))
    return torch.tensor(pw, dtype=torch.float32)


def train_eval(mode, substrate, seed, tr, va, te, ds, inner, epochs, batch=128):
    torch.manual_seed(seed)
    model = Model(mode, substrate, inner)
    pw = pos_weight(ds, tr)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    tr_loader = DataLoader([ds[i] for i in tr], batch_size=batch, shuffle=True)
    va_loader = DataLoader([ds[i] for i in va], batch_size=256)
    te_loader = DataLoader([ds[i] for i in te], batch_size=256)

    def predict(loader):
        model.eval(); P, Y = [], []
        with torch.no_grad():
            for b in loader:
                P.append(torch.sigmoid(model(b)).numpy()); Y.append(b.y[:, :N_TASKS].numpy())
        return np.vstack(P), np.vstack(Y)

    best_va, best = -1.0, None
    for _ in range(epochs):
        model.train()
        for b in tr_loader:
            loss = masked_bce(model(b), b.y[:, :N_TASKS], pw)
            opt.zero_grad(); loss.backward(); opt.step()
        pv, yv = predict(va_loader)
        vr = np.nanmean(per_task_auc(yv, pv))
        if vr > best_va:
            best_va = vr
            pt, yt = predict(te_loader); best = (pt, yt)
    return best, n_params(model)


def run(substrate, folds, seeds, epochs, inner):
    ds, _ = get_or_build_merged_dataset(root_dir='.', datasets=('Tox21', 'ToxCast'),
                                        cache_path='data/featurized_Tox21_ToxCast.pt')
    scaf = list(ds.scaffolds)
    uniq = {s: i for i, s in enumerate(sorted(set(scaf)))}
    groups = np.array([uniq[s] for s in scaf]); idx = np.arange(len(ds))
    gkf = GroupKFold(n_splits=folds)
    folds_list = []
    for tr_all, te in gkf.split(idx, groups=groups):
        g = np.unique(groups[tr_all]); rng = np.random.default_rng(0); rng.shuffle(g)
        vg = set(g[:max(1, int(0.1 * len(g)))].tolist())
        isv = np.array([gg in vg for gg in groups[tr_all]])
        folds_list.append((tr_all[~isv], tr_all[isv], te))

    N = len(ds); pcounts = {}
    seed_auc = {'decomposed': [], 'uniform': []}
    for seed in seeds:
        pools = {m: np.full((N, N_TASKS), np.nan) for m in ('decomposed', 'uniform')}
        for fi, (tr, va, te) in enumerate(folds_list):
            for mode in ('decomposed', 'uniform'):
                (pt, yt), npar = train_eval(mode, substrate, seed, tr, va, te, ds, inner, epochs)
                pools[mode][te] = pt
                pcounts[mode] = npar
            print(f"  [{substrate}] seed{seed} fold{fi} done", flush=True)
        for mode in ('decomposed', 'uniform'):
            yt_all = np.vstack([ds[i].y[:, :N_TASKS].numpy() for i in range(N)])
            seed_auc[mode].append(per_task_auc(yt_all, pools[mode]))
        ad, au = seed_auc['decomposed'][-1], seed_auc['uniform'][-1]
        print(f"  [{substrate}] seed{seed}: decomposed {np.nanmean(ad):.4f}  uniform {np.nanmean(au):.4f}"
              f"  d-u {np.nanmean(ad)-np.nanmean(au):+.4f}", flush=True)

    Ad = np.nanmean(np.vstack(seed_auc['decomposed']), 0)
    Au = np.nanmean(np.vstack(seed_auc['uniform']), 0)
    m = ~np.isnan(Ad) & ~np.isnan(Au); d = Ad[m] - Au[m]
    npos = int((d > 0).sum()); n = len(d)
    sgn = binomtest(npos, n, 0.5, alternative='greater').pvalue
    try:
        wp = wilcoxon(d, alternative='greater').pvalue
    except ValueError:
        wp = float('nan')
    print(f"\n==== RQ1 [{substrate}] decomposed vs uniform ====", flush=True)
    print(f"  decomposed mean ROC {np.nanmean(Ad):.4f} ({pcounts['decomposed']} params)  "
          f"uniform {np.nanmean(Au):.4f} ({pcounts['uniform']} params)", flush=True)
    print(f"  per-task median d-u {np.median(d):+.4f}  {npos}/{n} tasks favour decomposed  "
          f"sign p={sgn:.4g}  Wilcoxon p={wp:.4g}", flush=True)
    return dict(substrate=substrate, dec_roc=float(np.nanmean(Ad)), uni_roc=float(np.nanmean(Au)),
                median=float(np.median(d)), npos=npos, n=n, sign_p=float(sgn), wil_p=float(wp),
                dec_params=pcounts['decomposed'], uni_params=pcounts['uniform'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--substrate', type=str, default='classical', choices=['classical', 'quantum'])
    ap.add_argument('--folds', type=int, default=3)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--inner', type=int, default=48)
    ap.add_argument('--out', type=str, default='')
    args = ap.parse_args()
    res = run(args.substrate, args.folds, args.seeds, args.epochs, args.inner)
    if args.out:
        import json
        json.dump(res, open(args.out, 'w'), indent=2)
        print(f"saved -> {args.out}", flush=True)


if __name__ == '__main__':
    main()
