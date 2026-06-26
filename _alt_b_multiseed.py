"""Multi-seed paired test of the graph-topology inductive bias.

Reuses the cached coarse graphs from _alt_b_probe.py (_alt_b_coarse.npz). For each seed we
draw a fresh train/test split and fresh init, then train BOTH structured (true bond
adjacency gates IsingXX) and scrambled (random adjacency, same edge density) on the SAME
split -> a paired comparison. We report the per-seed BIAS delta = ROC_structured -
ROC_scrambled and its mean/spread, plus separable/classical context on seed 0 only.

If the delta is centered on ~0 across seeds, the graph-topology inductive bias carries no
signal even in this clean setting (the entanglement topology is raw per-molecule DATA that
no upstream linear layer can absorb, unlike the main benchmark's absorbable permutation).
"""
import numpy as np, torch, torch.nn as nn
import pennylane as qml
from sklearn.metrics import roc_auc_score

K, FDIM = 4, 5
SEEDS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
EPOCHS = 30

z = np.load('_alt_b_coarse.npz')
QF0, AT, AR, Y = z['QF'], z['AT'], z['AR'], z['Y']
print(f"coarse set {QF0.shape}  adj nnz/mol {(AT>0).reshape(len(AT),-1).sum(1).mean():.2f}", flush=True)

PAIRS = [(i, j) for i in range(K) for j in range(i + 1, K)]

class GraphQ(nn.Module):
    def __init__(self, variant='structured', n_layers=2, out_dim=12):
        super().__init__()
        self.variant = variant; entangle = (variant != 'separable')
        dev = qml.device('default.qubit', wires=K)
        @qml.qnode(dev, interface='torch')
        def circ(ry, rz, adj, theta, ringp, pairp, enc):
            for l in range(n_layers):
                for i in range(K):
                    qml.RY(enc[0] * ry[:, i], wires=i); qml.RZ(enc[1] * rz[:, i], wires=i)
                if entangle:
                    for pidx, (i, j) in enumerate(PAIRS):
                        qml.IsingXX(adj[:, i, j] * pairp[l, pidx], wires=[i, j])
                for i in range(K):
                    qml.RY(theta[l, i, 0], wires=i); qml.RZ(theta[l, i, 1], wires=i)
                if entangle:
                    for i in range(K):
                        qml.CRZ(ringp[l, i], wires=[i, (i + 1) % K])
            return ([qml.expval(qml.PauliX(i)) for i in range(K)] +
                    [qml.expval(qml.PauliY(i)) for i in range(K)] +
                    [qml.expval(qml.PauliZ(i)) for i in range(K)])
        self.circ = circ
        self.feat = nn.Linear(FDIM, 2)
        self.theta = nn.Parameter(torch.randn(n_layers, K, 2) * 0.1)
        self.ringp = nn.Parameter(torch.randn(n_layers, K) * 0.1)
        self.pairp = nn.Parameter(torch.randn(n_layers, len(PAIRS)) * 0.1)
        self.enc = nn.Parameter(torch.ones(2))
        self.head = nn.Linear(3 * K, out_dim)
    def forward(self, qf, adj):
        a = torch.atan(self.feat(qf))
        out = self.circ(a[:, :, 0], a[:, :, 1], adj, self.theta, self.ringp, self.pairp, self.enc)
        return self.head(torch.stack(out, -1).float())

class ClassicalRef(nn.Module):
    def __init__(self, out_dim=12, h=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(K * FDIM + K * K, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, out_dim))
    def forward(self, qf, adj):
        return self.net(torch.cat([qf.reshape(qf.size(0), -1), adj.reshape(adj.size(0), -1)], -1))

def masked_bce(logits, y, pw):
    m = ~torch.isnan(y); yt = torch.where(m, y, torch.zeros_like(y))
    l = torch.nn.functional.binary_cross_entropy_with_logits(logits, yt, reduction='none', pos_weight=pw)
    return (l * m.float()).sum() / m.sum().clamp(min=1)

def roc12(logits, y):
    p = 1 / (1 + np.exp(-logits)); a = []
    for t in range(12):
        v = ~np.isnan(y[:, t])
        if len(np.unique(y[v, t])) > 1: a.append(roc_auc_score(y[v, t], p[v, t]))
    return float(np.mean(a)) if a else float('nan')

def train_eval(variant, seed, tr, te, QF):
    torch.manual_seed(seed)
    model = ClassicalRef() if variant == 'classical' else GraphQ(variant)
    adj = AR if variant == 'scrambled' else AT
    QFt, At, Yt = torch.tensor(QF), torch.tensor(adj), torch.tensor(Y)
    pw = []
    for t in range(12):
        yt = Y[tr][:, t]; v = ~np.isnan(yt); pos = np.nansum(yt[v]); neg = v.sum() - pos
        pw.append(min(neg / (pos + 1e-5), 20.0))
    pw = torch.tensor(pw, dtype=torch.float32)
    qkeys = ('theta', 'ringp', 'pairp', 'enc')
    opt = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if any(k in n for k in qkeys)], 'lr': 1e-2},
        {'params': [p for n, p in model.named_parameters() if not any(k in n for k in qkeys)], 'lr': 1e-3},
    ], weight_decay=1e-4)
    best, B = 0.0, 128
    for ep in range(EPOCHS):
        model.train(); o = torch.randperm(len(tr))
        for s in range(0, len(tr), B):
            bi = tr[o[s:s + B]]
            loss = masked_bce(model(QFt[bi], At[bi]), Yt[bi], pw)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            lg = model(QFt[te], At[te]).numpy()
        best = max(best, roc12(lg, Y[te]))
    return best

deltas = []
for seed in SEEDS:
    rng = np.random.default_rng(seed); idx = rng.permutation(len(QF0)); nte = int(0.3 * len(QF0))
    te, tr = idx[:nte], idx[nte:]
    mu, sd = QF0[tr].reshape(-1, FDIM).mean(0), QF0[tr].reshape(-1, FDIM).std(0) + 1e-6
    QF = (QF0 - mu) / sd
    st = train_eval('structured', seed, tr, te, QF)
    sc = train_eval('scrambled', seed, tr, te, QF)
    d = st - sc; deltas.append(d)
    extra = ""
    if seed == 0:
        sep = train_eval('separable', seed, tr, te, QF)
        cl = train_eval('classical', seed, tr, te, QF)
        extra = f"  separable {sep:.4f}  classical {cl:.4f}"
    print(f"seed {seed}: structured {st:.4f}  scrambled {sc:.4f}  BIAS {d:+.4f}{extra}", flush=True)

deltas = np.array(deltas)
print(f"\nBIAS (structured - scrambled) over {len(SEEDS)} seeds:", flush=True)
print(f"  mean {deltas.mean():+.4f}  std {deltas.std():.4f}  "
      f"min {deltas.min():+.4f}  max {deltas.max():+.4f}", flush=True)
print(f"  seeds with structured > scrambled: {(deltas > 0).sum()}/{len(SEEDS)}", flush=True)
se = deltas.std(ddof=1) / np.sqrt(len(deltas))
print(f"  mean +/- 1.96*SE: [{deltas.mean()-1.96*se:+.4f}, {deltas.mean()+1.96*se:+.4f}]", flush=True)
print("MULTISEED_DONE", flush=True)
