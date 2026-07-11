"""Alt B + E prototype: molecular-graph entanglement, encoder bypassed.

Each molecule is coarse-grained to k=n_qubits clusters (spectral). The circuit feeds the
raw coarse chemistry into per-qubit rotations and entangles qubit pairs with IsingXX gated
by the TRUE coarse bond-adjacency -> the entanglement topology IS the molecular graph,
which a linear layer cannot absorb.

  structured : IsingXX angle = true_adjacency[i,j] * theta
  scrambled  : IsingXX angle = random_adjacency[i,j] * theta  (same edge density, fixed/mol)
  separable  : no IsingXX (and no variational ring)

If structured > scrambled here, the graph-topology inductive bias carries real signal.
"""
import os, numpy as np, torch, torch.nn as nn
import pennylane as qml
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.cluster import SpectralClustering
from sklearn.metrics import roc_auc_score
from src.data_loader import CachedGraphDataset, build_merged_dataframe

K = 4                      # qubits / coarse clusters
FDIM = 5                   # coarse atom features
CACHE = '_alt_b_coarse.npz'

# ---------- coarse-graph featurization (cheap; from SMILES, no 3D) ----------
def coarse_graph(smiles, k=K):
    m = Chem.MolFromSmiles(smiles)
    if m is None: return None
    n = m.GetNumAtoms()
    try: AllChem.ComputeGasteigerCharges(m)
    except Exception: pass
    feats = []
    for a in m.GetAtoms():
        try: q = float(a.GetProp('_GasteigerCharge'))
        except Exception: q = 0.0
        if not np.isfinite(q): q = 0.0
        feats.append([a.GetAtomicNum(), q, a.GetDegree(), int(a.GetIsAromatic()), int(a.IsInRing())])
    feats = np.asarray(feats, float)
    A = np.zeros((n, n))
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx(); w = b.GetBondTypeAsDouble()
        A[i, j] = A[j, i] = w
    if n <= k:
        labels = np.arange(n)
    else:
        try:
            labels = SpectralClustering(n_clusters=k, affinity='precomputed',
                                        assign_labels='discretize', random_state=0).fit_predict(A + 1e-6)
        except Exception:
            labels = np.arange(n) % k
    qf = np.zeros((k, FDIM))
    for c in range(k):
        msk = labels == c
        if msk.any(): qf[c] = feats[msk].mean(0)
    Ac = np.zeros((k, k))
    src, dst = np.nonzero(A)
    for i, j in zip(src, dst):
        if labels[i] != labels[j]: Ac[labels[i], labels[j]] += A[i, j]
    Ac /= (Ac.max() + 1e-9)
    return qf.astype(np.float32), Ac.astype(np.float32)

def random_adj_like(A, seed):
    rng = np.random.default_rng(seed)
    iu = np.triu_indices(K, 1); vals = A[iu].copy(); rng.shuffle(vals)
    R = np.zeros((K, K), np.float32); R[iu] = vals; return R + R.T

# ---------- build aligned (qfeat, adj_true, adj_rand, y[:12]) ----------
def smiles_in_order():
    df, _ = build_merged_dataframe('.', datasets=('Tox21', 'ToxCast'))
    out = []
    for smi in df['smiles']:
        mm = Chem.MolFromSmiles(smi)
        if mm is None: continue
        try: Chem.Kekulize(mm, clearAromaticFlags=False)
        except Exception: continue
        out.append(smi)
    return out

if os.path.exists(CACHE):
    z = np.load(CACHE)
    QF, AT, AR, Y = z['QF'], z['AT'], z['AR'], z['Y']
else:
    p = torch.load('data/featurized_Tox21_ToxCast.pt', weights_only=False)
    ds = CachedGraphDataset(p['data'], p['slices']); y12 = ds.data.y[:, :12].numpy()
    smis = smiles_in_order()
    assert len(smis) == len(ds), (len(smis), len(ds))
    QF, AT, AR, Y = [], [], [], []
    for i, smi in enumerate(smis):
        yi = y12[i]
        if np.all(np.isnan(yi)): continue          # keep Tox21-labelled molecules
        cg = coarse_graph(smi)
        if cg is None: continue
        qf, ac = cg
        QF.append(qf); AT.append(ac); AR.append(random_adj_like(ac, i)); Y.append(yi)
    QF, AT, AR, Y = map(lambda L: np.asarray(L, np.float32), (QF, AT, AR, Y))
    np.savez(CACHE, QF=QF, AT=AT, AR=AR, Y=Y)
print('coarse set', QF.shape, 'adj nnz/mol mean', (AT > 0).reshape(len(AT), -1).sum(1).mean())

# standardize coarse features (train stats applied below)
rng = np.random.default_rng(0); idx = rng.permutation(len(QF)); nte = int(0.3 * len(QF))
te, tr = idx[:nte], idx[nte:]
mu, sd = QF[tr].reshape(-1, FDIM).mean(0), QF[tr].reshape(-1, FDIM).std(0) + 1e-6
QF = (QF - mu) / sd

# ---------- model ----------
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
                    for pidx, (i, j) in enumerate(PAIRS):       # GRAPH-GATED entanglement (the bias)
                        qml.IsingXX(adj[:, i, j] * pairp[l, pidx], wires=[i, j])
                for i in range(K):                               # trainable variational
                    qml.RY(theta[l, i, 0], wires=i); qml.RZ(theta[l, i, 1], wires=i)
                if entangle:
                    for i in range(K):
                        qml.CRZ(ringp[l, i], wires=[i, (i + 1) % K])
            return ([qml.expval(qml.PauliX(i)) for i in range(K)] +
                    [qml.expval(qml.PauliY(i)) for i in range(K)] +
                    [qml.expval(qml.PauliZ(i)) for i in range(K)])
        self.circ = circ
        self.feat = nn.Linear(FDIM, 2)                          # tiny encoder (Alt E)
        self.theta = nn.Parameter(torch.randn(n_layers, K, 2) * 0.1)
        self.ringp = nn.Parameter(torch.randn(n_layers, K) * 0.1)
        self.pairp = nn.Parameter(torch.randn(n_layers, len(PAIRS)) * 0.1)
        self.enc = nn.Parameter(torch.ones(2))
        self.head = nn.Linear(3 * K, out_dim)
    def forward(self, qf, adj):
        a = torch.atan(self.feat(qf))            # (B,K,2)
        out = self.circ(a[:, :, 0], a[:, :, 1], adj, self.theta, self.ringp, self.pairp, self.enc)
        return self.head(torch.stack(out, -1).float())

class ClassicalRef(nn.Module):
    def __init__(self, out_dim=12, h=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(K * FDIM + K * K, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, out_dim))
    def forward(self, qf, adj):
        return self.net(torch.cat([qf.reshape(qf.size(0), -1), adj.reshape(adj.size(0), -1)], -1))

# ---------- train ----------
dev = 'cpu'
def masked_bce(logits, y, pw):
    m = ~torch.isnan(y); yt = torch.where(m, y, torch.zeros_like(y))
    l = torch.nn.functional.binary_cross_entropy_with_logits(logits, yt, reduction='none', pos_weight=pw)
    l = l * m.float(); return l.sum() / m.sum().clamp(min=1)

def roc12(logits, y):
    p = 1 / (1 + np.exp(-logits)); a = []
    for t in range(12):
        v = ~np.isnan(y[:, t])
        if len(np.unique(y[v, t])) > 1: a.append(roc_auc_score(y[v, t], p[v, t]))
    return float(np.mean(a)) if a else float('nan')

QFt, ATt, ARt, Yt = (torch.tensor(x) for x in (QF, AT, AR, Y))
pw = []
for t in range(12):
    yt = Y[tr][:, t]; v = ~np.isnan(yt); pos = np.nansum(yt[v]); neg = v.sum() - pos
    pw.append(min(neg / (pos + 1e-5), 20.0))
pw = torch.tensor(pw, dtype=torch.float32)

def run(variant):
    torch.manual_seed(7)
    if variant == 'classical': model = ClassicalRef()
    else: model = GraphQ(variant)
    adj_tr = ARt if variant == 'scrambled' else ATt
    opt = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if any(k in n for k in ('theta','ringp','pairp','enc'))], 'lr': 1e-2},
        {'params': [p for n, p in model.named_parameters() if not any(k in n for k in ('theta','ringp','pairp','enc'))], 'lr': 1e-3},
    ], weight_decay=1e-4)
    best = 0.0; B = 128
    for ep in range(30):
        model.train(); o = torch.randperm(len(tr))
        for s in range(0, len(tr), B):
            bi = tr[o[s:s+B]]
            logits = model(QFt[bi], adj_tr[bi]); loss = masked_bce(logits, Yt[bi], pw)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            lg = model(QFt[te], (ARt if variant == 'scrambled' else ATt)[te]).numpy()
        best = max(best, roc12(lg, Y[te]))
    return best

res = {v: run(v) for v in ['structured', 'scrambled', 'separable', 'classical']}
for v in ['structured', 'scrambled', 'separable', 'classical']:
    print(f'{v:11s} best test ROC {res[v]:.4f}')
print(">> BIAS (structured - scrambled): %+.4f" % (res['structured'] - res['scrambled']))
print(">> ENT  (structured - separable): %+.4f" % (res['structured'] - res['separable']))
print('ALTB_DONE')
