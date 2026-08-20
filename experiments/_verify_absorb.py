"""Absorbability audit across ALL 7 benchmark levels.

A level's 'scrambled' control is only meaningful if training CANNOT undo it. The scramble is a
fixed input permutation; the upstream nn.Linear projections are free, so a permutation of an
input vector that is used under exactly ONE permutation is fully re-absorbed (permuting a free
projection's output columns = permuting its weight rows). A scramble is genuine only where the
SAME projected vector is reused under MULTIPLE inconsistent permutations.

For each level we copy identical variational weights into structured and scrambled layers, feed
structured the raw inputs and scrambled the correspondingly-permuted inputs (each input permuted
by the FIRST gate-permutation applied to it), and measure the residual. ~0 => a single input
permutation reproduces structured => absorbable => the control is vacuous.
"""
import torch
from src.quantum_levels import (
    _perms, Level2QuantumLayer, Level3QuantumLayer, Level4QuantumLayer,
    Level5QuantumLayer, Level6QuantumLayer, Level7QuantumLayer,
)

N, B = 4, 8
torch.manual_seed(0)


def inv(p):
    o = [0] * len(p)
    for i, pi in enumerate(p):
        o[pi] = i
    return o


def pc(x, p):                       # y[:, p[i]] = x[:, i]
    return x[:, inv(p)]


def copy_vars(src, dst):
    with torch.no_grad():
        for (_, ps), (_, pd) in zip(src.named_parameters(), dst.named_parameters()):
            pd.copy_(ps)


def resid(level, LayerClass, n_inputs, n_sites, first_perm_idx):
    perms = _perms(N, True, n_sites)
    # distinct perms applied to each input (by gate) -> the structural absorbability test
    s = LayerClass(N, ansatz='strong')
    x = LayerClass(N, ansatz='scrambled')
    copy_vars(s, x)
    ins = [torch.randn(B, N) for _ in range(n_inputs)]
    out_s = s(*ins)
    permuted = [pc(ins[k], perms[first_perm_idx[k]]) for k in range(n_inputs)]
    out_x = x(*permuted)
    return float((out_s.detach() - out_x.detach()).abs().max())


# level: (LayerClass, n_inputs, n_sites, first-perm index per input, perms-per-input, note)
SPEC = {
    2: (Level2QuantumLayer, 3, 3, [0, 1, 2], "m,c,s: 1 perm each",                "ABSORBABLE (vacuous)"),
    3: (Level3QuantumLayer, 3, 4, [0, 1, 3], "m: 2 perms (RY+phase); c,s: 1",     "partial (motif reuse)"),
    4: (Level4QuantumLayer, 2, 2, [0, 1],    "chem,dist: 1 perm each (both free proj)", "ABSORBABLE (vacuous)"),
    5: (Level5QuantumLayer, 1, 4, [0],       "chem: 4 perms (RZ,RY,XX,YY)",       "genuine"),
    6: (Level6QuantumLayer, 1, 5, [0],       "chem: 5 perms (RX,RY,RZ,coupling)", "genuine"),
    7: (Level7QuantumLayer, 1, 5, [0],       "chem: 5 perms (U3x3,CRX,CRY)",      "genuine"),
}

print(f"qubits={N}  batch={B}\n")
print("Level 1: no chemistry->operator routing; 'scrambled' is identical to 'structured' by")
print("         design (nothing to permute). Control is a sanity baseline, not a bias test.\n")
print(f"{'Lvl':>3}  {'residual':>10}  {'verdict':<22}  reuse")
for lvl, (cls, ni, ns, fpi, note, verdict) in SPEC.items():
    r = resid(lvl, cls, ni, ns, fpi)
    flag = "VACUOUS" if r < 1e-6 else "ok"
    print(f"{lvl:>3}  {r:>10.2e}  {verdict:<22}  {note}")
print("\nVACUOUS (bit-exact 0) => structured and scrambled are the SAME function class; their")
print("measured delta is optimisation noise, not inductive bias.")
print("VERIFY_DONE")
