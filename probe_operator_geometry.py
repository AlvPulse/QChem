"""probe_operator_geometry.py

Empirically validates T2: the bond-pooled feature covariance matrix has
effective rank Theta(K) -- i.e., the operator-geometry dimension grows
linearly with the number of qubits K.

Measures effective rank and eigenvalue decay of the bond-pooled feature
covariance matrix across K=4,6,8 using random-weight GraphG (levelG) models.

Feature blocks extracted before the linear head:
  Phi_sq  (N, 3K)  -- single-qubit X, Y, Z observables
  Phi_bp  (N, 2K)  -- bond-pooled ZZ and XX two-qubit correlators

Usage:
  python probe_operator_geometry.py --qubits 4 6 8 --n_samples 500
  python probe_operator_geometry.py --qubits 4 --n_samples 200 --batch_size 16
"""
import os
import sys
import json
import argparse

import numpy as np
import torch

from run_bias_probe import featurize, standardize, FDIM, N_TASKS
from run_levelG_probe import GraphG


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(model, QFt, At, batch_size=32):
    """Extract pre-head features for all molecules via GraphG internals.

    Replicates GraphG.forward() up to (but not including) self.head, so we
    can inspect the raw feature geometry.

    Returns
    -------
    Phi_sq : Tensor (N, 3K)  -- single-qubit X,Y,Z observables
    Phi_bp : Tensor (N, 2K)  -- bond-pooled ZZ+XX correlators (K each)
    """
    k = model.k
    P = model.P
    sq_chunks = []
    bp_chunks = []

    with torch.no_grad():
        for start in range(0, QFt.shape[0], batch_size):
            end = min(start + batch_size, QFt.shape[0])
            qf_b = QFt[start:end]
            adj_b = At[start:end]

            # Encode atom features to (ry, rz) angles -- matches GraphG.forward
            a = torch.atan(model.feat(qf_b))            # (B, K, 2)
            ry = a[:, :, 0]
            rz = a[:, :, 1]

            # Run the quantum circuit
            out = model.circ(ry, rz, adj_b,
                             model.theta, model.ringp, model.pairp, model.enc)
            out = [o.float() for o in out]

            # Single-qubit block: X[0..k-1], Y[0..k-1], Z[0..k-1]
            sq = torch.stack(out[:3 * k], dim=-1)       # (B, 3K)

            # Two-qubit correlators: ZZ then XX, each length P
            zz = torch.stack(out[3 * k:3 * k + P], dim=-1)      # (B, P)
            xx = torch.stack(out[3 * k + P:3 * k + 2 * P], dim=-1)  # (B, P)

            # Bond-pool: b[i] = sum_j A[i,j] * corr(i,j)
            bp_zz = model._bond_pool(zz, adj_b)         # (B, K)
            bp_xx = model._bond_pool(xx, adj_b)          # (B, K)
            bp = torch.cat([bp_zz, bp_xx], dim=-1)       # (B, 2K)

            sq_chunks.append(sq.cpu())
            bp_chunks.append(bp.cpu())

    Phi_sq = torch.cat(sq_chunks, dim=0)   # (N, 3K)
    Phi_bp = torch.cat(bp_chunks, dim=0)   # (N, 2K)
    return Phi_sq, Phi_bp


# ---------------------------------------------------------------------------
# Geometry metrics
# ---------------------------------------------------------------------------

def effective_rank(eigs):
    """Effective rank = (sum eigs)^2 / sum(eigs^2).

    Equals the number of dimensions that carry equal variance (entropy-based
    rank, Roy and Vetterli 2007).  For a K-dimensional uniform spectrum
    this returns K exactly.
    """
    s1 = eigs.sum()
    s2 = (eigs ** 2).sum()
    return float((s1 ** 2 / (s2 + 1e-15)).item())


def covariance_eigs(Phi):
    """Centre Phi, compute gram/N, return eigenvalues sorted descending."""
    Phi_c = Phi - Phi.mean(dim=0)
    C = Phi_c.T @ Phi_c / Phi_c.shape[0]
    eigs = torch.linalg.eigvalsh(C)        # ascending by convention
    eigs = eigs.flip(0).clamp(min=0.0)    # descending, clamp float noise
    return eigs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='Operator-geometry dimension probe (T2).')
    ap.add_argument('--qubits', type=int, nargs='+', default=[4, 6, 8],
                    help='Qubit counts to sweep (default: 4 6 8).')
    ap.add_argument('--n_samples', type=int, default=500,
                    help='Max molecules per K (subsampled from full dataset).')
    ap.add_argument('--out', type=str, default='results/probe_opgeom.json',
                    help='Output JSON path.')
    ap.add_argument('--datasets', type=str, nargs='+', default=['Tox21', 'ToxCast'],
                    help='Datasets to featurize.')
    ap.add_argument('--seed', type=int, default=0,
                    help='RNG seed for subsampling and model init.')
    ap.add_argument('--batch_size', type=int, default=32,
                    help='Forward-pass batch size (reduce if OOM).')
    ap.add_argument('--train_frac', type=float, default=0.8,
                    help='Fraction used as "train" for QF standardization.')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    rows = []

    col_w = 3, 7, 9, 8, 8, 36
    header = (f"{'K':>{col_w[0]}} | {'n_feats':>{col_w[1]}} | "
              f"{'eff_rank':>{col_w[2]}} | {'frac_SQ':>{col_w[3]}} | "
              f"{'frac_BP':>{col_w[4]}} | top3_eigs")
    sep = '-' * (sum(col_w) + 4 * 3 + 2)
    print(header, flush=True)
    print(sep, flush=True)

    for k in args.qubits:
        print(f"\n[K={k}] featurizing {args.datasets}...", flush=True)

        # (a) Featurize
        QF0, AT, AR, Y, SCAF = featurize(k, args.datasets)
        N_total = len(Y)
        print(f"[K={k}] total molecules: {N_total}", flush=True)

        # Subsample to n_samples
        if N_total > args.n_samples:
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(N_total, args.n_samples, replace=False)
            QF0 = QF0[idx]
            AT = AT[idx]
        N = QF0.shape[0]

        # (b) Standardize QF on a random train split
        n_train = max(2, int(args.train_frac * N))
        perm = np.random.permutation(N)
        tr_idx = perm[:n_train]
        QF = standardize(QF0, tr_idx)

        # (c) Create GraphG with graph entangler + graph readout (random weights)
        model = GraphG(k, entangler='graph', readout='graph')

        # (d) eval mode
        model.eval()

        P = model.P
        print(f"[K={k}] N={N}, P={P} pairs, extracting features "
              f"(batch={args.batch_size})...", flush=True)

        # (e, f) Extract Phi_sq (N,3K) and Phi_bp (N,2K)
        QFt = torch.tensor(QF)
        At = torch.tensor(AT)
        Phi_sq, Phi_bp = extract_features(model, QFt, At, batch_size=args.batch_size)

        n_sq = Phi_sq.shape[1]   # 3K
        n_bp = Phi_bp.shape[1]   # 2K
        n_feats = n_sq + n_bp    # 5K

        # (g) Covariance of the full feature matrix Phi = [Phi_sq | Phi_bp]
        Phi = torch.cat([Phi_sq, Phi_bp], dim=-1).float()  # (N, 5K)

        # (h) Eigenvalues sorted descending
        eigs = covariance_eigs(Phi)

        # (i) Effective rank
        eff_rank = effective_rank(eigs)

        # (j) Variance fraction in each block (trace of block covariance)
        Phi_c = Phi - Phi.mean(dim=0)
        var_sq = float((Phi_c[:, :n_sq] ** 2).mean().item()) * n_sq
        var_bp = float((Phi_c[:, n_sq:] ** 2).mean().item()) * n_bp
        total_var = var_sq + var_bp + 1e-15
        frac_sq = var_sq / total_var
        frac_bp = var_bp / total_var

        top3 = [float(v) for v in eigs[:3].tolist()]
        top3_str = '[' + ', '.join(f'{v:.4f}' for v in top3) + ']'

        print(f"{k:>{col_w[0]}} | {n_feats:>{col_w[1]}} | "
              f"{eff_rank:>{col_w[2]}.3f} | {frac_sq:>{col_w[3]}.4f} | "
              f"{frac_bp:>{col_w[4]}.4f} | {top3_str}", flush=True)

        row = dict(
            k=k,
            n_feats=n_feats,
            n_sq=n_sq,
            n_bp=n_bp,
            n_pairs=P,
            n_molecules=N,
            eff_rank=eff_rank,
            frac_SQ=frac_sq,
            frac_BP=frac_bp,
            top3_eigs=top3,
            all_eigs=[float(v) for v in eigs.tolist()],
        )
        rows.append(row)

    # Summary table
    print(f"\n{'='*60}", flush=True)
    print("OPERATOR-GEOMETRY SUMMARY (T2 validation: eff_rank ~ K)", flush=True)
    print(f"{'='*60}", flush=True)
    print(header, flush=True)
    print(sep, flush=True)
    for r in rows:
        top3_str = '[' + ', '.join(f'{v:.4f}' for v in r['top3_eigs']) + ']'
        print(f"{r['k']:>{col_w[0]}} | {r['n_feats']:>{col_w[1]}} | "
              f"{r['eff_rank']:>{col_w[2]}.3f} | {r['frac_SQ']:>{col_w[3]}.4f} | "
              f"{r['frac_BP']:>{col_w[4]}.4f} | {top3_str}", flush=True)

    # Check T2: eff_rank should grow with K
    if len(rows) >= 2:
        print("\nT2 check (eff_rank vs K):", flush=True)
        for r in rows:
            ratio = r['eff_rank'] / r['k'] if r['k'] else float('nan')
            print(f"  K={r['k']}: eff_rank={r['eff_rank']:.3f}  eff_rank/K={ratio:.3f}",
                  flush=True)

    # Save results
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nSaved {len(rows)} rows -> {args.out}", flush=True)


if __name__ == '__main__':
    main()
