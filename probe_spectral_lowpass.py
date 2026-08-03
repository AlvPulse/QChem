"""E3: spectral low-pass verification probe (T3 Lemma 3.5).

Decomposes coarse node features (qf) into the symmetric normalized Laplacian
eigenbasis of each molecule's K-qubit coarse graph and measures:
  frac_lowfreq -- fraction of feature energy in bottom-K//2 eigenmodes
  spectral_gap -- lambda[K//2] - lambda[K//2-1]  (cluster separability)
  disc_error   -- ||qf - Pi_{K//2} qf|| / ||qf|| (deviation from ideal low-pass)

Usage:
  python probe_spectral_lowpass.py --qubits 4 6 8 --n_mols 200 \
      --datasets Tox21 ToxCast --out results/e3_spectral.json
"""
import argparse, json, os
import numpy as np, torch
from run_bias_probe import featurize, FDIM


def spectral_stats(qf_np, at_np):
    """Spectral decomposition for one molecule's coarse graph.
    qf_np: (K, FDIM)  coarse node features
    at_np: (K, K)     normalized coarse adjacency
    """
    K = qf_np.shape[0]
    m = max(1, K // 2)
    qf = torch.tensor(qf_np, dtype=torch.float64)
    A  = torch.tensor(at_np, dtype=torch.float64)

    # Symmetric normalized Laplacian: L_sym = I - D^{-1/2} A D^{-1/2}
    d       = A.sum(dim=1).clamp(min=1e-9)
    D_isqrt = torch.diag(1.0 / d.sqrt())
    L_sym   = torch.eye(K, dtype=torch.float64) - D_isqrt @ A @ D_isqrt

    lam, U = torch.linalg.eigh(L_sym)   # ascending order, (K,), (K,K)

    qf_energy = (qf ** 2).sum().item()
    if qf_energy < 1e-12:
        nan = float('nan')
        return dict(frac_lowfreq=nan, spectral_gap=nan, disc_error=nan)

    # Energy fraction in bottom-m modes
    proj_m = U[:, :m] @ (U[:, :m].T @ qf)
    frac   = float((proj_m ** 2).sum().item() / qf_energy)

    # Spectral gap at midpoint (0-indexed: gap between mode m-1 and m)
    gap = float(lam[min(m, K - 1)].item() - lam[max(m - 1, 0)].item())

    # Discretization error: ||qf - proj_m|| / ||qf||
    err = float(((qf - proj_m) ** 2).sum().item() ** 0.5 / qf_energy ** 0.5)

    return dict(frac_lowfreq=frac, spectral_gap=gap, disc_error=err)


def run_k(k, datasets, n_mols):
    QF, AT, _, _, _ = featurize(k, datasets)
    n = min(n_mols, len(QF))
    fracs, gaps, errs = [], [], []
    for b in range(n):
        s = spectral_stats(QF[b], AT[b])
        if not any(np.isnan(v) for v in s.values()):
            fracs.append(s['frac_lowfreq'])
            gaps.append(s['spectral_gap'])
            errs.append(s['disc_error'])
    return dict(k=k, n_valid=len(fracs),
                mean_frac_lowfreq=float(np.mean(fracs)),
                median_frac_lowfreq=float(np.median(fracs)),
                mean_spectral_gap=float(np.mean(gaps)),
                mean_disc_error=float(np.mean(errs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qubits',   type=int, nargs='+', default=[4, 6, 8])
    ap.add_argument('--n_mols',   type=int,            default=200)
    ap.add_argument('--datasets', type=str, nargs='+', default=['Tox21', 'ToxCast'])
    ap.add_argument('--out',      type=str,            default='results/e3_spectral.json')
    args = ap.parse_args()

    print(f"E3 spectral low-pass | datasets={args.datasets} | n_mols={args.n_mols}", flush=True)
    hdr = f"{'K':>4} | {'n_valid':>7} | {'frac_low(K/2)':>13} | {'gap':>8} | {'disc_err':>9} | interp"
    print(hdr); print("-" * len(hdr), flush=True)

    results = []
    for k in args.qubits:
        r = run_k(k, args.datasets, args.n_mols)
        results.append(r)
        tag = "low-pass OK" if r['mean_frac_lowfreq'] > 0.5 else "NOT low-pass"
        print(f"{k:>4} | {r['n_valid']:>7} | {r['mean_frac_lowfreq']:>13.4f} | "
              f"{r['mean_spectral_gap']:>8.4f} | {r['mean_disc_error']:>9.4f} | {tag}", flush=True)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(dict(datasets=args.datasets, n_mols=args.n_mols, results=results), f, indent=2)
    print(f"\nSaved -> {args.out}", flush=True)


if __name__ == '__main__':
    main()
