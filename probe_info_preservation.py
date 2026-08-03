"""E4: Epsilon-Sufficiency / Information Preservation Probe.

Tests T7 Thm 3.8 TL(G) assumption: how often does spectral coarsening
preserve toxicophore substructures in a single cluster?

Run: python probe_info_preservation.py --qubits 4 6 8 --n_mols 500 --out results/e4_preservation.json
"""
import argparse, json, os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.cluster import SpectralClustering

from run_bias_probe import build_merged_dataframe, N_TASKS

TOXICOPHORE_SMARTS = {
    'aromatic_ring': 'c1ccccc1',
    'nitro':         '[N+](=O)[O-]',
    'halogen_aro':   'cF,cCl,cBr',
    'amine_aro':     'cN',
    'carbonyl':      'C=O',
    'amide':         'C(=O)N',
    'epoxide':       'C1OC1',
    'hydroxyl_aro':  'cO',
}
PATTERNS = {}
for k, v in TOXICOPHORE_SMARTS.items():
    p = Chem.MolFromSmarts(v)
    if p is not None:
        PATTERNS[k] = p


def cluster_labels(smiles, n_clusters):
    """Return per-atom cluster assignment (n_atoms,) for a SMILES at K clusters."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    n = mol.GetNumAtoms()
    A = np.zeros((n, n))
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        w = b.GetBondTypeAsDouble()
        A[i, j] = A[j, i] = w
    if n <= n_clusters:
        return np.arange(n) % n_clusters, mol
    try:
        labels = SpectralClustering(
            n_clusters=n_clusters, affinity='precomputed',
            assign_labels='discretize', random_state=0
        ).fit_predict(A + 1e-6)
    except Exception:
        labels = np.arange(n) % n_clusters
    return labels, mol


def preservation_rate(smiles_list, k):
    """Fraction of toxicophore matches that fall in a single cluster."""
    results = {name: {'intact': 0, 'total': 0} for name in PATTERNS}
    for smi in smiles_list:
        labels, mol = cluster_labels(smi, k)
        if labels is None:
            continue
        for name, pat in PATTERNS.items():
            for match in mol.GetSubstructMatches(pat):
                results[name]['total'] += 1
                if len(set(labels[i] for i in match if i < len(labels))) == 1:
                    results[name]['intact'] += 1
    out = {}
    for name, d in results.items():
        total = d['total']
        rate = d['intact'] / total if total > 0 else float('nan')
        out[name] = {'rate': round(rate, 4), 'count': total}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qubits', type=int, nargs='+', default=[4, 6, 8])
    ap.add_argument('--datasets', type=str, nargs='+', default=['Tox21', 'ToxCast'])
    ap.add_argument('--n_mols', type=int, default=500)
    ap.add_argument('--out', default='results/e4_preservation.json')
    args = ap.parse_args()

    print(f"E4 info-preservation probe | n_mols={args.n_mols} | datasets={args.datasets}")

    # Get SMILES
    df, _ = build_merged_dataframe('.', datasets=tuple(args.datasets))
    smiles = []
    for smi in df['smiles']:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            Chem.Kekulize(mol, clearAromaticFlags=False)
        except Exception:
            continue
        smiles.append(smi)
    smiles = smiles[:args.n_mols]
    print(f"  Using {len(smiles)} molecules.")

    rows = []
    for k in args.qubits:
        print(f"\n[K={k}] Substructure preservation rates ({len(smiles)} mols):")
        pres = preservation_rate(smiles, k)
        rates = [v['rate'] for v in pres.values() if not np.isnan(v['rate'])]
        mean_rate = float(np.mean(rates)) if rates else float('nan')
        for name, v in pres.items():
            print(f"  {name:20s}: {v['rate']:.3f} ({v['count']} matches)")
        print(f"  K={k} mean preservation: {mean_rate:.3f}")
        rows.append(dict(k=k, mean_preservation=mean_rate, by_toxicophore=pres))

    print("\n" + "=" * 50)
    print(f"{'K':>4} | {'mean_preservation':>18}")
    print("-" * 28)
    for r in rows:
        print(f"{r['k']:>4} | {r['mean_preservation']:>18.3f}")
    print()
    print("T7 Thm 3.8 TL(G) check: mean_preservation > 0.7 at K=8 -> TL(G) holds (conditionally).")

    with open(args.out, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f"Saved -> {args.out}")


if __name__ == '__main__':
    main()
