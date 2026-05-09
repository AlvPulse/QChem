import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import scipy.linalg

def extract_spectral_features(smiles, top_k=5):
    """
    Extracts the top-k non-zero eigenvalues of the Normalized Graph Laplacian.
    Returns a 1D tensor of size top_k.
    """
    m = Chem.MolFromSmiles(smiles)
    if m is None or m.GetNumAtoms() == 0:
        return torch.zeros(top_k)

    # Create NetworkX graph
    G = nx.Graph()
    for i in range(m.GetNumAtoms()):
        G.add_node(i)
    for bond in m.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    if G.number_of_nodes() <= 1:
        return torch.zeros(top_k)

    # Normalized Laplacian
    L = nx.normalized_laplacian_matrix(G).toarray()
    eigenvalues = np.linalg.eigvalsh(L)

    # Filter out zero eigenvalues (with some tolerance)
    non_zero_eigs = eigenvalues[eigenvalues > 1e-5]

    # Sort descending
    non_zero_eigs = np.sort(non_zero_eigs)[::-1]

    # Pad or truncate to top_k
    feat = np.zeros(top_k)
    n_eigs = min(top_k, len(non_zero_eigs))
    feat[:n_eigs] = non_zero_eigs[:n_eigs]

    return torch.tensor(feat, dtype=torch.float32)

def extract_motif_features(smiles, n_bits=128):
    """
    Extracts Morgan Fingerprint (Radius 2).
    Returns a 1D tensor of size n_bits.
    """
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return torch.zeros(n_bits)

    fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)

    return torch.tensor(arr, dtype=torch.float32)

def extract_diffusion_features(smiles, top_k=5, beta=1.0):
    """
    Extracts the top-k heat kernel diagonal values (e^{-beta * L}).
    Returns a 1D tensor of size top_k.
    """
    m = Chem.MolFromSmiles(smiles)
    if m is None or m.GetNumAtoms() == 0:
        return torch.zeros(top_k)

    G = nx.Graph()
    for i in range(m.GetNumAtoms()):
        G.add_node(i)
    for bond in m.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    if G.number_of_nodes() <= 1:
        return torch.zeros(top_k)

    L = nx.normalized_laplacian_matrix(G).toarray()

    # Heat Kernel H = exp(-beta * L)
    # Since L is symmetric, we can use eigendecomposition
    vals, vecs = np.linalg.eigh(L)
    heat_vals = np.exp(-beta * vals)
    H = vecs @ np.diag(heat_vals) @ vecs.T

    # Get diagonal elements (auto-diffusion)
    diag_H = np.diag(H)

    # Sort descending
    diag_H = np.sort(diag_H)[::-1]

    # Pad or truncate to top_k
    feat = np.zeros(top_k)
    n_diag = min(top_k, len(diag_H))
    feat[:n_diag] = diag_H[:n_diag]

    return torch.tensor(feat, dtype=torch.float32)

def extract_all_heterogeneous_features(smiles_list):
    """
    Convenience function to extract S, M, D features for a list of SMILES.
    Returns three tensors: X_S, X_M, X_D.
    """
    X_S, X_M, X_D = [], [], []
    for s in smiles_list:
        X_S.append(extract_spectral_features(s))
        X_M.append(extract_motif_features(s))
        X_D.append(extract_diffusion_features(s))

    return torch.stack(X_S), torch.stack(X_M), torch.stack(X_D)
