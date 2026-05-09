import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import scipy.linalg

def extract_spectral_features(smiles, top_k=5):
    """
    Extracts the top-k non-zero eigenvalues and corresponding eigenvectors
    of the Normalized Graph Laplacian.
    Returns:
      vals: 1D tensor of size top_k (eigenvalues)
      vecs: 2D tensor of size (top_k, top_k) (eigenvectors reduced/padded)
    """
    m = Chem.MolFromSmiles(smiles)
    if m is None or m.GetNumAtoms() == 0:
        return torch.zeros(top_k), torch.zeros(top_k, top_k)

    # Create NetworkX graph
    G = nx.Graph()
    for i in range(m.GetNumAtoms()):
        G.add_node(i)
    for bond in m.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    if G.number_of_nodes() <= 1:
        return torch.zeros(top_k), torch.zeros(top_k, top_k)

    # Normalized Laplacian
    L = nx.normalized_laplacian_matrix(G).toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(L)

    # Filter out zero eigenvalues (with some tolerance)
    valid_idx = np.where(eigenvalues > 1e-5)[0]

    # Sort descending
    sorted_valid_idx = valid_idx[np.argsort(eigenvalues[valid_idx])[::-1]]

    non_zero_eigs = eigenvalues[sorted_valid_idx]
    non_zero_vecs = eigenvectors[:, sorted_valid_idx]

    # Pad or truncate to top_k
    n_eigs = min(top_k, len(non_zero_eigs))

    feat_vals = np.zeros(top_k)
    feat_vals[:n_eigs] = non_zero_eigs[:n_eigs]

    # For eigenvectors, we want a (top_k, top_k) matrix to match motif dimension
    # We take the top_k rows (nodes) and top_k cols (eigenvectors).
    # If graph has fewer than top_k nodes, we pad with 0.
    feat_vecs = np.zeros((top_k, top_k))
    n_nodes = min(top_k, non_zero_vecs.shape[0])
    feat_vecs[:n_nodes, :n_eigs] = non_zero_vecs[:n_nodes, :n_eigs]

    return torch.tensor(feat_vals, dtype=torch.float32), torch.tensor(feat_vecs, dtype=torch.float32)

from rdkit.Chem import MACCSkeys

def extract_all_maccs(smiles):
    """
    Helper function to extract all 167 MACCS keys for correlation analysis.
    """
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return np.zeros(167, dtype=np.float32)
    fp = MACCSkeys.GenMACCSKeys(m)
    arr = np.zeros((167,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)

def extract_motif_features(smiles, selected_indices):
    """
    Extracts MACCS Keys and keeps only the selected indices.
    """
    maccs_full = extract_all_maccs(smiles)
    if selected_indices is not None and len(selected_indices) > 0:
        feat = maccs_full[selected_indices]
    else:
        feat = np.zeros(5, dtype=np.float32)
    return torch.tensor(feat, dtype=torch.float32)

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

def extract_all_heterogeneous_features(smiles_list, motif_selected_indices=None):
    """
    Convenience function to extract S, M, D features for a list of SMILES.
    Returns four tensors: X_S_vals, X_S_vecs, X_M, X_D.
    """
    X_S_vals, X_S_vecs, X_M, X_D = [], [], [], []
    for s in smiles_list:
        v, e = extract_spectral_features(s)
        X_S_vals.append(v)
        X_S_vecs.append(e)
        X_M.append(extract_motif_features(s, motif_selected_indices))
        X_D.append(extract_diffusion_features(s))

    return torch.stack(X_S_vals), torch.stack(X_S_vecs), torch.stack(X_M), torch.stack(X_D)
