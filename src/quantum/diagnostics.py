import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def compute_gram_matrix(q_features):
    """
    Computes the classical Gram matrix K(x, x') = <phi(x), phi(x')> 
    from the quantum feature vectors.
    """
    # Normalize features to unit vectors
    norms = torch.norm(q_features, p=2, dim=1, keepdim=True)
    q_features_norm = q_features / (norms + 1e-8)
    
    # Compute dot products
    gram_matrix = torch.mm(q_features_norm, q_features_norm.t())
    return gram_matrix.detach().cpu().numpy()

def analyze_kernel_variance(gram_matrix):
    """
    Diagnostic tool to check for Exponential Concentration (a relative to Barren Plateaus).
    If variance is extremely low, the kernel cannot distinguish between different molecules.
    """
    off_diagonals = gram_matrix[~np.eye(gram_matrix.shape[0], dtype=bool)]
    variance = np.var(off_diagonals)
    mean_val = np.mean(off_diagonals)
    
    print(f"--- Quantum Kernel Diagnostics ---")
    print(f"Mean Off-Diagonal Similarity: {mean_val:.4f} (Ideal: closer to 0)")
    print(f"Variance of Similarities: {variance:.6f} (Ideal: > 0.05)")
    
    if variance < 1e-4:
        print("WARNING: Exponential State Concentration detected. All molecules look the same to the quantum circuit.")
    elif mean_val > 0.95:
        print("WARNING: Kernel values are saturating. The state space is too narrow.")
    else:
        print("SUCCESS: Quantum kernel shows healthy variance and spread.")
        
    return mean_val, variance

def plot_gram_matrix(gram_matrix, save_path="gram_matrix.png"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(gram_matrix, cmap="viridis", vmin=-1, vmax=1)
    plt.title("Quantum Kernel Gram Matrix")
    plt.xlabel("Molecule Index")
    plt.ylabel("Molecule Index")
    plt.savefig(save_path)
    plt.close()
    print(f"Gram matrix heatmap saved to {save_path}")