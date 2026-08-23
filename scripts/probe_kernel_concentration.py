import torch
import pennylane as qml
import numpy as np
from run_levelG_probe import GraphG
from run_bias_probe import featurize

def compute_gram_matrix_variance(k=4, dataset='Tox21'):
    """
    Implements Phase E: Kernel Concentration check.
    If the variance of the off-diagonal elements is ~0, the kernel has concentrated
    (suffers from barren plateaus/exponential vanishing of distances).
    """
    QF, AT, AR, Y, SCAF = featurize(k, [dataset])
    model = GraphG(k, entangler='graph', readout='graph')

    # Take a small batch of 16 molecules to compute the Gram matrix
    B = min(16, len(QF))
    QFt, At = torch.tensor(QF[:B]), torch.tensor(AT[:B])

    # We use the final graph representation before the linear head as the "feature map"
    model.eval()
    with torch.no_grad():
        a = torch.atan(model.feat(QFt))
        out = model.circ(a[:, :, 0], a[:, :, 1], At, model.theta, model.ringp, model.pairp, model.enc)
        out = [o.float() for o in out]

        feats = [torch.stack(out[:3 * k], -1)]
        zz = torch.stack(out[3 * k:3 * k + model.P], -1)
        xx = torch.stack(out[3 * k + model.P:3 * k + 2 * model.P], -1)

        # Structure-weighted pooled features
        pooled_zz = model._bond_pool(zz, At)
        pooled_xx = model._bond_pool(xx, At)

        # State vector representations
        phi = torch.cat([feats[0].reshape(B, -1), pooled_zz, pooled_xx], dim=-1)

    # Normalize features to act as a proper kernel trace
    phi = torch.nn.functional.normalize(phi, p=2, dim=1)

    # Compute Gram Matrix K = phi * phi^T
    K = torch.matmul(phi, phi.T).numpy()

    # Extract off-diagonal elements
    off_diagonals = K[np.triu_indices_from(K, k=1)]
    variance = np.var(off_diagonals)
    mean = np.mean(off_diagonals)

    print(f"Kernel Concentration Diagnostics (Level G, K={k}):")
    print(f"Mean of off-diagonals: {mean:.4f}")
    print(f"Variance of off-diagonals: {variance:.6f}")

    if variance < 0.05:
        print("WARNING: Kernel is highly concentrated (Variance < 0.05). State space might be too narrow.")
    else:
        print("SUCCESS: Kernel avoids concentration. The bond-pooled readout successfully preserves feature diversity.")

if __name__ == '__main__':
    compute_gram_matrix_variance(k=4, dataset='Tox21')
