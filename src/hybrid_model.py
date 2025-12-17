
import torch
import torch.nn as nn
import torch.fft
from .classical_gnn import ClassicalGNN
from .quantum_layers import QuantumLayer

class HybridGNNVQC(nn.Module):
    def __init__(self, n_qubits=4, q_layers=2, reduction='linear', ansatz='strong'):
        super().__init__()
        self.reduction_type = reduction

        # 1. Classical Encoder
        self.gnn = ClassicalGNN()

        # 2. Dimensionality Reduction
        if reduction == 'linear':
            self.projection = nn.Sequential(
                nn.Linear(self.gnn.hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, n_qubits)
            )
        elif reduction == 'fft':
            # Parameter-free reduction: We will just truncate the FFT output
            # No projection layer needed here, but we might need to align dimensions
            # GNN out is 64. We will pick top-k freq.
            pass
        else:
            raise ValueError("Unknown reduction type")

        # 3. Quantum Layer
        self.vqc = QuantumLayer(n_qubits, n_layers=q_layers, ansatz=ansatz)

        # 4. Final Classification
        self.final_head = nn.Linear(n_qubits, 1)
        self.n_qubits = n_qubits

    def forward(self, data):
        # 1. Encode Graph
        graph_emb = self.gnn.forward_features(data) # (batch, 64)

        # 2. Reduce Dimension
        if self.reduction_type == 'linear':
            latent = self.projection(graph_emb) # (batch, n_qubits)
        elif self.reduction_type == 'fft':
            # Real FFT
            # graph_emb: (batch, 64)
            # rfft output dim: 64//2 + 1 = 33 complex numbers
            f = torch.fft.rfft(graph_emb, dim=1)
            # Take magnitudes
            m = f.abs()
            # Select low frequencies (first n_qubits)
            # Ensure we have enough dims
            if m.shape[1] < self.n_qubits:
                # Pad if somehow needed (unlikely with 64 dim)
                latent = F.pad(m, (0, self.n_qubits - m.shape[1]))
            else:
                latent = m[:, :self.n_qubits]

            # Normalize to avoid huge angles
            latent = torch.atan(latent)

        # 3. Quantum Processing
        q_out = self.vqc(latent) # (batch, n_qubits)
        q_out = q_out.float() # Ensure float for downstream

        # 4. Final Prediction
        logits = self.final_head(q_out)
        return logits.squeeze(-1)
