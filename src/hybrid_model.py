
import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F
from .classical_gnn import ClassicalGNN
from .quantum_layers import QuantumLayer

class HybridGNNVQC(nn.Module):
    def __init__(self, n_qubits=4, q_layers=2, reduction='linear', ansatz='strong', gnn_type='gine', dropout=0.2):
        super().__init__()
        self.reduction_type = reduction
        self.n_qubits = n_qubits

        # 1. Classical Encoder
        self.gnn = ClassicalGNN(gnn_type=gnn_type, dropout=dropout)

        # 2. Dimensionality Reduction
        if reduction == 'linear':
            self.projection = nn.Sequential(
                nn.Linear(self.gnn.hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, n_qubits)
            )
        elif reduction == 'fft':
            pass
        else:
            raise ValueError("Unknown reduction type")

        # 3. Quantum Layer
        self.vqc = QuantumLayer(n_qubits, n_layers=q_layers, ansatz=ansatz)

        # 4. Final Classification
        self.final_head = nn.Linear(n_qubits, 1)

    def forward(self, data):
        # 1. Encode Graph
        graph_emb = self.gnn.forward_features(data) # (batch, 64)

        # 2. Reduce Dimension
        if self.reduction_type == 'linear':
            latent = self.projection(graph_emb) # (batch, n_qubits)
        elif self.reduction_type == 'fft':
            f = torch.fft.rfft(graph_emb, dim=1)
            m = f.abs()
            if m.shape[1] < self.n_qubits:
                latent = F.pad(m, (0, self.n_qubits - m.shape[1]))
            else:
                latent = m[:, :self.n_qubits]

            latent = torch.atan(latent)

        # 3. Quantum Processing
        q_out = self.vqc(latent) # (batch, n_qubits)
        q_out = q_out.float()

        # 4. Final Prediction
        logits = self.final_head(q_out)
        return logits.squeeze(-1)
