import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

from .features.semantic_extractor import SemanticFeatureExtractor
from .quantum_layers import QuantumLayer

class Level1Classical(nn.Module):
    """
    Level 1 Classical: "Three MLPs + Attention"
    Features are routed to independent MLP models and aggregated.
    """
    def __init__(self, hidden_dim=64, out_dim=12, dropout=0.2, inner_dim=32):
        super().__init__()
        self.extractor = SemanticFeatureExtractor(hidden_dim=hidden_dim, dropout=dropout)

        # Three independent Classical MLPs replacing the quantum circuits
        self.mlp_motif = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, out_dim)
        )
        self.mlp_cycle = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, out_dim)
        )
        self.mlp_spectral = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, out_dim)
        )

        # Attention aggregation
        self.attn = nn.Sequential(
            nn.Linear(out_dim, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, data):
        m, c, s = self.extractor(data) # (B, hidden_dim)

        out_m = self.mlp_motif(m)       # (B, out_dim)
        out_c = self.mlp_cycle(c)       # (B, out_dim)
        out_s = self.mlp_spectral(s)    # (B, out_dim)

        # Stack for attention
        stacked = torch.stack([out_m, out_c, out_s], dim=1) # (B, 3, out_dim)

        # Calculate attention weights
        attn_weights = self.attn(stacked) # (B, 3, 1)

        # Aggregate
        out = torch.sum(stacked * attn_weights, dim=1) # (B, out_dim)
        return out


class Level1Quantum(nn.Module):
    """
    Level 1 Quantum: "Three Circuits + Attention"
    Features are routed to independent Quantum models and aggregated via classical attention.
    """
    def __init__(self, hidden_dim=64, n_qubits=4, q_layers=2, out_dim=12, dropout=0.2):
        super().__init__()
        self.extractor = SemanticFeatureExtractor(hidden_dim=hidden_dim, dropout=dropout)

        # Projection from hidden_dim to n_qubits
        self.proj_motif = nn.Linear(hidden_dim, n_qubits)
        self.proj_cycle = nn.Linear(hidden_dim, n_qubits)
        self.proj_spectral = nn.Linear(hidden_dim, n_qubits)

        # Three independent VQCs
        self.q_motif = QuantumLayer(n_qubits, n_layers=q_layers, ansatz='strong')
        self.q_cycle = QuantumLayer(n_qubits, n_layers=q_layers, ansatz='strong')
        self.q_spectral = QuantumLayer(n_qubits, n_layers=q_layers, ansatz='strong')

        # Output heads
        self.head_motif = nn.Linear(n_qubits, out_dim)
        self.head_cycle = nn.Linear(n_qubits, out_dim)
        self.head_spectral = nn.Linear(n_qubits, out_dim)

        # Attention aggregation
        self.attn = nn.Sequential(
            nn.Linear(out_dim, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, data):
        m, c, s = self.extractor(data)

        # Project
        m_q = self.proj_motif(m)
        c_q = self.proj_cycle(c)
        s_q = self.proj_spectral(s)

        # Quantum Processing
        out_m = self.head_motif(self.q_motif(m_q))
        out_c = self.head_cycle(self.q_cycle(c_q))
        out_s = self.head_spectral(self.q_spectral(s_q))

        # Stack for attention
        stacked = torch.stack([out_m, out_c, out_s], dim=1)

        # Calculate attention weights
        attn_weights = self.attn(stacked)

        # Aggregate
        out = torch.sum(stacked * attn_weights, dim=1)
        return out

class Level2Classical(nn.Module):
    """
    Level 2 Classical: "Chemical-to-Operator Correspondence Equivalent"
    We try to simulate the specific algebraic processing of features using MLPs.
    However, the classical MLPs lack the strict geometric inductive bias of quantum operators.
    """
    def __init__(self, hidden_dim=64, out_dim=12, dropout=0.2, inner_dim=32):
        super().__init__()
        self.extractor = SemanticFeatureExtractor(hidden_dim=hidden_dim, dropout=dropout)

        # Motif -> Local processing (Simulated by depth-wise / independent scalar processing)
        self.mlp_motif = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, inner_dim)
        )

        # Cycle -> Phase/Periodic processing (Simulated by Sine/Cosine activations)
        self.mlp_cycle_proj = nn.Linear(hidden_dim, inner_dim)
        self.mlp_cycle_out = nn.Linear(inner_dim, inner_dim)

        # Spectral -> Mixing/Interaction (Simulated by dense cross-attention like mixing)
        self.mlp_spectral = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, inner_dim)
        )

        # Aggregation
        self.agg = nn.Linear(inner_dim * 3, out_dim)

    def forward(self, data):
        m, c, s = self.extractor(data)

        out_m = self.mlp_motif(m)

        # Periodic activation for cycle
        out_c_proj = self.mlp_cycle_proj(c)
        out_c = self.mlp_cycle_out(torch.sin(out_c_proj) + torch.cos(out_c_proj))

        out_s = self.mlp_spectral(s)

        # Concatenate and project
        concat = torch.cat([out_m, out_c, out_s], dim=1)
        return self.agg(concat)


class Level2QuantumLayer(nn.Module):
    """
    Level 2 Quantum Layer: Maps specific features to specific quantum operator families.
    - motifs -> R_y (Local observables)
    - cycles -> R_z (Phase operators)
    - spectral -> Ising/XY Entanglement (Interaction Hamiltonians)
    """
    def __init__(self, n_qubits=4, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch")
        def circuit(m_inputs, c_inputs, s_inputs, weights):
            # weights shape: (n_layers, n_qubits, 3)
            for l in range(n_layers):
                for i in range(n_qubits):
                    # 1. Motifs -> R_y (Local rotations)
                    qml.RY(m_inputs[i] * weights[l, i, 0], wires=i)

                    # 2. Cycles -> R_z (Phase accumulations)
                    qml.RZ(c_inputs[i] * weights[l, i, 1], wires=i)

                # 3. Spectral -> Interaction Hamiltonians (Ising ZZ or XY coupling)
                # We use the spectral features to scale the entanglement.
                # For a fixed geometry, we just use a chain or ring.
                for i in range(n_qubits - 1):
                    # Ising XX coupling controlled by spectral input + weight
                    coupling_strength = s_inputs[i] * weights[l, i, 2]
                    qml.IsingXX(coupling_strength, wires=[i, i+1])
                # Close the ring
                qml.IsingXX(s_inputs[-1] * weights[l, -1, 2], wires=[n_qubits-1, 0])

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qnode = circuit
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3))

    def forward(self, m, c, s):
        m = torch.atan(m)
        c = torch.atan(c)
        s = torch.atan(s)

        # Process batch by batch
        batch_size = m.shape[0]
        res = []
        for b in range(batch_size):
            out = self.qnode(m[b], c[b], s[b], self.weights)
            res.append(torch.stack(out))
        return torch.stack(res, dim=0).float()

class Level2Quantum(nn.Module):
    def __init__(self, hidden_dim=64, n_qubits=4, q_layers=2, out_dim=12, dropout=0.2):
        super().__init__()
        self.extractor = SemanticFeatureExtractor(hidden_dim=hidden_dim, dropout=dropout)

        self.proj_motif = nn.Linear(hidden_dim, n_qubits)
        self.proj_cycle = nn.Linear(hidden_dim, n_qubits)
        self.proj_spectral = nn.Linear(hidden_dim, n_qubits)

        self.q_layer = Level2QuantumLayer(n_qubits=n_qubits, n_layers=q_layers)

        self.head = nn.Linear(n_qubits, out_dim)

    def forward(self, data):
        m, c, s = self.extractor(data)

        m_q = self.proj_motif(m)
        c_q = self.proj_cycle(c)
        s_q = self.proj_spectral(s)

        q_out = self.q_layer(m_q, c_q, s_q)
        return self.head(q_out)

class Level3Classical(nn.Module):
    """
    Level 3 Classical: "Feature-wise Linear Modulation (FiLM)"
    Instead of passing features to independent streams, features actively modulate
    the weights/activations of the networks processing other features.
    """
    def __init__(self, hidden_dim=64, out_dim=12, dropout=0.2, inner_dim=32):
        super().__init__()
        self.extractor = SemanticFeatureExtractor(hidden_dim=hidden_dim, dropout=dropout)

        # Motif processing network
        self.motif_net = nn.Linear(hidden_dim, inner_dim)

        # Modulators
        # Motif modulates Cycle phase
        self.motif_to_cycle_mod = nn.Linear(hidden_dim, inner_dim)
        # Spectral modulates Cycle interaction
        self.spectral_to_cycle_mod = nn.Linear(hidden_dim, inner_dim)

        self.cycle_net_1 = nn.Linear(hidden_dim, inner_dim)
        self.cycle_net_2 = nn.Linear(inner_dim, inner_dim)

        self.agg = nn.Linear(inner_dim * 2, out_dim)

    def forward(self, data):
        m, c, s = self.extractor(data)

        out_m = F.relu(self.motif_net(m))

        # Motif modulates Cycle (Shift)
        shift_m = self.motif_to_cycle_mod(m)

        # Spectral modulates Cycle (Scale)
        scale_s = torch.sigmoid(self.spectral_to_cycle_mod(s))

        # Cycle Processing modulated by M and S
        c_1 = self.cycle_net_1(c)
        c_mod = (c_1 * scale_s) + shift_m
        out_c = F.relu(self.cycle_net_2(c_mod))

        # Concatenate and project
        concat = torch.cat([out_m, out_c], dim=1)
        return self.agg(concat)


class Level3QuantumLayer(nn.Module):
    """
    Level 3 Quantum Layer: "Chemical Operator Geometry"
    Features directly alter the structure/geometry of operators.
    - Motifs modulate phase accumulation of cycles: R_z(c + alpha * m)
    - Spectral modulates entanglement geometry: e^(-i * s * Z_i Z_j)
    """
    def __init__(self, n_qubits=4, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch")
        def circuit(m_inputs, c_inputs, s_inputs, weights, alpha):
            # weights shape: (n_layers, n_qubits, 2)
            for l in range(n_layers):
                for i in range(n_qubits):
                    # Local Motif processing
                    qml.RY(m_inputs[i] * weights[l, i, 0], wires=i)

                    # Motif modulates Phase Accumulation of Cycle! (Cross-modality interaction)
                    # This represents geometric dependency.
                    phase = (c_inputs[i] * weights[l, i, 1]) + (alpha[i] * m_inputs[i])
                    qml.RZ(phase, wires=i)

                # Spectral dictates Entanglement Geometry directly!
                for i in range(n_qubits - 1):
                    # Here s_inputs directly controls the strength of the XX coupling
                    # without an independent learned parameter (or with a global one).
                    # The chemistry (spectral mode) defines the interaction topology.
                    qml.IsingXX(s_inputs[i], wires=[i, i+1])
                qml.IsingXX(s_inputs[-1], wires=[n_qubits-1, 0])

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qnode = circuit
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 2))
        self.alpha = nn.Parameter(torch.randn(n_qubits)) # Modulation scale

    def forward(self, m, c, s):
        m = torch.atan(m)
        c = torch.atan(c)
        s = torch.atan(s)

        # Process batch by batch
        batch_size = m.shape[0]
        res = []
        for b in range(batch_size):
            out = self.qnode(m[b], c[b], s[b], self.weights, self.alpha)
            res.append(torch.stack(out))
        return torch.stack(res, dim=0).float()

class Level3Quantum(nn.Module):
    def __init__(self, hidden_dim=64, n_qubits=4, q_layers=2, out_dim=12, dropout=0.2):
        super().__init__()
        self.extractor = SemanticFeatureExtractor(hidden_dim=hidden_dim, dropout=dropout)

        self.proj_motif = nn.Linear(hidden_dim, n_qubits)
        self.proj_cycle = nn.Linear(hidden_dim, n_qubits)
        self.proj_spectral = nn.Linear(hidden_dim, n_qubits)

        self.q_layer = Level3QuantumLayer(n_qubits=n_qubits, n_layers=q_layers)

        self.head = nn.Linear(n_qubits, out_dim)

    def forward(self, data):
        m, c, s = self.extractor(data)

        m_q = self.proj_motif(m)
        c_q = self.proj_cycle(c)
        s_q = self.proj_spectral(s)

        q_out = self.q_layer(m_q, c_q, s_q)
        return self.head(q_out)
