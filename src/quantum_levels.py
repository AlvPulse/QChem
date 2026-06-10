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
        m, c, s, _, desc_preds = self.extractor(data) # (B, hidden_dim)

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
        m, c, s, _, desc_preds = self.extractor(data)

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
        m, c, s, _, desc_preds = self.extractor(data)

        out_m = self.mlp_motif(m)

        # Periodic activation for cycle
        out_c_proj = self.mlp_cycle_proj(c)
        out_c = self.mlp_cycle_out(torch.sin(out_c_proj) + torch.cos(out_c_proj))

        out_s = self.mlp_spectral(s)

        # Concatenate and project
        concat = torch.cat([out_m, out_c, out_s], dim=1)
        logits = self.agg(concat)
        return logits, concat, desc_preds


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
        m, c, s, _, desc_preds = self.extractor(data)

        m_q = self.proj_motif(m)
        c_q = self.proj_cycle(c)
        s_q = self.proj_spectral(s)

        q_out = self.q_layer(m_q, c_q, s_q)
        logits = self.head(q_out)
        return logits, q_out, desc_preds

class Level3Classical(nn.Module):
    """
    Level 3 Classical: "Feature-wise Linear Modulation (FiLM)"
    Instead of passing features to independent streams, features actively modulate
    the weights/activations of the networks processing other features.
    Added LayerNorm and Residual connections to prevent catastrophic collapse.
    """
    def __init__(self, hidden_dim=64, out_dim=12, dropout=0.2, inner_dim=32):
        super().__init__()
        self.extractor = SemanticFeatureExtractor(hidden_dim=hidden_dim, dropout=dropout)

        # Motif processing network
        self.motif_net = nn.Linear(hidden_dim, inner_dim)
        self.norm_m = nn.LayerNorm(inner_dim)

        # Modulators
        # Motif modulates Cycle phase
        self.motif_to_cycle_mod = nn.Linear(hidden_dim, inner_dim)
        # Spectral modulates Cycle interaction
        self.spectral_to_cycle_mod = nn.Linear(hidden_dim, inner_dim)

        self.cycle_net_1 = nn.Linear(hidden_dim, inner_dim)
        self.norm_c1 = nn.LayerNorm(inner_dim)
        self.cycle_net_2 = nn.Linear(inner_dim, inner_dim)
        self.norm_c2 = nn.LayerNorm(inner_dim)

        self.agg = nn.Linear(inner_dim * 2, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        m, c, s, _, desc_preds = self.extractor(data)

        # Motif processing with LayerNorm
        out_m = F.relu(self.norm_m(self.motif_net(m)))

        # Motif modulates Cycle (Shift)
        shift_m = self.motif_to_cycle_mod(m)

        # Spectral modulates Cycle (Scale)
        # Use tanh to allow both positive/negative scaling, but limit exploding gradients
        scale_s = torch.tanh(self.spectral_to_cycle_mod(s))

        # Cycle Processing modulated by M and S with LayerNorm and Residual connection
        c_1 = self.norm_c1(self.cycle_net_1(c))
        c_mod = (c_1 * scale_s) + shift_m

        # Residual connection over modulation
        c_mod_res = c_1 + c_mod

        out_c = F.relu(self.norm_c2(self.cycle_net_2(c_mod_res)))

        # Concatenate and project with Dropout
        concat = self.dropout(torch.cat([out_m, out_c], dim=1))
        logits = self.agg(concat)
        return logits, concat, desc_preds


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
        m, c, s, _, desc_preds = self.extractor(data)

        m_q = self.proj_motif(m)
        c_q = self.proj_cycle(c)
        s_q = self.proj_spectral(s)

        q_out = self.q_layer(m_q, c_q, s_q)
        logits = self.head(q_out)
        return logits, q_out, desc_preds

from torch_geometric.nn import AttentionalAggregation

class Level4Classical(nn.Module):
    """
    Level 4 Classical: "3D Spatial Entanglement Equivalent"
    Mimics quantum distance-based interactions using dot-product cross-modality.
    """
    def __init__(self, hidden_dim=64, out_dim=12, dropout=0.2, inner_dim=32):
        super().__init__()
        self.extractor = SemanticFeatureExtractor(hidden_dim=hidden_dim, dropout=dropout)

        # Linear projection for continuous Euclidean distances
        self.dist_proj = nn.Linear(1, inner_dim)
        self.spatial_pool = AttentionalAggregation(nn.Sequential(nn.Linear(inner_dim, 1)))

        self.walk_net = nn.Sequential(
            nn.Linear(hidden_dim + inner_dim, inner_dim),
            nn.LayerNorm(inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, inner_dim),
            nn.LayerNorm(inner_dim),
            nn.ReLU()
        )
        self.agg = nn.Linear(inner_dim, out_dim)

    def forward(self, data):
        _, _, s, chem, desc_preds = self.extractor(data)

        # Aggregate the continuous 3D edges for classical network
        if hasattr(data, 'edge_attr_cont'):
            edge_embs = self.dist_proj(data.edge_attr_cont)
            # Pool edge embeddings into a graph-level feature
            # We use data.batch for edges by mapping edge->src_node->batch
            edge_batch = data.batch[data.edge_index[0]] if hasattr(data, 'batch') else torch.zeros(edge_embs.size(0), dtype=torch.long, device=edge_embs.device)
            dist_rep = self.spatial_pool(edge_embs, edge_batch)
        else:
            dist_rep = torch.zeros(chem.size(0), self.dist_proj.out_features, device=chem.device)

        # Combine chemistry and distance representations
        out = self.walk_net(torch.cat([chem, dist_rep], dim=-1))
        logits = self.agg(out)
        return logits, out, desc_preds

class Level4QuantumLayer(nn.Module):
    """
    Level 4 Quantum Layer: "Quantum 3D Spatial Entanglement"
    Utilizes explicitly provided 3D Euclidean distances to modulate entanglement gates (CRZ),
    achieving a natively Quantum Geometric Message Passing layer.
    """
    def __init__(self, n_qubits=4, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch")
        def circuit(chem_inputs, dist_inputs, weights):
            for l in range(n_layers):
                # 1. State Preparation (Chemical features)
                for i in range(n_qubits):
                    qml.RY(chem_inputs[i] * weights[l, i, 0], wires=i)

                # 2. 3D Spatial Entanglement (Native Quantum Edge Processing)
                # We map the pooled Euclidean distance directly into the phase of the entangling gate.
                # In a true sparse QGNN, we'd map specific distances to specific qubit pairs.
                # Here, we use the graph-level distance representation to scale the ring topology.
                for i in range(n_qubits - 1):
                    # Inversely scale entanglement by distance (closer = stronger)
                    coupling = weights[l, i, 1] / (1.0 + dist_inputs[i])
                    qml.CRZ(coupling, wires=[i, i+1])
                qml.CRZ(weights[l, -1, 1] / (1.0 + dist_inputs[-1]), wires=[n_qubits-1, 0])

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qnode = circuit
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 2))

    def forward(self, chem, dist):
        chem = torch.atan(chem)
        # Distances are strictly positive, scale using log
        dist = torch.log1p(torch.abs(dist))

        batch_size = chem.shape[0]
        res = []
        for b in range(batch_size):
            out = self.qnode(chem[b], dist[b], self.weights)
            res.append(torch.stack(out))
        return torch.stack(res, dim=0).float()

class Level4Quantum(nn.Module):
    def __init__(self, hidden_dim=64, n_qubits=4, q_layers=2, out_dim=12, dropout=0.2):
        super().__init__()
        self.extractor = SemanticFeatureExtractor(hidden_dim=hidden_dim, dropout=dropout)

        self.proj_chem = nn.Linear(hidden_dim, n_qubits)

        # Spatial pooling mapping explicitly to qubits
        self.dist_proj = nn.Linear(1, n_qubits)
        self.spatial_pool = AttentionalAggregation(nn.Sequential(nn.Linear(n_qubits, 1)))

        self.q_layer = Level4QuantumLayer(n_qubits=n_qubits, n_layers=q_layers)
        self.head = nn.Linear(n_qubits, out_dim)

    def forward(self, data):
        _, _, _, chem, desc_preds = self.extractor(data)

        if hasattr(data, 'edge_attr_cont'):
            edge_embs = self.dist_proj(data.edge_attr_cont)
            edge_batch = data.batch[data.edge_index[0]] if hasattr(data, 'batch') else torch.zeros(edge_embs.size(0), dtype=torch.long, device=edge_embs.device)
            dist_rep = self.spatial_pool(edge_embs, edge_batch)
        else:
            dist_rep = torch.zeros(chem.size(0), self.dist_proj.out_features, device=chem.device)

        chem_q = self.proj_chem(chem)
        q_out = self.q_layer(chem_q, dist_rep)
        logits = self.head(q_out)
        return logits, q_out, desc_preds

class Level5Classical(nn.Module):
    """
    Level 5 Classical: "Electronic Structure / Hückel Model Equivalent"
    """
    def __init__(self, hidden_dim=64, out_dim=12, dropout=0.2, inner_dim=32):
        super().__init__()
        self.extractor = SemanticFeatureExtractor(hidden_dim=hidden_dim, dropout=dropout)
        self.elec_net = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.LayerNorm(inner_dim),
            nn.Sigmoid(),
            nn.Linear(inner_dim, inner_dim),
            nn.LayerNorm(inner_dim),
            nn.Tanh()
        )
        self.agg = nn.Linear(inner_dim, out_dim)

    def forward(self, data):
        _, _, _, chem, desc_preds = self.extractor(data)
        out = self.elec_net(chem)
        logits = self.agg(out)
        return logits, out, desc_preds

class Level5QuantumLayer(nn.Module):
    """
    Level 5 Quantum Layer: "Electronic Structure / Hückel Model"
    Z-rotations reflect atomic electronic properties.
    Entanglements reflect chemical bond orders.
    """
    def __init__(self, n_qubits=4, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch")
        def circuit(chem_inputs, weights):
            for l in range(n_layers):
                for i in range(n_qubits):
                    # Z-rotations reflect electronegativity / partial charges
                    qml.RZ(chem_inputs[i] * weights[l, i, 0], wires=i)
                    qml.RY(chem_inputs[i] * weights[l, i, 1], wires=i)
                for i in range(n_qubits - 1):
                    # XX/YY Entanglement reflecting bonding
                    qml.IsingXX(chem_inputs[i] * weights[l, i, 2], wires=[i, i+1])
                    qml.IsingYY(chem_inputs[i+1] * weights[l, i, 3], wires=[i, i+1])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qnode = circuit
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 4))

    def forward(self, chem):
        chem = torch.atan(chem)
        batch_size = chem.shape[0]
        res = []
        for b in range(batch_size):
            out = self.qnode(chem[b], self.weights)
            res.append(torch.stack(out))
        return torch.stack(res, dim=0).float()

class Level5Quantum(nn.Module):
    def __init__(self, hidden_dim=64, n_qubits=4, q_layers=2, out_dim=12, dropout=0.2):
        super().__init__()
        self.extractor = SemanticFeatureExtractor(hidden_dim=hidden_dim, dropout=dropout)
        self.proj_chem = nn.Linear(hidden_dim, n_qubits)
        self.q_layer = Level5QuantumLayer(n_qubits=n_qubits, n_layers=q_layers)
        self.head = nn.Linear(n_qubits, out_dim)

    def forward(self, data):
        _, _, _, chem, desc_preds = self.extractor(data)
        chem_q = self.proj_chem(chem)
        q_out = self.q_layer(chem_q)
        logits = self.head(q_out)
        return logits, q_out, desc_preds

class Level6Classical(nn.Module):
    """
    Level 6 Classical: "3D Spatial / Electrostatic Mapping Equivalent"
    """
    def __init__(self, hidden_dim=64, out_dim=12, dropout=0.2, inner_dim=32):
        super().__init__()
        self.extractor = SemanticFeatureExtractor(hidden_dim=hidden_dim, dropout=dropout)
        self.spatial_net = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.LayerNorm(inner_dim),
            nn.PReLU(),
            nn.Linear(inner_dim, inner_dim),
            nn.LayerNorm(inner_dim),
            nn.PReLU()
        )
        self.agg = nn.Linear(inner_dim, out_dim)

    def forward(self, data):
        _, _, _, chem, desc_preds = self.extractor(data)
        out = self.spatial_net(chem)
        logits = self.agg(out)
        return logits, out, desc_preds

class Level6QuantumLayer(nn.Module):
    """
    Level 6 Quantum Layer: "3D Spatial / Electrostatic Mapping"
    Interactions mimic spatial folding and physical distances.
    """
    def __init__(self, n_qubits=4, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch")
        def circuit(chem_inputs, weights):
            for l in range(n_layers):
                for i in range(n_qubits):
                    qml.RX(chem_inputs[i] * weights[l, i, 0], wires=i)
                    qml.RY(chem_inputs[i] * weights[l, i, 1], wires=i)
                    qml.RZ(chem_inputs[i] * weights[l, i, 2], wires=i)
                # All-to-all entanglement weighted by 3D spatial chem feature
                for i in range(n_qubits):
                    for j in range(i + 1, n_qubits):
                        coupling = chem_inputs[i] * chem_inputs[j] * weights[l, i, 3]
                        qml.CRZ(coupling, wires=[i, j])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qnode = circuit
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 4))

    def forward(self, chem):
        chem = torch.atan(chem)
        batch_size = chem.shape[0]
        res = []
        for b in range(batch_size):
            out = self.qnode(chem[b], self.weights)
            res.append(torch.stack(out))
        return torch.stack(res, dim=0).float()

class Level6Quantum(nn.Module):
    def __init__(self, hidden_dim=64, n_qubits=4, q_layers=2, out_dim=12, dropout=0.2):
        super().__init__()
        self.extractor = SemanticFeatureExtractor(hidden_dim=hidden_dim, dropout=dropout)
        self.proj_chem = nn.Linear(hidden_dim, n_qubits)
        self.q_layer = Level6QuantumLayer(n_qubits=n_qubits, n_layers=q_layers)
        self.head = nn.Linear(n_qubits, out_dim)

    def forward(self, data):
        _, _, _, chem, desc_preds = self.extractor(data)
        chem_q = self.proj_chem(chem)
        q_out = self.q_layer(chem_q)
        logits = self.head(q_out)
        return logits, q_out, desc_preds

class Level7Classical(nn.Module):
    """
    Level 7 Classical: "Pharmacophore / Reactivity Mapping Equivalent"
    """
    def __init__(self, hidden_dim=64, out_dim=12, dropout=0.2, inner_dim=32):
        super().__init__()
        self.extractor = SemanticFeatureExtractor(hidden_dim=hidden_dim, dropout=dropout)
        self.pharm_net = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.LayerNorm(inner_dim),
            nn.LeakyReLU(),
            nn.Linear(inner_dim, inner_dim),
            nn.LayerNorm(inner_dim),
            nn.LeakyReLU()
        )
        self.agg = nn.Linear(inner_dim, out_dim)

    def forward(self, data):
        _, _, _, chem, desc_preds = self.extractor(data)
        out = self.pharm_net(chem)
        logits = self.agg(out)
        return logits, out, desc_preds

class Level7QuantumLayer(nn.Module):
    """
    Level 7 Quantum Layer: "Pharmacophore / Reactivity Mapping"
    Quantum states represent specific chemical reactivity sites.
    """
    def __init__(self, n_qubits=4, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch")
        def circuit(chem_inputs, weights):
            for l in range(n_layers):
                for i in range(n_qubits):
                    qml.U3(chem_inputs[i] * weights[l, i, 0],
                           chem_inputs[i] * weights[l, i, 1],
                           chem_inputs[i] * weights[l, i, 2], wires=i)
                for i in range(n_qubits - 1):
                    # Multi-controlled interactions representing complex pharmacophore dependencies
                    qml.CRX(chem_inputs[i] * weights[l, i, 3], wires=[i, i+1])
                    qml.CRY(chem_inputs[i+1] * weights[l, i, 4], wires=[i, i+1])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qnode = circuit
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 5))

    def forward(self, chem):
        chem = torch.atan(chem)
        batch_size = chem.shape[0]
        res = []
        for b in range(batch_size):
            out = self.qnode(chem[b], self.weights)
            res.append(torch.stack(out))
        return torch.stack(res, dim=0).float()

class Level7Quantum(nn.Module):
    def __init__(self, hidden_dim=64, n_qubits=4, q_layers=2, out_dim=12, dropout=0.2):
        super().__init__()
        self.extractor = SemanticFeatureExtractor(hidden_dim=hidden_dim, dropout=dropout)
        self.proj_chem = nn.Linear(hidden_dim, n_qubits)
        self.q_layer = Level7QuantumLayer(n_qubits=n_qubits, n_layers=q_layers)
        self.head = nn.Linear(n_qubits, out_dim)

    def forward(self, data):
        _, _, _, chem, desc_preds = self.extractor(data)
        chem_q = self.proj_chem(chem)
        q_out = self.q_layer(chem_q)
        logits = self.head(q_out)
        return logits, q_out, desc_preds
