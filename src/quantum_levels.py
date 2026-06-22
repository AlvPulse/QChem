import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

from .features.semantic_extractor import SemanticFeatureExtractor
from .quantum_layers import QuantumLayer


def _perms(n_qubits, scrambled, n_sites, seed=20240617):
    """Index permutations used to route input feature columns into gate arguments.

    Returns `n_sites` lists of length `n_qubits`.

    * Not scrambled -> every site is the identity, so feature column i drives the
      gate(s) on wire i consistently (the structured chemistry->operator geometry).
    * Scrambled -> each site gets a *different* fixed random permutation. Every gate,
      weight and wire is untouched (same parameter count, depth and entanglement), but
      the coherent "feature i <-> qubit i" correspondence is destroyed: the same physical
      feature drives unrelated qubits at different points in the circuit. Because the
      upstream learned projection emits a single vector, it cannot undo several
      inconsistent permutations at once, so the inductive bias is genuinely removed
      rather than re-absorbed by training.
    """
    ids = list(range(n_qubits))
    if not scrambled or n_qubits < 2:
        return [ids[:] for _ in range(n_sites)]
    g = torch.Generator().manual_seed(seed)
    out = []
    while len(out) < n_sites:
        p = torch.randperm(n_qubits, generator=g).tolist()
        if p == ids:                      # never the identity (would preserve the mapping)
            continue
        if out and p == out[-1]:          # consecutive sites must differ
            continue
        out.append(p)
    return out


# --- Trainability primitives shared by every level's quantum layer ---
# The original circuits encoded data directly into the variational weights and measured
# only <Z>. That collapsed molecular variation and made phase-based biases unmeasurable.
# The redesign instead: (1) re-uploads each level's chemistry->operator ENCODING every
# layer (data-dependent, scaled by a learnable enc_scale), (2) interleaves a SEPARATE
# trainable variational block, and (3) reads out <X>,<Y>,<Z> per qubit.
N_OBS_PER_QUBIT = 3


def _variational_block(theta_l, ent_l, n_qubits, entangle):
    """One trainable, data-independent variational layer: RY/RZ per qubit + CRZ ring.
    theta_l: (n_qubits, 2); ent_l: (n_qubits,)."""
    for i in range(n_qubits):
        qml.RY(theta_l[i, 0], wires=i)
        qml.RZ(theta_l[i, 1], wires=i)
    if entangle:
        for i in range(n_qubits):
            qml.CRZ(ent_l[i], wires=[i, (i + 1) % n_qubits])


def _xyz_measure(n_qubits):
    """Read <X>,<Y>,<Z> on every qubit -> 3*n_qubits features (X/Y expose phase info)."""
    return ([qml.expval(qml.PauliX(i)) for i in range(n_qubits)] +
            [qml.expval(qml.PauliY(i)) for i in range(n_qubits)] +
            [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)])


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
        latent = torch.cat([out_m, out_c, out_s], dim=1) # (B, 3*out_dim)
        return out, latent, desc_preds


class Level1Quantum(nn.Module):
    """
    Level 1 Quantum: "Three Circuits + Attention"
    Features are routed to independent Quantum models and aggregated via classical attention.
    """
    def __init__(self, hidden_dim=64, n_qubits=4, q_layers=2, out_dim=12, dropout=0.2, ansatz="strong"):
        super().__init__()
        self.extractor = SemanticFeatureExtractor(hidden_dim=hidden_dim, dropout=dropout)

        # Projection from hidden_dim to n_qubits
        self.proj_motif = nn.Linear(hidden_dim, n_qubits)
        self.proj_cycle = nn.Linear(hidden_dim, n_qubits)
        self.proj_spectral = nn.Linear(hidden_dim, n_qubits)

        # Three independent VQCs with the richer <X>,<Y>,<Z> readout (trainability).
        # The ansatz is configurable so the benchmark's 'separable' (no-entanglement)
        # ablation actually changes the circuit.
        self.q_motif = QuantumLayer(n_qubits, n_layers=q_layers, ansatz=ansatz, readout='xyz')
        self.q_cycle = QuantumLayer(n_qubits, n_layers=q_layers, ansatz=ansatz, readout='xyz')
        self.q_spectral = QuantumLayer(n_qubits, n_layers=q_layers, ansatz=ansatz, readout='xyz')

        # Output heads
        self.head_motif = nn.Linear(self.q_motif.n_obs, out_dim)
        self.head_cycle = nn.Linear(self.q_cycle.n_obs, out_dim)
        self.head_spectral = nn.Linear(self.q_spectral.n_obs, out_dim)

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
        latent = torch.cat([out_m, out_c, out_s], dim=1) # (B, 3*out_dim)
        return out, latent, desc_preds

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
            nn.Linear(hidden_dim, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, inner_dim)
        )

        # Cycle -> Phase/Periodic processing (Simulated by Sine/Cosine activations)
        self.mlp_cycle_proj = nn.Linear(hidden_dim, inner_dim)
        self.mlp_cycle_out = nn.Linear(inner_dim, inner_dim)

        # Spectral -> Mixing/Interaction (Simulated by dense cross-attention like mixing)
        self.mlp_spectral = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, inner_dim)
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
    def __init__(self, n_qubits=4, n_layers=2, ansatz='strong'):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_obs = N_OBS_PER_QUBIT * n_qubits
        # 'separable' ablation: drop the spectral coupling + trainable entangler.
        entangle = (ansatz != 'separable')
        # 'scrambled' control: break the motif/cycle/spectral -> qubit alignment.
        pr_m, pr_c, pe_s = _perms(n_qubits, ansatz == 'scrambled', 3)
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch")
        def circuit(m_inputs, c_inputs, s_inputs, theta, ent, enc):
            for l in range(n_layers):
                # --- ENCODING (re-uploaded): operator-family mapping ---
                for i in range(n_qubits):
                    qml.RY(enc[0] * m_inputs[:, pr_m[i]], wires=i)   # motifs -> R_y
                    qml.RZ(enc[1] * c_inputs[:, pr_c[i]], wires=i)   # cycles -> R_z
                if entangle:                                          # spectral -> IsingXX
                    for i in range(n_qubits - 1):
                        qml.IsingXX(enc[2] * s_inputs[:, pe_s[i]], wires=[i, i + 1])
                    qml.IsingXX(enc[2] * s_inputs[:, pe_s[n_qubits - 1]], wires=[n_qubits - 1, 0])
                # --- VARIATIONAL (trainable) ---
                _variational_block(theta[l], ent[l], n_qubits, entangle)
            return _xyz_measure(n_qubits)

        self.qnode = circuit
        self.theta = nn.Parameter(torch.randn(n_layers, n_qubits, 2) * 0.1)
        self.ent = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
        self.enc_scale = nn.Parameter(torch.ones(3))

    def forward(self, m, c, s):
        m = torch.atan(m)
        c = torch.atan(c)
        s = torch.atan(s)
        out = self.qnode(m, c, s, self.theta, self.ent, self.enc_scale)
        return torch.stack(out, dim=-1).float()         # (B, 3*n_qubits)

class Level2Quantum(nn.Module):
    def __init__(self, hidden_dim=64, n_qubits=4, q_layers=2, out_dim=12, dropout=0.2, ansatz='strong'):
        super().__init__()
        self.extractor = SemanticFeatureExtractor(hidden_dim=hidden_dim, dropout=dropout)

        self.proj_motif = nn.Linear(hidden_dim, n_qubits)
        self.proj_cycle = nn.Linear(hidden_dim, n_qubits)
        self.proj_spectral = nn.Linear(hidden_dim, n_qubits)

        self.q_layer = Level2QuantumLayer(n_qubits=n_qubits, n_layers=q_layers, ansatz=ansatz)

        self.head = nn.Linear(self.q_layer.n_obs, out_dim)

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
    Level 3 Quantum Layer: "Chemical Operator Geometry" (trainable redesign).

    Encoding (the inductive bias): motifs -> R_y, motifs modulate cycle phase
    R_z(c + alpha*m), spectral -> IsingXX entanglement geometry.

    Trainability redesign (the old Z-only version sat at chance):
      * data re-uploading -- the chemistry->operator encoding is re-applied EVERY layer,
        interleaved with a trainable variational block (separate params from the encoding);
      * richer readout -- measure <X>,<Y>,<Z> per qubit (3*n_qubits features). <X>/<Y>
        expose the R_z phase information that a Z-only measurement discarded, which had made
        this level's phase-based bias literally unmeasurable;
      * learnable per-modality encoding scale + small variational init to escape the
        near-identity collapse (expval std across molecules was ~0.01).
    """
    N_OBS_PER_QUBIT = 3

    def __init__(self, n_qubits=4, n_layers=2, ansatz='strong'):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_obs = self.N_OBS_PER_QUBIT * n_qubits
        # 'separable' ablation: drop entanglement (both the spectral-encoded IsingXX and
        # the trainable entangler), keeping single-qubit ops.
        entangle = (ansatz != 'separable')
        # 'scrambled' control: destroy the chemistry->qubit alignment (see _perms).
        pr_m, pr_c, pm_m, pe_s = _perms(n_qubits, ansatz == 'scrambled', 4)
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch")
        def circuit(m_inputs, c_inputs, s_inputs, theta, ent, alpha, enc):
            # Inputs (B, n_qubits) broadcast over the batch. enc: (3,) encoding scales.
            for l in range(n_layers):
                # --- ENCODING (re-uploaded chemistry->operator geometry) ---
                for i in range(n_qubits):
                    qml.RY(enc[0] * m_inputs[:, pr_m[i]], wires=i)
                    phase = enc[1] * c_inputs[:, pr_c[i]] + alpha[i] * m_inputs[:, pm_m[i]]
                    qml.RZ(phase, wires=i)
                if entangle:
                    for i in range(n_qubits - 1):
                        qml.IsingXX(enc[2] * s_inputs[:, pe_s[i]], wires=[i, i + 1])
                    qml.IsingXX(enc[2] * s_inputs[:, pe_s[n_qubits - 1]], wires=[n_qubits - 1, 0])
                # --- VARIATIONAL (trainable) ---
                for i in range(n_qubits):
                    qml.RY(theta[l, i, 0], wires=i)
                    qml.RZ(theta[l, i, 1], wires=i)
                if entangle:
                    for i in range(n_qubits):
                        qml.CRZ(ent[l, i], wires=[i, (i + 1) % n_qubits])

            return ([qml.expval(qml.PauliX(i)) for i in range(n_qubits)] +
                    [qml.expval(qml.PauliY(i)) for i in range(n_qubits)] +
                    [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)])

        self.qnode = circuit
        self.theta = nn.Parameter(torch.randn(n_layers, n_qubits, 2) * 0.1)
        self.ent = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
        self.alpha = nn.Parameter(torch.randn(n_qubits) * 0.1)  # motif->cycle modulation
        self.enc_scale = nn.Parameter(torch.ones(3))            # per-modality encoding scale

    def forward(self, m, c, s):
        m = torch.atan(m)
        c = torch.atan(c)
        s = torch.atan(s)
        # Vectorized over the batch via PennyLane broadcasting.
        out = self.qnode(m, c, s, self.theta, self.ent, self.alpha, self.enc_scale)
        return torch.stack(out, dim=-1).float()   # (B, 3*n_qubits)

class Level3Quantum(nn.Module):
    def __init__(self, hidden_dim=64, n_qubits=4, q_layers=2, out_dim=12, dropout=0.2, ansatz='strong'):
        super().__init__()
        self.extractor = SemanticFeatureExtractor(hidden_dim=hidden_dim, dropout=dropout)

        self.proj_motif = nn.Linear(hidden_dim, n_qubits)
        self.proj_cycle = nn.Linear(hidden_dim, n_qubits)
        self.proj_spectral = nn.Linear(hidden_dim, n_qubits)

        self.q_layer = Level3QuantumLayer(n_qubits=n_qubits, n_layers=q_layers, ansatz=ansatz)

        self.head = nn.Linear(self.q_layer.n_obs, out_dim)

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
    def __init__(self, n_qubits=4, n_layers=2, ansatz='strong'):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        # 'separable' ablation: keep chemical state prep but drop the
        # distance-modulated spatial entanglement.
        entangle = (ansatz != 'separable')
        # 'scrambled' control: the inductive bias is that the distance between a
        # specific qubit pair modulates *that pair's* coupling. Permuting the chemical
        # state-prep and the distance->coupling routing (independently) breaks which
        # distance governs which entangling gate.
        self.n_obs = N_OBS_PER_QUBIT * n_qubits
        pr_c, pe_d = _perms(n_qubits, ansatz == 'scrambled', 2)
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch")
        def circuit(chem_inputs, dist_inputs, theta, ent, enc):
            for l in range(n_layers):
                # --- ENCODING (re-uploaded): chem state-prep + distance-modulated entanglement ---
                for i in range(n_qubits):
                    qml.RY(enc[0] * chem_inputs[:, pr_c[i]], wires=i)
                if entangle:
                    # 3D distance inversely scales the coupling (closer atoms -> stronger).
                    for i in range(n_qubits - 1):
                        qml.CRZ(enc[1] / (1.0 + dist_inputs[:, pe_d[i]]), wires=[i, i + 1])
                    qml.CRZ(enc[1] / (1.0 + dist_inputs[:, pe_d[n_qubits - 1]]), wires=[n_qubits - 1, 0])
                # --- VARIATIONAL (trainable) ---
                _variational_block(theta[l], ent[l], n_qubits, entangle)
            return _xyz_measure(n_qubits)

        self.qnode = circuit
        self.theta = nn.Parameter(torch.randn(n_layers, n_qubits, 2) * 0.1)
        self.ent = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
        self.enc_scale = nn.Parameter(torch.ones(2))   # chem, distance

    def forward(self, chem, dist):
        chem = torch.atan(chem)
        # Distances are strictly positive, scale using log
        dist = torch.log1p(torch.abs(dist))
        out = self.qnode(chem, dist, self.theta, self.ent, self.enc_scale)
        return torch.stack(out, dim=-1).float()

class Level4Quantum(nn.Module):
    def __init__(self, hidden_dim=64, n_qubits=4, q_layers=2, out_dim=12, dropout=0.2, ansatz='strong'):
        super().__init__()
        self.extractor = SemanticFeatureExtractor(hidden_dim=hidden_dim, dropout=dropout)

        self.proj_chem = nn.Linear(hidden_dim, n_qubits)

        # Spatial pooling mapping explicitly to qubits
        self.dist_proj = nn.Linear(1, n_qubits)
        self.spatial_pool = AttentionalAggregation(nn.Sequential(nn.Linear(n_qubits, 1)))

        self.q_layer = Level4QuantumLayer(n_qubits=n_qubits, n_layers=q_layers, ansatz=ansatz)
        self.head = nn.Linear(self.q_layer.n_obs, out_dim)

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
    def __init__(self, n_qubits=4, n_layers=2, ansatz='strong'):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_obs = N_OBS_PER_QUBIT * n_qubits
        # 'separable' ablation: drop the XX/YY bonding entanglement + trainable entangler.
        entangle = (ansatz != 'separable')
        # 'scrambled' control: break the coherent "atomic feature i drives qubit i's
        # rotations AND its bond entanglement" alignment.
        p_rz, p_ry, p_xx, p_yy = _perms(n_qubits, ansatz == 'scrambled', 4)
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch")
        def circuit(chem_inputs, theta, ent, enc):
            for l in range(n_layers):
                # --- ENCODING (re-uploaded): electronic structure + bonding ---
                for i in range(n_qubits):
                    qml.RZ(enc[0] * chem_inputs[:, p_rz[i]], wires=i)  # electronegativity/charge
                    qml.RY(enc[1] * chem_inputs[:, p_ry[i]], wires=i)
                if entangle:
                    for i in range(n_qubits - 1):
                        qml.IsingXX(enc[2] * chem_inputs[:, p_xx[i]], wires=[i, i + 1])
                        qml.IsingYY(enc[2] * chem_inputs[:, p_yy[i + 1]], wires=[i, i + 1])
                # --- VARIATIONAL (trainable) ---
                _variational_block(theta[l], ent[l], n_qubits, entangle)
            return _xyz_measure(n_qubits)

        self.qnode = circuit
        self.theta = nn.Parameter(torch.randn(n_layers, n_qubits, 2) * 0.1)
        self.ent = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
        self.enc_scale = nn.Parameter(torch.ones(3))

    def forward(self, chem):
        chem = torch.atan(chem)
        out = self.qnode(chem, self.theta, self.ent, self.enc_scale)
        return torch.stack(out, dim=-1).float()

class Level5Quantum(nn.Module):
    def __init__(self, hidden_dim=64, n_qubits=4, q_layers=2, out_dim=12, dropout=0.2, ansatz='strong'):
        super().__init__()
        self.extractor = SemanticFeatureExtractor(hidden_dim=hidden_dim, dropout=dropout)
        self.proj_chem = nn.Linear(hidden_dim, n_qubits)
        self.q_layer = Level5QuantumLayer(n_qubits=n_qubits, n_layers=q_layers, ansatz=ansatz)
        self.head = nn.Linear(self.q_layer.n_obs, out_dim)

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
    def __init__(self, n_qubits=4, n_layers=2, ansatz='strong'):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_obs = N_OBS_PER_QUBIT * n_qubits
        # 'separable' ablation: drop the all-to-all electrostatic entanglement + entangler.
        entangle = (ansatz != 'separable')
        # 'scrambled' control: break the electrostatic product term chem_i*chem_j and its
        # coherence with the single-qubit rotations.
        p_rx, p_ry, p_rz, pc_i, pc_j = _perms(n_qubits, ansatz == 'scrambled', 5)
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch")
        def circuit(chem_inputs, theta, ent, enc):
            for l in range(n_layers):
                # --- ENCODING (re-uploaded): full single-qubit rotations + electrostatic coupling ---
                for i in range(n_qubits):
                    qml.RX(enc[0] * chem_inputs[:, p_rx[i]], wires=i)
                    qml.RY(enc[0] * chem_inputs[:, p_ry[i]], wires=i)
                    qml.RZ(enc[0] * chem_inputs[:, p_rz[i]], wires=i)
                if entangle:
                    for i in range(n_qubits):
                        for j in range(i + 1, n_qubits):
                            qml.CRZ(enc[1] * chem_inputs[:, pc_i[i]] * chem_inputs[:, pc_j[j]], wires=[i, j])
                # --- VARIATIONAL (trainable) ---
                _variational_block(theta[l], ent[l], n_qubits, entangle)
            return _xyz_measure(n_qubits)

        self.qnode = circuit
        self.theta = nn.Parameter(torch.randn(n_layers, n_qubits, 2) * 0.1)
        self.ent = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
        self.enc_scale = nn.Parameter(torch.ones(2))   # rotations, coupling

    def forward(self, chem):
        chem = torch.atan(chem)
        out = self.qnode(chem, self.theta, self.ent, self.enc_scale)
        return torch.stack(out, dim=-1).float()

class Level6Quantum(nn.Module):
    def __init__(self, hidden_dim=64, n_qubits=4, q_layers=2, out_dim=12, dropout=0.2, ansatz='strong'):
        super().__init__()
        self.extractor = SemanticFeatureExtractor(hidden_dim=hidden_dim, dropout=dropout)
        self.proj_chem = nn.Linear(hidden_dim, n_qubits)
        self.q_layer = Level6QuantumLayer(n_qubits=n_qubits, n_layers=q_layers, ansatz=ansatz)
        self.head = nn.Linear(self.q_layer.n_obs, out_dim)

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
    def __init__(self, n_qubits=4, n_layers=2, ansatz='strong'):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_obs = N_OBS_PER_QUBIT * n_qubits
        # 'separable' ablation: drop the controlled pharmacophore dependency + entangler.
        entangle = (ansatz != 'separable')
        # 'scrambled' control: break the reactivity-site -> U3/CRX/CRY correspondence.
        p_u3a, p_u3b, p_u3c, p_crx, p_cry = _perms(n_qubits, ansatz == 'scrambled', 5)
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch")
        def circuit(chem_inputs, theta, ent, enc):
            for l in range(n_layers):
                # --- ENCODING (re-uploaded): pharmacophore U3 + controlled dependencies ---
                for i in range(n_qubits):
                    qml.U3(enc[0] * chem_inputs[:, p_u3a[i]],
                           enc[0] * chem_inputs[:, p_u3b[i]],
                           enc[0] * chem_inputs[:, p_u3c[i]], wires=i)
                if entangle:
                    for i in range(n_qubits - 1):
                        qml.CRX(enc[1] * chem_inputs[:, p_crx[i]], wires=[i, i + 1])
                        qml.CRY(enc[1] * chem_inputs[:, p_cry[i + 1]], wires=[i, i + 1])
                # --- VARIATIONAL (trainable) ---
                _variational_block(theta[l], ent[l], n_qubits, entangle)
            return _xyz_measure(n_qubits)

        self.qnode = circuit
        self.theta = nn.Parameter(torch.randn(n_layers, n_qubits, 2) * 0.1)
        self.ent = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
        self.enc_scale = nn.Parameter(torch.ones(2))   # U3, controlled deps

    def forward(self, chem):
        chem = torch.atan(chem)
        out = self.qnode(chem, self.theta, self.ent, self.enc_scale)
        return torch.stack(out, dim=-1).float()

class Level7Quantum(nn.Module):
    def __init__(self, hidden_dim=64, n_qubits=4, q_layers=2, out_dim=12, dropout=0.2, ansatz='strong'):
        super().__init__()
        self.extractor = SemanticFeatureExtractor(hidden_dim=hidden_dim, dropout=dropout)
        self.proj_chem = nn.Linear(hidden_dim, n_qubits)
        self.q_layer = Level7QuantumLayer(n_qubits=n_qubits, n_layers=q_layers, ansatz=ansatz)
        self.head = nn.Linear(self.q_layer.n_obs, out_dim)

    def forward(self, data):
        _, _, _, chem, desc_preds = self.extractor(data)
        chem_q = self.proj_chem(chem)
        q_out = self.q_layer(chem_q)
        logits = self.head(q_out)
        return logits, q_out, desc_preds
