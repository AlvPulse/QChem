import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import AttentionalAggregation
import torch_geometric.utils as pyg_utils

class SemanticFeatureExtractor(nn.Module):
    """
    Extracts three distinct semantic continuous representations from the input graph:
    1. motif_rep: Localized features (e.g., node degrees, specific atoms, local GNN aggregations)
    2. cycle_rep: Aromatic and ring-based features.
    3. spectral_rep: Graph-level Laplacian/diffusion or global readout features.
    """
    def __init__(self, hidden_dim, node_vocab_sizes=(120,10,7,5,2), emb_dims=(64,16,8,8,4), dropout=0.2):
        super().__init__()

        # Node embeddings
        self.node_embs = nn.ModuleList([
            nn.Embedding(v, d) for v, d in zip(node_vocab_sizes, emb_dims)
        ])
        input_node_dim = sum(emb_dims)

        self.proj = nn.Linear(input_node_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim

        # 1. Motif Extraction (Local aggregations)
        # Using a simple linear transformation of the projected node embeddings
        self.motif_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.motif_pool = AttentionalAggregation(nn.Sequential(nn.Linear(hidden_dim, 1)))

        # 2. Cycle/Aromatic Extraction
        # Focuses on the 'aromatic' flag (index 4 in node features: 120,10,7,5,2 -> index 4 is size 2)
        # and 'cycle' paths.
        self.cycle_mlp = nn.Sequential(
            nn.Linear(hidden_dim + emb_dims[4], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.cycle_pool = AttentionalAggregation(nn.Sequential(nn.Linear(hidden_dim, 1)))

        # 3. Spectral Extraction (Global structure)
        # Uses normalized graph Laplacian eigenvectors/eigenvalues.
        # Since calculating eigenvectors batch-wise is slow/complex, we approximate
        # structural/spectral properties using deep pooling of degrees and global topology.
        self.spectral_mlp = nn.Sequential(
            nn.Linear(hidden_dim + emb_dims[1], hidden_dim), # Degree is index 1
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.spectral_pool = AttentionalAggregation(nn.Sequential(nn.Linear(hidden_dim, 1)))

        # Advanced Chemical Feature extractors for Levels 4-7
        # x_cont format: [partial_charge, en, x, y, z, is_donor, is_acceptor, is_hydrophobe]
        self.chem_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.chem_pool = AttentionalAggregation(nn.Sequential(nn.Linear(hidden_dim, 1)))

    def forward(self, data):
        # Extract and embed node features
        x_cat = []
        # fields: Z, degree, charge, Hs, aromatic
        for i, emb in enumerate(self.node_embs):
            feat = data.x[:, i]
            x_cat.append(emb(feat.clamp(0, emb.num_embeddings - 1)))

        x = torch.cat(x_cat, dim=-1)
        h = self.proj(x)
        h = self.dropout(h)

        batch = getattr(data, 'batch', torch.zeros(h.size(0), dtype=torch.long, device=h.device))

        # 1. Motif Representation
        motif_nodes = self.motif_mlp(h)
        motif_rep = self.motif_pool(motif_nodes, batch) # (B, hidden_dim)

        # 2. Cycle Representation
        # Enhance with aromatic feature explicitly
        aromatic_feat = x_cat[4]
        cycle_nodes = self.cycle_mlp(torch.cat([h, aromatic_feat], dim=-1))
        # Mask out non-aromatic nodes to focus entirely on cycles/aromaticity
        # Aromatic feature is binary, let's use the raw node feature as mask
        is_aromatic = data.x[:, 4].float().unsqueeze(-1)
        cycle_nodes = cycle_nodes * is_aromatic
        cycle_rep = self.cycle_pool(cycle_nodes, batch) # (B, hidden_dim)

        # 3. Spectral Representation
        # Enhance with degree feature
        degree_feat = x_cat[1]
        spectral_nodes = self.spectral_mlp(torch.cat([h, degree_feat], dim=-1))
        spectral_rep = self.spectral_pool(spectral_nodes, batch) # (B, hidden_dim)

        # 4. Native Chemical Representation (used by Levels 4-7)
        if hasattr(data, 'x_cont'):
            chem_nodes = self.chem_mlp(torch.cat([h, data.x_cont], dim=-1))
            chem_rep = self.chem_pool(chem_nodes, batch) # (B, hidden_dim)
        else:
            chem_rep = torch.zeros_like(spectral_rep)

        # We return chem_rep as well for the new levels to use.
        return motif_rep, cycle_rep, spectral_rep, chem_rep
