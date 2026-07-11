import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import AttentionalAggregation, GINEConv
import torch_geometric.utils as pyg_utils

# Categorical bond-feature vocab (edge_attr = [bond_type, aromatic, conjugated]; see
# bond_features in data_loader.py). Sizes are padded for safety.
_BOND_VOCAB = (8, 2, 2)
_BOND_EMB = (16, 4, 4)


class SemanticFeatureExtractor(nn.Module):
    """
    Extracts three distinct semantic continuous representations from the input graph:
    1. motif_rep: Localized features (e.g., node degrees, specific atoms, local GNN aggregations)
    2. cycle_rep: Aromatic and ring-based features.
    3. spectral_rep: Graph-level Laplacian/diffusion or global readout features.

    Node embeddings are first refined by a stack of GINEConv message-passing layers that
    consume bond connectivity (edge_index) and bond features (edge_attr / edge_attr_cont),
    so the representation is graph-aware rather than a bag of atoms.
    """
    def __init__(self, hidden_dim, node_vocab_sizes=(120,10,7,5,2), emb_dims=(64,16,8,8,4),
                 dropout=0.2, n_mp_layers=3):
        super().__init__()

        # Auxiliary descriptor prediction head
        # Because all quantum levels use this extractor as the base classical encoder,
        # we put the descriptor head here to ensure the gradient flows back to update the embeddings.
        self.desc_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 6) # 6 continuous target descriptors
        )

        # Node embeddings
        self.node_embs = nn.ModuleList([
            nn.Embedding(v, d) for v, d in zip(node_vocab_sizes, emb_dims)
        ])
        input_node_dim = sum(emb_dims)

        self.proj = nn.Linear(input_node_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim

        # --- Graph message passing (GINE) ---
        # Edge encoder: embed the 3 categorical bond features + project the continuous
        # bond distance, producing a hidden_dim edge embedding for GINEConv.
        self.bond_embs = nn.ModuleList([
            nn.Embedding(v, d) for v, d in zip(_BOND_VOCAB, _BOND_EMB)
        ])
        self.edge_encoder = nn.Linear(sum(_BOND_EMB) + 1, hidden_dim)
        self.convs = nn.ModuleList()
        self.mp_norms = nn.ModuleList()
        for _ in range(n_mp_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, train_eps=True))
            self.mp_norms.append(nn.LayerNorm(hidden_dim))

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

        # --- Graph message passing over bonds (GINE) ---
        # Build a hidden_dim edge embedding from categorical bond features + distance, then
        # refine node embeddings with residual GINEConv layers so they carry neighbourhood
        # and bond context (the streams below previously saw only per-atom features).
        edge_index = data.edge_index
        ea = data.edge_attr
        bond_cat = torch.cat(
            [emb(ea[:, k].clamp(0, emb.num_embeddings - 1)) for k, emb in enumerate(self.bond_embs)],
            dim=-1,
        )
        ea_cont = getattr(data, 'edge_attr_cont', None)
        if ea_cont is None:
            ea_cont = torch.zeros(ea.size(0), 1, device=h.device)
        edge_emb = self.edge_encoder(torch.cat([bond_cat, ea_cont], dim=-1))
        for conv, norm in zip(self.convs, self.mp_norms):
            h = norm(h + F.relu(conv(h, edge_index, edge_emb)))

        # 1. Motif Representation
        motif_nodes = self.motif_mlp(h)
        motif_rep = self.motif_pool(motif_nodes, batch) # (B, hidden_dim)

        # 2. Cycle Representation
        # Enhance with aromatic feature explicitly
        aromatic_feat = x_cat[4]
        cycle_nodes = self.cycle_mlp(torch.cat([h, aromatic_feat], dim=-1))
        # The aromatic embedding is already concatenated above, so the stream still
        # emphasizes aromaticity. We intentionally do NOT hard-mask by `is_aromatic`:
        # that zeroed the entire cycle representation for non-aromatic molecules.
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

        # Aggregate all representations to predict descriptors
        combined_rep = torch.cat([motif_rep, cycle_rep, spectral_rep, chem_rep], dim=-1)
        desc_preds = self.desc_head(combined_rep)

        # We return chem_rep as well for the new levels to use.
        return motif_rep, cycle_rep, spectral_rep, chem_rep, desc_preds
