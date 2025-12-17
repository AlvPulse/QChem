
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GlobalAttention

class GINEBlock(nn.Module):
    def __init__(self, hidden_dim, edge_emb_dim=8):
        super().__init__()
        self.edge_emb = nn.Embedding(16, edge_emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv = GINEConv(self.mlp, train_eps=True, edge_dim=edge_emb_dim)

    def forward(self, x, edge_index, edge_attr):
        # Hash 3-dim edge attr to 1 index for simple embedding
        h = (edge_attr[:,0] + 3*edge_attr[:,1] + 7*edge_attr[:,2]) % 16
        e = self.edge_emb(h)
        return self.conv(x, edge_index, e)

class ClassicalGNN(nn.Module):
    def __init__(self, hidden_dim=64, node_vocab_sizes=(120,10,7,5,2), emb_dims=(64,16,8,8,4), out_dim=1):
        super().__init__()
        # Node embeddings
        self.node_embs = nn.ModuleList([
            nn.Embedding(v, d) for v, d in zip(node_vocab_sizes, emb_dims)
        ])
        input_node_dim = sum(emb_dims)

        self.proj = nn.Linear(input_node_dim, hidden_dim)
        self.gnn1 = GINEBlock(hidden_dim)
        self.gnn2 = GINEBlock(hidden_dim)
        self.gnn3 = GINEBlock(hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)

        # Pooling
        self.gate_nn = nn.Sequential(nn.Linear(hidden_dim, 1))
        self.readout = GlobalAttention(self.gate_nn)

        # Classification Head (for pure classical baseline)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, out_dim)
        )
        self.hidden_dim = hidden_dim

    def forward_features(self, data):
        # Extract and embed node features
        x_cat = []
        # fields: Z, degree, charge, Hs, aromatic
        for i, emb in enumerate(self.node_embs):
            feat = data.x[:, i]
            # Clamping to vocab size just in case
            x_cat.append(emb(feat.clamp(0, emb.num_embeddings - 1)))

        x = torch.cat(x_cat, dim=-1)
        x = self.proj(x)

        x = F.relu(self.gnn1(x, data.edge_index, data.edge_attr))
        x = F.relu(self.gnn2(x, data.edge_index, data.edge_attr))
        x = self.bn(F.relu(self.gnn3(x, data.edge_index, data.edge_attr)))

        batch = getattr(data, 'batch', torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        graph_emb = self.readout(x, batch)
        return graph_emb

    def forward(self, data):
        h = self.forward_features(data)
        return self.head(h).squeeze(-1)
