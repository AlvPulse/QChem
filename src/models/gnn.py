
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GlobalAttention

class GINEBlock(nn.Module):
    def __init__(self, hidden, edge_emb_dim=8):
        super().__init__()
        self.edge_emb = nn.Embedding(16, edge_emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden)
        )
        self.conv = GINEConv(self.mlp, train_eps=True, edge_dim=edge_emb_dim)

    def forward(self, x, edge_index, edge_attr):
        h = (edge_attr[:,0] + 3*edge_attr[:,1] + 7*edge_attr[:,2]) % 16
        e = self.edge_emb(h)
        return self.conv(x, edge_index, e)

class MultiTaskGNN(nn.Module):
    def __init__(self, num_tasks=12, hidden=128, node_vocab_sizes=(120,10,7,5,2), emb_dims=(64,16,8,8,4)):
        super().__init__()
        self.node_embs = nn.ModuleList([
            nn.Embedding(v, d) for v, d in zip(node_vocab_sizes, emb_dims)
        ])
        node_dim = sum(emb_dims)
        self.proj = nn.Linear(node_dim, hidden)

        self.g1 = GINEBlock(hidden)
        self.g2 = GINEBlock(hidden) # Added depth
        self.g3 = GINEBlock(hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)

        self.gate_nn = nn.Sequential(nn.Linear(hidden, 1))
        self.readout = GlobalAttention(self.gate_nn)

        self.head = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_tasks) # Multi-task output
        )

    def forward(self, data):
        x_cat = []
        Z, deg, chg, Hs, aro = data.x[:,0], data.x[:,1], data.x[:,2], data.x[:,3], data.x[:,4]
        for emb, feat in zip(self.node_embs, [Z,deg,chg,Hs,aro]):
            x_cat.append(emb(feat.clamp(min=0, max=emb.num_embeddings-1)))
        x = torch.cat(x_cat, dim=-1)
        x = self.proj(x)

        x = F.relu(self.g1(x, data.edge_index, data.edge_attr))
        x = self.bn1(x)
        x = F.relu(self.g2(x, data.edge_index, data.edge_attr))
        x = self.bn2(x)
        x = F.relu(self.g3(x, data.edge_index, data.edge_attr))

        batch = getattr(data, 'batch', torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        hg = self.readout(x, batch)

        logits = self.head(hg)
        return logits
