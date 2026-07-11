import torch
import numpy as np
from torch_geometric.data import Data
from collections import defaultdict
import os

os.makedirs('data', exist_ok=True)

# Generate dummy featurized dataset
num_mols = 100
n_tasks = 12

data_list = []
for i in range(num_mols):
    x = torch.randn(5, 32)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
    edge_attr = torch.randn(8, 10)
    y = torch.randint(0, 2, (1, n_tasks)).float()
    data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))

# PyG collate manually
from torch_geometric.data.collate import collate
data, slices, _ = collate(data_list[0].__class__, data_list=data_list, increment=False, add_batch=False)
scaffolds = ['O=C1NC=C2CN=CN21' for _ in range(num_mols)]

payload = {'data': data, 'slices': slices, 'scaffolds': scaffolds}
torch.save(payload, 'data/featurized_Tox21.pt')
print("Dummy data saved!")
