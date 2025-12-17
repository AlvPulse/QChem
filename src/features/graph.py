
import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset

# Atom features
def atom_features(atom):
    return np.array([
        atom.GetAtomicNum(),
        atom.GetTotalDegree(),
        atom.GetFormalCharge(),
        int(atom.GetTotalNumHs()),
        int(atom.GetIsAromatic())
    ], dtype=np.int64)

# Edge features
def bond_features(bond):
    if bond is None:
        bt, ar, conj = 0, 0, 0
    else:
        bt = int(bond.GetBondTypeAsDouble())
        ar = int(bond.GetIsAromatic())
        conj = int(bond.GetIsConjugated())
    return np.array([bt, ar, conj], dtype=np.int64)

def mol_to_pyg(smiles, y):
    m = Chem.MolFromSmiles(smiles)
    if m is None: return None
    Chem.Kekulize(m, clearAromaticFlags=False)
    # nodes
    x = np.vstack([atom_features(a) for a in m.GetAtoms()]).astype(np.int64)
    # edges
    ei_src, ei_dst, eattr = [], [], []
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = bond_features(b)
        ei_src += [i, j]; ei_dst += [j, i]
        eattr += [bf, bf]
    if len(ei_src) == 0:
        ei_src = [0]; ei_dst = [0]; eattr = [bond_features(None)]

    edge_index = torch.tensor([ei_src, ei_dst], dtype=torch.long)
    edge_attr = torch.tensor(np.vstack(eattr), dtype=torch.long)
    x = torch.tensor(x, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.float32).view(1, -1) # Shape [1, num_tasks]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

class ToxDataset(InMemoryDataset):
    def __init__(self, df, transform=None):
        super().__init__('.', transform)
        data_list = []
        for s, y in zip(df['smiles'], df['label_parsed']):
            g = mol_to_pyg(s, y)
            if g is not None: data_list.append(g)
        self.data, self.slices = self.collate(data_list)
