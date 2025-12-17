
import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold

# Constants
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

def canonical_smiles(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else None

def atom_features(atom):
    return np.array([
        atom.GetAtomicNum(),
        atom.GetTotalDegree(),
        atom.GetFormalCharge(),
        int(atom.GetTotalNumHs()),
        int(atom.GetIsAromatic())
    ], dtype=np.int64)

def bond_features(bond):
    if bond is None:
        return np.array([0, 0, 0], dtype=np.int64)
    else:
        bt = int(bond.GetBondTypeAsDouble())
        ar = int(bond.GetIsAromatic())
        conj = int(bond.GetIsConjugated())
        return np.array([bt, ar, conj], dtype=np.int64)

def mol_to_pyg(smiles, y):
    m = Chem.MolFromSmiles(smiles)
    if m is None: return None
    Chem.Kekulize(m, clearAromaticFlags=False)

    # Nodes
    x = np.vstack([atom_features(a) for a in m.GetAtoms()]).astype(np.int64)

    # Edges
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
    y = torch.tensor([y], dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

def scaffold_split(df, seed=SEED):
    # Create scaffold buckets
    scaffolds = defaultdict(list)
    for idx, smiles in enumerate(df['smiles']):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            scaf = MurckoScaffold.GetScaffoldForMol(mol)
            scaf_smiles = Chem.MolToSmiles(scaf) if scaf else ''
            scaffolds[scaf_smiles].append(idx)

    # Sort buckets by size
    buckets = sorted(scaffolds.values(), key=lambda x: len(x), reverse=True)

    train_idx, val_idx, test_idx = [], [], []
    n_total = len(df)
    train_cutoff = int(0.8 * n_total)
    val_cutoff = int(0.9 * n_total)

    # Fill splits
    for bucket in buckets:
        if len(train_idx) + len(bucket) <= train_cutoff:
            train_idx.extend(bucket)
        elif len(train_idx) + len(val_idx) + len(bucket) <= val_cutoff:
            val_idx.extend(bucket)
        else:
            test_idx.extend(bucket)

    return df.iloc[train_idx].reset_index(drop=True), \
           df.iloc[val_idx].reset_index(drop=True), \
           df.iloc[test_idx].reset_index(drop=True)

class Tox21GraphDataset(InMemoryDataset):
    def __init__(self, root, df, transform=None):
        self.df = df
        super().__init__(root, transform)
        self.data, self.slices = self.process_df()

    def process_df(self):
        data_list = []
        for _, row in self.df.iterrows():
            graph = mol_to_pyg(row['smiles'], row['label'])
            if graph:
                data_list.append(graph)
        return self.collate(data_list)

def get_dataloaders(batch_size=64, task_name='SR-MMP'):
    # Load and preprocess
    df = pd.read_csv("EDA_dataset.csv")

    # Extract task label
    # Note: 'label' column in CSV is string "[0. 0. ...]", need to parse
    # Or rely on 'labels_SP_test' column logic from notebook if available
    # Parsing strictly from the csv format:

    # Mapping task to index (from notebook)
    tox21_tasks=['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase',
                 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma',
                 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
    task_idx = tox21_tasks.index(task_name)

    def parse_label(l_str):
        # Convert string representation of list to actual value
        try:
            # Clean string like '[0. 0. ...]'
            vals = l_str.strip('[]').split()
            return float(vals[task_idx])
        except:
            return float('nan')

    df['label'] = df['label'].apply(parse_label)

    # Clean
    df['smiles'] = df['smiles'].apply(canonical_smiles)
    df = df.dropna(subset=['smiles', 'label']).drop_duplicates(subset=['smiles'])

    # Scaffold Split
    tr_df, va_df, te_df = scaffold_split(df)

    # PyG Datasets
    tr_ds = Tox21GraphDataset('.', tr_df)
    va_ds = Tox21GraphDataset('.', va_df)
    te_ds = Tox21GraphDataset('.', te_df)

    # Loaders with Weighted Random Sampler for Train
    y_tr = np.array([d.y.item() for d in tr_ds])
    class_counts = np.bincount(y_tr.astype(int))
    # Handle potentially missing classes in small subsets, though unlikely in train
    if len(class_counts) < 2: class_counts = np.array([1, 1])

    weights = 1. / class_counts
    samples_weights = weights[y_tr.astype(int)]
    sampler = torch.utils.data.WeightedRandomSampler(samples_weights, len(samples_weights))

    train_loader = DataLoader(tr_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False)

    # Calculate pos_weight for loss function
    pos_weight = class_counts[0] / max(class_counts[1], 1)

    return train_loader, val_loader, test_loader, pos_weight
