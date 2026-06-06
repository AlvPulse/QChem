
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
import ast

# Constants
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

def canonical_smiles(s):
    try:
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m) if m else None
    except:
        return None

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
    try:
        Chem.Kekulize(m, clearAromaticFlags=False)
    except:
        return None

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

    # y is expected to be a list/array of floats (or NaNs)
    y = torch.tensor(y, dtype=torch.float32).view(1, -1)

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

def get_dataloaders(batch_size=64, root_dir='.'):
    # Load and preprocess
    csv_path = os.path.join(root_dir, "EDA_dataset.csv")
    if not os.path.exists(csv_path):
        # Fallback for when running from src or other dirs, try to find it
        if os.path.exists("EDA_dataset.csv"):
             csv_path = "EDA_dataset.csv"
        elif os.path.exists("../EDA_dataset.csv"):
             csv_path = "../EDA_dataset.csv"
        else:
             raise FileNotFoundError("EDA_dataset.csv not found")

    df = pd.read_csv(csv_path)

    def parse_label(l_str):
        # Convert string representation "[0. 0. ...]" to list of floats
        # Replace non-numeric values with NaN if any (though usually 0/1)
        try:
            # Remove brackets and split by space
            vals = l_str.strip('[]').split()
            # Convert to float, replacing '' with nan if needed
            return [float(v) if v != '' else float('nan') for v in vals]
        except:
            return None

    df['label'] = df['label'].apply(parse_label)
    df = df.dropna(subset=['label'])

    # Clean
    df['smiles'] = df['smiles'].apply(canonical_smiles)
    # Remove rows where smiles failed
    df = df.dropna(subset=['smiles'])
    # Remove duplicates
    df = df.drop_duplicates(subset=['smiles'])

    # Verify label dimension
    num_tasks = len(df.iloc[0]['label'])
    # Filter out rows with incorrect label length
    df = df[df['label'].apply(len) == num_tasks]

    # Scaffold Split
    tr_df, va_df, te_df = scaffold_split(df)

    # PyG Datasets
    tr_ds = Tox21GraphDataset('.', tr_df)
    va_ds = Tox21GraphDataset('.', va_df)
    te_ds = Tox21GraphDataset('.', te_df)

    # Calculate pos_weights for all tasks
    # Iterate over training data to count positives and negatives per task
    all_y_tr = np.vstack([d.y.numpy() for d in tr_ds]) # (N, num_tasks)

    pos_weights = []
    for i in range(num_tasks):
        y_task = all_y_tr[:, i]
        # Ignore NaNs
        valid_mask = ~np.isnan(y_task)
        if valid_mask.sum() > 0:
            y_valid = y_task[valid_mask]
            n_pos = (y_valid == 1).sum()
            n_neg = (y_valid == 0).sum()
            weight = n_neg / max(n_pos, 1)
        else:
            weight = 1.0
        pos_weights.append(weight)

    pos_weights = torch.tensor(pos_weights, dtype=torch.float32)

    # Note: WeightedRandomSampler is tricky with multi-task.
    # Usually standard shuffling is used, or a sampler based on the presence of ANY active task.
    # For now, we use standard shuffling for train_loader.

    train_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, pos_weights, num_tasks


def get_toxcast_dataloaders(batch_size=64, root_dir='.'):
    """
    Downloads and prepares the ToxCast dataset using PyTorch Geometric's MoleculeNet.
    Returns: train_loader, val_loader, test_loader, pos_weights, num_tasks
    """
    from torch_geometric.datasets import MoleculeNet

    # We download the dataset using PyG
    dataset = MoleculeNet(root=os.path.join(root_dir, 'data'), name='ToxCast')
    num_tasks = dataset[0].y.shape[1]

    # We want to format this into our scaffold split df format to keep things consistent.
    # PyG MoleculeNet provides smiles in data.smiles
    data_list = []
    for data in dataset:
        data_list.append({
            'smiles': data.smiles,
            'label': data.y.squeeze(0).tolist() # converting tensor of shape (1, num_tasks) to list
        })

    df = pd.DataFrame(data_list)

    # Clean
    df['smiles'] = df['smiles'].apply(canonical_smiles)
    df = df.dropna(subset=['smiles'])
    df = df.drop_duplicates(subset=['smiles'])

    # Filter out rows with incorrect label length
    df = df[df['label'].apply(len) == num_tasks]

    # Scaffold Split
    tr_df, va_df, te_df = scaffold_split(df)

    # PyG Datasets
    tr_ds = Tox21GraphDataset('.', tr_df)
    va_ds = Tox21GraphDataset('.', va_df)
    te_ds = Tox21GraphDataset('.', te_df)

    all_y_tr = np.vstack([d.y.numpy() for d in tr_ds]) # (N, num_tasks)

    pos_weights = []
    for i in range(num_tasks):
        y_task = all_y_tr[:, i]
        valid_mask = ~np.isnan(y_task)
        if valid_mask.sum() > 0:
            y_valid = y_task[valid_mask]
            n_pos = (y_valid == 1).sum()
            n_neg = (y_valid == 0).sum()
            weight = n_neg / max(n_pos, 1)
        else:
            weight = 1.0
        pos_weights.append(weight)

    pos_weights = torch.tensor(pos_weights, dtype=torch.float32)

    train_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, pos_weights, num_tasks
