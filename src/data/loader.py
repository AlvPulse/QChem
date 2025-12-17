
import os
import random
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from torch_geometric.data import Data, InMemoryDataset
from rdkit.Chem import MolToSmiles

# Set random seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def canonical_smiles(s):
    try:
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m) if m else None
    except:
        return None

def randomize_smiles(smiles, num_aug=5):
    """
    Generates a list of randomized SMILES strings from a single SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:  # Handle invalid SMILES strings
        return [None] * num_aug
    return [MolToSmiles(mol, doRandom=True) for _ in range(num_aug)]

def load_and_preprocess_data(csv_path, tox21_tasks, seed=42):
    df = pd.read_csv(csv_path)

    # Parse label column from string "[0. 0. ...]" to list of floats
    # Assuming space separated inside brackets
    def parse_label(label_str):
        if isinstance(label_str, str):
            # Remove brackets and split by space
            clean_str = label_str.strip('[]')
            # Split by space and filter empty strings
            parts = [p for p in clean_str.split(' ') if p]
            return [float(p) for p in parts]
        return label_str # already list/array?

    df['label_parsed'] = df['label'].apply(parse_label)

    # Check parsing
    # sample_label = df['label_parsed'].iloc[0]
    # print(f"Sample parsed label: {sample_label}, length: {len(sample_label)}")

    # Canonicalize SMILES
    df['smiles'] = df['smiles'].apply(canonical_smiles)
    df = df.dropna(subset=['smiles']).drop_duplicates(subset=['smiles']).reset_index(drop=True)

    return df

def scaffold_split(df, seed=42):
    # Scaffold splitting
    def scaffold(smiles):
        m = Chem.MolFromSmiles(smiles)
        if m is None: return None
        scaf = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(scaf) if scaf else None

    df['scaffold'] = df['smiles'].apply(scaffold)
    scaf2idxs = defaultdict(list)
    for i, s in enumerate(df['scaffold']):
        scaf2idxs[s].append(i)

    buckets = sorted(scaf2idxs.values(), key=lambda x: -len(x))
    N = len(df)
    tr_idx, va_idx, te_idx = [], [], []
    tr_budget, va_budget = int(0.8*N), int(0.1*N)

    for b in buckets:
        if len(tr_idx) + len(b) <= tr_budget:
            tr_idx += b
        elif len(va_idx) + len(b) <= va_budget:
            va_idx += b
        else:
            te_idx += b

    tr = df.iloc[tr_idx].reset_index(drop=True)
    va = df.iloc[va_idx].reset_index(drop=True)
    te = df.iloc[te_idx].reset_index(drop=True)

    return tr, va, te

def augment_training_data(tr_df, tasks):
    # Augment positive samples for ANY task?
    # Or just keep it simple. The original code only augmented for a single task.
    # For multi-task, it's harder to balance everything perfectly.
    # But we can try to augment rows that have at least one positive label.

    # Let's check how sparse the positives are.
    # For now, let's skip complex augmentation for multi-task as it might explode dataset size
    # without clear benefit if not carefully done per task.
    # However, user asked to "push classical to limits".
    # I will stick to class weighting in loss instead of oversampling by augmentation for now,
    # or maybe mild augmentation.

    # Let's implement the randomization augmentation used in the notebook but adapted.
    # In notebook: positive_class_df = tr[tr['labels_SP_test'] == 1]

    # Here labels are vectors.
    # Let's skip augmentation for now and rely on weighted loss/sampler which is more robust for multitask.
    return tr_df
