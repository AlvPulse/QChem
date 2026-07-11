
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

from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import os

# Initialize Pharmacophore feature factory
fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
try:
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
except:
    factory = None

def atom_features(atom, mol, conf=None, p_features=None):
    # Base features
    base_feat = np.array([
        atom.GetAtomicNum(),
        atom.GetTotalDegree(),
        atom.GetFormalCharge(),
        int(atom.GetTotalNumHs()),
        int(atom.GetIsAromatic())
    ], dtype=np.int64)

    # Advanced features (Continuous):
    # - Partial Charge (Gasteiger)
    # - Electronegativity (approximate using Pauling scale mapping)
    # - 3D coordinates (x, y, z)
    # - Pharmacophore tags (Donor, Acceptor, Hydrophobe)

    try:
        partial_charge = float(atom.GetProp('_GasteigerCharge'))
        if np.isnan(partial_charge) or np.isinf(partial_charge):
            partial_charge = 0.0
    except:
        partial_charge = 0.0

    # Pauling electronegativity approximation (subset)
    en_map = {1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66}
    en = en_map.get(atom.GetAtomicNum(), 2.5) # Default to Carbon

    # 3D Coordinates
    coords = [0.0, 0.0, 0.0]
    if conf is not None:
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords = [pos.x, pos.y, pos.z]

    # Pharmacophores
    is_donor = 0.0
    is_acceptor = 0.0
    is_hydrophobe = 0.0
    if p_features is not None:
        idx = atom.GetIdx()
        if idx in p_features['Donor']: is_donor = 1.0
        if idx in p_features['Acceptor']: is_acceptor = 1.0
        if idx in p_features['Hydrophobe']: is_hydrophobe = 1.0

    continuous_feat = np.array([partial_charge, en] + coords + [is_donor, is_acceptor, is_hydrophobe], dtype=np.float32)

    return base_feat, continuous_feat

def bond_features(bond, conf=None):
    if bond is None:
        return np.array([0, 0, 0], dtype=np.int64), np.array([0.0], dtype=np.float32)
    else:
        bt = int(bond.GetBondTypeAsDouble())
        ar = int(bond.GetIsAromatic())
        conj = int(bond.GetIsConjugated())

        # Calculate 3D distance if conformer exists
        distance = 0.0
        if conf is not None:
            pos_i = conf.GetAtomPosition(bond.GetBeginAtomIdx())
            pos_j = conf.GetAtomPosition(bond.GetEndAtomIdx())
            distance = pos_i.Distance(pos_j)

        return np.array([bt, ar, conj], dtype=np.int64), np.array([distance], dtype=np.float32)

from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler

# Global scaler to be fitted after dataset creation
_desc_scaler = StandardScaler()

def calculate_descriptors(m):
    # MolWt, LogP, TPSA, HBD, HBA, Rotatable Bonds
    try:
        molwt = Descriptors.MolWt(m)
        logp = Descriptors.MolLogP(m)
        tpsa = Descriptors.TPSA(m)
        hbd = Descriptors.NumHDonors(m)
        hba = Descriptors.NumHAcceptors(m)
        rotb = Descriptors.NumRotatableBonds(m)
        return [molwt, logp, tpsa, hbd, hba, rotb]
    except:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

def mol_to_pyg(smiles, y):
    m = Chem.MolFromSmiles(smiles)
    if m is None: return None
    try:
        Chem.Kekulize(m, clearAromaticFlags=False)
    except:
        return None

    # Calculate Auxiliary Descriptors before adding Hs
    desc = calculate_descriptors(m)

    # Add Hs for 3D embedding and charges
    m = Chem.AddHs(m)

    # Calculate Gasteiger Charges
    try:
        AllChem.ComputeGasteigerCharges(m)
    except:
        pass

    # Generate 3D Conformer
    try:
        res = AllChem.EmbedMolecule(m, randomSeed=SEED, maxAttempts=50)
        if res == -1: # Failed to embed
            conf = None
        else:
            # Optimize geometry
            AllChem.UFFOptimizeMolecule(m, maxIters=200)
            conf = m.GetConformer()
    except:
        conf = None

    # Extract Pharmacophores
    p_features = {'Donor': set(), 'Acceptor': set(), 'Hydrophobe': set()}
    if factory is not None:
        try:
            feats = factory.GetFeaturesForMol(m)
            for f in feats:
                fam = f.GetFamily()
                for atom_idx in f.GetAtomIds():
                    if fam == 'Donor': p_features['Donor'].add(atom_idx)
                    elif fam == 'Acceptor': p_features['Acceptor'].add(atom_idx)
                    elif fam == 'Hydrophobe': p_features['Hydrophobe'].add(atom_idx)
        except:
            pass

    # Nodes
    x_base = []
    x_cont = []
    for a in m.GetAtoms():
        b_f, c_f = atom_features(a, m, conf, p_features)
        x_base.append(b_f)
        x_cont.append(c_f)

    x_base = np.vstack(x_base).astype(np.int64)
    x_cont = np.vstack(x_cont).astype(np.float32)

    # Edges
    ei_src, ei_dst, eattr, eattr_cont = [], [], [], []
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf, bf_cont = bond_features(b, conf)
        ei_src += [i, j]; ei_dst += [j, i]
        eattr += [bf, bf]
        eattr_cont += [bf_cont, bf_cont]

    if len(ei_src) == 0:
        ei_src = [0]; ei_dst = [0]
        bf, bf_cont = bond_features(None)
        eattr = [bf]
        eattr_cont = [bf_cont]

    edge_index = torch.tensor([ei_src, ei_dst], dtype=torch.long)
    edge_attr = torch.tensor(np.vstack(eattr), dtype=torch.long)
    edge_attr_cont = torch.tensor(np.vstack(eattr_cont), dtype=torch.float32)
    x = torch.tensor(x_base, dtype=torch.long)
    x_cont = torch.tensor(x_cont, dtype=torch.float32)

    # y is expected to be a list/array of floats (or NaNs)
    y = torch.tensor(y, dtype=torch.float32).view(1, -1)

    # Descriptor target
    y_desc = torch.tensor(desc, dtype=torch.float32).view(1, -1)

    return Data(x=x, x_cont=x_cont, edge_index=edge_index, edge_attr=edge_attr, edge_attr_cont=edge_attr_cont, y=y, y_desc=y_desc)

def murcko_scaffold(smiles, mol=None):
    """Bemis-Murcko scaffold SMILES for a molecule (acyclic -> '' bucket).
    Pass an already-parsed `mol` to avoid re-parsing."""
    if mol is None:
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ''
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaf) if scaf else ''
    except Exception:
        return ''


def scaffold_split(df, seed=SEED):
    # Create scaffold buckets
    scaffolds = defaultdict(list)
    for idx, smiles in enumerate(df['smiles']):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            scaf_smiles = murcko_scaffold(smiles, mol)
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
        scaffolds = []
        for _, row in self.df.iterrows():
            graph = mol_to_pyg(row['smiles'], row['label'])
            if graph:
                data_list.append(graph)
                # Aligned with dataset order; used for scaffold-grouped CV.
                scaffolds.append(murcko_scaffold(row['smiles']))
        self.scaffolds = scaffolds
        return self.collate(data_list)

POS_WEIGHT_CAP = 20.0  # ceiling so a near-degenerate task can't dominate the gradient


def _compute_pos_weights(tr_ds, num_tasks, cap=POS_WEIGHT_CAP):
    """Per-task pos_weight = n_neg / n_pos over the training split, NaN-aware and clamped.
    The clamp avoids huge weights (e.g. 100+ for very rare tasks) destabilizing training."""
    all_y_tr = np.vstack([d.y.numpy() for d in tr_ds]) # (N, num_tasks)

    pos_weights = []
    for i in range(num_tasks):
        y_task = all_y_tr[:, i]
        valid_mask = ~np.isnan(y_task)
        if valid_mask.sum() > 0:
            y_valid = y_task[valid_mask]
            n_pos = (y_valid == 1).sum()
            n_neg = (y_valid == 0).sum()
            weight = min(n_neg / max(n_pos, 1), cap)
        else:
            weight = 1.0
        pos_weights.append(weight)

    return torch.tensor(pos_weights, dtype=torch.float32)


def _finalize_loaders(tr_df, va_df, te_df, num_tasks, batch_size):
    """
    Shared pipeline tail used by every dataset entry point: featurize the three
    scaffold splits, fit the descriptor scaler on train only, and build loaders.
    Returns: train_loader, val_loader, test_loader, pos_weights, num_tasks
    """
    # PyG Datasets (featurized via mol_to_pyg: graphs, 3D conformers, x_cont, descriptors)
    tr_ds = Tox21GraphDataset('.', tr_df)
    va_ds = Tox21GraphDataset('.', va_df)
    te_ds = Tox21GraphDataset('.', te_df)

    # Fit descriptor scaler on TRAIN only, then apply to all splits (no leakage).
    train_desc = np.vstack([tr_ds.data.y_desc[tr_ds.slices['y_desc'][i]:tr_ds.slices['y_desc'][i+1]].numpy() for i in range(len(tr_ds))])
    _desc_scaler.fit(train_desc)

    for ds in [tr_ds, va_ds, te_ds]:
        if hasattr(ds.data, 'y_desc') and ds.data.y_desc is not None:
            ds.data.y_desc = torch.tensor(_desc_scaler.transform(ds.data.y_desc.numpy()), dtype=torch.float32)

    pos_weights = _compute_pos_weights(tr_ds, num_tasks)

    # Note: WeightedRandomSampler is tricky with multi-task; we use standard shuffling.
    train_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, pos_weights, num_tasks


def _load_moleculenet_labels(name, root_dir='.'):
    """
    Load a MoleculeNet dataset and return {canonical_smiles: label_vector} plus the
    task count. Missing labels are already NaN in PyG's MoleculeNet.
    """
    from torch_geometric.datasets import MoleculeNet

    dataset = MoleculeNet(root=os.path.join(root_dir, 'data'), name=name)
    n_tasks = dataset[0].y.shape[1]

    label_map = {}
    for data in dataset:
        smi = canonical_smiles(data.smiles)
        if smi is None:
            continue
        # Last write wins on duplicate scaffolds/SMILES; acceptable for our use.
        label_map[smi] = data.y.squeeze(0).numpy().astype(np.float32)  # (n_tasks,) with NaN
    return label_map, n_tasks


def build_merged_dataframe(root_dir='.', datasets=('Tox21', 'ToxCast')):
    """
    Merge several MoleculeNet datasets into one multi-task DataFrame keyed by
    canonical SMILES. Each dataset occupies a contiguous block of label columns;
    a molecule absent from a dataset has NaN for that block (handled downstream by
    the masked BCE loss). Default Tox21 (12) + ToxCast (617) -> 629 tasks.
    Returns: (df with columns ['smiles', 'label'], total_num_tasks)
    """
    label_maps, task_counts = [], []
    for name in datasets:
        m, n = _load_moleculenet_labels(name, root_dir)
        label_maps.append(m)
        task_counts.append(n)

    total_tasks = int(sum(task_counts))
    offsets = np.cumsum([0] + task_counts[:-1]).astype(int)

    all_smiles = set()
    for m in label_maps:
        all_smiles.update(m.keys())

    rows = []
    for smi in sorted(all_smiles):
        label = np.full(total_tasks, np.nan, dtype=np.float32)
        for k, m in enumerate(label_maps):
            if smi in m:
                off = offsets[k]
                label[off:off + task_counts[k]] = m[smi]
        rows.append({'smiles': smi, 'label': label.tolist()})

    df = pd.DataFrame(rows)
    # Record the per-dataset block structure so downstream code can slice the merged
    # task axis into its source datasets (e.g. Tox21 vs ToxCast per-block metrics).
    df.attrs['block_tasks'] = [(name, int(n)) for name, n in zip(datasets, task_counts)]
    print(f"Merged {datasets} -> {len(df)} unique molecules, {total_tasks} tasks "
          f"(blocks: {dict(zip(datasets, task_counts))})")
    return df, total_tasks


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
    return _finalize_loaders(tr_df, va_df, te_df, num_tasks, batch_size)


def get_toxcast_dataloaders(batch_size=64, root_dir='.'):
    """
    Downloads and prepares the ToxCast dataset using PyTorch Geometric's MoleculeNet.
    Returns: train_loader, val_loader, test_loader, pos_weights, num_tasks
    """
    label_map, num_tasks = _load_moleculenet_labels('ToxCast', root_dir)
    df = pd.DataFrame([{'smiles': smi, 'label': lab.tolist()} for smi, lab in label_map.items()])

    # Filter out rows with incorrect label length (defensive)
    df = df[df['label'].apply(len) == num_tasks]

    tr_df, va_df, te_df = scaffold_split(df)
    return _finalize_loaders(tr_df, va_df, te_df, num_tasks, batch_size)


class CachedGraphDataset(Tox21GraphDataset):
    """An InMemoryDataset reconstructed from a cached (data, slices) payload,
    skipping the (slow) mol_to_pyg + 3D-conformer featurization."""
    def __init__(self, data, slices, root='.'):
        InMemoryDataset.__init__(self, root)
        self.data, self.slices = data, slices


def _merged_scaffolds_in_order(root_dir, datasets):
    """Recover the Murcko scaffold of each featurized graph, in dataset order, without
    re-running the (slow) 3D-conformer featurization. The merged dataframe is
    deterministic and `mol_to_pyg` drops a molecule iff RDKit fails to parse or
    kekulize it, so replicating just that cheap test reproduces the dataset order."""
    df, _ = build_merged_dataframe(root_dir, datasets=datasets)
    scaffolds = []
    for smi in df['smiles']:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        try:
            Chem.Kekulize(m, clearAromaticFlags=False)
        except Exception:
            continue
        scaffolds.append(murcko_scaffold(smi))
    return scaffolds


def _block_tasks_for(root_dir, datasets):
    """Per-dataset task counts [(name, n), ...] without featurization (label maps only)."""
    blocks = []
    for name in datasets:
        _, n = _load_moleculenet_labels(name, root_dir)
        blocks.append((name, int(n)))
    return blocks


def get_or_build_merged_dataset(root_dir='.', datasets=('Tox21', 'ToxCast'), cache_path=None):
    """
    Build the featurized merged dataset once and cache it to disk; subsequent calls
    load the cache instead of re-running 3D conformer embedding for ~10k molecules.
    The dataset object carries `.scaffolds` (one Murcko scaffold per graph, in order)
    for scaffold-grouped cross-validation and `.block_tasks` ([(dataset_name, n_tasks), ...]
    in task-column order) for per-source-dataset metric reporting.
    Returns: (dataset, num_tasks).
    """
    if cache_path is None:
        cache_path = os.path.join(root_dir, 'data', f"featurized_{'_'.join(datasets)}.pt")

    if os.path.exists(cache_path):
        payload = torch.load(cache_path, weights_only=False)
        ds = CachedGraphDataset(payload['data'], payload['slices'])
        scaffolds = payload.get('scaffolds')
        block_tasks = payload.get('block_tasks')
        if scaffolds is None or len(scaffolds) != len(ds) or block_tasks is None:
            # Upgrade an older cache in place (cheap: no re-featurization).
            scaffolds = _merged_scaffolds_in_order(root_dir, datasets)
            block_tasks = _block_tasks_for(root_dir, datasets)
            if len(scaffolds) == len(ds):
                payload['scaffolds'] = scaffolds
                payload['block_tasks'] = block_tasks
                torch.save(payload, cache_path)
            else:
                print(f"WARNING: recovered {len(scaffolds)} scaffolds for {len(ds)} graphs; "
                      f"scaffold CV unavailable, falling back to per-graph buckets.")
                scaffolds = [str(i) for i in range(len(ds))]
        ds.scaffolds = scaffolds
        ds.block_tasks = block_tasks
        print(f"Loaded cached featurized dataset from {cache_path} ({len(ds)} graphs, {payload['num_tasks']} tasks)")
        return ds, int(payload['num_tasks'])

    df, num_tasks = build_merged_dataframe(root_dir, datasets=datasets)
    block_tasks = df.attrs.get('block_tasks', [(d, None) for d in datasets])
    ds = Tox21GraphDataset('.', df)
    ds.block_tasks = block_tasks
    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
    torch.save({'data': ds.data, 'slices': ds.slices, 'num_tasks': num_tasks,
                'scaffolds': ds.scaffolds, 'block_tasks': block_tasks}, cache_path)
    print(f"Cached featurized dataset to {cache_path} ({len(ds)} graphs)")
    return ds, int(num_tasks)


def get_merged_dataloaders(batch_size=64, root_dir='.', datasets=('Tox21', 'ToxCast')):
    """
    Merge multiple MoleculeNet toxicity datasets into a single multi-task problem.
    Molecules missing from a given dataset get NaN labels for that block, which the
    masked BCE / contrastive losses already ignore. Default: Tox21 + ToxCast (629 tasks).
    Returns: train_loader, val_loader, test_loader, pos_weights, num_tasks
    """
    df, num_tasks = build_merged_dataframe(root_dir, datasets=datasets)
    tr_df, va_df, te_df = scaffold_split(df)
    return _finalize_loaders(tr_df, va_df, te_df, num_tasks, batch_size)
