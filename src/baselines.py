
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import torch.nn as nn
from .classical_gnn import ClassicalGNN

def get_morgan_fingerprints(smiles_list, radius=2, n_bits=2048):
    fps = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m:
            fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
            fps.append(np.array(fp))
        else:
            fps.append(np.zeros(n_bits))
    return np.array(fps)

def run_rf_baseline():
    from .data_loader import get_dataloaders
    train_loader, val_loader, test_loader, _ = get_dataloaders(batch_size=32)

    # Extract data
    def extract_data(loader):
        # Access underlying dataframe
        df = loader.dataset.df
        X = get_morgan_fingerprints(df['smiles'].tolist())
        # y is list of lists
        y = np.array(df['label'].tolist())
        return X, y

    print("Extracting features for Random Forest...")
    X_train, y_train = extract_data(train_loader)
    X_val, y_val = extract_data(val_loader)
    X_test, y_test = extract_data(test_loader)

    # Train per task
    n_tasks = y_train.shape[1]
    roc_scores = []
    pr_scores = []

    print(f"Training Random Forest on {n_tasks} tasks...")

    for i in range(n_tasks):
        # Filter valid
        mask_tr = ~np.isnan(y_train[:, i])
        if mask_tr.sum() < 10: continue

        X_tr_task = X_train[mask_tr]
        y_tr_task = y_train[mask_tr, i]

        # Check class balance
        if len(np.unique(y_tr_task)) < 2: continue

        clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        clf.fit(X_tr_task, y_tr_task)

        # Evaluate on Test
        mask_te = ~np.isnan(y_test[:, i])
        if mask_te.sum() < 10: continue

        X_te_task = X_test[mask_te]
        y_te_task = y_test[mask_te, i]

        if len(np.unique(y_te_task)) < 2: continue

        probs = clf.predict_proba(X_te_task)[:, 1]
        roc = roc_auc_score(y_te_task, probs)
        pr = average_precision_score(y_te_task, probs)

        roc_scores.append(roc)
        pr_scores.append(pr)
        print(f"Task {i}: ROC {roc:.3f} PR {pr:.3f}")

    mean_roc = np.mean(roc_scores) if roc_scores else 0.5
    mean_pr = np.mean(pr_scores) if pr_scores else 0.0
    print(f"Mean ROC: {mean_roc:.3f}")
    print(f"Mean PR: {mean_pr:.3f}")
    return mean_roc

# Classical Ensemble Baseline
class ClassicalEstimator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class ClassicalEnsemble(nn.Module):
    def __init__(self, input_dim, n_estimators=4, output_dim=12, split_input=True):
        super().__init__()
        self.input_dim = input_dim
        self.n_estimators = n_estimators
        self.split_input = split_input

        input_dim_per_est = input_dim // n_estimators if split_input else input_dim

        self.estimators = nn.ModuleList([
            ClassicalEstimator(input_dim_per_est, output_dim) for _ in range(n_estimators)
        ])

    def forward(self, x):
        all_logits = []
        for i, est in enumerate(self.estimators):
            if self.split_input:
                chunk_size = self.input_dim // self.n_estimators
                x_i = x[:, i*chunk_size : (i+1)*chunk_size]
            else:
                x_i = x
            all_logits.append(est(x_i))

        stacked = torch.stack(all_logits, dim=1)
        return torch.mean(stacked, dim=1), stacked

class HybridClassicalEnsemble(nn.Module):
    # Same as HybridEnsembleVQC but uses ClassicalEnsemble
    def __init__(self, n_estimators=4, n_qubits_per_est=4, gnn_type='gine', dropout=0.2, split_input=True, n_outputs=12):
        super().__init__()
        self.gnn = ClassicalGNN(gnn_type=gnn_type, dropout=dropout, out_dim=n_outputs)

        if split_input:
            self.latent_dim = n_estimators * n_qubits_per_est
        else:
            self.latent_dim = n_qubits_per_est

        self.projection = nn.Sequential(
            nn.Linear(self.gnn.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.latent_dim)
        )

        self.ensemble = ClassicalEnsemble(
            input_dim=self.latent_dim,
            n_estimators=n_estimators,
            output_dim=n_outputs,
            split_input=split_input
        )

    def forward(self, data):
        graph_emb = self.gnn.forward_features(data)
        latent = self.projection(graph_emb)
        logits, _ = self.ensemble(latent)
        return logits, latent, self.gnn.desc_head(graph_emb)
