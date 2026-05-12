import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from quantum.kernel import StructuredQuantumKernel
from features.heterogeneous import extract_all_heterogeneous_features

def parse_label(l_str):
    try:
        vals = l_str.strip('[]').split()
        return [float(v) if v != 'nan' and v != '' else float('nan') for v in vals]
    except:
        return [float('nan')] * 12

def subsample_data(df, target_size=1500, target_task_idx=11):
    """
    Subsamples the dataset using stratified sampling on a specific task
    to ensure we have enough positive and negative examples.
    Task index 11 is usually SR-MMP which is relatively balanced/dense in Tox21.
    """
    # Filter rows that have a valid label for the target task
    df['task_label'] = df['label_parsed'].apply(lambda x: x[target_task_idx])
    df_valid = df.dropna(subset=['task_label']).copy()

    # We want target_size, if valid is less we take what we have
    n_samples = min(target_size, len(df_valid))

    if len(np.unique(df_valid['task_label'])) < 2:
        # Fallback if somehow there's only one class
        return df_valid.sample(n_samples, random_state=42)

    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=42)
    for train_index, _ in sss.split(np.zeros(len(df_valid)), df_valid['task_label']):
        df_sub = df_valid.iloc[train_index].copy()
        break

    return df_sub.reset_index(drop=True)

def main():
    print("Loading EDA_dataset.csv...")
    csv_path = "EDA_dataset.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError("EDA_dataset.csv not found")

    df = pd.read_csv(csv_path)
    df['label_parsed'] = df['label'].apply(parse_label)

    # Clean smiles
    df = df.dropna(subset=['smiles'])
    df = df.drop_duplicates(subset=['smiles'])

    # Use Task 11 (SR-MMP) for stratification, or find the most balanced one
    print("Subsampling data to ~1500 molecules...")
    df_sub = subsample_data(df, target_size=1500, target_task_idx=11)
    print(f"Subsampled size: {len(df_sub)}")

    smiles_list = df_sub['smiles'].tolist()
    labels = np.array(df_sub['label_parsed'].tolist())

    print("Extracting S, M, D features...")
    X_S, X_M, X_D = extract_all_heterogeneous_features(smiles_list)

    X_S_np = X_S.numpy()
    X_M_np = X_M.numpy()
    X_D_np = X_D.numpy()

    # Deterministic Projection and Scaling
    print("Applying PCA and Scaling...")
    pca = PCA(n_components=5, random_state=42)
    X_M_5d = pca.fit_transform(X_M_np)

    scaler_S = StandardScaler()
    scaler_M = StandardScaler()
    scaler_D = StandardScaler()

    X_S_5d = scaler_S.fit_transform(X_S_np)
    X_M_5d = scaler_M.fit_transform(X_M_5d)
    X_D_5d = scaler_D.fit_transform(X_D_np)

    # Instantiate Quantum Kernel Module (without random projections)
    sqk = StructuredQuantumKernel(s_dim=5, m_dim=5, d_dim=5, n_qubits=5, use_projections=False)

    print("Generating Quantum Feature Map vectors (this may take a few minutes)...")
    with torch.no_grad():
        batch_size = 100
        qfm_list = []

        # Convert back to torch
        ts_S = torch.tensor(X_S_5d, dtype=torch.float32)
        ts_M = torch.tensor(X_M_5d, dtype=torch.float32)
        ts_D = torch.tensor(X_D_5d, dtype=torch.float32)

        for i in range(0, len(X_S), batch_size):
            qfm_batch = sqk(ts_S[i:i+batch_size], ts_M[i:i+batch_size], ts_D[i:i+batch_size])
            qfm_list.append(qfm_batch)
        QFM_features = torch.cat(qfm_list, dim=0).numpy()

    # --- Kernels ---
    print("Computing Kernel Matrices...")

    # 1. Quantum Kernel (Linear on QFM)
    K_quantum = QFM_features @ QFM_features.T

    # 2. Classical RBF on Full 1034D Concatenated (Upper Bound baseline)
    X_concat_full = np.concatenate([X_S_np, X_M_np, X_D_np], axis=1)
    gamma_full = 1.0 / (X_concat_full.shape[1] * X_concat_full.var()) if X_concat_full.var() > 0 else 1.0
    K_concat_full = rbf_kernel(X_concat_full, gamma=gamma_full)

    # 3. Classical RBF on Reduced 15D Concatenated (Fair comparison)
    X_concat_15d = np.concatenate([X_S_5d, X_M_5d, X_D_5d], axis=1)
    gamma_15d = 1.0 / (X_concat_15d.shape[1] * X_concat_15d.var()) if X_concat_15d.var() > 0 else 1.0
    K_concat_15d = rbf_kernel(X_concat_15d, gamma=gamma_15d)

    # 4. Additive RBF (Reduced)
    gamma_S = 1.0 / (X_S_5d.shape[1] * X_S_5d.var()) if X_S_5d.var() > 0 else 1.0
    gamma_M = 1.0 / (X_M_5d.shape[1] * X_M_5d.var()) if X_M_5d.var() > 0 else 1.0
    gamma_D = 1.0 / (X_D_5d.shape[1] * X_D_5d.var()) if X_D_5d.var() > 0 else 1.0

    K_S = rbf_kernel(X_S_5d, gamma=gamma_S)
    K_M = rbf_kernel(X_M_5d, gamma=gamma_M)
    K_D = rbf_kernel(X_D_5d, gamma=gamma_D)
    K_add = K_S + K_M + K_D

    # 5. Multiplicative RBF (Reduced)
    K_mult = K_S * K_M * K_D

    kernels = {
        'Quantum (SQK)': K_quantum,
        'Classical RBF (Full Concat)': K_concat_full,
        'Classical RBF (Reduced Concat)': K_concat_15d,
        'Classical RBF (Reduced Additive)': K_add,
        'Classical RBF (Reduced Multiplicative)': K_mult
    }

    # --- Evaluation ---
    print("Evaluating Kernels via SVC...")
    n_tasks = labels.shape[1]

    results = {k: {'roc': [], 'pr': []} for k in kernels.keys()}

    # Split into train/test indices
    indices = np.arange(len(df_sub))
    np.random.shuffle(indices)
    train_size = int(0.8 * len(indices))
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    for task_i in range(n_tasks):
        y_task = labels[:, task_i]

        # Filter valid
        mask = ~np.isnan(y_task)

        train_mask = mask[train_idx]
        test_mask = mask[test_idx]

        y_train = y_task[train_idx][train_mask]
        y_test = y_task[test_idx][test_mask]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        # Get actual valid indices in the original matrix
        valid_train_idx = train_idx[train_mask]
        valid_test_idx = test_idx[test_mask]

        for k_name, K in kernels.items():
            # Subset the precomputed kernel matrix
            K_train = K[np.ix_(valid_train_idx, valid_train_idx)]
            K_test = K[np.ix_(valid_test_idx, valid_train_idx)]

            clf = SVC(kernel='precomputed', probability=True, random_state=42)
            clf.fit(K_train, y_train)

            probs = clf.predict_proba(K_test)[:, 1]
            roc = roc_auc_score(y_test, probs)
            pr = average_precision_score(y_test, probs)

            results[k_name]['roc'].append(roc)
            results[k_name]['pr'].append(pr)

    print("\n=== Final Benchmark Results ===")
    for k_name, mets in results.items():
        mean_roc = np.mean(mets['roc']) if mets['roc'] else 0.0
        mean_pr = np.mean(mets['pr']) if mets['pr'] else 0.0
        print(f"{k_name}:")
        print(f"  Mean ROC AUC: {mean_roc:.4f}")
        print(f"  Mean PR AUC:  {mean_pr:.4f}")

if __name__ == "__main__":
    main()
