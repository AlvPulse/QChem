import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from scipy.stats import wilcoxon
from torch_geometric.loader import DataLoader
from src.train import Trainer
from src.quantum_levels import *
from src.baselines import *
from torch_geometric.datasets import MoleculeNet
import matplotlib.pyplot as plt
import seaborn as sns

def get_param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def match_classical_inner_dim(target_params, level, num_tasks, in_dim=64):
    best_inner = 1
    min_diff = float('inf')

    for inner in range(1, 513):
        if level == 1:
            temp_model = Level1Classical(hidden_dim=in_dim, out_dim=num_tasks, inner_dim=inner)
        elif level == 2:
            temp_model = Level2Classical(hidden_dim=in_dim, out_dim=num_tasks, inner_dim=inner)
        elif level == 3:
            temp_model = Level3Classical(hidden_dim=in_dim, out_dim=num_tasks, inner_dim=inner)
        elif level == 4:
            temp_model = Level4Classical(hidden_dim=in_dim, out_dim=num_tasks, inner_dim=inner)
        elif level == 5:
            temp_model = Level5Classical(hidden_dim=in_dim, out_dim=num_tasks, inner_dim=inner)
        elif level == 6:
            temp_model = Level6Classical(hidden_dim=in_dim, out_dim=num_tasks, inner_dim=inner)
        elif level == 7:
            temp_model = Level7Classical(hidden_dim=in_dim, out_dim=num_tasks, inner_dim=inner)

        params = get_param_count(temp_model)
        diff = abs(params - target_params)

        if diff < min_diff:
            min_diff = diff
            best_inner = inner

        if params > target_params and diff > min_diff:
            break

    return best_inner

def bootstrap_metrics(y_true, y_prob, n_resamples=1000):
    from sklearn.metrics import roc_auc_score, average_precision_score
    roc_aucs, pr_aucs = [], []
    n_samples = len(y_true)
    for _ in range(n_resamples):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_b = y_true[indices]
        y_prob_b = y_prob[indices]

        r, p = [], []
        # Speed optimization: Only sample 10 tasks instead of all 617 to speed up bootstrap
        tasks_to_eval = np.random.choice(y_true_b.shape[1], min(10, y_true_b.shape[1]), replace=False)
        for i in tasks_to_eval:
            valid = ~np.isnan(y_true_b[:, i])
            yt = y_true_b[valid, i]
            yp = y_prob_b[valid, i]
            if len(np.unique(yt)) > 1:
                try:
                    r.append(roc_auc_score(yt, yp))
                    p.append(average_precision_score(yt, yp))
                except:
                    pass
        if r: roc_aucs.append(np.mean(r))
        if p: pr_aucs.append(np.mean(p))

    if not roc_aucs: return (0.5, 0.5), (0.0, 0.0)
    return (np.percentile(roc_aucs, 2.5), np.percentile(roc_aucs, 97.5)), \
           (np.percentile(pr_aucs, 2.5), np.percentile(pr_aucs, 97.5))

def create_model(level, scale, layers, num_tasks, m_type):
    if m_type == 'quantum':
        if level == 1: return Level1Quantum(hidden_dim=64, n_qubits=scale, q_layers=layers, out_dim=num_tasks, ansatz='strong')
        if level == 2: return Level2Quantum(hidden_dim=64, n_qubits=scale, q_layers=layers, out_dim=num_tasks, ansatz='strong')
        if level == 3: return Level3Quantum(hidden_dim=64, n_qubits=scale, q_layers=layers, out_dim=num_tasks, ansatz='strong')
        if level == 4: return Level4Quantum(hidden_dim=64, n_qubits=scale, q_layers=layers, out_dim=num_tasks, ansatz='strong')
        if level == 5: return Level5Quantum(hidden_dim=64, n_qubits=scale, q_layers=layers, out_dim=num_tasks, ansatz='strong')
        if level == 6: return Level6Quantum(hidden_dim=64, n_qubits=scale, q_layers=layers, out_dim=num_tasks, ansatz='strong')
        if level == 7: return Level7Quantum(hidden_dim=64, n_qubits=scale, q_layers=layers, out_dim=num_tasks, ansatz='strong')
    elif m_type == 'separable':
        if level == 1: return Level1Quantum(hidden_dim=64, n_qubits=scale, q_layers=layers, out_dim=num_tasks, ansatz='separable')
        if level == 2: return Level2Quantum(hidden_dim=64, n_qubits=scale, q_layers=layers, out_dim=num_tasks, ansatz='separable')
        if level == 3: return Level3Quantum(hidden_dim=64, n_qubits=scale, q_layers=layers, out_dim=num_tasks, ansatz='separable')
        if level == 4: return Level4Quantum(hidden_dim=64, n_qubits=scale, q_layers=layers, out_dim=num_tasks, ansatz='separable')
        if level == 5: return Level5Quantum(hidden_dim=64, n_qubits=scale, q_layers=layers, out_dim=num_tasks, ansatz='separable')
        if level == 6: return Level6Quantum(hidden_dim=64, n_qubits=scale, q_layers=layers, out_dim=num_tasks, ansatz='separable')
        if level == 7: return Level7Quantum(hidden_dim=64, n_qubits=scale, q_layers=layers, out_dim=num_tasks, ansatz='separable')
    elif m_type == 'classical':
        temp_q = create_model(level, scale, layers, num_tasks, 'quantum')
        q_params = get_param_count(temp_q)
        inner = match_classical_inner_dim(q_params, level, num_tasks, 64)
        if level == 1: return Level1Classical(hidden_dim=64, out_dim=num_tasks, inner_dim=inner)
        if level == 2: return Level2Classical(hidden_dim=64, out_dim=num_tasks, inner_dim=inner)
        if level == 3: return Level3Classical(hidden_dim=64, out_dim=num_tasks, inner_dim=inner)
        if level == 4: return Level4Classical(hidden_dim=64, out_dim=num_tasks, inner_dim=inner)
        if level == 5: return Level5Classical(hidden_dim=64, out_dim=num_tasks, inner_dim=inner)
        if level == 6: return Level6Classical(hidden_dim=64, out_dim=num_tasks, inner_dim=inner)
        if level == 7: return Level7Classical(hidden_dim=64, out_dim=num_tasks, inner_dim=inner)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking on {device} (Phase 1 5-Fold CV Suite)")

    dataset_name = 'toxcast'
    batch_size = 128

    print(f"Loading {dataset_name.capitalize()} Data...")

    dataset = MoleculeNet(root='data', name='ToxCast')
    num_tasks = dataset[0].y.shape[1]

    # For StratifiedKFold, stratify by the first task
    y_stratify = []
    for i in range(len(dataset)):
        val = dataset[i].y[0, 0].item()
        y_stratify.append(val if not np.isnan(val) else 0.0)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    qubit_scales = [4, 6]
    layers = 2
    epochs = 200
    patience = 20

    all_results = []

    for level in [1, 2, 3, 4, 5, 6, 7]:
        for scale in qubit_scales:
            print(f"\n--- Evaluating Level {level} (Qubits: {scale}) ---")

            fold_rocs = {'quantum': [], 'separable': [], 'classical': []}
            pooled_probs = {'quantum': [], 'separable': [], 'classical': []}
            pooled_trues = {'quantum': [], 'separable': [], 'classical': []}

            # Array of indices to split
            indices = np.arange(len(dataset))

            for fold, (train_idx, test_idx) in enumerate(skf.split(indices, y_stratify)):
                print(f"Fold {fold+1}/5")

                # Validation set is 10% of training set
                np.random.seed(42 + fold)
                val_size = int(0.1 * len(train_idx))
                val_choice = np.random.choice(len(train_idx), val_size, replace=False)
                val_idx = train_idx[val_choice]
                train_idx = np.delete(train_idx, val_choice)

                tr_ds = dataset[train_idx]
                va_ds = dataset[val_idx]
                te_ds = dataset[test_idx]

                tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
                va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False)
                te_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False)

                # Pos weights handling imbalance
                # Extract all ys from train ds
                all_labels = torch.cat([data.y for data in tr_ds], dim=0)
                pos_weight = []
                for t in range(num_tasks):
                    valid = ~torch.isnan(all_labels[:, t])
                    pos = all_labels[valid, t].sum().item()
                    neg = valid.sum().item() - pos
                    pos_weight.append(neg / (pos + 1e-5))
                pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(device)

                models = ['classical', 'separable', 'quantum']
                for m_type in models:
                    model = create_model(level, scale, layers, num_tasks, m_type)
                    trainer = Trainer(model, device, pos_weight)

                    best_val_loss = float('inf')
                    epochs_no_improve = 0

                    for ep in range(epochs):
                        trainer.train_epoch(tr_loader)
                        va_mets = trainer.evaluate(va_loader)
                        trainer.scheduler.step(va_mets['loss'])

                        if va_mets['loss'] < best_val_loss:
                            best_val_loss = va_mets['loss']
                            epochs_no_improve = 0
                        else:
                            epochs_no_improve += 1
                            if epochs_no_improve >= patience:
                                print(f"Early stopping at epoch {ep} for {m_type}")
                                break

                    te_mets = trainer.evaluate(te_loader)
                    fold_rocs[m_type].append(te_mets['roc_auc'])

                    pooled_probs[m_type].append(te_mets['y_prob'])
                    pooled_trues[m_type].append(te_mets['y_true'])

            for m_type in models:
                pooled_probs[m_type] = np.vstack(pooled_probs[m_type])
                pooled_trues[m_type] = np.vstack(pooled_trues[m_type])

            try:
                p_classical = wilcoxon(fold_rocs['quantum'], fold_rocs['classical']).pvalue
            except ValueError:
                p_classical = 1.0

            try:
                p_separable = wilcoxon(fold_rocs['quantum'], fold_rocs['separable']).pvalue
            except ValueError:
                p_separable = 1.0

            p_classical = min(1.0, p_classical * 2)
            p_separable = min(1.0, p_separable * 2)

            print(f"Wilcoxon p-val (Q vs C): {p_classical:.4f}")
            print(f"Wilcoxon p-val (Q vs Sep): {p_separable:.4f}")

            for m_type in models:
                mean_roc = np.mean(fold_rocs[m_type])
                std_roc = np.std(fold_rocs[m_type])
                ci_roc, ci_pr = bootstrap_metrics(pooled_trues[m_type], pooled_probs[m_type])

                all_results.append({
                    'Level': level, 'Qubits': scale, 'Model': m_type,
                    'ROC_AUC_CV_Mean': mean_roc, 'ROC_AUC_CV_Std': std_roc,
                    'ROC_CI_95': ci_roc, 'PR_CI_95': ci_pr,
                    'p_val_vs_classical': p_classical if m_type == 'quantum' else np.nan,
                    'p_val_vs_separable': p_separable if m_type == 'quantum' else np.nan
                })

    df_res = pd.DataFrame(all_results)
    os.makedirs('results', exist_ok=True)
    df_res.to_csv('results/benchmark_cv_results.csv', index=False)
    print("Benchmark saved.")

if __name__ == "__main__":
    main()
