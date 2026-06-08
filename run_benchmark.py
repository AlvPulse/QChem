import os
import torch
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.data_loader import get_dataloaders, get_toxcast_dataloaders
from src.train import Trainer
from src.quantum_levels import (
    Level1Classical, Level1Quantum,
    Level2Classical, Level2Quantum,
    Level3Classical, Level3Quantum
)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def match_classical_inner_dim(target_params, level, num_tasks, in_dim=64):
    """
    Iteratively finds the inner_dim for a given Classical level model
    that makes its parameter count as close as possible to target_params.
    """
    best_inner = 1
    min_diff = float('inf')

    # We brute-force search the inner_dim to guarantee accurate parameter matching.
    # Typical inner_dims range from 1 to 512.
    for inner in range(1, 513):
        if level == 1:
            temp_model = Level1Classical(hidden_dim=in_dim, out_dim=num_tasks, inner_dim=inner)
        elif level == 2:
            temp_model = Level2Classical(hidden_dim=in_dim, out_dim=num_tasks, inner_dim=inner)
        elif level == 3:
            temp_model = Level3Classical(hidden_dim=in_dim, out_dim=num_tasks, inner_dim=inner)

        params = get_param_count(temp_model)
        diff = abs(params - target_params)

        if diff < min_diff:
            min_diff = diff
            best_inner = inner

        # If we overshoot and diff starts increasing, we can stop
        if params > target_params and diff > min_diff:
            break

    return best_inner

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking on {device}")

    os.makedirs('results', exist_ok=True)

    # Configuration
    dataset_name = 'toxcast'
    batch_size = 128
    epochs = 10
    qubit_scales = [4, 6]
    layers = 2

    print(f"Loading {dataset_name.capitalize()} Data...")
    if dataset_name == 'toxcast':
        train_loader, val_loader, test_loader, pos_weight, num_tasks = get_toxcast_dataloaders(batch_size=batch_size)
    else:
        train_loader, val_loader, test_loader, pos_weight, num_tasks = get_dataloaders(batch_size=batch_size)

    results = []
    learning_curves = []

    # Iterate over levels and scales
    for scale in qubit_scales:
        for level in [1, 2, 3]:
            # --- 1. Train Quantum to get target params ---
            q_model_name = f"level{level}_quantum"
            print(f"\n--- Initializing {q_model_name.upper()} (Qubits: {scale}) ---")

            if level == 1:
                q_model = Level1Quantum(hidden_dim=64, n_qubits=scale, q_layers=layers, out_dim=num_tasks)
            elif level == 2:
                q_model = Level2Quantum(hidden_dim=64, n_qubits=scale, q_layers=layers, out_dim=num_tasks)
            elif level == 3:
                q_model = Level3Quantum(hidden_dim=64, n_qubits=scale, q_layers=layers, out_dim=num_tasks)

            q_params = get_param_count(q_model)
            q_model.to(device)

            # Train Quantum
            q_trainer = Trainer(q_model, device=device, pos_weight=pos_weight)
            q_best_val = 0
            q_train_roc, q_val_roc = [], []
            for ep in range(epochs):
                tr_mets = q_trainer.train_epoch(train_loader)
                va_mets = q_trainer.evaluate(val_loader)
                q_train_roc.append(tr_mets['roc_auc'])
                q_val_roc.append(va_mets['roc_auc'])
                if va_mets['roc_auc'] > q_best_val:
                    q_best_val = va_mets['roc_auc']

            q_test_mets = q_trainer.evaluate(test_loader)
            print(f"Q-Test ROC: {q_test_mets['roc_auc']:.4f}")

            learning_curves.append({
                'model': q_model_name, 'scale': scale, 'type': 'quantum',
                'train_roc': q_train_roc, 'val_roc': q_val_roc
            })

            desc = "Independent Feature-to-Operator Routing"
            if level == 2: desc = "Chemical-to-Operator Mapping"
            if level == 3: desc = "Dynamic Operator Geometry"

            results.append({
                'Model_Name': f"Level {level} Quantum",
                'Description': desc,
                'Level': level, 'Type': 'Quantum', 'Qubits': scale,
                'Params': q_params, 'Test_ROC': q_test_mets['roc_auc'], 'Test_PR': q_test_mets['pr_auc'],
                'Test_Brier': q_test_mets['brier'], 'Test_F1': q_test_mets['f1']
            })

            # --- 2. Train Classical with Matched Params ---
            c_model_name = f"level{level}_classical"
            inner_dim = match_classical_inner_dim(q_params, level, num_tasks, 64)
            print(f"\n--- Initializing {c_model_name.upper()} (Matched Inner Dim: {inner_dim}) ---")

            if level == 1:
                c_model = Level1Classical(hidden_dim=64, out_dim=num_tasks, inner_dim=inner_dim)
            elif level == 2:
                c_model = Level2Classical(hidden_dim=64, out_dim=num_tasks, inner_dim=inner_dim)
            elif level == 3:
                c_model = Level3Classical(hidden_dim=64, out_dim=num_tasks, inner_dim=inner_dim)

            c_params = get_param_count(c_model)
            c_model.to(device)

            # Train Classical
            c_trainer = Trainer(c_model, device=device, pos_weight=pos_weight)
            c_best_val = 0
            c_train_roc, c_val_roc = [], []
            for ep in range(epochs):
                tr_mets = c_trainer.train_epoch(train_loader)
                va_mets = c_trainer.evaluate(val_loader)
                c_train_roc.append(tr_mets['roc_auc'])
                c_val_roc.append(va_mets['roc_auc'])
                if va_mets['roc_auc'] > c_best_val:
                    c_best_val = va_mets['roc_auc']

            c_test_mets = c_trainer.evaluate(test_loader)
            print(f"C-Test ROC: {c_test_mets['roc_auc']:.4f} (Params: {c_params} vs Q:{q_params})")

            learning_curves.append({
                'model': c_model_name, 'scale': scale, 'type': 'classical',
                'train_roc': c_train_roc, 'val_roc': c_val_roc
            })

            results.append({
                'Model_Name': f"Level {level} Classical",
                'Description': "Parameter-Matched Classical Counterpart",
                'Level': level, 'Type': 'Classical', 'Qubits': scale,
                'Params': c_params, 'Test_ROC': c_test_mets['roc_auc'], 'Test_PR': c_test_mets['pr_auc'],
                'Test_Brier': c_test_mets['brier'], 'Test_F1': c_test_mets['f1']
            })

    # Save Results
    df_res = pd.DataFrame(results)
    df_res.to_csv('results/benchmark_results.csv', index=False)
    print("\nBenchmark Complete. Results saved to results/benchmark_results.csv")

    # --- PLOTTING ---
    # 1. Qubit Scaling Plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_res, x='Qubits', y='Test_ROC', hue='Type', style='Level', markers=True, dashes=False)
    plt.title("Scaling: Qubits vs Test ROC (ToxCast)")
    plt.ylabel("Test ROC AUC")
    plt.xticks(qubit_scales)
    plt.grid(True)
    plt.savefig('results/qubit_scaling.png')
    plt.close()

    # 2. Level Comparison Bar Chart (at max qubits)
    max_qubits = max(qubit_scales)
    df_max = df_res[df_res['Qubits'] == max_qubits]
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df_max, x='Level', y='Test_ROC', hue='Type')
    plt.title(f"Level Comparison at {max_qubits} Qubits")
    plt.ylabel("Test ROC AUC")
    plt.ylim(0.4, 0.8) # Adjust based on expected bounds
    plt.savefig('results/level_comparison.png')
    plt.close()

    # 3. Learning Curves Grid
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle('Learning Curves (Train vs Val ROC) at Max Qubits', fontsize=16)

    for i, level in enumerate([1, 2, 3]):
        # Quantum
        q_curve = next(c for c in learning_curves if c['model'] == f"level{level}_quantum" and c['scale'] == max_qubits)
        axes[i, 0].plot(q_curve['train_roc'], label='Train ROC')
        axes[i, 0].plot(q_curve['val_roc'], label='Val ROC')
        axes[i, 0].set_title(f"Level {level} Quantum")
        axes[i, 0].set_xlabel("Epochs")
        axes[i, 0].set_ylabel("ROC AUC")
        axes[i, 0].legend()

        # Classical
        c_curve = next(c for c in learning_curves if c['model'] == f"level{level}_classical" and c['scale'] == max_qubits)
        axes[i, 1].plot(c_curve['train_roc'], label='Train ROC')
        axes[i, 1].plot(c_curve['val_roc'], label='Val ROC')
        axes[i, 1].set_title(f"Level {level} Classical")
        axes[i, 1].set_xlabel("Epochs")
        axes[i, 1].set_ylabel("ROC AUC")
        axes[i, 1].legend()

    plt.tight_layout()
    plt.savefig('results/learning_curves.png')
    plt.close()

if __name__ == "__main__":
    main()
