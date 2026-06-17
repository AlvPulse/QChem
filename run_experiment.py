import argparse
import torch
import numpy as np
import random
import os

from src.data_loader import get_dataloaders, get_toxcast_dataloaders
from src.train import Trainer
from src.hybrid_model import HybridEnsembleVQC
from src.baselines import HybridClassicalEnsemble, run_rf_baseline
from src.classical_gnn import ClassicalGNN
from src.models.structured_kernel import HybridStructuredQGNN 
from src.quantum.diagnostics import compute_gram_matrix, analyze_kernel_variance

# Import new levels
from src.quantum_levels import (
    Level1Classical, Level1Quantum,
    Level2Classical, Level2Quantum,
    Level3Classical, Level3Quantum,
    Level4Classical, Level4Quantum,
    Level5Classical, Level5Quantum,
    Level6Classical, Level6Quantum,
    Level7Classical, Level7Quantum
)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="Run Tox21 Experiments")
    parser.add_argument('--model', type=str, default='hybrid_ensemble',
                        choices=[
                            'hybrid_ensemble', 'classical_ensemble', 'rf', 'classical_gnn', 'hybrid_kernel',
                            'level1_classical', 'level1_quantum',
                            'level2_classical', 'level2_quantum',
                            'level3_classical', 'level3_quantum',
                            'level4_classical', 'level4_quantum',
                            'level5_classical', 'level5_quantum',
                            'level6_classical', 'level6_quantum',
                            'level7_classical', 'level7_quantum'
                        ])
    parser.add_argument('--estimators', type=int, default=4, help='Number of estimators in ensemble')
    parser.add_argument('--qubits', type=int, default=4, help='Number of qubits per estimator')
    parser.add_argument('--layers', type=int, default=2, help='Number of quantum layers per estimator')
    parser.add_argument('--alpha', type=float, default=0.1, help='Weight for contrastive loss')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--ansatz', type=str, default='hea', choices=['strong', 'hea', 'reupload', 'mps'])
    parser.add_argument('--split', action='store_true', default=True, help='Split latent vector among estimators')
    parser.add_argument('--no-split', action='store_false', dest='split', help='Copy latent vector to estimators')
    parser.add_argument('--gnn', type=str, default='gine', choices=['gine', 'gat'])
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--diagnose', action='store_true', help='Run kernel diagnostics before training')
    parser.add_argument('--dataset', type=str, default='tox21', choices=['tox21', 'toxcast'], help='Dataset to use')

    args = parser.parse_args()

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Experiment: {args.model} | Device: {device}")

    if args.model == 'rf':
        print("Running Random Forest Baseline...")
        run_rf_baseline()
        return

    # Load Data
    print(f"Loading {args.dataset.capitalize()} Data...")
    if args.dataset == 'toxcast':
        train_loader, val_loader, test_loader, pos_weight, num_tasks = get_toxcast_dataloaders(batch_size=args.batch_size)
    else:
        train_loader, val_loader, test_loader, pos_weight, num_tasks = get_dataloaders(batch_size=args.batch_size)

    print(f"Number of Tasks: {num_tasks}")

    # Initialize Model
    if args.model == 'hybrid_ensemble':
        model = HybridEnsembleVQC(
            n_estimators=args.estimators,
            n_qubits_per_est=args.qubits,
            q_layers=args.layers,
            ansatz=args.ansatz,
            gnn_type=args.gnn,
            dropout=args.dropout,
            split_input=args.split,
            n_outputs=num_tasks
        )
    elif args.model == 'classical_ensemble':
        model = HybridClassicalEnsemble(
            n_estimators=args.estimators,
            n_qubits_per_est=args.qubits,
            gnn_type=args.gnn,
            dropout=args.dropout,
            split_input=args.split,
            n_outputs=num_tasks
        )
    elif args.model == 'classical_gnn':
        model = ClassicalGNN(
            gnn_type=args.gnn,
            dropout=args.dropout,
            out_dim=num_tasks
        )
    elif args.model == 'hybrid_kernel':
        print(f"Initializing Structured Quantum Kernel with {args.qubits} qubits...")
        model = HybridStructuredQGNN(
            num_tasks=num_tasks,
            hidden=64, 
            n_qubits=args.qubits
        )
    # --- NEW LEVELS ---
    elif args.model == 'level1_classical':
        model = Level1Classical(hidden_dim=64, out_dim=num_tasks, dropout=args.dropout)
    elif args.model == 'level1_quantum':
        model = Level1Quantum(hidden_dim=64, n_qubits=args.qubits, q_layers=args.layers, out_dim=num_tasks, dropout=args.dropout)
    elif args.model == 'level2_classical':
        model = Level2Classical(hidden_dim=64, out_dim=num_tasks, dropout=args.dropout)
    elif args.model == 'level2_quantum':
        model = Level2Quantum(hidden_dim=64, n_qubits=args.qubits, q_layers=args.layers, out_dim=num_tasks, dropout=args.dropout)
    elif args.model == 'level3_classical':
        model = Level3Classical(hidden_dim=64, out_dim=num_tasks, dropout=args.dropout)
    elif args.model == 'level3_quantum':
        model = Level3Quantum(hidden_dim=64, n_qubits=args.qubits, q_layers=args.layers, out_dim=num_tasks, dropout=args.dropout)
    elif args.model == 'level4_classical':
        model = Level4Classical(hidden_dim=64, out_dim=num_tasks, dropout=args.dropout)
    elif args.model == 'level4_quantum':
        model = Level4Quantum(hidden_dim=64, n_qubits=args.qubits, q_layers=args.layers, out_dim=num_tasks, dropout=args.dropout)
    elif args.model == 'level5_classical':
        model = Level5Classical(hidden_dim=64, out_dim=num_tasks, dropout=args.dropout)
    elif args.model == 'level5_quantum':
        model = Level5Quantum(hidden_dim=64, n_qubits=args.qubits, q_layers=args.layers, out_dim=num_tasks, dropout=args.dropout)
    elif args.model == 'level6_classical':
        model = Level6Classical(hidden_dim=64, out_dim=num_tasks, dropout=args.dropout)
    elif args.model == 'level6_quantum':
        model = Level6Quantum(hidden_dim=64, n_qubits=args.qubits, q_layers=args.layers, out_dim=num_tasks, dropout=args.dropout)
    elif args.model == 'level7_classical':
        model = Level7Classical(hidden_dim=64, out_dim=num_tasks, dropout=args.dropout)
    elif args.model == 'level7_quantum':
        model = Level7Quantum(hidden_dim=64, n_qubits=args.qubits, q_layers=args.layers, out_dim=num_tasks, dropout=args.dropout)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model.to(device)
    
    # --- OPTIONAL DIAGNOSTICS ---
    if args.model == 'hybrid_kernel' and args.diagnose:
        print("Running Pre-train Diagnostics...")
        model.eval()
        sample_batch = next(iter(val_loader)).to(device)
        with torch.no_grad():
            _, q_features = model(sample_batch)
        gram = compute_gram_matrix(q_features)
        analyze_kernel_variance(gram)

    # Train
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    trainer = Trainer(model, device=device, pos_weight=pos_weight, alpha=args.alpha)

    best_val_roc = 0.0

    for epoch in range(1, args.epochs + 1):
        tr_metrics = trainer.train_epoch(train_loader)
        va_metrics = trainer.evaluate(val_loader)

        print(f"Ep {epoch:02d} | Tr Loss: {tr_metrics['loss']:.4f} ROC: {tr_metrics['roc_auc']:.3f} | "
              f"Va Loss: {va_metrics['loss']:.4f} ROC: {va_metrics['roc_auc']:.3f}")

        if va_metrics['roc_auc'] > best_val_roc:
            best_val_roc = va_metrics['roc_auc']

    # Final Test
    print("Evaluating on Test Set...")
    te_metrics = trainer.evaluate(test_loader)
    print(f"Final Test ROC: {te_metrics['roc_auc']:.4f} PR: {te_metrics['pr_auc']:.4f}")

if __name__ == "__main__":
    main()
