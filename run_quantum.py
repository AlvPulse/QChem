
import os
import torch
import numpy as np
from src.data.loader import load_and_preprocess_data, scaffold_split, set_seed
from src.features.graph import ToxDataset
from src.models.quantum import HybridQGNN
from src.train import Trainer
from torch_geometric.loader import DataLoader
import torch.nn as nn

def main():
    set_seed(42)
    device = torch.device('cpu') # PennyLane usually runs on CPU unless using specific plugins
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading data...")
    tox21_tasks=['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
                 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']

    df = load_and_preprocess_data("EDA_dataset.csv", tox21_tasks)
    tr, va, te = scaffold_split(df)

    # Subset for QML speed if needed, but let's try full dataset first.
    # QML simulation is slow.
    # We might want to reduce batch size or data size for this demonstration if it takes too long.
    # Let's use a smaller subset for demonstration if it's too slow.
    # For now, full dataset.

    print(f"Train size: {len(tr)}, Val size: {len(va)}, Test size: {len(te)}")

    # 2. Prepare Datasets
    print("Creating Graph Datasets...")
    ds_tr = ToxDataset(tr)
    ds_va = ToxDataset(va)
    ds_te = ToxDataset(te)

    batch_size = 32 # Smaller batch size for QML

    train_loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ds_va, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(ds_te, batch_size=batch_size, shuffle=False)

    # 3. Model Setup
    # Hybrid QGNN
    # hidden=64 -> mapped to n_qubits=4 via linear layer in QuantumLayer
    model = HybridQGNN(num_tasks=len(tox21_tasks), hidden=64, n_qubits=4, n_layers=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Compute positive weights for BCEWithLogitsLoss
    y_tr_all = np.array(tr['label_parsed'].tolist())
    pos_counts = np.sum(y_tr_all, axis=0)
    neg_counts = len(y_tr_all) - pos_counts
    pos_weights = neg_counts / (pos_counts + 1e-9)
    pos_weights_t = torch.tensor(pos_weights, dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights_t)

    trainer = Trainer(model, optimizer, criterion, device)

    # 4. Training Loop
    print("Starting QML training (this may be slow)...")
    best_pr = -1
    patience = 5 # Lower patience for QML
    bad_epochs = 0
    best_state = None

    # Train for fewer epochs because QML is slow
    for epoch in range(1, 11):
        loss, y_true_tr, y_score_tr = trainer.train_epoch(train_loader)
        roc_tr, pr_tr, _, _ = trainer.compute_metrics(y_true_tr, y_score_tr)

        loss_va, y_true_va, y_score_va = trainer.evaluate(val_loader)
        roc_va, pr_va, _, _ = trainer.compute_metrics(y_true_va, y_score_va)

        print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Tr ROC: {roc_tr:.4f} PR: {pr_tr:.4f} | Va ROC: {roc_va:.4f} PR: {pr_va:.4f}")

        if pr_va > best_pr:
            best_pr = pr_va
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping")
                break

    # 5. Testing
    print("Testing best QML model...")
    if best_state:
        model.load_state_dict(best_state)
    loss_te, y_true_te, y_score_te = trainer.evaluate(test_loader)
    roc_te, pr_te, roc_per_task, pr_per_task = trainer.compute_metrics(y_true_te, y_score_te)

    print(f"TEST GLOBAL | ROC-AUC: {roc_te:.4f} | PR-AUC: {pr_te:.4f}")

    target_idx = tox21_tasks.index('SR-MMP')
    print(f"TEST SR-MMP | ROC-AUC: {roc_per_task[target_idx]:.4f} | PR-AUC: {pr_per_task[target_idx]:.4f}")

if __name__ == "__main__":
    main()
