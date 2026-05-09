import torch
import numpy as np
import random
from src.data.loader import load_and_preprocess_data, scaffold_split, set_seed
from src.features.graph import ToxDataset
from src.models.structured_kernel import HybridStructuredQGNN
from src.quantum.diagnostics import compute_gram_matrix, analyze_kernel_variance, plot_gram_matrix
from src.train import Trainer
from torch_geometric.loader import DataLoader
import torch.nn as nn

def subsample_dataset(dataset, max_samples=500):
    """Helper to reduce dataset size for manageable QML runtimes on laptop."""
    if len(dataset) > max_samples:
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        return torch.utils.data.Subset(dataset, indices[:max_samples])
    return dataset

def main():
    set_seed(42)
    device = torch.device('cpu') 
    
    # 1. Load Data
    print("Loading data...")
    tox21_tasks=['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
                 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']

    df = load_and_preprocess_data("EDA_dataset.csv", tox21_tasks)
    tr, va, te = scaffold_split(df)

    ds_tr = ToxDataset(tr)
    ds_va = ToxDataset(va)
    ds_te = ToxDataset(te)

    # Subsample for overnight run! (Crucial for quantum simulation)
    ds_tr = subsample_dataset(ds_tr, max_samples=1500)
    ds_va = subsample_dataset(ds_va, max_samples=300)
    ds_te = subsample_dataset(ds_te, max_samples=300)

    batch_size = 32
    train_loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ds_va, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(ds_te, batch_size=batch_size, shuffle=False)

    # 2. Model Setup
    model = HybridStructuredQGNN(num_tasks=len(tox21_tasks), hidden=64, n_qubits=5).to(device)
    # We only optimize the classical parameters (GNN + Linear layers). The QFM is fixed!
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss() # Add your pos_weights calculation here as before

    # Initialize Trainer (You will need to update Trainer to handle the tuple return: logits, q_features)
    trainer = Trainer(model, optimizer, criterion, device)

    # 3. Pre-Training Diagnostics (Check initial state of the Quantum Feature Map)
    print("Running initial quantum kernel diagnostics on validation batch...")
    model.eval()
    sample_batch = next(iter(val_loader)).to(device)
    with torch.no_grad():
        _, initial_q_features = model(sample_batch)
    
    gram_mat = compute_gram_matrix(initial_q_features)
    analyze_kernel_variance(gram_mat)
    plot_gram_matrix(gram_mat, save_path="initial_kernel.png")

    # 4. Training Loop
    print("Starting Hybrid training...")
    # ... (Keep your standard training loop here) ...
    # IMPORTANT: Ensure your trainer unpacks `logits, _ = model(data)`

if __name__ == "__main__":
    main()