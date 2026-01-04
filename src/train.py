
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from tqdm import tqdm
from .loss import SupervisedContrastiveLoss

class Trainer:
    def __init__(self, model, optimizer=None, criterion=None, device='cpu', pos_weight=None, contrastive_weight=0.0):
        self.model = model.to(device)
        self.device = device
        self.contrastive_weight = contrastive_weight

        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        if criterion:
            self.criterion = criterion
        else:
            # Loss with imbalance handling
            if pos_weight:
                if isinstance(pos_weight, (int, float)):
                    pw = torch.tensor([pos_weight], device=device)
                else:
                    pw = pos_weight.to(device)
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
            else:
                self.criterion = nn.BCEWithLogitsLoss()

        if contrastive_weight > 0:
            self.contrastive_loss_fn = SupervisedContrastiveLoss()

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        all_y, all_probs = [], []

        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            if self.contrastive_weight > 0:
                # Expecting model to support `return_embeddings`
                embeddings = self.model(batch, return_embeddings=True)
                # Re-run forward for logits (inefficient but safe without refactoring entire forward)
                # Or better: `logits = self.model.final_head(embeddings)` if we knew the structure.
                # Given `HybridGNNVQC` structure, logits = head(embeddings).
                if hasattr(self.model, 'final_head'):
                    logits = self.model.final_head(embeddings).squeeze(-1)
                else:
                    # Fallback for models without separated head (e.g. classical baseline if not updated)
                    logits = self.model(batch)

                ce_loss = self.criterion(logits, batch.y)
                con_loss = self.contrastive_loss_fn(embeddings, batch.y)
                loss = ce_loss + self.contrastive_weight * con_loss
            else:
                logits = self.model(batch)
                loss = self.criterion(logits, batch.y)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            all_y.extend(batch.y.cpu().numpy())
            all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy())

        metrics = self.calculate_metrics(all_y, all_probs)
        metrics['loss'] = total_loss / len(loader)
        return metrics

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        all_y, all_probs = [], []

        for batch in loader:
            batch = batch.to(self.device)
            logits = self.model(batch)
            loss = self.criterion(logits, batch.y)

            total_loss += loss.item()
            all_y.extend(batch.y.cpu().numpy())
            all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy())

        metrics = self.calculate_metrics(all_y, all_probs)
        metrics['loss'] = total_loss / len(loader)
        return metrics

    def calculate_metrics(self, y_true, y_prob):
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        # Handle edge cases with 1 class
        if len(np.unique(y_true)) < 2:
            return {'roc_auc': 0.5, 'pr_auc': 0.0}

        return {
            'roc_auc': roc_auc_score(y_true, y_prob),
            'pr_auc': average_precision_score(y_true, y_prob)
        }

def run_benchmark(model_type='classical', n_qubits=4, epochs=10):
    from .data_loader import get_dataloaders
    from .classical_gnn import ClassicalGNN
    from .hybrid_model import HybridGNNVQC

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device} | Model: {model_type}")

    # Load Data
    train_loader, val_loader, test_loader, pos_weight = get_dataloaders(batch_size=32)

    contrastive_weight = 0.0

    # Init Model
    if model_type == 'classical':
        model = ClassicalGNN(gnn_type='gine', dropout=0.2)
    elif model_type == 'classical_gat':
        model = ClassicalGNN(gnn_type='gat', dropout=0.2)
    elif model_type == 'hybrid_linear':
        model = HybridGNNVQC(n_qubits=n_qubits, reduction='linear', ansatz='strong', gnn_type='gine')
    elif model_type == 'hybrid_fft':
        model = HybridGNNVQC(n_qubits=n_qubits, reduction='fft', ansatz='strong', gnn_type='gine')
    elif model_type == 'mps':
        model = HybridGNNVQC(n_qubits=n_qubits, reduction='linear', ansatz='mps', gnn_type='gine')
    elif model_type == 'hybrid_gat':
        # GAT encoder + Linear Projection + Strong Entangling
        model = HybridGNNVQC(n_qubits=n_qubits, reduction='linear', ansatz='strong', gnn_type='gat', dropout=0.2)
    elif model_type == 'hybrid_reupload':
        # GINE encoder + Linear + Reuploading VQC
        model = HybridGNNVQC(n_qubits=n_qubits, q_layers=4, reduction='linear', ansatz='reupload', gnn_type='gine', dropout=0.2)
    elif model_type == 'hybrid_gat_reupload':
        # Best of both worlds?
        model = HybridGNNVQC(n_qubits=n_qubits, q_layers=4, reduction='linear', ansatz='reupload', gnn_type='gat', dropout=0.2)
    elif model_type == 'hybrid_residual':
        # Hybrid GAT + Reuploading + Residual Connection
        model = HybridGNNVQC(n_qubits=n_qubits, q_layers=4, reduction='linear', ansatz='reupload', gnn_type='gat', dropout=0.2, residual=True)
    elif model_type == 'hybrid_contrastive':
        # Hybrid GAT + Reuploading + Contrastive Loss
        model = HybridGNNVQC(n_qubits=n_qubits, q_layers=4, reduction='linear', ansatz='reupload', gnn_type='gat', dropout=0.2)
        contrastive_weight = 0.1
    else:
        raise ValueError("Unknown model type")

    trainer = Trainer(model, device=device, pos_weight=pos_weight, contrastive_weight=contrastive_weight)

    # Loop
    best_val_roc = 0
    for epoch in range(1, epochs+1):
        tr_metrics = trainer.train_epoch(train_loader)
        va_metrics = trainer.evaluate(val_loader)

        print(f"Ep {epoch} | Tr Loss: {tr_metrics['loss']:.4f} ROC: {tr_metrics['roc_auc']:.3f} | "
              f"Va Loss: {va_metrics['loss']:.4f} ROC: {va_metrics['roc_auc']:.3f}")

        if va_metrics['roc_auc'] > best_val_roc:
            best_val_roc = va_metrics['roc_auc']

    # Final Test
    te_metrics = trainer.evaluate(test_loader)
    print(f"Final Test ROC: {te_metrics['roc_auc']:.4f} PR: {te_metrics['pr_auc']:.4f}")

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else 'classical'
    run_benchmark(mode)
