
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score
import numpy as np
from tqdm import tqdm
from .loss import MaskedBCEWithLogitsLoss, MultiTaskSupervisedContrastiveLoss

class Trainer:
    def __init__(self, model, device='cpu', pos_weight=None, alpha=0.1):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.alpha = alpha

        # Loss with imbalance handling
        if pos_weight is not None:
            # pos_weight is a tensor of shape (12,)
            self.criterion_bce = MaskedBCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.criterion_bce = MaskedBCEWithLogitsLoss()

        self.criterion_sup = MultiTaskSupervisedContrastiveLoss(temperature=0.07)

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        all_y, all_probs = [], []

        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            out = self.model(batch)

            # Check if model returns (logits, latent) or just logits
            if isinstance(out, tuple):
                logits, latent = out
                loss_bce = self.criterion_bce(logits, batch.y)
                loss_sup = self.criterion_sup(latent, batch.y)
                loss = loss_bce + self.alpha * loss_sup
            else:
                logits = out
                loss = self.criterion_bce(logits, batch.y)

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
            out = self.model(batch)

            if isinstance(out, tuple):
                logits, _ = out
            else:
                logits = out

            loss = self.criterion_bce(logits, batch.y)

            total_loss += loss.item()
            all_y.extend(batch.y.cpu().numpy())
            all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy())

        metrics = self.calculate_metrics(all_y, all_probs)
        metrics['loss'] = total_loss / len(loader)
        return metrics

    def calculate_metrics(self, y_true, y_prob):
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        # y_true: (N, num_tasks), y_prob: (N, num_tasks)

        roc_aucs = []
        pr_aucs = []
        brier_scores = []
        f1_scores = []

        n_tasks = y_true.shape[1]
        for i in range(n_tasks):
            y_t = y_true[:, i]
            p_t = y_prob[:, i]

            # Filter NaNs
            valid_mask = ~np.isnan(y_t)
            if valid_mask.sum() < 2:
                continue

            y_t = y_t[valid_mask]
            p_t = p_t[valid_mask]

            # Check if we have both classes
            if len(np.unique(y_t)) < 2:
                continue

            try:
                roc_aucs.append(roc_auc_score(y_t, p_t))
                pr_aucs.append(average_precision_score(y_t, p_t))
                brier_scores.append(brier_score_loss(y_t, p_t))

                # F1 needs binary predictions
                y_pred = (p_t > 0.5).astype(int)
                f1_scores.append(f1_score(y_t, y_pred, zero_division=0))
            except ValueError:
                pass

        if len(roc_aucs) == 0:
            return {'roc_auc': 0.5, 'pr_auc': 0.0, 'brier': 0.0, 'f1': 0.0}

        return {
            'roc_auc': np.mean(roc_aucs),
            'pr_auc': np.mean(pr_aucs),
            'brier': np.mean(brier_scores),
            'f1': np.mean(f1_scores)
        }

def run_benchmark(model_type='classical', n_qubits=4, epochs=10, batch_size=32):
    from .data_loader import get_dataloaders
    from .classical_gnn import ClassicalGNN
    from .hybrid_model import HybridGNNVQC, HybridEnsembleVQC

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device} | Model: {model_type}")

    # Load Data
    train_loader, val_loader, test_loader, pos_weight = get_dataloaders(batch_size=batch_size)

    # Init Model
    if model_type == 'classical':
        # ClassicalGNN's head output dimension needs to be 12 now.
        # But ClassicalGNN init takes 'out_dim'.
        model = ClassicalGNN(gnn_type='gine', dropout=0.2, out_dim=12)
    elif model_type == 'classical_gat':
        model = ClassicalGNN(gnn_type='gat', dropout=0.2, out_dim=12)
    elif model_type == 'hybrid_ensemble':
        # Our new model
        model = HybridEnsembleVQC(
            n_estimators=4,
            n_qubits_per_est=4,
            q_layers=2,
            ansatz='hea',
            gnn_type='gine',
            dropout=0.2,
            split_input=True,
            n_outputs=12
        )
    elif model_type == 'hybrid_ensemble_reupload':
        # Reuploading ensemble
        model = HybridEnsembleVQC(
            n_estimators=4,
            n_qubits_per_est=4,
            q_layers=4,
            ansatz='reupload',
            gnn_type='gine',
            dropout=0.2,
            split_input=True,
            n_outputs=12
        )
    else:
        # Fallback to old models, but they output 1 dim which is incompatible with multi-task
        # We should probably adapt them or raise error.
        raise ValueError(f"Model type {model_type} not supported for multi-task benchmark")

    trainer = Trainer(model, device, pos_weight)

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
