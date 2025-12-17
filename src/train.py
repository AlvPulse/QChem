
import os
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

class Trainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        all_y_true = []
        all_y_scores = []

        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(batch)

            # Direct loss calculation assuming full labels
            # logits: [batch, num_tasks], batch.y: [batch, num_tasks]
            # BCEWithLogitsLoss with pos_weight=[num_tasks] handles this correctly
            loss = self.criterion(logits, batch.y)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            all_y_true.append(batch.y.detach().cpu().numpy())
            all_y_scores.append(torch.sigmoid(logits).detach().cpu().numpy())

        avg_loss = total_loss / len(loader)
        y_true = np.concatenate(all_y_true)
        y_scores = np.concatenate(all_y_scores)

        return avg_loss, y_true, y_scores

    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        all_y_true = []
        all_y_scores = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                logits = self.model(batch)
                loss = self.criterion(logits, batch.y)
                total_loss += loss.item()

                all_y_true.append(batch.y.detach().cpu().numpy())
                all_y_scores.append(torch.sigmoid(logits).detach().cpu().numpy())

        avg_loss = total_loss / len(loader)
        y_true = np.concatenate(all_y_true)
        y_scores = np.concatenate(all_y_scores)

        return avg_loss, y_true, y_scores

    def compute_metrics(self, y_true, y_scores):
        # y_true: [N, num_tasks]
        # Calculate per task and average
        roc_list = []
        pr_list = []
        num_tasks = y_true.shape[1]

        for i in range(num_tasks):
            # Check if valid labels exist for this task
            yt = y_true[:, i]
            ys = y_scores[:, i]
            # Ignore missing labels if we supported them, but here we assume filled or check.
            # Only calculate if we have both classes
            if len(np.unique(yt)) > 1:
                roc_list.append(roc_auc_score(yt, ys))
                pr_list.append(average_precision_score(yt, ys))
            else:
                # If only one class is present in the split for this task,
                # we can't compute AUC.
                pass

        mean_roc = np.mean(roc_list) if roc_list else 0.0
        mean_pr = np.mean(pr_list) if pr_list else 0.0

        # Fill missing tasks with 0 or nan for per-task list to align indices
        roc_list_aligned = []
        pr_list_aligned = []
        for i in range(num_tasks):
            yt = y_true[:, i]
            ys = y_scores[:, i]
            if len(np.unique(yt)) > 1:
                roc_list_aligned.append(roc_auc_score(yt, ys))
                pr_list_aligned.append(average_precision_score(yt, ys))
            else:
                roc_list_aligned.append(0.5) # Default/Neutral
                pr_list_aligned.append(0.0)

        return mean_roc, mean_pr, roc_list_aligned, pr_list_aligned
