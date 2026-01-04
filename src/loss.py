
import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: hidden vector of shape [batch_size, n_dim]
            labels: ground truth of shape [batch_size]
        """
        # Normalize features
        features = F.normalize(features, p=2, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Create mask for same-class positives
        # labels: (batch_size) -> (batch_size, 1)
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()

        # Remove self-contrast
        logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=mask.device)
        mask = mask * logits_mask

        # Compute log_prob
        # exp(sim)
        exp_sim = torch.exp(similarity_matrix) * logits_mask

        # Sum of exp(sim) for all negatives and positives (denominator)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)

        # Mean log-likelihood for positive pairs
        # Sum over positives / count of positives
        # Avoid division by zero
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        # Loss
        loss = -mean_log_prob_pos
        return loss.mean()
