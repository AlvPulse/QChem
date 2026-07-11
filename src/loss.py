
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedMultiTaskFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=0.0):
        super().__init__()
        # alpha here refers to the positive class weight (pos_weight) passed down.
        # gamma is the focal modulating exponent; gamma=0 -> plain (weighted) BCE.
        # NOTE: gamma defaults to 0. Stacking focal modulation (1-p_t)^gamma ON TOP of a
        # large pos_weight double-counts class imbalance and was collapsing the multi-task
        # outputs toward uninformative (base-rate) rankings -> ROC ~ 0.5 everywhere.
        self.alpha_weight = alpha
        self.gamma = gamma

    def forward(self, logits, target):
        # target shape: (B, num_tasks), logits shape: (B, num_tasks)
        # target may contain NaNs
        mask = ~torch.isnan(target)
        target_clean = torch.where(mask, target, torch.zeros_like(target))

        # Compute standard BCE loss per element without reduction
        if self.alpha_weight is not None:
            if self.alpha_weight.device != logits.device:
                self.alpha_weight = self.alpha_weight.to(logits.device)
            # Apply pos_weight via BCEWithLogitsLoss
            bce_loss = F.binary_cross_entropy_with_logits(
                logits, target_clean, reduction='none', pos_weight=self.alpha_weight
            )
        else:
            bce_loss = F.binary_cross_entropy_with_logits(
                logits, target_clean, reduction='none'
            )

        # Optional focal modulation. With gamma=0 this factor is identically 1.0 and the
        # loss is exactly masked (weighted) BCE.
        if self.gamma and self.gamma > 0:
            probs = torch.sigmoid(logits)
            p_t = target_clean * probs + (1 - target_clean) * (1 - probs)
            bce_loss = torch.pow(1.0 - p_t, self.gamma) * bce_loss

        # Apply missing-label mask
        masked_loss = bce_loss * mask.float()

        # Average over valid elements
        num_valid = mask.sum()
        if num_valid > 0:
            return masked_loss.sum() / num_valid
        else:
            # No valid labels in this batch: return a zero that stays connected to the
            # graph (a detached tensor would silently break backprop for the step).
            return logits.sum() * 0.0

# Keep the original wrapper name for compatibility with other files.
# Defaults to plain masked BCE (gamma=0); pass gamma>0 to opt back into focal loss.
class MaskedBCEWithLogitsLoss(MaskedMultiTaskFocalLoss):
    def __init__(self, pos_weight=None, gamma=0.0):
        super().__init__(alpha=pos_weight, gamma=gamma)

class MultiTaskSupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features: (B, dim) - normalized or not? Usually expected to be normalized for cosine sim.
        # labels: (B, 12) - can have NaNs

        # Normalize features
        features = F.normalize(features, dim=1)

        device = features.device
        batch_size = features.shape[0]
        n_tasks = labels.shape[1]

        total_loss = 0
        valid_tasks = 0

        # Create similarity matrix for features: (B, B)
        # sim_ij = z_i . z_j / temp
        sim_matrix = torch.div(torch.matmul(features, features.T), self.temperature)

        # For numerical stability
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_max.detach()

        # Exp
        exp_sim = torch.exp(sim_matrix)

        # Mask for self-contrast (exclude diagonal)
        logits_mask = torch.ones_like(sim_matrix) - torch.eye(batch_size, device=device)

        # Denominator: sum over all other samples
        # sum_{a \in A(i)} exp(z_i . z_a / tau)
        # We sum exp_sim * logits_mask
        denom = (exp_sim * logits_mask).sum(dim=1, keepdim=True) # (B, 1)

        # Iterate over tasks
        for t in range(n_tasks):
            y_t = labels[:, t] # (B,)
            mask_t = ~torch.isnan(y_t) # (B,)

            # If not enough valid samples, skip
            if mask_t.sum() < 2:
                continue

            # We only consider samples valid for this task
            # But the contrastive loss usually considers the whole batch in the denominator.
            # Let's keep the denominator as is (all samples against all samples),
            # but only compute the numerator (positives) for valid samples that have valid pairs.

            # Valid labels for this task
            # 1 if same class, 0 otherwise.
            # Also both must be valid.

            # mask_matrix[i, j] = 1 if mask[i] and mask[j]
            valid_pair_mask = mask_t.unsqueeze(0) * mask_t.unsqueeze(1) # (B, B)

            # label_match[i, j] = 1 if y[i] == y[j]
            # y_t contains 0, 1, nan. NaNs handled by valid_pair_mask.
            y_clean = torch.where(mask_t, y_t, torch.zeros_like(y_t))
            label_match = y_clean.unsqueeze(0) == y_clean.unsqueeze(1) # (B, B)

            # Positive mask: valid pair AND same label AND not self
            pos_mask = valid_pair_mask & label_match & (logits_mask.bool())

            # If a sample i has no positives, it contributes 0 loss
            # We compute log_prob = log(exp_sim / denom) = sim - log(denom)
            log_prob = sim_matrix - torch.log(denom + 1e-6)

            # Sum log_prob over positives
            # For each anchor i, we want: -1/|P(i)| * sum_{p \in P(i)} log_prob[i, p]

            # Numerator term per anchor
            # (log_prob * pos_mask).sum(dim=1)

            # Count positives per anchor
            pos_count = pos_mask.sum(dim=1) # (B,)

            # Loss per anchor
            # Avoid division by zero
            loss_anchor = -(log_prob * pos_mask.float()).sum(dim=1) / (pos_count + 1e-6)

            # Only count anchors that had at least one positive
            valid_anchors = pos_count > 0
            if valid_anchors.sum() > 0:
                loss_task = loss_anchor[valid_anchors].mean()
                total_loss += loss_task
                valid_tasks += 1

        if valid_tasks > 0:
            return total_loss / valid_tasks
        else:
            # Zero that stays connected to the graph (avoids a detached-tensor no-op).
            return features.sum() * 0.0
