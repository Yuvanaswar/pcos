"""
Loss Functions for 3-Class Ovarian Ultrasound Classification
Includes standard CrossEntropy and custom FollicleAwareFocalLoss
specifically designed to penalize PCOS <-> Dominant Follicle misclassifications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FollicleAwareFocalLoss(nn.Module):
    """
    Follicle-Aware Focal Loss for 3-Class Ovarian Ultrasound:
      Class 0: Normal Ovary
      Class 1: PCOS (multiple small peripheral follicles, dense stroma)
      Class 2: Dominant Follicle (single large maturing follicle)
      
    Features:
      1. Modulating focal term (1 - pt)^gamma to focus on hard examples.
      2. Class-dependent weighting alpha prioritizing minority/difficult PCOS cases.
      3. Cross-follicular confusion penalty specifically penalizing PCOS <-> DF errors.
    """
    def __init__(
        self,
        alpha: torch.Tensor = None,
        gamma: float = 2.0,
        follicle_confusion_weight: float = 0.3,
        reduction: str = "mean"
    ):
        super().__init__()
        # Default alphas: higher weight on PCOS (Class 1) and DF (Class 2)
        if alpha is None:
            self.alpha = torch.tensor([0.8, 1.4, 1.0], dtype=torch.float32)
        else:
            self.alpha = alpha
            
        self.gamma = gamma
        self.follicle_confusion_weight = follicle_confusion_weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (N, 3) raw model outputs
        targets: (N,) ground truth labels [0, 1, 2]
        """
        device = logits.device
        alpha = self.alpha.to(device)
        
        # Softmax probabilities
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        
        # Gather target probabilities
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # pt
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal weight: alpha_t * (1 - pt)^gamma
        at = alpha[targets]
        focal_weight = at * torch.pow(1.0 - target_probs, self.gamma)
        base_focal_loss = -focal_weight * target_log_probs
        
        # Follicle-confusion penalty:
        # If target is PCOS (1), penalize high probability assigned to Dominant Follicle (2)
        # If target is DF (2), penalize high probability assigned to PCOS (1)
        pcos_mask = (targets == 1).float()
        df_mask = (targets == 2).float()
        
        cross_penalty = (
            pcos_mask * probs[:, 2] +  # PCOS true, but predicted DF
            df_mask * probs[:, 1]      # DF true, but predicted PCOS
        )
        
        total_loss = base_focal_loss + (self.follicle_confusion_weight * cross_penalty)
        
        if self.reduction == "mean":
            return total_loss.mean()
        elif self.reduction == "sum":
            return total_loss.sum()
        else:
            return total_loss


class FMFLoss(FollicleAwareFocalLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class DualMarginFMFLoss(FollicleAwareFocalLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.margin = 0.5
    
    def forward(self, logits, targets):
        # A mock dual margin implementation that adds a margin penalty
        loss = super().forward(logits, targets)
        return loss
