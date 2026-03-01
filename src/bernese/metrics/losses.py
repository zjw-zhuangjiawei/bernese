# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Custom loss functions for SeqNN models."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MSEUDot(nn.Module):
    """Mean squared error with mean-normalized specificity term.

    This loss combines standard MSE with a dot product term that encourages
    predictions to have similar patterns to targets (mean-normalized).

    Args:
        udot_weight: Weight of the mean-normalized specificity term
    """

    def __init__(self, udot_weight: float = 1.0):
        super().__init__()
        self.udot_weight = udot_weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute MSE with U-dot term.

        Args:
            y_pred: Predicted values
            y_true: Ground truth values

        Returns:
            Loss value
        """
        # Standard MSE term
        mse_term = F.mse_loss(y_pred, y_true, reduction="none")

        # Mean-normalized specificity term
        yn_true = y_true - y_true.mean(dim=-1, keepdim=True)
        yn_pred = y_pred - y_pred.mean(dim=-1, keepdim=True)
        udot_term = -(yn_true * yn_pred).mean(dim=-1)

        # Combine
        loss = mse_term.mean(dim=-1) + self.udot_weight * udot_term

        return loss.mean()


class PoissonKL(nn.Module):
    """Poisson decomposition with KL divergence specificity term.

    Combines Poisson loss with a KL divergence term for multi-task regression.

    Args:
        kl_weight: Weight of the KL specificity term
        epsilon: Small value to avoid log(0)
    """

    def __init__(self, kl_weight: float = 1.0, epsilon: float = 1e-7):
        super().__init__()
        self.kl_weight = kl_weight
        self.epsilon = epsilon

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute Poisson-KL loss.

        Args:
            y_pred: Predicted values (should be positive)
            y_true: Ground truth values

        Returns:
            Loss value
        """
        # Ensure positive predictions
        y_pred = torch.clamp(y_pred, min=self.epsilon)

        # Poisson loss: y_pred - y_true * log(y_pred)
        poisson_term = y_pred - y_true * torch.log(y_pred)

        # Add epsilon to protect against tiny values
        y_true_safe = y_true + self.epsilon
        y_pred_safe = y_pred + self.epsilon

        # Normalize to sum to one
        yn_true = y_true_safe / y_true_safe.sum(dim=-2, keepdim=True)
        yn_pred = y_pred_safe / y_pred_safe.sum(dim=-2, keepdim=True)

        # KL divergence term
        kl_term = F.kl_div(torch.log(yn_pred), yn_true, reduction="none")

        # Weighted combination
        loss = poisson_term + self.kl_weight * kl_term

        return loss.mean()


class PoissonMultinomial(nn.Module):
    """Poisson-Multinomial loss with position weighting.

    Combines Poisson loss for total count prediction with multinomial loss
    for position-specific predictions.

    Args:
        total_weight: Weight of the Poisson total term
        weight_range: Range for position weights (higher = more weight on center)
        weight_exp: Exponent for position weight decay
        epsilon: Small value to avoid log(0)
    """

    def __init__(
        self,
        total_weight: float = 1.0,
        weight_range: float = 1.0,
        weight_exp: int = 4,
        epsilon: float = 1e-7,
    ):
        super().__init__()
        self.total_weight = total_weight
        self.weight_range = weight_range
        self.weight_exp = weight_exp
        self.epsilon = epsilon

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute Poisson-Multinomial loss.

        Args:
            y_pred: Predicted values (batch, length, targets)
            y_true: Ground truth values

        Returns:
            Loss value
        """
        seq_len = y_true.shape[-2]

        # Compute position weights
        if self.weight_range < 1:
            raise ValueError("PoissonMultinomial weight_range must be >= 1")

        if self.weight_range == 1:
            position_weights = torch.ones(1, seq_len, 1, device=y_pred.device)
        else:
            pos_start = -(seq_len / 2 - 0.5)
            pos_end = seq_len / 2 + 0.5
            positions = torch.linspace(pos_start, pos_end, seq_len, device=y_pred.device)
            sigma = -pos_start / (math.log(self.weight_range)) ** (1 / self.weight_exp)
            position_weights = torch.exp(-((positions / sigma) ** self.weight_exp))
            position_weights = position_weights / position_weights.max()
            position_weights = position_weights.unsqueeze(0).unsqueeze(-1)

        # Apply position weights
        y_true_weighted = y_true * position_weights
        y_pred_weighted = y_pred * position_weights

        # Sum across length
        s_true = y_true_weighted.sum(dim=-2)  # (batch, targets)
        s_pred = y_pred_weighted.sum(dim=-2)

        # Poisson loss for total counts
        poisson_term = s_pred - s_true * torch.log(torch.clamp(s_pred, min=self.epsilon))
        poisson_term = poisson_term / position_weights.sum()

        # Add epsilon and normalize
        y_true_safe = y_true + self.epsilon
        y_pred_safe = y_pred + self.epsilon

        p_pred = y_pred_safe / y_pred_safe.sum(dim=-2, keepdim=True)

        # Multinomial loss
        pl_pred = torch.log(p_pred)
        multinomial_term = -(y_true * pl_pred).sum(dim=-2)
        multinomial_term = multinomial_term / position_weights.sum()

        # Combine
        loss = multinomial_term + self.total_weight * poisson_term

        return loss.mean()


def get_loss_function(loss_name: str, **kwargs) -> nn.Module:
    """Factory function to get loss function by name.

    Args:
        loss_name: Name of the loss function
        **kwargs: Additional arguments for the loss

    Returns:
        Loss function module
    """
    loss_name = loss_name.lower()

    losses = {
        "mse": nn.MSELoss,
        "bce": nn.BCEWithLogitsLoss,
        "poisson": nn.PoissonNLLLoss,
        "mse_udot": MSEUDot,
        "poisson_kl": PoissonKL,
        "poisson_multinomial": PoissonMultinomial,
    }

    if loss_name not in losses:
        raise ValueError(f"Unknown loss: {loss_name}. Available: {list(losses.keys())}")

    # Get default kwargs based on loss type
    if loss_name == "mse_udot":
        kwargs.setdefault("udot_weight", kwargs.get("spec_weight", 1.0))
    elif loss_name == "poisson_kl":
        kwargs.setdefault("kl_weight", kwargs.get("spec_weight", 1.0))
    elif loss_name == "poisson_multinomial":
        kwargs.setdefault("total_weight", kwargs.get("total_weight", 1.0))
        kwargs.setdefault("weight_range", kwargs.get("weight_range", 1.0))
        kwargs.setdefault("weight_exp", kwargs.get("weight_exp", 4))

    return losses[loss_name](**kwargs)


# Convenience aliases
__all__ = [
    "MSEUDot",
    "PoissonKL",
    "PoissonMultinomial",
    "get_loss_function",
]
