# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Metrics for evaluating SeqNN models."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from sklearn.metrics import roc_auc_score, average_precision_score


class PearsonR(nn.Module):
    """Pearson correlation coefficient metric for multi-task regression.

    Args:
        num_targets: Number of prediction targets
        summarize: Whether to return mean correlation across targets
    """

    def __init__(self, num_targets: int, summarize: bool = True):
        super().__init__()
        self.num_targets = num_targets
        self.summarize = summarize

        # Accumulator variables
        self.register_buffer("_product", torch.zeros(num_targets))
        self.register_buffer("_true_sum", torch.zeros(num_targets))
        self.register_buffer("_true_sumsq", torch.zeros(num_targets))
        self.register_buffer("_pred_sum", torch.zeros(num_targets))
        self.register_buffer("_pred_sumsq", torch.zeros(num_targets))
        self.register_buffer("_count", torch.zeros(num_targets))

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Update metric state with batch of predictions.

        Args:
            y_true: Ground truth targets (batch, ..., num_targets)
            y_pred: Predicted targets (batch, ..., num_targets)
        """
        # Flatten batch dimensions
        y_true = y_true.reshape(-1, self.num_targets)
        y_pred = y_pred.reshape(-1, self.num_targets)

        # Update accumulators
        self._product += torch.sum(y_true * y_pred, dim=0)
        self._true_sum += torch.sum(y_true, dim=0)
        self._true_sumsq += torch.sum(y_true**2, dim=0)
        self._pred_sum += torch.sum(y_pred, dim=0)
        self._pred_sumsq += torch.sum(y_pred**2, dim=0)
        self._count += y_true.shape[0]

        return self.compute()

    def compute(self) -> torch.Tensor:
        """Compute PearsonR from accumulated state."""
        true_mean = self._true_sum / self._count
        pred_mean = self._pred_sum / self._count

        covariance = (
            self._product
            - true_mean * self._pred_sum
            - pred_mean * self._true_sum
            + self._count * true_mean * pred_mean
        )

        true_var = self._true_sumsq - self._count * true_mean**2
        pred_var = self._pred_sumsq - self._count * pred_mean**2

        # Avoid division by zero
        pred_var = torch.where(pred_var > 1e-12, pred_var, torch.ones_like(pred_var) * float("inf"))

        correlation = covariance / torch.sqrt(true_var * pred_var)

        if self.summarize:
            return torch.mean(correlation)
        return correlation

    def reset(self):
        """Reset metric state."""
        self._product.zero_()
        self._true_sum.zero_()
        self._true_sumsq.zero_()
        self._pred_sum.zero_()
        self._pred_sumsq.zero_()
        self._count.zero_()


class R2(nn.Module):
    """R-squared (coefficient of determination) metric.

    Args:
        num_targets: Number of prediction targets
        summarize: Whether to return mean R2 across targets
    """

    def __init__(self, num_targets: int, summarize: bool = True):
        super().__init__()
        self.num_targets = num_targets
        self.summarize = summarize

        # Accumulator variables
        self.register_buffer("_true_sum", torch.zeros(num_targets))
        self.register_buffer("_true_sumsq", torch.zeros(num_targets))
        self.register_buffer("_product", torch.zeros(num_targets))
        self.register_buffer("_pred_sumsq", torch.zeros(num_targets))
        self.register_buffer("_count", torch.zeros(num_targets))

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Update metric state with batch of predictions."""
        y_true = y_true.reshape(-1, self.num_targets)
        y_pred = y_pred.reshape(-1, self.num_targets)

        self._true_sum += torch.sum(y_true, dim=0)
        self._true_sumsq += torch.sum(y_true**2, dim=0)
        self._product += torch.sum(y_true * y_pred, dim=0)
        self._pred_sumsq += torch.sum(y_pred**2, dim=0)
        self._count += y_true.shape[0]

        return self.compute()

    def compute(self) -> torch.Tensor:
        """Compute R2 from accumulated state."""
        true_mean = self._true_sum / self._count
        total = self._true_sumsq - self._count * true_mean**2

        resid = self._pred_sumsq - 2 * self._product + self._true_sumsq

        r2 = 1 - resid / total

        if self.summarize:
            return torch.mean(r2)
        return r2

    def reset(self):
        """Reset metric state."""
        self._true_sum.zero_()
        self._true_sumsq.zero_()
        self._product.zero_()
        self._pred_sumsq.zero_()
        self._count.zero_()


class SeqAUC(nn.Module):
    """AUC metric for binary classification tasks.

    Supports both ROC-AUC and PR-AUC for multi-task binary classification.
    Uses sklearn for computation but works with PyTorch tensors.

    Args:
        curve: 'ROC' or 'PR' (precision-recall)
        summarize: Whether to return mean AUC across targets
    """

    def __init__(self, curve: str = "ROC", summarize: bool = True):
        super().__init__()
        self.curve = curve
        self.summarize = summarize

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Compute AUC metric.

        Args:
            y_true: Ground truth binary labels (batch, length, targets)
            y_pred: Predicted probabilities (batch, length, targets)

        Returns:
            AUC score(s)
        """
        # Flatten batch and sequence dimensions
        y_true_flat = y_true.reshape(-1, y_true.shape[-1])
        y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])

        # Compute AUC for each target
        aucs = []
        for i in range(y_true.shape[-1]):
            yt = y_true_flat[:, i].cpu().numpy()
            yp = y_pred_flat[:, i].cpu().numpy()

            # Skip targets with only one class
            if len(np.unique(yt)) < 2:
                continue

            try:
                if self.curve == "ROC":
                    auc = roc_auc_score(yt, yp)
                else:  # PR curve
                    auc = average_precision_score(yt, yp)
                aucs.append(auc)
            except ValueError:
                # Handle edge cases
                continue

        if len(aucs) == 0:
            return torch.tensor(0.0, device=y_true.device)

        auc_tensor = torch.tensor(aucs, device=y_true.device)

        if self.summarize:
            return auc_tensor.mean()
        return auc_tensor


def get_metric(metric_name: str, num_targets: int = 1, **kwargs) -> nn.Module:
    """Factory function to get metric by name.

    Args:
        metric_name: Name of the metric
        num_targets: Number of targets (for per-target metrics)
        **kwargs: Additional arguments

    Returns:
        Metric module
    """
    metric_name = metric_name.lower()

    metrics = {
        "pearsonr": PearsonR,
        "r2": R2,
        "auc": SeqAUC,
        "auroc": lambda n, s: SeqAUC("ROC", s),
        "auprc": lambda n, s: SeqAUC("PR", s),
    }

    if metric_name not in metrics:
        raise ValueError(f"Unknown metric: {metric_name}. Available: {list(metrics.keys())}")

    return metrics[metric_name](num_targets, kwargs.get("summarize", True))


# Convenience aliases
__all__ = [
    "PearsonR",
    "R2",
    "SeqAUC",
    "get_metric",
]
