# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Custom loss functions for SeqNN models.

This module provides Keras 3 native loss functions. All losses inherit from
keras.losses.Loss and use keras.ops for tensor operations, ensuring
compatibility with the Keras 3 training workflow and JIT compilation.
"""

import math

import keras
from keras import losses as keras_losses
from keras import ops


class MSEUDot(keras_losses.Loss):
    """Mean squared error with mean-normalized specificity term.

    This loss combines standard MSE with a dot product term that encourages
    predictions to have similar patterns to targets (mean-normalized).

    Args:
        udot_weight: Weight of the mean-normalized specificity term
        reduction: Reduction type for the loss
    """

    def __init__(self, udot_weight: float = 1.0, reduction: str = "sum_over_batch_size"):
        super().__init__(reduction=reduction, name="mse_udot")
        self.udot_weight = udot_weight

    def call(self, y_true, y_pred):
        """Compute MSE with U-dot term.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            Loss value
        """
        # Standard MSE term
        mse_term = ops.square(y_true - y_pred)

        # Mean-normalized specificity term
        yn_true = y_true - ops.mean(y_true, axis=-1, keepdims=True)
        yn_pred = y_pred - ops.mean(y_pred, axis=-1, keepdims=True)
        udot_term = -ops.mean(yn_true * yn_pred, axis=-1)

        # Combine
        loss = ops.mean(mse_term, axis=-1) + self.udot_weight * udot_term

        return ops.mean(loss)


class PoissonKL(keras_losses.Loss):
    """Poisson decomposition with KL divergence specificity term.

    Combines Poisson loss with a KL divergence term for multi-task regression.

    Args:
        kl_weight: Weight of the KL specificity term
        epsilon: Small value to avoid log(0)
        reduction: Reduction type for the loss
    """

    def __init__(
        self, kl_weight: float = 1.0, epsilon: float = 1e-7, reduction: str = "sum_over_batch_size"
    ):
        super().__init__(reduction=reduction, name="poisson_kl")
        self.kl_weight = kl_weight
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        """Compute Poisson-KL loss.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values (should be positive)

        Returns:
            Loss value
        """
        # Ensure positive predictions
        y_pred = ops.clip(y_pred, minval=self.epsilon)

        # Poisson loss: y_pred - y_true * log(y_pred)
        poisson_term = y_pred - y_true * ops.log(y_pred)

        # Add epsilon to protect against tiny values
        y_true_safe = y_true + self.epsilon
        y_pred_safe = y_pred + self.epsilon

        # Normalize to sum to one (sum over length dimension, axis=-2)
        yn_true = y_true_safe / ops.sum(y_true_safe, axis=-2, keepdims=True)
        yn_pred = y_pred_safe / ops.sum(y_pred_safe, axis=-2, keepdims=True)

        # KL divergence term: y_true * log(y_true/y_pred) = y_true * (log(y_true) - log(y_pred))
        kl_term = yn_true * (ops.log(yn_true) - ops.log(yn_pred))

        # Weighted combination
        loss = poisson_term + self.kl_weight * kl_term

        return ops.mean(loss)


class PoissonMultinomial(keras_losses.Loss):
    """Poisson-Multinomial loss with position weighting.

    Combines Poisson loss for total count prediction with multinomial loss
    for position-specific predictions.

    Args:
        total_weight: Weight of the Poisson total term
        weight_range: Range for position weights (higher = more weight on center)
        weight_exp: Exponent for position weight decay
        epsilon: Small value to avoid log(0)
        reduction: Reduction type for the loss
    """

    def __init__(
        self,
        total_weight: float = 1.0,
        weight_range: float = 1.0,
        weight_exp: int = 4,
        epsilon: float = 1e-7,
        reduction: str = "sum_over_batch_size",
    ):
        super().__init__(reduction=reduction, name="poisson_multinomial")
        self.total_weight = total_weight
        self.weight_range = weight_range
        self.weight_exp = weight_exp
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        """Compute Poisson-Multinomial loss.

        Args:
            y_true: Ground truth values (batch, length, targets)
            y_pred: Predicted values (batch, length, targets)

        Returns:
            Loss value
        """
        seq_len = ops.shape(y_true)[-2]

        # Compute position weights
        if self.weight_range < 1:
            raise ValueError("PoissonMultinomial weight_range must be >= 1")

        if self.weight_range == 1:
            position_weights = ops.ones((1, seq_len, 1))
        else:
            pos_start = -(seq_len / 2 - 0.5)
            pos_end = seq_len / 2 + 0.5
            positions = ops.linspace(pos_start, pos_end, seq_len)
            sigma = -pos_start / (math.log(self.weight_range)) ** (1 / self.weight_exp)
            position_weights = ops.exp(-((positions / sigma) ** self.weight_exp))
            position_weights = position_weights / ops.max(position_weights)
            position_weights = ops.reshape(position_weights, (1, seq_len, 1))

        # Apply position weights
        y_true_weighted = y_true * position_weights
        y_pred_weighted = y_pred * position_weights

        # Sum across length
        s_true = ops.sum(y_true_weighted, axis=-2)  # (batch, targets)
        s_pred = ops.sum(y_pred_weighted, axis=-2)

        # Poisson loss for total counts
        poisson_term = s_pred - s_true * ops.log(ops.clip(s_pred, minval=self.epsilon))
        poisson_term = poisson_term / ops.sum(position_weights)

        # Add epsilon and normalize
        y_true_safe = y_true + self.epsilon
        y_pred_safe = y_pred + self.epsilon

        p_pred = y_pred_safe / ops.sum(y_pred_safe, axis=-2, keepdims=True)

        # Multinomial loss
        pl_pred = ops.log(p_pred)
        multinomial_term = -ops.sum(y_true * pl_pred, axis=-2)
        multinomial_term = multinomial_term / ops.sum(position_weights)

        # Combine
        loss = multinomial_term + self.total_weight * poisson_term

        return ops.mean(loss)


# Aliases for Keras 3 compatible losses (wrappers around keras.losses)
class _MSELoss:
    """MSE loss compatible with Keras 3 training workflow."""

    def __init__(self, reduction: str = "sum_over_batch_size"):
        self.reduction = reduction
        self._loss = keras_losses.MeanSquaredError(reduction=reduction)

    def __call__(self, y_true, y_pred):
        return self._loss(y_true, y_pred)


class _PoissonLoss:
    """Poisson loss compatible with Keras 3 training workflow."""

    def __init__(self, reduction: str = "sum_over_batch_size"):
        self.reduction = reduction
        self._loss = keras_losses.Poisson(reduction=reduction)

    def __call__(self, y_true, y_pred):
        return self._loss(y_true, y_pred)


class _BCELoss:
    """Binary cross-entropy loss compatible with Keras 3 training workflow."""

    def __init__(self, reduction: str = "sum_over_batch_size", pos_weight=None):
        self.reduction = reduction
        self.pos_weight = pos_weight
        # Note: BCEWithLogitsLoss is not directly available in keras.losses
        # We use BinaryCrossentropy with from_logits=True
        self._loss = keras_losses.BinaryCrossentropy(from_logits=True, reduction=reduction, axis=-1)

    def __call__(self, y_true, y_pred):
        return self._loss(y_true, y_pred)


def get_loss_function(loss_name: str, **kwargs):
    """Factory function to get loss function by name.

    This function returns Keras 3 compatible loss functions.

    Args:
        loss_name: Name of the loss function
        **kwargs: Additional arguments for the loss

    Returns:
        Loss function compatible with Keras 3
    """
    loss_name = loss_name.lower()
    reduction = kwargs.get("reduction", "sum_over_batch_size")

    # Map to Keras 3 compatible losses
    if loss_name == "mse":
        return _MSELoss(reduction=reduction)
    elif loss_name == "bce":
        pos_weight = kwargs.get("pos_weight")
        return _BCELoss(reduction=reduction, pos_weight=pos_weight)
    elif loss_name == "poisson":
        return _PoissonLoss(reduction=reduction)
    elif loss_name == "mse_udot":
        udot_weight = kwargs.get("udot_weight", kwargs.get("spec_weight", 1.0))
        return MSEUDot(udot_weight=udot_weight, reduction=reduction)
    elif loss_name == "poisson_kl":
        kl_weight = kwargs.get("kl_weight", kwargs.get("spec_weight", 1.0))
        epsilon = kwargs.get("epsilon", 1e-7)
        return PoissonKL(kl_weight=kl_weight, epsilon=epsilon, reduction=reduction)
    elif loss_name == "poisson_multinomial":
        total_weight = kwargs.get("total_weight", 1.0)
        weight_range = kwargs.get("weight_range", 1.0)
        weight_exp = kwargs.get("weight_exp", 4)
        epsilon = kwargs.get("epsilon", 1e-7)
        return PoissonMultinomial(
            total_weight=total_weight,
            weight_range=weight_range,
            weight_exp=weight_exp,
            epsilon=epsilon,
            reduction=reduction,
        )
    else:
        raise ValueError(
            f"Unknown loss: {loss_name}. Available: mse, bce, poisson, mse_udot, poisson_kl, poisson_multinomial"
        )


# Convenience aliases
__all__ = [
    "MSEUDot",
    "PoissonKL",
    "PoissonMultinomial",
    "get_loss_function",
    "_MSELoss",
    "_PoissonLoss",
    "_BCELoss",
]
