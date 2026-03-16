# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Type-safe Pydantic models for Trainer configuration.

This module provides fully typed configuration models for the Trainer,
replacing runtime dict-based configuration with compile-time type checking.
Reference: bernese.models.config for pattern design.
"""

import json
from pathlib import Path
from typing import Literal, Optional, Union, Annotated

from pydantic import BaseModel, Field


###############################################################################
# Optimizer Configurations
###############################################################################


class SGDConfig(BaseModel):
    """SGD Optimizer configuration.

    Corresponds to: keras.optimizers.SGD
    """

    name: Literal["sgd"] = "sgd"
    learning_rate: float = 0.01
    momentum: float = 0.0
    weight_decay: float = 0.0
    nesterov: bool = False

    model_config = {"extra": "forbid"}


class AdamConfig(BaseModel):
    """Adam Optimizer configuration.

    Corresponds to: keras.optimizers.Adam
    """

    name: Literal["adam"] = "adam"
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-7

    model_config = {"extra": "forbid"}


class AdamWConfig(BaseModel):
    """AdamW Optimizer configuration.

    Corresponds to: keras.optimizers.AdamW
    """

    name: Literal["adamw"] = "adamw"
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-7

    model_config = {"extra": "forbid"}


# Discriminated union for optimizer configuration
OptimizerConfig = Annotated[
    Union[SGDConfig, AdamConfig, AdamWConfig],
    Field(discriminator="name"),
]


###############################################################################
# Loss Configurations
###############################################################################


class MSELossConfig(BaseModel):
    """Mean Squared Error loss configuration.

    Corresponds to: torch.nn.MSELoss
    """

    name: Literal["mse"] = "mse"

    model_config = {"extra": "forbid"}


class BCELossConfig(BaseModel):
    """Binary Cross-Entropy loss configuration.

    Corresponds to: torch.nn.BCEWithLogitsLoss
    """

    name: Literal["bce"] = "bce"
    reduction: Literal["none", "mean", "sum"] = "mean"
    pos_weight: Optional[list[float]] = None

    model_config = {"extra": "forbid"}


class PoissonLossConfig(BaseModel):
    """Poisson Negative Log-Likelihood loss configuration.

    Corresponds to: torch.nn.PoissonNLLLoss
    """

    name: Literal["poisson"] = "poisson"
    log_input: bool = True
    reduction: Literal["none", "mean", "sum"] = "mean"

    model_config = {"extra": "forbid"}


class MSEUDotLossConfig(BaseModel):
    """MSE with mean-normalized specificity term loss configuration.

    Corresponds to: bernese.metrics.losses.MSEUDot
    """

    name: Literal["mse_udot"] = "mse_udot"
    udot_weight: float = 1.0

    model_config = {"extra": "forbid"}


class PoissonKLLossConfig(BaseModel):
    """Poisson KL divergence loss configuration.

    Corresponds to: bernese.metrics.losses.PoissonKL
    """

    name: Literal["poisson_kl"] = "poisson_kl"
    kl_weight: float = 1.0
    epsilon: float = 1e-7

    model_config = {"extra": "forbid"}


class PoissonMultinomialLossConfig(BaseModel):
    """Poisson-Multinomial loss configuration.

    Corresponds to: bernese.metrics.losses.PoissonMultinomial
    """

    name: Literal["poisson_multinomial"] = "poisson_multinomial"
    total_weight: float = 1.0
    weight_range: float = 1.0
    weight_exp: int = 4
    epsilon: float = 1e-7

    model_config = {"extra": "forbid"}


# Discriminated union for loss configuration
LossConfig = Annotated[
    Union[
        MSELossConfig,
        BCELossConfig,
        PoissonLossConfig,
        MSEUDotLossConfig,
        PoissonKLLossConfig,
        PoissonMultinomialLossConfig,
    ],
    Field(discriminator="name"),
]


###############################################################################
# Metrics Configurations
###############################################################################


class PearsonRMetricConfig(BaseModel):
    """Pearson correlation coefficient metric configuration.

    Corresponds to: bernese.metrics.metrics.PearsonR
    """

    name: Literal["pearsonr"] = "pearsonr"
    summarize: bool = True

    model_config = {"extra": "forbid"}


class R2MetricConfig(BaseModel):
    """R-squared (coefficient of determination) metric configuration.

    Corresponds to: bernese.metrics.metrics.R2
    """

    name: Literal["r2"] = "r2"
    summarize: bool = True

    model_config = {"extra": "forbid"}


class AUROCMetricConfig(BaseModel):
    """Area Under ROC Curve metric configuration.

    Corresponds to: bernese.metrics.metrics.SeqAUC with ROC curve
    """

    name: Literal["auroc"] = "auroc"
    summarize: bool = True

    model_config = {"extra": "forbid"}


class AUPRCMetricConfig(BaseModel):
    """Area Under Precision-Recall Curve metric configuration.

    Corresponds to: bernese.metrics.metrics.SeqAUC with PR curve
    """

    name: Literal["auprc"] = "auprc"
    summarize: bool = True

    model_config = {"extra": "forbid"}


# Union type for metric configuration
MetricConfig = Union[
    PearsonRMetricConfig,
    R2MetricConfig,
    AUROCMetricConfig,
    AUPRCMetricConfig,
]


# Default metrics
DEFAULT_METRICS: list[MetricConfig] = [PearsonRMetricConfig(name="pearsonr")]


###############################################################################
# Scheduler Configurations
###############################################################################


class ConstantSchedulerConfig(BaseModel):
    """Constant learning rate scheduler configuration.

    Uses fixed learning rate throughout training.
    """

    name: Literal["constant"] = "constant"

    model_config = {"extra": "forbid"}


class ExponentialSchedulerConfig(BaseModel):
    """Exponential decay learning rate scheduler configuration.

    Corresponds to: keras.optimizers.schedules.ExponentialDecay
    """

    name: Literal["exponential"] = "exponential"
    decay_rate: float = 0.96
    decay_steps: int = 1000
    staircase: bool = False

    model_config = {"extra": "forbid"}


class CyclicalSchedulerConfig(BaseModel):
    """Cyclical learning rate scheduler configuration.

    Implements cyclical learning rate schedule.
    """

    name: Literal["cyclical"] = "cyclical"
    base_lr: float
    max_lr: float
    step_size: int
    mode: Literal["triangular", "triangular2", "exp_range"] = "triangular"
    gamma: float = 1.0

    model_config = {"extra": "forbid"}


class WarmupSchedulerConfig(BaseModel):
    """Learning rate warmup scheduler configuration.

    Implements learning rate warmup followed by another scheduler.
    """

    name: Literal["warmup"] = "warmup"
    warmup_steps: int
    warmup_lr: float = 1e-6

    model_config = {"extra": "forbid"}


# Discriminated union for scheduler configuration
SchedulerConfig = Annotated[
    Union[
        ConstantSchedulerConfig,
        ExponentialSchedulerConfig,
        CyclicalSchedulerConfig,
        WarmupSchedulerConfig,
    ],
    Field(discriminator="name"),
]


###############################################################################
# Callback Configurations
###############################################################################


class EarlyStoppingConfig(BaseModel):
    """Early stopping callback configuration.

    Monitors a metric and stops training when no improvement is seen.
    """

    monitor: Literal["val_loss", "train_loss"] = "val_loss"
    patience: int = 10
    min_delta: float = 0.0001
    mode: Literal["min", "max"] = "min"
    restore_best_weights: bool = True

    model_config = {"extra": "forbid"}


class CheckpointConfig(BaseModel):
    """Model checkpoint callback configuration.

    Saves model weights at specified intervals.
    """

    save_interval: int = 1
    save_best_only: bool = True
    monitor: Literal["val_loss", "train_loss"] = "val_loss"
    mode: Literal["min", "max"] = "min"
    max_to_keep: int = 5

    model_config = {"extra": "forbid"}


###############################################################################
# Data Configuration
###############################################################################


class DataConfig(BaseModel):
    """Data loading configuration."""

    batch_size: int = 64
    num_workers: int = 0
    prefetch_factor: Optional[int] = 2
    pin_memory: bool = True
    shuffle_train: bool = True

    model_config = {"extra": "forbid"}


###############################################################################
# Main Trainer Configuration
###############################################################################


class TrainerConfig(BaseModel):
    """Complete training configuration with full type safety.

    This is the main configuration class that combines all parameters
    needed to train a SeqNN model. Uses Pydantic for validation.

    Example:
        >>> config = TrainerConfig.from_json("train_config.json")
        >>> builder = TrainerBuilder(config)
        >>> trainer = builder.build()
    """

    # Required fields
    num_targets: int

    # Data configuration
    batch_size: int = 64
    num_workers: int = 0
    prefetch_factor: Optional[int] = 2
    pin_memory: bool = True

    # Optimizer & Loss (required - no defaults for type safety)
    optimizer: OptimizerConfig
    loss: LossConfig

    # Metrics (with default)
    metrics: list[MetricConfig] = Field(
        default_factory=lambda: [PearsonRMetricConfig(name="pearsonr")]
    )

    # Scheduler & Callbacks (optional)
    scheduler: Optional[SchedulerConfig] = None
    early_stopping: Optional[EarlyStoppingConfig] = None
    checkpoint: Optional[CheckpointConfig] = None

    # Training loop
    max_epochs: int = 100
    grad_clip_norm: float = 0.0  # 0 = disabled
    eval_interval: int = 1

    # Device & Reproducibility
    device: str = "cuda"
    seed: int = 42

    # Output
    output_dir: str = "train_out"
    verbose: bool = True

    model_config = {"extra": "forbid"}

    def model_post_init(self, context):
        """Apply defaults after initialization."""
        # Set default callbacks if not provided
        if self.early_stopping is None:
            self.early_stopping = EarlyStoppingConfig()

        if self.checkpoint is None:
            self.checkpoint = CheckpointConfig()

        return super().model_post_init(context)

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainerConfig":
        """Load config from JSON file.

        Args:
            path: Path to JSON file.

        Returns:
            TrainerConfig instance.
        """
        with open(path) as f:
            params = json.load(f)
        return cls(**params)

    def to_json(self, path: str | Path):
        """Save config to JSON file.

        Args:
            path: Path to save JSON file.
        """
        with open(path, "w") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "TrainerConfig":
        """Create config from dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            TrainerConfig instance.
        """
        return cls(**data)

    def to_dict(self) -> dict:
        """Convert config to dictionary.

        Returns:
            Configuration dictionary.
        """
        return self.model_dump(mode="json")


# Convenience function for creating configs
def create_trainer_config(
    num_targets: int,
    optimizer: OptimizerConfig,
    loss: LossConfig,
    *,
    batch_size: int = 64,
    max_epochs: int = 100,
    learning_rate: Optional[float] = None,
    metrics: Optional[list[MetricConfig]] = None,
    device: str = "cuda",
    output_dir: str = "train_out",
) -> TrainerConfig:
    """Create a TrainerConfig with sensible defaults.

    This is a convenience function for quick config creation.

    Args:
        num_targets: Number of prediction targets.
        optimizer: Optimizer configuration.
        loss: Loss function configuration.
        batch_size: Batch size for training.
        max_epochs: Maximum training epochs.
        learning_rate: Override learning rate in optimizer config.
        metrics: List of metric configurations.
        device: Device to train on.
        output_dir: Output directory for checkpoints.

    Returns:
        TrainerConfig instance.
    """
    # Override learning rate if provided
    if learning_rate is not None:
        match optimizer:
            case SGDConfig() as cfg:
                cfg.learning_rate = learning_rate
            case AdamConfig() as cfg:
                cfg.learning_rate = learning_rate
            case AdamWConfig() as cfg:
                cfg.learning_rate = learning_rate

    return TrainerConfig(
        num_targets=num_targets,
        optimizer=optimizer,
        loss=loss,
        batch_size=batch_size,
        max_epochs=max_epochs,
        metrics=metrics or DEFAULT_METRICS,
        device=device,
        output_dir=output_dir,
    )


# Export all config types
__all__ = [
    # Optimizer configs
    "OptimizerConfig",
    "SGDConfig",
    "AdamConfig",
    "AdamWConfig",
    # Loss configs
    "LossConfig",
    "MSELossConfig",
    "BCELossConfig",
    "PoissonLossConfig",
    "MSEUDotLossConfig",
    "PoissonKLLossConfig",
    "PoissonMultinomialLossConfig",
    # Metric configs
    "MetricConfig",
    "PearsonRMetricConfig",
    "R2MetricConfig",
    "AUROCMetricConfig",
    "AUPRCMetricConfig",
    "DEFAULT_METRICS",
    # Scheduler configs
    "SchedulerConfig",
    "ConstantSchedulerConfig",
    "ExponentialSchedulerConfig",
    "CyclicalSchedulerConfig",
    "WarmupSchedulerConfig",
    # Callback configs
    "EarlyStoppingConfig",
    "CheckpointConfig",
    # Data config
    "DataConfig",
    # Main config
    "TrainerConfig",
    "create_trainer_config",
]
