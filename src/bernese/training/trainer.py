# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Comprehensive training loop and utilities for SeqNN models.

This module provides:
- TrainerConfig: Fully typed Pydantic configuration
- TrainerBuilder: Builder class for constructing Trainer
- Trainer: Comprehensive training loop with Keras 3 + PyTorch DataLoader

This module implements a hybrid approach:
- Keras 3 model for forward pass and weight updates
- PyTorch DataLoader for data iteration
- PyTorch losses and metrics for evaluation
"""

import itertools
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

import keras
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bernese.models.seqnn import SeqNN
from bernese.metrics.losses import (
    MSEUDot,
    PoissonKL,
    PoissonMultinomial,
    get_loss_function as get_keras_loss,
    _MSELoss,
    _PoissonLoss,
    _BCELoss,
)
from bernese.metrics.metrics import (
    PearsonR,
    R2,
    SeqAUC,
    get_metric,
)

# Import Pydantic configurations
from bernese.training.config import (
    TrainerConfig,
    OptimizerConfig,
    LossConfig,
    MetricConfig,
    SchedulerConfig,
    EarlyStoppingConfig,
    CheckpointConfig,
    # Optimizer configs
    SGDConfig,
    AdamConfig,
    AdamWConfig,
    # Loss configs
    MSELossConfig,
    BCELossConfig,
    PoissonLossConfig,
    MSEUDotLossConfig,
    PoissonKLLossConfig,
    PoissonMultinomialLossConfig,
    # Metric configs
    PearsonRMetricConfig,
    R2MetricConfig,
    AUROCMetricConfig,
    AUPRCMetricConfig,
    # Scheduler configs
    ConstantSchedulerConfig,
    ExponentialSchedulerConfig,
    CyclicalSchedulerConfig,
    WarmupSchedulerConfig,
)


class TrainerBuilder:
    """Builder class for constructing Trainer with type-safe configuration.

    This class handles the trainer construction logic, separating construction
    concerns from the trainer itself. Uses Pydantic config for validation.

    Example:
        >>> config = TrainerConfig.from_json("train_config.json")
        >>> builder = TrainerBuilder(config)
        >>> trainer = builder.with_model(model).build()
    """

    def __init__(self, config: TrainerConfig):
        """Initialize builder with configuration.

        Args:
            config: TrainerConfig Pydantic model instance.
        """
        if not isinstance(config, TrainerConfig):
            raise TypeError(
                f"config must be a TrainerConfig instance, got {type(config).__name__}. "
                f"Use TrainerConfig.from_json() to load from file."
            )

        self.config = config
        self.model: Optional[SeqNN] = None
        self.train_loader: Optional[DataLoader | list[DataLoader]] = None
        self.val_loader: Optional[DataLoader | list[DataLoader]] = None

    def with_model(self, model: SeqNN) -> "TrainerBuilder":
        """Set the model to train.

        Args:
            model: SeqNN model instance.

        Returns:
            Self for method chaining.
        """
        self.model = model
        return self

    def with_train_loader(self, loader: DataLoader | list[DataLoader]) -> "TrainerBuilder":
        """Set the training data loader.

        Args:
            loader: Training data loader or list of loaders.

        Returns:
            Self for method chaining.
        """
        self.train_loader = loader
        return self

    def with_val_loader(self, loader: DataLoader | list[DataLoader]) -> "TrainerBuilder":
        """Set the validation data loader.

        Args:
            loader: Validation data loader or list of loaders.

        Returns:
            Self for method chaining.
        """
        self.val_loader = loader
        return self

    def build(self) -> "Trainer":
        """Build the complete Trainer.

        Returns:
            Configured Trainer instance.

        Raises:
            ValueError: If required components are not set.
        """
        if self.model is None:
            raise ValueError("Model must be set before building Trainer")
        if self.train_loader is None:
            raise ValueError("Training data loader must be set")
        if self.val_loader is None:
            raise ValueError("Validation data loader must be set")

        # Build components
        optimizer = self._build_optimizer()
        loss_fn = self._build_loss()
        metrics = self._build_metrics()
        scheduler = self._build_scheduler()

        # Create trainer
        return Trainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metrics=metrics,
            scheduler=scheduler,
            config=self.config,
            device=self.config.device,
        )

    def _build_optimizer(self) -> keras.optimizers.Optimizer:
        """Build optimizer with type-safe dispatch and gradient clipping.

        Returns:
            Keras optimizer instance with gradient clipping enabled.
        """
        # Get gradient clipping norm from config (default to 1.0 for stability)
        clipnorm = self.config.grad_clip_norm if self.config.grad_clip_norm > 0 else None

        match self.config.optimizer:
            case SGDConfig() as cfg:
                return keras.optimizers.SGD(
                    learning_rate=cfg.learning_rate,
                    momentum=cfg.momentum,
                    weight_decay=cfg.weight_decay,
                    clipnorm=clipnorm,
                )
            case AdamConfig() as cfg:
                return keras.optimizers.Adam(
                    learning_rate=cfg.learning_rate,
                    weight_decay=cfg.weight_decay,
                    beta_1=cfg.beta_1,
                    beta_2=cfg.beta_2,
                    epsilon=cfg.epsilon,
                    clipnorm=clipnorm,
                )
            case AdamWConfig() as cfg:
                return keras.optimizers.AdamW(
                    learning_rate=cfg.learning_rate,
                    weight_decay=cfg.weight_decay,
                    beta_1=cfg.beta_1,
                    beta_2=cfg.beta_2,
                    epsilon=cfg.epsilon,
                    clipnorm=clipnorm,
                )
            case _:
                raise ValueError(
                    f"Unknown optimizer config: {type(self.config.optimizer).__name__}"
                )

    def _build_loss(self):
        """Build loss function with type-safe dispatch.

        Returns:
            Keras 3 compatible loss function.
        """
        match self.config.loss:
            case MSELossConfig():
                return _MSELoss(reduction="sum_over_batch_size")
            case BCELossConfig() as cfg:
                return _BCELoss(reduction="sum_over_batch_size", pos_weight=cfg.pos_weight)
            case PoissonLossConfig() as cfg:
                return _PoissonLoss(reduction="sum_over_batch_size")
            case MSEUDotLossConfig() as cfg:
                return MSEUDot(udot_weight=cfg.udot_weight, reduction="sum_over_batch_size")
            case PoissonKLLossConfig() as cfg:
                return PoissonKL(
                    kl_weight=cfg.kl_weight, epsilon=cfg.epsilon, reduction="sum_over_batch_size"
                )
            case PoissonMultinomialLossConfig() as cfg:
                return PoissonMultinomial(
                    total_weight=cfg.total_weight,
                    weight_range=cfg.weight_range,
                    weight_exp=cfg.weight_exp,
                    epsilon=cfg.epsilon,
                    reduction="sum_over_batch_size",
                )
            case _:
                raise ValueError(f"Unknown loss config: {type(self.config.loss).__name__}")

    def _build_metrics(self) -> dict[str, nn.Module]:
        """Build metrics dictionary with type-safe dispatch.

        Returns:
            Dictionary of metric modules.
        """
        metrics = {}
        num_targets = self.config.num_targets

        for metric_cfg in self.config.metrics:
            match metric_cfg:
                case PearsonRMetricConfig() as cfg:
                    metrics["pearsonr"] = PearsonR(num_targets, cfg.summarize)
                case R2MetricConfig() as cfg:
                    metrics["r2"] = R2(num_targets, cfg.summarize)
                case AUROCMetricConfig() as cfg:
                    metrics["auroc"] = SeqAUC("ROC", cfg.summarize)
                case AUPRCMetricConfig() as cfg:
                    metrics["auprc"] = SeqAUC("PR", cfg.summarize)
                case _:
                    raise ValueError(f"Unknown metric config: {type(metric_cfg).__name__}")

        return metrics

    def _build_scheduler(self) -> Optional[keras.optimizers.schedules.LearningRateSchedule]:
        """Build learning rate scheduler with type-safe dispatch.

        Returns:
            Keras learning rate scheduler or None.
        """
        if self.config.scheduler is None:
            return None

        # Get base learning rate from optimizer config
        base_lr = self.config.optimizer.learning_rate

        match self.config.scheduler:
            case ConstantSchedulerConfig():
                return base_lr
            case ExponentialSchedulerConfig() as cfg:
                return keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=base_lr,
                    decay_rate=cfg.decay_rate,
                    decay_steps=cfg.decay_steps,
                    staircase=cfg.staircase,
                )
            case CyclicalSchedulerConfig() as cfg:
                # Keras doesn't have built-in cyclical scheduler, use custom callback
                return None
            case WarmupSchedulerConfig() as cfg:
                # Warmup will be handled by a callback
                return None
            case _:
                return None


class Trainer:
    """Comprehensive trainer class for SeqNN models.

    This trainer uses a hybrid approach:
    - Keras 3 model for forward pass and weight updates
    - PyTorch DataLoader for data iteration
    - PyTorch losses and metrics for evaluation

    Args:
        model: The SeqNN model to train.
        train_loader: Training data loader (or list of loaders for multi-dataset).
        val_loader: Validation data loader (or list of loaders).
        optimizer: Keras optimizer instance.
        loss_fn: PyTorch loss module.
        metrics: Dictionary of PyTorch metric modules.
        scheduler: Optional Keras learning rate scheduler.
        config: TrainerConfig instance.
        device: Device to train on (cuda/cpu).
    """

    def __init__(
        self,
        model: SeqNN,
        train_loader: DataLoader | list[DataLoader],
        val_loader: DataLoader | list[DataLoader],
        optimizer: keras.optimizers.Optimizer,
        loss_fn: nn.Module,
        metrics: dict[str, nn.Module],
        scheduler: Optional[keras.optimizers.schedules.LearningRateSchedule],
        config: TrainerConfig,
        device: str = "cuda",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.scheduler = scheduler
        self.config = config
        self.device = device

        # Get num_targets from model
        self.num_targets = model.get_num_targets()

        # Compile Keras model
        self.model.model.compile(optimizer=self.optimizer, loss=self.loss_fn)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history = {
            "train_loss": [],
            "val_loss": [],
        }

    def fit(
        self,
        epochs: Optional[int] = None,
        out_dir: Optional[str] = None,
        resume_from: Optional[str] = None,
    ) -> dict:
        """Train the model.

        Args:
            epochs: Number of training epochs (overrides config).
            out_dir: Output directory for checkpoints (overrides config).
            resume_from: Checkpoint path to resume from.

        Returns:
            Training history dictionary.
        """
        epochs = epochs or self.config.max_epochs
        out_dir = out_dir or self.config.output_dir

        # Create output directory
        os.makedirs(out_dir, exist_ok=True)

        # Resume from checkpoint if specified
        if resume_from:
            self._load_checkpoint(resume_from)

        # Get optimizer and loss info for printing
        opt_name = type(self.config.optimizer).__name__.replace("Config", "")
        lr = self.config.optimizer.learning_rate
        loss_name = type(self.config.loss).__name__.replace("Config", "")

        print(f"Starting training for {epochs} epochs...")
        print(f"Optimizer: {opt_name}")
        print(f"Learning rate: {lr}")
        print(f"Loss: {loss_name}")
        print(f"Batch size: {self.config.batch_size}")
        print("-" * 60)

        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch

            # Train one epoch
            train_loss = self._train_epoch()
            self.history["train_loss"].append(train_loss)

            print(f"Epoch {epoch + 1}/{epochs} - loss: {train_loss:.6f}")

            # Validate
            if (epoch + 1) % self.config.eval_interval == 0:
                val_loss, val_metrics = self._validate()
                self.history["val_loss"].append(val_loss)

                # Print metrics
                metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
                print(f"Epoch {epoch + 1}/{epochs} - val_loss: {val_loss:.6f} - {metrics_str}")

                # Early stopping check
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    # Save best model
                    if self.config.checkpoint.save_best_only:
                        best_path = os.path.join(out_dir, "best_model.weights.h5")
                        self.model.model.save_weights(best_path)
                        print(f"  -> Saved best model (val_loss: {val_loss:.6f})")
                else:
                    self.patience_counter += 1

                # Check early stopping
                if (
                    self.config.early_stopping
                    and self.patience_counter >= self.config.early_stopping.patience
                ):
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            # Periodic checkpoint
            if self.config.checkpoint and (epoch + 1) % self.config.checkpoint.save_interval == 0:
                checkpoint_path = os.path.join(out_dir, f"checkpoint_{epoch + 1}.weights.h5")
                self.model.model.save_weights(checkpoint_path)

        print("Training complete!")
        return self.history

    def _train_epoch(self) -> float:
        """Train for one epoch.

        Returns:
            Average training loss.
        """
        self.model.model.trainable = True

        # Handle multiple data loaders (multi-dataset)
        loaders = self.train_loader if isinstance(self.train_loader, list) else [self.train_loader]

        total_loss = 0.0
        num_batches = 0

        # Round-robin through data loaders
        if len(loaders) > 1:
            # Multi-dataset: round-robin sampling
            iterators = [iter(loader) for loader in loaders]

            while True:
                has_data = False
                for it in iterators:
                    try:
                        batch = next(it)
                        sequences, targets = batch
                        has_data = True

                        # Convert to numpy
                        X = sequences.numpy().astype("float32")
                        y = targets.numpy().astype("float32")

                        # Forward pass via Keras train_on_batch
                        loss = self.model.model.train_on_batch(X, y)

                        total_loss += loss
                        num_batches += 1
                    except StopIteration:
                        pass

                if not has_data:
                    break
        else:
            # Single dataset
            for batch in loaders[0]:
                sequences, targets = batch

                # Convert to numpy
                X = sequences.numpy().astype("float32")
                y = targets.numpy().astype("float32")

                # Forward pass via Keras train_on_batch
                loss = self.model.model.train_on_batch(X, y)

                total_loss += loss
                num_batches += 1

        return total_loss / max(num_batches, 1)

    def _validate(self) -> tuple[float, dict]:
        """Validate the model.

        Returns:
            Tuple of (validation loss, metrics dict).
        """
        self.model.model.trainable = False

        # Handle multiple data loaders
        loaders = self.val_loader if isinstance(self.val_loader, list) else [self.val_loader]

        total_loss = 0.0
        num_batches = 0

        # Reset metrics
        for metric in self.metrics.values():
            if hasattr(metric, "reset"):
                metric.reset()

        # Collect all predictions and targets for metric computation
        all_preds = []
        all_targets = []

        for loader in loaders:
            for batch in loader:
                sequences, targets = batch

                # Convert to numpy
                X = sequences.numpy().astype("float32")
                y = targets.numpy().astype("float32")

                # Predict
                preds = self.model.model.predict(X, verbose=0)

                # Compute loss using Keras compatible loss function
                # Keras losses expect (y_true, y_pred) order
                loss_result = self.loss_fn(y, preds)
                # Handle both scalar and tensor results
                if hasattr(loss_result, "item"):
                    loss = loss_result.item()
                else:
                    loss = float(loss_result)

                total_loss += loss
                num_batches += 1

                # Collect for metrics (convert to torch tensors if needed)
                all_preds.append(
                    torch.from_numpy(preds) if isinstance(preds, np.ndarray) else preds
                )
                all_targets.append(torch.from_numpy(y) if isinstance(y, np.ndarray) else y)

        # Compute metrics
        metrics = {}
        if len(all_preds) > 0:
            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

            for metric_name, metric_fn in self.metrics.items():
                try:
                    metric_value = metric_fn(all_targets, all_preds)
                    if hasattr(metric_value, "item"):
                        metric_value = metric_value.item()
                    metrics[metric_name] = metric_value
                except Exception as e:
                    metrics[metric_name] = 0.0

        val_loss = total_loss / max(num_batches, 1)
        return val_loss, metrics

    def _save_checkpoint(self, path: str):
        """Save model weights and training state.

        Args:
            path: Path to save checkpoint.
        """
        # Save Keras model weights
        self.model.model.save_weights(path)

        # Save training state (JSON)
        state_path = path.replace(".weights.h5", "_state.json")
        state = {
            "epoch": self.current_epoch + 1,
            "best_val_loss": self.best_val_loss,
            "patience_counter": self.patience_counter,
            "config": self.config.model_dump(),
        }
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

    def _load_checkpoint(self, path: str):
        """Load model weights and training state.

        Args:
            path: Path to checkpoint.
        """
        # Load Keras weights
        self.model.model.load_weights(path)

        # Load training state
        state_path = path.replace(".weights.h5", "_state.json")
        if os.path.exists(state_path):
            with open(state_path) as f:
                state = json.load(f)
            self.current_epoch = state.get("epoch", 0)
            self.best_val_loss = state.get("best_val_loss", float("inf"))
            self.patience_counter = state.get("patience_counter", 0)
            print(f"Resumed from epoch {self.current_epoch}")


def create_trainer_from_config(
    model: SeqNN,
    train_loader: DataLoader | list[DataLoader],
    val_loader: DataLoader | list[DataLoader],
    config: TrainerConfig,
    device: str = "cuda",
) -> Trainer:
    """Factory function to create Trainer from TrainerConfig.

    This is a convenience function that creates a Trainer from a TrainerConfig.

    Args:
        model: The SeqNN model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        config: TrainerConfig instance.
        device: Device to train on.

    Returns:
        Trainer instance.
    """
    builder = TrainerBuilder(config)
    return (
        builder.with_model(model)
        .with_train_loader(train_loader)
        .with_val_loader(val_loader)
        .build()
    )


# Export all classes and functions
__all__ = [
    "TrainerConfig",
    "TrainerBuilder",
    "Trainer",
    "create_trainer_from_config",
]
