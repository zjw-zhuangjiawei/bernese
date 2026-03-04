# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Comprehensive training loop and utilities for SeqNN models.

This module provides a Trainer class that combines:
- Keras 3 SeqNN model for forward pass
- PyTorch DataLoader for data iteration
- PyTorch losses and metrics for training evaluation
"""

import itertools
import json
import os
from pathlib import Path
from typing import TypedDict, Optional

import numpy as np

import keras
from torch.utils.data import DataLoader

from bernese.models.seqnn import SeqNN
from bernese.metrics.losses import get_loss_function
from bernese.metrics.metrics import get_metric


class TrainConfig(TypedDict, total=False):
    """Training configuration matching train.json fields.

    Attributes:
        batch_size: Training batch size
        optimizer: Optimizer type (sgd, adam, adamw)
        learning_rate: Initial learning rate
        momentum: Momentum for SGD optimizer
        weight_decay: Weight decay for regularization
        loss: Loss function (mse, bce, poisson, mse_udot)
        patience: Early stopping patience (epochs)
        clip_norm: Gradient clipping norm (0 = disabled)
        train_epochs_max: Maximum training epochs
        eval_interval: Validation evaluation interval (epochs)
        save_interval: Checkpoint save interval (epochs)
        metrics: List of metrics to compute (pearsonr, r2, auroc, auprc)
    """

    batch_size: int
    optimizer: str
    learning_rate: float
    momentum: float
    weight_decay: float
    loss: str
    patience: int
    clip_norm: float
    train_epochs_max: int
    eval_interval: int
    save_interval: int
    metrics: list[str]


class Trainer:
    """Comprehensive trainer class for SeqNN models.

    This trainer uses a hybrid approach:
    - Keras 3 model for forward pass and weight updates
    - PyTorch DataLoader for data iteration
    - PyTorch losses and metrics for evaluation

    Args:
        model: The SeqNN model to train
        train_loader: Training data loader (or list of loaders for multi-dataset)
        val_loader: Validation data loader (or list of loaders)
        config: Training configuration dictionary
        device: Device to train on (cuda/cpu)
    """

    def __init__(
        self,
        model: SeqNN,
        train_loader: DataLoader | list[DataLoader],
        val_loader: DataLoader | list[DataLoader],
        config: TrainConfig,
        device: str = "cuda",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Get num_targets from model
        self.num_targets = model.get_num_targets()

        # Create optimizer
        self.optimizer = self._create_optimizer()

        # Create loss function (PyTorch)
        self.loss_fn = get_loss_function(config.get("loss", "mse"))

        # Create metrics (PyTorch)
        self.metrics = self._create_metrics()

        self.model.model.compile(optimizer=self.optimizer, loss=self.loss_fn)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history = {
            "train_loss": [],
            "val_loss": [],
        }

    def _create_optimizer(self) -> keras.optimizers.Optimizer:
        """Create Keras optimizer from config."""
        opt_type = self.config.get("optimizer", "adam").lower()
        lr = self.config.get("learning_rate", 0.001)
        momentum = self.config.get("momentum", 0.0)
        weight_decay = self.config.get("weight_decay", 0.0)

        if opt_type == "sgd":
            return keras.optimizers.SGD(
                learning_rate=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        elif opt_type == "adam":
            return keras.optimizers.Adam(
                learning_rate=lr,
                weight_decay=weight_decay,
            )
        elif opt_type == "adamw":
            return keras.optimizers.AdamW(
                learning_rate=lr,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")

    def _create_metrics(self) -> dict:
        """Create metrics dictionary."""
        metrics_config = self.config.get("metrics", ["pearsonr"])
        metrics = {}

        for metric_name in metrics_config:
            metrics[metric_name] = get_metric(
                metric_name,
                num_targets=self.num_targets,
                summarize=True,
            )

        return metrics

    def fit(
        self,
        epochs: Optional[int] = None,
        out_dir: str = "train_out",
        resume_from: Optional[str] = None,
    ) -> dict:
        """Train the model.

        Args:
            epochs: Number of training epochs (overrides config)
            out_dir: Output directory for checkpoints
            resume_from: Checkpoint path to resume from

        Returns:
            Training history dictionary
        """
        epochs = epochs or self.config.get("train_epochs_max", 100)

        # Create output directory
        os.makedirs(out_dir, exist_ok=True)

        # Resume from checkpoint if specified
        if resume_from:
            self._load_checkpoint(resume_from)

        print(f"Starting training for {epochs} epochs...")
        print(f"Optimizer: {self.config.get('optimizer', 'adam')}")
        print(f"Learning rate: {self.config.get('learning_rate', 0.001)}")
        print(f"Loss: {self.config.get('loss', 'mse')}")
        print(f"Batch size: {self.config.get('batch_size', 64)}")
        print("-" * 60)

        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch

            # Train one epoch
            train_loss = self._train_epoch()
            self.history["train_loss"].append(train_loss)

            print(f"Epoch {epoch + 1}/{epochs} - loss: {train_loss:.6f}")

            # Validate
            if (epoch + 1) % self.config.get("eval_interval", 1) == 0:
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
                    best_path = os.path.join(out_dir, "best_model.weights.h5")
                    self.model.model.save_weights(best_path)
                    print(f"  -> Saved best model (val_loss: {val_loss:.6f})")
                else:
                    self.patience_counter += 1

                if self.patience_counter >= self.config.get("patience", 10):
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            # Periodic checkpoint
            if (epoch + 1) % self.config.get("save_interval", 1) == 0:
                checkpoint_path = os.path.join(out_dir, f"checkpoint_{epoch + 1}.weights.h5")
                self.model.model.save_weights(checkpoint_path)

        print("Training complete!")
        return self.history

    def _train_epoch(self) -> float:
        """Train for one epoch.

        Returns:
            Average training loss
        """
        self.model.model.trainable = True

        # Handle multiple data loaders (multi-dataset)
        loaders = self.train_loader if isinstance(self.train_loader, list) else [self.train_loader]
        num_loaders = len(loaders)

        total_loss = 0.0
        num_batches = 0

        # Round-robin through data loaders
        if num_loaders > 1:
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
            Tuple of (validation loss, metrics dict)
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

                # Compute loss (PyTorch style for metrics)
                # Convert back to torch for metric computation
                import torch

                preds_tensor = torch.from_numpy(preds) if isinstance(preds, np.ndarray) else preds
                targets_tensor = torch.from_numpy(y) if isinstance(y, np.ndarray) else y

                # Compute loss
                if hasattr(self.loss_fn, "forward"):
                    loss = self.loss_fn(preds_tensor, targets_tensor).item()
                else:
                    loss = np.mean((preds - y) ** 2)

                total_loss += loss
                num_batches += 1

                # Collect for metrics
                all_preds.append(
                    preds_tensor
                    if isinstance(preds_tensor, torch.Tensor)
                    else torch.from_numpy(preds)
                )
                all_targets.append(
                    targets_tensor
                    if isinstance(targets_tensor, torch.Tensor)
                    else torch.from_numpy(y)
                )

        # Compute metrics
        metrics = {}
        if len(all_preds) > 0:
            import torch

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
            path: Path to save checkpoint
        """
        # Save Keras model weights
        self.model.model.save_weights(path)

        # Save training state (JSON)
        state_path = path.replace(".weights.h5", "_state.json")
        state = {
            "epoch": self.current_epoch + 1,
            "best_val_loss": self.best_val_loss,
            "patience_counter": self.patience_counter,
            "config": dict(self.config),
        }
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

    def _load_checkpoint(self, path: str):
        """Load model weights and training state.

        Args:
            path: Path to checkpoint
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
    config: dict,
    device: str = "cuda",
) -> Trainer:
    """Factory function to create Trainer from config dict.

    Args:
        model: The SeqNN model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration dictionary (e.g., from train.json)
        device: Device to train on

    Returns:
        Trainer instance
    """
    # Convert dict to TrainConfig with defaults
    train_config: TrainConfig = {
        "batch_size": config.get("batch_size", 64),
        "optimizer": config.get("optimizer", "adam"),
        "learning_rate": config.get("learning_rate", 0.001),
        "momentum": config.get("momentum", 0.0),
        "weight_decay": config.get("weight_decay", 0.0),
        "loss": config.get("loss", "mse"),
        "patience": config.get("patience", 10),
        "clip_norm": config.get("clip_norm", 0.0),
        "train_epochs_max": config.get("train_epochs_max", 100),
        "eval_interval": config.get("eval_interval", 1),
        "save_interval": config.get("save_interval", 1),
        "metrics": config.get("metrics", ["pearsonr"]),
    }

    return Trainer(model, train_loader, val_loader, train_config, device)
