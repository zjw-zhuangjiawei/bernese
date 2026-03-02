# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Comprehensive training loop and utilities for SeqNN models."""

import json
import math
import os
import time
from typing import Optional, Dict, List, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler

import numpy as np

from bernese.metrics import losses as loss_functions
from bernese.metrics import metrics as metric_functions


class Trainer:
    """Comprehensive trainer class for SeqNN models.

    Args:
        model: The SeqNN model to train
        train_loader: Training data loader (or list of loaders for multi-dataset)
        val_loader: Validation data loader (or list of loaders)
        config: Training configuration dictionary
        device: Device to train on
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: Union[DataLoader, List[DataLoader]],
        val_loader: Union[DataLoader, List[DataLoader]],
        config: dict,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.config = config

        # Handle single or multiple data loaders
        if isinstance(train_loader, list):
            self.train_loaders = train_loader
            self.num_datasets = len(train_loader)
        else:
            self.train_loaders = [train_loader]
            self.num_datasets = 1

        if isinstance(val_loader, list):
            self.val_loaders = val_loader
        else:
            self.val_loaders = [val_loader]

        # Loss function
        self.loss_fn = self._create_loss_function(config)

        # Metrics
        self._setup_metrics(config)

        # Optimizer
        self.optimizer = self._create_optimizer(config)

        # Learning rate scheduler
        self.scheduler = self._create_scheduler(config)

        # Training state
        self.current_epoch = 0
        self.best_metric = float("-inf")
        self.best_metric_name = config.get("best_metric", "val_loss")
        self.patience = config.get("patience", 20)
        self.train_epochs_min = config.get("train_epochs_min", 1)
        self.train_epochs_max = config.get("train_epochs_max", 10000)

        # Gradient clipping
        self.clip_norm = config.get("clip_norm", None)
        self.global_clipnorm = config.get("global_clipnorm", None)

    def _create_loss_function(self, config: dict) -> nn.Module:
        """Create loss function from config."""
        loss_name = str(config.get("loss", "mse")).lower()

        # Get loss-specific parameters
        loss_kwargs = {}
        if loss_name == "poisson_kl" or loss_name == "mse_udot":
            loss_kwargs["udot_weight"] = config.get("spec_weight", 1.0)
        elif loss_name == "poisson_multinomial":
            loss_kwargs["total_weight"] = config.get("total_weight", 1.0)
            loss_kwargs["weight_range"] = config.get("weight_range", 1.0)
            loss_kwargs["weight_exp"] = config.get("weight_exp", 4)

        return loss_functions.get_loss_function(loss_name, **loss_kwargs)

    def _setup_metrics(self, config: dict):
        """Setup metrics for training and validation."""
        self.train_metrics: dict[str, nn.Module] = {}
        self.val_metrics: dict[str, nn.Module] = {}

        # Get num_targets from model or config
        num_targets = config.get("num_targets", 1)
        loss = str(config.get("loss", "mse")).lower()

        if loss == "bce":
            # Binary classification metrics
            self.val_metrics["auroc"] = metric_functions.SeqAUC("ROC", summarize=True)
            self.val_metrics["auprc"] = metric_functions.SeqAUC("PR", summarize=True)
        else:
            # Regression metrics
            self.val_metrics["pearsonr"] = metric_functions.PearsonR(num_targets, summarize=True)
            self.val_metrics["r2"] = metric_functions.R2(num_targets, summarize=True)

    def _create_optimizer(self, config: dict) -> torch.optim.Optimizer:
        """Create optimizer from config."""
        optimizer_type = str(config.get("optimizer", "adam")).lower()
        lr = config.get("learning_rate", config.get("initial_learning_rate", 0.001))

        # Weight decay
        weight_decay = config.get("weight_decay", 0.0)

        if optimizer_type == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                betas=(config.get("adam_beta1", 0.9), config.get("adam_beta2", 0.999)),
                weight_decay=weight_decay,
            )
        elif optimizer_type == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                betas=(config.get("adam_beta1", 0.9), config.get("adam_beta2", 0.999)),
                weight_decay=weight_decay,
            )
        elif optimizer_type in ["sgd", "momentum"]:
            return torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=config.get("momentum", 0.99),
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

    def _create_scheduler(self, config: dict) -> Optional[_LRScheduler]:
        """Create learning rate scheduler from config."""
        # Check for cyclical LR
        has_cyclical = (
            "initial_learning_rate" in config
            and "maximal_learning_rate" in config
            and "final_learning_rate" in config
            and "train_epochs_cycle1" in config
        )

        if has_cyclical:
            # Calculate step size
            batches_per_epoch = len(self.train_loaders[0])
            step_size = int(config["train_epochs_cycle1"]) * batches_per_epoch

            return Cyclical1LearningRate(
                initial_learning_rate=float(config["initial_learning_rate"]),
                maximal_learning_rate=float(config["maximal_learning_rate"]),
                final_learning_rate=float(config["final_learning_rate"]),
                step_size=step_size,
            )
        elif "warmup_steps" in config:
            # Warmup + decay
            return WarmUpScheduler(
                initial_learning_rate=float(config.get("learning_rate", 0.001)),
                warmup_steps=int(config["warmup_steps"]),
                decay_type=str(config.get("decay_type", "exponential")),
                decay_steps=int(config.get("decay_steps", 100000)),
                decay_rate=float(config.get("decay_rate", 0.96)),
                optimizer=self.optimizer,
            )
        elif "decay_steps" in config:
            # Exponential decay
            return torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=float(config.get("decay_rate", 0.96)),
            )

        return None

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Handle multiple datasets (round-robin)
        if self.num_datasets > 1:
            # Get iterators for each dataset
            train_iters = [iter(loader) for loader in self.train_loaders]

            # Calculate total batches
            total_batches = sum(len(loader) for loader in self.train_loaders)

            # Create dataset indices
            dataset_indexes = []
            for di, loader in enumerate(self.train_loaders):
                dataset_indexes.extend([di] * len(loader))
            dataset_indexes = np.array(dataset_indexes)
            np.random.shuffle(dataset_indexes)

            for di in dataset_indexes:
                try:
                    sequences, targets = next(train_iters[di])
                except StopIteration:
                    train_iters[di] = iter(self.train_loaders[di])
                    sequences, targets = next(train_iters[di])

                loss = self._train_step(sequences, targets)
                total_loss += loss
                num_batches += 1
        else:
            # Single dataset
            for sequences, targets in self.train_loaders[0]:
                loss = self._train_step(sequences, targets)
                total_loss += loss
                num_batches += 1

        return {"loss": total_loss / max(num_batches, 1)}

    def _train_step(self, sequences: torch.Tensor, targets: torch.Tensor) -> float:
        """Single training step.

        Args:
            sequences: Input sequences
            targets: Target values

        Returns:
            Loss value
        """
        sequences = sequences.to(self.device)
        targets = targets.to(self.device)

        self.optimizer.zero_grad()

        # Forward pass
        predictions = self.model(sequences)

        # Handle multi-head output
        if isinstance(predictions, list):
            predictions = predictions[0]

        # Compute loss
        loss = self.loss_fn(predictions, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if self.clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
        elif self.global_clipnorm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.global_clipnorm)

        self.optimizer.step()

        return loss.item()

    def validate(self) -> Dict[str, float]:
        """Validate the model.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        # Reset metrics
        for metric in self.val_metrics.values():
            if hasattr(metric, "reset"):
                metric.reset()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for loader in self.val_loaders:
                for sequences, targets in loader:
                    sequences = sequences.to(self.device)
                    targets = targets.to(self.device)

                    # Forward pass
                    predictions = self.model(sequences)

                    if isinstance(predictions, list):
                        predictions = predictions[0]

                    # Compute loss
                    loss = self.loss_fn(predictions, targets)
                    total_loss += loss.item()
                    num_batches += 1

                    # Update metrics
                    for metric in self.val_metrics.values():
                        metric(predictions, targets)

        # Compute metrics
        metrics = {"val_loss": total_loss / max(num_batches, 1)}

        for name, metric in self.val_metrics.items():
            if hasattr(metric, "compute"):
                metrics[f"val_{name}"] = metric.compute().item()
            elif hasattr(metric, "forward"):
                # Already computed in loop
                pass

        return metrics

    def fit(
        self,
        epochs: Optional[int] = None,
        out_dir: Optional[str] = None,
        resume_from: Optional[str] = None,
    ):
        """Train the model for specified number of epochs.

        Args:
            epochs: Number of epochs to train (default: from config)
            out_dir: Directory for checkpoints
            resume_from: Path to checkpoint to resume from
        """
        # Determine epochs
        if epochs is None:
            epochs = self.train_epochs_max

        # Setup directories
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # Resume from checkpoint
        start_epoch = 0
        if resume_from and os.path.exists(resume_from):
            self.load_checkpoint(resume_from)
            start_epoch = self.current_epoch + 1
            print(f"Resumed from epoch {start_epoch}")

        patience_counter = 0

        for epoch in range(start_epoch, epochs):
            # Check early stopping
            if epoch >= self.train_epochs_min and patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                break

            self.current_epoch = epoch
            t0 = time.time()

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Compute epoch time
            epoch_time = time.time() - t0

            # Print progress
            self._print_progress(epoch, epoch_time, train_metrics, val_metrics)

            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()

            # Determine metric for early stopping and best model
            if self.best_metric_name == "val_loss":
                current_metric = -val_metrics.get("val_loss", 0)
            else:
                current_metric = val_metrics.get(f"val_{self.best_metric_name}", 0)

            # Check for improvement
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                patience_counter = 0

                # Save best model
                if out_dir:
                    self.save_checkpoint(os.path.join(out_dir, "best_model.pt"))
            else:
                patience_counter += 1

            # Save checkpoint
            if out_dir:
                self.save_checkpoint(os.path.join(out_dir, f"checkpoint_epoch_{epoch}.pt"))

        print(f"Training complete. Best {self.best_metric_name}: {self.best_metric:.4f}")

    def _print_progress(
        self, epoch: int, epoch_time: float, train_metrics: Dict, val_metrics: Dict
    ):
        """Print training progress."""
        loss_str = f"loss: {train_metrics.get('loss', 0):.4f}"
        val_str = f"val_loss: {val_metrics.get('val_loss', 0):.4f}"

        # Add metrics
        for name, metric in self.val_metrics.items():
            if hasattr(metric, "compute"):
                val = metric.compute().item()
                val_str += f" - val_{name}: {val:.4f}"

        print(f"Epoch {epoch} - {epoch_time:.1f}s - {loss_str} - {val_str}", flush=True)

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save(
            {
                "epoch": self.current_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "best_metric": self.best_metric,
                "config": self.config,
            },
            path,
        )

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.best_metric = checkpoint["best_metric"]


# Learning rate schedulers


class Cyclical1LearningRate(_LRScheduler):
    """Cyclical learning rate schedule.

    Args:
        initial_learning_rate: Starting learning rate
        maximal_learning_rate: Peak learning rate
        final_learning_rate: Minimum learning rate after cycle
        step_size: Number of steps per half cycle
        optimizer: Wrapped optimizer
    """

    def __init__(
        self,
        initial_learning_rate: float,
        maximal_learning_rate: float,
        final_learning_rate: float,
        step_size: int,
        optimizer: torch.optim.Optimizer,
    ):
        self.initial_learning_rate = initial_learning_rate
        self.maximal_learning_rate = maximal_learning_rate
        self.final_learning_rate = final_learning_rate
        self.step_size = step_size
        super().__init__(optimizer)

    def get_lr(self):
        """Compute learning rate for current step."""
        cycle = math.floor(1 + self.last_epoch / (2 * self.step_size))
        x = abs(self.last_epoch / self.step_size - 2 * cycle + 1)

        lr = torch.where(
            torch.tensor(self.last_epoch) > 2 * self.step_size,
            torch.tensor(self.final_learning_rate),
            torch.tensor(self.initial_learning_rate)
            + (torch.tensor(self.maximal_learning_rate) - torch.tensor(self.initial_learning_rate))
            * torch.max(torch.tensor(0), 1 - x),
        )

        return [lr.item()] * len(self.base_lrs)


class WarmUpScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Learning rate scheduler with warmup.

    Args:
        initial_learning_rate: Learning rate after warmup
        warmup_steps: Number of warmup steps
        decay_type: Type of decay after warmup ('exponential' or 'linear')
        decay_steps: Steps for decay
        decay_rate: Decay rate
        optimizer: Wrapped optimizer
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_learning_rate: float,
        warmup_steps: int,
        decay_type: str = "exponential",
        decay_steps: int = 100000,
        decay_rate: float = 0.96,
    ):
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.decay_type = decay_type
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        super().__init__(optimizer)

    def get_lr(self):
        """Compute learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.last_epoch / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Decay
            if self.decay_type == "exponential":
                decay_factor = self.decay_rate ** (
                    (self.last_epoch - self.warmup_steps) / self.decay_steps
                )
            else:
                decay_factor = 1 - (self.last_epoch - self.warmup_steps) / self.decay_steps
                decay_factor = max(decay_factor, 0)

            return [base_lr * decay_factor for base_lr in self.base_lrs]


def create_trainer_from_config(
    model: nn.Module,
    train_loader: Union[DataLoader, List[DataLoader]],
    val_loader: Union[DataLoader, List[DataLoader]],
    config: Dict,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Trainer:
    """Factory function to create trainer from config.

    Args:
        model: The model to train
        train_loader: Training data loader(s)
        val_loader: Validation data loader(s)
        config: Training configuration
        device: Device to train on

    Returns:
        Trainer instance
    """
    return Trainer(model, train_loader, val_loader, config, device)


# Convenience function to load config and create trainer


def train_from_json(
    model: nn.Module,
    config_path: str,
    train_loader: Union[DataLoader, List[DataLoader]],
    val_loader: Union[DataLoader, List[DataLoader]],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Trainer:
    """Create trainer from JSON config file.

    Args:
        model: The model to train
        config_path: Path to JSON config file
        train_loader: Training data loader(s)
        val_loader: Validation data loader(s)
        device: Device to train on

    Returns:
        Trainer instance
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    return create_trainer_from_config(model, train_loader, val_loader, config, device)
