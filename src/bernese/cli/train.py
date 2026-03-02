# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Training command-line interface for bernese.

This module provides the 'bernese train' command for training SeqNN models.
Based on hound_train.py from Basenji/Baskerville.
"""

import json
import os
import random
import shutil
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import typer

from bernese.models import create_seqnn
from bernese.data import create_data_loaders
from bernese.training import create_trainer_from_config


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(
    params_file: Path = typer.Argument(..., exists=True, help="JSON file with model and training parameters"),
    data_dirs: List[Path] = typer.Argument(..., help="Training data directory(ies)"),
    out_dir: Path = typer.Option("train_out", "-o", help="Output directory for checkpoints"),
    epochs: Optional[int] = typer.Option(None, "--epochs", help="Number of training epochs (default: from config)"),
    batch_size: Optional[int] = typer.Option(None, "--batch_size", help="Batch size (default: from config)"),
    lr: Optional[float] = typer.Option(None, "--lr", "--learning_rate", help="Learning rate (default: from config)"),
    optimizer: Optional[str] = typer.Option(None, "--optimizer", help="Optimizer type: adam, adamw, or sgd (default: from config)"),
    loss: Optional[str] = typer.Option(None, "--loss", help="Loss function: mse, bce, poisson, poisson_kl, poisson_multinomial, or mse_udot (default: from config)"),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu", "-d", "--device", help="Device to train on"),
    num_workers: int = typer.Option(0, "--num_workers", help="Number of data loading workers"),
    resume: Optional[Path] = typer.Option(None, "--resume", help="Resume from checkpoint"),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
    copy_params: bool = typer.Option(False, "--copy_params", help="Copy params file to output directory"),
) -> None:
    """Train a SeqNN model for regulatory genomics predictions.

    Example:
        bernese train params.json data_dir/ -o train_out
        bernese train params.json data_dir/ --epochs 100 --lr 0.001
    """
    # Set random seed
    set_seed(seed)

    # Create output directory
    os.makedirs(out_dir, exist_ok=True)

    # Copy params file
    if copy_params and params_file != Path(out_dir) / "params.json":
        shutil.copy(params_file, Path(out_dir) / "params.json")

    # Load configuration
    with open(params_file, "r") as f:
        params = json.load(f)

    params_model = params.get("model", {})
    params_train = params.get("train", {})

    # Override config with command-line args
    if batch_size is not None:
        params_train["batch_size"] = batch_size
    if lr is not None:
        params_train["learning_rate"] = lr
    if optimizer is not None:
        params_train["optimizer"] = optimizer
    if loss is not None:
        params_train["loss"] = loss
    if epochs is not None:
        params_train["train_epochs_max"] = epochs

    # Get batch size
    batch_size = params_train.get("batch_size", 64)

    # Create data loaders
    print(f"Loading data from: {[str(d) for d in data_dirs]}")

    train_loaders = []
    val_loaders = []

    for data_dir in data_dirs:
        train_loader, val_loader, _ = create_data_loaders(
            str(data_dir),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle_train=True,
        )
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

    # Use first data dir for model config
    num_targets = train_loaders[0].dataset.num_targets
    params_model["num_targets"] = num_targets

    # Print model info
    print(f"Number of targets: {num_targets}")
    print(f"Training sequences: {train_loaders[0].dataset.num_seqs}")
    print(f"Validation sequences: {val_loaders[0].dataset.num_seqs}")

    # Create model
    print("Creating model...")
    model = create_seqnn(params_model)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create trainer
    print("Creating trainer...")
    trainer = create_trainer_from_config(
        model=model,
        train_loader=train_loaders if len(train_loaders) > 1 else train_loaders[0],
        val_loader=val_loaders if len(val_loaders) > 1 else val_loaders[0],
        config=params_train,
        device=device,
    )

    # Train
    num_epochs = epochs or params_train.get("train_epochs_max", 100)
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Output directory: {out_dir}")
    print("-" * 60)

    trainer.fit(
        epochs=epochs,
        out_dir=str(out_dir),
        resume_from=str(resume) if resume else None,
    )

    print("Training complete!")
