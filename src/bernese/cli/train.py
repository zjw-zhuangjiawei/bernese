# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Training command-line interface for bernese."""

import argparse
import json
import os
import sys
import shutil

import torch

from bernese.models import SeqNN, create_seqnn
from bernese.data import create_data_loaders
from bernese.training import Trainer, create_trainer_from_config


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a SeqNN model for regulatory genomics predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Positional arguments
    parser.add_argument(
        "params_file",
        help="JSON file with model and training parameters",
    )
    parser.add_argument(
        "data_dirs",
        nargs="+",
        help="Training data directory(ies)",
    )

    # Output options
    parser.add_argument(
        "-o",
        "--out_dir",
        default="train_out",
        help="Output directory for checkpoints",
    )

    # Training options
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (default: from config)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (default: from config)",
    )
    parser.add_argument(
        "--lr",
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (default: from config)",
    )
    parser.add_argument(
        "--optimizer",
        choices=["adam", "adamw", "sgd"],
        default=None,
        help="Optimizer type (default: from config)",
    )
    parser.add_argument(
        "--loss",
        choices=["mse", "bce", "poisson", "poisson_kl", "poisson_multinomial", "mse_udot"],
        default=None,
        help="Loss function (default: from config)",
    )

    # Device options
    parser.add_argument(
        "-d",
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of data loading workers",
    )

    # Resume options
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume from checkpoint",
    )

    # Other options
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--copy_params",
        action="store_true",
        help="Copy params file to output directory",
    )

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    """Main training function."""
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Copy params file
    if args.copy_params and args.params_file != f"{args.out_dir}/params.json":
        shutil.copy(args.params_file, f"{args.out_dir}/params.json")

    # Load configuration
    with open(args.params_file, "r") as f:
        params = json.load(f)

    params_model = params.get("model", {})
    params_train = params.get("train", {})

    # Override config with command-line args
    if args.batch_size is not None:
        params_train["batch_size"] = args.batch_size
    if args.lr is not None:
        params_train["learning_rate"] = args.lr
    if args.optimizer is not None:
        params_train["optimizer"] = args.optimizer
    if args.loss is not None:
        params_train["loss"] = args.loss
    if args.epochs is not None:
        params_train["train_epochs_max"] = args.epochs

    # Get batch size
    batch_size = params_train.get("batch_size", 64)

    # Create data loaders
    print(f"Loading data from: {args.data_dirs}")

    train_loaders = []
    val_loaders = []

    for data_dir in args.data_dirs:
        train_loader, val_loader, _ = create_data_loaders(
            data_dir,
            batch_size=batch_size,
            num_workers=args.num_workers,
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
        device=args.device,
    )

    # Train
    print(
        f"\nStarting training for {args.epochs or params_train.get('train_epochs_max', 100)} epochs..."
    )
    print(f"Output directory: {args.out_dir}")
    print("-" * 60)

    trainer.fit(
        epochs=args.epochs,
        out_dir=args.out_dir,
        resume_from=args.resume,
    )

    print("Training complete!")


if __name__ == "__main__":
    main()
