# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Summary command-line interface for bernese.

This module provides the 'bernese summary' command for visualizing
SeqNN model architecture.
"""

from pathlib import Path
from typing import Optional

import keras
import torch
import typer

from bernese.models import SeqNN, SeqNNConfig


def summary(
    params_file: Path = typer.Argument(..., exists=True, help="JSON file with model parameters"),
    num_targets: Optional[int] = typer.Option(
        None, "--num_targets", help="Number of prediction targets (default: from config or 1)"
    ),
    seq_length: Optional[int] = typer.Option(
        None, "--seq_length", help="Input sequence length (default: from config or 1344)"
    ),
    seq_depth: int = typer.Option(
        4, "--seq_depth", help="Input sequence depth (channels, default: 4 for DNA)"
    ),
    device: str = typer.Option(
        "cuda" if torch.cuda.is_available() else "cpu", "-d", "--device", help="Device for model"
    ),
    verbose: int = typer.Option(1, "-v", "--verbose", help="Verbosity level (0-2)"),
) -> None:
    """Display a summary of the SeqNN model architecture.

    Loads model configuration from a JSON file and displays a detailed
    architecture summary.

    Example:
        bernese summary params.json
        bernese summary params.json --num_targets 100
        bernese summary params.json --seq_length 2048 --device cpu
    """
    # Load configuration using Pydantic
    config = SeqNNConfig.from_json(str(params_file))

    # Override with command-line arguments
    if num_targets is not None:
        config.num_targets = num_targets
    if seq_length is not None:
        config.seq_length = seq_length
    config.seq_depth = seq_depth
    config.verbose = verbose > 0

    # Ensure num_targets has a default
    if config.num_targets is None:
        config.num_targets = 1

    # Create model
    print(f"Loading model from: {params_file}")
    print(f"Keras backend: {keras.backend.backend()}")
    print(f"Device: {device}")
    print("-" * 60)

    # Create model
    model = SeqNN(config)

    # Get model info
    seq_len = config.seq_length
    seq_depth_val = config.seq_depth
    num_targets_val = model.get_num_targets()

    # Print summary
    print("-" * 60)
    print(f"Model configuration:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Sequence depth: {seq_depth_val}")
    print(f"  Number of targets: {num_targets_val}")

    # Print total parameters
    total_params = model.model.count_params()
    print(f"\nTotal parameters: {total_params:,}")

    if verbose > 1:
        print("-" * 60)
        print("Layer summary:")
        print(model.model.summary(line_length=100))
