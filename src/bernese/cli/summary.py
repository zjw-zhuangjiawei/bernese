# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Summary command-line interface for bernese.

This module provides the 'bernese summary' command for visualizing
SeqNN model architecture using torchinfo.
"""

from pathlib import Path
from typing import Optional

import torch
import torchinfo
import typer

from bernese.models import create_seqnn


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
    verbose: int = typer.Option(1, "-v", "--verbose", help="Verbosity level for torchinfo (0-3)"),
    row_settings: Optional[list[str]] = typer.Option(
        None, "--row_settings", help="Row settings for torchinfo (e.g., aliases,depth)"
    ),
) -> None:
    """Display a summary of the SeqNN model architecture.

    Loads model configuration from a JSON file and displays a detailed
    architecture summary using torchinfo.

    Example:
        bernese summary params.json
        bernese summary params.json --num_targets 100
        bernese summary params.json --seq_length 2048 --device cpu
    """
    import json

    # Load configuration
    with open(params_file, "r") as f:
        params = json.load(f)

    # Extract model parameters
    params_model = params.get("model", params)

    # Override with command-line arguments
    if num_targets is not None:
        params_model["num_targets"] = num_targets
    if seq_length is not None:
        params_model["seq_length"] = seq_length

    # Set seq_depth
    params_model["seq_depth"] = seq_depth

    # Ensure num_targets has a default
    if "num_targets" not in params_model:
        params_model["num_targets"] = 1

    # Create model
    print(f"Loading model from: {params_file}")
    print(f"Device: {device}")
    print("-" * 60)

    model = create_seqnn(params_model)
    model = model.to(device)

    # Get input size - SeqNN expects (batch, seq_length, seq_depth)
    batch_size = 1
    seq_len = params_model.get("seq_length", 1344)
    seq_depth_val = params_model.get("seq_depth", 4)

    # Use torchinfo's detailed_print for better output with custom input
    # The model expects (batch, seq_length, channels) format
    input_size = (batch_size, seq_len, seq_depth_val)

    # Prepare torchinfo kwargs - use col_names for detailed output
    torchinfo_kwargs = {
        "model": model,
        "input_size": input_size,
        "device": device,
        "verbose": verbose,
        "col_names": ["input_size", "output_size", "num_params", "trainable"],
        "col_width": 35,
    }

    if row_settings:
        torchinfo_kwargs["row_settings"] = row_settings

    # Print summary
    torchinfo.summary(**torchinfo_kwargs)

    # Print additional info
    print("-" * 60)
    print(f"Model configuration:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Sequence depth: {seq_depth_val}")
    print(f"  Number of targets: {params_model['num_targets']}")

    # Print total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
