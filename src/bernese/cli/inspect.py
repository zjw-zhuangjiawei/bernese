# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Dataset inspection command-line interface for bernese.

This module provides the 'bernese inspect' command for inspecting
prepared datasets.
"""

from pathlib import Path
from typing import Optional

import typer

from bernese.data.backends import HDF5Backend


def inspect(
    data_dir: Path = typer.Argument(..., exists=True, help="Dataset directory to inspect"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Show detailed output"),
) -> None:
    """Display information about a prepared dataset.

    Example:
        bernese inspect data_dir/
        bernese inspect data_dir/ -v
    """
    # Check for manifest.json or sequences.h5
    manifest_path = data_dir / "manifest.json"
    sequences_h5 = data_dir / "sequences.h5"
    
    if not manifest_path.exists() and not sequences_h5.exists():
        raise typer.BadParameter(
            f"Directory '{data_dir}' does not contain a valid dataset. "
            "Expected manifest.json (v2 format) or sequences.h5 (legacy format)."
        )

    # Create backend to load metadata
    try:
        backend = HDF5Backend(data_dir)
        metadata = backend.metadata
    except Exception as e:
        raise typer.BadParameter(
            f"Failed to load dataset from '{data_dir}': {e}"
        )

    # Print header
    print(f"Dataset: {data_dir}")
    print(f"Version: {metadata.version}")
    print("-" * 60)

    # Sequence information
    print("Sequence Information:")
    print(f"  Sequence length: {metadata.seq_length:,} bp")
    print(f"  Sequence depth: {metadata.seq_depth} (channels)")

    # Target information
    print("\nTarget Information:")
    print(f"  Target type: {metadata.target_type}")
    print(f"  Number of targets: {metadata.num_targets}")
    print(f"  Target length: {metadata.target_length}")
    print(f"  Pool width: {metadata.pool_width}")
    if metadata.diagonal_offset > 0:
        print(f"  Diagonal offset: {metadata.diagonal_offset}")

    # Split information
    print("\nSplit Information:")
    total_seqs = 0
    for split in ["train", "valid", "test"]:
        if split in metadata.splits:
            num = metadata.splits[split].num_seqs
            total_seqs += num
            print(f"  {split}: {num:,} sequences")
    print(f"  Total: {total_seqs:,} sequences")

    # Target tracks
    if metadata.targets:
        print("\nTarget Tracks:")
        for i, target in enumerate(metadata.targets):
            print(f"  {i+1}. {target.name}")
            if verbose:
                print(f"     File: {target.file}")
                if target.clip:
                    print(f"     Clip: {target.clip}")

    # Genome information
    if metadata.genome:
        print("\nGenome Information:")
        print(f"  Name: {metadata.genome.name}")
        print(f"  FASTA: {metadata.genome.fasta_path}")

    # Verbose: show sample coordinates
    if verbose:
        print("\nSample Coordinates:")
        for split in ["train", "valid", "test"]:
            if split in metadata.splits:
                num_to_show = min(3, metadata.splits[split].num_seqs)
                coords = backend.get_coordinates(split, list(range(num_to_show)))
                print(f"  {split}:")
                for chrom, start, end in coords:
                    print(f"    {chrom}:{start}-{end}")
