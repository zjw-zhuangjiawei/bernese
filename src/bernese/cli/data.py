# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Data preparation command-line interface for bernese.

This module provides the 'bernese prepare' command for preparing genomic data
for training SeqNN models.
"""

from pathlib import Path

import typer

from bernese.data.preparation import DataPreparator, PreparationConfig

# Create parent app for prepare command group
prepare_app = typer.Typer(help="Prepare genomic data for training SeqNN models.")


@prepare_app.command(name="hic")
def hic(
    fasta_file: Path = typer.Argument(..., exists=True, help="Genome FASTA file"),
    targets_file: Path = typer.Argument(..., exists=True, help="Targets file (TSV with target information)"),
    out_dir: Path = typer.Option("data_out", "-o", help="Output directory"),
    seq_length: int = typer.Option(131072, "-l", "--seq_length", help="Sequence length in bp"),
    crop_bp: int = typer.Option(0, "-c", "--crop_bp", help="Crop bp from each end"),
    pool_width: int = typer.Option(128, "-w", "--pool_width", help="Pool width for targets"),
    diagonal_offset: int = typer.Option(2, "-d", "--diagonal_offset", help="Diagonal offset for Hi-C"),
    test_pct: float = typer.Option(0.05, "--test_pct", help="Test set proportion"),
    valid_pct: float = typer.Option(0.05, "--valid_pct", help="Validation set proportion"),
    stride_train: float = typer.Option(1.0, "--stride_train", help="Stride for training sequences (fraction or bp)"),
    stride_test: float = typer.Option(1.0, "--stride_test", help="Stride for test/valid sequences"),
    sample: float = typer.Option(1.0, "--sample", help="Down-sample proportion"),
    seed: int = typer.Option(44, "--seed", help="Random seed"),
) -> None:
    """Prepare Hi-C data for training.

    This command processes a genome FASTA file and Hi-C targets file to create
    training data in v2 format.

    Example:
        bernese prepare hic genome.fa targets.tsv -o data_out
    """
    _run_preparation(
        fasta_file=fasta_file,
        targets_file=targets_file,
        out_dir=out_dir,
        seq_length=seq_length,
        crop_bp=crop_bp,
        pool_width=pool_width,
        target_type="hic",
        diagonal_offset=diagonal_offset,
        test_pct=test_pct,
        valid_pct=valid_pct,
        stride_train=stride_train,
        stride_test=stride_test,
        sample=sample,
        seed=seed,
    )


@prepare_app.command(name="bigwig")
def bigwig(
    fasta_file: Path = typer.Argument(..., exists=True, help="Genome FASTA file"),
    targets_file: Path = typer.Argument(..., exists=True, help="Targets file (TSV with target information)"),
    out_dir: Path = typer.Option("data_out", "-o", help="Output directory"),
    seq_length: int = typer.Option(131072, "-l", "--seq_length", help="Sequence length in bp"),
    crop_bp: int = typer.Option(0, "-c", "--crop_bp", help="Crop bp from each end"),
    pool_width: int = typer.Option(128, "-w", "--pool_width", help="Pool width for targets"),
    aggregation: str = typer.Option("mean", "-a", "--aggregation", help="Aggregation method: mean, sum, max, min"),
    test_pct: float = typer.Option(0.05, "--test_pct", help="Test set proportion"),
    valid_pct: float = typer.Option(0.05, "--valid_pct", help="Validation set proportion"),
    stride_train: float = typer.Option(1.0, "--stride_train", help="Stride for training sequences (fraction or bp)"),
    stride_test: float = typer.Option(1.0, "--stride_test", help="Stride for test/valid sequences"),
    sample: float = typer.Option(1.0, "--sample", help="Down-sample proportion"),
    seed: int = typer.Option(44, "--seed", help="Random seed"),
) -> None:
    """Prepare BigWig data for training.

    This command processes a genome FASTA file and BigWig targets file to create
    training data in v2 format.

    Example:
        bernese prepare bigwig genome.fa targets.tsv -o data_out
    """
    _run_preparation(
        fasta_file=fasta_file,
        targets_file=targets_file,
        out_dir=out_dir,
        seq_length=seq_length,
        crop_bp=crop_bp,
        pool_width=pool_width,
        target_type="bigwig",
        aggregation=aggregation,
        diagonal_offset=0,
        test_pct=test_pct,
        valid_pct=valid_pct,
        stride_train=stride_train,
        stride_test=stride_test,
        sample=sample,
        seed=seed,
    )


def _run_preparation(
    fasta_file: Path,
    targets_file: Path,
    out_dir: Path,
    seq_length: int,
    crop_bp: int,
    pool_width: int,
    target_type: str,
    diagonal_offset: int = 0,
    aggregation: str = "mean",
    test_pct: float = 0.05,
    valid_pct: float = 0.05,
    stride_train: float = 1.0,
    stride_test: float = 1.0,
    sample: float = 1.0,
    seed: int = 44,
) -> None:
    """Run the data preparation with common parameters."""
    # Parse stride values
    if stride_train <= 1:
        stride_train_bp = int(stride_train * seq_length)
    else:
        stride_train_bp = int(stride_train)

    if stride_test <= 1:
        stride_test_bp = int(stride_test * seq_length)
    else:
        stride_test_bp = int(stride_test)

    # Create config
    config = PreparationConfig(
        seq_length=seq_length,
        crop_bp=crop_bp,
        pool_width=pool_width,
        target_type=target_type,
        diagonal_offset=diagonal_offset,
        test_pct=test_pct,
        valid_pct=valid_pct,
        stride_train=stride_train_bp,
        stride_test=stride_test_bp,
        sample_pct=sample,
        seed=seed,
    )

    # Create output directory
    if out_dir.exists() and (out_dir / "manifest.json").exists():
        raise typer.BadParameter(f"Output directory {out_dir} already has manifest.json. Remove or use a different directory.")

    # Run preparation
    print(f"Preparing dataset...")
    print(f"  Output: {out_dir}")
    print(f"  Genome: {fasta_file}")
    print(f"  Targets: {targets_file}")
    print(f"  Sequence length: {seq_length}")
    print(f"  Target type: {target_type}")
    print()

    preparator = DataPreparator(out_dir, fasta_file, targets_file, config)
    metadata = preparator.prepare()

    print()
    print("Dataset statistics:")
    print(f"  Sequence length: {metadata.seq_length}")
    print(f"  Target length: {metadata.target_length}")
    print(f"  Number of targets: {metadata.num_targets}")
    for split, info in metadata.splits.items():
        print(f"  {split}: {info.num_seqs} sequences")


def main() -> None:
    """Main entry point."""
    prepare_app()


if __name__ == "__main__":
    main()
