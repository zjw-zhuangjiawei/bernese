# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Data preparation command-line interface for bernese.

This module provides the 'bernese prepare' command for preparing genomic data
for training SeqNN models with Hi-C/coverage data.

Based on the akita_data.py from Basenji.
"""

import json
import os
import random
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from bernese.data import contacts

# Create internal app for decorators - not exposed as subcommand
_app = typer.Typer(help="Prepare genomic data for training SeqNN models.")


def parse_pct_or_chr(value: str) -> float | list[str]:
    """Parse percentage or chromosome list.

    Args:
        value: String like '0.05' or 'chr1,chr2'

    Returns:
        Float (percentage) or list of strings (chromosomes)
    """
    try:
        return float(value)
    except ValueError:
        return [c.strip() for c in value.split(",")]


@_app.command()
def prepare(
    fasta_file: Path = typer.Argument(..., exists=True, help="Genome FASTA file"),
    targets_file: Path = typer.Argument(..., exists=True, help="Targets file (TSV with target information)"),
    out_dir: Path = typer.Option("data_out", "-o", help="Output directory"),
    restart: bool = typer.Option(False, "--restart", help="Continue from checkpoint"),
    seq_length: int = typer.Option(131072, "-l", "--seq_length", help="Sequence length in bp"),
    crop_bp: int = typer.Option(0, "-c", "--crop_bp", help="Crop bp from each end"),
    pool_width: int = typer.Option(128, "-w", "--pool_width", help="Pool width for targets"),
    snap: int = typer.Option(1, "--snap", help="Snap sequences to multiple of this value"),
    folds: Optional[int] = typer.Option(None, "-f", "--folds", help="Number of cross-validation folds"),
    test_pct: str = typer.Option("0.05", "-t", "--test_pct", help="Test set proportion or chromosome list (comma-separated)"),
    valid_pct: str = typer.Option("0.05", "-v", "--valid_pct", help="Validation set proportion or chromosome list (comma-separated)"),
    stride_train: float = typer.Option(1.0, "--stride_train", help="Stride for training sequences (fraction or bp)"),
    stride_test: float = typer.Option(1.0, "--stride_test", help="Stride for test/valid sequences (fraction or bp)"),
    sample: float = typer.Option(1.0, "--sample", help="Down-sample proportion"),
    seed: int = typer.Option(44, "--seed", help="Random seed"),
    break_threshold: int = typer.Option(8388608, "--break_threshold", help="Break large contigs above this length"),
    gaps_file: Optional[Path] = typer.Option(None, "-g", "--gaps_file", help="Genome assembly gaps BED file"),
    limit_bed: Optional[Path] = typer.Option(None, "--limit_bed", help="Limit to segments overlapping BED file"),
    umap_bed: Optional[Path] = typer.Option(None, "-u", "--umap_bed", help="Unmappable regions BED file"),
    umap_midpoints: Optional[Path] = typer.Option(None, "--umap_midpoints", help="Midpoints to exclude (for 4C/Hi-C)"),
    umap_t: float = typer.Option(0.3, "--umap_t", help="Maximum unmappable bin fraction"),
    blacklist_bed: Optional[Path] = typer.Option(None, "-b", "--blacklist_bed", help="Blacklist regions BED file"),
    diagonal_offset: int = typer.Option(2, "-d", "--diagonal_offset", help="Positions on diagonal to ignore"),
    kernel_stddev: int = typer.Option(0, "-k", "--kernel_stddev", help="Gaussian kernel stddev for smoothing"),
    as_obsexp: bool = typer.Option(False, "--as_obsexp", help="Save targets as observed/expected"),
    global_obsexp: bool = typer.Option(False, "--global_obsexp", help="Use pre-calculated per-chromosome obs/exp"),
    no_log: bool = typer.Option(False, "--no_log", help="Don't take log for obs/exp"),
    processes: Optional[int] = typer.Option(None, "-p", "--processes", help="Number of parallel processes"),
    local: bool = typer.Option(False, "--local", help="Run locally instead of SLURM"),
    split_test: bool = typer.Option(False, "--split_test", help="Exit after splitting test set"),
    write_hdf5: bool = typer.Option(False, "--write_hdf5", help="Write HDF5 training files"),
    hdf5_seqs_per_file: int = typer.Option(128, "--hdf5_seqs_per_file", help="Number of sequences per HDF5 file"),
) -> None:
    """Prepare genomic data for training.

    This command processes a genome FASTA file and targets file to create
    training data for SeqNN models with Hi-C/coverage data.

    Example:
        bernese prepare genome.fa targets.tsv -o data_out
    """
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)

    # Validate snap
    if seq_length % snap != 0:
        raise typer.BadParameter("seq_length must be a multiple of snap")
    if stride_train <= 1:
        stride_train_bp = int(stride_train * seq_length)
    else:
        stride_train_bp = int(stride_train)
    if stride_test <= 1:
        stride_test_bp = int(stride_test * seq_length)
    else:
        stride_test_bp = int(stride_test)

    if stride_train_bp % snap != 0:
        raise typer.BadParameter("stride_train must be a multiple of snap")
    if stride_test_bp % snap != 0:
        raise typer.BadParameter("stride_test must be a multiple of snap")

    # Validate break threshold
    if break_threshold < seq_length:
        raise typer.BadParameter("break_threshold cannot be less than seq_length")

    # Parse test/valid as pct or chr
    test_pct_or_chr = parse_pct_or_chr(test_pct)
    valid_pct_or_chr = parse_pct_or_chr(valid_pct)

    # Create output directory
    if os.path.isdir(out_dir) and not restart:
        raise typer.BadParameter(f"Remove output directory {out_dir} or use --restart")
    elif not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Save options
    options = {
        "fasta_file": str(fasta_file),
        "targets_file": str(targets_file),
        "out_dir": str(out_dir),
        "restart": restart,
        "seq_length": seq_length,
        "crop_bp": crop_bp,
        "pool_width": pool_width,
        "snap": snap,
        "folds": folds,
        "test_pct": test_pct,
        "valid_pct": valid_pct,
        "stride_train": stride_train,
        "stride_test": stride_test,
        "sample": sample,
        "seed": seed,
        "break_threshold": break_threshold,
        "gaps_file": str(gaps_file) if gaps_file else None,
        "limit_bed": str(limit_bed) if limit_bed else None,
        "umap_bed": str(umap_bed) if umap_bed else None,
        "umap_midpoints": str(umap_midpoints) if umap_midpoints else None,
        "umap_t": umap_t,
        "blacklist_bed": str(blacklist_bed) if blacklist_bed else None,
        "diagonal_offset": diagonal_offset,
        "kernel_stddev": kernel_stddev,
        "as_obsexp": as_obsexp,
        "global_obsexp": global_obsexp,
        "no_log": no_log,
        "processes": processes,
        "local": local,
        "split_test": split_test,
    }

    with open(os.path.join(out_dir, "options.json"), "w") as f:
        json.dump(options, f, indent=4)

    seq_tlength = seq_length - 2 * crop_bp

    if not restart:
        print("Loading genome...")
        chrom_contigs = contacts.load_genome(fasta_file)

        # Remove gaps
        if gaps_file:
            print("Splitting at gaps...")
            from bernese.data import genomics
            chrom_contigs = genomics.split_contigs_by_gaps(chrom_contigs, gaps_file)

        # Convert to list
        contigs = contacts.contigs_to_list(chrom_contigs)
        print(f"Loaded {len(contigs)} contigs")

        # Limit to BED file
        if limit_bed is not None:
            print("Limiting to BED regions...")
            contigs = contacts.limit_contigs_to_bed(contigs, limit_bed)

        # Filter for large enough
        contigs = [ctg for ctg in contigs if ctg.end - ctg.start >= seq_tlength]
        print(f"Filtered to {len(contigs)} contigs >= {seq_tlength} bp")

        # Break large contigs
        print(f"Breaking contigs > {break_threshold} bp...")
        contigs = contacts.break_large_contigs(contigs, break_threshold)
        print(f"Broken into {len(contigs)} contigs")

    # Set up fold labels
    if folds is not None:
        fold_labels = [f"fold{i}" for i in range(folds)]
        num_folds = folds
    else:
        fold_labels = ["train", "valid", "test"]
        num_folds = 3

    if not restart:
        # Divide contigs
        if folds is not None:
            fold_contigs = contacts.divide_contigs_by_folds(contigs, folds)
        else:
            # Try to parse as float
            try:
                valid_pct_val = float(valid_pct_or_chr)
                test_pct_val = float(test_pct_or_chr)

                if not (0 <= valid_pct_val <= 1 and 0 <= test_pct_val <= 1):
                    raise ValueError("Percentages must be between 0 and 1")

                fold_contigs = contacts.divide_contigs_by_pct(contigs, test_pct_val, valid_pct_val)
                # Convert tuple to list for mutability
                fold_contigs = list(fold_contigs)
            except (ValueError, TypeError):
                # Parse as chromosome lists
                valid_chrs = (
                    valid_pct_or_chr
                    if isinstance(valid_pct_or_chr, list)
                    else valid_pct_or_chr.split(",")
                )
                test_chrs = (
                    test_pct_or_chr
                    if isinstance(test_pct_or_chr, list)
                    else test_pct_or_chr.split(",")
                )

                fold_contigs = contacts.divide_contigs_by_chr(contigs, test_chrs, valid_chrs)
                # Convert tuple to list for mutability
                fold_contigs = list(fold_contigs)

        # Rejoin broken contigs within each fold
        for fi in range(len(fold_contigs)):
            fold_contigs[fi] = contacts.rejoin_broken_contigs(fold_contigs[fi])

        # Write contigs to BED
        ctg_bed_file = os.path.join(out_dir, "contigs.bed")
        with open(ctg_bed_file, "w") as f:
            for fi in range(len(fold_contigs)):
                for ctg in fold_contigs[fi]:
                    print(f"{ctg.chr}\t{ctg.start}\t{ctg.end}\t{fold_labels[fi]}", file=f)

        print("Contigs written to contigs.bed")

    if split_test:
        print("Exiting after split (--split_test)")
        return

    if not restart:
        fold_mseqs = []

        for fi in range(num_folds):
            if fold_labels[fi] in ["valid", "test"]:
                stride_fold = stride_test_bp
            else:
                stride_fold = stride_train_bp

            # Create sequences
            mseqs = contacts.create_model_sequences(
                fold_contigs[fi],
                seq_tlength,
                stride_fold,
                snap,
                fold_labels[fi],
            )

            # Shuffle
            random.shuffle(mseqs)

            # Down-sample
            if sample < 1.0:
                mseqs = random.sample(mseqs, int(sample * len(mseqs)))

            fold_mseqs.append(mseqs)
            print(f"{fold_labels[fi]}: {len(mseqs)} sequences")

        # Merge into one list
        all_mseqs = [ms for fm in fold_mseqs for ms in fm]

    if not restart:
        # Check for bedtools if mappability is requested
        if umap_bed is not None or umap_midpoints is not None:
            if shutil.which("bedtools") is None:
                raise typer.ClickException("bedtools is required for mappability filtering")

        if umap_bed is not None:
            print("Annotating mappability...")
            seq_unmap = contacts.annotate_mappability(
                all_mseqs,
                umap_bed,
                seq_length,
                pool_width,
            )

            # Filter
            map_mask = seq_unmap.mean(axis=1) < umap_t
            all_mseqs = [all_mseqs[i] for i in range(len(all_mseqs)) if map_mask[i]]
            seq_unmap = seq_unmap[map_mask, :]

            # Save
            np.save(os.path.join(out_dir, "mseqs_unmap.npy"), seq_unmap)
            print(f"Filtered to {len(all_mseqs)} sequences with <{umap_t} unmappable")

        if umap_midpoints is not None:
            print("Annotating midpoint mappability...")
            seq_unmap_mid = contacts.annotate_mappability(
                all_mseqs,
                umap_midpoints,
                seq_length,
                pool_width,
            )

            # Filter to exclude midpoints
            mid_idx = seq_unmap_mid.shape[1] // 2
            map_mask = (seq_unmap_mid[:, mid_idx - 1 : mid_idx + 1].sum(axis=1)) == 0

            all_mseqs = [all_mseqs[i] for i in range(len(all_mseqs)) if map_mask[i]]

            np.save(os.path.join(out_dir, "mseqs_unmap_midpoints.npy"), seq_unmap_mid[map_mask, :])
            print(f"Filtered to {len(all_mseqs)} sequences with mappable midpoints")

        # Write sequences to BED
        print("Writing sequences to BED...")
        seqs_bed_file = os.path.join(out_dir, "sequences.bed")
        contacts.write_sequences_bed(seqs_bed_file, all_mseqs, include_labels=True)
    else:
        # Load from existing files
        seqs_bed_file = os.path.join(out_dir, "sequences.bed")
        all_mseqs = contacts.read_sequences_bed(seqs_bed_file)

        # Rebuild fold lists
        fold_mseqs = [[] for _ in range(num_folds)]
        for ms in all_mseqs:
            if ms.label == "train":
                fold_mseqs[0].append(ms)
            elif ms.label == "valid":
                fold_mseqs[1].append(ms)
            elif ms.label == "test":
                fold_mseqs[2].append(ms)
            elif ms.label and ms.label.startswith("fold"):
                fold_idx = int(ms.label.replace("fold", ""))
                fold_mseqs[fold_idx].append(ms)

    shutil.copy(targets_file, os.path.join(out_dir, "targets.txt"))

    # Read targets from .cool files
    print("Reading Hi-C targets...")
    seqs_cov_dir = os.path.join(out_dir, "seqs_cov")
    os.makedirs(seqs_cov_dir, exist_ok=True)

    targets_df = pd.read_csv(targets_file, index_col=0, sep="\t")
    for ti in range(targets_df.shape[0]):
        genome_hic_file = targets_df["file"].iloc[ti]
        seqs_cov_file = os.path.join(seqs_cov_dir, f"{ti}.h5")

        clip_ti = targets_df["clip"].iloc[ti] if "clip" in targets_df.columns else None

        print(f"  Target {ti}: {genome_hic_file}")

        contacts.read_cool_targets(
            genome_hic_file,
            seqs_bed_file,
            seqs_cov_file,
            crop_bp=crop_bp,
            pool_width=pool_width,
            diagonal_offset=diagonal_offset,
            blacklist_bed=str(blacklist_bed) if blacklist_bed else None,
            as_obsexp=as_obsexp,
            global_obsexp=global_obsexp,
            no_log=no_log,
            clip=clip_ti,
        )

    fold_seqs = {fold_labels[fi]: len(fold_mseqs[fi]) for fi in range(num_folds)}

    stats = contacts.compute_hic_statistics(
        num_targets=1,  # Will be updated based on targets file
        seq_length=seq_length,
        pool_width=pool_width,
        crop_bp=crop_bp,
        diagonal_offset=diagonal_offset,
        fold_seqs=fold_seqs,
    ).to_dict()

    # Update with actual target count
    stats["num_targets"] = targets_df.shape[0]

    with open(os.path.join(out_dir, "statistics.json"), "w") as f:
        json.dump(stats, f, indent=4)

    # Phase 7: Write HDF5 files
    if write_hdf5:
        print("\nWriting HDF5 files...")
        hdf5_dir = os.path.join(out_dir, "hdf5")
        os.makedirs(hdf5_dir, exist_ok=True)

        # Get sequence indices for each fold
        seqs_bed_file = os.path.join(out_dir, "sequences.bed")
        all_mseqs = contacts.read_sequences_bed(seqs_bed_file)

        # Build index mapping for each fold
        fold_start_idx = {}
        current_idx = 0
        for fi, label in enumerate(fold_labels):
            fold_start_idx[label] = current_idx
            current_idx += len(fold_mseqs[fi])

        # Write HDF5 for each fold
        for fi, label in enumerate(fold_labels):
            fold_start = fold_start_idx[label]
            fold_end = fold_start + len(fold_mseqs[fi])

            # Write in chunks
            chunk_size = hdf5_seqs_per_file
            num_chunks = (len(fold_mseqs[fi]) + chunk_size - 1) // chunk_size

            for ci in range(num_chunks):
                chunk_start = ci * chunk_size
                chunk_end = min((ci + 1) * chunk_size, len(fold_mseqs[fi]))

                seq_start = fold_start + chunk_start
                seq_end = fold_start + chunk_end

                output_h5 = os.path.join(hdf5_dir, f"{label}-{ci}.h5")

                print(f"  Writing {label}-{ci}.h5 ({chunk_end - chunk_start} sequences)...")

                contacts.write_sequences_hdf5(
                    str(fasta_file),
                    seqs_bed_file,
                    os.path.join(out_dir, "seqs_cov"),
                    output_h5,
                    start_idx=seq_start,
                    end_idx=seq_end,
                )

        print(f"HDF5 files written to {hdf5_dir}/")

    print("\nData preparation complete!")
    print(f"Output directory: {out_dir}")
    print("Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
