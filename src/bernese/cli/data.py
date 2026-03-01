# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Data preparation command-line interface for bernese.

This module provides the 'bernese data' command for preparing genomic data
for training SeqNN models with Hi-C/coverage data.

Based on the akita_data.py from Basenji.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

from bernese.data import akita


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description="Prepare genomic data for training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Positional arguments
    parser.add_argument(
        "fasta_file",
        help="Genome FASTA file",
    )
    parser.add_argument(
        "targets_file",
        help="Targets file (TSV with target information)",
    )

    # Output options
    parser.add_argument(
        "-o",
        "--out_dir",
        default="data_out",
        help="Output directory",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Continue from checkpoint",
    )

    # Sequence parameters
    parser.add_argument(
        "-l",
        "--seq_length",
        type=int,
        default=131072,
        help="Sequence length in bp",
    )
    parser.add_argument(
        "-c",
        "--crop_bp",
        type=int,
        default=0,
        help="Crop bp from each end",
    )
    parser.add_argument(
        "-w",
        "--pool_width",
        type=int,
        default=128,
        help="Pool width for targets",
    )
    parser.add_argument(
        "--snap",
        type=int,
        default=1,
        help="Snap sequences to multiple of this value",
    )

    # Split parameters
    parser.add_argument(
        "-f",
        "--folds",
        type=int,
        default=None,
        help="Number of cross-validation folds",
    )
    parser.add_argument(
        "-t",
        "--test_pct",
        default="0.05",
        help="Test set proportion or chromosome list (comma-separated)",
    )
    parser.add_argument(
        "-v",
        "--valid_pct",
        default="0.05",
        help="Validation set proportion or chromosome list (comma-separated)",
    )
    parser.add_argument(
        "--stride_train",
        type=float,
        default=1.0,
        help="Stride for training sequences (fraction or bp)",
    )
    parser.add_argument(
        "--stride_test",
        type=float,
        default=1.0,
        help="Stride for test/valid sequences (fraction or bp)",
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=1.0,
        help="Down-sample proportion",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=44,
        help="Random seed",
    )

    # Contig handling
    parser.add_argument(
        "--break_threshold",
        type=int,
        default=8388608,
        help="Break large contigs above this length",
    )
    parser.add_argument(
        "-g",
        "--gaps_file",
        default=None,
        help="Genome assembly gaps BED file",
    )
    parser.add_argument(
        "--limit_bed",
        default=None,
        help="Limit to segments overlapping BED file",
    )

    # Mappability
    parser.add_argument(
        "-u",
        "--umap_bed",
        default=None,
        help="Unmappable regions BED file",
    )
    parser.add_argument(
        "--umap_midpoints",
        default=None,
        help="Midpoints to exclude (for 4C/Hi-C)",
    )
    parser.add_argument(
        "--umap_t",
        type=float,
        default=0.3,
        help="Maximum unmappable bin fraction",
    )

    # Blacklist
    parser.add_argument(
        "-b",
        "--blacklist_bed",
        default=None,
        help="Blacklist regions BED file",
    )

    # Hi-C specific options
    parser.add_argument(
        "-d",
        "--diagonal_offset",
        type=int,
        default=2,
        help="Positions on diagonal to ignore",
    )
    parser.add_argument(
        "-k",
        "--kernel_stddev",
        type=int,
        default=0,
        help="Gaussian kernel stddev for smoothing",
    )
    parser.add_argument(
        "--as_obsexp",
        action="store_true",
        help="Save targets as observed/expected",
    )
    parser.add_argument(
        "--global_obsexp",
        action="store_true",
        help="Use pre-calculated per-chromosome obs/exp",
    )
    parser.add_argument(
        "--no_log",
        action="store_true",
        help="Don't take log for obs/exp",
    )

    # Other options
    parser.add_argument(
        "-p",
        "--processes",
        type=int,
        default=None,
        help="Number of parallel processes",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run locally instead of SLURM",
    )
    parser.add_argument(
        "--split_test",
        action="store_true",
        help="Exit after splitting test set",
    )

    return parser


def parse_pct_or_chr(value: str) -> Any:
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


def process_arguments(args: argparse.Namespace) -> dict[str, Any]:
    """Process and validate arguments.

    Args:
        args: Parsed arguments

    Returns:
        Processed options dictionary
    """
    options = vars(args).copy()

    # Convert strides from fraction to bp
    if options["stride_train"] <= 1:
        options["stride_train"] = int(options["stride_train"] * options["seq_length"])
    else:
        options["stride_train"] = int(options["stride_train"])

    if options["stride_test"] <= 1:
        options["stride_test"] = int(options["stride_test"] * options["seq_length"])
    else:
        options["stride_test"] = int(options["stride_test"])

    # Validate snap
    if options["snap"] is not None:
        if options["seq_length"] % options["snap"] != 0:
            raise ValueError("seq_length must be a multiple of snap")
        if options["stride_train"] % options["snap"] != 0:
            raise ValueError("stride_train must be a multiple of snap")
        if options["stride_test"] % options["snap"] != 0:
            raise ValueError("stride_test must be a multiple of snap")

    # Validate break threshold
    if options["break_threshold"] is not None:
        if options["break_threshold"] < options["seq_length"]:
            raise ValueError("break_threshold cannot be less than seq_length")

    # Parse test/valid as pct or chr
    options["test_pct_or_chr"] = parse_pct_or_chr(options["test_pct"])
    options["valid_pct_or_chr"] = parse_pct_or_chr(options["valid_pct"])

    return options


def run_data_prep(options: dict[str, Any]) -> None:
    """Run the data preparation pipeline.

    Args:
        options: Processed options dictionary
    """
    # Set random seeds
    random.seed(options["seed"])
    np.random.seed(options["seed"])

    fasta_file = options["fasta_file"]
    targets_file = options["targets_file"]
    out_dir = options["out_dir"]

    # Create output directory
    if os.path.isdir(out_dir) and not options["restart"]:
        print(f"Remove output directory {out_dir} or use --restart")
        sys.exit(1)
    elif not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Save options
    with open(os.path.join(out_dir, "options.json"), "w") as f:
        json.dump(options, f, indent=4)

    seq_length = options["seq_length"]
    crop_bp = options["crop_bp"]
    pool_width = options["pool_width"]
    diagonal_offset = options["diagonal_offset"]

    # =========================================================================
    # Define genomic contigs
    # =========================================================================
    if not options["restart"]:
        print("Loading genome...")
        chrom_contigs = akita.load_genome(fasta_file)

        # Remove gaps
        if options["gaps_file"]:
            print("Splitting at gaps...")
            chrom_contigs = akita.split_contigs_by_gaps(chrom_contigs, options["gaps_file"])

        # Convert to list
        contigs = akita.contigs_to_list(chrom_contigs)
        print(f"Loaded {len(contigs)} contigs")

        # Limit to BED file
        if options["limit_bed"] is not None:
            print("Limiting to BED regions...")
            contigs = akita.limit_contigs_to_bed(contigs, options["limit_bed"])

        # Filter for large enough
        seq_tlength = seq_length - 2 * crop_bp
        contigs = [ctg for ctg in contigs if ctg.end - ctg.start >= seq_tlength]
        print(f"Filtered to {len(contigs)} contigs >= {seq_tlength} bp")

        # Break large contigs
        if options["break_threshold"] is not None:
            print(f"Breaking contigs > {options['break_threshold']} bp...")
            contigs = akita.break_large_contigs(contigs, options["break_threshold"])
            print(f"Broken into {len(contigs)} contigs")

    # =========================================================================
    # Divide between train/valid/test
    # =========================================================================
    # Set up fold labels
    if options["folds"] is not None:
        fold_labels = [f"fold{i}" for i in range(options["folds"])]
        num_folds = options["folds"]
    else:
        fold_labels = ["train", "valid", "test"]
        num_folds = 3

    if not options["restart"]:
        # Get proper fold value
        num_folds = options.get("folds", 3)

        # Divide contigs
        test_pct_or_chr = options["test_pct_or_chr"]
        valid_pct_or_chr = options["valid_pct_or_chr"]

        if options["folds"] is not None:
            fold_contigs = akita.divide_contigs_by_folds(contigs, options["folds"])
        else:
            # Try to parse as float
            try:
                valid_pct = float(valid_pct_or_chr)
                test_pct = float(test_pct_or_chr)

                if not (0 <= valid_pct <= 1 and 0 <= test_pct <= 1):
                    raise ValueError("Percentages must be between 0 and 1")

                fold_contigs = akita.divide_contigs_by_pct(contigs, test_pct, valid_pct)
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

                fold_contigs = akita.divide_contigs_by_chr(contigs, test_chrs, valid_chrs)

        # Rejoin broken contigs within each fold
        for fi in range(len(fold_contigs)):
            fold_contigs[fi] = akita.rejoin_broken_contigs(fold_contigs[fi])

        # Write contigs to BED
        ctg_bed_file = os.path.join(out_dir, "contigs.bed")
        with open(ctg_bed_file, "w") as f:
            for fi in range(len(fold_contigs)):
                for ctg in fold_contigs[fi]:
                    print(f"{ctg.chr}\t{ctg.start}\t{ctg.end}\t{fold_labels[fi]}", file=f)

        print("Contigs written to contigs.bed")

    if options.get("split_test"):
        print("Exiting after split (--split_test)")
        return

    # =========================================================================
    # Define model sequences
    # =========================================================================
    if not options["restart"]:
        fold_mseqs = []

        for fi in range(num_folds):
            if fold_labels[fi] in ["valid", "test"]:
                stride_fold = options["stride_test"]
            else:
                stride_fold = options["stride_train"]

            # Create sequences
            mseqs = akita.create_model_sequences(
                fold_contigs[fi],
                seq_tlength,
                stride_fold,
                options["snap"],
                fold_labels[fi],
            )

            # Shuffle
            random.shuffle(mseqs)

            # Down-sample
            if options["sample"] < 1.0:
                mseqs = random.sample(mseqs, int(options["sample"] * len(mseqs)))

            fold_mseqs.append(mseqs)
            print(f"{fold_labels[fi]}: {len(mseqs)} sequences")

        # Merge into one list
        all_mseqs = [ms for fm in fold_mseqs for ms in fm]

    # =========================================================================
    # Mappability
    # =========================================================================
    if not options["restart"]:
        # Check for bedtools if mappability is requested
        if options["umap_bed"] is not None or options["umap_midpoints"] is not None:
            if shutil.which("bedtools") is None:
                print("Error: bedtools is required for mappability filtering", file=sys.stderr)
                sys.exit(1)

        if options["umap_bed"] is not None:
            print("Annotating mappability...")
            seq_unmap = akita.annotate_mappability(
                all_mseqs,
                options["umap_bed"],
                seq_length,
                pool_width,
            )

            # Filter
            map_mask = seq_unmap.mean(axis=1) < options["umap_t"]
            all_mseqs = [all_mseqs[i] for i in range(len(all_mseqs)) if map_mask[i]]
            seq_unmap = seq_unmap[map_mask, :]

            # Save
            np.save(os.path.join(out_dir, "mseqs_unmap.npy"), seq_unmap)
            print(f"Filtered to {len(all_mseqs)} sequences with <{options['umap_t']} unmappable")

        if options["umap_midpoints"] is not None:
            print("Annotating midpoint mappability...")
            seq_unmap_mid = akita.annotate_mappability(
                all_mseqs,
                options["umap_midpoints"],
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
        akita.write_sequences_bed(seqs_bed_file, all_mseqs, include_labels=True)
    else:
        # Load from existing files
        seqs_bed_file = os.path.join(out_dir, "sequences.bed")
        all_mseqs = akita.read_sequences_bed(seqs_bed_file)

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

    # =========================================================================
    # Copy targets file
    # =========================================================================
    shutil.copy(targets_file, os.path.join(out_dir, "targets.txt"))

    # =========================================================================
    # Write statistics
    # =========================================================================
    fold_seqs = {fold_labels[fi]: len(fold_mseqs[fi]) for fi in range(num_folds)}

    stats = akita.compute_statistics(
        num_targets=1,  # Will be updated based on targets file
        seq_length=seq_length,
        pool_width=pool_width,
        crop_bp=crop_bp,
        diagonal_offset=diagonal_offset,
        fold_seqs=fold_seqs,
    )

    # Update with actual target count
    import pandas as pd

    targets_df = pd.read_csv(targets_file, index_col=0, sep="\t")
    stats["num_targets"] = targets_df.shape[0]

    with open(os.path.join(out_dir, "statistics.json"), "w") as f:
        json.dump(stats, f, indent=4)

    print("\nData preparation complete!")
    print(f"Output directory: {out_dir}")
    print(f"Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    try:
        options = process_arguments(args)
        run_data_prep(options)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
