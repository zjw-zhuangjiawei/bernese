# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Data preparation utilities for Akita/Hi-C style data.

This module provides functions for preparing genomic data for training,
including loading FASTA sequences, creating model sequences, handling
mappability, and reading Hi-C/coverage data.

Based on the akita_data.py from Basenji.
"""

from __future__ import annotations

import collections
import gzip
import heapq
import json
import math
import os
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

import h5py
import numpy as np
import pandas as pd


# Named tuples for genomic data
Contig = collections.namedtuple("Contig", ["chr", "start", "end"])
ModelSeq = collections.namedtuple("ModelSeq", ["chr", "start", "end", "label"])


# =============================================================================
# FASTA and Genome Loading
# =============================================================================


def load_genome(fasta_file: str) -> dict[str, list[tuple[int, int]]]:
    """Load chromosomes from a FASTA file.

    Args:
        fasta_file: Path to the FASTA file

    Returns:
        Dictionary mapping chromosome names to list of (start, end) tuples
    """
    chrom_contigs: dict[str, list[tuple[int, int]]] = {}

    # Handle gzipped files
    if fasta_file.endswith(".gz"):
        fopen = gzip.open(fasta_file, "rt")
    else:
        fopen = open(fasta_file, "r")

    with fopen as f:
        current_chrom = None
        chrom_start = 0

        for line in f:
            if line.startswith(">"):
                # Save previous chromosome
                if current_chrom is not None:
                    chrom_contigs[current_chrom] = [(0, chrom_start)]

                # Parse new chromosome
                current_chrom = line[1:].strip().split()[0]
                chrom_start = 0
            else:
                chrom_start += len(line.strip())

        # Save last chromosome
        if current_chrom is not None:
            chrom_contigs[current_chrom] = [(0, chrom_start)]

    return chrom_contigs


def split_contigs_by_gaps(
    contigs: dict[str, list[tuple[int, int]]], gaps_file: str
) -> dict[str, list[tuple[int, int]]]:
    """Split contigs at gap regions.

    Args:
        contigs: Dictionary of chromosome -> list of (start, end)
        gaps_file: BED file with gap regions

    Returns:
        Updated contigs dictionary with gaps split
    """
    # Read gap positions
    gaps: dict[str, list[tuple[int, int]]] = {}
    with open(gaps_file, "r") as f:
        for line in f:
            parts = line.split()
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            gaps.setdefault(chrom, []).append((start, end))

    # Split contigs at gaps
    new_contigs: dict[str, list[tuple[int, int]]] = {}

    for chrom, positions in contigs.items():
        if chrom not in gaps or not gaps[chrom]:
            new_contigs[chrom] = positions
            continue

        chrom_gaps = sorted(gaps[chrom])
        new_positions = []

        for start, end in positions:
            current = start
            for gap_start, gap_end in chrom_gaps:
                if gap_end < start:
                    continue
                if gap_start > end:
                    break

                # Add region before gap
                if current < gap_start:
                    new_positions.append((current, gap_start))

                current = max(current, gap_end)

            # Add final region
            if current < end:
                new_positions.append((current, end))

        new_contigs[chrom] = new_positions

    return new_contigs


def contigs_to_list(
    chrom_contigs: dict[str, list[tuple[int, int]]],
) -> list[Contig]:
    """Convert chromosome contigs to a flat list of Contig namedtuples.

    Args:
        chrom_contigs: Dictionary mapping chromosome to list of (start, end)

    Returns:
        List of Contig namedtuples
    """
    contigs = []
    for chrom, positions in chrom_contigs.items():
        for start, end in positions:
            contigs.append(Contig(chrom, start, end))
    return contigs


# =============================================================================
# Model Sequence Generation
# =============================================================================


def create_model_sequences(
    contigs: list[Contig],
    seq_length: int,
    stride: int,
    snap: int = 1,
    label: Optional[str] = None,
) -> list[ModelSeq]:
    """Break contigs into model sequences.

    Args:
        contigs: List of Contig namedtuples
        seq_length: Length of each model sequence
        stride: Step size between sequences
        snap: Snap start positions to multiples of this value
        label: Label for these sequences (e.g., 'train', 'valid')

    Returns:
        List of ModelSeq namedtuples
    """
    mseqs = []
    for ctg in contigs:
        # Snap start position
        seq_start = int(math.ceil(ctg.start / snap) * snap)
        seq_end = seq_start + seq_length

        while seq_end <= ctg.end:
            mseqs.append(ModelSeq(ctg.chr, seq_start, seq_end, label))
            seq_start += stride
            seq_end += stride

    return mseqs


# =============================================================================
# Contig Splitting
# =============================================================================


def divide_contigs_by_chr(
    contigs: list[Contig],
    test_chrs: list[str],
    valid_chrs: list[str],
) -> tuple[list[Contig], list[Contig], list[Contig]]:
    """Divide contigs into train/valid/test by chromosome.

    Args:
        contigs: List of all contigs
        test_chrs: List of chromosome names for test set
        valid_chrs: List of chromosome names for validation set

    Returns:
        Tuple of (train_contigs, valid_contigs, test_contigs)
    """
    train_contigs = []
    valid_contigs = []
    test_contigs = []

    for ctg in contigs:
        if ctg.chr in test_chrs:
            test_contigs.append(ctg)
        elif ctg.chr in valid_chrs:
            valid_contigs.append(ctg)
        else:
            train_contigs.append(ctg)

    return train_contigs, valid_contigs, test_contigs


def divide_contigs_by_pct(
    contigs: list[Contig],
    test_pct: float,
    valid_pct: float,
    pct_abstain: float = 0.2,
) -> tuple[list[Contig], list[Contig], list[Contig]]:
    """Divide contigs by percentage, trying to match target percentages.

    Args:
        contigs: List of all contigs
        test_pct: Target test percentage
        valid_pct: Target validation percentage
        pct_abstain: Tolerance for abstaining from assigning a contig

    Returns:
        Tuple of (train_contigs, valid_contigs, test_contigs)
    """
    # Sort by length descending
    length_contigs = [(ctg.end - ctg.start, ctg) for ctg in contigs]
    length_contigs.sort(reverse=True)

    total_nt = sum(lc[0] for lc in length_contigs)

    test_nt_aim = test_pct * total_nt
    valid_nt_aim = valid_pct * total_nt
    train_nt_aim = total_nt - valid_nt_aim - test_nt_aim

    train_nt = 0
    valid_nt = 0
    test_nt = 0

    train_contigs = []
    valid_contigs = []
    test_contigs = []

    for ctg_len, ctg in length_contigs:
        # Compute gaps
        test_nt_gap = max(0, test_nt_aim - test_nt)
        valid_nt_gap = max(0, valid_nt_aim - valid_nt)
        train_nt_gap = max(1, train_nt_aim - train_nt)

        # Skip if contig too large
        if ctg_len > pct_abstain * test_nt_gap:
            test_nt_gap = 0
        if ctg_len > pct_abstain * valid_nt_gap:
            valid_nt_gap = 0

        # Compute probabilities
        gap_sum = train_nt_gap + valid_nt_gap + test_nt_gap
        test_pct_gap = test_nt_gap / gap_sum
        valid_pct_gap = valid_nt_gap / gap_sum
        train_pct_gap = train_nt_gap / gap_sum

        # Sample assignment
        ri = np.random.choice(3, p=[train_pct_gap, valid_pct_gap, test_pct_gap])
        if ri == 0:
            train_contigs.append(ctg)
            train_nt += ctg_len
        elif ri == 1:
            valid_contigs.append(ctg)
            valid_nt += ctg_len
        else:
            test_contigs.append(ctg)
            test_nt += ctg_len

    return train_contigs, valid_contigs, test_contigs


def divide_contigs_by_folds(
    contigs: list[Contig],
    num_folds: int,
) -> list[list[Contig]]:
    """Divide contigs into cross-validation folds.

    Args:
        contigs: List of all contigs
        num_folds: Number of folds

    Returns:
        List of fold contig lists
    """
    # Sort by length descending
    length_contigs = [(ctg.end - ctg.start, ctg) for ctg in contigs]
    length_contigs.sort(reverse=True)

    total_nt = sum(lc[0] for lc in length_contigs)
    fold_nt_aim = int(math.ceil(total_nt / num_folds))

    fold_nt = np.zeros(num_folds)
    fold_contigs: list[list[Contig]] = [[] for _ in range(num_folds)]

    for ctg_len, ctg in length_contigs:
        # Compute gaps
        fold_nt_gap = fold_nt_aim - fold_nt
        fold_nt_gap = np.clip(fold_nt_gap, 0, np.inf)

        # Compute probabilities
        fold_prob = fold_nt_gap / fold_nt_gap.sum()

        # Sample fold
        fi = np.random.choice(num_folds, p=fold_prob)
        fold_contigs[fi].append(ctg)
        fold_nt[fi] += ctg_len

    return fold_contigs


def break_large_contigs(
    contigs: list[Contig],
    break_threshold: int,
) -> list[Contig]:
    """Break large contigs into smaller pieces.

    Args:
        contigs: List of contigs
        break_threshold: Maximum contig length

    Returns:
        List of contigs with large ones broken
    """
    # Use heap for efficient processing
    contig_heap = []
    for ctg in contigs:
        ctg_len = ctg.end - ctg.start
        heapq.heappush(contig_heap, (-ctg_len, ctg))

    broken_contigs = []

    while contig_heap:
        neg_len, ctg = heapq.heappop(contig_heap)
        ctg_len = -neg_len

        if ctg_len <= break_threshold:
            broken_contigs.append(ctg)
            continue

        # Break in half
        ctg_mid = ctg.start + ctg_len // 2

        ctg_left = Contig(ctg.chr, ctg.start, ctg_mid)
        ctg_right = Contig(ctg.chr, ctg_mid, ctg.end)

        heapq.heappush(contig_heap, (-(ctg_left.end - ctg_left.start), ctg_left))
        heapq.heappush(contig_heap, (-(ctg_right.end - ctg_right.start), ctg_right))

    return broken_contigs


def rejoin_broken_contigs(contigs: list[Contig]) -> list[Contig]:
    """Rejoin contigs that were previously broken by chromosome.

    Args:
        contigs: List of contigs

    Returns:
        List with contigs rejoined where possible
    """
    # Group by chromosome
    chr_contigs: dict[str, list[Contig]] = {}
    for ctg in contigs:
        chr_contigs.setdefault(ctg.chr, []).append(ctg)

    # Sort within chromosome
    for chrm in chr_contigs:
        chr_contigs[chrm].sort(key=lambda x: x.start)

    # Rejoin adjacent contigs
    result = []
    for chrm, ctgs in chr_contigs.items():
        if not ctgs:
            continue

        ongoing = ctgs[0]
        for i in range(1, len(ctgs)):
            this_ctg = ctgs[i]
            if ongoing.end == this_ctg.start:
                ongoing = Contig(chrm, ongoing.start, this_ctg.end)
            else:
                result.append(ongoing)
                ongoing = this_ctg

        result.append(ongoing)

    return result


def limit_contigs_to_bed(
    contigs: list[Contig],
    filter_bed: str,
) -> list[Contig]:
    """Limit contigs to regions overlapping a BED file.

    Args:
        contigs: List of contigs
        filter_bed: BED file to filter by

    Returns:
        Filtered list of contigs
    """
    # Write contigs to temp file
    fd, temp_bed = tempfile.mkstemp()
    with os.fdopen(fd, "w") as f:
        for ctg in contigs:
            f.write(f"{ctg.chr}\t{ctg.start}\t{ctg.end}\n")

    # Intersect with filter BED
    result = []
    p = subprocess.Popen(
        ["bedtools", "intersect", "-a", temp_bed, "-b", filter_bed],
        stdout=subprocess.PIPE,
        text=True,
    )

    for line in p.stdout:
        parts = line.strip().split()
        chrom = parts[0]
        start = int(parts[1])
        end = int(parts[2])
        result.append(Contig(chrom, start, end))

    p.wait()
    os.remove(temp_bed)

    return result


# =============================================================================
# Mappability
# =============================================================================


def annotate_mappability(
    mseqs: list[ModelSeq],
    unmap_bed: str,
    seq_length: int,
    pool_width: int,
) -> np.ndarray:
    """Annotate sequences with mappability information.

    Args:
        mseqs: List of model sequences
        unmap_bed: BED file with unmappable regions
        seq_length: Sequence length
        pool_width: Pool width for binning

    Returns:
        Binary array (num_seqs x pool_seq_length) indicating unmappable bins
    """
    # Write sequences to temp file
    fd, seqs_bed = tempfile.mkstemp()
    with os.fdopen(fd, "w") as f:
        for ms in mseqs:
            f.write(f"{ms.chr}\t{ms.start}\t{ms.end}\n")

    # Hash sequences to indexes
    chr_start_idx: dict[tuple[str, int], int] = {}
    for i, ms in enumerate(mseqs):
        chr_start_idx[(ms.chr, ms.start)] = i

    # Initialize unmappable array
    pool_seq_length = seq_length // pool_width
    seqs_unmap = np.zeros((len(mseqs), pool_seq_length), dtype=bool)

    # Intersect with unmappable regions
    p = subprocess.Popen(
        ["bedtools", "intersect", "-wo", "-a", seqs_bed, "-b", unmap_bed],
        stdout=subprocess.PIPE,
        text=True,
    )

    for line in p.stdout:
        parts = line.strip().split()

        seq_chrom = parts[0]
        seq_start = int(parts[1])
        seq_end = int(parts[2])
        seq_key = (seq_chrom, seq_start)

        if seq_key not in chr_start_idx:
            continue

        idx = chr_start_idx[seq_key]

        unmap_start = int(parts[4])
        unmap_end = int(parts[5])

        overlap_start = max(seq_start, unmap_start)
        overlap_end = min(seq_end, unmap_end)

        pool_start = math.floor((overlap_start - seq_start) / pool_width)
        pool_end = math.ceil((overlap_end - seq_start) / pool_width)

        # Skip minor overlaps
        first_start = seq_start + pool_start * pool_width
        first_end = first_start + pool_width
        first_overlap = first_end - overlap_start
        if first_overlap < 0.1 * pool_width:
            pool_start += 1

        last_start = seq_start + (pool_end - 1) * pool_width
        last_overlap = overlap_end - last_start
        if last_overlap < 0.1 * pool_width:
            pool_end -= 1

        # Mark as unmappable
        seqs_unmap[idx, pool_start:pool_end] = True

    p.wait()
    os.remove(seqs_bed)

    return seqs_unmap


def filter_by_mappability(
    mseqs: list[ModelSeq],
    seq_unmap: np.ndarray,
    threshold: float = 0.5,
) -> tuple[list[ModelSeq], np.ndarray]:
    """Filter sequences by mappability threshold.

    Args:
        mseqs: List of model sequences
        seq_unmap: Unmappable array from annotate_mappability
        threshold: Maximum allowed unmappable fraction

    Returns:
        Tuple of (filtered_mseqs, filtered_unmap)
    """
    # Compute mean unmappable fraction
    map_mask = seq_unmap.mean(axis=1) < threshold

    filtered_mseqs = [mseqs[i] for i in range(len(mseqs)) if map_mask[i]]
    filtered_unmap = seq_unmap[map_mask, :]

    return filtered_mseqs, filtered_unmap


# =============================================================================
# BED File Operations
# =============================================================================


def write_sequences_bed(
    bed_file: str,
    mseqs: list[ModelSeq],
    include_labels: bool = False,
) -> None:
    """Write model sequences to BED file.

    Args:
        bed_file: Output BED file path
        mseqs: List of model sequences
        include_labels: Whether to include sequence labels
    """
    with open(bed_file, "w") as f:
        for ms in mseqs:
            line = f"{ms.chr}\t{ms.start}\t{ms.end}"
            if include_labels and ms.label:
                line += f"\t{ms.label}"
            print(line, file=f)


def read_sequences_bed(bed_file: str) -> list[ModelSeq]:
    """Read model sequences from BED file.

    Args:
        bed_file: Input BED file path

    Returns:
        List of ModelSeq namedtuples
    """
    mseqs = []
    with open(bed_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            ms = ModelSeq(
                parts[0],
                int(parts[1]),
                int(parts[2]),
                parts[3] if len(parts) > 3 else None,
            )
            mseqs.append(ms)
    return mseqs


# =============================================================================
# Statistics
# =============================================================================


def compute_statistics(
    num_targets: int,
    seq_length: int,
    pool_width: int,
    crop_bp: int,
    diagonal_offset: int,
    fold_seqs: dict[str, int],
) -> dict[str, Any]:
    """Compute dataset statistics.

    Args:
        num_targets: Number of target tracks
        seq_length: Sequence length in bp
        pool_width: Pooling width
        crop_bp: Cropped bp from each end
        diagonal_offset: Diagonal offset for Hi-C
        fold_seqs: Dictionary mapping fold name to sequence count

    Returns:
        Statistics dictionary
    """
    stats: dict[str, Any] = {}

    stats["num_targets"] = num_targets
    stats["seq_length"] = seq_length
    stats["seq_1hot"] = True
    stats["pool_width"] = pool_width
    stats["crop_bp"] = crop_bp
    stats["diagonal_offset"] = diagonal_offset

    # Compute target length (triangular for Hi-C)
    target_length = seq_length - 2 * crop_bp
    target_length = target_length // pool_width
    target_length = target_length - diagonal_offset
    target_length = target_length * (target_length + 1) // 2

    stats["target_length"] = target_length

    # Add sequence counts for each fold
    for fold_name, num_seqs in fold_seqs.items():
        stats[f"{fold_name}_seqs"] = num_seqs

    return stats


# =============================================================================
# HDF5 Writing
# =============================================================================


def write_hdf5_dataset(
    output_dir: str,
    fasta_file: str,
    seqs_bed: str,
    targets_file: str,
    fold_mseqs: dict[str, list[ModelSeq]],
    seq_length: int = 131072,
    seq_depth: int = 4,
) -> dict[str, Any]:
    """Write sequences and targets to HDF5 format.

    This is a simplified version that writes pre-extracted data.
    For full functionality, additional helper scripts would be needed.

    Args:
        output_dir: Output directory
        fasta_file: FASTA file path
        seqs_bed: Sequences BED file
        targets_file: Targets file
        fold_mseqs: Dictionary mapping fold to sequences
        seq_length: Sequence length
        seq_depth: Sequence depth (4 for DNA)

    Returns:
        Statistics dictionary
    """
    os.makedirs(output_dir, exist_ok=True)

    # Copy targets file
    import shutil

    shutil.copy(targets_file, os.path.join(output_dir, "targets.txt"))

    # Load targets info
    targets_df = pd.read_csv(targets_file, index_col=0, sep="\t")
    num_targets = targets_df.shape[0]

    # For now, create placeholder statistics
    # Full implementation would extract sequences and targets from FASTA/BED
    fold_seqs = {fold: len(mseqs) for fold, mseqs in fold_mseqs.items()}

    return {
        "num_targets": num_targets,
        "seq_length": seq_length,
        "seq_1hot": True,
        "pool_width": 128,
        "crop_bp": 0,
        "target_length": (seq_length // 128) * (seq_length // 128 + 1) // 2,
        **fold_seqs,
    }
