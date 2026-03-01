# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""
BED file handling utilities for genomic sequence extraction.

This module provides functions for reading BED files, extracting sequences from
FASTA files, and writing prediction results as BEDgraph files.

Functions:
    make_bed_seqs: Extract sequences from BED regions centered on each region
    make_ntwise_bed_seqs: Extract sequences centered at every nucleotide
    read_bed_coords: Read BED coordinates
    write_bedgraph: Write predictions as BEDgraph files
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import pandas as pd
import pysam


def make_bed_seqs(
    bed_file: str,
    fasta_file: str,
    seq_len: int,
    stranded: bool = False,
) -> tuple[list[str], list[tuple]]:
    """Return BED regions as sequences and coordinates.

    Extracts sequences centered on each BED region, extended to seq_len.

    Args:
        bed_file: Path to BED file
        fasta_file: Path to FASTA genome file
        seq_len: Length of sequences to extract
        stranded: If True, reverse complement minus-strand sequences

    Returns:
        Tuple of (list of DNA sequences, list of coordinate tuples).
        Coordinate tuples are (chrom, start, end) or (chrom, start, end, strand).

    Example:
        >>> seqs, coords = make_bed_seqs("peaks.bed", "genome.fa", 1000)
    """
    fasta = pysam.Fastafile(fasta_file)

    seqs_dna = []
    seqs_coords = []

    with open(bed_file) as f:
        for line in f:
            parts = line.split()
            chrom = parts[0]
            start = int(float(parts[1]))
            end = int(float(parts[2]))
            strand = parts[5] if len(parts) >= 6 else "+"

            # Determine sequence limits (centered on region)
            mid = (start + end) // 2
            seq_start = mid - seq_len // 2
            seq_end = seq_start + seq_len

            # Save coordinates
            if stranded:
                seqs_coords.append((chrom, seq_start, seq_end, strand))
            else:
                seqs_coords.append((chrom, seq_start, seq_end))

            # Initialize sequence
            seq_dna = ""

            # Add N's for left overreach
            if seq_start < 0:
                seq_dna = "N" * (-seq_start)
                seq_start = 0

            # Get DNA from FASTA
            seq_dna += fasta.fetch(chrom, seq_start, seq_end).upper()

            # Add N's for right overreach
            if len(seq_dna) < seq_len:
                seq_dna += "N" * (seq_len - len(seq_dna))

            # Reverse complement for minus strand
            if stranded and strand == "-":
                seq_dna = dna_rc(seq_dna)

            seqs_dna.append(seq_dna)

    fasta.close()

    return seqs_dna, seqs_coords


def make_ntwise_bed_seqs(
    bed_file: str,
    fasta_file: str,
    seq_len: int,
    stranded: bool = False,
) -> tuple[dict, dict, list[int]]:
    """Return BED regions as sequences centered at every nucleotide.

    For each BED region of length N, creates N sequences - one centered
    at each nucleotide position.

    Args:
        bed_file: Path to BED file
        fasta_file: Path to FASTA genome file
        seq_len: Length of sequences to extract
        stranded: If True, reverse complement minus-strand sequences

    Returns:
        Tuple of:
            - dict: DNA sequences indexed by BED line number
            - dict: Coordinates indexed by BED line number
            - list: Lengths of each BED region

    Example:
        >>> seqs_dict, coords_dict, lengths = make_ntwise_bed_seqs("peaks.bed", "genome.fa", 1000)
    """
    fasta = pysam.Fastafile(fasta_file)

    seqs_dna: dict[int, list[str]] = {}
    seqs_coords: dict[int, list[tuple]] = {}
    ism_lengths: list[int] = []

    with open(bed_file) as f:
        for il, line in enumerate(f):
            parts = line.split()
            chrom = parts[0]
            start = int(float(parts[1]))
            end = int(float(parts[2]))
            strand = parts[5] if len(parts) >= 6 else "+"

            num_pos = end - start

            seqs_dna[il] = []
            seqs_coords[il] = []

            for ni in range(num_pos):
                # Center on each nucleotide
                mid = start + ni
                seq_start = mid - seq_len // 2
                seq_end = seq_start + seq_len

                # Save coordinates
                if stranded:
                    seqs_coords[il].append((chrom, seq_start, seq_end, strand))
                else:
                    seqs_coords[il].append((chrom, seq_start, seq_end))

                # Initialize sequence
                seq_dna = ""

                # Add N's for left overreach
                if seq_start < 0:
                    seq_dna = "N" * (-seq_start)
                    seq_start = 0

                # Get DNA from FASTA
                seq_dna += fasta.fetch(chrom, seq_start, seq_end).upper()

                # Add N's for right overreach
                if len(seq_dna) < seq_len:
                    seq_dna += "N" * (seq_len - len(seq_dna))

                # Reverse complement for minus strand
                if stranded and strand == "-":
                    seq_dna = dna_rc(seq_dna)

                seqs_dna[il].append(seq_dna)

            ism_lengths.append(num_pos)

    fasta.close()

    return seqs_dna, seqs_coords, ism_lengths


def read_bed_coords(
    bed_file: str,
    seq_len: int,
) -> list[tuple[int, int, int]]:
    """Read BED coordinates and extend to specified length.

    Args:
        bed_file: Path to BED file
        seq_len: Length to extend sequences to

    Returns:
        List of (chrom, start, end) coordinate tuples.

    Example:
        >>> coords = read_bed_coords("peaks.bed", 1000)
    """
    seqs_coords = []

    with open(bed_file) as f:
        for line in f:
            parts = line.split()
            chrom = parts[0]
            start = int(float(parts[1]))
            end = int(float(parts[2]))

            # Determine sequence limits (centered on region)
            mid = (start + end) // 2
            seq_start = mid - seq_len // 2
            seq_end = seq_start + seq_len

            seqs_coords.append((chrom, seq_start, seq_end))

    return seqs_coords


def write_bedgraph(
    preds: np.ndarray,
    targets: np.ndarray,
    data_dir: str,
    out_dir: str,
    split_label: str,
    bedgraph_indexes: Optional[list[int]] = None,
) -> None:
    """Write BEDgraph files for predictions and targets.

    Args:
        preds: Predictions array of shape (num_seqs, target_length, num_targets)
        targets: Targets array of shape (num_seqs, target_length, num_targets)
        data_dir: Data directory containing sequences.bed and statistics.json
        out_dir: Output directory for BEDgraph files
        split_label: Split label (e.g., 'train', 'valid', 'test')
        bedgraph_indexes: List of target indexes to write. If None, writes all.

    Example:
        >>> write_bedgraph(preds, targets, "data/", "output/", "test")
    """
    # Get shapes
    num_seqs, target_length, num_targets = targets.shape

    # Set bedgraph indexes
    if bedgraph_indexes is None:
        bedgraph_indexes = list(range(num_targets))

    # Read data parameters
    stats_file = os.path.join(data_dir, "statistics.json")
    with open(stats_file) as f:
        data_stats = json.load(f)
    pool_width = data_stats["pool_width"]

    # Read sequence positions
    seqs_df = pd.read_csv(
        os.path.join(data_dir, "sequences.bed"),
        sep="\t",
        names=["chr", "start", "end", "split"],
    )
    seqs_df = seqs_df[seqs_df.split == split_label]
    assert seqs_df.shape[0] == num_seqs, f"Expected {num_seqs} sequences, got {seqs_df.shape[0]}"

    # Initialize output directory
    os.makedirs(out_dir, exist_ok=True)

    for ti in bedgraph_indexes:
        # Slice preds/targets
        preds_ti = preds[:, :, ti]
        targets_ti = targets[:, :, ti]

        # Open output files
        preds_out = open(os.path.join(out_dir, f"preds_t{ti}.bedgraph"), "w")
        targets_out = open(os.path.join(out_dir, f"targets_t{ti}.bedgraph"), "w")

        # Write predictions and targets
        for si, seq in enumerate(seqs_df.itertuples()):
            bin_start = seq.start
            for bi in range(target_length):
                bin_end = bin_start + pool_width

                # Write prediction
                print(f"{seq.chr}\t{bin_start}\t{bin_end}\t{preds_ti[si, bi]:.2f}", file=preds_out)

                # Write target
                print(
                    f"{seq.chr}\t{bin_start}\t{bin_end}\t{targets_ti[si, bi]:.2f}", file=targets_out
                )

                bin_start = bin_end

        preds_out.close()
        targets_out.close()


def extract_sequence_from_fasta(
    fasta_file: str,
    chrom: str,
    start: int,
    end: int,
    strand: str = "+",
) -> str:
    """Extract a DNA sequence from a FASTA file.

    Args:
        fasta_file: Path to FASTA file
        chrom: Chromosome name
        start: Start position (0-indexed)
        end: End position (0-indexed, exclusive)
        strand: Strand (+ or -). If -, returns reverse complement.

    Returns:
        DNA sequence string.

    Example:
        >>> seq = extract_sequence_from_fasta("genome.fa", "chr1", 1000, 2000)
    """
    fasta = pysam.Fastafile(fasta_file)

    # Handle negative coordinates
    if start < 0:
        start = 0

    seq = fasta.fetch(chrom, start, end).upper()

    if strand == "-":
        seq = dna_rc(seq)

    fasta.close()

    return seq


def write_bed(
    bed_file: str,
    intervals: list[tuple],
    names: Optional[list[str]] = None,
    scores: Optional[list[float]] = None,
    strands: Optional[list[str]] = None,
) -> None:
    """Write intervals to a BED file.

    Args:
        bed_file: Output BED file path
        intervals: List of (chrom, start, end) tuples
        names: Optional list of names (fourth column)
        scores: Optional list of scores (fifth column)
        strands: Optional list of strands (sixth column)

    Example:
        >>> intervals = [("chr1", 1000, 2000), ("chr2", 5000, 6000)]
        >>> write_bed("output.bed", intervals)
    """
    with open(bed_file, "w") as f:
        for i, (chrom, start, end) in enumerate(intervals):
            line = f"{chrom}\t{start}\t{end}"

            if names is not None:
                line += f"\t{names[i]}"
            if scores is not None:
                line += f"\t{scores[i]}"
            if strands is not None:
                line += f"\t{strands[i]}"

            print(line, file=f)


def read_bed(
    bed_file: str,
    require_strand: bool = False,
) -> pd.DataFrame:
    """Read a BED file into a DataFrame.

    Args:
        bed_file: Path to BED file
        require_strand: If True, expect strand column

    Returns:
        DataFrame with columns: chrom, start, end, [name, score, strand]

    Example:
        >>> df = read_bed("peaks.bed")
    """
    col_names = ["chrom", "start", "end"]

    if require_strand:
        col_names.extend(["name", "score", "strand"])
    else:
        # Try to detect optional columns
        with open(bed_file) as f:
            first_line = f.readline()
            num_cols = len(first_line.split())
            if num_cols >= 4:
                col_names.append("name")
            if num_cols >= 5:
                col_names.append("score")
            if num_cols >= 6:
                col_names.append("strand")

    df = pd.read_csv(bed_file, sep="\t", names=col_names)
    return df


# Helper function (imported from dna module)
def dna_rc(seq: str) -> str:
    """Reverse complement a DNA sequence.

    Args:
        seq: DNA sequence string.

    Returns:
        Reverse complement of the input sequence.
    """
    complement_map = str.maketrans("ATCGatcg", "TAGCtagc")
    return seq.translate(complement_map)[::-1]
