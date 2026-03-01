# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""
Hi-C Contact Data Preparation Module.

This module provides functions for preparing Hi-C/contact matrix data for training
SeqNN models. It handles genomic contig processing, sequence extraction, mappability
filtering, and Hi-C matrix extraction from .cool files.

Based on akita_data.py from Basenji.

Classes:
    HicTarget: Target dataset configuration
    HicStats: Statistics for Hi-C data

Functions:
    load_genome: Load genome from FASTA file
    contigs_to_list: Convert chromosome dict to list of Contig
    limit_contigs_to_bed: Filter contigs by BED regions
    break_large_contigs: Break large contigs for parallel processing
    divide_contigs_by_pct: Split contigs by percentage
    divide_contigs_by_chr: Split contigs by chromosome
    divide_contigs_by_folds: Split contigs into k folds
    rejoin_broken_contigs: Rejoin previously broken contigs
    create_model_sequences: Create sequences from contigs
    annotate_mappability: Annotate mappability from BED
    filter_mappability: Filter sequences by mappability
    read_sequences_bed: Read sequences from BED file
    write_sequences_bed: Write sequences to BED file
    read_cool_targets: Extract Hi-C matrices from .cool
    compute_hic_statistics: Compute statistics JSON
    write_sequences_hdf5: Write sequences + targets to HDF5
"""

import collections
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
import pandas as pd
import pysam
import intervaltree
import cooler
from cooltools.lib.numutils import (
    observed_over_expected,
    adaptive_coarsegrain,
    interpolate_bad_singletons,
    set_diag,
    interp_nan,
)
from astropy.convolution import Gaussian2DKernel, convolve

from bernese.data import genomics


# Named tuples for genomic coordinates
Contig = collections.namedtuple("Contig", ["genome", "chr", "start", "end"])
ModelSeq = collections.namedtuple("ModelSeq", ["genome", "chr", "start", "end", "label"])


@dataclass
class HicTarget:
    """Hi-C target dataset configuration.

    Attributes:
        file: Path to .cool file
        clip: Optional clip value for targets
        index: Target index in multi-target datasets
    """

    file: str
    clip: Optional[float] = None
    index: int = 0


@dataclass
class HicStats:
    """Statistics for Hi-C dataset.

    Attributes:
        num_targets: Number of target tracks
        seq_length: Sequence length in bp
        seq_1hot: Whether sequences are one-hot encoded
        pool_width: Pooling width
        crop_bp: Crop from each end
        diagonal_offset: Diagonal offset
        target_length: Length of flattened target vector
        train_seqs: Number of training sequences
        valid_seqs: Number of validation sequences
        test_seqs: Number of test sequences
    """

    num_targets: int
    seq_length: int
    seq_1hot: bool = True
    pool_width: int = 128
    crop_bp: int = 0
    diagonal_offset: int = 2
    target_length: int = 0
    train_seqs: int = 0
    valid_seqs: int = 0
    test_seqs: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "num_targets": self.num_targets,
            "seq_length": self.seq_length,
            "seq_1hot": self.seq_1hot,
            "pool_width": self.pool_width,
            "crop_bp": self.crop_bp,
            "diagonal_offset": self.diagonal_offset,
            "target_length": self.target_length,
            "train_seqs": self.train_seqs,
            "valid_seqs": self.valid_seqs,
            "test_seqs": self.test_seqs,
        }


def load_genome(fasta_file: str) -> dict[str, list[tuple[int, int]]]:
    """Load genome from FASTA file.

    Args:
        fasta_file: Path to FASTA file

    Returns:
        Dictionary mapping chromosome to list of (start, end) segments
    """
    return genomics.load_chromosomes(fasta_file)


def contigs_to_list(
    chrom_contigs: dict[str, list[tuple[int, int]]],
    genome: str = "",
) -> list[Contig]:
    """Convert chromosome dict to list of Contig.

    Args:
        chrom_contigs: Dict mapping chromosome to list of (start, end)
        genome: Genome name

    Returns:
        List of Contig objects
    """
    contigs = []
    for chrom in chrom_contigs:
        for start, end in chrom_contigs[chrom]:
            contigs.append(Contig(genome, chrom, start, end))
    return contigs


def limit_contigs_to_bed(
    contigs: list[Contig],
    bed_file: str,
) -> list[Contig]:
    """Limit contigs to regions overlapping a BED file.

    Args:
        contigs: List of Contig objects
        bed_file: BED file with regions to keep

    Returns:
        Filtered list of Contig objects
    """
    import intervaltree

    # Build interval tree from BED
    bed_intervals = intervaltree.IntervalTree()
    with open(bed_file) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            bed_intervals.addi(start, end, chrom)

    # Filter contigs
    filtered = []
    for ctg in contigs:
        overlaps = bed_intervals.overlap(ctg.start, ctg.end)
        if overlaps:
            filtered.append(ctg)

    return filtered


def break_large_contigs(
    contigs: list[Contig],
    break_threshold: int,
    verbose: bool = False,
) -> list[Contig]:
    """Break large contigs into smaller pieces.

    Args:
        contigs: List of Contig objects
        break_threshold: Size threshold - contigs larger than this are broken
        verbose: Print progress

    Returns:
        List of broken contigs
    """
    return genomics.break_large_contigs(contigs, break_threshold, verbose)


def rejoin_broken_contigs(contigs: list[Contig]) -> list[Contig]:
    """Rejoin contigs that were previously broken.

    Args:
        contigs: List of Contig objects

    Returns:
        List of rejoined contigs
    """
    return genomics.rejoin_broken_contigs(contigs)


def divide_contigs_by_pct(
    contigs: list[Contig],
    test_pct: float,
    valid_pct: float,
) -> tuple[list[Contig], list[Contig], list[Contig]]:
    """Divide contigs by percentage into train/valid/test.

    Args:
        contigs: List of Contig objects
        test_pct: Test set percentage (0-1)
        valid_pct: Validation set percentage (0-1)

    Returns:
        Tuple of (train, valid, test) contig lists
    """
    # Shuffle contigs
    indices = np.arange(len(contigs))
    np.random.shuffle(indices)

    n = len(contigs)
    n_test = int(n * test_pct)
    n_valid = int(n * valid_pct)

    test_contigs = [contigs[i] for i in indices[:n_test]]
    valid_contigs = [contigs[i] for i in indices[n_test : n_test + n_valid]]
    train_contigs = [contigs[i] for i in indices[n_test + n_valid :]]

    return train_contigs, valid_contigs, test_contigs


def divide_contigs_by_chr(
    contigs: list[Contig],
    test_chrs: list[str],
    valid_chrs: list[str],
) -> tuple[list[Contig], list[Contig], list[Contig]]:
    """Divide contigs by chromosome into train/valid/test.

    Args:
        contigs: List of Contig objects
        test_chrs: List of chromosome names for test set
        valid_chrs: List of chromosome names for validation set

    Returns:
        Tuple of (train, valid, test) contig lists
    """
    test_set = set(test_chrs)
    valid_set = set(valid_chrs)

    train_contigs = []
    valid_contigs = []
    test_contigs = []

    for ctg in contigs:
        if ctg.chr in test_set:
            test_contigs.append(ctg)
        elif ctg.chr in valid_set:
            valid_contigs.append(ctg)
        else:
            train_contigs.append(ctg)

    return train_contigs, valid_contigs, test_contigs


def divide_contigs_by_folds(
    contigs: list[Contig],
    num_folds: int,
) -> list[list[Contig]]:
    """Divide contigs into k folds for cross-validation.

    Args:
        contigs: List of Contig objects
        num_folds: Number of folds

    Returns:
        List of fold contig lists
    """
    # Shuffle indices
    indices = np.arange(len(contigs))
    np.random.shuffle(indices)

    # Divide into folds
    fold_size = len(contigs) // num_folds
    folds = []
    for i in range(num_folds):
        start = i * fold_size
        if i == num_folds - 1:
            # Last fold gets remainder
            end = len(contigs)
        else:
            end = (i + 1) * fold_size
        fold_indices = indices[start:end]
        folds.append([contigs[j] for j in fold_indices])

    return folds


def create_model_sequences(
    contigs: list[Contig],
    seq_length: int,
    stride: int,
    snap: int = 1,
    label: Optional[str] = None,
) -> list[ModelSeq]:
    """Create model-length sequences from contigs.

    Args:
        contigs: List of Contig objects
        seq_length: Length of sequences
        stride: Stride between sequences
        snap: Snap start positions to multiples of this
        label: Optional label for sequences

    Returns:
        List of ModelSeq objects
    """
    return genomics.contig_sequences(contigs, seq_length, stride, snap, label)


def annotate_mappability(
    seqs: list[ModelSeq],
    umap_bed: str,
    seq_length: int,
    pool_width: int,
) -> np.ndarray:
    """Annotate sequences with mappability from BED file.

    Args:
        seqs: List of ModelSeq objects
        umap_bed: BED file with unmappable regions
        seq_length: Sequence length
        pool_width: Pooling width

    Returns:
        Array of shape (num_seqs, seq_length // pool_width) with mappability
    """
    if shutil.which("bedtools") is None:
        raise RuntimeError("bedtools is required for mappability annotation")

    # Get pool length
    pool_length = seq_length // pool_width

    # Initialize unmappable array
    seq_unmap = np.zeros((len(seqs), pool_length), dtype=np.float32)

    # Create BED for sequences
    seq_bed_tmp = "/tmp/seqs_tmp.bed"
    umap_bed_tmp = "/tmp/umap_tmp.bed"

    # Write sequences to temp BED
    with open(seq_bed_tmp, "w") as f:
        for i, seq in enumerate(seqs):
            print(f"{seq.chr}\t{seq.start}\t{seq.end}\t{i}", file=f)

    # Create link to umap file (bedtools requires file existence)
    if not os.path.isabs(umap_bed):
        umap_bed = os.path.abspath(umap_bed)
    shutil.copy(umap_bed, umap_bed_tmp)

    # Run bedtools coverage
    cov_cmd = f"bedtools coverage -a {seq_bed_tmp} -b {umap_bed_tmp}"
    cov_result = subprocess.run(
        cov_cmd,
        shell=True,
        capture_output=True,
        text=True,
    )

    if cov_result.returncode != 0:
        print(f"Warning: bedtools coverage failed: {cov_result.stderr}", file=sys.stderr)

    # Parse coverage output
    for line in cov_result.stdout.strip().split("\n"):
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        seq_idx = int(parts[3])
        # Coverage is in columns 4+ (one per BED region)
        if len(parts) > 4:
            # Sum coverage across all umap regions
            cov_vals = [float(x) for x in parts[4:]]
            seq_unmap[seq_idx, :] = sum(cov_vals)

    # Cleanup
    os.remove(seq_bed_tmp)
    os.remove(umap_bed_tmp)

    return seq_unmap


def filter_mappability(
    seqs: list[ModelSeq],
    seq_unmap: np.ndarray,
    umap_threshold: float,
) -> tuple[list[ModelSeq], np.ndarray]:
    """Filter sequences by mappability.

    Args:
        seqs: List of ModelSeq objects
        seq_unmap: Mappability array from annotate_mappability
        umap_threshold: Maximum allowed unmappable fraction

    Returns:
        Tuple of (filtered_seqs, filtered_unmap)
    """
    # Compute mean unmappable fraction per sequence
    map_mask = seq_unmap.mean(axis=1) < umap_threshold
    filtered_seqs = [seqs[i] for i in range(len(seqs)) if map_mask[i]]
    filtered_unmap = seq_unmap[map_mask, :]

    return filtered_seqs, filtered_unmap


def write_sequences_bed(
    bed_file: str,
    seqs: list[ModelSeq],
    include_labels: bool = False,
) -> None:
    """Write sequences to BED file.

    Args:
        bed_file: Output BED file path
        seqs: List of ModelSeq objects
        include_labels: Include label in fourth column
    """
    genomics.write_sequences_bed(bed_file, seqs, include_labels)


def read_sequences_bed(bed_file: str) -> list[ModelSeq]:
    """Read sequences from BED file.

    Args:
        bed_file: Input BED file path

    Returns:
        List of ModelSeq objects
    """
    return genomics.read_sequences_bed(bed_file)


def read_cool_targets(
    cool_file: str,
    seqs_bed_file: str,
    output_h5: str,
    crop_bp: int = 0,
    pool_width: int = 128,
    diagonal_offset: int = 2,
    kernel_stddev: int = 0,
    blacklist_bed: Optional[str] = None,
    as_obsexp: bool = False,
    global_obsexp: bool = False,
    no_log: bool = False,
    clip: Optional[float] = None,
) -> None:
    """Extract Hi-C matrices from .cool file and save to HDF5.

    Args:
        cool_file: Input .cool file
        seqs_bed_file: BED file with sequence coordinates
        output_h5: Output HDF5 file
        crop_bp: Crop bp from each end
        pool_width: Pooling width
        diagonal_offset: Diagonal positions to ignore
        kernel_stddev: Gaussian kernel stddev for smoothing
        blacklist_bed: BED file with blacklist regions
        as_obsexp: Save as observed/expected
        global_obsexp: Use pre-computed per-chromosome obs/exp
        no_log: Don't take log for obs/exp
        clip: Clip values to this maximum
    """
    # Read sequences
    seqs = []
    with open(seqs_bed_file) as f:
        for line in f:
            parts = line.split()
            seqs.append(ModelSeq("", parts[0], int(parts[1]), int(parts[2]), None))

    # Read blacklist
    black_chr_trees = {}
    if blacklist_bed:
        black_chr_trees = read_blacklist(blacklist_bed)

    # Compute dimensions
    num_seqs = len(seqs)
    seq_len_nt = seqs[0].end - seqs[0].start
    seq_len_pool = seq_len_nt // pool_width

    if crop_bp == 0:
        seq_len_crop = seq_len_pool
    else:
        crop_start = crop_bp // pool_width
        crop_end = seq_len_pool - crop_start
        seq_len_crop = seq_len_pool - 2 * crop_start

    # Compute upper triangular indices
    triu_tup = np.triu_indices(seq_len_crop, diagonal_offset)
    seq_len_nodiag = seq_len_crop - diagonal_offset
    seq_len_hic = seq_len_nodiag * (seq_len_nodiag + 1) // 2

    # Initialize HDF5
    with h5py.File(output_h5, "w") as h5f:
        h5f.create_dataset("targets", shape=(num_seqs, seq_len_hic), dtype=np.float32)

    # Initialize kernel
    if kernel_stddev > 0:
        kernel = Gaussian2DKernel(x_stddev=kernel_stddev)
    else:
        kernel = None

    # Open cooler
    cool = cooler.Cooler(cool_file)

    # Check for chr prefix
    has_chr_prefix = "chr1" in cool.chromnames

    # Load global expected if needed
    if global_obsexp:
        expected_file = cool_file.replace(".cool", ".expected")
        try:
            genome_expected = pd.read_csv(expected_file, sep="\t")
        except FileNotFoundError:
            raise ValueError(f"Expected file not found: {expected_file}")

    # Verify resolution
    if pool_width != cool.info["bin-size"]:
        raise ValueError(
            f"Pool width {pool_width} doesn't match cooler resolution {cool.info['bin-size']}"
        )

    # Process each sequence
    for si, mseq in enumerate(seqs):
        try:
            # Format chromosome string
            if has_chr_prefix:
                mseq_str = f"{mseq.chr}:{mseq.start}-{mseq.end}"
            else:
                mseq_str = f"{mseq.chr[3:] if mseq.chr.startswith('chr') else mseq.chr}:{mseq.start}-{mseq.end}"

            # Fetch raw Hi-C matrix
            seq_hic_raw = cool.matrix(balance=True).fetch(mseq_str)
            seq_hic_nan = np.isnan(seq_hic_raw)

            # Check filtering
            num_filtered_bins = np.sum(np.sum(seq_hic_nan, axis=0) == len(seq_hic_nan))
            if num_filtered_bins > 0.5 * len(seq_hic_nan):
                print(
                    f"Warning: {cool_file} {mseq_str} has >50% filtered bins",
                    file=sys.stderr,
                )

            # Set blacklist to NaN
            if mseq.chr in black_chr_trees:
                for interval in black_chr_trees[mseq.chr][mseq.start : mseq.end]:
                    black_start = (interval.begin - mseq.start) // pool_width
                    black_end = int(np.ceil((interval.end - mseq.start) / pool_width))
                    seq_hic_raw[:, black_start:black_end] = np.nan
                    seq_hic_raw[black_start:black_end, :] = np.nan
                seq_hic_nan = np.isnan(seq_hic_raw)

            # Clip diagonals and high values
            clipval = np.nanmedian(np.diag(seq_hic_raw, diagonal_offset))
            for i in range(-diagonal_offset + 1, diagonal_offset):
                set_diag(seq_hic_raw, clipval, i)
            seq_hic_raw = np.clip(seq_hic_raw, 0, clipval)
            seq_hic_raw[seq_hic_nan] = np.nan

            # Adaptively coarsegrain
            seq_hic_smoothed = adaptive_coarsegrain(
                seq_hic_raw,
                cool.matrix(balance=False).fetch(mseq_str),
                cutoff=2,
                max_levels=8,
            )
            seq_hic_nan = np.isnan(seq_hic_smoothed)

            if as_obsexp:
                # Compute observed/expected
                if global_obsexp:
                    exp_chr = genome_expected.iloc[genome_expected["chrom"].values == mseq.chr][
                        :seq_len_pool
                    ]
                    if len(exp_chr) == 0:
                        raise ValueError(f"No expected values for {mseq.chr}")

                    exp_map = np.zeros((seq_len_pool, seq_len_pool))
                    for i in range(seq_len_pool):
                        set_diag(exp_map, exp_chr["balanced.avg"].values[i], i)
                        set_diag(exp_map, exp_chr["balanced.avg"].values[i], -i)

                    seq_hic_obsexp = seq_hic_smoothed / exp_map
                    for i in range(-diagonal_offset + 1, diagonal_offset):
                        set_diag(seq_hic_obsexp, 1.0, i)
                    seq_hic_obsexp[seq_hic_nan] = np.nan
                else:
                    seq_hic_obsexp = observed_over_expected(seq_hic_smoothed, ~seq_hic_nan)[0]

                # Apply log transform
                if not no_log:
                    seq_hic_obsexp = np.log(seq_hic_obsexp)
                    if clip is not None:
                        seq_hic_obsexp = np.clip(seq_hic_obsexp, -clip, clip)
                    seq_hic_obsexp = interp_nan(seq_hic_obsexp)
                    for i in range(-diagonal_offset + 1, diagonal_offset):
                        set_diag(seq_hic_obsexp, 0, i)
                else:
                    if clip is not None:
                        seq_hic_obsexp = np.clip(seq_hic_obsexp, 0, clip)
                    seq_hic_obsexp = interp_nan(seq_hic_obsexp)
                    for i in range(-diagonal_offset + 1, diagonal_offset):
                        set_diag(seq_hic_obsexp, 1, i)

                # Apply kernel smoothing
                if kernel is not None:
                    seq_hic = convolve(seq_hic_obsexp, kernel)
                else:
                    seq_hic = seq_hic_obsexp

            else:
                # Interpolate missing bins
                seq_hic_interpolated = interp_nan(seq_hic_smoothed)

                # Rescale and reclip
                seq_hic = 100000 * seq_hic_interpolated
                clipval = np.nanmedian(np.diag(seq_hic, diagonal_offset))
                for i in range(-diagonal_offset + 1, diagonal_offset):
                    set_diag(seq_hic, clipval, i)
                seq_hic = np.clip(seq_hic, 0, clipval)

                if kernel is not None:
                    seq_hic = convolve(seq_hic, kernel)

        except ValueError as e:
            print(f"Warning: {cool_file} doesn't see {mseq_str}. Setting to zeros. ({e})")
            seq_hic = np.zeros((seq_len_pool, seq_len_pool), dtype=np.float32)

        # Crop
        if crop_bp > 0:
            crop_start = crop_bp // pool_width
            crop_end = seq_len_pool - crop_start
            seq_hic = seq_hic[crop_start:crop_end, crop_start:crop_end]

        # Unroll upper triangular
        seq_hic = seq_hic[triu_tup]

        # Write to HDF5
        with h5py.File(output_h5, "a") as h5f:
            h5f["targets"][si, :] = seq_hic.astype(np.float32)


def read_blacklist(
    blacklist_bed: str,
) -> dict[str, intervaltree.IntervalTree]:
    """Read blacklist BED file.

    Args:
        blacklist_bed: BED file with blacklist regions

    Returns:
        Dictionary mapping chromosome to interval tree
    """

    black_chr_trees = {}
    with open(blacklist_bed) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])

            if chrom not in black_chr_trees:
                black_chr_trees[chrom] = intervaltree.IntervalTree()
            black_chr_trees[chrom].addi(start, end)

    return black_chr_trees


def compute_hic_statistics(
    num_targets: int,
    seq_length: int,
    pool_width: int,
    crop_bp: int,
    diagonal_offset: int,
    fold_seqs: dict[str, int],
) -> HicStats:
    """Compute statistics for Hi-C dataset.

    Args:
        num_targets: Number of target tracks
        seq_length: Sequence length in bp
        pool_width: Pooling width
        crop_bp: Crop from each end
        diagonal_offset: Diagonal offset
        fold_seqs: Dictionary mapping fold name to sequence count

    Returns:
        HicStats object
    """
    # Compute target length
    target_len = seq_length - 2 * crop_bp
    target_len = target_len // pool_width
    target_len = target_len - diagonal_offset
    target_length = target_len * (target_len + 1) // 2

    # Get sequence counts
    train_seqs = fold_seqs.get("train", 0)
    valid_seqs = fold_seqs.get("valid", 0)
    test_seqs = fold_seqs.get("test", 0)

    # Handle folds
    for key, val in fold_seqs.items():
        if key.startswith("fold"):
            if "train" not in fold_seqs:
                train_seqs = val
            elif "valid" not in fold_seqs and "valid" in key:
                valid_seqs = val
            elif "test" not in fold_seqs and "test" in key:
                test_seqs = val

    return HicStats(
        num_targets=num_targets,
        seq_length=seq_length,
        seq_1hot=True,
        pool_width=pool_width,
        crop_bp=crop_bp,
        diagonal_offset=diagonal_offset,
        target_length=target_length,
        train_seqs=train_seqs,
        valid_seqs=valid_seqs,
        test_seqs=test_seqs,
    )


def write_sequences_hdf5(
    fasta_file: str,
    seqs_bed_file: str,
    seqs_cov_dir: str,
    output_h5: str,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
) -> None:
    """Write sequences and targets to HDF5.

    Args:
        fasta_file: FASTA file with genome sequence
        seqs_bed_file: BED file with sequence coordinates
        seqs_cov_dir: Directory with target HDF5 files
        output_h5: Output HDF5 file
        start_idx: Start sequence index
        end_idx: End sequence index (None = all)
    """
    # Read sequences
    seqs = read_sequences_bed(seqs_bed_file)

    if end_idx is None:
        end_idx = len(seqs)

    seqs = seqs[start_idx:end_idx]
    num_seqs = len(seqs)

    # Find target files
    seqs_cov_files = []
    ti = 0
    cov_file = os.path.join(seqs_cov_dir, f"{ti}.h5")
    while os.path.isfile(cov_file):
        seqs_cov_files.append(cov_file)
        ti += 1
        cov_file = os.path.join(seqs_cov_dir, f"{ti}.h5")

    if len(seqs_cov_files) == 0:
        raise FileNotFoundError(f"No target files found in {seqs_cov_dir}")

    # Get dimensions from first target file
    with h5py.File(seqs_cov_files[0], "r") as f:
        seq_len_hic = f["targets"].shape[1]
        num_targets = len(seqs_cov_files)

    # Initialize arrays
    seqs_1hot = np.zeros((num_seqs, seq_len_hic, 4), dtype=np.float32)
    targets = np.zeros((num_seqs, seq_len_hic, num_targets), dtype=np.float32)

    # Open FASTA
    fasta = pysam.Fastafile(fasta_file)

    # Read sequences and targets
    for si, seq in enumerate(seqs):
        # Read from FASTA
        seq_dna = fasta.fetch(seq.chr, seq.start, seq.end)

        # One-hot encode
        seq_1hot[si] = dna_1hot(seq_dna)

        # Read targets
        for ti, cov_file in enumerate(seqs_cov_files):
            with h5py.File(cov_file, "r") as f:
                targets[si, :, ti] = f["targets"][start_idx + si, :]

    fasta.close()

    # Write to HDF5
    with h5py.File(output_h5, "w") as f:
        f.create_dataset("seqs_1hot", data=seqs_1hot, compression="gzip")
        f.create_dataset("targets", data=targets, compression="gzip")


def dna_1hot(seq: str) -> np.ndarray:
    """One-hot encode DNA sequence.

    Args:
        seq: DNA sequence string

    Returns:
        Array of shape (length, 4) with one-hot encoding
    """
    seq = seq.upper().replace("A", "0").replace("C", "1")
    seq = seq.replace("G", "2").replace("T", "3").replace("N", "0")

    seq_1hot = np.zeros((len(seq), 4), dtype=np.float32)
    for i, c in enumerate(seq):
        ci = ord(c) - ord("0")
        if 0 <= ci <= 3:
            seq_1hot[i, ci] = 1.0

    return seq_1hot


def prepare_contacts_data(
    fasta_file: str,
    targets_file: str,
    out_dir: str,
    seq_length: int = 131072,
    crop_bp: int = 0,
    pool_width: int = 128,
    diagonal_offset: int = 2,
    snap: int = 1,
    stride_train: int = 131072,
    stride_test: int = 131072,
    test_pct: float = 0.05,
    valid_pct: float = 0.05,
    test_chrs: Optional[list[str]] = None,
    valid_chrs: Optional[list[str]] = None,
    sample_pct: float = 1.0,
    seed: int = 44,
    break_threshold: int = 8388608,
    gaps_file: Optional[str] = None,
    limit_bed: Optional[str] = None,
    umap_bed: Optional[str] = None,
    umap_threshold: float = 0.3,
    blacklist_bed: Optional[str] = None,
    as_obsexp: bool = False,
    global_obsexp: bool = False,
    no_log: bool = False,
    restart: bool = False,
    split_test: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Prepare Hi-C contact data for training.

    This is the main entry point for data preparation.

    Args:
        fasta_file: Genome FASTA file
        targets_file: TSV file with target information
        out_dir: Output directory
        seq_length: Sequence length in bp
        crop_bp: Crop from each end
        pool_width: Pooling width
        diagonal_offset: Diagonal offset
        snap: Snap to multiple
        stride_train: Training stride
        stride_test: Test/valid stride
        test_pct: Test percentage (if not using test_chrs)
        valid_pct: Validation percentage (if not using valid_chrs)
        test_chrs: Test chromosomes (overrides test_pct)
        valid_chrs: Validation chromosomes (overrides valid_pct)
        sample_pct: Sample proportion
        seed: Random seed
        break_threshold: Break contigs larger than this
        gaps_file: Gaps BED file
        limit_bed: Limit to BED regions
        umap_bed: Mappability BED file
        umap_threshold: Maximum unmappable fraction
        blacklist_bed: Blacklist BED file
        as_obsexp: Save as observed/expected
        global_obsexp: Use global expected
        no_log: Don't log transform obs/exp
        restart: Continue from checkpoint
        split_test: Exit after splitting
        verbose: Print progress

    Returns:
        Statistics dictionary
    """
    np.random.seed(seed)

    # Calculate effective sequence length after cropping
    seq_tlength = seq_length - 2 * crop_bp

    # Create output directory
    if os.path.isdir(out_dir) and not restart:
        raise FileExistsError(f"Remove {out_dir} or use --restart")
    elif not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Save options
    options = {
        "fasta_file": fasta_file,
        "targets_file": targets_file,
        "seq_length": seq_length,
        "crop_bp": crop_bp,
        "pool_width": pool_width,
        "diagonal_offset": diagonal_offset,
        "snap": snap,
        "stride_train": stride_train,
        "stride_test": stride_test,
        "test_pct": test_pct,
        "valid_pct": valid_pct,
        "test_chrs": test_chrs,
        "valid_chrs": valid_chrs,
        "sample_pct": sample_pct,
        "seed": seed,
        "break_threshold": break_threshold,
        "gaps_file": gaps_file,
        "limit_bed": limit_bed,
        "umap_bed": umap_bed,
        "umap_threshold": umap_threshold,
        "blacklist_bed": blacklist_bed,
        "as_obsexp": as_obsexp,
        "global_obsexp": global_obsexp,
        "no_log": no_log,
    }

    with open(os.path.join(out_dir, "options.json"), "w") as f:
        json.dump(options, f, indent=4)

    # Phase 1: Define genomic contigs
    if not restart:
        if verbose:
            print("Loading genome...")

        chrom_contigs = load_genome(fasta_file)

        # Remove gaps
        if gaps_file:
            if verbose:
                print("Splitting at gaps...")
            chrom_contigs = genomics.split_contigs_by_gaps(chrom_contigs, gaps_file)

        # Convert to list
        contigs = contigs_to_list(chrom_contigs)
        if verbose:
            print(f"Loaded {len(contigs)} contigs")

        # Limit to BED
        if limit_bed:
            if verbose:
                print("Limiting to BED regions...")
            contigs = limit_contigs_to_bed(contigs, limit_bed)

        # Filter by size
        contigs = [ctg for ctg in contigs if ctg.end - ctg.start >= seq_tlength]
        if verbose:
            print(f"Filtered to {len(contigs)} contigs >= {seq_tlength} bp")

        # Break large contigs
        if break_threshold:
            if verbose:
                print(f"Breaking contigs > {break_threshold} bp...")
            contigs = break_large_contigs(contigs, break_threshold, verbose)
            if verbose:
                print(f"Broken into {len(contigs)} contigs")

    # Phase 2: Split into train/valid/test
    fold_labels = ["train", "valid", "test"]

    if not restart:
        if test_chrs is not None and valid_chrs is not None:
            fold_contigs = divide_contigs_by_chr(contigs, test_chrs, valid_chrs)
        else:
            fold_contigs = divide_contigs_by_pct(contigs, test_pct, valid_pct)

        # Rejoin broken contigs within each fold
        fold_contigs = [rejoin_broken_contigs(fc) for fc in fold_contigs]

        # Write contigs to BED
        ctg_bed = os.path.join(out_dir, "contigs.bed")
        with open(ctg_bed, "w") as f:
            for fi, fc in enumerate(fold_contigs):
                for ctg in fc:
                    print(f"{ctg.chr}\t{ctg.start}\t{ctg.end}\t{fold_labels[fi]}", file=f)

        if verbose:
            print("Contigs written to contigs.bed")

    if split_test:
        if verbose:
            print("Exiting after split (--split_test)")
        return {}

    # Phase 3: Create model sequences
    if not restart:
        fold_mseqs = []

        for fi, label in enumerate(fold_labels):
            if label in ["valid", "test"]:
                stride_fold = stride_test
            else:
                stride_fold = stride_train

            mseqs = create_model_sequences(fold_contigs[fi], seq_tlength, stride_fold, snap, label)

            # Shuffle
            np.random.shuffle(mseqs)

            # Down-sample
            if sample_pct < 1.0:
                n_sample = int(sample_pct * len(mseqs))
                mseqs = list(np.random.choice(mseqs, n_sample, replace=False))

            fold_mseqs.append(mseqs)
            if verbose:
                print(f"{label}: {len(mseqs)} sequences")

        # Merge
        all_mseqs = [ms for fm in fold_mseqs for ms in fm]

    # Phase 4: Mappability filtering
    if not restart:
        if umap_bed:
            if verbose:
                print("Annotating mappability...")
            seq_unmap = annotate_mappability(all_mseqs, umap_bed, seq_length, pool_width)

            # Filter
            all_mseqs, seq_unmap = filter_mappability(all_mseqs, seq_unmap, umap_threshold)

            # Save
            np.save(os.path.join(out_dir, "mseqs_unmap.npy"), seq_unmap)
            if verbose:
                print(f"Filtered to {len(all_mseqs)} sequences with <{umap_threshold} unmappable")

        # Write sequences to BED
        seqs_bed = os.path.join(out_dir, "sequences.bed")
        write_sequences_bed(seqs_bed, all_mseqs, include_labels=True)

    else:
        # Load from existing
        seqs_bed = os.path.join(out_dir, "sequences.bed")
        all_mseqs = read_sequences_bed(seqs_bed)

        # Rebuild fold lists
        fold_mseqs = [[], [], []]
        for ms in all_mseqs:
            if ms.label == "train":
                fold_mseqs[0].append(ms)
            elif ms.label == "valid":
                fold_mseqs[1].append(ms)
            elif ms.label == "test":
                fold_mseqs[2].append(ms)

    # Copy targets file
    shutil.copy(targets_file, os.path.join(out_dir, "targets.txt"))

    # Phase 5: Read targets
    targets_df = pd.read_csv(targets_file, index_col=0, sep="\t")
    seqs_cov_dir = os.path.join(out_dir, "seqs_cov")
    os.makedirs(seqs_cov_dir, exist_ok=True)

    if verbose:
        print("Reading Hi-C targets...")

    for ti in range(targets_df.shape[0]):
        genome_hic_file = targets_df["file"].iloc[ti]
        seqs_cov_file = os.path.join(seqs_cov_dir, f"{ti}.h5")

        clip_ti = targets_df["clip"].iloc[ti] if "clip" in targets_df.columns else None

        if verbose:
            print(f"  Target {ti}: {genome_hic_file}")

        read_cool_targets(
            genome_hic_file,
            seqs_bed,
            seqs_cov_file,
            crop_bp=crop_bp,
            pool_width=pool_width,
            diagonal_offset=diagonal_offset,
            blacklist_bed=blacklist_bed,
            as_obsexp=as_obsexp,
            global_obsexp=global_obsexp,
            no_log=no_log,
            clip=clip_ti,
        )

    # Phase 6: Compute statistics
    fold_seqs = {fold_labels[fi]: len(fold_mseqs[fi]) for fi in range(3)}
    stats = compute_hic_statistics(
        num_targets=targets_df.shape[0],
        seq_length=seq_length,
        pool_width=pool_width,
        crop_bp=crop_bp,
        diagonal_offset=diagonal_offset,
        fold_seqs=fold_seqs,
    )

    with open(os.path.join(out_dir, "statistics.json"), "w") as f:
        json.dump(stats.to_dict(), f, indent=4)

    if verbose:
        print(f"\nData preparation complete!")
        print(f"Output directory: {out_dir}")
        print(f"Statistics:")
        for key, value in stats.to_dict().items():
            print(f"  {key}: {value}")

    return stats.to_dict()


__all__ = [
    # Classes
    "HicTarget",
    "HicStats",
    # Contig handling
    "load_genome",
    "contigs_to_list",
    "limit_contigs_to_bed",
    "break_large_contigs",
    "rejoin_broken_contigs",
    # Splitting
    "divide_contigs_by_pct",
    "divide_contigs_by_chr",
    "divide_contigs_by_folds",
    # Sequences
    "create_model_sequences",
    "annotate_mappability",
    "filter_mappability",
    "write_sequences_bed",
    "read_sequences_bed",
    # Hi-C
    "read_cool_targets",
    "read_blacklist",
    # Statistics
    "compute_hic_statistics",
    # HDF5
    "write_sequences_hdf5",
    "dna_1hot",
    # Main
    "prepare_contacts_data",
]
