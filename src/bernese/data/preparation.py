# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Data preparation utilities for creating v2 datasets.

This module provides functions for preparing genomic data in the v2 format
with manifest.json and split-based organization.
"""

from __future__ import annotations

import json
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pysam
from pydantic import TypeAdapter

from bernese.data.backends import HDF5Writer
from bernese.data.config import TargetConfig
from bernese.data.targets import TargetProcessorRegistry
from bernese.data.targets.pool import PoolStat


# Basenji-style namedtuples
Contig = namedtuple("Contig", ["chr", "start", "end"])
GenomicRegion = namedtuple("GenomicRegion", ["chrom", "start", "end", "label"])


@dataclass
class PreparationConfig:
    """Configuration for data preparation."""

    # Sequence parameters
    seq_length: int = 131072
    crop_bp: int = 0
    pool_width: int = 128

    # Target parameters
    target_type: str = "hic"
    diagonal_offset: int = 2

    # Split parameters
    test_pct: float = 0.05
    valid_pct: float = 0.05
    folds: Optional[int] = None

    # Block-based region creation
    block_size: int = 1048576  # 1 Mb
    join_blocks: bool = True  # Join adjacent blocks after splitting

    # Stride parameters
    stride_train: float = 1.0
    stride_test: float = 1.0
    snap: int = 1

    # Filtering
    sample_pct: float = 1.0

    # Random
    seed: int = 44


class DataPreparator:
    """Prepares genomic data in v2 format.

    This class handles the complete data preparation pipeline:
    1. Pre-encode entire genome to 1hot (genome.h5)
    2. Create sequence coordinates and write indices
    3. Split into train/valid/test using basenji algorithm
    4. Extract targets using target processors
    5. Write manifest and HDF5 files using DatasetWriter
    """

    def __init__(
        self,
        output_dir: str | Path,
        fasta_file: str | Path,
        targets_file: str | Path,
        config: Optional[PreparationConfig] = None,
    ):
        self.output_dir = Path(output_dir)
        self.fasta_file = Path(fasta_file)
        self.targets_file = Path(targets_file)
        self.config = config or PreparationConfig()

        # Load targets info from JSON
        with open(targets_file) as f:
            self.targets_list: list[TargetConfig] = TypeAdapter(
                list[TargetConfig]
            ).validate_json(f.read())

        # Calculate target length
        self.target_length = self._calculate_target_length()

        # Create writer
        self.writer = HDF5Writer(
            output_dir=self.output_dir,
            seq_length=self.config.seq_length,
            seq_depth=4,
            target_length=self.target_length,
            num_targets=len(self.targets_list),
        )

        # Setup random
        np.random.seed(self.config.seed)

        # Track chromsome mapping (name -> index)
        self._chrom_to_idx: dict[str, int] = {}

    def _calculate_target_length(self) -> int:
        """Calculate target length based on target type."""
        seq_length = self.config.seq_length - 2 * self.config.crop_bp
        pool_length = seq_length // self.config.pool_width
        seq_len_nodiag = pool_length - self.config.diagonal_offset
        return seq_len_nodiag * (seq_len_nodiag + 1) // 2

    def prepare(self):
        """Run complete data preparation.

        Returns:
            DatasetMetadata for the prepared dataset
        """
        # Step 1: Pre-encode entire genome to 1hot (streaming)
        print("Pre-encoding genome...")
        self._preencode_genome()

        # Step 2: Create sequence coordinates and extract targets
        # This is now combined into a single atomic write per split
        print("Creating regions and extracting targets...")
        self._extract_targets_and_write()

        # Step 3: Finalize (creates manifest.json)
        print("Creating manifest...")
        target_info = []
        target_lengths: dict[int, int] = {}
        for i, target in enumerate(self.targets_list):
            target_info.append(
                {
                    "name": target.name,
                    "target_type": target.target_type,
                    "clip": target.clip,
                    "metadata": target.metadata,
                }
            )
            # Calculate target length for this target
            # Note: target_length depends on target-specific parameters
            # We store what was computed during extraction

        # Build target_lengths from writer's understanding
        # For now, use the same target_length for all (will be updated per-target)
        for i in range(len(self.targets_list)):
            target_lengths[i] = self.target_length

        metadata = self.writer.finalize(
            genome_name=self.fasta_file.stem,
            target_type=self.config.target_type,
            pool_width=self.config.pool_width,
            diagonal_offset=self.config.diagonal_offset,
            target_info=target_info,
            target_lengths=target_lengths,
        )

        print(f"Data preparation complete: {self.output_dir}")
        return metadata

    def _preencode_genome(self) -> None:
        """Stream one chromosome at a time to genome.h5.

        This method fetches each chromosome, encodes it internally in HDF5Writer,
        and immediately releases memory - avoiding OOM for large genomes.
        """
        # Open FASTA file
        fasta = pysam.FastaFile(str(self.fasta_file))

        try:
            # Stream each chromosome directly to HDF5
            for chrom, length in zip(fasta.references, fasta.lengths):
                print(f"  Encoding {chrom} ({length} bp)")

                # Fetch sequence (string)
                seq_dna = fasta.fetch(chrom)

                # Write directly to HDF5 (encoding happens inside writer)
                self.writer.write_chromosome(chrom, seq_dna)

                # Memory is released after each iteration

        finally:
            fasta.close()

    def _extract_targets_and_write(self) -> None:
        """Create sequence regions, extract targets, and write atomically.

        This combines region creation, target extraction, and writing into a single
        pass per split using write_split() for atomic operation.
        """
        # Step 1: Create sequence regions
        regions_by_split = self._create_sequence_regions()

        # Track target_lengths for manifest
        target_lengths: dict[int, int] = {}

        # Step 2: Process each split atomically
        for split, regions in regions_by_split.items():
            if len(regions) == 0:
                continue

            print(f"  Processing {split}: {len(regions)} regions")

            # Extract coordinates
            chrom_names = []
            starts = []
            ends = []

            for region in regions:
                chrom_names.append(region.chrom)
                starts.append(region.start)
                ends.append(region.end)

            # Convert regions to tuple format for target processor
            region_tuples = [(r.chrom, r.start, r.end) for r in regions]

            # Collect targets list (each target单独存储)
            targets_list_write: list[np.ndarray] = []

            # Process each target
            for i, target in enumerate(self.targets_list):
                # Build params from target-specific parameters
                params = dict(target.parameters)
                params["pool_width"] = self.config.pool_width
                params["crop_bp"] = self.config.crop_bp
                if target.target_type == "hic":
                    params["diagonal_offset"] = self.config.diagonal_offset

                # Extract pool_stat from parameters and convert to enum
                pool_stat_str = params.pop("pool_stat", "mean")
                try:
                    pool_stat = PoolStat(pool_stat_str)
                except ValueError:
                    pool_stat = PoolStat.MEAN  # Default to mean
                params["pool_stat"] = pool_stat

                # Create processor for this target type
                processor = TargetProcessorRegistry.create(
                    target.target_type, **params
                )

                # Process
                targets = processor.process(target.file, region_tuples)

                # Apply clip if specified
                if target.clip is not None:
                    targets = np.clip(targets, None, target.clip)

                # Store target (num_seqs, target_length_i)
                targets_list_write.append(targets)

                # Track target length for manifest
                target_lengths[i] = targets.shape[1]

            # Atomic write using write_split (indices + targets list in one file)
            self.writer.write_split(
                split, chrom_names, starts, ends, targets_list_write, self.targets_list
            )

    def _load_genome(self) -> dict[str, int]:
        """Load genome chromosome sizes."""
        fasta = pysam.FastaFile(str(self.fasta_file))
        chrom_sizes = dict(zip(fasta.references, fasta.lengths))
        fasta.close()
        return chrom_sizes

    def _create_sequence_regions(self) -> dict[str, list[GenomicRegion]]:
        """Create sequence regions for each split.

        Uses block-based approach:
        1. Divide genome into fixed-size blocks (block_size, default 1 Mb)
        2. Split blocks into train/valid/test
        3. Optionally join adjacent blocks within each split
        4. Generate sliding windows within each block
        """
        chrom_sizes = self._load_genome()

        # Use full sequence length for genomic regions
        # Crop will be applied only to targets (Hi-C matrix), not to sequences
        seq_length = self.config.seq_length

        # Create fixed-size blocks
        print(f"  Creating blocks of size {self.config.block_size}...")
        blocks = self._create_genomic_blocks(chrom_sizes)

        # Filter for minimum length
        blocks = [b for b in blocks if b.end - b.start >= seq_length]
        print(f"  After length filter: {len(blocks)} blocks")

        # Divide into train/valid/test
        print(f"  Dividing blocks: test={self.config.test_pct}, valid={self.config.valid_pct}")
        fold_blocks = self._divide_contigs_pct(
            blocks,
            self.config.test_pct,
            self.config.valid_pct,
        )

        # Optionally join adjacent blocks within each fold
        if self.config.join_blocks:
            print(f"  Joining adjacent blocks within each split...")
            for i in range(len(fold_blocks)):
                fold_blocks[i] = self._rejoin_large_contigs(fold_blocks[i])

        # Convert to GenomicRegion with labels
        train_regions = self._contig_sequences(fold_blocks[0], seq_length, self.config.stride_train)
        valid_regions = self._contig_sequences(fold_blocks[1], seq_length, self.config.stride_test)
        test_regions = self._contig_sequences(fold_blocks[2], seq_length, self.config.stride_test)

        # Shuffle each fold
        np.random.shuffle(train_regions)
        np.random.shuffle(valid_regions)
        np.random.shuffle(test_regions)

        # Apply sampling
        if self.config.sample_pct < 1.0 and len(train_regions) > 0:
            n_train = int(len(train_regions) * self.config.sample_pct)
            indices = np.random.choice(len(train_regions), n_train, replace=False)
            train_regions = [train_regions[i] for i in indices]

        # Add labels
        train_labeled = [GenomicRegion(r.chrom, r.start, r.end, "train") for r in train_regions]
        valid_labeled = [GenomicRegion(r.chrom, r.start, r.end, "valid") for r in valid_regions]
        test_labeled = [GenomicRegion(r.chrom, r.start, r.end, "test") for r in test_regions]

        return {
            "train": train_labeled,
            "valid": valid_labeled,
            "test": test_labeled,
        }

    def _create_genomic_blocks(self, chrom_sizes: dict[str, int]) -> list[Contig]:
        """Divide genome into fixed-size blocks.

        Creates non-overlapping blocks of block_size (default 1 Mb) from each chromosome.
        This ensures blocks are substantially larger than sequence length to preserve
        local genomic structure and prevent information leakage.

        Args:
            chrom_sizes: Dictionary mapping chromosome name to length

        Returns:
            List of Contig objects representing fixed-size blocks
        """
        blocks = []
        for chrom, length in chrom_sizes.items():
            pos = 0
            while pos + self.config.block_size <= length:
                blocks.append(Contig(chrom, pos, pos + self.config.block_size))
                pos += self.config.block_size
            # Keep remainder if >= seq_length
            remainder = length - pos
            if remainder >= self.config.seq_length - 2 * self.config.crop_bp:
                blocks.append(Contig(chrom, pos, length))

        print(f"  Created {len(blocks)} blocks of size {self.config.block_size}")
        return blocks

    def _divide_contigs_pct(
        self,
        contigs: list[Contig],
        test_pct: float,
        valid_pct: float,
        pct_abstain: float = 0.2,
    ) -> list[list[Contig]]:
        """Divide contigs into train/valid/test by nucleotide percentage."""
        # Sort contigs descending by length
        length_contigs = [(ctg.end - ctg.start, ctg) for ctg in contigs]
        length_contigs.sort(reverse=True)

        # Compute total nucleotides
        total_nt = sum(lc[0] for lc in length_contigs)

        # Compute aimed train/valid/test nucleotides
        test_nt_aim = test_pct * total_nt
        valid_nt_aim = valid_pct * total_nt
        train_nt_aim = total_nt - valid_nt_aim - test_nt_aim

        # Initialize current nucleotides
        train_nt = 0
        valid_nt = 0
        test_nt = 0

        # Initialize contig lists
        train_contigs = []
        valid_contigs = []
        test_contigs = []

        # Process contigs
        for ctg_len, ctg in length_contigs:
            # Compute gap between current and aim
            test_nt_gap = max(0, test_nt_aim - test_nt)
            valid_nt_gap = max(0, valid_nt_aim - valid_nt)
            train_nt_gap = max(1, train_nt_aim - train_nt)

            # Skip if too large
            if ctg_len > pct_abstain * test_nt_gap:
                test_nt_gap = 0
            if ctg_len > pct_abstain * valid_nt_gap:
                valid_nt_gap = 0

            # Compute remaining percentages
            gap_sum = train_nt_gap + valid_nt_gap + test_nt_gap
            if gap_sum == 0:
                # All targets reached, add to train
                train_contigs.append(ctg)
                train_nt += ctg_len
                continue

            test_pct_gap = test_nt_gap / gap_sum
            valid_pct_gap = valid_nt_gap / gap_sum
            train_pct_gap = train_nt_gap / gap_sum

            # Sample train/valid/test
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

        print(f"  Train: {len(train_contigs)} contigs, {train_nt} nt ({train_nt / total_nt:.4f})")
        print(f"  Valid: {len(valid_contigs)} contigs, {valid_nt} nt ({valid_nt / total_nt:.4f})")
        print(f"  Test:  {len(test_contigs)} contigs, {test_nt} nt ({test_nt / total_nt:.4f})")

        return [train_contigs, valid_contigs, test_contigs]

    def _rejoin_large_contigs(self, contigs: list[Contig]) -> list[Contig]:
        """Rejoin contigs that were broken up before the split."""
        if not contigs:
            return contigs

        # Group by chromosome
        chr_contigs = {}
        for ctg in contigs:
            chr_contigs.setdefault(ctg.chr, []).append(ctg)

        result = []
        for chrom in chr_contigs:
            # Sort by start position
            chr_contigs[chrom].sort(key=lambda x: x.start)

            ctg_ongoing = chr_contigs[chrom][0]
            for i in range(1, len(chr_contigs[chrom])):
                ctg_this = chr_contigs[chrom][i]
                if ctg_ongoing.end == ctg_this.start:
                    # Join
                    ctg_ongoing = Contig(ctg_ongoing.chr, ctg_ongoing.start, ctg_this.end)
                else:
                    # Conclude ongoing
                    result.append(ctg_ongoing)
                    ctg_ongoing = ctg_this

            # Conclude final
            result.append(ctg_ongoing)

        return result

    def _contig_sequences(
        self,
        contigs: list[Contig],
        seq_length: int,
        stride: float,
    ) -> list[GenomicRegion]:
        """Convert contigs to sequence regions."""
        regions = []
        for ctg in contigs:
            # Calculate start position (snapped)
            seq_start = int(np.ceil(ctg.start / self.config.snap) * self.config.snap)
            seq_end = seq_start + seq_length

            while seq_end <= ctg.end:
                regions.append(GenomicRegion(ctg.chr, seq_start, seq_end, None))
                seq_start += int(stride)
                seq_end += int(stride)

        return regions


def prepare_dataset(
    output_dir: str | Path,
    fasta_file: str | Path,
    targets_file: str | Path,
    config: Optional[PreparationConfig] = None,
    **kwargs,
):
    """Prepare a genomic dataset in v2 format.

    Args:
        output_dir: Output directory
        fasta_file: Genome FASTA file
        targets_file: Targets file (TSV)
        config: Preparation config
        **kwargs: Additional config options

    Returns:
        DatasetMetadata for the prepared dataset
    """
    # Merge config and kwargs
    if config is None:
        config = PreparationConfig()

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Create preparator
    preparator = DataPreparator(output_dir, fasta_file, targets_file, config)

    # Run preparation
    return preparator.prepare()
