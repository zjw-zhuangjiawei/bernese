# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Hi-C target processor implementation.

This module provides a target processor for Hi-C/cooler files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from bernese.data.targets.registry import TargetProcessor, TargetProcessorRegistry


@TargetProcessorRegistry.register("hic")
class HiCTargetProcessor(TargetProcessor):
    """Processor for Hi-C/cooler files.

    Extracts and processes Hi-C matrices from .cool files.
    Supports observed/expected transformation, normalization, and
    triangular matrix encoding.
    """

    def __init__(
        self,
        pool_width: int = 128,
        diagonal_offset: int = 2,
        as_obsexp: bool = False,
        global_obsexp: bool = False,
        no_log: bool = False,
        clip: float | None = None,
        crop_bp: int = 0,
    ):
        self.pool_width = pool_width
        self.diagonal_offset = diagonal_offset
        self.as_obsexp = as_obsexp
        self.global_obsexp = global_obsexp
        self.no_log = no_log
        self.clip = clip
        self.crop_bp = crop_bp

    @property
    def target_type(self) -> str:
        return "hic"

    def compute_target_length(self, seq_length: int) -> int:
        """Compute target length for Hi-C triangular matrix.

        Args:
            seq_length: Sequence length in bp

        Returns:
            Target length (flattened triangular matrix size)
        """
        # Account for cropping (applied to Hi-C matrix only)
        seq_len_after_crop = seq_length - 2 * self.crop_bp

        # Apply pooling
        seq_len_pool = seq_len_after_crop // self.pool_width

        # Subtract diagonal offset
        seq_len_nodiag = seq_len_pool - self.diagonal_offset

        # Compute triangular size
        return seq_len_nodiag * (seq_len_nodiag + 1) // 2

    def process(
        self,
        input_file: str,
        regions: list[tuple[str, int, int]],
        **kwargs,
    ) -> np.ndarray:
        """Process Hi-C data for genomic regions.

        Args:
            input_file: Path to .cool file
            regions: List of (chrom, start, end) tuples
            **kwargs: Additional options

        Returns:
            Array of shape (num_regions, target_length)
        """
        import cooler

        # Open cooler
        cool = cooler.Cooler(input_file)

        # Check for chr prefix
        has_chr_prefix = "chr1" in cool.chromnames

        # Compute target length (accounts for crop_bp)
        seq_len = regions[0][2] - regions[0][1]
        target_length = self.compute_target_length(seq_len)

        # Compute effective length after crop for Hi-C matrix
        seq_len_after_crop = seq_len - 2 * self.crop_bp

        # Compute upper triangular indices using cropped length
        seq_len_pool = seq_len_after_crop // self.pool_width

        triu_tup = np.triu_indices(seq_len_pool, self.diagonal_offset)

        # Initialize output
        num_targets = len(regions)
        targets = np.zeros((num_targets, target_length), dtype=np.float32)

        # Process each region
        for ri, (chrom, start, end) in enumerate(regions):
            try:
                # Format chromosome string
                if has_chr_prefix:
                    chrom_str = f"{chrom}:{start}-{end}"
                else:
                    chrom_str = f"{chrom[3:] if chrom.startswith('chr') else chrom}:{start}-{end}"

                # Fetch raw Hi-C matrix
                seq_hic = cool.matrix(balance=True).fetch(chrom_str)

                # Handle NaN - interpolate missing values
                from cooltools.lib.numutils import interp_nan, set_diag

                seq_hic_nan = np.isnan(seq_hic)

                # Interpolate NaN values before clipping (matching Basenji akita_data_read.py)
                seq_hic = interp_nan(seq_hic)

                # Clip diagonals
                clipval = np.nanmedian(np.diag(seq_hic, self.diagonal_offset))

                for i in range(-self.diagonal_offset + 1, self.diagonal_offset):
                    set_diag(seq_hic, clipval, i)

                seq_hic = np.clip(seq_hic, 0, clipval)

                if self.as_obsexp:
                    from cooltools.lib.numutils import observed_over_expected

                    # Compute observed/expected
                    seq_hic_obsexp = observed_over_expected(seq_hic, ~seq_hic_nan)[0]

                    # Apply log transform
                    if not self.no_log:
                        seq_hic_obsexp = np.log(seq_hic_obsexp)
                        if self.clip is not None:
                            seq_hic_obsexp = np.clip(seq_hic_obsexp, -self.clip, self.clip)
                    else:
                        if self.clip is not None:
                            seq_hic_obsexp = np.clip(seq_hic_obsexp, 0, self.clip)

                # Unroll upper triangular
                seq_hic = seq_hic[triu_tup]
                targets[ri] = seq_hic.astype(np.float32)

            except Exception as e:
                # Return zeros on error
                print(f"Warning: Could not process {chrom}:{start}-{end}: {e}")
                targets[ri] = 0

        return targets

    def compute_statistics(self, data: np.ndarray) -> dict[str, Any]:
        """Compute statistics for Hi-C data.

        Args:
            data: Target array

        Returns:
            Dictionary of statistics
        """
        # Compute per-target statistics
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        percentiles = np.nanpercentile(data, [1, 5, 25, 50, 75, 95, 99], axis=0)

        return {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "percentiles": percentiles.tolist(),
            "min": float(np.nanmin(data)),
            "max": float(np.nanmax(data)),
        }

    def save(
        self,
        output_path: str,
        targets: np.ndarray,
        mode: str = "w",
    ) -> None:
        """Save targets to HDF5.

        Args:
            output_path: Output HDF5 file path
            targets: Target array
            mode: File mode ('w' or 'a')
        """
        with h5py.File(output_path, mode) as f:
            f.create_dataset("data", data=targets, chunks=(1024, -1), compression="gzip")
