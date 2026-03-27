# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""BigWig target processor implementation.

This module provides a target processor for BigWig files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pysam

import numpy as np

from bernese.data.targets.pool import PoolStat
from bernese.data.targets.registry import TargetProcessor, TargetProcessorRegistry


@TargetProcessorRegistry.register("bigwig")
class BigWigTargetProcessor(TargetProcessor):
    """Processor for BigWig files.

    Extracts and processes signal tracks from BigWig files.
    Supports mean, sum, and other aggregation methods.
    """

    def __init__(
        self,
        pool_width: int = 128,
        pool_stat: PoolStat = PoolStat.MEAN,
        crop_bp: int = 0,
    ):
        self.pool_width = pool_width
        self.pool_stat = pool_stat
        self.crop_bp = crop_bp

    @property
    def target_type(self) -> str:
        return "bigwig"

    def compute_target_length(self, seq_length: int) -> int:
        """Compute target length after pooling.

        Args:
            seq_length: Sequence length in bp

        Returns:
            Target length
        """
        # Apply crop
        seq_length = seq_length - 2 * self.crop_bp

        # Pool
        return seq_length // self.pool_width

    def _aggregate(self, values: np.ndarray, stat: PoolStat | None = None) -> float:
        """Apply pooling aggregation.

        Args:
            values: Array of values to aggregate
            stat: PoolStat to use (defaults to self.pool_stat)

        Returns:
            Aggregated value
        """
        if stat is None:
            stat = self.pool_stat

        match stat:
            case PoolStat.SUM:
                return np.sum(values) if len(values) > 0 else 0.0
            case PoolStat.SUM_SQRT:
                return -1 + np.sqrt(1 + np.sum(values)) if len(values) > 0 else 0.0
            case PoolStat.MEAN:
                return np.mean(values) if len(values) > 0 else 0.0
            case PoolStat.MEAN_SQRT:
                return -1 + np.sqrt(1 + np.mean(values)) if len(values) > 0 else 0.0
            case PoolStat.MEDIAN:
                return np.median(values) if len(values) > 0 else 0.0
            case PoolStat.MAX:
                return np.max(values) if len(values) > 0 else 0.0
            case PoolStat.MIN:
                return np.min(values) if len(values) > 0 else 0.0
            case PoolStat.PEAK:
                mean_val = np.mean(values) if len(values) > 0 else 0.0
                return np.clip(np.sqrt(mean_val * 4), 0, 1)

    def process(
        self,
        input_file: str,
        regions: list[tuple[str, int, int]],
        **kwargs,
    ) -> np.ndarray:
        """Process BigWig data for genomic regions.

        Args:
            input_file: Path to BigWig file
            regions: List of (chrom, start, end) tuples
            **kwargs: Additional options

        Returns:
            Array of shape (num_regions, target_length)
        """
        # Open BigWig
        bw = pysam.TabixFile(input_file)

        # Compute target length
        seq_len = regions[0][2] - regions[0][1]
        target_length = self.compute_target_length(seq_len)

        # Initialize output
        num_regions = len(regions)
        targets = np.zeros((num_regions, target_length), dtype=np.float32)

        # Process each region
        for ri, (chrom, start, end) in enumerate(regions):
            try:
                # Fetch data
                if chrom.startswith("chr"):
                    chrom_query = chrom
                else:
                    chrom_query = f"chr{chrom}"

                # Get pileup
                try:
                    # Using pysam for BigWig access
                    values = []
                    for start_bin in range(start, end, self.pool_width):
                        end_bin = min(start_bin + self.pool_width, end)

                        # Get values in region
                        try:
                            fetched = bw.fetch(chrom_query, start_bin, end_bin)
                            vals = [float(x) for x in fetched if x != "."]
                        except:
                            vals = []

                        values.append(self._aggregate(vals))

                    # Pad if needed
                    while len(values) < target_length:
                        values.append(0.0)

                    targets[ri] = values[:target_length]

                except Exception as e:
                    print(f"Warning: Could not process {chrom}:{start}-{end}: {e}")
                    targets[ri] = 0

            except Exception as e:
                print(f"Warning: Error processing {chrom}:{start}-{end}: {e}")
                targets[ri] = 0

        bw.close()

        return targets

    def compute_statistics(self, data: np.ndarray) -> dict[str, Any]:
        """Compute statistics for BigWig data.

        Args:
            data: Target array

        Returns:
            Dictionary of statistics
        """
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)

        return {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
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
