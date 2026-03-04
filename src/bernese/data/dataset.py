# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Genomic dataset with backend and transform support.

This module provides a unified PyTorch Dataset interface that supports
multiple storage backends and composable transforms.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from bernese.data.backends import DataBackend, HDF5Backend, HDF5Writer
from bernese.data.transforms import TransformPipeline


class GenomicDataset(Dataset):
    """Unified genomic dataset with backend and transform support.

    This dataset supports multiple storage backends and composable transforms,
    providing a clean interface for loading genomic data for training.

    Args:
        backend: Data backend (HDF5Backend, etc.)
        split: Dataset split (train/valid/test)
        transform: Optional transform pipeline for sequences/targets
        target_transform: Optional transform pipeline for targets only
        shuffle: Whether to shuffle indices
        random_seed: Random seed for shuffling
    """

    def __init__(
        self,
        backend: DataBackend,
        split: str = "train",
        transform: Optional[TransformPipeline] = None,
        target_transform: Optional[TransformPipeline] = None,
        shuffle: bool = False,
        random_seed: int = 42,
    ):
        self.backend = backend
        self.split = split
        self.transform = transform or TransformPipeline()
        self.target_transform = target_transform or TransformPipeline()
        self.shuffle = shuffle

        # Get split metadata
        metadata = backend.metadata
        if split in metadata.splits:
            split_info = metadata.splits[split]
            self.num_seqs = split_info.num_seqs
        else:
            self.num_seqs = 0

        # Set up indexing
        self._indices = np.arange(self.num_seqs)
        if self.shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(self._indices)

        # Cache for data
        self._seq_cache: dict[int, torch.Tensor] = {}
        self._tgt_cache: dict[int, torch.Tensor] = {}
        self._cache_size = 1000  # Number of samples to cache

    def __len__(self) -> int:
        return self.num_seqs

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample.

        Args:
            idx: Sample index (in shuffled order if shuffle=True)

        Returns:
            Tuple of (sequence, targets) tensors
        """
        # Map to original index
        if self.shuffle:
            orig_idx = self._indices[idx]
        else:
            orig_idx = idx

        # Try cache first
        if orig_idx in self._seq_cache:
            seq = self._seq_cache[orig_idx]
        else:
            # Load from backend
            seq = self.backend.get_sequences(self.split, [orig_idx])[0]

            # Add to cache
            if len(self._seq_cache) < self._cache_size:
                self._seq_cache[orig_idx] = seq

        if orig_idx in self._tgt_cache:
            targets = self._tgt_cache[orig_idx]
        else:
            targets = self.backend.get_targets(self.split, [orig_idx])[0]

            if len(self._tgt_cache) < self._cache_size:
                self._tgt_cache[orig_idx] = targets

        # Apply transforms
        from bernese.data.transforms.base import Sample

        sample = Sample(sequences=seq.unsqueeze(0), targets=targets.unsqueeze(0))

        if len(self.transform) > 0:
            sample = self.transform(sample)

        if len(self.target_transform) > 0:
            sample = self.target_transform(sample)

        return sample.sequences.squeeze(0), sample.targets.squeeze(0)

    def get_batch(
        self,
        start: int,
        end: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of samples efficiently.

        Args:
            start: Start index
            end: End index

        Returns:
            Tuple of (sequences, targets) tensors
        """
        indices = list(range(start, end))

        # Load all at once (more efficient)
        sequences = self.backend.get_sequences(self.split, indices)
        targets = self.backend.get_targets(self.split, indices)

        return sequences, targets

    def get_coordinates(self, indices: Optional[np.ndarray] = None) -> list[tuple[str, int, int]]:
        """Get genomic coordinates for indices.

        Args:
            indices: Indices to get coordinates for, or None for all

        Returns:
            List of (chrom, start, end) tuples
        """
        return self.backend.get_coordinates(self.split, indices)

    @property
    def seq_length(self) -> int:
        """Return sequence length."""
        return self.backend.metadata.seq_length

    @property
    def seq_depth(self) -> int:
        """Return sequence depth (number of channels for one-hot)."""
        return self.backend.metadata.seq_depth

    @property
    def target_length(self) -> int:
        """Return target length."""
        return self.backend.metadata.target_length

    @property
    def num_targets(self) -> int:
        """Return number of targets."""
        return self.backend.metadata.num_targets

    def clear_cache(self) -> None:
        """Clear the sample cache."""
        self._seq_cache.clear()
        self._tgt_cache.clear()


def create_data_loaders_v2(
    data_dir: str | Path,
    batch_size: int = 64,
    num_workers: int = 0,
    shuffle_train: bool = True,
    seq_length_crop: int = 0,
    target_length_crop: int = 0,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Create train, validation, and test data loaders (v2 API).

    Args:
        data_dir: Path to data directory
        batch_size: Batch size for loading
        num_workers: Number of worker processes
        shuffle_train: Whether to shuffle training data
        seq_length_crop: Crop length from sequence ends (deprecated, use transforms)
        target_length_crop: Crop length from target ends (deprecated, use transforms)
        pin_memory: Whether to pin memory for GPU transfer

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create backend
    backend = HDF5Backend(data_dir)

    # Get available splits
    splits = list(backend.metadata.splits.keys())

    # Create datasets
    train_dataset = None
    val_dataset = None
    test_dataset = None

    if "train" in splits:
        train_dataset = GenomicDataset(backend, split="train", shuffle=shuffle_train)

    if "valid" in splits:
        val_dataset = GenomicDataset(backend, split="valid", shuffle=False)

    if "test" in splits:
        test_dataset = GenomicDataset(backend, split="test", shuffle=False)

    # Create loaders
    train_loader = None
    val_loader = None
    test_loader = None

    if train_dataset is not None:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return train_loader, val_loader, test_loader


# Backwards compatibility alias
create_data_loaders = create_data_loaders_v2


class DatasetWriter:
    """Backward-compatible wrapper for HDF5Writer.

    This class delegates to HDF5Writer for all operations.
    See HDF5Writer for full documentation.

    Example:
        writer = DatasetWriter("output_dir")

        # Write pre-encoded genome
        writer.write_genome(genome_dict)

        # Write indices
        writer.write_indices("train", chrom_indices, starts, ends)

        # Write targets
        writer.write_targets("train", targets)

        # Finalize
        writer.finalize(...)
    """

    def __init__(
        self,
        output_dir: str | Path,
        seq_length: int = 131072,
        seq_depth: int = 4,
        target_length: int = 0,
        num_targets: int = 1,
    ):
        # Delegate to HDF5Writer
        self._writer = HDF5Writer(
            output_dir=output_dir,
            seq_length=seq_length,
            seq_depth=seq_depth,
            target_length=target_length,
            num_targets=num_targets,
        )

        # Expose attributes for backward compatibility
        self.output_dir = self._writer.output_dir
        self.seq_length = self._writer.seq_length
        self.seq_depth = self._writer.seq_depth
        self.target_length = self._writer.target_length
        self.num_targets = self._writer.num_targets

    def write_genome(
        self,
        genome_dict: dict[int, np.ndarray],
    ) -> None:
        """Write pre-encoded genome to HDF5.

        Args:
            genome_dict: Dictionary mapping chrom_idx to 1hot encoded array
        """
        self._writer.write_genome(genome_dict)

    def write_indices(
        self,
        split: str,
        chrom_indices: list[int],
        starts: list[int],
        ends: list[int],
    ) -> None:
        """Write sequence indices for a split.

        Args:
            split: Split name (train/valid/test)
            chrom_indices: List of chromosome indices
            starts: List of start positions
            ends: List of end positions
        """
        self._writer.write_indices(split, chrom_indices, starts, ends)

    def write_coordinates(
        self,
        split: str,
        coordinates: list[tuple[str, int, int]],
    ) -> None:
        """Write genomic coordinates for a split."""
        self._writer.write_coordinates(split, coordinates)

    def write_sequences(
        self,
        split: str,
        sequences: np.ndarray,
        chunk_size: int = 256,
    ) -> None:
        """Write pre-extracted sequences for a split (legacy compatibility)."""
        self._writer.write_sequences(split, sequences, chunk_size)

    def write_targets(
        self,
        split: str,
        targets: np.ndarray,
        chunk_size: int = 1024,
    ) -> None:
        """Write targets for a split."""
        self._writer.write_targets(split, targets, chunk_size)

    def finalize(
        self,
        genome_name: str = "",
        target_type: str = "unknown",
        pool_width: int = 1,
        diagonal_offset: int = 0,
        target_info: list[dict] | None = None,
    ) -> "DatasetMetadata":
        """Finalize dataset by creating manifest.json."""
        return self._writer.finalize(
            genome_name=genome_name,
            target_type=target_type,
            pool_width=pool_width,
            diagonal_offset=diagonal_offset,
            target_info=target_info,
        )


class MultiDatasetWrapper(Dataset):
    """Wrapper for combining multiple datasets.

    This enables training on multiple datasets with round-robin sampling.
    """

    def __init__(
        self,
        datasets: list[GenomicDataset],
        weights: Optional[list[float]] = None,
    ):
        self.datasets = datasets

        if weights is None:
            self.weights = [1.0] * len(datasets)
        else:
            self.weights = weights

        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

        # Calculate total length
        self.total_len = sum(int(d.num_seqs * w) for d, w in zip(datasets, self.weights))

        # Build index mapping
        self._build_index_map()

    def _build_index_map(self):
        """Build mapping from global index to (dataset, local_idx)."""
        self._index_map = []

        for di, (dataset, weight) in enumerate(zip(self.datasets, self.weights)):
            num_samples = int(dataset.num_seqs * weight)
            for _ in range(num_samples):
                self._index_map.append(di)

    def __len__(self) -> int:
        return self.total_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get sample from wrapped datasets."""
        di = self._index_map[idx % len(self._index_map)]
        local_idx = idx % len(self.datasets[di])
        return self.datasets[di][local_idx]
