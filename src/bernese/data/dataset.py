# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Genomic dataset with backend and transform support.

This module provides a unified PyTorch Dataset interface that supports
multiple storage backends and composable transforms.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from bernese.data.backends import DataBackend, HDF5Backend
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


def create_data_loaders(
    data_dir: str | Path,
    batch_size: int = 64,
    num_workers: int = 0,
    shuffle_train: bool = True,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Create train, validation, and test data loaders.

    Args:
        data_dir: Path to data directory
        batch_size: Batch size for loading
        num_workers: Number of worker processes
        shuffle_train: Whether to shuffle training data
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
