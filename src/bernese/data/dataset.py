# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""PyTorch Dataset for genomic sequences with HDF5 support."""

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Tuple, TypeVar

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = "torch.Tensor"  # type: ignore[misc,assignment]

# Type variable for transform functions
T = TypeVar("T", bound=Tensor)


class SeqDataset(Dataset):
    """Sequence dataset for regulatory activity prediction.

    Supports loading from HDF5 files or memory-mapped arrays.

    Args:
        data_dir: Path to data directory containing sequences.h5 and statistics.json
        split: Dataset split - 'train', 'valid', or 'test'
        batch_size: Batch size for loading (used for batch count calculation)
        shuffle: Whether to shuffle the dataset (for training)
        seq_length_crop: Crop length from sequence ends
        transform: Optional transform to apply to sequences
        target_transform: Optional transform to apply to targets
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        batch_size: int = 64,
        shuffle: bool = True,
        seq_length_crop: int = 0,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seq_length_crop = seq_length_crop
        self.transform = transform
        self.target_transform = target_transform

        # Load statistics
        stats_file = self.data_dir / "statistics.json"
        with open(stats_file) as f:
            stats = json.load(f)

        self.seq_length = stats["seq_length"]
        self.seq_depth = stats.get("seq_depth", 4)
        self.target_length = stats["target_length"]
        self.num_targets = stats["num_targets"]
        self.pool_width = stats.get("pool_width", 1)

        # Calculate actual sequence length after cropping
        self._actual_seq_length = self.seq_length
        if seq_length_crop > 0:
            self._actual_seq_length = self.seq_length - seq_length_crop

        # Load sequence count
        self.num_seqs = stats.get(f"{split}_seqs", 0)

        # Track indices for shuffling
        self._indices = np.arange(self.num_seqs)
        if self.shuffle and self.num_seqs > 0:
            np.random.shuffle(self._indices)

        # Load HDF5 data
        self._sequences = None
        self._targets = None
        self._load_data()

    def _load_data(self):
        """Load sequences and targets from HDF5 file."""
        h5_file = self.data_dir / "sequences.h5"

        if h5_file.exists():
            with h5py.File(h5_file, "r") as f:
                # Load sequences
                if self.split in f["sequences"]:
                    seqs = f[f"sequences/{self.split}"]
                    # Handle optional cropping
                    if self.seq_length_crop > 0:
                        crop_len = self.seq_length_crop // 2
                        self._sequences = seqs[crop_len:-crop_len]
                    else:
                        self._sequences = seqs[:]

                # Load targets
                if self.split in f["targets"]:
                    tgts = f[f"targets/{self.split}"]
                    self._targets = tgts[:]
        else:
            # Try loading from numpy files as fallback
            seq_file = self.data_dir / f"sequences_{self.split}.npy"
            tgt_file = self.data_dir / f"targets_{self.split}.npy"

            if seq_file.exists():
                self._sequences = np.load(seq_file)
                if self.seq_length_crop > 0:
                    crop_len = self.seq_length_crop // 2
                    self._sequences = self._sequences[:, crop_len:-crop_len, :]

            if tgt_file.exists():
                self._targets = np.load(tgt_file)

    def __len__(self) -> int:
        return self.num_seqs

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sequence and its targets.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (sequence, targets) tensors
        """
        # Apply index shuffling if enabled
        if self.shuffle:
            idx = self._indices[idx]

        # Get sequence
        if self._sequences is not None:
            seq = torch.from_numpy(self._sequences[idx].copy())
        else:
            # Generate placeholder if no data loaded
            seq = torch.zeros(self._actual_seq_length, self.seq_depth, dtype=torch.float32)

        # Get targets
        if self._targets is not None:
            targets = torch.from_numpy(self._targets[idx].copy())
        else:
            targets = torch.zeros(self.target_length, self.num_targets, dtype=torch.float32)

        # Apply transforms
        if self.transform is not None:
            seq = self.transform(seq)

        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return seq, targets

    def num_batches(self) -> int:
        """Return number of batches per epoch."""
        return max(1, self.num_seqs // self.batch_size)

    def get_statistics(self) -> dict[str, Any]:
        """Return dataset statistics."""
        return {
            "seq_length": self.seq_length,
            "seq_depth": self.seq_depth,
            "target_length": self.target_length,
            "num_targets": self.num_targets,
            "num_seqs": self.num_seqs,
            "pool_width": self.pool_width,
        }


class SeqDatasetLazy(IterableDataset):
    """Lazy-loading sequence dataset for large-scale training.

    Loads data on-demand from HDF5 files without loading entire dataset into memory.

    Args:
        data_dir: Path to data directory containing sequences.h5 and statistics.json
        split: Dataset split - 'train', 'valid', or 'test'
        batch_size: Batch size for loading
        shuffle: Whether to shuffle the dataset
        seq_length_crop: Crop length from sequence ends
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        batch_size: int = 64,
        shuffle: bool = True,
        seq_length_crop: int = 0,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seq_length_crop = seq_length_crop

        # Load statistics
        stats_file = self.data_dir / "statistics.json"
        with open(stats_file) as f:
            stats = json.load(f)

        self.seq_length = stats["seq_length"]
        self.seq_depth = stats.get("seq_depth", 4)
        self.target_length = stats["target_length"]
        self.num_targets = stats["num_targets"]

        # Calculate actual sequence length after cropping
        self._actual_seq_length = self.seq_length
        if seq_length_crop > 0:
            self._actual_seq_length = self.seq_length - seq_length_crop

        # Load sequence count
        self.num_seqs = stats.get(f"{split}_seqs", 0)

        # Track indices for shuffling
        self._indices = np.arange(self.num_seqs)

    def __len__(self) -> int:
        return self.num_seqs

    def __iter__(self):
        """Iterate over batches of sequences and targets."""
        # Shuffle indices if needed
        indices = self._indices.copy()
        if self.shuffle:
            np.random.shuffle(indices)

        # Open HDF5 file
        h5_file = self.data_dir / "sequences.h5"

        if h5_file.exists():
            with h5py.File(h5_file, "r") as f:
                seqs = f[f"sequences/{self.split}"]
                tgts = f[f"targets/{self.split}"]

                for start in range(0, self.num_seqs, self.batch_size):
                    end = min(start + self.batch_size, self.num_seqs)
                    batch_indices = indices[start:end]

                    # Load batch
                    batch_seqs = torch.from_numpy(seqs[batch_indices].astype(np.float32))
                    batch_targets = torch.from_numpy(tgts[batch_indices].astype(np.float32))

                    # Apply cropping if needed
                    if self.seq_length_crop > 0:
                        crop_len = self.seq_length_crop // 2
                        batch_seqs = batch_seqs[:, crop_len:-crop_len, :]

                    yield batch_seqs, batch_targets
        else:
            # Fallback to numpy files
            seq_file = self.data_dir / f"sequences_{self.split}.npy"
            tgt_file = self.data_dir / f"targets_{self.split}.npy"

            if seq_file.exists() and tgt_file.exists():
                sequences = np.load(seq_file)
                targets = np.load(tgt_file)

                for start in range(0, self.num_seqs, self.batch_size):
                    end = min(start + self.batch_size, self.num_seqs)
                    batch_indices = indices[start:end]

                    batch_seqs = torch.from_numpy(sequences[batch_indices].astype(np.float32))
                    batch_targets = torch.from_numpy(targets[batch_indices].astype(np.float32))

                    if self.seq_length_crop > 0:
                        crop_len = self.seq_length_crop // 2
                        batch_seqs = batch_seqs[:, crop_len:-crop_len, :]

                    yield batch_seqs, batch_targets
            else:
                # No data files - yield empty batches
                batch_seqs = torch.zeros(0, self._actual_seq_length, self.seq_depth)
                batch_targets = torch.zeros(0, self.target_length, self.num_targets)
                yield batch_seqs, batch_targets


def create_data_loaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 0,
    shuffle_train: bool = True,
    seq_length_crop: int = 0,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Create train, validation, and test data loaders.

    Args:
        data_dir: Path to data directory
        batch_size: Batch size for loading
        num_workers: Number of worker processes for data loading
        shuffle_train: Whether to shuffle training data
        seq_length_crop: Crop length from sequence ends
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Determine available splits
    stats_file = Path(data_dir) / "statistics.json"
    with open(stats_file) as f:
        stats = json.load(f)

    # Create datasets
    train_dataset = SeqDataset(
        data_dir,
        split="train",
        batch_size=batch_size,
        shuffle=shuffle_train,
        seq_length_crop=seq_length_crop,
    )
    val_dataset = SeqDataset(
        data_dir,
        split="valid",
        batch_size=batch_size,
        shuffle=False,
        seq_length_crop=seq_length_crop,
    )

    # Check if test split exists
    test_dataset = None
    if "test_seqs" in stats and stats["test_seqs"] > 0:
        test_dataset = SeqDataset(
            data_dir,
            split="test",
            batch_size=batch_size,
            shuffle=False,
            seq_length_crop=seq_length_crop,
        )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return train_loader, val_loader, test_loader


def load_targets_info(data_dir: str) -> pd.DataFrame:
    """Load targets information from targets.txt.

    Args:
        data_dir: Path to data directory

    Returns:
        DataFrame with target information
    """
    targets_file = Path(data_dir) / "targets.txt"
    if targets_file.exists():
        return pd.read_csv(targets_file, sep="\t", index_col=0)
    return None


def create_collate_fn(pad_value: float = 0.0) -> Callable:
    """Create a collate function for batching variable-length sequences.

    Args:
        pad_value: Value to use for padding

    Returns:
        Collate function
    """

    def collate_fn(
        batch: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequences, targets = zip(*batch)
        return torch.stack(sequences), torch.stack(targets)

    return collate_fn


################################################################################
# Target Transformation Functions (from baskerville)
################################################################################


def untransform_preds(
    preds: np.ndarray,
    targets_df: pd.DataFrame,
    unscale: bool = False,
    unclip: bool = True,
) -> np.ndarray:
    """Undo the squashing transformations performed for the tasks.

    This reverses the soft clipping and sqrt transformations applied during
    target preprocessing.

    Args:
        preds: Predictions array of shape (target_length, num_targets)
        targets_df: Targets information DataFrame with columns:
            - clip_soft: Soft clipping threshold
            - sum_stat: Sum statistics (e.g., 'sum', 'mean', '_sqrt')
            - scale: Scale factor
        unscale: If True, also undo the scaling transformation

    Returns:
        Untransformed predictions array of same shape as input

    Example:
        >>> # Assuming targets_df has clip_soft, sum_stat, and scale columns
        >>> untransformed = untransform_preds(preds, targets_df)
    """
    preds = preds.copy()

    # Unclip soft (reverse soft clipping)
    if unclip:
        clip_soft = np.expand_dims(np.array(targets_df["clip_soft"]), axis=0)
        preds_unclip = clip_soft - 1 + (preds - clip_soft + 1) ** 2
        preds = np.where(preds > clip_soft, preds_unclip, preds)

    # Unsquash sqrt
    sqrt_mask = np.array([ss.find("_sqrt") != -1 for ss in targets_df["sum_stat"]])
    preds[:, sqrt_mask] = -1 + (preds[:, sqrt_mask] + 1) ** 2  # (4 / 3)

    # Unscale
    if unscale:
        scale = np.expand_dims(np.array(targets_df["scale"]), axis=0)
        preds = preds / scale

    return preds


def untransform_preds1(
    preds: np.ndarray,
    targets_df: pd.DataFrame,
    unscale: bool = False,
    unclip: bool = True,
) -> np.ndarray:
    """Undo squashing transformations (alternative version).

    This is an alternative implementation that applies transformations in
    a different order.

    Args:
        preds: Predictions array of shape (target_length, num_targets)
        targets_df: Targets information DataFrame
        unscale: If True, also undo the scaling transformation
        unclip: If True, also undo the soft clipping transformation

    Returns:
        Untransformed predictions array of same shape as input
    """
    preds = preds.copy()

    # Scale
    scale = np.expand_dims(np.array(targets_df["scale"]), axis=0)
    preds = preds / scale

    # Unclip soft
    if unclip:
        clip_soft = np.expand_dims(np.array(targets_df["clip_soft"]), axis=0)
        preds_unclip = clip_soft + (preds - clip_soft) ** 2
        preds = np.where(preds > clip_soft, preds_unclip, preds)

    # Un-squash power (0.75)
    sqrt_mask = np.array([ss.find("_sqrt") != -1 for ss in targets_df["sum_stat"]])
    preds[:, sqrt_mask] = preds[:, sqrt_mask] ** (4 / 3)

    # Unscale
    if not unscale:
        preds = preds * scale

    return preds


def untransform_preds_torch(
    preds: torch.Tensor,
    targets_df: pd.DataFrame,
    unscale: bool = False,
    unclip: bool = True,
) -> torch.Tensor:
    """Undo squashing transformations (PyTorch version).

    Args:
        preds: Predictions tensor of shape (target_length, num_targets) or (batch, target_length, num_targets)
        targets_df: Targets information DataFrame
        unscale: If True, also undo the scaling transformation
        unclip: If True, also undo the soft clipping transformation

    Returns:
        Untransformed predictions tensor
    """
    preds = preds.clone()

    # Handle both 2D and 3D tensors
    squeeze_batch = False
    if preds.ndim == 3:
        squeeze_batch = True
        batch_size = preds.shape[0]
        preds = preds.view(-1, preds.shape[-1])

    # Unclip soft
    if unclip:
        clip_soft = torch.from_numpy(np.expand_dims(np.array(targets_df["clip_soft"]), axis=0)).to(
            preds.device
        )
        preds_unclip = clip_soft - 1 + (preds - clip_soft + 1) ** 2
        preds = torch.where(preds > clip_soft, preds_unclip, preds)

    # Unsquash sqrt
    sqrt_mask = np.array([ss.find("_sqrt") != -1 for ss in targets_df["sum_stat"]])
    sqrt_mask_tensor = torch.from_numpy(sqrt_mask).to(preds.device)
    preds[:, sqrt_mask_tensor] = -1 + (preds[:, sqrt_mask_tensor] + 1) ** 2

    # Unscale
    if unscale:
        scale = torch.from_numpy(np.expand_dims(np.array(targets_df["scale"]), axis=0)).to(
            preds.device
        )
        preds = preds / scale

    # Reshape back if needed
    if squeeze_batch:
        preds = preds.view(batch_size, -1, preds.shape[-1])

    return preds


def make_strand_transform(
    targets_df: pd.DataFrame,
    targets_strand_df: pd.DataFrame,
) -> np.ndarray:
    """Make a sparse matrix to sum strand pairs.

    Creates a transformation matrix that sums predictions from both strands
    of stranded target pairs.

    Args:
        targets_df: Full targets DataFrame with strand information
        targets_strand_df: Collapsed stranded targets DataFrame

    Returns:
        Sparse matrix (dok format) to transform predictions from full to stranded

    Example:
        >>> strand_transform = make_strand_transform(targets_df, targets_strand_df)
        >>> stranded_preds = strand_transform @ preds
    """
    from scipy.sparse import dok_matrix

    # Initialize sparse matrix
    num_rows = targets_df.shape[0]
    num_cols = targets_strand_df.shape[0]
    strand_transform = dok_matrix((num_rows, num_cols), dtype=np.float32)

    # Track which strand pairs we've seen
    seen_pairs = set()

    # Fill in matrix
    ti = 0  # target index in full DataFrame
    sti = 0  # target index in stranded DataFrame
    for _, target in targets_df.iterrows():
        strand_transform[ti, sti] = True
        if target.get("strand_pair", target.name) == target.name:
            # Unstranded target - move to next stranded target
            sti += 1
        else:
            # Stranded target
            strand_pair = target.get("strand_pair", "")
            if strand_pair in seen_pairs:
                # This is the second member of the pair
                sti += 1
            else:
                # This is the first member of the pair
                seen_pairs.add(target.name)
        ti += 1

    return strand_transform


def targets_prep_strand(targets_df: pd.DataFrame) -> pd.DataFrame:
    """Adjust targets table for merged stranded datasets.

    Collapses stranded targets (forward and reverse) into a single row.

    Args:
        targets_df: Targets DataFrame with strand information

    Returns:
        DataFrame with stranded targets collapsed

    Example:
        >>> collapsed = targets_prep_strand(targets_df)
    """
    # Attach strand
    targets_strand = []
    for _, target in targets_df.iterrows():
        strand_pair = target.get("strand_pair", target.name)
        if strand_pair == target.name:
            targets_strand.append(".")
        else:
            # Extract strand from identifier
            identifier = str(target.get("identifier", ""))
            targets_strand.append(identifier[-1] if identifier else ".")
    targets_df = targets_df.copy()
    targets_df["strand"] = targets_strand

    # Collapse stranded
    strand_mask = targets_df["strand"] != "-"
    targets_strand_df = targets_df[strand_mask]

    return targets_strand_df
