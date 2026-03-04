# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""HDF5 data backend implementation.

This module provides HDF5-based storage with split organization.
Sequences are loaded from pre-encoded genome.h5 using indices.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from bernese.data.backends.base import (
    DatasetMetadata,
    GenomeInfo,
    SplitMetadata,
    TargetInfo,
)


class HDF5Backend:
    """HDF5-based data backend with split organization.

    Expected folder structure:
        data_dir/
        ├── manifest.json
        ├── genome.h5              # Pre-encoded genome (chrom × 4)
        ├── sequences/
        │   ├── train/indices.h5    # (chrom_idx, start, end)
        │   ├── valid/indices.h5
        │   └── test/indices.h5
        └── targets/
            ├── train.h5
            ├── valid.h5
            └── test.h5

    Args:
        data_dir: Path to dataset directory
    """

    def __init__(
        self,
        data_dir: str | Path,
        preload_sequences: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.preload_sequences = preload_sequences

        # Load manifest
        manifest_path = self.data_dir / "manifest.json"
        if manifest_path.exists():
            self._metadata = DatasetMetadata.from_manifest(manifest_path)
        else:
            # Fallback: try to infer from legacy format
            self._metadata = self._infer_metadata_legacy()

        # Track open file handles
        self._genome_file = None
        self._tgt_files: dict[str, h5py.File] = {}
        self._coord_cache: dict[str, np.ndarray] = {}

    def _infer_metadata_legacy(self) -> DatasetMetadata:
        """Infer metadata from legacy format (statistics.json)."""
        stats_file = self.data_dir / "sequences.h5"
        if not stats_file.exists():
            stats_file = self.data_dir.parent / "sequences.h5"

        # Try to load from statistics.json
        import json

        stats = {}
        for name in ["statistics.json", "statistics_v2.json"]:
            path = self.data_dir / name
            if path.exists():
                with open(path) as f:
                    stats = json.load(f)
                break

        # Build splits from stats
        splits = {}
        for split in ["train", "valid", "test"]:
            key = f"{split}_seqs"
            if key in stats and stats[key] > 0:
                splits[split] = SplitMetadata(
                    name=split,
                    num_seqs=stats[key],
                    coordinate_file=f"sequences/{split}/indices.h5",
                    target_file=f"targets/{split}.h5",
                )

        return DatasetMetadata(
            version="1.0",
            seq_length=stats.get("seq_length", 0),
            seq_depth=stats.get("seq_depth", 4),
            target_length=stats.get("target_length", 0),
            num_targets=stats.get("num_targets", 0),
            target_type="hic",
            pool_width=stats.get("pool_width", 1),
            diagonal_offset=stats.get("diagonal_offset", 2),
            splits=splits,
        )

    @property
    def metadata(self) -> DatasetMetadata:
        """Return dataset metadata."""
        return self._metadata

    def _get_split_info(self, split: str) -> SplitMetadata:
        """Get split metadata."""
        if split in self._metadata.splits:
            return self._metadata.splits[split]
        raise ValueError(f"Unknown split: {split}")

    def _ensure_genome_loaded(self) -> None:
        """Load genome file if not already open."""
        if self._genome_file is None:
            genome_file = self.data_dir / "genome.h5"
            if genome_file.exists():
                self._genome_file = h5py.File(genome_file, "r")

    def _ensure_coord_loaded(self, split: str) -> None:
        """Load coordinates into cache if not already loaded."""
        if split in self._coord_cache:
            return

        split_info = self._get_split_info(split)
        coord_file = self.data_dir / split_info.coordinate_file

        if coord_file.exists():
            with h5py.File(coord_file, "r") as f:
                # Try new format with "chrom" (string names), fallback to "chrom_idx" (legacy)
                if "chrom" in f:
                    chrom_data = f["chrom"][:]
                    # Handle both fixed-length and variable-length string types
                    if isinstance(chrom_data, bytes):
                        chroms = [c.decode("utf-8") for c in chrom_data]
                    else:
                        chroms = list(chrom_data)
                    self._coord_cache[split] = {
                        "chrom": np.array(chroms),
                        "start": f["start"][:],
                        "end": f["end"][:],
                    }
                elif "chrom_idx" in f:
                    # Legacy format: convert integer indices to string representation
                    self._coord_cache[split] = {
                        "chrom": np.array([str(i) for i in f["chrom_idx"][:]]),
                        "start": f["start"][:],
                        "end": f["end"][:],
                    }
                else:
                    # No chrom data found
                    self._coord_cache[split] = {
                        "chrom": np.array([]),
                        "start": np.array([]),
                        "end": np.array([]),
                    }

    def _ensure_target_file_open(self, split: str) -> None:
        """Ensure target HDF5 file is open for split."""
        if split not in self._tgt_files:
            split_info = self._get_split_info(split)
            tgt_file = self.data_dir / split_info.target_file
            if tgt_file.exists():
                self._tgt_files[split] = h5py.File(tgt_file, "r")

    def get_sequences(
        self,
        split: str,
        indices: np.ndarray | list[int] | slice | None = None,
    ) -> torch.Tensor:
        """Load sequences for a split.

        Args:
            split: Dataset split name (train/valid/test)
            indices: Specific indices to load, or None for all

        Returns:
            Tensor of shape (batch, seq_length, seq_depth)
        """
        self._ensure_coord_loaded(split)
        self._ensure_genome_loaded()

        if split not in self._coord_cache:
            # No coordinates - return zeros
            split_info = self._get_split_info(split)
            num = split_info.num_seqs if indices is None else len(indices)
            return torch.zeros(num, self._metadata.seq_length, self._metadata.seq_depth)

        coords = self._coord_cache[split]
        seq_length = self._metadata.seq_length
        seq_depth = self._metadata.seq_depth

        # Normalize indices
        if indices is None:
            idx = np.arange(coords["chrom"].shape[0])
        else:
            idx = np.asarray(indices)

        # Extract sequences from pre-encoded genome
        seqs = []
        if self._genome_file is not None:
            for i in idx:
                chrom_name = coords["chrom"][i]
                start = coords["start"][i]
                end = coords["end"][i]

                # Extract from genome.h5 - each chromosome is a separate dataset
                # Dataset name is "chrom_{chrom_name}" (e.g., "chrom_chr1")
                chrom_key = f"chrom_{chrom_name}"
                if chrom_key in self._genome_file:
                    chrom_seq = self._genome_file[chrom_key]
                    # Extract region: (start:end, :), shape (seq_length, 4)
                    region = chrom_seq[start:end, :]
                    # Pad if needed
                    if region.shape[0] < seq_length:
                        padding = np.zeros((seq_length - region.shape[0], seq_depth), dtype=np.float32)
                        region = np.vstack([region, padding])
                    elif region.shape[0] > seq_length:
                        region = region[:seq_length, :]
                    seqs.append(region)  # (seq_length, 4)
                else:
                    # Chromosome not found - return zeros
                    seqs.append(np.zeros((seq_length, seq_depth), dtype=np.float32))
        else:
            # No genome file - return zeros
            for _ in idx:
                seqs.append(np.zeros((seq_length, seq_depth), dtype=np.float32))

        return torch.from_numpy(np.stack(seqs).astype(np.float32))

    def get_targets(
        self,
        split: str,
        indices: np.ndarray | list[int] | slice | None = None,
    ) -> torch.Tensor:
        """Load targets for a split.

        Args:
            split: Dataset split name (train/valid/test)
            indices: Specific indices to load, or None for all

        Returns:
            Tensor of shape (batch, target_length, num_targets)
        """
        split_info = self._get_split_info(split)
        self._ensure_target_file_open(split)

        if split not in self._tgt_files:
            # Return zeros if no targets file
            num = split_info.num_seqs if indices is None else len(indices)
            return torch.zeros(num, self._metadata.target_length, self._metadata.num_targets)

        with h5py.File(self._tgt_files[split].filename, "r") as f:
            # Try different dataset names
            for ds_name in ["data", "targets"]:
                if ds_name in f:
                    ds = f[ds_name]
                    if indices is not None:
                        data = ds[indices]
                    else:
                        data = ds[:]
                    return torch.from_numpy(data.astype(np.float32))

        # Fallback: zeros
        num = split_info.num_seqs if indices is None else len(indices)
        return torch.zeros(num, self._metadata.target_length, self._metadata.num_targets)

    def get_coordinates(
        self,
        split: str,
        indices: np.ndarray | list[int] | slice | None = None,
    ) -> list[tuple[str, int, int]]:
        """Load genomic coordinates for sequences.

        Args:
            split: Dataset split name
            indices: Specific indices to load, or None for all

        Returns:
            List of (chrom, start, end) tuples
        """
        self._ensure_coord_loaded(split)

        if split not in self._coord_cache:
            return []

        coords = self._coord_cache[split]

        # Normalize indices
        if indices is None:
            idx = np.arange(coords["chrom"].shape[0])
        else:
            idx = np.asarray(indices)

        return [
            (str(coords["chrom"][i]), int(coords["start"][i]), int(coords["end"][i])) for i in idx
        ]

    def close(self) -> None:
        """Close all open file handles."""
        if self._genome_file is not None:
            self._genome_file.close()
            self._genome_file = None

        for f in self._tgt_files.values():
            f.close()
        self._tgt_files.clear()

        self._coord_cache.clear()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()

    def __len__(self) -> int:
        """Return total number of sequences across all splits."""
        return sum(s.num_seqs for s in self._metadata.splits.values())

    def __getitem__(self, key: str) -> Any:
        """Allow dict-style access to metadata."""
        return getattr(self._metadata, key, None)


class HDF5Writer:
    """HDF5-based data writer for creating datasets in v2 format.

    This class handles writing genomic datasets with:
    - Pre-encoded genome (genome.h5)
    - Sequence indices (chrom_idx, start, end)
    - Target arrays per split
    - Manifest.json creation

    Example:
        writer = HDF5Writer("output_dir")

        # Write pre-encoded genome
        writer.write_genome(genome_1hot, chrom_lengths)

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
        self.output_dir = Path(output_dir)
        self.seq_length = seq_length
        self.seq_depth = seq_depth
        self.target_length = target_length
        self.num_targets = num_targets

        # Track what has been written
        self._splits_written: set[str] = set()
        self._chrom_lengths: dict[int, int] = {}

        # Create directory structure
        self._create_directories()

    def _create_directories(self) -> None:
        """Create output directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for split in ["train", "valid", "test"]:
            (self.output_dir / "sequences" / split).mkdir(parents=True, exist_ok=True)

        (self.output_dir / "targets").mkdir(parents=True, exist_ok=True)

    def write_genome(
        self,
        genome_dict: dict[str, np.ndarray],
    ) -> None:
        """Write pre-encoded genome to HDF5.

        Args:
            genome_dict: Dictionary mapping chrom_name to 1hot encoded array (length × 4)
                          e.g., {"chr1": array, "chr2": array, ...}
        """
        genome_file = self.output_dir / "genome.h5"

        with h5py.File(genome_file, "w") as f:
            for chrom_name, seq_1hot in genome_dict.items():
                # Store with "chrom_" prefix: "chrom_chr1", "chrom_chr2", etc.
                chrom_key = f"chrom_{chrom_name}"
                f.create_dataset(chrom_key, data=seq_1hot, compression="gzip")
                self._chrom_lengths[chrom_name] = seq_1hot.shape[0]

    def write_indices(
        self,
        split: str,
        chrom_names: list[str],
        starts: list[int],
        ends: list[int],
    ) -> None:
        """Write sequence indices for a split.

        Args:
            split: Split name (train/valid/test)
            chrom_names: List of chromosome names (e.g., "chr1", "chr2")
            starts: List of start positions
            ends: List of end positions
        """
        coord_file = self.output_dir / "sequences" / split / "indices.h5"

        with h5py.File(coord_file, "w") as f:
            # Store chromosome names as strings (variable-length UTF-8)
            dt = h5py.string_dtype()
            f.create_dataset("chrom", data=np.array(chrom_names, dtype=dt))
            f.create_dataset("start", data=np.array(starts, dtype=np.int32))
            f.create_dataset("end", data=np.array(ends, dtype=np.int32))

        self._splits_written.add(split)

    def write_coordinates(
        self,
        split: str,
        coordinates: list[tuple[str, int, int]],
    ) -> None:
        """Write genomic coordinates for a split.

        Args:
            split: Split name (train/valid/test)
            coordinates: List of (chrom, start, end) tuples
        """
        # Extract chromosome names directly from coordinates
        chrom_names = []
        starts = []
        ends = []

        for chrom, start, end in coordinates:
            chrom_names.append(chrom)
            starts.append(start)
            ends.append(end)

        self.write_indices(split, chrom_names, starts, ends)

    def write_sequences(
        self,
        split: str,
        sequences: np.ndarray,
        chunk_size: int = 256,
    ) -> None:
        """Write pre-extracted sequences for a split (legacy compatibility).

        Args:
            split: Split name
            sequences: Array of shape (num_seqs, seq_depth, seq_length)
            chunk_size: HDF5 chunk size
        """
        # This is kept for backward compatibility but deprecated
        # New code should use write_genome + write_indices
        seq_file = self.output_dir / "sequences" / split / "data.h5"

        with h5py.File(seq_file, "w") as f:
            f.create_dataset(
                "data",
                data=sequences.astype(np.float32),
                chunks=(chunk_size, self.seq_depth, self.seq_length),
                compression="gzip",
            )

    def write_targets(
        self,
        split: str,
        targets: np.ndarray,
        chunk_size: int = 1024,
    ) -> None:
        """Write targets for a split.

        Args:
            split: Split name
            targets: Array of shape (num_seqs, target_length, num_targets)
            chunk_size: HDF5 chunk size (will be reduced if smaller than data)
        """
        tgt_file = self.output_dir / "targets" / f"{split}.h5"

        # Adapt chunk size to data size for small datasets
        actual_chunk_size = min(chunk_size, targets.shape[0])

        with h5py.File(tgt_file, "w") as f:
            f.create_dataset(
                "data",
                data=targets.astype(np.float32),
                chunks=(actual_chunk_size, self.target_length, self.num_targets),
                compression="gzip",
            )

    def finalize(
        self,
        genome_name: str = "",
        target_type: str = "unknown",
        pool_width: int = 1,
        diagonal_offset: int = 0,
        target_info: list[dict] | None = None,
    ) -> DatasetMetadata:
        """Finalize dataset by creating manifest.json.

        Args:
            genome_name: Genome name
            target_type: Target type identifier
            pool_width: Pool width
            diagonal_offset: Diagonal offset
            target_info: List of target info dicts

        Returns:
            DatasetMetadata object
        """
        import json

        # Build splits
        splits = {}
        for split in self._splits_written:
            splits[split] = SplitMetadata(
                name=split,
                num_seqs=0,  # Will be updated when indices are written
                coordinate_file=f"sequences/{split}/indices.h5",
                target_file=f"targets/{split}.h5",
            )

        # Update num_seqs from indices files
        for split in splits:
            coord_file = self.output_dir / splits[split].coordinate_file
            if coord_file.exists():
                with h5py.File(coord_file, "r") as f:
                    # Try new format "chrom", fallback to legacy "chrom_idx"
                    if "chrom" in f:
                        num_seqs = f["chrom"].shape[0]
                    elif "chrom_idx" in f:
                        num_seqs = f["chrom_idx"].shape[0]
                    else:
                        num_seqs = 0
                    splits[split] = SplitMetadata(
                        name=split,
                        num_seqs=num_seqs,
                        coordinate_file=splits[split].coordinate_file,
                        target_file=splits[split].target_file,
                    )

        # Build targets (file paths no longer stored)
        targets = []
        if target_info:
            for i, info in enumerate(target_info):
                targets.append(
                    TargetInfo(
                        name=info.get("name", f"target_{i}"),
                        clip=info.get("clip"),
                        index=i,
                    )
                )

        # Create metadata
        metadata = DatasetMetadata(
            version="2.0",
            name=self.output_dir.name,
            created=datetime.utcnow().isoformat() + "Z",
            genome=GenomeInfo(name=genome_name),
            seq_length=self.seq_length,
            seq_depth=self.seq_depth,
            target_length=self.target_length,
            num_targets=self.num_targets,
            target_type=target_type,
            pool_width=pool_width,
            diagonal_offset=diagonal_offset,
            splits=splits,
            targets=targets,
        )

        # Write manifest
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        return metadata
