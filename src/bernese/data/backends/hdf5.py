# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""HDF5 data backend implementation - Modern v2.0+ format.

This module provides high-performance HDF5-based storage with:
- Flat file layout: {split}.h5 (e.g., train.h5, valid.h5)
- Pre-loaded indices in memory
- Sorted batch I/O for optimal throughput
- Fixed-length strings (S16) for chromosomes
- uint8 compression for genome storage
- SWMR support for multi-process training
- Strict mode (no silent zero fallback)
"""

from __future__ import annotations

from contextlib import contextmanager
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
from bernese.data.config import TargetConfig


# Fixed-length string dtype for chromosome names (supports most standard chrom names)
# "chr1" -> 4 chars, "chrM" -> 4 chars, "Chr1" -> 4 chars, max 16 is safe
CHROM_DTYPE = "S16"


class HDF5Backend:
    """High-performance HDF5-based data backend.

    Expected folder structure (v2.0+ flat format):
        data_dir/
        ├── manifest.json          # Version 2.0+ metadata
        ├── genome.h5              # Pre-encoded genome (uint8, chrom × 4)
        ├── train.h5               # Indices + targets for train split
        ├── valid.h5               # Indices + targets for valid split
        └── test.h5                # Indices + targets for test split

    HDF5 internal structure for {split}.h5:
        /indices/
            /chrom    - Fixed-length strings (S16)
            /start    - int32 array
            /end      - int32 array
        /targets/
            /data     - float32 array (N, target_length, num_targets)

    Args:
        data_dir: Path to dataset directory
        preload_indices: Whether to preload indices into memory (default: True)
        strict_mode: If True, raise exceptions on missing data (default: True)
    """

    def __init__(
        self,
        data_dir: str | Path,
        preload_indices: bool = True,
        strict_mode: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.preload_indices = preload_indices
        self.strict_mode = strict_mode

        # Load manifest
        manifest_path = self.data_dir / "manifest.json"
        self._metadata = DatasetMetadata.from_manifest(manifest_path)

        # Track open file handles
        self._genome_file: h5py.File | None = None
        self._split_files: dict[str, h5py.File] = {}

        # Pre-loaded indices in memory (key optimization)
        self._indices_cache: dict[str, dict[str, np.ndarray]] = {}

        # Preload indices if requested
        if self.preload_indices:
            self._preload_all_indices()

    def _preload_all_indices(self) -> None:
        """Pre-load all split indices into memory."""
        for split in self._metadata.splits:
            self._load_indices_to_memory(split)

    def _load_indices_to_memory(self, split: str) -> None:
        """Load indices for a split into memory."""
        if split in self._indices_cache:
            return

        split_info = self._metadata.splits[split]
        split_file = self.data_dir / split_info.split_file

        if split_file.exists():
            with h5py.File(split_file, "r") as f:
                if "indices" in f:
                    indices_grp = f["indices"]

                    # Load chrom (fixed-length strings)
                    chrom_data = indices_grp["chrom"][:]
                    if isinstance(chrom_data[0], bytes):
                        # Fixed-length bytes, decode
                        chroms = np.array(
                            [c.decode("utf-8").strip() for c in chrom_data], dtype=object
                        )
                    elif isinstance(chrom_data[0], str):
                        # Already strings
                        chroms = np.array(list(chrom_data), dtype=object)
                    else:
                        # Integers - convert back to strings
                        chroms = np.array([f"chr{i}" for i in chrom_data], dtype=object)

                    self._indices_cache[split] = {
                        "chrom": chroms,
                        "start": indices_grp["start"][:].astype(np.int64),
                        "end": indices_grp["end"][:].astype(np.int64),
                    }

    @property
    def metadata(self) -> DatasetMetadata:
        """Return dataset metadata."""
        return self._metadata

    def _get_split_info(self, split: str) -> SplitMetadata:
        """Get split metadata."""
        if split in self._metadata.splits:
            return self._metadata.splits[split]
        raise ValueError(f"Unknown split: {split}")

    def _ensure_genome_open(self) -> None:
        """Open genome file with SWMR support if not already open."""
        if self._genome_file is None:
            genome_file = self.data_dir / "genome.h5"
            if genome_file.exists():
                # SWMR mode for multi-process reading
                self._genome_file = h5py.File(genome_file, "r", libver="latest", swmr=True)

    def _ensure_split_file_open(self, split: str) -> None:
        """Open split HDF5 file with SWMR support."""
        if split not in self._split_files:
            split_info = self._get_split_info(split)
            split_file = self.data_dir / split_info.split_file
            if split_file.exists():
                self._split_files[split] = h5py.File(split_file, "r", libver="latest", swmr=True)

    def _sort_indices_for_io(
        self, indices: np.ndarray, coords: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sort indices by chromosome and start position for optimal I/O.

        Returns:
            Tuple of (sorted_indices, original_order)
        """
        if len(indices) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        # Get coordinates for requested indices
        chroms = coords["chrom"][indices]
        starts = coords["start"][indices]

        # Create sorting key: (chrom, start)
        # Convert chrom to Unicode string for zfill (np.char.zfill doesn't support object dtype)
        chrom_keys = np.char.zfill(chroms.astype("U16"), 16)  # Pad for string sorting

        # Sort by chrom first, then by start
        sort_idx = np.lexsort((starts, chrom_keys))

        return indices[sort_idx], sort_idx

    def get_sequences(
        self,
        split: str,
        indices: np.ndarray | list[int] | slice | None = None,
    ) -> torch.Tensor:
        """Load sequences for a split with sorted batch I/O.

        Args:
            split: Dataset split name (train/valid/test)
            indices: Specific indices to load, or None for all

        Returns:
            Tensor of shape (batch, seq_length, seq_depth)
        """
        # Ensure indices are loaded
        if split not in self._indices_cache:
            self._load_indices_to_memory(split)

        if split not in self._indices_cache:
            # No data available
            if self.strict_mode:
                raise KeyError(f"No indices found for split: {split}")
            split_info = self._get_split_info(split)
            num = split_info.num_seqs if indices is None else len(indices)
            return torch.zeros(num, self._metadata.seq_length, self._metadata.seq_depth)

        coords = self._indices_cache[split]
        seq_length = self._metadata.seq_length
        seq_depth = self._metadata.seq_depth

        # Normalize indices
        if indices is None:
            idx = np.arange(coords["chrom"].shape[0], dtype=np.int64)
        else:
            idx = np.asarray(indices, dtype=np.int64)

        # Sort indices for optimal I/O
        sorted_idx, sort_order = self._sort_indices_for_io(idx, coords)

        # Extract sequences from genome
        self._ensure_genome_open()

        seqs = []
        if self._genome_file is not None:
            # Group by chromosome for batch reading
            chrom_groups: dict[str, list[tuple[int, int, int]]] = {}

            for sorted_i, orig_idx in enumerate(sorted_idx):
                chrom_name = str(coords["chrom"][orig_idx])
                start = int(coords["start"][orig_idx])
                end = int(coords["end"][orig_idx])

                if chrom_name not in chrom_groups:
                    chrom_groups[chrom_name] = []
                chrom_groups[chrom_name].append((sorted_i, start, end))

            # Read each chromosome contiguously
            results = [None] * len(sorted_idx)

            for chrom_name, regions in chrom_groups.items():
                if chrom_name in self._genome_file:
                    chrom_data = self._genome_file[chrom_name]

                    for sorted_i, start, end in regions:
                        region = chrom_data[start:end, :]

                        # Handle length mismatches
                        if region.shape[0] < seq_length:
                            padding = np.zeros(
                                (seq_length - region.shape[0], seq_depth), dtype=np.uint8
                            )
                            region = np.vstack([region, padding])
                        elif region.shape[0] > seq_length:
                            region = region[:seq_length, :]

                        results[sorted_i] = region
                else:
                    # Chromosome not found in genome
                    if self.strict_mode:
                        raise KeyError(f"Chromosome {chrom_name} not found in genome.h5")
                    for sorted_i, start, end in regions:
                        results[sorted_i] = np.zeros((seq_length, seq_depth), dtype=np.uint8)

            seqs = results
        else:
            # No genome file
            if self.strict_mode:
                raise FileNotFoundError("genome.h5 not found")
            for _ in sorted_idx:
                seqs.append(np.zeros((seq_length, seq_depth), dtype=np.uint8))

        # Convert to tensor and restore original order
        seqs_tensor = torch.from_numpy(np.stack(seqs).astype(np.float32))

        # Restore original order
        if len(sort_order) > 0:
            inverse_order = np.argsort(sort_order)
            seqs_tensor = seqs_tensor[inverse_order]

        return seqs_tensor

    def get_targets(
        self,
        split: str,
        indices: np.ndarray | list[int] | slice | None = None,
        target_index: int | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Load targets for a split.

        Args:
            split: Dataset split name (train/valid/test)
            indices: Specific indices to load, or None for all
            target_index: Specific target index to load, or None for all targets

        Returns:
            If target_index is specified: Tensor of shape (batch, target_length)
            If target_index is None: List of tensors, one per TargetConfig
        """
        split_info = self._get_split_info(split)
        self._ensure_split_file_open(split)

        if split not in self._split_files:
            if self.strict_mode:
                raise KeyError(f"No target file found for split: {split}")
            num = split_info.num_seqs if indices is None else len(indices)
            return torch.zeros(num, self._metadata.target_length, self._metadata.num_targets)

        # Read from split file
        with h5py.File(self._split_files[split].filename, "r") as f:
            if "targets" not in f:
                if self.strict_mode:
                    raise KeyError(f"No targets found in {split_info.split_file}")
                num = split_info.num_seqs if indices is None else len(indices)
                return [torch.zeros(num, self._metadata.target_length) for _ in range(self._metadata.num_targets)]

            targets_grp = f["targets"]

            if target_index is not None:
                # Read specific target: /targets/{target_index}
                target_path = str(target_index)
                if target_path not in targets_grp:
                    raise KeyError(f"Target {target_index} not found in {split}")

                targets_ds = targets_grp[target_path]
                if indices is not None:
                    data = targets_ds[indices]
                else:
                    data = targets_ds[:]
                return torch.from_numpy(data.astype(np.float32))
            else:
                # Read all targets
                results = []
                num_targets = self._metadata.num_targets
                for i in range(num_targets):
                    target_path = str(i)
                    if target_path not in targets_grp:
                        raise KeyError(f"Target {i} not found in {split}")

                    targets_ds = targets_grp[target_path]
                    if indices is not None:
                        data = targets_ds[indices]
                    else:
                        data = targets_ds[:]
                    results.append(torch.from_numpy(data.astype(np.float32)))
                return results

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
        # Ensure indices are loaded
        if split not in self._indices_cache:
            self._load_indices_to_memory(split)

        if split not in self._indices_cache:
            return []

        coords = self._indices_cache[split]

        # Normalize indices
        if indices is None:
            idx = np.arange(coords["chrom"].shape[0])
        else:
            idx = np.asarray(indices)

        return [
            (str(coords["chrom"][i]), int(coords["start"][i]), int(coords["end"][i])) for i in idx
        ]

    @contextmanager
    def open(self):
        """Context manager for proper file handle management."""
        try:
            yield self
        finally:
            self.close()

    def close(self) -> None:
        """Close all open file handles."""
        if self._genome_file is not None:
            self._genome_file.close()
            self._genome_file = None

        for f in self._split_files.values():
            f.close()
        self._split_files.clear()

        # Keep indices in memory (they're small)

    def __del__(self):
        """Cleanup on deletion."""
        self.close()

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.close()

    def __len__(self) -> int:
        """Return total number of sequences across all splits."""
        return sum(s.num_seqs for s in self._metadata.splits.values())

    def __getitem__(self, key: str) -> Any:
        """Allow dict-style access to metadata."""
        return getattr(self._metadata, key, None)


class HDF5Writer:
    """HDF5-based data writer for creating datasets in v2.0+ flat format.

    This class handles writing genomic datasets with:
    - Flat file layout: {split}.h5
    - Pre-encoded genome (uint8, compressed)
    - Fixed-length strings for chromosomes
    - Sorted indices for optimal I/O
    - Manifest.json creation

    Example:
        writer = HDF5Writer("output_dir")

        # Write pre-encoded genome (uint8)
        writer.write_genome(genome_1hot)

        # Write split (indices + targets in one file)
        writer.write_split("train", chrom_names, starts, ends, targets)

        # Finalize
        writer.finalize(...)
    """

    def __init__(
        self,
        output_dir: str | Path,
        seq_length: int = 131072,
        seq_depth: int = 4,
        target_length: int | dict[str, int] = 0,
        num_targets: int = 1,
    ):
        self.output_dir = Path(output_dir)
        self.seq_length = seq_length
        self.seq_depth = seq_depth
        # Store target_length as dict for multi-type support
        self._target_length = target_length if isinstance(target_length, dict) else {"default": target_length}
        self.num_targets = num_targets

        # Track what has been written
        self._splits_written: set[str] = set()
        self._chrom_lengths: dict[str, int] = {}

        # Create directory structure
        self._create_directories()

    @property
    def target_length(self) -> int | dict[str, int]:
        """Return target_length (for backward compatibility, returns int if uniform)."""
        if len(self._target_length) == 1 and "default" in self._target_length:
            return self._target_length["default"]
        return self._target_length

    @target_length.setter
    def target_length(self, value: int | dict[str, int]) -> None:
        """Set target_length (supports both int and dict formats)."""
        self._target_length = value if isinstance(value, dict) else {"default": value}

    def _create_directories(self) -> None:
        """Create output directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_chromosome(self, chrom_name: str, dna_seq: str) -> None:
        """Stream one chromosome directly to HDF5 with uint8 compression.

        This method performs one-hot encoding internally and writes immediately,
        avoiding memory accumulation of the entire genome.

        Args:
            chrom_name: Chromosome name (e.g., "chr1", "chr2")
            dna_seq: Raw DNA sequence string (ACGTN characters)
        """
        genome_file = self.output_dir / "genome.h5"

        # One-hot encode: A=0, C=1, G=2, T=3, N=0
        seq_upper = dna_seq.upper()
        seq_length = len(seq_upper)

        # Pre-allocate uint8 array (4 channels)
        seq_1hot = np.zeros((seq_length, 4), dtype=np.uint8)

        # Vectorized encoding using lookup
        for i, char in enumerate(seq_upper):
            if char == "A":
                seq_1hot[i, 0] = 1
            elif char == "C":
                seq_1hot[i, 1] = 1
            elif char == "G":
                seq_1hot[i, 2] = 1
            elif char == "T":
                seq_1hot[i, 3] = 1
            # N stays as all zeros (background)

        # Write to file in append mode
        with h5py.File(genome_file, "a") as f:
            # Set attributes on first write
            if "version" not in f.attrs:
                f.attrs["version"] = "2.0"
                f.attrs["seq_length"] = self.seq_length
                f.attrs["seq_depth"] = self.seq_depth
                f.attrs["dtype"] = "uint8"

            chrom_key = str(chrom_name)
            if chrom_key in f:
                del f[chrom_key]

            f.create_dataset(
                chrom_key,
                data=seq_1hot,
                compression="gzip",
                compression_opts=4,
            )

        self._chrom_lengths[chrom_name] = seq_length

    def write_genome(
        self,
        genome_dict: dict[str, np.ndarray],
    ) -> None:
        """Write pre-encoded genome to HDF5 with uint8 compression.

        Args:
            genome_dict: Dictionary mapping chrom_name to 1hot encoded array (length × 4)
                          e.g., {"chr1": array, "chr2": array, ...}

        Note: For large genomes, prefer using write_chromosome() in streaming mode
              to avoid memory accumulation.
        """
        genome_file = self.output_dir / "genome.h5"

        with h5py.File(genome_file, "w") as f:
            # Store metadata in file attributes
            f.attrs["version"] = "2.0"
            f.attrs["seq_length"] = self.seq_length
            f.attrs["seq_depth"] = self.seq_depth
            f.attrs["dtype"] = "uint8"

            for chrom_name, seq_1hot in genome_dict.items():
                # Convert to uint8 for 75% space savings
                if seq_1hot.dtype != np.uint8:
                    seq_1hot = (seq_1hot * 255).astype(np.uint8)

                chrom_key = str(chrom_name)
                f.create_dataset(chrom_key, data=seq_1hot, compression="gzip", compression_opts=4)
                self._chrom_lengths[chrom_name] = seq_1hot.shape[0]

    def write_split(
        self,
        split: str,
        chrom_names: list[str],
        starts: list[int],
        ends: list[int],
        targets: list[np.ndarray] | None = None,
        target_configs: list[TargetConfig] | None = None,
        chunk_size: int = 1024,
    ) -> None:
        """Write indices and targets for a split in a single flat file.

        Args:
            split: Split name (train/valid/test)
            chrom_names: List of chromosome names (e.g., "chr1", "chr2")
            starts: List of start positions
            ends: List of end positions
            targets: List of target arrays, each of shape (num_seqs, target_length_i)
            target_configs: List of TargetConfig for each target
            chunk_size: HDF5 chunk size for targets
        """
        split_file = self.output_dir / f"{split}.h5"

        num_seqs = len(chrom_names)

        # Sort by chromosome and start position for optimal I/O
        sorted_data = self._sort_by_chrom_start(chrom_names, starts, ends, targets)
        sorted_chroms = sorted_data["chroms"]
        sorted_starts = sorted_data["starts"]
        sorted_ends = sorted_data["ends"]
        sorted_targets = sorted_data["targets"]

        with h5py.File(split_file, "w") as f:
            # Write indices group
            indices_grp = f.create_group("indices")

            # Fixed-length strings (S16) for better performance
            chrom_array = np.array(sorted_chroms, dtype=object)
            # Pad strings to fixed length
            padded_chroms = np.array(
                [c.encode("utf-8").ljust(16, b"\x00") for c in chrom_array], dtype=CHROM_DTYPE
            )
            indices_grp.create_dataset("chrom", data=padded_chroms)
            indices_grp.create_dataset("start", data=np.array(sorted_starts, dtype=np.int32))
            indices_grp.create_dataset("end", data=np.array(sorted_ends, dtype=np.int32))

            # Write targets if provided (v3.0+ format: /targets/{i} as dataset)
            if sorted_targets is not None and target_configs is not None:
                targets_grp = f.create_group("targets")

                for i, (target_data, config) in enumerate(zip(sorted_targets, target_configs)):
                    # Create /targets/{i} dataset directly (not a group)
                    target_length = target_data.shape[1]

                    # Adapt chunk size to data size
                    actual_chunk_size = min(chunk_size, target_data.shape[0])

                    targets_grp.create_dataset(
                        str(i),
                        data=target_data.astype(np.float32),
                        chunks=(actual_chunk_size, target_length),
                        compression="gzip",
                        compression_opts=4,
                    )

        self._splits_written.add(split)

    def _sort_by_chrom_start(
        self,
        chrom_names: list[str],
        starts: list[int],
        ends: list[int],
        targets: list[np.ndarray] | None = None,
    ) -> dict:
        """Sort data by chromosome and start position.

        Returns:
            Dictionary with sorted chroms, starts, ends, and targets
        """
        # Create index array
        idx = np.arange(len(chrom_names))

        # Sort by chrom first, then by start
        sorted_idx = sorted(idx, key=lambda i: (chrom_names[i].lower(), starts[i]))
        sorted_idx = np.array(sorted_idx, dtype=np.int64)

        # Apply sorting
        sorted_chroms = [chrom_names[i] for i in sorted_idx]
        sorted_starts = [starts[i] for i in sorted_idx]
        sorted_ends = [ends[i] for i in sorted_idx]

        sorted_targets = None
        if targets is not None:
            # targets is list[np.ndarray], each target_data shape: (num_seqs, target_length_i)
            sorted_targets = [target_data[sorted_idx] for target_data in targets]

        return {
            "chroms": sorted_chroms,
            "starts": sorted_starts,
            "ends": sorted_ends,
            "targets": sorted_targets,
        }

    def write_coordinates(
        self,
        split: str,
        coordinates: list[tuple[str, int, int]],
        targets: list[np.ndarray] | None = None,
        target_configs: list[TargetConfig] | None = None,
    ) -> None:
        """Write genomic coordinates and targets for a split.

        Args:
            split: Split name (train/valid/test)
            coordinates: List of (chrom, start, end) tuples
            targets: List of target arrays, each of shape (num_seqs, target_length_i)
            target_configs: List of TargetConfig for each target
        """
        chrom_names = []
        starts = []
        ends = []

        for chrom, start, end in coordinates:
            chrom_names.append(chrom)
            starts.append(start)
            ends.append(end)

        self.write_split(split, chrom_names, starts, ends, targets, target_configs)

    def finalize(
        self,
        genome_name: str = "",
        target_type: str = "unknown",
        pool_width: int = 1,
        diagonal_offset: int = 0,
        target_info: list[dict] | None = None,
        target_lengths: dict[int, int] | None = None,
    ) -> DatasetMetadata:
        """Finalize dataset by creating manifest.json.

        Args:
            genome_name: Genome name
            target_type: Target type identifier
            pool_width: Pool width
            diagonal_offset: Diagonal offset
            target_info: List of target info dicts
            target_lengths: Dictionary mapping target index to target length

        Returns:
            DatasetMetadata object
        """
        import json

        # Build splits with flat file format
        splits = {}
        for split in self._splits_written:
            split_file = self.output_dir / f"{split}.h5"

            if split_file.exists():
                with h5py.File(split_file, "r") as f:
                    if "indices" in f and "chrom" in f["indices"]:
                        num_seqs = f["indices"]["chrom"].shape[0]
                    else:
                        num_seqs = 0

                    splits[split] = SplitMetadata(
                        name=split,
                        num_seqs=num_seqs,
                        split_file=f"{split}.h5",
                    )

        # Build targets
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
            version="3.0",
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
            target_lengths=target_lengths or {},
        )

        # Write manifest
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        return metadata
