# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Abstract data backend protocol and metadata classes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class DataWriter(Protocol):
    """Abstract protocol for data storage writers.

    This defines the interface that all data writers must implement.
    Implementations can support various storage formats (HDF5, Zarr, etc.).
    """

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
        ...

    def write_sequences(
        self,
        split: str,
        sequences: np.ndarray,
    ) -> None:
        """Write pre-extracted sequences for a split.

        Args:
            split: Split name
            sequences: Array of shape (num_seqs, seq_depth, seq_length)
        """
        ...

    def write_targets(
        self,
        split: str,
        targets: np.ndarray,
    ) -> None:
        """Write targets for a split.

        Args:
            split: Split name
            targets: Array of shape (num_seqs, target_length, num_targets)
        """
        ...

    def finalize(
        self,
        genome_name: str = "",
        fasta_path: str = "",
        target_type: str = "unknown",
        pool_width: int = 1,
        diagonal_offset: int = 0,
        target_info: list[dict] | None = None,
    ) -> "DatasetMetadata":
        """Finalize dataset by creating manifest.

        Args:
            genome_name: Genome name
            fasta_path: Path to FASTA file
            target_type: Target type identifier
            pool_width: Pool width
            diagonal_offset: Diagonal offset
            target_info: List of target info dicts

        Returns:
            DatasetMetadata object
        """
        ...


@dataclass
class SplitMetadata:
    """Metadata for a dataset split (train/valid/test)."""

    name: str
    num_seqs: int
    coordinate_file: str
    target_file: str | None = None


@dataclass
class GenomeInfo:
    """Genome reference information."""

    name: str


@dataclass
class TargetInfo:
    """Target track information."""

    name: str
    clip: float | None = None
    index: int = 0


@dataclass
class DatasetMetadata:
    """Complete dataset metadata."""

    version: str = "2.0"
    name: str = ""
    created: str = ""
    genome: GenomeInfo | None = None
    seq_length: int = 0
    seq_depth: int = 4
    target_length: int = 0
    num_targets: int = 0
    target_type: str = "unknown"
    pool_width: int = 1
    diagonal_offset: int = 0
    splits: dict[str, SplitMetadata] = field(default_factory=dict)
    targets: list[TargetInfo] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_manifest(cls, manifest_path: Path) -> "DatasetMetadata":
        """Load metadata from manifest.json.
        
        Relative paths in the manifest are resolved from the manifest's directory.
        """
        # Get manifest directory for resolving relative paths
        manifest_dir = Path(manifest_path).parent

        def resolve_path(path: str | None) -> str:
            """Resolve relative paths from manifest directory."""
            if not path:
                return ""
            p = Path(path)
            if p.is_absolute():
                return path
            # Resolve relative to manifest directory
            resolved = manifest_dir / p
            return str(resolved.resolve())

        with open(manifest_path) as f:
            data = json.load(f)

        # Parse genome
        genome = None
        if "genome" in data:
            g = data["genome"]
            genome = GenomeInfo(
                name=g.get("name", ""),
            )

        # Parse splits
        splits = {}
        if "sequences" in data and "splits" in data["sequences"]:
            for name, info in data["sequences"]["splits"].items():
                splits[name] = SplitMetadata(
                    name=name,
                    num_seqs=info.get("num_seqs", 0),
                    coordinate_file=resolve_path(info.get("coordinate_file", "")),
                    target_file=resolve_path(info.get("target_file")),
                )

        # Parse targets
        targets = []
        if "targets" in data and "info" in data["targets"]:
            for i, info in enumerate(data["targets"]["info"]):
                targets.append(
                    TargetInfo(
                        name=info.get("name", ""),
                        clip=info.get("clip"),
                        index=i,
                    )
                )

        return cls(
            version=data.get("version", "2.0"),
            name=data.get("name", ""),
            created=data.get("created", ""),
            genome=genome,
            seq_length=data.get("sequences", {}).get("seq_length", 0),
            seq_depth=data.get("sequences", {}).get("seq_depth", 4),
            target_length=data.get("targets", {}).get("target_length", 0),
            num_targets=data.get("targets", {}).get("num_targets", 0),
            target_type=data.get("targets", {}).get("target_type", "unknown"),
            pool_width=data.get("targets", {}).get("pool_width", 1),
            diagonal_offset=data.get("targets", {}).get("diagonal_offset", 0),
            splits=splits,
            targets=targets,
            statistics=data.get("statistics", {}),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "version": self.version,
            "name": self.name,
            "created": self.created,
            "sequences": {
                "seq_length": self.seq_length,
                "seq_depth": self.seq_depth,
                "splits": {},
            },
            "targets": {
                "target_type": self.target_type,
                "num_targets": self.num_targets,
                "target_length": self.target_length,
                "pool_width": self.pool_width,
                "diagonal_offset": self.diagonal_offset,
                "info": [],
            },
            "statistics": self.statistics,
        }

        if self.genome:
            result["genome"] = {
                "name": self.genome.name,
            }

        for name, split in self.splits.items():
            result["sequences"]["splits"][name] = {
                "num_seqs": split.num_seqs,
                "coordinate_file": split.coordinate_file,
                "target_file": split.target_file,
            }

        for target in self.targets:
            info = {"name": target.name}
            if target.clip is not None:
                info["clip"] = target.clip
            result["targets"]["info"].append(info)

        return result


@runtime_checkable
class DataBackend(Protocol):
    """Abstract protocol for data storage backends.

    This defines the interface that all data backends must implement.
    Implementations can support various storage formats (HDF5, Zarr, etc.)
    and loading strategies (pre-extracted, lazy, etc.).
    """

    @property
    def metadata(self) -> DatasetMetadata:
        """Return dataset metadata."""
        ...

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
            Tensor of shape (batch, seq_depth, seq_length)
        """
        ...

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
        ...

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
        ...
