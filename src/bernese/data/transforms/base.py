# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Base classes for data transformations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from dataclasses import dataclass


@dataclass
class Sample:
    """Single sample from the dataset.

    Attributes:
        sequences: Input sequences tensor (batch, seq_depth, seq_length)
        targets: Target values tensor (batch, target_length, num_targets)
        coordinates: Optional genomic coordinates (batch,) of (chrom, start, end)
        metadata: Optional metadata dict
    """

    sequences: torch.Tensor
    targets: torch.Tensor
    coordinates: list[tuple[str, int, int]] | None = None
    metadata: dict | None = None


class Transform(ABC):
    """Abstract base class for data transforms.

    Transforms can operate on sequences, targets, or both.
    They are applied lazily at training time.
    """

    @abstractmethod
    def __call__(self, sample: Sample) -> Sample:
        """Apply transform to a sample.

        Args:
            sample: Input sample

        Returns:
            Transformed sample
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class TransformPipeline:
    """Composable pipeline of transforms.

    Transforms are applied in order.

    Args:
        transforms: List of transforms to apply
    """

    def __init__(self, transforms: list[Transform] | None = None):
        self.transforms = transforms or []

    def __call__(self, sample: Sample) -> Sample:
        """Apply all transforms in order.

        Args:
            sample: Input sample

        Returns:
            Transformed sample
        """
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def __repr__(self) -> str:
        transform_str = ", ".join(repr(t) for t in self.transforms)
        return f"TransformPipeline([{transform_str}])"

    def __len__(self) -> int:
        return len(self.transforms)

    def append(self, transform: Transform) -> "TransformPipeline":
        """Add a transform to the end of the pipeline.

        Args:
            transform: Transform to add

        Returns:
            Self for chaining
        """
        self.transforms.append(transform)
        return self

    def prepend(self, transform: Transform) -> "TransformPipeline":
        """Add a transform to the beginning of the pipeline.

        Args:
            transform: Transform to add

        Returns:
            Self for chaining
        """
        self.transforms.insert(0, transform)
        return self

    @classmethod
    def compose(cls, *transforms: Transform) -> "TransformPipeline":
        """Create pipeline from multiple transforms.

        Args:
            transforms: Transforms to compose

        Returns:
            New TransformPipeline
        """
        return cls(list(transforms))
