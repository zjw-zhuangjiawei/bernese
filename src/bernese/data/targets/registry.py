# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Target processor registry system.

This module provides a registry for target processors that handle
different target data types (Hi-C, BigWig, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np


@dataclass
class TargetMetadata:
    """Metadata for a target track."""

    name: str
    file: str
    target_type: str
    clip: float | None = None
    index: int = 0
    statistics: dict[str, Any] = field(default_factory=dict)


class TargetProcessor(ABC):
    """Abstract base class for target processors."""

    @abstractmethod
    def process(
        self,
        input_file: str,
        regions: list[tuple[str, int, int]],
        **kwargs,
    ) -> np.ndarray:
        """Process target data for given genomic regions.

        Args:
            input_file: Path to input data file
            regions: List of (chrom, start, end) tuples
            **kwargs: Additional processor-specific options

        Returns:
            Array of shape (num_regions, target_length, ...)
        """
        ...

    @abstractmethod
    def compute_statistics(self, data: np.ndarray) -> dict[str, Any]:
        """Compute statistics for target data.

        Args:
            data: Target array

        Returns:
            Dictionary of statistics
        """
        ...

    @property
    @abstractmethod
    def target_type(self) -> str:
        """Return the target type identifier."""
        ...


class TargetProcessorRegistry:
    """Registry for target processors.

    Processors are registered by target type and can be retrieved
    for processing specific target data.
    """

    _processors: dict[str, type[TargetProcessor]] = {}
    _decorator_func: Callable[[type[TargetProcessor]], type[TargetProcessor]] | None = None

    @classmethod
    def register(cls, target_type: str) -> Callable[[type[TargetProcessor]], type[TargetProcessor]]:
        """Decorator to register a target processor.

        Args:
            target_type: Target type identifier

        Returns:
            Decorator function
        """

        def decorator(processor_cls: type[TargetProcessor]) -> type[TargetProcessor]:
            cls._processors[target_type] = processor_cls
            return processor_cls

        return decorator

    @classmethod
    def get(cls, target_type: str) -> type[TargetProcessor]:
        """Get processor class for target type.

        Args:
            target_type: Target type identifier

        Returns:
            Processor class

        Raises:
            ValueError: If target type not registered
        """
        if target_type not in cls._processors:
            raise ValueError(
                f"Unknown target type: {target_type}. Available: {list(cls._processors.keys())}"
            )
        return cls._processors[target_type]

    @classmethod
    def create(cls, target_type: str, **kwargs) -> TargetProcessor:
        """Create a target processor instance.

        Args:
            target_type: Target type identifier
            **kwargs: Arguments to pass to processor constructor

        Returns:
            TargetProcessor instance
        """
        processor_cls = cls.get(target_type)
        return processor_cls(**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        """Get list of available target types.

        Returns:
            List of target type identifiers
        """
        return list(cls._processors.keys())


def target_processor(target_type: str) -> Callable[[type[TargetProcessor]], type[TargetProcessor]]:
    """Decorator to register a target processor.

    This is a convenience alias for TargetProcessorRegistry.register.

    Args:
        target_type: Target type identifier

    Returns:
        Decorator function
    """
    return TargetProcessorRegistry.register(target_type)
