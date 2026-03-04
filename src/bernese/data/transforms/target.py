# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Target transformation classes.

This module provides transforms for target preprocessing and normalization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from bernese.data.transforms.base import Sample, Transform

if TYPE_CHECKING:
    from numpy.typing import NDArray


class TargetNormalize(Transform):
    """Normalize targets using precomputed statistics.
    
    Applies z-score normalization or min-max scaling to targets.
    
    Args:
        mean: Mean values per target
        std: Standard deviation per target
        mode: Normalization mode ("zscore", "minmax", or "none")
    """
    
    def __init__(
        self,
        mean: NDArray[np.float32] | torch.Tensor | None = None,
        std: NDArray[np.float32] | torch.Tensor | None = None,
        mode: str = "zscore",
    ):
        self.mode = mode
        
        # Convert to tensors
        if mean is not None:
            if isinstance(mean, np.ndarray):
                self.mean = torch.from_numpy(mean)
            else:
                self.mean = mean
        else:
            self.mean = None
            
        if std is not None:
            if isinstance(std, np.ndarray):
                self.std = torch.from_numpy(std)
            else:
                self.std = std
        else:
            self.std = None
    
    def __call__(self, sample: Sample) -> Sample:
        """Apply normalization to targets."""
        targets = sample.targets
        
        if self.mode == "none" or self.mean is None or self.std is None:
            return sample
        
        # Reshape for broadcasting
        mean = self.mean.to(targets.device)
        std = self.std.to(targets.device)
        
        if self.mode == "zscore":
            # Z-score normalization
            if mean.ndim == 1:
                mean = mean.view(1, 1, -1)
            if std.ndim == 1:
                std = std.view(1, 1, -1)
            
            targets = (targets - mean) / (std + 1e-8)
        
        elif self.mode == "minmax":
            # Min-max to [0, 1]
            if mean.ndim == 1:
                mean = mean.view(1, 1, -1)
            if std.ndim == 1:
                # std used as (max - min)
                std = std.view(1, 1, -1)
            
            targets = (targets - mean) / (std + 1e-8)
            targets = torch.clamp(targets, 0, 1)
        
        return Sample(
            sequences=sample.sequences,
            targets=targets,
            coordinates=sample.coordinates,
            metadata=sample.metadata,
        )


class TargetClamp(Transform):
    """Clamp target values to a range.
    
    Args:
        min_val: Minimum value
        max_val: Maximum value
    """
    
    def __init__(self, min_val: float = -10.0, max_val: float = 10.0):
        self.min_val = min_val
        self.max_val = max_val
    
    def __call__(self, sample: Sample) -> Sample:
        """Apply clamping to targets."""
        targets = torch.clamp(sample.targets, self.min_val, self.max_val)
        
        return Sample(
            sequences=sample.sequences,
            targets=targets,
            coordinates=sample.coordinates,
            metadata=sample.metadata,
        )


class TargetLogTransform(Transform):
    """Apply log transform to targets.
    
    Adds a small offset before log to handle zeros.
    
    Args:
        offset: Offset to add before log (default: 1)
    """
    
    def __init__(self, offset: float = 1.0):
        self.offset = offset
    
    def __call__(self, sample: Sample) -> Sample:
        """Apply log transform to targets."""
        targets = torch.log(sample.targets + self.offset)
        
        return Sample(
            sequences=sample.sequences,
            targets=targets,
            coordinates=sample.coordinates,
            metadata=sample.metadata,
        )


class TargetUntransform(Transform):
    """Undo transformations applied during preprocessing.
    
    This is the inverse of TargetNormalize for predictions.
    
    Args:
        mean: Mean values used in normalization
        std: Std values used in normalization
        mode: Original normalization mode
    """
    
    def __init__(
        self,
        mean: NDArray[np.float32] | torch.Tensor,
        std: NDArray[np.float32] | torch.Tensor,
        mode: str = "zscore",
    ):
        self.mode = mode
        
        if isinstance(mean, np.ndarray):
            self.mean = torch.from_numpy(mean)
        else:
            self.mean = mean
            
        if isinstance(std, np.ndarray):
            self.std = torch.from_numpy(std)
        else:
            self.std = std
    
    def __call__(self, sample: Sample) -> Sample:
        """Undo normalization."""
        targets = sample.targets
        
        mean = self.mean.to(targets.device)
        std = self.std.to(targets.device)
        
        if self.mode == "zscore":
            if mean.ndim == 1:
                mean = mean.view(1, 1, -1)
            if std.ndim == 1:
                std = std.view(1, 1, -1)
            
            targets = targets * std + mean
        
        return Sample(
            sequences=sample.sequences,
            targets=targets,
            coordinates=sample.coordinates,
            metadata=sample.metadata,
        )


def create_target_transforms(
    statistics: dict | None = None,
    normalize: bool = True,
    log_transform: bool = False,
    clamp: tuple[float, float] | None = None,
) -> list[Transform]:
    """Create standard target transform pipeline.
    
    Args:
        statistics: Dict with mean, std, etc.
        normalize: Whether to apply z-score normalization
        log_transform: Whether to apply log transform
        clamp: Optional (min, max) tuple for clamping
        
    Returns:
        List of transforms
    """
    transforms = []
    
    if log_transform:
        transforms.append(TargetLogTransform())
    
    if clamp is not None:
        transforms.append(TargetClamp(min_val=clamp[0], max_val=clamp[1]))
    
    if normalize and statistics is not None:
        mean = statistics.get("mean")
        std = statistics.get("std")
        if mean is not None and std is not None:
            transforms.append(TargetNormalize(mean=mean, std=std, mode="zscore"))
    
    return transforms
