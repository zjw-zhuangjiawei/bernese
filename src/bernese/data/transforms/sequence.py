# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Sequence transformation classes.

This module provides transforms for DNA sequence augmentation
and preprocessing.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import torch

from bernese.data.transforms.base import Sample, Transform

if TYPE_CHECKING:
    from typing import Sequence


class RandomShift(Transform):
    """Randomly shift sequences by a random offset.
    
    Shifts the sequence left or right by a random amount,
    wrapping around (circular shift).
    
    Args:
        max_shift: Maximum shift amount in bases (both directions)
        pad_value: Value to use for padding when shift exceeds sequence
    """
    
    def __init__(self, max_shift: int = 128, pad_value: float = 0.0):
        self.max_shift = max_shift
        self.pad_value = pad_value
    
    def __call__(self, sample: Sample) -> Sample:
        """Apply random shift to sequences."""
        shift = random.randint(-self.max_shift, self.max_shift)
        
        if shift == 0:
            return sample
        
        seq = sample.sequences
        
        if shift > 0:
            # Shift right - wrap around
            shifted = torch.zeros_like(seq)
            shifted[:, :, shift:] = seq[:, :, :-shift]
            shifted[:, :, :shift] = self.pad_value
        else:
            # Shift left - wrap around
            shifted = torch.zeros_like(seq)
            shifted[:, :, :shift] = seq[:, :, -shift:]
            shifted[:, :, shift:] = self.pad_value
        
        return Sample(
            sequences=shifted,
            targets=sample.targets,
            coordinates=sample.coordinates,
            metadata=sample.metadata,
        )


class ReverseComplement(Transform):
    """Randomly apply reverse complement to sequences.
    
    With probability 0.5, applies reverse complement to the sequence.
    Also reverses the targets (for symmetric targets like Hi-C).
    
    Args:
        probability: Probability of applying reverse complement
        reverse_targets: Whether to also reverse target tracks
    """
    
    def __init__(self, probability: float = 0.5, reverse_targets: bool = True):
        self.probability = probability
        self.reverse_targets = reverse_targets
    
    def __call__(self, sample: Sample) -> Sample:
        """Apply reverse complement with given probability."""
        if random.random() > self.probability:
            return sample
        
        seq = sample.sequences
        
        # Reverse
        seq_rc = torch.flip(seq, dims=[-1])
        # Complement: A<->T, C<->G
        # (0,1,2,3) -> (3,2,1,0)
        seq_rc = torch.flip(seq_rc, dims=[1])
        
        targets = sample.targets
        if self.reverse_targets and targets.numel() > 0:
            targets = torch.flip(targets, dims=[1])
        
        return Sample(
            sequences=seq_rc,
            targets=targets,
            coordinates=sample.coordinates,
            metadata=sample.metadata,
        )


class RandomCrop(Transform):
    """Randomly crop sequences from both ends.
    
    Crops a random amount from each end of the sequence.
    
    Args:
        max_crop: Maximum crop amount from each end
    """
    
    def __init__(self, max_crop: int = 256):
        self.max_crop = max_crop
    
    def __call__(self, sample: Sample) -> Sample:
        """Apply random crop to sequences."""
        crop = random.randint(0, self.max_crop)
        
        if crop == 0:
            return sample
        
        seq = sample.sequences
        targets = sample.targets
        
        # Crop from both ends
        seq = seq[:, :, crop:-crop] if crop > 0 else seq
        targets = targets[:, crop:-crop, :] if crop > 0 else targets
        
        return Sample(
            sequences=seq,
            targets=targets,
            coordinates=sample.coordinates,
            metadata=sample.metadata,
        )


class CenterCrop(Transform):
    """Crop sequences to a smaller size from the center.
    
    Args:
        target_length: Desired length after cropping
    """
    
    def __init__(self, target_length: int):
        self.target_length = target_length
    
    def __call__(self, sample: Sample) -> Sample:
        """Apply center crop to sequences."""
        seq = sample.sequences
        targets = sample.targets
        
        seq_len = seq.shape[-1]
        
        if seq_len <= self.target_length:
            return sample
        
        # Calculate crop amounts
        start = (seq_len - self.target_length) // 2
        end = start + self.target_length
        
        seq = seq[:, :, start:end]
        
        if targets.numel() > 0:
            tgt_len = targets.shape[1]
            tgt_start = (tgt_len - self.target_length) // 2
            targets = targets[:, tgt_start:tgt_start + self.target_length, :]
        
        return Sample(
            sequences=seq,
            targets=targets,
            coordinates=sample.coordinates,
            metadata=sample.metadata,
        )


class SequenceInvert(Transform):
    """Randomly invert the sequence (swap A/T, C/G).
    
    With probability 0.5, inverts the nucleotide encoding.
    
    Args:
        probability: Probability of applying inversion
    """
    
    def __init__(self, probability: float = 0.5):
        self.probability = probability
    
    def __call__(self, sample: Sample) -> Sample:
        """Apply sequence inversion."""
        if random.random() > self.probability:
            return sample
        
        # Swap A<->T and C<->G
        # (0,1,2,3) -> (3,2,1,0)
        seq = torch.flip(sample.sequences, dims=[1])
        
        return Sample(
            sequences=seq,
            targets=sample.targets,
            coordinates=sample.coordinates,
            metadata=sample.metadata,
        )


class Compose(Transform):
    """Compose multiple transforms into one.
    
    Args:
        transforms: List of transforms to apply in order
    """
    
    def __init__(self, transforms: list[Transform]):
        self.transforms = transforms
    
    def __call__(self, sample: Sample) -> Sample:
        """Apply all transforms in order."""
        for transform in self.transforms:
            sample = transform(sample)
        return sample


def create_augmentation_transforms(
    augmentation: str | None = None,
    rc_prob: float = 0.5,
    shift_max: int = 128,
) -> TransformPipeline:
    """Create standard augmentation pipeline.
    
    Args:
        augmentation: Augmentation type ("none", "standard", "full")
        rc_prob: Reverse complement probability
        shift_max: Maximum shift amount
        
    Returns:
        TransformPipeline with augmentations
    """
    if augmentation is None or augmentation == "none":
        return TransformPipeline()
    
    transforms = []
    
    if augmentation in ("standard", "full"):
        transforms.append(ReverseComplement(probability=rc_prob))
    
    if augmentation == "full":
        transforms.append(RandomShift(max_shift=shift_max))
    
    return TransformPipeline(transforms)
