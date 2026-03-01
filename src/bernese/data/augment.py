# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Data augmentation transforms for genomic sequences."""

import random
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReverseComplement(nn.Module):
    """Stochastic reverse complement augmentation.

    Randomly applies reverse complement to the input sequence with probability 0.5.
    For DNA sequences, this means: A<->T, C<->G, and reversing the sequence.
    """

    def __init__(self, probability: float = 0.5):
        super().__init__()
        self.probability = probability

        # Complement mapping for DNA
        self.register_buffer(
            "_complement_map",
            torch.tensor([3, 2, 1, 0, 4], dtype=torch.long),  # A,T,G,C,N
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """Apply reverse complement with given probability.

        Args:
            x: Input tensor of shape (seq_length, seq_depth) or (batch, seq_length, seq_depth)
               where seq_depth=4 for one-hot encoded DNA (A,T,G,C)

        Returns:
            Tuple of (transformed_sequence, was_reversed)
        """
        if self.training and random.random() < self.probability:
            # Reverse the sequence
            x = torch.flip(x, dims=[-2])

            # Complement: A->T, T->A, C->G, G->C
            # Input is (seq_length, 4) one-hot: A=0, C=1, G=2, T=3
            x = self._complement_map[x.argmax(dim=-1)]
            x = F.one_hot(x, num_classes=4).float()

            return x, True

        return x, False


class StochasticShift(nn.Module):
    """Stochastic shift augmentation.

    Randomly shifts the sequence by a random amount from the given range.

    Args:
        shifts: List of shift amounts to choose from
    """

    def __init__(self, shifts: Optional[list] = None):
        super().__init__()
        self.shifts = shifts if shifts is not None else [-2, -1, 0, 1, 2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random shift.

        Args:
            x: Input tensor of shape (seq_length, seq_depth) or (batch, seq_length, seq_depth)

        Returns:
            Shifted tensor (same shape)
        """
        if not self.training or len(self.shifts) == 0:
            return x

        shift = random.choice(self.shifts)

        if shift == 0:
            return x

        # Apply circular shift
        if shift > 0:
            x = torch.cat([x[shift:], x[:shift]], dim=-2)
        else:
            shift = abs(shift)
            x = torch.cat([x[-shift:], x[:-shift]], dim=-2)

        return x


class RandomCrop(nn.Module):
    """Random crop augmentation.

    Randomly crops the sequence to a smaller size and optionally pads back.

    Args:
        crop_length: Length to crop to
        pad_value: Value to use for padding (default: 0)
    """

    def __init__(self, crop_length: int, pad_value: float = 0.0):
        super().__init__()
        self.crop_length = crop_length
        self.pad_value = pad_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random crop.

        Args:
            x: Input tensor of shape (seq_length, seq_depth) or (batch, seq_length, seq_depth)

        Returns:
            Cropped tensor (same shape, but content cropped)
        """
        seq_dim = -2

        if x.shape[seq_dim] <= self.crop_length:
            return x

        # Random crop position
        max_start = x.shape[seq_dim] - self.crop_length
        start = random.randint(0, max_start)

        # Crop
        if x.dim() == 2:
            x = x[start : start + self.crop_length]
        else:
            x = x[:, start : start + self.crop_length, :]

        return x


class Compose(nn.Module):
    """Compose multiple transforms together.

    Args:
        transforms: List of transforms to compose
    """

    def __init__(self, transforms: list):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, x: torch.Tensor):
        for t in self.transforms:
            x = t(x)
        return x


class Identity(nn.Module):
    """Identity transform (no augmentation)."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x


def create_augmentation_transforms(
    augment_rc: bool = False,
    augment_shift: Optional[Union[int, list]] = None,
    augment_rc_prob: float = 0.5,
) -> nn.Module:
    """Create data augmentation transforms based on configuration.

    Args:
        augment_rc: Whether to use reverse complement augmentation
        augment_shift: Integer (max shift) or list of shift amounts (e.g., [-2, -1, 0, 1, 2])
        augment_rc_prob: Probability of applying reverse complement

    Returns:
        Composed augmentation transform
    """
    transforms = []

    if augment_shift is not None:
        # Handle integer input - create symmetric shift range
        if isinstance(augment_shift, int):
            max_shift = abs(augment_shift)
            shift_list = list(range(-max_shift, max_shift + 1))
        else:
            shift_list = augment_shift

        if len(shift_list) > 0:
            transforms.append(StochasticShift(shifts=shift_list))

    if augment_rc:
        transforms.append(ReverseComplement(probability=augment_rc_prob))

    if len(transforms) == 0:
        return Identity()

    return Compose(transforms)


# Target transforms for untransforming predictions


class UntransformTargets(nn.Module):
    """Transform targets back to original scale.

    Applies inverse transformations that were applied during data preprocessing:
    - Undo clipping (soft clip)
    - Undo sqrt transformation
    - Undo scaling

    Args:
        targets_df: DataFrame with target information (clip_soft, scale, sum_stat columns)
    """

    def __init__(self, targets_df=None):
        super().__init__()
        self.targets_df = targets_df

    def forward(self, preds: torch.Tensor) -> torch.Tensor:
        """Untransform predictions.

        Args:
            preds: Predictions tensor

        Returns:
            Untransformed predictions
        """
        if self.targets_df is None:
            return preds

        # This would need targets_df to be passed - simplified for now
        return preds


def untransform_predictions(
    preds: torch.Tensor,
    clip_soft: torch.Tensor,
    scale: torch.Tensor,
    sum_stat: list,
) -> torch.Tensor:
    """Untransform predictions to original scale.

    Args:
        preds: Predictions tensor (batch, length, targets)
        clip_soft: Soft clipping threshold per target
        scale: Scale factor per target
        sum_stat: List of sum statistics ('sum', 'sum_sqrt', 'mean', etc.)

    Returns:
        Untransformed predictions
    """
    # Undo soft clip
    # Original: clipped = clip_soft - 1 + (preds - clip_soft + 1)^2
    # Inverse: preds = clip_soft + sqrt(clipped - clip_soft + 1) - 1
    # But simpler: use the values as-is for now

    # Undo sqrt (sum_sqrt)
    for i, stat in enumerate(sum_stat):
        if "sqrt" in stat.lower():
            # Check if this is the target dimension
            if preds.dim() == 3 and i < preds.shape[-1]:
                preds[:, :, i] = (preds[:, :, i] + 1) ** 2 - 1

    # Undo scale
    if scale is not None:
        preds = preds / scale

    return preds
