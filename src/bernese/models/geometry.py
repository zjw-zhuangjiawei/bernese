# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Model geometry tracking for SeqNN.

This module provides the ModelGeometry dataclass to track input/output shapes,
strides, and other geometric properties of the model.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelGeometry:
    """Tracks geometric properties of a SeqNN model.

    Attributes:
        input_length: Original input sequence length
        input_channels: Number of input channels (4 for DNA)
        trunk_output_channels: Number of output channels from trunk
        trunk_output_length: Output sequence length from trunk
        head_output_shapes: List of (batch, length, channels) tuples per head
        strides: List of effective strides per head
        target_lengths: List of target lengths per head
        target_crops: List of crop amounts per head
    """

    input_length: int = 1344
    input_channels: int = 4
    trunk_output_channels: int = 0
    trunk_output_length: int = 0
    head_output_shapes: list[tuple[int, int, int]] = field(default_factory=list)
    strides: list[int] = field(default_factory=list)
    target_lengths: list[int] = field(default_factory=list)
    target_crops: list[int] = field(default_factory=list)

    def num_heads(self) -> int:
        """Return number of prediction heads."""
        return len(self.head_output_shapes)

    def get_head_shape(self, head_i: int) -> tuple[int, int, int]:
        """Get output shape for a specific head.

        Args:
            head_i: Head index

        Returns:
            Tuple of (batch, length, channels)
        """
        if head_i >= self.num_heads():
            raise IndexError(f"Head index {head_i} out of range [0, {self.num_heads()})")
        return self.head_output_shapes[head_i]

    def get_head_output_channels(self, head_i: int) -> int:
        """Get output channels for a specific head."""
        _, _, channels = self.get_head_shape(head_i)
        return channels

    def get_head_output_length(self, head_i: int) -> int:
        """Get output length for a specific head."""
        _, length, _ = self.get_head_shape(head_i)
        return length


def compute_trunk_output_length(
    seq_length: int,
    trunk_config: list[dict],
) -> int:
    """Compute the output length after the trunk.

    Args:
        seq_length: Input sequence length
        trunk_config: List of trunk block configurations

    Returns:
        Output sequence length after trunk
    """
    current_length = seq_length

    for block_config in trunk_config:
        name = block_config.get("name", "")
        pool_size = block_config.get("pool_size", 1)
        stride = block_config.get("stride", 1)

        # Pooling affects length
        if pool_size > 1:
            if "tower" in name:
                repeat = block_config.get("repeat", 1)
                current_length = current_length // (pool_size**repeat)
            else:
                current_length = current_length // pool_size

        # Striding affects length
        if stride > 1:
            current_length = current_length // stride

    return current_length


def compute_trunk_output_channels(
    seq_depth: int,
    trunk_config: list[dict],
) -> int:
    """Compute the output channels from the trunk.

    Args:
        seq_depth: Input sequence depth (channels)
        trunk_config: List of trunk block configurations

    Returns:
        Output channels from trunk
    """
    import math

    current_channels = seq_depth

    for block_config in trunk_config:
        block_name = block_config.get("name", "conv_block")

        if block_name in ["conv_tower", "conv_tower_nac", "res_tower"]:
            filters_init = block_config.get("filters_init", current_channels)
            filters_end = block_config.get("filters_end", filters_init * 8)
            repeat = block_config.get("repeat", 4)

            if block_name == "res_tower":
                current_channels = filters_end
            else:
                filters_mult = block_config.get(
                    "filters_mult",
                    math.exp(math.log(filters_end / filters_init) / (repeat - 1))
                    if repeat > 1
                    else 1,
                )
                current_channels = int(filters_init * (filters_mult ** (repeat - 1)))
        else:
            current_channels = block_config.get("filters", current_channels)

    return current_channels


def compute_head_output_shape(
    trunk_output_channels: int,
    trunk_output_length: int,
    head_config: list[dict],
    num_targets: int = 1,
) -> tuple[int, int]:
    """Compute the output shape for a head without building it.

    Args:
        trunk_output_channels: Number of channels from trunk
        trunk_output_length: Sequence length from trunk
        head_config: List of block configurations for the head
        num_targets: Number of prediction targets

    Returns:
        Tuple of (output_length, output_channels)
    """
    in_channels = trunk_output_channels
    current_length = trunk_output_length

    # Check if this is a 2D head (has one_to_two)
    has_2d = any(bc.get("name") == "one_to_two" for bc in head_config)

    # Find special blocks
    cropping_2d_idx = None
    upper_tri_idx = None
    for i, bc in enumerate(head_config):
        name = bc.get("name", "")
        if name == "cropping_2d":
            cropping_2d_idx = i
        elif name == "upper_tri":
            upper_tri_idx = i

    # For 2D heads, compute upper_tri length
    if has_2d:
        # Determine the length that upper_tri operates on
        upper_tri_length = current_length
        if cropping_2d_idx is not None and upper_tri_idx is not None:
            if cropping_2d_idx < upper_tri_idx:
                for bc in head_config:
                    if bc.get("name") == "cropping_2d":
                        crop = bc.get("cropping", 0)
                        upper_tri_length = max(1, upper_tri_length - 2 * crop)

        # Get diagonal offset
        diagonal_offset = 0
        for bc in head_config:
            if bc.get("name") == "upper_tri":
                diagonal_offset = bc.get("diagonal_offset", 0)
                break

        # Compute triu_count
        effective_length = upper_tri_length - diagonal_offset
        current_length = effective_length * (effective_length + 1) // 2

        # Track channels through 2D blocks
        for bc in head_config:
            bc_name = bc.get("name", "")
            if bc_name == "concat_dist_2d":
                num_features = bc.get("num_features", 5)
                in_channels = in_channels + num_features
            elif bc_name == "conv_block_2d":
                in_channels = bc.get("filters", in_channels)
            elif bc_name == "dilated_residual_2d":
                in_channels = bc.get("filters", in_channels)
    else:
        # 1D head - track through blocks
        for bc in head_config:
            bc_name = bc.get("name", "")
            if bc_name == "final":
                # Final block determines output
                out_units = bc.get("out_units", bc.get("units", num_targets))
                in_channels = out_units
            elif bc_name == "cropping_2d":
                # 1D cropping
                cropping = bc.get("cropping", 0)
                current_length = max(1, current_length - 2 * cropping)
            elif bc_name not in ["one_to_two", "upper_tri", "concat_dist_2d"]:
                # Regular 1D block
                in_channels = bc.get("filters", in_channels)
                # Check for pooling in nested configs
                if "pool_size" in bc and bc.get("pool_size", 1) > 1:
                    current_length = current_length // bc.get("pool_size", 1)

    return current_length, in_channels
