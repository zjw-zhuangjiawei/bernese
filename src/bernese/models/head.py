# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Head builder for SeqNN models.

This module provides the HeadBuilder class for constructing prediction heads
from configuration dictionaries. Heads take the trunk output and produce
final predictions (either 1D or 2D contact maps).
"""

from typing import Any

import torch.nn as nn

from bernese.models.blocks import create_block, Final
from bernese.models.geometry import compute_head_output_shape


class HeadBuilder:
    """Builds prediction heads from configurations.

    A head takes the trunk output and produces final predictions.
    Multiple heads can be used for multi-task learning.

    Args:
        trunk_output_channels: Number of channels from trunk output
        trunk_output_length: Sequence length from trunk output
        num_targets: Number of prediction targets
    """

    def __init__(
        self,
        trunk_output_channels: int,
        trunk_output_length: int,
        num_targets: int = 1,
    ):
        """Initialize the head builder.

        Args:
            trunk_output_channels: Number of channels from trunk
            trunk_output_length: Sequence length from trunk
            num_targets: Number of prediction targets
        """
        self.trunk_output_channels = trunk_output_channels
        self.trunk_output_length = trunk_output_length
        self.num_targets = num_targets

    def build(self, config: list[dict[str, Any]]) -> nn.Module:
        """Build a prediction head from configuration.

        Args:
            config: List of block configuration dictionaries for this head

        Returns:
            Sequential head module
        """
        # Pre-compute the flattened size for 2D heads
        precomputed_flat_size = self._precompute_flat_size(config)

        # Build blocks
        layers = []
        in_channels = self.trunk_output_channels

        for block_config in config:
            block_name = block_config.get("name", "conv_block")

            if block_name == "final":
                # Final layer needs special handling
                block = self._build_final_block(block_config, precomputed_flat_size)
                layers.append(block)
                in_channels = block_config.get(
                    "out_units",
                    block_config.get("units", self.num_targets)
                )
            else:
                # Regular block
                block = self._build_block(block_config, in_channels)
                layers.append(block)
                in_channels = self._update_channels(in_channels, block_config, block_name)

        return nn.Sequential(*layers)

    def compute_output_shape(
        self, config: list[dict[str, Any]]
    ) -> tuple[int, int]:
        """Compute output shape without building the head.

        Args:
            config: List of block configurations

        Returns:
            Tuple of (output_length, output_channels)
        """
        return compute_head_output_shape(
            self.trunk_output_channels,
            self.trunk_output_length,
            config,
            self.num_targets,
        )

    def _precompute_flat_size(self, config: list[dict[str, Any]]) -> int | None:
        """Pre-compute the flattened size for 2D heads.

        Args:
            config: List of block configurations

        Returns:
            Pre-computed flat size or None for 1D heads
        """
        # Check if this is a 2D head
        has_2d = any(bc.get("name") == "one_to_two" for bc in config)
        if not has_2d:
            return None

        output_length, output_channels = self.compute_output_shape(config)
        return output_length * output_channels

    def _build_final_block(
        self, config: dict[str, Any], precomputed_flat_size: int | None
    ) -> nn.Module:
        """Build a final prediction block.

        Args:
            config: Block configuration
            precomputed_flat_size: Pre-computed flat size for 2D heads

        Returns:
            Final block module
        """
        params = {k: v for k, v in config.items() if k != "name"}

        # Use precomputed_flat_size to determine if 2D head
        if "flatten" not in params and precomputed_flat_size is not None:
            params["flatten"] = True

        if params.get("flatten", False) and precomputed_flat_size is not None:
            params["in_channels"] = precomputed_flat_size
        else:
            params["in_channels"] = self.trunk_output_channels

        # Map 'units' to 'out_units'
        if "units" in params and "out_units" not in params:
            params["out_units"] = params.pop("units")
        elif "units" not in params:
            params["out_units"] = self.num_targets

        return Final(**params)

    def _build_block(self, config: dict[str, Any], in_channels: int) -> nn.Module:
        """Build a regular block.

        Args:
            config: Block configuration
            in_channels: Number of input channels

        Returns:
            Block module
        """
        params = {k: v for k, v in config.items() if k != "name"}
        params["in_channels"] = in_channels

        # Map 'filters' to 'out_channels'
        if "filters" in params and "out_channels" not in params:
            params["out_channels"] = params.pop("filters")

        return create_block(config.get("name", "conv_block"), **params)

    def _update_channels(
        self, in_channels: int, config: dict[str, Any], block_name: str
    ) -> int:
        """Update channel count based on block type.

        Args:
            in_channels: Current channel count
            config: Block configuration
            block_name: Name of the block

        Returns:
            Updated channel count
        """
        if block_name == "concat_dist_2d":
            num_features = config.get("num_features", 5)
            return in_channels + num_features
        elif block_name == "cropping_2d":
            # Cropping doesn't change channels
            return in_channels
        elif block_name == "upper_tri":
            # Upper triangular extraction doesn't change channels
            return in_channels
        elif block_name in ["one_to_two", "symmetrize_2d"]:
            # 2D transformation doesn't change channels
            return in_channels
        else:
            # Regular block - use filters
            return config.get("filters", in_channels)
