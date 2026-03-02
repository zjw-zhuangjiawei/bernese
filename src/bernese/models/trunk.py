# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Trunk builder for SeqNN models.

This module provides the TrunkBuilder class for constructing the trunk (backbone)
of the SeqNN model from configuration dictionaries.
"""

import math
from typing import Any

import torch.nn as nn

from bernese.models.blocks import create_block
from bernese.models.geometry import compute_trunk_output_channels, compute_trunk_output_length


class TrunkBuilder:
    """Builds the trunk (backbone) of a SeqNN model.

    The trunk is a shared feature extraction backbone that processes
    input sequences through a series of convolutional blocks.

    Args:
        seq_length: Input sequence length
        seq_depth: Number of input channels (4 for DNA)
    """

    def __init__(self, seq_length: int, seq_depth: int):
        """Initialize the trunk builder.

        Args:
            seq_length: Input sequence length
            seq_depth: Number of input channels
        """
        self.seq_length = seq_length
        self.seq_depth = seq_depth
        self._output_channels: int = 0
        self._output_length: int = 0

    @property
    def output_channels(self) -> int:
        """Return the number of output channels from the trunk."""
        return self._output_channels

    @property
    def output_length(self) -> int:
        """Return the output sequence length from the trunk."""
        return self._output_length

    def build(self, config: list[dict[str, Any]]) -> nn.Module:
        """Build the trunk from configuration.

        Args:
            config: List of block configuration dictionaries

        Returns:
            Sequential trunk module

        Raises:
            ValueError: If block configuration is invalid
        """
        layers = []
        current_channels = self.seq_depth
        current_length = self.seq_length

        for block_config in config:
            block_name = block_config.get("name", "conv_block")

            # Handle tower blocks - they contain multiple sub-blocks
            if block_name in ["conv_tower", "conv_tower_nac", "res_tower"]:
                tower = self._build_block(block_config, current_channels)
                layers.append(tower)

                # Update channels based on tower output
                if hasattr(tower, "reprs") and tower.reprs:
                    current_channels = tower.reprs[-1]
                elif hasattr(tower, "blocks") and len(tower.blocks) > 0:
                    current_channels = self._get_block_output_channels(tower, current_channels)
            else:
                # Regular single block
                block = self._build_block(block_config, current_channels)
                layers.append(block)
                current_channels = block_config.get("filters", current_channels)

            # Track length changes from pooling/striding
            if "pool_size" in block_config and block_config["pool_size"] > 1:
                current_length = current_length // block_config["pool_size"]
            if "stride" in block_config and block_config["stride"] > 1:
                current_length = current_length // block_config["stride"]

        # Store computed geometry
        self._output_channels = compute_trunk_output_channels(self.seq_depth, config)
        self._output_length = compute_trunk_output_length(self.seq_length, config)

        return nn.Sequential(*layers)

    def _build_block(
        self, config: dict[str, Any], in_channels: int
    ) -> nn.Module:
        """Build a single block from configuration.

        Args:
            config: Block configuration dictionary
            in_channels: Number of input channels

        Returns:
            Block module
        """
        # Extract common parameters, excluding 'name'
        params = {k: v for k, v in config.items() if k != "name"}
        params["in_channels"] = in_channels

        # Map 'filters' to 'out_channels' for compatibility
        if "filters" in params and "out_channels" not in params:
            params["out_channels"] = params.pop("filters")

        return create_block(config.get("name", "conv_block"), **params)

    def _get_block_output_channels(self, block: nn.Module, in_channels: int) -> int:
        """Get output channels from a block.

        Args:
            block: Block module
            in_channels: Fallback input channels

        Returns:
            Number of output channels
        """
        if hasattr(block, "out_channels"):
            return block.out_channels
        elif hasattr(block, "conv_layers") and len(block.conv_layers) > 0:
            for layer in block.conv_layers:
                if isinstance(layer, nn.Conv1d):
                    return layer.out_channels
        return in_channels
