# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Main SeqNN model class for regulatory genomics predictions.

This module provides a PyTorch implementation of the Sequence Neural Network (SeqNN)
model for regulatory activity prediction, migrated from the TensorFlow baskerville
implementation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from bernese.models import blocks
from bernese.types import ModelConfig, BlockConfig, BlockName

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = "torch.Tensor"  # type: ignore[misc,assignment]


class SeqNN(nn.Module):
    """Sequence neural network model for regulatory activity prediction.

    Args:
        params (dict): Model specification and parameters including:
            - seq_length: Input sequence length
            - seq_depth: Number of channels (4 for DNA)
            - trunk: List of block configurations for the model trunk
            - heads: List of head configurations for prediction heads
            - num_targets: Number of prediction targets
            - target_length: Output sequence length
            - augment_rc: Whether to use reverse complement augmentation
            - augment_shift: List of shift amounts for augmentation
            - strand_pair: List of strand pairs for merging predictions
            - activation: Default activation function
    """

    def __init__(self, params: dict[str, Any]):
        super().__init__()
        self.params = params
        self._set_defaults()
        self._build_model()

    def _set_defaults(self):
        """Set default parameters."""
        defaults = {
            "augment_rc": False,
            "augment_shift": [0],
            "strand_pair": [],
            "verbose": True,
            "activation": "relu",
            "seq_length": 1344,
            "seq_depth": 4,
        }
        for key, value in defaults.items():
            if key not in self.params:
                self.params[key] = value

    def _build_model(self):
        """Build the model architecture."""
        seq_length = self.params.get("seq_length", 1344)
        seq_depth = self.params.get("seq_depth", 4)

        # Build trunk
        trunk_config = self.params.get("trunk", [])
        self.trunk = self._build_trunk(trunk_config, seq_length, seq_depth)

        # Track trunk output shape
        self._trunk_output_channels = self._get_trunk_output_channels(trunk_config, seq_depth)

        # Final activation
        activation = self.params.get("activation", "relu")
        self.final_activation = blocks.get_activation(activation)

        # Build heads - support both 'heads' and 'head_hic' (legacy)
        # head_hic can be either a flat list of blocks or a list of heads (each head is a list of blocks)
        head_configs = self.params.get("heads", self.params.get("head_hic", []))

        # If head_hic is provided as a flat list of blocks (each has 'name' key), wrap in a list
        if "head_hic" in self.params and "heads" not in self.params:
            if head_configs and isinstance(head_configs[0], dict) and "name" in head_configs[0]:
                head_configs = [head_configs]  # Wrap flat list in outer list

        self.heads = nn.ModuleList([self._build_head(hc) for hc in head_configs])

        # Track if we predict upper triangular
        self.preds_triu = False

        # Track stride and cropping
        self.model_strides = [1]
        self.target_lengths = []
        self.target_crops = []

    def _build_trunk(
        self, config: list[dict[str, Any]], seq_length: int, seq_depth: int
    ) -> nn.Module:
        """Build the model trunk from block configurations.

        Args:
            config: List of block configuration dicts
            seq_length: Input sequence length
            seq_depth: Number of input channels

        Returns:
            Sequential trunk module
        """
        layers = []
        current_channels = seq_depth
        current_length = seq_length

        for block_config in config:
            block_name = block_config.get("name", "conv_block")

            # Handle tower specially - it contains multiple blocks
            if block_name in ["conv_tower", "conv_tower_nac", "res_tower"]:
                tower = self._build_block(block_config, current_channels)
                layers.append(tower)
                # Update channels based on tower output
                if hasattr(tower, "reprs") and tower.reprs:
                    current_channels = tower.reprs[-1]
                elif hasattr(tower, "blocks") and len(tower.blocks) > 0:
                    # Get output from last block
                    current_channels = self._get_block_output_channels(tower, current_channels)
            else:
                block = self._build_block(block_config, current_channels)
                layers.append(block)
                # Update channels
                current_channels = block_config.get("filters", current_channels)

            # Track length changes from pooling/striding
            if "pool_size" in block_config and block_config["pool_size"] > 1:
                current_length = current_length // block_config["pool_size"]
            if "stride" in block_config and block_config["stride"] > 1:
                current_length = current_length // block_config["stride"]

        return nn.Sequential(*layers)

    def _get_trunk_output_channels(self, config: list[dict[str, Any]], seq_depth: int) -> int:
        """Get the number of output channels from the trunk."""
        current_channels = seq_depth
        for block_config in config:
            block_name = block_config.get("name", "conv_block")

            if block_name in ["conv_tower", "conv_tower_nac", "res_tower"]:
                filters_init = block_config.get("filters_init", current_channels)
                filters_end = block_config.get("filters_end", filters_init * 8)
                repeat = block_config.get("repeat", 4)
                if block_name == "res_tower":
                    # Use filters_end as output
                    current_channels = filters_end
                else:
                    # For towers, final filters depend on multiplier
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

    def _get_block_output_channels(self, block, in_channels: int) -> int:
        """Get output channels from a block."""
        if hasattr(block, "out_channels"):
            return block.out_channels
        elif hasattr(block, "conv_layers") and len(block.conv_layers) > 0:
            # Try to get from conv layer
            for layer in block.conv_layers:
                if isinstance(layer, nn.Conv1d):
                    return layer.out_channels
        return in_channels

    def _build_block(self, config: dict[str, Any], in_channels: int) -> nn.Module:
        """Build a single block from configuration.

        Args:
            config: Block configuration dict
            in_channels: Number of input channels

        Returns:
            Block module
        """
        block_name = config.get("name", "conv_block")

        # Extract common parameters
        params = {k: v for k, v in config.items() if k != "name"}
        params["in_channels"] = in_channels

        # Map 'filters' to 'out_channels' for compatibility with existing configs
        # (The config uses 'filters', but the blocks expect 'out_channels')
        if "filters" in params and "out_channels" not in params:
            params["out_channels"] = params.pop("filters")

        # Create block
        return blocks.create_block(block_name, **params)

    def _build_head(self, config: list[dict[str, Any]]) -> nn.Module:
        """Build a prediction head from configuration.

        Args:
            config: List of block configurations for this head

        Returns:
            Sequential head module
        """
        # First pass: compute the final state before building blocks
        # This is needed to know the correct input size for Final block

        # Check if this is a 2D head (has one_to_two)
        has_2d = any(bc.get("name") == "one_to_two" for bc in config)

        # Compute trunk output length
        seq_length = self.params.get("seq_length", 1344)
        for tc in self.params.get("trunk", []):
            name = tc.get("name", "")
            pool_size = tc.get("pool_size", 1)
            stride = tc.get("stride", 1)
            repeat = tc.get("repeat", 1)

            if pool_size > 1:
                if "tower" in name:
                    seq_length = seq_length // (pool_size**repeat)
                else:
                    seq_length = seq_length // pool_size
            if stride > 1:
                seq_length = seq_length // stride

        # Account for potential +1 from dilated convolutions with 'same' padding
        # This is a simplified heuristic - actual length may vary
        seq_length = seq_length + 1

        # For 2D heads, we need to compute the flattened size correctly
        # We need to know the actual size that upper_tri will produce

        precomputed_flat_size = None
        if has_2d:
            # Find the positions of cropping_2d and upper_tri in the config
            cropping_2d_idx = None
            upper_tri_idx = None
            for i, bc in enumerate(config):
                if bc.get("name") == "cropping_2d":
                    cropping_2d_idx = i
                elif bc.get("name") == "upper_tri":
                    upper_tri_idx = i

            # Determine the length that upper_tri operates on
            # If cropping comes AFTER upper_tri, use the original length
            # If cropping comes BEFORE upper_tri, use the post-cropping length
            upper_tri_length = seq_length
            if cropping_2d_idx is not None and upper_tri_idx is not None:
                if cropping_2d_idx < upper_tri_idx:
                    # Cropping comes BEFORE upper_tri, apply cropping first
                    for bc in config:
                        if bc.get("name") == "cropping_2d":
                            crop = bc.get("cropping", 0)
                            upper_tri_length = max(1, upper_tri_length - 2 * crop)
                # else: cropping AFTER upper_tri, use original length (no change needed)

            # Get diagonal offset
            diagonal_offset = 0
            for bc in config:
                if bc.get("name") == "upper_tri":
                    diagonal_offset = bc.get("diagonal_offset", 0)
                    break

            # Compute triu_count
            effective_len = max(1, upper_tri_length - abs(diagonal_offset))
            triu_count = effective_len * (effective_len + 1) // 2

            # Track channels through head blocks to get final channel count
            in_channels = self._trunk_output_channels
            for bc in config:
                bc_name = bc.get("name", "")
                if bc_name == "concat_dist_2d":
                    num_features = bc.get("num_features", 5)
                    in_channels = in_channels + num_features
                elif bc_name == "conv_block_2d":
                    in_channels = bc.get("filters", in_channels)
                elif bc_name == "dilated_residual_2d":
                    in_channels = bc.get("filters", in_channels)

            precomputed_flat_size = in_channels * triu_count

        # Now build the actual head
        layers = []
        in_channels = self._trunk_output_channels
        current_length = seq_length

        for block_config in config:
            block_name = block_config.get("name", "conv_block")

            if block_name == "final":
                # Final layer - use Final class
                params = {k: v for k, v in block_config.items() if k != "name"}

                # For 2D heads with flatten, use pre-computed flattened size
                if has_2d and "flatten" not in params:
                    params["flatten"] = True

                if params.get("flatten", False):
                    # Use the pre-computed flattened size (channels * triu_count)
                    params["in_channels"] = precomputed_flat_size
                else:
                    params["in_channels"] = in_channels

                # Map 'units' to 'out_units' for Final block compatibility
                if "units" in params and "out_units" not in params:
                    params["out_units"] = params.pop("units")
                elif "units" not in params:
                    params["out_units"] = self.params.get("num_targets", 1)

                block = blocks.Final(**params)
                layers.append(block)
                in_channels = params.get("out_units", 1)
            else:
                block = self._build_block(block_config, in_channels)
                layers.append(block)
                # Update channels - handle special 2D blocks
                if block_name == "concat_dist_2d":
                    num_features = block_config.get("num_features", 5)
                    in_channels = in_channels + num_features
                elif block_name == "cropping_2d":
                    cropping = block_config.get("cropping", 0)
                    current_length = max(1, current_length - 2 * cropping)
                elif block_name == "upper_tri":
                    diagonal_offset = block_config.get("diagonal_offset", 0)
                    effective_len = max(1, current_length - abs(diagonal_offset))
                    current_length = effective_len * (effective_len + 1) // 2
                else:
                    in_channels = block_config.get("filters", in_channels)

        return nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, head_i: Optional[int] = None
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (batch, seq_length, seq_depth)
               or (batch, seq_depth, seq_length)
            head_i: Optional head index for single-head output

        Returns:
            Predictions tensor if head_i is specified, otherwise list of predictions
        """
        # Handle input shape (batch, seq_length, seq_depth) -> (batch, seq_depth, seq_length)
        if x.dim() == 3 and x.shape[-1] == 4:
            # Shape is (batch, seq_length, seq_depth) - transpose
            x = x.transpose(1, 2)

        # Trunk forward
        trunk_output = self.trunk(x)

        # Final activation
        trunk_output = self.final_activation(trunk_output)

        # Head forward
        if head_i is not None:
            return self.heads[head_i](trunk_output)

        return [head(trunk_output) for head in self.heads]

    def num_targets(self, head_i: Optional[int] = None) -> int:
        """Return number of prediction targets.

        Args:
            head_i: Optional head index

        Returns:
            Number of targets
        """
        if head_i is not None:
            head = self.heads[head_i]
        else:
            head = self.heads[0] if len(self.heads) > 0 else None

        if head is None:
            return self.params.get("num_targets", 0)

        # Try to get from final layer
        for layer in head:
            if isinstance(layer, blocks.Final):
                return layer.dense.out_features

        return self.params.get("num_targets", 0)

    def get_output_shape(self, head_i: int = 0) -> Tuple[int, int, int]:
        """Get output shape for a head.

        Args:
            head_i: Head index

        Returns:
            Tuple of (batch, length, channels)
        """
        # Use dummy forward to get shape
        seq_length = self.params.get("seq_length", 1344)
        seq_depth = self.params.get("seq_depth", 4)

        dummy_input = torch.zeros(1, seq_length, seq_depth)
        with torch.no_grad():
            output = self.forward(dummy_input, head_i)

        if isinstance(output, list):
            output = output[0]

        return output.shape

    def track_sequence(self, seq_length: int):
        """Track pooling, striding, and cropping of sequence.

        Args:
            seq_length: Original input sequence length
        """
        self.model_strides = []
        self.target_lengths = []
        self.target_crops = []

        for head in self.heads:
            # Determine model stride
            stride = 1
            for name, module in self.trunk.named_modules():
                if isinstance(module, (nn.MaxPool1d, nn.AvgPool1d)):
                    stride *= module.kernel_size
                elif hasattr(module, "pool") and module.pool is not None:
                    stride *= module.pool.kernel_size

            self.model_strides.append(stride)

            # Determine target length after striding
            target_full_length = seq_length // stride

            # Determine predictions length (simplified - assumes no cropping)
            self.target_lengths.append(target_full_length)
            self.target_crops.append(0)

    def save(self, path: str):
        """Save model weights.

        Args:
            path: Path to save model
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location: str = "cpu"):
        """Load model weights.

        Args:
            path: Path to load model from
            map_location: Device to map tensors to
        """
        self.load_state_dict(torch.load(path, map_location=map_location))

    def __repr__(self) -> str:
        return f"SeqNN(seq_length={self.params.get('seq_length', 'N/A')}, num_targets={self.num_targets()})"


def create_seqnn(params: dict[str, Any]) -> SeqNN:
    """Factory function to create a SeqNN model.

    Args:
        params: Model configuration dictionary

    Returns:
        SeqNN model instance
    """
    return SeqNN(params)


# Example configuration
DEFAULT_CONFIG = {
    "seq_length": 1344,
    "seq_depth": 4,
    "activation": "relu",
    "trunk": [
        {
            "name": "conv_tower",
            "filters_init": 48,
            "filters_end": 512,
            "repeat": 6,
            "kernel_size": 3,
            "norm_type": "batch",
            "activation": "relu",
        },
    ],
    "heads": [
        [
            {
                "name": "conv_block",
                "filters": 256,
                "kernel_size": 1,
                "norm_type": "batch",
                "activation": "relu",
            },
            {
                "name": "conv_block",
                "filters": 256,
                "kernel_size": 1,
                "norm_type": "batch",
                "activation": "relu",
            },
            {"name": "final", "units": 1, "activation": "linear"},
        ],
    ],
    "num_targets": 1,
}
