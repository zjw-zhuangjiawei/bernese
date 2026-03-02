# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Main SeqNN model class for regulatory genomics predictions.

This module provides a PyTorch implementation of the Sequence Neural Network (SeqNN)
model for regulatory activity prediction, migrated from the TensorFlow baskerville
implementation.
"""

from typing import Any, Optional, Union

import torch
import torch.nn as nn

from bernese.models.blocks import get_activation, Final
from bernese.models.geometry import (
    ModelGeometry,
)
from bernese.models.trunk import TrunkBuilder
from bernese.models.head import HeadBuilder


# Default configuration
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


class SeqNN(nn.Module):
    """Sequence neural network model for regulatory activity prediction.

    Args:
        params (dict): Model specification and parameters including:
            - seq_length: Input sequence length
            - seq_depth: Number of channels (4 for DNA)
            - trunk: List of block configurations for the model trunk
            - heads: List of head configurations for prediction heads
            - num_targets: Number of prediction targets
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
        """Build the model architecture using builders."""
        seq_length = self.params.get("seq_length", 1344)
        seq_depth = self.params.get("seq_depth", 4)

        # Build trunk using TrunkBuilder
        trunk_config = self.params.get("trunk", [])
        trunk_builder = TrunkBuilder(seq_length, seq_depth)
        self.trunk = trunk_builder.build(trunk_config)

        # Get trunk output geometry
        trunk_output_channels = trunk_builder.output_channels
        trunk_output_length = trunk_builder.output_length

        # Final activation
        activation = self.params.get("activation", "relu")
        self.final_activation = get_activation(activation)

        # Build heads using HeadBuilder
        num_targets = self.params.get("num_targets", 1)
        head_builder = HeadBuilder(
            trunk_output_channels,
            trunk_output_length,
            num_targets,
        )

        # Support both 'heads' and 'head_hic' (legacy)
        head_configs = self.params.get("heads", self.params.get("head_hic", []))

        # If head_hic is provided as a flat list of blocks, wrap in a list
        if "head_hic" in self.params and "heads" not in self.params:
            if head_configs and isinstance(head_configs[0], dict) and "name" in head_configs[0]:
                head_configs = [head_configs]

        self.heads = nn.ModuleList([head_builder.build(hc) for hc in head_configs])

        # Track geometry
        self.geometry = ModelGeometry(
            input_length=seq_length,
            input_channels=seq_depth,
            trunk_output_channels=trunk_output_channels,
            trunk_output_length=trunk_output_length,
        )

        # Compute head output shapes
        for hc in head_configs:
            length, channels = head_builder.compute_output_shape(hc)
            self.geometry.head_output_shapes.append((0, length, channels))

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
            if isinstance(layer, Final):
                return layer.dense.out_features

        return self.params.get("num_targets", 0)

    def get_output_shape(self, head_i: int = 0) -> tuple[int, int, int]:
        """Get output shape for a head.

        Args:
            head_i: Head index

        Returns:
            Tuple of (batch, length, channels)
        """
        # Use geometry if available
        if hasattr(self, "geometry") and head_i < len(self.geometry.head_output_shapes):
            _, length, channels = self.geometry.head_output_shapes[head_i]
            batch = 0  # Unknown at this point
            return (batch, length, channels)

        # Fallback to dummy forward
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
        self.geometry.strides = []
        self.geometry.target_lengths = []
        self.geometry.target_crops = []

        # Determine model stride
        stride = 1
        for module in self.trunk.modules():
            if isinstance(module, (nn.MaxPool1d, nn.AvgPool1d)):
                stride *= module.kernel_size
            elif hasattr(module, "pool") and module.pool is not None:
                stride *= module.pool.kernel_size

        for _ in self.heads:
            self.geometry.strides.append(stride)
            target_full_length = seq_length // stride
            self.geometry.target_lengths.append(target_full_length)
            self.geometry.target_crops.append(0)

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
