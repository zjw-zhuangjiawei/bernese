# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Dense (fully connected) blocks for SeqNN models.

This module provides dense/linear layer blocks used in the output heads
of the SeqNN model.
"""

from typing import Optional

import torch
import torch.nn as nn

from bernese.models.blocks.base import ActivationType, NormType, get_activation, get_norm_layer


class DenseBlock(nn.Module):
    """Dense block with optional flattening.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        activation: Activation function
        flatten: Whether to flatten spatial dimensions
        dropout: Dropout probability
        residual: Whether to add residual connection
        norm_type: Normalization type
        bn_momentum: BatchNorm momentum
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: ActivationType = "relu",
        flatten: bool = False,
        dropout: float = 0.0,
        residual: bool = False,
        norm_type: NormType = None,
        bn_momentum: float = 0.99,
        **kwargs,
    ):
        super().__init__()

        self.flatten = flatten
        self.residual = residual

        # Activation
        self.activation = get_activation(activation)

        # Dense
        self.dense = nn.Linear(in_channels, out_channels, bias=(norm_type is None))

        # Norm
        self.norm = None
        if norm_type:
            self.norm = get_norm_layer(norm_type, out_channels, bn_momentum=bn_momentum)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten if needed
        if self.flatten:
            b = x.shape[0]
            x = x.view(b, -1)

        x = self.activation(x)
        x = self.dense(x)

        if self.norm is not None:
            x = self.norm(x)

        if self.dropout is not None:
            x = self.dropout(x)

        return x


class Final(nn.Module):
    """Final dense layer for predictions.

    Args:
        in_channels: Number of input channels
        out_units: Number of output units
        activation: Output activation
        flatten: Whether to flatten spatial dimensions
    """

    def __init__(
        self,
        in_channels: int,
        out_units: int,
        activation: ActivationType = "linear",
        flatten: bool = False,
    ):
        super().__init__()

        self.flatten = flatten

        # Dense
        self.dense = nn.Linear(in_channels if not flatten else in_channels, out_units)

        # Activation
        self.activation = get_activation(activation) if activation != "linear" else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten if needed
        if self.flatten:
            b = x.shape[0]
            x = x.view(b, -1)

        x = self.dense(x)

        if self.activation is not None:
            x = self.activation(x)

        return x
