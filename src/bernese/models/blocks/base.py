# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Base utilities for neural network blocks.

This module provides activation functions, normalization layers, and other
foundational components used throughout the blocks package.
"""

from typing import Any, Literal, TypedDict

import torch
import torch.nn as nn

# Type aliases for block parameters
ActivationType = Literal[
    "relu", "gelu", "sigmoid", "tanh", "elu", "selu", "softplus", "linear", "none", None
]
NormType = Literal["batch", "layer", "batch_sync", None]
PoolType = Literal["max", "avg", "softmax", None]
PaddingType = Literal["same", "valid"] | int | None
OperationType = Literal["mean", "outer", "product", None]
SymmetrizeMode = Literal["mean", "max", "min"]


# TypedDict configs for block creation
class ConvBlockConfig(TypedDict, total=False):
    """Configuration for ConvBlock."""
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    dilation: int
    padding: PaddingType
    activation: ActivationType
    activation_end: ActivationType
    norm_type: NormType
    dropout: float
    residual: bool
    bn_momentum: float
    pool_size: int
    pool_type: PoolType
    conv_type: Literal["standard", "separable"]
    kernel_initializer: str
    norm_gamma: float | str | None


class ConvNACConfig(TypedDict, total=False):
    """Configuration for ConvNAC."""
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    dilation: int
    padding: PaddingType
    activation: ActivationType
    norm_type: NormType
    dropout: float
    residual: bool
    bn_momentum: float
    pool_size: int
    pool_type: PoolType
    se: bool


class ConvDNAConfig(TypedDict, total=False):
    """Configuration for ConvDNA."""
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    activation: ActivationType
    norm_type: NormType
    dropout: float
    residual: bool
    bn_momentum: float
    se: bool
    pool_size: int
    pool_type: PoolType
    dropout_residual: float


class DenseBlockConfig(TypedDict, total=False):
    """Configuration for DenseBlock."""
    in_channels: int
    out_channels: int
    activation: ActivationType
    flatten: bool
    dropout: float
    residual: bool
    norm_type: NormType
    bn_momentum: float


class FinalConfig(TypedDict, total=False):
    """Configuration for Final block."""
    in_channels: int
    out_units: int
    activation: ActivationType
    flatten: bool


class DilatedResidualConfig(TypedDict, total=False):
    """Configuration for DilatedResidual."""
    in_channels: int
    out_channels: int
    kernel_size: int
    rate_mult: float
    dropout: float
    activation: ActivationType
    norm_type: NormType
    bn_momentum: float


class TowerConfig(TypedDict, total=False):
    """Configuration for Tower blocks."""
    in_channels: int
    filters_init: int
    filters_end: int | None
    filters_mult: float | None
    divisible_by: int
    repeat: int
    kernel_size: int
    activation: ActivationType
    norm_type: NormType
    dropout: float
    bn_momentum: float
    pool_size: int
    pool_type: PoolType
    num_convs: int


def get_activation(activation: ActivationType) -> nn.Module:
    """Get activation function by name.

    Args:
        activation: Name of activation function

    Returns:
        Activation module
    """
    activations: dict[str, nn.Module] = {
        "relu": nn.ReLU(inplace=True),
        "gelu": nn.GELU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "elu": nn.ELU(inplace=True),
        "selu": nn.SELU(inplace=True),
        "softplus": nn.Softplus(),
        "linear": nn.Identity(),
        "none": nn.Identity(),
    }
    if activation is None or activation.lower() not in activations:
        raise ValueError(f"Unknown activation: {activation}")
    return activations[activation.lower()]


def get_norm_layer(norm_type: NormType | None, num_features: int, **kwargs: Any) -> nn.Module:
    """Get normalization layer by name.

    Args:
        norm_type: Type of normalization ('batch', 'layer', or None)
        num_features: Number of features
        **kwargs: Additional arguments for norm layer

    Returns:
        Normalization module or Identity if None
    """
    if norm_type is None:
        return nn.Identity()

    norm_type_lower = norm_type.lower()
    if norm_type_lower == "batch":
        momentum = kwargs.get("bn_momentum", 0.99)
        return nn.BatchNorm1d(num_features, momentum=1 - momentum)
    elif norm_type_lower == "layer":
        return nn.LayerNorm(num_features)
    elif norm_type_lower == "batch_sync":
        # PyTorch DDP handles sync internally; use standard BatchNorm
        return nn.BatchNorm1d(num_features, momentum=1 - kwargs.get("bn_momentum", 0.99))
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")


class Scale(nn.Module):
    """Learnable channel-wise scaling.

    Works with both Conv1D (channels in dim=1) and linear (features).

    Args:
        num_features: Number of features to scale
        init_value: Initial value for scale (default: 0 for residual blocks)
    """

    def __init__(self, num_features: int, init_value: float = 0.0):
        super().__init__()
        self.num_features = num_features
        self.scale = nn.Parameter(torch.full((num_features,), init_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle Conv1D (B, C, L) - scale channels
        if x.dim() == 3:
            return x * self.scale.view(1, self.num_features, 1)
        # Handle Linear (B, C) or (B, L, C)
        elif x.dim() == 2:
            return x * self.scale
        else:
            return x * self.scale
