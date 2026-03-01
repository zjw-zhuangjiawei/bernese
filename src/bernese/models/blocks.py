# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Neural network blocks for SeqNN models.

This module provides PyTorch implementations of convolutional and dense blocks
for regulatory genomics predictions, migrated from the TensorFlow baskerville
implementation.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List, Optional, Any, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from bernese.types import (
    ActivationType,
    NormType,
    PoolType,
    BlockName,
    OneToTwoOperation,
    SymmetrizeMode,
)

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = "torch.Tensor"  # type: ignore[misc,assignment]


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

    norm_type = norm_type.lower()
    if norm_type == "batch":
        momentum = kwargs.get("bn_momentum", 0.99)
        return nn.BatchNorm1d(num_features, momentum=1 - momentum)
    elif norm_type == "layer":
        return nn.LayerNorm(num_features)
    elif norm_type == "batch_sync":
        # PyTorch DDP handles sync internally; use standard BatchNorm
        return nn.BatchNorm1d(num_features, momentum=1 - kwargs.get("bn_momentum", 0.99))
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")


class Scale(nn.Module):
    """Learnable channel-wise scaling.

    Args:
        num_features: Number of features to scale
        init_value: Initial value for scale (default: 0 for residual blocks)
    """

    def __init__(self, num_features: int, init_value: float = 0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.full((num_features,), init_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale.unsqueeze(0)


class ConvBlock(nn.Module):
    """Basic convolution block with normalization and activation.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        dilation: Dilation rate
        padding: Padding mode
        activation: Activation function name
        norm_type: Normalization type ('batch', 'layer', or None)
        dropout: Dropout probability
        residual: Whether to add residual connection
        l2_scale: L2 regularization weight (handled via optimizer)
        bn_momentum: BatchNorm momentum
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        dilation: int = 1,
        padding: str = "same",
        activation: str = "relu",
        norm_type: Optional[str] = None,
        dropout: float = 0.0,
        residual: bool = False,
        l2_scale: float = 0.0,
        bn_momentum: float = 0.99,
        activation_end: Optional[str] = None,
        pool_size: int = 1,
        pool_type: str = "max",
        **kwargs,
    ):
        super().__init__()

        self.residual = residual
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Calculate padding
        if padding == "same":
            pad = (kernel_size - 1) * dilation // 2
        else:
            pad = 0

        # Build layers
        layers = []

        # Initial activation
        if activation:
            layers.append(get_activation(activation))

        # Convolution
        layers.append(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=pad,
                dilation=dilation,
                bias=(norm_type is None),
            )
        )

        # Normalization
        if norm_type:
            layers.append(get_norm_layer(norm_type, out_channels, bn_momentum=bn_momentum))

        # Dropout
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        self.conv_layers = nn.Sequential(*layers)

        # Residual connection
        self.residual_add = None
        self.residual_scale = None
        if residual and in_channels != out_channels:
            self.residual_add = nn.Identity()
            self.residual_scale = Scale(out_channels, init_value=0.0)
        elif residual:
            self.residual_scale = Scale(out_channels, init_value=0.0)

        # End activation
        self.activation_end = get_activation(activation_end) if activation_end else None

        # Pooling
        self.pool = None
        if pool_size > 1:
            # Pooling padding should be pool_size // 2 for 'same' behavior
            pool_pad = pool_size // 2
            if pool_type == "max":
                self.pool = nn.MaxPool1d(pool_size, padding=pool_pad)
            elif pool_type == "avg":
                self.pool = nn.AvgPool1d(pool_size, padding=pool_pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_layers(x)

        # Residual connection
        if self.residual:
            if self.residual_scale is not None:
                out = self.residual_scale(out)
            if self.residual_add is not None or self.residual_scale is not None:
                # Handle channel mismatch via 1x1 conv if needed
                if self.in_channels != self.out_channels:
                    residual = F.interpolate(
                        x, size=out.shape[-1], mode="linear", align_corners=False
                    )
                    if residual.shape[1] != out.shape[1]:
                        # Use 1D interpolation then project
                        residual = F.conv1d(
                            residual,
                            torch.eye(
                                self.out_channels, self.in_channels, device=out.device
                            ).unsqueeze(-1),
                            padding=0,
                        )
                else:
                    residual = x
                # Match lengths
                if residual.shape[-1] != out.shape[-1]:
                    residual = F.interpolate(
                        residual, size=out.shape[-1], mode="linear", align_corners=False
                    )
                out = out + residual

        # End activation
        if self.activation_end is not None:
            out = self.activation_end(out)

        # Pooling
        if self.pool is not None:
            out = self.pool(out)

        return out


class ConvNAC(nn.Module):
    """Norm -> Activation -> Conv block.

    This is the NAC (Norm-Act-Conv) pattern used in ResNet-style towers.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        dilation: Dilation rate
        activation: Activation function name
        norm_type: Normalization type
        dropout: Dropout probability
        residual: Whether to add residual connection
        l2_scale: L2 regularization weight
        bn_momentum: BatchNorm momentum
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        dilation: int = 1,
        padding: str = "same",
        activation: str = "relu",
        norm_type: Optional[str] = None,
        dropout: float = 0.0,
        residual: bool = False,
        l2_scale: float = 0.0,
        bn_momentum: float = 0.99,
        pool_size: int = 1,
        pool_type: str = "max",
        se: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.residual = residual
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Calculate padding
        if padding == "same":
            pad = (kernel_size - 1) * dilation // 2
        else:
            pad = 0

        # Build layers: norm -> act -> conv
        layers = []

        # Normalization first (NAC pattern)
        if norm_type:
            layers.append(get_norm_layer(norm_type, in_channels, bn_momentum=bn_momentum))

        # Activation
        if activation:
            layers.append(get_activation(activation))

        # Convolution
        layers.append(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=pad,
                dilation=dilation,
                bias=True,  # NAC uses bias after norm
            )
        )

        # Squeeze-excite
        self.se = None
        if se:
            self.se = SqueezeExcite(out_channels)

        # Dropout
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        self.conv_layers = nn.Sequential(*layers)

        # Residual
        self.residual_conv = None
        if residual and in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, 1)

        # Pooling
        self.pool = None
        if pool_size > 1:
            if pool_type == "max":
                self.pool = nn.MaxPool1d(pool_size, padding=pad)
            elif pool_type == "avg":
                self.pool = nn.AvgPool1d(pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_layers(x)

        # SE
        if self.se is not None:
            out = self.se(out)

        # Residual
        if self.residual:
            if self.residual_conv is not None:
                residual = self.residual_conv(x)
            else:
                residual = x

            # Match lengths if needed
            if residual.shape[-1] != out.shape[-1]:
                residual = F.interpolate(
                    residual, size=out.shape[-1], mode="linear", align_corners=False
                )

            out = out + residual

        # Pooling
        if self.pool is not None:
            out = self.pool(out)

        return out


class ConvDNA(nn.Module):
    """DNA-specific convolution block with optional residual.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size (default: 15 for DNA)
        stride: Convolution stride
        activation: Activation function name
        norm_type: Normalization type
        dropout: Dropout probability
        residual: Whether to add residual connection
        l2_scale: L2 regularization weight
        bn_momentum: BatchNorm momentum
        se: Whether to use squeeze-excitation
        pool_size: Pooling size
        pool_type: Pooling type
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 15,
        stride: int = 1,
        activation: str = "relu",
        norm_type: Optional[str] = None,
        dropout: float = 0.0,
        residual: bool = False,
        l2_scale: float = 0.0,
        bn_momentum: float = 0.99,
        se: bool = False,
        pool_size: int = 1,
        pool_type: str = "max",
        dropout_residual: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        self.residual = residual

        # Calculate padding for 'same'
        pad = (kernel_size - 1) // 2

        # First conv without activation (preserve DNA patterns)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=pad,
            bias=(norm_type is None and not residual),
        )

        # SE
        self.se = None
        if se:
            self.se = SqueezeExcite(out_channels)

        # Residual path
        if residual:
            self.residual_conv = ConvNAC(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                activation=activation,
                norm_type=norm_type,
                dropout=dropout_residual,
                bn_momentum=bn_momentum,
            )
            self.residual_scale = Scale(out_channels, init_value=0.0)

        # Norm after conv
        self.norm = None
        if norm_type and not residual:
            self.norm = get_norm_layer(norm_type, out_channels, bn_momentum=bn_momentum)

        # Activation
        self.activation = get_activation(activation) if activation else None

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Pooling
        self.pool = None
        if pool_size > 1:
            if pool_type == "max":
                self.pool = nn.MaxPool1d(pool_size)
            elif pool_type == "softmax":
                self.pool = SoftmaxPool1D(pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)

        # SE
        if self.se is not None:
            out = self.se(out)

        # Residual
        if self.residual:
            residual = self.residual_conv(out)
            out = self.residual_scale(out + residual)
        else:
            # Norm + activation
            if self.norm is not None:
                out = self.norm(out)
            if self.activation is not None:
                out = self.activation(out)

        # Dropout
        if self.dropout is not None:
            out = self.dropout(out)

        # Pooling
        if self.pool is not None:
            out = self.pool(out)

        return out


class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation block.

    Args:
        channels: Number of channels
        reduction: Squeeze reduction ratio
        activation: Activation function
        additive: Whether to use additive attention
    """

    def __init__(
        self, channels: int, reduction: int = 8, activation: str = "relu", additive: bool = False
    ):
        super().__init__()
        self.additive = additive

        # Squeeze
        self.squeeze = nn.AdaptiveAvgPool1d(1)

        # Excitation
        self.excite = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            get_activation(activation),
            nn.Linear(channels // reduction, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.shape

        # Squeeze
        s = self.squeeze(x).view(b, c)

        # Excite
        e = self.excite(s).view(b, c, 1)

        if self.additive:
            return x + e
        else:
            return x * torch.sigmoid(e)


class SoftmaxPool1D(nn.Module):
    """Learnable softmax pooling.

    Args:
        pool_size: Size of pooling window
        per_channel: Whether to compute weights per channel
        init_gain: Initial gain for softmax weights
    """

    def __init__(self, pool_size: int = 2, per_channel: bool = False, init_gain: float = 2.0):
        super().__init__()
        self.pool_size = pool_size
        self.per_channel = per_channel

        # Learnable logits
        self.logit_linear = nn.Conv1d(
            1 if not per_channel else pool_size, pool_size if per_channel else 1, 1, bias=False
        )

        # Initialize
        if init_gain != 0:
            nn.init.zeros_(self.logit_linear.weight)
            nn.init.constant_(self.logit_linear.weight, init_gain / pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, l = x.shape

        # Reshape for pooling
        if l % self.pool_size != 0:
            # Pad
            pad = self.pool_size - (l % self.pool_size)
            x = F.pad(x, (0, pad))
            b, c, l = x.shape

        # Reshape
        x = x.view(b, c, l // self.pool_size, self.pool_size)

        # Compute weights
        # For simplicity, use standard softmax pooling
        weights = F.softmax(x, dim=-1)

        return (x * weights).sum(dim=-1)


class ConvTower(nn.Module):
    """Convolutional tower with increasing filter counts.

    This implements the pyramidal filter growth pattern where each block
    increases the number of filters exponentially.

    Args:
        in_channels: Number of input channels
        filters_init: Initial number of filters
        filters_end: Target number of filters (reached after `repeat` blocks)
        filters_mult: Multiplicative factor for filter growth
        repeat: Number of blocks in the tower
        divisible_by: Round filters to be divisible by this number
        kernel_size: Kernel size for conv blocks
        activation: Activation function
        norm_type: Normalization type
        dropout: Dropout probability
        l2_scale: L2 regularization weight
        bn_momentum: BatchNorm momentum
    """

    def __init__(
        self,
        in_channels: int,
        filters_init: int,
        filters_end: Optional[int] = None,
        filters_mult: Optional[float] = None,
        divisible_by: int = 1,
        repeat: int = 1,
        kernel_size: int = 1,
        activation: str = "relu",
        norm_type: Optional[str] = None,
        dropout: float = 0.0,
        l2_scale: float = 0.0,
        bn_momentum: float = 0.99,
        **kwargs,
    ):
        super().__init__()

        self.reprs = []  # Store intermediate representations

        def _round(x):
            return int(round(x / divisible_by) * divisible_by)

        # Determine multiplier
        if filters_mult is None:
            assert filters_end is not None, "Either filters_mult or filters_end must be provided"
            filters_mult = math.exp(math.log(filters_end / filters_init) / (repeat - 1))

        current_channels = in_channels
        rep_filters = filters_init

        layers = []
        for ri in range(repeat):
            out_channels = _round(rep_filters)

            block = ConvBlock(
                in_channels=current_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                activation=activation,
                norm_type=norm_type,
                dropout=dropout,
                l2_scale=l2_scale,
                bn_momentum=bn_momentum,
                **kwargs,
            )
            layers.append(block)

            # Save representation
            self.reprs.append(out_channels)

            # Update channels
            current_channels = out_channels
            rep_filters *= filters_mult

        self.blocks = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, save_reprs: bool = True) -> torch.Tensor:
        reprs = []
        for block in self.blocks:
            x = block(x)
            if save_reprs:
                reprs.append(x)
        return x

    def get_repr_channels(self) -> List[int]:
        return self.reprs


class ConvTowerNAC(nn.Module):
    """Convolutional tower using NAC (Norm-Act-Conv) blocks.

    Similar to ConvTower but uses ConvNAC blocks instead of ConvBlock.
    """

    def __init__(
        self,
        in_channels: int,
        filters_init: int,
        filters_end: Optional[int] = None,
        filters_mult: Optional[float] = None,
        divisible_by: int = 1,
        repeat: int = 1,
        kernel_size: int = 1,
        activation: str = "relu",
        norm_type: Optional[str] = None,
        dropout: float = 0.0,
        l2_scale: float = 0.0,
        bn_momentum: float = 0.99,
        **kwargs,
    ):
        super().__init__()

        self.reprs = []

        def _round(x):
            return int(round(x / divisible_by) * divisible_by)

        # Determine multiplier
        if filters_mult is None:
            assert filters_end is not None
            filters_mult = math.exp(math.log(filters_end / filters_init) / (repeat - 1))

        current_channels = in_channels
        rep_filters = filters_init

        layers = []
        for ri in range(repeat):
            out_channels = _round(rep_filters)

            block = ConvNAC(
                in_channels=current_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                activation=activation,
                norm_type=norm_type,
                dropout=dropout,
                l2_scale=l2_scale,
                bn_momentum=bn_momentum,
                **kwargs,
            )
            layers.append(block)

            self.reprs.append(out_channels)

            current_channels = out_channels
            rep_filters *= filters_mult

        self.blocks = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, save_reprs: bool = True) -> torch.Tensor:
        reprs = []
        for block in self.blocks:
            x = block(x)
            if save_reprs:
                reprs.append(x)
        return x


class ResTower(nn.Module):
    """Residual tower with pooling between blocks.

    Args:
        in_channels: Number of input channels
        filters_init: Initial number of filters
        filters_end: Target number of filters
        filters_mult: Multiplicative factor
        kernel_size: Kernel size for convolutions
        dropout: Dropout between residual blocks
        pool_size: Pooling size between blocks
        pool_type: Type of pooling
        divisible_by: Round filters to be divisible by
        repeat: Number of residual blocks
        num_convs: Number of convolutions per residual block
    """

    def __init__(
        self,
        in_channels: int,
        filters_init: int,
        filters_end: Optional[int] = None,
        filters_mult: Optional[float] = None,
        kernel_size: int = 1,
        dropout: float = 0.0,
        pool_size: int = 2,
        pool_type: str = "max",
        divisible_by: int = 1,
        repeat: int = 1,
        num_convs: int = 2,
        activation: str = "relu",
        norm_type: Optional[str] = None,
        l2_scale: float = 0.0,
        bn_momentum: float = 0.99,
        **kwargs,
    ):
        super().__init__()

        self.reprs = []

        def _round(x):
            return int(round(x / divisible_by) * divisible_by)

        # Determine multiplier
        if filters_mult is None:
            assert filters_end is not None
            filters_mult = math.exp(math.log(filters_end / filters_init) / (repeat - 1))

        current_channels = in_channels
        rep_filters = filters_init

        layers = []
        for ri in range(repeat):
            out_channels = _round(rep_filters)

            # First conv
            convs = []
            convs.append(
                ConvNAC(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    activation=activation,
                    norm_type=norm_type,
                    dropout=0,  # No dropout in first conv
                    l2_scale=l2_scale,
                    bn_momentum=bn_momentum,
                    **kwargs,
                )
            )

            # Subsequent convs
            for ci in range(1, num_convs):
                convs.append(
                    ConvNAC(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        activation=activation,
                        norm_type=norm_type,
                        dropout=dropout if ci == num_convs - 1 else 0,
                        l2_scale=l2_scale,
                        bn_momentum=bn_momentum,
                        **kwargs,
                    )
                )

            # Build residual block
            if len(convs) > 1:
                res_block = nn.Sequential(*convs)
                if num_convs > 1:
                    res_scale = Scale(out_channels, init_value=0.0)

                    # Wrap for residual
                    class ResidualBlock(nn.Module):
                        def __init__(self, block, scale):
                            super().__init__()
                            self.block = block
                            self.scale = scale

                        def forward(self, x):
                            return self.scale(self.block(x) + x)

                    res_block = ResidualBlock(res_block, res_scale)
            else:
                res_block = convs[0]

            layers.append(res_block)

            # Pooling
            if pool_size > 1:
                if pool_type == "max":
                    layers.append(nn.MaxPool1d(pool_size))
                elif pool_type == "softmax":
                    layers.append(SoftmaxPool1D(pool_size))
                elif pool_type == "avg":
                    layers.append(nn.AvgPool1d(pool_size))

            self.reprs.append(out_channels)

            current_channels = out_channels
            rep_filters *= filters_mult

        self.blocks = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, save_reprs: bool = True) -> torch.Tensor:
        reprs = []
        for module in self.blocks:
            x = module(x)
            if save_reprs and hasattr(module, "block"):
                reprs.append(x)
        return x


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
        activation: str = "relu",
        flatten: bool = False,
        dropout: float = 0.0,
        residual: bool = False,
        norm_type: Optional[str] = None,
        bn_momentum: float = 0.99,
        l2_scale: float = 0.0,
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
        activation: str = "linear",
        flatten: bool = False,
        l2_scale: float = 0.0,
        **kwargs,
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


# Block factory functions


def create_block(block_name: str, in_channels: int, **kwargs) -> nn.Module:
    """Factory function to create blocks by name.

    Args:
        block_name: Name of the block type
        in_channels: Number of input channels
        **kwargs: Additional arguments for the block

    Returns:
        Instantiated block module
    """
    block_registry = {
        "conv_block": ConvBlock,
        "conv_nac": ConvNAC,
        "conv_dna": ConvDNA,
        "conv_tower": ConvTower,
        "conv_tower_nac": ConvTowerNAC,
        "res_tower": ResTower,
        "dense_block": DenseBlock,
        "final": Final,
        "dilated_residual": DilatedResidual,
        "one_to_two": OneToTwo,
        "concat_dist_2d": ConcatDist2d,
        "conv_block_2d": ConvBlock2d,
        "symmetrize_2d": Symmetrize2d,
        "dilated_residual_2d": DilatedResidual2d,
        "cropping_2d": Cropping2d,
        "upper_tri": UpperTri,
    }

    if block_name not in block_registry:
        raise ValueError(f"Unknown block: {block_name}. Available: {list(block_registry.keys())}")

    return block_registry[block_name](in_channels=in_channels, **kwargs)


class DilatedResidual(nn.Module):
    """Dilated residual block for 1D sequences.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        rate_mult: Multiplicative factor for dilation rate
        dropout: Dropout probability
        activation: Activation function name
        norm_type: Normalization type
        bn_momentum: BatchNorm momentum
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        rate_mult: float = 1.0,
        dropout: float = 0.0,
        activation: str = "relu",
        norm_type: Optional[str] = "batch",
        bn_momentum: float = 0.99,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Calculate base dilation rate
        base_dilation = 1
        dilation = int(base_dilation * rate_mult)

        # First conv with dilation
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation,
            bias=(norm_type is None),
        )

        # Norm
        self.norm1 = (
            get_norm_layer(norm_type, out_channels, bn_momentum=bn_momentum) if norm_type else None
        )

        # Activation
        self.activation = get_activation(activation) if activation else None

        # Second conv (1x1)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            1,
            bias=(norm_type is None),
        )

        # Norm 2
        self.norm2 = (
            get_norm_layer(norm_type, out_channels, bn_momentum=bn_momentum) if norm_type else None
        )

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Residual projection if channels differ
        self.residual_proj = None
        if in_channels != out_channels:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # First conv
        out = self.conv1(x)
        if self.norm1:
            out = self.norm1(out)
        if self.activation:
            out = self.activation(out)

        # Second conv
        out = self.conv2(out)
        if self.norm2:
            out = self.norm2(out)

        # Residual
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)

        # Add residual with scale
        out = out + residual

        if self.dropout:
            out = self.dropout(out)

        return out


class OneToTwo(nn.Module):
    """Convert 1D sequence to 2D contact map by computing pairwise interactions.

    Args:
        in_channels: Number of input channels
        operation: Operation to use ('mean', 'outer', 'product')
    """

    def __init__(
        self,
        in_channels: int,
        operation: str = "mean",
        **kwargs,
    ):
        super().__init__()
        self.operation = operation.lower()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert 1D to 2D contact map.

        Args:
            x: Input tensor of shape (batch, channels, length)

        Returns:
            Output tensor of shape (batch, channels, length, length)
        """
        batch, channels, length = x.shape

        if self.operation == "mean":
            # Compute outer product with mean
            x_expanded = x.unsqueeze(-1)  # (batch, channels, length, 1)
            x_expanded_t = x.unsqueeze(-2)  # (batch, channels, 1, length)
            out = (x_expanded + x_expanded_t) / 2
        elif self.operation == "outer" or self.operation == "product":
            # Outer product
            x_expanded = x.unsqueeze(-1)  # (batch, channels, length, 1)
            x_expanded_t = x.unsqueeze(-2)  # (batch, channels, 1, length)
            out = x_expanded * x_expanded_t
        else:
            raise ValueError(f"Unknown operation: {self.operation}")

        return out


class ConcatDist2d(nn.Module):
    """Concatenate distance-based features to 2D input.

    Args:
        num_features: Number of distance features to add
    """

    def __init__(
        self,
        in_channels: int,
        num_features: int = 5,
        **kwargs,
    ):
        super().__init__()
        self.num_features = num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add distance-based features.

        Args:
            x: Input tensor of shape (batch, channels, length, length)

        Returns:
            Output with distance features concatenated
        """
        batch, channels, length, _ = x.shape

        # Create distance indices
        dist_indices = torch.arange(length, device=x.device).float()
        dist_matrix = torch.abs(
            dist_indices.unsqueeze(0) - dist_indices.unsqueeze(1)
        )  # (length, length)

        # Normalize
        dist_matrix = dist_matrix / (length - 1 + 1e-8)

        # Create multiple distance-based features
        features = []
        for i in range(self.num_features):
            features.append(torch.sin((i + 1) * torch.pi * dist_matrix))

        # Stack features
        dist_features = torch.stack(features, dim=0)  # (num_features, length, length)
        dist_features = dist_features.unsqueeze(0).expand(
            batch, -1, -1, -1
        )  # (batch, num_features, length, length)

        # Concatenate along channel dimension
        out = torch.cat([x, dist_features], dim=1)

        return out


class ConvBlock2d(nn.Module):
    """2D convolution block with normalization and activation.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        dilation: Dilation rate
        activation: Activation function name
        norm_type: Normalization type ('batch', 'layer', or None)
        dropout: Dropout probability
        bn_momentum: BatchNorm momentum
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        padding: str = "same",
        activation: str = "relu",
        norm_type: Optional[str] = None,
        dropout: float = 0.0,
        bn_momentum: float = 0.99,
        **kwargs,
    ):
        super().__init__()

        # Calculate padding
        if padding == "same":
            pad = (kernel_size - 1) * dilation // 2
        else:
            pad = 0

        layers = []

        # Activation
        if activation:
            layers.append(get_activation(activation))

        # Conv2d
        layers.append(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=pad,
                dilation=dilation,
                bias=(norm_type is None),
            )
        )

        # Normalization
        if norm_type:
            if norm_type.lower() == "batch":
                layers.append(nn.BatchNorm2d(out_channels, momentum=1 - bn_momentum))
            elif norm_type.lower() == "layer":
                layers.append(nn.LayerNorm([out_channels, 1, 1]))  # Simplified

        # Dropout
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        self.conv_layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_layers(x)


class Symmetrize2d(nn.Module):
    """Symmetrize a 2D tensor (make it symmetric).

    Args:
        mode: How to symmetrize ('mean', 'max', 'min')
    """

    def __init__(
        self,
        in_channels: int,
        mode: str = "mean",
        **kwargs,
    ):
        super().__init__()
        self.mode = mode.lower()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Make tensor symmetric.

        Args:
            x: Input tensor of shape (batch, channels, length, length)

        Returns:
            Symmetrized tensor
        """
        x_lower = torch.tril(x, diagonal=-1)
        x_upper = torch.triu(x, diagonal=1)
        x_diag = torch.diagonal(x, dim1=2, dim2=3)

        if self.mode == "mean":
            sym = (x_lower + x_upper.transpose(2, 3)) / 2
        elif self.mode == "max":
            sym = torch.maximum(x_lower, x_upper.transpose(2, 3))
        elif self.mode == "min":
            sym = torch.minimum(x_lower, x_upper.transpose(2, 3))
        else:
            raise ValueError(f"Unknown symmetrize mode: {self.mode}")

        # Add diagonal back
        sym = sym + torch.diag_embed(x_diag, dim1=2, dim2=3)

        return sym


class DilatedResidual2d(nn.Module):
    """Dilated residual block for 2D contact maps.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        rate_mult: Multiplicative factor for dilation rate
        dropout: Dropout probability
        activation: Activation function name
        norm_type: Normalization type
        bn_momentum: BatchNorm momentum
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        rate_mult: float = 1.0,
        dropout: float = 0.0,
        activation: str = "relu",
        norm_type: Optional[str] = "batch",
        bn_momentum: float = 0.99,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Calculate dilation
        base_dilation = 1
        dilation = int(base_dilation * rate_mult)

        # Padding for 'same'
        pad = (kernel_size - 1) * dilation // 2

        # First conv
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=pad,
            dilation=dilation,
            bias=(norm_type is None),
        )

        # Norm 1
        self.norm1 = nn.BatchNorm2d(out_channels, momentum=1 - bn_momentum) if norm_type else None

        # Activation
        self.activation = get_activation(activation) if activation else None

        # Second conv (1x1)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            1,
            bias=(norm_type is None),
        )

        # Norm 2
        self.norm2 = nn.BatchNorm2d(out_channels, momentum=1 - bn_momentum) if norm_type else None

        # Dropout
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

        # Residual projection
        self.residual_proj = None
        if in_channels != out_channels:
            self.residual_proj = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # First conv
        out = self.conv1(x)
        if self.norm1:
            out = self.norm1(out)
        if self.activation:
            out = self.activation(out)

        # Second conv
        out = self.conv2(out)
        if self.norm2:
            out = self.norm2(out)

        # Residual
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)

        out = out + residual

        if self.dropout:
            out = self.dropout(out)

        return out


class Cropping2d(nn.Module):
    """Crop a 2D tensor.

    Args:
        cropping: Number of pixels to crop from each side
    """

    def __init__(
        self,
        in_channels: int,
        cropping: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.cropping = cropping

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Crop 2D tensor.

        Args:
            x: Input tensor of shape (batch, channels, length, length)

        Returns:
            Cropped tensor
        """
        if self.cropping == 0:
            return x

        return x[:, :, self.cropping : -self.cropping, self.cropping : -self.cropping]


class UpperTri(nn.Module):
    """Extract upper triangular part of a 2D tensor.

    Args:
        diagonal_offset: Offset for diagonal (positive = above diagonal)
    """

    def __init__(
        self,
        in_channels: int,
        diagonal_offset: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.diagonal_offset = diagonal_offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract upper triangular part.

        Args:
            x: Input tensor of shape (batch, channels, length, length)

        Returns:
            Flattened upper triangular tensor
        """
        batch, channels, length, _ = x.shape

        # torch.triu returns elements on and above the main diagonal
        # diagonal_offset > 0 shifts the diagonal down (includes more elements above diagonal)
        # For diagonal_offset=k, we want elements where j - i >= k, i.e., j >= i + k
        upper = torch.triu(x, diagonal=self.diagonal_offset)

        # Build index tensors for valid upper triangular elements
        # Valid elements are where j >= i + diagonal_offset
        row_indices = []
        col_indices = []
        for i in range(length):
            for j in range(i + self.diagonal_offset, length):
                row_indices.append(i)
                col_indices.append(j)

        row_indices = torch.tensor(row_indices, device=x.device, dtype=torch.long)
        col_indices = torch.tensor(col_indices, device=x.device, dtype=torch.long)

        # Use advanced indexing to extract
        out = upper[:, :, row_indices, col_indices]  # (batch, channels, numel)

        return out


# Block registry for SeqNN model
BLOCK_REGISTRY = {
    "conv_block": ConvBlock,
    "conv_nac": ConvNAC,
    "conv_dna": ConvDNA,
    "conv_tower": ConvTower,
    "conv_tower_nac": ConvTowerNAC,
    "res_tower": ResTower,
    "dense_block": DenseBlock,
    "final": Final,
    "dilated_residual": DilatedResidual,
    "one_to_two": OneToTwo,
    "concat_dist_2d": ConcatDist2d,
    "conv_block_2d": ConvBlock2d,
    "symmetrize_2d": Symmetrize2d,
    "dilated_residual_2d": DilatedResidual2d,
    "cropping_2d": Cropping2d,
    "upper_tri": UpperTri,
}
