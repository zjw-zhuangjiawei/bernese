# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""1D convolution blocks for SeqNN models.

This module provides convolutional building blocks including ConvBlock, ConvNAC,
ConvDNA, SqueezeExcitation, and dilated residual blocks.
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from bernese.models.blocks.base import (
    ActivationType,
    NormType,
    PaddingType,
    PoolType,
    Scale,
    get_activation,
    get_norm_layer,
)


class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation block.

    Args:
        channels: Number of channels
        reduction: Squeeze reduction ratio
        activation: Activation function
        additive: Whether to use additive attention
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 8,
        activation: ActivationType = "relu",
        additive: bool = False,
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


class ConvBlock(nn.Module):
    """Basic convolution block with normalization and activation.

    Migrated from baskerville/blocks.py::conv_block.
    Pattern: Act -> Conv -> Norm -> Dropout -> [Residual Add] -> ActEnd -> Pool

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        dilation: Dilation rate
        padding: Padding mode ('same', 'valid')
        activation: Activation function name ('relu', 'gelu', etc.)
        activation_end: Activation after residual and before pool
        norm_type: Normalization type ('batch', 'layer', 'batch_sync', or None)
        dropout: Dropout probability
        residual: Whether to add residual connection
        bn_momentum: BatchNorm momentum
        conv_type: Convolution type ('standard', 'separable')
        kernel_initializer: Weight initialization ('he_normal', 'glorot_uniform', etc.)
        norm_gamma: Normalization gamma initializer (None, 'ones', 'zeros', or float)
        pool_size: Pooling window size
        pool_type: Pooling type ('max', 'avg', 'softmax')
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        dilation: int = 1,
        padding: PaddingType = "same",
        activation: ActivationType = "relu",
        norm_type: NormType = None,
        dropout: float = 0.0,
        residual: bool = False,
        bn_momentum: float = 0.99,
        activation_end: ActivationType = None,
        pool_size: int = 1,
        pool_type: PoolType = "max",
        conv_type: Literal["standard", "separable"] = "standard",
        kernel_initializer: str = "he_normal",
        norm_gamma: float | str | None = None,
    ):
        super().__init__()

        self.residual = residual
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pool_type = pool_type

        # Build layers as sequential chain
        layers: list[nn.Module] = []

        # 1. Initial activation
        layers.append(get_activation(activation) if activation else nn.Identity())

        # 2. Convolution layer
        conv = self._make_conv(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            conv_type,
            (norm_type is None),
        )
        layers.append(conv)

        # Apply kernel initialization
        self._apply_initializer(conv, kernel_initializer)

        # 3. Normalization
        if norm_type:
            norm_layer = get_norm_layer(norm_type, out_channels, bn_momentum=bn_momentum)
            self._apply_norm_gamma(norm_layer, norm_gamma)
            layers.append(norm_layer)
        else:
            layers.append(nn.Identity())

        # 4. Dropout
        layers.append(nn.Dropout(dropout) if dropout > 0 else nn.Identity())

        self.main = nn.Sequential(*layers)

        # 5. Residual connection components
        self.residual_scale = Scale(out_channels, init_value=0.0) if residual else nn.Identity()
        self.residual_proj = (
            nn.Conv1d(in_channels, out_channels, 1)
            if (residual and in_channels != out_channels)
            else nn.Identity()
        )

        # 6. End activation
        layers_end: list[nn.Module] = []
        layers_end.append(get_activation(activation_end) if activation_end else nn.Identity())

        # 7. Pooling
        if pool_size > 1:
            if pool_type == "max":
                # Use ceil_mode=True to match Keras 'same' padding behavior
                layers_end.append(nn.MaxPool1d(pool_size, ceil_mode=True))
            elif pool_type == "avg":
                layers_end.append(nn.AvgPool1d(pool_size, ceil_mode=True))
            elif pool_type == "softmax":
                layers_end.append(SoftmaxPool1D(pool_size))
        else:
            layers_end.append(nn.Identity())

        self.end = nn.Sequential(*layers_end)

    def _make_conv(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: PaddingType,
        dilation: int,
        conv_type: str,
        bias: bool,
    ) -> nn.Module:
        """Create convolution layer based on type."""
        if conv_type == "separable":
            # PyTorch equivalent: depthwise + pointwise
            # Use groups to achieve separable convolution
            return nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,  # type: ignore
                dilation=dilation,
                bias=bias,
                groups=in_channels,  # Depthwise
            )
        else:
            return nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,  # type: ignore
                dilation=dilation,
                bias=bias,
            )

    def _apply_initializer(self, module: nn.Module, initializer: str) -> None:
        """Apply weight initialization to module."""
        if isinstance(module, nn.Conv1d):
            if initializer == "he_normal" or initializer == "he_uniform":
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif initializer == "glorot_uniform":
                nn.init.xavier_uniform_(module.weight)
            elif initializer == "glorot_normal":
                nn.init.xavier_normal_(module.weight)
            elif initializer == "orthogonal":
                nn.init.orthogonal_(module.weight)
            # Default: leave as is (Kaiming normal is Keras he_normal default)

    def _apply_norm_gamma(self, module: nn.Module, gamma: float | str | None) -> None:
        """Apply normalization gamma initialization."""
        if gamma is None:
            return
        if isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
            if isinstance(gamma, (int, float)):
                nn.init.constant_(module.weight, gamma)
            elif gamma == "ones":
                nn.init.ones_(module.weight)
            elif gamma == "zeros":
                nn.init.zeros_(module.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Main path: activation -> conv -> norm -> dropout
        out = self.main(x)

        # Residual connection
        if self.residual:
            # Get residual from input
            residual = self.residual_proj(x)
            # Scale residual (must be after projection for proper shape)
            if self.residual_scale is not None:
                residual = self.residual_scale(residual)
            # Add residual
            out = out + residual

        # End activation and pooling
        return self.end(out)


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
        bn_momentum: BatchNorm momentum
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        dilation: int = 1,
        padding: PaddingType = "same",
        activation: ActivationType = "relu",
        norm_type: NormType = None,
        dropout: float = 0.0,
        residual: bool = False,
        bn_momentum: float = 0.99,
        pool_size: int = 1,
        pool_type: PoolType = "max",
        se: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.residual = residual
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Calculate padding
        if isinstance(padding, str):
            if padding == "same":
                pad = (kernel_size - 1) * dilation // 2
            else:
                pad = 0
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
        activation: ActivationType = "relu",
        norm_type: NormType = None,
        dropout: float = 0.0,
        residual: bool = False,
        bn_momentum: float = 0.99,
        se: bool = False,
        pool_size: int = 1,
        pool_type: PoolType = "max",
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
        activation: ActivationType = "relu",
        norm_type: NormType = "batch",
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
