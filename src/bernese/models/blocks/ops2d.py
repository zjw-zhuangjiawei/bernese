# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""2D operations for contact map prediction.

This module provides 2D convolution blocks and tensor transformations
used in the Hi-C/contact map prediction heads of the SeqNN model.
"""

from typing import Optional

import torch
import torch.nn as nn

from bernese.models.blocks.base import ActivationType, NormType, OperationType, SymmetrizeMode, get_activation


class OneToTwo(nn.Module):
    """Convert 1D sequence to 2D contact map by computing pairwise interactions.

    Args:
        in_channels: Number of input channels
        operation: Operation to use ('mean', 'outer', 'product')
    """

    def __init__(
        self,
        in_channels: int,
        operation: OperationType = "mean",
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
        activation: ActivationType = "relu",
        norm_type: NormType = None,
        dropout: float = 0.0,
        bn_momentum: float = 0.99,
        **kwargs,
    ):
        super().__init__()

        # Calculate padding
        if isinstance(padding, str):
            if padding == "same":
                pad = (kernel_size - 1) * dilation // 2
            else:
                pad = 0
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
        mode: SymmetrizeMode = "mean",
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
        activation: ActivationType = "relu",
        norm_type: NormType = "batch",
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
            Flattened upper triangular tensor of shape (batch, channels, triu_count)
        """
        batch, channels, length, _ = x.shape

        # Compute upper triangular indices
        # For diagonal_offset=k, valid where j >= i + k
        # Count = (n-k) * (n-k+1) / 2 where n = length

        # Create row and column indices for upper triangular elements
        row_indices = []
        col_indices = []
        for i in range(length):
            for j in range(i + self.diagonal_offset, length):
                row_indices.append(i)
                col_indices.append(j)

        row_indices = torch.tensor(row_indices, device=x.device, dtype=torch.long)
        col_indices = torch.tensor(col_indices, device=x.device, dtype=torch.long)

        # Use advanced indexing to extract the upper triangular elements
        # Shape: (batch, channels, triu_count)
        out = x[:, :, row_indices, col_indices]

        return out
