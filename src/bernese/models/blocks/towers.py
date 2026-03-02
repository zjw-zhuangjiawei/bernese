# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Tower modules for SeqNN models.

This module provides tower architectures that stack multiple blocks with
increasing filter counts (pyramidal structure).
"""

import math
from typing import Any, List, Optional

import torch
import torch.nn as nn

from bernese.models.blocks.base import ActivationType, NormType, PoolType, Scale
from bernese.models.blocks.conv import ConvBlock, ConvNAC, SoftmaxPool1D


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
        activation: ActivationType = "relu",
        norm_type: NormType = None,
        dropout: float = 0.0,
        bn_momentum: float = 0.99,
        **kwargs: Any,
    ):
        super().__init__()

        self.reprs: list[int] = []

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
        activation: ActivationType = "relu",
        norm_type: NormType = None,
        dropout: float = 0.0,
        bn_momentum: float = 0.99,
        **kwargs: Any,
    ):
        super().__init__()

        self.reprs: list[int] = []

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
        pool_type: PoolType = "max",
        divisible_by: int = 1,
        repeat: int = 1,
        num_convs: int = 2,
        activation: ActivationType = "relu",
        norm_type: NormType = None,
        bn_momentum: float = 0.99,
        **kwargs: Any,
    ):
        super().__init__()

        self.reprs: list[int] = []

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
