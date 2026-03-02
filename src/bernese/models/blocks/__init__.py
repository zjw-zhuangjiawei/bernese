# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Neural network blocks for SeqNN models.

This package provides PyTorch implementations of convolutional and dense blocks
for regulatory genomics predictions, migrated from the TensorFlow baskerville
implementation.

Modules:
- base: Activation functions, normalization layers, and Scale
- conv: 1D convolution blocks (ConvBlock, ConvNAC, ConvDNA, etc.)
- towers: Tower architectures (ConvTower, ConvTowerNAC, ResTower)
- dense: Dense/linear blocks (DenseBlock, Final)
- ops2d: 2D operations for contact map prediction
"""

from typing import Any, Literal

import torch.nn as nn

# Re-export type aliases from base
from bernese.models.blocks.base import (
    ActivationType,
    ConvBlockConfig,
    ConvDNAConfig,
    ConvNACConfig,
    DenseBlockConfig,
    DilatedResidualConfig,
    FinalConfig,
    NormType,
    OperationType,
    PaddingType,
    PoolType,
    Scale,
    SymmetrizeMode,
    TowerConfig,
    get_activation,
    get_norm_layer,
)
from bernese.models.blocks.conv import (
    ConvBlock,
    ConvDNA,
    ConvNAC,
    DilatedResidual,
    SqueezeExcite,
    SoftmaxPool1D,
)
from bernese.models.blocks.towers import (
    ConvTower,
    ConvTowerNAC,
    ResTower,
)
from bernese.models.blocks.dense import (
    DenseBlock,
    Final,
)
from bernese.models.blocks.ops2d import (
    ConcatDist2d,
    ConvBlock2d,
    Cropping2d,
    DilatedResidual2d,
    OneToTwo,
    Symmetrize2d,
    UpperTri,
)

__all__ = [
    # Type aliases
    "ActivationType",
    "NormType",
    "PoolType",
    "PaddingType",
    "OperationType",
    "SymmetrizeMode",
    # TypedDict configs
    "ConvBlockConfig",
    "ConvNACConfig",
    "ConvDNAConfig",
    "DenseBlockConfig",
    "FinalConfig",
    "DilatedResidualConfig",
    "TowerConfig",
    # Base
    "Scale",
    "get_activation",
    "get_norm_layer",
    # Conv
    "ConvBlock",
    "ConvDNA",
    "ConvNAC",
    "DilatedResidual",
    "SqueezeExcite",
    "SoftmaxPool1D",
    # Towers
    "ConvTower",
    "ConvTowerNAC",
    "ResTower",
    # Dense
    "DenseBlock",
    "Final",
    # 2D Ops
    "ConcatDist2d",
    "ConvBlock2d",
    "Cropping2d",
    "DilatedResidual2d",
    "OneToTwo",
    "Symmetrize2d",
    "UpperTri",
    # Factory
    "create_block",
    "BLOCK_REGISTRY",
]

# Block name type
BlockNameType = Literal[
    "conv_block",
    "conv_nac",
    "conv_dna",
    "conv_tower",
    "conv_tower_nac",
    "res_tower",
    "dense_block",
    "final",
    "dilated_residual",
    "one_to_two",
    "concat_dist_2d",
    "conv_block_2d",
    "symmetrize_2d",
    "dilated_residual_2d",
    "cropping_2d",
    "upper_tri",
]


# Block registry for SeqNN model
BLOCK_REGISTRY: dict[str, type[nn.Module]] = {
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


def create_block(block_name: BlockNameType, in_channels: int, **kwargs: Any) -> nn.Module:
    """Factory function to create blocks by name.

    Args:
        block_name: Name of the block type
        in_channels: Number of input channels
        **kwargs: Additional arguments for the block

    Returns:
        Instantiated block module

    Raises:
        ValueError: If block_name is not recognized
    """
    if block_name not in BLOCK_REGISTRY:
        raise ValueError(f"Unknown block: {block_name}. Available: {list(BLOCK_REGISTRY.keys())}")

    return BLOCK_REGISTRY[block_name](in_channels=in_channels, **kwargs)
