# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Type-safe Pydantic models for SeqNN configuration.

This module provides fully typed configuration models for SeqNN blocks,
replacing runtime dict-based configuration with compile-time type checking.
"""

from typing import Literal, Optional, Union, Annotated
from pydantic import BaseModel, Field


############################################################
# Convolution Blocks
############################################################


class ConvBlockConfig(BaseModel):
    """Configuration for standard convolution block.

    Corresponds to: conv_block function
    """

    name: Literal["conv_block"] = "conv_block"
    filters: Optional[int] = None
    kernel_size: int = 1
    stride: int = 1
    dilation_rate: int = 1
    activation: str = "relu"
    activation_end: Optional[str] = None
    conv_type: str = "standard"
    residual: bool = False
    norm_type: Optional[str] = None
    bn_momentum: float = 0.99
    norm_gamma: Optional[str] = None
    l2_scale: float = 0
    l1_scale: float = 0
    dropout: float = 0
    kernel_initializer: str = "he_normal"
    padding: str = "same"
    pool_size: int = 1
    pool_type: str = "max"

    model_config = {"extra": "forbid"}


class ConvNACConfig(BaseModel):
    """Configuration for NAC (Norm-Act-Conv) block.

    Corresponds to: conv_nac function
    """

    name: Literal["conv_nac"] = "conv_nac"
    filters: Optional[int] = None
    kernel_size: int = 1
    stride: int = 1
    dilation_rate: int = 1
    activation: str = "relu"
    conv_type: str = "standard"
    residual: bool = False
    se: bool = False
    norm_type: Optional[str] = None
    bn_momentum: float = 0.99
    norm_gamma: Optional[str] = None
    l2_scale: float = 0
    dropout: float = 0
    kernel_initializer: str = "he_normal"
    padding: str = "same"
    pool_size: int = 1
    pool_type: str = "max"

    model_config = {"extra": "forbid"}


class ConvDNAConfig(BaseModel):
    """Configuration for DNA convolution block.

    Corresponds to: conv_dna function
    """

    name: Literal["conv_dna"] = "conv_dna"
    filters: Optional[int] = None
    kernel_size: int = 15
    stride: int = 1
    activation: str = "relu"
    l2_scale: float = 0
    residual: bool = False
    dropout: float = 0
    dropout_residual: float = 0
    norm_type: Optional[str] = None
    bn_momentum: float = 0.99
    norm_gamma: Optional[str] = None
    use_bias: Optional[bool] = None
    se: bool = False
    conv_type: str = "standard"
    kernel_initializer: str = "he_normal"
    padding: str = "same"
    pool_size: int = 1
    pool_type: str = "max"

    model_config = {"extra": "forbid"}


class ConvBlock2DConfig(BaseModel):
    """Configuration for 2D convolution block.

    Corresponds to: conv_block_2d function
    """

    name: Literal["conv_block_2d"] = "conv_block_2d"
    filters: int = 128
    kernel_size: int = 1
    stride: int = 1
    dilation_rate: int = 1
    activation: str = "relu"
    conv_type: str = "standard"
    l2_scale: float = 0
    dropout: float = 0
    norm_type: Optional[str] = None
    bn_momentum: float = 0.99
    norm_gamma: str = "ones"
    kernel_initializer: str = "he_normal"
    symmetric: bool = False
    pool_size: int = 1

    model_config = {"extra": "forbid"}


############################################################
# Tower Blocks
############################################################


class ConvTowerConfig(BaseModel):
    """Configuration for convolution tower.

    Corresponds to: conv_tower function
    """

    name: Literal["conv_tower"] = "conv_tower"
    filters_init: int
    filters_end: Optional[int] = None
    filters_mult: Optional[float] = None
    kernel_size: int = 3
    divisible_by: int = 1
    repeat: int = 1
    activation: str = "relu"
    activation_end: Optional[str] = None
    conv_type: str = "standard"
    residual: bool = False
    norm_type: Optional[str] = None
    bn_momentum: float = 0.99
    norm_gamma: Optional[str] = None
    l2_scale: float = 0
    l1_scale: float = 0
    dropout: float = 0
    kernel_initializer: str = "he_normal"
    padding: str = "same"
    pool_size: int = 1
    pool_type: str = "max"

    model_config = {"extra": "forbid"}


class ConvTowerNACConfig(BaseModel):
    """Configuration for NAC convolution tower.

    Corresponds to: conv_tower_nac function
    """

    name: Literal["conv_tower_nac"] = "conv_tower_nac"
    filters_init: int
    filters_end: Optional[int] = None
    filters_mult: Optional[float] = None
    divisible_by: int = 1
    repeat: int = 1
    activation: str = "relu"
    activation_end: Optional[str] = None
    conv_type: str = "standard"
    residual: bool = False
    norm_type: Optional[str] = None
    bn_momentum: float = 0.99
    norm_gamma: Optional[str] = None
    l2_scale: float = 0
    dropout: float = 0
    kernel_initializer: str = "he_normal"
    padding: str = "same"
    pool_size: int = 1
    pool_type: str = "max"

    model_config = {"extra": "forbid"}


class ResTowerConfig(BaseModel):
    """Configuration for residual tower.

    Corresponds to: res_tower function
    """

    name: Literal["res_tower"] = "res_tower"
    filters_init: int
    filters_end: Optional[int] = None
    filters_mult: Optional[float] = None
    kernel_size: int = 1
    activation: str = "relu"
    dropout: float = 0
    pool_size: int = 2
    pool_type: str = "max"
    divisible_by: int = 1
    repeat: int = 1
    num_convs: int = 2
    conv_type: str = "standard"
    residual: bool = False
    norm_type: Optional[str] = None
    bn_momentum: float = 0.99
    norm_gamma: Optional[str] = None
    l2_scale: float = 0
    kernel_initializer: str = "he_normal"

    model_config = {"extra": "forbid"}


############################################################
# Dense Blocks
############################################################


class DenseBlockConfig(BaseModel):
    """Configuration for dense (fully connected) block.

    Corresponds to: dense_block function
    """

    name: Literal["dense_block"] = "dense_block"
    units: Optional[int] = None
    activation: str = "relu"
    activation_end: Optional[str] = None
    flatten: bool = False
    dropout: float = 0
    l2_scale: float = 0
    l1_scale: float = 0
    residual: bool = False
    norm_type: Optional[str] = None
    bn_momentum: float = 0.99
    norm_gamma: Optional[str] = None
    kernel_initializer: str = "he_normal"

    model_config = {"extra": "forbid"}


class FinalConfig(BaseModel):
    """Configuration for final output block.

    Corresponds to: final function
    """

    name: Literal["final"] = "final"
    units: int
    activation: str = "linear"
    flatten: bool = False
    kernel_initializer: str = "he_normal"
    l2_scale: float = 0
    l1_scale: float = 0

    model_config = {"extra": "forbid"}


############################################################
# Dilated Blocks
############################################################


class DilatedResidualConfig(BaseModel):
    """Configuration for dilated residual block.

    Corresponds to: dilated_residual function
    """

    name: Literal["dilated_residual"] = "dilated_residual"
    filters: int
    kernel_size: int = 3
    rate_mult: float = 2
    dropout: float = 0
    repeat: int = 1
    conv_type: str = "standard"
    activation: str = "relu"
    norm_type: Optional[str] = None
    round_dilation: bool = False
    l2_scale: float = 0
    l1_scale: float = 0
    kernel_initializer: str = "he_normal"
    padding: str = "same"

    model_config = {"extra": "forbid"}


class DilatedResidual2DConfig(BaseModel):
    """Configuration for 2D dilated residual block.

    Corresponds to: dilated_residual_2d function
    """

    name: Literal["dilated_residual_2d"] = "dilated_residual_2d"
    filters: int
    kernel_size: int = 3
    rate_mult: float = 2
    dropout: float = 0
    repeat: int = 1
    activation: str = "relu"
    symmetric: bool = True
    conv_type: str = "standard"
    norm_type: Optional[str] = None
    l2_scale: float = 0
    kernel_initializer: str = "he_normal"

    model_config = {"extra": "forbid"}


############################################################
# 2D Operation Blocks
############################################################


class OneToTwoConfig(BaseModel):
    """Configuration for 1D to 2D transformation.

    Corresponds to: one_to_two function
    """

    name: Literal["one_to_two"] = "one_to_two"
    operation: str = "mean"

    model_config = {"extra": "forbid"}


class ConcatDist2DConfig(BaseModel):
    """Configuration for distance concatenation.

    Corresponds to: concat_dist_2d function
    """

    name: Literal["concat_dist_2d"] = "concat_dist_2d"

    model_config = {"extra": "forbid"}


class Symmetrize2DConfig(BaseModel):
    """Configuration for 2D symmetrization.

    Corresponds to: symmetrize_2d function
    """

    name: Literal["symmetrize_2d"] = "symmetrize_2d"

    model_config = {"extra": "forbid"}


class UpperTriConfig(BaseModel):
    """Configuration for upper triangular extraction.

    Corresponds to: upper_tri function
    """

    name: Literal["upper_tri"] = "upper_tri"
    diagonal_offset: int = 2

    model_config = {"extra": "forbid"}


class Cropping2DConfig(BaseModel):
    """Configuration for 2D cropping.

    Corresponds to: cropping_2d function
    """

    name: Literal["cropping_2d"] = "cropping_2d"
    cropping: int

    model_config = {"extra": "forbid"}


class SqueezeExciteConfig(BaseModel):
    """Configuration for squeeze-and-excitation block.

    Corresponds to: squeeze_excite function
    """

    name: Literal["squeeze_excite"] = "squeeze_excite"
    activation: str = "relu"
    additive: bool = False
    bottleneck_ratio: int = 8

    model_config = {"extra": "forbid"}


############################################################
# Block Union Type
############################################################


# All block configurations as a discriminated union
BlockConfig = Annotated[
    Union[
        ConvBlockConfig,
        ConvNACConfig,
        ConvDNAConfig,
        ConvBlock2DConfig,
        ConvTowerConfig,
        ConvTowerNACConfig,
        ResTowerConfig,
        DenseBlockConfig,
        FinalConfig,
        DilatedResidualConfig,
        DilatedResidual2DConfig,
        OneToTwoConfig,
        ConcatDist2DConfig,
        Symmetrize2DConfig,
        UpperTriConfig,
        Cropping2DConfig,
        SqueezeExciteConfig,
    ],
    Field(discriminator="name"),
]


# Type alias for trunk (list of blocks)
TrunkConfig = list[BlockConfig]

# Type alias for heads (list of list of blocks)
HeadsConfig = list[list[BlockConfig]]


############################################################
# SeqNN Configuration
############################################################


class SeqNNConfig(BaseModel):
    """Complete configuration for SeqNN model.

    This is the main configuration class that combines all parameters
    needed to build a SeqNN model. Uses Pydantic for validation.

    Note: This configuration uses strict typing. All fields must be
    provided with correct types. Use from_json() to load from file.
    """

    # Required fields
    seq_length: int = 1344
    seq_depth: int = 4
    num_targets: int = 1

    # Model architecture - must be properly typed
    trunk: TrunkConfig = Field(default_factory=list)
    heads: HeadsConfig = Field(default_factory=list)

    # Augmentation
    augment_rc: bool = False
    augment_shift: list[int] = Field(default_factory=lambda: [0])
    strand_pair: list[Optional[list[int]]] = Field(default_factory=list)

    # Regularization and initialization
    activation: str = "relu"
    l2_scale: float = 0
    l1_scale: float = 0
    kernel_initializer: str = "he_normal"
    norm_type: Optional[str] = None
    bn_momentum: float = 0.99

    # Output control
    verbose: bool = True
    diagonal_offset: int = 0

    model_config = {"extra": "forbid"}

    def model_post_init(self, context):
        """Apply defaults after initialization."""
        # Default trunk configuration
        if not self.trunk:
            self.trunk = [
                ConvTowerConfig(
                    name="conv_tower",
                    filters_init=48,
                    filters_end=512,
                    repeat=6,
                    kernel_size=3,
                    norm_type="batch",
                    activation="relu",
                )
            ]

        # Default heads configuration
        if not self.heads:
            self.heads = [
                [
                    ConvBlockConfig(
                        name="conv_block",
                        filters=256,
                        kernel_size=1,
                        norm_type="batch",
                        activation="relu",
                    ),
                    ConvBlockConfig(
                        name="conv_block",
                        filters=256,
                        kernel_size=1,
                        norm_type="batch",
                        activation="relu",
                    ),
                    FinalConfig(
                        name="final",
                        units=self.num_targets,
                        activation="linear",
                    ),
                ]
            ]

        # Extend strand_pair if needed
        while len(self.strand_pair) < len(self.heads):
            self.strand_pair.append(None)

        return super().model_post_init(context)

    @classmethod
    def from_json(cls, path: str) -> "SeqNNConfig":
        """Load config from JSON file.

        Args:
            path: Path to JSON file.

        Returns:
            SeqNNConfig instance.
        """
        import json

        with open(path) as f:
            params = json.load(f)
        return cls(**params)


# Export all config types
__all__ = [
    "BlockConfig",
    "TrunkConfig",
    "HeadsConfig",
    "SeqNNConfig",
    # Block configs
    "ConvBlockConfig",
    "ConvNACConfig",
    "ConvDNAConfig",
    "ConvBlock2DConfig",
    "ConvTowerConfig",
    "ConvTowerNACConfig",
    "ResTowerConfig",
    "DenseBlockConfig",
    "FinalConfig",
    "DilatedResidualConfig",
    "DilatedResidual2DConfig",
    "OneToTwoConfig",
    "ConcatDist2DConfig",
    "Symmetrize2DConfig",
    "UpperTriConfig",
    "Cropping2DConfig",
    "SqueezeExciteConfig",
]
