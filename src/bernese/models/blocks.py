# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Block functions for SeqNN models using Keras 3.

This module provides functional block definitions ported from the TensorFlow
baskerville implementation, adapted for Keras 3.

Each block function now accepts a Pydantic Config object as its second argument.
"""

import math
from typing import Optional, List, Any, Union

import keras
from keras import ops

from bernese.models.layers import (
    Scale,
    SoftmaxPool1D,
    SqueezeExcite,
    OneToTwo,
    ConcatDist2D,
    Symmetrize2D,
    UpperTri,
    activate as activate_layer,
)
from bernese.models.config import (
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
)


# Type alias for tensor
Tensor = Any

# Type alias for block config
BlockConfig = Union[
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
]


############################################################
# Convolution Blocks
############################################################


def conv_block(inputs: Tensor, config: ConvBlockConfig) -> Tensor:
    """Construct a single convolution block.

    Args:
        inputs: Input tensor [batch_size, seq_length, features]
        config: ConvBlockConfig configuration

    Returns:
        Output tensor [batch_size, seq_length, features]
    """
    current = inputs
    c = config

    # Choose convolution type
    if c.conv_type == "separable":
        conv_layer = keras.layers.SeparableConv1D
    else:
        conv_layer = keras.layers.Conv1D

    filters = c.filters if c.filters is not None else inputs.shape[-1]

    # Activation
    current = activate_layer(current, c.activation)

    # Convolution
    current = conv_layer(
        filters=filters,
        kernel_size=c.kernel_size,
        strides=c.stride,
        padding=c.padding,
        use_bias=(c.norm_type is None),
        dilation_rate=c.dilation_rate,
        kernel_initializer=c.kernel_initializer,
        kernel_regularizer=keras.regularizers.l1_l2(c.l1_scale, c.l2_scale)
        if c.l1_scale > 0 or c.l2_scale > 0
        else None,
    )(current)

    # Normalize
    if c.norm_type == "batch":
        current = keras.layers.BatchNormalization(
            momentum=c.bn_momentum,
            gamma_initializer=c.norm_gamma or ("zeros" if c.residual else "ones"),
        )(current)
    elif c.norm_type == "layer":
        current = keras.layers.LayerNormalization(
            gamma_initializer=c.norm_gamma or ("zeros" if c.residual else "ones"),
        )(current)

    # Dropout
    if c.dropout > 0:
        current = keras.layers.Dropout(rate=c.dropout)(current)

    # Residual add
    if c.residual:
        current = keras.layers.Add()([inputs, current])

    # End activation
    if c.activation_end is not None:
        current = activate_layer(current, c.activation_end)

    # Pool
    if c.pool_size > 1:
        if c.pool_type == "softmax":
            current = SoftmaxPool1D(pool_size=c.pool_size)(current)
        elif c.pool_type == "avg":
            current = keras.layers.AvgPool1D(pool_size=c.pool_size, padding=c.padding)(current)
        else:
            current = keras.layers.MaxPool1D(pool_size=c.pool_size, padding=c.padding)(current)

    return current


def conv_nac(inputs: Tensor, config: ConvNACConfig) -> Tensor:
    """Construct a NAC (Norm-Act-Conv) block.

    Args:
        inputs: Input tensor [batch_size, seq_length, features]
        config: ConvNACConfig configuration

    Returns:
        Output tensor [batch_size, seq_length, features]
    """
    current = inputs
    c = config

    # Choose convolution type
    if c.conv_type == "separable":
        conv_layer = keras.layers.SeparableConv1D
    else:
        conv_layer = keras.layers.Conv1D

    filters = c.filters if c.filters is not None else inputs.shape[-1]

    # Normalize (NAC pattern: norm first)
    if c.norm_type == "batch":
        current = keras.layers.BatchNormalization(
            momentum=c.bn_momentum,
            gamma_initializer=c.norm_gamma or ("zeros" if c.residual else "ones"),
        )(current)
    elif c.norm_type == "layer":
        current = keras.layers.LayerNormalization(
            gamma_initializer=c.norm_gamma or ("zeros" if c.residual else "ones"),
        )(current)

    # Activation
    current = activate_layer(current, c.activation)

    # Convolution
    current = conv_layer(
        filters=filters,
        kernel_size=c.kernel_size,
        strides=c.stride,
        padding=c.padding,
        use_bias=True,
        dilation_rate=c.dilation_rate,
        kernel_initializer=c.kernel_initializer,
        kernel_regularizer=keras.regularizers.l2(c.l2_scale) if c.l2_scale > 0 else None,
    )(current)

    # Squeeze-excitation
    if c.se:
        current = SqueezeExcite(rank=8)(current)

    # Dropout
    if c.dropout > 0:
        current = keras.layers.Dropout(rate=c.dropout)(current)

    # Residual add
    if c.residual:
        current = keras.layers.Add()([inputs, current])

    # Pool
    if c.pool_size > 1:
        if c.pool_type == "softmax":
            current = SoftmaxPool1D(pool_size=c.pool_size)(current)
        elif c.pool_type == "avg":
            current = keras.layers.AvgPool1D(pool_size=c.pool_size, padding=c.padding)(current)
        else:
            current = keras.layers.MaxPool1D(pool_size=c.pool_size, padding=c.padding)(current)

    return current


def conv_dna(inputs: Tensor, config: ConvDNAConfig) -> Tensor:
    """Construct a DNA convolution block.

    Args:
        inputs: Input tensor [batch_size, seq_length, features]
        config: ConvDNAConfig configuration

    Returns:
        Output tensor [batch_size, seq_length, features]
    """
    current = inputs
    c = config

    # Choose convolution type
    if c.conv_type == "separable":
        conv_layer = keras.layers.SeparableConv1D
    else:
        conv_layer = keras.layers.Conv1D

    filters = c.filters if c.filters is not None else inputs.shape[-1]

    # Determine bias
    use_bias = c.use_bias if c.use_bias is not None else (c.norm_type is None and not c.residual)

    # Convolution
    current = conv_layer(
        filters=filters,
        kernel_size=c.kernel_size,
        strides=c.stride,
        padding=c.padding,
        use_bias=use_bias,
        kernel_initializer=c.kernel_initializer,
        kernel_regularizer=keras.regularizers.l2(c.l2_scale) if c.l2_scale > 0 else None,
    )(current)

    # Squeeze-excitation
    if c.se:
        current = SqueezeExcite(rank=8)(current)

    if c.residual:
        # Residual conv block using conv_nac
        residual_config = ConvNACConfig(
            activation=c.activation,
            l2_scale=c.l2_scale,
            dropout=c.dropout_residual,
            conv_type=c.conv_type,
            norm_type=c.norm_type,
            se=c.se,
            bn_momentum=c.bn_momentum,
            kernel_initializer=c.kernel_initializer,
        )
        rcurrent = conv_nac(current, residual_config)

        # Residual add with scale
        rcurrent = Scale()(rcurrent)
        current = keras.layers.Add()([current, rcurrent])

    else:
        # Normalize
        if c.norm_type == "batch":
            current = keras.layers.BatchNormalization(momentum=c.bn_momentum)(current)
        elif c.norm_type == "layer":
            current = keras.layers.LayerNormalization()(current)

        # Activation
        current = activate_layer(current, c.activation)

    # Dropout
    if c.dropout > 0:
        current = keras.layers.Dropout(rate=c.dropout)(current)

    # Pool
    if c.pool_size > 1:
        if c.pool_type == "softmax":
            current = SoftmaxPool1D(pool_size=c.pool_size)(current)
        elif c.pool_type == "avg":
            current = keras.layers.AvgPool1D(pool_size=c.pool_size, padding=c.padding)(current)
        else:
            current = keras.layers.MaxPool1D(pool_size=c.pool_size, padding=c.padding)(current)

    return current


def conv_block_2d(inputs: Tensor, config: ConvBlock2DConfig) -> Tensor:
    """Construct a single 2D convolution block.

    Args:
        inputs: Input tensor
        config: ConvBlock2DConfig configuration

    Returns:
        Output tensor
    """
    current = inputs
    c = config

    # Activation
    current = activate_layer(current, c.activation)

    # Choose convolution type
    if c.conv_type == "separable":
        conv_layer = keras.layers.SeparableConv2D
    else:
        conv_layer = keras.layers.Conv2D

    # Convolution
    current = conv_layer(
        filters=c.filters,
        kernel_size=c.kernel_size,
        strides=c.stride,
        padding="same",
        use_bias=(c.norm_type is None),
        dilation_rate=c.dilation_rate,
        kernel_initializer=c.kernel_initializer,
        kernel_regularizer=keras.regularizers.l2(c.l2_scale) if c.l2_scale > 0 else None,
    )(current)

    # Normalize
    if c.norm_type == "batch":
        current = keras.layers.BatchNormalization(
            momentum=c.bn_momentum,
            gamma_initializer=c.norm_gamma,
        )(current)
    elif c.norm_type == "layer":
        current = keras.layers.LayerNormalization(gamma_initializer=c.norm_gamma)(current)

    # Dropout
    if c.dropout > 0:
        current = keras.layers.Dropout(rate=c.dropout)(current)

    # Pool
    if c.pool_size > 1:
        current = keras.layers.MaxPool2D(pool_size=c.pool_size, padding="same")(current)

    # Symmetric
    if c.symmetric:
        current = Symmetrize2D()(current)

    return current


############################################################
# Towers
############################################################


def conv_tower(
    inputs: Tensor, config: ConvTowerConfig, reprs: Optional[list[keras.KerasTensor]] = None
) -> Tensor:
    """Construct a reducing convolution tower.

    Args:
        inputs: Input tensor [batch_size, seq_length, features]
        config: ConvTowerConfig configuration
        reprs: List to save representations

    Returns:
        Output tensor [batch_size, seq_length, features]
    """
    c = config

    def _round(x):
        return int(round(x / c.divisible_by) * c.divisible_by)

    current = inputs
    rep_filters = float(c.filters_init)

    # Determine multiplier
    if c.filters_mult is None:
        c.filters_mult = math.exp(math.log(c.filters_end / c.filters_init) / (c.repeat - 1))

    for ri in range(c.repeat):
        # Create config for conv_block
        block_config = ConvBlockConfig(
            filters=_round(rep_filters),
            kernel_size=c.kernel_size,
            activation=c.activation,
            activation_end=c.activation_end,
            stride=1,
            dilation_rate=1,
            conv_type=c.conv_type,
            residual=c.residual,
            norm_type=c.norm_type,
            bn_momentum=c.bn_momentum,
            norm_gamma=c.norm_gamma,
            l2_scale=c.l2_scale,
            l1_scale=c.l1_scale,
            dropout=c.dropout,
            kernel_initializer=c.kernel_initializer,
            padding=c.padding,
            pool_size=c.pool_size,
            pool_type=c.pool_type,
        )

        # Convolution
        current = conv_block(current, block_config)

        # Save representation
        if reprs is not None:
            reprs.append(current)

        # Update filters
        rep_filters *= c.filters_mult

    return current


def conv_tower_nac(
    inputs: Tensor, config: ConvTowerNACConfig, reprs: Optional[list[keras.KerasTensor]] = None
) -> Tensor:
    """Construct a reducing convolution tower using NAC blocks.

    Args:
        inputs: Input tensor [batch_size, seq_length, features]
        config: ConvTowerNACConfig configuration
        reprs: List to save representations

    Returns:
        Output tensor [batch_size, seq_length, features]
    """
    c = config

    def _round(x):
        return int(round(x / c.divisible_by) * c.divisible_by)

    current = inputs
    rep_filters = float(c.filters_init)

    # Determine multiplier
    if c.filters_mult is None:
        c.filters_mult = math.exp(math.log(c.filters_end / c.filters_init) / (c.repeat - 1))

    for ri in range(c.repeat):
        # Create config for conv_nac
        block_config = ConvNACConfig(
            filters=_round(rep_filters),
            kernel_size=c.kernel_size,
            activation=c.activation,
            stride=1,
            dilation_rate=1,
            conv_type=c.conv_type,
            residual=c.residual,
            norm_type=c.norm_type,
            bn_momentum=c.bn_momentum,
            norm_gamma=c.norm_gamma,
            l2_scale=c.l2_scale,
            dropout=c.dropout,
            kernel_initializer=c.kernel_initializer,
            padding=c.padding,
            pool_size=c.pool_size,
            pool_type=c.pool_type,
        )

        # Convolution
        current = conv_nac(current, block_config)

        # Save representation
        if reprs is not None:
            reprs.append(current)

        # Update filters
        rep_filters *= c.filters_mult

    return current


def res_tower(
    inputs: Tensor, config: ResTowerConfig, reprs: Optional[list[keras.KerasTensor]] = None
) -> Tensor:
    """Construct a residual tower with pooling between blocks.

    Args:
        inputs: Input tensor [batch_size, seq_length, features]
        config: ResTowerConfig configuration
        reprs: List to save representations

    Returns:
        Output tensor [batch_size, seq_length, features]
    """
    c = config

    def _round(x):
        return int(round(x / c.divisible_by) * c.divisible_by)

    current = inputs
    rep_filters = float(c.filters_init)

    # Determine multiplier
    if c.filters_mult is None:
        c.filters_mult = math.exp(math.log(c.filters_end / c.filters_init) / (c.repeat - 1))

    for ri in range(c.repeat):
        rep_filters_int = _round(rep_filters)

        # Initial conv
        block_config0 = ConvNACConfig(
            filters=rep_filters_int,
            kernel_size=c.kernel_size,
            activation=c.activation,
            conv_type=c.conv_type,
            residual=c.residual,
            norm_type=c.norm_type,
            bn_momentum=c.bn_momentum,
            norm_gamma=c.norm_gamma,
            l2_scale=c.l2_scale,
            kernel_initializer=c.kernel_initializer,
        )
        current0 = conv_nac(current, block_config0)

        # Subsequent convs
        current = current0
        for ci in range(1, c.num_convs):
            block_config = ConvNACConfig(
                filters=rep_filters_int,
                kernel_size=c.kernel_size,
                activation=c.activation,
                conv_type=c.conv_type,
                residual=c.residual,
                norm_type=c.norm_type,
                bn_momentum=c.bn_momentum,
                norm_gamma=c.norm_gamma,
                l2_scale=c.l2_scale,
                kernel_initializer=c.kernel_initializer,
            )
            current = conv_nac(current, block_config)

        # Residual add with scale
        if c.num_convs > 1:
            current = Scale()(current)
            current = keras.layers.Add()([current0, current])

        # Dropout
        if c.dropout > 0:
            current = keras.layers.Dropout(rate=c.dropout)(current)

        # Save representation
        if reprs is not None:
            reprs.append(current)

        # Pool
        if c.pool_size > 1:
            if c.pool_type == "softmax":
                current = SoftmaxPool1D(pool_size=c.pool_size)(current)
            elif c.pool_type == "avg":
                current = keras.layers.AvgPool1D(pool_size=c.pool_size, padding="same")(current)
            else:
                current = keras.layers.MaxPool1D(pool_size=c.pool_size, padding="same")(current)

        # Update filters
        rep_filters *= c.filters_mult

    return current


############################################################
# Dense Blocks
############################################################


def dense_block(inputs: Tensor, config: DenseBlockConfig) -> Tensor:
    """Construct a dense (fully connected) block.

    Args:
        inputs: Input tensor
        config: DenseBlockConfig configuration

    Returns:
        Output tensor
    """
    current = inputs
    c = config

    units = c.units if c.units is not None else inputs.shape[-1]

    # Activation
    current = activate_layer(current, c.activation)

    # Flatten
    if c.flatten:
        current = keras.layers.Flatten()(current)

    # Dense
    current = keras.layers.Dense(
        units=units,
        use_bias=(c.norm_type is None),
        kernel_initializer=c.kernel_initializer,
        kernel_regularizer=keras.regularizers.l1_l2(c.l1_scale, c.l2_scale)
        if c.l1_scale > 0 or c.l2_scale > 0
        else None,
    )(current)

    # Normalize
    norm_gamma = c.norm_gamma if c.norm_gamma is not None else ("zeros" if c.residual else "ones")
    if c.norm_type == "batch":
        current = keras.layers.BatchNormalization(
            momentum=c.bn_momentum,
            gamma_initializer=norm_gamma,
        )(current)
    elif c.norm_type == "layer":
        current = keras.layers.LayerNormalization(gamma_initializer=norm_gamma)(current)

    # Dropout
    if c.dropout > 0:
        current = keras.layers.Dropout(rate=c.dropout)(current)

    # Residual add
    if c.residual:
        # Need matching shapes - apply projection if needed
        if inputs.shape[-1] != units:
            inputs_proj = keras.layers.Dense(units)(inputs)
        else:
            inputs_proj = inputs
        current = keras.layers.Add()([inputs_proj, current])

    # End activation
    if c.activation_end is not None:
        current = activate_layer(current, c.activation_end)

    return current


def final(inputs: Tensor, config: FinalConfig) -> Tensor:
    """Final simple transformation before comparison to targets.

    Args:
        inputs: Input tensor [batch_size, seq_length, features]
        config: FinalConfig configuration

    Returns:
        Output tensor [batch_size, seq_length(?), units]
    """
    current = inputs
    c = config

    # Flatten
    if c.flatten:
        current = keras.layers.Flatten()(current)

    # Dense
    current = keras.layers.Dense(
        units=c.units,
        use_bias=True,
        activation=c.activation,
        kernel_initializer=c.kernel_initializer,
        kernel_regularizer=keras.regularizers.l1_l2(c.l1_scale, c.l2_scale)
        if c.l1_scale > 0 or c.l2_scale > 0
        else None,
    )(current)

    return current


############################################################
# Dilated Blocks
############################################################


def dilated_residual(inputs: Tensor, config: DilatedResidualConfig) -> Tensor:
    """Construct a residual dilated convolution block.

    Args:
        inputs: Input tensor
        config: DilatedResidualConfig configuration

    Returns:
        Output tensor
    """
    current = inputs
    c = config
    dilation_rate = 1.0

    for ri in range(c.repeat):
        rep_input = current

        # Create config for first conv_block
        conv1_config = ConvBlockConfig(
            filters=c.filters,
            kernel_size=c.kernel_size,
            activation=c.activation,
            dilation_rate=int(round(dilation_rate)),
            conv_type=c.conv_type,
            norm_type=c.norm_type,
            norm_gamma="ones",
            l2_scale=c.l2_scale,
            l1_scale=c.l1_scale,
            kernel_initializer=c.kernel_initializer,
            padding=c.padding,
        )

        # Dilated conv
        current = conv_block(current, conv1_config)

        # Create config for second conv_block
        conv2_config = ConvBlockConfig(
            filters=rep_input.shape[-1],
            activation=c.activation,
            dilation_rate=1,
            conv_type=c.conv_type,
            norm_type=c.norm_type,
            norm_gamma="zeros",
            dropout=c.dropout,
            l2_scale=c.l2_scale,
            l1_scale=c.l1_scale,
            kernel_initializer=c.kernel_initializer,
            padding=c.padding,
        )

        # Return conv
        current = conv_block(current, conv2_config)

        # Residual add with scale
        if c.norm_type is None:
            current = Scale()(current)

        current = keras.layers.Add()([rep_input, current])

        # Update dilation rate
        dilation_rate *= c.rate_mult
        if c.round_dilation:
            dilation_rate = round(dilation_rate)

    return current


def dilated_residual_2d(inputs: Tensor, config: DilatedResidual2DConfig) -> Tensor:
    """Construct a residual dilated convolution block for 2D.

    Args:
        inputs: Input tensor
        config: DilatedResidual2DConfig configuration

    Returns:
        Output tensor
    """
    current = inputs
    c = config
    dilation_rate = 1.0

    for ri in range(c.repeat):
        rep_input = current

        # Create config for first conv_block_2d
        conv1_config = ConvBlock2DConfig(
            filters=c.filters,
            kernel_size=c.kernel_size,
            activation=c.activation,
            dilation_rate=int(round(dilation_rate)),
            conv_type=c.conv_type,
            norm_type=c.norm_type,
            norm_gamma="ones",
            l2_scale=c.l2_scale,
            kernel_initializer=c.kernel_initializer,
        )

        # Dilated conv
        current = conv_block_2d(current, conv1_config)

        # Create config for second conv_block_2d
        conv2_config = ConvBlock2DConfig(
            filters=rep_input.shape[-1],
            activation=c.activation,
            dilation_rate=1,
            conv_type=c.conv_type,
            norm_type=c.norm_type,
            norm_gamma="zeros",
            dropout=c.dropout,
            l2_scale=c.l2_scale,
            kernel_initializer=c.kernel_initializer,
        )

        # Return conv
        current = conv_block_2d(current, conv2_config)

        # Residual add
        current = keras.layers.Add()([rep_input, current])

        # Enforce symmetry
        if c.symmetric:
            current = Symmetrize2D()(current)

        # Update dilation rate
        dilation_rate *= c.rate_mult

    return current


############################################################
# 2D Operations
############################################################


def one_to_two(inputs: Tensor, config: OneToTwoConfig) -> Tensor:
    """Convert 1D to 2D contact map."""
    return OneToTwo(operation=config.operation)(inputs)


def symmetrize_2d(inputs: Tensor, config: Symmetrize2DConfig) -> Tensor:
    """Symmetrize a 2D tensor."""
    return Symmetrize2D()(inputs)


def upper_tri(inputs: Tensor, config: UpperTriConfig) -> Tensor:
    """Extract upper triangular part."""
    return UpperTri(diagonal_offset=config.diagonal_offset)(inputs)


def concat_dist_2d(inputs: Tensor, config: ConcatDist2DConfig) -> Tensor:
    """Concatenate distance features to 2D input."""
    return ConcatDist2D()(inputs)


def cropping_2d(inputs: Tensor, config: Cropping2DConfig) -> Tensor:
    """Crop a 2D tensor."""
    return keras.layers.Cropping2D(cropping=config.cropping)(inputs)


def squeeze_excite(inputs: Tensor, config: SqueezeExciteConfig) -> Tensor:
    """Squeeze-and-excitation block."""
    return SqueezeExcite(
        activation=config.activation,
        additive=config.additive,
        rank=config.bottleneck_ratio,
    )(inputs)


############################################################
# Block Registry
############################################################

# Dictionary mapping block names to functions
name_func = {
    "center_slice": None,  # Not implemented
    "center_average": None,  # Not implemented
    "concat_dist_2d": concat_dist_2d,
    "conv_block": conv_block,
    "conv_dna": conv_dna,
    "conv_nac": conv_nac,
    "conv_block_2d": conv_block_2d,
    "conv_tower": conv_tower,
    "conv_tower_nac": conv_tower_nac,
    "cropping_2d": cropping_2d,
    "dense_block": dense_block,
    "dilated_residual": dilated_residual,
    "dilated_residual_2d": dilated_residual_2d,
    "final": final,
    "one_to_two": one_to_two,
    "symmetrize_2d": symmetrize_2d,
    "squeeze_excite": squeeze_excite,
    "res_tower": res_tower,
    "upper_tri": upper_tri,
}

# Dictionary for standard Keras layers
keras_func = {
    "Conv1D": keras.layers.Conv1D,
    "Cropping1D": keras.layers.Cropping1D,
    "Cropping2D": keras.layers.Cropping2D,
    "Dense": keras.layers.Dense,
    "Flatten": keras.layers.Flatten,
    "MaxPool1D": keras.layers.MaxPool1D,
    "MaxPool2D": keras.layers.MaxPool2D,
    "AvgPool1D": keras.layers.AvgPool1D,
    "AvgPool2D": keras.layers.AvgPool2D,
    "Dropout": keras.layers.Dropout,
    "BatchNormalization": keras.layers.BatchNormalization,
    "LayerNormalization": keras.layers.LayerNormalization,
}


# Export all functions
__all__ = [
    "conv_block",
    "conv_dna",
    "conv_nac",
    "conv_block_2d",
    "conv_tower",
    "conv_tower_nac",
    "res_tower",
    "dense_block",
    "final",
    "dilated_residual",
    "dilated_residual_2d",
    "one_to_two",
    "symmetrize_2d",
    "upper_tri",
    "concat_dist_2d",
    "cropping_2d",
    "squeeze_excite",
    "name_func",
    "keras_func",
    "BlockConfig",
]
