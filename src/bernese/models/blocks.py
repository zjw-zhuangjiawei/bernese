# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Block functions for SeqNN models using Keras 3.

This module provides functional block definitions ported from the TensorFlow
baskerville implementation, adapted for Keras 3.
"""

import math
from typing import Optional, List, Any

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


# Type alias for tensor
Tensor = Any


############################################################
# Convolution Blocks
############################################################


def conv_block(
    inputs: Tensor,
    filters: Optional[int] = None,
    kernel_size: int = 1,
    activation: str = "relu",
    activation_end: Optional[str] = None,
    stride: int = 1,
    dilation_rate: int = 1,
    l2_scale: float = 0,
    l1_scale: float = 0,
    dropout: float = 0,
    conv_type: str = "standard",
    pool_size: int = 1,
    pool_type: str = "max",
    norm_type: Optional[str] = None,
    bn_momentum: float = 0.99,
    norm_gamma: Optional[str] = None,
    residual: bool = False,
    kernel_initializer: str = "he_normal",
    padding: str = "same",
) -> Tensor:
    """Construct a single convolution block.

    Args:
        inputs: Input tensor [batch_size, seq_length, features]
        filters: Conv1D filters
        kernel_size: Conv1D kernel_size
        activation: relu/gelu/etc
        activation_end: Activation after residual and before pool
        stride: Conv1D stride
        dilation_rate: Conv1D dilation rate
        l2_scale: L2 regularization weight
        l1_scale: L1 regularization weight
        dropout: Dropout rate probability
        conv_type: Conv1D layer type
        residual: Residual connection boolean
        pool_size: Max pool width
        norm_type: Apply batch or layer normalization
        bn_momentum: BatchNorm momentum
        norm_gamma: BatchNorm gamma initializer
        kernel_initializer: Weight initialization

    Returns:
        Output tensor [batch_size, seq_length, features]
    """
    current = inputs

    # Choose convolution type
    if conv_type == "separable":
        conv_layer = keras.layers.SeparableConv1D
    else:
        conv_layer = keras.layers.Conv1D

    if filters is None:
        filters = inputs.shape[-1]

    # Activation
    current = activate_layer(current, activation)

    # Convolution
    current = conv_layer(
        filters=filters,
        kernel_size=kernel_size,
        strides=stride,
        padding=padding,
        use_bias=(norm_type is None),
        dilation_rate=dilation_rate,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=keras.regularizers.l1_l2(l1_scale, l2_scale) if l1_scale > 0 or l2_scale > 0 else None,
    )(current)

    # Normalize
    if norm_type == "batch":
        current = keras.layers.BatchNormalization(
            momentum=bn_momentum,
            gamma_initializer=norm_gamma or ("zeros" if residual else "ones"),
        )(current)
    elif norm_type == "layer":
        current = keras.layers.LayerNormalization(
            gamma_initializer=norm_gamma or ("zeros" if residual else "ones"),
        )(current)

    # Dropout
    if dropout > 0:
        current = keras.layers.Dropout(rate=dropout)(current)

    # Residual add
    if residual:
        current = keras.layers.Add()([inputs, current])

    # End activation
    if activation_end is not None:
        current = activate_layer(current, activation_end)

    # Pool
    if pool_size > 1:
        if pool_type == "softmax":
            current = SoftmaxPool1D(pool_size=pool_size)(current)
        elif pool_type == "avg":
            current = keras.layers.AvgPool1D(pool_size=pool_size, padding=padding)(current)
        else:
            current = keras.layers.MaxPool1D(pool_size=pool_size, padding=padding)(current)

    return current


def conv_nac(
    inputs: Tensor,
    filters: Optional[int] = None,
    kernel_size: int = 1,
    activation: str = "relu",
    stride: int = 1,
    dilation_rate: int = 1,
    l2_scale: float = 0,
    dropout: float = 0,
    conv_type: str = "standard",
    residual: bool = False,
    pool_size: int = 1,
    pool_type: str = "max",
    norm_type: Optional[str] = None,
    bn_momentum: float = 0.99,
    norm_gamma: Optional[str] = None,
    kernel_initializer: str = "he_normal",
    padding: str = "same",
    se: bool = False,
) -> Tensor:
    """Construct a NAC (Norm-Act-Conv) block.

    Args:
        inputs: Input tensor [batch_size, seq_length, features]
        filters: Conv1D filters
        kernel_size: Conv1D kernel_size
        activation: relu/gelu/etc
        stride: Conv1D stride
        dilation_rate: Conv1D dilation rate
        l2_scale: L2 regularization weight
        dropout: Dropout rate probability
        conv_type: Conv1D layer type
        residual: Residual connection boolean
        pool_size: Max pool width
        pool_type: Pool type
        norm_type: Apply batch or layer normalization
        bn_momentum: BatchNorm momentum
        norm_gamma: BatchNorm gamma initializer
        kernel_initializer: Weight initialization
        se: Use squeeze-excitation

    Returns:
        Output tensor [batch_size, seq_length, features]
    """
    current = inputs

    # Choose convolution type
    if conv_type == "separable":
        conv_layer = keras.layers.SeparableConv1D
    else:
        conv_layer = keras.layers.Conv1D

    if filters is None:
        filters = inputs.shape[-1]

    # Normalize (NAC pattern: norm first)
    if norm_type == "batch":
        current = keras.layers.BatchNormalization(
            momentum=bn_momentum,
            gamma_initializer=norm_gamma or ("zeros" if residual else "ones"),
        )(current)
    elif norm_type == "layer":
        current = keras.layers.LayerNormalization(
            gamma_initializer=norm_gamma or ("zeros" if residual else "ones"),
        )(current)

    # Activation
    current = activate_layer(current, activation)

    # Convolution
    current = conv_layer(
        filters=filters,
        kernel_size=kernel_size,
        strides=stride,
        padding=padding,
        use_bias=True,
        dilation_rate=dilation_rate,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=keras.regularizers.l2(l2_scale) if l2_scale > 0 else None,
    )(current)

    # Squeeze-excitation
    if se:
        current = SqueezeExcite(rank=8)(current)

    # Dropout
    if dropout > 0:
        current = keras.layers.Dropout(rate=dropout)(current)

    # Residual add
    if residual:
        current = keras.layers.Add()([inputs, current])

    # Pool
    if pool_size > 1:
        if pool_type == "softmax":
            current = SoftmaxPool1D(pool_size=pool_size)(current)
        elif pool_type == "avg":
            current = keras.layers.AvgPool1D(pool_size=pool_size, padding=padding)(current)
        else:
            current = keras.layers.MaxPool1D(pool_size=pool_size, padding=padding)(current)

    return current


def conv_dna(
    inputs: Tensor,
    filters: Optional[int] = None,
    kernel_size: int = 15,
    activation: str = "relu",
    stride: int = 1,
    l2_scale: float = 0,
    residual: bool = False,
    dropout: float = 0,
    dropout_residual: float = 0,
    pool_size: int = 1,
    pool_type: str = "max",
    norm_type: Optional[str] = None,
    bn_momentum: float = 0.99,
    norm_gamma: Optional[str] = None,
    use_bias: Optional[bool] = None,
    se: bool = False,
    conv_type: str = "standard",
    kernel_initializer: str = "he_normal",
    padding: str = "same",
) -> Tensor:
    """Construct a DNA convolution block.

    Args:
        inputs: Input tensor [batch_size, seq_length, features]
        filters: Conv1D filters
        kernel_size: Conv1D kernel_size
        activation: relu/gelu/etc
        stride: Conv1D stride
        l2_scale: L2 regularization weight
        dropout: Dropout rate probability
        conv_type: Conv1D layer type
        pool_size: Max pool width
        norm_type: Apply batch or layer normalization
        bn_momentum: BatchNorm momentum

    Returns:
        Output tensor [batch_size, seq_length, features]
    """
    current = inputs

    # Choose convolution type
    if conv_type == "separable":
        conv_layer = keras.layers.SeparableConv1D
    else:
        conv_layer = keras.layers.Conv1D

    if filters is None:
        filters = inputs.shape[-1]

    # Determine bias
    if use_bias is None:
        use_bias = norm_type is None and not residual

    # Convolution
    current = conv_layer(
        filters=filters,
        kernel_size=kernel_size,
        strides=stride,
        padding=padding,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=keras.regularizers.l2(l2_scale) if l2_scale > 0 else None,
    )(current)

    # Squeeze-excitation
    if se:
        current = SqueezeExcite(rank=8)(current)

    if residual:
        # Residual conv block
        rcurrent = conv_nac(
            current,
            activation=activation,
            l2_scale=l2_scale,
            dropout=dropout_residual,
            conv_type=conv_type,
            norm_type=norm_type,
            se=se,
            bn_momentum=bn_momentum,
            kernel_initializer=kernel_initializer,
        )

        # Residual add with scale
        rcurrent = Scale()(rcurrent)
        current = keras.layers.Add()([current, rcurrent])

    else:
        # Normalize
        if norm_type == "batch":
            current = keras.layers.BatchNormalization(momentum=bn_momentum)(current)
        elif norm_type == "layer":
            current = keras.layers.LayerNormalization()(current)

        # Activation
        current = activate_layer(current, activation)

    # Dropout
    if dropout > 0:
        current = keras.layers.Dropout(rate=dropout)(current)

    # Pool
    if pool_size > 1:
        if pool_type == "softmax":
            current = SoftmaxPool1D(pool_size=pool_size)(current)
        elif pool_type == "avg":
            current = keras.layers.AvgPool1D(pool_size=pool_size, padding=padding)(current)
        else:
            current = keras.layers.MaxPool1D(pool_size=pool_size, padding=padding)(current)

    return current


def conv_block_2d(
    inputs: Tensor,
    filters: int = 128,
    activation: str = "relu",
    conv_type: str = "standard",
    kernel_size: int = 1,
    stride: int = 1,
    dilation_rate: int = 1,
    l2_scale: float = 0,
    dropout: float = 0,
    pool_size: int = 1,
    norm_type: Optional[str] = None,
    bn_momentum: float = 0.99,
    norm_gamma: str = "ones",
    kernel_initializer: str = "he_normal",
    symmetric: bool = False,
) -> Tensor:
    """Construct a single 2D convolution block."""
    current = inputs

    # Activation
    current = activate_layer(current, activation)

    # Choose convolution type
    if conv_type == "separable":
        conv_layer = keras.layers.SeparableConv2D
    else:
        conv_layer = keras.layers.Conv2D

    # Convolution
    current = conv_layer(
        filters=filters,
        kernel_size=kernel_size,
        strides=stride,
        padding="same",
        use_bias=(norm_type is None),
        dilation_rate=dilation_rate,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=keras.regularizers.l2(l2_scale) if l2_scale > 0 else None,
    )(current)

    # Normalize
    if norm_type == "batch":
        current = keras.layers.BatchNormalization(
            momentum=bn_momentum,
            gamma_initializer=norm_gamma,
        )(current)
    elif norm_type == "layer":
        current = keras.layers.LayerNormalization(gamma_initializer=norm_gamma)(current)

    # Dropout
    if dropout > 0:
        current = keras.layers.Dropout(rate=dropout)(current)

    # Pool
    if pool_size > 1:
        current = keras.layers.MaxPool2D(pool_size=pool_size, padding="same")(current)

    # Symmetric
    if symmetric:
        current = Symmetrize2D()(current)

    return current


############################################################
# Towers
############################################################


def conv_tower(
    inputs: Tensor,
    filters_init: int,
    filters_end: Optional[int] = None,
    filters_mult: Optional[float] = None,
    divisible_by: int = 1,
    repeat: int = 1,
    reprs: Optional[List[Tensor]] = None,
    **kwargs,
) -> Tensor:
    """Construct a reducing convolution tower.

    Args:
        inputs: Input tensor [batch_size, seq_length, features]
        filters_init: Initial Conv1D filters
        filters_end: End Conv1D filters
        filters_mult: Multiplier for Conv1D filters
        divisible_by: Round filters to be divisible by
        repeat: Tower repetitions
        reprs: List to save representations

    Returns:
        Output tensor [batch_size, seq_length, features]
    """
    def _round(x):
        return int(round(x / divisible_by) * divisible_by)

    current = inputs
    rep_filters = float(filters_init)

    # Determine multiplier
    if filters_mult is None:
        assert filters_end is not None
        filters_mult = math.exp(math.log(filters_end / filters_init) / (repeat - 1))

    for ri in range(repeat):
        # Convolution
        current = conv_block(current, filters=_round(rep_filters), **kwargs)

        # Save representation
        if reprs is not None:
            reprs.append(current)

        # Update filters
        rep_filters *= filters_mult

    return current


def conv_tower_nac(
    inputs: Tensor,
    filters_init: int,
    filters_end: Optional[int] = None,
    filters_mult: Optional[float] = None,
    divisible_by: int = 1,
    repeat: int = 1,
    reprs: Optional[List[Tensor]] = None,
    **kwargs,
) -> Tensor:
    """Construct a reducing convolution tower using NAC blocks."""
    def _round(x):
        return int(round(x / divisible_by) * divisible_by)

    current = inputs
    rep_filters = float(filters_init)

    # Determine multiplier
    if filters_mult is None:
        assert filters_end is not None
        filters_mult = math.exp(math.log(filters_end / filters_init) / (repeat - 1))

    for ri in range(repeat):
        # Convolution
        current = conv_nac(current, filters=_round(rep_filters), **kwargs)

        # Save representation
        if reprs is not None:
            reprs.append(current)

        # Update filters
        rep_filters *= filters_mult

    return current


def res_tower(
    inputs: Tensor,
    filters_init: int,
    filters_end: Optional[int] = None,
    filters_mult: Optional[float] = None,
    kernel_size: int = 1,
    dropout: float = 0,
    pool_size: int = 2,
    pool_type: str = "max",
    divisible_by: int = 1,
    repeat: int = 1,
    num_convs: int = 2,
    reprs: Optional[List[Tensor]] = None,
    **kwargs,
) -> Tensor:
    """Construct a residual tower with pooling between blocks."""
    def _round(x):
        return int(round(x / divisible_by) * divisible_by)

    current = inputs
    rep_filters = float(filters_init)

    # Determine multiplier
    if filters_mult is None:
        assert filters_end is not None
        filters_mult = math.exp(math.log(filters_end / filters_init) / (repeat - 1))

    for ri in range(repeat):
        rep_filters_int = _round(rep_filters)

        # Initial conv
        current0 = conv_nac(
            current,
            filters=rep_filters_int,
            kernel_size=kernel_size,
            **kwargs
        )

        # Subsequent convs
        current = current0
        for ci in range(1, num_convs):
            current = conv_nac(
                current,
                filters=rep_filters_int,
                **kwargs
            )

        # Residual add with scale
        if num_convs > 1:
            current = Scale()(current)
            current = keras.layers.Add()([current0, current])

        # Dropout
        if dropout > 0:
            current = keras.layers.Dropout(rate=dropout)(current)

        # Save representation
        if reprs is not None:
            reprs.append(current)

        # Pool
        if pool_size > 1:
            if pool_type == "softmax":
                current = SoftmaxPool1D(pool_size=pool_size)(current)
            elif pool_type == "avg":
                current = keras.layers.AvgPool1D(pool_size=pool_size, padding="same")(current)
            else:
                current = keras.layers.MaxPool1D(pool_size=pool_size, padding="same")(current)

        # Update filters
        rep_filters *= filters_mult

    return current


############################################################
# Dense Blocks
############################################################


def dense_block(
    inputs: Tensor,
    units: Optional[int] = None,
    activation: str = "relu",
    activation_end: Optional[str] = None,
    flatten: bool = False,
    dropout: float = 0,
    l2_scale: float = 0,
    l1_scale: float = 0,
    residual: bool = False,
    norm_type: Optional[str] = None,
    bn_momentum: float = 0.99,
    norm_gamma: Optional[str] = None,
    kernel_initializer: str = "he_normal",
    **kwargs,
) -> Tensor:
    """Construct a dense (fully connected) block.

    Args:
        inputs: Input tensor
        units: Dense units
        activation: Activation function
        activation_end: Activation after other operations
        flatten: Flatten across positional axis
        dropout: Dropout rate probability
        l2_scale: L2 regularization weight
        l1_scale: L1 regularization weight
        residual: Residual connection boolean
        norm_type: Apply batch or layer normalization
        bn_momentum: BatchNorm momentum
        norm_gamma: BatchNorm gamma initializer
        kernel_initializer: Weight initialization

    Returns:
        Output tensor
    """
    current = inputs

    if units is None:
        units = inputs.shape[-1]

    # Activation
    current = activate_layer(current, activation)

    # Flatten
    if flatten:
        current = keras.layers.Flatten()(current)

    # Dense
    current = keras.layers.Dense(
        units=units,
        use_bias=(norm_type is None),
        kernel_initializer=kernel_initializer,
        kernel_regularizer=keras.regularizers.l1_l2(l1_scale, l2_scale) if l1_scale > 0 or l2_scale > 0 else None,
    )(current)

    # Normalize
    if norm_gamma is None:
        norm_gamma = "zeros" if residual else "ones"
    if norm_type == "batch":
        current = keras.layers.BatchNormalization(
            momentum=bn_momentum,
            gamma_initializer=norm_gamma,
        )(current)
    elif norm_type == "layer":
        current = keras.layers.LayerNormalization(gamma_initializer=norm_gamma)(current)

    # Dropout
    if dropout > 0:
        current = keras.layers.Dropout(rate=dropout)(current)

    # Residual add
    if residual:
        # Need matching shapes - apply projection if needed
        if inputs.shape[-1] != units:
            inputs_proj = keras.layers.Dense(units)(inputs)
        else:
            inputs_proj = inputs
        current = keras.layers.Add()([inputs_proj, current])

    # End activation
    if activation_end is not None:
        current = activate_layer(current, activation_end)

    return current


def final(
    inputs: Tensor,
    units: int,
    activation: str = "linear",
    flatten: bool = False,
    kernel_initializer: str = "he_normal",
    l2_scale: float = 0,
    l1_scale: float = 0,
    **kwargs,
) -> Tensor:
    """Final simple transformation before comparison to targets.

    Args:
        inputs: Input tensor [batch_size, seq_length, features]
        units: Dense units
        activation: Output activation
        flatten: Flatten positional axis
        kernel_initializer: Weight initialization
        l2_scale: L2 regularization weight
        l1_scale: L1 regularization weight

    Returns:
        Output tensor [batch_size, seq_length(?), units]
    """
    current = inputs

    # Flatten
    if flatten:
        current = keras.layers.Flatten()(current)

    # Dense
    current = keras.layers.Dense(
        units=units,
        use_bias=True,
        activation=activation,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=keras.regularizers.l1_l2(l1_scale, l2_scale) if l1_scale > 0 or l2_scale > 0 else None,
    )(current)

    return current


############################################################
# Dilated Blocks
############################################################


def dilated_residual(
    inputs: Tensor,
    filters: int,
    kernel_size: int = 3,
    rate_mult: float = 2,
    dropout: float = 0,
    repeat: int = 1,
    conv_type: str = "standard",
    norm_type: Optional[str] = None,
    round_dilation: bool = False,
    **kwargs,
) -> Tensor:
    """Construct a residual dilated convolution block."""
    current = inputs
    dilation_rate = 1.0

    for ri in range(repeat):
        rep_input = current

        # Dilated conv
        current = conv_block(
            current,
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=int(round(dilation_rate)),
            conv_type=conv_type,
            norm_type=norm_type,
            norm_gamma="ones",
            **kwargs,
        )

        # Return conv
        current = conv_block(
            current,
            filters=rep_input.shape[-1],
            dropout=dropout,
            norm_type=norm_type,
            norm_gamma="zeros",
            **kwargs,
        )

        # Residual add with scale
        if norm_type is None:
            current = Scale()(current)

        current = keras.layers.Add()([rep_input, current])

        # Update dilation rate
        dilation_rate *= rate_mult
        if round_dilation:
            dilation_rate = round(dilation_rate)

    return current


def dilated_residual_2d(
    inputs: Tensor,
    filters: int,
    kernel_size: int = 3,
    rate_mult: float = 2,
    dropout: float = 0,
    repeat: int = 1,
    symmetric: bool = True,
    **kwargs,
) -> Tensor:
    """Construct a residual dilated convolution block for 2D."""
    current = inputs
    dilation_rate = 1.0

    for ri in range(repeat):
        rep_input = current

        # Dilated conv
        current = conv_block_2d(
            current,
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=int(round(dilation_rate)),
            norm_gamma="ones",
            **kwargs,
        )

        # Return conv
        current = conv_block_2d(
            current,
            filters=rep_input.shape[-1],
            dropout=dropout,
            norm_gamma="zeros",
            **kwargs,
        )

        # Residual add
        current = keras.layers.Add()([rep_input, current])

        # Enforce symmetry
        if symmetric:
            current = Symmetrize2D()(current)

        # Update dilation rate
        dilation_rate *= rate_mult

    return current


############################################################
# 2D Operations
############################################################


def one_to_two(
    inputs: Tensor,
    operation: str = "mean",
    **kwargs,
) -> Tensor:
    """Convert 1D to 2D contact map."""
    return OneToTwo(operation=operation)(inputs)


def symmetrize_2d(
    inputs: Tensor,
    **kwargs,
) -> Tensor:
    """Symmetrize a 2D tensor."""
    return Symmetrize2D()(inputs)


def upper_tri(
    inputs: Tensor,
    diagonal_offset: int = 2,
    **kwargs,
) -> Tensor:
    """Extract upper triangular part."""
    return UpperTri(diagonal_offset=diagonal_offset)(inputs)


def concat_dist_2d(
    inputs: Tensor,
    **kwargs,
) -> Tensor:
    """Concatenate distance features to 2D input."""
    return ConcatDist2D()(inputs)


def cropping_2d(
    inputs: Tensor,
    cropping: int,
    **kwargs,
) -> Tensor:
    """Crop a 2D tensor."""
    return keras.layers.Cropping2D(cropping=cropping)(inputs)


def squeeze_excite(
    inputs: Tensor,
    activation: str = "relu",
    additive: bool = False,
    bottleneck_ratio: int = 8,
    **kwargs,
) -> Tensor:
    """Squeeze-and-excitation block."""
    return SqueezeExcite(
        activation=activation,
        additive=additive,
        rank=bottleneck_ratio,
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
]
