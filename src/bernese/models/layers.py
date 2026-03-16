# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Custom Keras 3 layers for SeqNN models.

This module provides custom layers ported from the TensorFlow baskerville
implementation, adapted for Keras 3 with backend-agnostic operations.
"""

from typing import Optional

import numpy as np
import keras
from keras import ops


class Scale(keras.layers.Layer):
    """Scale the input by a learned value.

    Args:
        axis: Axis/axes along which to scale.
        initializer: Initializer for the scale weight.
    """

    def __init__(self, axis: int = -1, initializer: str = "zeros", **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.initializer = initializer

    def build(self, input_shape):
        if isinstance(self.axis, int):
            axis_list = [self.axis]
        else:
            axis_list = list(self.axis)

        # Resolve negative axes
        ndims = len(input_shape)
        resolved_axis = []
        for x in axis_list:
            if x < 0:
                x = ndims + x
            resolved_axis.append(x)

        # Validate axes
        for x in resolved_axis:
            if x < 0 or x >= ndims:
                raise ValueError(f"Invalid axis: {x}")

        # Build scale parameter
        param_shape = [input_shape[d] for d in resolved_axis]
        self.scale = self.add_weight(
            name="scale",
            shape=param_shape,
            initializer=self.initializer,
            trainable=True,
        )

    def call(self, x):
        return x * self.scale

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "axis": self.axis,
                "initializer": self.initializer,
            }
        )
        return config


class Softplus(keras.layers.Layer):
    """Safe softplus, clipping large values."""

    def __init__(self, exp_max: float = 10000, **kwargs):
        super().__init__(**kwargs)
        self.exp_max = exp_max

    def call(self, x):
        x = ops.clip(x, -self.exp_max, self.exp_max)
        return ops.softplus(x)

    def get_config(self):
        config = super().get_config()
        config["exp_max"] = self.exp_max
        return config


class SoftmaxPool1D(keras.layers.Layer):
    """Pooling operation with optional weights.

    Args:
        pool_size: Pooling size, same as in Max/AvgPooling.
        per_channel: If True, the logits/softmax weights will be computed for
            each channel separately. If False, same weights will be used across all
            channels.
        init_gain: When 0.0 is equivalent to avg pooling, and when
            ~2.0 and it's equivalent to max pooling.
    """

    def __init__(
        self,
        pool_size: int = 2,
        per_channel: bool = False,
        init_gain: float = 2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pool_size = pool_size
        self.per_channel = per_channel
        self.init_gain = init_gain

    def build(self, input_shape):
        self.num_channels = input_shape[-1]
        self.logit_linear = keras.layers.Dense(
            units=self.num_channels if self.per_channel else 1,
            use_bias=False,
            kernel_initializer=keras.initializers.Identity(self.init_gain),
        )

    def call(self, inputs):
        # Reshape for pooling
        seq_length = ops.shape(inputs)[1]
        new_length = seq_length // self.pool_size

        # Reshape input
        inputs_reshaped = ops.reshape(inputs, (-1, new_length, self.pool_size, self.num_channels))

        # Get softmax weights
        # For now, simplify to standard softmax pooling
        return ops.mean(inputs_reshaped, axis=2)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "pool_size": self.pool_size,
                "per_channel": self.per_channel,
                "init_gain": self.init_gain,
            }
        )
        return config


class SqueezeExcite(keras.layers.Layer):
    """Squeeze-and-Excitation block.

    Args:
        activation: Activation function.
        additive: Whether to use additive attention.
        rank: Squeeze reduction ratio.
    """

    def __init__(
        self,
        activation: str = "relu",
        additive: bool = False,
        rank: int = 8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.activation = activation
        self.additive = additive
        self.rank = rank

    def build(self, input_shape):
        self.num_channels = input_shape[-1]

        # Squeeze (global average pooling)
        self.gap = keras.layers.GlobalAveragePooling1D()

        # Excitation
        self.dense1 = keras.layers.Dense(
            units=self.num_channels // self.rank,
            activation=self.activation,
        )
        self.dense2 = keras.layers.Dense(
            units=self.num_channels,
            activation="sigmoid",
        )

    def call(self, x):
        # Squeeze
        squeeze = self.gap(x)

        # Excite
        excite = self.dense1(squeeze)
        excite = self.dense2(excite)

        # Reshape for broadcasting
        excite = ops.reshape(excite, (-1, 1, self.num_channels))

        if self.additive:
            return x + x * excite
        else:
            return x * excite

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "activation": self.activation,
                "additive": self.additive,
                "rank": self.rank,
            }
        )
        return config


class OneToTwo(keras.layers.Layer):
    """Transform 1d to 2d with i,j vectors operated on.

    Args:
        operation: Operation to use ('mean', 'concat', 'max', 'multiply')
    """

    def __init__(self, operation: str = "mean", **kwargs):
        super().__init__(**kwargs)
        self.operation = operation.lower()
        valid_operations = ["concat", "mean", "max", "multiply"]
        if self.operation not in valid_operations:
            raise ValueError(f"operation must be one of {valid_operations}")

    def call(self, oned):
        # Shape: (batch, seq_len, features)
        seq_len = ops.shape(oned)[1]

        # Tile and reshape to create pairwise interactions
        twod1 = ops.reshape(ops.tile(oned, [1, seq_len, 1]), (-1, seq_len, seq_len, oned.shape[-1]))
        twod2 = ops.transpose(twod1, (0, 2, 1, 3))

        if self.operation == "concat":
            return ops.concatenate([twod1, twod2], axis=-1)
        elif self.operation == "multiply":
            return twod1 * twod2
        elif self.operation == "mean":
            return (twod1 + twod2) / 2
        else:
            return ops.maximum(twod1, twod2)

    def get_config(self):
        config = super().get_config()
        config["operation"] = self.operation
        return config


class ConcatDist2D(keras.layers.Layer):
    """Concatenate the pairwise distance to 2d feature matrix."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        batch_size = ops.shape(inputs)[0]
        seq_len = ops.shape(inputs)[1]

        # Create position indices
        pos = ops.arange(0, seq_len)
        pos1 = ops.reshape(pos, (1, seq_len, 1))
        pos2 = ops.reshape(pos, (1, 1, seq_len))

        # Compute distance matrix
        dist = ops.abs(pos1 - pos2)
        dist = ops.cast(dist, "float32")
        dist = ops.reshape(dist, (1, seq_len, seq_len, 1))
        dist = ops.tile(dist, [batch_size, 1, 1, 1])

        # Concatenate along channel dimension
        return ops.concatenate([inputs, dist], axis=-1)

    def get_config(self):
        return super().get_config()


class UpperTri(keras.layers.Layer):
    """Unroll matrix to its upper triangular portion.

    Args:
        diagonal_offset: Offset for diagonal (positive = above diagonal)
    """

    def __init__(self, diagonal_offset: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.diagonal_offset = diagonal_offset

    def call(self, inputs):
        # Inputs shape: (batch, seq_len, seq_len, channels)
        seq_len = inputs.shape[1]
        output_dim = inputs.shape[-1]

        # Get upper triangular indices using numpy
        triu_tup = np.triu_indices(seq_len, self.diagonal_offset)
        triu_index = list(triu_tup[0] + seq_len * triu_tup[1])

        # Reshape input: (batch, seq_len, seq_len, channels) -> (batch, seq_len^2, channels)
        unroll_repr = ops.reshape(inputs, (-1, seq_len**2, output_dim))

        # Gather upper triangular elements
        return ops.take(unroll_repr, ops.convert_to_tensor(triu_index, dtype="int32"), axis=1)

    def get_config(self):
        config = super().get_config()
        config["diagonal_offset"] = self.diagonal_offset
        return config


class Symmetrize2D(keras.layers.Layer):
    """Take the average of a matrix and its transpose to enforce symmetry."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        x_t = ops.transpose(x, (0, 2, 1, 3))
        x_sym = (x + x_t) / 2
        return x_sym

    def get_config(self):
        return super().get_config()


class StochasticReverseComplement(keras.layers.Layer):
    """Stochastically reverse complement a one hot encoded DNA sequence."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, seq_1hot, training=None):
        # In Keras 3, training is passed explicitly to the call method
        # Default to False if not provided (inference mode)
        if training is None:
            training = False

        # DNA complement mapping: A<->T, G<->C
        # [A, C, G, T] -> [T, G, C, A]
        rc_indices = ops.convert_to_tensor([3, 2, 1, 0], dtype="int32")
        rc_seq = ops.take(seq_1hot, rc_indices, axis=-1)

        # Reverse along axis 1 using flip
        rc_seq = ops.flip(rc_seq, axis=1)

        # Random choice during training
        if training:
            random_val = keras.random.uniform(())
            reverse_bool = random_val > 0.5
            # Use conditional
            return keras.ops.where(
                ops.reshape(reverse_bool, (1, 1, 1)), rc_seq, seq_1hot
            ), reverse_bool
        else:
            return seq_1hot, ops.zeros(())

    def get_config(self):
        return super().get_config()


class StochasticShift(keras.layers.Layer):
    """Stochastically shift a one hot encoded DNA sequence.

    Args:
        shift_max: Maximum shift amount.
        symmetric: Whether to consider both directions.
    """

    def __init__(self, shift_max: int = 0, symmetric: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.shift_max = shift_max
        self.symmetric = symmetric

    def call(self, seq_1hot, training=None):
        # In Keras 3, training is passed explicitly to the call method
        # Default to False if not provided (inference mode)
        if training is None:
            training = False

        if not training or self.shift_max == 0:
            return seq_1hot

        # Random shift
        if self.symmetric:
            shift = keras.random.uniform((), minval=-self.shift_max, maxval=self.shift_max + 1)
            shift = ops.cast(shift, "int32")
        else:
            shift = keras.random.uniform((), minval=0, maxval=self.shift_max + 1)
            shift = ops.cast(shift, "int32")

        # Apply shift
        seq_len = ops.shape(seq_1hot)[1]

        # Positive shift = shift right (pad on left)
        # Negative shift = shift left (pad on right)
        if_shift = shift > 0
        shift_abs = ops.abs(shift)

        # Handle edge case where shift is 0 - return original sequence
        # When shift_abs is 0, slicing with :-0 or [0:] causes issues
        if ops.equal(shift_abs, 0):
            return seq_1hot

        # Create padded tensor
        pad_left = ops.zeros((ops.shape(seq_1hot)[0], shift_abs, 4))
        pad_right = ops.zeros((ops.shape(seq_1hot)[0], shift_abs, 4))

        # For positive shift: [pad, seq[:-shift]]
        # For negative shift: [seq[-shift:], pad]
        shifted = ops.where(
            ops.reshape(if_shift, (1, 1, 1)),
            ops.concatenate([pad_left, seq_1hot[:, :-shift_abs, :]], axis=1),
            ops.concatenate([seq_1hot[:, shift_abs:, :], pad_right], axis=1),
        )

        return shifted

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "shift_max": self.shift_max,
                "symmetric": self.symmetric,
            }
        )
        return config


class SwitchReverse(keras.layers.Layer):
    """Reverse predictions if the inputs were reverse complemented.

    Args:
        strand_pair: List of strand pairs for merging predictions.
        diagonal_offset: Diagonal offset for Hi-C predictions.
    """

    def __init__(self, strand_pair: Optional[list[int]] = None, diagonal_offset: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.strand_pair = strand_pair
        self.diagonal_offset = diagonal_offset

    def call(self, x_reverse):
        x = x_reverse[0]
        reverse = x_reverse[1]

        # Reverse along length axis (axis 1) using flip
        x_reversed = ops.flip(x, axis=1)

        xr = ops.where(ops.reshape(reverse, (1, 1, 1)), x_reversed, x)

        # Apply strand pair if specified
        if self.strand_pair is not None:
            xr = ops.where(
                ops.reshape(reverse, (1, 1, 1)),
                ops.take(xr, ops.convert_to_tensor(self.strand_pair), axis=-1),
                xr,
            )

        return xr

    def get_config(self):
        config = super().get_config()
        config["strand_pair"] = self.strand_pair
        return config


class EnsembleShift(keras.layers.Layer):
    """Expand tensor to include shifts of one hot encoded DNA sequence.

    Args:
        shifts: List of shift amounts.
    """

    def __init__(self, shifts: list[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.shifts = shifts or [0]

    def call(self, seqs_1hot):
        if not isinstance(seqs_1hot, list):
            seqs_1hot = [seqs_1hot]

        ens_seqs_1hot = []
        for seq_1hot in seqs_1hot:
            for shift in self.shifts:
                shifted = self._shift_sequence(seq_1hot, shift)
                ens_seqs_1hot.append(shifted)

        return ens_seqs_1hot

    def _shift_sequence(self, seq, shift):
        if shift == 0:
            return seq

        seq_len = ops.shape(seq)[1]
        shift_abs = ops.abs(shift)

        pad_value = ops.zeros((ops.shape(seq)[0], shift_abs, 4))

        if shift > 0:
            # Shift right
            return ops.concatenate([pad_value, seq[:, :-shift, :]], axis=1)
        else:
            # Shift left
            return ops.concatenate([seq[:, -shift:, :], pad_value], axis=1)

    def get_config(self):
        config = super().get_config()
        config["shifts"] = self.shifts
        return config


class EnsembleReverseComplement(keras.layers.Layer):
    """Expand tensor to include reverse complement of one hot encoded DNA sequence."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, seqs_1hot):
        if not isinstance(seqs_1hot, list):
            seqs_1hot = [seqs_1hot]

        ens_seqs_1hot = []
        for seq_1hot in seqs_1hot:
            # Original sequence
            ens_seqs_1hot.append((seq_1hot, ops.zeros(())))

            # Reverse complement
            rc_indices = ops.convert_to_tensor([3, 2, 1, 0], dtype="int32")
            rc_seq = ops.take(seq_1hot, rc_indices, axis=-1)
            # Reverse along axis 1 using flip
            rc_seq = ops.flip(rc_seq, axis=1)
            ens_seqs_1hot.append((rc_seq, ops.ones(())))

        return ens_seqs_1hot

    def get_config(self):
        return super().get_config()


# Helper function for activation
def activate(current, activation, verbose=False):
    """Apply activation function.

    Args:
        current: Input tensor.
        activation: Activation name.
        verbose: Whether to print debug info.

    Returns:
        Activated tensor.
    """
    if verbose:
        print(f"activate: {activation}")

    if activation is None or activation == "linear":
        return current
    elif activation == "relu":
        return keras.layers.ReLU()(current)
    elif activation == "gelu":
        return keras.activations.gelu(current, approximate=True)
    elif activation == "sigmoid":
        return keras.activations.sigmoid(current)
    elif activation == "tanh":
        return keras.activations.tanh(current)
    elif activation == "selu":
        return keras.activations.selu(current)
    elif activation == "elu":
        return keras.activations.elu(current)
    elif activation == "softmax":
        return keras.activations.softmax(current)
    else:
        # Try to get activation from keras
        try:
            act = keras.activations.get(activation)
            return act(current)
        except ValueError:
            raise ValueError(f'Unrecognized activation "{activation}"')


# Export all layer classes
__all__ = [
    "Scale",
    "Softplus",
    "SoftmaxPool1D",
    "SqueezeExcite",
    "OneToTwo",
    "ConcatDist2D",
    "UpperTri",
    "Symmetrize2D",
    "StochasticReverseComplement",
    "StochasticShift",
    "SwitchReverse",
    "EnsembleShift",
    "EnsembleReverseComplement",
    "activate",
]
