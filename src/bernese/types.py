# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Type definitions for Bernese configuration schemas.

This module provides TypedDict definitions for type-safe configuration handling
across the Bernese project.
"""

from typing import TYPE_CHECKING, TypedDict, Literal

# Activation functions
ActivationType = Literal[
    "relu", "gelu", "sigmoid", "tanh", "elu", "selu", "softplus", "linear", "none"
]

# Normalization types
NormType = Literal["batch", "layer", "batch_sync", "none"]

# Pooling types
PoolType = Literal["max", "avg", "softmax"]

# Padding types
PaddingType = str  # Can be "same" or an integer

# Padding literal (for type-safe "same" padding)
PaddingLiteral = Literal["same"]

# Optimizer types
OptimizerType = Literal["adam", "adamw", "sgd", "momentum"]

# Loss function types
LossType = Literal["mse", "bce", "poisson", "poisson_kl", "poisson_multinomial", "mse_udot"]

# Decay types for learning rate
DecayType = Literal["exponential", "linear"]

# Block type names
BlockName = Literal[
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

# OneToTwo operations
OneToTwoOperation = Literal["mean", "outer", "product"]

# Symmetrize modes
SymmetrizeMode = Literal["mean", "max", "min"]

# AUC curve types
AUCCurve = Literal["ROC", "PR"]

# Metric names
MetricName = Literal["pearsonr", "r2", "auroc", "auprc", "val_loss"]

if TYPE_CHECKING:
    import torch
    import numpy as np
    from torch import nn
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import _LRScheduler
    from torch.utils.data import DataLoader
    from collections.abc import Callable

    # Proper type aliases
    Tensor = torch.Tensor
    Module = nn.Module
    Optimizer = Optimizer
    Scheduler = _LRScheduler
    DataLoader = DataLoader
    NPArray = np.ndarray
    TransformCallable = Callable[[Tensor], Tensor]
else:
    # Runtime fallbacks (for when torch/numpy aren't available at import time)
    # These use type: comments for backwards compatibility with older Python versions
    Tensor = "torch.Tensor"  # type: ignore[misc,assignment]
    Module = "nn.Module"  # type: ignore[misc,assignment]
    Optimizer = "torch.optim.Optimizer"  # type: ignore[misc,assignment]
    Scheduler = "torch.optim.lr_scheduler._LRScheduler"  # type: ignore[misc,assignment]
    DataLoader = "torch.utils.data.DataLoader"  # type: ignore[misc,assignment]
    NPArray = "numpy.ndarray"  # type: ignore[misc,assignment]
    TransformCallable = "Callable[[Tensor], Tensor]"  # type: ignore[misc,assignment]

class BlockConfig(TypedDict, total=False):
    """Configuration for a single neural network block.

    Attributes:
        name: Block type name (e.g., 'conv_block', 'conv_tower', 'final')
        filters: Number of output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        dilation: Dilation rate
        padding: Padding mode ('same' or integer)
        activation: Activation function name
        norm_type: Normalization type ('batch', 'layer', or None)
        dropout: Dropout probability
        residual: Whether to add residual connection
        pool_size: Pooling size
        pool_type: Type of pooling ('max', 'avg', 'softmax')
    """

    name: BlockName
    filters: int | None
    kernel_size: int
    stride: int
    dilation: int
    padding: PaddingType
    activation: ActivationType
    norm_type: NormType
    dropout: float
    residual: bool
    l2_scale: float
    bn_momentum: float
    activation_end: ActivationType
    pool_size: int
    pool_type: PoolType


class ConvTowerBlockConfig(BlockConfig):
    """Configuration for convolutional tower blocks.

    Additional attributes:
        filters_init: Initial number of filters
        filters_end: Target number of filters
        filters_mult: Multiplicative factor for filter growth
        divisible_by: Round filters to be divisible by this number
        repeat: Number of blocks in the tower
    """

    filters_init: int
    filters_end: int | None
    filters_mult: float | None
    divisible_by: int
    repeat: int


class FinalBlockConfig(BlockConfig):
    """Configuration for final prediction blocks.

    Additional attributes:
        out_units: Number of output units
        flatten: Whether to flatten spatial dimensions
    """

    out_units: int
    flatten: bool


class ConvTowerNACBlockConfig(ConvTowerBlockConfig):
    """Configuration for NAC-style tower blocks."""

    pass


class ResTowerBlockConfig(ConvTowerBlockConfig):
    """Configuration for residual tower blocks.

    Additional attributes:
        num_convs: Number of convolutions per residual block
    """

    num_convs: int


class UpperTriBlockConfig(BlockConfig):
    """Configuration for upper triangular extraction blocks.

    Additional attributes:
        diagonal_offset: Offset for diagonal
    """

    diagonal_offset: int


class Cropping2dBlockConfig(BlockConfig):
    """Configuration for 2D cropping blocks.

    Additional attributes:
        cropping: Number of pixels to crop from each side
    """

    cropping: int


class ConvBlock2dBlockConfig(BlockConfig):
    """Configuration for 2D convolution blocks."""

    pass


class DilatedResidual2dBlockConfig(BlockConfig):
    """Configuration for 2D dilated residual blocks."""

    pass


class OneToTwoBlockConfig(BlockConfig):
    """Configuration for 1D to 2D conversion blocks.

    Additional attributes:
        operation: Operation to use ('mean', 'outer', 'product')
    """

    operation: str


class ConcatDist2dBlockConfig(BlockConfig):
    """Configuration for distance feature concatenation blocks.

    Additional attributes:
        num_features: Number of distance features to add
    """

    num_features: int


class Symmetrize2dBlockConfig(BlockConfig):
    """Configuration for 2D symmetrization blocks.

    Additional attributes:
        mode: How to symmetrize ('mean', 'max', 'min')
    """

    mode: str


class ModelConfig(TypedDict, total=False):
    """Configuration for SeqNN model.

    Attributes:
        seq_length: Input sequence length
        seq_depth: Number of channels (4 for DNA)
        trunk: List of block configurations for the model trunk
        heads: List of head configurations for prediction heads
        num_targets: Number of prediction targets
        target_length: Output sequence length
        augment_rc: Whether to use reverse complement augmentation
        augment_shift: List of shift amounts for augmentation
        strand_pair: List of strand pairs for merging predictions
        activation: Default activation function
        verbose: Whether to print verbose output
    """

    seq_length: int
    seq_depth: int
    trunk: list[BlockConfig]
    heads: list[list[BlockConfig]]
    num_targets: int
    target_length: int
    augment_rc: bool
    augment_shift: list[int]
    strand_pair: list[tuple[int, int]]
    activation: ActivationType
    verbose: bool
    # Legacy/alternative keys
    head_hic: list[list[BlockConfig]]
    # Internal computed properties (not from config)
    _trunk_output_channels: int
    preds_triu: bool
    model_strides: list[int]
    target_lengths: list[int]
    target_crops: list[int]


class TrainingConfig(TypedDict, total=False):
    """Configuration for training.

    Attributes:
        loss: Loss function name
        optimizer: Optimizer type ('adam', 'adamw', 'sgd')
        learning_rate: Learning rate
        initial_learning_rate: Initial learning rate for schedulers
        maximal_learning_rate: Peak learning rate for cyclical LR
        final_learning_rate: Minimum learning rate for cyclical LR
        weight_decay: L2 weight decay
        adam_beta1: Adam beta1 parameter
        adam_beta2: Adam beta2 parameter
        momentum: SGD momentum
        best_metric: Metric to use for early stopping
        patience: Early stopping patience
        train_epochs_min: Minimum training epochs
        train_epochs_max: Maximum training epochs
        clip_norm: Gradient clipping norm
        global_clipnorm: Global gradient clipping norm
        # Loss-specific parameters
        spec_weight: Weight for specificity terms
        total_weight: Weight for total count terms
        weight_range: Range for position weights
        weight_exp: Exponent for position weight decay
        # Scheduler parameters
        train_epochs_cycle1: Epochs for first cycle
        warmup_steps: Warmup steps
        decay_type: Decay type ('exponential', 'linear')
        decay_steps: Decay steps
        decay_rate: Decay rate
    """

    loss: LossType
    optimizer: OptimizerType
    learning_rate: float
    initial_learning_rate: float
    maximal_learning_rate: float
    final_learning_rate: float
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    momentum: float
    best_metric: MetricName
    patience: int
    train_epochs_min: int
    train_epochs_max: int
    clip_norm: float | None
    global_clipnorm: float | None
    spec_weight: float
    total_weight: float
    weight_range: float
    weight_exp: int
    train_epochs_cycle1: int
    warmup_steps: int
    decay_type: DecayType
    decay_steps: int
    decay_rate: float
    num_targets: int
    batch_size: int
    num_workers: int
    shuffle_train: bool
    seq_length_crop: int
    pin_memory: bool
    warmup_epochs: int
    seed: int


class DataConfig(TypedDict, total=False):
    """Configuration for data loading.

    Attributes:
        data_dir: Path to data directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle_train: Whether to shuffle training data
        seq_length_crop: Crop length from sequence ends
        pin_memory: Whether to pin memory for GPU transfer
    """

    data_dir: str
    batch_size: int
    num_workers: int
    shuffle_train: bool
    seq_length_crop: int
    pin_memory: bool


class AugmentationConfig(TypedDict, total=False):
    """Configuration for data augmentation.

    Attributes:
        augment_rc: Whether to use reverse complement
        augment_shift: Shift amounts (int for max, or list)
        augment_rc_prob: Probability of reverse complement
    """

    augment_rc: bool
    augment_shift: int | list[int]
    augment_rc_prob: float


def validate_model_config(config: dict[str, object]) -> ModelConfig:
    """Validate and cast a model configuration dictionary.

    Args:
        config: Raw configuration dictionary

    Returns:
        Validated ModelConfig

    Raises:
        KeyError: If required keys are missing
        TypeError: If values have incorrect types
    """
    required_keys = ["seq_length", "seq_depth", "num_targets"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required key in model config: {key}")

    # Validate types
    if not isinstance(config["seq_length"], int):
        raise TypeError(f"seq_length must be int, got {type(config['seq_length'])}")
    if not isinstance(config["seq_depth"], int):
        raise TypeError(f"seq_depth must be int, got {type(config['seq_depth'])}")
    if not isinstance(config["num_targets"], int):
        raise TypeError(f"num_targets must be int, got {type(config['num_targets'])}")

    # Cast to ModelConfig ( TypedDict is a dict subtype )
    return config  # type: ignore[return-value]


def validate_training_config(config: dict[str, object]) -> TrainingConfig:
    """Validate and cast a training configuration dictionary.

    Args:
        config: Raw configuration dictionary

    Returns:
        Validated TrainingConfig
    """
    # Set defaults
    defaults: dict[str, object] = {
        "loss": "mse",
        "optimizer": "adam",
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "best_metric": "val_loss",
        "patience": 20,
        "train_epochs_min": 1,
        "train_epochs_max": 10000,
    }

    # Merge with provided config (all values in config override defaults)
    result: dict[str, object] = {**defaults, **config}

    # Cast to TrainingConfig (TypedDict is a dict subtype)
    return result  # type: ignore[return-value]
