# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""SeqNN model class for regulatory genomics predictions using Keras 3.

This module provides a Keras 3 implementation of the Sequence Neural Network (SeqNN)
model for regulatory activity prediction, ported from the TensorFlow baskerville
implementation.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Any

import keras

from bernese.models import blocks
from bernese.models import layers as custom_layers


@dataclass
class SeqNNConfig:
    """Configuration for SeqNN model.

    Attributes:
        seq_length: Input sequence length
        seq_depth: Number of channels (4 for DNA)
        trunk: List of block configurations for the model trunk
        heads: List of head configurations (list of lists of block dicts)
        num_targets: Number of prediction targets
        augment_rc: Whether to use reverse complement augmentation
        augment_shift: List of shift amounts for augmentation
        strand_pair: List of strand pairs for merging predictions
        activation: Default activation function
        l2_scale: L2 regularization weight
        l1_scale: L1 regularization weight
        kernel_initializer: Weight initialization
        norm_type: Normalization type
        bn_momentum: BatchNorm momentum
        verbose: Whether to print model summary
        diagonal_offset: Diagonal offset for Hi-C predictions
    """

    # Required fields
    seq_length: int = 1344
    seq_depth: int = 4
    trunk: List[dict] = field(default_factory=list)
    heads: List[List[dict]] = field(default_factory=list)
    num_targets: int = 1

    # Augmentation
    augment_rc: bool = False
    augment_shift: List[int] = field(default_factory=lambda: [0])
    strand_pair: List = field(default_factory=list)

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

    def __post_init__(self):
        """Validate and normalize configuration."""
        # Handle augment_shift as int (legacy format)
        if isinstance(self.augment_shift, int):
            self.augment_shift = [self.augment_shift]

        # Normalize trunk - ensure it's a list of dicts
        if not self.trunk:
            self.trunk = [
                {
                    "name": "conv_tower",
                    "filters_init": 48,
                    "filters_end": 512,
                    "repeat": 6,
                    "kernel_size": 3,
                    "norm_type": "batch",
                    "activation": "relu",
                }
            ]

        # Normalize heads - ensure it's list of lists
        if not self.heads:
            self.heads = [
                [
                    {
                        "name": "conv_block",
                        "filters": 256,
                        "kernel_size": 1,
                        "norm_type": "batch",
                        "activation": "relu",
                    },
                    {
                        "name": "conv_block",
                        "filters": 256,
                        "kernel_size": 1,
                        "norm_type": "batch",
                        "activation": "relu",
                    },
                    {"name": "final", "units": self.num_targets, "activation": "linear"},
                ]
            ]
        elif isinstance(self.heads, list) and len(self.heads) > 0:
            # Check if it's a list of dicts (single head) or list of lists
            if isinstance(self.heads[0], dict):
                self.heads = [self.heads]

        # Extend strand_pair if needed
        while len(self.strand_pair) < len(self.heads):
            self.strand_pair.append(None)

    @classmethod
    def from_dict(cls, params: dict) -> "SeqNNConfig":
        """Create config from dictionary, handling None values.

        Args:
            params: Dictionary of parameters.

        Returns:
            SeqNNConfig instance.
        """
        # Filter out None values to use defaults
        filtered = {k: v for k, v in params.items() if v is not None}

        # Handle legacy 'head_hic' as 'heads'
        if "head_hic" in filtered and "heads" not in filtered:
            filtered["heads"] = filtered.pop("head_hic")

        # Filter to only known fields
        known_fields = {
            "seq_length",
            "seq_depth",
            "trunk",
            "heads",
            "num_targets",
            "augment_rc",
            "augment_shift",
            "strand_pair",
            "activation",
            "l2_scale",
            "l1_scale",
            "kernel_initializer",
            "norm_type",
            "bn_momentum",
            "verbose",
            "diagonal_offset",
        }
        filtered = {k: v for k, v in filtered.items() if k in known_fields}

        return cls(**filtered)


class SeqNNBuilder:
    """Builder class for constructing SeqNN models.

    This class handles the model building logic, separating construction
    concerns from the model itself.
    """

    def __init__(self, config: SeqNNConfig):
        """Initialize builder with configuration.

        Args:
            config: SeqNN configuration.
        """
        self.config = config
        self.sequence = None
        self.trunk_output = None
        self.reverse_bool = None
        self.preds_triu = False
        self.reprs = []

    def build(self) -> "SeqNN":
        """Build the complete SeqNN model.

        Returns:
            Configured SeqNN instance with built models.
        """
        self._build_inputs()
        current = self._build_augmentation()
        current = self._build_trunk(current)
        current = custom_layers.activate(current, self.config.activation)

        self.trunk_output = current
        self._build_trunk_model()

        head_outputs = self._build_heads()
        models = self._create_models(head_outputs)

        # Create SeqNN instance
        seqnn = SeqNN.__new__(SeqNN)
        seqnn.config = self.config
        seqnn.model = models[0]
        seqnn.models = models
        seqnn.model_trunk = self.trunk_model
        seqnn.ensemble = None
        seqnn.preds_triu = self.preds_triu
        seqnn.reprs = self.reprs

        # Track geometry
        seqnn.model_strides = []
        seqnn.target_lengths = []
        seqnn.target_crops = []

        for model in models:
            stride_factor = 1
            for layer in model.layers:
                if hasattr(layer, "strides") and layer.strides[0] > 1:
                    stride_factor *= layer.strides[0]
                if hasattr(layer, "pool_size") and isinstance(layer.pool_size, int):
                    stride_factor *= layer.pool_size

            seqnn.model_strides.append(int(stride_factor))
            output_shape = model.output_shape
            target_length = output_shape[1] if len(output_shape) == 3 else output_shape[1]
            seqnn.target_lengths.append(target_length)
            target_full_length = self.config.seq_length // stride_factor
            seqnn.target_crops.append((target_full_length - target_length) // 2)

        if self.config.verbose:
            print(seqnn.model.summary())
            print("model_strides", seqnn.model_strides)
            print("target_lengths", seqnn.target_lengths)
            print("target_crops", seqnn.target_crops)

        return seqnn

    def _build_inputs(self):
        """Build input layers."""
        self.sequence = keras.Input(
            shape=(self.config.seq_length, self.config.seq_depth), name="sequence"
        )

    def _build_augmentation(self) -> keras.Layer:
        """Build augmentation layers.

        Returns:
            Augmented tensor.
        """
        current = self.sequence

        if self.config.augment_rc:
            current, self.reverse_bool = custom_layers.StochasticReverseComplement()(current)

        if self.config.augment_shift != [0]:
            shift_max = max(self.config.augment_shift)
            current = custom_layers.StochasticShift(shift_max=shift_max)(current)

        return current

    def _build_trunk(self, current: keras.Layer) -> keras.Layer:
        """Build trunk blocks.

        Args:
            current: Input tensor.

        Returns:
            Trunk output tensor.
        """
        for block_params in self.config.trunk:
            current = self._build_block(current, block_params)

        return current

    def _build_block(self, current: keras.Layer, block_params: dict) -> keras.Layer:
        """Build a single block.

        Args:
            current: Input tensor.
            block_params: Block configuration dictionary.

        Returns:
            Block output tensor.
        """
        block_name = block_params.get("name", "")

        # Track upper_tri for Hi-C predictions
        self.preds_triu = self.preds_triu or block_name == "upper_tri"

        # Determine block function
        block_func = None
        if block_name and block_name[0].isupper():
            if block_name in blocks.keras_func:
                block_func = blocks.keras_func[block_name]
        elif block_name in blocks.name_func:
            block_func = blocks.name_func[block_name]

        # Get allowed parameters for this block
        allowed_params = set()
        if block_func is not None:
            try:
                import inspect

                if hasattr(block_func, "__code__"):
                    allowed_params = set(block_func.__code__.co_varnames)
                elif hasattr(block_func, "__init__") and hasattr(block_func.__init__, "__code__"):
                    allowed_params = set(block_func.__init__.__code__.co_varnames)
            except Exception:
                pass

        # Start with block-specific params
        block_args = {k: v for k, v in block_params.items() if k != "name"}

        # Add global defaults only if they're in allowed params
        global_vars = [
            "activation",
            "batch_norm",
            "bn_momentum",
            "norm_type",
            "l2_scale",
            "l1_scale",
            "padding",
            "kernel_initializer",
        ]
        for gv in global_vars:
            if gv in allowed_params:
                gv_value = getattr(self.config, gv, None)
                if gv_value is not None and gv not in block_args:
                    block_args[gv] = gv_value

        # Save representations for towers
        if "tower" in block_name:
            block_args["reprs"] = self.reprs

        # Execute block
        if block_func is not None and block_name and block_name[0].isupper():
            current = block_func(**block_args)(current)
        elif block_name in blocks.name_func:
            block_func = blocks.name_func[block_name]
            if block_func is not None:
                current = block_func(current, **block_args)
            else:
                raise ValueError(f"Block {block_name} not implemented")
        else:
            raise ValueError(f"Unknown block: {block_name}")

        return current

    def _build_trunk_model(self):
        """Build trunk model."""
        self.trunk_model = keras.Model(
            inputs=self.sequence, outputs=self.trunk_output, name="trunk"
        )

    def _build_heads(self) -> List[keras.Layer]:
        """Build prediction heads.

        Returns:
            List of head output tensors.
        """
        head_outputs = []

        for hi, head in enumerate(self.config.heads):
            # Reset to trunk output
            current = self.trunk_output

            # Build blocks for this head
            for block_params in head:
                if isinstance(block_params, dict):
                    current = self._build_block(current, block_params)

            # Get strand pair for this head
            strand_pair = self.config.strand_pair[hi] if hi < len(self.config.strand_pair) else None

            # Transform back from reverse complement
            if self.config.augment_rc:
                if self.preds_triu:
                    current = custom_layers.SwitchReverse(
                        diagonal_offset=self.config.diagonal_offset
                    )([current, self.reverse_bool])
                else:
                    current = custom_layers.SwitchReverse(strand_pair=strand_pair)(
                        [current, self.reverse_bool]
                    )

            head_outputs.append(current)

        return head_outputs

    def _create_models(self, head_outputs: List[keras.Layer]) -> List[keras.Model]:
        """Create Keras models for each head.

        Args:
            head_outputs: List of head output tensors.

        Returns:
            List of Keras models.
        """
        models = []
        for ho in head_outputs:
            models.append(keras.Model(inputs=self.sequence, outputs=ho))

        return models


class SeqNN:
    """Sequence neural network model for regulatory activity prediction.

    This class builds a Keras model from a SeqNNConfig, using the builder pattern
    for clean separation of concerns.

    Args:
        config: SeqNNConfig instance or dict of parameters.
    """

    def __init__(self, config: SeqNNConfig | dict[str, Any]):
        # Handle dict input
        if isinstance(config, dict):
            config = SeqNNConfig.from_dict(config)

        # Build model using builder
        builder = SeqNNBuilder(config)
        built_seqnn = builder.build()

        # Copy attributes from built model
        self.config = built_seqnn.config
        self.model = built_seqnn.model
        self.models = built_seqnn.models
        self.model_trunk = built_seqnn.model_trunk
        self.ensemble = None
        self.preds_triu = built_seqnn.preds_triu
        self.reprs = built_seqnn.reprs
        self.model_strides = built_seqnn.model_strides
        self.target_lengths = built_seqnn.target_lengths
        self.target_crops = built_seqnn.target_crops

    @property
    def seq_length(self) -> int:
        return self.config.seq_length

    @property
    def seq_depth(self) -> int:
        return self.config.seq_depth

    @property
    def augment_rc(self) -> bool:
        return self.config.augment_rc

    @property
    def augment_shift(self) -> List[int]:
        return self.config.augment_shift

    @property
    def strand_pair(self) -> List:
        return self.config.strand_pair

    def get_num_targets(self, head_i: Optional[int] = None) -> int:
        """Return number of targets.

        Args:
            head_i: Optional head index.

        Returns:
            Number of targets.
        """
        if head_i is None:
            return self.model.output_shape[-1]
        else:
            return self.models[head_i].output_shape[-1]

    def __call__(self, x, head_i: Optional[int] = None, dtype="float32"):
        """Predict targets for single batch.

        Args:
            x: Input sequences.
            head_i: Optional head index.

        Returns:
            Predictions.
        """
        # Choose model
        if head_i is not None:
            model = self.models[head_i]
        else:
            model = self.model

        preds = model(x, training=False)

        # Convert to numpy - handle PyTorch tensors from Keras backend
        if hasattr(preds, "device") and hasattr(preds, "detach"):
            preds = preds.detach().cpu().numpy()
        elif hasattr(preds, "numpy"):
            preds = preds.numpy()

        return preds.astype(dtype)

    def predict(
        self,
        dataset,
        head_i: Optional[int] = None,
        **kwargs,
    ):
        """Predict targets for dataset.

        Args:
            dataset: Input dataset.
            head_i: Optional head index.

        Returns:
            Predictions.
        """
        if head_i is not None:
            model = self.models[head_i]
        else:
            model = self.model

        return model.predict(dataset, **kwargs)

    def evaluate(self, dataset, head_i: Optional[int] = None, **kwargs):
        """Evaluate model on dataset.

        Args:
            dataset: Input dataset.
            head_i: Optional head index.

        Returns:
            Evaluation metrics.
        """
        if head_i is not None:
            model = self.models[head_i]
        else:
            model = self.model

        return model.evaluate(dataset, **kwargs)

    def restore(self, model_file: str, head_i: int = 0, trunk: bool = False):
        """Restore weights from saved model.

        Args:
            model_file: Path to saved model.
            head_i: Head index.
            trunk: Whether to restore trunk only.
        """
        if trunk:
            self.model_trunk.load_weights(model_file)
        else:
            self.models[head_i].load_weights(model_file)
            if head_i == 0:
                self.model = self.models[head_i]

    def save(self, model_file: str, trunk: bool = False):
        """Save model weights to file.

        Args:
            model_file: Path to save model weights.
            trunk: Save trunk weights only.
        """
        if trunk:
            self.model_trunk.save_weights(model_file)
        else:
            self.model.save_weights(model_file)

    def build_ensemble(self, ensemble_rc: bool = False, ensemble_shifts: List[int] = None):
        """Build ensemble of models with augmented inputs.

        Args:
            ensemble_rc: Whether to include reverse complement.
            ensemble_shifts: List of shift amounts.
        """
        if ensemble_shifts is None:
            ensemble_shifts = [0]

        shift_bool = len(ensemble_shifts) > 1 or ensemble_shifts[0] != 0

        if not ensemble_rc and not shift_bool:
            return

        sequence = keras.Input(shape=(self.seq_length, self.seq_depth), name="sequence")
        sequences = [sequence]

        if shift_bool:
            sequences = custom_layers.EnsembleShift(shifts=ensemble_shifts)(sequences)

        if ensemble_rc:
            sequences_rev = custom_layers.EnsembleReverseComplement()(sequences)
        else:
            sequences_rev = [(seq, keras.ops.zeros(())) for seq in sequences]

        strand_pair = self.strand_pair[0] if self.strand_pair else None

        preds = []
        for seq, rp in sequences_rev:
            if self.preds_triu:
                pred = custom_layers.SwitchReverse(diagonal_offset=self.config.diagonal_offset)(
                    [self.model(seq), rp]
                )
            else:
                pred = custom_layers.SwitchReverse(strand_pair=strand_pair)([self.model(seq), rp])
            preds.append(pred)

        preds_avg = keras.layers.Average()(preds)
        self.ensemble = keras.Model(inputs=sequence, outputs=preds_avg)

    def build_slice(self, target_slice: Optional[List[int]] = None, target_sum: bool = False):
        """Slice and/or sum across tasks in graph.

        Args:
            target_slice: Indices of targets to keep.
            target_sum: Whether to sum across targets.
        """
        if target_slice is None and not target_sum:
            return

        sequence = keras.Input(shape=(self.seq_length, self.seq_depth), name="sequence")
        predictions = self.model(sequence)

        if target_slice is not None:
            predictions = keras.layers.Lambda(
                lambda x: keras.ops.take(x, keras.ops.convert_to_tensor(target_slice), axis=-1)
            )(predictions)

        if target_sum:
            predictions = keras.layers.Lambda(lambda x: keras.ops.sum(x, axis=-1, keepdims=True))(
                predictions
            )

        self.model = keras.Model(inputs=sequence, outputs=predictions)

    def get_output_shape(self, head_i: int = 0) -> tuple:
        """Get output shape for a head.

        Args:
            head_i: Head index.

        Returns:
            Output shape tuple.
        """
        return self.models[head_i].output_shape

    def __repr__(self) -> str:
        return f"SeqNN(seq_length={self.seq_length}, num_targets={self.get_num_targets()})"


def create_seqnn(params: dict[str, Any]) -> SeqNN:
    """Factory function to create a SeqNN model.

    Args:
        params: Model configuration dictionary.

    Returns:
        SeqNN model instance.
    """
    return SeqNN(params)


# Export classes and functions
__all__ = [
    "SeqNN",
    "SeqNNConfig",
    "SeqNNBuilder",
    "create_seqnn",
]
