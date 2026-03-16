# Training subpackage

# Import trainer and builder
from bernese.training.trainer import (
    Trainer,
    TrainerBuilder,
    create_trainer_from_config,
)

# Import Pydantic configurations
from bernese.training.config import (
    # Main config
    TrainerConfig,
    # Optimizer configs
    OptimizerConfig,
    SGDConfig,
    AdamConfig,
    AdamWConfig,
    # Loss configs
    LossConfig,
    MSELossConfig,
    BCELossConfig,
    PoissonLossConfig,
    MSEUDotLossConfig,
    PoissonKLLossConfig,
    PoissonMultinomialLossConfig,
    # Metric configs
    MetricConfig,
    PearsonRMetricConfig,
    R2MetricConfig,
    AUROCMetricConfig,
    AUPRCMetricConfig,
    DEFAULT_METRICS,
    # Scheduler configs
    SchedulerConfig,
    ConstantSchedulerConfig,
    ExponentialSchedulerConfig,
    CyclicalSchedulerConfig,
    WarmupSchedulerConfig,
    # Callback configs
    EarlyStoppingConfig,
    CheckpointConfig,
    # Data config
    DataConfig,
    # Convenience function
    create_trainer_config,
)

__all__ = [
    # Main classes
    "Trainer",
    "TrainerBuilder",
    "TrainerConfig",
    # Factory functions
    "create_trainer_from_config",
    "create_trainer_config",
    # Optimizer configs
    "OptimizerConfig",
    "SGDConfig",
    "AdamConfig",
    "AdamWConfig",
    # Loss configs
    "LossConfig",
    "MSELossConfig",
    "BCELossConfig",
    "PoissonLossConfig",
    "MSEUDotLossConfig",
    "PoissonKLLossConfig",
    "PoissonMultinomialLossConfig",
    # Metric configs
    "MetricConfig",
    "PearsonRMetricConfig",
    "R2MetricConfig",
    "AUROCMetricConfig",
    "AUPRCMetricConfig",
    "DEFAULT_METRICS",
    # Scheduler configs
    "SchedulerConfig",
    "ConstantSchedulerConfig",
    "ExponentialSchedulerConfig",
    "CyclicalSchedulerConfig",
    "WarmupSchedulerConfig",
    # Callback configs
    "EarlyStoppingConfig",
    "CheckpointConfig",
    # Data config
    "DataConfig",
]
