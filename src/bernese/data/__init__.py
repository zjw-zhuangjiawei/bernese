# Data subpackage

# Import dataset
from bernese.data.dataset import (
    GenomicDataset,
    create_data_loaders,
    MultiDatasetWrapper,
)

# Import backends
from bernese.data.backends import (
    DataBackend,
    DatasetMetadata,
    SplitMetadata,
    HDF5Backend,
    HDF5Writer,
)

# Import transforms
from bernese.data.transforms import (
    Transform,
    TransformPipeline,
    RandomShift,
    ReverseComplement,
    RandomCrop,
    TargetNormalize,
)

# Import target processors
from bernese.data.targets import (
    TargetProcessorRegistry,
    target_processor,
    HiCTargetProcessor,
    BigWigTargetProcessor,
)

# Import preparation
from bernese.data.preparation import (
    DataPreparator,
    PreparationConfig,
    prepare_dataset,
)


__all__ = [
    # Dataset
    "GenomicDataset",
    "create_data_loaders",
    "MultiDatasetWrapper",
    # Backends
    "DataBackend",
    "DatasetMetadata",
    "SplitMetadata",
    "HDF5Backend",
    "HDF5Writer",
    # Transforms
    "Transform",
    "TransformPipeline",
    "RandomShift",
    "ReverseComplement",
    "RandomCrop",
    "TargetNormalize",
    # Target processors
    "TargetProcessorRegistry",
    "target_processor",
    "HiCTargetProcessor",
    "BigWigTargetProcessor",
    # Preparation
    "DataPreparator",
    "PreparationConfig",
    "prepare_dataset",
]
