# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Target processor registry and implementations.

This module provides processors for different target data types
(Hi-C, BigWig, etc.) used in regulatory genomics.
"""

from bernese.data.targets.registry import TargetProcessorRegistry, target_processor
from bernese.data.targets.hic import HiCTargetProcessor
from bernese.data.targets.bigwig import BigWigTargetProcessor

__all__ = [
    "TargetProcessorRegistry",
    "target_processor",
    "HiCTargetProcessor",
    "BigWigTargetProcessor",
]
