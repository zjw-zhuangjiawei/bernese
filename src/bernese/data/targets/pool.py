# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Pooling statistics enum and helpers.

This module provides the PoolStat enum for specifying pooling aggregation
methods used by target processors.
"""

from enum import Enum


class PoolStat(Enum):
    """Pooling summary statistics.
    
    Attributes:
        SUM: Sum of values
        SUM_SQRT: sqrt(1 + sum(values)) - 1 (shifted square root of sum)
        MEAN: Mean of values
        MEAN_SQRT: sqrt(1 + mean(values)) - 1 (shifted square root of mean)
        MEDIAN: Median of values
        MAX: Maximum of values
        MIN: Minimum of values
        PEAK: Clipped sqrt(mean * 4) for peak detection
    """
    SUM = "sum"
    SUM_SQRT = "sum_sqrt"
    MEAN = "mean"
    MEAN_SQRT = "mean_sqrt"
    MEDIAN = "median"
    MAX = "max"
    MIN = "min"
    PEAK = "peak"
