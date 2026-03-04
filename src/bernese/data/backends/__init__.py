# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Data backend interfaces and implementations.

This module provides abstract backends for different data storage formats,
enabling flexible data loading and writing strategies.

Backends (Reading):
    DataBackend: Abstract protocol for data storage
    HDF5Backend: HDF5-based storage with split organization

Writers:
    DataWriter: Abstract protocol for data storage writers
    HDF5Writer: HDF5-based data writer for creating datasets
"""

from bernese.data.backends.base import DataBackend, DataWriter, DatasetMetadata, SplitMetadata
from bernese.data.backends.hdf5 import HDF5Backend, HDF5Writer

__all__ = [
    "DataBackend",
    "DataWriter",
    "DatasetMetadata",
    "SplitMetadata",
    "HDF5Backend",
    "HDF5Writer",
]
