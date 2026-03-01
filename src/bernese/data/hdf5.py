# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""
HDF5 utilities for distributed computing and data processing.

This module provides functions for collecting and merging HDF5 files from
parallel jobs, commonly used in distributed variant effect prediction.

Functions:
    collect_h5: Collect and merge HDF5 files from parallel jobs
    collect_h5_borzoi: Collect Borzoi-style HDF5 output files
"""

import os
from typing import Optional

import h5py
import numpy as np


def collect_h5(
    file_name: str,
    out_dir: str,
    num_procs: int,
    progress: bool = True,
) -> None:
    """Collect and merge HDF5 files from parallel jobs.

    Reads HDF5 output files from job directories (job0, job1, etc.) and
    merges them into a single output file.

    Args:
        file_name: Name of the HDF5 file to collect (e.g., 'sad.h5')
        out_dir: Output directory containing job subdirectories
        num_procs: Number of parallel jobs
        progress: If True, print progress information

    Example:
        >>> collect_h5("sad.h5", "output/", 10)
    """
    # Count total variants
    num_variants = 0
    for pi in range(num_procs):
        job_h5_file = os.path.join(out_dir, f"job{pi}", file_name)
        with h5py.File(job_h5_file, "r") as job_h5:
            num_variants += len(job_h5["snp"])
            if progress:
                print(f"Job {pi}: {len(job_h5['snp'])} variants")

    if progress:
        print(f"Total variants: {num_variants}")

    # Initialize final h5
    final_h5_file = os.path.join(out_dir, file_name)
    final_h5 = h5py.File(final_h5_file, "w")

    # Keep dict for string values
    final_strings: dict[str, list] = {}

    # Get schema from first job
    job0_h5_file = os.path.join(out_dir, "job0", file_name)
    with h5py.File(job0_h5_file, "r") as job0_h5:
        for key in job0_h5.keys():
            if key in ["percentiles", "target_ids", "target_labels"]:
                # Copy metadata directly
                final_h5.create_dataset(key, data=job0_h5[key][:])
            elif key.endswith("_pct"):
                # Initialize percentiles with zeros
                values = np.zeros(job0_h5[key].shape)
                final_h5.create_dataset(key, data=values)
            elif job0_h5[key].dtype.kind == "S":
                # String dataset - collect from all jobs
                final_strings[key] = []
            elif job0_h5[key].ndim == 1:
                # 1D dataset
                final_h5.create_dataset(key, shape=(num_variants,), dtype=job0_h5[key].dtype)
            else:
                # nD dataset (e.g., predictions)
                num_targets = job0_h5[key].shape[1]
                final_h5.create_dataset(
                    key, shape=(num_variants, num_targets), dtype=job0_h5[key].dtype
                )

    # Set values from all jobs
    vi = 0
    for pi in range(num_procs):
        if progress:
            print(f"Processing job {pi}...")

        job_h5_file = os.path.join(out_dir, f"job{pi}", file_name)
        with h5py.File(job_h5_file, "r") as job_h5:
            # Append to final
            for key in job_h5.keys():
                if key in ["percentiles", "target_ids", "target_labels"]:
                    # Already copied from job0
                    pass
                elif key.endswith("_pct"):
                    # Average across jobs
                    existing = np.array(final_h5[key])
                    new_vals = np.array(job_h5[key])
                    final_h5[key][:] = existing + (new_vals - existing) / (pi + 1)
                else:
                    if job_h5[key].dtype.kind == "S":
                        # Collect strings
                        final_strings[key].extend(list(job_h5[key]))
                    else:
                        job_variants = job_h5[key].shape[0]
                        try:
                            final_h5[key][vi : vi + job_variants] = job_h5[key][:]
                        except Exception as e:
                            raise RuntimeError(
                                f"{job_h5_file} {key} has wrong shape. "
                                f"Remove this file and rerun. Error: {e}"
                            )

        vi += job_variants
        if progress:
            print(f"  Processed {job_variants} variants (total: {vi})")

    # Create final string datasets
    for key in final_strings:
        string_data = np.array(final_strings[key], dtype="S")
        final_h5.create_dataset(key, data=string_data)

    final_h5.close()

    if progress:
        print(f"Done! Collected {num_variants} variants into {final_h5_file}")


def collect_h5_borzoi(
    out_dir: str,
    num_procs: int,
    sad_stat: str = "sad",
    progress: bool = True,
) -> None:
    """Collect Borzoi-style HDF5 output files from parallel jobs.

    Borzoi uses a specific HDF5 format with scores_f0c0.h5 containing
    various prediction statistics.

    Args:
        out_dir: Output directory containing job subdirectories
        num_procs: Number of parallel jobs
        sad_stat: Name of the main statistic dataset (default: 'sad')
        progress: If True, print progress information

    Example:
        >>> collect_h5_borzoi("output/", 10)
    """
    h5_file = "scores_f0c0.h5"

    # Count total sequences
    num_seqs = 0
    seq_len = 0
    num_targets = 0

    for pi in range(num_procs):
        job_h5_file = os.path.join(out_dir, f"job{pi}", h5_file)
        with h5py.File(job_h5_file, "r") as job_h5:
            num_seqs += job_h5[sad_stat].shape[0]
            seq_len = job_h5[sad_stat].shape[1]
            num_targets = job_h5[sad_stat].shape[-1]

    if progress:
        print(f"Total sequences: {num_seqs}, length: {seq_len}, targets: {num_targets}")

    # Initialize final h5
    final_h5_file = os.path.join(out_dir, h5_file)
    final_h5 = h5py.File(final_h5_file, "w")

    # Keep dict for string values
    final_strings: dict[str, list] = {}

    # Get schema from first job
    job0_h5_file = os.path.join(out_dir, "job0", h5_file)
    with h5py.File(job0_h5_file, "r") as job0_h5:
        for key in job0_h5.keys():
            key_shape = list(job0_h5[key].shape)
            key_shape[0] = num_seqs
            key_shape = tuple(key_shape)

            if job0_h5[key].dtype.kind == "S":
                final_strings[key] = []
            else:
                final_h5.create_dataset(key, shape=key_shape, dtype=job0_h5[key].dtype)

    # Set values from all jobs
    si = 0
    for pi in range(num_procs):
        if progress:
            print(f"Processing job {pi}...")

        job_h5_file = os.path.join(out_dir, f"job{pi}", h5_file)
        with h5py.File(job_h5_file, "r") as job_h5:
            job_seqs = job_h5[sad_stat].shape[0]

            # Append to final
            for key in job_h5.keys():
                if job_h5[key].dtype.kind == "S":
                    final_strings[key].extend(list(job_h5[key]))
                else:
                    final_h5[key][si : si + job_seqs] = job_h5[key][:]

        si += job_seqs
        if progress:
            print(f"  Processed {job_seqs} sequences (total: {si})")

    # Create final string datasets
    for key in final_strings:
        string_data = np.array(final_strings[key], dtype="S")
        final_h5.create_dataset(key, data=string_data)

    final_h5.close()

    if progress:
        print(f"Done! Collected {num_seqs} sequences into {final_h5_file}")


def create_h5_writer(
    output_file: str,
    schema: dict,
    num_samples: int,
    compression: Optional[str] = "gzip",
    compression_opts: int = 4,
) -> h5py.File:
    """Create an HDF5 file with specified schema for writing.

    Args:
        output_file: Path to output HDF5 file
        schema: Dictionary mapping dataset names to shapes and dtypes
        num_samples: Number of samples/variants
        compression: Compression type ('gzip', 'lzf', or None)
        compression_opts: Compression level (1-9 for gzip)

    Returns:
        Open HDF5 file handle for writing

    Example:
        >>> schema = {
        ...     "snp": (num_variants,),
        ...     "preds": (num_variants, num_targets),
        ... }
        >>> with create_h5_writer("output.h5", schema, num_variants) as f:
        ...     f["snp"][:] = snp_data
        ...     f["preds"][:] = preds_data
    """
    h5_file = h5py.File(output_file, "w")

    for name, (shape, dtype) in schema.items():
        if isinstance(shape, tuple):
            # Variable first dimension
            actual_shape = (num_samples,) + shape[1:]
        else:
            actual_shape = (num_samples,)

        h5_file.create_dataset(
            name,
            shape=actual_shape,
            dtype=dtype,
            compression=compression,
            compression_opts=compression_opts if compression == "gzip" else None,
        )

    return h5_file


def read_h5_schema(h5_file: str) -> dict:
    """Read the schema (dataset names, shapes, dtypes) from an HDF5 file.

    Args:
        h5_file: Path to HDF5 file

    Returns:
        Dictionary mapping dataset names to (shape, dtype) tuples
    """
    schema = {}
    with h5py.File(h5_file, "r") as f:
        for key in f.keys():
            schema[key] = (f[key].shape, f[key].dtype)
    return schema
