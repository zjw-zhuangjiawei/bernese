# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""
DNA encoding and decoding utilities for genomic sequences.

This module provides functions for converting between DNA sequences and their
numerical representations (1-hot encoding, index encoding), as well as
reverse complement operations and sequence augmentation.

Functions:
    dna_rc: Reverse complement a DNA sequence string.
    dna_1hot: Convert DNA string to 1-hot encoding.
    dna_1hot_index: Convert DNA string to index encoding.
    hot1_rc: Reverse complement 1-hot encoded sequences.
    hot1_dna: Convert 1-hot encoding back to DNA string.
    hot1_get: Get nucleotide at a position from 1-hot encoding.
    hot1_set: Set nucleotide at a position in 1-hot encoding.
    hot1_insert: Insert sequence at a position.
    hot1_delete: Delete nucleotides at a position.
    hot1_augment: Augment 1-hot sequences with shift and reverse complement.
"""

import random
from typing import Optional

import numpy as np
import torch


def dna_rc(seq: str) -> str:
    """Reverse complement a DNA sequence.

    Args:
        seq: DNA sequence string (upper or lower case).

    Returns:
        Reverse complement of the input sequence.

    Example:
        >>> dna_rc("ATCG")
        'CGAT'
    """
    complement_map = str.maketrans("ATCGatcg", "TAGCtagc")
    return seq.translate(complement_map)[::-1]


def dna_1hot(
    seq: str,
    seq_len: Optional[int] = None,
    n_uniform: bool = False,
    n_sample: bool = False,
) -> np.ndarray:
    """Convert a DNA sequence to a 1-hot encoding.

    Args:
        seq: DNA sequence string.
        seq_len: Length to extend/trim sequences to. If None, uses sequence length.
        n_uniform: If True, represent N's as 0.25 (uniform distribution).
            Otherwise, use 0/1 encoding.
        n_sample: If True, randomly sample ACGT for N positions.

    Returns:
        1-hot encoded sequence as numpy array of shape (seq_len, 4).
        Order: A, C, G, T

    Example:
        >>> dna_1hot("ACGT", seq_len=6)
        array([[1., 0., 0., 0.],
               [0., 1., 0., 0.],
               [0., 0., 1., 0.],
               [0., 0., 0., 1.],
               [0.25, 0.25, 0.25, 0.25],
               [0.25, 0.25, 0.25, 0.25]])
    """
    if seq_len is None:
        seq_len = len(seq)
        seq_start = 0
    else:
        if seq_len <= len(seq):
            # trim the sequence
            seq_trim = (len(seq) - seq_len) // 2
            seq = seq[seq_trim : seq_trim + seq_len]
            seq_start = 0
        else:
            seq_start = (seq_len - len(seq)) // 2

    seq = seq.upper()

    # map nucleotides to a matrix len(seq) x 4 of 0's and 1's
    if n_uniform:
        seq_code = np.zeros((seq_len, 4), dtype=np.float32)
    else:
        seq_code = np.zeros((seq_len, 4), dtype=np.float32)

    for i in range(seq_len):
        if i >= seq_start and i - seq_start < len(seq):
            nt = seq[i - seq_start]
            if nt == "A":
                seq_code[i, 0] = 1
            elif nt == "C":
                seq_code[i, 1] = 1
            elif nt == "G":
                seq_code[i, 2] = 1
            elif nt == "T":
                seq_code[i, 3] = 1
            else:
                if n_uniform:
                    seq_code[i, :] = 0.25
                elif n_sample:
                    ni = random.randint(0, 3)
                    seq_code[i, ni] = 1

    return seq_code


def dna_1hot_index(seq: str, n_sample: bool = False) -> np.ndarray:
    """Convert a DNA sequence to an index encoding.

    Args:
        seq: DNA sequence string.
        n_sample: If True, randomly sample ACGT for N positions.
            Otherwise, use 4 for unknown nucleotides.

    Returns:
        Index-encoded sequence as numpy array of shape (seq_len,).
        Index mapping: A=0, C=1, G=2, T=3, N=4

    Example:
        >>> dna_1hot_index("ACGTN")
        array([0, 1, 2, 3, 4], dtype=uint8)
    """
    seq_len = len(seq)
    seq = seq.upper()

    # map nucleotides to indices 0, 1, 2, 3, 4
    seq_code = np.zeros(seq_len, dtype=np.uint8)

    for i in range(seq_len):
        nt = seq[i]
        if nt == "A":
            seq_code[i] = 0
        elif nt == "C":
            seq_code[i] = 1
        elif nt == "G":
            seq_code[i] = 2
        elif nt == "T":
            seq_code[i] = 3
        else:
            if n_sample:
                seq_code[i] = random.randint(0, 3)
            else:
                seq_code[i] = 4

    return seq_code


def hot1_rc(seqs_1hot: np.ndarray) -> np.ndarray:
    """Reverse complement a batch of 1-hot coded sequences.

    Handles additional tracks beyond the four nucleotides correctly.

    Args:
        seqs_1hot: 1-hot encoded sequences.
            Can be 2D (Lx4) or 3D (BxLx4).

    Returns:
        Reverse complemented sequences with same shape.

    Example:
        >>> seq = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        >>> hot1_rc(seq)
        array([[0., 0., 0., 1.],
               [0., 0., 1., 0.],
               [0., 1., 0., 0.],
               [1., 0., 0., 0.]])
    """
    if seqs_1hot.ndim == 2:
        singleton = True
        seqs_1hot = np.expand_dims(seqs_1hot, axis=0)
    else:
        singleton = False

    seqs_1hot_rc = seqs_1hot.copy()

    # reverse
    seqs_1hot_rc = seqs_1hot_rc[:, ::-1, :]

    # swap A and T (columns 0 and 3)
    seqs_1hot_rc[:, :, [0, 3]] = seqs_1hot_rc[:, :, [3, 0]]

    # swap C and G (columns 1 and 2)
    seqs_1hot_rc[:, :, [1, 2]] = seqs_1hot_rc[:, :, [2, 1]]

    if singleton:
        seqs_1hot_rc = seqs_1hot_rc[0]

    return seqs_1hot_rc


def hot1_rc_torch(seqs_1hot: torch.Tensor) -> torch.Tensor:
    """Reverse complement a batch of 1-hot coded sequences (PyTorch version).

    Handles additional tracks beyond the four nucleotides correctly.

    Args:
        seqs_1hot: 1-hot encoded sequences as PyTorch tensor.
            Can be 2D (Lx4) or 3D (BxLx4).

    Returns:
        Reverse complemented sequences with same shape.
    """
    if seqs_1hot.ndim == 2:
        singleton = True
        seqs_1hot = seqs_1hot.unsqueeze(0)
    else:
        singleton = False

    seqs_1hot_rc = seqs_1hot.clone()

    # reverse
    seqs_1hot_rc = seqs_1hot_rc[:, ::-1, :]

    # swap A and T (columns 0 and 3)
    seqs_1hot_rc[:, :, [0, 3]] = seqs_1hot_rc[:, :, [3, 0]]

    # swap C and G (columns 1 and 2)
    seqs_1hot_rc[:, :, [1, 2]] = seqs_1hot_rc[:, :, [2, 1]]

    if singleton:
        seqs_1hot_rc = seqs_1hot_rc[0]

    return seqs_1hot_rc


def hot1_dna(seqs_1hot: np.ndarray) -> list[str] | str:
    """Convert 1-hot coded sequences to DNA strings.

    Args:
        seqs_1hot: 1-hot encoded sequences.
            Can be 2D (Lx4) or 3D (BxLx4).

    Returns:
        DNA sequence string(s). Returns a single string if input was 2D,
        or a list of strings if input was 3D.

    Example:
        >>> seq = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        >>> hot1_dna(seq)
        'ACGT'
    """
    singleton = False
    if seqs_1hot.ndim == 2:
        singleton = True
        seqs_1hot = np.expand_dims(seqs_1hot, axis=0)

    seqs = []
    for si in range(seqs_1hot.shape[0]):
        seq_list = ["A"] * seqs_1hot.shape[1]
        for li in range(seqs_1hot.shape[1]):
            if seqs_1hot[si, li, 0] == 1:
                seq_list[li] = "A"
            elif seqs_1hot[si, li, 1] == 1:
                seq_list[li] = "C"
            elif seqs_1hot[si, li, 2] == 1:
                seq_list[li] = "G"
            elif seqs_1hot[si, li, 3] == 1:
                seq_list[li] = "T"
            else:
                seq_list[li] = "N"

        seqs.append("".join(seq_list))

    if singleton:
        seqs = seqs[0]

    return seqs


def hot1_get(seqs_1hot: np.ndarray, pos: int) -> str:
    """Get the nucleotide at a position from 1-hot encoding.

    Args:
        seqs_1hot: 1-hot encoded sequence (Lx4).
        pos: Position to get nucleotide.

    Returns:
        Nucleotide character (A, C, G, T, or N).

    Example:
        >>> seq = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        >>> hot1_get(seq, 0)
        'A'
    """
    if seqs_1hot[pos, 0] == 1:
        return "A"
    elif seqs_1hot[pos, 1] == 1:
        return "C"
    elif seqs_1hot[pos, 2] == 1:
        return "G"
    elif seqs_1hot[pos, 3] == 1:
        return "T"
    else:
        return "N"


def hot1_set(seq_1hot: np.ndarray, pos: int, nt: str) -> None:
    """Set position in a 1-hot encoded sequence to given nucleotide.

    Args:
        seq_1hot: 1-hot encoded sequence (Lx4). Modified in place.
        pos: Position to set nucleotide.
        nt: Nucleotide to set (A, C, G, T, or N).

    Example:
        >>> seq = np.zeros((4, 4), dtype=np.float32)
        >>> hot1_set(seq, 0, 'A')
        >>> seq[0]
        array([1., 0., 0., 0.])
    """
    # reset
    seq_1hot[pos, :4] = 0

    # set
    if nt == "A":
        seq_1hot[pos, 0] = 1
    elif nt == "C":
        seq_1hot[pos, 1] = 1
    elif nt == "G":
        seq_1hot[pos, 2] = 1
    elif nt == "T":
        seq_1hot[pos, 3] = 1
    elif nt != "N":
        raise ValueError(f"Invalid nucleotide: {nt}")


def hot1_insert(seq_1hot: np.ndarray, pos: int, insert_seq: str) -> None:
    """Insert sequence at a given position in the 1-hot encoded sequence.

    Args:
        seq_1hot: 1-hot encoded sequence (Lx4). Modified in place.
        pos: Position to insert sequence.
        insert_seq: Sequence to insert.

    Example:
        >>> seq = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        >>> hot1_insert(seq, 1, "GC")
        >>> seq
        array([[1., 0., 0., 0.],
               [0., 1., 0., 0.],
               [0., 0., 1., 0.],
               [0., 0., 0., 1.],
               [0., 1., 0., 0.],
               [0., 0., 1., 0.]])
    """
    insert_len = len(insert_seq)
    seq_len = seq_1hot.shape[0]

    # shift right
    if pos + insert_len < seq_len:
        seq_1hot[pos + insert_len :, :4] = seq_1hot[pos : seq_len - insert_len, :4]

    # reset inserted region
    seq_1hot[pos : pos + insert_len, :4] = 0

    # encode inserted sequence
    for i in range(insert_len):
        nt = insert_seq[i]
        if nt == "A":
            seq_1hot[pos + i, 0] = 1
        elif nt == "C":
            seq_1hot[pos + i, 1] = 1
        elif nt == "G":
            seq_1hot[pos + i, 2] = 1
        elif nt == "T":
            seq_1hot[pos + i, 3] = 1
        else:
            raise ValueError(f"Invalid nucleotide in insert: {nt}")


def hot1_delete(seq_1hot: np.ndarray, pos: int, delete_len: int) -> None:
    """Delete nucleotides starting at a given position in the 1-hot encoded sequence.

    Args:
        seq_1hot: 1-hot encoded sequence (Lx4). Modified in place.
        pos: Position to start deleting.
        delete_len: Number of nucleotides to delete.

    Example:
        >>> seq = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        >>> hot1_delete(seq, 1, 2)
        >>> seq
        array([[1., 0., 0., 0.],
               [0., 0., 0., 1.],
               [0.25, 0.25, 0.25, 0.25],
               [0.25, 0.25, 0.25, 0.25]])
    """
    seq_len = seq_1hot.shape[0]

    # shift left
    if pos + delete_len < seq_len:
        seq_1hot[pos : seq_len - delete_len, :] = seq_1hot[pos + delete_len :, :]

    # change right end to N's (using 0.25 for each)
    seq_1hot[seq_len - delete_len :, :4] = 0.25


def hot1_augment(
    Xb: np.ndarray,
    fwdrc: bool = True,
    shift: int = 0,
) -> np.ndarray:
    """Transform a batch of 1-hot coded sequences for data augmentation.

    Args:
        Xb: Batch of 1-hot coded sequences (BxLx4 or Lx4).
        fwdrc: If True, keep forward strand. If False, apply reverse complement.
        shift: Number of positions to shift. Positive shifts left, negative shifts right.

    Returns:
        Transformed batch of sequences.

    Example:
        >>> seq = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        >>> hot1_augment(seq, fwdrc=False, shift=1)
    """
    if Xb.ndim == 2:
        singleton = True
        Xb = np.expand_dims(Xb, axis=0)
    else:
        singleton = False

    # Determine pad value based on dtype
    if Xb.dtype == bool:
        nval = 0
    else:
        nval = 0.25

    if shift == 0:
        Xbt = Xb
    elif shift > 0:
        Xbt = np.zeros(Xb.shape, dtype=Xb.dtype)

        # fill in left unknowns with N
        Xbt[:, :shift, :] = nval

        # fill in sequence
        Xbt[:, shift:, :] = Xb[:, :-shift, :]
    else:
        Xbt = np.zeros(Xb.shape, dtype=Xb.dtype)

        # fill in right unknowns with N
        Xbt[:, shift:, :] = nval

        # fill in sequence
        Xbt[:, :shift, :] = Xb[:, -shift:, :]

    if not fwdrc:
        Xbt = hot1_rc(Xbt)

    if singleton:
        Xbt = Xbt[0]

    return Xbt


def hot1_augment_torch(
    Xb: torch.Tensor,
    fwdrc: bool = True,
    shift: int = 0,
) -> torch.Tensor:
    """Transform a batch of 1-hot coded sequences for data augmentation (PyTorch version).

    Args:
        Xb: Batch of 1-hot coded sequences as PyTorch tensor (BxLx4 or Lx4).
        fwdrc: If True, keep forward strand. If False, apply reverse complement.
        shift: Number of positions to shift. Positive shifts left, negative shifts right.

    Returns:
        Transformed batch of sequences.
    """
    if Xb.ndim == 2:
        singleton = True
        Xb = Xb.unsqueeze(0)
    else:
        singleton = False

    # Determine pad value based on dtype
    if Xb.dtype == torch.bool:
        nval = 0
    else:
        nval = 0.25

    if shift == 0:
        Xbt = Xb
    elif shift > 0:
        Xbt = torch.zeros_like(Xb)

        # fill in left unknowns with N
        Xbt[:, :shift, :] = nval

        # fill in sequence
        Xbt[:, shift:, :] = Xb[:, :-shift, :]
    else:
        Xbt = torch.zeros_like(Xb)

        # fill in right unknowns with N
        Xbt[:, shift:, :] = nval

        # fill in sequence
        Xbt[:, :shift, :] = Xb[:, -shift:, :]

    if not fwdrc:
        Xbt = hot1_rc_torch(Xbt)

    if singleton:
        Xbt = Xbt[0]

    return Xbt
