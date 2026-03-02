# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""
Genomic data structures and utilities for regulatory genomics.

This module provides classes and functions for working with genomic coordinates,
contigs, genes, and genomes.

Classes:
    Contig: Named tuple for genomic contigs (genome, chr, start, end)
    ModelSeq: Named tuple for model sequences (genome, chr, start, end, label)
    GenomicInterval: Basic genomic interval representation
    Gene: Gene with exons for isoform-agnostic analysis
    Transcriptome: Collection of genes from GTF file

Functions:
    load_chromosomes: Load genome from FASTA or chromosome length table
    contig_sequences: Break contigs into model-length sequences
    break_large_contigs: Split large contigs for parallel processing
    rejoin_broken_contigs: Rejoin contigs after processing
    split_contigs: Split assembly by gaps
"""

import collections
import gzip
import heapq
import json
from typing import Optional

import numpy as np

# Named tuples for genomic coordinates
Contig = collections.namedtuple("Contig", ["genome", "chr", "start", "end"])
ModelSeq = collections.namedtuple("ModelSeq", ["genome", "chr", "start", "end", "label"])


class GenomicInterval:
    """Basic genomic interval representation.

    Attributes:
        start: Start position (0-indexed, inclusive)
        end: End position (0-indexed, exclusive)
        chrom: Chromosome name (optional)
        strand: Strand (+, -, or None)
    """

    def __init__(
        self,
        start: int,
        end: int,
        chrom: Optional[str] = None,
        strand: Optional[str] = None,
    ):
        self.start = start
        self.end = end
        self.chrom = chrom
        self.strand = strand

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GenomicInterval):
            return NotImplemented
        return self.start == other.start

    def __lt__(self, other: "GenomicInterval") -> bool:
        return self.start < other.start

    def __len__(self) -> int:
        return self.end - self.start

    def __str__(self) -> str:
        if self.chrom is None:
            return f"[{self.start}-{self.end}]"
        return f"{self.chrom}:{self.start}-{self.end}"

    def __repr__(self) -> str:
        return f"GenomicInterval(start={self.start}, end={self.end}, chrom={self.chrom!r}, strand={self.strand!r})"


class Gene:
    """Gene with exons for isoform-agnostic analysis.

    This class represents a gene by taking the union of exons across all isoforms.
    It provides methods for computing gene spans, midpoints, and output slices
    for model predictions.

    Attributes:
        chrom: Chromosome name
        strand: Strand (+ or -)
        kv: Dictionary of gene attributes from GTF
        name: Gene name (optional)
    """

    def __init__(
        self,
        chrom: str,
        strand: str,
        kv: dict,
        name: Optional[str] = None,
    ):
        self.chrom = chrom
        self.strand = strand
        self.kv = kv
        self.name = name
        self._exons: list[tuple[int, int]] = []

    def add_exon(self, start: int, end: int) -> None:
        """Add an exon to the gene.

        Args:
            start: Exon start position (0-indexed, BED convention)
            end: Exon end position (0-indexed, exclusive)
        """
        self._exons.append((start, end))

    def get_exons(self) -> list[tuple[int, int]]:
        """Get merged exons sorted by position.

        Returns:
            List of (start, end) tuples for merged exons.
        """
        if not self._exons:
            return []

        # Sort by start position
        sorted_exons = sorted(self._exons, key=lambda x: x[0])

        # Merge overlapping exons
        merged = [sorted_exons[0]]
        for start, end in sorted_exons[1:]:
            if start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        self._exons = merged
        return merged

    def midpoint(self) -> int:
        """Get the midpoint of the gene.

        Returns:
            Integer position of the gene midpoint.
        """
        positions = []
        for start, end in self.get_exons():
            positions.extend(range(start, end))
        return int(np.mean(positions))

    def span(self) -> tuple[int, int]:
        """Get the gene span (min start, max end).

        Returns:
            Tuple of (min_start, max_end).
        """
        if not self._exons:
            return (0, 0)
        starts = [exon[0] for exon in self._exons]
        ends = [exon[1] for exon in self._exons]
        return (min(starts), max(ends))

    def output_slice(
        self,
        seq_start: int,
        seq_len: int,
        model_stride: int,
        span: bool = False,
        majority_overlap: bool = True,
    ) -> np.ndarray:
        """Compute the output slice for this gene given sequence coordinates.

        Args:
            seq_start: Start position of the sequence.
            seq_len: Length of the sequence.
            model_stride: Model pooling stride.
            span: If True, use gene span instead of individual exons.
            majority_overlap: If True, require >50% overlap. If False, any overlap.

        Returns:
            Array of output bin indices that overlap with the gene.
        """
        gene_slice: list[int] = []

        def clip_boundaries(slice_start: int, slice_end: int) -> tuple[int, int]:
            slice_max = seq_len // model_stride
            slice_start = min(slice_start, slice_max)
            slice_end = min(slice_end, slice_max)
            slice_start = max(slice_start, 0)
            slice_end = max(slice_end, 0)
            return slice_start, slice_end

        if span:
            gene_start, gene_end = self.span()

            # Clip left boundaries
            gene_seq_start = max(0, gene_start - seq_start)
            gene_seq_end = max(0, gene_end - seq_start)

            # Calculate slice
            if majority_overlap:
                slice_start = int(np.round(gene_seq_start / model_stride))
                slice_end = int(np.round(gene_seq_end / model_stride))
            else:
                slice_start = int(np.floor(gene_seq_start / model_stride))
                slice_end = int(np.ceil(gene_seq_end / model_stride))

            # Clip boundaries
            slice_start, slice_end = clip_boundaries(slice_start, slice_end)

            # Add to gene slice
            if slice_start < slice_end:
                gene_slice = list(range(slice_start, slice_end))
        else:
            for exon_start, exon_end in self.get_exons():
                # Clip left boundaries
                exon_seq_start = max(0, exon_start - seq_start)
                exon_seq_end = max(0, exon_end - seq_start)

                if majority_overlap:
                    slice_start = int(np.round(exon_seq_start / model_stride))
                    slice_end = int(np.round(exon_seq_end / model_stride))
                else:
                    slice_start = int(np.floor(exon_seq_start / model_stride))
                    slice_end = int(np.ceil(exon_seq_end / model_stride))

                # Clip boundaries
                slice_start, slice_end = clip_boundaries(slice_start, slice_end)

                # Add to gene slice
                if slice_start < slice_end:
                    gene_slice.extend(range(slice_start, slice_end))

        # Collapse overlaps
        return np.unique(gene_slice)


class Transcriptome:
    """Collection of genes from a GTF file.

    Attributes:
        genes: Dictionary mapping gene_id to Gene objects
    """

    def __init__(self, gtf_file: str):
        """Initialize transcriptome from GTF file.

        Args:
            gtf_file: Path to GTF file (can be gzipped)
        """
        self.genes: dict[str, Gene] = {}
        self.read_gtf(gtf_file)

    def read_gtf(self, gtf_file: str) -> None:
        """Read genes from GTF file.

        Args:
            gtf_file: Path to GTF file (can be gzipped)
        """
        if gtf_file.endswith(".gz"):
            gtf_in = gzip.open(gtf_file, "rt")
        else:
            gtf_in = open(gtf_file)

        # Skip header
        line = gtf_in.readline()
        while line.startswith("#"):
            line = gtf_in.readline()

        while line:
            fields = line.split("\t")
            if len(fields) < 9:
                line = gtf_in.readline()
                continue

            if fields[2] == "exon":
                chrom = fields[0]
                start = int(fields[3])
                end = int(fields[4])
                strand = fields[6]
                kv = self._gtf_kv(fields[8])
                gene_id = kv.get("gene_id", "")
                gene_name = kv.get("gene_name", None)

                # Initialize gene if needed
                if gene_id not in self.genes:
                    self.genes[gene_id] = Gene(chrom, strand, kv, gene_name)

                # Add exon (GTF is 1-indexed, convert to BED 0-indexed)
                self.genes[gene_id].add_exon(start - 1, end)

            line = gtf_in.readline()

        gtf_in.close()

    @staticmethod
    def _gtf_kv(s: str) -> dict[str, str]:
        """Convert GTF attribute string to dictionary.

        Args:
            s: GTF attribute field (column 9)

        Returns:
            Dictionary of key-value pairs
        """
        d = {}
        parts = s.split(";")
        for key_val in parts:
            key_val = key_val.strip()
            if not key_val:
                continue

            # Handle both key=value and key "value" formats
            if "=" in key_val:
                kvs = key_val.split("=")
            else:
                kvs = key_val.split()

            if len(kvs) < 2:
                continue

            key = kvs[0]
            val = " ".join(kvs[1:])

            # Remove quotes
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1]

            d[key] = val

        return d


################################################################################
# Helper Functions
################################################################################


def load_chromosomes(genome_file: str) -> dict[str, list[tuple[int, int]]]:
    """Load genome segments from FASTA file or chromosome length table.

    Args:
        genome_file: Path to FASTA file or chromosome length table.
            If FASTA, should have .fa or .fasta extension.
            Otherwise, expects tab-separated (chrom, length) per line.

    Returns:
        Dictionary mapping chromosome names to list of (start, end) segments.

    Example:
        >>> chroms = load_chromosomes("genome.fa")
        >>> print(chroms)
        {'chr1': [(0, 248956422)], 'chr2': [(0, 242193529)], ...}
    """
    # Check if FASTA
    with open(genome_file) as f:
        first_char = f.readline()[0]

    is_fasta = first_char == ">"

    if is_fasta:
        import pysam

        fasta = pysam.Fastafile(genome_file)
        chrom_segments = {
            ref: [(0, length)] for ref, length in zip(fasta.references, fasta.lengths)
        }
        fasta.close()
    else:
        chrom_segments = {}
        with open(genome_file) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    chrom = parts[0]
                    length = int(parts[1])
                    chrom_segments[chrom] = [(0, length)]

    return chrom_segments


def contig_sequences(
    contigs: list[Contig],
    seq_length: int,
    stride: int,
    snap: int = 1,
    label: Optional[str] = None,
) -> list[ModelSeq]:
    """Break up a list of Contigs into model-length sequence segments.

    Args:
        contigs: List of Contig objects
        seq_length: Length of sequences to extract
        stride: Stride between consecutive sequences
        snap: Snap start positions to multiples of this value
        label: Optional label for sequences

    Returns:
        List of ModelSeq objects

    Example:
        >>> contigs = [Contig("hg38", "chr1", 0, 100000)]
        >>> mseqs = contig_sequences(contigs, seq_length=1000, stride=500)
    """
    mseqs = []

    for ctg in contigs:
        # Snap start position
        seq_start = int(np.ceil(ctg.start / snap) * snap)
        seq_end = seq_start + seq_length

        while seq_end <= ctg.end:
            mseqs.append(ModelSeq(ctg.genome, ctg.chr, seq_start, seq_end, label))
            seq_start += stride
            seq_end += stride

    return mseqs


def break_large_contigs(
    contigs: list[Contig],
    break_t: int,
    verbose: bool = False,
) -> list[Contig]:
    """Break large contigs in half until all contigs are under the size threshold.

    Uses a heap-based approach to break the largest contigs first.

    Args:
        contigs: List of Contig objects
        break_t: Size threshold - contigs larger than this will be broken
        verbose: If True, print progress

    Returns:
        List of broken contigs

    Example:
        >>> contigs = [Contig("hg38", "chr1", 0, 1000000)]
        >>> broken = break_large_contigs(contigs, break_t=500000)
    """
    # Initialize heap with contigs and lengths (negative for max-heap)
    contig_heap = []
    for ctg in contigs:
        ctg_len = ctg.end - ctg.start
        heapq.heappush(contig_heap, (-ctg_len, ctg))

    # Collect contigs that don't exceed the threshold
    result_contigs = []

    while contig_heap:
        # Pop largest contig
        ctg_neg_len, ctg = heapq.heappop(contig_heap)
        ctg_len = -ctg_neg_len

        # If too large, break in two
        if ctg_len > break_t:
            if verbose:
                print(f"Breaking {ctg.chr}:{ctg.start}-{ctg.end} ({ctg_len} nt)")

            ctg_mid = ctg.start + ctg_len // 2
            ctg_left = Contig(ctg.genome, ctg.chr, ctg.start, ctg_mid)
            ctg_right = Contig(ctg.genome, ctg.chr, ctg_mid, ctg.end)

            # Add both halves back to heap
            heapq.heappush(contig_heap, (-(ctg_left.end - ctg_left.start), ctg_left))
            heapq.heappush(contig_heap, (-(ctg_right.end - ctg_right.start), ctg_right))
        else:
            # Contig is small enough, keep it
            result_contigs.append(ctg)

    return result_contigs


def rejoin_broken_contigs(contigs: list[Contig]) -> list[Contig]:
    """Rejoin contigs that were previously broken for parallel processing.

    Args:
        contigs: List of Contig objects that may have been broken

    Returns:
        List of rejoined contigs

    Example:
        >>> broken = [Contig("hg38", "chr1", 0, 500), Contig("hg38", "chr1", 500, 1000)]
        >>> rejoined = rejoin_broken_contigs(broken)
    """
    if not contigs:
        return []

    # Group by genome/chromosome
    gchr_contigs: dict[tuple[str, str], list[Contig]] = {}
    for ctg in contigs:
        key = (ctg.genome, ctg.chr)
        gchr_contigs.setdefault(key, []).append(ctg)

    result = []
    for key in gchr_contigs:
        # Sort within chromosome by start position
        sorted_contigs = sorted(gchr_contigs[key], key=lambda x: x.start)

        # Merge adjacent contigs
        current = sorted_contigs[0]
        for i in range(1, len(sorted_contigs)):
            next_ctg = sorted_contigs[i]
            if current.end == next_ctg.start:
                # Join contigs
                current = Contig(current.genome, current.chr, current.start, next_ctg.end)
            else:
                # Conclude current and move to next
                result.append(current)
                current = next_ctg

        # Add final contig
        result.append(current)

    return result


def split_contigs_by_gaps(
    chrom_segments: dict[str, list[tuple[int, int]]],
    gaps_file: str,
) -> dict[str, list[tuple[int, int]]]:
    """Split assembly contigs by assembly gaps.

    Args:
        chrom_segments: Dictionary mapping chromosome to list of (start, end) segments
        gaps_file: File specifying assembly gaps (BED-like format: chrom start end)

    Returns:
        Updated chrom_segments with gaps split

    Example:
        >>> chroms = {"chr1": [(0, 248956422)]}
        >>> split = split_contigs_by_gaps(chroms, "gaps.bed")
    """
    chrom_events: dict[str, list[tuple[int, str]]] = {}

    # Add known segments
    for chrom in chrom_segments:
        if len(chrom_segments[chrom]) > 1:
            raise ValueError(f"Multiple segments for {chrom}, expected single segment")

        cstart, cend = chrom_segments[chrom][0]
        chrom_events.setdefault(chrom, []).append((cstart, "cstart"))
        chrom_events[chrom].append((cend, "cend"))

    # Add gaps
    with open(gaps_file) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            chrom = parts[0]
            gstart = int(parts[1])
            gend = int(parts[2])

            if chrom in chrom_events:
                chrom_events[chrom].append((gstart, "gstart"))
                chrom_events[chrom].append((gend, "gend"))

    # Process events
    for chrom in chrom_events:
        # Sort events
        chrom_events[chrom].sort()

        # Read out segments
        chrom_segments[chrom] = []
        for i in range(len(chrom_events[chrom]) - 1):
            pos1, event1 = chrom_events[chrom][i]
            pos2, event2 = chrom_events[chrom][i + 1]

            event1 = event1.lower()
            event2 = event2.lower()

            # Determine if we should emit this segment
            emit = False
            if event1 == "cstart" and event2 == "cend":
                emit = True
            elif event1 == "cstart" and event2 == "gstart":
                emit = True
            elif event1 == "gend" and event2 == "gstart":
                emit = True
            elif event1 == "gend" and event2 == "cend":
                emit = True

            if emit and pos1 < pos2:
                chrom_segments[chrom].append((pos1, pos2))

    return chrom_segments


def write_sequences_bed(
    bed_file: str,
    seqs: list[ModelSeq],
    labels: bool = False,
) -> None:
    """Write sequences to BED file.

    Args:
        bed_file: Output BED file path
        seqs: List of ModelSeq objects
        labels: If True, write label as fourth column

    Example:
        >>> mseqs = [ModelSeq("hg38", "chr1", 0, 1000, "train")]
        >>> write_sequences_bed("sequences.bed", mseqs, labels=True)
    """
    with open(bed_file, "w") as f:
        for seq in seqs:
            line = f"{seq.chr}\t{seq.start}\t{seq.end}"
            if labels and seq.label is not None:
                line += f"\t{seq.label}"
            print(line, file=f)


def read_sequences_bed(bed_file: str) -> list[ModelSeq]:
    """Read sequences from BED file.

    Args:
        bed_file: Input BED file path

    Returns:
        List of ModelSeq objects

    Example:
        >>> mseqs = read_sequences_bed("sequences.bed")
    """
    seqs = []
    with open(bed_file) as f:
        for line in f:
            parts = line.split()
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            label = parts[3] if len(parts) > 3 else None
            seqs.append(ModelSeq(genome="", chr=chrom, start=start, end=end, label=label))
    return seqs
