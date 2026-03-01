# Code Style Rules

**Summary**: Code style guidelines for consistent, readable Python code in the Bernese project.

**Why**: Consistent code style improves readability, reduces cognitive load, and makes code reviews easier. These rules follow PEP 8 and project-specific conventions.

Last updated: 2026-03-01

---

## Heavy Section Dividers Forbidden

**Summary**: Do not use heavy section dividers (decorative comment blocks) to separate code sections.

**Why**: Heavy section dividers (like `# =============================================================================`) add unnecessary visual clutter, reduce code density, and create inconsistency. Modern IDEs provide navigation features that make these separators obsolete. The project already removed these from `contacts.py`.

### ❌ Don't Do This

```python
# =============================================================================
# Genome and Contig Handling
# =============================================================================

def load_genome(fasta_file: str) -> dict[str, list[tuple[int, int]]]:
    ...

# =============================================================================
# Train/Valid/Test Splitting
# =============================================================================

def divide_contigs_by_pct(...):
    ...
```

### ✅ Do This

Write functions/classes in logical order with clear docstrings - no separator comments needed:

```python
def load_genome(fasta_file: str) -> dict[str, list[tuple[int, int]]]:
    """Load genome from FASTA file.
    
    Args:
        fasta_file: Path to FASTA file
    
    Returns:
        Dictionary mapping chromosome to list of (start, end) segments
    """
    return genomics.load_chromosomes(fasta_file)


def divide_contigs_by_pct(
    contigs: list[Contig],
    test_pct: float,
    valid_pct: float,
) -> tuple[list[Contig], list[Contig], list[Contig]]:
    """Divide contigs by percentage into train/valid/test.
    
    Args:
        contigs: List of Contig objects
        test_pct: Test set percentage (0-1)
        valid_pct: Validation set percentage (0-1)
    
    Returns:
        Tuple of (train, valid, test) contig lists
    """
    ...
```

### Exceptions

- **Lone modules**: A single-module file may use a light separator (e.g., `# --- Functions ---`) if truly needed for clarity
- **Legacy code**: Existing files with these dividers don't need to be cleaned up unless you're already editing that file

---

## Related Rules

- [UV Python Rules](1-uv-python-rules.md) - Package management
- [Global Rules](../Cline/Rules/) - Environment and best practice rules
