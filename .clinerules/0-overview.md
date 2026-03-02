# Bernese Clinerules Overview

**Summary**: Project-specific rules and guidelines for developing Bernese, a Keras 3-based library for regulatory genomics predictions.

**Why**: These rules ensure consistent development practices, proper dependency management, and maintainable code across the Bernese project.

Last updated: 2026-03-02

---

## Project Overview

Bernese is a **Keras 3-based deep learning library for regulatory genomics predictions**. It implements the SeqNN (Sequence Neural Network) model for predicting regulatory activity from DNA sequences.

| Aspect | Details |
|--------|---------|
| **Type** | Python library for genomic/molecular biology deep learning |
| **Framework** | Keras 3 (with optional PyTorch/CUDA support) |
| **Core Model** | SeqNN - Sequential Neural Network for regulatory activity prediction |
| **Data Format** | HDF5 files (sequences.h5) with JSON metadata (statistics.json) |
| **Python** | >=3.13 |
| **Key Dependencies** | numpy, pandas, scipy, h5py, scikit-learn, keras, torchinfo |

---

## Navigating the Rules

| Document | Description |
|----------|-------------|
| [`1-uv-python-rules.md`](1-uv-python-rules.md) | UV Python package manager usage guide |
| [`2-code-style-rules.md`](2-code-style-rules.md) | Code style guidelines |
| [`3-common-operations.md`](3-common-operations.md) | Common CLI operations reference |

---

## Quick Reference

### Common Commands

| Task | Command |
|------|---------|
| Install dependencies (GPU) | `uv sync --extra cu130` |
| Install dependencies (CPU) | `uv sync` |
| Add dependency | `uv add <package>` |
| Inspect model | `uv run bernese summary <params_file>` |
| Train model | `uv run bernese train <params_file> <data_dirs>...` |
| Run tests | `uv run pytest` |
| Build package | `uv build` |

### Project Structure

```
bernese/
├── .clinerules/           # This directory
├── src/bernese/           # Package source code
│   ├── cli/               # CLI entry points
│   ├── data/              # Dataset classes
│   ├── interpretation/    # Interpretation tools
│   ├── metrics/           # Loss functions and metrics
│   ├── models/            # Model architectures
│   └── training/          # Training utilities
└── tests/                 # Test files
```

---

## Key Components

### Models

- **`SeqNN`**: Main model class for regulatory activity prediction (Keras 3)
  - Location: `src/bernese/models/seqnn.py`
  - Factory function: `create_seqnn()`

- **`layers`**: Keras 3 layer definitions
  - Location: `src/bernese/models/layers.py`

- **`blocks`**: Keras 3 model blocks (convolution, dense, towers)
  - Location: `src/bernese/models/blocks.py`

### Data

- **`SeqDataset`**: PyTorch Dataset for genomic sequences
  - Location: `src/bernese/data/dataset.py`
  - Supports HDF5 and numpy formats

### Training

- **`Trainer`**: Comprehensive training loop with:
  - Multiple loss functions (MSE, Poisson KL, BCE)
  - Metrics (PearsonR, R2, AUROC, AUPRC)
  - Learning rate schedulers (cyclical, warmup, exponential)
  - Checkpointing and early stopping

---

## Related Rules

- [Global Rules](../Cline/Rules/) - Environment and best practice rules
- [UV Python Rules](1-uv-python-rules.md) - Package management

---

## Notes

- This project uses **CUDA 13.0** for PyTorch (configured in `pyproject.toml`)
- Data directories should contain `sequences.h5` and `statistics.json`
- Model configurations are typically stored as JSON files
