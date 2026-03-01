# Bernese Clinerules Overview

**Summary**: Project-specific rules and guidelines for developing Bernese, a PyTorch-based library for regulatory genomics predictions.

**Why**: These rules ensure consistent development practices, proper dependency management, and maintainable code across the Bernese project.

Last updated: 2026-02-28

---

## Project Overview

Bernese is a **PyTorch-based deep learning library for regulatory genomics predictions**. It implements the SeqNN (Sequence Neural Network) model for predicting regulatory activity from DNA sequences.

| Aspect | Details |
|--------|---------|
| **Type** | Python library for genomic/molecular biology deep learning |
| **Core Model** | SeqNN - Sequential Neural Network for regulatory activity prediction |
| **Data Format** | HDF5 files (sequences.h5) with JSON metadata (statistics.json) |
| **Python** | >=3.13 |
| **Key Dependencies** | numpy, pandas, scipy, h5py, scikit-learn, torch |

---

## Navigating the Rules

| Document | Description |
|----------|-------------|
| [`1-uv-python-rules.md`](1-uv-python-rules.md) | UV Python package manager usage guide |

---

## Quick Reference

### Common Commands

| Task | Command |
|------|---------|
| Install dependencies | `uv sync` |
| Add dependency | `uv add <package>` |
| Run CLI | `uv run bernese train <params_file> <data_dirs>...` |
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

- **`SeqNN`**: Main model class for regulatory activity prediction
  - Location: `src/bernese/models/seqnn.py`
  - Configuration: Dictionary-based params (see `DEFAULT_CONFIG`)

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
