# Common Operations

**Summary**: Quick reference for common CLI operations in Bernese.

**Why**: This document provides quick command references for everyday development tasks.

Last updated: 2026-03-01

---

## Installation & Dependencies

| Task | Command |
|------|---------|
| Install dependencies (with GPU/CUDA) | `uv sync --extra cu130` |
| Install dependencies (CPU only) | `uv sync` |
| Add dependency | `uv add <package>` |
| Add dev dependency | `uv add --dev <package>` |
| Update lock file | `uv lock` |

**Note**: Always use `--extra cu130` for GPU support (CUDA 13.0). Omit for CPU-only.

---

## Running the Project

| Task | Command |
|------|---------|
| Inspect model architecture | `uv run bernese summary <params_file>` |
| Train model | `uv run bernese train <params_file> <data_dirs>...` |
| Prepare genomic data | `uv run bernese prepare <genome> <targets> -o <output_dir>` |
| Run tests | `uv run pytest` |
| Build package | `uv build` |

---

## Model Inspection

Use `bernese summary` to visualize model architecture:

```bash
# Basic usage
uv run bernese summary params.json

# With options
uv run bernese summary params.json --num_targets 100
uv run bernese summary params.json --device cpu
```

This uses `torchinfo` to display:
- Layer types and order
- Input/output shapes
- Parameter counts
- Memory usage estimates

---

## Related Rules

- [Overview](0-overview.md) - Project overview
- [UV Python Rules](1-uv-python-rules.md) - Package management details
- [Code Style Rules](2-code-style-rules.md) - Code style guidelines
