# UV Python Project Rules

**Summary**: Comprehensive rules for developing this Python library project using `uv`, the fast Python package manager.

**Why**: `uv` provides significantly faster dependency resolution and installation compared to pip. These rules ensure consistent workflows across the project and leverage uv's best features for library development.

Last updated: 2026-02-28

---

## Table of Contents

1. [Project Initialization](#1-project-initialization)
2. [Virtual Environment Management](#2-virtual-environment-management)
3. [Dependency Management](#3-dependency-management)
4. [Running the Project](#4-running-the-project)
5. [Building & Publishing](#5-building--publishing)
6. [Development Workflow](#6-development-workflow)
7. [Project Structure](#7-project-structure)

---

## 1. Project Initialization

**Summary**: How to set up a new Python project with `uv`.

**Why**: Proper initialization ensures reproducible builds and correct dependency management from the start.

### ✅ Do This

```powershell
# Initialize a new project in the current directory
uv init

# Initialize with specific options
uv init --name "bernese" --python "3.13" --no-readme
```

### ❌ Don't Do This

```powershell
# Don't use pip/pipenv for this project - use uv exclusively
pip install -e .

# Don't create virtual environments manually
python -m venv .venv
```

**Why**: `uv init` automatically creates the correct `pyproject.toml`, `.python-version`, and sets up the project structure properly. It also initializes git if available.

### Setting Up Existing Project

```powershell
# Install dependencies from pyproject.toml
uv sync

# Or with specific Python version
uv sync --python 3.13
```

---

## 2. Virtual Environment Management

**Summary**: Managing virtual environments with `uv`.

**Why**: `uv` handles virtual environments more efficiently and integrates them with its workflow commands.

### ✅ Do This

```powershell
# Create and sync virtual environment
uv venv
uv sync

# Or in one command
uv sync --venv .venv

# Activate the virtual environment (for manual work)
.venv\Scripts\activate

# Check which Python version is being used
uv python list

# Install a specific Python version
uv python install 3.13
```

### ❌ Don't Do This

```powershell
# Don't manually create venv
python -m venv .venv
source .venv/bin/activate  # Unix style on Windows

# Don't install packages directly into system Python
pip install package_name
```

**Why**: `uv venv` creates optimized virtual environments. The `.venv` directory is the conventional name used by uv and other tools (like VS Code).

### Virtual Environment Location

The project uses `.venv` in the project root. This is configured in `.gitignore` and should not be committed.

---

## 3. Dependency Management

**Summary**: Adding, updating, and managing dependencies with `uv`.

**Why**: `uv` provides fast resolution and proper lock file management for reproducible builds.

### Adding Dependencies

```powershell
# Add a runtime dependency
uv add requests

# Add a development dependency (for testing, linting, etc.)
uv add --dev pytest

# Add an optional dependency group
uv add --optional httpx

# Add from a specific source
uv add package_name --index https://pypi.org/simple/
```

### Dependency Groups

```powershell
# Add to a specific dependency group
uv add pytest --group test

# View all dependency groups
uv pip tree
```

### Updating Dependencies

```powershell
# Update all dependencies to latest compatible versions
uv update

# Update a specific package
uv update requests

# Upgrade a package to latest (including breaking changes)
uv update requests --upgrade
```

### Lock File Management

```powershell
# Lock dependencies (generate/update uv.lock)
uv lock

# Sync with lock file
uv sync

# Lock with specific Python version
uv lock --python 3.13
```

### Optional Dependencies

This project uses optional dependencies for GPU support (CUDA). These are defined in `pyproject.toml` under `[project.optional-dependencies]`.

```powershell
# Sync base dependencies only (CPU)
uv sync

# Sync with optional dependencies (GPU/CUDA support)
uv sync --extra cu130

# Sync with multiple extras
uv sync --extra cu130 --extra <another-extra>

# Run with specific extra
uv run --extra cu130 your_script.py
```

**Why**: Optional dependencies allow installing platform-specific packages (like PyTorch with CUDA) only when needed. The project uses `cu130` for PyTorch with CUDA 13.0 support.

### Removing Dependencies

```powershell
# Remove a dependency
uv remove requests

# Remove a dev dependency
uv remove --dev pytest
```

### ❌ Don't Do This

```powershell
# Don't edit pyproject.toml manually for basic operations
# Use uv add/remove commands instead

# Don't use pip alongside uv for the same project
pip install package  # This can cause conflicts
```

**Why**:

- `uv lock` ensures reproducible builds across machines
- Using uv commands keeps `pyproject.toml` and `uv.lock` in sync
- Mixing pip and uv can cause dependency conflicts

---

## 4. Running the Project

**Summary**: Executing Python code and scripts with `uv`.

**Why**: `uv run` provides a convenient way to execute code with the correct environment.

### ✅ Do This

```powershell
# Run the main script
uv run main.py

# Run with specific Python version
uv run --python 3.13 main.py

# Run a function from a module
uv run -m module_name function_name

# Execute inline Python code
uv run -c "print('Hello')"
```

### Running Scripts from pyproject.toml

If you define scripts in `pyproject.toml`:

```toml
[project.scripts]
bernese = "bernese.__main__:main"
```

Run them with:

```powershell
uv run bernese
```

### Using the Virtual Environment

```powershell
# Activate venv and run (alternative to uv run)
.venv\Scripts\python main.py

# Run pytest through uv (when added)
uv run pytest
```

### ❌ Don't Do This

```powershell
# Don't run with system Python when venv exists
python main.py  # May use wrong Python

# Don't use uv run for interactive shells
uv run ipython  # Use activated venv instead
```

**Why**: `uv run` automatically uses the project's virtual environment and ensures proper Python version compatibility.

---

## 5. Building & Publishing

**Summary**: Building and publishing the library package.

**Why**: Proper build configuration ensures the library can be distributed via PyPI.

### Building the Package

```powershell
# Build source distribution and wheel
uv build

# Build only source distribution
uv build --sdist

# Build only wheel
uv build --wheel
```

### Publishing to PyPI

```powershell
# Publish to PyPI (requires configuration)
uv publish

# Publish with specific token
uv publish --token <pypi-token>

# Publish to Test PyPI first
uv publish --index https://test.pypi.org/simple/
```

### Package Configuration

Ensure `pyproject.toml` has proper metadata:

```toml
[project]
name = "bernese"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = []

# Optional: Add classifiers
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]

# Optional: Configure package layout
[tool.uv]
package = true
```

### ❌ Don't Do This

```powershell
# Don't build with setup.py if using pyproject.toml
python setup.py build

# Don't publish without testing on Test PyPI first
uv publish  # Direct to PyPI
```

**Why**: Test PyPI lets you verify the package works before releasing to production.

---

## 6. Development Workflow

**Summary**: Recommended development practices for this library.

**Why**: Consistent workflows improve productivity and code quality.

### Typical Development Cycle

```powershell
# 1. Create/update virtual environment
uv venv
uv sync --extra cu130  # Include GPU support (optional)

# 2. Make code changes

# 3. Test locally
uv run main.py

# 4. Add new dependencies as needed
uv add package_name

# 5. Update lock file
uv lock

# 6. Build and verify
uv build
```

**Note**: Use `uv sync --extra cu130` if you need PyTorch with CUDA support. Omit `--extra cu130` for CPU-only development.

### Adding Testing (When Needed)

```powershell
# Add pytest as dev dependency
uv add --dev pytest

# Run tests
uv run pytest

# Run with coverage (if added)
uv run pytest --cov=bernese --cov-report=html
```

### Code Quality

```powershell
# Check for issues (when linting tools added)
uv run ruff check .

# Format code (when formatter added)
uv run ruff format .
```

### ❌ Don't Do This

```powershell
# Don't skip lock file updates
# Always run uv lock after adding dependencies

# Don't test only in your local environment
# Ensure the package builds correctly
```

**Why**: Regular lock file updates ensure consistency across development environments and CI/CD.

---

## 7. Project Structure

**Summary**: Recommended directory structure for this Python library.

**Why**: Following standard Python packaging conventions ensures compatibility with uv and PyPI.

### Recommended Structure

```
bernese/
├── .clinerules/           # Cline rules (this directory)
├── .git/                  # Git repository
├── .gitignore
├── .python-version        # Python version specification
├── .venv/                 # Virtual environment (not committed)
├── README.md              # Project documentation
├── pyproject.toml         # Project configuration
├── uv.lock                # Locked dependencies
├── src/
│   └── bernese/           # Package source code
│       ├── __init__.py
│       └── __main__.py    # CLI entry point (if needed)
├── tests/                 # Test files (when added)
│   └── test_bernese.py
└── docs/                  # Documentation (if needed)
```

### Source Code Location

**Always use `src/` layout for libraries**:

```
src/
└── bernese/
    ├── __init__.py
    └── ...
```

This prevents import issues when the package is installed.

### ❌ Don't Do This

```powershell
# Don't use implicit package discovery (root-level modules)
# Bad:
bernese/
├── __init__.py  # This is a namespace package, not a proper package

# Don't put tests at root level
# Bad:
bernese/
├── test_*.py  # These won't be excluded from the package properly
```

**Why**: The `src/` layout is the modern standard for Python libraries, recommended by uv and pyproject.toml best practices.

---

## Quick Reference

### Common Commands

| Task | Command |
|------|---------|
| Initialize project | `uv init` |
| Create venv | `uv venv` |
| Install dependencies | `uv sync` |
| Install with extras | `uv sync --extra cu130` |
| Add dependency | `uv add <package>` |
| Add dev dependency | `uv add --dev <package>` |
| Remove dependency | `uv remove <package>` |
| Update all | `uv update` |
| Lock dependencies | `uv lock` |
| Run script | `uv run <script>` |
| Run module | `uv run -m <module>` |
| Build package | `uv build` |
| Publish | `uv publish` |
| Check Python versions | `uv python list` |
| Install Python | `uv python install <version>` |
