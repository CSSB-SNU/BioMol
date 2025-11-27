# BioMol

<p align="center">
  <img src="https://raw.githubusercontent.com/CSSB-SNU/BioMol/main/docs/_static/logo-light.png" width="450" alt="BioMol Logo">
</p>

BioMol is a molecular data engine that provides PyMOL-like selections and NumPy-style
operations, designed for machine learning and large-scale molecular analysis.

For more details, see the [documentation](https://biomol.readthedocs.io/en/latest/index.html).

## Installation

BioMol supports Python 3.10 and above.

### Install from PyPI

```bash
pip install biomol
```

### Install from Source

For development, BioMol uses [uv](https://docs.astral.sh/uv/) for fast dependency management:

```bash
git clone https://github.com/CSSB-SNU/BioMol
cd BioMol
uv sync
```

For development with all tools (linting, testing, etc.):

```bash
uv sync --extra dev
```
