"""ML-ready biomolecular data handling package.

This package provides tools for representing, manipulating, and analyzing biomolecular
structures and sequences. It is designed to integrate seamlessly with machine learning
workflows by offering standardized data representations (e.g., arrays) and utilities for
scientific computing.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"

from . import enums, exceptions
from .core import BioMol

__all__ = [
    "BioMol",
    "__version__",
    "enums",
    "exceptions",
]
