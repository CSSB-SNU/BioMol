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
from .cif import CIFMol
from .core import BioMol
from .core.utils import load_bytes, to_bytes

__all__ = [
    "BioMol",
    "CIFMol",
    "__version__",
    "enums",
    "exceptions",
    "load_bytes",
    "to_bytes",
]
