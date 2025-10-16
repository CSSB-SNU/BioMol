"""The core data model for the BioMol package."""

from __future__ import annotations

from .mol import CIFMol, CIFAtomView, CIFResidueView, CIFChainView

__all__ = [
    "CIFMol",
    "CIFAtomView",
    "CIFResidueView",
    "CIFChainView",
]
