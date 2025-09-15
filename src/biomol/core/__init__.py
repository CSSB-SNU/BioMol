"""The core data model for the BioMol package."""

from __future__ import annotations

from .biomol import BioMol
from .container import (
    AtomContainer,
    ChainContainer,
    FeatureContainer,
    ResidueContainer,
)
from .feature import EdgeFeature, Feature, NodeFeature
from .index import IndexTable
from .view import AtomView, ChainView, ResidueView, ViewProtocol

__all__ = [
    "AtomContainer",
    "AtomView",
    "BioMol",
    "ChainContainer",
    "ChainView",
    "EdgeFeature",
    "Feature",
    "FeatureContainer",
    "IndexTable",
    "NodeFeature",
    "ResidueContainer",
    "ResidueView",
    "ViewProtocol",
]
