"""The core data model for the BioMol package."""

from __future__ import annotations

from .biomol import BioMol
from .container import FeatureContainer
from .feature import EdgeFeature, Feature, NodeFeature
from .index import IndexTable
from .view import ViewProtocol

__all__ = [
    "BioMol",
    "EdgeFeature",
    "Feature",
    "FeatureContainer",
    "IndexTable",
    "NodeFeature",
    "ViewProtocol",
]
