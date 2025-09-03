from __future__ import annotations

import enum
from typing import Protocol, runtime_checkable

from typing_extensions import TypeVar


@enum.unique
class MoleculeType(enum.Enum):
    """Molecule types commonly found in biomolecular structures."""

    POLYMER = "polymer"
    NONPOLYMER = "non-polymer"
    BRANCHED = "branched"
    WATER = "water"
    BIOASSEMBLY = "bioassembly"


@enum.unique
class PolymerType(enum.Enum):
    """Types of polymer molecules."""

    PROTEIN = "polypeptide(L)"
    PROTEIN_D = "polypeptide(D)"
    PNA = "peptide nucleic acid"
    RNA = "polyribonucleotide"
    DNA = "polydeoxyribonucleotide"
    NA_HYBRID = "polydeoxyribonucleotide/polyribonucleotide hybrid"
    ETC = "etc"


@enum.unique
class StructureLevel(enum.Enum):
    """Levels of structural hierarchy in biomolecules."""

    ATOM = "atom"
    RESIDUE = "residue"
    CHAIN = "chain"


@runtime_checkable
class ViewProtocol(Protocol):
    """Protocol for views."""


AtomProtoT = TypeVar("AtomProtoT", default=ViewProtocol)
ResidueProtoT = TypeVar("ResidueProtoT", default=ViewProtocol)
ChainProtoT = TypeVar("ChainProtoT", default=ViewProtocol)
