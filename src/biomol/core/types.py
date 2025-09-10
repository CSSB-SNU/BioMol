from __future__ import annotations

import enum


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
