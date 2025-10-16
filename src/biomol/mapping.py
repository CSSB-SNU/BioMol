from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, ClassVar

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


class PolymerType(Enum):
    """Enumeration of biological polymer types."""

    PROTEIN = "protein"
    RNA = "rna"
    DNA = "dna"
    LIGAND = "ligand"


class BaseResidueView:
    """Base view class for residue-to-index mappings."""

    GAP_INDEX: ClassVar[int] = 31

    def __init__(self, mapping_table: ResidueMapping) -> None:
        """Initialize view with a reference to the parent mapping table."""
        self._table = mapping_table

    def map(self, residues: Sequence[str] | np.ndarray) -> np.ndarray:
        """Map a sequence of residue symbols into integer indices."""
        if isinstance(residues, str):
            residues = [residues]
        # Use np.fromiter for efficient conversion
        return np.fromiter((self._map_single(r) for r in residues), dtype=np.int32)

    def _map_single(self, residue: str) -> int:
        """Map a single residue symbol to an integer index (to be implemented)."""
        raise NotImplementedError


class ProteinView(BaseResidueView):
    """Mapping view for protein amino acids."""

    def _map_single(self, aa: str) -> int:
        if aa == "-":
            return self.GAP_INDEX
        return self._table.AA2NUM.get(aa, self._table.AA_UNKNOWN)


class RNAView(BaseResidueView):
    """Mapping view for RNA nucleotides."""

    def _map_single(self, base: str) -> int:
        if base == "-":
            return self.GAP_INDEX
        return self._table.RNA2NUM.get(base, self._table.RNA_UNKNOWN)


class DNAView(BaseResidueView):
    """Mapping view for DNA nucleotides."""

    def _map_single(self, base: str) -> int:
        if base == "-":
            return self.GAP_INDEX
        return self._table.DNA2NUM.get(base, self._table.DNA_UNKNOWN)


class LigandView(BaseResidueView):
    """Mapping view for ligands or non-polymeric residues."""

    def _map_single(self, _: str) -> int:
        return self._table.LIGAND_INDEX


class ResidueMapping:
    """Unified residue-to-index mapping system providing specialized views."""

    # --- Protein constants ---
    NUM2AA: ClassVar[list[str]] = [
        "A",
        "R",
        "N",
        "D",
        "C",
        "Q",
        "E",
        "G",
        "H",
        "I",
        "L",
        "K",
        "M",
        "F",
        "P",
        "S",
        "T",
        "W",
        "Y",
        "V",
        "X",
    ]
    AA2NUM: ClassVar[dict[str, int]] = {x: i for i, x in enumerate(NUM2AA)}
    AA2NUM.update(
        {
            "B": 3,  # Asx (D/N)
            "J": 20,  # X
            "O": 20,  # Pyrrolysine
            "U": 4,  # Selenocysteine
            "Z": 6,  # Glx (E/Q)
        },
    )
    AA_UNKNOWN: ClassVar[int] = 20

    # --- RNA constants ---
    RNA2NUM: ClassVar[dict[str, int]] = {"A": 21, "U": 22, "G": 23, "C": 24}
    RNA_UNKNOWN: ClassVar[int] = 25

    # --- DNA constants ---
    DNA2NUM: ClassVar[dict[str, int]] = {"A": 26, "T": 27, "G": 28, "C": 29}
    DNA_UNKNOWN: ClassVar[int] = 30

    # --- Special tokens ---
    LIGAND_INDEX: ClassVar[int] = 20

    MAX_INDEX: ClassVar[int] = 31  # Including gap

    def __init__(self) -> None:
        """Initialize all mapping views for different polymer types."""
        self.protein = ProteinView(self)
        self.rna = RNAView(self)
        self.dna = DNAView(self)
        self.ligand = LigandView(self)

    def get_view(self, polymer_type: PolymerType) -> BaseResidueView:
        """Return the corresponding mapping view for the given polymer type."""
        match polymer_type:
            case PolymerType.PROTEIN:
                return self.protein
            case PolymerType.RNA:
                return self.rna
            case PolymerType.DNA:
                return self.dna
            case _:
                return self.ligand
