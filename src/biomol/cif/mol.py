from biomol import BioMol
from biomol.core import EdgeFeature, NodeFeature, ViewProtocol


class CIFAtomView(
    ViewProtocol["CIFAtomView", "CIFResidueView", "CIFChainView", "CIFMol"],
):
    """View class for CIF atoms."""

    @property
    def coord(self) -> NodeFeature:
        """XYZ coordinates of atoms."""

    @property
    def name(self) -> NodeFeature:
        """Atom names."""

    @property
    def element(self) -> NodeFeature:
        """Atom elements."""

    @property
    def bond(self) -> EdgeFeature:
        """Covalent bond."""


class CIFResidueView(
    ViewProtocol["CIFAtomView", "CIFResidueView", "CIFChainView", "CIFMol"],
):
    """View class for CIF residues."""

    @property
    def id(self) -> NodeFeature:
        """Residue indices."""

    @property
    def name(self) -> NodeFeature:
        """Residue names."""

    @property
    def wc_pair(self) -> EdgeFeature:
        """Watson-Crick base pair."""

    @property
    def non_wc_pair(self) -> EdgeFeature:
        """Non Watson-Crick base pair."""


class CIFChainView(
    ViewProtocol["CIFAtomView", "CIFResidueView", "CIFChainView", "CIFMol"],
):
    """View class for CIF chains."""

    @property
    def name(self) -> NodeFeature:
        """Chain names."""


class CIFMol(BioMol["CIFAtomView", "CIFResidueView", "CIFChainView"]):
    """Class for CIF molecules."""
