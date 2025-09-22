from biomol.core.biomol import BioMol
from biomol.core.feature import EdgeFeature, NodeFeature
from biomol.core.view import ViewProtocol


class AtomProtocol(
    ViewProtocol["AtomProtocol", "ResidueProtocol", "ChainProtocol", "CCDBioMol"]
):
    @property
    def ideal_coords(self) -> NodeFeature: ...

    @property
    def type(self) -> NodeFeature:
        """Type of the atom."""
        ...

    @property
    def chiral(self) -> NodeFeature:
        """Chirality of the atom."""
        ...

    @property
    def charge(self) -> NodeFeature:
        """Charge of the atom."""
        ...

    @property
    def bond_order(self) -> EdgeFeature:
        """Bond order of the atom."""
        ...

    @property
    def bond_aromacity(self) -> EdgeFeature:
        """Bond aromacity of the atom."""
        ...

    def bond_stereo(self) -> EdgeFeature:
        """Bond stereo of the atom."""
        ...


class ResidueProtocol(
    ViewProtocol["AtomProtocol", "ResidueProtocol", "ChainProtocol", "CCDBioMol"]
):
    @property
    def id(self) -> NodeFeature:
        """ID of the CCD."""
        ...

    @property
    def name(self) -> NodeFeature:
        """Name of the CCD."""
        ...

    @property
    def formula(self) -> NodeFeature:
        """Chemical formula of the CCD."""
        ...


class ChainProtocol(
    ViewProtocol["AtomProtocol", "ResidueProtocol", "ChainProtocol", "CCDBioMol"]
): ...


class CCDBioMol(BioMol[AtomProtocol, ResidueProtocol, ChainProtocol]):
    """BioMol for CCD structures."""
