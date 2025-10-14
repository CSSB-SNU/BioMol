from biomol import BioMol
from biomol.core import EdgeFeature, NodeFeature, ViewProtocol


class CIFAtomView(
    ViewProtocol["CIFAtomView", "CIFResidueView", "CIFChainView", "CIFMol"],
):
    """View class for CIF atoms."""

    @property
    def atom_id(self) -> NodeFeature:
        """Atom IDs. Example: 'N', 'CA', 'C', 'O', etc."""

    @property
    def element(self) -> NodeFeature:
        """Atom elements. Example: 'C', 'N', 'O', etc."""

    @property
    def aromatic(self) -> NodeFeature:
        """Aromatic flag."""

    @property
    def stereo(self) -> NodeFeature:
        """Stereochemistry flag."""

    @property
    def charge(self) -> NodeFeature:
        """Formal charge of atoms."""

    @property
    def model_xyz(self) -> NodeFeature:
        """Model XYZ coordinates of atoms in chemical component."""

    @property
    def xyz(self) -> NodeFeature:
        """XYZ coordinates of atoms."""

    @property
    def b_factor(self) -> NodeFeature:
        """B-factors of atoms."""

    @property
    def occupancy(self) -> NodeFeature:
        """Occupancy of atoms."""

    @property
    def bond_type(self) -> EdgeFeature:
        """Bond types between atoms."""

    @property
    def bond_aromatic(self) -> EdgeFeature:
        """Aromatic bonds between atoms."""

    @property
    def bond_stereo(self) -> EdgeFeature:
        """Bond stereochemistry between atoms."""


class CIFResidueView(
    ViewProtocol["CIFAtomView", "CIFResidueView", "CIFChainView", "CIFMol"],
):
    """View class for CIF residues."""

    @property
    def name(self) -> NodeFeature:
        """Residue names."""

    @property
    def formula(self) -> NodeFeature:
        """Residue formulas."""

    @property
    def one_letter_code_can(self) -> NodeFeature:
        """One-letter code (canonical)."""

    @property
    def one_letter_code(self) -> NodeFeature:
        """One-letter code (not canonical)."""

    @property
    def cif_idx(self) -> NodeFeature:
        """CIF residue indices."""

    @property
    def auth_idx(self) -> NodeFeature:
        """Author residue indices."""

    @property
    def chem_comp(self) -> NodeFeature:
        """Chemical component IDs."""

    @property
    def hetero(self) -> NodeFeature:
        """Hetero flag."""

    @property
    def residue_bond(self) -> EdgeFeature:
        """Residue-level bonds 1 if exists else not."""


class CIFChainView(
    ViewProtocol["CIFAtomView", "CIFResidueView", "CIFChainView", "CIFMol"],
):
    """View class for CIF chains."""

    @property
    def entity_id(self) -> NodeFeature:
        """Entity IDs."""

    @property
    def chain_id(self) -> NodeFeature:
        """Chain IDs. asym_id_oper_id. Example: 'A_1', 'B_1', etc."""


class CIFMol(BioMol["CIFAtomView", "CIFResidueView", "CIFChainView"]):
    """Class for CIF molecules."""
