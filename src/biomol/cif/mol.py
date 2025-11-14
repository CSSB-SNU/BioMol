# pyright: reportReturnType=false
from biomol import BioMol
from biomol.core import EdgeFeature, NodeFeature, View


class CIFAtomView(
    View["CIFAtomView", "CIFResidueView", "CIFChainView", "CIFMol"],
):
    """View class for CIF atoms."""

    @property
    def name(self) -> NodeFeature:
        """Atom IDs. Example: 'N', 'CA', 'C', 'O', etc."""

    @property
    def element(self) -> NodeFeature:
        """Atom elements. Example: 'C', 'N', 'O', etc."""

    @property
    def aromatic(self) -> NodeFeature:
        """Aromatic."""

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
    View["CIFAtomView", "CIFResidueView", "CIFChainView", "CIFMol"],
):
    """View class for CIF residues."""

    @property
    def name(self) -> NodeFeature:
        """Residue names."""

    @property
    def formula(self) -> NodeFeature:
        """Residue chemical formulas."""

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
    def chem_comp_id(self) -> NodeFeature:
        """Chemical component IDs."""

    @property
    def hetero(self) -> NodeFeature:
        """Hetero flag."""

    @property
    def bond(self) -> EdgeFeature:
        """Residue-level bonds 1 if exists else not."""

    @property
    def struct_conn(self) -> EdgeFeature:
        """struct_conn of residues."""


class CIFChainView(
    View["CIFAtomView", "CIFResidueView", "CIFChainView", "CIFMol"],
):
    """View class for CIF chains."""

    @property
    def entity_id(self) -> NodeFeature:
        """Entity IDs."""

    @property
    def entity_type(self) -> NodeFeature:
        """Entity types. Example: 'polymer', 'non-polymer', etc."""

    @property
    def chain_id(self) -> NodeFeature:
        """Chain IDs. asym_id_oper_id. Example: 'A_1', 'B_1', etc."""

    @property
    def auth_asym_id(self) -> NodeFeature:
        """Author chain IDs. Example: 'A', 'B', etc."""

    @property
    def contact(self) -> EdgeFeature:
        """Contact graph of chains."""


class CIFMol(BioMol["CIFAtomView", "CIFResidueView", "CIFChainView"]):
    """Class for CIF molecules."""

    @property
    def id(self) -> str:
        """PDB ID of the molecule."""
        return self.metadata["id"]

    @property
    def assembly_id(self) -> str:
        """Assembly ID of the molecule."""
        return self.metadata["assembly_id"]

    @property
    def model_id(self) -> int:
        """Model ID of the molecule."""
        return self.metadata["model_id"]

    @property
    def alt_id(self) -> str:
        """Alternate location ID of the molecule."""
        return self.metadata["alt_id"]
