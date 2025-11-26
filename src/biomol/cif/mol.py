# pyright: reportReturnType=false
from biomol.core import BioMol, EdgeFeature, NodeFeature, View


class CIFAtomView(
    View["CIFAtomView", "CIFResidueView", "CIFChainView", "CIFMol"],
):
    """View class for CIF atoms."""

    @property
    def id(self) -> NodeFeature:
        """Atom identifiers from the CIF file."""

    @property
    def element(self) -> NodeFeature:
        """Chemical element symbols (e.g., 'C', 'N', 'O', 'S', 'Fe')."""

    @property
    def aromatic(self) -> NodeFeature:
        """Aromaticity flag for atoms."""

    @property
    def stereo(self) -> NodeFeature:
        """Stereochemistry configuration at chiral centers.

        Labels:
        - 'R' : R configuration
        - 'S' : S configuration
        - 'N' : no stereochemistry
        """

    @property
    def charge(self) -> NodeFeature:
        """Formal charge on each atom."""

    @property
    def model_xyz(self) -> NodeFeature:
        """Idealized coordinates from chemical component dictionary."""

    @property
    def xyz(self) -> NodeFeature:
        """Experimental 3D coordinates of atoms."""

    @property
    def b_factor(self) -> NodeFeature:
        """Temperature factors (B-factors)."""

    @property
    def occupancy(self) -> NodeFeature:
        """Fractional occupancy of atomic positions."""

    @property
    def bond_type(self) -> EdgeFeature:
        """Chemical bond types between atoms.

        Labels:
        - 'SING' : Single bond
        - 'DOUB' : Double bond
        - 'TRIP' : Triple bond
        """

    @property
    def bond_aromatic(self) -> EdgeFeature:
        """Aromaticity flag for bonds."""

    @property
    def bond_stereo(self) -> EdgeFeature:
        """Stereochemical configuration of bonds.

        Labels:
        - 'E' : E configuration
        - 'Z' : Z configuration
        - 'N' : no stereochemistry
        """


class CIFResidueView(
    View["CIFAtomView", "CIFResidueView", "CIFChainView", "CIFMol"],
):
    """View class for CIF residues."""

    @property
    def name(self) -> NodeFeature:
        """Residue names."""

    @property
    def formula(self) -> NodeFeature:
        """Chemical formulas of residues (e.g., 'C3 H7 N O2' for alanine)."""

    @property
    def one_letter_code_can(self) -> NodeFeature:
        """Canonical one-letter amino acid codes."""

    @property
    def one_letter_code(self) -> NodeFeature:
        """One-letter codes including non-canonical residues."""

    @property
    def cif_idx(self) -> NodeFeature:
        """CIF residue indices."""

    @property
    def auth_idx(self) -> NodeFeature:
        """Author-provided residue indices."""

    @property
    def chem_comp_id(self) -> NodeFeature:
        """Chemical component identifiers."""

    @property
    def hetero(self) -> NodeFeature:
        """Heteroatom flag."""

    @property
    def bond(self) -> EdgeFeature:
        """Residue-level connectivity."""

    @property
    def struct_conn(self) -> EdgeFeature:
        """Structural connections between residues.

        Information about non-standard connections such as disulfide bonds, metal
        coordination, or covalent modifications.
        """


class CIFChainView(
    View["CIFAtomView", "CIFResidueView", "CIFChainView", "CIFMol"],
):
    """View class for CIF chains."""

    @property
    def entity_id(self) -> NodeFeature:
        """Entity identifiers."""

    @property
    def entity_type(self) -> NodeFeature:
        """Entity types."""

    @property
    def chain_id(self) -> NodeFeature:
        """Chain identifiers with symmetry operator."""

    @property
    def auth_asym_id(self) -> NodeFeature:
        """Author-provided chain identifiers."""

    @property
    def contact(self) -> EdgeFeature:
        """Inter-chain contact graph."""

    @property
    def cluster_id(self) -> NodeFeature:
        """Sequence cluster identifiers."""

    @property
    def seq_id(self) -> NodeFeature:
        """Unique sequence identifiers."""


class CIFMol(BioMol["CIFAtomView", "CIFResidueView", "CIFChainView"]):
    """A class representing a biomolecular structure in CIF format."""


    @property
    def id(self) -> str:
        """PDB identifier."""
        return self.metadata["id"]

    @property
    def assembly_id(self) -> str:
        """Biological assembly identifier."""
        return self.metadata["assembly_id"]

    @property
    def model_id(self) -> int:
        """Model number in multi-model structures."""
        return self.metadata["model_id"]

    @property
    def alt_id(self) -> str:
        """Alternate location identifier."""
        return self.metadata["alt_id"]
