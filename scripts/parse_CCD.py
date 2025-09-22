import sys
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from Bio.PDB.MMCIF2Dict import MMCIF2Dict as mmcif2dict

from biomol.core.biomol import BioMol
from biomol.core.view import ViewProtocol
from biomol.core.feature import EdgeFeature, NodeFeature
from biomol.core.index import IndexTable
from biomol.io.cache import ParsingCache
from biomol.io.cooker import Cooker
from biomol.io.builder import MolBuilder


@runtime_checkable
class AtomProtocol(
    ViewProtocol["AtomProtocol", "ResidueProtocol", "ChainProtocol", "CCDBioMol"],
    Protocol,
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

    @property
    def bond_stereo(self) -> EdgeFeature:
        """Bond stereo of the atom."""
        ...


@runtime_checkable
class ResidueProtocol(
    ViewProtocol["AtomProtocol", "ResidueProtocol", "ChainProtocol", "CCDBioMol"],
    Protocol,
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


@runtime_checkable
class ChainProtocol(
    ViewProtocol["AtomProtocol", "ResidueProtocol", "ChainProtocol", "CCDBioMol"],
    Protocol,
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


class CCDBioMol(BioMol[AtomProtocol, ResidueProtocol, ChainProtocol]):
    """BioMol for CCD structures."""


def main():
    if len(sys.argv) < 3:
        print("Usage: python parse_CCD.py <recipe_path> <path_to_cif>")
        sys.exit(1)

    recipe_path = sys.argv[1].strip()
    cif_path = sys.argv[2].strip()

    cache = ParsingCache()
    cif_data = mmcif2dict(cif_path)
    cooker = Cooker(parse_cache=cache, recipebook=recipe_path)
    cooker.prep(cif_data, fields=list(cif_data.keys()))
    cooker.cook()

    # NOTE: below is temporal code for CCD parsing.
    # it's bad example, so never access ParsingCache directly like below in your code.
    ### only for CCD, if you need dynamic index table description, it's up to you ###
    atom_to_res = np.zeros((len(cache["atom_id"])), dtype=int)
    res_to_chain = np.zeros((1,), dtype=int)
    num_chain = 1
    index_table = IndexTable.from_parents(atom_to_res, res_to_chain, num_chain)
    cache.add_data("index_table", index_table)

    mol_builder = MolBuilder(parsing_cache=cache, mol_guide=CCDBioMol)
    bio_mol = mol_builder.build()
    print(bio_mol)


if __name__ == "__main__":
    main()
