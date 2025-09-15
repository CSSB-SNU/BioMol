import numpy as np
from Bio.PDB.MMCIF2Dict import MMCIF2Dict as mmcif2dict

from biomol.io.parser import Parser
from biomol.io.schema import MappingSpec, FeatureSpec, FeatureKind, FeatureLevel
from biomol.io.registry import MapperRegistry
from biomol.core.biomol import BioMol
from biomol.core.container import AtomContainer, ResidueContainer, ChainContainer
from biomol.core.index import IndexTable


CCD_PIPELINE_CONFIG = [
    # molecular metadata parsing
    MappingSpec(
        name="parse_chem_comp_properties",
        mapper="identity",
        inputs={"fields": ["_chem_comp.id", "_chem_comp.name", "_chem_comp.formula"]},
        outputs=[
            FeatureSpec(
                name="id",
                kind=FeatureKind.NODE,
                level=FeatureLevel.STRUCTURE,
                dtype=str,
            ),
            FeatureSpec(
                name="name",
                kind=FeatureKind.NODE,
                level=FeatureLevel.STRUCTURE,
                dtype=str,
            ),
            FeatureSpec(
                name="formula",
                kind=FeatureKind.NODE,
                level=FeatureLevel.STRUCTURE,
                dtype=str,
            ),
        ],
    ),
    # 2. atomic id
    MappingSpec(
        name="parse_atoms_as_vocab",
        mapper="identity",
        inputs={"fields": ["_chem_comp_atom.atom_id", "_chem_comp_atom.type_symbol"]},
        outputs=[
            # atomic id will be only used as vocab for bond parsing,
            # if you want to keep it as a feature, change FeatureKind.AUX -> FeatureKind.NODE
            FeatureSpec(
                name="atom_id",
                kind=FeatureKind.AUX,
                level=FeatureLevel.ATOM,
                dtype=str,
            ),
            FeatureSpec(
                name="atom_symbol",
                kind=FeatureKind.NODE,
                level=FeatureLevel.ATOM,
                dtype=str,
                on_missing={"?": "X"},
            ),
        ],
    ),
    # 3. Coordinates
    MappingSpec(
        name="parse_ideal_coordinates",
        mapper="stack",
        inputs={
            "fields": [
                "_chem_comp_atom.pdbx_model_Cartn_x_ideal",
                "_chem_comp_atom.pdbx_model_Cartn_y_ideal",
                "_chem_comp_atom.pdbx_model_Cartn_z_ideal",
            ]
        },
        outputs=[
            FeatureSpec(
                name="ideal_coords",
                kind=FeatureKind.NODE,
                level=FeatureLevel.ATOM,
                dtype=float,
                on_missing={"?": 0.0},
            )
        ],
    ),
    # 4. 2D Features: 결합(bond) 정보 파싱
    MappingSpec(
        name="parse_bonds",
        mapper="bond",
        inputs={
            "fields": [
                "_chem_comp_bond.atom_id_1",  # source node ID
                "_chem_comp_bond.atom_id_2",  # target node ID
                "_chem_comp_bond.value_order",  # edge feature 1
                "_chem_comp_bond.pdbx_aromatic_flag",  # edge feature 2
                "_chem_comp_bond.pdbx_stereo_config",  # edge feature 3
            ],
            "context": ["atom_id"],  # depend on the parsed atom_id vocab
        },
        outputs=[
            FeatureSpec(
                name="bond_type",
                kind=FeatureKind.EDGE,
                level=FeatureLevel.ATOM,
                dtype=str,
            ),
            FeatureSpec(
                name="aromatic",
                kind=FeatureKind.EDGE,
                level=FeatureLevel.ATOM,
                dtype=str,
            ),
            FeatureSpec(
                name="stereo", kind=FeatureKind.EDGE, level=FeatureLevel.ATOM, dtype=str
            ),
        ],
    ),
]


def parse_cif(cif_file: str) -> BioMol:
    """Parse a CIF file and return a Biomol object.

    Args:
        cif_file (str): Path to the CIF file.

    Returns:
        Biomol: Parsed Biomol object.
    """
    # Load the CIF file into a dictionary
    cif_dict = mmcif2dict(cif_file)

    # Initialize the parser with the CCD pipeline configuration
    parser = Parser(pipeline=CCD_PIPELINE_CONFIG)

    # Parse the CIF dictionary to create a Biomol object
    # TODO: separate building BioMol from ParsingContext into builder class
    parsed_features = parser.parse(cif_dict)
    atom_container = AtomContainer(
        node_features=parsed_features.get_features(
            FeatureSpec(
                name="", kind=FeatureKind.NODE, level=FeatureLevel.ATOM, dtype=None
            )
        ),
        edge_features=parsed_features.get_features(
            FeatureSpec(
                name="", kind=FeatureKind.EDGE, level=FeatureLevel.ATOM, dtype=None
            )
        ),
    )
    residue_container = ResidueContainer(
        node_features=parsed_features.get_features(
            FeatureSpec(
                name="", kind=FeatureKind.NODE, level=FeatureLevel.STRUCTURE, dtype=None
            )
        ),
        edge_features={},
    )
    chain_container = ChainContainer(
        node_features=parsed_features.get_features(
            FeatureSpec(
                name="", kind=FeatureKind.NODE, level=FeatureLevel.STRUCTURE, dtype=None
            )
        ),
        edge_features={},
    )

    num_atoms = len(parsed_features.get_feature("atom_id"))
    index_table = IndexTable.from_parents(
        atom_to_res=np.zeros((num_atoms,), dtype=np.int32),
        res_to_chain=np.zeros((1,), dtype=np.int32),
        n_chain=1,
    )

    return BioMol(atom_container, residue_container, chain_container, index_table)


if __name__ == "__main__":
    import sys
    import biomol.io.mappers.ccd_mappers

    cif_path = sys.argv[1]
    biomol = parse_cif(cif_path)
    print(biomol)
