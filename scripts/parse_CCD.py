import os
import sys

from biomol.io.blueprint import Blueprint
from biomol.io.factory import BioMolFactory
from Bio.PDB.MMCIF2Dict import MMCIF2Dict as mmcif2dict

import biomol.io.instructions.common_instructions


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_CCD.py <path_to_file_>")
        sys.exit(1)

    cifs_path = sys.argv[1]
    cif_list = open(cifs_path, "r").readlines()

    # ---- define the parsing plan ----#
    ccd_plan = (
        Blueprint()
        .stage("parse_chem_comp_properties")
        .using("identity")
        .from_fields("_chem_comp.id", "_chem_comp.name", "_chem_comp.formula")
        .to_residue_nodes(id=str, name=str, formula=str)
    )
    ccd_plan = (
        ccd_plan.stage("parse_atoms")
        .using("identity")
        .from_fields("_chem_comp_atom.atom_id", "_chem_comp_atom.type_symbol")
        .to_atom_nodes(atom_id=str, atom_symbol=(str, {"?": "X"}))
    )
    ccd_plan = (
        ccd_plan.stage("parse_ideal_coordinates")
        .using("stack")
        .from_fields(
            "_chem_comp_atom.pdbx_model_Cartn_x_ideal",
            "_chem_comp_atom.pdbx_model_Cartn_y_ideal",
            "_chem_comp_atom.pdbx_model_Cartn_z_ideal",
        )
        .to_atom_nodes(ideal_coords=(float, {"?": 0.0}))
    )
    ccd_plan = (
        ccd_plan.stage("parse_bonds")
        .using("bond")
        .from_fields(
            "_chem_comp_bond.atom_id_1",
            "_chem_comp_bond.atom_id_2",
            "_chem_comp_bond.value_order",
            "_chem_comp_bond.pdbx_aromatic_flag",
            "_chem_comp_bond.pdbx_stereo_config",
        )
        .with_context("atom_id")
        .to_atom_edges(bond_type=str, aromatic=str, stereo=str)
    )
    ccd_plan = ccd_plan.build()

    factory = BioMolFactory(num_workers=4)
    factory.load_plan(ccd_plan)

    cif_data = []
    for cif in cif_list:
        cif_dict = mmcif2dict(cif.strip())
        cif_data.append(cif_dict)

    biomol_list = factory.produce(dataset=cif_data)

    # 결과 출력
    if biomol_list:
        print("Successfully parsed BioMol object:")
        for biomol in biomol_list:
            print(biomol)


if __name__ == "__main__":
    main()
