from pathlib import Path

from biomol.io.instructions.cif_instructions import (
    attach_entity,
    build_assembly_dict,
    build_full_length_asym_dict,
    compare_chem_comp,
    extract_float_single,
    get_smaller_dict,
    get_struct_oper,
    key_stack,
    merge_dict,
    parse_assembly_dict,
    parse_chem_comp,
    parse_entity_dict,
    parse_scheme_dict,
    rearrange_atom_site_dict,
    remove_unknown_atom_site,
    remove_unknown_from_struct_conn,
    single_value_instruction,
)
from biomol.io.recipe import RecipeBook

"""Build a CIF-specific Cooker.

This factory function constructs and returns a Cooker preconfigured
with CIF parsing recipes and instructions.
"""

cif_recipe = RecipeBook()

"""
1. Metadata Extraction
    - Extract basic metadata like id, resolution, etc.
2. Merge Instructions
    - Merge multiple sources of data into a single small dict like chem comp, etc.
3. Calculate chain num, length, etc.

"""


# 1. Metadata Extraction
cif_recipe.add(
    targets=[
        ("deposition_date", str),
    ],
    instruction=single_value_instruction(dtype=str),
    inputs={
        "args": (("_pdbx_database_status.recvd_initial_deposition_date", str),),
    },
)

cif_recipe.add(
    targets=(("resolution", float),),
    instruction=extract_float_single,
    inputs={
        "args": (
            ("_refine.ls_d_res_high", str | None),
            ("_em_3d_reconstruction.resolution", str | None),
        ),
    },
)


cif_recipe.add(
    targets=(("metadata_dict", dict | None),),
    instruction=key_stack(),
    inputs={
        "kwargs": {
            "id": ("_entry.id", list),
            "deposition_date": ("deposition_date", str),
            "resolution": ("resolution", float | None),
        },
    },
)

# 2. Merge Instructions

cif_recipe.add(
    targets=[
        (("_chem_comp_dict", dict),),
        (("_chem_comp_atom_dict", dict),),
        (("_chem_comp_bond_dict", dict),),
        (("_pdbx_poly_seq_scheme_dict", dict | None),),
        (("_pdbx_nonpoly_scheme_dict", dict | None),),
        (("_pdbx_branch_scheme_dict", dict | None),),
        (("_atom_site_dict", dict),),
        (("_entity_dict", dict),),
        (("_entity_poly_dict", dict | None),),
        (("_entity_poly_seq_dict", dict | None),),
        (("_entity_nonpoly_dict", dict | None),),
        (("_entity_branch_descriptor_dict", dict | None),),
        (("_entity_branch_link_dict", dict | None),),
        (("_entity_branch_list_dict", dict | None),),
        (("_struct_oper_dict", dict | None),),
        (("_struct_assembly_gen_dict", dict | None),),
        (("_struct_conn_dict", dict | None),),
    ],
    instruction=get_smaller_dict(dtype=str),
    inputs=[
        {
            "kwargs": {"cif_raw_dict": ("_chem_comp", str | None)},
            "params": {
                "tied_to": "id",
                "columns": ["name", "formula"],
            },
        },
        {
            "kwargs": {"cif_raw_dict": ("_chem_comp_atom", str | None)},
            "params": {
                "tied_to": "comp_id",
                "columns": [
                    "atom_id",
                    "type_symbol",
                    "charge",
                    "pdbx_aromatic_flag",
                    "pdbx_stereo_config",
                ],
            },
        },
        {
            "kwargs": {"cif_raw_dict": ("_chem_comp_bond", str | None)},
            "params": {
                "tied_to": "comp_id",
                "columns": [
                    "atom_id_1",
                    "atom_id_2",
                    "value_order",
                    "pdbx_aromatic_flag",
                    "pdbx_stereo_config",
                ],
            },
        },
        {
            "kwargs": {"cif_raw_dict": ("_pdbx_poly_seq_scheme", str | None)},
            "params": {
                "tied_to": "asym_id",
                "columns": [
                    "entity_id",
                    "seq_id",
                    "mon_id",
                    "pdb_seq_num",
                    "pdb_ins_code",
                    "hetero",
                ],
            },
        },
        {
            "kwargs": {"cif_raw_dict": ("_pdbx_nonpoly_scheme", str | None)},
            "params": {
                "tied_to": "asym_id",
                "columns": ["entity_id", "mon_id", "pdb_seq_num", "pdb_ins_code"],
            },
        },
        {
            "kwargs": {"cif_raw_dict": ("_pdbx_branch_scheme", str | None)},
            "params": {
                "tied_to": "asym_id",
                "columns": ["entity_id", "mon_id", "pdb_seq_num", "hetero"],
            },
        },
        {
            "kwargs": {"cif_raw_dict": ("_atom_site", str)},
            "params": {
                "tied_to": "label_asym_id",
                "columns": [
                    "pdbx_PDB_model_num",
                    "label_alt_id",
                    "label_seq_id",
                    "auth_seq_id",
                    "pdbx_PDB_ins_code",
                    "Cartn_x",
                    "Cartn_y",
                    "Cartn_z",
                    "occupancy",
                    "B_iso_or_equiv",
                    "label_atom_id",
                    "type_symbol",
                    "label_comp_id",
                    "auth_asym_id",
                ],
            },
        },
        {
            "kwargs": {"cif_raw_dict": ("_entity", str | None)},
            "params": {
                "tied_to": "id",
                "columns": ["type"],
            },
        },
        {
            "kwargs": {"cif_raw_dict": ("_entity_poly", str | None)},
            "params": {
                "tied_to": "entity_id",
                "columns": [
                    "type",
                    "pdbx_seq_one_letter_code_can",
                    "pdbx_seq_one_letter_code",
                ],
            },
        },
        {
            "kwargs": {"cif_raw_dict": ("_entity_poly_seq", str | None)},
            "params": {
                "tied_to": "entity_id",
                "columns": ["num", "mon_id", "hetero"],
            },
        },
        {
            "kwargs": {"cif_raw_dict": ("_pdbx_entity_nonpoly", str | None)},
            "params": {
                "tied_to": "entity_id",
                "columns": ["comp_id"],
            },
        },
        {
            "kwargs": {"cif_raw_dict": ("_pdbx_entity_branch_descriptor", str | None)},
            "params": {
                "tied_to": "entity_id",
                "columns": ["descriptor"],
            },
        },
        {
            "kwargs": {"cif_raw_dict": ("_pdbx_entity_branch_link", str | None)},
            "params": {
                "tied_to": "entity_id",
                "columns": [
                    "comp_id_1",
                    "comp_id_2",
                    "atom_id_1",
                    "atom_id_2",
                    "leaving_atom_id_1",
                    "leaving_atom_id_2",
                    "value_order",
                    "entity_branch_list_num_1",
                    "entity_branch_list_num_2",
                ],
            },
        },
        {
            "kwargs": {"cif_raw_dict": ("_pdbx_entity_branch_list", str | None)},
            "params": {
                "tied_to": "entity_id",
                "columns": ["num", "comp_id", "hetero"],
            },
        },
        {
            "kwargs": {"cif_raw_dict": ("_pdbx_struct_oper_list", str | None)},
            "params": {
                "tied_to": "id",
                "columns": [
                    "matrix[1][1]",
                    "matrix[1][2]",
                    "matrix[1][3]",
                    "matrix[2][1]",
                    "matrix[2][2]",
                    "matrix[2][3]",
                    "matrix[3][1]",
                    "matrix[3][2]",
                    "matrix[3][3]",
                    "vector[1]",
                    "vector[2]",
                    "vector[3]",
                ],
            },
        },
        {
            "kwargs": {"cif_raw_dict": ("_pdbx_struct_assembly_gen", str | None)},
            "params": {
                "tied_to": "assembly_id",
                "columns": [
                    "oper_expression",
                    "asym_id_list",
                ],
            },
        },
        {
            "kwargs": {"cif_raw_dict": ("_struct_conn", str | None)},
            "params": {
                "tied_to": ("ptnr1_label_asym_id", "ptnr2_label_asym_id"),
                "columns": [
                    "conn_type_id",
                    "pdbx_ptnr1_label_alt_id",
                    "pdbx_ptnr2_label_alt_id",
                    "ptnr1_label_comp_id",
                    "ptnr2_label_comp_id",
                    "ptnr1_auth_seq_id",
                    "ptnr2_auth_seq_id",
                    "pdbx_ptnr1_PDB_ins_code",
                    "pdbx_ptnr2_PDB_ins_code",
                    "ptnr1_label_atom_id",
                    "ptnr2_label_atom_id",
                    "pdbx_value_order",
                ],
            },
        },
    ],
)


cif_recipe.add(
    targets=(("chem_comp_dict", dict),),
    instruction=parse_chem_comp(),
    inputs={
        "kwargs": {
            "chem_comp_dict": ("_chem_comp_dict", dict | None),
            "chem_comp_atom_dict": ("_chem_comp_atom_dict", dict | None),
            "chem_comp_bond_dict": ("_chem_comp_bond_dict", dict | None),
        },
        "params": {
            "remove_hydrogen": True,
        },
    },
)


cif_recipe.add(
    targets=(("chem_comp_full_dict", dict),),
    instruction=compare_chem_comp(),
    inputs={
        "kwargs": {
            "chem_comp_dict": ("chem_comp_dict", dict | None),
            "ccd_db_path": ("ccd_db_path", Path | None),
        },
    },
)

cif_recipe.add(
    targets=(("_entity_polymer", dict),),
    instruction=merge_dict(),
    inputs={
        "args": (
            ("_entity_poly_dict", dict | None),
            ("_entity_poly_seq_dict", dict | None),
        ),
    },
)
cif_recipe.add(
    targets=(("_entity_branched", dict | None),),
    instruction=merge_dict(),
    inputs={
        "args": (
            ("_entity_branch_descriptor_dict", dict | None),
            ("_entity_branch_link_dict", dict | None),
            ("_entity_branch_list_dict", dict | None),
        ),
    },
)


cif_recipe.add(
    targets=[
        (("entity_polymer", dict | None),),
        (("entity_nonpolymer", dict | None),),
        (("entity_branched", dict | None),),
    ],
    instruction=parse_entity_dict(),
    inputs=[
        {
            "kwargs": {"entity_dict": ("_entity_polymer", dict | None)},
        },
        {
            "kwargs": {"entity_dict": ("_entity_nonpoly_dict", dict | None)},
        },
        {
            "kwargs": {"entity_dict": ("_entity_branched", dict | None)},
        },
    ],
)

cif_recipe.add(
    targets=(("entity_dict", dict),),
    instruction=merge_dict(),
    inputs={
        "args": (
            ("_entity_dict", dict),
            ("entity_polymer", dict | None),
            ("entity_nonpolymer", dict | None),
            ("entity_branched", dict | None),
        ),
    },
)


cif_recipe.add(
    targets=[
        (("_poly_seq_scheme", dict | None),),
        (("_nonpoly_scheme", dict | None),),
        (("_branch_scheme", dict | None),),
    ],
    instruction=parse_scheme_dict(),
    inputs=[
        {
            "kwargs": {"asym_scheme_dict": ("_pdbx_poly_seq_scheme_dict", dict | None)},
        },
        {
            "kwargs": {"asym_scheme_dict": ("_pdbx_nonpoly_scheme_dict", dict | None)},
        },
        {
            "kwargs": {"asym_scheme_dict": ("_pdbx_branch_scheme_dict", dict | None)},
        },
    ],
)


cif_recipe.add(
    targets=(("_asym_scheme_dict", dict),),
    instruction=merge_dict(),
    inputs={
        "args": (
            ("_poly_seq_scheme", dict | None),
            ("_nonpoly_scheme", dict | None),
            ("_branch_scheme", dict | None),
        ),
    },
)


cif_recipe.add(
    targets=(("_atom_site_wo_unknown_dict", dict | None),),
    instruction=remove_unknown_atom_site(),
    inputs={
        "args": (("_atom_site_dict", dict | None),),
    },
)

cif_recipe.add(
    targets=(("_struct_conn_wo_unknown_dict", dict | None),),
    instruction=remove_unknown_from_struct_conn(),
    inputs={
        "args": (("_struct_conn_dict", dict | None),),
    },
)

cif_recipe.add(
    targets=(("model_alt_atom_site_dict", dict | None),),
    instruction=rearrange_atom_site_dict(),
    inputs={
        "kwargs": {
            "atom_site_dict": ("_atom_site_wo_unknown_dict", dict | None),
        },
        "params": {
            "remove_hydrogen": True,
        },
    },
)

# TODOs
# 3. attach chem comp to atom site dict
# 4. build biomol


cif_recipe.add(
    targets=(("_asym_scheme_entity_dict", dict | None),),
    instruction=attach_entity(),
    inputs={
        "kwargs": {
            "asym_dict": ("_asym_scheme_dict", dict | None),
            "entity_dict": ("entity_dict", dict | None),
        },
    },
)


cif_recipe.add(
    targets=(("_asym_dict", dict),),
    instruction=merge_dict(),
    inputs={
        "args": (
            ("_asym_scheme_entity_dict", dict | None),
            ("model_alt_atom_site_dict", dict | None),
        ),
    },
)


cif_recipe.add(
    targets=(("asym_dict", dict),),
    instruction=build_full_length_asym_dict(),
    inputs={
        "kwargs": {
            "asym_dict": ("_asym_dict", dict | None),
            "chem_comp_dict": ("chem_comp_full_dict", dict | None),
        },
    },
)


cif_recipe.add(
    targets=(("struct_assembly_dict", dict | None),),
    instruction=parse_assembly_dict(),
    inputs={
        "kwargs": {
            "struct_assembly_gen_dict": ("_struct_assembly_gen_dict", dict | None),
        },
    },
)

cif_recipe.add(
    targets=(("struct_oper_dict", dict | None),),
    instruction=get_struct_oper(),
    inputs={
        "kwargs": {
            "struct_oper_dict": ("_struct_oper_dict", dict),
        },
    },
)

cif_recipe.add(
    targets=(("assembly_dict", dict | None),),
    instruction=build_assembly_dict(),
    inputs={
        "kwargs": {
            "asym_dict": ("asym_dict", dict),
            "struct_assembly_dict": ("struct_assembly_dict", dict | None),
            "struct_oper_dict": ("struct_oper_dict", dict),
            "struct_conn_dict": ("_struct_conn_wo_unknown_dict", dict | None),
        },
    },
)


RECIPE = cif_recipe
TARGETS = ["assembly_dict", "metadata_dict"]
