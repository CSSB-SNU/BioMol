import numpy as np

from typing import Any
from biomol.io.cache import ParsingCache
from biomol.io.recipe import RecipeBook
from biomol.io.instructions.cif_instructions import (
    single_value_instruction,
    get_smaller_dict,
    merge_dict,
)

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


def extract_single(*args: str | None) -> float:
    none_mask = [a is None for a in args]
    none_mask = np.array(none_mask)
    if np.all(none_mask):
        msg = "All inputs are None"
        raise ValueError(msg)
    if np.sum(none_mask) > 1:
        msg = "More than one input is not None"
        raise ValueError(msg)
    valid_idx = np.where(~none_mask)[0].item()
    return args[valid_idx][0]


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
    instruction=extract_single,
    inputs={
        "args": (
            ("_refine.ls_d_res_high", str | None),
            ("_em_3d_reconstruction.resolution", str | None),
        ),
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
                    "pdb_seq_numpdb_ins_code",
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
                    "label_entity_id",
                    "auth_asym_id",
                ],
            },
        },
    ],
)

# remove unl from atom site dict

cif_recipe.add(
    targets=(("chem_comp_dict", dict),),
    instruction=merge_dict(),
    inputs={
        "args": (
            ("_chem_comp_dict", dict),
            ("_chem_comp_atom_dict", dict),
            ("_chem_comp_bond_dict", dict),
        ),
    },
)

cif_recipe.add(
    targets=(("asym_dict", dict),),
    instruction=merge_dict(),
    inputs={
        "args": (
            ("_pdbx_poly_seq_scheme_dict", dict | None),
            ("_pdbx_nonpoly_scheme_dict", dict | None),
            ("_pdbx_branch_scheme_dict", dict | None),
            ("_atom_site_dict", dict | None),
        ),
    },
)

RECIPE = cif_recipe
