import numpy as np

from biomol.core.feature import EdgeFeature, NodeFeature
from biomol.io.cache import ParsingCache
from biomol.io.instructions.ccd_instructions import (
    bond_instruction,
    identity_instruction,
    stack_instruction,
)
from biomol.io.recipe import RecipeBook

"""Build a CCD-specific Cooker.

This factory function constructs and returns a Cooker preconfigured
with CCD parsing recipes and instructions.
"""
parse_cache = ParsingCache()
ccd_recipe = RecipeBook()

ccd_recipe.add(
    targets=[
        (("id", NodeFeature),),
        (("name", NodeFeature),),
        (("formula", NodeFeature),),
    ],
    instruction=identity_instruction(dtype=str),
    inputs=[
        {"args": (("_chem_comp.id", str),), "params": {"description": "chem comp id"}},
        {
            "args": (("_chem_comp.name", str),),
            "params": {"description": "chem comp name"},
        },
        {
            "args": (("_chem_comp.formula", str),),
            "params": {"description": "chem comp formula"},
        },
    ],
)

ccd_recipe.add(
    targets=[
        (("atom_id", NodeFeature), ("atom_id_mask", NodeFeature)),
        (("type", NodeFeature), ("type_mask", NodeFeature)),
        (("chiral", NodeFeature),),
        (("charge", NodeFeature),),
    ],
    instruction=identity_instruction(dtype=str),
    inputs=[
        {
            "args": (("_chem_comp_atom.atom_id", str),),
            "params": {"description": "atom id", "on_missing": {"?": "X"}},
        },
        {
            "args": (("_chem_comp_atom.type_symbol", str),),
            "params": {"description": "atom symbol", "on_missing": {"?": "X"}},
        },
        {
            "args": (("_chem_comp_atom.pdbx_stereo_config", str),),
            "params": {
                "description": "atom symbol",
            },
        },
        {
            "args": (("_chem_comp_atom.charge", int),),
            "params": {
                "description": "atom symbol",
            },
        },
    ],
)

bond_kwargs = {
    "src": ("_chem_comp_bond.atom_id_1", str),
    "dst": ("_chem_comp_bond.atom_id_2", str),
    "atom_id": ("atom_id", str),
}
ccd_recipe.add(
    targets=[
        (("bond_order", EdgeFeature),),
        (("bond_aromacity", EdgeFeature),),
        (("bond_stereo", EdgeFeature),),
    ],
    instruction=bond_instruction(dtype=str),
    inputs=[
        {
            "kwargs": bond_kwargs,
            "args": (("_chem_comp_bond.value_order", str),),
            "params": {"description": "bond_type"},
        },
        {
            "kwargs": bond_kwargs,
            "args": (("_chem_comp_bond.pdbx_aromatic_flag", str),),
            "params": {"description": "bond_aromacity"},
        },
        {
            "kwargs": bond_kwargs,
            "args": (("_chem_comp_bond.pdbx_stereo_config", str),),
            "params": {"description": "bond_stereo"},
        },
    ],
)

ccd_recipe.add(
    targets=(("ideal_coords", NodeFeature), ("ideal_coords_mask", NodeFeature)),
    instruction=stack_instruction(dtype=float),
    inputs={
        "args": (
            ("_chem_comp_atom.pdbx_model_Cartn_x_ideal", float),
            ("_chem_comp_atom.pdbx_model_Cartn_y_ideal", float),
            ("_chem_comp_atom.pdbx_model_Cartn_z_ideal", float),
        ),
        "params": {
            "description": "Ideal_coords recorded in CCD. Missing values are set to nan",
            "on_missing": {"?": np.nan},
        },
    },
)

RECIPE = ccd_recipe
