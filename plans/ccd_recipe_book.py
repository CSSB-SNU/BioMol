from biomol.io.instructions.cif_instructions import (
    get_smaller_dict,
    parse_chem_comp,
)
from biomol.io.recipe import RecipeBook

"""Build a CIF-specific Cooker.

This factory function constructs and returns a Cooker preconfigured
with CIF parsing recipes and instructions.
"""

ccd_recipe = RecipeBook()


ccd_recipe.add(
    targets=[
        (("_chem_comp_dict", dict),),
        (("_chem_comp_atom_dict", dict),),
        (("_chem_comp_bond_dict", dict),),
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
                    "model_Cartn_x",
                    "model_Cartn_y",
                    "model_Cartn_z",
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
    ],
)


ccd_recipe.add(
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
            "unwrap": True,
        },
    },
)


RECIPE = ccd_recipe
TARGETS = "chem_comp_dict"
