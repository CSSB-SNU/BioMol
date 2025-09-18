import numpy as np

from biomol.core.feature import EdgeFeature, NodeFeature
from biomol.io.cache import ParsingCache
from biomol.io.instructions.ccd_instructions import (
    bond_instruction,
    identity_instruction,
    stack_instruction,
)
from biomol.io.recipe import Constant, RecipeBook

"""Build a CCD-specific Cooker.

This factory function constructs and returns a Cooker preconfigured
with CCD parsing recipes and instructions.
"""
parse_cache = ParsingCache()
ccd_recipe = RecipeBook()

ccd_recipe.add(
    target=[{"id": NodeFeature}, {"name": NodeFeature}, {"formula": NodeFeature}],
    instruction=identity_instruction,
    data=["_chem_comp.id", "_chem_comp.name", "_chem_comp.formula"],
    dtype=[str, str, str],
    description=[
        Constant("chem comp id."),
        Constant("chem comp name."),
        Constant("chem comp formula."),
    ],
    group=True,
)

ccd_recipe.add(
    target=[
        {"atom_id": NodeFeature, "atom_id_mask": NodeFeature},
        {"atom_symbol": NodeFeature, "atom_symbol_mask": NodeFeature},
    ],
    instruction=identity_instruction,
    data=["_chem_comp_atom.atom_id", "_chem_comp_atom.type_symbol"],
    dtype=[str, str],
    on_missing=[Constant({"?": "X"}), Constant({"?": "X"})],
    description=[Constant("atom id"), Constant("atom symbol")],
    group=True,
)

ccd_recipe.add(
    target={"bonds": EdgeFeature},
    instruction=bond_instruction,
    value1="_chem_comp_bond.value_order",
    src="_chem_comp_bond.atom_id_1",
    dst="_chem_comp_bond.atom_id_2",
    atom_id="atom_id",
    dtype=str,
    description=Constant("bond_type"),
)

ccd_recipe.add(
    target={"ideal_coords": NodeFeature, "ideal_coords_mask": NodeFeature},
    instruction=stack_instruction,
    value_1="_chem_comp_atom.pdbx_model_Cartn_x_ideal",
    value_2="_chem_comp_atom.pdbx_model_Cartn_y_ideal",
    value_3="_chem_comp_atom.pdbx_model_Cartn_z_ideal",
    dtype=float,
    on_missing=Constant({"?": np.nan}),
    description=Constant(
        "Ideal_coords recorded in CCD. Missing values are set to nan"
    ),
)

RECIPE = ccd_recipe
