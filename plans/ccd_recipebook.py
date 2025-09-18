from collections.abc import Sequence, Callable
from typing import Any

import numpy as np

from biomol.io.std_instructions.mapping_instr import name2index
from biomol.io.cache import ParsingCache
from biomol.io.cooker import Cooker
from biomol.io.recipe import RecipeBook


parse_cache = ParsingCache()
recipe_book = RecipeBook()

rename = lambda x: x  # identity function
# sturcture metadatas
recipe_book.add(target={"id": str}, instruction=rename, x="_chem_comp.id")
recipe_book.add(target={"name": str}, instruction=rename, x="_chem_comp.name")
recipe_book.add(target={"formula": str}, instruction=rename, x="_chem_comp.formula")

# atom symbol
recipe_book.add(
    target={"atom_symbol": list[str]},
    instruction=rename,
    x="_chem_comp_atom.type_symbol",
)
recipe_book.add(
    target={"atom_id": list[str]},
    instruction=rename,
    x="_chem_comp_atom.atom_id",
)


# ideal coordinates
### replacing mask token to 0.0, make each mask
find_mask = lambda x: [x_i == "?" for x_i in x]
replace_mask = lambda x: [0.0 if x_i == "?" else float(x_i) for x_i in x]
recipe_book.add(
    target={"temp_x": list[float]},
    instruction=replace_mask,
    x="_chem_comp_atom.pdbx_model_Cartn_x_ideal",
)
recipe_book.add(
    target={"temp_y": list[float]},
    instruction=replace_mask,
    x="_chem_comp_atom.pdbx_model_Cartn_y_ideal",
)
recipe_book.add(
    target={"temp_z": list[float]},
    instruction=replace_mask,
    x="_chem_comp_atom.pdbx_model_Cartn_z_ideal",
)
recipe_book.add(
    target={"mask_x": list},
    instruction=find_mask,
    x="_chem_comp_atom.pdbx_model_Cartn_x_ideal",
)
recipe_book.add(
    target={"mask_y": list},
    instruction=find_mask,
    x="_chem_comp_atom.pdbx_model_Cartn_y_ideal",
)
recipe_book.add(
    target={"mask_z": list},
    instruction=find_mask,
    x="_chem_comp_atom.pdbx_model_Cartn_z_ideal",
)


### stacking x,y,z
def stack_coords(x: list[float], y: list[float], z: list[float]) -> list:
    return np.stack([x, y, z], axis=-1).tolist()


# recipe_book.add(
#    target={"ideal_coords": np.ndarray},
#    instruction=stack_coords,
#    x="temp_x",
#    y="temp_y",
#    z="temp_z",
# )

### and operations on masks
# recipe_book.add(
#    target={"ideal_coords_mask": np.ndarray},
#    instruction=lambda mx, my, mz: [bool(a and b and c) for a, b, c in zip(mx, my, mz)],
#    mx="mask_x",
#    my="mask_y",
#    mz="mask_z",
# )

# bond feature
###node id to node index mapping
recipe_book.add(
    target={"node_id2idx": dict[str, int]},
    instruction=lambda x: {x_i: i for i, x_i in enumerate(x)},
    x="atom_id",
)


###source node id, target node id
def id2idx(nodes, mapping):
    return [mapping[n] for n in nodes]


recipe_book.add(
    target={"source_id": list[int]},
    instruction=id2idx,
    nodes="_chem_comp_bond.atom_id_1",
    mapping="node_id2idx",
)
recipe_book.add(
    target={"target_id": list[int]},
    instruction=id2idx,
    nodes="_chem_comp_bond.atom_id_2",
    mapping="node_id2idx",
)
###each bond feature
recipe_book.add(
    target={"bond_order": list[int]},
    instruction=rename,
    x="_chem_comp_bond.value_order",
)

RECIPE = recipe_book

# TODO: more complex type value, support multipleoutput
