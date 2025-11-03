from biomol.cif.mol import CIFMol
from biomol.io.recipe import RecipeBook
from postprocessing.instructions.seq_instructions import build_seq_hash_map

"""Build a CIFMol->fasta Cooker."""

hash_map_recipe = RecipeBook()

hash_map_recipe.add(
    targets=[
        ("seq_hash_map", dict),
    ],
    instruction=build_seq_hash_map(),
    inputs={
        "kwargs": {
            "fasta_dict": ("fasta_dict", dict | None),
        },
    },
)

RECIPE = hash_map_recipe
TARGETS = ["seq_hash_map"]
