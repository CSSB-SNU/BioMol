from biomol.cif.mol import CIFMol
from biomol.io.recipe import RecipeBook
from postprocessing.instructions.graph_cluster_instructions import (
    extract_graph_per_cifmol,
)

"""Build a CIFMol->fasta Cooker."""

gc_recipe = RecipeBook()

gc_recipe.add(
    targets=[
        ("graph_bytes", dict[str, bytes]),
    ],
    instruction=extract_graph_per_cifmol(),
    inputs={
        "kwargs": {
            "cifmol": ("cifmol", list[CIFMol] | None),
            "seq_to_seq_hash_list": ("seq_to_seq_hash_list", dict | None),
            "seq_hash_to_cluster": ("seq_hash_to_cluster", dict | None),
        },
    },
)


RECIPE = gc_recipe
TARGETS = ["graph_bytes"]
