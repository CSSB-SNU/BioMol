from biomol.cif.mol import CIFMol
from biomol.io.recipe import RecipeBook
from postprocessing.instructions.seq_instructions import build_fasta

"""Build a CIFMol->fasta Cooker."""

fasta_recipe = RecipeBook()

fasta_recipe.add(
    targets=[
        ("fasta", str),
    ],
    instruction=build_fasta(),
    inputs={
        "kwargs": {
            "cifmol": ("cifmol", CIFMol | None),
        },
    },
)

RECIPE = fasta_recipe
TARGETS = ["fasta"]
