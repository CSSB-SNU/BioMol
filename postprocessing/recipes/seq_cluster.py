from pathlib import Path

from biomol.io.recipe import RecipeBook
from postprocessing.instructions.seq_cluster_instructions import (
    separate_sequences,
    protein_cluster,
    antibody_cluster,
    write_cluster,
)

"""Build a sequence clustering Cooker."""

seq_cluster_recipe = RecipeBook()

seq_cluster_recipe.add(
    targets=[
        ("fasta_path_dict", dict),
    ],
    instruction=separate_sequences(),
    inputs={
        "kwargs": {
            "tmp_dir": ("tmp_dir", Path),
            "seq_hash_map": ("seq_hash_map", Path),
        },
    },
)

seq_cluster_recipe.add(
    targets=[
        ("protein_cluster_dict", dict),
        ("protein_d_cluster_dict", dict),
    ],
    instruction=protein_cluster(),
    inputs={
        "kwargs": {
            "tmp_dir": ("tmp_dir", Path),
            "fasta_path_dict": ("fasta_path_dict", Path),
        },
    },
)

seq_cluster_recipe.add(
    targets=[
        ("antibody_cluster_dict", dict),
    ],
    instruction=antibody_cluster(),
    inputs={
        "kwargs": {
            "tmp_dir": ("tmp_dir", Path),
            "fasta_path_dict": ("fasta_path_dict", Path),
        },
    },
)

seq_cluster_recipe.add(
    targets=[
        ("cluster_dict", dict),
    ],
    instruction=write_cluster(),
    inputs={
        "kwargs": {
            "fasta_path_dict": ("fasta_path_dict", dict),
            "protein_cluster_dict": ("protein_cluster_dict", dict),
            "protein_d_cluster_dict": ("protein_d_cluster_dict", dict),
            "antibody_cluster_dict": ("antibody_cluster_dict", dict),
        },
    },
)

RECIPE = seq_cluster_recipe
TARGETS = ["cluster_dict"]
