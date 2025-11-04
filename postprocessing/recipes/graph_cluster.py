from pathlib import Path

import networkx as nx

from biomol.cif.mol import CIFMol
from biomol.core.container import FeatureContainer
from biomol.io.recipe import RecipeBook
from postprocessing.instructions.graph_cluster_instructions import (
    build_graph_hash,
    convert_graph_to_bytes,
    extract_graphs,
    graph_edge_cluster,
)

"""Build a CIFMol->fasta Cooker."""

gc_recipe = RecipeBook()


gc_recipe.add(
    targets=[
        ("unique_map", dict[str, nx.Graph]),  # graph_hash -> graph
        ("graph_hash_map", dict[str, int]),  # cif_ID -> graph_hash
    ],
    instruction=build_graph_hash(),
    inputs={
        "kwargs": {
            "graph_map": ("graph_map", dict[str, nx.Graph] | None),
        },
    },
)

gc_recipe.add(
    targets=[
        (
            "graph_clusters",
            dict[int, list[str]],
        ),  # set of graph Ids per cluster Ex [{0, 1, 3}, {2, 5}, {4, 7}, {6}, {8}]
        ("comp_map", dict[str, int]),
    ],
    instruction=graph_edge_cluster(),
    inputs={
        "kwargs": {
            "unique_graphs": ("unique_map", list[nx.Graph] | None),
        },
    },
)

gc_recipe.add(
    targets=[
        ("graph_dict", dict[str, bytes]),
    ],
    instruction=convert_graph_to_bytes(),
    inputs={
        "kwargs": {
            "graph_map": ("unique_map", dict[str, nx.Graph] | None),
        },
    },
)

RECIPE = gc_recipe
TARGETS = ["graph_clusters", "graph_hash_map", "unique_map", "graph_dict"]
