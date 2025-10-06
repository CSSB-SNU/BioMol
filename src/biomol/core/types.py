from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray


class NodeFeatureDict(TypedDict):
    """TypedDict for node features."""

    value: NDArray[np.generic]
    description: str | None


class EdgeFeatureDict(TypedDict):
    """TypedDict for edge features."""

    src_indices: NDArray[np.integer]
    dst_indices: NDArray[np.integer]
    value: NDArray[np.generic]
    description: str | None


class FeatureContainerDict(TypedDict):
    """TypedDict for feature container."""

    nodes: dict[str, NodeFeatureDict]
    edges: dict[str, EdgeFeatureDict]


class IndexTableDict(TypedDict):
    """TypedDict for index table."""

    atom_to_res: NDArray[np.integer]
    res_to_chain: NDArray[np.integer]
    res_atom_indptr: NDArray[np.integer]
    res_atom_indices: NDArray[np.integer]
    chain_res_indptr: NDArray[np.integer]
    chain_res_indices: NDArray[np.integer]


class BioMolDict(TypedDict):
    """TypedDict for BioMol object."""

    atoms: FeatureContainerDict
    residues: FeatureContainerDict
    chains: FeatureContainerDict
    index_table: IndexTableDict
    metadata: dict[str, Any]
