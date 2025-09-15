from typing import TypedDict

import numpy as np


class NodeFeatureDict(TypedDict):
    """TypedDict for node features."""

    value: np.ndarray
    description: str | None


class EdgeFeatureDict(TypedDict):
    """TypedDict for edge features."""

    src_indices: np.ndarray
    dst_indices: np.ndarray
    value: np.ndarray
    description: str | None


class FeatureContainerDict(TypedDict):
    """TypedDict for feature container."""

    nodes: dict[str, NodeFeatureDict]
    edges: dict[str, EdgeFeatureDict]


class IndexTableDict(TypedDict):
    """TypedDict for index table."""

    atom_to_res: np.ndarray
    res_to_chain: np.ndarray
    res_atom_indptr: np.ndarray
    res_atom_indices: np.ndarray
    chain_res_indptr: np.ndarray
    chain_res_indices: np.ndarray


class BioMolDict(TypedDict):
    """TypedDict for BioMol object."""

    atoms: FeatureContainerDict
    residues: FeatureContainerDict
    chains: FeatureContainerDict
    index_table: IndexTableDict
