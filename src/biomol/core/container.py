from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

import numpy as np

from biomol.exceptions import FeatureKeyError, IndexMismatchError, IndexOutOfBoundsError

from .feature import EdgeFeature, Feature, NodeFeature

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray

    from .types import FeatureContainerDict


class FeatureContainer:
    """Container for holding either node or edge features.

    Parameters
    ----------
    features: Mapping[str, Feature]
        A mapping of feature keys to Feature objects. Features can be either NodeFeature
        or EdgeFeature.

    Notes
    -----
    features must contain at least one NodeFeature. All NodeFeatures must have the same
    length.
    """

    def __init__(self, features: Mapping[str, Feature]) -> None:
        self._features = dict(features)

        self._check_node_lengths()
        self._check_edge_indices()

    def __len__(self) -> int:
        """Return the number of nodes in the container."""
        node_lengths = {
            len(f.value) for f in self._features.values() if isinstance(f, NodeFeature)
        }
        return node_lengths.pop()

    def __getitem__(self, key: str) -> Feature:
        """Get a feature by its key."""
        if key in self._features:
            return self._features[key]
        raise FeatureKeyError(key)

    def __repr__(self) -> str:
        """Return a string representation of the container."""
        return f"FeatureContainer(keys={list(self._features.keys())})"

    def keys(self) -> list[str]:
        """List of all features keys in the container."""
        return list(self._features.keys())

    def crop(self, indices: NDArray[np.integer]) -> FeatureContainer:
        """Crop all features to only include the specified indices.

        Parameters
        ----------
        indices: NDArray[np.integer]
            1D array of global node indices to keep. Only integer arrays is allowed.
        """
        return FeatureContainer(
            {key: feat.crop(indices) for key, feat in self._features.items()},
        )

    def to_dict(self) -> FeatureContainerDict:
        """Convert the container to a dictionary."""
        nodes = {
            key: asdict(values)
            for key, values in self._features.items()
            if isinstance(values, NodeFeature)
        }
        edges = {
            key: asdict(values)
            for key, values in self._features.items()
            if isinstance(values, EdgeFeature)
        }
        return {"nodes": nodes, "edges": edges}  # pyright: ignore[reportReturnType]

    @classmethod
    def from_dict(cls, data: FeatureContainerDict) -> FeatureContainer:
        """Create a FeatureContainer from a dictionary.

        Parameters
        ----------
        data : FeatureContainerDict
            Dictionary containing node and edge features.
        """
        nodes = {
            key: NodeFeature(**values) for key, values in data.get("nodes", {}).items()
        }
        edges = {
            key: EdgeFeature(**values) for key, values in data.get("edges", {}).items()
        }
        if nodes.keys() & edges.keys():
            overlap_keys = nodes.keys() & edges.keys()
            msg = f"Feature keys cannot be both node and edge features: {overlap_keys}"
            raise FeatureKeyError(msg)
        return FeatureContainer(features={**nodes, **edges})

    def _check_node_lengths(self) -> None:
        node_lengths = {
            len(f.value) for f in self._features.values() if isinstance(f, NodeFeature)
        }
        if not node_lengths:
            msg = "FeatureContainer must contain at least one node feature."
            raise FeatureKeyError(msg)
        if len(node_lengths) > 1:
            msg = f"Inconsistent node feature lengths {node_lengths}"
            raise IndexMismatchError(msg)

    def _check_edge_indices(self) -> None:
        length = len(self)
        for key, feat in self._features.items():
            if not isinstance(feat, EdgeFeature):
                continue
            if np.any(feat.src_indices >= length) or np.any(feat.dst_indices >= length):
                msg = (
                    f"Pair feature '{key}' has out-of-bounds indices. "
                    f"Max index allowed is {length - 1}, "
                    f"but got src_indices max={feat.src_indices.max()} and "
                    f"dst_indices max={feat.dst_indices.max()}."
                )
                raise IndexOutOfBoundsError(msg)
