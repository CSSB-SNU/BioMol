from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any, cast

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

    def update(self, **features: Feature | NDArray[Any]) -> FeatureContainer:
        """Update the container with new or modified features.

        Parameters
        ----------
        **features: Feature | NDArray[Any]
            Key-value pairs of features to add or update. Values can be either Feature
            objects or numpy arrays (which will be converted to NodeFeature).

        Notes
        -----
        This method returns a new FeatureContainer instance and does not modify the
        current instance.

        Examples
        --------
        .. code-block:: python

            container = FeatureContainer(...)
            new_container = container.update(coord=container["coord"] + 1.0)

        """
        _features = dict(self._features)
        _features.update(
            {
                key: value if isinstance(value, Feature) else NodeFeature(value)
                for key, value in features.items()
            },
        )
        return FeatureContainer(_features)

    def remove(self, *keys: str) -> FeatureContainer:
        """Remove features by their keys.

        Parameters
        ----------
        *keys: str
            Keys of the features to remove.

        Notes
        -----
        This method returns a new FeatureContainer instance and does not modify the
        current instance.

        Examples
        --------
        .. code-block:: python

            container = FeatureContainer(...)
            new_container = container.remove("coord", "element")

        """
        _features = dict(self._features)
        for key in keys:
            if key not in _features:
                raise FeatureKeyError(key)
            del _features[key]
        return FeatureContainer(_features)

    @classmethod
    def concat(cls, containers: list[FeatureContainer]) -> FeatureContainer:
        """Concatenate multiple FeatureContainer instances.

        Parameters
        ----------
        containers: list[FeatureContainer]
            List of FeatureContainer instances to concatenate.

        Returns
        -------
        FeatureContainer
            Concatenated FeatureContainer.

        Notes
        -----
        All containers must have the same set of feature keys.

        Examples
        --------
        .. code-block:: python

            container1 = FeatureContainer(...)
            container2 = FeatureContainer(...)
            concatenated = FeatureContainer.concat([container1, container2])

        """
        if not containers:
            msg = "No FeatureContainer instances provided for concatenation."
            raise ValueError(msg)

        base_keys = containers[0].keys()
        for container in containers[1:]:
            if set(container.keys()) != set(base_keys):
                msg = (
                    "All containers must have the same feature keys. "
                    f"Missing keys: {set(base_keys) - set(container.keys())}. "
                    f"Extra keys: {set(container.keys()) - set(base_keys)}."
                )
                raise FeatureKeyError(msg)

        new_features: dict[str, Feature] = {}
        for key in base_keys:
            features = [container[key] for container in containers]
            if all(isinstance(feat, NodeFeature) for feat in features):
                new_features[key] = NodeFeature(
                    np.concatenate([feature.value for feature in features], axis=0),
                )
            elif all(isinstance(feat, EdgeFeature) for feat in features):
                features = cast("list[EdgeFeature]", features)
                counts = [len(c) for c in containers]
                offsets = np.cumsum([0, *counts[:-1]])
                all_src = [
                    feature.src_indices + offset
                    for feature, offset in zip(features, offsets, strict=True)
                ]
                all_dst = [
                    feature.dst_indices + offset
                    for feature, offset in zip(features, offsets, strict=True)
                ]
                new_features[key] = EdgeFeature(
                    np.concatenate([feature.value for feature in features], axis=0),
                    src_indices=np.concatenate(all_src, axis=0),
                    dst_indices=np.concatenate(all_dst, axis=0),
                )
            else:
                msg = (
                    f"Feature '{key}' has mixed types across containers: "
                    f"{ {type(f) for f in features} }"
                )
                raise FeatureKeyError(msg)

        return FeatureContainer(new_features)

    def copy(self) -> FeatureContainer:
        """Create a deep copy of the FeatureContainer."""
        return FeatureContainer(
            {key: feat.copy() for key, feat in self._features.items()},
        )

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
