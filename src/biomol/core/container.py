from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import ClassVar

import numpy as np

from .exceptions import FeatureIndicesError, FeatureKeyError, FeatureShapeError
from .feature import EdgeFeature, Feature, NodeFeature
from .types import StructureLevel


@dataclass(frozen=True, slots=True)
class FeatureContainer:
    """Container for holding either node or pair features."""

    node_features: Mapping[str, NodeFeature]
    pair_features: Mapping[str, EdgeFeature]

    level: ClassVar[StructureLevel]

    def __post_init__(self) -> None:  # noqa: D105
        self._check_node_lengths()
        self._check_pair_indices()
        self._check_duplicate_keys()

    def _check_node_lengths(self) -> None:
        if not self.node_features:
            msg = "No node features defined."
            raise FeatureKeyError(msg)
        node_lengths = {len(f.value) for f in self.node_features.values()}
        if len(node_lengths) > 1:
            msg = f"Inconsistent node feature lengths {node_lengths}"
            raise FeatureShapeError(msg)

    def _check_pair_indices(self) -> None:
        length = len(next(iter(self.node_features.values())).value)
        for key, feat in self.pair_features.items():
            if np.any(feat.src_indices >= length) or np.any(feat.dst_indices >= length):
                msg = (
                    f"Pair feature '{key}' has out-of-bounds indices. "
                    f"Max index allowed is {length - 1}, "
                    f"but got src_indices max={feat.src_indices.max()} and "
                    f"dst_indices max={feat.dst_indices.max()}."
                )
                raise FeatureIndicesError(msg)

    def _check_duplicate_keys(self) -> None:
        if len(self.keys) != len(set(self.keys)):
            duplicate_keys = {key for key in self.keys if self.keys.count(key) > 1}
            msg = f"Duplicate feature keys found in features: {duplicate_keys}"
            raise FeatureKeyError(msg)

    @property
    def keys(self) -> list[str]:
        """List of all feature keys in the container."""
        return list(self.node_features.keys()) + list(self.pair_features.keys())

    def __getattr__(self, key: str) -> Feature:
        """Get a feature by its key."""
        return self.__getitem__(key)

    def __getitem__(self, key: str) -> Feature:
        """Get a feature by its key."""
        if key in self.node_features:
            return self.node_features[key]
        if key in self.pair_features:
            return self.pair_features[key]
        raise FeatureKeyError(key)

    def crop(self, indices: np.ndarray) -> "FeatureContainer":
        """Crop all features to only include the specified indices.

        Parameters
        ----------
        indices : np.ndarray
            Indices to keep. Supported formats:
            - 1D integer array of indices (must be unique)
            - 1D boolean mask (must have same length as node features)

        Returns
        -------
        FeatureContainer
            A new FeatureContainer containing only the specified indices.
        """
        node_features = {
            key: feat.crop(indices) for key, feat in self.node_features.items()
        }
        pair_features = {
            key: feat.crop(indices, remapping=True)
            for key, feat in self.pair_features.items()
        }
        return replace(self, node_features=node_features, pair_features=pair_features)


@dataclass(frozen=True, slots=True)
class AtomContainer(FeatureContainer):
    """Container for atom-level features."""

    node_features: Mapping[str, NodeFeature]
    pair_features: Mapping[str, EdgeFeature]

    level: ClassVar[StructureLevel] = StructureLevel.ATOM


class ResidueContainer(FeatureContainer):
    """Container for residue-level features."""

    node_features: Mapping[str, NodeFeature]
    pair_features: Mapping[str, EdgeFeature]

    level: ClassVar[StructureLevel] = StructureLevel.RESIDUE


class ChainContainer(FeatureContainer):
    """Container for chain-level features."""

    node_features: Mapping[str, NodeFeature]
    pair_features: Mapping[str, EdgeFeature]

    level: ClassVar[StructureLevel] = StructureLevel.CHAIN
