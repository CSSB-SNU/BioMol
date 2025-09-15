from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from .enums import StructureLevel
from .exceptions import FeatureKeyError, IndexMismatchError, IndexOutOfBoundsError
from .feature import EdgeFeature, Feature, NodeFeature
from .types import FeatureContainerDict


@dataclass(frozen=True, slots=True)
class FeatureContainer:
    """Container for holding either node or edge features."""

    node_features: Mapping[str, NodeFeature]
    edge_features: Mapping[str, EdgeFeature]

    level: ClassVar[StructureLevel]

    def __post_init__(self) -> None:  # noqa: D105
        self._check_node_lengths()
        self._check_edge_indices()
        self._check_duplicate_keys()

    def __len__(self) -> int:
        """Return the number of nodes in the container."""
        return len(next(iter(self.node_features.values())).value)

    def __getattr__(self, key: str) -> Feature:
        """Get a feature by its key."""
        return self.__getitem__(key)

    def __getitem__(self, key: str) -> Feature:
        """Get a feature by its key."""
        if key in self.node_features:
            return self.node_features[key]
        if key in self.edge_features:
            return self.edge_features[key]
        raise FeatureKeyError(key)

    @property
    def keys(self) -> list[str]:
        """List of all features keys in the container."""
        return list(self.node_features.keys()) + list(self.edge_features.keys())

    def crop(self, indices: NDArray[np.integer]) -> Self:
        """Crop all features to only include the specified indices.

        Parameters
        ----------
        indices: NDArray[np.integer]
            1D array of global node indices to keep. Only integer arrays is allowed.
        """
        node_features = {
            key: feat.crop(indices) for key, feat in self.node_features.items()
        }
        edge_features = {
            key: feat.crop(indices) for key, feat in self.edge_features.items()
        }
        return replace(self, node_features=node_features, edge_features=edge_features)

    def to_dict(self) -> FeatureContainerDict:
        """Convert the container to a dictionary."""
        return {
            "nodes": {k: asdict(v) for k, v in self.node_features.items()},
            "edges": {k: asdict(v) for k, v in self.edge_features.items()},
        }

    @classmethod
    def from_dict(cls, data: FeatureContainerDict) -> Self:
        """Create a FeatureContainer from a dictionary.

        Parameters
        ----------
        data : FeatureContainerDict
            Dictionary containing node and edge features.
        """
        node_features = {
            key: NodeFeature(**values) for key, values in data.get("nodes", {}).items()
        }
        edge_features = {
            key: EdgeFeature(**values) for key, values in data.get("edges", {}).items()
        }
        return cls(node_features=node_features, edge_features=edge_features)

    def _check_node_lengths(self) -> None:
        if not self.node_features:
            msg = "No node features defined."
            raise FeatureKeyError(msg)
        node_lengths = {len(f.value) for f in self.node_features.values()}
        if len(node_lengths) > 1:
            msg = f"Inconsistent node feature lengths {node_lengths}"
            raise IndexMismatchError(msg)

    def _check_edge_indices(self) -> None:
        length = len(next(iter(self.node_features.values())).value)
        for key, feat in self.edge_features.items():
            if np.any(feat.src_indices >= length) or np.any(feat.dst_indices >= length):
                msg = (
                    f"Pair feature '{key}' has out-of-bounds indices. "
                    f"Max index allowed is {length - 1}, "
                    f"but got src_indices max={feat.src_indices.max()} and "
                    f"dst_indices max={feat.dst_indices.max()}."
                )
                raise IndexOutOfBoundsError(msg)

    def _check_duplicate_keys(self) -> None:
        if len(self.keys) != len(set(self.keys)):
            duplicate_keys = {key for key in self.keys if self.keys.count(key) > 1}
            msg = f"Duplicate feature keys found in features: {duplicate_keys}"
            raise FeatureKeyError(msg)


@dataclass(frozen=True, slots=True)
class AtomContainer(FeatureContainer):
    """Container for atom-level features."""

    level: ClassVar[StructureLevel] = StructureLevel.ATOM


@dataclass(frozen=True, slots=True)
class ResidueContainer(FeatureContainer):
    """Container for residue-level features."""

    level: ClassVar[StructureLevel] = StructureLevel.RESIDUE


@dataclass(frozen=True, slots=True)
class ChainContainer(FeatureContainer):
    """Container for chain-level features."""

    level: ClassVar[StructureLevel] = StructureLevel.CHAIN
