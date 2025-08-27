import dataclasses
from collections.abc import Iterable
from typing import Any

import numpy as np

from .feature import (
    Feature,
    Feature0D,
    Feature1D,
    FeatureMap0D,
    FeatureMap1D,
    FeatureMapPair,
    FeaturePair,
)
from .types import MoleculeType, StructureLevel


@dataclasses.dataclass(slots=True)
class FeatureContainer:
    feature_map_0D: FeatureMap0D
    feature_map_1D: FeatureMap1D
    feature_map_pair: FeatureMapPair
    level: StructureLevel

    def __init__(
        self,
        feature_map_0D: FeatureMap0D | None = None,
        feature_map_1D: FeatureMap1D | None = None,
        feature_map_pair: FeatureMapPair | None = None,
    ):
        self.feature_map_0D = feature_map_0D or FeatureMap0D({})
        self.feature_map_1D = feature_map_1D or FeatureMap1D({})
        self.feature_map_pair = feature_map_pair or FeatureMapPair({})
        self._check_duplicate_keys()
        self._check_length()
        self._check_level()

    def __getitem__(self, idx: str | int | slice | tuple[int, int]) -> Feature:
        if isinstance(idx, str):
            if idx in self.feature_map_0D:
                return self.feature_map_0D[idx]
            if idx in self.feature_map_1D:
                return self.feature_map_1D[idx]
            return self.feature_map_pair[idx]
        if isinstance(idx, int | slice):
            return self.feature_map_1D[idx]
        if isinstance(idx, tuple):
            return self.feature_map_pair[idx]
        raise TypeError(f"Unsupported index type: {type(idx).__name__}")

    def __getattr__(self, name: str) -> str | int | float | bool | np.ndarray:
        """
        Allow accessing features by attribute syntax.

        container.key1  <=> container.key_to_map["key1"].
        """
        if name in self.feature_map_0D:
            return self.feature_map_0D[name].value
        if name in self.feature_map_1D:
            return self.feature_map_1D[name].value
        if name in self.feature_map_pair:
            return self.feature_map_pair[name].value

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def keys(self) -> list[str]:
        return (
            list(self.feature_map_0D.keys())
            + list(self.feature_map_1D.keys())
            + list(self.feature_map_pair.keys())
        )

    def get_0d_features(self) -> FeatureMap0D:
        return self.feature_map_0D

    def get_1d_features(self) -> FeatureMap1D:
        return self.feature_map_1D

    def get_pair_features(self) -> FeatureMapPair:
        return self.feature_map_pair

    def crop(self, indices: np.ndarray) -> "FeatureContainer":
        """
        Crop the feature container to only include the specified indices.

        Parameters
        ----------
        indices : np.ndarray
            1D array of indices to keep.

        Returns
        -------
        FeatureContainer
            A new FeatureContainer containing only the specified indices.
        """
        cropped_1D = self.feature_map_1D.crop(indices)
        cropped_pair = self.feature_map_pair.crop(indices)
        return FeatureContainer(self.feature_map_0D, cropped_1D, cropped_pair)

    def to_numpy(self) -> dict[str, Any]:
        return {
            "0D": {
                k: {
                    "value": v.value,
                    "level": v.level.value,
                    "info": v.additional_info,
                }
                for k, v in self.feature_map_0D.feature_map.items()
            },
            "1D": {
                k: {
                    "value": v.value,
                    "level": v.level.value,
                    "info": v.additional_info,
                }
                for k, v in self.feature_map_1D.feature_map.items()
            },
            "pair": {
                k: {
                    "value": v.value,
                    "level": v.level.value,
                    "info": v.additional_info,
                }
                for k, v in self.feature_map_pair.feature_map.items()
            },
        }

    @classmethod
    def from_numpy(cls, data: dict[str, Any]) -> "FeatureContainer":
        map_0d: dict[str, Feature0D] = {}
        for k, d in (data.get("0D") or {}).items():
            level = d.get("level")
            level_enum = (
                level if isinstance(level, StructureLevel) else StructureLevel(level)
            )
            map_0d[k] = Feature0D(
                value=d.get("value"),
                level=level_enum,
                additional_info=d.get("info"),
            )

        map_1d: dict[str, Feature1D] = {}
        for k, d in (data.get("1D") or {}).items():
            level = d.get("level")
            level_enum = (
                level if isinstance(level, StructureLevel) else StructureLevel(level)
            )
            map_1d[k] = Feature1D(
                value=d.get("value"),
                level=level_enum,
                additional_info=d.get("info"),
            )

        map_pair: dict[str, FeaturePair] = {}
        for k, d in (data.get("pair") or {}).items():
            level = d.get("level")
            level_enum = (
                level if isinstance(level, StructureLevel) else StructureLevel(level)
            )
            fp = FeaturePair(
                value=d.get("value"),
                level=level_enum,
                additional_info=d.get("info"),
            )
            map_pair[k] = fp

        fm0d = FeatureMap0D(map_0d)
        fm1d = FeatureMap1D(map_1d)
        fmpair = FeatureMapPair(map_pair)
        return cls(fm0d, fm1d, fmpair)

    def _check_duplicate_keys(self) -> None:
        keys_0D = self.feature_map_0D.keys()
        keys_1D = self.feature_map_1D.keys()
        keys_pair = self.feature_map_pair.keys()
        total_keys = list(keys_0D) + list(keys_1D) + list(keys_pair)
        if len(set(total_keys)) != len(total_keys):
            raise ValueError("Duplicate keys found in feature maps")

    def _check_length(self) -> None:
        if len(self.feature_map_1D) == 0:
            return
        length = len(self.feature_map_1D.values()[0])
        for pair_feature in self.feature_map_pair.values():
            ii, jj = pair_feature.indices()
            if ii >= length or jj >= length:
                raise ValueError(f"Feature pair indices out of bounds: {ii}, {jj}")

    def _check_level(self):
        levels = {
            "0D": self.feature_map_0D.level,
            "1D": self.feature_map_1D.level,
            "pair": self.feature_map_pair.level,
        }
        if any(level != self.level for level in levels.values()):
            raise ValueError(f"All feature maps must have level {self.level}.")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FeatureContainer):
            return NotImplemented
        return (
            self.level == other.level
            and self.feature_map_0D == other.feature_map_0D
            and self.feature_map_1D == other.feature_map_1D
            and self.feature_map_pair == other.feature_map_pair
        )


@dataclasses.dataclass(slots=True, eq=False)
class AtomContainer(FeatureContainer):
    """
    Container for atom-level features.

    Notes
    -----
    - Represents the lowest-level features (individual atoms).
    - Conceptually corresponds to **Residue** or **ChemComp** in structural biology.
    - All feature maps inside this container must have `level = FeatureLevel.ATOM`.
    """

    feature_map_0D: FeatureMap0D
    feature_map_1D: FeatureMap1D
    feature_map_pair: FeatureMapPair
    level: StructureLevel = StructureLevel.ATOM


@dataclasses.dataclass(slots=True)
class ResidueContainer(FeatureContainer):
    """
    Container for residue-level features.

    Notes
    -----
    - Groups atoms into residues.
    - Conceptually corresponds to **Entity** or **Chain** in structural biology.
    - All feature maps inside this container must have `level = FeatureLevel.RESIDUE`.
    """

    feature_map_0D: FeatureMap0D
    feature_map_1D: FeatureMap1D
    feature_map_pair: FeatureMapPair
    level: StructureLevel = StructureLevel.RESIDUE
    type: MoleculeType = None

    def get_type(self):
        return self.type


@dataclasses.dataclass(slots=True)
class ChainContainer(FeatureContainer):
    """
    Container for chain-level features.

    Notes
    -----
    - Groups residues into chains or higher-level assemblies.
    - Conceptually corresponds to a **Biological Assembly** in structural biology.
    - All feature maps inside this container must have `level = FeatureLevel.CHAIN`.
    """

    feature_map_0D: FeatureMap0D
    feature_map_1D: FeatureMap1D
    feature_map_pair: FeatureMapPair
    level: StructureLevel = StructureLevel.CHAIN


def combine_container(
    list_of_container: Iterable[FeatureContainer],
) -> FeatureContainer:
    containers = list(list_of_container)
    if not containers:
        raise ValueError("combine_container requires at least one container.")

    cls = type(containers[0])
    if not all(isinstance(c, cls) for c in containers):
        raise TypeError("All containers must be of the same class.")

    merged_0d = {}
    merged_1d = {}
    merged_pair = {}

    seen_0d, seen_1d, seen_pair = set(), set(), set()
    for c in containers:
        # 0D
        overlap = seen_0d.intersection(c.feature_map_0D.feature_map.keys())
        if overlap:
            raise ValueError(f"Duplicate 0D feature keys: {sorted(overlap)}")
        merged_0d.update(c.feature_map_0D.feature_map)
        seen_0d.update(c.feature_map_0D.feature_map.keys())

        # 1D
        overlap = seen_1d.intersection(c.feature_map_1D.feature_map.keys())
        if overlap:
            raise ValueError(f"Duplicate 1D feature keys: {sorted(overlap)}")
        merged_1d.update(c.feature_map_1D.feature_map)
        seen_1d.update(c.feature_map_1D.feature_map.keys())

        # pair
        overlap = seen_pair.intersection(c.feature_map_pair.feature_map.keys())
        if overlap:
            raise ValueError(f"Duplicate pair feature keys: {sorted(overlap)}")
        merged_pair.update(c.feature_map_pair.feature_map)
        seen_pair.update(c.feature_map_pair.feature_map.keys())

    fm0d = FeatureMap0D(merged_0d)
    fm1d = FeatureMap1D(merged_1d)
    fmpair = FeatureMapPair(merged_pair)

    return cls(fm0d, fm1d, fmpair)
