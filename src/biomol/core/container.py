import numpy as np
import dataclasses
import enum
from collections.abc import Iterable
from typing import Any

from biomol.core.feature import (
    FeatureLevel,
    Feature0D,
    Feature1D,
    FeaturePair,
    FeatureMap0D,
    FeatureMap1D,
    FeatureMapPair,
)


@enum.unique
class MoleculeType(enum.Enum):
    POLYMER = "polymer"
    NONPOLYMER = "non-polymer"
    BRANCHED = "branched"
    WATER = "water"
    BIOASSEMBLY = "bioassembly"


@enum.unique
class PolymerType(enum.Enum):
    PROTEIN = "polypeptide(L)"
    PROTEIN_D = "polypeptide(D)"
    PNA = "peptide nucleic acid"
    RNA = "polyribonucleotide"
    DNA = "polydeoxyribonucleotide"  # TODO
    NA_HYBRID = "polydeoxyribonucleotide/polyribonucleotide hybrid"
    ETC = "etc"


@dataclasses.dataclass(slots=True)
class FeatureContainer:
    feature_map_0D: FeatureMap0D
    feature_map_1D: FeatureMap1D
    feature_map_pair: FeatureMapPair

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
        self.key_to_map = {
            **self.feature_map_0D.feature_map,
            **self.feature_map_1D.feature_map,
            **self.feature_map_pair.feature_map,
        }

    def __getitem__(self, idx: str | int | slice | tuple[int, int]) -> Any:
        if isinstance(idx, str):
            return self.key_to_map[idx]
        elif isinstance(idx, int | slice):
            return self.feature_map_1D[idx]
        elif isinstance(idx, tuple):
            return self.feature_map_pair[idx]
        else:
            raise TypeError(f"Unsupported index type: {type(idx).__name__}")

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
                    "mask": v.mask,
                    "level": v.level.value,
                    "info": v.additional_info,
                }
                for k, v in self.feature_map_1D.feature_map.items()
            },
            "pair": {
                k: {
                    "ijv": v.to_numpy(),  # (N,3) stacked [i,j,v]
                    "unidirectional": v.unidirectional,
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
                level if isinstance(level, FeatureLevel) else FeatureLevel(level)
            )
            map_0d[k] = Feature0D(
                key=k,
                value=d.get("value"),
                level=level_enum,
                additional_info=d.get("info"),
            )

        map_1d: dict[str, Feature1D] = {}
        for k, d in (data.get("1D") or {}).items():
            level = d.get("level")
            level_enum = (
                level if isinstance(level, FeatureLevel) else FeatureLevel(level)
            )
            map_1d[k] = Feature1D(
                key=k,
                value=d.get("value"),
                mask=d.get("mask"),
                level=level_enum,
                additional_info=d.get("info"),
            )

        map_pair: dict[str, FeaturePair] = {}
        for k, d in (data.get("pair") or {}).items():
            level = d.get("level")
            level_enum = (
                level if isinstance(level, FeatureLevel) else FeatureLevel(level)
            )
            ijv = d.get("ijv")
            if ijv is None:
                raise ValueError(f"Missing 'ijv' for pair feature '{k}'")
            ijv = np.asarray(ijv)
            if ijv.ndim != 2 or ijv.shape[1] != 3:
                raise ValueError(
                    f"'ijv' must have shape (N,3); got {ijv.shape} for '{k}'"
                )
            fp = FeaturePair.from_numpy(
                key=k,
                data=ijv,
                level=level_enum,
                unidirectional=bool(d.get("unidirectional", False)),
                additional_info=d.get("info"),
            )
            map_pair[k] = fp

        fm0d = FeatureMap0D(map_0d)
        fm1d = FeatureMap1D(map_1d)
        fmpair = FeatureMapPair(map_pair)
        return cls(fm0d, fm1d, fmpair)


@dataclasses.dataclass(slots=True)
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

    def __post_init__(self):
        self._check_level()

    def _check_level(self):
        levels = {
            "0D": self.feature_map_0D.level,
            "1D": self.feature_map_1D.level,
            "pair": self.feature_map_pair.level,
        }
        if any(level != FeatureLevel.ATOM for level in levels.values()):
            raise ValueError("All feature maps must have level ATOM.")


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

    def __post_init__(self):
        self._check_level()

    def _check_level(self):
        levels = {
            "0D": self.feature_map_0D.level,
            "1D": self.feature_map_1D.level,
            "pair": self.feature_map_pair.level,
        }
        if any(level != FeatureLevel.RESIDUE for level in levels.values()):
            raise ValueError("All feature maps must have level RESIDUE.")


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

    def __post_init__(self):
        self._check_level()

    def _check_level(self):
        levels = {
            "0D": self.feature_map_0D.level,
            "1D": self.feature_map_1D.level,
            "pair": self.feature_map_pair.level,
        }
        if any(level != FeatureLevel.CHAIN for level in levels.values()):
            raise ValueError("All feature maps must have level CHAIN.")


def combine_container(list_of_container: Iterable[FeatureContainer]) -> FeatureContainer:
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
