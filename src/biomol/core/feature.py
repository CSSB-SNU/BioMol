import numpy as np
import dataclasses
import enum
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any


@enum.unique
class FeatureLevel(enum.Enum):
    ATOM = "atom"
    CHEMCOMP = "chem_comp"
    ENTITY = "entity"
    CHAIN = "chain"
    UNIT = "asymmetric_unit"
    FULL = "full"


@dataclasses.dataclass(frozen=True, slots=True)
class Help:
    """Stores descriptive help information."""

    text: str | Sequence[str]

    def __call__(self) -> str | Sequence[str]:
        return self.text


class Feature(ABC):
    @abstractmethod
    def describe(self) -> str | Sequence[str]:
        pass


class FeatureMap(ABC):
    @abstractmethod
    def keys(self) -> list[str]:
        """Return the keys of the feature map."""
        pass

    @abstractmethod
    def values(self) -> list[Feature]:
        """Return the values of the feature map."""
        pass

    @abstractmethod
    def __getitem__(self, key) -> Any:
        """Get feature by key, index, or slice."""
        pass

    @abstractmethod
    def __contains__(self, key: str) -> bool:
        """Check if the feature map contains a feature by key."""
        pass

    @abstractmethod
    def describe(self, key: str | Sequence[str] | None = None) -> str | Sequence[str]:
        pass


class _HelpMixin:
    """Reusable logic for collecting help/description strings."""

    @staticmethod
    def _collect_help(feature_map: Mapping[str, Feature]) -> list[str]:
        help_list: list[str] = []
        for items in feature_map.values():
            h = items.describe()
            if isinstance(h, str):
                help_list.append(h)
            elif isinstance(h, Sequence):
                help_list.extend(h)
            else:
                raise ValueError("Description should be a string or a list of strings")
        return help_list


@dataclasses.dataclass(frozen=True, slots=True)
class Feature0D(Feature):
    key: str
    value: Any
    level: FeatureLevel
    additional_info: Any

    def describe(self) -> str | Sequence[str]:
        return self.additional_info  # TODO : parse additional_info


@dataclasses.dataclass(frozen=True, slots=True)
class FeatureMap0D(FeatureMap, _HelpMixin):
    feature_map: Mapping[str, Feature0D]

    def keys(self) -> list[str]:
        return list(self.feature_map.keys())

    def values(self) -> list[Feature0D]:
        return list(self.feature_map.values())

    def describe(self, key: str | None = None) -> str | Sequence[str]:
        if key is None:
            return self._collect_help(self.feature_map)
        return self.feature_map[key].describe()

    def __contains__(self, key: str) -> bool:
        return key in self.feature_map

    def __repr__(self) -> str:
        output = "FeatureMap0D(\n"
        for value in self.feature_map.values():
            output += f"  {value}\n"
        output += ")"
        return output

    def __getitem__(self, key: str) -> Feature0D:
        if not isinstance(key, str):
            raise ValueError("Key should be a string for 0D feature map")
        return self.feature_map[key]


@dataclasses.dataclass(frozen=True, slots=True)
class Feature1D(Feature):
    key: str
    value: np.ndarray  # (L,) or (L, C)
    mask: np.ndarray | None
    level: FeatureLevel
    additional_info: Any

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        return self.value[idx], self.mask[idx] if self.mask is not None else None

    def __len__(self) -> int:
        return len(self.value)

    def describe(self) -> str | Sequence[str]:
        return self.additional_info  # TODO : parse additional_info


@dataclasses.dataclass()
class FeatureMap1D(FeatureMap, _HelpMixin):
    feature_map: Mapping[str, Feature1D]

    def __post_init__(self):
        self._check_length()

    def _check_length(self):
        lengths = [len(v) for v in self.values()]
        if not all(length == lengths[0] for length in lengths):
            raise ValueError("All features must have the same length")

    def keys(self) -> list[str]:
        return list(self.feature_map.keys())

    def values(self) -> list[Feature1D]:
        return list(self.feature_map.values())

    def __getitem__(self, key: str | int | slice) -> Any:
        if isinstance(key, str):
            return self.feature_map[key]
        return {k: v[key] for k, v in self.feature_map.items()}

    def __contains__(self, key: str) -> bool:
        return key in self.feature_map

    def describe(self, key: str | None = None) -> Sequence[str]:
        if key is None:
            return self._collect_help(self.feature_map)
        return self.feature_map[key].describe()

    def __repr__(self) -> str:
        output = "FeatureMap1D(\n"
        for value in self.feature_map.values():
            output += f"  {value}\n"
        output += ")"
        return output


@dataclasses.dataclass(frozen=True)
class FeaturePair(Feature):
    key: str
    value: Mapping[tuple[int, int], Any]
    level: FeatureLevel
    unidirectional: bool
    additional_info: Any

    def __getitem__(self, idx: tuple[int, int]) -> Any:
        return self.value[idx]

    def describe(self) -> str | Sequence[str]:
        return self.additional_info  # TODO : parse additional_info

    def indices(self) -> list[tuple[int, int]]:
        return list(self.value.keys())

    def to_numpy(self) -> dict[str, np.ndarray]:
        """Convert pair mapping into numpy arrays (I, J, V)."""
        I, J, V = [], [], []
        for (i, j), v in self.value.items():
            I.append(i)
            J.append(j)
            V.append(v)

        # stack : (i,j,v)
        stacked = np.stack((I, J, V), axis=-1)
        return stacked

    @classmethod
    def from_numpy(
        cls,
        key: str,
        data: np.ndarray,
        level: FeatureLevel,
        unidirectional: bool,
        additional_info: Any,
    ) -> "FeaturePair":
        """Rebuild FeaturePair from numpy arrays."""
        I, J, V = data[:, 0], data[:, 1], data[:, 2]
        mapping = {(int(i), int(j)): v for i, j, v in zip(I, J, V)}
        return cls(
            key=key,
            value=mapping,
            level=level,
            unidirectional=unidirectional,
            additional_info=additional_info,
        )


@dataclasses.dataclass()
class FeatureMapPair(FeatureMap, _HelpMixin):
    feature_map: Mapping[str, FeaturePair]

    def keys(self) -> list[str]:
        return list(self.feature_map.keys())

    def values(self) -> list[FeaturePair]:
        return list(self.feature_map.values())

    def __getitem__(self, key: str | tuple[int, int]) -> Any:
        if isinstance(key, str):
            return self.feature_map[key]
        return {k: v[key] for k, v in self.feature_map.items()}

    def __contains__(self, key: str) -> bool:
        return key in self.feature_map

    def describe(self, key: str | None = None) -> Sequence[str]:
        if key is None:
            return self._collect_help(self.feature_map)
        return self.feature_map[key].describe()

    def __repr__(self) -> str:
        output = "FeatureMapPair(\n"
        for value in self.feature_map.values():
            output += f"  {value}\n"
        output += ")"
        return output


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
        # Ensure consistent types for empty defaults
        self.feature_map_0D = feature_map_0D or FeatureMap0D({})
        self.feature_map_1D = feature_map_1D or FeatureMap1D({})
        self.feature_map_pair = feature_map_pair or FeatureMapPair({})
        self._check_duplicate_keys()
        self._check_length()
        self._check_feature_level()
        self.key_to_map = {
            **self.feature_map_0D.feature_map,
            **self.feature_map_1D.feature_map,
            **self.feature_map_pair.feature_map,
        }

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

    def _check_feature_level(self) -> None:
        levels = {
            **self.feature_map_0D.levels(),
            **self.feature_map_1D.levels(),
            **self.feature_map_pair.levels(),
        }
        for key, level in levels.items():
            if level != FeatureLevel.FULL:
                raise ValueError(f"Feature '{key}' is not at FULL level")

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
        # 0D (supports legacy: value is stored directly)
        f0d_items: dict[str, Feature0D] = {}
        for k, raw in data.get("0D", {}).items():
            if isinstance(raw, dict) and "value" in raw:
                val = raw["value"]
                lvl = FeatureLevel(raw.get("level", FeatureLevel.FULL.value))
                info = raw.get("info", None)
            else:
                val = raw
                lvl = FeatureLevel.FULL
                info = None
            f0d_items[k] = Feature0D(key=k, value=val, level=lvl, additional_info=info)
        f0d = FeatureMap0D(f0d_items)

        # 1D (supports legacy: only {"value","mask"})
        f1d_items: dict[str, Feature1D] = {}
        for k, raw in data.get("1D", {}).items():
            val = raw["value"] if isinstance(raw, dict) else raw
            msk = raw.get("mask", None) if isinstance(raw, dict) else None
            lvl = (
                FeatureLevel(raw.get("level", FeatureLevel.FULL.value))
                if isinstance(raw, dict)
                else FeatureLevel.FULL
            )
            info = raw.get("info", None) if isinstance(raw, dict) else None
            f1d_items[k] = Feature1D(
                key=k, value=val, mask=msk, level=lvl, additional_info=info
            )
        f1d = FeatureMap1D(f1d_items)

        # pair (supports legacy: value was a bare (N,3) array)
        fpair_items: dict[str, FeaturePair] = {}
        for k, raw in data.get("pair", {}).items():
            if isinstance(raw, dict) and "ijv" in raw:
                ijv = raw["ijv"]
                unidir = bool(raw.get("unidirectional", True))
                lvl = FeatureLevel(raw.get("level", FeatureLevel.FULL.value))
                info = raw.get("info", None)
            else:
                ijv = raw
                unidir = True
                lvl = FeatureLevel.FULL
                info = None
            fpair_items[k] = FeaturePair.from_numpy(
                key=k,
                data=ijv,
                level=lvl,
                unidirectional=unidir,
                additional_info=info,
            )
        fpair = FeatureMapPair(fpair_items)

        return cls(f0d, f1d, fpair)

@dataclasses.dataclass(slots=True)
class AtomFeatureContainer(FeatureContainer):


def combine_feature_map_container():
    pass


def crop():
    pass
