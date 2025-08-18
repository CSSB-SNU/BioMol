import numpy as np
import dataclasses
import enum
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any


@enum.unique
class FeatureLevel(enum.Enum):
    ATOM = "atom"
    RESIDUE = "residue"
    CHAIN = "chain"


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

    def __post_init__(self):
        self._check_level()

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

    def __contains__(self, key: str) -> bool:
        return key in self.feature_map

    def keys(self) -> list[str]:
        return list(self.feature_map.keys())

    def values(self) -> list[Feature0D]:
        return list(self.feature_map.values())

    def describe(self, key: str | None = None) -> str | Sequence[str]:
        if key is None:
            return self._collect_help(self.feature_map)
        return self.feature_map[key].describe()

    def _check_level(self):
        levels = {feature.level for feature in self.feature_map.values()}
        if len(levels) != 1:
            raise ValueError(f"Invalid feature levels: {levels}")
        self.level = levels.pop()  # Set the level to the single level found


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

    def crop(self, indices: np.ndarray) -> "Feature1D":
        """
        Crop Feature1D along the first axis.

        Parameters
        ----------
        indices : np.ndarray
            - 1D integer array of indices (negative allowed)
            - 1D boolean mask (must have same length as value)
            - slice object (will be converted internally to np.arange)

        Returns
        -------
        Feature1D
            Cropped feature.
        """
        L = self.value.shape[0]

        if isinstance(indices, slice):
            sel = np.arange(L)[indices]
        else:
            sel = np.asarray(indices)

        if sel.dtype == bool:
            if sel.shape[0] != L:
                raise ValueError(
                    f"Boolean mask must have shape ({L},), got {sel.shape}."
                )
            new_value = self.value[sel]
            new_mask = None if self.mask is None else self.mask[sel]
        elif np.issubdtype(sel.dtype, np.integer):
            if sel.ndim != 1:
                raise ValueError("Integer indices must be a 1D array.")
            if ((sel >= L) | (sel < -L)).any():
                raise IndexError(f"Indices out of range for length {L}.")
            new_value = self.value[sel]
            new_mask = None if self.mask is None else self.mask[sel]
        else:
            raise TypeError(
                f"indices must be int array, bool mask, or slice; got {sel.dtype}"
            )

        return Feature1D(
            key=self.key,
            value=new_value,
            mask=new_mask,
            level=self.level,
            additional_info=self.additional_info,
        )


@dataclasses.dataclass()
class FeatureMap1D(FeatureMap, _HelpMixin):
    feature_map: Mapping[str, Feature1D]

    def __post_init__(self):
        self._check_length()
        self._check_level()

    def __repr__(self) -> str:
        output = "FeatureMap1D(\n"
        for value in self.feature_map.values():
            output += f"  {value}\n"
        output += ")"
        return output

    def __getitem__(self, key: str | int | slice) -> Any:
        if isinstance(key, str):
            return self.feature_map[key]
        return {k: v[key] for k, v in self.feature_map.items()}

    def __contains__(self, key: str) -> bool:
        return key in self.feature_map

    def keys(self) -> list[str]:
        return list(self.feature_map.keys())

    def values(self) -> list[Feature1D]:
        return list(self.feature_map.values())

    def describe(self, key: str | None = None) -> Sequence[str]:
        if key is None:
            return self._collect_help(self.feature_map)
        return self.feature_map[key].describe()

    def crop(self, indices: np.ndarray) -> "FeatureMap1D":
        """
        Crop FeatureMap1D along the first axis.

        Parameters
        ----------
        indices : np.ndarray
            - 1D integer array of indices (negative allowed)
            - 1D boolean mask (must have same length as value)
            - slice object (will be converted internally to np.arange)

        Returns
        -------
        FeatureMap1D
            Cropped feature map.
        """
        cropped_map = {}
        for key, feature in self.feature_map.items():
            cropped_map[key] = feature.crop(indices)
        return FeatureMap1D(cropped_map)

    def _check_length(self):
        lengths = [len(v) for v in self.values()]
        if not all(length == lengths[0] for length in lengths):
            raise ValueError("All features must have the same length")

    def _check_level(self):
        levels = {feature.level for feature in self.feature_map.values()}
        if len(levels) != 1:
            raise ValueError(f"Invalid feature levels: {levels}")
        self.level = levels.pop()  # Set the level to the single level found


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

    def crop(self, indices: np.ndarray) -> "FeaturePair":
        """
        Keep only pairs (i, j) whose both endpoints are in `indices`,
        and remap old node indices -> new compact indices [0..K-1]
        in the given order.

        Parameters
        ----------
        indices : np.ndarray
            - 1D integer array of UNIQUE, NON-NEGATIVE node indices to keep
              (defines new ordering)
            - OR 1D boolean mask where True means keep (length = #nodes)

        Returns
        -------
        FeaturePair
            Cropped pair feature with remapped node indices.

        Notes
        -----
        - Negative integer indices are not supported because FeaturePair
          does not know the global length.
        - Duplicate integer indices are not allowed.
        - The unidirectional flag is preserved. Only the existing edges
          are filtered and remapped; no symmetric edges are added.
        """
        if not isinstance(indices, np.ndarray):
            raise TypeError("indices must be a numpy.ndarray")

        if indices.dtype == bool:
            if indices.ndim != 1:
                raise ValueError("Boolean mask for FeaturePair.crop must be 1D.")
            selected = np.flatnonzero(indices)
        elif np.issubdtype(indices.dtype, np.integer):
            if indices.ndim != 1:
                raise ValueError("Integer indices for FeaturePair.crop must be 1D.")
            if (indices < 0).any():
                raise ValueError(
                    "Negative indices are not supported for FeaturePair.crop."
                )
            if np.unique(indices).shape[0] != indices.shape[0]:
                raise ValueError("Integer indices must be unique.")
            selected = indices
        else:
            raise TypeError("indices must be a 1D integer array or a 1D boolean mask.")

        index_map = {int(old): int(new) for new, old in enumerate(selected)}
        selected_set = set(index_map.keys())

        new_mapping: dict[tuple[int, int], Any] = {}
        for (i, j), v in self.value.items():
            if (i in selected_set) and (j in selected_set):
                ni = index_map[i]
                nj = index_map[j]
                new_mapping[(ni, nj)] = v

        return FeaturePair(
            key=self.key,
            value=new_mapping,
            level=self.level,
            unidirectional=self.unidirectional,
            additional_info=self.additional_info,
        )

    def to_numpy(self) -> dict[str, np.ndarray]:
        """Convert pair mapping into numpy arrays (I, J, V)."""
        I, J, V = [], [], []
        for (i, j), v in self.value.items():
            I.append(i)
            J.append(j)
            V.append(v)
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

    def __post_init__(self):
        self._check_level()

    def __repr__(self) -> str:
        output = "FeatureMapPair(\n"
        for value in self.feature_map.values():
            output += f"  {value}\n"
        output += ")"
        return output

    def __getitem__(self, key: str | tuple[int, int]) -> Any:
        if isinstance(key, str):
            return self.feature_map[key]
        return {k: v[key] for k, v in self.feature_map.items()}

    def __contains__(self, key: str) -> bool:
        return key in self.feature_map

    def keys(self) -> list[str]:
        return list(self.feature_map.keys())

    def values(self) -> list[FeaturePair]:
        return list(self.feature_map.values())

    def crop(self, indices: np.ndarray) -> "FeatureMapPair":
        """
        Crop the feature map pair to only include the specified indices.

        Parameters
        ----------
        indices : np.ndarray
            1D array of indices to keep.

        Returns
        -------
        FeatureMapPair
            A new FeatureMapPair containing only the specified indices.
        """
        cropped_feature_map = {
            key: feature_pair.crop(indices)
            for key, feature_pair in self.feature_map.items()
        }
        return FeatureMapPair(cropped_feature_map)

    def describe(self, key: str | None = None) -> Sequence[str]:
        if key is None:
            return self._collect_help(self.feature_map)
        return self.feature_map[key].describe()

    def _check_level(self):
        levels = {feature.level for feature in self.feature_map.values()}
        if len(levels) != 1:
            raise ValueError(f"Invalid feature levels: {levels}")
        self.level = levels.pop()  # Set the level to the single level found
