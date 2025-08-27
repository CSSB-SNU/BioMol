import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, Generic, TypeVar

import numpy as np

from .types import StructureLevel


@dataclasses.dataclass(frozen=True, slots=True)
class Help:
    """Stores descriptive help information."""

    text: str | Sequence[str]

    def __call__(self) -> str | Sequence[str]:
        return self.text


@dataclasses.dataclass(frozen=True, slots=True)
class Feature:
    """Represents a feature with a value and metadata."""

    value: Any
    level: StructureLevel
    additional_info: Any | Help | None = None

    def describe(self) -> str | Sequence[str]:
        return self.additional_info if self.additional_info is not None else ""


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
                msg = "Description should be a string or a list of strings"
                raise TypeError(msg)
        return help_list


T_Feature = TypeVar("T_Feature", bound=Feature)


class FeatureMap(Generic[T_Feature], _HelpMixin):
    """Base class for feature maps with shared behaviors."""

    feature_map: Mapping[str, Feature]

    def __post_init__(self) -> None:
        self._check_level()

    def __getattr__(self, name: str) -> Any:
        """Allow attribute access to values by their names."""
        if name in self.feature_map:
            return self.feature_map[name].value

        msg = f"'{self.__class__.__name__}' object has no attribute '{name}'"
        raise AttributeError(msg)

    def __contains__(self, key: str) -> bool:
        return key in self.feature_map

    def keys(self) -> list[str]:
        return list(self.feature_map.keys())

    def __getitem__(self, key: str) -> T_Feature:
        return self.feature_map[key]

    def __repr__(self) -> str:
        """Return a string representation of the feature map."""
        lines = [f"{self.__class__.__name__}("]
        for k, v in self.feature_map.items():
            lines.append(f"  {k}: {v}")
        lines.append(")")
        return "\n".join(lines)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FeatureMap):
            return NotImplemented
        return self.feature_map == other.feature_map

    def describe(self, key: str | None = None) -> str | Sequence[str]:
        if key is None:
            return self._collect_help(self.feature_map)
        return self.feature_map[key].describe()

    def _check_level(self) -> None:
        levels = {feature.level for feature in self.feature_map.values()}
        if len(levels) != 1:
            raise ValueError(f"Invalid feature levels: {levels}")

    def values(self) -> list[T_Feature]:
        """Return the values of the feature map."""
        return list(self.feature_map.values())

    @property
    def level(self) -> StructureLevel:
        return self.feature_map[next(iter(self.feature_map))].level


@dataclasses.dataclass(frozen=True, slots=True)
class Feature0D(Feature):
    """0D feature with a single scalar value."""

    value: str | int | float | bool
    level: StructureLevel
    additional_info: Any = None

    # Equality and inequality based on .value
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Feature0D):
            return self.value == other.value
        return self.value == other

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    # Optional: ordering support if value is comparable
    def __lt__(self, other: object) -> bool:
        return self.value < (other.value if isinstance(other, Feature0D) else other)

    def __le__(self, other: object) -> bool:
        return self.value <= (other.value if isinstance(other, Feature0D) else other)

    def __gt__(self, other: object) -> bool:
        return self.value > (other.value if isinstance(other, Feature0D) else other)

    def __ge__(self, other: object) -> bool:
        return self.value >= (other.value if isinstance(other, Feature0D) else other)

    # String and repr
    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return f"Feature0D(value={self.value!r}, level={self.level}, info={self.additional_info!r})"


@dataclasses.dataclass(frozen=True, slots=True)
class FeatureMap0D(FeatureMap[Feature0D], _HelpMixin):
    """Represents a mapping of 0D features."""

    feature_map: Mapping[str, Feature0D]


@dataclasses.dataclass(frozen=True, slots=True)
class Feature1D:
    """A thin wrapper around np.ndarray that participates in NumPy dispatch."""

    value: np.ndarray
    level: StructureLevel
    additional_info: Any = None

    def __len__(self) -> int:
        return self.value.shape[0]

    # --- NumPy protocol: array conversion ---
    def __array__(self, dtype: Any | None = None) -> np.ndarray:
        """Allow NumPy to view this object as an ndarray."""
        return self.value.astype(dtype, copy=False) if dtype is not None else self.value

    __array_priority__ = 1000

    # --- Helpers for (un)wrapping ---
    @staticmethod
    def _unwrap(x: Any) -> Any:
        """Recursively unwrap Feature1D to ndarray for arrays/containers."""
        if isinstance(x, Feature1D):
            return x.value
        if isinstance(x, (list, tuple)):
            return type(x)(Feature1D._unwrap(t) for t in x)
        if isinstance(x, dict):
            return {k: Feature1D._unwrap(v) for k, v in x.items()}
        return x

    @staticmethod
    def _first_feature(
        args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> "Feature1D | None":
        """Find the first Feature1D among positional/keyword args."""

        def _iter(o: Any) -> Sequence["Feature1D"]:
            if isinstance(o, Feature1D):
                yield o
            elif isinstance(o, (list, tuple)):
                for t in o:
                    yield from _iter(t)
            elif isinstance(o, dict):
                for v in o.values():
                    yield from _iter(v)

        for a in args:
            for f in _iter(a):
                return f
        for v in (kwargs or {}).values():
            for f in _iter(v):
                return f
        return None

    @staticmethod
    def _wrap(result: Any, ref_len: int, level: StructureLevel, info: Any) -> Any:
        """Wrap arrays back into Feature1D if leading length matches ref_len."""

        def _maybe(arr: Any) -> Any:
            if (
                isinstance(arr, np.ndarray)
                and arr.ndim >= 1
                and arr.shape[0] == ref_len
            ):
                return Feature1D(arr, level, info)
            return arr

        if isinstance(result, tuple):
            return tuple(Feature1D._wrap(r, ref_len, level, info) for r in result)
        if isinstance(result, list):
            return [Feature1D._wrap(r, ref_len, level, info) for r in result]
        if isinstance(result, dict):
            return {
                k: Feature1D._wrap(v, ref_len, level, info) for k, v in result.items()
            }
        return _maybe(result)

    # --- NumPy protocol: ufuncs (np.add, np.exp, ...) ---
    def __array_ufunc__(
        self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any
    ) -> Any:
        ref = Feature1D._first_feature(inputs, kwargs)
        ref_len = ref.value.shape[0] if ref is not None else None
        level = ref.level if ref is not None else self.level
        info = ref.additional_info if ref is not None else self.additional_info

        unwrapped = tuple(Feature1D._unwrap(x) for x in inputs)
        result = getattr(ufunc, method)(*unwrapped, **kwargs)
        if ref_len is None:
            return result
        return Feature1D._wrap(result, ref_len, level, info)

    # --- NumPy protocol: numpy.* functions (np.stack, np.where, ...) ---
    def __array_function__(
        self,
        func: Any,
        types: tuple[type, ...],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if not any(issubclass(t, Feature1D) for t in types):
            return NotImplemented

        ref = Feature1D._first_feature(args, kwargs)
        ref_len = ref.value.shape[0] if ref is not None else None
        level = ref.level if ref is not None else self.level
        info = ref.additional_info if ref is not None else self.additional_info

        # special-case: np.where
        if func is np.where:
            if len(args) == 1 and not kwargs:
                (cond,) = args
                cond = Feature1D._unwrap(cond)
                return np.asarray(cond).nonzero()  # indices tuple

            if len(args) == 3:
                cond, x, y = args
                out = np.where(
                    Feature1D._unwrap(cond), Feature1D._unwrap(x), Feature1D._unwrap(y)
                )
                return (
                    out
                    if ref_len is None
                    else Feature1D._wrap(out, ref_len, level, info)
                )

        out = func(
            *(Feature1D._unwrap(a) for a in args),
            **(
                {}
                if kwargs is None
                else {k: Feature1D._unwrap(v) for k, v in kwargs.items()}
            ),
        )
        return out if ref_len is None else Feature1D._wrap(out, ref_len, level, info)

    # --- Forwarding ---
    def __getattr__(self, name: str) -> Any:
        """Forward ndarray attributes (shape, dtype, etc.)."""
        return getattr(self.value, name)

    # --- Comparisons ---
    def _cmp_op(self, other: Any, op: Any) -> np.ndarray:
        rhs = other.value if isinstance(other, Feature1D) else other
        return op(self.value, rhs)

    def __lt__(self, other: Any) -> np.ndarray:
        return self._cmp_op(other, np.less)

    def __le__(self, other: Any) -> np.ndarray:
        return self._cmp_op(other, np.less_equal)

    def __gt__(self, other: Any) -> np.ndarray:
        return self._cmp_op(other, np.greater)

    def __ge__(self, other: Any) -> np.ndarray:
        return self._cmp_op(other, np.greater_equal)

    def __eq__(self, other: Any) -> np.ndarray:
        if isinstance(other, Feature1D):
            return np.array_equal(self.value, other.value)
        return self._cmp_op(other, np.equal)

    def __ne__(self, other: Any) -> np.ndarray:
        if isinstance(other, Feature1D):
            return not np.array_equal(self.value, other.value)
        return self._cmp_op(other, np.not_equal)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.shape


@dataclasses.dataclass()
class FeatureMap1D(FeatureMap[Feature1D], _HelpMixin):
    feature_map: Mapping[str, Feature1D]

    def __post_init__(self):
        self._check_length()
        self._check_level()

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


@dataclasses.dataclass(frozen=True)
class FeaturePair(Feature):
    value: np.ndarray  # (i,j, v1, v2, ...)
    level: StructureLevel
    additional_info: Any

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FeaturePair):
            return NotImplemented
        return np.array_equal(self.value, other.value) and self.level == other.level

    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.shape

    def crop(self, indices: np.ndarray) -> "FeaturePair":
        """
        Keep only pairs (i, j) whose both endpoints are in `indices` and remap old indices -> [0..K-1] following the order of `indices`.

        indices:
            - 1D bool mask  -> kept = np.flatnonzero(indices)
            - 1D int array  -> kept = indices (must be unique, non-negative)
        """
        if not isinstance(indices, np.ndarray):
            raise TypeError("indices must be a numpy.ndarray")

        # resolve kept node ids in the given order
        if indices.dtype == bool:
            if indices.ndim != 1:
                raise ValueError("Boolean mask for FeaturePair.crop must be 1D.")
            kept = np.flatnonzero(indices)
        elif np.issubdtype(indices.dtype, np.integer):
            if indices.ndim != 1:
                raise ValueError("Integer indices for FeaturePair.crop must be 1D.")
            if (indices < 0).any():
                raise ValueError(
                    "Negative indices are not supported for FeaturePair.crop."
                )
            if np.unique(indices).size != indices.size:
                raise ValueError("Integer indices must be unique.")
            kept = indices.astype(np.int64, copy=False)
        else:
            raise TypeError("indices must be a 1D integer array or a 1D boolean mask.")

        # fast exit on empty input/value
        if self.value.size == 0 or kept.size == 0:
            empty = np.empty((0, self.value.shape[1]), dtype=self.value.dtype)
            return FeaturePair(
                value=empty,
                level=self.level,
                additional_info=self.additional_info,
            )

        # filter rows where both endpoints are in kept
        ij = self.value[:, :2].astype(np.int64, copy=False)  # [N, 2]
        i_col, j_col = ij[:, 0], ij[:, 1]
        in_kept_i = np.isin(i_col, kept, assume_unique=False)
        in_kept_j = np.isin(j_col, kept, assume_unique=False)
        row_mask = in_kept_i & in_kept_j
        if not row_mask.any():
            empty = np.empty((0, self.value.shape[1]), dtype=self.value.dtype)
            return FeaturePair(
                value=empty,
                level=self.level,
                additional_info=self.additional_info,
            )

        sub = self.value[row_mask].copy()

        # remap i,j -> positions in kept (order-preserving)
        # build old->new map; dict + vectorized lookup is robust for sparse/large ids
        index_map = {int(old): int(new) for new, old in enumerate(kept)}
        # vectorized remap
        mapper = np.vectorize(index_map.__getitem__, otypes=[np.int64])
        sub[:, 0] = mapper(sub[:, 0].astype(np.int64))
        sub[:, 1] = mapper(sub[:, 1].astype(np.int64))

        return FeaturePair(
            value=sub,
            level=self.level,
            additional_info=self.additional_info,
        )


@dataclasses.dataclass()
class FeatureMapPair(FeatureMap[FeaturePair], _HelpMixin):
    feature_map: Mapping[str, FeaturePair]

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
