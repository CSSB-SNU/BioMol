from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Final

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from typing_extensions import Self, override

from biomol.exceptions import (
    FeatureOperationError,
    IndexInvalidError,
    IndexMismatchError,
)

if TYPE_CHECKING:
    from numpy.typing import DTypeLike, NDArray


_LOGICAL_UFUNCS: Final[set[np.ufunc]] = {
    np.logical_and,
    np.logical_or,
    np.logical_xor,
    np.logical_not,
}

_COMPARISON_UFUNCS: Final[set[np.ufunc]] = {
    np.equal,
    np.not_equal,
    np.less,
    np.less_equal,
    np.greater,
    np.greater_equal,
}


@dataclass(frozen=True, slots=True, eq=False)
class Feature(NDArrayOperatorsMixin, ABC):
    """A base class for features in a structure.

    This class supports numpy operations and can be indexed and cropped.
    """

    value: NDArray[Any]
    """The underlying numpy array representing the feature data."""

    __array_priority__ = 1000

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the feature."""
        return self.value.shape

    @property
    def ndim(self) -> int:
        """Return the number of dimensions of the feature."""
        return self.value.ndim

    @property
    def dtype(self) -> DTypeLike:
        """Return the data type of the feature."""
        return self.value.dtype

    @property
    def size(self) -> int:
        """Return the total number of elements in the feature."""
        return self.value.size

    def mean(self, axis: int | None = None, **kwargs: Any) -> Any:  # noqa: ANN401
        """Return the mean of the feature along the specified axis."""
        return self.value.mean(axis=axis, **kwargs)

    def sum(self, axis: int | None = None, **kwargs: Any) -> Any:  # noqa: ANN401
        """Return the sum of the feature along the specified axis."""
        return self.value.sum(axis=axis, **kwargs)

    def min(self, axis: int | None = None, **kwargs: Any) -> Any:  # noqa: ANN401
        """Return the minimum of the feature along the specified axis."""
        return self.value.min(axis=axis, **kwargs)

    def max(self, axis: int | None = None, **kwargs: Any) -> Any:  # noqa: ANN401
        """Return the maximum of the feature along the specified axis."""
        return self.value.max(axis=axis, **kwargs)

    @abstractmethod
    def crop(self, indices: NDArray[np.integer]) -> Self:
        """Crop the feature to only include the specified indices.

        Parameters
        ----------
        indices: NDArray[np.integer]
            1D array of node indices to keep. Only integer arrays is allowed.
        """

    @abstractmethod
    def copy(self) -> Self:
        """Return a deep copy of the feature.

        Returns
        -------
        Self
            A new instance with copied numpy arrays.
        """

    @abstractmethod
    def __getitem__(self, key: Any) -> Self:  # noqa: ANN401
        """Get a subset of the feature."""

    def __len__(self) -> int:
        """Return the number of entries in the feature."""
        return len(self.value)

    def __bool__(self) -> bool:
        """Prevent ambiguous truth value evaluation."""
        return bool(self.value)

    def __array__(self, dtype: DTypeLike | None = None) -> NDArray[Any]:
        """Convert the feature to a numpy array.

        This method is called when numpy functions are applied to the feature.
        """
        return np.asarray(self.value, dtype=dtype)

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> Any:  # noqa: ANN401
        """Support numpy universal functions (ufuncs).

        This method is called when numpy ufuncs are applied to the feature.
        """
        if method == "at":
            msg = (
                f"{type(self).__name__} is immutable; "
                "in-place operations are not supported."
            )
            raise FeatureOperationError(msg)
        if "out" in kwargs and kwargs["out"] is not None:
            outs = kwargs["out"]
            if not isinstance(outs, tuple):
                outs = (outs,)
            if any(isinstance(o, Feature) for o in outs):
                msg = (
                    f"{type(self).__name__} is immutable; "
                    "in-place operations are not supported."
                )
                raise FeatureOperationError(msg)

        args = [x.value if isinstance(x, Feature) else x for x in inputs]
        res = getattr(ufunc, method)(*args, **kwargs)
        if method in ("reduce", "reduceat", "accumulate", "outer", "inner"):
            return res
        if ufunc in _COMPARISON_UFUNCS or ufunc in _LOGICAL_UFUNCS:
            return res
        if isinstance(res, tuple):
            return tuple(
                replace(self, value=r)
                if isinstance(r, np.ndarray) and r.shape == self.shape
                else r
                for r in res
            )
        if isinstance(res, np.ndarray) and res.shape == self.shape:
            return replace(self, value=res)
        return res


@dataclass(frozen=True, slots=True, eq=False)
class NodeFeature(Feature):
    """A feature associated with nodes in a structure.

    Parameters
    ----------
    value: np.ndarray
        A numpy array where the first dimension corresponds to the nodes.
    """

    @override
    def crop(self, indices: NDArray[np.integer]) -> Self:
        return self[indices]

    @override
    def copy(self) -> Self:
        return replace(self, value=self.value.copy())

    @override
    def __getitem__(self, key: Any) -> Self:
        return replace(self, value=self.value[key])


@dataclass(frozen=True, slots=True, eq=False)
class EdgeFeature(Feature):
    """A feature associated with edges (pairs of nodes) in a structure.

    Parameters
    ----------
    value: np.ndarray
        A numpy array where the first dimension corresponds to the edges.
    src_indices: NDArray[np.integer]
        A 1D numpy array of source node indices for each edge.
    dst_indices: NDArray[np.integer]
        A 1D numpy array of destination node indices for each edge.
    """

    src_indices: NDArray[np.integer]
    """Source node indices of the edges."""

    dst_indices: NDArray[np.integer]
    """Destination node indices of the edges."""

    def __post_init__(self) -> None:  # noqa: D105
        if not (self.src_indices.ndim == 1 and self.dst_indices.ndim == 1):
            msg = (
                "src_indices and dst_indices must be 1D arrays. Got "
                f"src_indices={self.src_indices.ndim}, "
                f"dst_indices={self.dst_indices.ndim}"
            )
            raise IndexInvalidError(msg)
        if not (len(self.value) == len(self.src_indices) == len(self.dst_indices)):
            msg = (
                "All arrays must have the same length. Got "
                f"value={len(self.value)}, src_indices={len(self.src_indices)}, "
                f"dst_indices={len(self.dst_indices)}"
            )
            raise IndexMismatchError(msg)
        if np.any(self.src_indices < 0) or np.any(self.dst_indices < 0):
            msg = "src_indices and dst_indices must be non-negative."
            raise IndexInvalidError(msg)

    @property
    def src(self) -> NDArray[np.integer]:
        """Return the source node indices of the edges."""
        return self.src_indices

    @property
    def dst(self) -> NDArray[np.integer]:
        """Return the destination node indices of the edges."""
        return self.dst_indices

    @property
    def nodes(self) -> NDArray[np.integer]:
        """Return the unique node indices involved in the edges."""
        return np.unique(np.concatenate([self.src_indices, self.dst_indices]))

    @override
    def crop(self, indices: NDArray[np.integer]) -> Self:
        """Crop the feature to only include the specified indices.

        Keep only pairs (i, j) whose both endpoints are in `indices`.

        Parameters
        ----------
        indices: NDArray[np.integer]
            1D array of node indices to keep. Only integer arrays is allowed.
        """
        if not isinstance(indices, np.ndarray):
            msg = f"Indices must be a numpy.ndarray, got {type(indices)}"
            raise IndexInvalidError(msg)
        if indices.ndim != 1:
            msg = f"Indices must be a 1D array, got {indices.ndim}D array"
            raise IndexInvalidError(msg)
        if not np.issubdtype(indices.dtype, np.integer):
            msg = f"Indices must be a integer array, got {indices.dtype}"
            raise IndexInvalidError(msg)
        if np.any(indices < 0):
            msg = "Negative indices are not allowed."
            raise IndexInvalidError(msg)
        if self.value.size == 0 or indices.size == 0:
            return self._empty_like()

        kept, idx = np.unique(indices, return_index=True)
        src_in_kept = np.isin(self.src_indices, kept, assume_unique=True)
        dst_in_kept = np.isin(self.dst_indices, kept, assume_unique=True)
        row_mask = src_in_kept & dst_in_kept
        if not row_mask.any():
            return self._empty_like()

        new_src = idx[np.searchsorted(kept, self.src_indices[row_mask])]
        new_dst = idx[np.searchsorted(kept, self.dst_indices[row_mask])]
        return replace(
            self,
            value=self.value[row_mask],
            src_indices=new_src,
            dst_indices=new_dst,
        )

    @override
    def copy(self) -> Self:
        """Return a deep copy of the edge feature.

        Returns
        -------
        Self
            A new instance with all numpy arrays copied.
        """
        return replace(
            self,
            value=self.value.copy(),
            src_indices=self.src_indices.copy(),
            dst_indices=self.dst_indices.copy(),
        )

    def _empty_like(self) -> Self:
        empty = np.empty((0, *self.value.shape[1:]), dtype=self.value.dtype)
        ind = np.empty((0,), dtype=self.src_indices.dtype)
        return replace(self, value=empty, src_indices=ind, dst_indices=ind)

    @override
    def __getitem__(self, key: Any) -> Self:
        return replace(
            self,
            value=self.value[key],
            src_indices=self.src_indices[key],
            dst_indices=self.dst_indices[key],
        )
