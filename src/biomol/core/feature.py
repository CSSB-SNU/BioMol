from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
from typing_extensions import Self, override

from .exceptions import FeatureIndicesError, FeatureShapeError

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True, kw_only=True)
class Feature(ABC):
    """A base class for features in a structure."""

    value: np.ndarray
    description: str | None = None

    @abstractmethod
    def __getitem__(self, key: Any) -> Self:  # noqa: ANN401
        """Get a subset of the feature."""

    @abstractmethod
    def crop(self, indices: NDArray[np.integer]) -> Self:
        """Crop the feature to only include the specified indices.

        Parameters
        ----------
        indices: NDArray[np.integer]
            1D array of node indices to keep. Only integer arrays is allowed.
        """


@dataclass(frozen=True, slots=True, kw_only=True)
class NodeFeature(Feature):
    """A feature associated with nodes in a structure."""

    @override
    def __getitem__(self, key: Any) -> Self:
        return replace(self, value=self.value[key])

    @override
    def crop(self, indices: NDArray[np.integer]) -> Self:
        return self[indices]


@dataclass(frozen=True, slots=True, kw_only=True)
class EdgeFeature(Feature):
    """A feature associated with edges (pairs of nodes) in a structure."""

    src_indices: NDArray[np.integer]
    dst_indices: NDArray[np.integer]

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

    def __post_init__(self) -> None:  # noqa: D105
        if not (self.src_indices.ndim == 1 and self.dst_indices.ndim == 1):
            msg = (
                "src_indices and dst_indices must be 1D arrays. Got "
                f"src_indices={self.src_indices.ndim}, "
                f"dst_indices={self.dst_indices.ndim}"
            )
            raise FeatureShapeError(msg)
        if not (len(self.value) == len(self.src_indices) == len(self.dst_indices)):
            msg = (
                "All arrays must have the same length. Got "
                f"value={len(self.value)}, src_indices={len(self.src_indices)}, "
                f"dst_indices={len(self.dst_indices)}"
            )
            raise FeatureShapeError(msg)
        if np.any(self.src_indices < 0) or np.any(self.dst_indices < 0):
            msg = "src_indices and dst_indices must be non-negative."
            raise FeatureIndicesError(msg)

    @override
    def __getitem__(self, key: Any) -> Self:
        return replace(
            self,
            value=self.value[key],
            src_indices=self.src_indices[key],
            dst_indices=self.dst_indices[key],
        )

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
            raise FeatureIndicesError(msg)
        if indices.ndim != 1:
            msg = f"Indices must be a 1D array, got {indices.ndim}D array"
            raise FeatureIndicesError(msg)
        if not np.issubdtype(indices.dtype, np.integer):
            msg = f"Indices must be a integer array, got {indices.dtype}"
            raise FeatureIndicesError(msg)
        if np.any(indices < 0):
            msg = "Negative indices are not allowed."
            raise FeatureIndicesError(msg)
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

    def _empty_like(self) -> Self:
        empty = np.empty((0,) + self.value.shape[1:], dtype=self.value.dtype)
        ind = np.empty((0,), dtype=self.src_indices.dtype)
        return replace(self, value=empty, src_indices=ind, dst_indices=ind)
