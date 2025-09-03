from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np
from typing_extensions import Self

from .exceptions import FeatureIndicesError, FeatureShapeError


class Feature(Protocol):
    """Protocol for feature objects."""

    value: np.ndarray
    description: str | None

    def crop(self, indices: np.ndarray) -> Self: ...


@dataclass(frozen=True, slots=True)
class NodeFeature:
    """A feature associated with nodes in a structure."""

    value: np.ndarray
    description: str | None = None

    def crop(self, indices: np.ndarray) -> Self:
        """Crop the feature to only include the specified indices.

        Parameters
        ----------
        indices : np.ndarray
            Indices to keep. Supported formats:
            - 1D integer array of indices (must be unique)
            - 1D boolean mask (must have same length as value)

        Returns
        -------
        NodeFeature
            A new NodeFeature containing only the specified indices.
        """
        if not isinstance(indices, np.ndarray):
            msg = "Indices must be a numpy.ndarray"
            raise FeatureIndicesError(msg)
        if indices.ndim != 1:
            msg = "Indices must be a 1D array"
            raise FeatureIndicesError(msg)
        return replace(self, value=self.value[indices])

    def __getitem__(self, indices: np.ndarray) -> Self:
        return self.crop(indices)


@dataclass(frozen=True, slots=True)
class EdgeFeature:
    """A feature associated with edges (pairs of nodes) in a structure."""

    value: np.ndarray
    src_indices: np.ndarray
    dst_indices: np.ndarray
    description: str | None = None

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

    def __getitem__(self, indices: np.ndarray) -> Self:
        return self.crop(indices)

    def crop(self, indices: np.ndarray, remapping: bool = False) -> Self:
        """Crop the feature to only include the specified indices.

        Keep only pairs (i, j) whose both endpoints are in `indices`.

        Parameters
        ----------
        indices: np.ndarray
            Indices to keep. Supported formats:
            - 1D integer array of indices (negative disallowed, must be unique)
            - 1D boolean mask (must have same length as value)
        """
        if not isinstance(indices, np.ndarray):
            msg = "Indices must be a numpy.ndarray"
            raise FeatureIndicesError(msg)
        if indices.ndim != 1:
            msg = f"Indices must be a 1D array, got {indices.ndim}D array"
            raise FeatureIndicesError(msg)

        if indices.dtype == bool:
            if len(indices) != len(self.value):
                msg = (
                    "Boolean mask must have the same length as the number of edges. "
                    f"Got mask length={len(indices)}, value length={len(self.value)}."
                )
                raise FeatureIndicesError(msg)
            kept = np.flatnonzero(indices)
        elif np.issubdtype(indices.dtype, np.integer):
            if np.any(indices < 0):
                msg = "Negative indices are not allowed."
                raise FeatureIndicesError(msg)
            if np.unique(indices).size != indices.size:
                msg = "Integer indices must be unique."
                raise FeatureIndicesError(msg)
            kept = indices
        else:
            msg = "indices must be a boolean or integer array"
            raise FeatureIndicesError(msg)

        if self.value.size == 0 or kept.size == 0:
            empty = np.empty((0,) + self.value.shape[1:], dtype=self.value.dtype)
            ind = np.empty((0,), dtype=self.src_indices.dtype)
            return replace(self, value=empty, src_indices=ind, dst_indices=ind)

        src_in_kept = np.isin(self.src_indices, kept, assume_unique=False)
        dst_in_kept = np.isin(self.dst_indices, kept, assume_unique=False)
        row_mask = src_in_kept & dst_in_kept
        if not row_mask.any():
            empty = np.empty((0,) + self.value.shape[1:], dtype=self.value.dtype)
            ind = np.empty((0,), dtype=self.src_indices.dtype)
            return replace(self, value=empty, src_indices=ind, dst_indices=ind)

        return replace(
            self,
            value=self.value[row_mask],
            src_indices=self.src_indices[row_mask],
            dst_indices=self.dst_indices[row_mask],
        )
