from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar, Final, Protocol, runtime_checkable

import numpy as np
from typing_extensions import Self, TypeVar, override

from .exceptions import IndexInvalidError, IndexOutOfBoundsError, ViewOperationError
from .types import StructureLevel

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray

    from .biomol import BioMol
    from .container import FeatureContainer
    from .feature import Feature


A_co = TypeVar("A_co", bound="ViewProtocol", default="ViewProtocol", covariant=True)
R_co = TypeVar("R_co", bound="ViewProtocol", default="ViewProtocol", covariant=True)
C_co = TypeVar("C_co", bound="ViewProtocol", default="ViewProtocol", covariant=True)
M_co = TypeVar("M_co", bound="BioMol", default="BioMol", covariant=True)


@runtime_checkable
class ViewProtocol(Protocol[A_co, R_co, C_co, M_co]):
    """Protocol for all views."""

    def __init__(
        self,
        mol: M_co,
        indices: NDArray[np.integer],
    ) -> None: ...

    @property
    def atoms(self) -> A_co:
        """View of the atoms in the selection."""

    @property
    def residues(self) -> R_co:
        """View of the residues in the selection."""

    @property
    def chains(self) -> C_co:
        """View of the chains in the selection."""

    @property
    def level(self) -> StructureLevel:
        """The structural level of the view."""

    @property
    def mol(self) -> M_co:
        """Return the parent molecule."""

    @property
    def indices(self) -> NDArray[np.integer]:
        """Return the indices of the view."""

    @cached_property
    def unique_indices(self) -> NDArray[np.integer]:
        """Return the unique indices of the view, preserving first-occurrence order."""

    def get_feature(self, key: str) -> Feature:
        """Return the feature for the given key, cropped to the view's indices."""

    def get_features(self) -> FeatureContainer:
        """Return the features of the view, cropped to the view's indices."""

    def unique(self) -> Self:
        """Return a new view with unique indices, preserving first-occurrence order."""

    def new(self, indices: NDArray[np.integer]) -> Self:
        """Return a new view with the specified indices."""

    def __repr__(self) -> str:
        """Return a string representation of the view."""

    def __len__(self) -> int:
        """Return the number of elements in the view."""

    def __getattr__(self, key: str) -> Feature:
        """Return the feature for the given key, cropped to the view's indices."""

    def __getitem__(self, key: Any) -> Self:  # noqa: ANN401
        """Return a new view with the specified indices."""

    def __iter__(self) -> Iterator[Self]:
        """Iterate over the elements in the view, yielding single-element views."""

    def __add__(self, other: Self) -> Self:
        """Return a new view with concatenated indices.

        Note that the result may contain duplicate indices.
        """

    def __sub__(self, other: Self) -> Self:
        """Return a new view with indices in self but not in other.

        Note that the result contains only unique indices.
        """

    def __and__(self, other: Self) -> Self:
        """Return a new view with indices in both self and other.

        Note that the result contains only unique indices.
        """

    def __or__(self, other: Self) -> Self:
        """Return a new view with indices in either self or other.

        Note that the result contains only unique indices.
        """

    def __xor__(self, other: Self) -> Self:
        """Return a new view with indices in self or other but not both.

        Note that the result contains only unique indices.
        """

    def __invert__(self) -> Self:
        """Return a new view with indices not in the current view.

        Note that the result contains only unique indices.
        """


class BaseView(ViewProtocol[A_co, R_co, C_co, M_co]):
    """Base class for all views."""

    _level: ClassVar[StructureLevel]

    def __init__(
        self,
        mol: M_co,
        indices: NDArray[np.integer],
    ) -> None:
        indices = np.atleast_1d(indices)
        if indices.ndim != 1:
            msg = f"Indices must be 1-dimensional, but got {indices.ndim}D."
            raise IndexInvalidError(msg)
        max_index = len(mol.get_container(self.level)) - 1
        out_of_bounds = (indices < 0) | (indices > max_index)
        if np.any(out_of_bounds):
            invalid_indices = indices[out_of_bounds]
            msg = (
                f"Indices contain out-of-bounds values: {invalid_indices}. "
                f"Valid range is [0, {max_index}]."
            )
            raise IndexOutOfBoundsError(msg)

        self._mol = mol
        self._indices = indices

    @property
    @override
    def level(self) -> StructureLevel:
        return self._level

    @property
    @override
    def mol(self) -> M_co:
        return self._mol

    @property
    @override
    def indices(self) -> NDArray[np.integer]:
        return self._indices

    @cached_property
    @override
    def unique_indices(self) -> NDArray[np.integer]:
        uniq, idx = np.unique(self.indices, return_index=True)
        order = np.argsort(idx, kind="stable")
        return uniq[order]

    @override
    def get_feature(self, key: str) -> Feature:
        return self._mol.get_container(self.level)[key].crop(self.indices)

    @override
    def get_features(self) -> FeatureContainer:
        return self._mol.get_container(self.level).crop(self.indices)

    @override
    def unique(self) -> Self:
        return self.new(self.unique_indices)

    @override
    def new(self, indices: NDArray[np.integer]) -> Self:
        return self.__class__(self._mol, indices)

    def _check_same_level(self, other: Self) -> None:
        if not isinstance(other, BaseView):
            msg = f"Invalid view type: {type(other)}"
            raise ViewOperationError(msg)
        if self.mol is not other.mol:
            msg = "Cannot operate on views from different molecules."
            raise ViewOperationError(msg)
        if self.level != other.level:
            msg = (
                f"Cannot operate on views of different levels: "
                f"{self.level} and {other.level}"
            )
            raise ViewOperationError(msg)

    @override
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} with {len(self)} elements>"

    @override
    def __len__(self) -> int:
        return len(self.indices)

    @override
    def __getattr__(self, key: str) -> Feature:
        return self.get_feature(key)

    @override
    def __getitem__(self, key: Any) -> Self:
        return self.new(self.indices[key])

    @override
    def __iter__(self) -> Iterator[Self]:
        for i in range(len(self)):
            yield self[i]

    @override
    def __add__(self, other: Self) -> Self:
        self._check_same_level(other)
        return self.new(np.concatenate((self.indices, other.indices)))

    @override
    def __sub__(self, other: Self) -> Self:
        self._check_same_level(other)
        mask = np.isin(self.unique_indices, other.indices, invert=True)
        return self.new(self.unique_indices[mask])

    @override
    def __and__(self, other: Self) -> Self:
        self._check_same_level(other)
        mask = np.isin(self.unique_indices, other.indices)
        return self.new(self.unique_indices[mask])

    @override
    def __or__(self, other: Self) -> Self:
        return (self + other).unique()

    @override
    def __xor__(self, other: Self) -> Self:
        return (self | other) - (self & other)

    @override
    def __invert__(self) -> Self:
        all_indices = np.arange(len(self._mol.get_container(self.level)))
        mask = np.isin(all_indices, self.indices, invert=True)
        return self.new(all_indices[mask])


class AtomView(BaseView):
    """View of the atoms in the selection."""

    _level: Final = StructureLevel.ATOM

    @property
    @override
    def atoms(self) -> Self:
        return self

    @property
    @override
    def residues(self) -> ResidueView:
        indices = self._mol.index_table.atoms_to_residues(self.indices)
        return ResidueView(self._mol, indices)

    @property
    @override
    def chains(self) -> ChainView:
        indices = self._mol.index_table.atoms_to_chains(self.indices)
        return ChainView(self._mol, indices)


class ResidueView(BaseView):
    """View of the residues in the selection."""

    _level: Final = StructureLevel.RESIDUE

    @property
    @override
    def atoms(self) -> AtomView:
        indices = self._mol.index_table.residues_to_atoms(self.indices)
        return AtomView(self._mol, indices)

    @property
    @override
    def residues(self) -> Self:
        return self

    @property
    @override
    def chains(self) -> ChainView:
        indices = self._mol.index_table.residues_to_chains(self.indices)
        return ChainView(self._mol, indices)


class ChainView(BaseView):
    """View of the chains in the selection."""

    _level: Final = StructureLevel.CHAIN

    @property
    @override
    def atoms(self) -> AtomView:
        indices = self._mol.index_table.chains_to_atoms(self.indices)
        return AtomView(self._mol, indices)

    @property
    @override
    def residues(self) -> ResidueView:
        res_idx = self._mol.index_table.chains_to_residues(self.indices)
        return ResidueView(self._mol, res_idx)

    @property
    @override
    def chains(self) -> Self:
        return self
