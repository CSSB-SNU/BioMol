from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

import numpy as np
from typing_extensions import Self, TypeVar, override

from biomol.enums import StructureLevel
from biomol.exceptions import (
    IndexInvalidError,
    IndexOutOfBoundsError,
    ViewOperationError,
)

from .feature import EdgeFeature

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
    """Protocol for all views.

    This protocol defines the common interface for all views, including methods for
    converting between different structural levels, accessing features, and performing
    set operations. User can also implement this protocol to create custom views.

    Parameters
    ----------
    mol : BioMol
        The parent molecule.
    indices : NDArray[np.integer]
        The indices of the elements in the view.
    """

    @property
    def atoms(self) -> A_co:
        """View of the atoms in the selection."""
        ...

    @property
    def residues(self) -> R_co:
        """View of the residues in the selection."""
        ...

    @property
    def chains(self) -> C_co:
        """View of the chains in the selection."""
        ...

    @property
    def level(self) -> StructureLevel:
        """The structural level of the view."""
        ...

    @property
    def mol(self) -> M_co:
        """Return the parent molecule."""
        ...

    @property
    def indices(self) -> NDArray[np.integer]:
        """Return the indices of the view."""
        ...

    @cached_property
    def unique_indices(self) -> NDArray[np.integer]:
        """Return the unique indices of the view, preserving first-occurrence order."""
        ...

    def to_atoms(self, *, unique: bool = False) -> A_co:
        """Return an AtomView of the atoms in the selection.

        If `unique` is True, the resulting view will contain only unique indices,
        preserving first-occurrence order. Default is False.
        """
        ...

    def to_residues(self, *, unique: bool = False) -> R_co:
        """Return a ResidueView of the residues in the selection.

        If `unique` is True, the resulting view will contain only unique indices,
        preserving first-occurrence order. Default is False.
        """
        ...

    def to_chains(self, *, unique: bool = False) -> C_co:
        """Return a ChainView of the chains in the selection.

        If `unique` is True, the resulting view will contain only unique indices,
        preserving first-occurrence order. Default is False.
        """
        ...

    def get_feature(self, key: str) -> Feature:
        """Return the feature for the given key, cropped to the view's indices."""
        ...

    def get_container(self) -> FeatureContainer:
        """Return the feature container cropped to the view's indices."""
        ...

    def unique(self) -> Self:
        """Return a new view with unique indices, preserving first-occurrence order."""
        ...

    def new(self, indices: NDArray[np.integer]) -> Self:
        """Return a new view with the specified indices."""
        ...

    def sort(self) -> Self:
        """Return a new view with sorted indices."""
        ...

    def is_empty(self) -> bool:
        """Return True if the view is empty (contains no elements)."""
        ...

    def is_subset(self, other: Self) -> bool:
        """Return True if the view is a subset of another view."""
        ...

    def select(self, **kwargs: Any) -> Self:  # noqa: ANN401
        """Return a new view filtered by feature values.

        This method allows filtering based on both node and edge features.
        Each keyword argument corresponds to a feature name and its desired value(s).

        Parameters
        ----------
        **kwargs : Any
            Feature names and values to filter by.

        Notes
        -----
        **Node Feature Selection**
            - Filtering is applied **only to the elements within the current view**.
            - Only elements that match the criteria are retained.

        **Edge Feature Selection**
            - Filtering is applied to **edges within the current view**.
            - However, **all edges in the entire molecule** are considered during
              matching.
            - It then returns the elements from the current view that participate in any
              matching edges.
            - This allows selecting nodes based on their interactions with elements
              outside the current view.

        **Value Matching**
            - **Single value** → the feature must **match exactly**.
            - **Sequence** (list, tuple, set, or ndarray) → the feature can **match any
              value** in the sequence.

        Examples
        --------
        Select atoms with name ``CA`` and residue id ``10`` from an ``atom_view``

        .. code-block:: python

            selected_atoms = atom_view.select(name='CA', id=10)

        Select residues with name ``ALA`` or ``GLY`` from a ``residue_view``

        .. code-block:: python

            selected_residues = residue_view.select(name=['ALA', 'GLY'])

        Select atoms from ``chain_view`` that form disulfide bonds.

        .. code-block:: python

            selected_atoms = chain_view.select(bond="disulfide")

        """
        ...

    def __repr__(self) -> str:
        """Return a string representation of the view."""
        ...

    def __len__(self) -> int:
        """Return the number of elements in the view."""
        ...

    def __getattr__(self, key: str) -> Feature:
        """Return the feature for the given key, cropped to the view's indices."""
        ...

    def __getitem__(self, key: Any) -> Self:  # noqa: ANN401
        """Return a new view with the specified indices."""
        ...

    def __iter__(self) -> Iterator[Self]:
        """Iterate over the elements in the view, yielding single-element views."""
        ...

    def __add__(self, other: Self) -> Self:
        """Return a new view with concatenated indices.

        Note that the result may contain duplicate indices.
        """
        ...

    def __sub__(self, other: Self) -> Self:
        """Return a new view with indices in self but not in other.

        Note that the result contains only unique indices.
        """
        ...

    def __and__(self, other: Self) -> Self:
        """Return a new view with indices in both self and other.

        Note that the result contains only unique indices.
        """
        ...

    def __or__(self, other: Self) -> Self:
        """Return a new view with indices in either self or other.

        Note that the result contains only unique indices.
        """
        ...

    def __xor__(self, other: Self) -> Self:
        """Return a new view with indices in self or other but not both.

        Note that the result contains only unique indices.
        """
        ...

    def __invert__(self) -> Self:
        """Return a new view with indices not in the current view.

        Note that the result contains only unique indices.
        """
        ...


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
    def atoms(self) -> AtomView:
        return self.to_atoms(unique=True)

    @property
    @override
    def residues(self) -> ResidueView:
        return self.to_residues(unique=True)

    @property
    @override
    def chains(self) -> ChainView:
        return self.to_chains(unique=True)

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
    def to_atoms(self, *, unique: bool = False) -> AtomView:  # pyright: ignore[reportIncompatibleMethodOverride]
        indices = self.mol.index_table.convert(
            self.indices,
            source=self.level,
            target=StructureLevel.ATOM,
        )
        view = AtomView(self.mol, indices)
        return view.unique() if unique else view

    @override
    def to_residues(self, *, unique: bool = False) -> ResidueView:  # pyright: ignore[reportIncompatibleMethodOverride]
        indices = self.mol.index_table.convert(
            self.indices,
            source=self.level,
            target=StructureLevel.RESIDUE,
        )
        view = ResidueView(self.mol, indices)
        return view.unique() if unique else view

    @override
    def to_chains(self, *, unique: bool = False) -> ChainView:  # pyright: ignore[reportIncompatibleMethodOverride]
        indices = self.mol.index_table.convert(
            self.indices,
            source=self.level,
            target=StructureLevel.CHAIN,
        )
        view = ChainView(self.mol, indices)
        return view.unique() if unique else view

    @override
    def get_feature(self, key: str) -> Feature:
        return self.mol.get_container(self.level)[key].crop(self.indices)

    @override
    def get_container(self) -> FeatureContainer:
        return self.mol.get_container(self.level).crop(self.indices)

    @override
    def unique(self) -> Self:
        return self.new(self.unique_indices)

    @override
    def new(self, indices: NDArray[np.integer]) -> Self:
        return self.__class__(self.mol, indices)

    @override
    def sort(self) -> Self:
        return self.new(np.sort(self.indices))

    @override
    def is_empty(self) -> bool:
        return len(self) == 0

    @override
    def is_subset(self, other: Self) -> bool:
        self._check_same_level(other)
        return all(np.isin(self.indices, other.indices))

    @override
    def select(self, **kwargs: Any) -> Self:
        if not kwargs:
            return self

        mask = np.ones(len(self), dtype=bool)
        for key, value in kwargs.items():
            feature = self.mol.get_container(self.level)[key]
            if not isinstance(feature, EdgeFeature):
                feature = feature.crop(self.indices)
            feature_value = np.asarray(feature)
            if isinstance(value, (list, tuple, set, np.ndarray)):
                feature_mask = np.isin(feature_value, list(value))
            else:
                feature_mask = feature_value == value
            if isinstance(feature, EdgeFeature):
                indices = feature[feature_mask].nodes
                mask &= np.isin(self.indices, indices)
            else:
                mask &= feature_mask

        return self.new(self.indices[mask])

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
        all_indices = np.arange(len(self.mol.get_container(self.level)))
        mask = np.isin(all_indices, self.indices, invert=True)
        return self.new(all_indices[mask])


class AtomView(BaseView):
    """View of the atoms in the selection."""

    _level: ClassVar[StructureLevel] = StructureLevel.ATOM


class ResidueView(BaseView):
    """View of the residues in the selection."""

    _level: ClassVar[StructureLevel] = StructureLevel.RESIDUE


class ChainView(BaseView):
    """View of the chains in the selection."""

    _level: ClassVar[StructureLevel] = StructureLevel.CHAIN
