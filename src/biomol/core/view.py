from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Final, Generic

from typing_extensions import Self, TypeVar, override

from .types import AtomProtoT, ChainProtoT, ResidueProtoT, StructureLevel

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from .biomol import BioMol
    from .container import FeatureContainer
    from .feature import Feature


LevelProtoT = TypeVar("LevelProtoT", default=Any)


class ViewLike(ABC, Generic[AtomProtoT, ResidueProtoT, ChainProtoT, LevelProtoT]):
    """A generic interface for views."""

    @property
    @abstractmethod
    def atoms(self) -> AtomView[AtomProtoT, ResidueProtoT, ChainProtoT] | AtomProtoT:
        """View of the atoms in the selection."""

    @property
    @abstractmethod
    def residues(
        self,
    ) -> ResidueView[AtomProtoT, ResidueProtoT, ChainProtoT] | ResidueProtoT:
        """View of the residues in the selection."""

    @property
    @abstractmethod
    def chains(self) -> ChainView[AtomProtoT, ResidueProtoT, ChainProtoT] | ChainProtoT:
        """View of the chains in the selection."""


class BaseView(ViewLike[AtomProtoT, ResidueProtoT, ChainProtoT, LevelProtoT]):
    """Base class for all views."""

    _level: ClassVar[StructureLevel]

    def __init__(
        self,
        mol: BioMol[AtomProtoT, ResidueProtoT, ChainProtoT],
        indices: NDArray[np.integer],
    ) -> None:
        if indices.ndim != 1:
            msg = f"Indices must be 1-dimensional, but got {indices.ndim}D."
            raise ValueError(msg)
        self._mol = mol
        self._indices = indices

    @property
    def level(self) -> StructureLevel:
        """The structural level of the view."""
        return self._level

    @property
    def mol(self) -> BioMol[AtomProtoT, ResidueProtoT, ChainProtoT]:
        """Return the parent molecule."""
        return self._mol

    @property
    def features(self) -> FeatureContainer:
        """Return the features of the view."""
        return self._mol.get_container(self.level)

    def _check_same_level(self, other: Self) -> None:
        if not isinstance(other, BaseView):
            msg = f"Invalid view type: {type(other)}"
            raise TypeError(msg)
        if self.level != other.level:
            msg = (
                f"Cannot operate on views of different levels: "
                f"{self.level} and {other.level}"
            )
            raise TypeError(msg)

    def __getattr__(self, key: str) -> Feature:
        """Return the feature for the given key, cropped to the view's indices."""
        return self.features[key].crop(self._indices)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, key: Any) -> Self | LevelProtoT:  # noqa: ANN401
        """Return a new view with the specified indices."""
        return self.__class__(self._mol, self._indices[key])

    def __and__(self, other: Self) -> Self:
        self._check_same_level(other)
        raise NotImplementedError

    def __or__(self, other: Self) -> Self:
        return self.__add__(other)

    def __xor__(self, other: Self) -> Self:
        self._check_same_level(other)
        raise NotImplementedError

    def __add__(self, other: Self) -> Self:
        self._check_same_level(other)
        raise NotImplementedError

    def __sub__(self, other: Self) -> Self:
        self._check_same_level(other)
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError

    def __iter__(self) -> Self:
        raise NotImplementedError


class AtomView(BaseView[AtomProtoT, ResidueProtoT, ChainProtoT, AtomProtoT]):
    """View of the atoms in the selection."""

    _level: Final = StructureLevel.ATOM

    @property
    @override
    def atoms(self) -> Self | AtomProtoT:
        return self

    @property
    @override
    def residues(
        self,
    ) -> ResidueView[AtomProtoT, ResidueProtoT, ChainProtoT] | ResidueProtoT:
        raise NotImplementedError

    @property
    @override
    def chains(self) -> ChainView[AtomProtoT, ResidueProtoT, ChainProtoT] | ChainProtoT:
        raise NotImplementedError


class ResidueView(BaseView[AtomProtoT, ResidueProtoT, ChainProtoT, ResidueProtoT]):
    """View of the residues in the selection."""

    _level: Final = StructureLevel.RESIDUE

    @property
    @override
    def atoms(self) -> AtomView[AtomProtoT, ResidueProtoT, ChainProtoT] | AtomProtoT:
        raise NotImplementedError

    @property
    @override
    def residues(self) -> Self | ResidueProtoT:
        return self

    @property
    @override
    def chains(self) -> ChainView[AtomProtoT, ResidueProtoT, ChainProtoT] | ChainProtoT:
        raise NotImplementedError


class ChainView(BaseView[AtomProtoT, ResidueProtoT, ChainProtoT, ChainProtoT]):
    """View of the chains in the selection."""

    _level: Final = StructureLevel.CHAIN

    @property
    @override
    def atoms(self) -> AtomView[AtomProtoT, ResidueProtoT, ChainProtoT] | AtomProtoT:
        raise NotImplementedError

    @property
    @override
    def residues(
        self,
    ) -> ResidueView[AtomProtoT, ResidueProtoT, ChainProtoT] | ResidueProtoT:
        raise NotImplementedError

    @property
    @override
    def chains(self) -> ChainView[AtomProtoT, ResidueProtoT, ChainProtoT] | ChainProtoT:
        return self
