from typing import Generic, get_args, get_origin

import numpy as np

from .container import AtomContainer, ChainContainer, FeatureContainer, ResidueContainer
from .exceptions import (
    FeatureKeyError,
    IndexMismatchError,
    StructureLevelError,
    ViewProtocolError,
)
from .index import IndexTable
from .types import StructureLevel
from .view import A_co, AtomView, C_co, ChainView, R_co, ResidueView, ViewProtocol


class BioMol(Generic[A_co, R_co, C_co]):
    """A class representing a biomolecular structure."""

    def __init__(
        self,
        atom_container: AtomContainer,
        residue_container: ResidueContainer,
        chain_container: ChainContainer,
        index_table: IndexTable,
    ) -> None:
        self._atom_container = atom_container
        self._residue_container = residue_container
        self._chain_container = chain_container
        self._index = index_table
        self._check_protocol_type()
        self._check_lengths()

    @property
    def atoms(self) -> A_co:
        """View of the atoms in the selection."""
        return AtomView(self, np.arange(len(self._atom_container)))

    @property
    def residues(self) -> R_co:
        """View of the residues in the selection."""
        return ResidueView(self, np.arange(len(self._residue_container)))

    @property
    def chains(self) -> C_co:
        """View of the chains in the selection."""
        return ChainView(self, np.arange(len(self._chain_container)))

    @property
    def index_table(self) -> IndexTable:
        """The index table mapping atoms, residues, and chains."""
        return self._index

    def get_container(self, level: StructureLevel) -> FeatureContainer:
        """Get the feature container for a specific structure level.

        Parameters
        ----------
        level: StructureLevel
            The structure level for which to get the feature container.

        Returns
        -------
        FeatureContainer
            The feature container for the specified structure level.
        """
        match level:
            case StructureLevel.ATOM:
                return self._atom_container
            case StructureLevel.RESIDUE:
                return self._residue_container
            case StructureLevel.CHAIN:
                return self._chain_container
            case _:
                msg = f"Invalid structure level: {level}."
                raise StructureLevelError(msg)

    def _check_protocol_type(self) -> None:
        """Check if the view types satisfy the specified Protocols."""
        _orig_class = [
            c for c in self.__class__.__orig_bases__ if get_origin(c) is BioMol
        ]
        if len(_orig_class) == 0:
            return

        args = get_args(_orig_class[0])
        for i, proto in enumerate(args):
            bases = getattr(proto, "__mro__", ()) or getattr(proto, "__bases__", ())
            if not any(b is ViewProtocol for b in bases):
                msg = f"{proto.__name__} must inherit from ViewProtocol."
                raise ViewProtocolError(msg)
            if not getattr(proto, "_is_protocol", False):
                msg = f"{proto.__name__} is not a Protocol."
                raise ViewProtocolError(msg)

            name = ["atoms", "residues", "chains"][i]
            view = getattr(self, name)
            try:
                _is_satisfied = isinstance(view, proto)
            except TypeError as e:
                msg = (
                    f"{proto.__name__} must be marked with @runtime_checkable "
                    "to allow runtime Protocol checks."
                )
                raise ViewProtocolError(msg) from e
            except FeatureKeyError as e:
                msg = f"Feature key '{e.args[0]}' not found on {name} view."
                raise ViewProtocolError(msg) from e
            if not _is_satisfied:
                msg = f"{name} view must satisfy {proto.__name__}."
                raise ViewProtocolError(msg)

    def _check_lengths(self) -> None:
        """Check if the lengths of the containers and index table are consistent."""
        if len(self._atom_container) != len(self._index.atom_to_res):
            msg = (
                "Atom length mismatch: "
                f"atom_container has length {len(self._atom_container)}, "
                f"but index table has length {len(self._index.atom_to_res)}."
            )
            raise IndexMismatchError(msg)

        if len(self._residue_container) != len(self._index.res_to_chain):
            msg = (
                "Residue length mismatch: "
                f"residue_container has length {len(self._residue_container)}, "
                f"but index table has length {len(self._index.res_to_chain)}."
            )
            raise IndexMismatchError(msg)

        if len(self._chain_container) != len(self._index.chain_res_indptr) - 1:
            msg = (
                "Chain length mismatch: "
                f"chain_container has length {len(self._chain_container)}, "
                f"but index table has length {len(self._index.chain_res_indptr) - 1}."
            )
            raise IndexMismatchError(msg)

    def __repr__(self) -> str:
        """Return a string representation of the BioMol object."""
        return (
            f"<{self.__class__.__name__} with {len(self._atom_container)} atoms, "
            f"{len(self._residue_container)} residues, "
            f"and {len(self._chain_container)} chains>"
        )
