from typing import Any, get_args, get_origin

import numpy as np
from typing_extensions import override

from .container import AtomContainer, ChainContainer, FeatureContainer, ResidueContainer
from .exceptions import FeatureKeyError, StructureLevelError, ViewProtocolError
from .types import AtomProtoT, ChainProtoT, ResidueProtoT, StructureLevel
from .view import AtomView, ChainView, ResidueView, ViewLike


class BioMol(ViewLike[AtomProtoT, ResidueProtoT, ChainProtoT, Any]):
    """A class representing a biomolecular structure."""

    def __init__(
        self,
        atom_container: AtomContainer,
        residue_container: ResidueContainer,
        chain_container: ChainContainer,
    ) -> None:
        self._atom_container = atom_container
        self._residue_container = residue_container
        self._chain_container = chain_container
        self._check_protocol_type()

    @property
    @override
    def atoms(self) -> AtomView[AtomProtoT, ResidueProtoT, ChainProtoT] | AtomProtoT:
        return AtomView(self, np.arange(len(self._atom_container)))

    @property
    @override
    def residues(
        self,
    ) -> ResidueView[AtomProtoT, ResidueProtoT, ChainProtoT] | ResidueProtoT:
        return ResidueView(self, np.arange(len(self._residue_container)))

    @property
    @override
    def chains(self) -> ChainView[AtomProtoT, ResidueProtoT, ChainProtoT] | ChainProtoT:
        return ChainView(self, np.arange(len(self._chain_container)))

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
        if level == StructureLevel.ATOM:
            return self._atom_container
        if level == StructureLevel.RESIDUE:
            return self._residue_container
        if level == StructureLevel.CHAIN:
            return self._chain_container
        msg = f"Invalid structure level: {level}"
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
            if proto is type(None):
                continue
            if not getattr(proto, "_is_protocol", False):
                msg = f"{proto.__name__} is not a Protocol."
                raise TypeError(msg)

            name = ["atoms", "residues", "chains"][i]
            view = getattr(self, name)
            try:
                _is_satisfied = isinstance(view, proto)
            except TypeError as e:
                msg = (
                    f"{proto.__name__} must be marked with @runtime_checkable "
                    "to allow runtime Protocol checks."
                )
                raise TypeError(msg) from e
            except FeatureKeyError as e:
                msg = f"Feature key '{e.args[0]}' not found on {name} view."
                raise ViewProtocolError(msg) from e
            if not _is_satisfied:
                msg = f"{name} view must satisfy {proto.__name__}."
                raise ViewProtocolError(msg)
