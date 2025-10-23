# pyright: reportImportCycles=none

import json
from collections.abc import Mapping
from dataclasses import asdict
from io import BytesIO
from typing import Any, Generic

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self
from zstandard import ZstdCompressor, ZstdDecompressor

from biomol.enums import StructureLevel
from biomol.exceptions import IndexMismatchError, StructureLevelError

from .container import FeatureContainer
from .feature import Feature
from .index import IndexTable
from .types import BioMolDict
from .view import A_co, AtomView, C_co, ChainView, R_co, ResidueView


class BioMol(Generic[A_co, R_co, C_co]):
    """A class representing a biomolecular structure.

    Parameters
    ----------
    atom_container: FeatureContainer
        The container holding atom-level features.
    residue_container: FeatureContainer
        The container holding residue-level features.
    chain_container: FeatureContainer
        The container holding chain-level features.
    index_table: IndexTable
        The index table mapping atoms, residues, and chains.
    metadata: dict[str, Any] | None, optional
        Additional metadata associated with the biomolecular structure.
    """

    def __init__(
        self,
        atom_container: FeatureContainer,
        residue_container: FeatureContainer,
        chain_container: FeatureContainer,
        index_table: IndexTable,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._atom_container = atom_container
        self._residue_container = residue_container
        self._chain_container = chain_container
        self._index = index_table
        self._metadata = metadata or {}
        self._check_lengths()

    @property
    def atoms(self) -> A_co:
        """View of the atoms in the selection."""
        return AtomView(self, np.arange(len(self._atom_container)))  # pyright: ignore[reportReturnType]

    @property
    def residues(self) -> R_co:
        """View of the residues in the selection."""
        return ResidueView(self, np.arange(len(self._residue_container)))  # pyright: ignore[reportReturnType]

    @property
    def chains(self) -> C_co:
        """View of the chains in the selection."""
        return ChainView(self, np.arange(len(self._chain_container)))  # pyright: ignore[reportReturnType]

    @property
    def index_table(self) -> IndexTable:
        """The index table mapping atoms, residues, and chains."""
        return self._index

    @property
    def metadata(self) -> dict[str, Any]:
        """The metadata associated with the biomolecular structure."""
        return self._metadata

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

    def to_dict(self) -> BioMolDict:
        """Convert the BioMol object to a dictionary."""
        return {
            "atoms": self._atom_container.to_dict(),
            "residues": self._residue_container.to_dict(),
            "chains": self._chain_container.to_dict(),
            "index_table": asdict(self._index),  # pyright: ignore[reportReturnType]
            "metadata": self._metadata,
        }

    @classmethod
    def from_dict(cls, data: BioMolDict) -> Self:
        """Create a BioMol object from a dictionary.

        Parameters
        ----------
        data: BioMolDict
            A dictionary containing the data to create the BioMol object.

        Returns
        -------
        BioMol
            The created BioMol object.
        """
        return cls(
            FeatureContainer.from_dict(data["atoms"]),
            FeatureContainer.from_dict(data["residues"]),
            FeatureContainer.from_dict(data["chains"]),
            IndexTable(**data["index_table"]),
            data["metadata"],
        )

    def to_bytes(self, level: int = 6) -> bytes:
        """Serialize the container to zstd-compressed bytes.

        Parameters
        ----------
        level: int, optional
            The compression level for zstd (default is 6).
        """

        def _flatten_data(
            data: Mapping[str, Any],
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            template = {}
            flatten = {}
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    _key = str(id(value))
                    template[key] = _key
                    buffer = BytesIO()
                    np.save(buffer, np.ascontiguousarray(value), allow_pickle=False)
                    flatten[_key] = buffer.getvalue()
                elif isinstance(value, dict):
                    _template, _flatten = _flatten_data(value)
                    template[key] = _template
                    flatten.update(_flatten)
                else:
                    template[key] = value
            return template, flatten

        data = self.to_dict()
        template, flatten_data = _flatten_data(data)
        header = {
            "class": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "template": template,
            "arrays": {key: len(value) for key, value in flatten_data.items()},
        }
        header_bytes = json.dumps(header).encode("utf-8")
        payload = b"".join(flatten_data[key] for key in flatten_data)
        raw = len(header_bytes).to_bytes(8, "little") + header_bytes + payload
        return ZstdCompressor(level=level).compress(raw)

    @classmethod
    def from_bytes(cls, byte_data: bytes) -> Self:
        """Deserialize the container from zstd-compressed bytes."""

        def _reconstruct_data(
            template: dict[str, Any],
            flatten: dict[str, Any],
        ) -> dict[str, Any]:
            data = {}
            for key, value in template.items():
                if isinstance(value, str) and value in flatten:
                    buffer = BytesIO(flatten[value])
                    buffer.seek(0)
                    arr = np.load(buffer, allow_pickle=False)
                    data[key] = arr
                elif isinstance(value, dict):
                    data[key] = _reconstruct_data(value, flatten)
                else:
                    data[key] = value
            return data

        raw = ZstdDecompressor().decompress(byte_data)
        hlen = int.from_bytes(raw[:8], "little")
        header = json.loads(raw[8 : 8 + hlen].decode("utf-8"))
        payload = raw[8 + hlen :]

        offset = 0
        flatten_data = {}
        for key, ln in header["arrays"].items():
            chunk = payload[offset : offset + ln]
            offset += ln
            flatten_data[key] = chunk

        template_dict = header["template"]
        data = _reconstruct_data(template_dict, flatten_data)
        return cls.from_dict(data)  # pyright: ignore[reportArgumentType]

    def update_features(
        self,
        level: StructureLevel,
        **features: Feature | NDArray[Any],
    ) -> Self:
        """Update features at a specific structure level.

        Parameters
        ----------
        level: StructureLevel
            The structure level at which to update features.
        **features: Feature | NDArray[Any]
            Key-value pairs of features to update. Values can be either Feature
            objects or numpy arrays (which will be converted to NodeFeature).

        Returns
        -------
        mol
            Updated BioMol object.

        Notes
        -----
        Does not modify the current BioMol instance; instead, returns a new one.

        Examples
        --------
        .. code-block:: python

            mol = BioMol(...)
            new_mol = mol.update_features(
                StructureLevel.ATOM,
                coord=mol.atoms.coord + 1.0,
            )

        """
        containers = {
            StructureLevel.ATOM: self._atom_container,
            StructureLevel.RESIDUE: self._residue_container,
            StructureLevel.CHAIN: self._chain_container,
        }
        containers[level] = containers[level].update(**features)
        return self.__class__(
            containers[StructureLevel.ATOM],
            containers[StructureLevel.RESIDUE],
            containers[StructureLevel.CHAIN],
            self.index_table,
            self.metadata,
        )

    def remove_features(self, level: StructureLevel, *keys: str) -> Self:
        """Remove features at a specific structure level.

        Parameters
        ----------
        level: StructureLevel
            The structure level at which to remove features.
        *keys: str
            Keys of the features to remove.

        Returns
        -------
        mol
            Updated BioMol object.

        Notes
        -----
        Does not modify the current BioMol instance; instead, returns a new one.

        Examples
        --------
        .. code-block:: python

            mol = BioMol(...)
            new_mol = mol.remove_features(StructureLevel.ATOM, "coord", "element")

        """
        containers = {
            StructureLevel.ATOM: self._atom_container,
            StructureLevel.RESIDUE: self._residue_container,
            StructureLevel.CHAIN: self._chain_container,
        }
        containers[level] = containers[level].remove(*keys)
        return self.__class__(
            containers[StructureLevel.ATOM],
            containers[StructureLevel.RESIDUE],
            containers[StructureLevel.CHAIN],
            self.index_table,
            self.metadata,
        )

    @classmethod
    def concat(cls, mols: list[Self]) -> Self:
        """Concatenate multiple BioMol objects.

        Parameters
        ----------
        mols: list[Self]
            List of BioMol objects to concatenate.

        Returns
        -------
        Self
            Concatenated BioMol object.

        Notes
        -----
        All containers must have the same set of feature keys.
        Metadata from the first BioMol object is retained.

        Examples
        --------
        .. code-block:: python

            mol1 = BioMol(...)
            mol2 = BioMol(...)
            concatenated_mol = BioMol.concat([mol1, mol2])

        """
        if not mols:
            msg = "Cannot concatenate an empty list of BioMol objects."
            raise ValueError(msg)
        if len(mols) == 1:
            return mols[0]

        atom_containers = [mol.get_container(StructureLevel.ATOM) for mol in mols]
        residue_containers = [mol.get_container(StructureLevel.RESIDUE) for mol in mols]
        chain_containers = [mol.get_container(StructureLevel.CHAIN) for mol in mols]

        residue_counts = [len(container) for container in residue_containers]
        residue_offsets = np.cumsum([0, *residue_counts[:-1]])
        atom_to_res = [
            mol.index_table.atom_to_res + offset
            for mol, offset in zip(mols, residue_offsets, strict=True)
        ]

        chain_counts = [len(container) for container in chain_containers]
        chain_offsets = np.cumsum([0, *chain_counts[:-1]])
        res_to_chain = [
            mol.index_table.res_to_chain + offset
            for mol, offset in zip(mols, chain_offsets, strict=True)
        ]

        concatenated_table = IndexTable.from_parents(
            atom_to_res=np.concatenate(atom_to_res, axis=0),
            res_to_chain=np.concatenate(res_to_chain, axis=0),
            n_chain=sum(chain_counts),
        )
        return cls(
            FeatureContainer.concat(atom_containers),
            FeatureContainer.concat(residue_containers),
            FeatureContainer.concat(chain_containers),
            concatenated_table,
            metadata=mols[0].metadata.copy(),
        )

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

    def __add__(self, other: Self) -> Self:
        """Concatenate two BioMol objects using the + operator.

        Note
        ----
        For concatenating more than two objects, use BioMol.concat([mol1, mol2, mol3])
        for better performance.
        """
        return self.concat([self, other])
