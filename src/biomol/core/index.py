from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from biomol.enums import StructureLevel
from biomol.exceptions import (
    IndexInvalidError,
    IndexOutOfBoundsError,
    StructureLevelError,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _build_csr(
    parent_of_child: NDArray[np.integer],
    n_parent: int,
) -> tuple[NDArray[np.integer], NDArray[np.integer]]:
    counts = np.bincount(parent_of_child, minlength=n_parent)
    indptr = np.empty(n_parent + 1, dtype=int)
    indptr[0] = 0
    np.cumsum(counts, dtype=int, out=indptr[1:])
    indices = np.empty_like(parent_of_child)
    offsets = indptr[:-1].copy()
    for child_idx, parent_idx in enumerate(parent_of_child):
        pos = offsets[parent_idx]
        indices[pos] = child_idx
        offsets[parent_idx] += 1
    return indptr, indices


@dataclass(frozen=True, slots=True)
class IndexTable:
    """Index mapping between structural levels.

    This class stores forward parent mappings and reverse CSR mappings to
    efficiently move between atoms, residues, and chains.

    Parameters
    ----------
    atom_to_res : NDArray[np.integer]
        1D array mapping each atom index to its parent residue index.
    res_to_chain : NDArray[np.integer]
        1D array mapping each residue index to its parent chain index.
    res_atom_indptr : NDArray[np.integer]
        CSR index pointer array for residues to atoms mapping.
    res_atom_indices : NDArray[np.integer]
        CSR indices array for residues to atoms mapping.
    chain_res_indptr : NDArray[np.integer]
        CSR index pointer array for chains to residues mapping.
    chain_res_indices : NDArray[np.integer]
        CSR indices array for chains to residues mapping.

    Examples
    --------
    .. code-block:: python

        >>> table = IndexTable.from_parents(
        ...     atom_to_res=np.array([0, 0, 1, 1, 2]),
        ...     res_to_chain=np.array([0, 0, 1]),
        ... )

        >>> table.atoms_to_residues(np.array([0, 2, 4]))
        array([0, 1, 2])

        >>> table.residues_to_chains(np.array([0, 2]))
        array([0, 1])

        >>> table.chains_to_residues(np.array([0, 1]))
        array([0, 1, 2])
    """

    atom_to_res: NDArray[np.integer]
    """1D array mapping each atom index to its parent residue index."""

    res_to_chain: NDArray[np.integer]
    """1D array mapping each residue index to its parent chain index."""

    res_atom_indptr: NDArray[np.integer]
    """CSR index pointer array for residues to atoms mapping."""

    res_atom_indices: NDArray[np.integer]
    """CSR indices array for residues to atoms mapping."""

    chain_res_indptr: NDArray[np.integer]
    """CSR index pointer array for chains to residues mapping."""

    chain_res_indices: NDArray[np.integer]
    """CSR indices array for chains to residues mapping."""

    _converter_table: ClassVar[dict[tuple[StructureLevel, StructureLevel], str]] = {
        (StructureLevel.ATOM, StructureLevel.RESIDUE): "atoms_to_residues",
        (StructureLevel.ATOM, StructureLevel.CHAIN): "atoms_to_chains",
        (StructureLevel.RESIDUE, StructureLevel.ATOM): "residues_to_atoms",
        (StructureLevel.RESIDUE, StructureLevel.CHAIN): "residues_to_chains",
        (StructureLevel.CHAIN, StructureLevel.RESIDUE): "chains_to_residues",
        (StructureLevel.CHAIN, StructureLevel.ATOM): "chains_to_atoms",
    }

    @classmethod
    def from_parents(
        cls,
        atom_to_res: NDArray[np.integer],
        res_to_chain: NDArray[np.integer],
        n_chain: int | None = None,
    ) -> IndexTable:
        """Create IndexTable from forward parent mappings.

        Parameters
        ----------
        atom_to_res : NDArray[np.integer]
            1D array mapping each atom index to its parent residue index.
        res_to_chain : NDArray[np.integer]
            1D array mapping each residue index to its parent chain index.
        n_chain : int | None, optional
            Total number of chains. If None, inferred as max(res_to_chain) + 1.
        """
        cls._check_indices(atom_to_res)
        cls._check_indices(res_to_chain)

        n_residue = len(res_to_chain)
        if atom_to_res.max() >= n_residue:
            msg = (
                f"atom_to_res has out-of-range values={atom_to_res.max()} "
                f"for {n_residue=}"
            )
            raise IndexOutOfBoundsError(msg)

        if n_chain is None:
            n_chain = int(res_to_chain.max()) + 1
        elif n_chain <= 0:
            msg = f"n_chain must be positive, got {n_chain}"
            raise IndexOutOfBoundsError(msg)
        elif res_to_chain.max() >= n_chain:
            msg = (
                f"res_to_chains has out-of-range values={res_to_chain.max()} "
                f"for {n_chain=}"
            )
            raise IndexOutOfBoundsError(msg)

        res_atom_indptr, res_atom_indices = _build_csr(atom_to_res, n_residue)
        chain_res_indptr, chain_res_indices = _build_csr(res_to_chain, n_chain)
        return cls(
            atom_to_res=atom_to_res,
            res_to_chain=res_to_chain,
            res_atom_indptr=res_atom_indptr,
            res_atom_indices=res_atom_indices,
            chain_res_indptr=chain_res_indptr,
            chain_res_indices=chain_res_indices,
        )

    def atoms_to_residues(self, indices: NDArray[np.integer]) -> NDArray[np.integer]:
        """Map atom indices to residue indices."""
        return self.atom_to_res[np.asarray(indices)]

    def residues_to_chains(self, indices: NDArray[np.integer]) -> NDArray[np.integer]:
        """Map residue indices to chain indices."""
        return self.res_to_chain[np.asarray(indices)]

    def atoms_to_chains(self, indices: NDArray[np.integer]) -> NDArray[np.integer]:
        """Map atom indices to chain indices."""
        res_indices = self.atoms_to_residues(indices)
        return self.residues_to_chains(res_indices)

    def residues_to_atoms(self, indices: NDArray[np.integer]) -> NDArray[np.integer]:
        """Map residue indices to concatenated atom indices."""
        indices = np.asarray(indices)
        if indices.size == 0:
            return indices
        parts = [
            self.res_atom_indices[
                self.res_atom_indptr[idx] : self.res_atom_indptr[idx + 1]
            ]
            for idx in indices
        ]
        if not parts:
            return np.empty((0,), dtype=self.res_atom_indices.dtype)
        return np.concatenate(parts)

    def chains_to_residues(self, indices: NDArray[np.integer]) -> NDArray[np.integer]:
        """Map chain indices to concatenated residue indices."""
        indices = np.asarray(indices)
        if indices.size == 0:
            return indices.astype(int)
        parts = [
            self.chain_res_indices[
                self.chain_res_indptr[idx] : self.chain_res_indptr[idx + 1]
            ]
            for idx in indices
        ]
        if not parts:
            return np.empty((0,), dtype=self.chain_res_indices.dtype)
        return np.concatenate(parts)

    def chains_to_atoms(self, indices: NDArray[np.integer]) -> NDArray[np.integer]:
        """Map chain indices to concatenated atom indices."""
        res_indices = self.chains_to_residues(indices)
        return self.residues_to_atoms(res_indices)

    def convert(
        self,
        indices: NDArray[np.integer],
        source: StructureLevel,
        target: StructureLevel,
    ) -> NDArray[np.integer]:
        """Convert indices between structural levels.

        Parameters
        ----------
        indices : NDArray[np.integer]
            1D array of indices at the source level.
        source : StructureLevel
            The structural level of the input indices.
        target : StructureLevel
            The structural level to convert the indices to.

        Returns
        -------
        NDArray[np.integer]
            1D array of indices at the target level.
        """
        if source == target:
            return indices

        if (source, target) not in self._converter_table:
            msg = f"Invalid level conversion: {source} -> {target}"
            raise StructureLevelError(msg)
        method_name = self._converter_table[(source, target)]
        return getattr(self, method_name)(indices)

    @staticmethod
    def _check_indices(indices: NDArray[np.integer]) -> None:
        if indices.ndim != 1:
            msg = f"Indices must be 1D, got shape {indices.shape}"
            raise IndexInvalidError(msg)
        if indices.size == 0:
            msg = "Indices must be non-empty"
            raise IndexInvalidError(msg)
        if np.any(indices < 0):
            msg = "Indices contain negative values"
            raise IndexInvalidError(msg)

    def copy(self) -> IndexTable:
        """Create a deep copy of the IndexTable."""
        return IndexTable(
            self.atom_to_res.copy(),
            self.res_to_chain.copy(),
            self.res_atom_indptr.copy(),
            self.res_atom_indices.copy(),
            self.chain_res_indptr.copy(),
            self.chain_res_indices.copy(),
        )
