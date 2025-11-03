from collections.abc import Callable
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from biomol.cif.mol import CIFMol
from biomol.core.container import FeatureContainer
from biomol.core.feature import EdgeFeature

InputType = TypeVar("InputType", str, int, float)
FeatureType = TypeVar("FeatureType")
NumericType = TypeVar("NumericType", int, float)


def single_value_instruction(
    *,
    dtype: type[InputType],
) -> Callable[..., type[InputType]]:
    """
    Return a configured instruction function that maps fields to node features.

    The returned function 'remembers' the dtype via closure.
    """

    def _worker(
        data: list[InputType] | NDArray,
    ) -> type[InputType]:
        formatted_data = [dtype(datum) for datum in data]
        if len(formatted_data) != 1:
            msg = f"Expected single value, got {len(formatted_data)}"
            raise ValueError(msg)
        return formatted_data[0]

    return _worker


def neighbor_list_grid(  # noqa: C901, PLR0912, PLR0915
    xyz: np.ndarray,
    d_thr: float,
    n_max: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute (n_atom, n_max) neighbor indices using a uniform grid of cell size d_thr.

    Vectorized over atoms; only a tiny fixed loop over 27 neighbor-cell offsets.

    Parameters
    ----------
    xyz : (n_atom, 3) float32/64
        Coordinates; any row containing NaN is ignored.
    d_thr : float
        L2 distance threshold for neighbor definition.
    n_max : int
        Maximum number of neighbors to keep per atom.

    Returns
    -------
    nbrs : (n_atom, n_max) int64
        Neighbor indices, padded with -1.
    counts : (n_atom,) int32
        Actual neighbor counts for each atom (self excluded).
    """
    n_atom = xyz.shape[0]
    nbrs = np.full((n_atom, n_max), -1, dtype=np.int64)
    counts = np.zeros(n_atom, dtype=np.int32)

    # Mask invalid atoms (any NaN)
    valid = np.all(np.isfinite(xyz), axis=1)
    if not np.any(valid):
        return nbrs, counts

    # Compressed point array (n_valid,3)
    valid_xyz = xyz[valid]
    n_valid = valid_xyz.shape[0]

    # Discretize into cells of side length d_thr
    cell = np.floor(valid_xyz / d_thr).astype(np.int64)  # (n_valid,3)

    # Group points by cell via lexicographic sort
    order = np.lexsort((cell[:, 2], cell[:, 1], cell[:, 0]))
    cell_sorted = cell[order]

    # Identify unique cells and their [start, end) spans in the sorted arrays
    if n_valid > 1:
        change = np.any(np.diff(cell_sorted, axis=0) != 0, axis=1)
        starts = np.concatenate(([0], np.nonzero(change)[0] + 1))
    else:
        starts = np.array([0], dtype=np.int64)
    ends = np.concatenate((starts[1:], [n_valid]))
    unique_cells = cell_sorted[starts]  # (n_unique,3)
    n_unique = unique_cells.shape[0]

    # Pack 3 int64s into a byte blob so we can binary-search with searchsorted
    def pack_cells(arr_i64x3: np.ndarray) -> np.ndarray:
        return arr_i64x3.view(np.dtype((np.void, arr_i64x3.dtype.itemsize * 3))).ravel()

    packed_unique = pack_cells(unique_cells)

    # Cell id for each point in original compressed order
    cell_id_sorted = np.empty(n_valid, dtype=np.int64)
    for cid, (s, e) in enumerate(zip(starts, ends, strict=True)):
        cell_id_sorted[s:e] = cid
    inv_order = np.empty(n_valid, dtype=np.int64)
    inv_order[order] = np.arange(n_valid)

    # 27 neighbor-cell offsets (-1,0,1)^3
    offsets = (
        np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij"))
        .reshape(3, -1)
        .T
    )  # (27,3)

    # Accumulate candidate (i,j) pairs in compressed indices
    pair_i = []
    pair_j = []

    # For each offset, find matching neighbor cells and generate Cartesian pairs
    for off in offsets:
        nei_cells = unique_cells + off  # (n_unique,3)
        packed_nei = pack_cells(nei_cells)

        # searchsorted may return n_unique (out of bounds); guard before indexing
        pos = np.searchsorted(packed_unique, packed_nei, side="left")
        in_bounds = pos < n_unique

        ok = np.zeros_like(in_bounds, dtype=bool)
        if np.any(in_bounds):
            ok[in_bounds] = packed_unique[pos[in_bounds]] == packed_nei[in_bounds]

        if not np.any(ok):
            continue

        src_c = np.nonzero(ok)[0]  # source cell ids
        dst_c = pos[ok]  # matching neighbor cell ids

        # Build ranges for each cell span
        src_ranges = [np.arange(starts[c], ends[c], dtype=np.int64) for c in src_c]
        dst_ranges = [np.arange(starts[c], ends[c], dtype=np.int64) for c in dst_c]

        # Cartesian product per (src, dst) cell pair (vectorized at cell level)
        if len(src_ranges) == 0:
            continue
        src_idx = np.concatenate(
            [
                np.repeat(r, len(d))
                for r, d in zip(src_ranges, dst_ranges, strict=False)
            ],
        )  # sorted space
        dst_idx = np.concatenate(
            [np.tile(d, len(r)) for r, d in zip(src_ranges, dst_ranges, strict=False)],
        )

        if src_idx.size == 0:
            continue

        # Map back to compressed order (unsorted)
        src_idx = order[src_idx]
        dst_idx = order[dst_idx]

        # Drop self-pairs
        keep = src_idx != dst_idx
        if not np.any(keep):
            continue

        pair_i.append(src_idx[keep])
        pair_j.append(dst_idx[keep])

    if not pair_i:
        # No candidate pairs; return empty neighbor lists
        return nbrs, counts

    pair_i = np.concatenate(pair_i)
    pair_j = np.concatenate(pair_j)

    # Distance filtering: keep pairs with ||valid_xyz[i]-valid_xyz[j]|| <= d_thr
    dvec = valid_xyz[pair_i] - valid_xyz[pair_j]
    dist2 = np.einsum("ij,ij->i", dvec, dvec)
    keep = dist2 <= (d_thr * d_thr)
    if not np.any(keep):
        return nbrs, counts

    pair_i = pair_i[keep]
    pair_j = pair_j[keep]

    # Remove duplicate pairs
    # (the same (i,j) can appear via multiple neighbor-cell offsets)
    ij = np.stack([pair_i, pair_j], axis=1).astype(np.int64)
    ij_packed = ij.view(np.dtype((np.void, ij.dtype.itemsize * 2))).ravel()
    uniq_idx = np.unique(ij_packed, return_index=True)[1]
    ij = ij[uniq_idx]
    pair_i, pair_j = ij[:, 0], ij[:, 1]

    # Map compressed indices back to original n_atom-space
    valid_true_idx = np.flatnonzero(valid)
    gi = valid_true_idx[pair_i]
    gj = valid_true_idx[pair_j]

    # Fill neighbor matrix: group by gi, keep up to n_max in order of (gi, gj)
    order_fill = np.lexsort((gj, gi))
    gi = gi[order_fill]
    gj = gj[order_fill]

    uniq_i, first_pos, counts_all = np.unique(gi, return_index=True, return_counts=True)
    take_counts = np.minimum(counts_all, n_max)

    if uniq_i.size > 0:
        gather_idx = np.concatenate(
            [
                np.arange(s, s + t, dtype=np.int64)
                for s, t in zip(first_pos, take_counts, strict=True)
            ],
        )
        gi_take = gi[gather_idx]
        gj_take = gj[gather_idx]

        # Relative slot [0..taken-1] within each group
        rel = np.arange(gj.size, dtype=np.int64) - np.repeat(first_pos, counts_all)
        rel = rel[gather_idx]

        nbrs[gi_take, rel] = gj_take
        counts[uniq_i] = take_counts.astype(np.int32)

    return nbrs, counts


def extract_contact_graph(
    d_thr: float = 6.0,
    n_max: int = 128,
) -> Callable[[CIFMol], FeatureContainer]:
    """Return a configured instruction function that extracts contact chains.

    Edge values will be the number of atom-atom contacts (pairs) between chain pairs.
    """

    def _worker(cifmol: CIFMol) -> FeatureContainer:
        # 1) Coordinates and per-atom chain indices
        xyz = cifmol.atoms.xyz.value  # (L, 3)
        chain_idx = cifmol.index_table.atoms_to_chains(np.arange(xyz.shape[0]))  # (L,)

        # 2) Neighborhood via grid cells
        nbrs, counts = neighbor_list_grid(
            xyz,
            d_thr,
            n_max,
        )  # nbrs: (L, N_max), -1 padded

        # 3) Build (chain_i, chain_j) for every valid inter-chain atom pair
        #    - broadcast chain_i over neighbor slots
        chain_idx_i_matrix = np.broadcast_to(chain_idx[:, np.newaxis], nbrs.shape)

        #    - map neighbor indices (including -1) to chain_j using padding trick
        padded_chain_idx = np.append(chain_idx, -1)
        chain_idx_j_matrix = padded_chain_idx[nbrs]

        #    - keep only valid neighbors and inter-chain pairs
        valid_neighbor_mask = nbrs != -1
        inter_chain_mask = chain_idx_i_matrix != chain_idx_j_matrix
        final_mask = valid_neighbor_mask & inter_chain_mask

        if not np.any(final_mask):
            # No contacts at all: return empty edge set
            contact_nodes = cifmol.chains.chain_id
            contact_edges = EdgeFeature(
                value=np.empty((0,), dtype=np.int32),
                src_indices=np.empty((0,), dtype=chain_idx.dtype),
                dst_indices=np.empty((0,), dtype=chain_idx.dtype),
            )
            return FeatureContainer(
                nodes=contact_nodes,
                edges={"contact": contact_edges},
            )

        # 4) Collect chain pair list for all contacting atom pairs
        chain_pairs_i = chain_idx_i_matrix[final_mask]
        chain_pairs_j = chain_idx_j_matrix[final_mask]

        # 5) Make edges undirected by sorting endpoints within each pair
        chain_pairs = np.stack([chain_pairs_i, chain_pairs_j], axis=1)  # (E_atom, 2)
        sorted_edges = np.sort(chain_pairs, axis=1)  # ensure (min, max)

        # 6) Count how many atom-atom contacts per chain-pair
        #    Use unique by rows with counts
        contact_edges_unique, counts_per_edge = np.unique(
            sorted_edges,
            axis=0,
            return_counts=True,
        )  # contact_edges_unique: (E_chain, 2), counts_per_edge: (E_chain,)

        # 7) Build EdgeFeature with counts as values
        contact_src = contact_edges_unique[:, 0]
        contact_dst = contact_edges_unique[:, 1]

        contact_nodes = cifmol.chains.chain_id
        contact_edges = EdgeFeature(
            value=counts_per_edge.astype(np.int32),  # number of atom-atom contacts
            src_indices=contact_src,
            dst_indices=contact_dst,
        )

        return FeatureContainer(
            nodes=contact_nodes,
            edges={"contact": contact_edges},
        )

    return _worker
