from collections import deque
from collections.abc import Callable, Iterable
from typing import TypeVar

from numpy.typing import NDArray

from biomol.cif.mol import CIFMol

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


def graph_to_canonical_sequence(
    seq_list: Iterable[str],
    src_indices: Iterable[int],
    dst_indices: Iterable[int],
    max_wl_iter: int = 100,
) -> str:
    """
    Canonical-ish serialization for simple, undirected, labeled graphs (no loops / no multi-edges).

    Steps:
      1) Build adjacency; initial colors = (label, degree)
      2) 1-WL refinement to stability
      3) Color-aware lex-BFS traversals from every node in the globally-smallest color class;
         pick the lexicographically smallest traversal *code* (node-color sequence, edge list in visit-order).
      4) Serialize nodes/edges using that canonical order.

    Deterministic and index-order-invariant. For exotic 1-WL-indistinguishable families, use a full canonical
    labeling (e.g., nauty/Traces) if theoretical guarantees are required.
    """
    # -------- 0) Normalize inputs (handle numpy arrays, generators, etc.) --------
    labels = list(seq_list)
    src = list(src_indices)
    dst = list(dst_indices)
    n = len(labels)
    if len(src) != len(dst):
        msg = "src_indices and dst_indices must have the same length"
        raise ValueError(msg)
    if n == 0:
        return "|"

    # -------- 1) Build simple undirected adjacency --------
    adj: list[list[int]] = [[] for _ in range(n)]
    edge_set: set[tuple[int, int]] = set()
    for s, d in zip(src, dst, strict=True):
        if s == d:
            continue  # no self-loops by assumption
        a, b = (s, d) if s < d else (d, s)
        if (a, b) in edge_set:
            continue  # no multi-edges by assumption
        edge_set.add((a, b))
        adj[a].append(b)
        adj[b].append(a)
    for i in range(n):
        adj[i].sort()
    degrees = [len(adj[i]) for i in range(n)]

    # -------- 2) 1-WL color refinement --------
    def remap_to_int(keys: list[tuple]) -> list[int]:
        table: dict[tuple, int] = {}
        out: list[int] = []
        for k in keys:
            if k not in table:
                table[k] = len(table)
            out.append(table[k])
        return out

    color: list[int] = remap_to_int([(labels[i], degrees[i]) for i in range(n)])

    for _ in range(max_wl_iter):
        sigs: list[tuple] = []
        for i in range(n):
            # neighbor colors, sorted by neighbor id (adj is sorted)
            neigh = tuple(color[j] for j in adj[i])
            sigs.append((color[i], neigh))
        new_color = remap_to_int(sigs)
        if new_color == color:
            break
        color = new_color

    # -------- 3) Color-aware lex-BFS; return (code, order) --------
    from collections import defaultdict as _dd

    by_color: dict[int, list[int]] = _dd(list)
    for i, c in enumerate(color):
        by_color[c].append(i)

    for c, nodes in by_color.items():
        nodes.sort()
        by_color[c] = nodes

    # pick smallest color class; if anything goes wrong, fall back to all nodes
    starts: list[int]
    if by_color:
        min_color = min(by_color.keys())
        starts = by_color[min_color]
    else:
        starts = list(range(n))

    # Safety: should never be empty, but guard anyway
    if not starts:
        starts = list(range(n))

    def traversal_code(
        start: int,
    ) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...], list[int]]:
        visited = [-1] * n
        order: list[int] = []
        q = deque([start])

        def push_component(seed: int) -> None:
            if visited[seed] != -1:
                return
            q.append(seed)
            while q:
                u = q.popleft()
                if visited[u] != -1:
                    continue
                visited[u] = len(order)
                order.append(u)
                # expand unvisited neighbors in (color, degree, id) order
                cand = [(color[v], degrees[v], v) for v in adj[u] if visited[v] == -1]
                cand.sort()
                for _, _, v in cand:
                    if visited[v] == -1:
                        q.append(v)

        # first component from 'start'
        push_component(start)

        # if disconnected, visit remaining components by (color, degree, id)
        if len(order) < n:
            remaining = [i for i in range(n) if visited[i] == -1]
            remaining.sort(key=lambda i: (color[i], degrees[i], i))
            for seed in remaining:
                push_component(seed)

        # build code
        node_code = tuple(color[u] for u in order)

        pos = {u: i for i, u in enumerate(order)}
        edges_in_order: list[tuple[int, int]] = []
        for a, b in edge_set:
            i, j = pos[a], pos[b]
            if i > j:
                i, j = j, i
            edges_in_order.append((i + 1, j + 1))  # 1-based for readability
        edges_in_order.sort()
        edge_code = tuple(edges_in_order)
        return node_code, edge_code, order

    best_code: tuple[tuple[int, ...], tuple[tuple[int, int], ...]] | None = None
    best_order: list[int] | None = None
    for s in starts:
        node_code, edge_code, order = traversal_code(s)
        code = (node_code, edge_code)
        if (best_code is None) or (code < best_code):
            best_code = code
            best_order = order

    # robust fallback (should not trigger)
    if best_order is None:
        # fall back to a purely structural order
        best_order = sorted(range(n), key=lambda i: (color[i], degrees[i], i))

    # -------- 4) Serialize using canonical order --------
    pos = {u: i for i, u in enumerate(best_order)}
    nodes_str = "".join(f"({labels[u]})" for u in best_order)

    edges_norm: list[tuple[int, int]] = []
    for a, b in edge_set:
        i, j = pos[a], pos[b]
        if i > j:
            i, j = j, i
        edges_norm.append((i + 1, j + 1))
    edges_norm.sort()
    edges_str = "".join(f"({i},{j})" for (i, j) in edges_norm)

    return f"{nodes_str}|{edges_str}"


def build_fasta() -> Callable[..., type[InputType]]:
    """
    Read CIFMol and build sequence string.

    Headers:
    >(PDB_ID)_(Chain_ID) | (Molecule_Type) | (if polymer, Polymer_Type)
    Sequence:
        - If polymer, use one-letter codes (canonical).
        - If non-polymer, use three-letter codes separated by spaces. Ex) (SO4)
        - If branched, use three-letter codes with edge information. (NAG)(NAG)  | (0, 1, 2)
    Skip residues with unknown types or water.
    """

    def _worker(
        cifmol: CIFMol,
    ) -> dict[str, str]:
        fasta_dict = {}
        chain_ids = cifmol.chains.chain_id.value
        for full_chain_id in chain_ids:
            chain_id = full_chain_id.split("_")[0]
            if chain_id in fasta_dict:
                continue  # skip duplicate chains (due to symmetry operators)
            entity_type = cifmol.chains[
                cifmol.chains.chain_id == full_chain_id
            ].entity_type.value[0]
            header = f">{cifmol.id[0]}_{chain_id} | {entity_type}"
            if entity_type == "non-polymer":
                seq = cifmol.chains[
                    cifmol.chains.chain_id == full_chain_id
                ].residues.chem_comp.value
                seq = f"({seq[0]})"
            elif entity_type == "branched":
                seq_list = cifmol.chains[
                    cifmol.chains.chain_id == full_chain_id
                ].residues.chem_comp.value
                bonds = cifmol.chains[
                    cifmol.chains.chain_id == full_chain_id
                ].residues.bond
                seq = graph_to_canonical_sequence(
                    seq_list,
                    bonds.src_indices,
                    bonds.dst_indices,
                )
            else:  # polymer
                seq = cifmol.chains[
                    cifmol.chains.chain_id == full_chain_id
                ].residues.one_letter_code_can.value
                seq = "".join(seq)

            fasta_dict[chain_id] = f"{header}\n{seq}\n"

        return fasta_dict

    return _worker


def build_seq_hash_map() -> Callable[..., type[InputType]]:
    """
    Build a sequence hash map from CIFMol sequences.

    The returned function 'remembers' nothing via closure.
    """

    def _worker(
        fasta_dict: dict[str, str],
    ) -> type[InputType]:
        seq_hash_map = {}
        seq_hash = 0

        for header, sequence in fasta_dict.items():
            cif_ID = header.split("|")[0].strip().split("_")[0]
            mol_type = header.split("|")[1].strip()
            match mol_type:
                case "polypeptide(L)":
                    mol_identifier = "P"
                case "polypeptide(D)":
                    mol_identifier = "Q"
                case "polydeoxyribonucleotide":
                    mol_identifier = "D"
                case "polyribonucleotide":
                    mol_identifier = "R"
                case "polydeoxyribonucleotide/polyribonucleotide hybrid":
                    mol_identifier = "N"
                case "branched":
                    mol_identifier = "B"
                case "non-polymer":
                    mol_identifier = "L"
                case _:
                    mol_identifier = "X"
            key = f"{mol_identifier}{sequence}"
            if key in seq_hash_map:
                continue  # skip duplicate sequences
            _seq_hash = f"{mol_identifier}{seq_hash:07d}"
            seq_hash_map[key] = _seq_hash
            seq_hash += 1
        new_seq_hash_map = {}
        for key, value in seq_hash_map.items():
            sequence = key[1:]
            new_seq_hash_map[value] = sequence
        seq_hash_map = new_seq_hash_map

        return seq_hash_map

    return _worker
