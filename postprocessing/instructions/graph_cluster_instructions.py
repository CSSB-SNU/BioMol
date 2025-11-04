import re
import string
from collections import defaultdict, deque
from collections.abc import Callable
from typing import TypeVar, Iterable

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from pathlib import Path

from biomol.core.container import FeatureContainer
from biomol.core.feature import NodeFeature, EdgeFeature
from biomol.core.utils import to_bytes
from biomol.mapping import ResidueMapping
from biomol.cif.mol import CIFMol
from joblib import Parallel, delayed
import itertools
from collections import Counter, defaultdict

from postprocessing.instructions.seq_instructions import graph_to_canonical_sequence


InputType = TypeVar("InputType", str, int, float)
FeatureType = TypeVar("FeatureType")
NumericType = TypeVar("NumericType", int, float)


def extract_graph_per_cifmol() -> Callable[..., bytes]:
    """Read CIFMol and extract chain-level contact graphs with cluster labels."""

    def _worker(
        cifmol: CIFMol,
        seq_to_seq_hash_list: dict[str, list[str]],
        seq_hash_to_cluster: dict[str, str],
    ) -> type[InputType]:
        chain_id_to_cluster = {}
        chain_ids = cifmol.chains.chain_id.value

        for full_chain_id in chain_ids:
            entity_type = cifmol.chains[
                cifmol.chains.chain_id == full_chain_id
            ].entity_type.value[0]
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

            match entity_type:
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

            seq_hash_list = seq_to_seq_hash_list[seq]
            seq_hash = [h for h in seq_hash_list if h.startswith(mol_identifier)][0]
            chain_id_to_cluster[full_chain_id] = seq_hash_to_cluster[seq_hash]

        # chain level contact graph
        contact_graph = cifmol.chains.contact
        chain_id_list = cifmol.chains.chain_id.value
        cluster_list = [chain_id_to_cluster[chain_id] for chain_id in chain_id_list]
        src, dst = contact_graph.src_indices, contact_graph.dst_indices

        cluster_list = np.array(cluster_list)
        features = {
            "seq_clusters": NodeFeature(cluster_list),
            "contact_edges": EdgeFeature(
                value=np.array([1] * len(src)),
                src_indices=np.array(src),
                dst_indices=np.array(dst),
            ),
        }
        container = FeatureContainer(features)
        return to_bytes({"cluster_graph": container})

    return _worker


def extract_graphs(n_jobs: int = -1) -> Callable[..., type[InputType]]:
    """Read CIFMol and extract chain-level contact graphs with cluster labels."""

    def _single_function(
        cifmol: CIFMol,
        seq_to_cluster: dict[str, str],
    ) -> type[InputType]:
        chain_id_to_cluster = {}
        chain_ids = cifmol.chains.chain_id.value
        for full_chain_id in chain_ids:
            entity_type = cifmol.chains[
                cifmol.chains.chain_id == full_chain_id
            ].entity_type.value[0]
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
            chain_id_to_cluster[full_chain_id] = seq_to_cluster[seq]

        # chain level contact graph
        contact_graph = cifmol.chains.contact
        chain_id_list = cifmol.chains.chain_id.value
        src, dst = contact_graph.src_indices, contact_graph.dst_indices
        graph = nx.Graph()
        for ii, chain_id in enumerate(chain_id_list):
            graph.add_node(ii, label=chain_id_to_cluster[chain_id])
        for s, d in zip(src, dst, strict=True):
            graph.add_edge(s, d)

        return graph

    def _worker(
        cifmol_dict: dict[str, CIFMol],
        seq_hash_map: Path,  # seq to seq hash
        seq_cluster_map: Path,  # seq hash to cluster
    ) -> dict[str, nx.Graph]:
        seq_to_hash: dict[str, int] = {}
        with seq_hash_map.open("r") as f:
            for line in f:
                seq_hash, seq = line.strip().split("\t")
                seq_to_hash[seq] = seq_hash
        seq_hash_to_cluster: dict[int, str] = {}
        with seq_cluster_map.open("r") as f:
            for line in f:
                rep, members = line.strip().split("\t")
                members = members.split(",")
                for m in members:
                    seq_hash_to_cluster[m] = rep

        seq_to_cluster: dict[str, str] = {}
        for seq, seq_hash in seq_to_hash.items():
            cluster_id = seq_hash_to_cluster[seq_hash]
            seq_to_cluster[seq] = cluster_id
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(_single_function)(cifmol, seq_to_cluster)
            for cifmol in cifmol_dict.values()
        )
        return dict(zip(cifmol_dict.keys(), results, strict=True))

    return _worker


def has_common_node(g1: nx.Graph, g2: nx.Graph) -> bool:
    """Check if two graphs have common node labels."""
    labels_g1 = {data.get("label") for node, data in g1.nodes(data=True)}
    labels_g2 = {data.get("label") for node, data in g2.nodes(data=True)}
    return bool(labels_g1.intersection(labels_g2))


def get_edge_labels(graph: nx.Graph) -> set[tuple[str, str]]:
    """Get edge labels from a graph."""
    edge_labels = set()
    for u, v in graph.edges():
        label_u = graph.nodes[u].get("label")
        label_v = graph.nodes[v].get("label")
        edge_labels.add(tuple(sorted((label_u, label_v))))
    return edge_labels


def has_common_edge(g1: nx.Graph, g2: nx.Graph) -> bool:
    """Check if two graphs have common edge labels."""
    edge_labels_g1 = get_edge_labels(g1)
    edge_labels_g2 = get_edge_labels(g2)
    return bool(edge_labels_g1.intersection(edge_labels_g2))


def graph_isomorphism(graph1: nx.Graph, graph2: nx.Graph) -> bool:
    """Check if two graphs are isomorphic."""

    def node_match(attr1: dict, attr2: dict) -> bool:
        return attr1.get("label") == attr2.get("label")

    gm = nx.isomorphism.GraphMatcher(graph1, graph2, node_match=node_match)
    return gm.is_isomorphic()


def build_graph_hash(n_jobs: int = -1) -> Callable[..., type[InputType]]:
    """Deduplicate graphs up to isomorphism using node 'label' attribute."""

    def _worker(
        graph_map: dict[str, nx.Graph],
    ) -> dict[str, dict[str, str]]:
        """
        Deduplicate graphs up to isomorphism using node 'label' attribute.

        - Uses strong invariants (WL hash + degree multiset + label multiset) to bucket graphs.
        - Within each bucket, runs exact isomorphism checks with categorical node match.
        - Assigns contiguous global cluster IDs starting from `start_idx`.

        Args:
            graph_map: Mapping from user graph key -> NetworkX Graph.
                    Each node must have a 'label' attribute (hashable).
            start_idx: Starting integer for cluster IDs.
            n_jobs:    Parallel workers for invariant computation; use -1 for all cores.

        Returns:
        -------
            unique_map: {cluster_id -> representative Graph}
            gid_to_id:  {input_key -> cluster_id}

        Notes:
            - Bucketing by invariants prunes most non-isomorphic pairs before exact checks.
            - Exact checks use GraphMatcher with categorical_node_match('label', None).
            - Determinism: per-bucket keys are processed in sorted order of input keys.
        """
        if not graph_map:
            return {}, {}

        # ---------- 1) Compute invariants in parallel ----------
        # Invariants are strong but cheap: WL hash (label-aware), degree multiset, label multiset.
        def _invariants(item: tuple[str | int, nx.Graph]) -> tuple[str | int, tuple]:
            gid, G = item
            # WL hash aware of node label
            wl = nx.weisfeiler_lehman_graph_hash(G, node_attr="label")
            # degree multiset
            degs = tuple(sorted(d for _, d in G.degree()))
            # node-label multiset
            lbls = tuple(sorted(Counter(nx.get_node_attributes(G, "label")).items()))
            return gid, (wl, degs, lbls)

        items = list(graph_map.items())
        inv_list = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(_invariants)(it) for it in items
        )
        inv_map: dict[str | int, tuple] = dict(inv_list)

        # ---------- 2) Bucket by invariants ----------
        buckets: dict[tuple, list[str | int]] = defaultdict(list)
        for gid, key in inv_map.items():
            buckets[key].append(gid)

        # Sort keys inside each bucket for determinism
        for key in buckets:
            buckets[key].sort(key=lambda x: (str(type(x)), str(x)))

        # ---------- 3) Per-bucket exact clustering (isomorphism with labels) ----------
        # Use threading backend to avoid heavy pickling of NetworkX graphs.
        node_match = nx.algorithms.isomorphism.categorical_node_match("label", None)

        def _cluster_bucket(
            gids: list[str | int],
        ) -> list[tuple[list[str | int], str | int]]:
            """
            Cluster one bucket into isomorphism classes.

            Returns a list of clusters as (member_gids, representative_gid).
            """
            reps: list[tuple[str | int, nx.Graph]] = []  # [(rep_gid, rep_graph)]
            clusters: list[list[str | int]] = []

            for gid in gids:
                G = graph_map[gid]
                placed = False
                # Try match against existing representatives
                for c_idx, (rep_gid, repG) in enumerate(reps):
                    GM = nx.isomorphism.GraphMatcher(G, repG, node_match=node_match)
                    if GM.is_isomorphic():
                        clusters[c_idx].append(gid)
                        placed = True
                        break
                if not placed:
                    reps.append((gid, G))
                    clusters.append([gid])

            # Pair each cluster with its representative gid (first in deterministic order)
            return [(cls, reps[i][0]) for i, cls in enumerate(clusters)]

        bucket_results = Parallel(n_jobs=n_jobs, backend="threading", verbose=5)(
            delayed(_cluster_bucket)(gids) for _, gids in buckets.items()
        )

        # Flatten list of clusters
        clusters_all: list[tuple[list[str | int], str | int]] = list(
            itertools.chain.from_iterable(bucket_results)
        )

        # ---------- 4) Assign global cluster IDs and build outputs ----------
        unique_map: dict[int, nx.Graph] = {}
        gid_to_id: dict[str | int, int] = {}

        for cur, (members, rep_gid) in enumerate(clusters_all):
            unique_map[cur] = graph_map[rep_gid]
            for gid in members:
                gid_to_id[gid] = cur
        return unique_map, gid_to_id

    return _worker


class _UnionFind:
    """Disjoint-set (Union-Find) with path compression and union by rank."""

    def __init__(self, elems: Iterable[int]):
        self._index = {e: i for i, e in enumerate(sorted(elems))}
        n = len(self._index)
        self.parent = list(range(n))
        self.rank = [0] * n
        self.reverse = {i: e for e, i in self._index.items()}

    def _id(self, e: int) -> int:
        return self._index[e]

    def find(self, e: int) -> int:
        i = self._id(e)
        # path compression
        if self.parent[i] != i:
            self.parent[i] = self.find(self.reverse[self.parent[i]])
        return self.parent[i]

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # union by rank
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[rb] < self.rank[ra]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

    def components(self) -> list[set[int]]:
        roots_to_members: dict[int, set[int]] = defaultdict(set)
        for i, e in self.reverse.items():
            r = self.find(e)
            roots_to_members[r].add(e)
        # stable order by smallest member for determinism
        comps = list(roots_to_members.values())
        comps.sort(key=lambda s: min(s))
        return comps


def graph_edge_cluster(n_jobs: int = -1) -> Callable[..., type[InputType]]:
    """Cluster graphs that share at least one edge defined by node label pairs."""

    def _worker(
        unique_graphs: dict[int, nx.Graph],
    ) -> tuple[list[set[int]], dict[int, int]]:
        """
        Cluster graphs if they share at least one edge defined by node label pairs.

        An "edge key" is an unordered pair (min(label_u, label_v), max(label_u, label_v)).
        Two graphs belong to the same cluster if they share >= 1 such edge key.

        Args:
            unique_graphs: {graph_id -> NetworkX Graph} where each node has attribute 'label'.
            n_jobs:        Number of parallel workers for edge extraction (-1 = all cores).

        Returns:
            clusters: List of sets of graph_ids, one set per connected component.
            comp_map: Mapping {graph_id -> component_index} where components are 0..C-1
                    assigned in ascending order of each component's smallest graph_id.

        Notes:
            - Graphs with no edges (or single-node) become singleton clusters.
            - Uses threading backend implicitly (joblib default) to avoid heavy pickling.
            - Deterministic component indices via sorting by smallest member id.
        """
        if not unique_graphs:
            return [], {}

        graph_ids = sorted(unique_graphs.keys())

        # --- 1) Extract per-graph labeled edge keys in parallel -------------------
        def _extract_edge_keys(gid: int) -> tuple[int, list[tuple[object, object]]]:
            G = unique_graphs[gid]
            labels = nx.get_node_attributes(G, "label")
            # Build a set to avoid duplicates within the same graph
            keys: set[tuple[object, object]] = set()
            # If directed graphs appear, treat as undirected by sorting endpoints
            for u, v in G.edges():
                lu = labels.get(u, None)
                lv = labels.get(v, None)
                # Require both endpoints to have 'label'
                if lu is None or lv is None:
                    continue
                key = (lu, lv) if lu <= lv else (lv, lu)
                keys.add(key)
            return gid, sorted(keys)

        edge_key_lists: list[tuple[int, list[tuple[object, object]]]] = Parallel(
            n_jobs=n_jobs, verbose=5
        )(delayed(_extract_edge_keys)(gid) for gid in graph_ids)

        # --- 2) Build edge_key -> [graph_id, ...] inverted index ------------------
        edge_to_graphs: dict[tuple[object, object], list[int]] = defaultdict(list)
        for gid, keys in edge_key_lists:
            for k in keys:
                edge_to_graphs[k].append(gid)

        # --- 3) Union-Find: union all graphs that share any edge key --------------
        uf = _UnionFind(graph_ids)
        for gids in edge_to_graphs.values():
            if len(gids) <= 1:
                continue
            base = gids[0]
            for other in gids[1:]:
                uf.union(base, other)

        # --- 4) Collect components and build comp_map -----------------------------
        clusters = uf.components()  # sorted by smallest member id
        comp_map: dict[int, int] = {}
        for c_idx, members in enumerate(clusters):
            for gid in members:
                comp_map[gid] = c_idx

        return clusters, comp_map

    return _worker


def convert_graph_to_bytes(n_jobs: int = -1) -> Callable[..., bytes]:
    """Convert a NetworkX graph to a FeatureContainer."""

    def _function(
        graph: nx.Graph,
    ) -> bytes:
        label_list = []
        src_list, dst_list = [], []
        for _, data in graph.nodes(data=True):
            label = data.get("label")
            label_list.append(label)
        for u, v in graph.edges():
            src_list.append(u)
            dst_list.append(v)

        label_list = np.array(label_list)
        features = {
            "seq_clusters": NodeFeature(label_list),
            "contact_edges": EdgeFeature(
                value=np.array([1] * len(src_list)),
                src_indices=np.array(src_list),
                dst_indices=np.array(dst_list),
            ),
        }
        container = FeatureContainer(features)
        return to_bytes({"cluster_graph": container})

    def _worker(
        graph_map: dict[str, nx.Graph],
    ) -> dict[str, bytes]:
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(_function)(graph) for graph in graph_map.values()
        )
        return dict(zip(graph_map.keys(), results, strict=True))

    return _worker
