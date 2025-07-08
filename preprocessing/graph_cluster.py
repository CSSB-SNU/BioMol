import pickle
import os
import gc
import networkx as nx
from networkx.algorithms.isomorphism import categorical_node_match
from typing import List
import re
from joblib import Parallel, delayed
import torch
import numpy as np
from itertools import chain
from collections import defaultdict, Counter
from BioMol import DB_PATH, SEQ_TO_HASH_PATH
import json

all_mol_seq_to_cluster_path = f"{DB_PATH}/cluster/seq_clust/seq_to_cluster.pkl"
chain_to_cluster_path = f"{DB_PATH}/cluster/seq_clust/chain_ID_to_cluster.pkl"

protein_cluster_path = (
    f"{DB_PATH}/cluster/seq_clust/protein_seq_clust/v2_chainID_to_cluster.pkl"
)

with open(SEQ_TO_HASH_PATH, "rb") as f:
    seq_to_hash = pickle.load(f)
with open(all_mol_seq_to_cluster_path, "rb") as f:
    seq_to_cluster = pickle.load(f)

# breakpoint()

with open(chain_to_cluster_path, "rb") as f:
    chain_to_cluster = pickle.load(f)

# print(51759 in chain_to_cluster.values())
# print(51759 in seq_to_cluster.values())

# cluster_to_chain = {}
# for chain_ID, cluster in chain_to_cluster.items():
#     if cluster not in cluster_to_chain:
#         cluster_to_chain[cluster] = []
#     cluster_to_chain[cluster].append(chain_ID)

# breakpoint()


seq_hash_to_cluster = {}
for seq, cluster in seq_to_cluster.items():
    seq_hash = seq_to_hash[seq]

    seq_hash_to_cluster[seq_hash] = cluster

protein_chain_ID_to_cluster = {}
with open(protein_cluster_path, "rb") as f:
    protein_chain_ID_to_cluster = pickle.load(f)


# Regex pattern to capture the three groups inside the tuple
pattern = re.compile(r"\('([^']+)', '([^']+)', '([^']+)'\)")


class custom_graph(nx.Graph):
    def __init__(self, graph: nx.Graph):
        super().__init__(graph)
        self.graph = graph

    def get_edges(self):
        return self.graph.edges()


def format_tuple(s):
    match = pattern.search(s)
    if match:
        # Combine the captured groups with underscores
        return f"{match.group(1)}_{match.group(2)}_{match.group(3)}"
    return None


def parse_graph(file_path: str) -> List[nx.Graph]:
    graph_ID = file_path.split("/")[-1].split(".")[0]
    hash_G_dict = {}
    cluster_G_dict = {}
    hash_G = nx.Graph()  # Use nx.DiGraph() for directed graphs
    cluster_G = nx.Graph()
    ID = None
    with open(file_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "t":  # New graph line: "t ID"
                if hash_G.number_of_nodes() > 0:
                    hash_G_dict[ID] = hash_G
                    cluster_G_dict[ID] = cluster_G
                    hash_G = nx.Graph()
                    cluster_G = nx.Graph()
                parts = line.strip().split("#")
                ID = format_tuple(parts[1])
                ID = f"{graph_ID}_{ID}"
            elif parts[0] == "v":  # Vertex line: "v id label"
                node_id = int(parts[1])
                seq_hash = parts[2]
                try:
                    seq_cluster = seq_hash_to_cluster[seq_hash]
                except:
                    print(file_path)
                    assert file_path == 0, (
                        f"file_path: {file_path}, seq_hash: {seq_hash}"
                    )
                hash_G.add_node(node_id, label=seq_hash)
                cluster_G.add_node(node_id, label=seq_cluster)
            elif parts[0] == "e":  # Edge line: "e src tgt"
                src, tgt = int(parts[1]), int(parts[2])
                hash_G.add_edge(src, tgt)
                cluster_G.add_edge(src, tgt)
        if hash_G.number_of_nodes() > 0:
            hash_G_dict[ID] = hash_G
            cluster_G_dict[ID] = cluster_G
    return hash_G_dict, cluster_G_dict


# graph_path = f"{DB_PATH}/contact_graphs/rs/3rsq.graph"
# out = parse_graph(graph_path)
# breakpoint()


def has_common_node(G1, G2):
    # same node label
    labels_G1 = {data.get("label") for node, data in G1.nodes(data=True)}
    labels_G2 = {data.get("label") for node, data in G2.nodes(data=True)}
    if labels_G1.intersection(labels_G2):
        return True
    return False


def get_edge_labels(graph):
    edge_labels = set()
    for u, v in graph.edges():
        # 각 노드의 label을 가져온 후, 정렬된 튜플로 만듦 (방향 없는 edge이므로)
        label_u = graph.nodes[u].get("label")
        label_v = graph.nodes[v].get("label")
        edge_labels.add(tuple(sorted((label_u, label_v))))
    return edge_labels


def has_common_edge(G1, G2):
    edge_labels_G1 = get_edge_labels(G1)
    edge_labels_G2 = get_edge_labels(G2)
    if edge_labels_G1.intersection(edge_labels_G2):
        return True
    return False


def graph_isomorphism(graph1, graph2):
    node_match = lambda attr1, attr2: attr1.get("label") == attr2.get("label")
    GM = nx.isomorphism.GraphMatcher(graph1, graph2, node_match=node_match)
    return GM.is_isomorphic()


def unique_graphs(
    graphs: list[tuple[str, nx.Graph]], start_idx: int = 0, n_jobs: int = -1
) -> tuple[dict[int, nx.Graph], dict[str, int]]:
    """
    Identify unique graphs up to isomorphism across multiple invariant‐buckets.
    Uses NumPy‐unique on flattened adjacency matrices to cluster each bucket
    in parallel via joblib (threading backend, to avoid pickle overhead).

    Args:
      graphs:    List of (graph_id, Graph), all same node-count per bucket.
      start_idx: Starting cluster index.
      n_jobs:    Number of parallel jobs for joblib.

    Returns:
      unique:    dict[cluster_idx -> representative Graph]
      graph_map: dict[graph_id -> cluster_idx]
    """
    if not graphs:
        return {}, {}

    # 1) Build a map for quick lookup
    G_map = {gid: G for gid, G in graphs}
    nm = nx.algorithms.isomorphism.categorical_node_match("label", None)

    # 2) Compute combined‐invariant keys in parallel
    def _inv(item):
        gid, G = item
        h = nx.weisfeiler_lehman_graph_hash(G, node_attr="label")
        deg_seq = tuple(sorted(d for _, d in G.degree()))
        labels = tuple(sorted(Counter(nx.get_node_attributes(G, "label")).items()))
        return gid, (h, deg_seq, labels)

    inv_items = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_inv)(item) for item in graphs
    )
    inv_map = dict(inv_items)

    # 3) Bucket GIDs by invariant key
    buckets: dict[tuple, list[str]] = {}
    for gid, key in inv_map.items():
        buckets.setdefault(key, []).append(gid)

    # 4) Prepare bucket_data: (nodes, mats, gid_list) per bucket
    bucket_data = []
    for gid_list in buckets.values():
        # assume all graphs in this bucket share the same node set
        nodes = sorted(G_map[gid_list[0]].nodes())
        mats = np.stack(
            [
                nx.to_numpy_array(G_map[gid], nodelist=nodes, dtype=np.uint8).ravel()
                for gid in gid_list
            ],
            axis=0,
        )  # shape = (B_i, N*N)
        bucket_data.append((nodes, mats, gid_list))

    # 5) Define bucket‐level clustering using NumPy.unique
    def _cluster_bucket(nodes, mats, gid_list):
        # mats: np.ndarray shape (B_i, N²)
        if len(gid_list) == 1:
            # only one graph in this bucket, no clustering needed
            return [gid_list[0]], {gid_list[0]: gid_list[0]}
        uniq_rows, indices, inverse = np.unique(
            mats, axis=0, return_index=True, return_inverse=True
        )
        # rep_gids: first‐occurrence indices → actual GIDs
        rep_gids = [gid_list[i] for i in indices]
        # map each gid to its rep_gid
        gid_to_rep = {
            gid_list[j]: gid_list[indices[inverse[j]]] for j in range(len(gid_list))
        }
        return rep_gids, gid_to_rep

    # 6) Run clustering on all buckets in parallel (threading backend)
    results = Parallel(n_jobs=n_jobs, backend="threading", verbose=5)(
        delayed(_cluster_bucket)(nodes, mats, gid_list)
        for nodes, mats, gid_list in bucket_data
    )

    # 7) Merge results and assign global cluster indices
    unique = {}
    graph_map = {}
    next_idx = start_idx

    for rep_gids, gid_to_rep in results:
        # assign one cluster index per rep_gid
        rep_to_global = {}
        for rep in rep_gids:
            rep_to_global[rep] = next_idx
            unique[next_idx] = G_map[rep]
            next_idx += 1
        # now map every gid in this bucket to its cluster index
        for gid, rep in gid_to_rep.items():
            graph_map[gid] = rep_to_global[rep]

    return unique, graph_map


def process_level(
    graphs: dict[str, nx.Graph],
    csv_path: str,
    graph_path: str,
    start_idx: int = 0,
    n_jobs: int = -1,
) -> tuple[dict[int, nx.Graph], dict[str, int], int]:
    """
    Cluster a dict of graphs, write CSV & pickle.
    Buckets by node‐count and calls unique_graphs **sequentially**,
    relying on its internal parallelism.
    """
    # Bucket by node count
    groups: dict[int, list[tuple[str, nx.Graph]]] = {}
    for gid, G in sorted(graphs.items(), key=lambda x: len(x[1].nodes())):
        groups.setdefault(len(G.nodes()), []).append((gid, G))

    unique_all = {}
    map_all = {}
    idx = start_idx

    # sort groups by node number, last group = smallest node count
    groups = {k: v for k, v in sorted(groups.items(), key=lambda x: x[0], reverse=True)}

    print(f"Total groups: {len(groups)}")

    for ii, n_nodes in enumerate(groups):
        print(f"Processing bucket {ii + 1}/{len(groups)}: {n_nodes} nodes")
        bucket = groups[n_nodes]
        uniq_local, map_local = unique_graphs(bucket, start_idx=idx, n_jobs=n_jobs)
        unique_all.update(uniq_local)
        map_all.update(map_local)
        idx = max(unique_all.keys()) + 1

    # Write out CSV
    with open(csv_path, "w") as f:
        for gid, cid in map_all.items():
            f.write(f"{gid},{cid}\n")

    # Pickle representative graphs
    with open(graph_path, "wb") as f:
        pickle.dump(unique_all, f, protocol=pickle.HIGHEST_PROTOCOL)

    return unique_all, map_all, idx


def level0_graph_cluster(
    graph_dir: str,
    hash_level_csv: str,
    hash_level_graph: str,
    cluster_level_csv: str,
    cluster_level_graph: str,
):
    # collect and parse
    files = []
    for root, _, filenames in os.walk(graph_dir):
        for fn in filenames:
            if fn.endswith(".graph"):
                files.append(os.path.join(root, fn))

    tmp = "tmp/parsed_graphs.pkl"
    os.makedirs("tmp", exist_ok=True)
    if not os.path.exists(tmp):
        print("Parsing graphs...")
        parsed = Parallel(n_jobs=-1, verbose=10)(delayed(parse_graph)(f) for f in files)
        with open(tmp, "wb") as f:
            pickle.dump(parsed, f)
    else:
        print("Loading from tmp...")
        with open(tmp, "rb") as f:
            parsed = pickle.load(f)

    hash_dict: dict[str, nx.Graph] = {}
    cluster_dict: dict[str, nx.Graph] = {}
    for hmap, cmap in parsed:
        hash_dict.update(hmap)
        cluster_dict.update(cmap)

    print(f"Total graphs: hash={len(hash_dict)}, cluster={len(cluster_dict)}")

    # process both levels, chaining indices
    _, _, next_idx = process_level(
        hash_dict, hash_level_csv, hash_level_graph, start_idx=0
    )
    print(f"Processed hash level, next index: {next_idx}")
    process_level(
        cluster_dict, cluster_level_csv, cluster_level_graph, start_idx=next_idx
    )
    print("Processed cluster level.")
    gc.collect()


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        # path-compression
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        # union by rank
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[ry] < self.rank[rx]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1


def level1_graph_cluster(pickle_path, edge_to_graph_save_path, cluster_save_path):
    # Load unique_graph and build hash list
    with open(pickle_path, "rb") as f:
        unique_graph = pickle.load(f)
    hash_list = sorted(unique_graph.keys())
    hash_to_idx = {h: i for i, h in enumerate(hash_list)}
    N = len(hash_list)

    # Step 1: Build or load edge -> [graph_idx] mapping
    if not os.path.exists(edge_to_graph_save_path):
        graphs = [(h, G) for h, G in unique_graph.items() if G.number_of_nodes() > 1]

        def extract_edges(h, G):
            idx = hash_to_idx[h]
            out = []
            node_dict = G.nodes(data="label")
            for u, v in G.edges():
                u, v = node_dict[u], node_dict[v]
                edge = (u, v) if u <= v else (v, u)
                out.append((edge, idx))
            return out

        all_pairs = Parallel(n_jobs=-1, verbose=10)(
            delayed(extract_edges)(h, G) for h, G in graphs
        )
        flat = chain.from_iterable(all_pairs)

        edge_to_graphs = defaultdict(list)
        for edge, idx in flat:
            edge_to_graphs[edge].append(idx)

        with open(edge_to_graph_save_path, "wb") as f:
            pickle.dump(edge_to_graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(edge_to_graph_save_path, "rb") as f:
            edge_to_graphs = pickle.load(f)

    # Step 2: Union-Find to cluster graphs by shared edges
    uf = UnionFind(N)
    for idxs in edge_to_graphs.values():
        if len(idxs) <= 1:
            continue
        rep = idxs[0]
        for i in idxs[1:]:
            uf.union(rep, i)

    # Step 3: Collect clusters
    clusters = defaultdict(set)
    for i, h in enumerate(hash_list):
        root = uf.find(i)
        clusters[root].add(h)

    # Save clusters to file
    with open(cluster_save_path, "w") as f:
        for comp in clusters.values():
            f.write(f"{comp}\n")

    print(f"Extracted {len(clusters)} clusters from {N} graphs.")

    return list(clusters.values())


def separate_and_map_graphs(
    hash_level_csv_path,
    cluster_level_csv_path,
    edge_level_csv_path,
    train_graph_hash_path,
    valid_graph_hash_path,
    graph_cluster_metadata_path,
):
    pdb_ID_to_hash_hash = {}
    pdb_ID_to_cluster_hash = {}
    hash_hash_to_pdbID = {}

    cluster_to_hash = {}

    with open(hash_level_csv_path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            pdb_ID, graph_hash = line.split(",")
            pdb_ID_to_hash_hash[pdb_ID] = graph_hash
            if graph_hash not in hash_hash_to_pdbID:
                hash_hash_to_pdbID[graph_hash] = []
            hash_hash_to_pdbID[graph_hash].append(pdb_ID)

    with open(cluster_level_csv_path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            pdb_ID, cluster_hash = line.split(",")
            pdb_ID_to_cluster_hash[pdb_ID] = cluster_hash

    pdb_ID_list = sorted(pdb_ID_to_hash_hash.keys())
    for pdb_ID in pdb_ID_list:
        graph_hash = pdb_ID_to_hash_hash[pdb_ID]
        cluster_hash = pdb_ID_to_cluster_hash[pdb_ID]
        if cluster_hash not in cluster_to_hash:
            cluster_to_hash[cluster_hash] = []
        cluster_to_hash[cluster_hash].append(graph_hash)

    cluster_to_hash_hash = {}
    cluster_to_cluster_hash = {}
    with open(edge_level_csv_path) as f:
        lines = f.readlines()
        for ii, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            line = line[1:-1]  # Remove the surrounding brackets
            cluster_ids = line.split(",")
            cluster_ids = [cluster_id.strip() for cluster_id in cluster_ids]
            hash_ids = [cluster_to_hash[cluster_id] for cluster_id in cluster_ids]
            hash_ids = [item for sublist in hash_ids for item in sublist]
            cluster_to_hash_hash[ii] = set(hash_ids)
            cluster_to_cluster_hash[ii] = set(cluster_ids)

    # sort by
    graph_cluster_ids = list(cluster_to_hash_hash.keys())
    graph_cluster_ids = sorted(
        graph_cluster_ids, key=lambda x: len(cluster_to_cluster_hash[x]), reverse=True
    )

    total_num = sum(
        len(cluster_to_cluster_hash[cluster_id]) for cluster_id in graph_cluster_ids
    )

    train_valid_split = 0.8
    train_clusters_ids = []
    valid_clusters_ids = []
    train_num = 0
    add_to_train = True
    for cluster_id in graph_cluster_ids:
        cluster_num = len(cluster_to_cluster_hash[cluster_id])
        if add_to_train:
            train_clusters_ids.append(cluster_id)
            train_num += cluster_num
        else:
            valid_clusters_ids.append(cluster_id)
        if train_num >= total_num * train_valid_split:
            add_to_train = False

    train_graph_hash = []
    valid_graph_hash = []

    for cluster_id in train_clusters_ids:
        cluster_hashes = cluster_to_cluster_hash[cluster_id]
        train_graph_hash.extend(cluster_hashes)
    for cluster_id in valid_clusters_ids:
        cluster_hashes = cluster_to_hash_hash[cluster_id]
        valid_graph_hash.extend(cluster_hashes)

    graph_cluster_metadata = {}
    for cluster_hash in pdb_ID_to_cluster_hash.values():
        if cluster_hash not in graph_cluster_metadata:
            graph_cluster_metadata[cluster_hash] = {}
        hash_hash = cluster_to_hash[cluster_hash]
        for graph_hash in hash_hash:
            graph_cluster_metadata[cluster_hash][graph_hash] = hash_hash_to_pdbID[
                graph_hash
            ]
    print(f"Total graphs: {len(train_graph_hash) + len(valid_graph_hash)}")
    print(f"Train graphs: {len(train_graph_hash)}")
    print(f"Valid graphs: {len(valid_graph_hash)}")

    # sort the graph hashes
    train_graph_hash = sorted(train_graph_hash, key=lambda x: int(x))
    valid_graph_hash = sorted(valid_graph_hash, key=lambda x: int(x))

    # save the graph hashes
    with open(train_graph_hash_path, "w") as f:
        for graph_hash in train_graph_hash:
            f.write(f"{graph_hash}\n")
    with open(valid_graph_hash_path, "w") as f:
        for graph_hash in valid_graph_hash:
            f.write(f"{graph_hash}\n")

    # save the graph cluster metadata as JSON
    with open(graph_cluster_metadata_path, "w") as f:
        json.dump(graph_cluster_metadata, f, indent=4)


def unittest_level1_graph(meta_graph_path, unique_graph_path):
    """
    Given a meta_graph (a torch.bool matrix where meta_graph[i,j] is True if graph i and j share an edge)
    and a dictionary of unique graphs (mapping graph identifiers to graphs),
    this function clusters the graphs and then performs two tests in parallel:
      1. Checks that in each cluster, every graph has at least one common edge with another graph in the same cluster.
      2. Randomly samples pairs (from different clusters) and checks that if they share a common node,
         they do not also share a common edge.
    """
    # Load meta-graph and unique graphs
    with open(meta_graph_path, "rb") as f:
        meta_graph = pickle.load(f)
    with open(unique_graph_path, "rb") as f:
        unique_graph = pickle.load(f)

    # Get the sorted list of graph identifiers
    hash_list = sorted(unique_graph.keys())

    # Convert the torch tensor to a numpy array
    meta_graph_np = meta_graph.numpy()

    # Create an undirected graph from the numpy array and compute connected components
    G = nx.from_numpy_array(meta_graph_np)
    clusters_indices = list(nx.connected_components(G))
    clusters = [{hash_list[i] for i in comp} for comp in clusters_indices]

    # Test 1: For each cluster with more than one graph, ensure each graph has at least one common edge with another graph.
    def test_cluster(cluster):
        cluster_list = list(cluster)
        for idx, graph_id in enumerate(cluster_list):
            # Flag to track if this graph has a common edge with at least one other graph in the same cluster
            at_least_one_common_edge = False
            for jdx, other_graph_id in enumerate(cluster_list):
                if idx == jdx:
                    continue
                if has_common_edge(unique_graph[graph_id], unique_graph[other_graph_id]):
                    at_least_one_common_edge = True
                    break
            if not at_least_one_common_edge:
                print(
                    f"Error: Graph {graph_id} in cluster {cluster} does not share an edge with any other graph."
                )
                breakpoint()  # For debugging purposes; you can remove or replace this as needed.

    # Parallelize test 1 across clusters with more than one element
    Parallel(n_jobs=-1)(
        delayed(test_cluster)(cluster) for cluster in clusters if len(cluster) > 1
    )

    # Test 2: Randomly sample pairs and ensure that if two graphs share a common node, they do not also share a common edge.
    sample_num = 1000

    def test_pair(_):
        i, j = np.random.choice(len(hash_list), 2)
        g1, g2 = unique_graph[hash_list[i]], unique_graph[hash_list[j]]
        if has_common_node(g1, g2):
            if has_common_edge(g1, g2):
                print(
                    f"Error: Graphs {hash_list[i]} and {hash_list[j]} (sampled pair) share both a common node and a common edge."
                )

    # Parallelize the random pair tests
    Parallel(n_jobs=-1)(delayed(test_pair)(_) for _ in range(sample_num))


def test():
    pickle_path = f"{DB_PATH}/cluster/graph_cluster/cluster_level_unique_graphs.pkl"
    with open(pickle_path, "rb") as f:
        unique_graph = pickle.load(f)
    hash_list = sorted(unique_graph.keys())
    edge_to_graph_save_path = f"{DB_PATH}/cluster/graph_cluster/edge_to_graph.pkl"
    edge_level_save_path = f"{DB_PATH}/cluster/graph_cluster/edge_level_cluster.csv"
    with open(edge_to_graph_save_path, "rb") as f:
        edge_to_graphs = pickle.load(f)
    edge_to_graphs_key_list = list(edge_to_graphs.keys())
    # edge_to_graphs[(51178,51178)]
    N = 251082
    uf = UnionFind(N)
    for idxs in edge_to_graphs.values():
        if len(idxs) <= 1:
            continue
        rep = idxs[0]
        for i in idxs[1:]:
            uf.union(rep, i)

    with open(edge_level_save_path, "r") as f:
        lines = f.readlines()

    clusters = {}
    for ii, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        line = line[1:-1]  # Remove the surrounding brackets
        cluster_ids = line.split(",")
        clusters[ii] = set(cluster_ids)

    item_num_list = [len(cluster) for cluster in clusters.values()]
    item_num_list = sorted(item_num_list, reverse=True)

    breakpoint()

    # # 1ubq, 8yum
    # graph_dir = f"{DB_PATH}/contact_graphs/"
    # graph_1ubq = graph_dir + "ub/1ubq.graph"
    # graph_8yum = graph_dir + "yu/8yum.graph"
    # graph_1ubq, graph_8yum = parse_graph(graph_1ubq), parse_graph(graph_8yum)

    # print(graph_1ubq[1]["1ubq_1_1_."].nodes(data="label"))
    # print(graph_8yum[1]["8yum_1_1_."].nodes(data="label"))

    # breakpoint()


if __name__ == "__main__":
    graph_dir = f"{DB_PATH}/contact_graphs/"
    hash_level_csv_path = f"{DB_PATH}/cluster/graph_cluster/hash_level_cluster.csv"
    hash_level_graph_path = (
        f"{DB_PATH}/cluster/graph_cluster/hash_level_unique_graphs.pkl"
    )
    cluster_level_csv_path = f"{DB_PATH}/cluster/graph_cluster/cluster_level_cluster.csv"
    cluster_level_graph_path = (
        f"{DB_PATH}/cluster/graph_cluster/cluster_level_unique_graphs.pkl"
    )

    level0_graph_cluster(
        graph_dir,
        hash_level_csv_path,
        hash_level_graph_path,
        cluster_level_csv_path,
        cluster_level_graph_path,
    )

    # test()

    edge_to_graph_save_path = f"{DB_PATH}/cluster/graph_cluster/edge_to_graph.pkl"
    edge_level_csv_path = f"{DB_PATH}/cluster/graph_cluster/edge_level_cluster.csv"
    level1_graph_cluster(
        cluster_level_graph_path, edge_to_graph_save_path, edge_level_csv_path
    )

    # # unittest_level1_graph(meta_graph_path, unique_graph_path)

    train_graph_hash_path = f"{DB_PATH}/cluster/graph_cluster/train_graph_hash.txt"
    valid_graph_hash_path = f"{DB_PATH}/cluster/graph_cluster/valid_graph_hash.txt"
    graph_cluster_metadata_path = (
        f"{DB_PATH}/cluster/graph_cluster/graph_cluster_metadata.json"
    )
    separate_and_map_graphs(
        hash_level_csv_path,
        cluster_level_csv_path,
        edge_level_csv_path,
        train_graph_hash_path,
        valid_graph_hash_path,
        graph_cluster_metadata_path,
    )
