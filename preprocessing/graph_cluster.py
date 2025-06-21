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
from itertools import combinations, chain
from collections import defaultdict
from BioMol import DB_PATH, SEQ_TO_HASH_PATH
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

all_mol_seq_to_cluster_path = f"{DB_PATH}/cluster/seq_clust/seq_to_cluster.pkl"
protein_cluster_path = (
    f"{DB_PATH}/cluster/seq_clust/protein_seq_clust/v2_chainID_to_cluster.pkl"
)

with open(SEQ_TO_HASH_PATH, "rb") as f:
    seq_to_hash = pickle.load(f)
with open(all_mol_seq_to_cluster_path, "rb") as f:
    seq_to_cluster = pickle.load(f)

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
                seq_cluster = seq_hash_to_cluster[seq_hash]
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


def unique_graphs(graphs):
    """
    Given a list of (graph_id, G), returns
      - unique:   a list of representative graphs
      - graph_map: dict mapping each graph_id to the index of its representative in `unique`
    Uniqueness is up to full isomorphism (matching the 'label' node attribute).
    """
    # mapping: hash -> list of indices in `unique` whose graphs share this hash
    hash_buckets: dict[str, list[int]] = {}
    unique: list[nx.Graph] = []
    graph_map: dict = {}
    # for matching node attributes
    nm = categorical_node_match("label", default=None)

    for graph_id, G in graphs:
        h = nx.weisfeiler_lehman_graph_hash(G, node_attr="label")
        # if no bucket yet, this is definitely new
        if h not in hash_buckets:
            idx = len(unique)
            unique.append(G)
            hash_buckets[h] = [idx]
            graph_map[graph_id] = idx
        else:
            # try each existing representative in this bucket
            found = False
            for idx in hash_buckets[h]:
                H = unique[idx]
                if nx.is_isomorphic(G, H, node_match=nm):
                    graph_map[graph_id] = idx
                    found = True
                    break
            if not found:
                # truly new graph
                idx = len(unique)
                unique.append(G)
                hash_buckets[h].append(idx)
                graph_map[graph_id] = idx

    return unique, graph_map


def level0_graph_cluster(
    graph_dir,
    hash_level_csv_path,
    hash_level_graph_path,
    cluster_level_csv_path,
    cluster_level_graph_path,
):
    # 1. Collect all graph file paths.
    graph_files = []
    for root, dirs, files in os.walk(graph_dir):
        for file in files:
            if file.endswith(".graph"):
                graph_files.append(os.path.join(root, file))

    # 2. Parse graphs in parallel.
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(parse_graph)(graph_path) for graph_path in graph_files
    )
    hash_graph_dict = {}
    cluster_graph_dict = {}
    for d in results:
        hash_graph_dict.update(d[0])
        cluster_graph_dict.update(d[1])

    # 3. Sort graph_dict by the number of nodes.
    hash_graph_dict = dict(
        sorted(hash_graph_dict.items(), key=lambda x: len(x[1].nodes()))
    )
    cluster_graph_dict = dict(
        sorted(cluster_graph_dict.items(), key=lambda x: len(x[1].nodes()))
    )

    print(f"total hash graphs : {len(hash_graph_dict)}")
    print(f"total cluster graphs : {len(cluster_graph_dict)}")

    # 4. Group graphs by their node count.
    hash_groups = {}
    cluster_groups = {}
    for graph_id, graph in hash_graph_dict.items():
        n_nodes = len(graph.nodes())
        # groups.setdefault(n_nodes, []).append((graph_id, graph))
        hash_groups.setdefault(n_nodes, []).append((graph_id, graph))
    for graph_id, graph in cluster_graph_dict.items():
        n_nodes = len(graph.nodes())
        # groups.setdefault(n_nodes, []).append((graph_id, graph))
        cluster_groups.setdefault(n_nodes, []).append((graph_id, graph))

    # 5. Process each group and assign cluster IDs.
    hash_graph_ID_to_hash_map = {}
    hash_unique_graph = {}
    cluster_graph_ID_to_hash_map = {}
    cluster_unique_graph = {}
    # Optionally, process groups in sorted order (by node count) for reproducibility.
    print(f"total hash groups : {len(hash_groups)}")
    print(f"total cluster groups : {len(cluster_groups)}")
    for n_nodes in sorted(hash_groups.keys()):
        group = hash_groups[n_nodes]
        _unique_graph, _graph_hash_map = unique_graphs(group)
        hash_unique_graph.update(_unique_graph)
        hash_graph_ID_to_hash_map.update(_graph_hash_map)
    for n_nodes in sorted(cluster_groups.keys()):
        group = cluster_groups[n_nodes]
        _unique_graph, _graph_hash_map = unique_graphs(group)
        cluster_unique_graph.update(_unique_graph)
        cluster_graph_ID_to_hash_map.update(_graph_hash_map)

    hash_hash_list = list(hash_graph_ID_to_hash_map.values())
    hash_hash_list = list(set(hash_hash_list))
    cluster_hash_list = list(cluster_graph_ID_to_hash_map.values())
    cluster_hash_list = list(set(cluster_hash_list))

    hash_hash_map = {hash_val: i for i, hash_val in enumerate(hash_hash_list)}
    cluster_hash_map = {hash_val: i for i, hash_val in enumerate(cluster_hash_list)}
    hash_graph_ID_to_hash_map = {
        graph_id: hash_hash_map[hash_val]
        for graph_id, hash_val in hash_graph_ID_to_hash_map.items()
    }
    hash_unique_graph = {
        hash_hash_map[hash_val]: graph for hash_val, graph in hash_unique_graph.items()
    }
    cluster_graph_ID_to_hash_map = {
        graph_id: cluster_hash_map[hash_val]
        for graph_id, hash_val in cluster_graph_ID_to_hash_map.items()
    }
    cluster_unique_graph = {
        cluster_hash_map[hash_val]: graph
        for hash_val, graph in cluster_unique_graph.items()
    }
    print(f"total unique graphs : {len(hash_unique_graph)}")

    # 6. Save clustering results.
    with open(hash_level_csv_path, "w") as f:
        for graph_id, graph_hash in hash_graph_ID_to_hash_map.items():
            f.write(f"{graph_id},{graph_hash}\n")
    with open(cluster_level_csv_path, "w") as f:
        for graph_id, graph_hash in cluster_graph_ID_to_hash_map.items():
            f.write(f"{graph_id},{graph_hash}\n")

    # Save unique graphs
    with open(hash_level_graph_path, "wb") as f:
        pickle.dump(hash_unique_graph, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(cluster_level_graph_path, "wb") as f:
        pickle.dump(cluster_unique_graph, f, protocol=pickle.HIGHEST_PROTOCOL)

    gc.collect()


def level1_graph_cluster(pickle_path, edge_to_graph_save_path, meta_save_path):
    with open(pickle_path, "rb") as f:
        unique_graph = pickle.load(f)

    # Compute the number of nodes per graph
    hash_list = sorted(unique_graph.keys())

    graphs = [(h, G) for h, G in unique_graph.items() if G.number_of_nodes() > 1]
    hash_list = list(unique_graph.keys())
    hash_to_idx = {h: i for i, h in enumerate(hash_list)}
    n = len(hash_list)

    # 3) define per-graph extractor
    def extract_edges(h, G):
        idx = hash_to_idx[h]
        out = []
        for u, v in G.edges():
            edge = (u, v) if u <= v else (v, u)
            out.append((edge, idx))
        return out

    if not os.path.exists(edge_to_graph_save_path):
        # 4) run extraction in parallel
        all_pairs = Parallel(n_jobs=-1, verbose=10)(
            delayed(extract_edges)(h, G) for h, G in graphs
        )
        # flatten to a single list of (edge, idx)
        flat = chain.from_iterable(all_pairs)

        # 5) aggregate into edge→[graph_idxs]
        edge_to_graphs = defaultdict(list)
        for edge, idx in flat:
            edge_to_graphs[edge].append(idx)
        # save to file
        with open(edge_to_graph_save_path, "wb") as f:
            pickle.dump(edge_to_graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        # load from file
        with open(edge_to_graph_save_path, "rb") as f:
            edge_to_graphs = pickle.load(f)

    # 6) build sparse meta-graph
    meta = torch.eye(n, dtype=torch.bool)
    for idx_list in edge_to_graphs.values():
        if len(idx_list) > 1:
            idx = torch.tensor(idx_list, dtype=torch.long, device=meta.device)
            # this will set meta[i, j] = True for all i,j in idx_list
            meta[idx.unsqueeze(1), idx.unsqueeze(0)] = True

    # 7) save result
    with open(meta_save_path, "wb") as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
    gc.collect()


def _check_node(task):
    # Unpack task: (i, j, graph1, graph2)
    graph1, graph2 = task
    hash1, graph1 = graph1
    hash2, graph2 = graph2
    return (hash1, hash2) if has_common_node(graph1, graph2) else None


def extract_clusters(meta_graph_path, unique_graph_path, save_path):
    """
    Given:
      - meta_graph_path: pickle of an (N,N) torch.BoolTensor where entry (i,j)=True
          iff graph i and j share an edge
      - unique_graph_path: pickle of a dict mapping graph-hash -> graph-object
    Produces:
      - clusters: a list of sets of graph-hashes, one per connected component
      - writes them to save_path, one set per line.
    """
    # load data
    with open(meta_graph_path, "rb") as f:
        meta_graph = pickle.load(f)  # torch.BoolTensor, shape (N,N)
    with open(unique_graph_path, "rb") as f:
        unique_graph = pickle.load(f)  # dict: hash -> graph

    hash_list = sorted(unique_graph.keys())  # length N
    N = len(hash_list)

    # build sparse matrix from each row's nonzeros
    rows, cols = [], []
    for i in range(N):
        nbrs = meta_graph[i].nonzero(as_tuple=True)[0].cpu().numpy()
        rows.append(np.full_like(nbrs, i))
        cols.append(nbrs)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)

    A = csr_matrix((np.ones(len(rows), bool), (rows, cols)), shape=(N, N))
    n_comp, labels = connected_components(A, directed=False, return_labels=True)

    clusters = [set() for _ in range(n_comp)]
    for idx, lbl in enumerate(labels):
        clusters[lbl].add(hash_list[idx])

    with open(save_path, "w") as f:
        for c in clusters:
            f.write(f"{c}\n")
    return clusters


def separate_graphs(
    meta_data_txt,
    graph_hash_to_graph_cluster_path,
    train_graph_hash_path,
    valid_graph_hash_path,
):
    """
    Given a meta_data_txt file containing graph identifiers and their corresponding cluster IDs,
    this function separates the graphs into different folders based on their cluster IDs.
    """
    with open(meta_data_txt, "r") as f:
        lines = f.readlines()

    # Create a dictionary to store the clusters
    total_num = 0
    clusters = []
    # Parse the lines and populate the clusters dictionary
    for line in lines:
        line = line.strip()
        line = line[1:-1]  # Remove the surrounding brackets
        line = line.split(", ")
        clusters.append((len(line), line))
        total_num += len(line)
    clusters = sorted(clusters, key=lambda x: x[0], reverse=True)
    hash_to_cluster = {}
    for ii, cluster in enumerate(clusters):
        for graph in cluster[1]:
            hash_to_cluster[graph] = ii + 1

    with open(graph_hash_to_graph_cluster_path, "w") as f:
        for graph, cluster in hash_to_cluster.items():
            f.write(f"{graph},{cluster}\n")

    train_valid_split = 0.8
    train_clusters = []
    valid_clusters = []
    train_num = 0
    add_to_train = True
    for cluster in clusters:
        for graph in cluster[1]:
            if add_to_train:
                train_clusters.append(graph)
                train_num += 1
            else:
                valid_clusters.append(graph)
        if train_num >= total_num * train_valid_split:
            add_to_train = False

    # sort the clusters
    train_clusters = sorted(train_clusters, key=lambda x: int(x))
    valid_clusters = sorted(valid_clusters, key=lambda x: int(x))

    with open(train_graph_hash_path, "w") as f:
        for graph in train_clusters:
            f.write(f"{graph}\n")
    with open(valid_graph_hash_path, "w") as f:
        for graph in valid_clusters:
            f.write(f"{graph}\n")
    print(f"train num : {len(train_clusters)}")


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


def write_PDBID_to_graph_hash(level0_save_path, PDBID_to_graph_hash_path):
    """
    Given a meta_graph (a torch.bool matrix where meta_graph[i,j] is True if graph i and j share an edge)
    and a list of graph identifiers (hash_list) corresponding to the rows/columns of meta_graph,
    this function returns a list of clusters, each cluster being a set of graph identifiers.
    """

    # Load the meta-graph
    with open(level0_save_path, "r") as f:
        lines = f.readlines()

    # Create a dictionary to store the clusters
    hash_to_PDBID = {}
    # Parse the lines and populate the clusters dictionary
    for line in lines:
        line = line.strip()
        line = line.split(",")
        pdb_ID, graph_hash = line[0], line[1]
        if graph_hash not in hash_to_PDBID:
            hash_to_PDBID[graph_hash] = []
        hash_to_PDBID[graph_hash].append(pdb_ID)

    # sort the clusters
    for graph_hash in hash_to_PDBID:
        hash_to_PDBID[graph_hash] = sorted(hash_to_PDBID[graph_hash], key=lambda x: x)
    # sort the hash
    hash_list = sorted(hash_to_PDBID.keys(), key=lambda x: int(x))
    # save as txt
    with open(PDBID_to_graph_hash_path, "w") as f:
        for graph_hash in hash_list:
            pdb_IDs = hash_to_PDBID[graph_hash]
            pdb_IDs = ",".join(pdb_IDs)
            f.write(f"{graph_hash}:{pdb_IDs}\n")


def test():
    # hash_level_csv_path = f"{DB_PATH}/cluster/graph_cluster/hash_level_cluster.csv"
    # hash_level_graph_path = (
    #     f"{DB_PATH}/cluster/graph_cluster/hash_level_unique_graphs.pkl"
    # )
    # cluster_level_csv_path = f"{DB_PATH}/cluster/graph_cluster/cluster_level_cluster.csv"
    # cluster_level_graph_path = (
    #     f"{DB_PATH}/cluster/graph_cluster/hash_level_unique_graphs.pkl"
    # )
    # # with open(cluster_level_graph_path, "rb") as f:
    # #     cluster_graphs = pickle.load(f)

    # pdb_ID_to_graph_cluster = {}
    # pdb_ID_to_graph_hash = {}
    # with open(hash_level_csv_path, "r") as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         line = line.strip()
    #         pdb_ID, graph_hash = line.split(",")
    #         if pdb_ID not in pdb_ID_to_graph_hash:
    #             pdb_ID_to_graph_hash[pdb_ID] = []
    #         pdb_ID_to_graph_hash[pdb_ID].append(graph_hash)
    # with open(cluster_level_csv_path, "r") as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         line = line.strip()
    #         pdb_ID, graph_cluster = line.split(",")
    #         if pdb_ID not in pdb_ID_to_graph_cluster:
    #             pdb_ID_to_graph_cluster[pdb_ID] = []
    #         pdb_ID_to_graph_cluster[pdb_ID].append(graph_cluster)
    # cluster_list = list(pdb_ID_to_graph_cluster.values())
    # hash_list = list(pdb_ID_to_graph_hash.values())

    # cluster_list = [cluster for sublist in cluster_list for cluster in sublist]
    # hash_list = [hash for sublist in hash_list for hash in sublist]

    # hash_to_cluster = {
    #     _hash: _cluster for _hash, _cluster in zip(hash_list, cluster_list)
    # }

    # cluster_list = list(set(cluster_list))
    # hash_list = list(set(hash_list))

    # 1ubq, 8yum
    graph_dir = f"{DB_PATH}/contact_graphs/"
    graph_1ubq = graph_dir + "ub/1ubq.graph"
    graph_8yum = graph_dir + "yu/8yum.graph"
    graph_1ubq, graph_8yum = parse_graph(graph_1ubq), parse_graph(graph_8yum)

    print(graph_1ubq[1]["1ubq_1_1_."].nodes(data="label"))
    print(graph_8yum[1]["8yum_1_1_."].nodes(data="label"))

    breakpoint()


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
    edge_level_save_path = f"{DB_PATH}/cluster/graph_cluster/edge_level_meta_graph.pkl"
    # level1_graph_cluster(
    #     cluster_level_graph_path, edge_to_graph_save_path, edge_level_save_path
    # )

    # meta_graph_path = f'{DB_PATH}/protein_graph/level1_meta_graph.pkl'
    # unique_graph_path = f'{DB_PATH}/protein_graph/unique_graphs.pkl'
    # # clusters = extract_clusters(meta_graph_path, unique_graph_path)

    # unittest_level1_graph(meta_graph_path, unique_graph_path)

    # unique_graph_pickle_path = f'{DB_PATH}/protein_graph/unique_graphs.pkl'
    # level2_save_path = f'{DB_PATH}/protein_graph/level2_meta_graph.pkl'
    # level2_graph_cluster(unique_graph_pickle_path, level2_save_path)

    # level1_graph_based_cluster = f"{DB_PATH}/cluster/graph_cluster/level1_cluster.txt"

    # clusters = extract_clusters(
    #     edge_level_save_path, cluster_level_graph_path, level1_graph_based_cluster
    # )

    # graph_hash_to_graph_cluster_path = (
    #     f"{DB_PATH}/cluster/graph_hash_to_graph_cluster.txt"
    # )
    # train_graph_hash_path = f"{DB_PATH}/cluster/train_graph_hash.txt"
    # valid_graph_hash_path = f"{DB_PATH}/cluster/valid_graph_hash.txt"
    # separate_graphs(
    #     f"{DB_PATH}/cluster/level2_cluster.txt",
    #     graph_hash_to_graph_cluster_path,
    #     train_graph_hash_path,
    #     valid_graph_hash_path,
    # )
