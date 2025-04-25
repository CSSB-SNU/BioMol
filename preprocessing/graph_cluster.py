import pickle
import os
import gc
import networkx as nx
from typing import List
import re
from joblib import Parallel, delayed
import time
import torch
import numpy as np
from itertools import combinations

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
    G_dict = {}
    G = nx.Graph()  # Use nx.DiGraph() for directed graphs
    with open(file_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "t":  # New graph line: "t ID"
                if G.number_of_nodes() > 0:
                    G_dict[ID] = G
                    G = nx.Graph()
                parts = line.strip().split("#")
                ID = format_tuple(parts[1])
                ID = f"{graph_ID}_{ID}"
            elif parts[0] == "v":  # Vertex line: "v id label"
                node_id = int(parts[1])
                label = parts[2]
                G.add_node(node_id, label=label)
            elif parts[0] == "e":  # Edge line: "e src tgt"
                src, tgt = int(parts[1]), int(parts[2])
                G.add_edge(src, tgt)
        if G.number_of_nodes() > 0:
            G_dict[ID] = G
    return G_dict


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


import os
from joblib import Parallel, delayed


def unique_graphs(graphs):
    """
    Given a list of graphs, return a list of unique graphs (up to isomorphism),
    based on the WL graph hash using the 'label' attribute.
    """
    unique = {}
    graph_map = {}
    for graph_id, G in graphs:
        # Compute the WL hash for the graph using the node label attribute.
        try:
            graph_hash = nx.weisfeiler_lehman_graph_hash(G, node_attr="label")
        except:
            breakpoint()
        # If the hash is not seen before, store the graph.
        if graph_hash not in unique:
            unique[graph_hash] = G
        graph_map[graph_id] = graph_hash
    return unique, graph_map


def level0_graph_cluster(graph_dir, save_path1, save_path2):
    # 1. Collect all graph file paths.
    graph_files = []
    for root, dirs, files in os.walk(graph_dir):
        for file in files:
            if file.endswith(".graph"):
                graph_files.append(os.path.join(root, file))

    # 2. Parse graphs in parallel.
    results = Parallel(n_jobs=-1)(
        delayed(parse_graph)(graph_path) for graph_path in graph_files
    )
    graph_dict = {}
    for d in results:
        graph_dict.update(d)

    # 3. Sort graph_dict by the number of nodes.
    graph_dict = dict(sorted(graph_dict.items(), key=lambda x: len(x[1].nodes())))

    # 4. Group graphs by their node count.
    groups = {}
    for graph_id, graph in graph_dict.items():
        n_nodes = len(graph.nodes())
        # groups.setdefault(n_nodes, []).append((graph_id, graph))
        groups.setdefault(n_nodes, []).append((graph_id, graph))

    # 5. Process each group and assign cluster IDs.
    graph_hash_map = {}
    unique_graph = {}
    global_cluster_id = 0
    # Optionally, process groups in sorted order (by node count) for reproducibility.
    print(f"total groups : {len(groups)}")
    for n_nodes in sorted(groups.keys()):
        group = groups[n_nodes]
        cluster_start_time = time.time()
        print(f"{n_nodes} : {len(group)}")
        _unique_graph, _graph_hash_map = unique_graphs(group)
        unique_graph.update(_unique_graph)
        graph_hash_map.update(_graph_hash_map)

        print(f"cluster time : {time.time() - cluster_start_time}")
    hash_list = list(graph_hash_map.values())
    hash_set = set(hash_list)
    hash_list = list(hash_set)
    hash_map = {hash_val: i for i, hash_val in enumerate(hash_list)}
    graph_hash_map = {
        graph_id: hash_map[hash_val] for graph_id, hash_val in graph_hash_map.items()
    }
    unique_graph = {
        hash_map[hash_val]: graph for hash_val, graph in unique_graph.items()
    }
    print(f"total unique graphs : {len(unique_graph)}")

    # 6. Save clustering results.
    with open(save_path1, "w") as f:
        for graph_id, graph_hash in graph_hash_map.items():
            f.write(f"{graph_id},{graph_hash}\n")

    # 7. Save unique graphs.
    with open(save_path2, "wb") as f:
        pickle.dump(unique_graph, f, protocol=pickle.HIGHEST_PROTOCOL)
    gc.collect()

    # For debugging


def _check_edge(task):
    # Unpack task: (i, j, graph1, graph2)
    graph1, graph2 = task
    hash1, graph1 = graph1
    hash2, graph2 = graph2
    return (hash1, hash2) if has_common_edge(graph1, graph2) else None


def level1_graph_cluster(pickle_path, save_path):
    with open(pickle_path, "rb") as f:
        unique_graph = pickle.load(f)

    # Compute the number of nodes per graph
    hash_to_len = {h: len(graph.nodes()) for h, graph in unique_graph.items()}
    hash_list = sorted(unique_graph.keys())

    # Select only graphs with more than one node
    not_len1_list = [h for h in hash_list if hash_to_len[h] != 1]
    num_len1 = len(hash_list) - len(not_len1_list)

    # Build mapping from hash to index in the sorted list
    hash_to_idx = {h: i for i, h in enumerate(hash_list)}

    # Precompute lists of indices and graphs for non-singleton graphs
    indices = [hash_to_idx[h] for h in not_len1_list]
    graphs = [(h, unique_graph[h]) for h in not_len1_list]

    # Use combinations to generate tasks without the nested loop overhead.
    print(f"start")
    tasks = [
        (graphs[i], graphs[j]) for i, j in combinations(range(len(not_len1_list)), 2)
    ]
    print(f"len tasks : {len(tasks)}")

    # Create meta-graph as an identity matrix
    graph_num = len(hash_list)
    meta_graph = torch.eye(graph_num)

    # Parallel processing of the edge check
    results = Parallel(n_jobs=-1)(delayed(_check_edge)(task) for task in tasks)

    # Update the meta-graph with the results
    for res in results:
        if res is not None:
            i, j = res
            meta_graph[i, j] = 1
            meta_graph[j, i] = 1

    meta_graph = meta_graph.bool()

    # save meta_graph
    with open(save_path, "wb") as f:
        pickle.dump(meta_graph, f, protocol=pickle.HIGHEST_PROTOCOL)
    gc.collect()


def _check_node(task):
    # Unpack task: (i, j, graph1, graph2)
    graph1, graph2 = task
    hash1, graph1 = graph1
    hash2, graph2 = graph2
    return (hash1, hash2) if has_common_node(graph1, graph2) else None


def level2_graph_cluster(pickle_path, save_path):
    with open(pickle_path, "rb") as f:
        unique_graph = pickle.load(f)

    # Compute the number of nodes per graph
    hash_list = sorted(unique_graph.keys())

    # Precompute lists of indices and graphs for non-singleton graphs
    graphs = [(h, unique_graph[h]) for h in hash_list]

    # Use combinations to generate tasks without the nested loop overhead.
    print(f"start")
    tasks = [(graphs[i], graphs[j]) for i, j in combinations(range(len(hash_list)), 2)]
    print(f"len tasks : {len(tasks)}")

    # Create meta-graph as an identity matrix
    graph_num = len(hash_list)
    meta_graph = torch.eye(graph_num)

    # Parallel processing of the edge check
    results = Parallel(n_jobs=-1)(delayed(_check_node)(task) for task in tasks)

    # Update the meta-graph with the results
    for res in results:
        if res is not None:
            i, j = res
            meta_graph[i, j] = 1
            meta_graph[j, i] = 1

    meta_graph = meta_graph.bool()

    # save meta_graph
    with open(save_path, "wb") as f:
        pickle.dump(meta_graph, f, protocol=pickle.HIGHEST_PROTOCOL)
    gc.collect()


def extract_clusters(meta_graph_path, unique_graph_path, save_path):
    """
    Given a meta_graph (a torch.bool matrix where meta_graph[i,j] is True if graph i and j share an edge)
    and a list of graph identifiers (hash_list) corresponding to the rows/columns of meta_graph,
    this function returns a list of clusters, each cluster being a set of graph identifiers.
    """

    # Load the meta-graph
    with open(meta_graph_path, "rb") as f:
        meta_graph = pickle.load(f)

    # Load the unique graphs
    with open(unique_graph_path, "rb") as f:
        unique_graph = pickle.load(f)

    # # Compute the number of nodes per graph
    # hash_to_len = {h: len(graph.nodes()) for h, graph in unique_graph.items()}
    # hash_list = sorted(unique_graph.keys())

    # # Select only graphs with more than one node
    # not_len1_list = [h for h in hash_list if hash_to_len[h] != 1]
    # num_len1 = len(hash_list) - len(not_len1_list)
    # not_len1 = len(not_len1_list)

    # # fix meta_graph
    # num_hash = len(hash_list)
    # meta_graph_fixed = torch.eye(num_hash).bool()
    # meta_graph_fixed[num_len1:,num_len1:] = meta_graph[:not_len1, :not_len1]

    hash_list = sorted(unique_graph.keys())

    # Convert the torch tensor to a numpy array
    # meta_graph_np = meta_graph_fixed.numpy()
    meta_graph_np = meta_graph.numpy()

    # Create an undirected graph from the numpy array
    G = nx.from_numpy_array(meta_graph_np)

    # Get connected components as sets of indices
    clusters_indices = list(nx.connected_components(G))

    # Map the indices back to your graph hash identifiers
    clusters = [{hash_list[i] for i in comp} for comp in clusters_indices]

    # Save the clusters
    with open(save_path, "wb") as f:
        pickle.dump(clusters, f, protocol=pickle.HIGHEST_PROTOCOL)

    # save as txt
    save_txt = "./level2_cluster.txt"
    with open(save_txt, "w") as f:
        for cluster in clusters:
            f.write(f"{cluster}\n")

    breakpoint()

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


if __name__ == "__main__":
    # graph_dir = '/data/psk6950/PDB_2024Mar18/protein_graph/'
    # level0_save_path1 = '/data/psk6950/PDB_2024Mar18/protein_graph/level0_cluster.csv'
    # level0_save_path2 = '/data/psk6950/PDB_2024Mar18/protein_graph/unique_graphs.pkl'
    # level0_graph_cluster(graph_dir, level0_save_path1, level0_save_path2)

    # level1_pickle_path = '/data/psk6950/PDB_2024Mar18/protein_graph/unique_graphs.pkl'
    # level1_save_path = '/data/psk6950/PDB_2024Mar18/protein_graph/level1_meta_graph.pkl'
    # level1_graph_cluster(level1_pickle_path, level1_save_path)

    # meta_graph_path = '/data/psk6950/PDB_2024Mar18/protein_graph/level1_meta_graph.pkl'
    # unique_graph_path = '/data/psk6950/PDB_2024Mar18/protein_graph/unique_graphs.pkl'
    # # clusters = extract_clusters(meta_graph_path, unique_graph_path)

    # unittest_level1_graph(meta_graph_path, unique_graph_path)

    # unique_graph_pickle_path = '/data/psk6950/PDB_2024Mar18/protein_graph/unique_graphs.pkl'
    # level2_save_path = '/data/psk6950/PDB_2024Mar18/protein_graph/level2_meta_graph.pkl'
    # level2_graph_cluster(unique_graph_pickle_path, level2_save_path)

    # meta_graph_path = '/data/psk6950/PDB_2024Mar18/protein_graph/level2_meta_graph.pkl'
    # unique_graph_path = '/data/psk6950/PDB_2024Mar18/protein_graph/unique_graphs.pkl'
    # level2_cluster_path = '/data/psk6950/PDB_2024Mar18/protein_graph/level2_cluster.pkl'
    # clusters = extract_clusters(meta_graph_path, unique_graph_path, level2_cluster_path)

    # graph_hash_to_graph_cluster_path = '/data/psk6950/PDB_2024Mar18/cluster/graph_hash_to_graph_cluster.txt'
    # train_graph_hash_path = '/data/psk6950/PDB_2024Mar18/cluster/train_graph_hash.txt'
    # valid_graph_hash_path = '/data/psk6950/PDB_2024Mar18/cluster/valid_graph_hash.txt'
    # separate_graphs('/data/psk6950/PDB_2024Mar18/cluster/level2_cluster.txt',
    #                 graph_hash_to_graph_cluster_path,
    #                 train_graph_hash_path,
    #                 valid_graph_hash_path)
    level0_save_path1 = "/data/psk6950/PDB_2024Mar18/protein_graph/level0_cluster.csv"
    PDBID_to_graph_hash_path = (
        "/data/psk6950/PDB_2024Mar18/cluster/PDBID_to_graph_hash.txt"
    )
    write_PDBID_to_graph_hash(level0_save_path1, PDBID_to_graph_hash_path)
