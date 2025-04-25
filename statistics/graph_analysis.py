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
import matplotlib.pyplot as plt

# Regex pattern to capture the three groups inside the tuple
pattern = re.compile(r"\('([^']+)', '([^']+)', '([^']+)'\)")


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


def get_frequency(graph_dir, node_save_path, edge_save_path, graph_save_path):
    # 1. Collect all graph file paths.
    graph_files = []
    for root, dirs, files in os.walk(graph_dir):
        for file in files:
            if file.endswith(".graph"):
                graph_files.append(os.path.join(root, file))

    # 2. Parse graphs in parallel.
    # results = Parallel(n_jobs=-1)(delayed(parse_graph)(graph_path) for graph_path in graph_files)
    # graph_files = graph_files[:10]
    results = Parallel(n_jobs=1)(
        delayed(parse_graph)(graph_path) for graph_path in graph_files
    )
    graph_dict = {}
    for d in results:
        graph_dict.update(d)

    pdb_ID_to_graph_num = {}
    for k in graph_dict.keys():
        pdb_ID = k.split("_")[0]
        if pdb_ID not in pdb_ID_to_graph_num:
            pdb_ID_to_graph_num[pdb_ID] = 1
        else:
            pdb_ID_to_graph_num[pdb_ID] += 1

    node_freq = {}
    edge_freq = {}
    graph_freq = {}
    for k, v in graph_dict.items():
        graph_num = pdb_ID_to_graph_num[k.split("_")[0]]
        temp_label_data = []
        for node in v.nodes(data=True):
            label = node[1]["label"]
            if label in temp_label_data:
                continue
            if label not in node_freq:
                node_freq[label] = 1 / graph_num
            else:
                node_freq[label] += 1 / graph_num
            temp_label_data.append(label)
        temp_label_data = []
        for edge in v.edges():
            src_label = v.nodes[edge[0]]["label"]
            tgt_label = v.nodes[edge[1]]["label"]
            edge_label = (src_label, tgt_label)
            if edge_label in temp_label_data:
                continue
            if edge_label not in edge_freq:
                edge_freq[edge_label] = 1 / graph_num
            else:
                edge_freq[edge_label] += 1 / graph_num
            temp_label_data.append(edge_label)

        graph_hash = nx.weisfeiler_lehman_graph_hash(v, node_attr="label")
        if graph_hash not in graph_freq:
            graph_freq[graph_hash] = (1 / graph_num, v, [k])
        else:
            graph_freq[graph_hash] = (
                graph_freq[graph_hash][0] + 1 / graph_num,
                v,
                graph_freq[graph_hash][2] + [k],
            )

    # save node_freq, edge_freq
    with open(node_save_path, "wb") as f:
        pickle.dump(node_freq, f)
    with open(edge_save_path, "wb") as f:
        pickle.dump(edge_freq, f)
    with open(graph_save_path, "wb") as f:
        pickle.dump(graph_freq, f)


def plot_histogram(node_pickle_path, edge_pickle_path, node_path, edge_path):
    with open(node_pickle_path, "rb") as f:
        node_freq = pickle.load(f)
    with open(edge_pickle_path, "rb") as f:
        edge_freq = pickle.load(f)

    # x axis : frequencty of node/edge
    # y axis : frequency of frequency

    node_x = list(node_freq.values())

    node_y = {"<=1": 0, "1~10": 0, "10~100": 0, "100~1000": 0, "1000~": 0}
    for v in node_x:
        if v <= 1:
            node_y["<=1"] += 1
        elif v > 1 and v <= 10:
            node_y["1~10"] += 1
        elif v > 10 and v <= 100:
            node_y["10~100"] += 1
        elif v > 100 and v <= 1000:
            node_y["100~1000"] += 1
        else:
            node_y["1000~"] += 1
    # sort by x
    plt.bar(list(node_y.keys()), list(node_y.values()), align="center")
    plt.xlabel("Frequency of Node")
    plt.ylabel("Frequency of Frequency")
    plt.savefig("./node_freq.png")
    plt.close()

    edge_x = list(edge_freq.values())
    # group x 1, 2~10, 11~100, 101~1000, 1001~

    edge_y = {"<=1": 0, "1~10": 0, "10~100": 0, "100~1000": 0, "1000~": 0}
    for v in edge_x:
        if v <= 1:
            edge_y["<=1"] += 1
        elif v > 1 and v <= 10:
            edge_y["1~10"] += 1
        elif v > 10 and v <= 100:
            edge_y["10~100"] += 1
        elif v > 100 and v <= 1000:
            edge_y["100~1000"] += 1
        else:
            edge_y["1000~"] += 1
    plt.bar(list(edge_y.keys()), list(edge_y.values()), align="center")
    plt.xlabel("Frequency of Edge")
    plt.ylabel("Frequency of Frequency")
    plt.savefig("./edge_freq.png")
    plt.close()


def get_most_frequent_item(
    node_pickle_path,
    edge_pickle_path,
    graph_pickle_path,
    chain_ID_to_cluster_path,
    top_k,
):
    with open(node_pickle_path, "rb") as f:
        node_freq = pickle.load(f)
    with open(edge_pickle_path, "rb") as f:
        edge_freq = pickle.load(f)
    with open(graph_pickle_path, "rb") as f:
        graph_freq = pickle.load(f)
    with open(chain_ID_to_cluster_path, "rb") as f:
        chain_ID_to_cluster = pickle.load(f)

    cluster_to_chain_ID_list = {}
    for chain_ID, cluster in chain_ID_to_cluster.items():
        if cluster not in cluster_to_chain_ID_list:
            cluster_to_chain_ID_list[cluster] = [chain_ID]
        else:
            cluster_to_chain_ID_list[cluster].append(chain_ID)

    node_freq_list = sorted(node_freq.items(), key=lambda x: x[1], reverse=True)
    edge_freq_list = sorted(edge_freq.items(), key=lambda x: x[1], reverse=True)
    graph_freq_list = sorted(graph_freq.items(), key=lambda x: x[1][0], reverse=True)

    frequent_nodes = [node[0] for node in node_freq_list[:top_k]]
    frequent_edges = [edge[0] for edge in edge_freq_list[:top_k]]
    frequent_graphs = []
    for graph in graph_freq_list:
        graph, pdb_id_list = graph[1][1], graph[1][2]
        if len(graph.nodes()) == 1:
            continue
        frequent_graphs.append((graph, pdb_id_list))
        if len(frequent_graphs) == top_k:
            break
    frequent_chain_IDs = [cluster_to_chain_ID_list[int(i)] for i in frequent_nodes]
    frequent_edge_chain_IDs = {}
    for rank, edge in enumerate(frequent_edges):
        src, tgt = edge
        src_chain_IDs = cluster_to_chain_ID_list[int(src)]
        tgt_chain_IDs = cluster_to_chain_ID_list[int(tgt)]
        frequent_edge_chain_IDs[rank] = [
            (src_chain, tgt_chain)
            for src_chain, tgt_chain in zip(src_chain_IDs, tgt_chain_IDs)
        ]

    frequent_graph_chain_IDs = {}
    # for rank, graph in enumerate(frequent_graphs):
    breakpoint()
    # graph_freq = sorted(graph_freq.items(), key=lambda x: x[1], reverse=True)
    # return node_freq[:top_k], edge_freq[:top_k], graph_freq[:top_k]


def test_most_frequent_node(frequent_node_label="37802"):
    # check the number of graph including the most frequent node
    graph_dir = "/data/psk6950/PDB_2024Mar18/protein_graph/"
    graph_files = []
    for root, dirs, files in os.walk(graph_dir):
        for file in files:
            if file.endswith(".graph"):
                graph_files.append(os.path.join(root, file))
    results = Parallel(n_jobs=1)(
        delayed(parse_graph)(graph_path) for graph_path in graph_files
    )
    graph_dict = {}
    for d in results:
        graph_dict.update(d)
    count = 0
    for k, v in graph_dict.items():
        for node in v.nodes(data=True):
            label = node[1]["label"]
            if label == frequent_node_label:
                count += 1
                print(k)
                break
    print(count)


def has_common_node(G1, G2):
    # same node label
    labels_G1 = {data.get("label") for node, data in G1.nodes(data=True)}
    labels_G2 = {data.get("label") for node, data in G2.nodes(data=True)}
    if labels_G1.intersection(labels_G2):
        return True
    return False


def is_connected(matrix):
    n = matrix.shape[0]
    visited = np.zeros(n, dtype=bool)
    stack = [0]  # start from node 0

    while stack:
        node = stack.pop()
        if not visited[node]:
            visited[node] = True
            for neighbor in range(n):
                if matrix[node, neighbor] == 1 and not visited[neighbor]:
                    stack.append(neighbor)
    return visited.all()


def compute_connection(i, j, G1, G2):
    # Return tuple with indices and connection (1 if common node, 0 otherwise)
    return i, j, 1 if has_common_node(G1, G2) else 0


def unittest_level2_Cluster(level0_save_path2, level2_cluster_path):
    connectivity_matrix_path = "./biggest_cluster_connectivity.npy"
    if not os.path.exists(connectivity_matrix_path):
        # Load level 0 graphs from file.
        with open(level0_save_path2, "rb") as f:
            level0_graphs = pickle.load(f)

        # Read and parse level 2 clusters from file.
        with open(level2_cluster_path, "r") as f:
            lines = f.readlines()
        level2_clusters = []
        for line in lines:
            line = line.strip()
            line = line[1:-1]  # Remove surrounding brackets.
            line = line.split(",")
            line = [i.strip() for i in line]
            level2_clusters.append(line)

        # Sort clusters by the number of graphs (largest first) and choose the biggest cluster.
        level2_clusters = sorted(level2_clusters, key=lambda x: len(x), reverse=True)
        biggest_cluster = level2_clusters[0]
        n_cluster = len(biggest_cluster)

        graph_list = [level0_graphs[int(i)] for i in biggest_cluster]

        # Initialize connectivity matrix.
        connectivity = np.zeros((n_cluster, n_cluster))

        # Prepare list of pairs (only upper triangle indices since the matrix is symmetric).
        pairs = [
            (i, j, graph_list[i], graph_list[j])
            for i, j in combinations(range(n_cluster), 2)
        ]

        # Compute connections in parallel.
        cpu_num = os.cpu_count()
        print(f"CPU NUM: {cpu_num}")
        print(f"Number of pairs: {len(pairs)}")
        results = Parallel(n_jobs=cpu_num)(
            delayed(compute_connection)(i, j, graph1, graph2)
            for i, j, graph1, graph2 in pairs
        )

        # Fill the connectivity matrix with the results.
        for i, j, conn in results:
            connectivity[i, j] = conn
            connectivity[j, i] = conn  # Symmetric matrix

        # Save the connectivity matrix.
        np.save("./biggest_cluster_connectivity.npy", connectivity)

    else:
        # Load the connectivity matrix from file.
        connectivity = np.load(connectivity_matrix_path)
        print("Loaded connectivity matrix from file.")
    # Check if the biggest cluster is connected (not divisible)
    print(is_connected(connectivity))
    diameter = calculate_graph_diameter(connectivity)
    print(f"Diameter of the biggest cluster: {diameter}")


def check_unittest():
    connectivity = np.load("./biggest_cluster_connectivity.npy")
    breakpoint()


def calculate_graph_diameter(adj_matrix):
    """
    Calculate the graph diameter from an adjacency matrix.

    The diameter is defined as the longest shortest-path distance (in number
    of edges) between any pair of nodes in the graph.

    Parameters:
        adj_matrix (np.ndarray): A symmetric connectivity matrix where an entry of 1
                                 indicates a direct connection between nodes.

    Returns:
        int: The graph diameter.
    """
    n = adj_matrix.shape[0]

    def bfs_max_distance(start):
        # Initialize distances as -1 (unvisited)
        distances = [-1] * n
        distances[start] = 0
        queue = [start]
        while queue:
            current = queue.pop(0)
            for neighbor in range(n):
                if adj_matrix[current, neighbor] == 1 and distances[neighbor] == -1:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        return max(distances)

    # Use parallel computation to run BFS from every node.
    max_distances = Parallel(n_jobs=-1)(delayed(bfs_max_distance)(i) for i in range(n))
    diameter = max(max_distances)
    return diameter


if __name__ == "__main__":
    graph_dir = "/data/psk6950/PDB_2024Mar18/protein_graph/"
    node_save_path = "/data/psk6950/PDB_2024Mar18/protein_graph/node_freq.pkl"
    edge_save_path = "/data/psk6950/PDB_2024Mar18/protein_graph/edge_freq.pkl"
    graph_save_path = "/data/psk6950/PDB_2024Mar18/protein_graph/graph_freq.pkl"
    node_path = "./node_freq.png"
    edge_path = "./edge_freq.png"
    # get_frequency(graph_dir, node_save_path, edge_save_path, graph_save_path)
    # plot_histogram(node_save_path, edge_save_path, node_path, edge_path)

    chain_ID_to_cluster_path = (
        "/data/psk6950/PDB_2024Mar18/protein_seq_clust/v2_chainID_to_cluster.pkl"
    )

    # get_most_frequent_item(node_save_path, edge_save_path, graph_save_path, chain_ID_to_cluster_path, 10)
    # test_most_frequent_node()
    level0_save_path1 = "/data/psk6950/PDB_2024Mar18/protein_graph/level0_cluster.csv"
    level0_save_path2 = "/data/psk6950/PDB_2024Mar18/protein_graph/unique_graphs.pkl"
    unittest_level2_Cluster(level0_save_path2, "./level2_cluster.txt")
    # check_unittest()
