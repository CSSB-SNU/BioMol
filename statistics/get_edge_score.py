import pickle
import os
import networkx as nx
import re
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

# Regex pattern to capture the three groups inside the tuple
pattern = re.compile(r"\('([^']+)', '([^']+)', '([^']+)'\)")


def format_tuple(s):
    match = pattern.search(s)
    if match:
        # Combine the captured groups with underscores
        return f"{match.group(1)}_{match.group(2)}_{match.group(3)}"
    return None


def parse_graph(file_path: str) -> list[nx.Graph]:
    graph_ID = file_path.split("/")[-1].split(".")[0]
    G_dict = {}
    G = nx.Graph()  # Use nx.DiGraph() for directed graphs
    ID = None
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
    for root, _, files in os.walk(graph_dir):
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

    pdb_ID_to_graph_num = {}
    for k in graph_dict.keys():
        pdb_ID = k.split("_")[0]
        if pdb_ID not in pdb_ID_to_graph_num:
            pdb_ID_to_graph_num[pdb_ID] = 1
        else:
            pdb_ID_to_graph_num[pdb_ID] += 1

    edge_score_dict = {}
    for k, v in graph_dict.items():
        graph_num = pdb_ID_to_graph_num[k.split("_")[0]]
        edge_num = len(v.edges())
        temp_label_data = []
        for edge in v.edges():
            src_label = v.nodes[edge[0]]["label"]
            tgt_label = v.nodes[edge[1]]["label"]
            edge_label = (src_label, tgt_label)
            if edge_label in temp_label_data:
                continue
            edge_score = 1/ (edge_num*graph_num)
            if edge_label not in edge_score_dict:
                edge_score_dict[edge_label] = edge_score
            else:
                edge_score_dict[edge_label] += edge_score
            temp_label_data.append(edge_label)

    # print max and min edge score and its ID
    max_edge, max_score = max(edge_score_dict.items(), key=lambda kv: kv[1])
    min_edge, min_score = min(edge_score_dict.items(), key=lambda kv: kv[1])
    print(f"Max-scoring edge {max_edge!r} → {max_score:.6f}")
    print(f"Min-scoring edge {min_edge!r} → {min_score:.6f}")

    # save node_freq, edge_score_dict
    with open(edge_save_path, "wb") as f:
        pickle.dump(edge_score_dict, f)

# def plot_histogram(node_pickle_path, edge_pickle_path, node_path, edge_path):
#     with open(edge_pickle_path, "rb") as f:
#         edge_score_dict = pickle.load(f)

#     # x axis : frequencty of node/edge
#     # y axis : frequency of frequency

#     node_x = list(node_freq.values())

#     node_y = {"<=1": 0, "1~10": 0, "10~100": 0, "100~1000": 0, "1000~": 0}
#     for v in node_x:
#         if v <= 1:
#             node_y["<=1"] += 1
#         elif v > 1 and v <= 10:
#             node_y["1~10"] += 1
#         elif v > 10 and v <= 100:
#             node_y["10~100"] += 1
#         elif v > 100 and v <= 1000:
#             node_y["100~1000"] += 1
#         else:
#             node_y["1000~"] += 1
#     # sort by x
#     plt.bar(list(node_y.keys()), list(node_y.values()), align="center")
#     plt.xlabel("Frequency of Node")
#     plt.ylabel("Frequency of Frequency")
#     plt.savefig("./node_freq.png")
#     plt.close()

#     edge_x = list(edge_score_dict.values())
#     # group x 1, 2~10, 11~100, 101~1000, 1001~

#     edge_y = {"<=1": 0, "1~10": 0, "10~100": 0, "100~1000": 0, "1000~": 0}
#     for v in edge_x:
#         if v <= 1:
#             edge_y["<=1"] += 1
#         elif v > 1 and v <= 10:
#             edge_y["1~10"] += 1
#         elif v > 10 and v <= 100:
#             edge_y["10~100"] += 1
#         elif v > 100 and v <= 1000:
#             edge_y["100~1000"] += 1
#         else:
#             edge_y["1000~"] += 1
#     plt.bar(list(edge_y.keys()), list(edge_y.values()), align="center")
#     plt.xlabel("Frequency of Edge")
#     plt.ylabel("Frequency of Frequency")
#     plt.savefig("./edge_score_dict.png")
#     plt.close()


if __name__ == "__main__":
    graph_dir = "/data/psk6950/PDB_2024Oct21/protein_graph/"
    node_save_path = "/data/psk6950/PDB_2024Oct21/protein_graph/node_freq.pkl"
    edge_save_path = "/data/psk6950/PDB_2024Oct21/protein_graph/edge_score_dict.pkl"
    graph_save_path = "/data/psk6950/PDB_2024Oct21/protein_graph/graph_freq.pkl"
    node_path = "./node_freq.png"
    edge_path = "./edge_score_dict.png"
    get_frequency(graph_dir, node_save_path, edge_save_path, graph_save_path)
    # plot_histogram(node_save_path, edge_save_path, node_path, edge_path)
