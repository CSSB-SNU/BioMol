import pickle
import re
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

# Regex pattern to capture the three groups inside the tuple
pattern = re.compile(r"\('([^']+)', '([^']+)', '([^']+)'\)")

def get_unique_graphs(unique_graph_path):
    with open(unique_graph_path, "rb") as f:
        unique_graphs = pickle.load(f)
    return unique_graphs


def get_frequency(unique_graph_path, node_save_path, edge_save_path):
    unique_graphs = get_unique_graphs(unique_graph_path)

    node_score_dict = {}
    edge_score_dict = {}

    # sort by graph_hash
    unique_graphs = dict(sorted(unique_graphs.items(), key=lambda x: x[0]))

    for graph_hash, graph in unique_graphs.items():
        node_num = len(graph.nodes())
        edge_num = len(graph.edges())
        temp_node_label_data = []
        temp_edge_label_data = []
        for edge in graph.edges():
            src_label = graph.nodes[edge[0]]["label"]
            tgt_label = graph.nodes[edge[1]]["label"]
            edge_label = tuple(sorted((src_label, tgt_label)))
            if edge_label in temp_edge_label_data:
                continue
            edge_score = 1/ (edge_num)
            if edge_label not in edge_score_dict:
                edge_score_dict[edge_label] = edge_score
            else:
                edge_score_dict[edge_label] += edge_score
            temp_edge_label_data.append(edge_label)

        for node in graph.nodes():
            node_label = graph.nodes[node]["label"]
            if node_label in temp_node_label_data:
                continue
            node_score = 1 / (node_num)
            if node_label not in node_score_dict:
                node_score_dict[node_label] = node_score
            else:
                node_score_dict[node_label] += node_score
            temp_node_label_data.append(node_label)
        breakpoint()

    # print max and min edge score and its ID
    max_node, max_node_score = max(node_score_dict.items(), key=lambda kv: kv[1])
    min_node, min_node_score = min(node_score_dict.items(), key=lambda kv: kv[1])
    print(f"Max-scoring node {max_node!r} → {max_node_score:.6f}")
    print(f"Min-scoring node {min_node!r} → {min_node_score:.6f}")

    max_edge, max_edge_score = max(edge_score_dict.items(), key=lambda kv: kv[1])
    min_edge, min_edge_score = min(edge_score_dict.items(), key=lambda kv: kv[1])
    print(f"Max-scoring edge {max_edge!r} → {max_edge_score:.6f}")
    print(f"Min-scoring edge {min_edge!r} → {min_edge_score:.6f}")

    # save node_freq, edge_score_dict
    with open(node_save_path, "wb") as f:
        pickle.dump(node_score_dict, f)

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
    unique_graphs_path = "/public_data/BioMolDB_2024Oct21/protein_graph/unique_graphs.pkl"
    node_save_path = "/public_data/BioMolDB_2024Oct21/protein_graph/node_score.pkl"
    edge_save_path = "/public_data/BioMolDB_2024Oct21/protein_graph/edge_score.pkl"
    get_frequency(unique_graphs_path, node_save_path, edge_save_path)
    # node_path = "./node_freq.png"
    # edge_path = "./edge_score_dict.png"
    # plot_histogram(node_save_path, edge_save_path, node_path, edge_path)
