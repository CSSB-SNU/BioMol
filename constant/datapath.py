import os
# load base path from ./configs/datapath.json
import json
with open("configs/datapath.json", "r") as f:
    base_path = json.load(f)["base_path"]

CONTACT_GRAPH_PATH = os.path.join(base_path, "protein_graph/")
MSA_PATH = os.path.join(base_path, "a3m/")
CIF_PATH = os.path.join(base_path, "cif/")
SEQ_TO_HASH_PATH = os.path.join(base_path, "entity/sequence_hashes.pkl")
GRAPH_HASH_PATH = os.path.join(base_path, "protein_graph/level0_cluster.csv")
GRAPH_CLUSTER_PATH = os.path.join(base_path, "cluster/graph_hash_to_graph_cluster.txt")
MSADB_PATH = os.path.join(base_path, "MSA.lmdb")
CIFDB_PATH = os.path.join(base_path, "cif_protein_only.lmdb")
CCD_DB_PATH = os.path.join(base_path, "ligand_info.lmdb")
IDEAL_LIGAND_PATH = os.path.join(base_path, "metadata/ideal_ligand_list.pkl")
SIGNALP_PATH = os.path.join(base_path, "signalp")

__all__ = [
    "CONTACT_GRAPH_PATH",
    "MSA_PATH",
    "CIF_PATH",
    "SEQ_TO_HASH_PATH",
    "GRAPH_HASH_PATH",
    "GRAPH_CLUSTER_PATH",
    "MSADB_PATH",
    "CIFDB_PATH",
    "CCD_DB_PATH",
    "IDEAL_LIGAND_PATH",
    "SIGNALP_PATH"
]

for path in __all__:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist. Please check the path.")
