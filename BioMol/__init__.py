import os
from pathlib import Path
import json

# Locate the package directory and config file
MODULE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = MODULE_DIR / "configs" / "datapath.json"

ALL_TYPE_CONFIG_PATH = MODULE_DIR / "configs" / "types" / "base.json"
PROTEIN_ONLY_CONFIG_PATH = MODULE_DIR / "configs" / "types" / "protein_only.json"

# 1) Ensure config file exists
if not CONFIG_PATH.is_file():
    raise ImportError(
        f"🐛 BioMol configuration file not found: {CONFIG_PATH}\n"
        "Please run:\n"
        "  biomol configure --config-file /path/to/datapath.json"
    )

# 2) Load and parse JSON
try:
    with open(CONFIG_PATH) as f:
        _cfg = json.load(f)
except json.JSONDecodeError as err:
    raise ImportError(f"🐛 Invalid JSON in {CONFIG_PATH}: {err}") from err

# 3) Validate required keys
_required = ["CCD_PATH", "DB_PATH"]
_missing = [k for k in _required if k not in _cfg]
if _missing:
    raise ImportError(
        "🐛 Missing required key(s) in BioMol config: " + ", ".join(_missing) + "\n"
        "Please update datapath.json and re-run `biomol configure`."
    )

# 4) Define data paths
DB_PATH               = _cfg["DB_PATH"]
CCD_PATH              = _cfg["CCD_PATH"]
CONTACT_GRAPH_PATH    = os.path.join(DB_PATH, "protein_graph")
MSA_PATH              = os.path.join(DB_PATH, "a3m")
CIF_PATH              = os.path.join(DB_PATH, "cif")
SEQ_TO_HASH_PATH      = os.path.join(DB_PATH, "entity", "sequence_hashes.pkl")
GRAPH_HASH_PATH       = os.path.join(DB_PATH, "protein_graph", "level0_cluster.csv")
GRAPH_CLUSTER_PATH    = os.path.join(DB_PATH, "cluster", "graph_hash_to_graph_cluster.txt")  # noqa: E501
MSADB_PATH            = os.path.join(DB_PATH, "MSA.lmdb")
# CIFDB_PATH            = os.path.join(DB_PATH, "cif_protein_only.lmdb")
IDEAL_LIGAND_PATH     = os.path.join(DB_PATH, "metadata", "ideal_ligand_list.pkl")
SIGNALP_PATH          = os.path.join(DB_PATH, "signalp")
CCD_DB_PATH           = os.path.join(CCD_PATH, "ligand_info.lmdb")

# 5) Verify each path exists
__all__ = [
    "ALL_TYPE_CONFIG_PATH",
    "PROTEIN_ONLY_CONFIG_PATH",
    "CONTACT_GRAPH_PATH",
    "MSA_PATH",
    "DB_PATH",
    "SEQ_TO_HASH_PATH",
    "GRAPH_HASH_PATH",
    "GRAPH_CLUSTER_PATH",
    "MSADB_PATH",
    # "CIFDB_PATH",
    # "CCD_DB_PATH",
    "IDEAL_LIGAND_PATH",
    "SIGNALP_PATH",
]

for _name in __all__:
    _path = globals()[_name]
    if not os.path.exists(_path):
        raise FileNotFoundError(
            f"🐛 Required data path '{_name}' not found: {_path}\n"
            "Please check your datapath.json and directory layout."
        )
