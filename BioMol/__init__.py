import os
from pathlib import Path
import json

MODULE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = MODULE_DIR / "configs" / "datapath.json"
ALL_TYPE_CONFIG_PATH = MODULE_DIR / "configs" / "types" / "base.json"
PROTEIN_ONLY_CONFIG_PATH = MODULE_DIR / "configs" / "types" / "protein_only.json"

# don’t touch the file system at import time:
_cfg = None


def _load_config():
    global _cfg
    if _cfg is None:
        if not CONFIG_PATH.is_file():
            raise ImportError(
                f"🐛 BioMol config file not found: {CONFIG_PATH}\n"
                "Please run:\n"
                "    biomol configure --config-file /path/to/datapath.json"
            )
        try:
            data = json.loads(CONFIG_PATH.read_text())
        except json.JSONDecodeError as err:
            raise ImportError(f"🐛 Invalid JSON in {CONFIG_PATH}: {err}") from err

        for key in ("DB_PATH", "CCD_PATH"):
            if key not in data:
                raise ImportError(f"🐛 Missing '{key}' in {CONFIG_PATH}")
        _cfg = data
    return _cfg


def get_paths():
    """Return a dict of all the BioMol data paths."""
    cfg = _load_config()
    base = cfg["DB_PATH"]
    ccd = cfg["CCD_PATH"]
    return {
        "DB_PATH": base,
        "CCD_PATH": ccd,
        "CONTACT_GRAPH_PATH": os.path.join(base, "contact_graphs"),
        "SEQ_TO_HASH_PATH": os.path.join(base, "entity", "sequence_hashes.pkl"),

        # "GRAPH_HASH_PATH": os.path.join(base, "protein_graph", "level0_cluster.csv"),  # noqa: E501
        # "GRAPH_CLUSTER_PATH": os.path.join(
        #     base, "cluster/graph_hash_to_graph_cluster.txt"
        # ),  # noqa: E501
        "A3MDB_PATH": os.path.join(base, "a3m.lmdb"), 
        "MSADB_PATH": os.path.join(base, "MSA.lmdb"),
        "CIFDB_PATH": os.path.join(base, "cif_protein_only.lmdb"),
        "IDEAL_LIGAND_PATH": os.path.join(base, "metadata/ideal_ligand_list.pkl"),
        "SIGNALP_PATH": os.path.join(base, "signalp"),
        "CCD_DB_PATH": os.path.join(ccd, "ligand_info.lmdb"),
    }


# Optionally, for convenience, you can expose attributes directly:
def __getattr__(name):
    paths = get_paths()
    if name in paths:
        return paths[name]
    if name in ("ALL_TYPE_CONFIG", "PROTEIN_ONLY_CONFIG"):
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
