#!/usr/bin/env python3
"""
Multi-node LMDB ingestion for CIF files using SLURM Array and per-task shards.
Each task processes a distinct subset of CIF files and writes to its own LMDB shard.
"""
import os
import math
import lmdb
import torch
import pickle
import gzip
from joblib import Parallel, delayed

from BioMol.utils.parser import parse_cif
from BioMol import DB_PATH

# Base LMDB environment path for storing parsed CIF assemblies
DB_ENV = f"{DB_PATH}/cif_all_molecules.lmdb"
# Directory containing .cif.gz files
CIF_DIR = f"{DB_PATH}/cif/"
# Path to CIF parsing configuration
CIF_CONFIG_PATH = "./BioMol/configs/types/base.json"
# Whether to remove signal peptide during parsing
REMOVE_SIGNAL_PEPTIDE = True


def already_parsed(env_path: str) -> set[str]:
    """
    Retrieve keys already present in the LMDB database to avoid reprocessing.
    """
    try:
        env = lmdb.open(env_path, readonly=True, lock=False)
        with env.begin() as txn:
            keys = {key.decode() for key, _ in txn.cursor()}
        env.close()
        return keys
    except lmdb.Error as e:
        print(f"No existing LMDB database found at {env_path}: {e}")
        return set()


def get_all_cif_files(cif_dir: str) -> list[str]:
    """
    Walk through the CIF directory and collect all .cif.gz file paths.
    """
    files = []
    for root, _, filenames in os.walk(cif_dir):
        for fname in filenames:
            if fname.endswith(".cif.gz"):
                files.append(os.path.join(root, fname))
    return sorted(files)


def process_file(cif_path: str) -> tuple[str, bytes]:
    """
    Parse a single CIF file and return its hash plus compressed serialized bioassembly.
    """
    # Use a single thread per task for safe parallel execution
    torch.set_num_threads(1)
    cif_hash = os.path.basename(cif_path).split(".")[0]
    bioassembly = parse_cif(cif_path, CIF_CONFIG_PATH, REMOVE_SIGNAL_PEPTIDE)
    serialized = pickle.dumps(bioassembly, protocol=pickle.HIGHEST_PROTOCOL)
    compressed = gzip.compress(serialized)
    return cif_hash, compressed


def lmdb_cif_multi(env_path: str, cif_dir: str, n_jobs: int = 1) -> None:
    """
    Distribute CIF parsing across SLURM Array tasks, each writing to its own LMDB shard.

    Args:
        env_path: Base path for LMDB environment.
        cif_dir: Directory containing CIF files.
        n_jobs: Number of parallel jobs per task (usually SLURM_CPUS_ON_NODE).
    """
    # Determine already processed keys
    processed = already_parsed(env_path)

    # Gather all CIF files and filter out processed ones
    all_files = get_all_cif_files(cif_dir)
    to_process = [f for f in all_files
                  if os.path.basename(f).split(".")[0] not in processed]
    to_process = sorted(to_process)
    total = len(to_process)

    # Use SLURM Array environment variables to split work
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))
    chunk_size = math.ceil(total / num_tasks)
    start = task_id * chunk_size
    end = min(start + chunk_size, total)
    subset = to_process[start:end]
    print(f"[Task {task_id+1}/{num_tasks}] Processing {len(subset)} of {total} files: indices {start}-{end-1}")

    # Create and open shard-specific LMDB environment
    shard_path = f"{env_path}.shard_{task_id}"
    os.makedirs(shard_path, exist_ok=True)
    env = lmdb.open(shard_path, map_size=1 * 1024**4)

    # Parse and write each CIF in parallel
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_file)(path) for path in subset
    )
    with env.begin(write=True) as txn:
        for key, data in results:
            txn.put(key.encode(), data)
    env.close()

    print(f"[Task {task_id}] Completed shard with {len(subset)} entries.")


if __name__ == "__main__":
    # Use SLURM_CPUS_ON_NODE to automatically set parallel jobs per task
    cpus = int(os.getenv("SLURM_CPUS_ON_NODE", "1"))
    lmdb_cif_multi(DB_ENV, CIF_DIR, n_jobs=cpus)
