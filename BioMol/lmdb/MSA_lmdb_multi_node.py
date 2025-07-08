import os
import math
import lmdb
import pickle
import gzip
from joblib import Parallel, delayed

from BioMol.utils.MSA import MSA
from BioMol import DB_PATH, SEQ_TO_HASH_PATH

# Base LMDB environment path for storing parsed MSA objects
db_env = os.path.join(DB_PATH, "MSA.lmdb")
# Path to sequence-to-hash mapping
seq_to_hash_path = SEQ_TO_HASH_PATH

# Default batch size to avoid OOM
DEFAULT_BATCH_SIZE = 1000


def load_seq_to_hash(path: str = seq_to_hash_path) -> dict[int, str]:
    """
    Load the mapping from sequence IDs to zero-padded hash strings.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def already_parsed(env_path: str) -> set[str]:
    """
    Retrieve keys already present in an LMDB environment to avoid reprocessing.
    """
    try:
        env = lmdb.open(env_path, readonly=True, lock=False)
        with env.begin() as txn:
            keys = {key.decode() for key, _ in txn.cursor()}
        env.close()
        print(f"[Shard] Found {len(keys)} existing keys in {env_path}")
        return keys
    except lmdb.Error:
        print(f"[Shard] No existing LMDB at {env_path}, starting fresh.")
        return set()


def process_hash(seq_hash: str) -> tuple[bytes, bytes]:
    """
    Build the MSA object for a given hash, then pickle+gzip it.
    """
    msa = MSA(seq_hash, use_lmdb=True)
    packed = pickle.dumps(msa, protocol=pickle.HIGHEST_PROTOCOL)
    blob = gzip.compress(packed)
    return seq_hash.encode(), blob


def lmdb_MSA_multi_batch(
    env_path: str = db_env, n_jobs: int = 1, batch_size: int = DEFAULT_BATCH_SIZE
) -> None:
    # 1) Load and format list of all hashes
    seq_to_hash = load_seq_to_hash()
    all_hashes = [str(h).zfill(6) for h in seq_to_hash.values()]
    total = len(all_hashes)
    print(f"[Main] Total MSAs: {total}")

    # 2) Determine task slice via SLURM_ARRAY env vars
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))
    # shard-specific env path
    shard_env = f"{env_path}.shard_{task_id}"
    os.makedirs(shard_env, exist_ok=True)

    # 3) Filter out already-parsed keys in this shard
    parsed = already_parsed(shard_env)
    remaining = [h for h in all_hashes if h not in parsed]
    remaining.sort()
    rem_total = len(remaining)
    print(f"[Task {task_id}] Remaining to write: {rem_total}")

    # 4) Compute this task's subset indices
    chunk = math.ceil(rem_total / num_tasks)
    start = task_id * chunk
    end = min(start + chunk, rem_total)
    subset = remaining[start:end]
    print(
        f"[Task {task_id + 1}/{num_tasks}] "
        f"Processing {len(subset)} hashes (indices {start}-{end - 1})"
    )

    # 5) Open task‐specific LMDB
    env = lmdb.open(shard_env, map_size=(2048 + 128) * 1024**3)

    # 6) Process in batches to avoid OOM
    for batch_start in range(0, len(subset), batch_size):
        batch_end = min(batch_start + batch_size, len(subset))
        batch_hashes = subset[batch_start:batch_end]
        print(
            f"[Task {task_id}] Batch {batch_start}-{batch_end - 1}: {len(batch_hashes)} items"
        )

        # parallel processing for this batch
        results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(process_hash)(h) for h in batch_hashes
        )

        # write this batch in one transaction
        with env.begin(write=True) as txn:
            for key, blob in results:
                txn.put(key, blob)

        # free memory before next batch
        del results

    env.close()
    print(f"[Task {task_id}] Done. Wrote {len(subset)} entries to {shard_env}")


if __name__ == "__main__":
    cpus = int(os.environ.get("SLURM_CPUS_ON_NODE", "1"))
    batch_size = int(os.environ.get("BATCH_SIZE", str(DEFAULT_BATCH_SIZE)))
    lmdb_MSA_multi_batch(db_env, n_jobs=cpus, batch_size=batch_size)
