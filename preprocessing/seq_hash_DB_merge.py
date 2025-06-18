#!/usr/bin/env python3
"""
Merge multiple LMDB shard environments into a single consolidated LMDB.
Each shard directory should have keys unique across shards.
"""

import os
import lmdb
import argparse


def merge_shards(shard_dirs, output_env, map_size=None):
    """
    Merge LMDB shards into a single environment.

    Args:
        shard_dirs (list[str]): Paths to shard LMDB directories.
        output_env (str): Path to the merged LMDB directory (will be created).
        map_size (int, optional): Maximum size of the output database in bytes.
                                  If None, defaults to sum of shard sizes * 2.
    """
    # Estimate total size if not provided
    if map_size is None:
        total_bytes = 0
        for shard in shard_dirs:
            data_file = os.path.join(shard, "data.mdb")
            if os.path.exists(data_file):
                total_bytes += os.path.getsize(data_file)
        # provide headroom
        map_size = total_bytes * 2 if total_bytes > 0 else 1 * 1024**4

    os.makedirs(output_env, exist_ok=True)
    env_out = lmdb.open(output_env, map_size=map_size)

    # Iterate over each shard and copy entries
    with env_out.begin(write=True) as txn_out:
        for shard in shard_dirs:
            print(f"Merging shard: {shard}")
            env_in = lmdb.open(shard, readonly=True, lock=False)
            with env_in.begin() as txn_in:
                cursor = txn_in.cursor()
                for key, value in cursor:
                    # Overwrite or skip if key exists?
                    txn_out.put(key, value)
            env_in.close()
    env_out.close()
    print(f"Merged {len(shard_dirs)} shards into {output_env}")


def find_shards(base_path):
    """
    Find all shard directories under base_path matching pattern '*.shard_*'.
    """
    return sorted(
        os.path.join(base_path, d)
        for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d)) and "shard_" in d
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge LMDB shard directories into one LMDB environment"
    )
    parser.add_argument(
        "base_env",
        help="Base LMDB path (e.g., cif_protein_only.lmdb) that contains shard subdirectories",
    )
    parser.add_argument(
        "output_env", help="Target LMDB directory for the merged database"
    )
    parser.add_argument(
        "--map-size",
        type=int,
        default=None,
        help="Optional map size in bytes for the merged LMDB",
    )
    args = parser.parse_args()

    shards = find_shards(args.base_env)
    if not shards:
        print(f"No shard directories found. Exiting. {args.base_env}")
        exit(1)

    merge_shards(shards, args.output_env, map_size=args.map_size)

    # map_size : 2
    # python seq_hash_DB_merge.py /data/psk6950/BioMolDB_2024Oct21/seq_to_str/ /data/psk6950/BioMolDB_2024Oct21/seq_to_str/atom.lmdb --map-size 2000000000000
