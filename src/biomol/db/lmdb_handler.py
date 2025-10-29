from collections.abc import Callable
from pathlib import Path

import lmdb
from joblib import Parallel, delayed

from biomol.core.utils import from_bytes, to_bytes


def already_parsed_keys(env_path: Path) -> set[str]:
    """
    Retrieve all keys from the LMDB database.

    Args:
        env_path: Path to the LMDB environment.
    Returns
    -------
        set
            A set of all keys in the LMDB database.
    """
    env = lmdb.open(str(env_path), readonly=True, lock=False)
    with env.begin() as txn:
        key_set = {
            key.decode() for key in txn.cursor().iternext(keys=True, values=False)
        }
    env.close()
    return key_set


def build_lmdb(
    *data_list: Path,
    env_path: Path,
    recipe: Path,
    parser: Callable,
    n_jobs: int = -1,
    map_size: int = 1e12,  # ~1TB
    ccd_db_path: Path | None = None,
) -> None:
    """
    Build an LMDB database from parsed data.

    Args:
        env_path: Path to the LMDB environment.
        data_list: List of paths to data files to parse.
        parser: Function to parse individual data files.
        n_jobs: Number of parallel jobs for parsing.
        map_size: Maximum size of the LMDB database in bytes.
    """
    env = lmdb.open(str(env_path), map_size=int(map_size))

    def _process_file(data_file: Path) -> tuple[bytes, bytes, Exception | None]:
        """Parse a single file and return (key, compressed_data, error)."""
        key = data_file.stem
        key = key.split(".cif")[0]
        try:
            if ccd_db_path is not None:
                data_dict = parser(recipe, data_file, ccd_db_path=ccd_db_path)
            else:
                data_dict = parser(recipe, data_file)
            zcompressed_data = to_bytes(data_dict)
            return key.encode(), zcompressed_data, None
        except Exception as error:
            return key.encode(), to_bytes({}), error

    # remove UNL
    data_list = [data for data in data_list if data.stem != "UNL"]
    _already_parsed_keys = already_parsed_keys(env_path)
    print(f"Already parsed {len(_already_parsed_keys)} entries.")
    data_list = [
        data
        for data in data_list
        if data.stem.split(".cif")[0] not in _already_parsed_keys
    ]

    # --- Parallel processing ---
    chunk = 10_000
    for i in range(0, len(data_list), chunk):
        print(
            f"Processing files {i} to {min(i + chunk, len(data_list))} / {len(data_list)}"
        )
        data_chunk = data_list[i : i + chunk]
        results = Parallel(n_jobs=n_jobs, verbose=10, prefer="processes")(
            delayed(_process_file)(data_file) for data_file in data_chunk
        )

        # --- Write results to LMDB ---
        with env.begin(write=True) as txn:
            for key, zcompressed_data, error in results:
                if error is not None:
                    # Print error message but continue
                    print(f"Error processing {key.decode()}: {error}")
                    continue
                txn.put(key, zcompressed_data)

    env.close()


def merge_lmdb_shards(
    shard_paths: list[Path],
    merged_env_path: Path,
    map_size: int = int(1e12),
    overwrite: bool = False,
) -> None:
    """
    Merge multiple LMDB shard databases into a single LMDB file.

    Args:
        shard_paths: List of LMDB shard directories to merge.
        merged_env_path: Output LMDB path for the merged database.
        map_size: Maximum size of the merged LMDB in bytes.
        overwrite: Whether to overwrite existing merged file if it exists.
    """
    if merged_env_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{merged_env_path} already exists. Use overwrite=True to replace it."
            )

    # --- Create a new LMDB environment for the merged database ---
    merged_env = lmdb.open(str(merged_env_path), map_size=map_size)

    total_keys = 0

    # --- Iterate through each shard and copy all entries ---
    for shard_path in shard_paths:
        print(f"Merging shard: {shard_path}")
        shard_env = lmdb.open(str(shard_path), readonly=True, lock=False)
        with shard_env.begin() as shard_txn, merged_env.begin(write=True) as merged_txn:
            cursor = shard_txn.cursor()
            for key, value in cursor:
                merged_txn.put(key, value)
                total_keys += 1
        shard_env.close()

    merged_env.sync()
    merged_env.close()

    print(f"[Done] Merged {len(shard_paths)} shards into {merged_env_path}")
    print(f"Total keys merged: {total_keys}")


def read_lmdb(env_path: Path, key: str) -> bytes:
    """
    Read a value from the LMDB database by key.

    Args:
        env_path: Path to the LMDB environment.
        key: Key of the data to retrieve.

    Returns
    -------
        dict
            The data dictionary retrieved from the LMDB database.
    """
    env = lmdb.open(str(env_path), readonly=True, lock=False)
    with env.begin() as txn:
        value = txn.get(key.encode())
    env.close()
    if value is None:
        msg = f"Key '{key}' not found in LMDB database."
        raise KeyError(msg)
    return from_bytes(value)
