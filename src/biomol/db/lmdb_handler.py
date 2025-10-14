from collections.abc import Callable
from pathlib import Path

import lmdb
from joblib import Parallel, delayed

from biomol.core.utils import from_bytes, to_bytes


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
            data_dict = parser(ccd_db_path, recipe, data_file)
            zcompressed_data = to_bytes(data_dict)
            return key.encode(), zcompressed_data, None
        except Exception as error:
            return key.encode(), to_bytes({}), error

    # remove UNL
    data_list = [data for data in data_list if data.stem != "UNL"]

    # --- Parallel processing ---
    results = Parallel(n_jobs=n_jobs, verbose=10, prefer="processes")(
        delayed(_process_file)(data_file) for data_file in data_list
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
