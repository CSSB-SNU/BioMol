import lmdb
import pickle
import gzip
from BioMol import CIFDB_PATH, MSADB_PATH


def read_cif_lmdb(key: str):
    env = lmdb.open(CIFDB_PATH, readonly=True, lock=False)

    with env.begin() as txn:
        compressed_pickled_data = txn.get(key.encode())
    env.close()

    if compressed_pickled_data is None:
        print(f"No data found for key: {key}")
        return None

    # First, decompress the data using gzip
    decompressed_data = gzip.decompress(compressed_pickled_data)

    # Then, unpickle the decompressed data to retrieve the original object
    cif_data = pickle.loads(decompressed_data)

    return cif_data


def read_MSA_lmdb(key: str):
    env = lmdb.open(MSADB_PATH, readonly=True, lock=False)

    with env.begin() as txn:
        compressed_pickled_data = txn.get(key.encode())
    env.close()

    if compressed_pickled_data is None:
        print(f"No data found for key: {key}")
        return None

    # First, decompress the data using gzip
    decompressed_data = gzip.decompress(compressed_pickled_data)

    # Then, unpickle the decompressed data to retrieve the original object
    cif_data = pickle.loads(decompressed_data)

    return cif_data
