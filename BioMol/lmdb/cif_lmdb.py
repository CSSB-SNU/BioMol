import os
import lmdb
import torch
import pickle
import gzip
from joblib import Parallel, delayed
# from BioMol import CIFDB_PATH
from BioMol.utils.parser import parse_cif


db_env = "/data/psk6950/PDB_2024Mar18/cif_protein_only.lmdb"
protein_graph_dir = "/data/psk6950/PDB_2024Mar18/protein_graph/"
cif_dir = "/data/psk6950/PDB_2024Mar18/cif/"
cif_config_path = "./BioMol/configs/types/protein_only.json"
CIFDB_PATH = "/data/psk6950/PDB_2024Mar18/cif_protein_only.lmdb"
remove_signal_peptide = True


def already_parsed(env_path=db_env):
    # Open LMDB in readonly mode to get already parsed keys
    env = lmdb.open(env_path, map_size=1 * 1024**4)  # 1TB
    with env.begin() as txn:
        keys = {key.decode() for key, _ in txn.cursor()}
    env.close()
    print(f"Already parsed keys: {len(keys)}")
    return keys


def get_cif_files(cif_dir, protein_graph_dir=protein_graph_dir):
    # Walk through cif_dir to gather all .cif.gz files
    protein_graph_files = []
    for _, _, files in os.walk(protein_graph_dir):
        for file in files:
            if file.endswith(".graph"):
                protein_graph_files.append(file.split(".")[0])
    cif_files = []
    for root, _, files in os.walk(cif_dir):
        for file in files:
            if file.endswith(".cif.gz"):
                cif_ID = file.split(".")[0]
                if cif_ID in protein_graph_files:
                    cif_files.append(os.path.join(root, file))
    return cif_files


def process_file(cif_path):
    # Extract cif_hash from the file name (assumes filename structure hash.xxx.cif.gz)
    file_name = os.path.basename(cif_path)
    cif_hash = file_name.split(".")[0]

    # Parse the cif file and process it
    torch.set_num_threads(1)
    bioassembly = parse_cif(cif_path, cif_config_path, remove_signal_peptide)

    # Serialize the bioassembly and compress it using gzip
    serialized_data = pickle.dumps(bioassembly, protocol=pickle.HIGHEST_PROTOCOL)
    compressed_data = gzip.compress(serialized_data)

    return cif_hash, compressed_data


def lmdb_cif(env_path=db_env, n_jobs=-1, batch=200):
    parsed_keys = already_parsed()
    cif_files = get_cif_files(cif_dir)

    # Filter out files that are already processed
    files_to_process = [
        f for f in cif_files if os.path.basename(f).split(".")[0] not in parsed_keys
    ]
    print(f"Files to process: {len(files_to_process)}")

    # Parallel processing of files using joblib.
    env = lmdb.open(env_path, map_size=1 * 1024**4)  # 1TB
    # n_jobs=-1 uses all available cores. Adjust as needed.
    for _ in range(0, len(files_to_process), batch):
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_file)(cif_path) for cif_path in files_to_process
        )
        # Open LMDB environment for writing
        with env.begin(write=True) as txn:
            for cif_hash, compressed_data in results:
                txn.put(cif_hash.encode(), compressed_data)
    env.close()


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


if __name__ == "__main__":
    lmdb_cif()
    # read_cif_lmdb('1v9u')
