import torch
import torch.nn.functional as F
import pickle
from BioMol import SEQ_TO_HASH_PATH, DB_PATH
from BioMol.BioMol import BioMol
from joblib import Parallel, delayed
import os
import lmdb

merged_fasta_path = f"{DB_PATH}/entity/merged_protein.fasta"
hash_to_full_IDs_path = f"{DB_PATH}/metadata/hash_to_full_IDs.pkl"
db_env = f"{DB_PATH}/seq_to_str/residue.lmdb"

def load_hash_to_seq():
    return pickle.load(open(SEQ_TO_HASH_PATH, "rb"))

def load_hash_to_pdbIDs():
    return pickle.load(open(hash_to_full_IDs_path, "rb"))

def find_pdbIDs_by_hashes(hash_list:list[str], hash_to_pdbIDs:dict[str, list[str]]) -> list[str]:
    """
    Given a list of hashes, return a dictionary mapping each hash to its corresponding PDB IDs.
    """
    pdb_IDs = {}
    for seq_hash in hash_list:
        if seq_hash in hash_to_pdbIDs:
            pdb_IDs[seq_hash] = hash_to_pdbIDs[seq_hash]
        else:
            raise ValueError(f"Hash {seq_hash} not found in the database.")
        
    # find common pdbIDs
    common_pdbIDs = set(pdb_IDs[hash_list[0]])
    for seq_hash in hash_list[1:]:
        common_pdbIDs.intersection_update(pdb_IDs[seq_hash])

    if len(common_pdbIDs) == 0:
        raise ValueError("No common PDB IDs found for the given hashes.")
    return sorted(list(common_pdbIDs))


# TODO expand it to full IDs. This version is only for testing
def make_seq_hash_to_structure_db(
    save_dir = f"{DB_PATH}/seq_to_str/",
    thread_num = 1,
    level: str = "residue", # or "atom"
    inner_dir_already = False,
) -> dict[str, tuple[list[str], list[torch.Tensor]]]:
    """
    Given a sequence hash, return a list of PDB IDs.
    """
    save_dir = f"{save_dir}/{level}/"
    metadata_path = f"{DB_PATH}/metadata/metadata_psk.csv" # protein only

    lines = open(metadata_path, "r").readlines()
    lines = lines[1:] # remove header
    pdb_IDs = [line.split(",")[0].split("_")[0] for line in lines]
    pdb_IDs = sorted(list(set(pdb_IDs)))

    seq_to_hash = load_hash_to_seq()
    hash_list = list(seq_to_hash.values())
    hash_list = [str(_hash).zfill(6) for _hash in hash_list]

    # pre generate inner_dirs
    if not inner_dir_already:
        for _hash in hash_list:
            inner_dir = f"{save_dir}/{_hash[0:3]}/{_hash[3:6]}"
            if not os.path.exists(inner_dir):
                os.makedirs(inner_dir)

    print(f"Number of PDB IDs: {len(pdb_IDs)}")

    results = Parallel(n_jobs=thread_num)(
        delayed(save_structures)(pdb_ID, save_dir, level) for pdb_ID in pdb_IDs
    )    


def process_file(_hash : str):
    save_dir = f"{DB_PATH}/seq_to_str/residue/{_hash[0:3]}/{_hash[3:6]}"
    IDs = os.listdir(save_dir)
    tensors = [os.path.join(save_dir, ID) for ID in IDs]
    tensors = [torch.load(tensor) for tensor in tensors]
    # if tensors.shape different, pad them to the same size
    tensor_shapes = [tensor.shape for tensor in tensors]
    if len(set(tensor_shapes)) > 1:
        max_len = max([shape[0] for shape in tensor_shapes])
        tensors = [
            F.pad(tensor, (0, 0, 0, max_len - tensor.shape[0]), "constant", float("nan"))
            for tensor in tensors
        ]
        print(f"Warning: tensors have different shapes. Padding to {max_len} for hash {_hash}.")

    tensors = torch.stack(tensors, dim=0) # (B, L, 10)
    return IDs, tensors

def lmdb_seq_to_str(env_path=db_env, n_jobs=-1):
    seq_to_hash = load_hash_to_seq()
    hash_list = list(seq_to_hash.values())
    hash_list = [str(_hash).zfill(6) for _hash in hash_list]
    
    # Filter out files that are already processed
    print(f"Files to process: {len(hash_list)}")

    # Parallel processing of files using joblib.
    env = lmdb.open(env_path, map_size=1 * 1024**4)  # 1TB
    # n_jobs=-1 uses all available cores. Adjust as needed.
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_file)(_hash) for _hash in hash_list
    )
    # Open LMDB environment for writing
    with env.begin(write=True) as txn:
        for _hash, (cif_IDs, tensors) in zip(hash_list, results):
            to_save = {
                "cif_IDs": cif_IDs,
                "tensors": tensors,
            }
            to_save = pickle.dumps(to_save, protocol=pickle.HIGHEST_PROTOCOL)
            txn.put(_hash.encode(), to_save)
    env.close()

def read_seq_lmdb(key: str):
    """
    Read a sequence from the LMDB database.
    """
    env = lmdb.open(db_env, readonly=True)
    with env.begin() as txn:
        data = txn.get(key.encode())
        if data is None:
            raise ValueError(f"Key {key} not found in the database.")
        data = pickle.loads(data)
    env.close()
    return data


if __name__ == "__main__":
    # Load the sequence hash from the file
    # make_seq_hash_to_structure_db(thread_num=-1, inner_dir_already=False)
    lmdb_seq_to_str()
