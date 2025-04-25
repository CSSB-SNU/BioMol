import os
import pickle
from joblib import Parallel, delayed


def check_MSA():
    hash_file = "/data/psk6950/PDB_2024Mar18/entity/sequence_hashes.pkl"
    metadata_csv = "/data/psk6950/PDB_2024Mar18/metadata/metadata_psk.csv"
    if not os.path.exists(hash_file):
        print(f"Hash file {hash_file} does not exist.")
        return False
    with open(hash_file, "rb") as pf:
        seq_hashes = pickle.load(pf)
    hash_to_seq = {v: k for k, v in seq_hashes.items()}

    hash_list = []
    with open(metadata_csv, "r") as f:
        lines = f.readlines()
        lines = lines[1:]  # Skip the header
        lines = [line.strip().split(",") for line in lines]
        hash_list = [line[3] for line in lines]

    hash_list = list(set(hash_list))
    seq_hash_list = list(set(seq_hashes.values()))

    a3m_dir1 = "/data/psk6950/PDB_2024Mar18/a3m"
    a3m_dir2 = "/data/psk6950/PDB_2024Mar18/new_hash_a3m"
    a3m_files1 = os.listdir(a3m_dir1)
    a3m_files1 = [
        file.split(".a3m.gz")[0] for file in a3m_files1 if file.endswith(".a3m.gz")
    ]
    a3m_files2 = os.listdir(a3m_dir2)
    error_list = []

    # # intersect the two lists
    a3m_intersection = list(set(a3m_files1).intersection(set(a3m_files2)))

    # union the two lists
    a3m_files = list(set(a3m_files1).union(set(a3m_files2)))
    test = list(set(a3m_files).intersection(set(hash_list)))
    diff = set(hash_list) - set(a3m_files)
    seqs_not_in_a3m = [hash_to_seq[int(hash)] for hash in diff]
    len_seqs_not_in_a3m = [len(seq) for seq in seqs_not_in_a3m]
    print(f"Number of files in {a3m_dir1}: {len(a3m_files1)}")

    breakpoint()
    for inner_dir in os.listdir(a3m_dir2):
        a3m_path = f"{a3m_dir2}/{inner_dir}/t000_msa0.a3m"
        if os.path.exists(a3m_path):
            a3m_files2.append(a3m_path)
        else:
            print(f"File {a3m_path} does not exist.")
            error_list.append(inner_dir)

    breakpoint()


def process_seq(seq_hash, a3m_dir1, a3m_dir2):
    a3m_path = os.path.join(a3m_dir2, seq_hash, "t000_msa0.a3m")
    if os.path.exists(a3m_path):
        # Copy the file
        dest_path = os.path.join(a3m_dir1, f"{seq_hash}.a3m")
        os.system(f"cp {a3m_path} {dest_path}")
        if os.path.exists(f"{dest_path}.gz"):
            os.remove(f"{dest_path}.gz")
        # Gzip the file
        os.system(f"gzip {dest_path}")
    else:
        print(f"File {a3m_path} does not exist.")


def copy_MSA():
    a3m_dir1 = "/data/psk6950/PDB_2024Mar18/a3m"
    a3m_dir2 = "/data/psk6950/PDB_2024Mar18/new_hash_a3m"
    seq_hashes = os.listdir(a3m_dir2)

    # Run the processing in parallel using all available cores
    Parallel(n_jobs=-1)(
        delayed(process_seq)(seq_hash, a3m_dir1, a3m_dir2) for seq_hash in seq_hashes
    )


if __name__ == "__main__":
    copy_MSA()
