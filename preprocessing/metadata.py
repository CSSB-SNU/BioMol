import os
from joblib import Parallel, delayed
import csv
from filelock import FileLock
import gc
from BioMol import BioMol
import pickle

chainID_to_cluster_path = "/public_data/BioMolDB_2024Oct21/protein_seq_clust/mmseqs2_seqid30_cov80_covmode0_clustmode1_chainID_to_cluster.pkl"
with open(chainID_to_cluster_path, "rb") as f:
    chainID_to_cluster = pickle.load(f)

seq_to_hash_path = "/public_data/BioMolDB_2024Oct21/entity/sequence_hashes.pkl"
with open(seq_to_hash_path, "rb") as f:
    seq_to_hash = pickle.load(f)


def _save_deposition_resolution(cif_path, save_path, lock_path):
    """Process a single CIF file and write the result to CSV immediately."""
    print(f"Loading {cif_path}")
    biomol = BioMol(cif_path)
    ID = biomol.bioassembly.ID
    deposition_date = biomol.bioassembly.deposition_date
    resolution = biomol.bioassembly.resolution
    if resolution is None:
        resoultion = "?"

    # Write the result immediately, using a file lock to avoid conflicts.
    with FileLock(lock_path):
        with open(save_path, "a") as f:
            f.write(f"{ID},{deposition_date},{resolution}\n")

    # gc.collect()

    return cif_path


def save_deposition_resolution(cif_dir, save_path="passed_cif.csv"):
    # Define a lock file path (it will create a lock file alongside your CSV)
    lock_path = save_path + ".lock"

    # Gather all CIF file paths from the inner directories.
    cif_path_list = []
    for inner_dir in os.listdir(cif_dir):
        inner_dir_path = os.path.join(cif_dir, inner_dir)
        if not os.path.isdir(inner_dir_path):
            continue

        for file_name in os.listdir(inner_dir_path):
            if file_name.endswith(".cif") or file_name.endswith(".cif.gz"):
                full_path = os.path.join(inner_dir_path, file_name)
                cif_path_list.append(full_path)

    print(f"Processing {len(cif_path_list)} CIF files...")

    Parallel(n_jobs=-1)(
        delayed(_save_deposition_resolution)(cif_path, save_path, lock_path)
        for cif_path in cif_path_list
    )


def make_metadata(merged_fasta_path, chainID_to_deposition_csv, save_path):
    header = "CHAINID,DEPOSITION,RESOLUTION,HASH,CLUSTER,SEQUENCE\n"
    lines = [header]

    hash_digits = 6
    cluster_digits = 5

    ID_to_deposition = {}
    with open(chainID_to_deposition_csv, "r") as f:
        for line in f:
            line = line.strip()
            ID, deposition_date, resolution = line.split(",")
            ID_to_deposition[ID] = (deposition_date, resolution)

    with open(merged_fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                ID = line.split("|")[0][1:].strip()
            else:
                seq = line
                hash = seq_to_hash[seq]
                cluster = chainID_to_cluster[ID]
                deposition_date, resolution = ID_to_deposition[ID.split("_")[0]]
                if resolution == "None" or resolution is None:
                    resolution = -1
                # hash -> 6 digits (123 -> 000123)
                # cluster -> 5 digits (123 -> 00123)
                hash = str(hash).zfill(hash_digits)
                cluster = str(cluster).zfill(cluster_digits)
                lines.append(
                    f"{ID},{deposition_date},{resolution},{hash},{cluster},{seq}\n"
                )

    # sort by deposition date
    header = lines[0]
    lines = lines[1:]
    lines = sorted(lines, key=lambda x: x.split(",")[1])
    lines = [header] + lines

    if '.pkl' in save_path:
        with open(save_path, "wb") as f:
            pickle.dump(lines, f)
    else :
        with open(save_path, "w") as f:
            f.writelines(lines)


def read_large_csv_parallel(file_path, chunk_size=10**4, n_jobs=-1):
    """
    Reads a large CSV file in parallel using joblib and Python's built-in csv module.

    Parameters:
        file_path (str): Path to the CSV file.
        chunk_size (int): Number of lines per chunk for parallel processing.
        n_jobs (int): Number of CPU cores to use (-1 for all cores).

    Returns:
        list: A list of rows from the CSV file.
    """

    def get_file_line_count(file_path):
        """Counts the total number of lines in the file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)

    def read_chunk(start, end, file_path):
        """Reads a chunk of CSV data from a file using Python's built-in csv module."""
        with open(file_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            return [line for i, line in enumerate(reader) if start <= i < end]

    # Get total number of lines in the file
    total_lines = get_file_line_count(file_path)

    # Generate chunk indices
    chunk_indices = [
        (i, min(i + chunk_size, total_lines)) for i in range(0, total_lines, chunk_size)
    ]

    # Read and process chunks in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(read_chunk)(start, end, file_path) for start, end in chunk_indices
    )

    # Flatten results into a single list
    return [row for chunk in results for row in chunk]


def compare(
    new_meta_csv, old_meta_csv, save_dir="/public_data/BioMolDB_2024Oct21/metadata"
):
    old = read_large_csv_parallel(old_meta_csv)
    new = read_large_csv_parallel(new_meta_csv)

    old = old[1:]
    new = new[1:]

    old_dict = {}
    new_dict = {}
    for line in old:
        chain_ID, deposition_date, resolution, hash, cluster, seq, _ = line
        old_dict[chain_ID] = (deposition_date, resolution, seq, hash)

    for line in new:
        chain_ID, deposition_date, resolution, hash, cluster, seq = line
        new_dict[chain_ID] = (deposition_date, resolution, seq, hash)

    # test if the two dictionaries are the same for shared keys
    old_keys = set(old_dict.keys())
    new_keys = set(new_dict.keys())
    shared_keys = set(old_dict.keys()).intersection(set(new_dict.keys()))
    old_only_keys = old_keys - shared_keys
    new_only_keys = new_keys - shared_keys

    weird_changes = {}

    valid_keys = []

    for key in shared_keys:
        old_val = old_dict[key]
        new_val = new_dict[key]
        old_seq = old_val[2]
        new_seq = new_val[2]
        if old_seq != new_seq:
            if len(old_seq) != len(new_seq):
                weird_changes[key] = ("len diff", old_val, new_val)
            else:
                candidate = None
                for ii, (old_res, new_res) in enumerate(zip(old_seq, new_seq)):
                    if old_res == new_res:
                        continue
                    elif old_res == "X":
                        if candidate != "mmcif fasta discrepancy":
                            candidate = "unknown residue mapping"
                    else:
                        candidate = "mmcif fasta discrepancy (maybe mmcif update)"
                if candidate is None:
                    breakpoint()
                assert candidate is not None
                weird_changes[key] = (candidate, old_val, new_val)
        else:
            valid_keys.append(key)
    valid_keys = set(valid_keys)
    # save weird changes to save_dir + weird_changes.csv
    with open(os.path.join(save_dir, "weird_changes.csv"), "w") as f:
        for key in weird_changes:
            f.write(f"{key},{weird_changes[key]}\n")

    hash_map = {}
    for key in valid_keys:
        old_hash = old_dict[key][3]
        new_hash = new_dict[key][3]
        hash_map[old_hash] = new_hash

    # save hash_map to save_dir + hash_map.csv
    with open(os.path.join(save_dir, "hash_map.csv"), "w") as f:
        for key in hash_map:
            f.write(f"{key},{hash_map[key]}\n")

    len_cut = 4

    have_to_find_msa_key = new_keys - valid_keys
    have_to_find_msa_hash = set()
    for key in have_to_find_msa_key:
        hash, seq = new_dict[key][3], new_dict[key][2]
        if len(seq) < len_cut:
            continue
        have_to_find_msa_hash.add((hash, seq))
    print(f"have_to_find_msa_hash : {len(have_to_find_msa_hash)}")
    # write save_dir/{hash}.fasta per hash
    # >hash
    # seq
    for hash, seq in have_to_find_msa_hash:
        with open(os.path.join(save_dir, f"fasta/{hash}.fasta"), "w") as f:
            f.write(f">{hash}\n{seq}\n")


def run_hhblit(command):
    os.system(command)


def run_hhblits(
    evalue,
    cov,
    fasta_dir,
    a3m_dir,
    db="/public_data/db_protSeq/uniref30/2022_02/UniRef30_2022_02",
):
    # run hhblits
    # hhblits -i {hash}.fasta -oa3m {hash}.a3m -d {db} -cov {cov} -evalue {evalue}
    fasta_files = os.listdir(fasta_dir)
    commands = []
    for fasta_file in fasta_files:
        hash = fasta_file.split(".")[0]
        command = f"hhblits -i {fasta_dir}/{fasta_file} -oa3m {a3m_dir}/{hash}.a3m -d {db} -cov {cov} -evalue {evalue}"
        commands.append(command)

    breakpoint()
    Parallel(n_jobs=-1)(delayed(run_hhblit)(command) for command in commands)
    pass

def pickle_deposition_resolution(
    csv_path, save_path="/public_data/BioMolDB_2024Oct21/metadata/ID_to_deposition.pkl"
):
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        data = {row[0]: (row[1], row[2]) for row in reader}

    with open(save_path, "wb") as f:
        pickle.dump(data, f)


# TODO filter out (small peptides, low resolution, too many unknown residues)

if __name__ == "__main__":
    cif_dir = "/public_data/BioMolDB_2024Oct21/cif/"
    ID_to_deposition = "/public_data/BioMolDB_2024Oct21/metadata/ID_to_deposition.csv"

    merged_fasta_path = "/public_data/BioMolDB_2024Oct21/entity/merged_protein.fasta"
    # save_deposition_resolution(cif_dir, save_path = ID_to_deposition)
    pickle_deposition_resolution_path = "/public_data/BioMolDB_2024Oct21/metadata/ID_to_deposition.pkl"
    pickle_deposition_resolution(ID_to_deposition, save_path = pickle_deposition_resolution_path)

    make_metadata(merged_fasta_path, ID_to_deposition, "/public_data/BioMolDB_2024Oct21/metadata/metadata_psk.csv")

    # new_meta_csv = "/public_data/BioMolDB_2024Oct21/metadata/metadata_psk.csv"
    # old_meta_csv = "/public_data/ml/RF2_train/PDB-2021AUG02/list_v02.csv"
    # compare(new_meta_csv, old_meta_csv)
