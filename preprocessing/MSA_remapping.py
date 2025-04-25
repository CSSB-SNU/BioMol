# due to re-hash of the sequence, I have to remap the a3m files.
# remap and save it using pickle and lmdb
import os
import lmdb
import pickle
import gzip

remap_path = "/data/psk6950/PDB_2024Mar18/metadata/hash_map.csv"
hash_map = {}
test = {}
with open(remap_path, "r") as f:
    for line in f:
        line = line.strip().split(",")
        hash_map[line[0]] = line[1]
        test[line[1]] = line[0]

a3m_dir = "/public_data/ml/RF2_train/PDB-2021AUG02/a3m"
to_dir = "/data/psk6950/PDB_2024Mar18/a3m"
db_env = "/data/psk6950/PDB_2024Mar18/MSA.lmdb"


def already_parsed(env_path=db_env):
    env = lmdb.open(env_path, map_size=1024**3)
    with env.begin() as txn:
        keys = [key.decode() for key, _ in txn.cursor()]
    env.close()
    print(f"already parsed keys: {len(keys)}")
    return keys


def rename_and_copy_files():
    for root, dirs, files in os.walk(a3m_dir):
        for file in files:
            if file.endswith(".a3m.gz"):
                old_hash = file.split(".")[0]
                if old_hash not in hash_map:
                    print(f"hash {old_hash} not found in hash_map")
                    continue
                new_hash = hash_map[old_hash]
                a3m_path = os.path.join(root, file)

                # Rename the file on disk from old hash to new hash
                new_file = f"{new_hash}.a3m.gz"
                new_path = os.path.join(to_dir, new_file)
                if os.path.exists(new_path):
                    print(f"File {new_path} already exists")
                else:
                    # copy
                    os.system(f"cp {a3m_path} {new_path}")
                    print(f"Copied {a3m_path} to {new_path}")


def lmdb_MSA(env_path=db_env):
    already_parsed_keys = already_parsed()
    env = lmdb.open(env_path, map_size=800 * 1024**3)
    for root, dirs, files in os.walk(to_dir):
        for file in files:
            if file.endswith(".a3m.gz"):
                a3m_path = os.path.join(root, file)
                a3m_hash = file.split(".")[0]
                if a3m_hash in already_parsed_keys:
                    print(f"hash {a3m_hash} already parsed")
                    continue

                # # Read and process the file if needed (e.g., decompress)
                # with gzip.open(a3m_path, 'rb') as f:
                #     msa_data = f.read()
                with open(a3m_path, "rb") as f:
                    msa_data = f.read()

                # Write the processed data into LMDB using new_hash as key
                with env.begin(write=True) as txn:
                    txn.put(
                        a3m_hash.encode(),
                        pickle.dumps(msa_data, protocol=pickle.HIGHEST_PROTOCOL),
                    )

    env.close()


def mv_signalp(
    hhblit_result_path="/data/psk6950/PDB_2024Mar18/new_hash_a3m/",
    to_dir="/data/psk6950/PDB_2024Mar18/signalp",
):
    inner_dir_list = os.listdir(hhblit_result_path)
    for inner_dir in inner_dir_list:
        inner_dir_path = os.path.join(hhblit_result_path, inner_dir)
        if not os.path.isdir(inner_dir_path):
            continue
        file = inner_dir_path + "/signalp/output.gff3"
        if not os.path.exists(file):
            print(f"File {file} does not exist")
            continue
        # Read and process the file if needed
        with open(file, "r") as f:
            lines = f.readlines()
            if len(lines) < 2:
                print(f"File {file} is empty")
                continue
        os.system(f"cp {file} {to_dir}/{inner_dir}.gff3")


if __name__ == "__main__":
    # rename_and_copy_files()
    # lmdb_MSA()
    mv_signalp()
