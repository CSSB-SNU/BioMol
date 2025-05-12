import lmdb
import pickle
import gzip
import os
from joblib import Parallel, delayed
from BioMol.utils.MSA import MSA
from BioMol import DB_PATH, SEQ_TO_HASH_PATH

db_env = os.path.join(DB_PATH, "MSA.lmdb")


def load_seq_to_hash():
    return pickle.load(open(SEQ_TO_HASH_PATH, "rb"))


def already_parsed(env_path=db_env):
    # Open LMDB in readonly mode to get already parsed keys
    env = lmdb.open(env_path, readonly=True)
    with env.begin() as txn:
        keys = {key.decode() for key, _ in txn.cursor()}
    env.close()
    print(f"Already parsed keys: {len(keys)}")
    return keys


def process_file(seq_hash: str):
    # build the MSA object and serialize+compress
    msa = MSA(seq_hash, use_lmdb=True)
    data = pickle.dumps(msa, protocol=pickle.HIGHEST_PROTOCOL)
    blob = gzip.compress(data)
    return seq_hash.encode(), blob


def lmdb_MSA(env_path=db_env, n_jobs=-1, batch_size=1000):
    seq_to_hash = load_seq_to_hash()
    hash_list = [str(h).zfill(6) for h in seq_to_hash.values()]
    total = len(hash_list)
    print(f"Total MSAs to write: {total}")

    if os.path.exists(env_path):
        already_parsed_keys = already_parsed(env_path)
        hash_list = [h for h in hash_list if h not in already_parsed_keys]
        print(f"MSAs to write: {len(hash_list)}")

    env = lmdb.open(env_path, map_size=(2048 + 128) * 1024**3)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = hash_list[start:end]
        print(f"Processing batch {start}–{end - 1}...")

        # parallel map over this batch only
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_file)(seq_hash) for seq_hash in batch
        )

        # write this batch in one write-transaction
        with env.begin(write=True) as txn:
            for key_bytes, blob in results:
                txn.put(key_bytes, blob)

        # free memory of results before next batch
        del results

    env.close()
    print("All done.")


if __name__ == "__main__":
    lmdb_MSA()
