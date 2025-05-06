import os
import torch
from joblib import Parallel, delayed
from BioMol import SEQ_TO_HASH_PATH, DB_PATH
import pickle
import gc
import lmdb
import copy

'''
This code requires seq_to_str DB.
'''

db_env = f"{DB_PATH}/seq_to_str/residue.lmdb"


def load_hash_to_seq():
    return pickle.load(open(SEQ_TO_HASH_PATH, "rb"))

def read_seq_lmdb(key: str):
    """
    Read a sequence from the LMDB database.
    """
    env = lmdb.open(db_env, readonly=True, lock=False)
    with env.begin() as txn:
        data = txn.get(key.encode())
        if data is None:
            raise ValueError(f"Key {key} not found in the database.")
        data = pickle.loads(data)
    env.close()
    return data['cif_IDs'], data['tensors']


def max_contact_diff(x, thr=12., sep=16, chunk=32):
    B,L,_ = x.shape
    m = x[...,0].bool()
    D = torch.cdist(x[...,1:], x[...,1:]) 
    r = torch.arange(L, device=x.device)
    inv = ~(m[:,:,None]&m[:,None,:]) | ((r[:,None]-r).abs().lt(sep))[None]
    A = (D.masked_fill(inv, float('inf')) < thr).triu(1)
    b1,b2 = torch.triu_indices(B, B, 1)
    score = -1
    if L > 1024 :
        chunk = 1
    for ii in range(0, b1.shape[0], chunk):
        i1 = b1[ii:ii+chunk]; i2 = b2[ii:ii+chunk]
        C1,C2 = A[i1], A[i2]; M1,M2 = m[i1], m[i2]
        N1 = C1 & (M2[:,None]&M2[:,:,None]); N2 = C2 & (M1[:,None]&M1[:,:,None])
        diff = (N1 ^ N2).sum(dim=(1,2)).float(); inter = (N1 & N2).sum(dim=(1,2)).float()
        score = max(score, (diff / (1 + inter)).max().item())
    return score

def process_sequence(sequence_hash, criteria):
    """
    Process a single multi-state sequence.
    """
    IDs, residue_tensor = read_seq_lmdb(str(sequence_hash))

    print(f"Processing sequence {sequence_hash} with IDs {len(IDs)}")

    if residue_tensor.shape[0] == 1:
        return (sequence_hash, IDs, 0.0, 'single_state')

    residue_tensor = residue_tensor[:,:,4:8]

    # Compute the contact pair
    diff_max = max_contact_diff(residue_tensor, thr=8., sep=16)

    # Determine the category based on the criteria
    if diff_max < criteria["single_state"][1]:
        category = "single_state"
    elif diff_max < criteria["flexible"][1]:
        category = "flexible"
    elif diff_max < criteria["conformation_change"][1]:
        category = "conformation_change"
    elif diff_max < criteria["dynamic"][1]:
        category = "dynamic"
    else:
        category = None

    return (sequence_hash, IDs, diff_max, category)


def categorize_multi_state_sequences():
    """
    Categorize multi-state sequences based on their contact differences.
    """
    criteria = {
        "single_state": (0, 0.15),
        "flexible": (0.15, 0.4),
        "conformation_change": (0.4, 0.6),
        "dynamic": (0.6, 1.0),
    }

    seq_to_hash = load_hash_to_seq()
    hash_list = list(seq_to_hash.values())

    # Process each sequence in parallel. Using n_jobs=-1 uses all available cores.
    results = Parallel(n_jobs=16)(
        delayed(process_sequence)(
            seq_hash, criteria
        )
        for seq_hash in hash_list
    )

    categories = {
        "single_state": {},
        "flexible": {},
        "conformation_change": {},
        "dynamic": {},
    }

    # collect results
    for seq_hash, chain_ids, diff_max, category in results:
        if seq_hash is None or category not in categories:
            continue
        categories[category][seq_hash] = (chain_ids, diff_max)

    # ensure save directory exists
    save_dir = f"{DB_PATH}/statistics/multi_state/"
    os.makedirs(save_dir, exist_ok=True)

    # dump each category to its own .pkl
    for name, data in categories.items():
        path = os.path.join(save_dir, f"{name}_sequences.pkl")
        with open(path, "wb") as pf:
            pickle.dump(data, pf, protocol=pickle.HIGHEST_PROTOCOL)

    # assign back to variables if needed
    single_state_sequences = categories["single_state"]
    flexible_sequences = categories["flexible"]
    conformation_change_sequences = categories["conformation_change"]
    dynamic_sequences = categories["dynamic"]

    return (
        single_state_sequences,
        flexible_sequences,
        conformation_change_sequences,
        dynamic_sequences,
    )

def filter_multi_state_sequences(
    seq_len : tuple[int, int] = (128, 1024),
):
    categories = {
        "single_state": {},
        "flexible": {},
        "conformation_change": {},
        "dynamic": {},
    }
    for name, data in categories.items():
        path = os.path.join(f"{DB_PATH}/statistics/multi_state/not_filtered", f"{name}_sequences.pkl")
        with open(path, "rb") as pf:
            categories[name] = pickle.load(pf)

    seq_to_hash = load_hash_to_seq()
    hash_to_seqlen = {v: len(k) for k, v in seq_to_hash.items()}

    to_path = f"{DB_PATH}/statistics/multi_state/filtered_{seq_len[0]}_{seq_len[1]}/"
    if not os.path.exists(to_path):
        os.makedirs(to_path)


    # print the number of sequences in each category
    for name, data in categories.items():
        print(f"{name}: {len(data)} sequences")

    filtered_categories = {
        "single_state": {},
        "flexible": {},
        "conformation_change": {},
        "dynamic": {},
    }
    for name, data in categories.items():
        for seq_hash, (chain_ids, diff_max) in data.items():
            if seq_hash not in hash_to_seqlen:
                raise ValueError(f"Sequence hash {seq_hash} not found in hash_to_seqlen.")
            _seq_len = hash_to_seqlen[seq_hash]
            if _seq_len < seq_len[0] or _seq_len > seq_len[1]:
                # remove the sequence from the category
                continue
            filtered_categories[name][seq_hash] = (chain_ids, diff_max)

        path = os.path.join(to_path, f"{name}_sequences.pkl")
        with open(path, "wb") as pf:
            pickle.dump(filtered_categories[name], pf, protocol=pickle.HIGHEST_PROTOCOL)

    # print the number of sequences in each category
    for name, data in filtered_categories.items():
        print(f"{name}: {len(data)} sequences")

    breakpoint()

if __name__ == "__main__":
    # filtered_sequences = categorize_multi_state_sequences()
    filter_multi_state_sequences(seq_len=(128, 1024))
