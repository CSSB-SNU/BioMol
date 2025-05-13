import os
import torch
from joblib import Parallel, delayed
from BioMol import SEQ_TO_HASH_PATH, DB_PATH
import pickle
import lmdb

"""
This code requires seq_to_str DB.
"""

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
    return data["cif_IDs"], data["tensors"]


def max_contact_diff(x, thr=12.0, sep=16, tol=2.0, chunk=32):
    B, L, _ = x.shape
    # boolean mask for valid points
    mask = x[..., 0].bool()

    # use half precision for coordinates to save memory
    # coords = x[..., 1:].to(torch.bfloat16)  # (B, L, 3)
    coords = x[..., 1:]
    coords = coords - coords.mean(dim=1, keepdim=True)  # (B, L, 3)
    coords = coords.to(torch.float16)  # convert to half precision
    # compute pairwise distances in half precision
    D = torch.norm(coords[:, :, None, :] - coords[:, None, :, :], dim=-1)  # (B, L, L)

    r = torch.arange(L, device=x.device)
    # invalidate pairs if either is masked-out or sequence separation < sep
    invalid = torch.logical_or(
        torch.logical_not(mask[:, :, None] & mask[:, None, :]),
        ((r[:, None] - r).abs() < sep)[None],
    )
    # build upper-triangular contact map (distance < thr)
    contact = torch.logical_and(~invalid, D < thr).triu(1)

    # generate all unique batch-pair indices
    b1, b2 = torch.triu_indices(B, B, 1)
    P = b1.numel()
    # preallocate score tensor on GPU
    diff_num = torch.empty(P, device=x.device, dtype=torch.int32)
    scores = torch.empty(P, device=x.device, dtype=torch.float32)

    # process pairs in chunks
    for idx in range(0, P, chunk):
        i1 = b1[idx : idx + chunk]
        i2 = b2[idx : idx + chunk]

        C1 = contact[i1]  # shape: (chunk, L, L)
        C2 = contact[i2]
        M1 = mask[i1]
        M2 = mask[i2]
        M1 = torch.logical_and(M1[:, :, None], M1[:, None, :])
        M2 = torch.logical_and(M2[:, :, None], M2[:, None, :])

        # valid contacts where both residues are present
        N1 = torch.logical_and(C1, M2)
        N2 = torch.logical_and(C2, M1)

        # corresponding distance slices
        D1 = D[i1]
        D2 = D[i2]

        # symmetric difference base
        diff_base = torch.logical_xor(N1, N2)
        # pairs where distance difference < tol
        near = torch.abs(D1 - D2) < tol

        # intersection: originally intersecting OR diff but distances close
        inter = torch.logical_or(
            torch.logical_and(N1, N2), torch.logical_and(diff_base, near)
        )
        # difference: diff_base AND NOT near
        diff = torch.logical_and(diff_base, ~near)

        # count events
        diff_count = diff.sum(dim=(1, 2))
        inter_count = inter.sum(dim=(1, 2))

        # compute chunk scores and write into the preallocated tensor
        diff_num[idx : idx + chunk] = diff_count
        scores[idx : idx + chunk] = diff_count.float() / (1 + inter_count.float())

    # find the batch-pair with maximum score
    max_idx = scores.argmax().item()
    return diff_num.tolist(), scores.tolist(), (b1[max_idx].item(), b2[max_idx].item())

def visualize_2D_tensor(tensor, save_path=None):
    """
    Visualize a 2D tensor as a heatmap.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))
    plt.imshow(tensor.cpu().numpy(), cmap="hot", interpolation="nearest")
    plt.colorbar()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def process_sequence(sequence_hash, criteria):
    """
    Process a single multi-state sequence.
    """
    torch.set_num_threads(1)
    IDs, residue_tensor = read_seq_lmdb(str(sequence_hash))

    if residue_tensor.shape[0] == 1:
        return (sequence_hash, IDs, 0, 0.0, None, "single_state")

    residue_tensor = residue_tensor[:, :, 4:8]

    # Compute the contact pair
    diff_nums, scores, (b1, b2) = max_contact_diff(residue_tensor, thr=8.0, tol=2.0, sep=16)
    diff_num_max = max(diff_nums)
    diff_max = max(scores)
    diff_max_IDs = (IDs[b1], IDs[b2])

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

    return (sequence_hash, IDs, diff_num_max, diff_max, diff_max_IDs, category)


def categorize_multi_state_sequences():
    """
    Categorize multi-state sequences based on their contact differences.
    """
    criteria = {
        "single_state": (0, 0.5),
        "flexible": (0.5, 1.0),
        "conformation_change": (1.0, 5.0),
        "dynamic": (5.0, float("inf")),
    }

    # for now, to save time filter out too long sequences
    seq_to_hash = load_hash_to_seq()
    seq_to_hash = {k : v for k, v in seq_to_hash.items() if len(k) < 2048}
    hash_list = [_hash for _, _hash in seq_to_hash.items()]

    not_filtered_path = f"{DB_PATH}/statistics/multi_state/not_filtered/multi_state_sequences.pkl"
    if os.path.exists(not_filtered_path):
        with open(not_filtered_path, "rb") as pf:
            results = pickle.load(pf)
    else :
        # Process each sequence in parallel. Using n_jobs=-1 uses all available cores.
        print(f"Processing {len(hash_list)} sequences...")
        results = Parallel(n_jobs=-1, verbose=10)(
            delayed(process_sequence)(seq_hash, criteria) for seq_hash in hash_list
        )

    # to prevent redundant work save the results
    with open(
        f"{DB_PATH}/statistics/multi_state/not_filtered/multi_state_sequences.pkl",
        "wb",
    ) as pf:
        pickle.dump(results, pf, protocol=pickle.HIGHEST_PROTOCOL)

    categories = {
        "single_state": {},
        "flexible": {},
        "conformation_change": {},
        "dynamic": {},
    }

    # collect results
    for seq_hash, chain_ids, diff_num_max, diff_max, diff_max_IDs, category in results:
        if seq_hash is None or category not in categories:
            continue
        categories[category][seq_hash] = (chain_ids, diff_num_max, diff_max, diff_max_IDs)

    # ensure save directory exists
    save_dir = f"{DB_PATH}/statistics/multi_state/not_filtered/"
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
    seq_len: tuple[int, int] = (128, 1024),
):
    categories = {
        "single_state": {},
        "flexible": {},
        "conformation_change": {},
        "dynamic": {},
    }
    for name, _ in categories.items():
        path = os.path.join(
            f"{DB_PATH}/statistics/multi_state/not_filtered", f"{name}_sequences.pkl"
        )
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
                raise ValueError(
                    f"Sequence hash {seq_hash} not found in hash_to_seqlen."
                )
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
    filtered_sequences = categorize_multi_state_sequences()
    # filter_multi_state_sequences(seq_len=(128, 1024))
