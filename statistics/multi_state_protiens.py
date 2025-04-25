import os
import torch
from BioMol import BioMol
from joblib import Parallel, delayed
import pickle
import gc

merged_fasta_path = "/data/psk6950/PDB_2024Mar18/entity/merged_protein.fasta"
protein_graph_path = "/data/psk6950/PDB_2024Mar18/cluster/PDBID_to_graph_hash.txt"
sequence_hash_path = "/data/psk6950/PDB_2024Mar18/entity/sequence_hashes.pkl"

with open(sequence_hash_path, "rb") as pf:
    sequence_hashes = pickle.load(pf)


def parse_protein_ID(protein_id: str) -> str:
    pdb_ID, bioasssembly_id, model_id, alt_id = protein_id.split("_")
    return pdb_ID, bioasssembly_id, model_id, alt_id


def parse_protein_graph():
    """
    Parse the protein graph file and extract protein IDs and their corresponding graph hashes.
    """  # noqa: E501
    assert os.path.exists(protein_graph_path), f"File not found: {protein_graph_path}"

    with open(protein_graph_path) as f:
        lines = f.readlines()

    protein_id_data = {}

    for line in lines:
        line = line.strip()
        if line:
            graph_hash, protein_ids = line.split(":")
            protein_ids = protein_ids.split(",")
            protein_ids = [id.strip() for id in protein_ids]
            for protein_id in protein_ids:
                pdb_ID, bioasssembly_id, model_id, alt_id = parse_protein_ID(protein_id)
                if pdb_ID not in protein_id_data:
                    protein_id_data[pdb_ID] = []
                protein_id_data[pdb_ID].append(
                    {
                        "bioasssembly_id": bioasssembly_id,
                        "model_id": model_id,
                        "alt_id": alt_id,
                        "graph_hash": graph_hash,
                    }
                )

    return protein_id_data


def get_multi_state_protein_seqs(length_filter=(128, 512)):
    """
    Parse the merged fasta file and extract protein sequences.
    """
    assert os.path.exists(merged_fasta_path), f"File not found: {merged_fasta_path}"

    with open(merged_fasta_path) as f:
        lines = f.readlines()

    protein_sequences = {}
    current_id = None
    current_sequence = []

    for line in lines:
        if line.startswith(">"):
            if current_id is not None:
                protein_sequences[current_id] = "".join(current_sequence)
            current_id = line[1:].strip()
            current_id = current_id.split("|")[0].strip()
            current_sequence = []
        else:
            current_sequence.append(line.strip())

    if current_id is not None:
        protein_sequences[current_id] = "".join(current_sequence)

    sequences_to_chain_ID = {}
    for protein_id, sequence in protein_sequences.items():
        if sequence not in sequences_to_chain_ID:
            sequences_to_chain_ID[sequence] = []
        sequences_to_chain_ID[sequence].append(protein_id)

    protein_id_data = parse_protein_graph()

    single_state_sequences = {}
    multi_state_sequences = {}
    for sequence, chain_ids in sequences_to_chain_ID.items():
        num_of_ids = 0
        for chain_id in chain_ids:
            pdb_ID = chain_id.split("_")[0]
            if pdb_ID in protein_id_data:
                num_of_ids += len(protein_id_data[pdb_ID])
        if num_of_ids == 1:
            single_state_sequences[sequence] = chain_ids
        else:
            multi_state_sequences[sequence] = chain_ids

    # filter by length
    single_state_sequences = {
        k: v
        for k, v in single_state_sequences.items()
        if length_filter[0] <= len(k) <= length_filter[1]
    }
    multi_state_sequences = {
        k: v
        for k, v in multi_state_sequences.items()
        if length_filter[0] <= len(k) <= length_filter[1]
    }

    single_state_sequences = {
        sequence_hashes[k]: v for k, v in single_state_sequences.items()
    }
    multi_state_sequences = {
        sequence_hashes[k]: v for k, v in multi_state_sequences.items()
    }

    print(f"Number of single state sequences: {len(single_state_sequences)}")
    print(f"Number of multi state sequences: {len(multi_state_sequences)}")

    return single_state_sequences, multi_state_sequences


def extract_chain_ids(biomol: BioMol, chain_ID, assembly_id, model_id, alt_id):
    """
    Extract chain IDs from the biomol object.
    """
    biomol.choose(assembly_id, model_id, alt_id)
    residue_chain_break = biomol.structure.residue_chain_break
    residue_chain_id_list = list(residue_chain_break.keys())
    residue_tensor = biomol.structure.residue_tensor

    find_chain = False
    for ch in residue_chain_id_list:
        if chain_ID == ch.split("_")[0]:
            chain_ID = ch
            find_chain = True
            break

    if not find_chain:
        return None

    residue_start, residue_end = residue_chain_break[chain_ID]
    residue_tensor = residue_tensor[residue_start:residue_end]
    return residue_tensor


def get_contact_pair(residue_tensor, contact_threshold=8.0):
    xyz = residue_tensor[:, 5:8]  # (L, 3)
    mask = residue_tensor[:, 4]  # (L, 1)
    L = xyz.shape[0]
    valid_reisude = torch.where(mask == 1)[0].tolist()
    dist_map = ((xyz[:, None, :] - xyz[None, :, :]) ** 2).sum(-1) ** 0.5  # (L, L)
    dist_map = dist_map * mask[:, None] * mask[None, :]  # (L, L)
    sequence_contact_map = torch.arange(L)[:, None] - torch.arange(L)[None, :]  # (L, L)
    sequence_contact_map = torch.abs(sequence_contact_map)
    sequence_contact_map = sequence_contact_map < 16

    # convert 0 to 999
    dist_map[dist_map == 0] = 999
    dist_map[sequence_contact_map] = 999

    contact_map = dist_map < contact_threshold
    contact_pair = torch.nonzero(contact_map)
    contact_pair = contact_pair[contact_pair[:, 0] < contact_pair[:, 1]].tolist()
    # to set
    contact_pair = set(map(tuple, contact_pair))
    return contact_pair, valid_reisude


def cal_contact_diff(contact_pair1, contact_pair2, mask1, mask2):
    """
    Calculate the difference between two contact pairs.
    """
    new_contact_pair1 = []
    new_contact_pair2 = []
    for pair in contact_pair1:
        if pair[0] in mask2 and pair[1] in mask2:
            new_contact_pair1.append(pair)
    for pair in contact_pair2:
        if pair[0] in mask1 and pair[1] in mask1:
            new_contact_pair2.append(pair)
    contact_pair1 = set(map(tuple, new_contact_pair1))
    contact_pair2 = set(map(tuple, new_contact_pair2))

    diff1 = contact_pair1 - contact_pair2
    diff2 = contact_pair2 - contact_pair1
    intersection = contact_pair1.intersection(contact_pair2)
    diff = diff1.union(diff2)
    return len(diff) / (1 + len(intersection))



def process_sequence(sequence_hash, chain_ids, protein_id_data, cif_dir, criteria):
    """
    Process a single multi-state sequence.
    """
    contact_pairs = {}
    residue_tensor_dict = {}
    for chain_id in chain_ids:
        pdb_ID = chain_id.split("_")[0]

        # Initialize the BioMol object
        biomol = BioMol(
            pdb_ID = pdb_ID,
            mol_types=["protein"],
        )

        # Look up protein id data
        ID_list = protein_id_data[pdb_ID]
        for ID in ID_list:
            bioasssembly_id = ID["bioasssembly_id"]
            model_id = ID["model_id"]
            alt_id = ID["alt_id"]
            chain_ID = chain_id.split("_")[1]

            residue_tensor = extract_chain_ids(
                biomol, chain_ID, bioasssembly_id, model_id, alt_id
            )
            if residue_tensor is None:
                continue

            # Compute the contact pair
            contact_pair, mask = get_contact_pair(residue_tensor)
            merged_ID = f"{pdb_ID}_{chain_ID}_{bioasssembly_id}_{model_id}_{alt_id}"
            contact_pairs[merged_ID] = (contact_pair, mask)
            residue_tensor_dict[merged_ID] = residue_tensor

        del biomol
        gc.collect()

    # Calculate pairwise differences
    diffs = {}
    ID_list = list(contact_pairs.keys())
    if len(ID_list) < 2:
        return (None, None, None, None, None)
    print(f"sequence_hash: {sequence_hash}, num of contact pairs: {len(contact_pairs)}")
    for i in range(len(contact_pairs)):
        for j in range(i + 1, len(contact_pairs)):
            ID1, ID2 = ID_list[i], ID_list[j]
            contact_pair1, mask1 = contact_pairs[ID1]
            contact_pair2, mask2 = contact_pairs[ID2]
            diff = cal_contact_diff(contact_pair1, contact_pair2, mask1, mask2)
            diffs[(ID1, ID2)] = diff

    # Find the maximum difference
    diff_max_ID = max(diffs, key=diffs.get)
    diff_max = diffs[diff_max_ID]

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

    return (sequence_hash, chain_ids, diff_max_ID, diff_max, category)


def filter_multi_state_sequences(multi_state_sequences):
    """
    Filter multi-state sequences based on specific criteria using parallel processing.
    """
    criteria = {
        "single_state": (0, 0.15),
        "flexible": (0.15, 0.4),
        "conformation_change": (0.4, 0.6),
        "dynamic": (0.6, 1.0),
    }

    cif_dir = "/data/psk6950/PDB_2024Mar18/cif/"
    protein_id_data = parse_protein_graph()

    # Process each sequence in parallel. Using n_jobs=-1 uses all available cores.
    results = Parallel(n_jobs=-1)(
        delayed(process_sequence)(
            sequence_hash, chain_ids, protein_id_data, cif_dir, criteria
        )
        for sequence_hash, chain_ids in multi_state_sequences.items()
    )
    # results = []
    # for sequence_hash, chain_ids in multi_state_sequences.items():
    #     result = process_sequence(sequence_hash, chain_ids, protein_id_data, cif_dir,
    # criteria)
    #     results.append(result)
    # 602833

    # Collect the results into the appropriate dictionaries.
    single_state_sequences = {}
    flexible_sequences = {}
    conformation_change_sequences = {}
    dynamic_sequences = {}

    for sequence_hash, chain_ids, diff_max_ID, diff_max, category in results:
        if sequence_hash is None:
            continue
        if category == "single_state":
            single_state_sequences[sequence_hash] = (chain_ids, diff_max_ID, diff_max)
        elif category == "flexible":
            flexible_sequences[sequence_hash] = (chain_ids, diff_max_ID, diff_max)
        elif category == "conformation_change":
            conformation_change_sequences[sequence_hash] = (
                chain_ids,
                diff_max_ID,
                diff_max,
            )
        elif category == "dynamic":
            dynamic_sequences[sequence_hash] = (chain_ids, diff_max_ID, diff_max)

    save_dir = "/data/psk6950/PDB_2024Mar18/statistics/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # pickle the results
    with open(os.path.join(save_dir, "single_state_sequences.pkl"), "wb") as pf:
        pickle.dump(single_state_sequences, pf, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(save_dir, "flexible_sequences.pkl"), "wb") as pf:
        pickle.dump(flexible_sequences, pf, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(save_dir, "conformation_change_sequences.pkl"), "wb") as pf:
        pickle.dump(conformation_change_sequences, pf, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(save_dir, "dynamic_sequences.pkl"), "wb") as pf:
        pickle.dump(dynamic_sequences, pf, protocol=pickle.HIGHEST_PROTOCOL)

    return (
        single_state_sequences,
        flexible_sequences,
        conformation_change_sequences,
        dynamic_sequences,
    )


if __name__ == "__main__":
    single_state_sequences, multi_state_sequences = get_multi_state_protein_seqs()
    filtered_sequences = filter_multi_state_sequences(multi_state_sequences)
