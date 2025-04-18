import os
import torch
from biomol import BioMol
import copy
from joblib import Parallel, delayed
import pickle
import gc
import matplotlib.pyplot as plt

merged_fasta_path = "/data/psk6950/PDB_2024Mar18/entity/merged_protein.fasta"
protein_graph_path = "/data/psk6950/PDB_2024Mar18/cluster/PDBID_to_graph_hash.txt"
sequence_hash_path = "/data/psk6950/PDB_2024Mar18/entity/sequence_hashes.pkl"

with open(sequence_hash_path, 'rb') as pf:
    sequence_hashes = pickle.load(pf)

def parse_protein_ID(protein_id: str) -> str:
    pdb_ID, bioasssembly_id, model_id, alt_id = protein_id.split('_')
    return pdb_ID, bioasssembly_id, model_id, alt_id

def parse_protein_graph():
    """
    Parse the protein graph file and extract protein IDs and their corresponding graph hashes.
    """
    assert os.path.exists(protein_graph_path), f"File not found: {protein_graph_path}"
    
    with open(protein_graph_path, 'r') as f:
        lines = f.readlines()
    
    protein_id_data = {}
    
    for line in lines:
        line = line.strip()
        if line:
            graph_hash, protein_ids = line.split(':')
            protein_ids = protein_ids.split(',')
            protein_ids = [id.strip() for id in protein_ids]
            for protein_id in protein_ids:
                pdb_ID, bioasssembly_id, model_id, alt_id = parse_protein_ID(protein_id)
                if pdb_ID not in protein_id_data:
                    protein_id_data[pdb_ID] = []
                protein_id_data[pdb_ID].append({
                    'bioasssembly_id': bioasssembly_id,
                    'model_id': model_id,
                    'alt_id': alt_id,
                    'graph_hash': graph_hash
                })

    
    return protein_id_data

def get_multi_pdb_ID_protein_seqs(length_filter = (128, 512), save_path = "./statistics/figures/"):
    """
    Parse the merged fasta file and extract protein sequences.
    """
    assert os.path.exists(merged_fasta_path), f"File not found: {merged_fasta_path}"
    
    with open(merged_fasta_path, 'r') as f:
        lines = f.readlines()
    
    protein_sequences = {}
    current_id = None
    current_sequence = []
    
    for line in lines:
        if line.startswith('>'):
            if current_id is not None:
                protein_sequences[current_id] = ''.join(current_sequence)
            current_id = line[1:].strip()
            current_id = current_id.split('|')[0].strip()
            current_sequence = []
        else:
            current_sequence.append(line.strip())
    
    if current_id is not None:
        protein_sequences[current_id] = ''.join(current_sequence)


    sequences_to_pdb_ID = {}
    for protein_id, sequence in protein_sequences.items():
        if sequence not in sequences_to_pdb_ID:
            sequences_to_pdb_ID[sequence] = []
        pdb_ID = protein_id.split('_')[0]
        if pdb_ID not in sequences_to_pdb_ID[sequence]:
            sequences_to_pdb_ID[sequence].append(pdb_ID)
        else:
            continue

    sequence_to_pdb_num = {}

    for sequence, pdb_ids in sequences_to_pdb_ID.items():
        sequence_to_pdb_num[sequence] = len(pdb_ids)

    num_to_num = {}
    num_to_num = {
        '1' : 0,
        '2-10' : 0,
        '11-30' : 0,
        '31-50' : 0,
        '51-100' : 0,
        '101-' : 0,
    }
    for k, v in sequence_to_pdb_num.items():
        if v == 1:
            num_to_num['1'] += 1
        elif v <= 10:
            num_to_num['2-10'] += 1
        elif v <= 30:
            num_to_num['11-30'] += 1
        elif v <= 50:
            num_to_num['31-50'] += 1
        elif v <= 100:
            num_to_num['51-100'] += 1
        else:
            num_to_num['101-'] += 1

    # draw histogram
    # fig size
    plt.figure(figsize=(10, 5))
    plt.bar(num_to_num.keys(), num_to_num.values())
    plt.xlabel('Number of PDB IDs')
    plt.ylabel('Number of sequences')
    plt.title('Number of sequences with different number of PDB IDs')
    plt.savefig(os.path.join(save_path, "number_of_sequences.png"))
    plt.close()

    most_frequent_sequences = sorted(sequence_to_pdb_num.items(), key=lambda x: x[1], reverse=True)[:10]

    breakpoint()

    # filter by length
    single_state_sequences = {k: v for k, v in single_state_sequences.items() if length_filter[0] <= len(k) <= length_filter[1]}
    multi_state_sequences = {k: v for k, v in multi_state_sequences.items() if length_filter[0] <= len(k) <= length_filter[1]}

    single_state_sequences = {sequence_hashes[k]: v for k, v in single_state_sequences.items()}
    multi_state_sequences = {sequence_hashes[k]: v for k, v in multi_state_sequences.items()}

    print(f"Number of single state sequences: {len(single_state_sequences)}")
    print(f"Number of multi state sequences: {len(multi_state_sequences)}")
    
    return single_state_sequences, multi_state_sequences


if __name__ == "__main__":
    single_state_sequences, multi_state_sequences = get_multi_pdb_ID_protein_seqs()
    # filtered_sequences = filter_multi_state_sequences(multi_state_sequences)