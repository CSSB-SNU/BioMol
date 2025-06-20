"""
mmseqs + AbAg clustring (H3/L3(only light chain)) + NA + Ligand (Following AF3)
"""

import os
import pickle
import copy
from BioMol import DB_PATH, SEQ_TO_HASH_PATH


with open(SEQ_TO_HASH_PATH, "rb") as f:
    seq_to_hash = pickle.load(f)
molecule_type_map = {
    'PolymerType.PROTEIN': '[PROTEIN]:',
    'PolymerType.DNA': '[DNA]:',
    'PolymerType.RNA': '[RNA]:',
    'PolymerType.NA_HYBRID': '[NA_HYBRID]:',
    'NONPOLYMER': '[NONPOLYMER]:',
    'BRANCHED': '[BRANCHED]:',
}

def parse_fasta(fasta_path):
    chain_ID_to_seq = {}

    molecule_type_tag = ''

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line[0] == ">":
                line = line[1:]  # Remove the '>' character
                chain_ID = line.split("| ")[0].strip()
                molecule_type_tag = molecule_type_map[line.split("| ")[-1]]
            else:
                if '  | ' in line:
                    line = line.replace('  | ', '|')
                seq = molecule_type_tag + line.strip()
                chain_ID_to_seq[chain_ID] = seq

    return chain_ID_to_seq

def seq_cluster(
    chain_ID_to_seq,
    protein_cluster_path, 
    all_mol_chain_ID_to_cluster_path,
    all_mol_seq_to_cluster_path,
    ):
    '''
    Following AF3, NA + ligand -> 100% identity clustering
    '''
    with open(protein_cluster_path, "rb") as f:
        protein_cluster = pickle.load(f)

    protein_cluster_hash = list(protein_cluster.values())
    protein_cluster_hash = sorted(list(set(protein_cluster_hash)))
    max_cluster = max(protein_cluster_hash)

    all_mol_chain_ID_list = list(chain_ID_to_seq.keys())

    hash_to_cluster = {}

    all_mol_chain_ID_to_cluster = copy.deepcopy(protein_cluster)
    all_mol_seq_to_cluster = {}

    for chain_ID in all_mol_chain_ID_list:
        seq = chain_ID_to_seq[chain_ID]
        seq_hash = seq_to_hash[seq]
        if seq_hash not in hash_to_cluster:
            max_cluster += 1
            cluster_id = max_cluster
        else:
            cluster_id = hash_to_cluster[seq_hash]

        all_mol_chain_ID_to_cluster[chain_ID] = cluster_id
        all_mol_seq_to_cluster[seq] = cluster_id

    with open(all_mol_chain_ID_to_cluster_path, "wb") as f:
        pickle.dump(all_mol_chain_ID_to_cluster, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(all_mol_seq_to_cluster_path, "wb") as f:
        pickle.dump(all_mol_seq_to_cluster, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    merged_fasta_path = f"{DB_PATH}/entity/merged_all_molecules.fasta"
    chain_ID_to_seq = parse_fasta(merged_fasta_path)

    protein_cluster_path = f"{DB_PATH}/seq_clust/protein_seq_clust/v2_chainID_to_cluster.pkl"
    all_mol_chain_ID_to_cluster_path = f"{DB_PATH}/seq_clust/chain_ID_to_cluster.pkl"
    all_mol_seq_to_cluster_path = f"{DB_PATH}/seq_clust/seq_to_cluster.pkl"

    seq_cluster(
        chain_ID_to_seq,
        protein_cluster_path, 
        all_mol_chain_ID_to_cluster_path,
        all_mol_seq_to_cluster_path,
    )
