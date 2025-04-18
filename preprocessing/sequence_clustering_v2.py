'''
mmseqs + AbAg clustring (H3/L3(only light chain))
'''

import os
import pickle
import copy

def parse_cdhit_result(path):
    with open(path, 'r') as f:
        text = f.read()

    clusters = []
    current_cluster = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Detect a new cluster
        if line.startswith('>Cluster'):
            if current_cluster:
                clusters.append(current_cluster)
                current_cluster = []
        else:
            # Split the line by comma to isolate the part with the chain id.
            parts = line.split(',')
            if len(parts) > 1:
                chain_part = parts[1].strip()
                if chain_part.startswith('>'):
                    # Extract the chain id from the first whitespace-separated token
                    chain_id = chain_part.split()[0]
                    # Remove trailing dots and the '>' sign
                    chain_id = chain_id.lstrip('>').rstrip('.')
                    current_cluster.append(chain_id)
    if current_cluster:
        clusters.append(current_cluster)
    return clusters

def v2_cluster(v1_cluster_path, v2_cluster_path, Ab_fasta_path, Ab_cluster_path):
    with open(v1_cluster_path, 'rb') as f:
        v1_cluster = pickle.load(f)

    v2_cluster = copy.deepcopy(v1_cluster)

    cluster_hash = list(v1_cluster.values())
    cluster_hash = sorted(list(set(cluster_hash)))
    max_cluster = max(cluster_hash)

    # parse cd-hit result
    cdhit_clusters = parse_cdhit_result(Ab_cluster_path)

    chain_conjugated = {}

    with open(Ab_fasta_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('>'):
            chain_id = line.split('|')[0][1:].strip()
            pdb_id = chain_id.split('_')[0]
            if pdb_id not in chain_conjugated:
                chain_conjugated[pdb_id] = []
            chain_conjugated[pdb_id].append(chain_id)

    for cdhit_cluster in cdhit_clusters:
        max_cluster += 1
        for chain_id in cdhit_cluster:
            pdb_id = chain_id.split('_')[0]
            for conjugated_chain_id in chain_conjugated[pdb_id]:
                v2_cluster[conjugated_chain_id] = max_cluster

    with open(v2_cluster_path, 'wb') as f:
        pickle.dump(v2_cluster, f, protocol=pickle.HIGHEST_PROTOCOL)
    breakpoint()

def test_v2_cluster():
    v2_cluster_path = '/public_data/psk6950/PDB_2024Mar18/protein_seq_clust/v2_chainID_to_cluster.pkl'
    with open(v2_cluster_path, 'rb') as f:
        v2_cluster = pickle.load(f)
    breakpoint()

if __name__ == '__main__':
    # v1_cluster_path = '/public_data/psk6950/PDB_2024Mar18/protein_seq_clust/mmseqs2_seqid30_cov80_covmode0_clustmode1_chainID_to_cluster.pkl'
    # v2_cluster_path = '/public_data/psk6950/PDB_2024Mar18/protein_seq_clust/v2_chainID_to_cluster.pkl'
    # Ab_fasta_path = "/public_data/psk6950/PDB_2024Mar18/AbAg/Ab.fasta"
    # Ab_cluster_path = "/public_data/psk6950/PDB_2024Mar18/AbAg/cd_hit_pdb_2024MAr18.dat.clstr"
    # v2_cluster(v1_cluster_path, v2_cluster_path, Ab_fasta_path, Ab_cluster_path)

    test_v2_cluster()
