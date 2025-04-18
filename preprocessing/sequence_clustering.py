import os
import pickle

def filter_protein(fasta_txt):
    first_line = fasta_txt.split('\n')[0]
    is_protein = first_line.split('| ')[-1] == 'PolymerType.PROTEIN'
    return is_protein

def read_fasta(fasta_path):
    with open(fasta_path, 'r') as f:
        fasta_txt = f.read()
    return fasta_txt

def merge_all_protein_seq(entity_dir, save_path):
    protein_fasta = []
    for inner_dir in os.listdir(entity_dir):
        for file in os.listdir(os.path.join(entity_dir, inner_dir)):
            if file.endswith('.fasta'):
                fasta_txt = read_fasta(os.path.join(entity_dir, inner_dir, file))
                if filter_protein(fasta_txt):
                    protein_fasta.append(fasta_txt)

    with open(save_path, 'w') as f:
        for fasta_txt in protein_fasta:
            f.write(fasta_txt)

def run_mmseqs_cluster(fasta_path, cluster_dir):
    os.system(f'mmseqs easy-cluster {fasta_path} {cluster_dir} /data/psk6950/PDB_2024Mar18/protein_seq_clust/tmp/ --min-seq-id 0.3 -c 0.8 --cov-mode 0 --cluster-mode 1')
    
def save_chainID_to_cluster(mmseq_tsv,
                        mmseq_rep_seq_fasta, 
                        save_path):
    # cluster_hash = idx of representative sequence (rep_seq.fasta)
    rep_seq = []
    with open(mmseq_rep_seq_fasta, 'r') as f:
        for line in f:
            if line.startswith('>'):
                line = line.strip()
                cluster_id = line.split('|')[0][1:].strip()
                rep_seq.append(cluster_id)
    cluster_hash = {rep_seq[i]: i for i in range(len(rep_seq))}

    chainID_to_cluster = {}

    with open(mmseq_tsv, 'r') as f:
        for line in f:
            line = line.strip()
            cluster_id, chain_ID = line.split('\t')
            chainID_to_cluster[chain_ID] = cluster_hash[cluster_id]

    with open(save_path, 'wb') as f:
        pickle.dump(chainID_to_cluster, f, protocol=pickle.HIGHEST_PROTOCOL)    

if __name__ == '__main__':
    entity_dir = '/data/psk6950/PDB_2024Mar18/entity'
    merged_fasta_path = '/data/psk6950/PDB_2024Mar18/entity/merged_protein.fasta'
    # merge_all_protein_seq(entity_dir, merged_fasta_path)
    # run_mmseqs_cluster(merged_fasta_path, '/data/psk6950/PDB_2024Mar18/protein_seq_clust/mmseqs2_seqid30_cov80_covmode0_clustmode1')

    mmseq_tsv = '/data/psk6950/PDB_2024Mar18/protein_seq_clust/mmseqs2_seqid30_cov80_covmode0_clustmode1_cluster.tsv'
    mmseq_rep_seq_fasta = '/data/psk6950/PDB_2024Mar18/protein_seq_clust/mmseqs2_seqid30_cov80_covmode0_clustmode1_rep_seq.fasta'
    save_path = '/data/psk6950/PDB_2024Mar18/protein_seq_clust/mmseqs2_seqid30_cov80_covmode0_clustmode1_chainID_to_cluster.pkl'
    save_chainID_to_cluster(mmseq_tsv, merged_fasta_path, mmseq_rep_seq_fasta, save_path)


