import os
import datetime
import pickle
from BioMol import BioMol
from joblib import Parallel, delayed
from Bio.PDB.MMCIF2Dict import MMCIF2Dict as mmcif2dict
import gzip

cif_dir = "/public_data/psk6950/PDB_2024Mar18/cif"


def list_AbAg(tsv_path, merged_fasta_path, save_path):
    chain_ID_to_seq = {}
    with open(merged_fasta_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                chain_id = line[1:].strip().split("|")[0].strip()
                chain_ID_to_seq[chain_id] = ""
            else:
                chain_ID_to_seq[chain_id] += line.strip()

    lines = []
    with open(tsv_path, "r") as f:
        for line in f:
            lines.append(line.strip().split("\t"))
    header = lines[0]
    lines = lines[1:]

    # filter out wo antigen data
    # with_antigen_lines = [line for line in lines if line[4] != '' and line[4] != 'NA']

    with_antigen_lines = []
    for line in lines:
        try:
            if line[4] != "" and line[4] != "NA":
                with_antigen_lines.append(line)
        except:
            breakpoint()
            print(line)

    # TODO : In this version, I only get protein or peptide antigen data
    # For the next version, I will get all antigen data including DNA, RNA, etc.
    antigen_type = [line[5] for line in with_antigen_lines]  # 'Protein'
    antigen_type = [
        "protein" in antigen.lower() or "peptide" in antigen.lower()
        for antigen in antigen_type
    ]
    protein_antigen_lines = [
        line for line, is_protein in zip(with_antigen_lines, antigen_type) if is_protein
    ]
    pdb_id = [line[0] for line in protein_antigen_lines]
    new_chain_ID_to_seq = {}
    for chain_id in chain_ID_to_seq:
        if chain_id.split("_")[0] in pdb_id:
            new_chain_ID_to_seq[chain_id] = chain_ID_to_seq[chain_id]
    chain_ID_to_seq = new_chain_ID_to_seq

    Ab_chain_ids = []

    def _get_ab_data(line):
        output = {}
        pdb_id, auth_id1, auth_id2 = line[0], line[1], line[2]
        cif_path = cif_dir + f"/{pdb_id[1:3]}/{pdb_id}.cif.gz"
        if not os.path.exists(cif_path):
            return None

        if cif_path.endswith(".cif"):
            mmcif_dict = mmcif2dict(cif_path)
        elif cif_path.endswith(".cif.gz"):
            with gzip.open(cif_path, "rt") as f:
                mmcif_dict = mmcif2dict(f)
        group_pdb = mmcif_dict["_atom_site.group_PDB"]
        label_asym_id = mmcif_dict["_atom_site.label_asym_id"]
        auth_asym_id = mmcif_dict["_atom_site.auth_asym_id"]
        pair = [
            (label, auth)
            for group, label, auth in zip(group_pdb, label_asym_id, auth_asym_id)
            if group == "ATOM"
        ]
        pair = list(set(pair))
        auth_chain_id_to_label_id = {auth: label for label, auth in pair}

        try:
            if auth_id1 != "NA" and auth_id1 != "":
                label_id1 = auth_chain_id_to_label_id[auth_id1]
                heavy_chain_id = f"{pdb_id}_{label_id1}"
                heavy_chain_seq = chain_ID_to_seq[heavy_chain_id]
                output["Heavy"] = (heavy_chain_id, heavy_chain_seq)
            if auth_id2 != "NA" and auth_id2 != "" and auth_id2.isupper():
                label_id2 = auth_chain_id_to_label_id[auth_id2]
                light_chain_id = f"{pdb_id}_{label_id2}"
                light_chain_seq = chain_ID_to_seq[light_chain_id]
                output["Light"] = (light_chain_id, light_chain_seq)
            return output
        except Exception as e:
            return e, line

    # using joblib to parallelize the process
    # protein_antigen_lines = protein_antigen_lines[:1000]
    print(f"Total number of protein antigen data: {len(protein_antigen_lines)}")
    Ab_chain_ids = Parallel(n_jobs=-1)(
        delayed(_get_ab_data)(line) for line in protein_antigen_lines
    )
    print(f"End of parallel processing")
    Ab_chain_ids = [data for data in Ab_chain_ids if data is not None]
    test = [data for data in Ab_chain_ids if type(data) == tuple]
    no_heavy_chain = [data for data in Ab_chain_ids if "Heavy" not in data]

    # pickle save
    with open("./Ab_chain_ids.pkl", "wb") as f:
        pickle.dump(Ab_chain_ids, f)

    with open(save_path, "w") as f:
        for data in Ab_chain_ids:
            if "Heavy" in data:
                heavy_chain_id, heavy_chain_seq = data["Heavy"]
                f.write(f">{heavy_chain_id} | Heavy\n{heavy_chain_seq}\n")
            if "Light" in data:
                light_chain_id, light_chain_seq = data["Light"]
                f.write(f">{light_chain_id} | Light\n{light_chain_seq}\n")
            if "Heavy" not in data and "Light" not in data:
                raise ValueError(f"Data: {data}")


def debug___():
    save_path = "/public_data/psk6950/PDB_2024Mar18/AbAg/Ag.fasta"
    Ab_chain_ids = pickle.load(open("./Ab_chain_ids.pkl", "rb"))

    errors = {}

    with open(save_path, "w") as f:
        for data in Ab_chain_ids:
            if type(data) == dict:
                if "Heavy" in data:
                    heavy_chain_id, heavy_chain_seq = data["Heavy"]
                    f.write(f">{heavy_chain_id} | Heavy\n{heavy_chain_seq}\n")
                if "Light" in data:
                    light_chain_id, light_chain_seq = data["Light"]
                    f.write(f">{light_chain_id} | Light\n{light_chain_seq}\n")
            else:
                pdb_id = data[0]
                errors[pdb_id] = data
                continue

    breakpoint()


if __name__ == "__main__":
    chain_ID_to_cluster_path = "/public_data/psk6950/PDB_2024Mar18/protein_seq_clust/mmseqs2_seqid30_cov80_covmode0_clustmode1_chainID_to_cluster.pkl"
    tsv_path = "/public_data/psk6950/PDB_2024Mar18/AbAg/sabdab_summary_fixed.tsv"
    sequence_hash_file = "/public_data/psk6950/PDB_2024Mar18/entity/sequence_hashes.pkl"
    merged_fasta_path = "/public_data/psk6950/PDB_2024Mar18/entity/merged_protein.fasta"
    save_path = "/public_data/psk6950/PDB_2024Mar18/AbAg/Ag.fasta"

    list_AbAg(tsv_path, merged_fasta_path, save_path)
    breakpoint()

    # debug___()
