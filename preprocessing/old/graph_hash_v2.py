import torch
from BioMol import BioMol
from utils.parser import parse_cif
from joblib import Parallel, delayed
from filelock import FileLock
import pickle
import os
import gc
import networkx as nx

# 20250303, protein only

chainID_to_cluster_path = (
    "/data/psk6950/PDB_2024Oct21/protein_seq_clust/v2_chainID_to_cluster.pkl"
)
with open(chainID_to_cluster_path, "rb") as f:
    chainID_to_cluster = pickle.load(f)

CONTACT_TH = 8.0


class TooManyChainsError(Exception):
    pass


def get_cluster_id(chain_id):
    return chainID_to_cluster[chain_id]


@torch.no_grad()
def get_graph_from_biomol_structure(biomol_structure):
    residue_tensor = biomol_structure.residue_tensor
    residue_chain_break = biomol_structure.residue_chain_break
    chain_num = len(residue_chain_break)
    if chain_num > 100:
        raise TooManyChainsError

    residue_mask = residue_tensor[:, 4].bool()
    residue_xyz = residue_tensor[:, 5:8]

    chain_list = list(residue_chain_break.keys())
    total_residue = residue_tensor.shape[0]
    residue_chain_idx = torch.zeros(total_residue, dtype=torch.long)
    for ii in range(chain_num):
        start_i, end_i = residue_chain_break[chain_list[ii]]
        residue_chain_idx[start_i : end_i + 1] = ii

    residue_chain_idx = torch.nn.functional.one_hot(
        residue_chain_idx, num_classes=chain_num
    )  # (total_residue, chain_num)
    residue_chain_idx = residue_chain_idx.to(torch.bfloat16)

    dist_map = torch.cdist(residue_xyz, residue_xyz)
    dist_map = dist_map + torch.eye(dist_map.shape[0]) * 1000
    dist_map[residue_mask == 0, :] = 1000
    dist_map[:, residue_mask == 0] = 1000
    contact_ij = dist_map < CONTACT_TH  # (total_residue, total_residue)
    contact_ij = contact_ij.to(torch.bfloat16)

    edge = []

    chain_contact = (
        residue_chain_idx.T @ contact_ij @ residue_chain_idx
    )  # (chain_batch, chain_batch)
    contact_ii, contact_jj = torch.where(chain_contact > 0)
    edge = [(i, j) for i, j in zip(contact_ii, contact_jj)]

    # remove self contact
    edge = [e for e in edge if e[0] < e[1]]

    ID = biomol_structure.ID
    chain_list = [chain_ID.split("_")[0] for chain_ID in chain_list]
    chain_list = [f"{ID}_{chain_list[i]}" for i in range(len(chain_list))]
    node = [get_cluster_id(chain_ID) for chain_ID in chain_list]

    return node, edge


save_dir = "/data/psk6950/PDB_2024Oct21/protein_graph/"


def save_graph_from_cif(cif_path):
    ID = cif_path.split("/")[-1].split(".")[0]
    save_path = f"{save_dir}{ID[1:3]}/{ID}.graph"
    biomol = BioMol(
        cif=cif_path, cif_config="./cif_configs/protein_only.json"
    )  # 20250303, protein only
    graphs = {}

    for assembly_id in biomol.bioassembly.keys():
        for model_id in biomol.bioassembly[assembly_id]:
            for alt_id in biomol.bioassembly[assembly_id][model_id]:
                biomol_structure = biomol.bioassembly[assembly_id][model_id][alt_id]
                graph = get_graph_from_biomol_structure(biomol_structure)
                graphs[(assembly_id, model_id, alt_id)] = graph
    if graphs == {}:  # no protein
        return

    # t # ID, assembly_ID, model_ID, alt_ID
    # v 0 node1
    # v 1 node2 ...
    # e 0 1
    # e 0 2 ...
    with open(save_path, "w") as f:
        for key in graphs.keys():
            node, edge = graphs[key]
            f.write(f"t # {key}\n")
            for i in range(len(node)):
                f.write(f"v {i} {node[i]}\n")
            for e in edge:
                f.write(f"e {e[0]} {e[1]}\n")
    return graphs


def _save_graph(cif_path):
    """Process a single CIF file and write the result to CSV immediately."""
    print(f"Loading {cif_path}")
    try:
        save_graph_from_cif(cif_path)
    except Exception as e:
        return cif_path

    return None


def save_graph_v2(cif_dir, Ab_fasta_path):
    remake_pdb_id = []
    with open(Ab_fasta_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith(">"):
            chain_id = line.split("|")[0][1:].strip()
            pdb_id = chain_id.split("_")[0]
            remake_pdb_id.append(pdb_id)
    remake_pdb_id = list(set(remake_pdb_id))

    save_dir = "/data/psk6950/PDB_2024Oct21/protein_graph/"

    # Gather all CIF file paths from the inner directories.
    cif_path_list = []
    for pdb_id in remake_pdb_id:
        cif_path = os.path.join(cif_dir, pdb_id[1:3], f"{pdb_id}.cif.gz")
        if os.path.exists(cif_path):
            cif_path_list.append(cif_path)

    print(f"Processing {len(cif_path_list)} CIF files...")
    output = Parallel(n_jobs=-1)(
        delayed(_save_graph)(cif_path) for cif_path in cif_path_list
    )

    # too many chains
    output = [o for o in output if o is not None]

    too_many_chain_save_path = "./too_many_chain.txt"
    with open(too_many_chain_save_path, "w") as f:
        for o in output:
            f.write(f"{o}\n")
    print(f"Too many chains : {len(output)}")


@torch.no_grad()
def get_graph_from_biomol_structure_for_large(biomol_structure, device="cpu"):
    residue_tensor = biomol_structure.residue_tensor
    residue_chain_break = biomol_structure.residue_chain_break
    residue_mask = residue_tensor[:, 4].bool().to(device)
    residue_xyz = residue_tensor[:, 5:8]
    residue_xyz.to(torch.float16)
    residue_xyz = residue_xyz.to(device)

    chain_num = len(residue_chain_break)

    edge = []

    chain_list = list(residue_chain_break.keys())
    total_residue = residue_tensor.shape[0]
    residue_chain_idx = torch.zeros(total_residue, dtype=torch.long, device=device)
    for ii in range(chain_num):
        start_i, end_i = residue_chain_break[chain_list[ii]]
        residue_chain_idx[start_i : end_i + 1] = ii

    # residue_chain_idx = torch.nn.functional.one_hot(residue_chain_idx, num_classes=chain_num) # (total_residue, chain_num)

    chain_batch = 5  # ...
    start_ii = 0
    for ii in range(0, chain_num, chain_batch):
        last_chain_id = chain_list[min(ii + chain_batch, chain_num) - 1]
        end_ii = residue_chain_break[last_chain_id][1]
        ii_idx = torch.arange(start_ii, end_ii + 1, device=device)
        start_jj = 0
        residue_chain_ii = residue_chain_idx[ii_idx]
        residue_chain_ii = torch.nn.functional.one_hot(
            residue_chain_ii, num_classes=chain_num
        ).to(torch.bfloat16)  # (ii, chain_num)
        residue_xyz_ii = residue_xyz[ii_idx]
        for jj in range(0, chain_num, chain_batch):
            last_chain_id = chain_list[min(jj + chain_batch, chain_num) - 1]
            end_jj = residue_chain_break[last_chain_id][1]
            jj_idx = torch.arange(start_jj, end_jj + 1, device=device)
            residue_xyz_jj = residue_xyz[jj_idx]
            dist_ij = (residue_xyz_ii[:, None, :] - residue_xyz_jj[None, :, :]).norm(
                dim=-1
            )
            dist_ij[residue_mask[ii_idx] == 0, :] = 1000
            dist_ij[:, residue_mask[jj_idx] == 0] = 1000
            contact_ij = dist_ij < CONTACT_TH  # (ii, jj)
            contact_ij = contact_ij.to(torch.bfloat16)

            # residue_chain_jj = residue_chain_idx[jj_idx].to(torch.bfloat16) # (jj, chain_num)
            residue_chain_jj = torch.nn.functional.one_hot(
                residue_chain_idx[jj_idx], num_classes=chain_num
            ).to(torch.bfloat16)  # (jj, chain_num)

            chain_contact = (
                residue_chain_ii.T @ contact_ij @ residue_chain_jj
            )  # (chain_num, chain_num)
            contact_ii, contact_jj = torch.where(chain_contact > 0)
            for i, j in zip(contact_ii, contact_jj):
                edge.append((i.item(), j.item()))

            start_jj = end_jj + 1
        start_ii = end_ii + 1
    # remove self contact
    edge = [e for e in edge if e[0] < e[1]]
    # remove duplicate
    edge = list(set(edge))
    ID = biomol_structure.ID
    chain_list = [chain_ID.split("_")[0] for chain_ID in chain_list]
    chain_list = [f"{ID}_{chain_list[i]}" for i in range(len(chain_list))]
    node = [get_cluster_id(chain_ID) for chain_ID in chain_list]

    return node, edge


def save_graph_from_cif_large_protein(cif_path, device="cpu"):
    save_dir = "/data/psk6950/PDB_2024Oct21/protein_graph/"
    ID = cif_path.split("/")[-1].split(".")[0]
    save_path = f"{save_dir}{ID[1:3]}/{ID}.graph"
    # if os.path.exists(save_path):
    #     return
    biomol = BioMol(
        cif=cif_path, cif_config="./cif_configs/protein_only.json"
    )  # 20250303, protein only
    graphs = {}

    for assembly_id in biomol.bioassembly.keys():
        for model_id in biomol.bioassembly[assembly_id]:
            for alt_id in biomol.bioassembly[assembly_id][model_id]:
                biomol_structure = biomol.bioassembly[assembly_id][model_id][alt_id]
                graph = get_graph_from_biomol_structure_for_large(
                    biomol_structure, device=device
                )
                graphs[(assembly_id, model_id, alt_id)] = graph
    if graphs == {}:  # no protein
        return

    # t # ID, assembly_ID, model_ID, alt_ID
    # v 0 node1
    # v 1 node2 ...
    # e 0 1
    # e 0 2 ...
    with open(save_path, "w") as f:
        for key in graphs.keys():
            node, edge = graphs[key]
            f.write(f"t # {key}\n")
            for i in range(len(node)):
                f.write(f"v {i} {node[i]}\n")
            for e in edge:
                f.write(f"e {e[0]} {e[1]}\n")
    gc.collect()
    return graphs


def save_graph_v2_too_many_chain():
    with open("./too_many_chain.txt", "r") as f:
        lines = f.readlines()
    cif_path_list = [line.strip() for line in lines]
    print(f"Processing {len(cif_path_list)} CIF files...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    for nn, cif_path in enumerate(cif_path_list):
        print(f"{nn} | cif_path: {cif_path}")
        save_graph_from_cif_large_protein(cif_path, device=device)


if __name__ == "__main__":
    # fix the graph for AbAg
    Ab_fasta_path = "/data/psk6950/PDB_2024Oct21/AbAg/Ab.fasta"
    cif_dir = "/data/psk6950/PDB_2024Oct21/cif/"
    save_graph_v2(cif_dir, Ab_fasta_path)
    # save_graph_v2_too_many_chain()
