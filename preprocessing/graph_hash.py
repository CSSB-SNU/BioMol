import torch
from biomol import BioMol
from utils.parser import parse_cif
from joblib import Parallel, delayed
from filelock import FileLock
import pickle
import os
import gc
import networkx as nx

# 20250303, protein only

chainID_to_cluster_path = '/public_data/psk6950/PDB_2024Mar18/protein_seq_clust/mmseqs2_seqid30_cov80_covmode0_clustmode1_chainID_to_cluster.pkl'
with open(chainID_to_cluster_path, 'rb') as f:
    chainID_to_cluster = pickle.load(f)

CONTACT_TH = 8.0

class TooManyChainsError(Exception):
    pass

def get_cluster_id(chain_id):
    return chainID_to_cluster[chain_id]

@torch.no_grad()
def get_graph_from_biomol_structure_for_large(biomol_structure, device='cpu'):
    residue_tensor = biomol_structure.residue_tensor
    residue_chain_break = biomol_structure.residue_chain_break
    residue_mask = residue_tensor[:,4].bool().to(device)
    residue_xyz = residue_tensor[:,5:8]
    residue_xyz.to(torch.float16)
    residue_xyz = residue_xyz.to(device)

    chain_num = len(residue_chain_break)

    # dist_map = torch.cdist(residue_xyz, residue_xyz)
    # dist_map = dist_map + torch.eye(dist_map.shape[0]) * 1000
    # dist_map[residue_mask == 0, :] = 1000
    # dist_map[:, residue_mask == 0] = 1000

    edge = []

    chain_list = list(residue_chain_break.keys())
    total_residue = residue_tensor.shape[0]
    residue_chain_idx = torch.zeros(total_residue, dtype=torch.long, device=device)
    for ii in range(chain_num):
        start_i, end_i = residue_chain_break[chain_list[ii]]
        residue_chain_idx[start_i:end_i+1] = ii

    # residue_chain_idx = torch.nn.functional.one_hot(residue_chain_idx, num_classes=chain_num) # (total_residue, chain_num)

    chain_batch = 5 # ...
    start_ii = 0
    for ii in range(0, chain_num, chain_batch):
        last_chain_id = chain_list[min(ii+chain_batch, chain_num)-1]
        end_ii = residue_chain_break[last_chain_id][1]
        ii_idx = torch.arange(start_ii, end_ii+1, device=device)
        start_jj = 0
        residue_chain_ii = residue_chain_idx[ii_idx]
        residue_chain_ii = torch.nn.functional.one_hot(residue_chain_ii, num_classes=chain_num).to(torch.bfloat16) # (ii, chain_num)
        residue_xyz_ii = residue_xyz[ii_idx]
        for jj in range(0, chain_num, chain_batch):
            last_chain_id = chain_list[min(jj+chain_batch, chain_num)-1]
            end_jj = residue_chain_break[last_chain_id][1]
            jj_idx = torch.arange(start_jj, end_jj+1, device=device)
            residue_xyz_jj = residue_xyz[jj_idx]
            dist_ij = (residue_xyz_ii[:, None, :] - residue_xyz_jj[None,:, :]).norm(dim=-1)
            dist_ij[residue_mask[ii_idx]==0, :] = 1000
            dist_ij[:, residue_mask[jj_idx]==0] = 1000
            contact_ij = dist_ij < CONTACT_TH # (ii, jj)
            contact_ij = contact_ij.to(torch.bfloat16)

            # residue_chain_jj = residue_chain_idx[jj_idx].to(torch.bfloat16) # (jj, chain_num)
            residue_chain_jj = torch.nn.functional.one_hot(residue_chain_idx[jj_idx], num_classes=chain_num).to(torch.bfloat16) # (jj, chain_num)

            chain_contact = residue_chain_ii.T @ contact_ij @ residue_chain_jj # (chain_num, chain_num)
            contact_ii, contact_jj = torch.where(chain_contact>0)
            for i, j in zip(contact_ii, contact_jj):
                edge.append((i.item(), j.item()))

            start_jj = end_jj + 1
        start_ii = end_ii + 1
    # remove self contact
    edge = [e for e in edge if e[0] < e[1]]
    # remove duplicate
    edge = list(set(edge))

    # if chain_num > 500 :
    #     raise TooManyChainsError

    # for ii in range(chain_num):
    #     start_i, end_i = residue_chain_break[chain_list[ii]]
    #     xyz_ii = residue_xyz[start_i:end_i+1]
    #     mask_ii = residue_mask[start_i:end_i+1]
    #     # dist_i = torch.cdist(xyz_ii, residue_xyz)
    #     dist_i = (xyz_ii[:, None, :] - residue_xyz[None, :, :]).norm(dim=-1)
    #     dist_i[mask_ii == 0, :] = 1000
    #     dist_i, _ = dist_i.min(dim=0)
    #     for jj in range(ii+1, chain_num):
    #         start_j, end_j = residue_chain_break[chain_list[jj]]
    #         mask_jj = residue_mask[start_j:end_j+1]

    #         dist_ij = dist_i[start_j:end_j+1]
    #         dist_ij[mask_jj == 0] = 1000
    #         if torch.min(dist_ij) < CONTACT_TH:
    #             edge.append((ii, jj))
    ID = biomol_structure.ID
    chain_list = [chain_ID.split('_')[0] for chain_ID in chain_list]
    chain_list = [f'{ID}_{chain_list[i]}' for i in range(len(chain_list))]
    node = [get_cluster_id(chain_ID) for chain_ID in chain_list]

    return node, edge

@torch.no_grad()
def get_graph_from_biomol_structure(biomol_structure):
    residue_tensor = biomol_structure.residue_tensor
    residue_chain_break = biomol_structure.residue_chain_break
    chain_num = len(residue_chain_break)
    if chain_num > 100 :
        raise TooManyChainsError
    
    residue_mask = residue_tensor[:,4].bool()
    residue_xyz = residue_tensor[:,5:8]

    chain_list = list(residue_chain_break.keys())
    total_residue = residue_tensor.shape[0]
    residue_chain_idx = torch.zeros(total_residue, dtype=torch.long)
    for ii in range(chain_num):
        start_i, end_i = residue_chain_break[chain_list[ii]]
        residue_chain_idx[start_i:end_i+1] = ii

    residue_chain_idx = torch.nn.functional.one_hot(residue_chain_idx, num_classes=chain_num) # (total_residue, chain_num)
    residue_chain_idx = residue_chain_idx.to(torch.bfloat16)

    dist_map = torch.cdist(residue_xyz, residue_xyz)
    dist_map = dist_map + torch.eye(dist_map.shape[0]) * 1000
    dist_map[residue_mask == 0, :] = 1000
    dist_map[:, residue_mask == 0] = 1000
    contact_ij = dist_map < CONTACT_TH # (total_residue, total_residue)
    contact_ij = contact_ij.to(torch.bfloat16)

    edge = []
    
    chain_contact = residue_chain_idx.T @ contact_ij @ residue_chain_idx # (chain_batch, chain_batch)
    contact_ii, contact_jj = torch.where(chain_contact>0)
    edge = [(i, j) for i, j in zip(contact_ii, contact_jj)]

    # remove self contact
    edge = [e for e in edge if e[0] < e[1]]
    
    ID = biomol_structure.ID
    chain_list = [chain_ID.split('_')[0] for chain_ID in chain_list]
    chain_list = [f'{ID}_{chain_list[i]}' for i in range(len(chain_list))]
    node = [get_cluster_id(chain_ID) for chain_ID in chain_list]

    return node, edge


def save_graph_from_cif(cif_path):
    save_dir = '/public_data/psk6950/PDB_2024Mar18/protein_graph/'
    ID = cif_path.split('/')[-1].split('.')[0]
    save_path = f'{save_dir}{ID[1:3]}/{ID}.graph'
    biomol = BioMol(cif=cif_path, cif_config='utils/cif_configs/protein_only.json') # 20250303, protein only
    graphs = {}

    for assembly_id in biomol.bioassembly.keys():
        for model_id in biomol.bioassembly[assembly_id]:
            for alt_id in biomol.bioassembly[assembly_id][model_id]:
                biomol_structure = biomol.bioassembly[assembly_id][model_id][alt_id]
                graph = get_graph_from_biomol_structure(biomol_structure)
                graphs[(assembly_id, model_id, alt_id)] = graph
    if graphs == {}: # no protein
        return

    # t # ID, assembly_ID, model_ID, alt_ID
    # v 0 node1
    # v 1 node2 ...
    # e 0 1
    # e 0 2 ...
    with open(save_path, 'w') as f:
        for key in graphs.keys():
            node, edge = graphs[key]
            f.write(f't # {key}\n')
            for i in range(len(node)):
                f.write(f'v {i} {node[i]}\n')
            for e in edge:
                f.write(f'e {e[0]} {e[1]}\n')
    return graphs


def save_graph_from_cif_large_protein(cif_path, device='cpu'):
    save_dir = '/public_data/psk6950/PDB_2024Mar18/protein_graph/'
    ID = cif_path.split('/')[-1].split('.')[0]
    save_path = f'{save_dir}{ID[1:3]}/{ID}.graph'
    # if os.path.exists(save_path):
    #     return
    biomol = BioMol(cif=cif_path, cif_config='utils/cif_configs/protein_only.json') # 20250303, protein only
    graphs = {}

    for assembly_id in biomol.bioassembly.keys():
        for model_id in biomol.bioassembly[assembly_id]:
            for alt_id in biomol.bioassembly[assembly_id][model_id]:
                biomol_structure = biomol.bioassembly[assembly_id][model_id][alt_id]
                graph = get_graph_from_biomol_structure_for_large(biomol_structure, device=device)
                graphs[(assembly_id, model_id, alt_id)] = graph
    if graphs == {}: # no protein
        return


    # t # ID, assembly_ID, model_ID, alt_ID
    # v 0 node1
    # v 1 node2 ...
    # e 0 1
    # e 0 2 ...
    with open(save_path, 'w') as f:
        for key in graphs.keys():
            node, edge = graphs[key]
            f.write(f't # {key}\n')
            for i in range(len(node)):
                f.write(f'v {i} {node[i]}\n')
            for e in edge:
                f.write(f'e {e[0]} {e[1]}\n')
    gc.collect()
    return graphs

def _save_graph(cif_path, pass_cif_path, lock_path):
    """Process a single CIF file and write the result to CSV immediately."""
    print(f"Loading {cif_path}")
    try:
        save_graph_from_cif(cif_path)
        # Write the result immediately, using a file lock to avoid conflicts.
        with FileLock(lock_path):
            with open(pass_cif_path, 'a') as f:
                f.write(f"{cif_path},-\n")
    except TooManyChainsError as e:
        # Write the error immediately, using a file lock to avoid conflicts.
        with FileLock(lock_path):
            with open(pass_cif_path, 'a') as f:
                f.write(f"{cif_path},TooManyChainsError\n")
    except Exception as e:
        print(f"Error processing {cif_path}: {e}")
        # Write the error immediately, using a file lock to avoid conflicts.
        with FileLock(lock_path):
            with open(pass_cif_path, 'a') as f:
                f.write(f"{cif_path},{e}\n")

    # save_graph_from_cif(cif_path)
    # # Write the result immediately, using a file lock to avoid conflicts.
    # with FileLock(lock_path):
    #     with open(pass_cif_path, 'a') as f:
    #         f.write(f"{cif_path},-\n")
    # gc.collect()
    
    return cif_path


def save_graph(cif_dir, passed_path="passed_cif.csv"):
    # Create the CSV file if it doesn't exist.
    if not os.path.exists(passed_path):
        with open(passed_path, 'w') as f:
            pass  # Just create an empty file
    
    # Define a lock file path (it will create a lock file alongside your CSV)
    lock_path = passed_path + ".lock"

    # Read previously processed CIF paths from the CSV.
    passed_cif_path_list = []
    not_passed_cif_path_list = []
    with open(passed_path, 'r') as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                cif_path, err = parts[0], parts[1]
                # if err in ["-", "water", "mmcif_error", "StructConnAmbiguityError"]:
                if err in ["-", "water", "mmcif_error"]:
                    passed_cif_path_list.append(cif_path)
                else:
                    not_passed_cif_path_list.append(cif_path)

    not_passed_cif_path_list = list(set(not_passed_cif_path_list))

    save_dir = '/public_data/psk6950/PDB_2024Mar18/protein_graph/'

    # Gather all CIF file paths from the inner directories.
    cif_path_list = []
    for inner_dir in os.listdir(cif_dir):
        inner_dir_path = os.path.join(cif_dir, inner_dir)
        if not os.path.isdir(inner_dir_path):
            continue
        
        if not os.path.exists(os.path.join(save_dir, inner_dir)):
            os.makedirs(os.path.join(save_dir, inner_dir))

        for file_name in os.listdir(inner_dir_path):
            if file_name.endswith('.cif') or file_name.endswith('.cif.gz'):
                full_path = os.path.join(inner_dir_path, file_name)
                if full_path not in passed_cif_path_list:
                    cif_path_list.append(full_path)

    print(f"Processing {len(cif_path_list)} CIF files...")

    # # Gather all CIF file paths from the inner directories.
    # cif_path_list = []
    # for file_name in os.listdir(cif_dir):
    #     if file_name.endswith('.cif') or file_name.endswith('.cif.gz'):
    #         full_path = os.path.join(cif_dir, file_name)
    #         cif_path_list.append(full_path)

    # Process each CIF file in parallel.
    # Each worker writes its result to the CSV immediately.
    Parallel(n_jobs=-1)(
        delayed(_save_graph)(cif_path, passed_path, lock_path)
        for cif_path in cif_path_list
    )


def save_graph_large_protein(cif_dir, passed_path="passed_cif.csv", device='cpu'):
    # Create the CSV file if it doesn't exist.
    if not os.path.exists(passed_path):
        with open(passed_path, 'w') as f:
            pass  # Just create an empty file
    
    # Define a lock file path (it will create a lock file alongside your CSV)
    lock_path = passed_path + ".lock"

    # Read previously processed CIF paths from the CSV.
    passed_cif_path_list = []
    not_passed_cif_path_list = []
    with open(passed_path, 'r') as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                cif_path, err = parts[0], parts[1]
                # if err in ["-", "water", "mmcif_error", "StructConnAmbiguityError"]:
                if err in ["-"]:
                    passed_cif_path_list.append(cif_path)
                else:
                    not_passed_cif_path_list.append(cif_path)

    not_passed_cif_path_list = list(set(not_passed_cif_path_list))

    save_dir = '/public_data/psk6950/PDB_2024Mar18/protein_graph/'

    # Gather all CIF file paths from the inner directories.
    cif_path_list = []
    for inner_dir in os.listdir(cif_dir):
        inner_dir_path = os.path.join(cif_dir, inner_dir)
        if not os.path.isdir(inner_dir_path):
            continue
        
        if not os.path.exists(os.path.join(save_dir, inner_dir)):
            os.makedirs(os.path.join(save_dir, inner_dir))

        for file_name in os.listdir(inner_dir_path):
            if file_name.endswith('.cif') or file_name.endswith('.cif.gz'):
                full_path = os.path.join(inner_dir_path, file_name)
                if full_path not in passed_cif_path_list:
                    cif_path_list.append(full_path)

    print(f"Processing {len(cif_path_list)} CIF files...")

    for nn,cif_path in enumerate(cif_path_list):
        print(f"{nn} | cif_path: {cif_path}")
        save_graph_from_cif_large_protein(cif_path, device=device)

def merge_graph_data(graph_path, save_path):
    pass


def parse_graph(file_path):
    G = nx.Graph()  # Use nx.DiGraph() for directed graphs
    with open(file_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts: 
                continue
            if parts[0] == 'v':  # Vertex line: "v id label"
                node_id = int(parts[1])
                label = parts[2]
                G.add_node(node_id, label=label)
            elif parts[0] == 'e':  # Edge line: "e src tgt"
                src, tgt = int(parts[1]), int(parts[2])
                G.add_edge(src, tgt)
    return G

# # Parse two graphs from their respective files
# G1 = parse_graph('graph1.txt')
# G2 = parse_graph('graph2.txt')

# # Check if they are isomorphic
# print(nx.is_isomorphic(G1, G2))




if __name__ == '__main__':
    cif_dir = "/public_data/psk6950/PDB_2024Mar18/cif/"
    # save_graph(cif_dir, passed_path="graph_hash_passed_cif.csv")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    save_graph_large_protein(cif_dir, passed_path="graph_hash_passed_cif.csv", device=device)
    # cif_path = '/public_data/psk6950/PDB_2024Mar18/cif/wd/6wds.cif.gz'
    # save_graph_from_cif_large_protein(cif_path, device=device)

    # test_cif = '/public_data/psk6950/PDB_2024Mar18/cif/q1/6q1f.cif.gz'
    # save_graph_from_cif_large_protein(test_cif, device=device)