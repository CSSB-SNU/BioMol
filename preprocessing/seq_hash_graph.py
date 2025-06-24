import os
import pickle
import math
from BioMol.BioMol import BioMol
from BioMol import DB_PATH, CONTACT_GRAPH_PATH
from joblib import Parallel, delayed
from BioMol.utils.hierarchy import MoleculeType
import torch

import networkx as nx
from typing import List
import re

metadata_path = os.path.join(DB_PATH, "metadata", "metadata_psk.csv")
hash_to_graph_path = os.path.join(CONTACT_GRAPH_PATH, "unique_graphs.pkl")
CIF_dir = os.path.join(DB_PATH, "cif")
CONTACT_TH = 6.0
seq_to_hash_path = f"{DB_PATH}/entity/sequence_hashes.pkl"
with open(seq_to_hash_path, "rb") as f:
    seq_hashes = pickle.load(f)
molecule_type_map = {
    "PolymerType.PROTEIN": "[PROTEIN]:",
    "PolymerType.DNA": "[DNA]:",
    "PolymerType.RNA": "[RNA]:",
    "PolymerType.NA_HYBRID": "[NA_HYBRID]:",
    "NONPOLYMER": "[NONPOLYMER]:",
    "BRANCHED": "[BRANCHED]:",
}


def load_biomol(cif_ID: str):
    """
    Load the BioMol object for a given CIF ID.

    Parameters:
    cif_ID (str): The CIF ID to load.

    Returns:
    BioMol: The BioMol object for the given CIF ID.
    """
    assert len(cif_ID) == 4, "CIF ID must be 4 characters long"
    path = os.path.join(CIF_dir, cif_ID[1:3], cif_ID + ".cif.gz")
    biomol = BioMol(
        cif=path,
        remove_signal_peptide=False,
        mol_types=["protein", "nucleic_acid", "ligand"],
        use_lmdb=False,
    )
    ID_list = []
    for assembly_ID in biomol.bioassembly.keys():
        for model_ID in biomol.bioassembly[assembly_ID].keys():
            for alt_ID in biomol.bioassembly[assembly_ID][model_ID].keys():
                ID_list.append((assembly_ID, model_ID, alt_ID))
    return biomol, ID_list


def get_sequence_hash(biomolstructure):
    chain_to_hash = {}
    for entity_idx, entity in enumerate(biomolstructure.entity_list):
        chain = list(biomolstructure.residue_chain_break.keys())[entity_idx]
        chain = chain.split("_")[0]
        match entity.get_type():
            case MoleculeType.POLYMER:
                sequence = entity.get_one_letter_code(canonical=True)
                molecule_type = str(entity.get_polymer_type())
            case MoleculeType.NONPOLYMER:
                chem_comp = entity.get_chem_comp()
                sequence = f"({chem_comp})"
                molecule_type = "NONPOLYMER"
            case MoleculeType.BRANCHED:
                chem_comp_list = entity.get_chem_comp_list()
                chem_comp_list = [str(chem_comp) for chem_comp in chem_comp_list]
                bond_list = entity.get_bonds(level="residue")
                bond_list = [
                    f"({idx1}, {idx2}, {conn_type})"
                    for idx1, idx2, conn_type in bond_list
                ]
                sequence = f"({')('.join(chem_comp_list)})|{','.join(bond_list)}"
                molecule_type = "BRANCHED"
            case MoleculeType.WATER:
                pass
        sequence = f"{molecule_type_map[molecule_type]}{sequence}"
        sequence_hash = seq_hashes.get(sequence, None)
        if sequence_hash is None:
            raise ValueError(
                f"Sequence {sequence} not found in sequence hashes. "
                f"Please check the sequence_hashes_all_molecules.pkl file."
            )
        chain_to_hash[chain] = sequence_hash
    return chain_to_hash


# Precompute neighbor offsets once (module-level)
NEIGHBOR_OFFSETS = torch.tensor(
    [[dx, dy, dz] for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)],
    dtype=torch.int64,
)


@torch.no_grad()
def get_contact_graph(biomol, chunk_size=int(3e5)):
    # Unpack structure
    struct = biomol.structure
    atom_tensor = struct.atom_tensor
    atom_chain_break = struct.atom_chain_break
    atom_mask = atom_tensor[:, 4].bool()
    atom_xyz = atom_tensor[:, 5:8]
    N = atom_xyz.size(0)
    device = atom_xyz.device

    # Build cell index mapping for spatial hashing
    cell_idx = torch.floor(atom_xyz / CONTACT_TH).to(torch.int64)
    cell_map = {}
    for i in range(N):
        if not atom_mask[i]:
            continue
        key = tuple(cell_idx[i].tolist())
        cell_map.setdefault(key, []).append(i)

    # Precompute atom->chain index
    chain_list = list(atom_chain_break.keys())
    chain_idx = torch.zeros(N, dtype=torch.long, device=device)
    for cid, chain_ID in enumerate(chain_list):
        start, end = atom_chain_break[chain_ID]
        chain_idx[start : end + 1] = cid

    # Collect contacts in chunks
    all_i = []
    all_j = []
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        L = end - start

        # Gather neighbor lists
        neighbor_lists = []
        maxM = 0
        for i in range(start, end):
            if not atom_mask[i]:
                neighbor_lists.append([])
                continue
            base = cell_idx[i]
            nbrs = []
            # Use cached offsets
            for off in NEIGHBOR_OFFSETS:
                cell = tuple((base + off).tolist())
                if cell in cell_map:
                    nbrs.extend(cell_map[cell])
            # only j < i to avoid duplicates
            nbrs = [j for j in nbrs if j < i]
            neighbor_lists.append(nbrs)
            if len(nbrs) > maxM:
                maxM = len(nbrs)

        # Allocate padded neighbor index tensor and mask
        idx_tensor = torch.full((L, maxM), -1, dtype=torch.long, device=device)
        mask_tensor = torch.zeros((L, maxM), dtype=torch.bool, device=device)
        for r, nbrs in enumerate(neighbor_lists):
            m = len(nbrs)
            if m:
                idx_tensor[r, :m] = torch.tensor(nbrs, device=device, dtype=torch.long)
                mask_tensor[r, :m] = True

        # Distance check
        chunk_xyz = atom_xyz[start:end].unsqueeze(1)  # (L,1,3)
        nbr_xyz = atom_xyz[idx_tensor.clamp(min=0)]  # (L, maxM,3)
        dist2 = (chunk_xyz - nbr_xyz).pow(2).sum(-1)  # (L, maxM)
        contact = mask_tensor & (dist2 <= CONTACT_TH**2)

        # Extract i,j indices
        rows = torch.arange(L, device=device).unsqueeze(1).expand_as(contact)
        i_idx = rows[contact] + start
        j_idx = idx_tensor[contact]
        all_i.append(i_idx)
        all_j.append(j_idx)

    # Concatenate and unique
    i_all = torch.cat(all_i)
    j_all = torch.cat(all_j)
    pairs = torch.stack([i_all, j_all], dim=1)
    uniq = torch.unique(pairs, dim=0)
    # remove self-contacts
    uniq = uniq[uniq[:, 0] != uniq[:, 1]]

    # Chain-wise edges
    c1 = chain_idx[uniq[:, 0]]
    c2 = chain_idx[uniq[:, 1]]
    edges = torch.stack([torch.min(c1, c2), torch.max(c1, c2)], dim=1)
    # remove self-edges
    edges = torch.unique(edges, dim=0)
    edges = edges[edges[:, 0] != edges[:, 1]].tolist()  # (num_edges, 2)

    # Nodes as sequence hashes
    chain_hash = get_sequence_hash(struct)
    nodes = [chain_hash[c.split("_")[0]] for c in chain_list]

    return nodes, edges


def get_cif_ID_list():
    """
    Get a list of CIF IDs from the metadata file.

    Returns:
    list: A list of CIF IDs.
    """
    cif_ids = []
    for dirpath, _, filenames in os.walk(CIF_dir):
        for fn in filenames:
            if fn.endswith(".cif"):
                cif_id = fn[:-4]
            elif fn.endswith(".cif.gz"):
                cif_id = fn[:-7]
            else:
                continue
            cif_ids.append(cif_id)
    cif_ids = sorted(cif_ids)
    return cif_ids


def get_already_saved_cif_ID_list(save_dir):
    """
    Get a list of already saved CIF IDs from the hash to graph mapping file.

    Returns:
    list: A list of already saved CIF IDs.
    """
    cif_IDs = []
    for dirpath, _, filenames in os.walk(save_dir):
        for fn in filenames:
            if fn.endswith(".graph"):
                cif_id = fn[:-6]
            elif fn.endswith(".graph.gz"):
                cif_id = fn[:-9]
            else:
                continue
            cif_IDs.append(cif_id)
    cif_IDs = sorted(cif_IDs)
    return cif_IDs


def save_graphs(cif_ID, save_path):
    # torch set thread
    torch.set_num_threads(1)
    graphs = {}
    biomol, ID_list = load_biomol(cif_ID)
    for ids in ID_list:
        assembly_ID, model_ID, alt_ID = ids
        biomol.choose(assembly_ID, model_ID, alt_ID)
        node, edge = get_contact_graph(biomol)
        graphs[(assembly_ID, model_ID, alt_ID)] = (node, edge)

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


def save_protein_graph(save_dir, n_jobs=100):
    """
    Get the ligand binding IDs from the BioMol database.

    Parameters:
    save_dir (str): The path where the BioMol database is saved.

    Returns:
    list: A list of ligand binding IDs.
    """

    cif_ID_list = get_cif_ID_list()
    inner_dir_list = [os.path.join(save_dir, cif_ID[1:3]) for cif_ID in cif_ID_list]
    for inner_dir in inner_dir_list:
        if not os.path.exists(inner_dir):
            os.makedirs(inner_dir)

    already_saved_cif_ID_list = get_already_saved_cif_ID_list(save_dir)
    cif_ID_list = [
        cif_ID for cif_ID in cif_ID_list if cif_ID not in already_saved_cif_ID_list
    ]

    save_path_list = [
        os.path.join(save_dir, cif_ID[1:3], f"{cif_ID}.graph") for cif_ID in cif_ID_list
    ]

    total = len(cif_ID_list)
    print(f"Total hashes: {total}")

    # 2) SLURM_ARRAY_TASK_ID 로 shard 결정
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    # 각 노드가 처리할 청크 범위
    chunk_size = math.ceil(total / num_tasks)
    start = task_id * chunk_size
    end = min(start + chunk_size, total)
    cif_ID_subset = cif_ID_list[start:end]
    save_path_subset = save_path_list[start:end]
    print(f"[Task {task_id}/{num_tasks}] Processing hashes {start}–{end - 1}")

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(save_graphs)(cif_ID, save_path)
        for cif_ID, save_path in zip(cif_ID_subset, save_path_subset)
    )


pattern = re.compile(r"\('([^']+)', '([^']+)', '([^']+)'\)")


def format_tuple(s):
    match = pattern.search(s)
    if match:
        # Combine the captured groups with underscores
        return f"{match.group(1)}_{match.group(2)}_{match.group(3)}"
    return None


def get_edge(file_path: str) -> list[nx.Graph]:
    graph_ID = file_path.split("/")[-1].split(".")[0]
    edges = set()
    G_dict = {}
    G = nx.Graph()  # Use nx.DiGraph() for directed graphs
    with open(file_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "t":  # New graph line: "t ID"
                if G.number_of_nodes() > 0:
                    G_dict[ID] = G
                    G = nx.Graph()
                parts = line.strip().split("#")
                ID = format_tuple(parts[1])
                ID = f"{graph_ID}_{ID}"
            elif parts[0] == "v":  # Vertex line: "v id label"
                node_id = int(parts[1])
                label = parts[2]
                G.add_node(node_id, label=label)
            elif parts[0] == "e":  # Edge line: "e src tgt"
                src, tgt = int(parts[1]), int(parts[2])
                G.add_edge(src, tgt)
        if G.number_of_nodes() > 0:
            G_dict[ID] = G

    # get edges (label1, label2)
    for G in G_dict.values():
        for u, v in G.edges():
            label1 = G.nodes[u]["label"]
            label2 = G.nodes[v]["label"]
            edges.add((label1, label2))
    # Convert set to list
    edges = list(edges)
    edges.sort(key=lambda x: (x[0], x[1]))  # Sort edges for consistency

    return edges


def get_graphs(file_path: str) -> list[nx.Graph]:
    graph_ID = file_path.split("/")[-1].split(".")[0]
    edges = set()
    G_dict = {}
    G = nx.Graph()  # Use nx.DiGraph() for directed graphs
    with open(file_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "t":  # New graph line: "t ID"
                if G.number_of_nodes() > 0:
                    G_dict[ID] = G
                    G = nx.Graph()
                parts = line.strip().split("#")
                ID = format_tuple(parts[1])
                ID = f"{graph_ID}_{ID}"
            elif parts[0] == "v":  # Vertex line: "v id label"
                node_id = int(parts[1])
                label = parts[2]
                G.add_node(node_id, label=label)
            elif parts[0] == "e":  # Edge line: "e src tgt"
                src, tgt = int(parts[1]), int(parts[2])
                G.add_edge(src, tgt)
        if G.number_of_nodes() > 0:
            G_dict[ID] = G

    return G_dict


def get_all_edge(graph_dir, save_path):
    graph_files = []
    for root, dirs, files in os.walk(graph_dir):
        for file in files:
            if file.endswith(".graph"):
                graph_files.append(os.path.join(root, file))
    print(f"Total graph files: {len(graph_files)}")
    edges = set()
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(get_edge)(file_path) for file_path in graph_files
    )
    for result in results:
        edges.update(result)
    edges = list(edges)
    edges.sort(key=lambda x: (x[0], x[1]))  # Sort edges for consistency
    print(f"Total unique edges: {len(edges)}")
    with open(save_path, "wb") as f:
        pickle.dump(edges, f)


def get_all_graphs(graph_dir, save_path):
    graph_files = []
    for root, dirs, files in os.walk(graph_dir):
        for file in files:
            if file.endswith(".graph"):
                graph_files.append(os.path.join(root, file))
    print(f"Total graph files: {len(graph_files)}")
    graphs = {}
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(get_graphs)(file_path) for file_path in graph_files
    )
    for result in results:
        graphs.update(result)
    print(f"Total unique graphs: {len(graphs)}")
    with open(save_path, "wb") as f:
        pickle.dump(graphs, f)


if __name__ == "__main__":
    # Example usage
    save_dir = os.path.join(DB_PATH, "contact_graphs")
    # save_protein_graph(save_dir, n_jobs=-1)
    # get_all_edge(save_dir, os.path.join(DB_PATH, 'statistics', "all_edges.pkl"))
    get_all_graphs(save_dir, os.path.join(DB_PATH, "statistics", "all_graphs.pkl"))
