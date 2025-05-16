import torch
import torch.nn.functional as F
import pickle
from BioMol import SEQ_TO_HASH_PATH, DB_PATH
from BioMol.BioMol import BioMol
from joblib import Parallel, delayed
import os
import lmdb

merged_fasta_path = f"{DB_PATH}/entity/merged_protein.fasta"
save_path = f"{DB_PATH}/metadata/hash_to_pdbIDs.pkl"
hash_to_full_IDs_path = f"{DB_PATH}/metadata/hash_to_full_IDs.pkl"
db_env = f"{DB_PATH}/seq_to_str/residue.lmdb"


def load_hash_to_seq():
    return pickle.load(open(SEQ_TO_HASH_PATH, "rb"))


def load_hash_to_pdbIDs():
    return pickle.load(open(save_path, "rb"))


def _gen_hash_to_pdbIDs():
    """
    Parses a merged FASTA file and returns a dictionary mapping sequence IDs to sequences.
    """
    seq_to_hash = load_hash_to_seq()
    hash_to_pdbIDs = {}
    with open(merged_fasta_path, "r") as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            seq_id = lines[i][1:].split("_")[0]
            seq = lines[i + 1].strip()
            seq_hash = seq_to_hash.get(seq)
            if seq_hash not in hash_to_pdbIDs:
                hash_to_pdbIDs[seq_hash] = []
            hash_to_pdbIDs[seq_hash].append(seq_id)

    with open(save_path, "wb") as f:
        pickle.dump(hash_to_pdbIDs, f)

    return hash_to_pdbIDs


def find_pdbIDs_by_hashes(
    hash_list: list[str], hash_to_pdbIDs: dict[str, list[str]]
) -> list[str]:
    """
    Given a list of hashes, return a dictionary mapping each hash to its corresponding PDB IDs.
    """
    pdb_IDs = {}
    for seq_hash in hash_list:
        if seq_hash in hash_to_pdbIDs:
            pdb_IDs[seq_hash] = hash_to_pdbIDs[seq_hash]
        else:
            raise ValueError(f"Hash {seq_hash} not found in the database.")

    # find common pdbIDs
    common_pdbIDs = set(pdb_IDs[hash_list[0]])
    for seq_hash in hash_list[1:]:
        common_pdbIDs.intersection_update(pdb_IDs[seq_hash])

    if len(common_pdbIDs) == 0:
        raise ValueError("No common PDB IDs found for the given hashes.")
    return sorted(list(common_pdbIDs))


def extract_chain_ids(
    biomol: BioMol, seq_hash: str, level: str = "residue"
) -> list[torch.Tensor]:
    """
    Extract chain IDs from the biomol object.
    """
    ids = []
    structures = []
    for bioassembly_id in biomol.bioassembly.assembly_dict.keys():
        for model_id in biomol.bioassembly.assembly_dict[bioassembly_id].keys():
            for alt_id in biomol.bioassembly.assembly_dict[bioassembly_id][
                model_id
            ].keys():
                biomol.choose(bioassembly_id, model_id, alt_id)
                if seq_hash not in biomol.structure.sequence_hash.values():
                    continue
                id = f"{biomol.ID}_{bioassembly_id}_{model_id}_{alt_id}"
                ids.append(id)
                chains = [
                    chain
                    for chain, _seq_hash in biomol.structure.sequence_hash.items()
                    if _seq_hash == seq_hash
                ]
                if level == "residue":
                    residue_chain_brerak = biomol.structure.residue_chain_break
                    for chain in chains:
                        residue_start, residue_end = residue_chain_brerak[chain]
                        structures.append(
                            biomol.structure.residue_tensor[residue_start:residue_end, :]
                        )
                elif level == "atom":
                    atom_chain_break = biomol.structure.atom_chain_break
                    for chain in chains:
                        atom_start, atom_end = atom_chain_break[chain]
                        structures.append(
                            biomol.structure.atom_tensor[atom_start:atom_end, :]
                        )
    return ids, structures


def seq_hash_to_structure(
    seq_hash: str,
    hash_to_pdbIDs: dict[str, list[str]],
    remove_signal_peptide=True,
    mol_type: str = "protein",
    level: str = "residue",  # or "atom"
) -> list[str]:
    """
    Given a sequence hash, return a list of PDB IDs.
    """
    pdb_IDs = hash_to_pdbIDs.get(seq_hash, [])
    if len(pdb_IDs) == 0:
        raise ValueError(f"Hash {seq_hash} not found in the database.")

    ids = []
    structures = []
    for pdb_ID in pdb_IDs:
        biomol = BioMol(
            pdb_ID=pdb_ID,
            mol_types=[mol_type],
            remove_signal_peptide=remove_signal_peptide,
            use_lmdb=False,
        )
        _ids, _structures = extract_chain_ids(biomol, seq_hash, level)
        if len(_structures) == 0:
            raise ValueError(
                f"No structures found for PDB ID {pdb_ID} and hash {seq_hash}."
            )
        ids.extend(_ids)
        structures.extend(_structures)

    return ids, structures


# save each monomer structures
def save_structures(
    pdb_ID: str, save_dir: str, level: str = "residue"
) -> list[torch.Tensor]:
    """
    Extract chain IDs from the biomol object.
    """
    biomol = BioMol(
        pdb_ID=pdb_ID,
        mol_types=["protein"],
    )
    for bioassembly_id in biomol.bioassembly.assembly_dict.keys():
        for model_id in biomol.bioassembly.assembly_dict[bioassembly_id].keys():
            for alt_id in biomol.bioassembly.assembly_dict[bioassembly_id][
                model_id
            ].keys():
                biomol.choose(bioassembly_id, model_id, alt_id)

                chains = biomol.structure.sequence_hash.keys()
                # remove bioassembly_id from chain ids
                chains_wo_oper_id = [chain.split("_")[0] for chain in chains]
                chains_wo_oper_id = sorted(list(set(chains_wo_oper_id)))

                for chain_id, seq_hash in biomol.structure.sequence_hash.items():
                    if chain_id.split("_")[0] not in chains_wo_oper_id:
                        continue
                    seq_hash = str(seq_hash).zfill(6)
                    inner_dir = f"{save_dir}/{seq_hash[0:3]}/{seq_hash[3:6]}"
                    chain_id_wo_oper_id = chain_id.split("_")[0]
                    str_id = f"{biomol.ID}_{bioassembly_id}_{model_id}_{alt_id}_{chain_id_wo_oper_id}"
                    if level == "residue":
                        residue_chain_brerak = biomol.structure.residue_chain_break
                        residue_start, residue_end = residue_chain_brerak[chain_id]
                        to_save_tensor = biomol.structure.residue_tensor[
                            residue_start : (residue_end + 1), :
                        ]
                    elif level == "atom":
                        atom_chain_break = biomol.structure.atom_chain_break
                        atom_start, atom_end = atom_chain_break[chain_id]
                        to_save_tensor = biomol.structure.atom_tensor[
                            atom_start : (atom_end + 1), :
                        ]
                    save_path = f"{inner_dir}/{str_id}.pt"
                    torch.save(to_save_tensor, save_path)
                    print(f"Saved {save_path}")


# TODO expand it to full IDs. This version is only for testing
def make_seq_hash_to_structure_db(
    save_dir=f"{DB_PATH}/seq_to_str/",
    thread_num=1,
    level: str = "residue",  # or "atom"
    inner_dir_already=False,
) -> dict[str, tuple[list[str], list[torch.Tensor]]]:
    """
    Given a sequence hash, return a list of PDB IDs.
    """
    save_dir = f"{save_dir}/{level}/"
    metadata_path = f"{DB_PATH}/metadata/metadata_psk.csv"  # protein only

    lines = open(metadata_path, "r").readlines()
    lines = lines[1:]  # remove header
    pdb_IDs = [line.split(",")[0].split("_")[0] for line in lines]
    pdb_IDs = sorted(list(set(pdb_IDs)))

    seq_to_hash = load_hash_to_seq()
    hash_list = list(seq_to_hash.values())
    hash_list = [str(_hash).zfill(6) for _hash in hash_list]

    # pre generate inner_dirs
    if not inner_dir_already:
        for _hash in hash_list:
            inner_dir = f"{save_dir}/{_hash[0:3]}/{_hash[3:6]}"
            if not os.path.exists(inner_dir):
                os.makedirs(inner_dir)

    print(f"Number of PDB IDs: {len(pdb_IDs)}")

    results = Parallel(n_jobs=thread_num, verbose=10)(
        delayed(save_structures)(pdb_ID, save_dir, level) for pdb_ID in pdb_IDs
    )


def process_file(_hash: str):
    save_dir = f"{DB_PATH}/seq_to_str/residue/{_hash[0:3]}/{_hash[3:6]}"
    IDs = os.listdir(save_dir)
    tensors = [os.path.join(save_dir, ID) for ID in IDs]
    tensors = [torch.load(tensor) for tensor in tensors]
    # if tensors.shape different, pad them to the same size
    tensor_shapes = [tensor.shape for tensor in tensors]
    if len(set(tensor_shapes)) > 1:
        max_len = max([shape[0] for shape in tensor_shapes])
        tensors = [
            F.pad(tensor, (0, 0, 0, max_len - tensor.shape[0]), "constant", float("nan"))
            for tensor in tensors
        ]
        print(
            f"Warning: tensors have different shapes. Padding to {max_len} for hash {_hash}."
        )

    tensors = torch.stack(tensors, dim=0)  # (B, L, 10)
    return IDs, tensors


def lmdb_seq_to_str(env_path=db_env, n_jobs=-1):
    seq_to_hash = load_hash_to_seq()
    hash_list = list(seq_to_hash.values())
    hash_list = [str(_hash).zfill(6) for _hash in hash_list]

    # Filter out files that are already processed
    print(f"Files to process: {len(hash_list)}")

    # Parallel processing of files using joblib.
    env = lmdb.open(env_path, map_size=1 * 1024**4)  # 1TB
    # n_jobs=-1 uses all available cores. Adjust as needed.
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_file)(_hash) for _hash in hash_list
    )

    # hash to full_IDs
    hash_to_full_IDs = {}

    # Open LMDB environment for writing
    with env.begin(write=True) as txn:
        for _hash, (cif_IDs, tensors) in zip(hash_list, results):
            # str hash to int hash
            _hash = str(int(_hash)).zfill(6)  # remove leading zeros
            to_save = {
                "cif_IDs": cif_IDs,
                "tensors": tensors,
            }
            to_save = pickle.dumps(to_save, protocol=pickle.HIGHEST_PROTOCOL)
            test = pickle.loads(to_save)
            txn.put(_hash.encode(), to_save)
            hash_to_full_IDs[_hash] = cif_IDs
    env.close()

    # save hash to full_IDs
    with open(hash_to_full_IDs_path, "wb") as f:
        pickle.dump(hash_to_full_IDs, f)


def read_seq_lmdb(key: str):
    """
    Read a sequence from the LMDB database.
    """
    env = lmdb.open(db_env, readonly=True)
    with env.begin() as txn:
        data = txn.get(key.encode())
        if data is None:
            raise ValueError(f"Key {key} not found in the database.")
        data = pickle.loads(data)
    env.close()
    return data


if __name__ == "__main__":
    # Load the sequence hash from the file
    make_seq_hash_to_structure_db(thread_num=-1, inner_dir_already=False, level="atom")
    # lmdb_seq_to_str()
