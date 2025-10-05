import os
import numpy as np
import concurrent.futures
import torch.distributed as dist
from BioMol.BioMol import BioMol
from BioMol import DB_PATH


def save_each_file(cif_path: str, to_save_path: str) -> None:
    try:
        to_save = {}
        biomol = BioMol(
            cif=cif_path,
            remove_signal_peptide=False,
            mol_types=["protein", "nucleic_acid", "ligand"],
            use_lmdb=False,
        )
        biomol.choose("1", "1", ".")
        for assembly_id in biomol.bioassembly.keys():
            for model_id in biomol.bioassembly[assembly_id].keys():
                for alt_id in biomol.bioassembly[assembly_id][model_id].keys():
                    key = f"{assembly_id}_{model_id}_{alt_id}"
                    structure = biomol.bioassembly[assembly_id][model_id][alt_id]
                    scheme = structure.scheme
                    atom_tensor = np.array(structure.atom_tensor)
                    atom_bond = np.array(structure.atom_bond)
                    atom_chain_break = structure.atom_chain_break
                    scheme = {
                        "cif_idx_list": np.array(scheme.cif_idx_list),
                        "auth_idx_list": np.array(scheme.auth_idx_list),
                        "chem_comp_list": np.array(scheme.chem_comp_list),
                        "hetero_list": np.array(scheme.hetero_list),
                    }
                    to_save[key] = {
                        "atom_tensor": atom_tensor,
                        "atom_bond": atom_bond,
                        "atom_chain_break": atom_chain_break,
                        "scheme": scheme,
                    }

        np.savez_compressed(to_save_path, **to_save)
    except Exception as e:
        print(f"[ERROR] {cif_path}: {e}")


def split_workload(files, world_size, rank):
    """Distribute files across nodes (multi-node via torch.distributed)."""
    n = len(files)
    chunk_size = (n + world_size - 1) // world_size
    start = rank * chunk_size
    end = min(start + chunk_size, n)
    return files[start:end]


def save_to_np(cif_dir, save_dir, num_workers=8):
    files = [f for f in os.listdir(cif_dir) if f.endswith(".cif.gz")]
    os.makedirs(save_dir, exist_ok=True)

    # Multi-node split (if torch.distributed is initialized)
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        files = split_workload(files, world_size, rank)
        print(f"[Rank {rank}] Processing {len(files)} files.")
    else:
        rank, world_size = 0, 1
        print(f"[Single Node] Processing {len(files)} files.")

    # Multi-thread (or process) parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for file in files:
            cif_path = os.path.join(cif_dir, file)
            to_save_path = os.path.join(save_dir, file.replace(".cif.gz", ".npz"))
            futures.append(executor.submit(save_each_file, cif_path, to_save_path))
        for future in concurrent.futures.as_completed(futures):
            _ = future.result()


def main():
    cif_dir = os.path.join(DB_PATH, "cif")
    save_dir = os.path.join(DB_PATH, "restored_np")
    os.makedirs(save_dir, exist_ok=True)

    # Initialize multi-node distributed environment if available
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
    else:
        print("Running in single-node mode.")

    save_to_np(cif_dir, save_dir, num_workers=8)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
