import os
import math
import numpy as np
from joblib import Parallel, delayed
from BioMol.BioMol import BioMol
from BioMol import DB_PATH


def save_each_file(cif_path: str, to_save_path: str) -> None:
    """Convert a single CIF file to compressed NumPy archive."""
    to_save = {}
    biomol = BioMol(
        cif=cif_path,
        remove_signal_peptide=False,
        mol_types=["protein", "nucleic_acid", "ligand"],
        use_lmdb=False,
    )
    for assembly_id in biomol.bioassembly.keys():
        for model_id in biomol.bioassembly[assembly_id].keys():
            for alt_id in biomol.bioassembly[assembly_id][model_id].keys():
                key = f"{assembly_id}_{model_id}_{alt_id}"
                structure = biomol.bioassembly[assembly_id][model_id][alt_id]
                scheme = structure.scheme
                atom_tensor = np.asarray(structure.atom_tensor)
                atom_bond = np.asarray(structure.atom_bond)
                atom_chain_break = structure.atom_chain_break
                scheme = {
                    "cif_idx_list": np.asarray(scheme.cif_idx_list),
                    "auth_idx_list": np.asarray(scheme.auth_idx_list),
                    "chem_comp_list": np.asarray(scheme.chem_comp_list),
                    "hetero_list": np.asarray(scheme.hetero_list),
                }
                to_save[key] = {
                    "atom_tensor": atom_tensor,
                    "atom_bond": atom_bond,
                    "atom_chain_break": atom_chain_break,
                    "scheme": scheme,
                }

    os.makedirs(os.path.dirname(to_save_path), exist_ok=True)
    np.savez_compressed(to_save_path, **to_save)


def chunk_files(files, num_chunks, chunk_idx):
    """Split files into chunks for Slurm job array tasks."""
    n = len(files)
    chunk_size = math.ceil(n / num_chunks)
    start = chunk_idx * chunk_size
    end = min(start + chunk_size, n)
    return files[start:end]


def find_all_cif_files(cif_dir: str):
    """Recursively find all .cif.gz files inside subdirectories."""
    all_files = []
    for root, _, files in os.walk(cif_dir):
        for f in files:
            if f.endswith(".cif.gz"):
                all_files.append(os.path.join(root, f))
    return sorted(all_files)


def subdir_from_cif_id(filename: str) -> str:
    """
    Extract middle two characters from CIF ID for subdirectory.
    e.g. "abcd.cif.gz" -> "bc"
         "1xyz.cif.gz" -> "xy"
         "7abcde.cif.gz" -> "cd" (central 2 letters)
    """
    base = os.path.basename(filename).replace(".cif.gz", "")
    n = len(base)
    if n < 2:
        return base
    mid = n // 2
    return base[mid - 1: mid + 1]


def main():
    cif_dir = os.path.join(DB_PATH, "cif/cif_raw/")
    save_dir = os.path.join(DB_PATH, "restored_np")
    os.makedirs(save_dir, exist_ok=True)

    all_files = find_all_cif_files(cif_dir)
    num_jobs = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))
    job_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))

    files = chunk_files(all_files, num_jobs, job_id)
    print(f"[Task {job_id}] Processing {len(files)} files out of {len(all_files)} total")

    save_paths = []
    for f in files:
        subdir = subdir_from_cif_id(f)
        out_dir = os.path.join(save_dir, subdir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, os.path.basename(f).replace(".cif.gz", ".npz"))
        save_paths.append(out_path)

    Parallel(n_jobs=-1, verbose=10)(
        delayed(save_each_file)(cif_path, save_path)
        for cif_path, save_path in zip(files, save_paths)
    )


if __name__ == "__main__":
    main()
