import os
import numpy as np

from BioMol.BioMol import BioMol
from BioMol import DB_PATH

def save_each_file(cif_path: str, to_save_path: str) -> None:
    to_save = {}
    biomol = BioMol(
        cif=cif_path,
        remove_signal_peptide=False,
        mol_types=["protein", "nucleic_acid", "ligand"],  # only protein
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


def save_to_np(cif_dir, save_dir):
    for file in os.listdir(cif_dir):
        if file.endswith(".cif.gz"):
            cif_path = os.path.join(cif_dir, file)
            to_save_path = os.path.join(save_dir, file.replace(".cif.gz", ".npz"))
            save_each_file(cif_path, to_save_path)


def main():
    cif_dir = os.path.join(DB_PATH, "cif")
    save_dir = os.path.join(DB_PATH, "restored_np")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_to_np(cif_dir, save_dir)

if __name__ == "__main__":
    main()