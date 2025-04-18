import torch
from utils.feature import *
from utils.hierarchy import *
from utils.parser_utils import *
import os
from joblib import Parallel, delayed
from filelock import FileLock
from constant.chemical import stereo_config_map
import gc
from utils.parser import parse_cif


def process_cif(cif_path, pass_cif_path, lock_path):
    """Process a single CIF file and write the result to CSV immediately."""
    print(f"Loading {cif_path}")
    try:
        parse_cif(cif_path)
        error = "-"
    except StructConnAmbiguityError:
        error = "StructConnAmbiguityError"
    except NonpolymerError:
        error = "NonpolymerError"
    except EntityMismatchError:
        error = "EntityMismatchError"
    except Exception as e:
        print(f"Error processing {cif_path}: {e}")
        error = "code_error"
    
    # Write the result immediately, using a file lock to avoid conflicts.
    with FileLock(lock_path):
        with open(pass_cif_path, 'a') as f:
            f.write(f"{cif_path},{error}\n")

    gc.collect()
    
    return cif_path, error

class BondMismatchError(Exception):
    pass

def _unittest_bond(cif_path, atom_bond_threshold=5.0, residue_bond_threshold=10.0):
    bio_assembly = parse_cif(cif_path)
    for assembly_id in bio_assembly.keys():
        for model_id in bio_assembly[assembly_id].keys():
            for alt_id in bio_assembly[assembly_id][model_id].keys():
                biomol_structure = bio_assembly[assembly_id][model_id][alt_id]
                atom_tensor = biomol_structure.atom_tensor
                residue_tensor = biomol_structure.residue_tensor

                atom_mask = atom_tensor[:,4]
                valid_atom_idx = torch.nonzero(atom_mask).squeeze()
                # print(biomol_structure)
                atom_bond = biomol_structure.atom_bond[:,:2]
                bond_mask = (biomol_structure.atom_bond[:,2] == 2) | (biomol_structure.atom_bond[:,2] == 4)
                atom_bond = atom_bond[bond_mask]
                atom_bond_mask1 = torch.isin(atom_bond[:,0], valid_atom_idx)
                atom_bond_mask2 = torch.isin(atom_bond[:,1], valid_atom_idx)
                atom_bond_mask = atom_bond_mask1 * atom_bond_mask2
                atom_bond = atom_bond[atom_bond_mask]
                atom_idx1, atom_idx2 = atom_bond[:,0], atom_bond[:,1]
                atom_xyz = atom_tensor[:,5:8]
                atom_xyz1, atom_xyz2 = atom_xyz[atom_idx1], atom_xyz[atom_idx2]
                distance = torch.norm(atom_xyz1 - atom_xyz2, dim=1)
                invalid_bond_mask = distance > atom_bond_threshold
                if torch.sum(invalid_bond_mask) > 0:
                    max_dist_idx = torch.argmax(distance)
                    max_dixt_xyz1, max_dist_xyz2 = atom_xyz1[max_dist_idx], atom_xyz2[max_dist_idx]
                    print(F"cif_path: {cif_path}")
                    print(f"Max distance: {distance[max_dist_idx]}")
                    print(f"Atom 1 xyz: {max_dixt_xyz1}")
                    print(f"Atom 2 xyz: {max_dist_xyz2}")
                    raise BondMismatchError

                # residue_mask = residue_tensor[:,4]
                # valid_residue_idx = torch.nonzero(residue_mask).squeeze()
                # residue_bond = biomol_structure.residue_bond[:,:2]
                # bond_mask = (biomol_structure.residue_bond[:,2] == 2) | (biomol_structure.residue_bond[:,2] == 4)
                # residue_bond = residue_bond[bond_mask]
                # residue_bond_mask1 = torch.isin(residue_bond[:,0], valid_residue_idx)
                # residue_bond_mask2 = torch.isin(residue_bond[:,1], valid_residue_idx)
                # residue_bond_mask = residue_bond_mask1 * residue_bond_mask2
                # residue_bond = residue_bond[residue_bond_mask]
                # residue_idx1, residue_idx2 = residue_bond[:,0], residue_bond[:,1]
                # residue_xyz = residue_tensor[:,5:8]
                # residue_xyz1, residue_xyz2 = residue_xyz[residue_idx1], residue_xyz[residue_idx2]
                # distance = torch.norm(residue_xyz1 - residue_xyz2, dim=1)
                # invalid_bond_mask = distance > residue_bond_threshold
                # if torch.sum(invalid_bond_mask) > 0:
                #     max_dist_idx = torch.argmax(distance)
                #     print(f"Max distance: {distance[max_dist_idx]}")
                #     print(f"Residue 1 xyz: {residue_xyz1[max_dist_idx]}")
                #     print(f"Residue 2 xyz: {residue_xyz2[max_dist_idx]}")
                #     breakpoint()
                #     raise BondMismatchError

def unittest_bond(cif_path, pass_cif_path, lock_path):
    """Process a single CIF file and write the result to CSV immediately."""
    print(f"Loading {cif_path}")
    try:
        _unittest_bond(cif_path)
        error = "-"
    except StructConnAmbiguityError:
        error = "StructConnAmbiguityError"
    except NonpolymerError:
        error = "NonpolymerError"
    except EntityMismatchError:
        error = "EntityMismatchError"
    except Exception as e:
        print(f"Error processing {cif_path}: {e}")
        error = "code_error"
    
    # Write the result immediately, using a file lock to avoid conflicts.
    with FileLock(lock_path):
        with open(pass_cif_path, 'a') as f:
            f.write(f"{cif_path},{error}\n")

    gc.collect()
    
    return cif_path, error


def test_load_cif_v2(cif_dir, passed_path="pass_cif.csv", to_path = "new_pass_cif.csv"):
    # Create the CSV file if it doesn't exist.
    if not os.path.exists(to_path):
        with open(to_path, 'w') as f:
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

    # # Gather all CIF file paths from the inner directories.
    # cif_path_list = []
    # for inner_dir in os.listdir(cif_dir):
    #     inner_dir_path = os.path.join(cif_dir, inner_dir)
    #     if not os.path.isdir(inner_dir_path):
    #         continue
    #     for file_name in os.listdir(inner_dir_path):
    #         if file_name.endswith('.cif') or file_name.endswith('.cif.gz'):
    #             full_path = os.path.join(inner_dir_path, file_name)
    #             if full_path not in passed_cif_path_list:
    #                 cif_path_list.append(full_path)

                        
    # # Gather all CIF file paths from the inner directories.
    # cif_path_list = []
    # for file_name in os.listdir(cif_dir):
    #     if file_name.endswith('.cif') or file_name.endswith('.cif.gz'):
    #         full_path = os.path.join(cif_dir, file_name)
    #         cif_path_list.append(full_path)

    # Process each CIF file in parallel.
    # Each worker writes its result to the CSV immediately.
    Parallel(n_jobs=-1)(
        delayed(process_cif)(cif_path, to_path, lock_path)
        for cif_path in not_passed_cif_path_list
    )

def test_load_cif(cif_dir, passed_path="passed_cif.csv"):
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

    # Gather all CIF file paths from the inner directories.
    cif_path_list = []
    for inner_dir in os.listdir(cif_dir):
        inner_dir_path = os.path.join(cif_dir, inner_dir)
        if not os.path.isdir(inner_dir_path):
            continue
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
        delayed(unittest_bond)(cif_path, passed_path, lock_path)
        for cif_path in cif_path_list
    )

    
def _save_seq(cif_path, pass_cif_path, lock_path):
    """Process a single CIF file and write the result to CSV immediately."""
    print(f"Loading {cif_path}")
    bio_assembly = parse_cif(cif_path)
    for assembly_id in bio_assembly.keys():
        for model_id in bio_assembly[assembly_id].keys():
            for alt_id in bio_assembly[assembly_id][model_id].keys():
                biomol_structure = bio_assembly[assembly_id][model_id][alt_id]
                biomol_structure.save_entities()
    
    # Write the result immediately, using a file lock to avoid conflicts.
    with FileLock(lock_path):
        with open(pass_cif_path, 'a') as f:
            f.write(f"{cif_path},\n")

    gc.collect()
    
    return cif_path

def save_seq(cif_dir, passed_path="passed_cif.csv"):
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

    save_dir = '/public_data/psk6950/PDB_2024Mar18/entity/'

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
        delayed(_save_seq)(cif_path, passed_path, lock_path)
        for cif_path in cif_path_list
    )

if __name__ == "__main__":
    # cif_dir = "/public_data/psk6950/PDB_2024Mar18/cif/"
    # test_load_cif(cif_dir, passed_path = "bond_unittest_passed_cif.csv")

    # cif_path = "/public_data/psk6950/PDB_2024Mar18/cif/lp/7lpm.cif.gz"
    # _unittest_bond(cif_path)

    cif_dir = "/public_data/psk6950/PDB_2024Mar18/cif/"
    save_seq(cif_dir, passed_path = "save_seq_passed_cif.csv")