import os
from pathlib import Path
from typing import TypeVar

import lmdb
import numpy as np
from joblib import Parallel, delayed

from biomol.cif import CIFAtomView, CIFChainView, CIFMol, CIFResidueView
from biomol.core.utils import from_bytes

ViewType = TypeVar("ViewType", CIFAtomView, CIFResidueView, CIFChainView)


atom_mapping = {
    "H": 0,
    "HE": 1,
    "LI": 2,
    "BE": 3,
    "B": 4,
    "C": 5,
    "N": 6,
    "O": 7,
    "F": 8,
    "NE": 9,
    "NA": 10,
    "MG": 11,
    "AL": 12,
    "SI": 13,
    "P": 14,
    "S": 15,
    "CL": 16,
    "AR": 17,
    "K": 18,
    "CA": 19,
    "SC": 20,
    "TI": 21,
    "V": 22,
    "CR": 23,
    "MN": 24,
    "FE": 25,
    "CO": 26,
    "NI": 27,
    "CU": 28,
    "ZN": 29,
    "GA": 30,
    "GE": 31,
    "AS": 32,
    "SE": 33,
    "BR": 34,
    "KR": 35,
    "RB": 36,
    "SR": 37,
    "Y": 38,
    "ZR": 39,
    "NB": 40,
    "MO": 41,
    "TC": 42,
    "RU": 43,
    "RH": 44,
    "PD": 45,
    "AG": 46,
    "CD": 47,
    "IN": 48,
    "SN": 49,
    "SB": 50,
    "TE": 51,
    "I": 52,
    "XE": 53,
    "CS": 54,
    "BA": 55,
    "LA": 56,
    "CE": 57,
    "PR": 58,
    "ND": 59,
    "PM": 60,
    "SM": 61,
    "EU": 62,
    "GD": 63,
    "TB": 64,
    "DY": 65,
    "HO": 66,
    "ER": 67,
    "TM": 68,
    "YB": 69,
    "LU": 70,
    "HF": 71,
    "TA": 72,
    "W": 73,
    "RE": 74,
    "OS": 75,
    "IR": 76,
    "PT": 77,
    "AU": 78,
    "HG": 79,
    "TL": 80,
    "PB": 81,
    "BI": 82,
    "PO": 83,
    "AT": 84,
    "RN": 85,
    "FR": 86,
    "RA": 87,
    "AC": 88,
    "TH": 89,
    "PA": 90,
    "U": 91,
    "NP": 92,
    "PU": 93,
    "AM": 94,
    "CM": 95,
    "BK": 96,
    "CF": 97,
    "ES": 98,
    "FM": 99,
    "MD": 100,
    "NO": 101,
    "LR": 102,
    "RF": 103,
    "DB": 104,
    "SG": 105,
    "BH": 106,
    "HS": 107,
    "MT": 108,
    "DS": 109,
    "RG": 110,
    "CN": 111,
    "NH": 112,
    "FL": 113,
    "MC": 114,
    "LV": 115,
    "TS": 116,
    "OG": 117,
    "X": 118,  # for unknown atoms
    "D": 119,  # for deuterium
}


def extract_key_list(env_path: Path) -> list[str]:
    """Extract all keys from the LMDB database."""
    env = lmdb.open(str(env_path), readonly=True, lock=False)
    with env.begin() as txn:
        key_list = [
            key.decode() for key in txn.cursor().iternext(keys=True, values=False)
        ]
    env.close()
    return key_list


def read_lmdb(env_path: Path, key: str) -> dict[str, CIFMol]:
    """
    Read a value from the LMDB database by key.

    Args:
        env_path: Path to the LMDB environment.
        key: Key of the data to retrieve.

    Returns
    -------
        dict
            The data dictionary retrieved from the LMDB database.
    """
    env = lmdb.open(str(env_path), readonly=True, lock=False)
    with env.begin() as txn:
        value = txn.get(key.encode())
    env.close()
    if value is None:
        msg = f"Key '{key}' not found in LMDB database."
        raise KeyError(msg)
    value = from_bytes(value)  # remove metadata dict
    value, metadata = value["assembly_dict"], value["metadata_dict"]

    cifmol_dict = {}
    for cif_key in value:
        item = value[cif_key]
        # model_id = cif_key.
        assembly_id, model_id, alt_id = cif_key.split("_")
        metadata["assembly_id"] = assembly_id
        metadata["model_id"] = model_id
        metadata["alt_id"] = alt_id
        item["metadata"] = metadata

        cifmol_dict[cif_key] = CIFMol.from_dict(item)
    return cifmol_dict


def load_old_data(
    key: str,
    old_dir: Path,
) -> dict:
    """Load old data from a specified directory."""
    # temporal
    key = key.split(".cif")[0]
    file_path = Path(old_dir) / f"{key[1:3]}" / f"{key}.npz"
    if not file_path.is_file():
        msg = f"Old data file '{file_path}' does not exist."
        raise FileNotFoundError(msg)
    return np.load(file_path, allow_pickle=True)


def compare_scheme(new_value: CIFMol, old_value: dict) -> None:
    """Compare scheme data between new and old values."""

    # auth_idx test
    new_auth_idx = new_value.residues.auth_idx
    old_auth_idx = old_value["scheme"]["auth_idx_list"].astype(new_auth_idx.dtype)
    if not np.array_equal(new_auth_idx, old_auth_idx):
        breakpoint()
        msg = "auth_idx mismatch"
        raise ValueError(msg)

    # chem_comp test
    new_chem_comp = new_value.residues.chem_comp
    old_chem_comp = old_value["scheme"]["chem_comp_list"].astype(new_chem_comp.dtype)
    if not np.array_equal(new_chem_comp, old_chem_comp):
        msg = "chem_comp mismatch"
        raise ValueError(msg)

    # # hetero test
    # new_hetero = new_value["residue"]["nodes"]["hetero"]["value"]
    # old_hetero = old_value["scheme"]["hetero_list"].astype(new_hetero.dtype)
    # if not np.array_equal(new_hetero, old_hetero):
    #     msg = "hetero mismatch"
    #     raise ValueError(msg)


def compare_atom_tensor(new_value: CIFMol, old_value: dict) -> None:
    """Compare atom tensor data between new and old values."""
    old_atom_tensor = old_value["atom_tensor"]
    new_element, new_xyz, new_occup, new_bfactor = (
        new_value.atoms.element,
        new_value.atoms.xyz,
        new_value.atoms.occupancy,
        new_value.atoms.b_factor,
    )

    new_mask = np.isnan(new_xyz.value)  # (N, 3)
    new_mask = np.all(new_mask == 0, axis=-1).astype(np.int8)  # (N,)
    old_element, old_xyz, old_occup, old_bfactor = (
        old_atom_tensor[:, 3],
        old_atom_tensor[:, 5:8],
        old_atom_tensor[:, 8],
        old_atom_tensor[:, 9],
    )
    old_mask = old_atom_tensor[:, 4].astype(np.int8)
    old_xyz[old_mask == 0] = np.nan
    old_occup[old_mask == 0] = np.nan
    old_bfactor[old_mask == 0] = np.nan

    new_element = np.array([atom_mapping.get(el, 118) for el in new_element.value])
    old_element = old_element.astype(new_element.dtype)
    new_element[new_mask == 0] = 0
    old_element[old_mask == 0] = 0

    # compare element
    if not np.array_equal(new_element, old_element):
        msg = "element mismatch"
        raise ValueError(msg)

    # compare mask
    if not np.array_equal(new_mask, old_mask):
        msg = "mask mismatch"
        raise ValueError(msg)

    # compare xyz
    if not np.allclose(new_xyz, old_xyz, equal_nan=True, atol=1e-3):
        msg = "xyz mismatch"
        raise ValueError(msg)

    # compare occupancy
    if not np.allclose(new_occup, old_occup, equal_nan=True):
        msg = "occupancy mismatch"
        raise ValueError(msg)

    # compare bfactor
    if not np.allclose(new_bfactor, old_bfactor, equal_nan=True):
        msg = "bfactor mismatch"
        raise ValueError(msg)


def compare_datum(key: str, new_db: Path, old_dir: str) -> None:
    """Compare new data from LMDB with old data from files."""
    try:
        new_data: dict[str, CIFMol] = read_lmdb(new_db, key)
        old_data = load_old_data(key, old_dir=old_dir)
    except Exception as error:
        return {key: str(error)}

    # key checking
    new_keys = set(new_data.keys())
    old_keys = set(old_data.keys())
    # if new_keys != old_keys:
    #     msg = f"Key mismatch for {key}: {new_keys} vs {old_keys}"
    #     return {key: msg}

    compare_results = {}
    for dict_key in new_keys:
        try:
            new_value = new_data[dict_key]
            old_value = old_data[dict_key].item()

            # remove water from new_value
            new_value = new_value.residues[new_value.residues.chem_comp != "HOH"]

            compare_scheme(new_value, old_value)
            compare_atom_tensor(new_value, old_value)
        except Exception as error:
            compare_results[f"{key}_{dict_key}"] = str(error)
            continue
        compare_results[f"{key}_{dict_key}"] = "OK"
    return compare_results


def compare_data(new_db: Path, old_dir: str, n_jobs: int = 1) -> None:
    """Compare all data in the LMDB database with old data from files."""
    key_list = extract_key_list(new_db)

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(compare_datum)(key, new_db, old_dir) for key in key_list
    )

    for key, result in zip(key_list, results):
        for dict_key, status in result.items():
            if status != "OK":
                print(f"Discrepancy in {key} ({dict_key}): {status}")
            else:
                # print(f"{key} ({dict_key}): OK")
                pass


if __name__ == "__main__":
    new_db = Path("/public_data/BioMolDBv2_2024Oct21/cif.lmdb")
    old_dir = "/public_data/BioMolDB_2024Oct21/restored_np/"
    # compare_data(new_db, old_dir, n_jobs=-1)
    test = compare_datum("8t2f", new_db, old_dir)
    print(test)
