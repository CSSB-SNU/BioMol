import os
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from joblib import Parallel, delayed
import random
import json


def check_condition(condition, cif_path):
    """
    Search features in the cif file.
    """
    key_list = list(condition.keys())
    mmcif_dict = MMCIF2Dict(cif_path)
    result_dict = {key: False for key in key_list}
    for key, value in condition.items():
        if key in mmcif_dict:
            if value is None:
                result_dict[key] = True
                continue
            (tf, value) = value
            if tf:
                if value in mmcif_dict[key]:
                    result_dict[key] = True
            else:
                # mmcif_dict[key] -> unique() ->
                temp = list(set(mmcif_dict[key]))
                if value not in temp:
                    result_dict[key] = True
    return result_dict


def process_file(condition, cif_path):
    """
    Check a single CIF file against the condition.
    """
    result_dict = check_condition(condition, cif_path)
    if any(result_dict.values()):
        return cif_path
    return None


def searc_cif(condition, cif_dir, n_jobs=-1):
    """
    Search cif files in the directory using parallel processing.
    """
    cif_files = []
    for root, dirs, files in os.walk(cif_dir):
        for file in files:
            if file.endswith(".cif"):
                cif_files.append(os.path.join(root, file))

    # Use joblib to process files in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_file)(condition, cif_path) for cif_path in cif_files
    )

    # Return the first non-None result
    for result in results:
        if result is not None:
            return result
    return None


def searc_unclassified_fields(field_json, cif_dir, n_sample=10):
    """
    Search unclassified fields in the directory using parallel processing.
    """
    cif_files = []
    for root, dirs, files in os.walk(cif_dir):
        for file in files:
            if file.endswith(".cif"):
                cif_files.append(os.path.join(root, file))
    cif_files = random.sample(cif_files, n_sample)

    fields_classified = {}
    # parse the json file
    with open(field_json, "r") as f:
        fields_classified = json.load(f)

    used_fields = fields_classified["used"]
    not_used_fields = (
        fields_classified["not_used"]
        + fields_classified["crystal_related"]
        + fields_classified["em_related"]
        + fields_classified["nmr_related"]
    )

    classified_fields = set(used_fields + not_used_fields)

    assert len(classified_fields) == len(used_fields) + len(not_used_fields)

    # ~_src_~ files include all fileds like _pdbx_entity_src_syn or _pdbx_entity_src

    special_fields = [field[2:-2] for field in classified_fields if "~" in field]

    # searc unclassified fields
    sampled_fields = {}
    for cif_file in cif_files:
        try:
            cif_dict = MMCIF2Dict(cif_file)
        except Exception as e:
            print(f"Error: {e}")
            continue
        sampled_fields[cif_file] = list(cif_dict.keys())

    unclassified_fields = []
    to_test_cif_list = []
    to_test_cif_dict = {}

    for cif_file, fields in sampled_fields.items():
        for field in fields:
            field = field.split(".")[0]
            if field not in classified_fields:
                field_split = field.split("_")
                is_unclassified = True
                for special_field in special_fields:
                    if special_field in field_split:
                        is_unclassified = False
                        break
                if is_unclassified:
                    unclassified_fields.append(field)
                    to_test_cif_list.append(
                        cif_file
                    ) if cif_file not in to_test_cif_list else None
                    to_test_cif_dict[field] = cif_file

    unclassified_fields = list(set(unclassified_fields))
    print(f"unclassified_fields: {unclassified_fields}")
    if len(unclassified_fields) < 5:
        print(f"to_test_cif_dict: {to_test_cif_dict}")

    if len(unclassified_fields) == 0:
        print("No unclassified fields found.")
        return

    # copy cif files in to_test_cif_list to ./utils/to_test_cif/

    if not os.path.exists("./utils/to_test_cif/"):
        os.makedirs("./utils/to_test_cif/")

    for cif_file in to_test_cif_list:
        os.system(f"cp {cif_file} ./utils/to_test_cif/")

    # TODO branch related fields see 2qwh ['_pdbx_entity_branch_link', '_pdbx_entity_branch_descriptor', '_pdbx_entity_branch', '_pdbx_branch_scheme', '_struct_site_keywords', '_pdbx_entity_branch_list', '_pdbx_deposit_group']

    breakpoint()


if __name__ == "__main__":
    # condition = {'_entity_poly.nstd_monomer': (False, 'no'),} # -> 400d

    # TODO
    # condition = {'_pdbx_entity_src_category.value': None,}
    # condition = {'_geom_hbond.atom_site_id_D': None,}
    # condition = {'_entity_poly_seq.hetero': (True, 'yes')}
    condition = {
        "_pdbx_poly_seq_scheme.pdb_ins_code": (False, ".")
    }  # /public_data/pdb_mmcif/mmcif_files/6ins.cif
    cif_dir = "/public_data/pdb_mmcif/mmcif_files/"

    cif_path = searc_cif(condition, cif_dir)
    print(cif_path)
