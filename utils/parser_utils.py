import pickle
import lmdb
import os
from constant.datapath import CCD_DB_PATH

def read_lmdb_value(key: str):
    """
    Reads the value corresponding to `key` from an LMDB database at `db_path`.
    Returns None if the key is not found.
    """
    # LMDB keys must be bytes, so encode the string key.
    key_bytes = key.encode("utf-8")

    # Open the environment in read-only mode
    # lock=False allows concurrent access in many read-only cases
    env = lmdb.open(CCD_DB_PATH, readonly=True, lock=False)
    
    try:
        # Start a read transaction
        with env.begin(write=False) as txn:
            data = txn.get(key_bytes)
            if data is None:
                return None
            return pickle.loads(data)
    finally:
        env.close()

def get_smaller_mmcif_dict(mmcif_dict, mmcif_dict_keys, fields, first_key= None, second_key = None, len1_to_scalar = False):
    '''
    # 
    loop_
    _entity_poly_seq.entity_id 
    _entity_poly_seq.num 
    _entity_poly_seq.mon_id 
    _entity_poly_seq.hetero 
    1 1   VAL n 
    1 2   LEU n 
    1 3   SER n 
    1 4   PRO n 
    1 5   ALA n 
    -> {1: {'num': [1, 2, 3, 4, 5], 'mon_id': ['VAL', 'LEU', 'SER', 'PRO', 'ALA'], 'hetero': ['n', 'n', 'n', 'n', 'n']}}
    '''
    key_list = [key for key in mmcif_dict_keys if key.split('.')[0] == fields]
    
    if len(key_list) == 0:
        return {}

    value_list = [mmcif_dict[key] for key in key_list]
    
    if first_key is None:
        first_key_idx = 0
    else:
        first_key_idx = key_list.index(f'{fields}.{first_key}')
    second_key_idx = None
    if second_key is not None :
        second_key_idx = key_list.index(f'{fields}.{second_key}')
        if second_key_idx < first_key_idx:
            temp = second_key_idx
            second_key_idx = first_key_idx
            first_key_idx = temp
        key_list = key_list[:first_key_idx] + key_list[first_key_idx+1:second_key_idx] + key_list[second_key_idx+1:]
        id_list = zip(value_list[first_key_idx], value_list[second_key_idx])
        id_list = [",".join(item) for item in id_list]
        value_list = value_list[:first_key_idx] + value_list[first_key_idx+1:second_key_idx] + value_list[second_key_idx+1:]
    else :
        key_list = key_list[:first_key_idx] + key_list[first_key_idx+1:]
        id_list = value_list[first_key_idx]
        value_list = value_list[:first_key_idx] + value_list[first_key_idx+1:]
    
    row_num = len(id_list)

    output_dict = {}
    for ii in range(row_num):
        if id_list[ii] not in output_dict:
            output_dict[id_list[ii]] = {key: [] for key in key_list}
        for key, value in zip(key_list, value_list):
            output_dict[id_list[ii]][key].append(value[ii])

    if len1_to_scalar:
        # len 1 list to scalar
        for key in output_dict:
            for key2 in output_dict[key]:
                if len(output_dict[key][key2]) == 1:
                    output_dict[key][key2] = output_dict[key][key2][0]
    
    return output_dict

def merge_dict(dict_list):
    merged_dict = {}
    for d in dict_list:
        for k1, v1 in d.items():
            if k1 not in merged_dict:
                merged_dict[k1] = {}
            for k2, v2 in v1.items():
                merged_dict[k1][k2] = v2 if k2 not in merged_dict[k1] else merged_dict[k1][k2] + v2
    return merged_dict

def get_all_cif(cif_dir,cif_path):
    cif_list = []
    for inner_dir in os.listdir(cif_dir):
        inner_dir_path = os.path.join(cif_dir, inner_dir)
        if os.path.isdir(inner_dir_path):
            for cif in os.listdir(inner_dir_path):
                if cif.endswith('.cif.gz'):
                    cif_list.append(os.path.join(inner_dir_path, cif))

    with open(cif_path, 'w') as f:
        for cif in cif_list:
            f.write(cif + '\n')

if __name__ == "__main__":
    cif_dir = "/public_data/psk6950/PDB_2024Mar18/cif/"
    cif_path = "/public_data/psk6950/PDB_2024Mar18/cif_list.txt"
    get_all_cif(cif_dir, cif_path)