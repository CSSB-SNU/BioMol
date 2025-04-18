import os
from Bio.PDB.MMCIF2Dict import MMCIF2Dict as mmcif2dict
from utils.feature import *
from utils.hierarchy import ChemComp
from constant.chemical import stereo_config_map
import pickle
import lmdb

ligand_configs = {
    "0D": {
        "id": ("_chem_comp.id",str),
        "name" : ("_chem_comp.name",str),
        "formula" : ("_chem_comp.formula",str),
    },
    "1D": {
        "full_atoms" : ("_chem_comp_atom.atom_id",str),
        "one_letter_atoms" : ("_chem_comp_atom.type_symbol",str),
        "ideal_x" : ("_chem_comp_atom.pdbx_model_Cartn_x_ideal",float),
        "ideal_y" : ("_chem_comp_atom.pdbx_model_Cartn_y_ideal",float),
        "ideal_z" : ("_chem_comp_atom.pdbx_model_Cartn_z_ideal",float),
    },
    "2D": {
        "bond" : "_chem_comp_bond"
    }
}
def split_cif(cif_path, temp_dir):
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    with open(cif_path, 'r') as f:
        lines = f.readlines()

    # split components.cif into individual files ("data_{three-letter-code}.cif")
    start = 0
    for i, line in enumerate(lines):
        if line.startswith('data_'):
            ID = lines[start].strip().split('_')[1]
            if not os.path.exists(f'{temp_dir}{ID[0]}'):
                os.makedirs(f'{temp_dir}{ID[0]}')
            if i > 0:
                with open(f'{temp_dir}{ID[0]}/{ID}.cif', 'w') as f:
                    f.writelines(lines[start:i])
            start = i

    # write the last component
    with open(f'{temp_dir}{lines[start].strip()}.cif', 'w') as f:
        f.writelines(lines[start:])

def parse_cif(cif_path):
    mmcif_dict = mmcif2dict(cif_path)

    config_0D = ligand_configs["0D"]
    config_1D = ligand_configs["1D"]
    feature_level = FeatureLevel.CHEMCOMP

    output_0D = {}
    for key in config_0D:
        key_in_mmcif_dict, data_type = config_0D[key]
        data = mmcif_dict[key_in_mmcif_dict]
        data = data[0] if len(data) == 1 else data
        data = data_type(data)
        data = Feature0D(key, data, feature_level, None)
        output_0D[key] = data

    output_0D = FeatureMap0D(output_0D)   

    if "UNL" in output_0D["id"].feature():
        full_atoms = []
        return ChemComp(output_0D, None, None)

    output_1D = {}

    full_atoms = mmcif_dict['_chem_comp_atom.atom_id']
    one_letter_atoms = mmcif_dict['_chem_comp_atom.type_symbol']
    # hydrogen_mask = [atom != 'H' for atom in one_letter_atoms]
    # full_atoms = [atom for atom, mask in zip(full_atoms, hydrogen_mask) if mask]
    # one_letter_atoms = [atom for atom, mask in zip(one_letter_atoms, hydrogen_mask) if mask]
    full_atoms_mask = [d != '?' for d in full_atoms]
    one_letter_atoms_mask = [d != '?' for d in one_letter_atoms]

    output_1D["full_atoms"] = Feature1D("full_atoms", full_atoms, full_atoms_mask, feature_level, None)
    output_1D["one_letter_atoms"] = Feature1D("one_letter_atoms", one_letter_atoms, one_letter_atoms_mask, feature_level, None)
    
    for key in ["ideal_x", "ideal_y", "ideal_z"]:
        key_in_mmcif_dict, data_type = config_1D[key]
        data = mmcif_dict[key_in_mmcif_dict]
        data = [float(d) if d != '?' else 0.0 for d in data]
        data = [data_type(d) for d in data]
        mask = [d != '?' for d in data]
        output_1D[key] = Feature1D(key, torch.tensor(data), torch.tensor(mask), feature_level, None)
    ideal_coords = torch.stack([output_1D["ideal_x"].feature()[0], output_1D["ideal_y"].feature()[0], output_1D["ideal_z"].feature()[0]]).T # (N, 3) or (3,)
    if ideal_coords.dim() == 1:
        ideal_coords = ideal_coords.unsqueeze(0)
    ideal_coords_mask = torch.stack([output_1D["ideal_x"].feature()[1], output_1D["ideal_y"].feature()[1], output_1D["ideal_z"].feature()[1]]).T # (N, 3)
    ideal_coords_mask = torch.all(ideal_coords_mask, dim=1) # (N,)
    ideal_coords = Feature1D("ideal_coords", ideal_coords, ideal_coords_mask, feature_level, None)
    output_1D['ideal_coords'] = ideal_coords

    # remove key ideal_x ~ model_z
    keys_to_remove = ["ideal_x", "ideal_y", "ideal_z"]
    for key in keys_to_remove:
        output_1D.pop(key)
    output_1D = FeatureMap1D(output_1D)

    head = 5 # (atom_idx1) (atom_idx2) (single, double, triple), (aromatic, non-aromatic), (stereo, non-stereo)

    if '_chem_comp_bond.atom_id_1' in mmcif_dict:
        _chem_comp_bond_atom_id1 = mmcif_dict['_chem_comp_bond.atom_id_1']
        _chem_comp_bond_atom_id2 = mmcif_dict['_chem_comp_bond.atom_id_2']
        _chem_comp_bond_value_order = mmcif_dict['_chem_comp_bond.value_order']
        _chem_comp_bond_aromatic_flag = mmcif_dict['_chem_comp_bond.pdbx_aromatic_flag']
        _chem_comp_bond_stereo_config = mmcif_dict['_chem_comp_bond.pdbx_stereo_config']
        bond_number = len(_chem_comp_bond_atom_id1)
        bond_type_dict = {}
        aromatic_flag_dict = {}
        stereo_config_dict = {}
        output_2D = {}

        for bond_idx in range(bond_number):
            atom1 = _chem_comp_bond_atom_id1[bond_idx]
            atom2 = _chem_comp_bond_atom_id2[bond_idx]
            if atom1 not in output_1D["full_atoms"].value or atom2 not in output_1D["full_atoms"].value:
                continue
            atom1_idx = output_1D["full_atoms"].value.index(atom1)
            atom2_idx = output_1D["full_atoms"].value.index(atom2)
            bond_type = _chem_comp_bond_value_order[bond_idx]
            aromatic_flag = 1 if _chem_comp_bond_aromatic_flag[bond_idx] == 'Y' else 0
            stereo_config = stereo_config_map.get(_chem_comp_bond_stereo_config[bond_idx], -1)

            bond_type_dict[(atom1_idx, atom2_idx)] = bond_type
            aromatic_flag_dict[(atom1_idx, atom2_idx)] = aromatic_flag
            stereo_config_dict[(atom1_idx, atom2_idx)] = stereo_config

        output_2D["bond_type"] = FeaturePair("bond_type", bond_type_dict, feature_level, True, None)
        output_2D["aromatic"] = FeaturePair("aromatic", aromatic_flag_dict, feature_level, True, None)
        output_2D["stereo"] = FeaturePair("stereo", stereo_config_dict, feature_level, True, None)
        output_2D = FeatureMapPair(output_2D)

    else:
        output_2D = FeatureMapPair({})
    output = ChemComp(output_0D, output_1D, output_2D)

    output.get_ideal_coords()
    return output

def save_ligand_info(temp_dir, env_path):
    env = lmdb.open(env_path, map_size=1024 ** 3)
    big_dict = {}
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            if file.endswith('.cif'):
                cif_path = os.path.join(root, file)
                ligand_info = parse_cif(cif_path)
                ligand_ID = ligand_info.get_code()
                big_dict[ligand_ID] = ligand_info
                ligand_info.get_bonds()

    # write to lmdb
    with env.begin(write=True) as txn:
        for key, value in big_dict.items():
            txn.put(key.encode(), pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))

    env.close()

if __name__ == '__main__':
    save_ligand_info('/public_data/psk6950/CCD/components_tmp/', '/public_data/psk6950/CCD/ligand_info.lmdb')
