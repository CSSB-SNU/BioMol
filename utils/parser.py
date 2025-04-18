from Bio.PDB.MMCIF2Dict import MMCIF2Dict as mmcif2dict
from Bio.PDB import PDBParser, MMCIFIO
import json
import torch
import pickle
import copy
import gzip
from utils.feature import *
from utils.hierarchy import *
from utils.parser_utils import *
from constant.chemical import stereo_config_map, num2residue
from constant.datapath import IDEAL_LIGAND_PATH


# read ideal_ligand_list using pickle
with open(IDEAL_LIGAND_PATH, "rb") as f:
    ideal_ligand_list = pickle.load(f)

chem_comp_configs = {
    "0D": {
        "id" : ("_chem_comp.id",str),
        "name" : ("_chem_comp.name",str),
        "formula" : ("_chem_comp.formula",str),
    },
    "1D": {
        "full_atoms" : ("_chem_comp_atom.atom_id",str),
        "one_letter_atoms" : ("_chem_comp_atom.type_symbol",str),
    },
    "2D": {
        "bond" : "_chem_comp_bond"
    }
}

def parse_chem_comp(chem_comp_id, chem_comp_item):
    feature_level = FeatureLevel.CHEMCOMP

    config_0D = chem_comp_configs["0D"]
    config_1D = chem_comp_configs["1D"]
    output_0D = {'id' : Feature0D('id', chem_comp_id, feature_level, None)}
    for key in config_0D:
        key_in_mmcif_dict, data_type = config_0D[key]
        if key_in_mmcif_dict in chem_comp_item:
            data = chem_comp_item[key_in_mmcif_dict]
            data = data_type(data)
            data = Feature0D(key, data, feature_level, None)
            output_0D[key] = data   

    output_1D = {}
    for key in config_1D:
        key_in_mmcif_dict, data_type = config_1D[key]
        if key_in_mmcif_dict not in chem_comp_item:
            continue
        data = chem_comp_item[key_in_mmcif_dict]
        data = [data] if type(data) == str else data
        if data_type == bool:
            # change ? to N
            data = [d == 'Y' for d in data]
        elif data_type == float:
            data = [d if d != '?' else 0.0 for d in data]
        data = [data_type(d) for d in data]
        mask = [d != '?' for d in data]
        if data_type != str:
            output_1D[key] = Feature1D(key, torch.tensor(data), torch.tensor(mask), feature_level, None)
        else:
            output_1D[key] = Feature1D(key, data, mask, feature_level, None)

    head = 5 # (atom_idx1) (atom_idx2) (single, double, triple), (aromatic, non-aromatic), (stereo, non-stereo)

    if '_chem_comp_bond.atom_id_1' in chem_comp_item:
        _chem_comp_bond_atom_id1 = chem_comp_item['_chem_comp_bond.atom_id_1']
        _chem_comp_bond_atom_id2 = chem_comp_item['_chem_comp_bond.atom_id_2']
        _chem_comp_bond_value_order = chem_comp_item['_chem_comp_bond.value_order']
        _chem_comp_bond_aromatic_flag = chem_comp_item['_chem_comp_bond.pdbx_aromatic_flag']
        _chem_comp_bond_stereo_config = chem_comp_item['_chem_comp_bond.pdbx_stereo_config']
        
        bond_number = len(_chem_comp_bond_atom_id1)
        bond_type_dict = {}
        aromatic_flag_dict = {}
        stereo_config_dict = {}
        output_2D = {}

        for bond_idx in range(bond_number):
            atom1 = _chem_comp_bond_atom_id1[bond_idx]
            atom2 = _chem_comp_bond_atom_id2[bond_idx]
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

    else:
        output_2D = {}

    output = {
        "0D": output_0D,
        "1D": output_1D,
        "2D": output_2D
    }

    return output

def compare_ideal_chem_comp(chem_comp):
    chem_comp_id = chem_comp['0D']['id'].value
    # ideal_chem_comp = ideal_ligand_info[chem_comp_id]
    ideal_chem_comp = read_lmdb_value(chem_comp_id)
    if chem_comp_id == "UNL":
        return ideal_chem_comp

    chem_comp_help = {}

    # compare 0d feature
    key_0d = ['name','formula']
    flag_0d = []
    for key in key_0d :
        if key in chem_comp['0D'].keys():
            in_item, ideal = chem_comp['0D'][key], ideal_chem_comp[key]
            if in_item != ideal :
                chem_comp_help[key] = f'{key} is different : load {in_item}, but {ideal} at external source.'
        else :
            chem_comp['0D'][key] = ideal_chem_comp[key]
            flag_0d.append(f'0D {key} is from external source')
    
    chem_comp['0D']['flag'] = Feature0D('flag', flag_0d, FeatureLevel.CHEMCOMP, None)

    # compare 1d feature
    key_1d = chem_comp_configs['1D'].keys()
    flag_1d = []
    for key in key_1d : 
        if key in chem_comp['1D']:
            in_item, ideal = chem_comp['1D'][key], ideal_chem_comp[key]
            if type(in_item) == list:
                if in_item != ideal:
                    if len(in_item) != len(ideal):
                        chem_comp_help[key] = f'{key} length between loaded item and external source is different. You should check it.'
                    else:
                        diff = [idx for idx, (x,y) in enumerate(zip(in_item, ideal)) if x!=y]
                        diff = ",".join(diff)
                        chem_comp_help[key] = f'{key} is different at {diff}'
            elif type(in_item) == torch.Tensor:
                if torch.any(in_item!=ideal):
                    if in_item.shape != ideal.shape:
                        chem_comp_help[key] = f'{key} shape between loaded item and external source is different. You should check it.'
                    else:
                        assert len(in_item.shape) == 1
                        diff = torch.where(in_item != ideal)[0].tolist()
                        diff = [str(item) for item in diff]
                        diff = ",".join(diff)
                        chem_comp_help[key] = f'{key} is different at {diff}'
            else :
                chem_comp_help[key] = f'{key} weird data type'
        else :
            ideal_value, ideal_mask = ideal_chem_comp[key].feature()
            additional_info = f'{key} is from external source'
            chem_comp['1D'][key] = Feature1D(key, ideal_value, ideal_mask, FeatureLevel.CHEMCOMP, additional_info)

    # copmare 2d features
    if '2D' not in chem_comp or chem_comp['2D'] is None  or len(chem_comp['2D']) == 0:
        chem_comp['2D'] = ideal_chem_comp.get_pair_features()
    # else :
    #     if chem_comp['2D'].shape != ideal_chem_comp['2D'].shape :
    #         chem_comp_help['bond'] = f'number of bonds is different from external source. please check it.'
    #     elif torch.any(chem_comp['2D']!=ideal_chem_comp['2D']) :
    #         chem_comp_help['bond'] = f'bond info is somewhat different from external source. please check it.'

    chem_comp['help'] = chem_comp_help

    chem_comp['0D'] = FeatureMap0D(chem_comp['0D'])
    chem_comp['1D'] = FeatureMap1D(chem_comp['1D'])
    chem_comp['2D'] = FeatureMapPair(chem_comp['2D'])

    chem_comp = ChemComp(chem_comp['0D'], chem_comp['1D'], chem_comp['2D'], chem_comp['help'])

    return chem_comp

def parse_chem_comp_dict(_chem_comp_dict):
    output_dict = {}

    for key in _chem_comp_dict:
        chem_comp_item = _chem_comp_dict[key]
        chem_comp = parse_chem_comp(key, chem_comp_item)
        chem_comp = compare_ideal_chem_comp(chem_comp)
        output_dict[key] = chem_comp

    return output_dict
    
def parse_struct_ref_dict(struct_ref_dict):
    # input struct_ref_dict key is align id. output key : entity_id
    if len(struct_ref_dict) == 0 :
        return struct_ref_dict
    output_dict = {}
    for strut_ref_align_id, value in struct_ref_dict.items() :
        key_list = list(value.keys())
        entity_id = value['_struct_ref.entity_id']
        if entity_id not in output_dict :
            output_dict[entity_id] = {key : value[key] for key in key_list if key != '_struct_ref.entity_id'}
        else :
            for key in output_dict[entity_id].keys():
                if type(output_dict[entity_id][key]) == str:
                    output_dict[entity_id][key] = [output_dict[entity_id][key]] + value[key]
                elif type(output_dict[entity_id][key]) == list:
                    output_dict[entity_id][key] = output_dict[entity_id][key] + value[key]

    return output_dict

def parse_polymer_entity(items, chem_comp_dict):
    '''
    "_entity_poly",
    "_entity_poly_seq",
    "_struct_ref",
    "_struct_ref_seq",
    "_struct_ref_seq_dif",

    0D feature : 
        polymer_type (required)
        auth_asym_id (required)
    1D feature :
        one_letter_code (required)
        one_letter_code_can (required)
        chem_comp_list (required)
        chem_comp_hetero (required)
    2D feature :
        parsed later
    Additional Info :
        struct_ref (optional)
    '''
    feature_level = FeatureLevel.ENTITY
    polymer_0D = {}
    polymer_1D = {}

    # parse sequence
    polymer_0D['polymer_type'] = Feature0D('polymer_type', PolymerType(items['_entity_poly.type']), feature_level, None) # protein : polypeptide(L)
    polymer_0D['auth_asym_id'] = Feature0D('auth_asym_id', items['_entity_poly.pdbx_strand_id'], feature_level, None)

    seq_one_letter_code = items['_entity_poly.pdbx_seq_one_letter_code'].replace("\n","")
    seq_one_letter_code_can = items['_entity_poly.pdbx_seq_one_letter_code_can'].replace("\n","")

    if len(seq_one_letter_code) != len(seq_one_letter_code_can):
        sequence_split = re.findall(r"\(.*?\)|.", seq_one_letter_code)
        if len(sequence_split) != len(seq_one_letter_code_can):
            sequence_split = [seq if '(' not in seq else "X" for seq in sequence_split]
            seq_one_letter_code_can = "".join(sequence_split)

    polymer_1D['one_letter_code'] = Feature1D('one_letter_code', seq_one_letter_code, None, feature_level, None)
    polymer_1D['one_letter_code_can'] = Feature1D('one_letter_code_can', seq_one_letter_code_can, None, feature_level, None)

    # handle heterogeneous sequence
    seq_num_list = items['_entity_poly_seq.num']
    chem_comp_list = ChemCompView(chem_comp_dict, items['_entity_poly_seq.mon_id'])
    hetero_list = items['_entity_poly_seq.hetero']

    hetero_list = [1 if h == 'y' or h == 'yes' else 0 for h in hetero_list]
    seq_num_to_chem_comp = {}
    seq_num_to_hetero = {}
    for seq_num, chem_comp, hetero in zip(seq_num_list, chem_comp_list, hetero_list):
        if seq_num not in seq_num_to_chem_comp:
            seq_num_to_chem_comp[seq_num] = []
            seq_num_to_hetero[seq_num] = []
        seq_num_to_chem_comp[seq_num].append(chem_comp)
        seq_num_to_hetero[seq_num].append(hetero)
    
    chem_comp_list = list(seq_num_to_chem_comp.values())
    hetero_list = list(seq_num_to_hetero.values())

    polymer_1D['chem_comp_list'] = Feature1D('chem_comp_list', chem_comp_list, None, feature_level, None)
    polymer_1D['chem_comp_hetero'] = Feature1D('chem_comp_hetero', hetero_list, None, feature_level, None)

    polymer_0D = FeatureMap0D(polymer_0D)
    polymer_1D = FeatureMap1D(polymer_1D)

    polymer = Polymer(polymer_0D, polymer_1D, None, None)
    return polymer

def parse_nonpolymer_entity(items, chem_comp_dict, is_water = False):
    '''
    "_pdbx_entity_nonpoly",
    0D feature : 
        name (required)
        chem_comp_id (required)
    1D feature :
        parsed later
    2D feature :
        parsed later
    '''
    
    nonpolymer_0D = {}
    feature_level = FeatureLevel.ENTITY
    nonpolymer_0D['name'] = Feature0D('name', items.get('_pdbx_entity_nonpoly.entity_nonpoly',None), feature_level, None)
    nonpolymer_0D['chem_comp'] = Feature0D('chem_comp',items.get('_pdbx_entity_nonpoly.comp_id',None), feature_level, None)
    nonpolymer_0D = FeatureMap0D(nonpolymer_0D)
    chem_comp = chem_comp_dict[nonpolymer_0D['chem_comp'].value]
    nonpolymer = NonPolymer(nonpolymer_0D, chem_comp, is_water=is_water)

    return nonpolymer

def parse_branched_entity(items, chem_comp_dict):
    feature_level = FeatureLevel.ENTITY
    
    branched_0D = {
        'descriptor': Feature0D('descriptor', items.get('_pdbx_entity_branch_descriptor.descriptor', None), feature_level, None),
        'type': Feature0D('type', items.get('_pdbx_entity_branch_descriptor.type', None), feature_level, None),
        'program': Feature0D('program', items.get('_pdbx_entity_branch_descriptor.program', None), feature_level,  None),
        'program_version': Feature0D('program_version', items.get('_pdbx_entity_branch_descriptor.program_version', None), feature_level,  None)
    }

    branched_0D = FeatureMap0D(branched_0D)

    chem_comp_list = items['_pdbx_entity_branch_list.comp_id']
    chem_comp_list = ChemCompView(chem_comp_dict, chem_comp_list)
    seq_num_list = items['_pdbx_entity_branch_list.num']
    hetero_list = items['_pdbx_entity_branch_list.hetero']


    # handle heterogeneous sequence
    hetero_list = [1 if h == 'y' or h == 'yes' else 0 for h in hetero_list]
    seq_num_to_chem_comp = {}
    seq_num_to_hetero = {}
    for seq_num, chem_comp, hetero in zip(seq_num_list, chem_comp_list, hetero_list):
        if seq_num not in seq_num_to_chem_comp:
            seq_num_to_chem_comp[seq_num] = []
            seq_num_to_hetero[seq_num] = []
        seq_num_to_chem_comp[seq_num].append(chem_comp)
        seq_num_to_hetero[seq_num].append(hetero)
    
    seq_num_list = list(seq_num_to_chem_comp.keys())
    chem_comp_list = list(seq_num_to_chem_comp.values())
    hetero_list = list(seq_num_to_hetero.values())

    branched_1D = {
        'chem_comp_list' : Feature1D('chem_comp_list', chem_comp_list, None, feature_level, None),
        'chem_comp_num' : Feature1D('chem_comp_num', seq_num_list, None, feature_level, None),
        'chem_comp_hetero' : Feature1D('chem_comp_hetero', hetero_list, None, feature_level, None)
    }
    branched_1D = FeatureMap1D(branched_1D)

    branch_link = {}

    comp_id_1_list = items.get('_pdbx_entity_branch_link.comp_id_1', None)
    comp_id_2_list = items.get('_pdbx_entity_branch_link.comp_id_2', None)
    atom_id_1_list = items.get('_pdbx_entity_branch_link.atom_id_1', None)
    atom_id_2_list = items.get('_pdbx_entity_branch_link.atom_id_2', None)
    leaving_atom_id_1_list = items.get('_pdbx_entity_branch_link.leaving_atom_id_1', None)
    leaving_atom_id_2_list = items.get('_pdbx_entity_branch_link.leaving_atom_id_2', None)
    bond_list = items.get('_pdbx_entity_branch_link.value_order', None)
    entity_branch_list_num_1_list = items.get('_pdbx_entity_branch_link.entity_branch_list_num_1', None)
    entity_branch_list_num_2_list = items.get('_pdbx_entity_branch_link.entity_branch_list_num_2', None)

    if type(comp_id_1_list) == str:
        comp_id_1_list = [comp_id_1_list]

    for idx in range(len(comp_id_1_list)):
        comp_id_1 = comp_id_1_list[idx]
        comp_id_2 = comp_id_2_list[idx]
        atom_id_1 = atom_id_1_list[idx]
        atom_id_2 = atom_id_2_list[idx]
        leaving_atom_id_1 = leaving_atom_id_1_list[idx]
        leaving_atom_id_2 = leaving_atom_id_2_list[idx]
        bond = bond_list[idx]
        entity_branch_list_num_1 = entity_branch_list_num_1_list[idx]
        entity_branch_list_num_2 = entity_branch_list_num_2_list[idx]    

        branch_link[(entity_branch_list_num_1, entity_branch_list_num_2)] = {'atom_id' : (atom_id_1, atom_id_2), 
                                           'leaving_atom_id' : (leaving_atom_id_1, leaving_atom_id_2), 
                                           'bond' : bond, 
                                           'comp_id' : (comp_id_1, comp_id_2)}
    
    branch_link = FeaturePair('branch_link', branch_link, feature_level, True, None)
    branch_link = {'branch_link' : branch_link}
    branch_link = FeatureMapPair(branch_link)

    branched = Branched(branched_0D, branched_1D, branch_link, None)

    return branched

def parse_entity_dict(entity_dict, chem_comp_dict):
    output_dict = {}
    for entity_id, items in entity_dict.items():
        molecule_type = MoleculeType(items['_entity.type'])
        match molecule_type:
            case MoleculeType.POLYMER:
                entity = parse_polymer_entity(items, chem_comp_dict)
            case MoleculeType.NONPOLYMER:
                entity = parse_nonpolymer_entity(items, chem_comp_dict)
            case MoleculeType.BRANCHED:
                entity = parse_branched_entity(items, chem_comp_dict)
            case MoleculeType.WATER:
                entity = parse_nonpolymer_entity(items, chem_comp_dict, is_water=True)
        output_dict[entity_id] = entity

    return output_dict

def to_float_tensor(input_list):
    if type(input_list) == str:
        return torch.tensor([float(input_list)])
    float_list = [float(item) for item in input_list]
    return torch.tensor(float_list)

def to_int_tensor(input_list):
    int_list = []
    for item in input_list:
        if item == ".":
            int_list.append(-1)
        else :
            int_list.append(int(item))
    return torch.tensor(int_list)

def parse_str_dict(str_dict,_struct_site_dict):
    output_dict = {}

    for asym_id, items in str_dict.items():
        idx_dict = {}
        str_dict = {}
        if len(set(items['_atom_site.label_entity_id'])) > 1 :
            print(F"{asym_id} has multiple entity") # which is not desired.
        idx_dict['model'] = to_int_tensor(items['_atom_site.pdbx_PDB_model_num'])
        idx_dict['label_alt_id'] = items['_atom_site.label_alt_id']
        label_alt_id_list = list(set(items['_atom_site.label_alt_id']) - set(["."]))

        idx_dict['raw_atom_list']  = items['_atom_site.type_symbol']
        idx_dict['full_atom_list'] = items['_atom_site.label_atom_id']
        idx_dict['comp_id_list'] = items['_atom_site.label_comp_id']
        idx_dict['cif_idx_list'] = to_int_tensor(items['_atom_site.label_seq_id']) # it can be empty for ligand or water.
        idx_dict['auth_idx_list'] = to_int_tensor(items['_atom_site.auth_seq_id'])
        idx_dict['auth_ins_code_list'] = items['_atom_site.pdbx_PDB_ins_code']
        idx_dict['entity_list'] = items['_atom_site.label_entity_id']

        x,y,z = items['_atom_site.Cartn_x'], items['_atom_site.Cartn_y'], items['_atom_site.Cartn_z']
        x,y,z = to_float_tensor(x), to_float_tensor(y), to_float_tensor(z)
        str_dict['coordinates'] = torch.stack([x,y,z],dim=-1) # (L,3)
        str_dict['occupancy'] = to_float_tensor(items['_atom_site.occupancy'])
        str_dict['B_factor'] = to_float_tensor(items['_atom_site.B_iso_or_equiv'])

        length = x.shape[0]
        modified = torch.zeros((length,))
        modified_details = None
        functioanl_site = torch.zeros((length,))
        functional_site_id = None
        functional_site_details = None

        if '_pdbx_struct_mod_residue.label_seq_id' in items :
            modified_idx = to_int_tensor(items['_pdbx_struct_mod_residue.label_seq_id'])
            modified[modified_idx] = 1
            modified_details = items['_pdbx_struct_mod_residue.details']
        if '_struct_site_gen.label_seq_id' in items :
            functinoal_site_idx = to_int_tensor(items['_struct_site_gen.label_seq_id'])
            functioanl_site[functinoal_site_idx] = 1

            functional_site_id = items['_struct_site_gen.site_id']
            functional_site_details = items['_struct_site_gen.details']
            assert len(functional_site_id) == len(functional_site_details)
            temp = []
            for id,detail in zip(functional_site_id,functional_site_details):
                detail2 =_struct_site_dict[id]
                if detail == "?":
                    detail = detail2
                else :
                    detail = f'{detail},{detail2}'
                temp.append(detail)
            functional_site_details = temp

        idx_dict['modified'] = modified
        idx_dict['modified_details'] = modified_details
        idx_dict['functional_site'] = functioanl_site
        idx_dict['functional_site_id'] = functional_site_id
        idx_dict['functional_site_details'] = functional_site_details

        output_dict[asym_id] = {'idx' : idx_dict, 'structure' : str_dict}
    return output_dict

def to_list(input_list):
    if not type(input_list) == list:
        return [input_list]
    return input_list

def parse_poly_scheme_dict(poly_scheme_dict):
    scheme_type = MoleculeType('polymer')

    entity_id = to_list(poly_scheme_dict['_pdbx_poly_seq_scheme.entity_id'])
    cif_idx = to_list(poly_scheme_dict['_pdbx_poly_seq_scheme.seq_id'])
    # auth_idx = to_list(poly_scheme_dict['_pdbx_poly_seq_scheme.auth_seq_num'])
    auth_idx = to_list(poly_scheme_dict['_pdbx_poly_seq_scheme.pdb_seq_num'])
    ins_code = to_list(poly_scheme_dict['_pdbx_poly_seq_scheme.pdb_ins_code'])
    hetero = to_list(poly_scheme_dict['_pdbx_poly_seq_scheme.hetero'])
    chem_comp_list = to_list(poly_scheme_dict['_pdbx_poly_seq_scheme.mon_id'])

    entity_id = list(set(entity_id))
    assert len(entity_id) == 1, f"entity_id is not unique : {entity_id}"
    entity_id = entity_id[0]
    cif_idx_list = [int(cc) for cc in cif_idx]
    auth_idx_list = [aa if ii == '.' else f'{aa}.{ii}' for aa, ii in zip(auth_idx, ins_code)]
    hetero_list = [1 if h == 'y' or h == 'yes' else 0 for h in hetero]

    # handle heterogeneous sequence
    auth_idx_to_chem_comp = {}
    auth_idx_to_hetero = {}
    new_auth_idx_list = []
    new_cif_idx_list = []
    for idx in range(len(auth_idx)):
        chem_comp = chem_comp_list[idx]
        cif_idx = cif_idx_list[idx]
        auth_idx = auth_idx_list[idx]
        hetero = hetero_list[idx]

        if auth_idx not in auth_idx_to_chem_comp:
            auth_idx_to_chem_comp[auth_idx] = []
            auth_idx_to_hetero[auth_idx] = []
        auth_idx_to_chem_comp[auth_idx].append(chem_comp)
        auth_idx_to_hetero[auth_idx].append(hetero)
        if auth_idx not in new_auth_idx_list:
            new_auth_idx_list.append(auth_idx)
            new_cif_idx_list.append(cif_idx)
    
    chem_comp_list = list(auth_idx_to_chem_comp.values())
    hetero_list = [1 if 1 in h else 0 for h in auth_idx_to_hetero.values()]
    cif_idx_list = new_cif_idx_list
    auth_idx_list = new_auth_idx_list

    return Scheme(entity_id, scheme_type, cif_idx_list, auth_idx_list, chem_comp_list, hetero_list)

def parse_nonpoly_scheme_dict(nonpoly_scheme_dict):
    entity_id = to_list(nonpoly_scheme_dict['_pdbx_nonpoly_scheme.entity_id'])
    auth_idx = to_list(nonpoly_scheme_dict['_pdbx_nonpoly_scheme.pdb_seq_num'])
    cif_idx = [ii for ii in range(len(auth_idx))]
    ins_code = to_list(nonpoly_scheme_dict['_pdbx_nonpoly_scheme.pdb_ins_code'])
    hetero = None
    chem_comp_list = to_list(nonpoly_scheme_dict['_pdbx_nonpoly_scheme.mon_id'])

    if list(set(chem_comp_list)) == ['HOH']:
        scheme_type = MoleculeType('water')
    else :
        scheme_type = MoleculeType('non-polymer')
    
    chem_comp_list = [[cc] for cc in chem_comp_list]

    entity_id = list(set(entity_id))
    assert len(entity_id) == 1, f"entity_id is not unique : {entity_id}"
    entity_id = entity_id[0]
    auth_idx = [aa if ii == '.' else f'{aa}.{ii}' for aa, ii in zip(auth_idx, ins_code)]
    return Scheme(entity_id, scheme_type, cif_idx, auth_idx, chem_comp_list, hetero)

def parse_branched_scheme_dict(branched_scheme_dict):
    entity_id = to_list(branched_scheme_dict['_pdbx_branch_scheme.entity_id'])
    auth_idx_list = to_list(branched_scheme_dict['_pdbx_branch_scheme.pdb_seq_num'])
    cif_idx_list = [ii for ii in range(len(auth_idx_list))]
    hetero_list = to_list(branched_scheme_dict['_pdbx_branch_scheme.hetero'])
    chem_comp_list = to_list(branched_scheme_dict['_pdbx_branch_scheme.mon_id'])

    scheme_type = MoleculeType('branched')

    hetero_list = [1 if h == 'y' or h == 'yes' else 0 for h in hetero_list]

    # handle heterogeneous sequence
    auth_idx_to_chem_comp = {}
    auth_idx_to_hetero = {}
    new_auth_idx_list = []
    new_cif_idx_list = []
    for idx in range(len(auth_idx_list)):
        chem_comp = chem_comp_list[idx]
        cif_idx = cif_idx_list[idx]
        auth_idx = auth_idx_list[idx]
        hetero = hetero_list[idx]

        if auth_idx not in auth_idx_to_chem_comp:
            auth_idx_to_chem_comp[auth_idx] = []
            auth_idx_to_hetero[auth_idx] = []
        auth_idx_to_chem_comp[auth_idx].append(chem_comp)
        auth_idx_to_hetero[auth_idx].append(hetero)
        if auth_idx not in new_auth_idx_list:
            new_auth_idx_list.append(auth_idx)
            new_cif_idx_list.append(cif_idx)
    
    chem_comp_list = list(auth_idx_to_chem_comp.values())
    hetero_list = [1 if 1 in h else 0 for h in auth_idx_to_hetero.values()]
    cif_idx_list = new_cif_idx_list
    auth_idx_list = new_auth_idx_list

    entity_id = list(set(entity_id))
    assert len(entity_id) == 1, f"entity_id is not unique : {entity_id}"
    entity_id = entity_id[0]

    return Scheme(entity_id, scheme_type, cif_idx_list, auth_idx_list, chem_comp_list, hetero_list)

def parse_scheme_dict(scheme_dict):
    output_dict = {}
    for asym_id, items in scheme_dict.items():
        if '_pdbx_poly_seq_scheme.entity_id' in items :
            output_dict[asym_id] = parse_poly_scheme_dict(items)
        elif '_pdbx_nonpoly_scheme.entity_id' in items :
            output_dict[asym_id] = parse_nonpoly_scheme_dict(items)
        elif '_pdbx_branch_scheme.entity_id' in items :
            output_dict[asym_id] = parse_branched_scheme_dict(items)

    return output_dict

def remove_UNL_from_atom_site(_atom_site_dict):
    new_dict = copy.deepcopy(_atom_site_dict)
    remove_UNL = False
    for asym_id in _atom_site_dict.keys():
        label_comp_id_list = _atom_site_dict[asym_id]['_atom_site.label_comp_id']
        UNL_mask = [cc != 'UNL' for cc in label_comp_id_list]
        if not remove_UNL and any(UNL_mask):
            remove_UNL = True
        if not any(UNL_mask):
            del new_dict[asym_id]
            continue
        for key in _atom_site_dict[asym_id].keys():
            new_dict[asym_id][key] = [_atom_site_dict[asym_id][key][idx] for idx in range(len(UNL_mask)) if UNL_mask[idx]]

    return new_dict, remove_UNL

def remove_UNL_from_struct_conn(_struct_conn_dict):
    new_dict = copy.deepcopy(_struct_conn_dict)
    for asym_pair in _struct_conn_dict.keys():
        label_comp_id1_list = _struct_conn_dict[asym_pair]['_struct_conn.ptnr1_label_comp_id']
        label_comp_id2_list = _struct_conn_dict[asym_pair]['_struct_conn.ptnr2_label_comp_id']
        UNL_mask = [cc1 != 'UNL' and cc2 != 'UNL' for cc1, cc2 in zip(label_comp_id1_list, label_comp_id2_list)]
        if not any(UNL_mask):
            del new_dict[asym_pair]
            continue
        for key in _struct_conn_dict[asym_pair].keys():
            new_dict[asym_pair][key] = [new_dict[asym_pair][key][idx] for idx in range(len(UNL_mask)) if UNL_mask[idx]]

    return new_dict

def parse_expression(expr: str):
    """
    Parse an expression into a list of strings, expanding ranges when present.
    
    Examples:
        '(1,2,6,10,23,24)' -> ['1','2','6','10','23','24']
        '(1-60)'           -> ['1','2','3', ..., '60']
        '1'                -> ['1']
        'P'                -> ['P']
        'P,Q'              -> ['P','Q']
        '1-10,21-25'       -> ['1','2', ..., '10', '21', ..., '25']
        '(X0)(1-60)'       -> None # 20250302, psk 2024Mar03  : (2xgk,1dwn,2vf9,1lp3,5fmo,3cji,3dpr,4ang,1al0,4nww,1cd3,4aed) -> No need to use it
    """
    if ')(' in expr:
        return None
    def split_top_level_commas(s):
        """Split the input string on commas that are not nested inside parentheses."""
        result = []
        current = []
        paren_depth = 0
        for char in s:
            if char == '(':
                paren_depth += 1
                current.append(char)
            elif char == ')':
                paren_depth -= 1
                current.append(char)
            elif char == ',' and paren_depth == 0:
                result.append("".join(current).strip())
                current = []
            else:
                current.append(char)
        if current:
            result.append("".join(current).strip())
        return result

    def parse_parenthesized(s):
        """
        Parse a string that may be a comma-separated list and/or a range.
        
        - If it contains commas, each piece is processed individually.
        - If a piece contains a hyphen and both parts are numeric,
          it is expanded into a numeric range.
        - Otherwise, the piece is returned as is.
        """
        # If there are commas, split and process each component.
        if ',' in s:
            items = [itm.strip() for itm in s.split(',')]
            result = []
            for item in items:
                if '-' in item:
                    # Expand numeric range if applicable.
                    start_str, end_str = item.split('-', 1)
                    start_str, end_str = start_str.strip(), end_str.strip()
                    if start_str.isdigit() and end_str.isdigit():
                        start_val = int(start_str)
                        end_val = int(end_str)
                        result.extend([str(x) for x in range(start_val, end_val + 1)])
                    else:
                        result.append(item)
                else:
                    result.append(item)
            return result
        elif '-' in s:
            # No commas; check if the whole thing is a range.
            start_str, end_str = s.split('-', 1)
            start_str, end_str = start_str.strip(), end_str.strip()
            if start_str.isdigit() and end_str.isdigit():
                start_val = int(start_str)
                end_val = int(end_str)
                return [str(x) for x in range(start_val, end_val + 1)]
            else:
                return [s]
        else:
            return [s]

    def parse_token(token):
        """
        Process a token which might be a mix of parenthesized groups and plain text.
        
        If the token does not contain any parentheses, we pass it to parse_parenthesized
        so that forms like "1-10" or "P,Q" are expanded.
        """
        token = token.strip()
        if '(' not in token and ')' not in token:
            # No parentheses at all; process the whole token.
            return parse_parenthesized(token)
        
        # Otherwise, extract the parts using regex.
        pattern = r'\([^)]*\)|[^()]+'
        parts = re.findall(pattern, token)
        results = []
        for part in parts:
            part = part.strip()
            if part.startswith('(') and part.endswith(')'):
                # Remove the outer parentheses and process the inside.
                inside = part[1:-1].strip()
                results.extend(parse_parenthesized(inside))
            elif part:
                # For any plain-text part, also process it to catch ranges like "1-10".
                results.extend(parse_parenthesized(part))
        return results

    # First split by top-level commas, then process each token.
    tokens = split_top_level_commas(expr)
    final_result = []
    for t in tokens:
        final_result.extend(parse_token(t))
    return final_result

def parse_assembly_dict(mmcif_dict, mmcif_dict_keys):
    assembly_id_list = mmcif_dict['_pdbx_struct_assembly_gen.assembly_id']
    oper_expression_list = mmcif_dict['_pdbx_struct_assembly_gen.oper_expression']
    asym_id_list = mmcif_dict['_pdbx_struct_assembly_gen.asym_id_list']

    output_dict = {}
    for ii in range(len(assembly_id_list)):
        assembly_id = assembly_id_list[ii]
        oper_expression = oper_expression_list[ii]
        oper_expression = parse_expression(oper_expression)
        if oper_expression is None :
            continue

        asym_ids = asym_id_list[ii].split(',')
        if assembly_id not in output_dict:
            output_dict[assembly_id] = {}
        for asym_id in asym_ids:
            if asym_id not in output_dict[assembly_id]:
                output_dict[assembly_id][asym_id] = []
            output_dict[assembly_id][asym_id].extend(oper_expression)
    return output_dict

def get_pdbx_struct_oper(pdbx_struct_oper_dict):
    output_dict = {}
    for struct_oper_id in pdbx_struct_oper_dict.keys():
        raw_dict = pdbx_struct_oper_dict[struct_oper_id]
        output_dict[struct_oper_id] = {}
        matrix = []
        for ii in range(1,4):
            row = []
            for jj in range(1,4):
                row.append(float(raw_dict[f'_pdbx_struct_oper_list.matrix[{ii}][{jj}]']))
            matrix.append(row)
        matrix = torch.tensor(matrix)
        output_dict[struct_oper_id]['matrix'] = matrix
        vector = []
        for ii in range(1,4):
            vector.append(float(raw_dict[f'_pdbx_struct_oper_list.vector[{ii}]']))
        vector = torch.tensor(vector)
        output_dict[struct_oper_id]['vector'] = vector
        output_dict[struct_oper_id]['type' ] = raw_dict['_pdbx_struct_oper_list.type']
        output_dict[struct_oper_id]['name'] = raw_dict['_pdbx_struct_oper_list.name']
        output_dict[struct_oper_id]['symmetry_operation'] = raw_dict['_pdbx_struct_oper_list.symmetry_operation']
    return output_dict

def parse_cif(cif_path, cif_configs=None, remove_signal_peptide=False):
    '''
    Parse a CIF file and return a dictionary of parsed data.

    Args:
        cif_path (str): Path to the CIF file to parse.
        cif_configs (dict): Dictionary of configuration options for parsing the CIF file.
            If None, default options are used.

    Returns:
        BioAssembly: A BioAssembly object containing the parsed data.
    '''
    with open(cif_configs, 'r') as f:
        cif_configs = json.load(f)
    types = cif_configs['types']
    cif_ID = cif_path.split("/")[-1].split(".cif")[0]

    if cif_path.endswith('.cif'):
        mmcif_dict = mmcif2dict(cif_path)
    elif cif_path.endswith('.cif.gz'):
        with gzip.open(cif_path, 'rt') as f:
            mmcif_dict = mmcif2dict(f)

    key_list = cif_configs['used']
    key_expr_list = [key[1:-1] for key in key_list if "~"==key[0]]
    full_key_list = list(mmcif_dict.keys())
    new_dict = {}
    for key in full_key_list :
        if key.split(".")[0] in key_list :
            new_dict[key] = mmcif_dict[key]
        else :
            for key_expr in key_expr_list : 
                if key_expr in key.split("_") :
                    new_dict[key] = mmcif_dict[key]
                    break

    mmcif_dict = new_dict
    mmcif_dict_keys = mmcif_dict.keys()

    # get initial deposition date
    deposition_date = mmcif_dict['_pdbx_database_status.recvd_initial_deposition_date'][0]

    # get resolution
    if '_refine.ls_d_res_high' in mmcif_dict_keys:
        resoultion_refine = mmcif_dict['_refine.ls_d_res_high'][0]
    if '_em_3d_reconstruction.resolution' in mmcif_dict_keys:
        resolution_em = mmcif_dict['_em_3d_reconstruction.resolution'][0]
    try :
        resolution_em = float(resolution_em)
    except :
        resolution_em = None
    try :
        resoultion_refine = float(resoultion_refine)
    except :
        resoultion_refine = None
    
    if resoultion_refine is None and resolution_em is None:
        resolution = None
    elif resoultion_refine is None:
        resolution = resolution_em
    elif resolution_em is None:
        resolution = resoultion_refine
    else :
        resolution = min(resoultion_refine, resolution_em) 

    # get chemical info
    _chem_comp_dict = get_smaller_mmcif_dict(mmcif_dict, mmcif_dict_keys,'_chem_comp')
    _chem_comp_atom_dict = get_smaller_mmcif_dict(mmcif_dict,mmcif_dict_keys,'_chem_comp_atom')
    _chem_comp_bond_dict = get_smaller_mmcif_dict(mmcif_dict,mmcif_dict_keys,'_chem_comp_bond')
    chem_comp_dict = merge_dict([_chem_comp_dict, _chem_comp_atom_dict, _chem_comp_bond_dict])
    chem_comp_dict = parse_chem_comp_dict(chem_comp_dict)

    # # structure parser using 
    # parser = MMCIFParser()
    # structure = parser.get_structure("test",cif_path)

    # # Access structural data
    # for model in structure:
    #     print(f"Model ID: {model.id}")
    #     for chain in model:
    #         print(f"  Chain ID: {chain.id}")
    #         for residue in chain:
    #             print(f"    Residue: {residue.resname} {residue.id}")
    #             for atom in residue:
    #                 print(f"      Atom: {atom.name} - {atom.coord}")

    _pdbx_poly_seq_scheme_dict = get_smaller_mmcif_dict(mmcif_dict,mmcif_dict_keys, '_pdbx_poly_seq_scheme', 'asym_id')
    _pdbx_nonpoly_scheme_dict = get_smaller_mmcif_dict(mmcif_dict,mmcif_dict_keys, '_pdbx_nonpoly_scheme', 'asym_id')
    _pdbx_branch_scheme_dict = get_smaller_mmcif_dict(mmcif_dict,mmcif_dict_keys, '_pdbx_branch_scheme', 'asym_id')
    scheme_dict = merge_dict([
        _pdbx_poly_seq_scheme_dict,
        _pdbx_nonpoly_scheme_dict,
        _pdbx_branch_scheme_dict,
    ])
    scheme_dict = parse_scheme_dict(scheme_dict)

    # by asym_id
    # _pdbx_struct_mod_residue_dict = get_smaller_mmcif_dict(mmcif_dict,mmcif_dict_keys,'_pdbx_struct_mod_residue','label_asym_id')
    _atom_site_dict = get_smaller_mmcif_dict(mmcif_dict,mmcif_dict_keys,'_atom_site','label_asym_id')
    _atom_site_dict, remove_UNL = remove_UNL_from_atom_site(_atom_site_dict)
    # _struct_site_gen_dict = get_smaller_mmcif_dict(mmcif_dict,mmcif_dict_keys,'_struct_site_gen','label_asym_id')
    # _struct_site_dict = get_smaller_mmcif_dict(mmcif_dict,mmcif_dict_keys,'_struct_site')

    # str_dict = merge_dict([_atom_site_dict,_pdbx_struct_mod_residue_dict,_struct_site_gen_dict])
    # str_dict = parse_str_dict(str_dict,_struct_site_dict)

    structures = {}
    for asym_id in _atom_site_dict.keys():
        structure = AsymmetricChainStructure(asym_id, _atom_site_dict[asym_id], None)
        structure.parse_structure(chem_comp_dict)
        structures[asym_id] = structure
    # structures['G']['1']['5'].get_bonds()

    # get entity seq info
    _entity_dict = get_smaller_mmcif_dict(mmcif_dict, mmcif_dict_keys, '_entity', 'id', len1_to_scalar=True)
    _entity_poly_dict = get_smaller_mmcif_dict(mmcif_dict,mmcif_dict_keys, '_entity_poly', 'entity_id', len1_to_scalar=True)
    _entity_poly_seq_dict = get_smaller_mmcif_dict(mmcif_dict,mmcif_dict_keys, '_entity_poly_seq', 'entity_id', len1_to_scalar=False)
    _pdbx_entity_nonpoly_dict = get_smaller_mmcif_dict(mmcif_dict,mmcif_dict_keys, '_pdbx_entity_nonpoly', 'entity_id', len1_to_scalar=True)
    _pdbx_entity_branch_descriptor_dict = get_smaller_mmcif_dict(mmcif_dict,mmcif_dict_keys, '_pdbx_entity_branch_descriptor', 'entity_id', len1_to_scalar=False)
    _pdbx_entity_branch_link_dict = get_smaller_mmcif_dict(mmcif_dict,mmcif_dict_keys,'_pdbx_entity_branch_link', 'entity_id', len1_to_scalar=False)
    _pdbx_entity_branch_list_dict = get_smaller_mmcif_dict(mmcif_dict,mmcif_dict_keys,'_pdbx_entity_branch_list', 'entity_id', len1_to_scalar=False)

    # _pdbx_entity_branch_list
    # _struct_ref_dict = get_smaller_mmcif_dict(mmcif_dict,mmcif_dict_keys,'_struct_ref','id')
    # _struct_ref_seq_dict = get_smaller_mmcif_dict(mmcif_dict,mmcif_dict_keys,'_struct_ref_seq','ref_id')
    # # _struct_ref_seq_dif_dict = get_smaller_mmcif_dict(mmcif_dict,mmcif_dict_keys,'_struct_ref_seq_dif','align_id')
    # struct_ref_dict = merge_dict([_struct_ref_dict,_struct_ref_seq_dict])
    # struct_ref_dict = parse_struct_ref_dict(struct_ref_dict)
    
    entity_dict = merge_dict([_entity_dict, 
                              _entity_poly_dict, 
                              _entity_poly_seq_dict,
                              _pdbx_entity_nonpoly_dict,
                              _pdbx_entity_branch_descriptor_dict,
                              _pdbx_entity_branch_link_dict,
                              _pdbx_entity_branch_list_dict,
                            #   struct_ref_dict
                              ])

    entity_dict = parse_entity_dict(entity_dict, chem_comp_dict)

    asym_chain_dict = {}
    for asym_id in structures.keys():
        asym_chain_structure = structures[asym_id]
        entity_id = asym_chain_structure.entity_id
        entity = entity_dict[entity_id]
        if asym_id in scheme_dict:
            scheme = scheme_dict[asym_id]
        else :
            if entity.get_type() == MoleculeType.WATER:
                continue
            else :
                raise Exception(f"Scheme not found for {asym_id}")
        asym_chain_dict[asym_id] = AsymmetricChain(asym_chain_structure, entity, scheme)
    # by pair of asym_id
    # _struct_conf_dict = get_smaller_mmcif_dict(mmcif_dict,mmcif_dict_keys, '_struct_conf', 'beg_label_asym_id', 'end_label_asym_id')
    _struct_conn_dict = get_smaller_mmcif_dict(mmcif_dict,mmcif_dict_keys, '_struct_conn', 'ptnr1_label_asym_id', 'ptnr2_label_asym_id')
    _struct_conn_dict = remove_UNL_from_struct_conn(_struct_conn_dict)
    # _struct_sheet_range_dict = get_smaller_mmcif_dict(mmcif_dict,mmcif_dict_keys, '_struct_sheet_range', 'beg_label_asym_id', 'end_label_asym_id')
    # _struct_mon_prot_cis_dict = get_smaller_mmcif_dict(mmcif_dict,mmcif_dict_keys, '_struct_mon_prot_cis', 'label_asym_id', 'pdbx_label_asym_id_2')
    # _ndb_struct_na_base_pair_dict = get_smaller_mmcif_dict(mmcif_dict,mmcif_dict_keys, '_ndb_struct_na_base_pair', 'i_label_asym_id', 'j_label_asym_id')

    # special_case
    # _ndb_struct_na_base_pair_step_dict = get_smaller_mmcif_dict(mmcif_dict,mmcif_dict_keys, '_ndb_struct_na_base_pair_step', 'i_label_asym_id_1', 'j_label_asym_id_1')

    # asym_pair_dict = merge_dict([
    #     _struct_conf_dict,
    #     _struct_conn_dict,
    #     _struct_sheet_range_dict,
    #     _struct_mon_prot_cis_dict,
    #     _ndb_struct_na_base_pair_dict
    # ])

    
    asym_pair_dict = merge_dict([
        # _struct_conf_dict,
        _struct_conn_dict,
        # _struct_sheet_range_dict,
        # _struct_mon_prot_cis_dict,
        # _ndb_struct_na_base_pair_dict
    ])

    # assembly
    # assembly_id_dict = get_smaller_mmcif_dict(mmcif_dict, mmcif_dict_keys, '_pdbx_struct_assembly_gen', 'assembly_id')
    assembly_id_dict = parse_assembly_dict(mmcif_dict, mmcif_dict_keys)
    _pdbx_struct_oper_dict = get_smaller_mmcif_dict(mmcif_dict,mmcif_dict_keys, '_pdbx_struct_oper_list', 'id', len1_to_scalar=True)
    _pdbx_struct_oper_dict = get_pdbx_struct_oper(_pdbx_struct_oper_dict)

    # Now we have all necessary information to create assembly
    # ChemComp idx
    chem_comp_idx_map = {chem_copm : ideal_ligand_list.index(chem_copm) for chem_copm in chem_comp_dict.keys()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bioassembly = BioAssembly(
                            cif_ID,
                            deposition_date,
                            resolution,
                            chem_comp_idx_map,
                            chem_comp_dict,
                            asym_chain_dict, 
                            asym_pair_dict, 
                            assembly_id_dict, 
                            _pdbx_struct_oper_dict, 
                            device,
                            types = types,
                            remove_signal_peptide = remove_signal_peptide,
                            )

    return bioassembly

def parse_simple_pdb(pdb_path, cif_configs=None):
    '''
    This function is used to parse a simple pdb file like fb pdb data.

    1. write ./utils/tmp/{pdb_id}.cif file
    2. parse the simple cif file
    3. remove the cif file
    '''
    parser = PDBParser()
    structure = parser.get_structure("pdb_structure", pdb_path)

    cif_path = "./utils/tmp/tmp.cif"
    io = MMCIFIO()
    io.set_structure(structure)
    io.save(cif_path)

    chem_comp_str_list = []
    for model in structure:
        for chain in model:
            for residue in chain:
                chem_comp_str_list.append(residue.get_resname())
        break

    one_letter_code = [num2AA[num2residue.index(residue)] for residue in chem_comp_str_list]
    one_letter_code = ''.join(one_letter_code)
    seq_num = len(one_letter_code)
    
    chem_comp_set = list(set(chem_comp_str_list))
    chem_comp_dict = {key : read_lmdb_value(key) for key in chem_comp_set}
    chem_comp_list = [[chem_comp_dict[chem_comp]] for chem_comp in chem_comp_str_list]
    chem_comp_str_list = [[chem_comp] for chem_comp in chem_comp_str_list]

    feature_level = FeatureLevel.ENTITY
    polymer_0D = {}
    polymer_1D = {}

    # parse sequence
    polymer_0D['polymer_type'] = Feature0D('polymer_type', PolymerType("polypeptide(L)"), feature_level, None) # protein : polypeptide(L)
    polymer_0D['auth_asym_id'] = Feature0D('auth_asym_id', 'A', feature_level, None)

    seq_one_letter_code = one_letter_code
    seq_one_letter_code_can = one_letter_code

    if len(seq_one_letter_code) != len(seq_one_letter_code_can):
        sequence_split = re.findall(r"\(.*?\)|.", seq_one_letter_code)
        if len(sequence_split) != len(seq_one_letter_code_can):
            sequence_split = [seq if '(' not in seq else "X" for seq in sequence_split]
            seq_one_letter_code_can = "".join(sequence_split)

    polymer_1D['one_letter_code'] = Feature1D('one_letter_code', seq_one_letter_code, None, feature_level, None)
    polymer_1D['one_letter_code_can'] = Feature1D('one_letter_code_can', seq_one_letter_code_can, None, feature_level, None)

    # handle heterogeneous sequence
    hetero_list = [0 for _ in range(seq_num)]
    polymer_1D['chem_comp_list'] = Feature1D('chem_comp_list', chem_comp_list, None, feature_level, None)
    polymer_1D['chem_comp_hetero'] = Feature1D('chem_comp_hetero', hetero_list, None, feature_level, None)


    polymer_0D = FeatureMap0D(polymer_0D)
    polymer_1D = FeatureMap1D(polymer_1D)

    polymer = Polymer(polymer_0D, polymer_1D, None, None)
    scheme_type = MoleculeType('polymer')
    cif_idx_list = [str(ii+1) for ii in range(seq_num)]
    scheme = Scheme('1', scheme_type, [ii+1 for ii in range(seq_num)], [str(ii+1) for ii in range(seq_num)], chem_comp_str_list, hetero_list)

    chem_comp_idx_map = {chem_copm : ideal_ligand_list.index(chem_copm) for chem_copm in chem_comp_dict.keys()}

    deposition_date = None
    resolution = None
    
    with open(cif_configs, 'r') as f:
        cif_configs = json.load(f)
    types = cif_configs['types']
    cif_ID = pdb_path.split("/")[-1].split(".cif")[0]

    if cif_path.endswith('.cif'):
        mmcif_dict = mmcif2dict(cif_path)
    elif cif_path.endswith('.cif.gz'):
        with gzip.open(cif_path, 'rt') as f:
            mmcif_dict = mmcif2dict(f)

    key_list = cif_configs['used']
    key_expr_list = [key[1:-1] for key in key_list if "~"==key[0]]
    full_key_list = list(mmcif_dict.keys())
    new_dict = {}
    for key in full_key_list :
        if key.split(".")[0] in key_list :
            new_dict[key] = mmcif_dict[key]
        else :
            for key_expr in key_expr_list : 
                if key_expr in key.split("_") :
                    new_dict[key] = mmcif_dict[key]
                    break

    mmcif_dict = new_dict
    mmcif_dict_keys = mmcif_dict.keys()

    _atom_site_dict = get_smaller_mmcif_dict(mmcif_dict,mmcif_dict_keys,'_atom_site','label_asym_id')
    _atom_site_dict, remove_UNL = remove_UNL_from_atom_site(_atom_site_dict)
    _atom_site_dict['A']['_atom_site.label_entity_id'] = ['1' for _ in range(len(_atom_site_dict['A']['_atom_site.label_entity_id']))]

    structure = AsymmetricChainStructure('A', _atom_site_dict['A'], None)
    structure.parse_structure(chem_comp_dict)
    
    asym_chain = AsymmetricChain(structure, polymer, scheme)
    asym_chain_dict = {'A' : asym_chain}
    assembly_id_dict = {'1': {'A': ['1']}}
    _pdbx_struct_oper_dict = {'1': {'matrix': torch.eye(3), 'vector': torch.zeros(3), 'type': 'identity operation', 'name': '1_555', 'symmetry_operation': 'x,y,z'}}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bioassembly = BioAssembly(
                            cif_ID,
                            deposition_date,
                            resolution,
                            chem_comp_idx_map,
                            chem_comp_dict,
                            asym_chain_dict, 
                            None, 
                            assembly_id_dict, 
                            _pdbx_struct_oper_dict, 
                            device,
                            types = types)
    os.remove(cif_path)
    return bioassembly



if __name__ == "__main__":
    # parse_cif('example_cif/1qrs.cif')
    pdb_test = "/public_data/ml/RF2_train/fb_af/pdb/ff/ff/UniRef50_W4HDV5.pdb"
    bioassembly = parse_simple_pdb(pdb_test)
    pass
