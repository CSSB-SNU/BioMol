from typing import Any
import dataclasses
from collections.abc import Sequence
import enum
import warnings
import torch
import re
import os
import pickle
import copy

from BioMol.utils.feature import (
    FeatureLevel,
    Feature1D,
    FeatureMap0D,
    FeatureMap1D,
    FeatureMapPair,
    FeatureMapContainer,
)
from BioMol.utils.error import (
    EmptyStructureError,
    AltIDError,
    StructConnAmbiguityError,
    NonpolymerError,
    EntityMismatchError,
)
from BioMol.utils.contact_graph import ContactGraph
from BioMol import (
    SEQ_TO_HASH_PATH,
    SIGNALP_PATH,
    CONTACT_GRAPH_PATH,
)
from BioMol.constant.chemical import (
    atom_mapping,
    _struct_conn_type,
    AA2num,
    DNA2num,
    RNA2num,
    NA2num,
    bond_order_map,
    num2AA,
    num2DNA,
    num2RNA,
    num2NA,
    num_to_atom,
)

@enum.unique
class MoleculeType(enum.Enum):
    POLYMER = "polymer"
    NONPOLYMER = "non-polymer"
    BRANCHED = "branched"
    WATER = "water"
    BIOASSEMBLY = "bioassembly"


@enum.unique
class PolymerType(enum.Enum):
    PROTEIN = "polypeptide(L)"
    PROTEIN_D = "polypeptide(D)"
    PNA = "peptide nucleic acid"
    RNA = "polyribonucleotide"
    DNA = "polydeoxyribonucleotide"  # TODO
    NA_HYBRID = "polydeoxyribonucleotide/polyribonucleotide hybrid"
    ETC = "etc"


molecule_type_map = {
    PolymerType.PROTEIN: '[PROTEIN]:',
    PolymerType.DNA: '[DNA]:',
    PolymerType.RNA: '[RNA]:',
    PolymerType.NA_HYBRID: '[NA_HYBRID]:',
    MoleculeType.NONPOLYMER: '[NONPOLYMER]:',
    MoleculeType.BRANCHED: '[BRANCHED]:',
}

with open(SEQ_TO_HASH_PATH, "rb") as f:
    seq_to_hash = pickle.load(f)



@dataclasses.dataclass(frozen=True)
class Scheme:
    entity_id: str
    scheme_type: MoleculeType
    cif_idx_list: Sequence[int]  # if there is no cif_idx, automatically set to 0~N-1.
    auth_idx_list: Sequence[str]  # concat of auth_seq_id and auth_ins_code
    chem_comp_list: Sequence[str] | Sequence[list[str]]  # chem_comp_id
    hetero_list: Sequence[bool] | None

    def __post_init__(self):
        self._check_length()

    def _check_length(self):
        len_auth = len(self.auth_idx_list)
        len_cif = len(self.cif_idx_list)
        len_chem_comp = len(self.chem_comp_list)
        len_hetero = len(self.hetero_list) if self.hetero_list is not None else -1
        assert all(
            l == len_auth or l == -1 for l in (len_cif, len_chem_comp, len_hetero)
        ), "Length mismatch"

    def get_auth_to_cif_map(self):
        return dict(zip(self.auth_idx_list, self.cif_idx_list))

    def get_cif_to_auth_map(self):
        return dict(zip(self.cif_idx_list, self.auth_idx_list))

    def get_auth_to_chem_comp_map(self):
        return dict(zip(self.auth_idx_list, self.chem_comp_list))

    def get_cif_to_chem_comp_map(self):
        return dict(zip(self.cif_idx_list, self.chem_comp_list))

    def auth_to_cif(self, auth_idx: str | list[str]) -> int | list[int]:
        assert self.scheme_type == MoleculeType.POLYMER, (
            "Non-polymer or branched does not have cif_idx"
        )

        auth_to_cif_map = self.get_auth_to_cif_map()
        if isinstance(auth_idx, str):
            return auth_to_cif_map[auth_idx]
        elif isinstance(auth_idx, list):
            assert all(a in self.auth_idx_list for a in auth_idx), (
                "auth_idx not found in auth_idx_list"
            )
            return [auth_to_cif_map[a] for a in auth_idx]

    def cif_to_auth(self, cif_idx: int | list[int] | slice) -> str | list[str]:
        assert self.scheme_type == MoleculeType.POLYMER, (
            "Non-polymer or branched does not have cif_idx"
        )

        cif_to_auth_map = self.get_cif_to_auth_map()

        if isinstance(cif_idx, int):
            return cif_to_auth_map[cif_idx]
        elif isinstance(cif_idx, list):
            assert all(c in self.cif_idx_list for c in cif_idx), (
                "cif_idx not found in cif_idx_list"
            )
            return [cif_to_auth_map[c] for c in cif_idx]
        elif isinstance(cif_idx, slice):
            cif_range = range(cif_idx.start or 0, cif_idx.stop, cif_idx.step or 1)
            assert all(c in self.cif_idx_list for c in cif_range), (
                "cif_idx in slice not found in cif_idx_list"
            )
            return [cif_to_auth_map[c] for c in cif_range]

    def __getitem__(
        self, idx: int | str | slice | Sequence[int] | Sequence[str]
    ) -> int | str | list[int] | list[str]:
        if isinstance(idx, int):
            return self.auth_to_cif(idx)
        elif isinstance(idx, str):
            return self.cif_to_auth(idx)
        elif isinstance(idx, slice):
            return self.cif_to_auth(idx)

    def type(self):
        return self.scheme_type

    def __len__(self):
        return len(self.auth_idx_list)

    def __repr__(self):
        length = len(self)
        if length > 6:
            if self.cif_idx_list is not None:
                cif_idx_list = self.cif_idx_list[:3] + ["..."] + self.cif_idx_list[-3:]
                cif_idx_list = [str(c) for c in cif_idx_list]
                cif_idx_list = ",".join(cif_idx_list)
            auth_idx_list = self.auth_idx_list[:3] + ["..."] + self.auth_idx_list[-3:]
            if self.chem_comp_list is not None:
                chem_comp_list = (
                    self.chem_comp_list[:3] + ["..."] + self.chem_comp_list[-3:]
                )
                if chem_comp_list[0] is list:
                    chem_comp_list = [chem_comp[0] for chem_comp in chem_comp_list]
                chem_comp_list = ",".join(chem_comp_list)
        else:
            if self.cif_idx_list is not None:
                cif_idx_list = self.cif_idx_list
                cif_idx_list = [str(c) for c in cif_idx_list]
                cif_idx_list = ",".join(cif_idx_list)
            auth_idx_list = self.auth_idx_list
            if self.chem_comp_list is not None:
                chem_comp_list = self.chem_comp_list
                assert len(chem_comp_list) != 0, "chem_comp_list is empty"
                if type(chem_comp_list[0]) is list:
                    chem_comp_list = [chem_comp[0] for chem_comp in chem_comp_list]
                chem_comp_list = ",".join(chem_comp_list)
        auth_idx_list = ",".join(auth_idx_list)
        output = "\033[1;43mScheme\033[0m(\n"
        output += f"  entity_id: {self.entity_id}\n"
        output += f"  scheme_type: {self.scheme_type}\n"
        if self.cif_idx_list is not None:
            output += f"  cif_idx_list: {cif_idx_list} ({length})\n"
        output += f"  auth_idx_list: {auth_idx_list}  ({length})\n"
        if self.chem_comp_list is not None:
            output += f"  chem_comp_list: {chem_comp_list} ({length})\n"
        output += ")"
        return output

    def __eq__(self, other):
        if not isinstance(other, Scheme):
            return False
        if self.entity_id != other.entity_id:
            return False
        if self.scheme_type != other.scheme_type:
            return False
        if self.cif_idx_list != other.cif_idx_list:
            return False
        if self.auth_idx_list != other.auth_idx_list:
            return False
        if self.chem_comp_list != other.chem_comp_list:
            return False
        return True


class ChemComp(FeatureMapContainer):
    def __init__(
        self,
        feature_map_0D: FeatureMap0D | None,
        feature_map_1D: FeatureMap1D | None,
        feature_map_pair: FeatureMapPair | None,
        help: Any = None,
    ):
        super().__init__(feature_map_0D, feature_map_1D, feature_map_pair)

    def get_level(self):
        return FeatureLevel.CHEMCOMP

    def get_name(self) -> str:
        return self.feature_map_0D["name"].value

    def get_code(self) -> str:
        return self.feature_map_0D["id"].value

    def get_atoms(self, one_letter=False) -> Sequence[str]:
        if self.get_code() == "UNL":
            return []
        if one_letter:
            return self.feature_map_1D["one_letter_atoms"].value
        return self.feature_map_1D["full_atoms"].value

    def get_representative_atom(self, including_hydrogen=False) -> tuple[str, int]:
        if self.get_code() == "UNL":
            return None, None
        atoms = self.get_atoms(one_letter=False)
        one_letter_atoms = self.get_atoms(one_letter=True)
        if not including_hydrogen:
            mask = [atom != "H" and atom != "D" for atom in one_letter_atoms]
        else:
            mask = [True for atom in atoms]
        masked_atoms = [atom for atom, mask in zip(atoms, mask) if mask]
        if "CA" in atoms:
            return "CA", masked_atoms.index("CA")
        # if not found, return first carbon atom
        masked_one_letter_atoms = [
            atom for atom, mask in zip(one_letter_atoms, mask) if mask
        ]
        for idx, atom in enumerate(masked_atoms):
            if masked_one_letter_atoms[idx] == "C":
                return atom, idx
        # if not found, return first atom which is not hydrogen
        if len(masked_atoms) > 0:
            return masked_atoms[0], 0
        # if not found, return first atom
        return atoms[0], 0

    def get_bonds(self, wo_hydrogen=True):
        # build bond tuple (bond_num, head); \
        # head = (atom_idx1, atom_idx2, bond_type, aromatic, stereo)
        if self.get_code() == "UNL":
            return None
        bonds = []
        bond_type_map = {
            "SING": 0,
            "DOUB": 1,
            "TRIP": 2,
            "sing": 0,
            "doub": 1,
            "trip": 2,
        }
        if len(self.feature_map_pair.keys()) == 0:
            return None

        hydrogen_mask = [
            atom != "H" and atom != "D" for atom in self.get_atoms(one_letter=True)
        ]
        not_hydrogen_idx = [idx for idx, mask in enumerate(hydrogen_mask) if mask]
        not_hydrogen_idx_map = {idx: i for i, idx in enumerate(not_hydrogen_idx)}

        bond_type = self.feature_map_pair["bond_type"]
        aromatic = self.feature_map_pair["aromatic"]
        stereo = self.feature_map_pair["stereo"]
        for key in bond_type.value:
            atom_idx1, atom_idx2 = key
            if wo_hydrogen:
                if (
                    atom_idx1 not in not_hydrogen_idx
                    or atom_idx2 not in not_hydrogen_idx
                ):
                    continue
                atom_idx1, atom_idx2 = (
                    not_hydrogen_idx_map[atom_idx1],
                    not_hydrogen_idx_map[atom_idx2],
                )
            bonds.append(
                (
                    atom_idx1,
                    atom_idx2,
                    bond_type_map[bond_type[key]],
                    aromatic[key],
                    stereo[key],
                )
            )
        return bonds

    def get_ideal_coords(self, remove_hydrogen=True):
        if self.get_code() == "UNL":
            return None
        ideal_coords = self.feature_map_1D["ideal_coords"].value
        mask = self.feature_map_1D["ideal_coords"].mask
        atoms = self.get_atoms(one_letter=True)
        atoms = [atom_mapping[atom] for atom in atoms]
        atoms = torch.tensor(atoms)
        # output : atom, x, y, z, mask
        output = torch.cat([atoms.unsqueeze(1), ideal_coords, mask.unsqueeze(1)], dim=1)
        if remove_hydrogen:
            hydrogen_mask = [
                atom != "H" and atom != "D" for atom in self.get_atoms(one_letter=True)
            ]
            output = output[hydrogen_mask]
        return output  # (N, 5) | atom, x, y, z, mask

    def get_charges(self, remove_hydrogen=True):
        if self.get_code() == "UNL":
            return None
        charges = self.feature_map_1D["charges"].value
        mask = self.feature_map_1D["charges"].mask
        atoms = self.get_atoms(one_letter=True)
        atoms = [atom_mapping[atom] for atom in atoms]
        atoms = torch.tensor(atoms)
        # output : atom, charge, mask
        output = torch.cat(
            [atoms.unsqueeze(1), charges.unsqueeze(1), mask.unsqueeze(1)], dim=1
        )
        if remove_hydrogen:
            hydrogen_mask = [
                atom != "H" and atom != "D" for atom in self.get_atoms(one_letter=True)
            ]
            output = output[hydrogen_mask]
        return output  # (N, 3) | atom, charge, mask

    def __repr__(self):
        output = "\033[1;43mChemComp\033[0m(\n"
        output += f"  name: {self.get_name()}\n"
        output += f"  code: {self.get_code()}\n"
        output += f"  formula: {self.feature_map_0D['formula'].value}\n"
        output += f"  atoms: {self.get_atoms()}\n"
        output += ")"
        return output

    def __str__(self):
        return self.get_code()


class ChemCompView:
    def __init__(self, chem_comp_dict, chem_comp_list):
        self.chem_comp_dict = chem_comp_dict
        self.chem_comp_list = chem_comp_list

    def __getitem__(self, index):
        id = self.chem_comp_list[index]
        if isinstance(id, str):
            return self.chem_comp_dict[id]
        elif isinstance(id, list):
            return [self.chem_comp_dict[i] for i in id]

    def __len__(self):
        return len(self.chem_comp_list)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __repr__(self):
        length = len(self)
        if length > 6:
            chem_comp_list = self.chem_comp_list[:3] + ["..."] + self.chem_comp_list[-3:]
        else:
            chem_comp_list = self.chem_comp_list
        chem_comp_list = ",".join(chem_comp_list)
        output = "\033[1;43mChemCompView\033[0m(\n"
        output += f"  chem_comp_list: {chem_comp_list} ({length})\n"
        output += ")"
        return output


class Polymer(FeatureMapContainer):
    def __init__(
        self,
        feature_map_0D: FeatureMap0D | None,
        feature_map_1D: FeatureMap1D | None,
        feature_map_pair: FeatureMapPair | None,
        help: Any = None,
    ):
        super().__init__(feature_map_0D, feature_map_1D, feature_map_pair)
        self.chem_comp_list = self.feature_map_1D["chem_comp_list"].value

    def get_level(self):
        return FeatureLevel.ENTITY

    def get_type(self):
        return MoleculeType.POLYMER

    def get_polymer_type(self):
        return self.feature_map_0D["polymer_type"].value

    def get_chem_comp_list(self) -> Sequence[str]:
        return self.chem_comp_list

    def get_one_letter_code(self, canonical=False) -> str:
        if canonical:
            return self.feature_map_1D["one_letter_code_can"].value
        else:
            return self.feature_map_1D["one_letter_code"].value

    def parse(self, scheme):
        chem_comp_list = self.feature_map_1D["chem_comp_list"].value

        scheme_chem_comp_list = scheme.chem_comp_list
        new_chem_comp_list = []
        for inner_list, scheme_chem_comp in zip(chem_comp_list, scheme_chem_comp_list):
            for chem_comp in inner_list:
                if chem_comp.get_code() == scheme_chem_comp:
                    new_chem_comp_list.append(chem_comp)
                    break
        chem_comp_list = new_chem_comp_list
        self.hetero_determined = True
        self.chem_comp_list = chem_comp_list

    def crop(self, mask: list[int] | torch.Tensor):
        if isinstance(mask, torch.Tensor):
            mask = mask.tolist()
        self.chem_comp_list = [
            self.chem_comp_list[ii] for ii, mm in enumerate(mask) if mm is True
        ]
        feature_level = FeatureLevel.ENTITY

        cropped_one_letter_code = "".join(
            [
                self.feature_map_1D["one_letter_code"].value[ii]
                for ii, mm in enumerate(mask)
                if mm is True
            ]
        )
        cropped_one_letter_code_can = "".join(
            [
                self.feature_map_1D["one_letter_code_can"].value[ii]
                for ii, mm in enumerate(mask)
                if mm is True
            ]
        )
        cropped_chem_comp_list = [
            self.feature_map_1D["chem_comp_list"].value[ii]
            for ii, mm in enumerate(mask)
            if mm is True
        ]
        cropped_chem_comp_hetero = [
            self.feature_map_1D["chem_comp_hetero"].value[ii]
            for ii, mm in enumerate(mask)
            if mm is True
        ]

        polymer_1D = {}
        polymer_1D["one_letter_code"] = Feature1D(
            "one_letter_code", cropped_one_letter_code, None, feature_level, None
        )
        polymer_1D["one_letter_code_can"] = Feature1D(
            "one_letter_code_can", cropped_one_letter_code_can, None, feature_level, None
        )

        # handle heterogeneous sequence
        polymer_1D["chem_comp_list"] = Feature1D(
            "chem_comp_list", cropped_chem_comp_list, None, feature_level, None
        )
        polymer_1D["chem_comp_hetero"] = Feature1D(
            "chem_comp_hetero", cropped_chem_comp_hetero, None, feature_level, None
        )
        polymer_1D = FeatureMap1D(polymer_1D)

        self.feature_map_1D = polymer_1D

    def __len__(self):
        return len(self.get_chem_comp_list())

    def __repr__(self):
        one_letter_code = self.get_one_letter_code()
        chem_comp_list = self.get_chem_comp_list()
        if type(chem_comp_list[0]) is list:
            chem_comp_list = [chem_comp[0] for chem_comp in chem_comp_list]
        else:
            chem_comp_list = [chem_comp.get_code() for chem_comp in chem_comp_list]
        length = len(chem_comp_list)

        if len(chem_comp_list) > 6:
            one_letter_code = "".join(re.findall(r"\([^)]*\)|.", one_letter_code))
            one_letter_code = one_letter_code[:3] + "..." + one_letter_code[-3:]
            chem_comp_list = (
                ",".join(chem_comp_list[:3]) + "..." + ",".join(chem_comp_list[-3:])
            )
        else:
            chem_comp_list = ",".join(chem_comp_list)
        output = "\033[42mPolymer\033[0m(\n"
        output += f"  polymer_type: {self.get_polymer_type().value}\n"
        output += f"  one_letter_code: {one_letter_code} ({length})\n"
        output += f"  chem_comp_list: {chem_comp_list} ({length})\n"
        output += ")"
        return output


class NonPolymer(FeatureMapContainer):
    def __init__(
        self,
        feature_map_0D: FeatureMap0D | None,
        chem_comp: ChemComp,
        is_water: bool = False,
    ):
        super().__init__(feature_map_0D, None, None)
        self.load_chem_comp(chem_comp)
        self.is_water = is_water

    def load_chem_comp(self, chem_comp: ChemComp):
        assert self.get_chem_comp() == chem_comp.get_code(), (
            "ChemComp code does not match"
        )
        self.chem_comp = chem_comp

    def get_level(self):
        return FeatureLevel.ENTITY

    def get_type(self):
        if self.is_water:
            return MoleculeType.WATER
        return MoleculeType.NONPOLYMER

    def get_name(self):
        return self.feature_map_0D["name"].value

    def get_chem_comp(self) -> str:
        return self.feature_map_0D["chem_comp"].value

    def get_atoms(self) -> Sequence[str]:
        return self.chem_comp.get_atoms()

    def get_bonds(self):
        return self.chem_comp.get_bonds()

    def __len__(self):
        return 1

    def __repr__(self):
        output = "\033[44mNonPolymer\033[0m(\n"
        output += f"  name: {self.get_name()}\n"
        output += f"  chem_comp: {self.get_chem_comp()}\n"
        output += f"  atoms: {self.get_atoms()}\n"
        output += ")"
        return output


class Branched(FeatureMapContainer):
    def __init__(
        self,
        feature_map_0D: FeatureMap0D | None,
        feature_map_1D: FeatureMap1D | None,
        feature_map_pair: FeatureMapPair | None,
        help: Any = None,
    ):
        super().__init__(feature_map_0D, feature_map_1D, feature_map_pair)
        self.hetero_determined = False
        self.chem_comp_list = self.feature_map_1D["chem_comp_list"].value

    def get_level(self):
        return FeatureLevel.ENTITY

    def get_type(self):
        return MoleculeType.BRANCHED

    def parse(self, scheme):
        chem_comp_list = self.feature_map_1D["chem_comp_list"].value
        chem_comp_num = self.feature_map_1D["chem_comp_num"].value
        branch_link = self.feature_map_pair["branch_link"].value
        atom_list = []

        scheme_chem_comp_list = scheme.chem_comp_list
        new_chem_comp_list = []
        for inner_list, scheme_chem_comp in zip(chem_comp_list, scheme_chem_comp_list):
            for chem_comp in inner_list:
                if chem_comp.get_code() == scheme_chem_comp:
                    new_chem_comp_list.append(chem_comp)
                    break
        chem_comp_list = new_chem_comp_list
        self.hetero_determined = True

        chem_comp_to_cum_sum = {}
        chem_comp_atom_dict = {}
        cum_sum = 0
        for ii, chem_comp in enumerate(chem_comp_list):
            _hydrogen_mask = [
                atom != "H" or atom != "D"
                for atom in chem_comp.get_atoms(one_letter=True)
            ]
            atom_list = [
                atom for atom, mask in zip(chem_comp.get_atoms(), _hydrogen_mask) if mask
            ]
            to_add = len(atom_list)
            chem_comp_to_cum_sum[ii] = cum_sum
            chem_comp_atom_dict[chem_comp.get_code()] = atom_list
            cum_sum += to_add

        # comp_id1, copm_id2, atom_id1, atom_id2, bond_type
        chem_comp_level_bond = []
        atom_level_bond = []
        for branch_key, branch_item in branch_link.items():
            comp_id1, comp_id2 = branch_key
            comp_idx1, comp_idx2 = (
                chem_comp_num.index(comp_id1),
                chem_comp_num.index(comp_id2),
            )
            chem_comp1, chem_comp2 = chem_comp_list[comp_idx1], chem_comp_list[comp_idx2]
            atom_id1, atom_id2 = branch_item["atom_id"]
            atom_idx1, atom_idx2 = (
                chem_comp_atom_dict[chem_comp1.get_code()].index(atom_id1),
                chem_comp_atom_dict[chem_comp2.get_code()].index(atom_id2),
            )
            atom_idx1 += chem_comp_to_cum_sum[comp_idx1]
            atom_idx2 += chem_comp_to_cum_sum[comp_idx2]
            comp_idx1, comp_idx2 = min(comp_idx1, comp_idx2), max(comp_idx1, comp_idx2)
            chem_comp_level_bond.append(
                [comp_idx1, comp_idx2, 2]
            )  # see _struct_conn_type

            bond_type = branch_item["bond"]
            bond_type = bond_order_map[bond_type]
            atom_idx1, atom_idx2 = min(atom_idx1, atom_idx2), max(atom_idx1, atom_idx2)
            atom_level_bond.append((atom_idx1, atom_idx2, bond_type, 0, 0, 2))
            # aromatic, stereo, _struct_conn_type
            # I assume that aromatic and stereo are always 0 for branched

        self.chem_comp_list = chem_comp_list
        self.chem_comp_level_bond = chem_comp_level_bond
        self.atom_level_bond = atom_level_bond

    def get_bonds(self, level: str = "atom"):
        assert self.hetero_determined, "Hetero atoms are not determined"
        if level == "atom":
            return self.atom_level_bond
        elif level == "chem_comp" or level == "residue":
            return self.chem_comp_level_bond
        else:
            raise ValueError("level should be 'atom' or 'chem_comp' or 'residue'")

    def find_connected(self, idx: int):
        assert self.hetero_determined, "Hetero atoms are not determined"
        connected = []
        for bond in self.chem_comp_level_bond:
            if int(bond[0]) == idx:
                connected.append(
                    f"{self.chem_comp_list[int(bond[1])].get_code()}.{bond[1] + 1}",
                )
            elif int(bond[1]) == idx:
                connected.append(
                    f"{self.chem_comp_list[int(bond[0])].get_code()}.{bond[0] + 1}",
                )
        return connected

    def __repr__(self):
        output = "\033[45mBranched\033[0m(\n"
        for idx, chem_comp in enumerate(self.chem_comp_list):
            if self.hetero_determined:
                connected = self.find_connected(idx)
                connected = " | ".join(connected)
                output += f"  {chem_comp.get_code()}.{idx + 1} || {connected}\n"
            else:
                output += f"  {chem_comp.get_code()}.{idx + 1}\n"
        output += ")"
        return output

    def get_chem_comp_list(self) -> Sequence[str]:
        return self.chem_comp_list

    def __len__(self):
        return len(self.get_chem_comp_list())


class Water:
    pass


@dataclasses.dataclass
class Atom:
    atom_name: str
    type_symbol: str
    label_alt_id: str
    x: float
    y: float
    z: float
    occupancy: float
    b_factor: float

    def __repr__(self):
        return f"\033[47mAtom\033[0m({self.atom_name}({self.type_symbol}) {self.label_alt_id} \
            | {self.x}, {self.y}, {self.z}, {self.occupancy}, {self.b_factor})"  # noqa: E501

    def __eq__(self, other):
        if not isinstance(other, Atom):
            return False
        if self.atom_name != other.atom_name:
            return False
        if self.type_symbol != other.type_symbol:
            return False
        # Allow label_alt_id to be different
        # if self.label_alt_id != other.label_alt_id:
        #     return False
        if self.x != other.x:
            return False
        if self.y != other.y:
            return False
        if self.z != other.z:
            return False
        if self.occupancy != other.occupancy:
            return False
        if self.b_factor != other.b_factor:
            return False
        return True


@dataclasses.dataclass
class Residue:
    atoms: dict[
        int, Atom | list[Atom]
    ]  # key : chem_comp_atom_idx, value : Atom or list[Atom]
    label_seq_id: int
    auth_seq_id: int
    auth_ins_code: str | int | None
    chem_comp: ChemComp

    def add_atom(self, atom: Atom):
        if self.chem_comp.get_code() == "UNL":
            if len(self.atoms) == 0:
                self.atoms[0] = atom
            else:
                atom_idx = max(self.atoms.keys()) + 1
                if self.atoms.get(atom_idx) is not None:
                    self.has_alternative_conformation = True
                    if isinstance(self.atoms[atom_idx], list):
                        self.atoms[atom_idx].append(atom)
                    else:
                        self.atoms[atom_idx] = [self.atoms[atom_idx], atom]
            return

        if atom.atom_name not in self.chem_comp.get_atoms():
            # simply do not add atom
            return
        atom_idx = self.chem_comp.get_atoms().index(atom.atom_name)
        if self.atoms.get(atom_idx) is not None:
            self.has_alternative_conformation = True
            if isinstance(self.atoms[atom_idx], list):
                self.atoms[atom_idx].append(atom)
            else:
                self.atoms[atom_idx] = [self.atoms[atom_idx], atom]
        else:
            self.atoms[atom_idx] = atom

    def get_atoms(self):
        atom_list = []
        for idx in range(len(self.chem_comp.get_atoms())):
            if self.atoms.get(idx) is not None:
                atom_list.append(self.atoms[idx])
            else:
                atom_list.append(None)
        return atom_list

    def get_bonds(self):  # intra-residue bonds
        chem_comp_bonds = self.chem_comp.get_bonds()
        intra_residue_bonds = []
        for bond in chem_comp_bonds:
            chem_comp_atom_idx1, chem_comp_atom_idx2, bond_type, aromatic, stereo = list(
                bond
            )
            intra_residue_bonds.append(
                (chem_comp_atom_idx1, chem_comp_atom_idx2, bond_type, aromatic, stereo)
            )
        intra_residue_bonds = torch.tensor(intra_residue_bonds)
        return intra_residue_bonds

    def __repr__(self):
        output = "\033[47mResidue\033[0m(\n"
        output += f"  label_seq_id: {self.label_seq_id}\n"
        output += (
            f"  auth_seq_id/auth_ins_code: {self.auth_seq_id}/{self.auth_ins_code}\n"
        )
        output += f"  chem_comp: {self.chem_comp.get_code()}\n"
        output += "  atoms:\n"
        atoms = self.get_atoms()
        if len(atoms) > 6:
            atoms = atoms[:3] + ["..."] + atoms[-3:]
        else:
            atoms = atoms
        for atom in atoms:
            output += f"    {atom}\n"
        output += ")"
        return output


class AsymmetricChainStructure:
    def __init__(
        self,
        asym_chain_id: str,
        _atom_site_dict: dict[str, list[Any]],
        additional_info: Any = None,
    ):
        self.asym_chain_id = asym_chain_id
        self.entity_id = list(set(_atom_site_dict["_atom_site.label_entity_id"]))
        if len(self.entity_id) != 1:
            raise EntityMismatchError(
                f"Multiple entity_id found in asym_chain_id {asym_chain_id}"
            )
        self.entity_id = self.entity_id[0]
        self._atom_site_dict = _atom_site_dict
        self.additional_info = additional_info

        self.is_model_chosen = False
        self.is_structure_parsed = False

    def parse_structure(self, chem_comp_dict: dict[str, ChemComp]):
        _parsed_structure = {}

        model_id_list = self._atom_site_dict["_atom_site.pdbx_PDB_model_num"]
        label_alt_id_list = self._atom_site_dict["_atom_site.label_alt_id"]
        label_seq_id_list = self._atom_site_dict["_atom_site.label_seq_id"]
        auth_seq_id_list = self._atom_site_dict["_atom_site.auth_seq_id"]
        auth_ins_code_list = self._atom_site_dict["_atom_site.pdbx_PDB_ins_code"]

        coord_x_list = self._atom_site_dict["_atom_site.Cartn_x"]
        coord_y_list = self._atom_site_dict["_atom_site.Cartn_y"]
        coord_z_list = self._atom_site_dict["_atom_site.Cartn_z"]
        coord_list = list(zip(coord_x_list, coord_y_list, coord_z_list))
        occup_list = self._atom_site_dict["_atom_site.occupancy"]
        b_factor_list = self._atom_site_dict["_atom_site.B_iso_or_equiv"]
        atom_list = self._atom_site_dict["_atom_site.label_atom_id"]
        type_symbol_list = self._atom_site_dict["_atom_site.type_symbol"]
        chem_comp_list = self._atom_site_dict["_atom_site.label_comp_id"]

        atom_site_length = len(model_id_list)

        model_to_alt_id_list = {}

        # before parsing xyz, check if there is alternative conformation
        for idx in range(atom_site_length):
            model_id = model_id_list[idx]
            label_alt_id = label_alt_id_list[idx]
            if model_to_alt_id_list.get(model_id) is None:
                model_to_alt_id_list[model_id] = []

            if label_alt_id not in model_to_alt_id_list[model_id]:
                model_to_alt_id_list[model_id].append(label_alt_id)

        for model_id in model_to_alt_id_list.keys():
            alt_id_list = model_to_alt_id_list[model_id]
            if _parsed_structure.get(model_id) is None:
                _parsed_structure[model_id] = {}
            if len(alt_id_list) == 0:
                _parsed_structure[model_id]["."] = {}
            else:
                for alt_id in alt_id_list:
                    _parsed_structure[model_id][alt_id] = {}

        for idx in range(atom_site_length):
            model_id = model_id_list[idx]
            label_alt_id = label_alt_id_list[idx]
            label_seq_id = label_seq_id_list[idx]
            auth_seq_id = auth_seq_id_list[idx]
            auth_ins_code = auth_ins_code_list[idx]

            x, y, z = coord_list[idx]
            occup = occup_list[idx]
            b_factor = b_factor_list[idx]
            atom_name = atom_list[idx]
            type_symbol = type_symbol_list[idx]
            atom = Atom(atom_name, type_symbol, label_alt_id, x, y, z, occup, b_factor)

            chem_comp = chem_comp_list[idx]
            chem_comp = chem_comp_dict[chem_comp]

            if auth_ins_code != "?":
                sequence_key = f"{auth_seq_id}.{auth_ins_code}"
            else:
                sequence_key = f"{auth_seq_id}"

            alt_id_list = model_to_alt_id_list[model_id]

            # if has_alternative_conformation:
            #     structure_key = f"{model_id}_{label_alt_id}"
            # else:
            #     structure_key = f"{model_id}"

            if label_alt_id == "." and len(alt_id_list) != 0:
                for alt_id in alt_id_list:
                    if _parsed_structure[model_id][alt_id].get(sequence_key) is None:
                        _parsed_structure[model_id][alt_id][sequence_key] = Residue(
                            atoms={},
                            label_seq_id=label_seq_id,
                            auth_seq_id=auth_seq_id,
                            auth_ins_code=auth_ins_code,
                            chem_comp=chem_comp,
                        )
                    _parsed_structure[model_id][alt_id][sequence_key].add_atom(atom)

            else:
                if _parsed_structure[model_id][label_alt_id].get(sequence_key) is None:
                    _parsed_structure[model_id][label_alt_id][sequence_key] = Residue(
                        atoms={},
                        label_seq_id=label_seq_id,
                        auth_seq_id=auth_seq_id,
                        auth_ins_code=auth_ins_code,
                        chem_comp=chem_comp,
                    )
                _parsed_structure[model_id][label_alt_id][sequence_key].add_atom(atom)

        self.models = _parsed_structure
        self.is_structure_parsed = True
        self.model_to_alt_id_list = model_to_alt_id_list

    def get_model_ids(self):
        if self.is_structure_parsed is False:
            raise ValueError("Structure is not parsed")
        return list(self.models.keys())

    def __getitem__(self, idx):
        if self.is_structure_parsed is False:
            raise ValueError("Structure is not parsed")
        return self.models[idx]

    def keys(self):
        return self.models.keys()

    def get_chem_comp_dict(self, model_id: str = None, alt_id: str = None):
        chem_comp_dict = {}
        for _model_id in self.models.keys():
            chem_comp_dict[_model_id] = {}
            for alt_id in self.models[_model_id].keys():
                chem_comp_dict[_model_id][alt_id] = {}
                for auth_seq_id, residue in self.models[_model_id][alt_id].items():
                    chem_comp_dict[_model_id][alt_id][auth_seq_id] = (
                        residue.chem_comp.get_code()
                    )
        if model_id is not None:
            assert alt_id is not None, "alt_id should be provided"
            return chem_comp_dict[model_id][alt_id]
        return chem_comp_dict

    def __repr__(self):
        # only model_id 1 or first model is shown
        model_id_list = list(self.models.keys())
        structure = self.models[model_id_list[0]]
        alt_id_list = list(structure.keys())
        first_alt_id = alt_id_list[0]
        structure = structure[first_alt_id]

        residue_key_list = list(structure.keys())
        if len(residue_key_list) > 2:
            residue_list = [
                structure[residue_key_list[0]],
                structure[residue_key_list[1]],
                "...",
            ]
        else:
            residue_list = [
                structure[residue_key_list[ii]] for ii in range(len(residue_key_list))
            ]
        output = "\033[1;43mAsymmetricChainStructure\033[40m(\n"
        output += f"  asym_chain_id: {self.asym_chain_id}\n"
        output += f"  entity_id: {self.entity_id}\n"
        output += f"  model_id: {model_id_list} | alt_id : {alt_id_list} \
                      (below is the first model of {first_alt_id} alt id)\n"
        output += f"  residues: {residue_list}\n"
        output += ")"
        return output


class AsymmetricChain:
    """
    scheme
    structure
    ...etc
    """

    def __init__(
        self,
        asym_chain_structure: AsymmetricChainStructure,
        entity: Polymer | NonPolymer | Branched | Water,
        scheme: Scheme,
    ):
        self.structure = asym_chain_structure
        self.entity = entity
        self.asym_chain_id = asym_chain_structure.asym_chain_id
        self.entity_id = asym_chain_structure.entity_id
        self.scheme = scheme
        self.auth_to_cif_map = scheme.get_auth_to_cif_map()
        self.cif_to_auth_map = scheme.get_cif_to_auth_map()
        self._check_entity_scheme_match()
        self._check_structure_scheme_match()
        self.scheme_dict = self._parse_scheme()

    def _parse_scheme(self):
        scheme_dict = {}
        entity_id = self.scheme.entity_id
        scheme_type = self.scheme.scheme_type
        cif_idx_list = self.scheme.cif_idx_list
        auth_idx_list = self.scheme.auth_idx_list
        hetero_list = None

        for model_id in self.structure.keys():
            scheme_dict[model_id] = {}
            for alt_id in self.structure[model_id].keys():
                scheme_dict[model_id][alt_id] = {}
                scheme_auth_to_chem_comp_map = self.scheme.get_auth_to_chem_comp_map()
                model_residue_dict = {
                    key: value.chem_comp.get_code()
                    for key, value in self.structure[model_id][alt_id].items()
                }
                chem_comp_list = []
                for auth_idx in auth_idx_list:
                    scheme_chem_comp_list = scheme_auth_to_chem_comp_map[auth_idx]
                    if auth_idx in model_residue_dict.keys():
                        model_chem_comp = model_residue_dict[auth_idx]
                        assert model_chem_comp in scheme_chem_comp_list, (
                            "Scheme and model chem_comp mismatch"
                        )
                        chem_comp_list.append(model_chem_comp)
                    else:
                        if len(scheme_chem_comp_list) == 1:
                            chem_comp_list.append(scheme_chem_comp_list[0])
                        else:
                            # Warning :
                            # This case indicates that there is missing & hetero residue
                            # in the structure. I just add first chem_comp in the list.
                            warnings.warn(
                                f"Missing&hetero residue found. auth_seq_id: {auth_idx}",
                                stacklevel=2,
                            )
                            chem_comp_list.append(scheme_chem_comp_list[0])
                scheme = Scheme(
                    entity_id,
                    scheme_type,
                    cif_idx_list,
                    auth_idx_list,
                    chem_comp_list,
                    hetero_list,
                )
                scheme_dict[model_id][alt_id] = scheme
        return scheme_dict

    def _check_entity_id(self):
        if self.entity.get_type() == MoleculeType.WATER:
            return
        assert (
            self.entity_id == self.scheme.entity_id
            and self.entity_id == self.structure.entity_id
        ), "Entity ID does not match"

    def _check_type(self):
        scheme_type = self.scheme.scheme_type
        entity_type = self.entity.get_type()
        if scheme_type != entity_type:
            if scheme_type == MoleculeType.WATER or entity_type == MoleculeType.WATER:
                self.type = MoleculeType.WATER
                return
            raise ValueError("Scheme type and entity type does not match")
        self.type = scheme_type

    def _check_length(self):
        # if water, no need to check
        if self.type == MoleculeType.WATER:
            return
        scheme_length = len(self.scheme)
        if self.type == MoleculeType.NONPOLYMER:
            if (
                self.entity.chem_comp.get_code() == "HOH"
                or self.entity.chem_comp.get_code() == "DOD"
            ):
                self.length = scheme_length
                return
            if scheme_length != 1:
                raise ValueError("Non-polymer should have only one chem_comp")
        else:
            entity_length = len(self.entity)
            if scheme_length != entity_length:
                raise ValueError("Scheme and entity length mismatch")
        self.length = scheme_length

    def _check_chem_comp_list(self):
        if self.type == MoleculeType.POLYMER or self.type == MoleculeType.BRANCHED:
            scheme_chem_comp_list = self.scheme.chem_comp_list
            entity_chem_comp_list = self.entity.get_chem_comp_list()
            entity_chem_comp_list = [
                [chem_comp.get_code() for chem_comp in inner_list]
                for inner_list in entity_chem_comp_list
            ]
            if scheme_chem_comp_list != entity_chem_comp_list:
                # warning
                warnings.warn("Scheme and entity chem_comp_list mismatch", stacklevel=2)
                for inner_scheme_list, inner_entity_list in zip(
                    scheme_chem_comp_list, entity_chem_comp_list
                ):
                    if not set(inner_scheme_list).issubset(set(inner_entity_list)):
                        raise ValueError("Scheme and entity chem_comp_list mismatch")
            self.chem_comp_list = scheme_chem_comp_list

        elif self.type == MoleculeType.NONPOLYMER or self.type == MoleculeType.WATER:
            scheme_chem_comp = self.scheme.chem_comp_list[0][0]
            entity_chem_comp = self.entity.get_chem_comp()
            if scheme_chem_comp != entity_chem_comp:
                raise ValueError("Scheme and entity chem_comp mismatch")
            self.chem_comp_list = [scheme_chem_comp]
        else:
            raise ValueError("Unknown molecule type, is it water?")

    def _check_entity_scheme_match(self):
        self._check_entity_id()
        self._check_type()
        self._check_length()
        self._check_chem_comp_list()

    def _check_structure_scheme_match(self):
        # if water, no need to check
        if self.type == MoleculeType.WATER:
            return
        # get {auth_seq_id: chem_comp code} from structure
        auth_seq_id_to_chem_comp_in_structure = self.structure.get_chem_comp_dict()
        auth_seq_id_to_chem_comp_in_scheme = self.scheme.get_auth_to_chem_comp_map()

        # all auth_seq_id map in structure should be subset of scheme
        for model_id in auth_seq_id_to_chem_comp_in_structure.keys():
            for alt_id in auth_seq_id_to_chem_comp_in_structure[model_id].keys():
                for auth_seq_id in auth_seq_id_to_chem_comp_in_structure[model_id][
                    alt_id
                ].keys():
                    if auth_seq_id not in auth_seq_id_to_chem_comp_in_scheme.keys():
                        if self.type == MoleculeType.NONPOLYMER:
                            raise NonpolymerError(
                                f"auth_seq_id {auth_seq_id} in structure not found in scheme"  # noqa: E501
                            )
                        raise ValueError("auth_seq_id in structure not found in scheme")
                    if (
                        auth_seq_id_to_chem_comp_in_structure[model_id][alt_id][
                            auth_seq_id
                        ]
                        not in auth_seq_id_to_chem_comp_in_scheme[auth_seq_id]
                    ):
                        if self.type == MoleculeType.NONPOLYMER:
                            raise NonpolymerError(
                                f"auth_seq_id {auth_seq_id} in structure not found in scheme"  # noqa: E501
                            )
                        raise ValueError("auth_seq_id to chem_comp code mismatch")
        self.auth_seq_id_to_chem_comp = auth_seq_id_to_chem_comp_in_scheme
        self.auth_seq_id_to_cif_idx = self.scheme.get_auth_to_cif_map()
        self.cif_idx_to_auth_seq_id = self.scheme.get_cif_to_auth_map()

    def __len__(self):
        return self.length

    def get_missing_atoms(self):
        pass

    def get_sequence(self, canonical: bool = True):
        # Warning : I don't recommend using this function because it is only for polymer,
        #           not for branched or non-polymer.
        # Use get_chem_comp_list() instead.
        entity_type = self.entity.get_type()
        if entity_type == MoleculeType.POLYMER:
            canonical_sequence = self.entity.get_one_letter_code(canonical=True)
            non_canonical_sequence = self.entity.get_one_letter_code(canonical=False)

            match self.entity.get_polymer_type():
                case PolymerType.PROTEIN:
                    one_letter_map = num2AA
                case PolymerType.PROTEIN_D:
                    one_letter_map = num2AA
                case PolymerType.DNA:
                    one_letter_map = num2DNA
                case PolymerType.RNA:
                    one_letter_map = num2RNA
                case PolymerType.NA_HYBRID:
                    one_letter_map = num2NA
                case PolymerType.PNA:
                    one_letter_map = num2NA
                case _:
                    raise ValueError("Unknown polymer type")

            if canonical:
                sequence_split = re.findall(r"\(.*?\)|.", non_canonical_sequence)
                if len(canonical_sequence) == len(sequence_split):
                    return canonical_sequence
                else:
                    # in this case some reisues are merged into single chem_comp so that
                    # canonical and non-canonical sequence length mismatch
                    # for consistency, I changed it into single unknown residue 'X'
                    sequence_split = [
                        seq[1:-1] if seq[0] == "(" and seq[-1] == ")" else seq
                        for seq in sequence_split
                    ]
                    sequence_split = [
                        seq if seq in one_letter_map else "X" for seq in sequence_split
                    ]
                    return "".join(sequence_split)
            else:
                return non_canonical_sequence
        else:
            raise ValueError(
                f"get_full_sequence() is only for polymer, not for {entity_type}"
            )

    def get_missing_residues(self, model_id: str, alt_id: str):
        if model_id not in self.structure.keys():
            raise ValueError(f"Model ID {model_id} not found")
        if alt_id is None:
            alt_id = "."
        if alt_id not in self.structure[model_id].keys():
            raise ValueError(f"Alt ID {alt_id} not found")
        observed_residue_idx = self.structure[model_id][alt_id].keys()
        missing_residue_idx = [
            idx for idx in self.scheme.auth_idx_list if idx not in observed_residue_idx
        ]
        return missing_residue_idx

    def get_chem_comp_list(self):
        return self.chem_comp_list

    def get_structure(self):
        return self.structure

    def to_tensor(self, alphabet: str):
        pass

    def get_auth_asym_id(self):
        auth_asym_id_list = self.structure._atom_site_dict["_atom_site.auth_asym_id"]
        auth_asym_id_list = list(set(auth_asym_id_list))
        assert len(auth_asym_id_list) == 1, "Multiple auth_asym_id found"
        return auth_asym_id_list[0]

    def __repr__(self):
        output = "\033[1;43mAsymmetricChain\033[0m(\n"
        output += f"  asym_chain_id: {self.asym_chain_id}\n"
        output += f"  entity_id: {self.entity_id}\n"
        output += f"  scheme: {self.scheme}\n"
        output += f"  structure: {self.structure}\n"
        output += ")"
        # TODO
        return output

    def get_connections(self):
        """
        token level connection
        """
        pass

    def get_covalent_bonds(self):
        """
        get all covalent bond (N,2)
        """
        pass

    def get_chem_comp(self):
        """
        token level representation
        """
        pass

    def get_atom(self):
        """
        atom level representation
        """
        pass


class BioAssembly:
    def __init__(
        self,
        ID: str,
        deposition_date: str | None,
        resolution: str | None,
        chem_comp_idx_map: dict[str, int],
        chem_comp_dict: dict[str, ChemComp],
        asym_chain_dict: dict[str, AsymmetricChain],
        asym_pair_dict: dict[str, dict[str, Any]] | None,
        assembly_id_dict: dict[str, list[str]] | None,
        _pdbx_struct_oper_dict: dict[str, dict[str, Any]] | None,
        only_polymer: bool = False,
        device: torch.device | str = "cpu",
        types: list[str] = None,
        remove_signal_peptide: bool = False,
    ):
        if types is None:
            types = ["polymer", "nucleic_acid", "ligand"]
        self.ID = ID
        self.deposition_date = deposition_date
        self.resolution = resolution
        self.device = torch.device(device) if isinstance(device, str) else device
        self.chain_break_constant = 100  # constant for chain break
        self.only_polmyer = only_polymer
        self.chem_comp_idx_map = chem_comp_idx_map
        self.asym_chain_dict = asym_chain_dict
        self.assembly_id_dict = assembly_id_dict
        self._pdbx_struct_oper_dict = _pdbx_struct_oper_dict
        self.asym_pair_dict = asym_pair_dict
        self._struct_conn_distance_cutoff = 5.0
        self._struct_conn_distance_tol = 0.5
        self.types = types
        self.remove_signal_peptide = remove_signal_peptide

        self._parse_chem_comp_dict(chem_comp_dict)
        if asym_pair_dict is not None:
            self._parse_asym_pair_dict()

        self._tensorize()
        self._filter_type()

    def _parse_asym_pair_dict(self):
        # get bond info from asym_pair_dict (_struct_conn)

        def _get_alt_id(alt_id1, alt_id2):
            if (alt_id1 == "?" or alt_id1 == ".") and (alt_id2 == "?" or alt_id2 == "."):
                return "."
            elif (alt_id1 == "?" or alt_id1 == ".") and (
                alt_id2 != "?" and alt_id2 != "."
            ):
                return alt_id2
            elif (alt_id1 != "?" and alt_id1 != ".") and (
                alt_id2 == "?" or alt_id2 == "."
            ):
                return alt_id1
            else:
                if alt_id1 != alt_id2:
                    # TODO alt_id1 and alt_id2 are coupled! (checked by depositer)
                    raise StructConnAmbiguityError(
                        f"Ambiguity in struct_conn. alt_id1: {alt_id1}, alt_id2: {alt_id2}"  # noqa: E501
                    )
                return alt_id1

        # before main for loop, get pair of alt ids which are coupled
        for asym_pair in self.asym_pair_dict.keys():
            items = self.asym_pair_dict[asym_pair]
            label_alt_id1_list = items["_struct_conn.pdbx_ptnr1_label_alt_id"]
            label_alt_id2_list = items["_struct_conn.pdbx_ptnr2_label_alt_id"]
            conn_type_id_list = items["_struct_conn.conn_type_id"]
            conn_type_id_list = [
                _struct_conn_type[conn_type_id] for conn_type_id in conn_type_id_list
            ]
            for idx in range(len(conn_type_id_list)):
                alt_id1, alt_id2 = label_alt_id1_list[idx], label_alt_id2_list[idx]
                conn_type_id = conn_type_id_list[idx]
                if conn_type_id not in [2]:
                    continue
                alt_id = _get_alt_id(alt_id1, alt_id2)

        bond_dict = {}
        for asym_pair in self.asym_pair_dict.keys():
            items = self.asym_pair_dict[asym_pair]
            conn_type_id_list = items["_struct_conn.conn_type_id"]
            conn_type_id_list = [
                _struct_conn_type[conn_type_id] for conn_type_id in conn_type_id_list
            ]

            label_alt_id1_list = items["_struct_conn.pdbx_ptnr1_label_alt_id"]
            label_alt_id2_list = items["_struct_conn.pdbx_ptnr2_label_alt_id"]
            label_alt_id_list = []
            for alt_id1, alt_id2 in zip(label_alt_id1_list, label_alt_id2_list):
                try:
                    alt_id = _get_alt_id(alt_id1, alt_id2)
                except StructConnAmbiguityError:
                    continue
                label_alt_id_list.append(alt_id)
            chem_comp1_str_list = items["_struct_conn.ptnr1_label_comp_id"]
            chem_comp2_str_list = items["_struct_conn.ptnr2_label_comp_id"]
            chem_comp1_list = [
                self.chem_comp_dict[chem_comp] for chem_comp in chem_comp1_str_list
            ]
            chem_comp2_list = [
                self.chem_comp_dict[chem_comp] for chem_comp in chem_comp2_str_list
            ]

            auth_seq_id1_list = items["_struct_conn.ptnr1_auth_seq_id"]
            auth_seq_id2_list = items["_struct_conn.ptnr2_auth_seq_id"]
            ins_code1_list = items["_struct_conn.pdbx_ptnr1_PDB_ins_code"]
            ins_code2_list = items["_struct_conn.pdbx_ptnr2_PDB_ins_code"]

            auth_idx1_list = [
                f"{auth_seq_id1}.{ins_code1}" if ins_code1 != "?" else f"{auth_seq_id1}"
                for auth_seq_id1, ins_code1 in zip(auth_seq_id1_list, ins_code1_list)
            ]
            auth_idx2_list = [
                f"{auth_seq_id2}.{ins_code2}" if ins_code2 != "?" else f"{auth_seq_id2}"
                for auth_seq_id2, ins_code2 in zip(auth_seq_id2_list, ins_code2_list)
            ]

            label_atom_id1_list = items["_struct_conn.ptnr1_label_atom_id"]
            label_atom_id2_list = items["_struct_conn.ptnr2_label_atom_id"]
            label_atom_id1_list = [
                chem_comp.get_atoms().index(atom)
                for chem_comp, atom in zip(chem_comp1_list, label_atom_id1_list)
            ]
            label_atom_id2_list = [
                chem_comp.get_atoms().index(atom)
                for chem_comp, atom in zip(chem_comp2_list, label_atom_id2_list)
            ]
            bond_order_list = items["_struct_conn.pdbx_value_order"]
            bond_order_list = [
                bond_order_map[bond_order] for bond_order in bond_order_list
            ]

            if asym_pair not in bond_dict.keys():
                bond_dict[asym_pair] = {}

            for idx in range(len(conn_type_id_list)):
                alt_id1, alt_id2 = label_alt_id1_list[idx], label_alt_id2_list[idx]
                try:
                    alt_id = _get_alt_id(alt_id1, alt_id2)
                except StructConnAmbiguityError:
                    continue

                chem_comp1, chem_comp2 = chem_comp1_list[idx], chem_comp2_list[idx]
                auth_idx1, auth_idx2 = auth_idx1_list[idx], auth_idx2_list[idx]
                label_atom_id1, label_atom_id2 = (
                    label_atom_id1_list[idx],
                    label_atom_id2_list[idx],
                )
                bond_order = bond_order_list[idx]
                conn_type_id = conn_type_id_list[idx]

                if bond_dict[asym_pair].get(alt_id) is None:
                    bond_dict[asym_pair][alt_id] = []
                bond_dict[asym_pair][alt_id].append(
                    (
                        chem_comp1,
                        chem_comp2,
                        auth_idx1,
                        auth_idx2,
                        label_atom_id1,
                        label_atom_id2,
                        bond_order,
                        conn_type_id,
                    )
                )

        self._struct_conn_dict = bond_dict

    def _get_sequence_from_asym_chain(self, asym_chain, length):
        is_polymer = asym_chain.entity.get_type() == MoleculeType.POLYMER
        if is_polymer:
            sequence = asym_chain.get_sequence()
            polymer = asym_chain.entity
            match polymer.get_polymer_type():
                case PolymerType.PROTEIN:
                    sequence = [aa if aa in AA2num else "X" for aa in sequence]
                    sequence = [AA2num[aa] for aa in sequence]
                case PolymerType.PROTEIN_D:
                    sequence = [aa if aa in AA2num else "X" for aa in sequence]
                    sequence = [AA2num[aa] for aa in sequence]
                case PolymerType.DNA:
                    sequence = [dna if dna in DNA2num else "X" for dna in sequence]
                    sequence = [DNA2num[dna] for dna in sequence]
                case PolymerType.RNA:
                    sequence = [rna if rna in RNA2num else "X" for rna in sequence]
                    sequence = [RNA2num[rna] for rna in sequence]
                case PolymerType.NA_HYBRID:
                    sequence = asym_chain.get_sequence(canonical=False)
                    sequence_split = re.findall(r"\(.*?\)|.", sequence)
                    sequence_split = [
                        na.replace("(", "").replace(")", "") for na in sequence_split
                    ]
                    sequence = [na if na in NA2num else "X" for na in sequence_split]
                    sequence = [NA2num[na] for na in sequence]
                case PolymerType.PNA:
                    sequence = [-1] * length
                case _:
                    raise ValueError("Unknown polymer type")
        else:
            sequence = [-1] * length
        return sequence

    def _get_model_to_alt_id_list_from_asym_chain(self, asym_chain):
        model_to_alt_id_list = asym_chain.structure.model_to_alt_id_list
        return model_to_alt_id_list

    def _get_info_from_scheme(self, scheme_dict, entity, sequence):
        tensor_dict = {}
        bond_dict = {}
        n_atom_dict = {}
        n_residue_dict = {}

        entity_type = entity.get_type()

        entity_dict = {}
        for model_id in scheme_dict.keys():
            tensor_dict[model_id] = {}
            bond_dict[model_id] = {}
            n_atom_dict[model_id] = {}
            n_residue_dict[model_id] = {}
            entity_dict[model_id] = {}
            for alt_id in scheme_dict[model_id].keys():
                tensor_dict[model_id][alt_id] = {}
                bond_dict[model_id][alt_id] = {}
                n_atom_dict[model_id][alt_id] = {}
                n_residue_dict[model_id][alt_id] = {}
                entity_dict[model_id][alt_id] = {}

                scheme = scheme_dict[model_id][alt_id]

                n_row = 0
                residue_idx = 0

                # convert scheme and sequence to tensor by residue level and atom level
                auth_idx_to_atom_idx_map = {}
                auth_idx_to_residue_idx_map = {}
                # In some case, auth_idx = "?" in poly_scheme.
                # I assume that there is no strctural information for "?" auth_idx
                residue_idx_full_atom_list = []
                chem_comp_full_atom_list = []
                chem_comp_idx_full_atom_list = []
                sequence_full_atom_list = []

                scheme_cif_idx_list = scheme.cif_idx_list
                scheme_auth_idx_list = scheme.auth_idx_list
                scheme_chem_comp_list = scheme.chem_comp_list

                if entity_type == MoleculeType.POLYMER:
                    entity.parse(scheme)
                    bond_residue = [(idx, idx + 1, 0) for idx in range(len(scheme) - 1)]
                    bond_atom = []
                elif entity_type == MoleculeType.BRANCHED:
                    entity.parse(scheme)
                    bond_residue = entity.get_bonds(level="residue")
                    bond_atom = entity.get_bonds(level="atom")
                else:
                    bond_residue = []
                    bond_atom = []
                entity_dict[model_id][alt_id] = entity

                # bond information (wo hydrogen)
                # firstly, get polymer-canonical bonds (consecutive bonds)
                #   For polypeptide, ith C atom is connected to (i+1)th N atom (bond_type = (0, 0, 0))  # noqa: E501
                #   For DNA, RNA ith O3' atom is connected to (i+1)th P atom (bond_type = (0, 0, 0))  # noqa: E501
                # Secondly, get intra-residue bonds (intra-residue bonds, using chem_comp.get_bonds())  # noqa: E501
                # each item :
                #   residue level : tuple(idx1, idx2, 0) | 0 means canonical bond
                #   atom level : tuple(idx1,idx2, bond_type, aromatic, stereo, 0) | 0 means canonical bond  # noqa: E501

                if entity_type == MoleculeType.POLYMER:
                    canonical_seqs = entity.get_one_letter_code(canonical=True)
                before_chem_comp = None
                before_canonical_seq = None

                for scheme_idx in range(len(scheme)):
                    # cif_idx = scheme_cif_idx_list[scheme_idx]
                    auth_idx = scheme_auth_idx_list[scheme_idx]
                    chem_comp_str = scheme_chem_comp_list[scheme_idx]
                    chem_comp_idx = self.chem_comp_idx_map[chem_comp_str]

                    auth_idx_to_atom_idx_map[auth_idx] = n_row
                    auth_idx_to_residue_idx_map[auth_idx] = residue_idx

                    residue_idx_full_atom_list.extend(
                        [residue_idx] * self.chem_comp_to_n_atom[chem_comp_str]
                    )
                    chem_comp_full_atom_list.extend(
                        [chem_comp_str] * self.chem_comp_to_n_atom[chem_comp_str]
                    )
                    chem_comp_idx_full_atom_list.extend(
                        [chem_comp_idx] * self.chem_comp_to_n_atom[chem_comp_str]
                    )
                    sequence_full_atom_list.extend(
                        [sequence[scheme_idx]] * self.chem_comp_to_n_atom[chem_comp_str]
                    )

                    _hydrogen_mask = self.hydrogen_mask[chem_comp_str]
                    chem_comp = self.chem_comp_dict[chem_comp_str]
                    bonds = chem_comp.get_bonds()
                    if bonds is not None:
                        bonds = [
                            (n_row + idx1, n_row + idx2, bond_type, aromatic, stereo, 0)
                            for idx1, idx2, bond_type, aromatic, stereo in bonds
                            if idx1 in _hydrogen_mask and idx2 in _hydrogen_mask
                        ]
                        if entity_type == MoleculeType.POLYMER:
                            canonical_seq = canonical_seqs[scheme_idx]
                            if before_chem_comp is not None:
                                polymer_type = entity.get_polymer_type()
                                if canonical_seq != "X" and before_canonical_seq != "X":
                                    # I don't define any polymer-canonical bond for unknown residue "X"  # noqa: E501
                                    # if the .index() arises error, it means that the atom is not found in the chem_comp.  # noqa: E501
                                    # In this case, I'll skip the bond
                                    try:
                                        match polymer_type:
                                            case PolymerType.PROTEIN:
                                                C_idx = (
                                                    n_row
                                                    + before_chem_comp.get_atoms().index(
                                                        "C"
                                                    )
                                                    - self.chem_comp_to_n_atom[
                                                        before_chem_comp.get_code()
                                                    ]
                                                )
                                                N_idx = (
                                                    n_row
                                                    + chem_comp.get_atoms().index("N")
                                                )
                                                bonds.append((C_idx, N_idx, 0, 0, 0, 0))
                                            case PolymerType.PROTEIN_D:
                                                C_idx = (
                                                    n_row
                                                    + before_chem_comp.get_atoms().index(
                                                        "C"
                                                    )
                                                    - self.chem_comp_to_n_atom[
                                                        before_chem_comp.get_code()
                                                    ]
                                                )
                                                N_idx = (
                                                    n_row
                                                    + chem_comp.get_atoms().index("N")
                                                )
                                                bonds.append((C_idx, N_idx, 0, 0, 0, 0))
                                            case PolymerType.DNA:
                                                O3_idx = (
                                                    n_row
                                                    + before_chem_comp.get_atoms().index(
                                                        "O3'"
                                                    )
                                                    - self.chem_comp_to_n_atom[
                                                        before_chem_comp.get_code()
                                                    ]
                                                )
                                                P_idx = (
                                                    n_row
                                                    + chem_comp.get_atoms().index("P")
                                                )
                                                bonds.append((O3_idx, P_idx, 0, 0, 0, 0))
                                            case PolymerType.RNA:
                                                O3_idx = (
                                                    n_row
                                                    + before_chem_comp.get_atoms().index(
                                                        "O3'"
                                                    )
                                                    - self.chem_comp_to_n_atom[
                                                        before_chem_comp.get_code()
                                                    ]
                                                )
                                                P_idx = (
                                                    n_row
                                                    + chem_comp.get_atoms().index("P")
                                                )
                                                bonds.append((O3_idx, P_idx, 0, 0, 0, 0))
                                            case PolymerType.NA_HYBRID:
                                                O3_idx = (
                                                    n_row
                                                    + before_chem_comp.get_atoms().index(
                                                        "O3'"
                                                    )
                                                    - self.chem_comp_to_n_atom[
                                                        before_chem_comp.get_code()
                                                    ]
                                                )
                                                P_idx = (
                                                    n_row
                                                    + chem_comp.get_atoms().index("P")
                                                )
                                                bonds.append((O3_idx, P_idx, 0, 0, 0, 0))
                                            case PolymerType.PNA:
                                                pass
                                            case _:
                                                raise AssertionError(
                                                    "Unknown polymer type"
                                                )
                                    except Exception:
                                        pass
                            before_canonical_seq = canonical_seq

                        bond_atom.extend(bonds)
                    before_chem_comp = chem_comp

                    n_row += self.chem_comp_to_n_atom[chem_comp_str]

                    residue_idx += 1

                bond_dict[model_id][alt_id] = {
                    "atom": bond_atom,
                    "residue": bond_residue,
                }
                n_atom_dict[model_id][alt_id] = n_row

                residue_idx_full_atom_list = torch.tensor(
                    residue_idx_full_atom_list, device=self.device, dtype=torch.int
                )  # - min_residue_idx # 1-based to 0-based
                chem_comp_idx_full_atom_list = torch.tensor(
                    chem_comp_idx_full_atom_list, device=self.device, dtype=torch.int
                )
                sequence_full_atom_list = torch.tensor(
                    sequence_full_atom_list, device=self.device, dtype=torch.int
                )

                cif_idx_residue_list = (
                    torch.tensor(
                        scheme_cif_idx_list, device=self.device, dtype=torch.int
                    )
                    - 1
                )  # 1-based to 0-based
                chem_comp_idx_residue_list = torch.tensor(
                    [
                        self.chem_comp_idx_map[chem_comp]
                        for chem_comp in scheme_chem_comp_list
                    ],
                    device=self.device,
                    dtype=torch.int,
                )
                sequence_residue_list = torch.tensor(
                    sequence, device=self.device, dtype=torch.int
                )

                tensor_dict[model_id][alt_id] = {
                    "auth_idx_to_atom_idx_map": auth_idx_to_atom_idx_map,
                    "auth_idx_to_residue_idx_map": auth_idx_to_residue_idx_map,
                    "residue_idx_full_atom_list": residue_idx_full_atom_list,
                    "chem_comp_full_atom_list": chem_comp_full_atom_list,
                    "chem_comp_idx_full_atom_list": chem_comp_idx_full_atom_list,
                    "sequence_full_atom_list": sequence_full_atom_list,
                    "cif_idx_residue_list": cif_idx_residue_list,
                    "chem_comp_idx_residue_list": chem_comp_idx_residue_list,
                    "sequence_residue_list": sequence_residue_list,
                }

                n_residue_dict[model_id][alt_id] = len(scheme)

        return tensor_dict, bond_dict, n_atom_dict, n_residue_dict, entity_dict

    def _parse_chem_comp_dict(self, chem_comp_dict):
        n_atom = {}
        hydrogen_mask = {}
        for chem_comp in chem_comp_dict.keys():
            atoms = chem_comp_dict[chem_comp].get_atoms(one_letter=True)
            # remove hydrogen
            not_hydrogen_idx = [
                idx for idx, atom in enumerate(atoms) if atom != "H" and atom != "D"
            ]
            atoms = [atom for atom in atoms if atom != "H" and atom != "D"]
            n_atom[chem_comp] = len(atoms)
            hydrogen_mask[chem_comp] = not_hydrogen_idx

        self.chem_comp_to_n_atom = n_atom
        self.hydrogen_mask = hydrogen_mask
        self.chem_comp_dict = chem_comp_dict

    def _get_atom_str_info(self, atom, mask=False):
        if mask:
            atom_idx = atom_mapping[atom]
            return torch.tensor(
                [float(atom_idx), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=self.device
            )
        atom_idx = atom_mapping[atom.type_symbol]
        return torch.tensor(
            [
                float(atom_idx),
                1,
                float(atom.x),
                float(atom.y),
                float(atom.z),
                float(atom.occupancy),
                float(atom.b_factor),
            ],
            device=self.device,
        )

    def _parse_asym_info(self):
        asym_chain_to_scheme = {}
        asym_chain_to_entity = {}
        asym_chain_to_tensor = {}
        asym_chain_to_bond = {}
        # intra-chain bonds which is not included in _struct_conn fields.
        # It treats 1) polymer canonical bonds 2) intra-residue bonds
        asym_chain_to_n_atom = {}
        asym_chain_to_n_residue = {}
        asym_chain_to_auth_idx_to_atom_idx_map = {}

        for asym_chain_id in self.asym_chain_dict.keys():
            asym_chain = self.asym_chain_dict[asym_chain_id]

            entity = asym_chain.entity
            scheme = asym_chain.scheme

            asym_chain_to_n_residue[asym_chain_id] = len(scheme)
            # get sequence for MSA matching. I'll remove if I think it is not necessary
            sequence = self._get_sequence_from_asym_chain(asym_chain, len(scheme))

            model_to_alt_id_list = self._get_model_to_alt_id_list_from_asym_chain(
                asym_chain
            )
            models = asym_chain.structure.models

            tensor_dict, bond_dict, n_atom_dict, n_residue_dict, entity_dict = (
                self._get_info_from_scheme(asym_chain.scheme_dict, entity, sequence)
            )
            asym_chain_to_entity[asym_chain_id] = entity_dict

            asym_chain_to_scheme[asym_chain_id] = {}
            asym_chain_to_tensor[asym_chain_id] = {}
            asym_chain_to_bond[asym_chain_id] = {}
            asym_chain_to_n_atom[asym_chain_id] = {}
            asym_chain_to_n_residue[asym_chain_id] = {}
            asym_chain_to_auth_idx_to_atom_idx_map[asym_chain_id] = {}

            for model_id in models.keys():
                asym_chain_to_scheme[asym_chain_id][model_id] = {}
                asym_chain_to_tensor[asym_chain_id][model_id] = {}
                asym_chain_to_bond[asym_chain_id][model_id] = {}
                asym_chain_to_n_atom[asym_chain_id][model_id] = {}
                asym_chain_to_n_residue[asym_chain_id][model_id] = {}
                asym_chain_to_auth_idx_to_atom_idx_map[asym_chain_id][model_id] = {}

                for alt_id in model_to_alt_id_list[model_id]:
                    asym_chain_to_scheme[asym_chain_id][model_id][alt_id] = (
                        asym_chain.scheme_dict[model_id][alt_id]
                    )
                    residues = models[model_id][alt_id]
                    inner_tensor_dict = tensor_dict[model_id][alt_id]
                    inner_bond_dict = bond_dict[model_id][alt_id]
                    n_atom = n_atom_dict[model_id][alt_id]
                    n_residue = n_residue_dict[model_id][alt_id]

                    auth_idx_to_atom_idx_map = inner_tensor_dict[
                        "auth_idx_to_atom_idx_map"
                    ]
                    auth_idx_to_residue_idx_map = inner_tensor_dict[
                        "auth_idx_to_residue_idx_map"
                    ]
                    residue_idx_full_atom_list = inner_tensor_dict[
                        "residue_idx_full_atom_list"
                    ]
                    chem_comp_idx_full_atom_list = inner_tensor_dict[
                        "chem_comp_idx_full_atom_list"
                    ]
                    sequence_full_atom_list = inner_tensor_dict[
                        "sequence_full_atom_list"
                    ]
                    cif_idx_residue_list = inner_tensor_dict["cif_idx_residue_list"]
                    chem_comp_idx_residue_list = inner_tensor_dict[
                        "chem_comp_idx_residue_list"
                    ]
                    sequence_residue_list = inner_tensor_dict["sequence_residue_list"]

                    asym_chain_to_n_atom[asym_chain_id][model_id][alt_id] = n_atom
                    asym_chain_to_n_residue[asym_chain_id][model_id][alt_id] = n_residue
                    asym_chain_to_bond[asym_chain_id][model_id][alt_id] = inner_bond_dict
                    asym_chain_to_auth_idx_to_atom_idx_map[asym_chain_id][model_id][
                        alt_id
                    ] = auth_idx_to_atom_idx_map

                    if not entity.get_type() == MoleculeType.WATER:
                        atom_idx_map = [
                            auth_idx_to_atom_idx_map[residue_auth_idx]
                            for residue_auth_idx in residues.keys()
                        ]
                        residue_idx_map = [
                            auth_idx_to_residue_idx_map[residue_auth_idx]
                            for residue_auth_idx in residues.keys()
                        ]
                        all_atoms = [
                            residue.get_atoms() for residue in residues.values()
                        ]
                    else:
                        # for water, auth_idx can be missing. In this case, I just drop
                        common_key = []
                        for residue_auth_idx in auth_idx_to_atom_idx_map.keys():
                            if residue_auth_idx in residues.keys():
                                common_key.append(residue_auth_idx)
                        atom_idx_map = [
                            auth_idx_to_atom_idx_map[residue_auth_idx]
                            for residue_auth_idx in common_key
                        ]
                        residue_idx_map = [
                            auth_idx_to_residue_idx_map[residue_auth_idx]
                            for residue_auth_idx in common_key
                        ]
                        all_atoms = [
                            residue.get_atoms()
                            for residue in [residues[key] for key in common_key]
                        ]

                    tensor_atom = torch.zeros(n_atom, 3 + 2 + 3 + 2, device=self.device)
                    # 3 for sequence, chem_comp_idx_atom, cif_idx,
                    # 2 for atom and mask, 3 for xyz, 2 for occupancy and b_factor
                    tensor_residue = torch.zeros(
                        n_residue, 3 + 2 + 3 + 2, device=self.device
                    )
                    # 3 for sequence, chem_comp_idx_residue, residue_idx,
                    # 2 for atom and mask, 3 for xyz, 2 for occupancy and b_factor
                    filtered_atoms = []
                    filtered_residues = []

                    for idx in range(len(all_atoms)):
                        atoms = all_atoms[idx]
                        chem_comp = residues[list(residues.keys())[idx]].chem_comp
                        representative_atom, representative_atom_idx = (
                            chem_comp.get_representative_atom()
                        )
                        _hydrogen_mask = self.hydrogen_mask[chem_comp.get_code()]
                        chem_comp_full_atom = residues[
                            list(residues.keys())[idx]
                        ].chem_comp.get_atoms(one_letter=True)

                        atoms = [
                            atom
                            for idx, atom in enumerate(atoms)
                            if idx in _hydrogen_mask
                        ]
                        chem_comp_full_atom = [
                            atom
                            for idx, atom in enumerate(chem_comp_full_atom)
                            if idx in _hydrogen_mask
                        ]
                        if len(atoms) == 0:
                            # hydrogen or deuterium
                            continue
                        tmp = []

                        for (
                            chem_comp_atom,
                            atom,
                        ) in zip(chem_comp_full_atom, atoms):
                            if atom is None:
                                tmp.append(
                                    self._get_atom_str_info(chem_comp_atom, mask=True)
                                )
                                continue
                            if type(atom) is list:
                                atom = [
                                    a
                                    for a in atom
                                    if a.label_alt_id == alt_id or a.label_alt_id == "."
                                ]
                                assert len(atom) <= 1, (
                                    "Alternative conformation mismatch"
                                )
                                if len(atom) == 0:
                                    tmp.append(
                                        self._get_atom_str_info(
                                            chem_comp_atom, mask=True
                                        )
                                    )
                                else:
                                    tmp.append(self._get_atom_str_info(atom[0]))
                            else:
                                if (
                                    atom.label_alt_id == alt_id
                                    or atom.label_alt_id == "."
                                ):
                                    tmp.append(self._get_atom_str_info(atom))
                                else:
                                    tmp.append(
                                        self._get_atom_str_info(
                                            chem_comp_atom, mask=True
                                        )
                                    )  # No alternative conformation found

                        if not entity.get_type() == MoleculeType.WATER:
                            filtered_residues.append(tmp[representative_atom_idx])
                        else:
                            if representative_atom_idx < len(tmp):
                                filtered_residues.append(tmp[representative_atom_idx])
                        if len(tmp) > 0:
                            filtered_atoms.append(torch.stack(tmp, dim=0))

                    tensor_residue[:, 0] = sequence_residue_list
                    tensor_residue[:, 1] = chem_comp_idx_residue_list
                    tensor_residue[:, 2] = cif_idx_residue_list
                    for residue_idx in range(len(filtered_residues)):
                        tensor_residue[residue_idx_map[residue_idx], 3:] = (
                            filtered_residues[residue_idx]
                        )

                    tensor_atom[:, 0] = sequence_full_atom_list
                    tensor_atom[:, 1] = chem_comp_idx_full_atom_list
                    tensor_atom[:, 2] = residue_idx_full_atom_list
                    for residue_idx in range(len(filtered_atoms)):
                        tensor_atom[
                            atom_idx_map[residue_idx] : atom_idx_map[residue_idx]
                            + self.chem_comp_to_n_atom[
                                residues[
                                    list(residues.keys())[residue_idx]
                                ].chem_comp.get_code()
                            ],
                            3:,
                        ] = filtered_atoms[residue_idx]

                    asym_chain_to_tensor[asym_chain_id][model_id][alt_id] = {
                        "residue": tensor_residue,
                        "atom": tensor_atom,
                    }

        return (
            asym_chain_to_scheme,
            asym_chain_to_entity,
            asym_chain_to_tensor,
            asym_chain_to_bond,
            asym_chain_to_n_atom,
            asym_chain_to_n_residue,
            asym_chain_to_auth_idx_to_atom_idx_map,
        )

    def _get_model_alt(self, dictionary, model_id, alt_id):
        if model_id not in dictionary.keys():
            raise ValueError(f"Model ID {model_id} not found")
        alt_list = list(dictionary[model_id].keys())
        if alt_list == ["."]:
            return dictionary[model_id]["."]
        if alt_id not in dictionary[model_id].keys():
            # simply return common alt_id
            if "." in dictionary[model_id].keys():
                return dictionary[model_id]["."]
            else:
                raise AltIDError(
                    f"Alternative ID {alt_id} not found in model ID {model_id}"
                )
        return dictionary[model_id][alt_id]

    def _tensorize(self):
        (
            asym_chain_to_scheme,
            asym_chain_to_entity,
            asym_chain_to_tensor,
            asym_chain_to_bond,
            asym_chain_to_n_atom,
            asym_chain_to_n_residue,
            asym_chain_to_auth_idx_to_atom_idx_map,
        ) = self._parse_asym_info()
        # scheme, entity만 model_id, alt_id없고 나머진 다 있음.

        assembly_dict = {}
        for bioassembly_id in self.assembly_id_dict.keys():
            assembly_dict[bioassembly_id] = {}

            asym_id_list = list(self.assembly_id_dict[bioassembly_id].keys())
            asym_id_list = [
                asym_id
                for asym_id in asym_id_list
                if asym_id in asym_chain_to_tensor.keys()
            ]  # some chain can be removed. Ex) UNL
            model_list = [
                list(asym_chain_to_tensor[asym_id].keys()) for asym_id in asym_id_list
            ]

            model_to_asym_id_list = {}
            for asym_id in asym_id_list:
                for _model_id in asym_chain_to_tensor[asym_id].keys():
                    if _model_id not in model_to_asym_id_list:
                        model_to_asym_id_list[_model_id] = []
                    model_to_asym_id_list[_model_id].append(asym_id)

            # find largest model_list
            model_list = max(model_list, key=lambda x: len(x))
            model_alt_to_asym_id_list = {}
            model_to_alt = {}
            for model_id in model_list:
                model_alt_to_asym_id_list[model_id] = {}
                model_asym_id_list = model_to_asym_id_list[model_id]
                alt_list = [
                    list(asym_chain_to_tensor[asym_id][model_id].keys())
                    for asym_id in model_asym_id_list
                ]
                # remove ["."] from alt_list if it exists
                alt_list = [alt for alt in alt_list if alt != ["."]]
                if len(alt_list) == 0:
                    alt_union = ["."]
                # assert that all kinds of z
                else:
                    alt_list = list({tuple(sorted(x)) for x in alt_list})
                    alt_union = []
                    for alts in alt_list:
                        alt_union.extend(alts)
                    alt_union = list(set(alt_union))

                model_to_alt[model_id] = alt_union

                for alt_id in alt_union:
                    model_alt_to_asym_id_list[model_id][alt_id] = []
                    for asym_id in model_asym_id_list:
                        if (
                            alt_id in asym_chain_to_tensor[asym_id][model_id].keys()
                            or "." in asym_chain_to_tensor[asym_id][model_id].keys()
                        ):
                            model_alt_to_asym_id_list[model_id][alt_id].append(asym_id)

            for model_id in model_list:
                alt_id_list = model_to_alt[model_id]
                assembly_dict[bioassembly_id][model_id] = {}
                for alt_id in alt_id_list:
                    model_alt_asym_id_list = model_alt_to_asym_id_list[model_id][alt_id]

                    # assembly ID : {asym_id}_{oper_id}
                    assembly_chain_id_list = []
                    temp_dict = {}
                    for asym_chain_id in model_alt_asym_id_list:
                        oper_expression_list = self.assembly_id_dict[bioassembly_id][
                            asym_chain_id
                        ]
                        for oper_id in oper_expression_list:
                            oper_expression = self._pdbx_struct_oper_dict[oper_id]
                            assembly_chain_id = f"{asym_chain_id}_{oper_id}"
                            assembly_chain_id_list.append(assembly_chain_id)
                            temp_dict[assembly_chain_id] = {}
                            tensor = self._get_model_alt(
                                asym_chain_to_tensor[asym_chain_id], model_id, alt_id
                            )

                            temp_dict[assembly_chain_id]["scheme"] = self._get_model_alt(
                                asym_chain_to_scheme[asym_chain_id], model_id, alt_id
                            )
                            temp_dict[assembly_chain_id]["bond"] = self._get_model_alt(
                                asym_chain_to_bond[asym_chain_id], model_id, alt_id
                            )
                            # rotate and translate
                            matrix, vector = (
                                oper_expression["matrix"],
                                oper_expression["vector"],
                            )
                            residue_tensor = tensor["residue"].clone()
                            atom_tensor = tensor["atom"].clone()

                            residue_structure = residue_tensor[:, 5:8].clone()
                            atom_structure = atom_tensor[:, 5:8].clone()
                            transformed_residue_structure = (
                                torch.einsum("ij,kj->ki", matrix, residue_structure)
                                + vector
                            )
                            transformed_atom_structure = (
                                torch.einsum("ij,kj->ki", matrix, atom_structure)
                                + vector
                            )
                            residue_tensor[:, 5:8] = transformed_residue_structure
                            atom_tensor[:, 5:8] = transformed_atom_structure
                            temp_dict[assembly_chain_id]["tensor"] = {
                                "residue": residue_tensor,
                                "atom": atom_tensor,
                            }

                    # interchain bond를 다루기 위해선 concat, index를 재조정해야 함.
                    # 그래서 assembly_chain_id를 기준으로 정렬한 뒤,
                    # concat하고 index를 재조정함.
                    # 1) scheme concat
                    entity_id_list = [
                        temp_dict[assembly_chain_id]["scheme"].entity_id
                        for assembly_chain_id in assembly_chain_id_list
                    ]
                    entity_id = bioassembly_id + "_" + ",".join(entity_id_list)
                    scheme_type = MoleculeType.BIOASSEMBLY
                    assembly_cif_idx_list = []
                    assembly_auth_idx_list = []
                    assembly_chem_comp_list = []
                    assembly_hetero_list = []
                    assembly_residue_chain_break = {}
                    assembly_atom_chain_break = {}

                    assembly_residue_tensor = []
                    assembly_atom_tensor = []
                    assembly_residue_bond = []
                    assembly_atom_bond = []

                    assembly_residue_idx = 0
                    assembly_atom_idx = 0
                    for ii, assembly_chain_id in enumerate(assembly_chain_id_list):
                        cif_idx_list = temp_dict[assembly_chain_id][
                            "scheme"
                        ].cif_idx_list
                        auth_idx_list = temp_dict[assembly_chain_id][
                            "scheme"
                        ].auth_idx_list
                        chem_comp_list = temp_dict[assembly_chain_id][
                            "scheme"
                        ].chem_comp_list
                        hetero_list = temp_dict[assembly_chain_id]["scheme"].hetero_list
                        if hetero_list is None:
                            hetero_list = [True] * len(cif_idx_list)
                        assembly_cif_idx_list.extend(cif_idx_list)
                        assembly_auth_idx_list.extend(auth_idx_list)
                        assembly_chem_comp_list.extend(chem_comp_list)
                        assembly_hetero_list.extend(hetero_list)

                        cif_idx_list = torch.tensor(
                            cif_idx_list, device=self.device, dtype=torch.int
                        )  # (L_chain,)
                        cif_idx_list = (
                            cif_idx_list
                            + assembly_residue_idx
                            - min(cif_idx_list)
                            + (ii) * self.chain_break_constant
                        )

                        assembly_chain_residue_tensor = temp_dict[assembly_chain_id][
                            "tensor"
                        ]["residue"]
                        assembly_chain_atom_tensor = temp_dict[assembly_chain_id][
                            "tensor"
                        ]["atom"]
                        assembly_chain_atom_tensor[:, 2] = (
                            assembly_chain_atom_tensor[:, 2] + assembly_residue_idx
                        )

                        assembly_residue_tensor.append(assembly_chain_residue_tensor)
                        assembly_atom_tensor.append(assembly_chain_atom_tensor)

                        asym_id = assembly_chain_id.split("_")[0]
                        assembly_bond = self._get_model_alt(
                            asym_chain_to_bond[asym_id], model_id, alt_id
                        )
                        residue_bond = assembly_bond["residue"]
                        residue_bond = [
                            (
                                idx1 + assembly_residue_idx,
                                idx2 + assembly_residue_idx,
                                bond_type,
                            )
                            for idx1, idx2, bond_type in residue_bond
                        ]

                        atom_bond = assembly_bond["atom"]
                        n_atom = self._get_model_alt(
                            asym_chain_to_n_atom[asym_id], model_id, alt_id
                        )

                        atom_bond = [
                            (
                                idx1 + assembly_atom_idx,
                                idx2 + assembly_atom_idx,
                                bond_type,
                                aromatic,
                                stereo,
                                conn_type_id,
                            )
                            for idx1, idx2, bond_type, aromatic, stereo, conn_type_id in atom_bond
                        ]

                        assembly_residue_bond.extend(residue_bond)
                        assembly_atom_bond.extend(atom_bond)

                        assembly_residue_chain_break[assembly_chain_id] = (
                            assembly_residue_idx,
                            assembly_residue_idx + len(cif_idx_list) - 1,
                        )
                        assembly_atom_chain_break[assembly_chain_id] = (
                            assembly_atom_idx,
                            assembly_atom_idx + n_atom - 1,
                        )
                        assembly_residue_idx += len(cif_idx_list)

                        assembly_atom_idx += n_atom
                    assembly_scheme = Scheme(
                        entity_id,
                        scheme_type,
                        assembly_cif_idx_list,
                        assembly_auth_idx_list,
                        assembly_chem_comp_list,
                        assembly_hetero_list,
                    )

                    assembly_residue_tensor = torch.cat(assembly_residue_tensor, dim=0)
                    assembly_atom_tensor = torch.cat(assembly_atom_tensor, dim=0)

                    assembly_chain_n_residue_cumsum = {}
                    assembly_chain_n_atom_cumsum = {}
                    n_residue_cumsum = 0
                    n_atom_cumsum = 0
                    for assembly_chain_id in assembly_chain_id_list:
                        asym_chain_id = assembly_chain_id.split("_")[0]
                        assembly_chain_n_residue_cumsum[assembly_chain_id] = (
                            n_residue_cumsum
                        )
                        assembly_chain_n_atom_cumsum[assembly_chain_id] = n_atom_cumsum
                        n_residue = self._get_model_alt(
                            asym_chain_to_n_residue[asym_chain_id], model_id, alt_id
                        )
                        n_atom = self._get_model_alt(
                            asym_chain_to_n_atom[asym_chain_id], model_id, alt_id
                        )
                        n_residue_cumsum += n_residue
                        n_atom_cumsum += n_atom

                    assembly_struct_conn_dict = {}
                    if self.asym_pair_dict is not None:
                        for asym_pair in self._struct_conn_dict.keys():
                            _struct_conn_items = self._struct_conn_dict[asym_pair]
                            if alt_id not in _struct_conn_items.keys():
                                continue
                            _struct_conn_items = _struct_conn_items[alt_id]
                            asym_id1, asym_id2 = asym_pair.split(",")
                            try:
                                auth_idx_to_atom_idx_map1 = self._get_model_alt(
                                    asym_chain_to_auth_idx_to_atom_idx_map[asym_id1],
                                    model_id,
                                    alt_id,
                                )
                                auth_idx_to_atom_idx_map2 = self._get_model_alt(
                                    asym_chain_to_auth_idx_to_atom_idx_map[asym_id2],
                                    model_id,
                                    alt_id,
                                )
                            except Exception:
                                continue

                            if (
                                asym_id1 in model_alt_asym_id_list
                                and asym_id2 in model_alt_asym_id_list
                            ):
                                oper_expression_list1 = self.assembly_id_dict[
                                    bioassembly_id
                                ][asym_id1]
                                oper_expression_list2 = self.assembly_id_dict[
                                    bioassembly_id
                                ][asym_id2]
                                if oper_expression_list1 == oper_expression_list2:
                                    for oper_id in oper_expression_list1:
                                        assembly_chain_id1 = f"{asym_id1}_{oper_id}"
                                        assembly_chain_id2 = f"{asym_id2}_{oper_id}"
                                        assembly_struct_conn_dict[
                                            f"{assembly_chain_id1},{assembly_chain_id2}"
                                        ] = _struct_conn_items
                                elif (
                                    len(oper_expression_list1) == 1
                                    and len(oper_expression_list2) == 1
                                ):
                                    assembly_chain_id1 = (
                                        f"{asym_id1}_{oper_expression_list1[0]}"
                                    )
                                    assembly_chain_id2 = (
                                        f"{asym_id2}_{oper_expression_list2[0]}"
                                    )
                                    assembly_struct_conn_dict[
                                        f"{assembly_chain_id1},{assembly_chain_id2}"
                                    ] = _struct_conn_items
                                else:
                                    # do combinatorial calculation
                                    new_items = {}
                                    for _struct_conn_item in _struct_conn_items:
                                        residue_idx1, residue_idx2 = (
                                            _struct_conn_item[2],
                                            _struct_conn_item[3],
                                        )
                                        atom_idx1, atom_idx2 = (
                                            _struct_conn_item[4],
                                            _struct_conn_item[5],
                                        )
                                        min_distance = 1e10
                                        distance_dict = {}
                                        for oper_id1 in oper_expression_list1:
                                            for oper_id2 in oper_expression_list2:
                                                assembly_chain_id1 = (
                                                    f"{asym_id1}_{oper_id1}"
                                                )
                                                assembly_chain_id2 = (
                                                    f"{asym_id2}_{oper_id2}"
                                                )

                                                idx1_to_add = auth_idx_to_atom_idx_map1[
                                                    residue_idx1
                                                ]
                                                idx2_to_add = auth_idx_to_atom_idx_map2[
                                                    residue_idx2
                                                ]

                                                xyz1 = temp_dict[assembly_chain_id1][
                                                    "tensor"
                                                ]["atom"][atom_idx1 + idx1_to_add, 5:8]
                                                xyz2 = temp_dict[assembly_chain_id2][
                                                    "tensor"
                                                ]["atom"][atom_idx2 + idx2_to_add, 5:8]
                                                distance = torch.norm(xyz1 - xyz2)
                                                if distance < min_distance:
                                                    min_distance = distance

                                                distance_dict[
                                                    f"{assembly_chain_id1},{assembly_chain_id2}"
                                                ] = distance.item()
                                        if (
                                            min_distance
                                            > self._struct_conn_distance_cutoff
                                        ):
                                            continue
                                        for key in distance_dict.keys():
                                            if (
                                                distance_dict[key] - min_distance
                                            ).abs() < 1e-1:
                                                if key not in new_items.keys():
                                                    new_items[key] = []
                                                new_items[key].append(_struct_conn_item)
                                    for key in new_items.keys():
                                        assembly_struct_conn_dict[key] = new_items[key]

                    for asym_pair in assembly_struct_conn_dict.keys():
                        assembly_id1, assembly_id2 = asym_pair.split(",")
                        asym_id1, asym_id2 = (
                            assembly_id1.split("_")[0],
                            assembly_id2.split("_")[0],
                        )
                        auth_idx_to_atom_idx_map1 = self._get_model_alt(
                            asym_chain_to_auth_idx_to_atom_idx_map[asym_id1],
                            model_id,
                            alt_id,
                        )
                        auth_idx_to_atom_idx_map2 = self._get_model_alt(
                            asym_chain_to_auth_idx_to_atom_idx_map[asym_id2],
                            model_id,
                            alt_id,
                        )

                        atom_idx1_to_add = assembly_atom_chain_break[assembly_id1][0]
                        atom_idx2_to_add = assembly_atom_chain_break[assembly_id2][0]

                        residue_idx_to_add1 = assembly_chain_n_residue_cumsum[
                            assembly_id1
                        ]
                        residue_idx_to_add2 = assembly_chain_n_residue_cumsum[
                            assembly_id2
                        ]
                        scheme1 = self._get_model_alt(
                            asym_chain_to_scheme[assembly_id1.split("_")[0]],
                            model_id,
                            alt_id,
                        )
                        scheme2 = self._get_model_alt(
                            asym_chain_to_scheme[assembly_id2.split("_")[0]],
                            model_id,
                            alt_id,
                        )
                        auth_idx1_list = scheme1.auth_idx_list
                        auth_idx2_list = scheme2.auth_idx_list

                        bond_list = assembly_struct_conn_dict[asym_pair]

                        bond_list = [
                            (
                                auth_idx1_list.index(idx1) + residue_idx_to_add1,
                                auth_idx2_list.index(idx2) + residue_idx_to_add2,
                                idx3
                                + auth_idx_to_atom_idx_map1[idx1]
                                + atom_idx1_to_add,
                                idx4
                                + auth_idx_to_atom_idx_map2[idx2]
                                + atom_idx2_to_add,
                                bond_type,
                                conn_type_id,
                            )
                            for _, _, idx1, idx2, idx3, idx4, bond_type, conn_type_id in bond_list
                        ]

                        struct_conn_residue_bond_list = [
                            (idx1, idx2, conn_type)
                            for idx1, idx2, _, _, _, conn_type in bond_list
                        ]
                        struct_conn_atom_bond_list = [
                            (idx1, idx2, bond_type, 0, 0, conn_type)
                            for _, _, idx1, idx2, bond_type, conn_type in bond_list
                        ]
                        assembly_residue_bond.extend(struct_conn_residue_bond_list)
                        assembly_atom_bond.extend(struct_conn_atom_bond_list)

                    assembly_residue_bond = list(set(assembly_residue_bond))
                    assembly_atom_bond = list(set(assembly_atom_bond))
                    # sort by idx1_idx2
                    assembly_residue_bond.sort(key=lambda x: (x[0], x[1]))
                    assembly_atom_bond.sort(key=lambda x: (x[0], x[1]))

                    assembly_residue_bond = torch.tensor(
                        assembly_residue_bond, device=self.device, dtype=torch.int
                    )
                    assembly_atom_bond = torch.tensor(
                        assembly_atom_bond, device=self.device, dtype=torch.int
                    )

                    # same_entity
                    chain_num = len(assembly_chain_id_list)
                    same_entity = torch.zeros(
                        chain_num, chain_num, device=self.device, dtype=torch.bool
                    )
                    for ii, assembly_chain_id1 in enumerate(assembly_chain_id_list):
                        for jj, assembly_chain_id2 in enumerate(assembly_chain_id_list):
                            asym_id1, asym_id2 = (
                                assembly_chain_id1.split("_")[0],
                                assembly_chain_id2.split("_")[0],
                            )
                            entity_id1, entity_id2 = (
                                self._get_model_alt(
                                    asym_chain_to_scheme[asym_id1], model_id, alt_id
                                ).entity_id,
                                self._get_model_alt(
                                    asym_chain_to_scheme[asym_id2], model_id, alt_id
                                ).entity_id,
                            )
                            same_entity[ii, jj] = entity_id1 == entity_id2

                    # entity_list
                    assembly_entity_list = [
                        self._get_model_alt(
                            asym_chain_to_entity[assembly_chain_id.split("_")[0]],
                            model_id,
                            alt_id,
                        )
                        for assembly_chain_id in assembly_chain_id_list
                    ]

                    try:
                        biomol_structure = BioMolStructure(
                            self.ID,
                            bioassembly_id,
                            model_id,
                            alt_id,
                            assembly_scheme,
                            assembly_entity_list,
                            assembly_residue_tensor,
                            assembly_atom_tensor,
                            assembly_residue_bond,
                            assembly_atom_bond,
                            assembly_residue_chain_break,
                            assembly_atom_chain_break,
                            self.chem_comp_dict,
                            same_entity,
                            device=self.device,
                            types=self.types,
                            remove_signal_peptide=self.remove_signal_peptide,
                        )
                    except EmptyStructureError:
                        continue
                    assembly_dict[bioassembly_id][model_id][alt_id] = biomol_structure
        self.assembly_dict = assembly_dict

    def _filter_type(self):
        _types = []
        for _type in self.types:
            if _type == "protein":
                _types.append(PolymerType.PROTEIN)
            elif _type == "nucleic_acid":
                _types.append(PolymerType.DNA)
                _types.append(PolymerType.RNA)
                _types.append(PolymerType.NA_HYBRID)
            elif _type == "ligand":
                _types.append(MoleculeType.NONPOLYMER)

        filtered_asym_chain_dict = {}
        for asym_chain_id in self.asym_chain_dict.keys():
            entity = self.asym_chain_dict[asym_chain_id].entity
            entity_type = entity.get_type()
            if entity_type == MoleculeType.POLYMER:
                entity_type = entity.get_polymer_type()
            if entity_type in _types:
                filtered_asym_chain_dict[asym_chain_id] = self.asym_chain_dict[
                    asym_chain_id
                ]
        self.asym_chain_dict = filtered_asym_chain_dict

    def __getitem__(self, key: str | int | tuple[str, int]) -> dict:
        return self.assembly_dict[key]

    def keys(self) -> list[str]:
        return self.assembly_dict.keys()

    def to(self, device):
        for assembly_id in self.assembly_dict.keys():
            for model_id in self.assembly_dict[assembly_id].keys():
                for alt_id in self.assembly_dict[assembly_id][model_id].keys():
                    self.assembly_dict[assembly_id][model_id][alt_id].to(device)
        return self

    def get_chain_id_map(self, label_to_auth=True):
        chain_id_map = {}
        for label_asym_id in self.asym_chain_dict.keys():
            if label_to_auth:
                chain_id_map[label_asym_id] = self.asym_chain_dict[
                    label_asym_id
                ].get_auth_asym_id()
            else:
                chain_id_map[self.asym_chain_dict[label_asym_id].get_auth_asym_id()] = (
                    label_asym_id
                )
        return chain_id_map

    def __repr__(self):
        return f"BioMolAssembly(ID={self.ID}, assembly_dict={self.assembly_dict.keys()})"


def parse_signalp(signalp_path: str) -> list[str]:
    """
    Parse the signalp output file and extract the sequence IDs.
    """
    if not os.path.exists(signalp_path):
        return None
    with open(signalp_path) as f:
        lines = f.readlines()

    result = lines[1].split("\t")
    return int(result[3]) - 1, int(result[4]) - 1


class BioMolStructure:
    def __init__(
        self,
        ID,
        bioassembly_id,
        model_id,
        alt_id,
        scheme: Scheme,
        entity_list: list[Polymer | NonPolymer | Branched | Water],
        residue_tensor: torch.Tensor,
        atom_tensor: torch.Tensor,
        residue_bond: torch.Tensor,
        atom_bond: torch.Tensor,
        residue_chain_break: dict[str, tuple[int, int]],
        atom_chain_break: dict[str, tuple[int, int]],
        chem_comp_dict: dict[str, ChemComp],
        same_entity: torch.Tensor,
        device: str = "cuda",
        types: list[str] = None,
        remove_signal_peptide: bool = False,
    ):
        if types is None:
            types = ["protein", "ligand", "nucleic_acid"]
        self.ID = ID
        self.bioassembly_id = bioassembly_id
        self.model_id = model_id
        self.alt_id = alt_id
        self.scheme = scheme
        self.entity_list = entity_list
        self.residue_tensor = residue_tensor
        self.atom_tensor = atom_tensor
        self.residue_bond = residue_bond
        self.atom_bond = atom_bond
        self.residue_chain_break = residue_chain_break
        self.atom_chain_break = atom_chain_break
        self.chem_comp_dict = chem_comp_dict
        self.same_entity = same_entity
        self.device = device
        self.types = types

        self._filtered = False

        self.filter_by_type(types)
        if self.__len__() == 0:
            raise EmptyStructureError(
                f"Empty structure for {ID} {bioassembly_id} {model_id} {alt_id}"
            )
        self._load_sequence_hash()  # WARNING!!! This function only works for protein
        self._load_contact_graph()
        if remove_signal_peptide:
            # it requires signalp which is precomputed. (SIGNALP_PATH)
            self.remove_signal_peptide()

    def get_residue_idx_to_atom_idx(self, atom_tensor_residue_idx, residue_idx):
        mask = torch.isin(atom_tensor_residue_idx, residue_idx)
        return torch.where(mask)[0]
        # return torch.where(self.atom_tensor[:, 2] == residue_idx)[0]

    def _load_sequence_hash(self):
        sequence_hash = {}
        for entity_idx, entity in enumerate(self.entity_list):
            chain = list(self.residue_chain_break.keys())[entity_idx]

            match entity.get_type():
                case MoleculeType.POLYMER:
                    polymer_type = entity.get_polymer_type()
                    tag = molecule_type_map[polymer_type]
                    sequence = entity.get_one_letter_code(canonical=True)
                    sequence = tag + sequence
                    sequence_hash[chain] = str(seq_to_hash[sequence])
                case MoleculeType.NONPOLYMER:
                    tag = molecule_type_map[MoleculeType.NONPOLYMER]
                    chem_comp = entity.get_chem_comp()
                    sequence = f"({chem_comp})"
                    sequence = tag + sequence
                    sequence_hash[chain] = str(seq_to_hash[sequence])
                    pass
                case MoleculeType.BRANCHED:
                    tag = molecule_type_map[MoleculeType.BRANCHED]
                    chem_comp_list = entity.get_chem_comp_list()
                    chem_comp_list = [str(chem_comp) for chem_comp in chem_comp_list]
                    bond_list = entity.get_bonds(level="residue")
                    bond_list = [
                        f"({idx1}, {idx2}, {conn_type})"
                        for idx1, idx2, conn_type in bond_list
                    ]
                    sequence = f"({')('.join(chem_comp_list)})|{','.join(bond_list)}"
                    sequence = tag + sequence
                    sequence_hash[chain] = str(seq_to_hash[sequence])
                    pass
                case MoleculeType.WATER:
                    pass
        self.sequence_hash = sequence_hash

    def _load_contact_graph(self):
        graph_path = f"{CONTACT_GRAPH_PATH}/{self.ID[1:3]}/{self.ID}.graph"
        contact_graph = ContactGraph(graph_path)
        contact_graph.choose((self.bioassembly_id, self.model_id, self.alt_id))
        self.contact_graph = contact_graph 

    def __len__(self, level: str = "residue") -> int:
        if level is None:
            return len(self.residue_tensor)
        if level == "residue":
            return len(self.residue_tensor)
        elif level == "atom":
            return len(self.atom_tensor)
        else:
            raise ValueError(f"Unknown level {level}")

    def _get_residue_length(self):
        return len(self.residue_tensor)

    def get_mask_by_type(self, molecule_type: str):
        len_residue = self._get_residue_length()
        residue_mask = torch.zeros(len_residue, device=self.device, dtype=torch.bool)
        entity_mask = []
        chain_mask = dict.fromkeys(self.residue_chain_break.keys(), False)
        for entity, chain in zip(self.entity_list, self.residue_chain_break.keys()):
            chain_start, chain_end = self.residue_chain_break[chain]
            condition = False
            match molecule_type:
                case "protein":
                    if entity.get_type() == MoleculeType.POLYMER:
                        if entity.get_polymer_type() == PolymerType.PROTEIN:
                            condition = True
                case "nucleic_acid":
                    if entity.get_type() == MoleculeType.POLYMER:
                        if (
                            entity.get_polymer_type() == PolymerType.RNA
                            or entity.get_polymer_type() == PolymerType.DNA
                            or entity.get_polymer_type() == PolymerType.NA_HYBRID
                        ):
                            condition = True
                case "nonpolymer":
                    if entity.get_type() == MoleculeType.NONPOLYMER:
                        condition = True
                case "branched":
                    if entity.get_type() == MoleculeType.BRANCHED:
                        condition = True
                case "ligand":
                    if (
                        entity.get_type() == MoleculeType.NONPOLYMER
                        or entity.get_type() == MoleculeType.BRANCHED
                    ):
                        condition = True
                case "water":
                    if entity.get_type() == MoleculeType.WATER:
                        condition = True
            if condition:
                residue_mask[chain_start : chain_end + 1] = True
                entity_mask.append(True)
                chain_mask[chain] = True
            else:
                entity_mask.append(False)

        entity_mask = torch.tensor(entity_mask, device=self.device, dtype=torch.bool)
        return residue_mask, entity_mask, chain_mask

    def _filter_tensor(self, residue_mask):
        def rearrange_bond(bond, mask):
            if bond.size(0) == 0:
                return bond
            mask = mask.to(torch.bool)
            L = len(mask)
            cumsum = torch.cumsum(mask, dim=0) - 1  # (1,1,1,0,0,1) -> (0,1,2,2,2,3)
            idx_map = torch.stack(
                [torch.arange(L, device=self.device), cumsum], dim=1
            )  # (L, 2)
            idx_map = idx_map[mask.bool()]
            allowed_old = idx_map[:, 0]
            new_idx = idx_map[:, 1]

            pairs = bond.clone()
            pair_mask = torch.isin(pairs[:, 0], allowed_old) & torch.isin(
                pairs[:, 1], allowed_old
            )

            filtered_pairs = pairs[pair_mask]
            allowed_old = allowed_old.contiguous()
            mapped_src = new_idx[
                torch.searchsorted(allowed_old, filtered_pairs[:, 0].contiguous())
            ]
            mapped_dst = new_idx[
                torch.searchsorted(allowed_old, filtered_pairs[:, 1].contiguous())
            ]
            remapped_pairs = filtered_pairs.clone()  # Clone to preserve other columns.
            remapped_pairs[:, 0] = mapped_src
            remapped_pairs[:, 1] = mapped_dst
            return remapped_pairs

        # mask : torch.Tensor (L_residue, )
        len_residue = len(residue_mask)
        assert len_residue == self._get_residue_length(), (
            "Atom level filtering is not supported yet"
        )
        residue_valid_idx = torch.arange(len_residue, device=self.device)[
            residue_mask.bool()
        ]
        remaining_residue_idx = self.atom_tensor[:, 2].unique()
        atom_valid_mask = torch.isin(
            self.atom_tensor[:, 2].int(), remaining_residue_idx[residue_valid_idx]
        )
        atom_valid_idx = torch.nonzero(atom_valid_mask, as_tuple=True)[0]
        atom_mask = torch.zeros(
            len(self.atom_tensor), device=self.device, dtype=torch.bool
        )
        atom_mask[atom_valid_idx] = True
        self.residue_tensor = self.residue_tensor[residue_mask.bool()]
        self.atom_tensor = self.atom_tensor[atom_mask]

        self.residue_bond = rearrange_bond(self.residue_bond, residue_mask)
        self.atom_bond = rearrange_bond(self.atom_bond, atom_mask)
        self._filtered = True

    def _filter_scheme(self, residue_mask, entity_mask):
        scheme_entity_id = self.scheme.entity_id
        assembly_id = scheme_entity_id.split("_")[0]
        entity_id_list = scheme_entity_id.split("_")[-1].split(",")
        entity_id_list = [
            entity_id for entity_id, mask in zip(entity_id_list, entity_mask) if mask
        ]
        scheme_entity_id = f"{assembly_id}_{','.join(entity_id_list)}"
        scheme_type = MoleculeType.BIOASSEMBLY
        scheme_auth_idx_list = self.scheme.auth_idx_list
        scheme_cif_idx_list = self.scheme.cif_idx_list
        scheme_chem_comp_list = self.scheme.chem_comp_list
        scheme_hetero_list = self.scheme.hetero_list

        scheme_auth_idx_list = [
            auth_idx
            for auth_idx, mask in zip(scheme_auth_idx_list, residue_mask)
            if mask
        ]
        scheme_cif_idx_list = [
            cif_idx for cif_idx, mask in zip(scheme_cif_idx_list, residue_mask) if mask
        ]
        scheme_chem_comp_list = [
            chem_comp
            for chem_comp, mask in zip(scheme_chem_comp_list, residue_mask)
            if mask
        ]
        scheme_hetero_list = [
            hetero for hetero, mask in zip(scheme_hetero_list, residue_mask) if mask
        ]

        self.scheme = Scheme(
            scheme_entity_id,
            scheme_type,
            scheme_cif_idx_list,
            scheme_auth_idx_list,
            scheme_chem_comp_list,
            scheme_hetero_list,
        )

    def _filter_entity(self, residue_mask, entity_mask):
        new_entity_list = []
        for chain, entity, _entity_mask in zip(
            self.residue_chain_break.keys(), self.entity_list, entity_mask
        ):
            if not _entity_mask:
                continue
            chain_start, chain_end = self.residue_chain_break[chain]
            mask = residue_mask[chain_start : chain_end + 1]
            new_entity = copy.deepcopy(entity)
            new_entity.crop(mask)
            new_entity_list.append(new_entity)
        self.entity_list = new_entity_list

    def filter_by_type(self, types: list[str]):  # type to be remained
        type_list = [
            "protein",
            "nucleic_acid",
            "nonpolymer",
            "branched",
            "ligand",
            "water",
        ]
        residue_mask_list = []
        entity_mask_list = []
        chain_mask_list = []
        for molecule_type in types:
            if molecule_type not in type_list:
                raise ValueError(f"Unknown molecule type {molecule_type}")
            residue_mask, entity_mask, chain_mask = self.get_mask_by_type(molecule_type)
            residue_mask_list.append(residue_mask)
            entity_mask_list.append(entity_mask)
            chain_mask_list.append(chain_mask)
        residue_mask = torch.stack(residue_mask_list, dim=0)  # (N, L_residue)
        residue_mask = torch.any(residue_mask, dim=0)  # (L_residue, )
        entity_mask = torch.stack(entity_mask_list, dim=0)  # (N, L_entity)
        entity_mask = torch.any(entity_mask, dim=0)  # (L_entity, )
        entity_mask = entity_mask.tolist()
        new_residue_chain_break = {}
        new_atom_chain_break = {}
        new_chain_mask = dict.fromkeys(self.residue_chain_break.keys(), False)
        for chain in new_chain_mask.keys():
            if new_chain_mask[chain]:
                continue
            for chain_mask in chain_mask_list:
                if chain_mask[chain]:
                    new_chain_mask[chain] = True

        residue_chain_start = 0
        atom_chain_start = 0
        for chain in new_chain_mask.keys():
            if new_chain_mask[chain]:
                residue_chain_idx_diff = (
                    self.residue_chain_break[chain][1]
                    - self.residue_chain_break[chain][0]
                    + 1
                )
                new_residue_chain_break[chain] = (
                    residue_chain_start,
                    residue_chain_start + residue_chain_idx_diff - 1,
                )
                atom_chain_idx_diff = (
                    self.atom_chain_break[chain][1] - self.atom_chain_break[chain][0] + 1
                )
                new_atom_chain_break[chain] = (
                    atom_chain_start,
                    atom_chain_start + atom_chain_idx_diff - 1,
                )
                residue_chain_start += residue_chain_idx_diff
                atom_chain_start += atom_chain_idx_diff

        self._filter_tensor(residue_mask)
        self._filter_scheme(residue_mask, entity_mask)
        self.entity_list = [
            entity for entity, mask in zip(self.entity_list, entity_mask) if mask
        ]
        self.same_entity = self.same_entity[entity_mask][:, entity_mask]
        self.residue_chain_break = new_residue_chain_break
        self.atom_chain_break = new_atom_chain_break

    def remove_signal_peptide(self):
        signalp_results = {}
        crop_indices = []
        for chain, sequence_hash in self.sequence_hash.items():
            sequence_hash = str(sequence_hash).zfill(6)
            signalp_path = SIGNALP_PATH + f"/{sequence_hash}.gff3"
            chain_start, chain_end = self.residue_chain_break[chain]
            if not os.path.exists(signalp_path):
                crop_indices.append(torch.arange(chain_start, chain_end + 1))
                continue
            start, end = parse_signalp(signalp_path)
            if start is None or end is None:
                crop_indices.append(torch.arange(chain_start, chain_end + 1))
                continue
            signalp_results[chain] = (start, end)
            crop_indices.append(torch.arange(chain_start, chain_end + 1)[end + 1 :])
        if len(signalp_results) == 0:
            return
        crop_indices = torch.cat(crop_indices, dim=0)

        chain_crop = self.get_chain_crop_indices(crop_indices)
        entity_mask = []
        chain_mask = {}
        for chain in self.residue_chain_break.keys():
            if chain not in chain_crop.keys():
                entity_mask.append(False)
            else:
                entity_mask.append(True)
                chain_mask[chain] = True

        new_residue_chain_break = {}

        residue_chain_start = 0
        for chain in chain_crop.keys():
            new_residue_chain_break[chain] = (
                residue_chain_start,
                residue_chain_start + chain_crop[chain].shape[0] - 1,
            )
            residue_chain_start += chain_crop[chain].shape[0]

        new_atom_chain_break = {}
        atom_chain_start = 0

        remaining_residue_idx = self.atom_tensor[:, 2].unique()
        remaining_residue_idx = torch.sort(remaining_residue_idx)[0].int()
        atom_tensor_residue_idx = self.atom_tensor[:, 2].int()
        for chain in chain_crop.keys():
            residue_start, residue_end = new_residue_chain_break[chain]
            atom_num = 0
            residue_cropped = crop_indices[residue_start : residue_end + 1]
            residue_cropped = remaining_residue_idx[residue_cropped]
            atom_num += self.get_residue_idx_to_atom_idx(
                atom_tensor_residue_idx, residue_cropped
            ).shape[0]
            new_atom_chain_break[chain] = (
                atom_chain_start,
                atom_chain_start + atom_num - 1,
            )
            atom_chain_start += atom_num

        residue_mask = torch.zeros(len(self), device=self.device, dtype=torch.bool)
        residue_mask[crop_indices] = True
        self._filter_tensor(residue_mask)
        self._filter_scheme(residue_mask, entity_mask)
        self._filter_entity(residue_mask, entity_mask)
        self.entity_list = [
            entity for entity, mask in zip(self.entity_list, entity_mask) if mask
        ]
        self.same_entity = self.same_entity[entity_mask][:, entity_mask]
        self.residue_chain_break = new_residue_chain_break
        self.atom_chain_break = new_atom_chain_break

    def get_chain_crop_indices(
        self, crop_indices: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        chain_crop = {}
        for chain in self.residue_chain_break.keys():
            chain_start, chain_end = self.residue_chain_break[chain]
            minus_start = crop_indices - chain_start
            minus_end = crop_indices - chain_end
            crop_chain = (minus_start >= 0) & (minus_end <= 0)
            if crop_chain.sum() == 0:
                continue
            chain_crop[chain] = crop_indices[crop_chain] - chain_start
        return chain_crop

    def crop(
        self,
        crop_indices: torch.Tensor,
    ):
        chain_crop = self.get_chain_crop_indices(crop_indices)
        entity_mask = []
        chain_mask = {}
        for chain in self.residue_chain_break.keys():
            if chain not in chain_crop.keys():
                entity_mask.append(False)
            else:
                entity_mask.append(True)
                chain_mask[chain] = True

        new_residue_chain_break = {}

        residue_chain_start = 0
        for chain in chain_crop.keys():
            new_residue_chain_break[chain] = (
                residue_chain_start,
                residue_chain_start + chain_crop[chain].shape[0] - 1,
            )
            residue_chain_start += chain_crop[chain].shape[0]

        new_atom_chain_break = {}
        atom_chain_start = 0
        remaining_residue_idx = self.atom_tensor[:, 2].unique()
        remaining_residue_idx = torch.sort(remaining_residue_idx)[0]
        atom_tensor_residue_idx = self.atom_tensor[:, 2].int()
        for chain in chain_crop.keys():
            residue_start, residue_end = new_residue_chain_break[chain]
            atom_num = 0
            residue_cropped = crop_indices[residue_start : residue_end + 1]
            if len(residue_cropped) == 0:
                continue
            residue_cropped = remaining_residue_idx[residue_cropped]
            atom_num += self.get_residue_idx_to_atom_idx(
                atom_tensor_residue_idx, residue_cropped
            ).shape[0]

            new_atom_chain_break[chain] = (
                atom_chain_start,
                atom_chain_start + atom_num - 1,
            )
            atom_chain_start += atom_num

        residue_mask = torch.zeros(len(self), device=self.device, dtype=torch.bool)
        residue_mask[crop_indices] = True
        self._filter_tensor(residue_mask)
        self._filter_scheme(residue_mask, entity_mask)
        self.entity_list = [
            entity for entity, mask in zip(self.entity_list, entity_mask) if mask
        ]
        self.same_entity = self.same_entity[entity_mask][:, entity_mask]
        self.residue_chain_break = new_residue_chain_break
        self.atom_chain_break = new_atom_chain_break

    def to(self, dtype: torch.dtype = torch.float32, device: torch.device = "cuda"):
        self.residue_tensor = self.residue_tensor.to(dtype=dtype, device=device)
        self.atom_tensor = self.atom_tensor.to(dtype=dtype, device=device)
        self.residue_bond = self.residue_bond.to(dtype=dtype, device=device)
        self.atom_bond = self.atom_bond.to(dtype=dtype, device=device)
        self.same_entity = self.same_entity.to(dtype=dtype, device=device)
        return self

    def to_mmcif(self, cif_path):
        """
        1. _atom_site
        2. _struct_conn
        3. ~_scheme
        4. ~_branch
        5. ~_entity
        """

        output = f"#\ndata_{self.ID}_{self.model_id}_{self.alt_id}\n"

        def _to_mmcif_format(list):
            list = [str(item) for item in list]
            max_length = max([len(item) for item in list])
            list = [item.ljust(max_length) for item in list]
            return list

        def _write_atom_site(self):
            """
            #
            loop_
            _atom_site.group_PDB
            _atom_site.id
            _atom_site.type_symbol
            _atom_site.label_atom_id
            _atom_site.label_alt_id
            _atom_site.label_comp_id
            _atom_site.label_asym_id
            _atom_site.label_entity_id
            _atom_site.label_seq_id
            _atom_site.pdbx_PDB_ins_code
            _atom_site.Cartn_x
            _atom_site.Cartn_y
            _atom_site.Cartn_z
            _atom_site.occupancy
            _atom_site.B_iso_or_equiv
            _atom_site.pdbx_formal_charge
            _atom_site.auth_seq_id
            _atom_site.auth_comp_id
            _atom_site.auth_asym_id
            _atom_site.auth_atom_id
            _atom_site.pdbx_PDB_model_num
            """
            header = [
                "_atom_site.group_PDB",
                "_atom_site.id",
                "_atom_site.type_symbol",
                "_atom_site.label_atom_id",
                "_atom_site.label_alt_id",
                "_atom_site.label_comp_id",
                "_atom_site.label_asym_id",
                "_atom_site.label_entity_id",
                "_atom_site.label_seq_id",
                "_atom_site.pdbx_PDB_ins_code",
                "_atom_site.Cartn_x",
                "_atom_site.Cartn_y",
                "_atom_site.Cartn_z",
                "_atom_site.occupancy",
                "_atom_site.B_iso_or_equiv",
                "_atom_site.pdbx_formal_charge",
                "_atom_site.auth_seq_id",
                "_atom_site.auth_comp_id",
                "_atom_site.auth_asym_id",
                "_atom_site.auth_atom_id",
                "_atom_site.pdbx_PDB_model_num",
            ]
            output = "#\n"
            output += "loop_\n"
            output += "\n".join(header) + "\n"

            atom_tensor = self.atom_tensor
            length = len(atom_tensor)

            residue_auth_idx_list = self.scheme.auth_idx_list
            residue_ins_code_list = [
                "?" if len(auth_idx.split(".")) == 1 else auth_idx.split(".")[1]
                for auth_idx in residue_auth_idx_list
            ]
            residue_chem_comp_list = self.scheme.chem_comp_list
            chem_comp_full_atom_dict = {}
            for chem_comp_str in self.scheme.chem_comp_list:
                chem_comp = self.chem_comp_dict[chem_comp_str]
                full_atom_list = chem_comp.get_atoms(one_letter=False)
                one_letter_atom_list = chem_comp.get_atoms(one_letter=True)
                hydrogen_mask = [atom == "H" for atom in one_letter_atom_list]
                chem_comp_full_atom_dict[chem_comp_str] = [
                    full_atom
                    for full_atom, mask in zip(full_atom_list, hydrogen_mask)
                    if not mask
                ]

            residue_idx_to_chain_entity = {}

            for entity_idx in range(len(self.entity_list)):
                chain = list(self.residue_chain_break.keys())[entity_idx]
                start, end = self.residue_chain_break[chain]
                for idx in range(start, end + 1):
                    residue_idx_to_chain_entity[idx] = (chain, entity_idx + 1)

            group_PDB_list = []
            auth_idx_list = []
            ins_code_list = []
            type_symbol_list = []
            label_atom_id_list = []
            label_comp_id_list = []
            label_entity_id_list = []
            label_asym_id_list = []
            label_seq_list = []
            before_residue_idx = -1
            scheme_idx = -1
            residue_start = 0
            for line_idx in range(length):
                hetero = int(atom_tensor[line_idx, 0].item()) == -1
                hetero = "HETATM" if hetero else "ATOM  "
                atom_idx = int(atom_tensor[line_idx, 3].item())
                residue_idx = int(atom_tensor[line_idx, 2].item())
                if before_residue_idx != residue_idx:
                    residue_start = 0
                    scheme_idx += 1
                chem_comp_str = residue_chem_comp_list[scheme_idx]
                chem_comp_full_atom = chem_comp_full_atom_dict[chem_comp_str]

                auth_idx = residue_auth_idx_list[scheme_idx]
                ins_code = residue_ins_code_list[scheme_idx]
                type_symbol = num_to_atom[atom_idx]

                label_atom_id = chem_comp_full_atom[residue_start]
                chain, entity_id = residue_idx_to_chain_entity[scheme_idx]

                group_PDB_list.append(hetero)
                auth_idx_list.append(auth_idx)
                ins_code_list.append(ins_code)
                type_symbol_list.append(type_symbol)
                label_atom_id_list.append(label_atom_id)
                label_comp_id_list.append(chem_comp_str)
                label_entity_id_list.append(entity_id)
                label_asym_id_list.append(chain)
                label_seq_list.append(residue_idx + 1)

                before_residue_idx = residue_idx
                residue_start += 1

            atom_tensor_mask = atom_tensor[:, 4].bool()
            atom_tensor = atom_tensor[atom_tensor_mask]  # (L, 10)

            group_PDB_list = [
                group_PDB_list[i] for i in range(length) if atom_tensor_mask[i]
            ]
            type_symbol_list = [
                type_symbol_list[i] for i in range(length) if atom_tensor_mask[i]
            ]
            label_atom_id_list = [
                label_atom_id_list[i] for i in range(length) if atom_tensor_mask[i]
            ]
            label_comp_id_list = [
                label_comp_id_list[i] for i in range(length) if atom_tensor_mask[i]
            ]
            label_asym_id_list = [
                label_asym_id_list[i] for i in range(length) if atom_tensor_mask[i]
            ]
            label_entity_id_list = [
                label_entity_id_list[i] for i in range(length) if atom_tensor_mask[i]
            ]
            label_seq_list = [
                label_seq_list[i] for i in range(length) if atom_tensor_mask[i]
            ]
            ins_code_list = [
                ins_code_list[i] for i in range(length) if atom_tensor_mask[i]
            ]
            auth_idx_list = [
                auth_idx_list[i] for i in range(length) if atom_tensor_mask[i]
            ]

            type_symbol_list = _to_mmcif_format(type_symbol_list)
            label_atom_id_list = _to_mmcif_format(label_atom_id_list)
            label_comp_id_list = _to_mmcif_format(label_comp_id_list)
            label_asym_id_list = _to_mmcif_format(label_asym_id_list)
            label_entity_id_list = _to_mmcif_format(label_entity_id_list)
            label_seq_list = _to_mmcif_format(label_seq_list)
            ins_code_list = _to_mmcif_format(ins_code_list)
            auth_idx_list = _to_mmcif_format(auth_idx_list)

            length = len(atom_tensor)

            atom_idx_list = [i + 1 for i in range(length)]
            atom_idx_list = _to_mmcif_format(atom_idx_list)

            x, y, z = atom_tensor[:, 5], atom_tensor[:, 6], atom_tensor[:, 7]
            x, y, z = x.tolist(), y.tolist(), z.tolist()
            x, y, z = (
                [round(i, 3) for i in x],
                [round(i, 3) for i in y],
                [round(i, 3) for i in z],
            )
            x, y, z = (
                [f"{i:.3f}" for i in x],
                [f"{i:.3f}" for i in y],
                [f"{i:.3f}" for i in z],
            )
            x, y, z = _to_mmcif_format(x), _to_mmcif_format(y), _to_mmcif_format(z)

            occup, bfactor = atom_tensor[:, 8].tolist(), atom_tensor[:, 9].tolist()
            occup, bfactor = [round(i, 3) for i in occup], [round(i, 3) for i in bfactor]
            occup, bfactor = [f"{i:.2f}" for i in occup], [f"{i:.2f}" for i in bfactor]
            occup, bfactor = _to_mmcif_format(occup), _to_mmcif_format(bfactor)
            formal_charge_list = ["?"] * length
            model_num_list = [self.model_id] * length
            label_alt_id_list = [self.alt_id] * length

            model_num_list = _to_mmcif_format(model_num_list)
            label_alt_id_list = _to_mmcif_format(label_alt_id_list)

            for idx in range(length):
                fields = [
                    group_PDB_list[idx],
                    atom_idx_list[idx],
                    type_symbol_list[idx],
                    label_atom_id_list[idx],
                    label_alt_id_list[idx],
                    label_comp_id_list[idx],
                    label_asym_id_list[idx],
                    label_entity_id_list[idx],
                    label_seq_list[idx],
                    ins_code_list[idx],
                    x[idx],
                    y[idx],
                    z[idx],
                    occup[idx],
                    bfactor[idx],
                    formal_charge_list[idx],
                    auth_idx_list[idx],
                    label_comp_id_list[idx],
                    label_asym_id_list[idx],
                    label_atom_id_list[idx],
                    model_num_list[idx],
                ]
                output += " ".join(map(str, fields)) + "\n"

            return output

        def _write_struct_conn(self):
            """
            #
            loop_
            _struct_conn.conn_id
            _struct_conn.ptnr1_label_asym_id
            _struct_conn.ptnr1_label_comp_id
            _struct_conn.ptnr1_label_seq_id
            _struct_conn.ptnr1_label_atom_id
            _struct_conn.ptnr1_auth_asym_id
            _struct_conn.ptnr1_auth_comp_id
            _struct_conn.ptnr1_auth_seq_id
            _struct_conn.ptnr1_auth_atom_id
            _struct_conn.ptnr1_symmetry
            _struct_conn.ptnr2_label_asym_id
            _struct_conn.ptnr2_label_comp_id
            _struct_conn.ptnr2_label_seq_id
            _struct_conn.ptnr2_label_atom_id
            _struct_conn.ptnr2_auth_asym_id
            _struct_conn.ptnr2_auth_comp_id
            _struct_conn.ptnr2_auth_seq_id
            _struct_conn.ptnr2_auth_atom_id
            _struct_conn.ptnr2_symmetry
            _struct_conn.conn_type_id
            _struct_conn.details
            """
            header = [
                "_struct_conn.conn_id",
                "_struct_conn.ptnr1_label_asym_id",
                "_struct_conn.ptnr1_label_comp_id",
                "_struct_conn.ptnr1_label_seq_id",
                "_struct_conn.ptnr1_label_atom_id",
                "_struct_conn.ptnr1_auth_asym_id",
                "_struct_conn.ptnr1_auth_comp_id",
                "_struct_conn.ptnr1_auth_seq_id",
                "_struct_conn.ptnr1_auth_atom_id",
                "_struct_conn.ptnr1_symmetry",
                "_struct_conn.ptnr2_label_asym_id",
                "_struct_conn.ptnr2_label_comp_id",
                "_struct_conn.ptnr2_label_seq_id",
                "_struct_conn.ptnr2_label_atom_id",
                "_struct_conn.ptnr2_auth_asym_id",
                "_struct_conn.ptnr2_auth_comp_id",
                "_struct_conn.ptnr2_auth_seq_id",
                "_struct_conn.ptnr2_auth_atom_id",
                "_struct_conn.ptnr2_symmetry",
                "_struct_conn.conn_type_id",
                "_struct_conn.details",
            ]
            output = "#\n"
            output += "loop_\n"
            output += "\n".join(header) + "\n"

            residue_bond = self.residue_bond
            length = len(residue_bond)

            conn_id_list = [i + 1 for i in range(length)]
            conn_id_list = _to_mmcif_format(conn_id_list)

        _atom_site = _write_atom_site(self)
        output += _atom_site
        # _struct_conn = _write_struct_conn(self) # TODO

        with open(cif_path, "w") as f:
            f.write(output)

    def get_sequence(self, canonical: bool = True):
        chain_to_sequence = {}
        for entity_idx, entity in enumerate(self.entity_list):
            chain = list(self.residue_chain_break.keys())[entity_idx]
            match entity.get_type():
                case MoleculeType.POLYMER:
                    sequence = entity.get_one_letter_code(canonical=canonical)
                case MoleculeType.NONPOLYMER:
                    chem_comp = entity.get_chem_comp()
                    sequence = f"({chem_comp})"
                case MoleculeType.BRANCHED:
                    chem_comp_list = entity.get_chem_comp_list()
                    chem_comp_list = [str(chem_comp) for chem_comp in chem_comp_list]
                    bond_list = entity.get_bonds(level="residue")
                    bond_list = [
                        f"({idx1}, {idx2}, {conn_type})"
                        for idx1, idx2, conn_type in bond_list
                    ]
                    sequence = f"({')('.join(chem_comp_list)})  | {','.join(bond_list)}"
                case MoleculeType.WATER:
                    sequence = None
            chain_to_sequence[chain] = sequence
        return chain_to_sequence

    def save_entities(self, save_dir="/data/psk6950/PDB_2024Mar18/entity/"):
        # polymer : save as fasta (canonical sequence)
        # nonpolymer : save chem_comp
        # branched : save residue level graph
        # water : pass
        ID = self.ID
        save_dir = f"{save_dir}/{ID[1:3]}"
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        for entity_idx, entity in enumerate(self.entity_list):
            chain = list(self.residue_chain_break.keys())[entity_idx]
            chain = chain.split("_")[0]
            save_path = f"{save_dir}/{ID}_{chain}.fasta"

            if os.path.exists(save_path):
                continue

            match entity.get_type():
                case MoleculeType.POLYMER:
                    fasta_header = (
                        f">{ID}_{chain} | POLYMER | {entity.get_polymer_type()}"
                    )
                    sequence = entity.get_one_letter_code(canonical=True)
                case MoleculeType.NONPOLYMER:
                    fasta_header = f">{ID}_{chain} | NONPOLYMER"
                    chem_comp = entity.get_chem_comp()
                    sequence = f"({chem_comp})"
                case MoleculeType.BRANCHED:
                    fasta_header = f">{ID}_{chain} | BRANCHED"
                    chem_comp_list = entity.get_chem_comp_list()
                    chem_comp_list = [str(chem_comp) for chem_comp in chem_comp_list]
                    bond_list = entity.get_bonds(level="residue")
                    bond_list = [
                        f"({idx1}, {idx2}, {conn_type})"
                        for idx1, idx2, conn_type in bond_list
                    ]
                    sequence = f"({')('.join(chem_comp_list)})  | {','.join(bond_list)}"
                case MoleculeType.WATER:
                    pass

            if not os.path.exists(save_path):
                with open(f"{save_dir}/{ID}_{chain}.fasta", "w") as f:
                    f.write(f"{fasta_header}\n")
                    f.write(f"{sequence}\n")

    def __repr__(self):
        output = "\033[1;43mBioMolStructure \033[0m[\n"
        output += f"\t\033[1;43mID \033[0m: {self.ID}\n"
        output += f"\t\033[1;43mBioassembly ID \033[0m: {self.bioassembly_id}\n"
        output += f"\t\033[1;43mModel ID \033[0m: {self.model_id}\n"
        output += f"\t\033[1;43mAlt ID \033[0m: {self.alt_id}\n"

        scheme_text = self.scheme.__repr__().split("\n")
        for text in scheme_text:
            output += f"\t{text}\n"

        sequence_hash = "None" if self.sequence_hash is None else self.sequence_hash
        output += f"\t\033[1;43mSequence Hash \033[0m: {sequence_hash}\n"

        output += (
            f"\t\033[1;43mChains \033[0m: {list(self.residue_chain_break.keys())}\n"
        )
        entity_list = self.entity_list
        entity_num = len(entity_list)
        max_print = 3
        same_entity_max_print = 8

        if entity_num > max_print:
            entity_list = entity_list[: max_print - 1] + [entity_list[-1]]
        if entity_num > same_entity_max_print:
            entity_idx = torch.arange(same_entity_max_print, device=self.device)
            entity_idx[-1] = entity_num - 1
            entity_idx[-2] = entity_num - 2
            same_entity = self.same_entity[entity_idx][:, entity_idx]
        else:
            same_entity = self.same_entity
        entity_repr_list = [entity.__repr__() for entity in entity_list]
        longest_len = 0
        for entity_repr in entity_repr_list:
            text = entity_repr.split("\n")
            for t in text:
                if len(t) > longest_len:
                    longest_len = len(t)
        ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

        def strip_ansi(text):
            """Remove ANSI escape codes from text."""
            return ANSI_ESCAPE.sub("", text)

        for _ in range(5):
            entity_text_list = [
                entity_repr.split("\n")[_] for entity_repr in entity_repr_list
            ]

            if entity_num > max_print:
                # Apply padding only on stripped lengths, then reapply ANSI text
                entity_text_list = [
                    text.ljust(longest_len + len(text) - len(strip_ansi(text)))
                    for text in entity_text_list[: max_print - 1]
                ]

                if _ != 2:
                    entity_text_list.append("".center(30))  # Empty placeholder centered
                else:
                    entity_text_list.append(
                        f"...({entity_num - max_print} more entities)...".center(30)
                    )

                entity_text_list.append(
                    entity_repr_list[-1].split("\n")[_].ljust(longest_len)
                )
            else:
                entity_text_list = [
                    text.ljust(longest_len + len(text) - len(strip_ansi(text)))
                    for text in entity_text_list
                ]

            output += "\t" + " | ".join(entity_text_list) + "\n"
        filled = "■"  # Filled square for True
        empty = "□"  # Empty square for False

        output += "\t\033[1;43mSame Entity Matrix \033[0m\n"
        for row_idx, row in enumerate(same_entity):
            output += "\t\t"
            if entity_num <= same_entity_max_print:
                for element in row:
                    output += filled if element else empty
            else:
                if row_idx == same_entity_max_print - 2:
                    output += ".\n\t\t.\n\t\t"
                for element in row[: same_entity_max_print - 2]:
                    output += filled if element else empty
                output += "..."
                output += filled if row[-2] else empty
                output += filled if row[-1] else empty
            output += "\n"
        output += "]"
        return output
