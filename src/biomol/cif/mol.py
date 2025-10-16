from pathlib import Path
import numpy as np

from biomol import BioMol
from biomol.core import EdgeFeature, NodeFeature, ViewProtocol


class CIFAtomView(
    ViewProtocol["CIFAtomView", "CIFResidueView", "CIFChainView", "CIFMol"],
):
    """View class for CIF atoms."""

    @property
    def atom_id(self) -> NodeFeature:
        """Atom IDs. Example: 'N', 'CA', 'C', 'O', etc."""

    @property
    def element(self) -> NodeFeature:
        """Atom elements. Example: 'C', 'N', 'O', etc."""

    @property
    def atom_aromatic(self) -> NodeFeature:
        """Aromatic flag."""

    @property
    def atom_stereo(self) -> NodeFeature:
        """Stereochemistry flag."""

    @property
    def charge(self) -> NodeFeature:
        """Formal charge of atoms."""

    @property
    def model_xyz(self) -> NodeFeature:
        """Model XYZ coordinates of atoms in chemical component."""

    @property
    def xyz(self) -> NodeFeature:
        """XYZ coordinates of atoms."""

    @property
    def b_factor(self) -> NodeFeature:
        """B-factors of atoms."""

    @property
    def occupancy(self) -> NodeFeature:
        """Occupancy of atoms."""

    @property
    def bond_type(self) -> EdgeFeature:
        """Bond types between atoms."""

    @property
    def bond_aromatic(self) -> EdgeFeature:
        """Aromatic bonds between atoms."""

    @property
    def bond_stereo(self) -> EdgeFeature:
        """Bond stereochemistry between atoms."""


class CIFResidueView(
    ViewProtocol["CIFAtomView", "CIFResidueView", "CIFChainView", "CIFMol"],
):
    """View class for CIF residues."""

    @property
    def name(self) -> NodeFeature:
        """Residue names."""

    @property
    def formula(self) -> NodeFeature:
        """Residue formulas."""

    @property
    def one_letter_code_can(self) -> NodeFeature:
        """One-letter code (canonical)."""

    @property
    def one_letter_code(self) -> NodeFeature:
        """One-letter code (not canonical)."""

    @property
    def cif_idx(self) -> NodeFeature:
        """CIF residue indices."""

    @property
    def auth_idx(self) -> NodeFeature:
        """Author residue indices."""

    @property
    def chem_comp(self) -> NodeFeature:
        """Chemical component IDs."""

    @property
    def hetero(self) -> NodeFeature:
        """Hetero flag."""

    @property
    def residue_bond(self) -> EdgeFeature:
        """Residue-level bonds 1 if exists else not."""


class CIFChainView(
    ViewProtocol["CIFAtomView", "CIFResidueView", "CIFChainView", "CIFMol"],
):
    """View class for CIF chains."""

    @property
    def entity_id(self) -> NodeFeature:
        """Entity IDs."""

    @property
    def chain_id(self) -> NodeFeature:
        """Chain IDs. asym_id_oper_id. Example: 'A_1', 'B_1', etc."""


class CIFMol(BioMol["CIFAtomView", "CIFResidueView", "CIFChainView"]):
    """Class for CIF molecules."""

    @property
    def id(self) -> str:
        """PDB ID of the molecule."""
        return self.metadata["id"]

    @property
    def assembly_id(self) -> str:
        """Assembly ID of the molecule."""
        return self.metadata["assembly_id"]

    @property
    def model_id(self) -> int:
        """Model ID of the molecule."""
        return self.metadata["model_id"]

    @property
    def alt_id(self) -> str:
        """Alternate location ID of the molecule."""
        return self.metadata["alt_id"]

    def to_cif(self, output_path: Path) -> None:
        """Write a CIFMol object to a CIF file."""

        """
        1. _atom_site
        2. _struct_conn (TODO)
        3. ~_scheme (TODO)
        4. ~_branch (TODO)
        5. ~_entity (TODO)
        """

        # output = f"#\ndata_{self.metadata['id']}_{self.model_id}_{self.alt_id}\n"
        output = f"#\ndata_{self.id}_{self.model_id}_{self.alt_id}\n"

        xyz = self.atoms.xyz
        mask = ~np.isnan(xyz).any(axis=1)
        length = mask.sum()
        atom_to_res = np.array(self.index_table.atom_to_res)
        res_to_chain = np.array(self.index_table.res_to_chain)
        atom_to_chain = res_to_chain[atom_to_res]

        group_PDB_list = self.residues.hetero[atom_to_res][mask]
        id_list = 1 + np.arange(length)
        type_symbol_list = self.atoms.element[mask]
        label_atom_id_list = self.atoms.atom_id[mask]
        label_alt_id_list = [self.alt_id] * length
        label_comp_id_list = self.residues.chem_comp[atom_to_res][mask]
        label_asym_id_list = self.chains.chain_id[atom_to_chain][mask]
        label_entity_id_list = self.chains.entity_id[atom_to_chain][mask]
        label_seq_id_list = self.residues.cif_idx[atom_to_res][mask]
        auth_idx_list = self.residues.auth_idx[atom_to_res][mask]
        auth_seq_id_list, ins_code_list = zip(
            *[
                (p[0], p[1] if len(p) > 1 else "?")
                for p in (s.split(".") for s in auth_idx_list)
            ],
            strict=False,
        )

        # ins_code_list = [
        Cartn_x_list = self.atoms.xyz[mask, 0]
        Cartn_y_list = self.atoms.xyz[mask, 1]
        Cartn_z_list = self.atoms.xyz[mask, 2]
        occupancy_list = self.atoms.occupancy[mask]
        B_iso_or_equiv_list = self.atoms.b_factor[mask]
        pdbx_formal_charge_list = self.atoms.charge[mask]
        pdbx_PDB_model_num_list = [self.model_id] * length

        def _to_mmcif_format(list):
            list = [str(item) for item in list]
            max_length = max([len(item) for item in list])
            list = [item.ljust(max_length) for item in list]
            return list

        def _write_atom_site(self):
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

            length = len(self.atoms)
            group_PDB_list = self.atoms.hetero
            id_list = 1 + np.arange(length)
            type_symbol_list = self.atoms.element
            label_atom_id_list = self.atoms.atom_id
            label_alt_id_list = [self.alt_id] * length

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
            occup, bfactor = (
                [round(i, 3) for i in occup],
                [round(i, 3) for i in bfactor],
            )
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

        _atom_site = _write_atom_site(self)
        output += _atom_site

        with open(cif_path, "w") as f:
            f.write(output)
