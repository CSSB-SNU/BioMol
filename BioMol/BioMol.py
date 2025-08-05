import os
from typing import Any
import random
import torch
import copy

from BioMol.utils.parser import parse_cif, parse_simple_pdb
from BioMol.utils.MSA import MSA, ComplexMSA
from BioMol.utils.crop import (
    # crop_contiguous,
    crop_contiguous_monomer,
    crop_spatial,
    crop_spatial_interface,
    get_chain_crop_indices,
)
from BioMol.utils.read_lmdb import read_cif_lmdb, read_MSA_lmdb
from BioMol import ALL_TYPE_CONFIG_PATH, PROTEIN_ONLY_CONFIG_PATH
from BioMol.utils.error import NoValidChainsError, NoInterfaceError

"""
BioMol class

It can be loaded from
- sequence file (fasta)
- structure file (pdb, cif)
- or tensor (sequence or structure tensor) <- strictly formatted
- or .pkl file

And this class can be saved as
- sequence file (fasta)
- structure file (pdb, cif)
- .pkl file

=== Attributes ===
- description: dict
- sequence: BioMolSequence object
- structure: BioMolStructure object
- MSA: BioMolMSA object
- Template: BioMolTemplate object
- Function: BioMolFunction object

=== Methods ===
- __init__()
- __str__() # return brief information
- __repr__() # return detailed information
- get_sequence() # return sequence
- get_structure() # return structure
- get_MSA() # return MSA
- get_template() # return Template
- get_function() # return Function
- set_sequence() # set sequence
- set_structure() # set structure
- load_MSA() # load MSA
- load_template() # load Template
- write_pdb() # write pdb file
- write_cif() # write cif file
- write_fasta() # write fasta file
- write_a3m() # write a3m file
- pickle() # save as .pkl file
- help() # print help

- get_biological_assembly() # return biological assembly structure
- choose_model() # choose model from multiple models
- select_chain() # select chain
- crop() # crop sequence and structure
"""


class BioMol:
    def __init__(self, *args, **kwargs):
        self.description = {}

        self.load_water = False if "load_water" not in kwargs else kwargs["load_water"]

        # Handle different types of arguments
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, str):  # Single string argument (e.g., path to a file)
                if arg.endswith(".cif") or arg.endswith(".cif.gz"):
                    self.load_cif(arg)
                elif arg.endswith(".pdb") or arg.endswith(".pdb.gz"):
                    self.load_pdb(arg)
                else:
                    raise ValueError("Unknown file format.")
            elif isinstance(arg, dict):  # Dictionary argument
                self._load_attributes(arg)
            else:
                raise ValueError("Unsupported input type.")
        else:
            # Handle key-value arguments normally
            self._load_attributes(kwargs)

    def __str__(self):
        return "BioMol object"  # TODO: add brief information

    def __repr__(self):
        return "BioMol object"  # TODO: add detailed information

    def _load_attributes(self, attributes: dict[str, Any]) -> None:
        """
        Allowed
        - multiple sequence, in this case first sequence is used as a query.
        - multiple structure (multiple models)
        - multiple cif or pdb files (dynamics or multiple models)

        Not allowed
        - input both cif and pdb. You should choose one format.
        - different number of chains between each sequence and structure
            Ex) sequence 1 : monomer, sequence 2 : dimer
        - multiple MSA files. You can assign MSA for each chain, but not for each seq.
          i.e., MSA is always assigned to query sequence.
        - First sequence of MSA should be the same as query sequence.
        - Different sequence at multiple structure.
            Ex) str1: seq1, str2: seq2 but seq1 != seq2
          (All structures should encode query sequence if sequence is provided)
        - Incompatible inputs like
            - sequence & structure
            - (cif or pdb) & sequence

        Not recommended
        - In the case of multiple sequence, it is not recommended to have very different
        sequence. Multiple sequence is
            mainly used for mutation analysis or partially missing sequence.
        - Very large assembly structure due to memory issue.

        Not implemented
        - function related attributes

        Hierarchy of attributes
        1. cif or pdb
        2. structure
        3. sequence
        4. MSA
        5. Template
        6. Function (not implemented)
        """

        # Check if cif and pdb are both provided
        if "cif" in attributes and "pdb" in attributes:
            raise ValueError(
                "Cannot provide both cif and pdb files. Please choose one format."
            )
        if "cif_ID" in attributes and "pdb_ID" in attributes:
            raise ValueError(
                "Cannot provide both cif_ID and pdb_ID. Please choose one format."
            )
        if "cif_ID" in attributes:
            self.pdb_ID = attributes["cif_ID"]
            # pop cif_ID from attributes
            attributes.pop("cif_ID")

        # Load attributes
        for name, value in attributes.items():
            setattr(self, name, value)

        allowed_mol_types = ["protein", "nucleic_acid", "ligand"]
        if "mol_types" not in attributes:
            self.mol_types = ["protein"]  # default to protein

        assert [_type in allowed_mol_types for _type in self.mol_types], (
            f"Invalid mol_types. Allowed types are {allowed_mol_types}."
        )

        if self.mol_types == ["protein"]:
            self.type_config_path = PROTEIN_ONLY_CONFIG_PATH
        elif self.mol_types == ["protein", "nucleic_acid", "ligand"]:
            self.type_config_path = ALL_TYPE_CONFIG_PATH
        else:
            raise NotImplementedError

        for name, value in self.__dict__.items():
            if name == "pdb_ID":
                self.load_cif_from_ID(value)
                break
            if name == "cif":
                self.load_cif_from_path(value, self.remove_signal_peptide, self.use_lmdb)
                break
            elif name == "pdb":
                self.load_pdb(value)
                break
            # elif name == "sequence":
            #     self.load_sequence(value)
            # elif name == "structure":
            #     self.load_structure(value)
            # elif name == "MSA":
            #     self.load_MSA(value)
            # elif name == "Template":
            #     self.load_template(value)

    def _get_attribute(self, name: str) -> Any:
        return getattr(self, name, None)

    def check_sequence(self):
        pass

    def load_cif_from_ID(
        self,
        ID: str,
    ) -> None:
        bioassembly = read_cif_lmdb(ID)
        if bioassembly is None:
            print(f"File not found in LMDB: {ID}")
            return
        self.ID = bioassembly.ID
        self.bioassembly = bioassembly
        self.cif_loaded = True

    def load_cif_from_path(
        self,
        path: str,
        remove_signal_peptide: bool = False,
        use_lmdb: bool = True,
    ) -> None:
        assert os.path.exists(path), f"File not found: {path}"
        assert path.endswith(".cif") or path.endswith(".cif.gz"), (
            "Invalid file format. Please provide a cif file."
        )
        if use_lmdb:
            print(f"Loading cif from LMDB: {path}")
            bioassembly = read_cif_lmdb(os.path.basename(path).split(".")[0])
            if bioassembly is None:
                print(f"File not found in LMDB: {path}")
                return
        else:
            bioassembly = parse_cif(path, self.type_config_path, remove_signal_peptide)
        self.ID = bioassembly.ID
        self.bioassembly = bioassembly
        self.cif_loaded = True

    def load_pdb(self, path: str, cif_config_path: str = None) -> None:
        """
        Load pdb file and parse it.
        WARNING : This function does not support general pdb file.
        It only supports simple pdb file like fb db or predicted structure.
        """
        assert os.path.exists(path), f"File not found: {path}"
        assert path.endswith(".pdb") or path.endswith(".pdb.gz"), (
            "Invalid file format. Please provide a pdb file."
        )
        bioassembly = parse_simple_pdb(path)
        self.bioassembly = bioassembly
        self.pdb_loaded = True

    def choose(
        self, bioassembly_id: int | str, model_id: int | str, label_alt_id: str
    ) -> None:
        assert self.cif_loaded or self.pdb_loaded, "CIF file is not loaded."
        self.bioassembly_id = bioassembly_id
        self.model_id = model_id
        self.label_alt_id = label_alt_id
        self.structure_loaded = True
        self.structure = self.bioassembly[bioassembly_id][model_id][label_alt_id]

    def get_crop_indices(
        self,
        chain_bias=None,  # for spatial crop
        interface_bias=None,  # for interface crop
        contiguous_crop_weight: float = 0.2,
        spatial_crop_weight: float = 0.4,
        interface_crop_weight: float = 0.4,
        crop_length: int = 384,
        level="residue",  # "residue" or "atom"
        monomer_only: bool = False,  # if True, only use monomer chains
    ) -> None:
        assert self.structure_loaded, "Structure is not loaded."

        crop_method_prob = [
            contiguous_crop_weight,
            spatial_crop_weight,
            interface_crop_weight,
        ]

        method = random.choices(
            ["contiguous", "spatial", "interface"], weights=crop_method_prob, k=1
        )[0]

        if len(self.structure.residue_chain_break) == 1 and method == "interface":
            # If there is only one chain, use spatial crop instead of interface crop
            method = "spatial"

        if method == "contiguous":
            # 20250620, Change contiguous crop to monomer version
            try:
                crop_indices, crop_chain = crop_contiguous_monomer(
                    chain_bias, self.structure, crop_length, level=level
                )
            except NoValidChainsError:
                # print("No valid chains found. Using spatial crop instead of contiguous crop.")
                crop_indices, crop_chain = crop_spatial(
                    chain_bias, self.structure, crop_length, level=level
                )
        elif method == "spatial":
            if monomer_only:
                crop_indices, crop_chain = crop_spatial(
                    chain_bias, self.structure, crop_length
                )
            else:
                crop_indices, crop_chain = crop_spatial(
                    chain_bias, self.structure, crop_length, level=level
                )
        elif method == "interface":
            if monomer_only:
                raise ValueError("Interface crop is not supported for monomer only.")
            try:
                crop_indices, crop_chain = crop_spatial_interface(
                    interface_bias, self.structure, crop_length, level=level
                )
            except NoInterfaceError:
                # print(
                #     "No interface found. Using spatial crop instead of interface crop."
                # )
                crop_indices, crop_chain = crop_spatial(
                    chain_bias, self.structure, crop_length, level=level
                )

        crop_sequence_hash = {
            chain: self.structure.sequence_hash[chain] for chain in crop_chain
        }

        seq_hash_to_crop_indices = {}
        for chain, seq_hash in crop_sequence_hash.items():
            if seq_hash not in seq_hash_to_crop_indices:
                seq_hash_to_crop_indices[seq_hash] = []
            seq_hash_to_crop_indices[seq_hash].append(crop_chain[chain])

        return crop_indices, seq_hash_to_crop_indices

    def crop(
        self,
        crop_indices: torch.Tensor,
        crop_MSA: bool = False,
        use_MSADB: bool = True,
    ) -> None:
        if crop_MSA:
            msa_list = []
            chain_crop = get_chain_crop_indices(
                self.structure.residue_chain_break, crop_indices
            )
            seq_hashes = [
                self.structure.sequence_hash[chain_id] for chain_id in chain_crop.keys()
            ]
            seq_hashes = [seq_hash.zfill(6) for seq_hash in seq_hashes]

            if use_MSADB:
                hash_to_MSA = {
                    seq_hash: read_MSA_lmdb(seq_hash) for seq_hash in set(seq_hashes)
                }
            else:
                hash_to_MSA = {
                    seq_hash: MSA(seq_hash, use_lmdb=True)
                    for seq_hash in set(seq_hashes)
                }

            for chain_id, chain_crop_indices in chain_crop.items():
                seq_hash = self.structure.sequence_hash[chain_id]
                # Ex) 21022 -> 021022
                seq_hash = seq_hash.zfill(6)
                msa = copy.deepcopy(hash_to_MSA[seq_hash])
                msa.crop(chain_crop_indices.numpy())
                msa_list.append(msa)
            complex_msa = ComplexMSA(msa_list)
            self.MSA = complex_msa
        self.structure.crop(crop_indices)
