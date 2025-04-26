from BioMol.BioMol import BioMol

if __name__ == "__main__":
    # BioMol automatically generate all biological assemblies.
    biomol = BioMol(
        pdb_ID="9dw6", # lower case
        mol_types=["protein"], # only protein
    )
    # biomol = BioMol(
    #     cif="9dw6.cif", # This file should be from the PDB database.
    #     mol_types=["protein","nucleic_acid", "ligand"],
    #     remove_signal_peptide=True,
    #     use_lmdb=False, # If you want to load NA or ligand you must set use_lmdb=False,
    # )

    # choose assembly id, model_id, alt_id
    biomol.choose("1", "1", ".")

    # If you want to save loaded structure, you can use the following code.
    biomol.structure.to_mmcif("loaded_9dw6.cif") # save the loaded structure

    # Cropping and loading MSA
    biomol.crop_and_load_msa(
        chain_bias = ('A_1'),
        interaction_bias=('A_1','C_1'),
        params= {
            "method_prob": [0.0, 0.0, 1.0], # Contiguous, spaital, interface
            "crop_size": 384,
        }
    )
