from BioMol.BioMol import BioMol

if __name__ == "__main__":
    # BioMol automatically generate all biological assemblies.
    biomol = BioMol(
        pdb_ID="9dw6", # lower case
        mol_types=["protein"], # only protein
    )
    # biomol = BioMol(
    #     cif="5hlt.cif", # This file should be from the PDB database.
    #     mol_types=["protein","nucleic_acid", "ligand"],
    #     remove_signal_peptide=True,
    #     use_lmdb=False, # If you want to load NA or ligand you must set use_lmdb=False,
    # )

    # choose assembly id, model_id, alt_id
    biomol.choose("1", "1", ".")

    crop_indices, seq_hash_to_crop_indices = biomol.get_crop_indices()
    biomol.crop(seq_hash_to_crop_indices)


    breakpoint()

    # If you want to save loaded structure, you can use the following code.
    biomol.structure.to_mmcif("loaded_5hlt.cif") # save the loaded structure

    # # Cropping and loading MSA
    # biomol.crop_and_load_msa(
    #     chain_bias = ('A_1'),
    #     interaction_bias=('A_1','C_1'),
    #     crop_method_prob = [0.2, 0.4, 0.4],
    #     crop_length=384,
    # )
