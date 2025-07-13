from BioMol.BioMol import BioMol
from BioMol import DB_PATH
from BioMol.utils.parser import parse_cif

if __name__ == "__main__":
    # BioMol automatically generate all biological assemblies.
    biomol = BioMol(
        # pdb_ID="6nu2", # lower case
        # cif=f"{DB_PATH}/cif/cif_raw/an/1an2.cif.gz",
        cif=f"{DB_PATH}/cif/cif_raw/s3/6s3z.cif.gz",
        # cif=f"{DB_PATH}/cif/cif_raw/b0/5b0w.cif.gz",
        remove_signal_peptide=True,
        mol_types=["protein", "nucleic_acid", "ligand"],  # only protein
        use_lmdb=False,
    )
    # biomol = BioMol(
    #     cif="5hlt.cif", # This file should be from the PDB database.
    #     mol_types=["protein","nucleic_acid", "ligand"],
    #     remove_signal_peptide=True,
    #     use_lmdb=False, # If you want to load NA or ligand you must set use_lmdb=False,
    # )
    biomol.choose("1", "1", ".")
    crop_indices, seq_hash_to_crop_indices = biomol.get_crop_indices(
        # chain_bias="A_1",
        contiguous_crop_weight=0.0,
        spatial_crop_weight=0.0,
        interface_crop_weight=1.0,
        crop_length=2560,
        level="atom"
    )
    breakpoint()

    biomol.crop(crop_indices)
    breakpoint()
    biomol.structure.to_mmcif("6s3z_cropped.cif")  # save the loaded structure

    breakpoint()

    bioassembly = parse_cif(
        cif_path=f"{DB_PATH}/cif/vy/2vy1.cif.gz",
        cif_configs="./BioMol/configs/types/base.json",
    )
    breakpoint()

    # biomol.choose("1", "1", "B")
    breakpoint()

    # choose assembly id, model_id, alt_id

    crop_indices, seq_hash_to_crop_indices = biomol.get_crop_indices()
    biomol.crop(seq_hash_to_crop_indices)

    breakpoint()

    # If you want to save loaded structure, you can use the following code.
    biomol.structure.to_mmcif("loaded_5hlt.cif")  # save the loaded structure

    # # Cropping and loading MSA
    # biomol.crop_and_load_msa(
    #     chain_bias = ('A_1'),
    #     interaction_bias=('A_1','C_1'),
    #     crop_method_prob = [0.2, 0.4, 0.4],
    #     crop_length=384,
    # )
