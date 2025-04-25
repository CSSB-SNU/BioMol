from BioMol.BioMol import BioMol


if __name__ == "__main__":
    biomol = BioMol(
        cif_ID="1v9u",
        mol_types=["protein"],
        remove_signal_peptide=True,
        use_lmdb=True,
    )
    biomol.choose("1", "1", ".")
    breakpoint()
    biomol.choose("1", "1", ".")
    biomol.structure.to_mmcif("test.cif")
