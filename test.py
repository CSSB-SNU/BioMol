from BioMol.BioMol import BioMol
import time

if __name__ == "__main__":
    start = time.time()
    biomol = BioMol(
        pdb_ID="9dw6",
        mol_types=["protein"],
    )
    biomol.choose("1", "1", ".")
    print(f"loading time: {time.time() - start:.2f} seconds")
    biomol.structure.to_mmcif("9dw6.cif")

    biomol.crop_and_load_msa(
        interaction_bias=('A_1','C_1'),
        params= {
            "method_prob": [0.0, 0.0, 1.0],
            "crop_size": 35,
        }
    )

    biomol.structure.to_mmcif("9dw6_cropped.cif")
    breakpoint()
