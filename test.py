from BioMol.BioMol import BioMol
import time

if __name__ == "__main__":
    start = time.time()
    biomol = BioMol(
        pdb_ID="1ubq",
        mol_types=["protein"],
    )
    biomol.choose("1", "1", ".")
    print(f"loading time: {time.time() - start:.2f} seconds")
    biomol.structure.to_mmcif("test.cif")
    breakpoint()
