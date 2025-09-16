import sys

from Bio.PDB.MMCIF2Dict import MMCIF2Dict as mmcif2dict

from biomol.io.blueprint import Blueprint
from biomol.io.factory import BioMolFactory

import biomol.io.instructions.common_instructions


def main():
    if len(sys.argv) < 3:
        print("Usage: python parse_CCD.py <plan_path> <path_to_cif_list>")
        sys.exit(1)

    plan_path = sys.argv[1].strip()
    cifs_path = sys.argv[2].strip()

    with open(cifs_path, "r") as f:
        cif_list = [line.strip() for line in f]

    factory = BioMolFactory(blueprint=plan_path, num_workers=4)

    cif_data: list[dict] = [mmcif2dict(cif) for cif in cif_list]

    biomol_list = factory.produce(dataset=cif_data)

    if biomol_list:
        print("Successfully parsed BioMol object:")
        for biomol in biomol_list:
            print(biomol)


if __name__ == "__main__":
    main()
