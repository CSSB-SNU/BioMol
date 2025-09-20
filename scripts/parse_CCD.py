import sys

from Bio.PDB.MMCIF2Dict import MMCIF2Dict as mmcif2dict

from biomol.io.cooker import Cooker
from biomol.io.cache import ParsingCache


def main():
    if len(sys.argv) < 3:
        print("Usage: python parse_CCD.py <recipe_path> <path_to_cif>")
        sys.exit(1)

    recipe_path = sys.argv[1].strip()
    cif_path = sys.argv[2].strip()

    cif_data = mmcif2dict(cif_path)
    breakpoint()
    cooker = Cooker(parse_cache=ParsingCache(), recipebook=recipe_path)
    cooker.prep(cif_data, fields=list(cif_data.keys()))
    cooker.cook()

    target_id = cooker.recipebook["id"]
    print(target_id)


if __name__ == "__main__":
    main()

