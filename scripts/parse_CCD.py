from pathlib import Path
import sys

from biomol.io.parsers.cif_parser import parse


def main():
    if len(sys.argv) < 3:
        print("Usage: python parse_CCD.py <recipe_path> <path_to_cif>")
        sys.exit(1)

    recipe_path = sys.argv[1].strip()
    cif_path = sys.argv[2].strip()

    parsed_data = parse(
        recipe_path=Path(recipe_path),
        file_path=Path(cif_path),
    )
    return parsed_data


if __name__ == "__main__":
    # python ./scripts/parse_CCD.py ./plans/ccd_recipe_book.py /public_data/CCD/components_tmp/0/01R.cif
    main()
