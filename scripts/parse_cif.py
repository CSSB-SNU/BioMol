import sys
from pathlib import Path

from Bio.PDB.MMCIF2Dict import MMCIF2Dict as mmcif2dict  # noqa: N813

from biomol.cif import CIFMol
from biomol.core.utils import to_dict
from biomol.io.cache import ParsingCache
from biomol.io.cooker import Cooker


def get_cif_data(cif_path: Path) -> dict:
    """Parse a CIF file and return its data as a dictionary."""
    if cif_path.suffix == ".gz":
        import gzip

        with gzip.open(cif_path, "rt") as f:
            cif_raw_data = mmcif2dict(f)
    elif cif_path.suffix == ".cif":
        cif_raw_data = mmcif2dict(cif_path)
    else:
        msg = f"Unsupported file format: {cif_path}"
        raise ValueError(msg)
    # Reformat the mmcif2dict output to a more organized structure
    # into a nested dictionary: {key1 : {key2: [values]}}
    organized_dict = {}
    key_list = list(cif_raw_data.keys())
    for key in key_list:
        if "." not in key:
            organized_dict[key] = cif_raw_data[key]
            continue
        main_key, sub_key = key.split(".")
        if main_key not in organized_dict:
            organized_dict[main_key] = {}
        organized_dict[main_key][sub_key] = cif_raw_data[key]
    return organized_dict


def dot_transform(key: str) -> list[str]:
    """Transform a dot-separated key into a list of keys."""
    return key.split(".")


def parse(
    ccd_db_path: Path,
    recipe_path: Path,
    cif_path: Path,
    targets: list[str] | None = None,
) -> dict:
    """Parse a CIF file using a predefined recipe."""
    cif_data = get_cif_data(cif_path)
    cif_data["ccd_db_path"] = ccd_db_path
    parse_cache = ParsingCache(dot_transform)
    cooker = Cooker(parse_cache=parse_cache, recipebook=str(recipe_path))
    cooker.prep(cif_data, fields=list(cif_data.keys()))
    cooker.cook()
    return cooker.serve(targets=targets)


def main() -> None:
    if len(sys.argv) < 4:
        print(
            "Usage: python parse_cif.py <CCD_DB> <recipe_path> <path_to_cif> <target1> <target2> ..."
        )
        sys.exit(1)

    ccd_db_path = Path(sys.argv[1].strip())
    recipe_path = Path(sys.argv[2].strip())
    cif_path = Path(sys.argv[3].strip())
    targets = [arg.strip() for arg in sys.argv[4:]]
    if len(targets) == 0:
        targets = None
    if not recipe_path.is_file():
        print(f"Error: Recipe file '{recipe_path}' does not exist.")
        sys.exit(1)
    if not cif_path.is_file():
        print(f"Error: CIF file '{cif_path}' does not exist.")
        sys.exit(1)
    result = parse(ccd_db_path, recipe_path, cif_path, targets=targets)
    result = to_dict(result)

    value, metadata = result["assembly_dict"], result["metadata_dict"]
    cifmol_dict = {}
    for cif_key in value:
        item = value[cif_key]
        # model_id = cif_key.
        assembly_id, model_id, alt_id = cif_key.split("_")
        metadata["assembly_id"] = assembly_id
        metadata["model_id"] = model_id
        metadata["alt_id"] = alt_id
        item["metadata"] = metadata

        cifmol_dict[cif_key] = CIFMol.from_dict(item)
        cifmol_dict[cif_key].to_cif(f"test_{cif_key}.cif")
    breakpoint()


if __name__ == "__main__":
    main()
    # python scripts/parse_cif.py /public_data/CCD/biomol_CCD.lmdb/ plans/cif_recipe_book.py /public_data/BioMolDB_2024Oct21/cif/cif_raw/e2/2e27.cif.gz
