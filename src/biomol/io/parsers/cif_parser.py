import sys
from pathlib import Path

from Bio.PDB.MMCIF2Dict import MMCIF2Dict as mmcif2dict  # noqa: N813

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
    ccd_db_path: Path | None,
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
