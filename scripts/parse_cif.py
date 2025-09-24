import sys

from Bio.PDB.MMCIF2Dict import MMCIF2Dict as mmcif2dict  # noqa: N813

from biomol.io.cache import ParsingCache
from biomol.io.cooker import Cooker


def get_cif_data(cif_path: str) -> dict:
    """Parse a CIF file and return its data as a dictionary."""
    if cif_path.endswith(".gz"):
        import gzip

        with gzip.open(cif_path, "rt") as f:
            cif_raw_data = mmcif2dict(f)
    elif cif_path.endswith(".cif"):
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
    return key.split(".")


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python parse_CCD.py <recipe_path> <path_to_cif>")
        sys.exit(1)

    recipe_path = sys.argv[1].strip()
    cif_path = sys.argv[2].strip()

    cif_data = get_cif_data(cif_path)
    parse_cache = ParsingCache(dot_transform)
    cooker = Cooker(parse_cache=parse_cache, recipebook=recipe_path)
    cooker.prep(cif_data, fields=list(cif_data.keys()))
    cooker.cook()
    assembly_dict = cooker.serve(targets="assembly_dict")
    # cooker.parse_cache._storage['_pdbx_poly_seq_scheme_dict']['A'].keys()
    breakpoint()


if __name__ == "__main__":
    main()
