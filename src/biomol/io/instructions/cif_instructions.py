import re
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from biomol.core.container import FeatureContainer
from biomol.core.feature import EdgeFeature, NodeFeature
from biomol.core.index import IndexTable
from biomol.core.utils import concat_containers
from biomol.db.lmdb_handler import read_lmdb
from biomol.exceptions import IndexMismatchError

InputType = TypeVar("InputType", str, int, float)
FeatureType = TypeVar("FeatureType")
NumericType = TypeVar("NumericType", int, float)


def single_value_instruction(
    *,
    dtype: type[InputType],
) -> Callable[..., type[InputType]]:
    """
    Return a configured instruction function that maps fields to node features.

    The returned function 'remembers' the dtype via closure.
    """

    def _worker(
        data: list[InputType] | NDArray,
    ) -> type[InputType]:
        formatted_data = [dtype(datum) for datum in data]
        if len(formatted_data) != 1:
            msg = f"Expected single value, got {len(formatted_data)}"
            raise ValueError(msg)
        return formatted_data[0]

    return _worker


def extract_float_single(*args: str | None) -> float | None:
    """Extract a single valid float from multiple string-like inputs."""
    pattern = re.compile(r"^-?\d+(?:\.\d+)?$")

    # Build mask: True if value is a valid float string
    mask = []
    for a in args:
        if a is None:
            mask.append(False)
        elif isinstance(a, (int, float)):
            mask.append(True)
        elif isinstance(a, str):
            mask.append(bool(pattern.match(a)))
        else:
            mask.append(False)
    mask = np.array(mask)

    # Check conditions
    n_valid = np.sum(mask)
    if n_valid == 0:
        return None
    if n_valid > 1:
        msg = "More than one input is a valid float"
        raise ValueError(msg)

    # Return the parsed float
    valid_idx = np.where(mask)[0].item()
    return float(args[valid_idx])


def key_stack() -> Callable[..., type[InputType]]:
    """Return a configured instruction function that maps fields to node features."""

    def _worker(
        **kwargs: list[InputType] | NDArray,
    ) -> type[InputType]:
        return kwargs

    return _worker


def get_smaller_dict(
    *,
    dtype: type[InputType],
) -> Callable[..., dict[str, dict[str, NDArray]]]:
    """
    Group rows of cif_raw_dict by tied_to column(s).

    Parameters
    ----------
    cif_raw_dict : dict[str, list[str]]
        Column -> values (all lists must be same length)
    tied_to : str | tuple[str, str]
        Column(s) used for grouping.
        If tuple, the two values are joined with "|" to form the key.

    Returns
    -------
    dict[str, dict[str, list[str]]]
        Outer dict: group key -> inner dict (col -> list of values).
    """

    def _worker(
        cif_raw_dict: dict[str, list[str]],
        tied_to: str | tuple[str, str],
        columns: list[str] | None = None,
    ) -> dict[str, dict[str, NDArray]]:
        if not cif_raw_dict:
            return {}

        n = len(next(iter(cif_raw_dict.values())))
        cols = set(cif_raw_dict.keys())
        if columns is not None:
            cols = cols & set(columns)
        cols = list(
            cols - {tied_to}
            if isinstance(tied_to, str)
            else cols - {tied_to[0], tied_to[1]},
        )

        result: dict[str, dict[str, NDArray]] = {}

        for i in range(n):
            row = {col: cif_raw_dict[col][i] for col in cols}

            if isinstance(tied_to, str):
                key = cif_raw_dict[tied_to][i]
            else:
                key = (cif_raw_dict[tied_to[0]][i], cif_raw_dict[tied_to[1]][i])

            if key not in result:
                result[key] = {col: [] for col in cols}

            for col in cols:
                if col == tied_to:
                    continue
                result[key][col].append(row[col])

        # convert lists to arrays
        for outer_dict in result.values():
            for col in outer_dict:
                outer_dict[col] = np.array(outer_dict[col], dtype=dtype)
        return result

    return _worker


def merge_dict() -> Callable[..., type[InputType]]:
    """
    Return a configured instruction function that maps fields to node features.

    The returned function 'remembers' the dtype via closure.
    """

    def _worker(
        *args: list[InputType] | NDArray,
    ) -> type[InputType]:
        _merged_dict = {}
        for _dict in args:
            if _dict is None:
                continue
            for row1, inner_dict in _dict.items():
                if row1 not in _merged_dict:
                    _merged_dict[row1] = {}
                for col, values in inner_dict.items():
                    if col in _merged_dict[row1]:
                        msg = f"Duplicate column '{col}' for row '{row1}'"
                        raise ValueError(msg)
                    _merged_dict[row1][col] = values
        return _merged_dict

    return _worker


def parse_chem_comp() -> Callable[..., dict[str, FeatureContainer]]:
    """
    Return a configured instruction function that maps fields to node features.

    The returned function 'remembers' the dtype via closure.
    """

    def _parse_each_chem_comp(
        chem_comp_dict: dict[str, NDArray] | None,
        chem_comp_atom_dict: dict[str, NDArray] | None,
        chem_comp_bond_dict: dict[str, NDArray] | None,
        remove_hydrogen: bool = True,
    ) -> dict[str, dict[str, NDArray]]:
        name = NodeFeature(
            value=chem_comp_dict["name"].astype(str),
            description="name of the chemical component",
        )
        formula = NodeFeature(
            value=chem_comp_dict["formula"].astype(str),
            description="chemical formula of the component",
        )
        elements = chem_comp_atom_dict["type_symbol"]
        if remove_hydrogen:
            atom_mask = ~np.isin(elements, ["H", "D"])
        else:
            atom_mask = np.array([True] * len(elements))

        atom_id = chem_comp_atom_dict["atom_id"][atom_mask].astype(str)
        element = elements[atom_mask].astype(str)
        atom_aromatic = chem_comp_atom_dict["pdbx_aromatic_flag"][atom_mask].astype(str)
        atom_stereo = chem_comp_atom_dict["pdbx_stereo_config"][atom_mask].astype(str)

        charge = chem_comp_atom_dict.get("charge", None)
        model_x = chem_comp_atom_dict.get("model_Cartn_x", None)
        model_y = chem_comp_atom_dict.get("model_Cartn_y", None)
        model_z = chem_comp_atom_dict.get("model_Cartn_z", None)

        atom_id = NodeFeature(
            value=atom_id,
            description="atom id of the chemical component",
        )
        element = NodeFeature(
            value=element,
            description="element symbol of the chemical component",
        )
        atom_aromatic = NodeFeature(
            value=atom_aromatic,
            description="aromatic flag of the atom",
        )
        atom_stereo = NodeFeature(
            value=atom_stereo,
            description="stereochemistry flag of the atom",
        )

        atom_node_features = {
            "atom_id": atom_id,
            "element": element,
            "atom_aromatic": atom_aromatic,
            "atom_stereo": atom_stereo,
        }

        if charge is not None:
            charge = NodeFeature(
                value=charge[atom_mask].astype(str),
                description="formal charge of the atom",
            )
            atom_node_features["charge"] = charge
        if model_x is not None:
            xyz = np.stack(
                [
                    model_x[atom_mask].astype(str),
                    model_y[atom_mask].astype(str),
                    model_z[atom_mask].astype(str),
                ],
                axis=-1,
            )
            model_xyz = NodeFeature(
                value=xyz,
                description="Cartesian coordinates of the atom",
            )
            atom_node_features["model_xyz"] = model_xyz

        # handle edge features
        if chem_comp_bond_dict is not None:
            src, dst = (
                chem_comp_bond_dict["atom_id_1"],
                chem_comp_bond_dict["atom_id_2"],
            )
            bond_type = chem_comp_bond_dict["value_order"].astype(str)
            bond_stereo = chem_comp_bond_dict["pdbx_stereo_config"].astype(str)
            bond_aromatic = chem_comp_bond_dict["pdbx_aromatic_flag"].astype(str)

            order = np.argsort(atom_id.value)

            # mask out src, dst that are not in atom_id (e.g. hydrogen if removed)
            valid_src_mask = np.isin(src, atom_id.value)
            valid_dst_mask = np.isin(dst, atom_id.value)
            valid_mask = valid_src_mask & valid_dst_mask
            src = src[valid_mask]
            dst = dst[valid_mask]
            bond_type = bond_type[valid_mask]
            bond_stereo = bond_stereo[valid_mask]
            bond_aromatic = bond_aromatic[valid_mask]

            src_indices = order[np.searchsorted(atom_id.value, src, sorter=order)]
            dst_indices = order[np.searchsorted(atom_id.value, dst, sorter=order)]

            bond_type = EdgeFeature(
                value=bond_type,
                src_indices=src_indices,
                dst_indices=dst_indices,
                description="bond type of the chemical component",
            )

            bond_aromatic = EdgeFeature(
                value=bond_aromatic,
                src_indices=src_indices,
                dst_indices=dst_indices,
                description="aromatic flag of the bond",
            )

            bond_stereo = EdgeFeature(
                value=bond_stereo,
                src_indices=src_indices,
                dst_indices=dst_indices,
                description="stereochemistry flag of the bond",
            )
            edge_features = {
                "bond_type": bond_type,
                "bond_aromatic": bond_aromatic,
                "bond_stereo": bond_stereo,
            }
        else:
            edge_features = {}

        residue_container = FeatureContainer(
            node_features={
                "name": name,
                "formula": formula,
            },
            edge_features={},
        )

        atom_container = FeatureContainer(
            node_features=atom_node_features,
            edge_features=edge_features,
        )

        return {
            "residue": residue_container,
            "atom": atom_container,
        }

    def _worker(
        chem_comp_dict: dict[str, dict[str, NDArray]] | None,
        chem_comp_atom_dict: dict[str, dict[str, NDArray]] | None,
        chem_comp_bond_dict: dict[str, dict[str, NDArray]] | None,
        remove_hydrogen: bool = True,
        unwrap: bool = False,
    ) -> dict[str, dict[str, NDArray]]:
        output = {}
        for chem_comp_id in chem_comp_dict:
            if chem_comp_id == "UNL":
                output[chem_comp_id] = None
                continue
            if chem_comp_id not in chem_comp_atom_dict:
                # some cif files do not have chem_comp_atom.
                output[chem_comp_id] = None
                continue
            _chem_comp_dict = chem_comp_dict[chem_comp_id]
            _chem_comp_atom_dict = chem_comp_atom_dict[chem_comp_id]
            _chem_comp_bond_dict = chem_comp_bond_dict.get(chem_comp_id, None)
            parsed = _parse_each_chem_comp(
                _chem_comp_dict,
                _chem_comp_atom_dict,
                _chem_comp_bond_dict,
                remove_hydrogen,
            )
            output[chem_comp_id] = parsed
        if unwrap:
            assert len(output) == 1, (
                "unwrap is only valid when there is a single chem_comp_id"
            )
            return next(iter(output.values()))
        return output

    return _worker


def compare_chem_comp() -> Callable[..., dict[str, FeatureContainer]]:
    """Compare and merge cif_chem_comp with ideal_chem_comp from CCD database."""

    def _compare_each_chem_comp(
        cif_chem_comp: dict[str, FeatureContainer],
        ideal_chem_comp: dict[str, FeatureContainer],
    ) -> dict[str, NDArray]:
        output = {}
        # for residue, just follow cif_chem_comp if exists
        if len(cif_chem_comp["residue"]) != 0:
            output["residue"] = cif_chem_comp["residue"]
        else:
            output["residue"] = ideal_chem_comp["residue"]
        ideal_atom_dict = ideal_chem_comp["atom"].to_dict()
        ideal_node_key_list = list(ideal_atom_dict["nodes"].keys())
        ideal_edge_key_list = list(ideal_atom_dict["edges"].keys())
        atom_node_features = {}
        atom_edge_features = {}
        for key in ideal_node_key_list:
            if key in cif_chem_comp["atom"].node_features:
                # follow cif_chem_comp if exists
                atom_node_features[key] = cif_chem_comp["atom"].node_features[key]
            else:
                atom_node_features[key] = ideal_chem_comp["atom"].node_features[key]
        for key in ideal_edge_key_list:
            if key in cif_chem_comp["atom"].edge_features:
                atom_edge_features[key] = cif_chem_comp["atom"].edge_features[key]
            else:
                atom_edge_features[key] = ideal_chem_comp["atom"].edge_features[key]
        try:
            output["atom"] = FeatureContainer(
                node_features=atom_node_features,
                edge_features=atom_edge_features,
            )
        except IndexMismatchError:
            output["atom"] = ideal_chem_comp["atom"]
        return output

    def _worker(
        chem_comp_dict: dict[str, dict[str, NDArray]] | None,
        ccd_db_path: Path | None,
    ) -> dict[str, dict[str, NDArray]]:
        output = {}
        for chem_comp_id in chem_comp_dict:
            if chem_comp_id == "UNL":
                output[chem_comp_id] = None
                continue
            ideal_chem_comp = read_lmdb(ccd_db_path, chem_comp_id)
            ideal_chem_comp = {
                "atom": FeatureContainer.from_dict(ideal_chem_comp["atom"]),
                "residue": FeatureContainer.from_dict(ideal_chem_comp["residue"]),
            }
            cif_chem_comp = chem_comp_dict[chem_comp_id]
            if cif_chem_comp is None:
                output[chem_comp_id] = ideal_chem_comp
                continue
            parsed = _compare_each_chem_comp(cif_chem_comp, ideal_chem_comp)
            output[chem_comp_id] = parsed
        return output

    return _worker


def parse_scheme_dict() -> Callable[..., dict[str, dict[str, NDArray]] | None]:
    """Parse a scheme dict to ensure it has the expected structure."""

    def _parse_each_asym_id(
        scheme_dict: dict[str, dict[str, NDArray]] | None,
    ) -> dict[str, NDArray] | None:
        entity_id = scheme_dict["entity_id"]
        if "seq_id" in scheme_dict:  # polymer
            cif_idx = scheme_dict["seq_id"]
        elif "num" in scheme_dict:  # branched
            cif_idx = scheme_dict["num"]
        else:  # non-polymer
            cif_idx = None
        chem_comp = scheme_dict["mon_id"]
        auth_idx = scheme_dict["pdb_seq_num"]
        ins_code = scheme_dict.get("pdb_ins_code", None)
        hetero = scheme_dict.get("hetero", None)

        if cif_idx is not None:
            cif_idx = cif_idx.astype(int)
        if hetero is None:
            hetero = np.zeros(len(auth_idx), dtype=int)
        else:
            hetero = np.array([1 if h in {1, "y", "yes"} else 0 for h in hetero])
        if ins_code is None:
            ins_code = np.array(["." for _ in range(len(auth_idx))])
        auth_idx = [
            aa if ii == "." else f"{aa}.{ii}"
            for aa, ii in zip(auth_idx, ins_code, strict=True)
        ]
        # handle heterogeneous sequence
        auth_idx_to_chem_comp = {}
        auth_idx_to_hetero = {}
        new_auth_idx_list = []
        new_cif_idx_list = []
        for idx in range(len(auth_idx)):
            _chem_comp = chem_comp[idx]
            _cif_idx = cif_idx[idx] if cif_idx is not None else -1
            _auth_idx = auth_idx[idx]
            _hetero = hetero[idx]

            if _auth_idx not in auth_idx_to_chem_comp:
                auth_idx_to_chem_comp[_auth_idx] = []
                auth_idx_to_hetero[_auth_idx] = []
            auth_idx_to_chem_comp[_auth_idx].append(_chem_comp)
            auth_idx_to_hetero[_auth_idx].append(_hetero)
            if _auth_idx not in new_auth_idx_list:
                new_auth_idx_list.append(_auth_idx)
                new_cif_idx_list.append(_cif_idx)
        if cif_idx is not None:
            cif_idx = np.array(new_cif_idx_list)
        else:
            cif_idx = np.arange(len(new_auth_idx_list)) + 1

        chem_comp = list(auth_idx_to_chem_comp.values())
        max_hetero = max(len(cc) for cc in chem_comp)
        chem_comp = [cc + [""] * (max_hetero - len(cc)) for cc in chem_comp]
        chem_comp = np.array(chem_comp, dtype=str)
        hetero = np.array([1 if 1 in h else 0 for h in auth_idx_to_hetero.values()])
        auth_idx = np.array(new_auth_idx_list)
        return {
            "entity_id": entity_id,
            "cif_idx": cif_idx.astype(str),
            "auth_idx": auth_idx.astype(str),
            "chem_comp": chem_comp,
            "hetero": hetero.astype(bool),
        }

    def _worker(
        asym_scheme_dict: dict[str, dict[str, NDArray]]
        | None,  # {asym_id : scheme_dict}
    ) -> dict[str, NDArray] | None:
        if asym_scheme_dict is None:
            return None
        output = {}
        for asym_id, scheme_dict in asym_scheme_dict.items():
            parsed = _parse_each_asym_id(scheme_dict)
            output[asym_id] = parsed
        return output

    return _worker


def parse_entity_dict() -> Callable[..., dict[str, dict[str, NDArray]] | None]:
    """Parse a entity dict to ensure it has the expected structure."""

    def _parse_each_entity(
        entity_dict: dict[str, dict[str, NDArray]] | None,
    ) -> dict[str, NDArray] | None:
        entity_type = None
        if "mon_id" in entity_dict:
            entity_type = "polymer"
        elif "comp_id" in entity_dict and "num" in entity_dict:
            entity_type = "branched"
        else:
            entity_type = "non-polymer"

        if entity_type in {"polymer", "branched"}:  # Polymer or Branched
            seq_num_list = entity_dict["num"]
            chem_comp_list = (
                entity_dict["mon_id"]
                if "mon_id" in entity_dict
                else entity_dict["comp_id"]
            )
            hetero_list = entity_dict["hetero"]
        else:  # Non-polymer
            seq_num_list = np.array([1])
            chem_comp_list = entity_dict["comp_id"]
            hetero_list = np.array([0])

        hetero_list = np.array(
            [1 if h in {"y", "yes"} else 0 for h in hetero_list],
        )

        seq_num_to_chem_comp = {}
        seq_num_to_hetero = {}
        for seq_num, chem_comp, hetero in zip(
            seq_num_list,
            chem_comp_list,
            hetero_list,
            strict=True,
        ):
            if seq_num not in seq_num_to_chem_comp:
                seq_num_to_chem_comp[seq_num] = []
                seq_num_to_hetero[seq_num] = []
            seq_num_to_chem_comp[seq_num].append(chem_comp)
            seq_num_to_hetero[seq_num].append(hetero)

        seq_num_list = np.array(list(seq_num_to_chem_comp.keys()))
        chem_comp_list = list(seq_num_to_chem_comp.values())
        max_hetero = max(len(cc) for cc in chem_comp_list)
        chem_comp_list = [cc + [""] * (max_hetero - len(cc)) for cc in chem_comp_list]
        chem_comp_list = np.array(chem_comp_list, dtype=str)
        hetero = np.array([1 if 1 in h else 0 for h in seq_num_to_hetero.values()])

        if entity_type == "polymer":
            one_letter_code_can = entity_dict["pdbx_seq_one_letter_code_can"][0]
            one_letter_code = entity_dict["pdbx_seq_one_letter_code"][0]
            one_letter_code_can = one_letter_code_can.replace("\n", "")
            one_letter_code = one_letter_code.replace("\n", "")
            one_letter_code_can = np.array(list(one_letter_code_can))
            one_letter_code = np.array(list(one_letter_code))
        else:  # Non-polymer or Branched
            one_letter_code_can = np.array(["X" for _ in chem_comp_list])
            one_letter_code = one_letter_code_can

        if len(one_letter_code) != len(one_letter_code_can):
            one_letter_code = "".join(one_letter_code)
            sequence_split = re.findall(r"\(.*?\)|.", one_letter_code)
            one_letter_code = np.array(sequence_split)
            if len(sequence_split) != len(one_letter_code_can):
                sequence_split = [
                    seq if "(" not in seq else "X" for seq in sequence_split
                ]
                one_letter_code_can = np.array(sequence_split)

        one_letter_code_can = np.array(one_letter_code_can)
        one_letter_code = np.array(one_letter_code)

        if entity_type == "branched":
            descriptor = (
                entity_dict["descriptor"] if "descriptor" in entity_dict else ""
            )
        elif entity_type == "polymer":
            descriptor = entity_dict["type"]
        else:
            descriptor = chem_comp_list[0]  # len 1.

        if entity_type == "branched":
            branch_link = {}

            comp_id_1 = entity_dict.get("comp_id_1", None)
            comp_id_2 = entity_dict.get("comp_id_2", None)
            atom_id_1 = entity_dict.get("atom_id_1", None)
            atom_id_2 = entity_dict.get("atom_id_2", None)
            leaving_atom_id_1 = entity_dict.get("leaving_atom_id_1", None)
            leaving_atom_id_2 = entity_dict.get("leaving_atom_id_2", None)
            bond_list = entity_dict.get("value_order", None)
            num_1_list = entity_dict.get("entity_branch_list_num_1", None)
            num_2_list = entity_dict.get("entity_branch_list_num_2", None)

            if type(comp_id_1) is str:
                comp_id_1 = [comp_id_1]

            for idx in range(len(comp_id_1)):
                _comp_id_1 = comp_id_1[idx]
                _comp_id_2 = comp_id_2[idx]
                _atom_id_1 = atom_id_1[idx]
                _atom_id_2 = atom_id_2[idx]
                _leaving_atom_id_1 = leaving_atom_id_1[idx]
                _leaving_atom_id_2 = leaving_atom_id_2[idx]
                _bond = bond_list[idx]
                _num_1 = num_1_list[idx]
                _num_2 = num_2_list[idx]

                branch_link[(str(_num_1), str(_num_2))] = {
                    "atom_id": (_atom_id_1, _atom_id_2),
                    "leaving_atom_id": (_leaving_atom_id_1, _leaving_atom_id_2),
                    "bond": _bond,
                    "comp_id": (_comp_id_1, _comp_id_2),
                }
        else:
            branch_link = None
        return {
            "seq_num": seq_num_list.astype(int),
            "chem_comp": chem_comp_list,
            "hetero": hetero_list.astype(bool),
            "one_letter_code_can": one_letter_code_can,
            "one_letter_code": one_letter_code,
            "descriptor": descriptor,
            "branch_link": branch_link,
        }

    def _worker(
        entity_dict: dict[str, dict[str, NDArray]] | None,  # {asym_id : scheme_dict}
    ) -> dict[str, NDArray] | None:
        output = {}
        for entity_id, _entity_dict in entity_dict.items():
            parsed = _parse_each_entity(_entity_dict)
            output[entity_id] = parsed
        return output

    return _worker


def remove_unknown_atom_site() -> Callable[..., type[InputType]]:
    """Remove UNL residues from atom_site_dict."""

    def _worker(
        atom_site_dict: dict[str, dict[str, NDArray]] | None,
    ) -> dict[str, dict[str, NDArray]] | None:
        new_dict = {}
        for asym_id in atom_site_dict:
            label_comp_id_list = atom_site_dict[asym_id]["label_comp_id"]
            unknown_mask = [cc in ("UNL", "UNK") for cc in label_comp_id_list]
            if all(unknown_mask):
                continue
            for key in atom_site_dict[asym_id]:
                values = atom_site_dict[asym_id][key]
                values = np.array(values)[~np.array(unknown_mask)]
                atom_site_dict[asym_id][key] = values
            new_dict[asym_id] = atom_site_dict[asym_id]
        return new_dict

    return _worker


def remove_unknown_from_struct_conn() -> Callable[
    ...,
    dict[str, dict[str, NDArray]] | None,
]:
    """Remove connections where any partner residue is unknown."""

    def _worker(
        struct_conn_dict: dict[str, dict[str, NDArray]] | None,
    ) -> dict[str, dict[str, NDArray]] | None:
        if struct_conn_dict is None:
            return None

        new_dict: dict[str, dict[str, NDArray]] = {}

        for asym_pair, conn_fields in struct_conn_dict.items():
            label_comp_id1_list = conn_fields["ptnr1_label_comp_id"]
            label_comp_id2_list = conn_fields["ptnr2_label_comp_id"]

            unl_mask = np.array(
                [
                    cc1 != "UNL" and cc2 != "UNL"
                    for cc1, cc2 in zip(
                        label_comp_id1_list,
                        label_comp_id2_list,
                        strict=True,
                    )
                ],
                dtype=bool,
            )

            if not np.any(unl_mask):
                continue

            new_dict[asym_pair] = {
                key: np.array(values)[unl_mask] for key, values in conn_fields.items()
            }

        return new_dict

    return _worker


def attach_entity() -> Callable[..., dict[str, dict[str, NDArray]] | None]:
    """Attach entity information to each asym_id."""

    def _attach_entity(
        asym_info: dict[str, NDArray],
        entity_info: dict[str, NDArray],
    ) -> dict[str, NDArray]:
        entity_type = entity_info["type"][0]
        if entity_type != "water":
            """
            Validate consistency between asym_info(scheme) and entity_info.

            For water, skip the validation as it often lacks proper entity definition.
            """
            entity_hetero = entity_info["hetero"]
            scheme_hetero = asym_info["hetero"]

            if not np.all(np.isin(scheme_hetero, entity_hetero)):
                msg = "Hetero values in scheme not consistent with entity."
                raise ValueError(msg)

            entity_chem_comp = entity_info["chem_comp"]
            scheme_chem_comp = asym_info["chem_comp"]

            if not np.all(np.isin(scheme_chem_comp, entity_chem_comp)):
                msg = "Chem_comp values in scheme not consistent with entity."
                raise ValueError(msg)

            entity_seq_num = entity_info["seq_num"].astype(str)
            scheme_cif_idx = asym_info["cif_idx"]
            if not np.all(np.isin(scheme_cif_idx, entity_seq_num)):
                msg = "Cif_idx values in scheme not consistent with entity."
                raise ValueError(msg)

        # for water we have to expand one_letter_code to match the length of asym_info
        if entity_type == "water":
            _len = len(asym_info["cif_idx"])
            asym_info["one_letter_code_can"] = np.array(["X"] * _len, dtype=str)
            asym_info["one_letter_code"] = np.array(["X"] * _len, dtype=str)
        else:
            asym_info["one_letter_code_can"] = entity_info["one_letter_code_can"]
            asym_info["one_letter_code"] = entity_info["one_letter_code"]
        asym_info["descriptor"] = entity_info["descriptor"]
        asym_info["branch_link"] = entity_info["branch_link"]
        return asym_info

    def _worker(
        asym_dict: dict[str, dict[str, NDArray]] | None,
        entity_dict: dict[str, dict[str, NDArray]] | None,
    ) -> dict[str, dict[str, NDArray]] | None:
        output = {}
        for asym_id, asym_info in asym_dict.items():
            entity_id = asym_info["entity_id"]
            entity_info = entity_dict[entity_id[0]]
            attached = _attach_entity(asym_info, entity_info)
            output[asym_id] = attached
        return output

    return _worker


def rearrange_atom_site_dict() -> Callable[..., dict | None]:
    """Rearrange atom_site_dict to have model id and alt is as keys."""

    def _remove_hydrogen(
        atom_site_dict: dict[str, dict[str, NDArray]] | None,
    ) -> dict[str, dict[str, NDArray]] | None:
        element = atom_site_dict["type_symbol"]
        hydrogen_mask = ~np.isin(element, ["H", "D"])

        for key in atom_site_dict:
            values = atom_site_dict[key]
            values = np.array(values)[hydrogen_mask]
            atom_site_dict[key] = values
        return atom_site_dict

    def _rearrange_each_asym_id(
        atom_site_dict: dict[str, NDArray] | None,
        remove_hydrogen: bool = True,
    ) -> dict[str, dict[str, NDArray]] | None:
        if remove_hydrogen:
            atom_site_dict = _remove_hydrogen(atom_site_dict)
        if atom_site_dict is None:
            return None

        # Before remapping, rearrange key name first e.g., 'Cartn_x, Cartn_y, Cartn_z' -> 'xyz'
        rearranged_dict = {}
        cif_idx = atom_site_dict["label_seq_id"]
        ins_code = atom_site_dict["pdbx_PDB_ins_code"]
        auth_seq_idx = atom_site_dict["auth_seq_id"]
        auth_idx = np.array(
            [
                f"{aa if ii in {'.', '?'} else f'{aa}.{ii}'}"
                for aa, ii in zip(auth_seq_idx, ins_code, strict=True)
            ],
        )
        xyz = np.stack(
            [
                atom_site_dict["Cartn_x"].astype(float),
                atom_site_dict["Cartn_y"].astype(float),
                atom_site_dict["Cartn_z"].astype(float),
            ],
            axis=-1,
        )
        rearranged_dict["cif_idx"] = cif_idx.astype(
            str,
        )  # In the case of nonpolymer, it can be '.'.
        rearranged_dict["auth_idx"] = auth_idx.astype(str)
        rearranged_dict["xyz"] = xyz.astype(float)
        rearranged_dict["b_factor"] = atom_site_dict["B_iso_or_equiv"].astype(float)
        rearranged_dict["occupancy"] = atom_site_dict["occupancy"].astype(float)
        rearranged_dict["element"] = atom_site_dict["type_symbol"].astype(str)
        rearranged_dict["atom"] = atom_site_dict["label_atom_id"].astype(str)
        rearranged_dict["chem_comp"] = atom_site_dict["label_comp_id"].astype(str)

        rearranged_dict["alt_id"] = atom_site_dict["label_alt_id"].astype(str)
        rearranged_dict["model_id"] = atom_site_dict["pdbx_PDB_model_num"].astype(str)

        key_list = list(rearranged_dict.keys() - {"alt_id", "model_id"})

        # if alt id is present like 'A', 'B', then it can takes also alt_id=='.'
        # which is wildcard in general.

        output = {}
        model_ids = np.unique(rearranged_dict["model_id"])
        for _model_id in model_ids:
            model_id = str(_model_id)
            model_mask = rearranged_dict["model_id"] == model_id
            alt_ids = np.unique(rearranged_dict["alt_id"][model_mask])
            output[model_id] = {}
            for _alt_id in alt_ids:
                atom_array = np.asarray(rearranged_dict["atom"])
                auth_idx_array = np.asarray(rearranged_dict["auth_idx"])
                determinant_array = np.asarray(
                    [f"{a}.{b}" for a, b in zip(atom_array, auth_idx_array)]
                )
                alt_id_array = np.asarray(rearranged_dict["alt_id"])
                mask = np.zeros_like(alt_id_array, dtype=bool)
                for determinant in np.unique(determinant_array):
                    group_mask = determinant_array == determinant
                    group_alt = alt_id_array[group_mask]
                    has_target = np.any(group_alt == _alt_id)

                    if has_target:
                        # Mark True only where alt_id equals the reference alt_id
                        mask[group_mask & (alt_id_array == _alt_id)] = True
                    else:
                        # If no reference alt_id exists, mark True where alt_id == '.'
                        mask[group_mask & (alt_id_array == ".")] = True
                combined_mask = model_mask & mask
                output[model_id][str(_alt_id)] = {}
                for key in key_list:
                    output[model_id][str(_alt_id)][key] = rearranged_dict[key][
                        combined_mask
                    ]
        return {"atom_site": output}

    def _worker(
        atom_site_dict: dict[str, dict[str, NDArray]] | None,
        remove_hydrogen: bool = True,
    ) -> dict[str, dict[str, NDArray]] | None:
        output = {}
        for asym_id, _atom_site_dict in atom_site_dict.items():
            rearranged = _rearrange_each_asym_id(_atom_site_dict, remove_hydrogen)
            output[asym_id] = rearranged
        return output

    return _worker


def build_full_length_asym_dict() -> Callable[..., dict | None]:
    """Build full length asym dict by combining chem comp and atom_site info."""

    def _parse_atom_site_dict(
        asym_dict: dict[str, NDArray] | None,
        atom_site_dict: dict[str, dict[str, NDArray]] | None,
        chem_comp_dict: dict[str, dict[str, NDArray]] | None,
    ) -> dict[str, dict[str, NDArray]] | None:
        full_chem_comp_list = asym_dict["chem_comp"]
        auth_idx_list = asym_dict["auth_idx"]
        cif_idx_list = asym_dict["cif_idx"]
        auth_idx_to_chem_comp = {
            str(aa): cc
            for aa, cc in zip(auth_idx_list, full_chem_comp_list, strict=False)
        }  # Handle heterogeneous sequence, cc is a list

        atom_auth_idx = atom_site_dict["auth_idx"]
        atom_atom = atom_site_dict["atom"]
        atom_chem_comp = atom_site_dict["chem_comp"]
        xyz, b_factor, occupancy = (
            atom_site_dict["xyz"],
            atom_site_dict["b_factor"],
            atom_site_dict["occupancy"],
        )
        chem_comp_list = []
        atom_count = 0
        residue_count = 0
        atom_to_residue_idx = []
        auth_idx_to_atom_idx = {}
        for auth_idx in auth_idx_list:
            _chem_comp_list = auth_idx_to_chem_comp[auth_idx]
            # remove empty string in case of heterogeneous sequence
            _chem_comp_list = [cc for cc in _chem_comp_list if cc != ""]
            if len(_chem_comp_list) == 0:
                msg = f"No chem_comp found for residue {auth_idx}."
                raise ValueError(msg)
            """
            If the residue is missing in atom_site, fill in with the first chem_comp.
            """
            atom_idx = np.where(atom_auth_idx == auth_idx)[0]
            if len(atom_idx) == 0:
                chem_comp = _chem_comp_list[0]
            else:
                atom_idx = atom_idx[0]
                atom_cc = atom_chem_comp[atom_idx]
                if atom_cc in _chem_comp_list:
                    chem_comp = atom_cc
                else:
                    msg = f"Chem_comp {atom_cc} not in entity definition {_chem_comp_list} for residue {auth_idx}."
                    raise ValueError(msg)
            chem_comp_list.append(chem_comp)
            atom_num = len(chem_comp_dict[chem_comp]["atom"]["atom_id"])
            atom_to_residue_idx.extend([residue_count] * atom_num)
            auth_idx_to_atom_idx[auth_idx] = atom_count

            residue_count += 1
            atom_count += atom_num

        chem_comp_list = np.array(
            chem_comp_list,
            dtype=str,
        )  # resolve heterogeneous sequence
        chem_comp_id_list = chem_comp_list

        full_atom_num = 0
        for chem_comp_name in chem_comp_list:
            chem_comp = chem_comp_dict[chem_comp_name]
            full_atom_num += len(chem_comp["atom"])

        chem_comp_list = [chem_comp_dict[cc] for cc in chem_comp_list]
        chem_comp_residue_list = [cc["residue"] for cc in chem_comp_list]
        chem_comp_atom_list = [cc["atom"] for cc in chem_comp_list]

        chem_comp_residue_container = concat_containers(*chem_comp_residue_list)
        chem_comp_atom_container = concat_containers(*chem_comp_atom_list)

        full_xyz = np.full((full_atom_num, 3), np.nan, dtype=float)
        full_b_factor = np.full(full_atom_num, np.nan, dtype=float)
        full_occupancy = np.full(full_atom_num, np.nan, dtype=float)
        for ii in range(len(atom_atom)):
            chem_comp = chem_comp_dict[atom_chem_comp[ii]]
            chem_comp_atom_id = chem_comp["atom"]["atom_id"]
            atom_id = atom_atom[ii]
            auth_idx = atom_auth_idx[ii]
            if auth_idx not in auth_idx_to_atom_idx:
                # For H2O, it happens that auth_idx is not matched.
                if chem_comp["residue"]["name"].value[0] != "WATER":
                    msg = f"Auth_idx {auth_idx} not found in scheme for atom {atom_id} in residue {atom_chem_comp[ii]}."
                    raise ValueError(msg)
                continue
            # for some cases (ex 7oui's JH AJP ligand), atom scheme is not consistent with chem_comp_atom
            # in this case, we skip the atom
            chem_comp_atom_idx = np.where(chem_comp_atom_id == atom_id)[0]
            if len(chem_comp_atom_idx) == 0:
                continue
            atom_idx = auth_idx_to_atom_idx[auth_idx] + chem_comp_atom_idx
            full_xyz[atom_idx] = xyz[ii]
            full_b_factor[atom_idx] = b_factor[ii]
            full_occupancy[atom_idx] = occupancy[ii]

        full_xyz = NodeFeature(
            value=full_xyz,
            description="atomic coordinates",
        )
        full_b_factor = NodeFeature(
            value=full_b_factor,
            description="B-factor",
        )
        full_occupancy = NodeFeature(
            value=full_occupancy,
            description="occupancy",
        )

        atom_node_features = chem_comp_atom_container.node_features
        atom_edge_features = chem_comp_atom_container.edge_features
        atom_node_features.update(
            {
                "xyz": full_xyz,
                "b_factor": full_b_factor,
                "occupancy": full_occupancy,
            },
        )

        def find_atom_index(residue_idx, atom_name):
            chem_comp = chem_comp_list[residue_idx]
            chem_comp_atom_id = chem_comp["atom"]["atom_id"]
            if atom_name not in chem_comp_atom_id:
                return None
            return (
                auth_idx_to_atom_idx[auth_idx_list[residue_idx]]
                + np.where(chem_comp_atom_id == atom_name)[0][0]
            )

        # add branch edge + canonical bonds
        if asym_dict["branch_link"] is not None:  # Branched
            for residue_idx1, residue_idx2 in asym_dict["branch_link"]:
                link_info = asym_dict["branch_link"][(residue_idx1, residue_idx2)]
                res_idx1 = np.where(cif_idx_list == residue_idx1)[0][0]
                res_idx2 = np.where(cif_idx_list == residue_idx2)[0][0]
                residue_bond = EdgeFeature(
                    value=np.array([1], dtype=int),
                    src_indices=np.array([res_idx1], dtype=int),
                    dst_indices=np.array([res_idx2], dtype=int),
                    description="bond between residues. boolean.",
                )
                atom_id1, atom_id2 = link_info["atom_id"]
                src_idx = find_atom_index(res_idx1, atom_id1)
                dst_idx = find_atom_index(res_idx2, atom_id2)
                if src_idx is None or dst_idx is None:
                    msg = f"Cannot find atom {atom_id1} in residue {residue_idx1} or atom {atom_id2} in residue {residue_idx2} for branch link."
                    raise ValueError(msg)
                atom_src = np.array(
                    [*atom_edge_features["bond_type"].src_indices, src_idx],
                    dtype=int,
                )
                atom_dst = np.array(
                    [*atom_edge_features["bond_type"].dst_indices, dst_idx],
                    dtype=int,
                )
                bond_type_value = np.array(
                    [*atom_edge_features["bond_type"].value, link_info["bond"]],
                    dtype=str,
                )
                atom_edge_features["bond_type"] = EdgeFeature(
                    value=bond_type_value,
                    src_indices=atom_src,
                    dst_indices=atom_dst,
                    description=atom_edge_features["bond_type"].description,
                )
                aromatic_value = np.array(
                    [*atom_edge_features["bond_aromatic"].value, "N"],
                    dtype=str,
                )
                atom_edge_features["bond_aromatic"] = EdgeFeature(
                    value=aromatic_value,
                    src_indices=atom_src,
                    dst_indices=atom_dst,
                    description=atom_edge_features["bond_aromatic"].description,
                )
                stereo_value = np.array(
                    [*atom_edge_features["bond_stereo"].value, "N"],
                    dtype=str,
                )
                atom_edge_features["bond_stereo"] = EdgeFeature(
                    value=stereo_value,
                    src_indices=atom_src,
                    dst_indices=atom_dst,
                    description=atom_edge_features["bond_stereo"].description,
                )
        else:  # Polymer or Non-polymer
            match asym_dict["descriptor"]:
                case "polypeptide(L)" | "polypeptide(D)":  # Protein
                    i_atom, i_1_atom = "C", "N"
                case (
                    "polydeoxyribonucleotide"
                    | "polyribonucleotide"
                    | "polydeoxyribonucleotide/polyribonucleotide hybrid"
                ):  # DNA/RNA
                    i_atom, i_1_atom = "O3", "P"
                case _:  # None-polymer or unknown polymer
                    i_atom, i_1_atom = None, None
            if residue_count == 1:
                residue_bond = None
            else:
                residue_src = np.arange(residue_count - 1)
                residue_dst = np.arange(1, residue_count)
                residue_bond = EdgeFeature(
                    value=np.array([1] * len(residue_src), dtype=int),
                    src_indices=residue_src,
                    dst_indices=residue_dst,
                    description="bond between residues. boolean.",
                )
            if i_atom is not None and i_1_atom is not None:
                atom_src = []
                atom_dst = []
                for res_idx in range(residue_count - 1):
                    src_idx = find_atom_index(res_idx, i_1_atom)
                    dst_idx = find_atom_index(res_idx + 1, i_atom)
                    if src_idx is not None and dst_idx is not None:
                        atom_src.append(src_idx)
                        atom_dst.append(dst_idx)
                atom_src = np.array(atom_src, dtype=int)
                atom_dst = np.array(atom_dst, dtype=int)
                atom_bond_type_can = np.array([1] * len(atom_src), dtype=int)
                atom_aromatic_can = np.array(["N"] * len(atom_src), dtype=str)
                atom_stereo_can = np.array(["N"] * len(atom_src), dtype=str)

                atom_src, atom_dst = (
                    np.concatenate(
                        [atom_edge_features["bond_type"].src_indices, atom_src],
                    ),
                    np.concatenate(
                        [atom_edge_features["bond_type"].dst_indices, atom_dst],
                    ),
                )
                bond_type_value = np.concatenate(
                    [atom_edge_features["bond_type"].value, atom_bond_type_can],
                )
                atom_edge_features["bond_type"] = EdgeFeature(
                    value=bond_type_value,
                    src_indices=atom_src,
                    dst_indices=atom_dst,
                    description=atom_edge_features["bond_type"].description,
                )
                aromatic_value = np.concatenate(
                    [atom_edge_features["bond_aromatic"].value, atom_aromatic_can],
                )
                atom_edge_features["bond_aromatic"] = EdgeFeature(
                    value=aromatic_value,
                    src_indices=atom_src,
                    dst_indices=atom_dst,
                    description=atom_edge_features["bond_aromatic"].description,
                )
                stereo_value = np.concatenate(
                    [atom_edge_features["bond_stereo"].value, atom_stereo_can],
                )
                atom_edge_features["bond_stereo"] = EdgeFeature(
                    value=stereo_value,
                    src_indices=atom_src,
                    dst_indices=atom_dst,
                    description=atom_edge_features["bond_stereo"].description,
                )

        atom_container = FeatureContainer(
            node_features=atom_node_features,
            edge_features=atom_edge_features,
        )
        residue_node_features = chem_comp_residue_container.node_features
        residue_node_features.update(
            {
                "cif_idx": NodeFeature(
                    value=asym_dict["cif_idx"],
                    description="CIF index of the residue",
                ),
                "auth_idx": NodeFeature(
                    value=asym_dict["auth_idx"],
                    description="Author index of the residue",
                ),
                "chem_comp": NodeFeature(
                    value=chem_comp_id_list,
                    description="Chemical component ID of the residue",
                ),
                "hetero": NodeFeature(
                    value=asym_dict["hetero"].astype(int),
                    description="Heterogeneous flag of the residue",
                ),
                "one_letter_code_can": NodeFeature(
                    value=asym_dict["one_letter_code_can"],
                    description="One letter code (canonical) of the residue",
                ),
                "one_letter_code": NodeFeature(
                    value=asym_dict["one_letter_code"],
                    description="One letter code (asym) of the residue",
                ),
            },
        )
        residue_edge_features = {} if residue_bond is None else {"bond": residue_bond}
        residue_container = FeatureContainer(
            node_features=residue_node_features,
            edge_features=residue_edge_features,
        )

        chain_node_feature = NodeFeature(
            value=np.array(asym_dict["entity_id"][0:1], dtype=str),
            description="Entity ID of the chain",
        )
        chain_container = FeatureContainer(
            node_features={"entity_id": chain_node_feature},
            edge_features={},
        )

        return {
            "atom": atom_container,
            "residue": residue_container,
            "chain": chain_container,
            "atom_to_residue_idx": np.array(atom_to_residue_idx, dtype=int),
        }

    def _function(
        _asym_dict: dict[str, NDArray] | None,
        chem_comp_dict: dict[str, dict[str, NDArray]] | None,
    ) -> dict[str, dict[str, NDArray]] | None:
        output = {}
        for model_id, alt_id_dict in _asym_dict["atom_site"].items():
            for alt_id, atom_site_dict in alt_id_dict.items():
                output[(model_id, alt_id)] = _parse_atom_site_dict(
                    _asym_dict,
                    atom_site_dict,
                    chem_comp_dict,
                )
        return output

    def _worker(
        asym_dict: dict[str, dict[str, NDArray]] | None,
        chem_comp_dict: dict[str, dict[str, NDArray]] | None,
    ) -> dict[str, dict[str, NDArray]] | None:
        output = {}
        for asym_id, _asym_dict in asym_dict.items():
            if "atom_site" not in _asym_dict or len(_asym_dict.keys()) == 1:
                # In case of UNL-only chain, atom_site cannot be constructed.
                # Or there can be only atom_site without scheme info.
                # We skip those chains.
                continue
            rearranged = _function(_asym_dict, chem_comp_dict)
            output[asym_id] = rearranged
        return output

    return _worker


def parse_expression(expr: str) -> list[str] | None:
    """
    Parse an expression into a list of strings, expanding ranges when present.

    Examples
    --------
        '(1,2,6,10,23,24)' -> ['1','2','6','10','23','24']
        '(1-60)'           -> ['1','2','3', ..., '60']
        '1'                -> ['1']
        'P'                -> ['P']
        'P,Q'              -> ['P','Q']
        '1-10,21-25'       -> ['1','2', ..., '10', '21', ..., '25']
        '(X0)(1-60)'       -> None # 20250302, \
                              psk 2024Mar03  : (2xgk,1dwn,2vf9,1lp3,5fmo,3cji,3dpr,4ang,\
                                                1al0,4nww,1cd3,4aed)
                              -> No need to use it
    """
    if ")(" in expr:
        return None

    def split_top_level_commas(s: str) -> list[str]:
        """Split the input string on commas that are not nested inside parentheses."""
        result = []
        current = []
        paren_depth = 0
        for char in s:
            if char == "(":
                paren_depth += 1
                current.append(char)
            elif char == ")":
                paren_depth -= 1
                current.append(char)
            elif char == "," and paren_depth == 0:
                result.append("".join(current).strip())
                current = []
            else:
                current.append(char)
        if current:
            result.append("".join(current).strip())
        return result

    def parse_parenthesized(s: str) -> list[str]:
        """
        Parse a string that may be a comma-separated list and/or a range.

        - If it contains commas, each piece is processed individually.
        - If a piece contains a hyphen and both parts are numeric,
          it is expanded into a numeric range.
        - Otherwise, the piece is returned as is.
        """
        # If there are commas, split and process each component.
        if "," in s:
            items = [itm.strip() for itm in s.split(",")]
            result = []
            for item in items:
                if "-" in item:
                    # Expand numeric range if applicable.
                    start_str, end_str = item.split("-", 1)
                    start_str, end_str = start_str.strip(), end_str.strip()
                    if start_str.isdigit() and end_str.isdigit():
                        start_val = int(start_str)
                        end_val = int(end_str)
                        result.extend([str(x) for x in range(start_val, end_val + 1)])
                    else:
                        result.append(item)
                else:
                    result.append(item)
            return result
        if "-" in s:
            # No commas; check if the whole thing is a range.
            start_str, end_str = s.split("-", 1)
            start_str, end_str = start_str.strip(), end_str.strip()
            if start_str.isdigit() and end_str.isdigit():
                start_val = int(start_str)
                end_val = int(end_str)
                return [str(x) for x in range(start_val, end_val + 1)]
            return [s]
        return [s]

    def parse_token(token: str) -> list[str]:
        """
        Process a token which might be a mix of parenthesized groups and plain text.

        If the token does not contain any parentheses, we pass it to parse_parenthesized
        so that forms like "1-10" or "P,Q" are expanded.
        """
        token = token.strip()
        if "(" not in token and ")" not in token:
            # No parentheses at all; process the whole token.
            return parse_parenthesized(token)

        # Otherwise, extract the parts using regex.
        pattern = r"\([^)]*\)|[^()]+"
        parts = re.findall(pattern, token)
        results = []
        for _part in parts:
            part = _part.strip()
            if part.startswith("(") and part.endswith(")"):
                # Remove the outer parentheses and process the inside.
                inside = part[1:-1].strip()
                results.extend(parse_parenthesized(inside))
            elif part:
                # For any plain-text part, also process it to catch ranges like "1-10".
                results.extend(parse_parenthesized(part))
        return results

    # First split by top-level commas, then process each token.
    tokens = split_top_level_commas(expr)
    final_result = []
    for t in tokens:
        final_result.extend(parse_token(t))
    return final_result


def parse_assembly_dict() -> Callable[..., dict[str, dict[str, NDArray]] | None]:
    """Parse struct_assembly_gen oper_expression into a list of oper_id."""

    def _worker(
        struct_assembly_gen_dict: dict[str, dict[str, NDArray]] | None,
    ) -> dict[str, dict[str, NDArray]] | None:
        output_dict = {}
        for assembly_id, infos in struct_assembly_gen_dict.items():
            if assembly_id not in output_dict:
                output_dict[assembly_id] = {}
            for _asym_ids, _oper_expression in zip(
                infos["asym_id_list"],
                infos["oper_expression"],
                strict=True,
            ):
                if len(_asym_ids) == 0 or len(_oper_expression) == 0:
                    continue
                asym_ids = _asym_ids.split(",")
                oper_expression = parse_expression(_oper_expression)
                if oper_expression is None:
                    continue
                for asym_id in asym_ids:
                    if asym_id not in output_dict[assembly_id]:
                        output_dict[assembly_id][asym_id] = []
                    output_dict[assembly_id][asym_id].extend(oper_expression)
        return output_dict

    return _worker


def get_struct_oper() -> Callable[..., dict[str, dict[str, NDArray]] | None]:
    """Parse struct_oper to get rotation matrix and translation vector."""

    def _worker(
        struct_oper_dict: dict[str, dict[str, NDArray]] | None,
    ) -> dict[str, dict[str, NDArray]] | None:
        output_dict = {}
        for struct_oper_id in struct_oper_dict:
            raw_dict = struct_oper_dict[struct_oper_id]
            output_dict[struct_oper_id] = {}
            matrix = []
            for ii in range(1, 4):
                row = [float(raw_dict[f"matrix[{ii}][{jj}]"]) for jj in range(1, 4)]
                matrix.append(row)
            matrix = np.array(matrix, dtype=float)
            output_dict[struct_oper_id]["matrix"] = matrix
            vector = [float(raw_dict[f"vector[{ii}]"]) for ii in range(1, 4)]
            vector = np.array(vector, dtype=float)
            output_dict[struct_oper_id]["vector"] = vector
        return output_dict

    return _worker


def build_assembly_dict() -> Callable[..., dict[str, dict[str, NDArray]] | None]:
    """Remove chains where all residues are UNL."""

    def _apply_RT(
        atom_container: FeatureContainer, matrix: NDArray, vector: NDArray
    ) -> FeatureContainer:
        xyz = atom_container.node_features["xyz"].value
        new_xyz = xyz @ matrix.T + vector
        new_atom_container = atom_container.copy()
        new_atom_container.node_features["xyz"] = NodeFeature(
            value=new_xyz,
            description=atom_container.node_features["xyz"].description,
        )
        return new_atom_container

    def _get_atom_indices(
        residue_indices: np.ndarray,
        atom_id_list: np.ndarray,
        index_table: IndexTable,
        atom_id_in_container: NodeFeature,
    ) -> np.ndarray:
        atom_indices = []
        for residue_idx, atom_id in zip(residue_indices, atom_id_list, strict=True):
            _atom_indices = index_table.residues_to_atoms(np.array([residue_idx]))

            _atom_indices = np.where(atom_id_in_container[_atom_indices] == atom_id)[0]
            if len(_atom_indices) == 0:
                atom_indices.append(np.array([-1], dtype=int))  # e.g. hydrogen
            else:
                atom_indices.append(_atom_indices)
        return np.concatenate(atom_indices)

    def _worker(
        asym_dict: dict[str, dict[str, FeatureContainer]] | None,
        struct_assembly_dict: dict[str, dict[str, NDArray]] | None,
        struct_oper_dict: dict[str, dict[str, NDArray]] | None,
        struct_conn_dict: dict[str, dict[str, NDArray]] | None,
    ) -> dict[str, dict[str, NDArray]] | None:
        output = {}
        full_asym_id_list = list(asym_dict.keys())
        for assembly_id in struct_assembly_dict:
            asym_id_list = list(struct_assembly_dict[assembly_id].keys())
            asym_id_list = [aid for aid in asym_id_list if aid in full_asym_id_list]
            if len(asym_id_list) == 0:
                # for the case where (X0) is included in oper_expression or unknown chain
                continue
            asym_id_to_model_alt_id = {}
            for asym_id in asym_id_list:
                model_alt_id_list = list(asym_dict[asym_id].keys())
                asym_id_to_model_alt_id[asym_id] = model_alt_id_list

            sets = [set(v) for v in asym_id_to_model_alt_id.values()]

            model_alt_id_set = set.union(*sets)

            for model_alt_id in model_alt_id_set:
                model_id, alt_id = model_alt_id
                wildcard_model_alt_id = (model_id, ".") if alt_id != "." else None
                key = f"{assembly_id}_{model_id}_{alt_id}"
                asym_containers = {}
                for asym_id in asym_id_list:
                    oper_id_list = struct_assembly_dict[assembly_id][asym_id]
                    if len(oper_id_list) == 0:
                        continue
                    if model_alt_id not in asym_id_to_model_alt_id[asym_id]:
                        if (
                            wildcard_model_alt_id is not None
                            and wildcard_model_alt_id
                            in asym_id_to_model_alt_id[asym_id]
                        ):
                            asym_container = asym_dict[asym_id][wildcard_model_alt_id]
                        else:
                            continue
                    else:
                        asym_container = asym_dict[asym_id][model_alt_id]
                    _atom_container = asym_container["atom"]
                    _residue_container = asym_container["residue"]
                    _chain_container = asym_container["chain"]
                    _atom_to_residue_idx = asym_container["atom_to_residue_idx"]

                    for oper_id in oper_id_list:
                        new_atom_container = _apply_RT(
                            _atom_container,
                            struct_oper_dict[oper_id]["matrix"],
                            struct_oper_dict[oper_id]["vector"],
                        )
                        asym_containers[f"{asym_id}_{oper_id}"] = {
                            "atom": new_atom_container,
                            "residue": _residue_container,
                            "chain": _chain_container,
                            "atom_to_residue_idx": _atom_to_residue_idx,
                        }

                chain_id_list = list(asym_containers.keys())
                atom_container_list = [
                    asym_containers[asym_id]["atom"] for asym_id in asym_containers
                ]
                residue_container_list = [
                    asym_containers[asym_id]["residue"] for asym_id in asym_containers
                ]
                residue_to_chain_idx = []
                for chain_idx in range(len(chain_id_list)):
                    residue_num = len(residue_container_list[chain_idx])
                    residue_to_chain_idx.extend([chain_idx] * residue_num)
                residue_to_chain_idx = np.array(residue_to_chain_idx, dtype=int)

                chain_container_list = [
                    asym_containers[asym_id]["chain"] for asym_id in asym_containers
                ]
                atom_to_residue_idx_list = [
                    asym_containers[asym_id]["atom_to_residue_idx"]
                    for asym_id in asym_containers
                ]
                atom_container = concat_containers(*atom_container_list)
                residue_container = concat_containers(*residue_container_list)
                chain_container = concat_containers(*chain_container_list)

                increments = np.array(
                    [arr.max() + 1 for arr in atom_to_residue_idx_list]
                )
                offsets = np.concatenate([[0], np.cumsum(increments[:-1])])
                atom_to_residue_idx = [
                    arr + off for arr, off in zip(atom_to_residue_idx_list, offsets)
                ]
                atom_to_residue_idx = np.concatenate(atom_to_residue_idx)

                chain_id = NodeFeature(
                    value=np.array(chain_id_list, dtype=str),
                    description="Chain ID in the assembly",
                )
                chain_container.node_features["chain_id"] = chain_id

                # Indextable TODO
                index_table = IndexTable.from_parents(
                    atom_to_res=atom_to_residue_idx,
                    res_to_chain=residue_to_chain_idx,
                    n_chain=len(chain_id_list),
                )
                quest_to_dot = lambda x: "." if x == "?" else f"{x}"

                oper_to_chains = {}
                for cid in chain_id_list:
                    chain, assembly = cid.split("_")
                    if assembly not in oper_to_chains:
                        oper_to_chains[assembly] = []
                    oper_to_chains[assembly].append(chain)

                auth_idx_in_container = residue_container["auth_idx"]
                atom_id_in_container = atom_container["atom_id"]

                residue_src = []
                residue_dst = []
                atom_src = []
                atom_dst = []
                atom_value = []
                for c1, c2 in struct_conn_dict:
                    chain_id_with_oper = [
                        (f"{c1}_{oper_id}", f"{c2}_{oper_id}")
                        for oper_id, chains in oper_to_chains.items()
                        if c1 in chains and c2 in chains
                    ]
                    if len(chain_id_with_oper) == 0:
                        continue
                    _items = struct_conn_dict[(c1, c2)]
                    auth_idx1 = _items["ptnr1_auth_seq_id"]
                    auth_idx2 = _items["ptnr2_auth_seq_id"]
                    ins_code1 = _items["pdbx_ptnr1_PDB_ins_code"]
                    ins_code2 = _items["pdbx_ptnr2_PDB_ins_code"]
                    alt_id1 = _items["pdbx_ptnr1_label_alt_id"]
                    alt_id2 = _items["pdbx_ptnr2_label_alt_id"]
                    atom_id1 = _items["ptnr1_label_atom_id"]
                    atom_id2 = _items["ptnr2_label_atom_id"]
                    conn_type = _items["conn_type_id"]
                    value_order = _items["pdbx_value_order"]
                    edge_value = np.stack([conn_type, value_order], axis=-1)

                    ins_code1 = np.array(
                        [quest_to_dot(x) for x in ins_code1],
                        dtype=str,
                    )
                    ins_code2 = np.array(
                        [quest_to_dot(x) for x in ins_code2],
                        dtype=str,
                    )
                    alt_id1 = np.array([quest_to_dot(x) for x in alt_id1], dtype=str)
                    alt_id2 = np.array([quest_to_dot(x) for x in alt_id2], dtype=str)
                    auth_idx1 = np.array(
                        [
                            aa if ii == "." else f"{aa}.{ii}"
                            for aa, ii in zip(auth_idx1, ins_code1, strict=True)
                        ],
                        dtype=str,
                    )
                    auth_idx2 = np.array(
                        [
                            aa if ii == "." else f"{aa}.{ii}"
                            for aa, ii in zip(auth_idx2, ins_code2, strict=True)
                        ],
                        dtype=str,
                    )
                    for chain1, chain2 in chain_id_with_oper:
                        chain_idx1 = chain_id_list.index(chain1)
                        chain_idx2 = chain_id_list.index(chain2)
                        residue_indices1 = index_table.chains_to_residues([chain_idx1])
                        residue_indices2 = index_table.chains_to_residues([chain_idx2])

                        auth_indices1 = auth_idx_in_container[residue_indices1]
                        auth_indices2 = auth_idx_in_container[residue_indices2]

                        mapping1 = {val: i for i, val in enumerate(auth_indices1.value)}
                        mapping2 = {val: i for i, val in enumerate(auth_indices2.value)}
                        residue_indices1 = residue_indices1[0] + np.array(
                            [mapping1[val] for val in auth_idx1]
                        )
                        residue_indices2 = residue_indices2[0] + np.array(
                            [mapping2[val] for val in auth_idx2]
                        )
                        residue_src.append(residue_indices1)
                        residue_dst.append(residue_indices2)

                        atom_indices1 = _get_atom_indices(
                            residue_indices1,
                            atom_id1,
                            index_table,
                            atom_id_in_container,
                        )
                        atom_indices2 = _get_atom_indices(
                            residue_indices2,
                            atom_id2,
                            index_table,
                            atom_id_in_container,
                        )

                        valid_mask = (atom_indices1 != -1) & (atom_indices2 != -1)
                        edge_value_masked = edge_value[valid_mask]
                        atom_indices1 = atom_indices1[valid_mask]
                        atom_indices2 = atom_indices2[valid_mask]

                        atom_src.append(atom_indices1)
                        atom_dst.append(atom_indices2)
                        atom_value.append(edge_value_masked)

                if len(residue_src) > 0 and len(atom_src) > 0:
                    residue_src = np.concatenate(residue_src)
                    residue_dst = np.concatenate(residue_dst)
                    atom_src = np.concatenate(atom_src)
                    atom_dst = np.concatenate(atom_dst)
                    atom_value = np.concatenate(atom_value)
                    residue_value = np.array(
                        [1] * len(residue_src),
                        dtype=int,
                    )
                    residue_struct_conn = EdgeFeature(
                        value=residue_value,
                        src_indices=residue_src,
                        dst_indices=residue_dst,
                        description="struct_conn between residues. boolean.",
                    )
                    atom_struct_conn = EdgeFeature(
                        value=atom_value,
                        src_indices=atom_src,
                        dst_indices=atom_dst,
                        description="struct_conn between atoms. (conn_type_id, pdbx_value_order)",
                    )
                    residue_container.edge_features["struct_conn"] = residue_struct_conn
                    atom_container.edge_features["struct_conn"] = atom_struct_conn

                output[key] = {
                    "atoms": atom_container,
                    "residues": residue_container,
                    "chains": chain_container,
                    "index_table": index_table,
                }
        return output

    return _worker
