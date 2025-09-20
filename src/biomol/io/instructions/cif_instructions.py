from collections import defaultdict
from collections.abc import Callable
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from biomol.core.feature import EdgeFeature, NodeFeature

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
            cols - {tied_to} if isinstance(tied_to, str) else {tied_to[0], tied_to[1]}
        )

        result: dict[str, dict[str, NDArray]] = {}

        for i in range(n):
            row = {col: cif_raw_dict[col][i] for col in cols}

            if isinstance(tied_to, str):
                key = cif_raw_dict[tied_to][i]
            else:
                key = f"{cif_raw_dict[tied_to[0]][i]}|{cif_raw_dict[tied_to[1]][i]}"

            if key not in result:
                result[key] = {col: [] for col in cols}

            for col in cols:
                if col == tied_to:
                    continue
                result[key][col].append(row[col])

        # convert lists to arrays
        for key in result:
            for col in result[key]:
                result[key][col] = np.array(result[key][col], dtype=dtype)
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
        # check whether key lists are consistent
        # key_list = [set(d.keys()) for d in args]
        # if not all(kl == key_list[0] for kl in key_list):
        #     msg = "All input dicts must have the same outer keys."
        #     raise ValueError(msg)

        merged_dict = {}
        for _dict in args:
            for row1, inner_dict in _dict.items():
                if row1 not in merged_dict:
                    merged_dict[row1] = {}
                for col, values in inner_dict.items():
                    if col in merged_dict[row1]:
                        msg = f"Duplicate column '{col}' for row '{row1}'"
                        raise ValueError(msg)
                    else:
                        merged_dict[row1][col] = values
        return merged_dict

    return _worker
