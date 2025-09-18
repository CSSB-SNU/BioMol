import numpy as np
from numpy.typing import NDArray

from biomol.core.feature import EdgeFeature, NodeFeature


def identity_instruction(
    data: list[str | int] | NDArray,
    on_missing: dict[str, float] | None = None,
    dtype: type = float,
    description: str | None = None,
) -> tuple[NodeFeature, NodeFeature] | NodeFeature:
    """
    Map fields to node features directly.

    the order of fields in the record should match the order of specs.
    """
    if on_missing:
        formatted_data = [
            dtype(datum) if datum not in on_missing else dtype(on_missing[datum])
            for datum in data
        ]
    else:
        formatted_data = [dtype(datum) for datum in data]
    formatted_data = np.array(formatted_data, dtype=dtype)
    data_feature = NodeFeature(value=formatted_data, description=description)
    if on_missing:
        mask = np.array(
            [datum not in on_missing for datum in formatted_data],
            dtype=bool,
        )
        mask_description = f"{description}_mask"
        mask_feature = NodeFeature(value=mask, description=mask_description)

        return data_feature, mask_feature
    return data_feature


def stack_instruction(
    *args: list[str | int] | NDArray,
    on_missing: dict[str, float],
    dtype: type = float,
    description: str | None = None,
) -> NodeFeature:
    """
    Stacking mapper for stacking multiple fields into one feature and one mask feature.

    each field should have the same length and type.
    the order of fields in the record should match the order of specs.
    the resulting feature will have shape (N, M) where N is the length of each field.
    the mask feature will have shape (N,) indicating valid entries.

    Example:
        field1: [0,1,2,3,4]
        field2: [7,8,9,10,'?']
        result: [[0,7], [1,8], [2,9], [3,10], [4,0]]
        mask: [True, True, True, True, False]
    """
    mask_description = f"{description}_mask"

    result_data = []
    mask = []
    for data in args:
        if len(data) != len(next(iter(args))):
            msg = "All fields must have the same length."
            raise ValueError(msg)

        mask.append([d not in on_missing for d in data])
        formatted_data = [
            dtype(x) if x not in on_missing else dtype(on_missing[x]) for x in data
        ]
        result_data.append(np.array(formatted_data, dtype=dtype))

    stacked = np.stack(result_data, axis=-1)
    stacked_mask = np.all(np.stack(mask, axis=-1), axis=-1).astype(bool)

    return (
        NodeFeature(value=stacked, description=description),
        NodeFeature(value=stacked_mask, description=mask_description),
    )


def bond_instruction(
    *args: list[str | int] | NDArray,
    src: list[str | int] | NDArray,
    dst: list[str | int] | NDArray,
    atom_id: NodeFeature,
    dtype: type = str,
    description: str = "",
) -> EdgeFeature:
    """
    Map source node id, target node id, bond feature to edge features.

    the length of all fields should match.
    when parsing any edge feature, the source and target node ids are required.
    the node id -> index mapping is retrieved from the context using the context_key.

    record: source ids, target ids, bond features

    """
    values = []
    src = np.array(src)
    dst = np.array(dst)
    order = np.argsort(atom_id.value)
    src_indices = order[np.searchsorted(atom_id.value, src, sorter=order)]
    dst_indices = order[np.searchsorted(atom_id.value, dst, sorter=order)]
    for data in args:
        if len(data) != len(next(iter(args))):
            msg = "All fields must have the same length."
            raise ValueError(msg)
        formatted_data = [dtype(x) for x in data]
        values.append(np.array(formatted_data, dtype=dtype))
    value = np.stack(values, axis=-1) if len(values) > 1 else values[0]
    value = value.astype(dtype)

    return EdgeFeature(
        value=value,
        src_indices=np.array(src_indices, dtype=int),
        dst_indices=np.array(dst_indices, dtype=int),
        description=description,
    )
