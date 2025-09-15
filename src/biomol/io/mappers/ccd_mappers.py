from typing import List, Type
import numpy as np
import re

from biomol.io.registry import MapperRegistry
from biomol.core.feature import NodeFeature, EdgeFeature
from biomol.io.schema import FeatureSpec
from biomol.io.context import ParsingContext


@MapperRegistry.register("identity")
def identity_mapper(record: dict, specs: List[FeatureSpec]) -> List[NodeFeature]:
    """
    Map fields to node features directly.

    the order of fields in the record should match the order of specs.
    """
    if len(record) != len([0 for spec in specs if "mask" not in spec.name]):
        raise ValueError("Number of fields and specs must match.")

    features = []
    for i, field_name in enumerate(record.keys()):
        spec = specs[i]
        description = spec.description or field_name
        dtype: Type = spec.dtype
        unk_char = spec.on_missing
        field_data = record[field_name]
        field_data = [
            dtype(data) if data not in unk_char else dtype(unk_char[data])
            for data in field_data
        ]
        field_data = np.array(field_data, dtype=dtype)
        data_feature = NodeFeature(value=field_data, description=description)
        features.append(data_feature)
        if unk_char:
            mask = np.array([data not in unk_char for data in field_data], dtype=bool)
            mask_description = f"{description}_mask"
            mask_feature = NodeFeature(value=mask, description=mask_description)
            features.append(mask_feature)

    return features


@MapperRegistry.register("stack")
def stack_mapper(record: dict, specs: List[FeatureSpec]) -> List[NodeFeature]:
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
    if len([0 for spec in specs if "mask" not in spec.name]) != 1:
        raise ValueError("Stack mapper supports less than 2 specs.")

    data_spec = specs[0]
    dtype: Type = data_spec.dtype
    description = data_spec.description

    unk_char = data_spec.on_missing
    mask_description = f"{description}_mask"

    result_data = []
    mask = []
    for data in record.values():
        if len(data) != len(next(iter(record.values()))):
            raise ValueError("All fields must have the same length.")

        mask.append([d not in unk_char for d in data])
        data = [dtype(x) if x not in unk_char else dtype(unk_char[x]) for x in data]
        result_data.append(np.array(data, dtype=dtype))

    stacked = np.stack(result_data, axis=-1)
    stacked_mask = np.all(np.stack(mask, axis=-1), axis=-1).astype(bool)

    return [
        NodeFeature(value=stacked, description=description),
        NodeFeature(value=stacked_mask, description=mask_description),
    ]


@MapperRegistry.register("bond")
def bond_mapper(
    record: dict,
    specs: List[FeatureSpec],
    context: ParsingContext,
    context_key: List[str] = [],
) -> List[EdgeFeature]:
    """
    Map source node id, target node id, bond feature to edge features.

    the length of all fields should match.
    when parsing any edge feature, the source and target node ids are required.
    the node id -> index mapping is retrieved from the context using the context_key.

    record: source ids, target ids, bond features

    """
    source_node_id = record.pop(next(iter(record.keys())))
    target_node_id = record.pop(next(iter(record.keys())))
    if len(source_node_id) != len(target_node_id):
        raise ValueError("Source and target node ids must have the same length.")

    id_mapping = context.get_lookup_dict(context_key[0])
    source_node_idx = np.array([id_mapping[sid] for sid in source_node_id])
    target_node_idx = np.array([id_mapping[tid] for tid in target_node_id])

    features = []
    for i, field_name in enumerate(record.keys()):
        data = record[field_name]
        if len(data) != len(source_node_id):
            raise ValueError(
                f"All fields must have the same length. Field {field_name} has length {len(data)}."
            )

        spec = specs[i]
        on_missing = spec.on_missing
        data = [spec.dtype(d) if d not in on_missing else on_missing[d] for d in data]
        description = spec.description or field_name
        features.append(
            EdgeFeature(
                value=np.array(data, dtype=spec.dtype),
                src_indices=source_node_idx,
                dst_indices=target_node_idx,
                description=description,
            )
        )

        if on_missing:
            mask = np.array([d not in on_missing for d in data], dtype=bool)
            mask_description = f"{spec.description}_mask"
            features.append(
                EdgeFeature(
                    value=mask,
                    src_indices=source_node_idx,
                    dst_indices=target_node_idx,
                    description=mask_description,
                )
            )

    return features
