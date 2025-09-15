import numpy as np

from biomol.core.feature import EdgeFeature, NodeFeature
from biomol.io.context import ParsingContext
from biomol.io.isa import InstructionSet
from biomol.io.schema import FeatureSpec


@InstructionSet.register("identity")
def identity_instruction(record: dict, specs: list[FeatureSpec]) -> list[NodeFeature]:
    """
    Map fields to node features directly.

    the order of fields in the record should match the order of specs.
    """
    if len(record) != len([0 for spec in specs if "mask" not in spec.name]):
        msg = "Number of fields and specs must match."
        raise ValueError(msg)

    features = []
    for i, field_name in enumerate(record.keys()):
        spec = specs[i]
        description = spec.description or field_name
        dtype = spec.dtype
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


@InstructionSet.register("stack")
def stack_instruction(record: dict, specs: list[FeatureSpec]) -> list[NodeFeature]:
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
        msg = "Stack mapper supports less than 2 specs."
        raise ValueError(msg)

    data_spec = specs[0]
    dtype = data_spec.dtype
    description = data_spec.description

    unk_char = data_spec.on_missing
    mask_description = f"{description}_mask"

    result_data = []
    mask = []
    for data in record.values():
        if len(data) != len(next(iter(record.values()))):
            msg = "All fields must have the same length."
            raise ValueError(msg)

        mask.append([d not in unk_char for d in data])
        formatted_data = [
            dtype(x) if x not in unk_char else dtype(unk_char[x]) for x in data
        ]
        result_data.append(np.array(formatted_data, dtype=dtype))

    stacked = np.stack(result_data, axis=-1)
    stacked_mask = np.all(np.stack(mask, axis=-1), axis=-1).astype(bool)

    return [
        NodeFeature(value=stacked, description=description),
        NodeFeature(value=stacked_mask, description=mask_description),
    ]


@InstructionSet.register("bond")
def bond_instruction(
    record: dict,
    specs: list[FeatureSpec],
    context: ParsingContext,
    context_key: tuple[str, ...] = (),
) -> list[EdgeFeature]:
    """
    Map source node id, target node id, bond feature to edge features.

    the length of all fields should match.
    when parsing any edge feature, the source and target node ids are required.
    the node id -> index mapping is retrieved from the context using the context_key.

    record: source ids, target ids, bond features

    """
    record_iter = iter(record.values())
    source_node_id = next(record_iter)
    target_node_id = next(record_iter)
    if len(source_node_id) != len(target_node_id):
        msg = "Source and target node ids must have the same length."
        raise ValueError(msg)

    id_mapping = context.get_lookup_dict(context_key[0])
    source_node_idx = np.array([id_mapping[sid] for sid in source_node_id])
    target_node_idx = np.array([id_mapping[tid] for tid in target_node_id])

    features = []
    data_specs = [spec for spec in specs if "mask" not in spec.name]
    for spec, data in zip(data_specs, record_iter):
        if len(data) != len(source_node_id):
            msg = "All fields must have the same length."
            raise ValueError(msg)

        on_missing = spec.on_missing
        formatted_data = [
            spec.dtype(d) if d not in on_missing else on_missing[d] for d in data
        ]
        description = spec.description or spec.name
        features.append(
            EdgeFeature(
                value=np.array(formatted_data, dtype=spec.dtype),
                src_indices=source_node_idx,
                dst_indices=target_node_idx,
                description=description,
            )
        )

        if on_missing:
            mask = np.array([d not in on_missing for d in data], dtype=bool)
            mask_description = f"{description}_mask"
            features.append(
                EdgeFeature(
                    value=mask,
                    src_indices=source_node_idx,
                    dst_indices=target_node_idx,
                    description=mask_description,
                )
            )

    return features
