import json
from io import BytesIO

import numpy as np
import uuid
from zstandard import ZstdCompressor, ZstdDecompressor

from biomol.core.container import FeatureContainer
from biomol.core.feature import EdgeFeature, NodeFeature
from biomol.core.index import IndexTable


def concat_containers(*containers: FeatureContainer) -> FeatureContainer:
    """Efficiently concatenate multiple FeatureContainer instances into one."""
    if not containers:
        msg = "No containers provided for concatenation."
        raise ValueError(msg)

    container_type = type(containers[0])
    if not all(isinstance(c, container_type) for c in containers):
        msg = "All containers must be of the same type."
        raise ValueError(msg)

    # --- Collect all feature values first ---
    node_value_lists: dict[str, list[np.ndarray]] = {}
    edge_src_lists: dict[str, list[np.ndarray]] = {}
    edge_dst_lists: dict[str, list[np.ndarray]] = {}
    edge_value_lists: dict[str, list[np.ndarray]] = {}
    edge_desc: dict[str, str] = {}
    node_desc: dict[str, str] = {}

    offset = 0
    for container in containers:
        # Edge features
        for key, feature in container.edge_features.items():
            edge_desc[key] = feature.description
            src_shifted = feature.src_indices + offset
            dst_shifted = feature.dst_indices + offset
            edge_src_lists.setdefault(key, []).append(src_shifted)
            edge_dst_lists.setdefault(key, []).append(dst_shifted)
            edge_value_lists.setdefault(key, []).append(feature.value)

        # Node features
        for key, feature in container.node_features.items():
            node_desc[key] = feature.description
            node_value_lists.setdefault(key, []).append(feature.value)

        # Update offset based on current node count
        if container.node_features:
            offset += len(next(iter(container.node_features.values())).value)

    # --- Perform concatenation once per key ---
    concatenated_edge_features = {
        key: EdgeFeature(
            src_indices=np.concatenate(edge_src_lists[key]),
            dst_indices=np.concatenate(edge_dst_lists[key]),
            value=np.concatenate(edge_value_lists[key]),
            description=edge_desc[key],
        )
        for key in edge_value_lists
    }

    concatenated_node_features = {
        key: NodeFeature(
            value=np.concatenate(node_value_lists[key]),
            description=node_desc[key],
        )
        for key in node_value_lists
    }

    return container_type(
        node_features=concatenated_node_features,
        edge_features=concatenated_edge_features,
    )


def flatten_data(data: dict) -> tuple[dict, dict]:
    """Flatten a nested dictionary."""
    template = {}
    flatten = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            _key = str(uuid.uuid4())
            template[key] = _key
            buffer = BytesIO()
            np.save(buffer, np.ascontiguousarray(value), allow_pickle=False)
            flatten[_key] = buffer.getvalue()
        elif isinstance(value, dict):
            _template, _flatten = flatten_data(value)
            template[key] = _template
            flatten.update(_flatten)
        elif isinstance(value, FeatureContainer | IndexTable):
            _template, _flatten = flatten_data(value.to_dict())
            template[key] = _template
            flatten.update(_flatten)
        else:
            template[key] = value
    return template, flatten


def to_bytes(data_dict: dict, level: int = 6) -> bytes:
    """Serialize the container to zstd-compressed bytes.

    Parameters
    ----------
    level: int, optional
        The compression level for zstd (default is 6).
    """
    template, flattened_data = flatten_data(data_dict)
    header = {
        "template": template,
        "arrays": {key: len(value) for key, value in flattened_data.items()},
    }
    header_bytes = json.dumps(header).encode("utf-8")
    payload = b"".join(flattened_data[key] for key in flattened_data)
    raw = len(header_bytes).to_bytes(8, "little") + header_bytes + payload
    return ZstdCompressor(level=level).compress(raw)


def reconstruct_data(template: dict, flatten: dict) -> dict:
    """Reconstruct a nested dictionary from its flattened form."""
    data = {}
    for key, value in template.items():
        if isinstance(value, str) and value in flatten:
            buffer = BytesIO(flatten[value])
            buffer.seek(0)
            arr = np.load(buffer, allow_pickle=False)
            data[key] = arr
        elif isinstance(value, dict):
            data[key] = reconstruct_data(value, flatten)
        else:
            data[key] = value
    return data


def from_bytes(byte_data: bytes) -> dict:
    """Deserialize the container from zstd-compressed bytes."""
    raw = ZstdDecompressor().decompress(byte_data)
    hlen = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8 : 8 + hlen].decode("utf-8"))
    payload = raw[8 + hlen :]

    offset = 0
    flatten_data = {}
    for key, ln in header["arrays"].items():
        chunk = payload[offset : offset + ln]
        offset += ln
        flatten_data[key] = chunk

    template_dict = header["template"]
    return reconstruct_data(template_dict, flatten_data)


def to_dict(data: dict) -> dict:
    """Convert all FeatureContainer and IndexTable instances in the dictionary to regular dictionaries."""
    result = {}
    for key, value in data.items():
        if isinstance(value, (FeatureContainer, IndexTable)):
            result[key] = value.to_dict()
        elif isinstance(value, dict):
            result[key] = to_dict(value)
        else:
            result[key] = value
    return result
