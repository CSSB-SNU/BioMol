import json
from io import BytesIO

import numpy as np
import uuid
from zstandard import ZstdCompressor, ZstdDecompressor

from biomol.core.container import FeatureContainer
from biomol.core.feature import EdgeFeature, NodeFeature


def concat_containers(*containers: FeatureContainer) -> FeatureContainer:
    """Concatenate multiple FeatureContainer instances into one.

    Parameters
    ----------
    *containers : FeatureContainer
        Instances of FeatureContainer (or its subclasses) to concatenate.

    Returns
    -------
    FeatureContainer
        A new FeatureContainer instance containing concatenated features.

    Raises
    ------
    ValueError
        If the containers are of different types or have inconsistent node feature
        lengths.
    """
    if not containers:
        msg = "No containers provided for concatenation."
        raise ValueError(msg)

    container_type = type(containers[0])
    if not all(isinstance(c, container_type) for c in containers):
        msg = "All containers must be of the same type."
        raise ValueError(msg)

    concatenated_node_features = {}
    concatenated_edge_features = {}

    for container in containers:
        if len(concatenated_node_features) == 0:
            offset = 0
        else:
            offset = len(next(iter(concatenated_node_features.values())).value)
        for key, feature in container.edge_features.items():
            if key in concatenated_edge_features:
                new_src_indices = feature.src_indices + offset
                new_dst_indices = feature.dst_indices + offset
                concatenated_edge_features[key] = EdgeFeature(
                    src_indices=np.concatenate(
                        [
                            concatenated_edge_features[key].src_indices,
                            new_src_indices,
                        ],
                    ),
                    dst_indices=np.concatenate(
                        [
                            concatenated_edge_features[key].dst_indices,
                            new_dst_indices,
                        ],
                    ),
                    value=np.concatenate(
                        [concatenated_edge_features[key].value, feature.value],
                    ),
                    description=feature.description,
                )
            else:
                concatenated_edge_features[key] = feature
        for key, feature in container.node_features.items():
            if key in concatenated_node_features:
                concatenated_node_features[key] = NodeFeature(
                    value=np.concatenate(
                        [concatenated_node_features[key].value, feature.value],
                    ),
                    description=feature.description,
                )
            else:
                concatenated_node_features[key] = feature
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
        elif isinstance(value, FeatureContainer):
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
