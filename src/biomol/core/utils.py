import json
from collections.abc import Mapping
from io import BytesIO
from typing import Any

import numpy as np
from zstandard import ZstdCompressor, ZstdDecompressor


def to_bytes(data: Mapping[str, Any], level: int = 6) -> bytes:
    """Serialize a dictionary containing NumPy arrays to zstd-compressed bytes.

    Parameters
    ----------
    data : Mapping[str, Any]
        A Mapping containing the NumPy arrays and other data to serialize.
    level : int, optional
        The compression level for zstd (default is 6).
    """

    def _flatten_data(
        data: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        template = {}
        flatten = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                _key = str(id(value))
                template[key] = _key
                buffer = BytesIO()
                np.save(buffer, np.ascontiguousarray(value), allow_pickle=False)
                flatten[_key] = buffer.getvalue()
            elif isinstance(value, dict):
                _template, _flatten = _flatten_data(value)
                template[key] = _template
                flatten.update(_flatten)
            else:
                template[key] = value
        return template, flatten

    template, flatten_data = _flatten_data(data)
    header = {
        "template": template,
        "arrays": {key: len(value) for key, value in flatten_data.items()},
    }
    header_bytes = json.dumps(header).encode("utf-8")
    payload = b"".join(flatten_data[key] for key in flatten_data)
    raw = len(header_bytes).to_bytes(8, "little") + header_bytes + payload
    return ZstdCompressor(level=level).compress(raw)


def load_bytes(byte_data: bytes) -> Mapping[str, Any]:
    """Deserialize zstd-compressed bytes back into a dictionary."""

    def _reconstruct_data(
        template: dict[str, Any],
        flatten: dict[str, Any],
    ) -> dict[str, Any]:
        data = {}
        for key, value in template.items():
            if isinstance(value, str) and value in flatten:
                buffer = BytesIO(flatten[value])
                buffer.seek(0)
                arr = np.load(buffer, allow_pickle=False)
                data[key] = arr
            elif isinstance(value, dict):
                data[key] = _reconstruct_data(value, flatten)
            else:
                data[key] = value
        return data

    with ZstdDecompressor().stream_reader(BytesIO(byte_data)) as reader:
        raw = reader.read()
    hlen = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8 : 8 + hlen].decode("utf-8"))
    payload = memoryview(raw)[8 + hlen :]

    offset = 0
    flatten_data = {}
    for key, ln in header["arrays"].items():
        chunk = payload[offset : offset + ln]
        offset += ln
        flatten_data[key] = chunk

    template_dict = header["template"]
    return _reconstruct_data(template_dict, flatten_data)
