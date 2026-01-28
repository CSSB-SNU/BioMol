"""Tests for to_bytes and load_bytes serialization utilities."""

import numpy as np

from biomol import load_bytes, to_bytes


class TestBytesRoundTrip:
    def test_single_array(self):
        data = {"coord": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}
        result = load_bytes(to_bytes(data))

        np.testing.assert_array_equal(result["coord"], data["coord"])

    def test_multiple_arrays(self):
        data = {
            "coord": np.array([[1.0, 2.0, 3.0]]),
            "element": np.array([6, 7, 8]),
        }
        result = load_bytes(to_bytes(data))

        for key, value in data.items():
            np.testing.assert_array_equal(result[key], value)

    def test_nested_dict(self):
        data = {
            "atoms": {
                "coord": np.array([[1.0, 2.0, 3.0]]),
                "element": np.array([6]),
            },
            "residues": {
                "name": np.array([0]),
            },
        }
        result = load_bytes(to_bytes(data))

        np.testing.assert_array_equal(
            result["atoms"]["coord"],
            data["atoms"]["coord"],
        )
        np.testing.assert_array_equal(
            result["atoms"]["element"],
            data["atoms"]["element"],
        )
        np.testing.assert_array_equal(
            result["residues"]["name"],
            data["residues"]["name"],
        )

    def test_mixed_types(self):
        data = {
            "coord": np.array([[1.0, 2.0, 3.0]]),
            "name": "test",
            "count": 42,
        }
        result = load_bytes(to_bytes(data))

        np.testing.assert_array_equal(result["coord"], data["coord"])
        assert result["name"] == "test"
        assert result["count"] == 42

    def test_large_array(self):
        rng = np.random.default_rng(42)
        data = {"big": rng.random((1000, 128))}
        result = load_bytes(to_bytes(data))

        np.testing.assert_array_equal(result["big"], data["big"])

    def test_dtypes_and_shapes_preserved(self):
        data = {
            "f32": np.array([[1.0, 2.0]], dtype=np.float32),
            "f64": np.array([1.0], dtype=np.float64),
            "i32": np.array([[[1]]], dtype=np.int32),
            "i64": np.array([1], dtype=np.int64),
        }
        result = load_bytes(to_bytes(data))

        for key, value in data.items():
            assert result[key].dtype == value.dtype
            assert result[key].shape == value.shape

    def test_empty_array(self):
        data = {"empty": np.array([])}
        result = load_bytes(to_bytes(data))

        np.testing.assert_array_equal(result["empty"], data["empty"])
