"""Tests for copy methods in Feature, FeatureContainer, IndexTable, and BioMol."""

import numpy as np

from biomol import BioMol
from biomol.core import EdgeFeature, FeatureContainer, IndexTable, NodeFeature


class TestFeatureCopy:
    def test_node_feature_copy_creates_new_instance(self):
        original = NodeFeature(np.array([[1, 2, 3], [4, 5, 6]]))
        copied = original.copy()

        assert copied is not original
        assert isinstance(copied, NodeFeature)

    def test_node_feature_copy_deep_copy(self):
        original = NodeFeature(np.array([[1, 2, 3], [4, 5, 6]]))
        copied = original.copy()
        copied_modified = NodeFeature(copied.value + 10)

        np.testing.assert_array_equal(original.value, np.array([[1, 2, 3], [4, 5, 6]]))
        np.testing.assert_array_equal(
            copied_modified.value,
            np.array([[11, 12, 13], [14, 15, 16]]),
        )

    def test_edge_feature_copy_creates_new_instance(self):
        original = EdgeFeature(
            np.array([1.0, 2.0, 3.0]),
            src_indices=np.array([0, 1, 2]),
            dst_indices=np.array([1, 2, 0]),
        )
        copied = original.copy()

        assert copied is not original
        assert isinstance(copied, EdgeFeature)

    def test_edge_feature_copy_deep_copy(self):
        original = EdgeFeature(
            np.array([1.0, 2.0, 3.0]),
            src_indices=np.array([0, 1, 2]),
            dst_indices=np.array([1, 2, 0]),
        )
        copied = original.copy()

        np.testing.assert_array_equal(copied.value, original.value)
        np.testing.assert_array_equal(copied.src_indices, original.src_indices)
        np.testing.assert_array_equal(copied.dst_indices, original.dst_indices)


class TestFeatureContainerCopy:
    def test_feature_container_copy_creates_new_instance(self):
        original = FeatureContainer(
            {"coord": NodeFeature(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))},
        )
        copied = original.copy()

        assert copied is not original
        assert isinstance(copied, FeatureContainer)

    def test_feature_container_copy_deep_copy(self):
        original = FeatureContainer(
            {"coord": NodeFeature(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))},
        )
        copied = original.copy()

        np.testing.assert_array_equal(copied["coord"].value, original["coord"].value)
        assert copied["coord"] is not original["coord"]

    def test_feature_container_copy_with_multiple_features(self):
        original = FeatureContainer(
            {
                "coord": NodeFeature(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
                "element": NodeFeature(np.array(["C", "N"])),
                "bond": EdgeFeature(
                    np.array([1.5]),
                    src_indices=np.array([0]),
                    dst_indices=np.array([1]),
                ),
            },
        )
        copied = original.copy()

        assert set(copied.keys()) == set(original.keys())

        keys = original.keys()
        for key in keys:
            assert copied[key] is not original[key]
            np.testing.assert_array_equal(copied[key].value, original[key].value)


class TestIndexTableCopy:
    def test_index_table_copy_creates_new_instance(self):
        original = IndexTable.from_parents(
            atom_to_res=np.array([0, 0, 1, 1, 2]),
            res_to_chain=np.array([0, 0, 1]),
        )
        copied = original.copy()

        assert copied is not original
        assert isinstance(copied, IndexTable)

    def test_index_table_copy_deep_copy(self):
        original = IndexTable.from_parents(
            atom_to_res=np.array([0, 0, 1, 1, 2]),
            res_to_chain=np.array([0, 0, 1]),
        )
        copied = original.copy()

        np.testing.assert_array_equal(copied.atom_to_res, original.atom_to_res)
        np.testing.assert_array_equal(copied.res_to_chain, original.res_to_chain)
        np.testing.assert_array_equal(copied.res_atom_indptr, original.res_atom_indptr)
        np.testing.assert_array_equal(
            copied.res_atom_indices,
            original.res_atom_indices,
        )
        np.testing.assert_array_equal(
            copied.chain_res_indptr,
            original.chain_res_indptr,
        )
        np.testing.assert_array_equal(
            copied.chain_res_indices,
            original.chain_res_indices,
        )

        assert copied.atom_to_res is not original.atom_to_res
        assert copied.res_to_chain is not original.res_to_chain
        assert copied.res_atom_indptr is not original.res_atom_indptr
        assert copied.res_atom_indices is not original.res_atom_indices
        assert copied.chain_res_indptr is not original.chain_res_indptr
        assert copied.chain_res_indices is not original.chain_res_indices

    def test_index_table_copy_preserves_mapping(self):
        original = IndexTable.from_parents(
            atom_to_res=np.array([0, 0, 1, 1, 2]),
            res_to_chain=np.array([0, 0, 1]),
        )
        copied = original.copy()

        np.testing.assert_array_equal(
            copied.atoms_to_residues(np.array([0, 2, 4])),
            original.atoms_to_residues(np.array([0, 2, 4])),
        )

        np.testing.assert_array_equal(
            copied.residues_to_chains(np.array([0, 2])),
            original.residues_to_chains(np.array([0, 2])),
        )


class TestBioMolCopy:
    def test_biomol_copy_creates_new_instance(self):
        original = BioMol(
            atom_container=FeatureContainer(
                {"coord": NodeFeature(np.array([[1.0, 2.0, 3.0]]))},
            ),
            residue_container=FeatureContainer(
                {"res_name": NodeFeature(np.array(["ALA"]))},
            ),
            chain_container=FeatureContainer(
                {"chain_id": NodeFeature(np.array(["A"]))},
            ),
            index_table=IndexTable.from_parents(
                atom_to_res=np.array([0]),
                res_to_chain=np.array([0]),
            ),
            metadata={"pdb_id": "1ABC"},
        )
        copied = original.copy()

        assert copied is not original
        assert isinstance(copied, BioMol)

    def test_biomol_copy_deep_copy(self):
        original = BioMol(
            atom_container=FeatureContainer(
                {"coord": NodeFeature(np.array([[1.0, 2.0, 3.0]]))},
            ),
            residue_container=FeatureContainer(
                {"res_name": NodeFeature(np.array(["ALA"]))},
            ),
            chain_container=FeatureContainer(
                {"chain_id": NodeFeature(np.array(["A"]))},
            ),
            index_table=IndexTable.from_parents(
                atom_to_res=np.array([0]),
                res_to_chain=np.array([0]),
            ),
            metadata={"pdb_id": "1ABC", "resolution": 2.5},
        )
        copied = original.copy()

        assert copied.atoms.get_container() is not original.atoms.get_container()
        assert copied.residues.get_container() is not original.residues.get_container()
        assert copied.chains.get_container() is not original.chains.get_container()
        assert copied.index_table is not original.index_table
        assert copied.metadata is not original.metadata

        np.testing.assert_array_equal(
            copied.atoms.get_feature("coord").value,
            original.atoms.get_feature("coord").value,
        )
        np.testing.assert_array_equal(
            copied.residues.get_feature("res_name").value,
            original.residues.get_feature("res_name").value,
        )
        np.testing.assert_array_equal(
            copied.chains.get_feature("chain_id").value,
            original.chains.get_feature("chain_id").value,
        )
        assert copied.metadata == original.metadata

    def test_biomol_copy_metadata_independence(self):
        """Test that modifying copied metadata doesn't affect original."""
        original = BioMol(
            atom_container=FeatureContainer(
                {"coord": NodeFeature(np.array([[1.0, 2.0, 3.0]]))},
            ),
            residue_container=FeatureContainer(
                {"res_name": NodeFeature(np.array(["ALA"]))},
            ),
            chain_container=FeatureContainer(
                {"chain_id": NodeFeature(np.array(["A"]))},
            ),
            index_table=IndexTable.from_parents(
                atom_to_res=np.array([0]),
                res_to_chain=np.array([0]),
            ),
            metadata={"pdb_id": "1ABC"},
        )
        copied = original.copy()
        copied.metadata["new_key"] = "new_value"

        assert "new_key" not in original.metadata
        assert original.metadata == {"pdb_id": "1ABC"}
        assert copied.metadata == {"pdb_id": "1ABC", "new_key": "new_value"}
