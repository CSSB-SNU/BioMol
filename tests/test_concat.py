"""Tests for concat methods in FeatureContainer and BioMol."""

import numpy as np
import pytest

from biomol import BioMol
from biomol.core import EdgeFeature, FeatureContainer, IndexTable, NodeFeature
from biomol.exceptions import FeatureKeyError


class TestFeatureContainerConcat:
    def test_concat_node_features(self):
        c1 = FeatureContainer({"coord": NodeFeature(np.array([[1, 2, 3], [4, 5, 6]]))})
        c2 = FeatureContainer({"coord": NodeFeature(np.array([[7, 8, 9]]))})

        result = FeatureContainer.concat([c1, c2])

        assert len(result) == 3
        np.testing.assert_array_equal(
            result["coord"].value,
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        )

    def test_concat_edge_features_with_offset(self):
        c1 = FeatureContainer(
            {
                "coord": NodeFeature(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
                "bond": EdgeFeature(
                    np.array([1.5]),
                    src_indices=np.array([0]),
                    dst_indices=np.array([1]),
                ),
            },
        )
        c2 = FeatureContainer(
            {
                "coord": NodeFeature(np.array([[7.0, 8.0, 9.0]])),
                "bond": EdgeFeature(
                    np.array([2.5]),
                    src_indices=np.array([0]),
                    dst_indices=np.array([0]),
                ),
            },
        )

        result = FeatureContainer.concat([c1, c2])

        bond: EdgeFeature = result["bond"]  # pyright: ignore[reportAssignmentType]
        np.testing.assert_array_equal(bond.src_indices, np.array([0, 2]))
        np.testing.assert_array_equal(bond.dst_indices, np.array([1, 2]))

    def test_concat_three_edge_features_cumulative_offset(self):
        c1 = FeatureContainer(
            {
                "pos": NodeFeature(np.array([[1.0], [2.0]])),
                "edge": EdgeFeature(
                    np.array([10.0]),
                    src_indices=np.array([0]),
                    dst_indices=np.array([1]),
                ),
            },
        )
        c2 = FeatureContainer(
            {
                "pos": NodeFeature(np.array([[3.0], [4.0], [5.0]])),
                "edge": EdgeFeature(
                    np.array([20.0, 30.0]),
                    src_indices=np.array([0, 1]),
                    dst_indices=np.array([1, 2]),
                ),
            },
        )
        c3 = FeatureContainer(
            {
                "pos": NodeFeature(np.array([[6.0]])),
                "edge": EdgeFeature(
                    np.array([40.0]),
                    src_indices=np.array([0]),
                    dst_indices=np.array([0]),
                ),
            },
        )

        result = FeatureContainer.concat([c1, c2, c3])

        edge: EdgeFeature = result["edge"]  # pyright: ignore[reportAssignmentType]
        np.testing.assert_array_equal(edge.src_indices, np.array([0, 2, 3, 5]))
        np.testing.assert_array_equal(edge.dst_indices, np.array([1, 3, 4, 5]))

    def test_concat_mismatched_keys_raises(self):
        c1 = FeatureContainer({"coord": NodeFeature(np.array([[1, 2, 3]]))})
        c2 = FeatureContainer({"position": NodeFeature(np.array([[4, 5, 6]]))})

        with pytest.raises(FeatureKeyError, match="same feature keys"):
            FeatureContainer.concat([c1, c2])

    def test_concat_empty_list_raises(self):
        with pytest.raises(ValueError, match="No FeatureContainer instances"):
            FeatureContainer.concat([])

    def test_concat_single_container_returns_same(self):
        c1 = FeatureContainer({"coord": NodeFeature(np.array([[1, 2, 3]]))})
        result = FeatureContainer.concat([c1])
        assert result is c1


class TestBioMolConcat:
    def test_concat_and_add_operator(self):
        mol1 = BioMol(
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
        mol2 = BioMol(
            atom_container=FeatureContainer(
                {"coord": NodeFeature(np.array([[4.0, 5.0, 6.0]]))},
            ),
            residue_container=FeatureContainer(
                {"res_name": NodeFeature(np.array(["GLY"]))},
            ),
            chain_container=FeatureContainer(
                {"chain_id": NodeFeature(np.array(["B"]))},
            ),
            index_table=IndexTable.from_parents(
                atom_to_res=np.array([0]),
                res_to_chain=np.array([0]),
            ),
        )

        result_concat = BioMol.concat([mol1, mol2])
        result_add = mol1 + mol2

        for result in [result_concat, result_add]:
            assert len(result.atoms) == 2
            assert len(result.residues) == 2
            assert len(result.chains) == 2

        assert result_concat.metadata == {"pdb_id": "1ABC"}

    def test_concat_three_biomols_cumulative_offset(self):
        mol1 = BioMol(
            atom_container=FeatureContainer(
                {"pos": NodeFeature(np.array([[1.0], [2.0]]))},
            ),
            residue_container=FeatureContainer({"r": NodeFeature(np.array([100]))}),
            chain_container=FeatureContainer({"c": NodeFeature(np.array([1000]))}),
            index_table=IndexTable.from_parents(
                atom_to_res=np.array([0, 0]),
                res_to_chain=np.array([0]),
            ),
        )
        mol2 = BioMol(
            atom_container=FeatureContainer(
                {"pos": NodeFeature(np.array([[3.0], [4.0], [5.0]]))},
            ),
            residue_container=FeatureContainer(
                {"r": NodeFeature(np.array([200, 300]))},
            ),
            chain_container=FeatureContainer({"c": NodeFeature(np.array([2000]))}),
            index_table=IndexTable.from_parents(
                atom_to_res=np.array([0, 1, 1]),
                res_to_chain=np.array([0, 0]),
            ),
        )
        mol3 = BioMol(
            atom_container=FeatureContainer(
                {"pos": NodeFeature(np.array([[6.0]]))},
            ),
            residue_container=FeatureContainer({"r": NodeFeature(np.array([400]))}),
            chain_container=FeatureContainer({"c": NodeFeature(np.array([3000]))}),
            index_table=IndexTable.from_parents(
                atom_to_res=np.array([0]),
                res_to_chain=np.array([0]),
            ),
        )

        result = BioMol.concat([mol1, mol2, mol3])

        np.testing.assert_array_equal(
            result.index_table.atom_to_res,
            np.array([0, 0, 1, 2, 2, 3]),
        )
        np.testing.assert_array_equal(
            result.index_table.res_to_chain,
            np.array([0, 1, 1, 2]),
        )
        assert len(result.atoms) == 6
        assert len(result.residues) == 4
        assert len(result.chains) == 3

    def test_concat_empty_list_raises(self):
        with pytest.raises(ValueError, match="Cannot concatenate an empty list"):
            BioMol.concat([])
