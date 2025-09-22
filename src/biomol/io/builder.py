from typing import Optional, TypeVar, get_args, overload

from biomol.core.biomol import BioMol
from biomol.core.container import (
    FeatureContainer,
    AtomContainer,
    ResidueContainer,
    ChainContainer,
)
from biomol.core.feature import EdgeFeature, NodeFeature
from biomol.core.index import IndexTable
from biomol.core.view import ViewProtocol
from biomol.io.cache import ParsingCache


T_MOL = TypeVar("T_MOL", bound=BioMol)


class MolBuilder:
    """Build a BioMol object from parsed data stored in ParsingCache."""

    @overload
    def __init__(self, parsing_cache: ParsingCache) -> None: ...
    @overload
    def __init__(self, parsing_cache: ParsingCache, mol_guide: type[T_MOL]) -> None: ...
    @overload
    def __init__(self, parsing_cache: ParsingCache, mol_guide: str) -> None: ...

    def __init__(
        self,
        parsing_cache: ParsingCache,
        mol_guide: Optional[type[T_MOL] | str] = None,
    ) -> None:
        if mol_guide is None:
            self.mol_guide = BioMol
        elif isinstance(mol_guide, str):
            module = __import__(mol_guide, fromlist=["MolType"])
            self.mol_guide = getattr(module, "MolType")
        else:
            self.mol_guide = mol_guide

        self.parsing_cache = parsing_cache

    def _get_featurenames(self) -> tuple[list[str], list[str], list[str]]:
        generic_biomol_base = [
            base
            for base in self.mol_guide.__orig_bases__
            if hasattr(base, "__origin__") and base.__origin__ is BioMol
        ]

        def _get_featurenames_unit(protocol: type[ViewProtocol]) -> list[str]:
            base_attrs = set(dir(ViewProtocol))
            protocol_attrs = set(dir(protocol))
            new_attrs = sorted(list(protocol_attrs - base_attrs))
            return [
                attr
                for attr in new_attrs
                if isinstance(getattr(protocol, attr, None), property)
            ]

        atom_feature_names, res_feature_names, chain_feature_names = map(
            _get_featurenames_unit, get_args(generic_biomol_base[0])
        )
        return atom_feature_names, res_feature_names, chain_feature_names

    def _build_container(self, feature_names: list[str]) -> FeatureContainer:
        features = {name: self.parsing_cache[name] for name in feature_names}
        node_features = {
            name: feat
            for name, feat in features.items()
            if isinstance(feat, NodeFeature)
        }
        edge_features = {
            name: feat
            for name, feat in features.items()
            if isinstance(feat, EdgeFeature)
        }
        return FeatureContainer(
            node_features=node_features, edge_features=edge_features
        )

    def _get_index(self) -> IndexTable:
        index_table_dict = self.parsing_cache[IndexTable]
        if len(index_table_dict) != 1:
            msg = "You must provide one index table for one biomol."
            raise ValueError(msg)

        index_table = next(iter(index_table_dict.values()))
        return index_table

    def build(self) -> BioMol:
        # TODO: use _build_container instead of repeating code after rebase docs/sphinx branch
        atom_feat_names, res_feat_names, chain_feat_names = self._get_featurenames()
        atom_container = AtomContainer(
            node_features={
                name: self.parsing_cache[name]
                for name in atom_feat_names
                if isinstance(self.parsing_cache[name], NodeFeature)
            },
            edge_features={
                name: self.parsing_cache[name]
                for name in atom_feat_names
                if isinstance(self.parsing_cache[name], EdgeFeature)
            },
        )
        res_container = ResidueContainer(
            node_features={
                name: self.parsing_cache[name]
                for name in res_feat_names
                if isinstance(self.parsing_cache[name], NodeFeature)
            },
            edge_features={
                name: self.parsing_cache[name]
                for name in res_feat_names
                if isinstance(self.parsing_cache[name], EdgeFeature)
            },
        )
        chain_container = ChainContainer(
            node_features={
                name: self.parsing_cache[name]
                for name in chain_feat_names
                if isinstance(self.parsing_cache[name], NodeFeature)
            },
            edge_features={
                name: self.parsing_cache[name]
                for name in chain_feat_names
                if isinstance(self.parsing_cache[name], EdgeFeature)
            },
        )

        index_table = self._get_index()

        return self.mol_guide(
            atom_container=atom_container,
            residue_container=res_container,
            chain_container=chain_container,
            index_table=index_table,
        )
