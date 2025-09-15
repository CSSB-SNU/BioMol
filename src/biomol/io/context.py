from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

from biomol.io.schema import FeatureSpec

if TYPE_CHECKING:
    from collections.abc import Mapping

    from biomol.core.feature import Feature


class ParsingContext:
    """
    Manages features and vocabularies during parsing.

    it stores features by name and allows building vocabularies
    it's like an excercise book keeping track of all features and vocabs
    it will be cleared after each structure is parsed, after build a BioMol.
    """

    def __init__(self) -> None:
        self._features: dict[str, Feature] = {}
        self._specs: dict[str, FeatureSpec] = {}
        # mapping from one feature value to an index
        self._lookup_cache: dict[str, dict[Any, int]] = {}

    def add_feature(
        self,
        spec: FeatureSpec,
        feature: Feature,
    ) -> None:
        """Add a feature to the context."""
        self._features[spec.name] = feature
        self._specs[spec.name] = spec

    def get_feature(self, key: str) -> Feature:
        """Get a feature by its name."""
        try:
            return self._features[key]
        except KeyError:
            msg = f"Feature '{key}' not found. Available features: {list(self._features.keys())}"
            raise KeyError(msg) from None

    @overload
    def get_features(self, key: FeatureSpec) -> Mapping[str, Feature]: ...

    @overload
    def get_features(self, key: list[str]) -> Mapping[str, Feature]: ...

    def get_features(self, key: list[str] | FeatureSpec) -> Mapping[str, Feature]:
        """Get a feature by name or all features at a specific level."""
        if isinstance(key, list):
            return {k: self.get_feature(k) for k in key}

        elif isinstance(key, FeatureSpec):
            return {
                fname: feature
                for fname, feature in self._features.items()
                if self._specs[fname].matches(key)
            }
        else:
            msg = "Key must be a string or FeatureLevel."
            raise NotImplementedError(msg)

    def get_lookup_dict(self, feature_name: str) -> dict[Any, int]:
        """Get the lookup dictionary for a feature, creating it if necessary."""
        if feature_name in self._lookup_cache:
            return self._lookup_cache[feature_name]

        aux_feature = self.get_feature(feature_name)
        if aux_feature is None:
            msg = f"Feature '{feature_name}' not found in context."
            raise ValueError(msg)

        lookup_dict: dict[Any, int] = {
            val: i for i, val in enumerate(aux_feature.value)
        }
        self._lookup_cache[feature_name] = lookup_dict
        return lookup_dict
