from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Literal
from collections.abc import Sequence


class FeatureLevel(Enum):
    """Enumeration for the hierarchical level of a feature."""

    STRUCTURE = "structure"
    CHAIN = "chain"
    RESIDUE = "residue"
    ATOM = "atom"


class FeatureKind(Enum):
    """Enumeration for the kind of a feature (node, edge, or auxiliary)."""

    NODE = "node"
    EDGE = "edge"
    AUX = "auxiliary"


@dataclass(frozen=True)
class FeatureSpec:
    """
    A blueprint for a feature, defining its properties and metadata.

    Attributes
    ----------
        name: The key name of the feature used in the context.
        kind: The kind of feature (e.g., NODE, EDGE).
        level: The hierarchical level of the feature (e.g., ATOM, RESIDUE).
        dtype: The expected data type of the feature's values.
        description: An optional description of the feature.
        on_missing: A dictionary defining how to handle missing values,
                    e.g., `{'?': 0.0}` maps '?' to a float of 0.0.
    """

    name: str
    kind: FeatureKind
    level: FeatureLevel
    dtype: Any
    description: str = ""
    # metadata of missing data
    on_missing: dict[str, Any] = field(
        default_factory=dict
    )  # <--- e.g., {'token': '?', 'fill_value': 0}

    def matches(self, other: object) -> bool:
        """
        Check if this spec's attributes match another spec's attributes.

        A match occurs if for every non-empty attribute in this spec, the
        corresponding attribute in the `other` spec is identical.

        Args:
            other: The other FeatureSpec object to compare against.

        Returns
        -------
            True if all specified fields match, False otherwise.
        """
        if not isinstance(other, FeatureSpec):
            return NotImplemented

        for f in fields(self):
            filter_value = getattr(self, f.name)
            target_value = getattr(other, f.name)

            # If the filter value is not None, it must match
            if filter_value and target_value:
                target_value = getattr(other, f.name)
                if filter_value != target_value:
                    return False  # Mismatch found
        return True  # All specified fields match


@dataclass
class MappingSpec:
    """
    Defines a single stage in the parsing pipeline.

    This class specifies which mapper to use, what input fields it needs, and
    what features (defined by FeatureSpecs) it will produce.

    Attributes
    ----------
        name: The name of the mapping stage (e.g., "parse_atoms").
        mapper: The registered name of the mapper function to use.
        outputs: A list of FeatureSpec objects describing the output features.
        inputs: A dictionary specifying input "fields" and/or "context" keys.
    """

    name: str

    instruction: str

    # output
    outputs: Sequence[FeatureSpec]
    # input
    inputs: dict[Literal["fields", "context"], list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        After initialization, automatically expand the outputs list.

        to include FeatureSpecs for any masks that need to be generated.
        """
        expanded_outputs = []
        for spec in self.outputs:
            expanded_outputs.append(spec)
            # If on_missing is defined, a mask will be generated.
            # So, we create and add its FeatureSpec explicitly.
            if spec.on_missing:
                mask_name = f"{spec.name}_mask"
                mask_spec = FeatureSpec(
                    name=mask_name,
                    kind=spec.kind,
                    level=spec.level,
                    dtype=bool,
                    description=f"masked_{spec.description}",
                )
                expanded_outputs.append(mask_spec)

        # Replace the original outputs with the expanded list
        self.outputs = tuple(expanded_outputs)
