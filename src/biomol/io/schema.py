from dataclasses import dataclass, field, fields, make_dataclass
from enum import Enum
from typing import List, Dict, Any, Literal


class FeatureLevel(Enum):
    """docstring."""

    STRUCTURE = "structure"
    CHAIN = "chain"
    RESIDUE = "residue"
    ATOM = "atom"


class FeatureKind(Enum):
    """docstring."""

    NODE = "node"
    EDGE = "edge"
    AUX = "auxiliary"


@dataclass(frozen=True)
class FeatureSpec:
    """bluprint of feature after parsing and mapping."""

    name: str  # <--- the key name of the feature used in the context
    kind: FeatureKind
    level: FeatureLevel
    dtype: Any
    description: str = ""
    # metadata of missing data
    on_missing: Dict[str, Any] = field(
        default_factory=dict
    )  # <--- e.g., {'token': '?', 'fill_value': 0}

    def matches(self, other: object) -> bool:
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
    """details of one parsing stage."""

    name: str  # <--- name of the mapping stage(e.g., "parse_atoms")

    mapper: str  # <--- name of the mapper name (e.g., "identity", "stack_coords")

    # output
    outputs: List[FeatureSpec]
    # input
    inputs: Dict[Literal["fields", "context"], List[str]] = field(default_factory=dict)

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
        self.outputs = expanded_outputs
        # To make the instance immutable again after modification
        frozen_class = make_dataclass(
            self.__class__.__name__, [f.name for f in fields(self)], frozen=True
        )
        # Swap the class of the current instance to the new frozen class
        self.__class__ = frozen_class
