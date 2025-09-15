from typing import Any

from biomol.io.schema import FeatureKind, FeatureLevel, MappingSpec, FeatureSpec


class Blueprint:
    """
    Complile user's parsing recipe to parser-friendly format.

    Usecase example
    ---------------

    compiler.stage("parse_chem_comp_properties") \
    .using("identity") \
    .from_fields("_chem_comp.id", "_chem_comp.name", "_chem_comp.formula") \
    .to(FeatureKind.NODE, FeatureLevel.STRUCTURE, "id(str)", "name(str)", "formula(str)")
    """

    def __init__(self) -> None:
        self._pipeline: list[MappingSpec] = []
        self._current_spec: dict[str, Any] = {}

    def _commit_previous_stage(self) -> None:
        """Add previous stage to the pipeline if exists."""
        if self._current_spec:
            self._pipeline.append(MappingSpec(**self._current_spec))
        self._current_spec = {}

    def stage(self, name: str) -> "Blueprint":
        """Define a new stage in the pipeline."""
        self._commit_previous_stage()
        self._current_spec["name"] = name
        return self

    def using(self, instruction: str) -> "Blueprint":
        """Define the instruction function to use in the current stage."""
        self._current_spec["instruction"] = instruction
        return self

    def from_fields(self, *fields: str) -> "Blueprint":
        """Define input fields for the current stage."""
        self._current_spec.setdefault("inputs", {})["fields"] = list(fields)
        return self

    def with_context(self, *context_keys: str) -> "Blueprint":
        """Define context keys for the current stage."""
        self._current_spec.setdefault("inputs", {})["context"] = list(context_keys)
        return self

    def _create_specs(
        self, kind: FeatureKind, level: FeatureLevel, **specs
    ) -> "Blueprint":
        """Private function create FeatureSpec list."""
        outputs = []
        for name, spec_info in specs.items():
            dtype, on_missing = (
                (spec_info, {}) if not isinstance(spec_info, tuple) else spec_info
            )
            spec = FeatureSpec(
                name=name, kind=kind, level=level, dtype=dtype, on_missing=on_missing
            )
            outputs.append(spec)
        self._current_spec["outputs"] = outputs
        return self

    # ---- Shortcuts for common feature specs ----
    def to_structure_nodes(self, **specs) -> "Blueprint":
        return self._create_specs(FeatureKind.NODE, FeatureLevel.STRUCTURE, **specs)

    def to_chain_nodes(self, **specs) -> "Blueprint":
        return self._create_specs(FeatureKind.NODE, FeatureLevel.CHAIN, **specs)

    def to_residue_nodes(self, **specs) -> "Blueprint":
        return self._create_specs(FeatureKind.NODE, FeatureLevel.RESIDUE, **specs)

    def to_atom_nodes(self, **specs) -> "Blueprint":
        return self._create_specs(FeatureKind.NODE, FeatureLevel.ATOM, **specs)

    def to_chain_edges(self, **specs) -> "Blueprint":
        return self._create_specs(FeatureKind.EDGE, FeatureLevel.CHAIN, **specs)

    def to_residue_edges(self, **specs) -> "Blueprint":
        return self._create_specs(FeatureKind.EDGE, FeatureLevel.RESIDUE, **specs)

    def to_atom_edges(self, **specs) -> "Blueprint":
        return self._create_specs(FeatureKind.EDGE, FeatureLevel.ATOM, **specs)

    def to_chain_aux(self, **specs) -> "Blueprint":
        return self._create_specs(FeatureKind.AUX, FeatureLevel.CHAIN, **specs)

    def to_resiude_aux(self, **specs) -> "Blueprint":
        return self._create_specs(FeatureKind.AUX, FeatureLevel.RESIDUE, **specs)

    def to_atom_aux(self, **specs) -> "Blueprint":
        return self._create_specs(FeatureKind.AUX, FeatureLevel.ATOM, **specs)

    def build(self) -> list[MappingSpec]:
        self._commit_previous_stage()
        return self._pipeline
