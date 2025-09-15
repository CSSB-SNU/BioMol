"""
IO module defines the Parser class for processing biological data records.

Design Overview:
- The user defines a pipeline of mapping specifications (MappingSpec).
- Each spec dictates how to map raw data fields to biological features.
- The Parser iterates through this pipeline, using registered "mappers"
  (callable functions) to perform the transformations.
- A ParsingContext object is used to store the results of each step.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

from biomol.io.context import ParsingContext
from biomol.io.isa import InstructionSet

if TYPE_CHECKING:
    from biomol.io.schema import MappingSpec


class Processor:
    """A pipeline-based parser for transforming raw data records into features."""

    def __init__(self, pipeline: list[MappingSpec]) -> None:
        """
        Initialize the Parser with a processing pipeline.

        Args:
            pipeline: A list of MappingSpec objects that define the parsing stages.
        """
        self.pipeline = pipeline

    def parse(self, record: dict) -> ParsingContext:
        """
        Execute the parsing pipeline on a given data record.

        Args:
            record: A dictionary containing the raw data to be parsed.

        Returns
        -------
            A ParsingContext object populated with the extracted features.
        """
        context = ParsingContext()

        for spec in self.pipeline:
            instruction = InstructionSet.get(spec.instruction)
            kwargs = {
                "record": {f: record.get(f) for f in spec.inputs.get("fields", [])},
                "specs": spec.outputs,
            }
            instruction_params = inspect.signature(instruction).parameters
            if "context" in instruction_params:
                kwargs["context"] = context
                kwargs["context_key"] = spec.inputs.get("context", [])

            new_features = instruction(**kwargs)

            if not isinstance(new_features, list):
                new_features = [new_features]

            for i, feature in enumerate(new_features):
                context.add_feature(spec.outputs[i], feature)

        return context
