from __future__ import annotations
import enum
from typing import TYPE_CHECKING, List

import inspect

from biomol.io.registry import MapperRegistry
from biomol.io.context import ParsingContext

if TYPE_CHECKING:
    from biomol.io.schema import MappingSpec

"""
brief design
    user should register field to feature level, feature type(node/edge), feature metadata field
    baseparser will parse field to feature with mapping callable
    user can design new callable for new mapping type
    basically, I will provide 3 callable (allocate, gather, scatter)
"""


class Parser:
    def __init__(self, pipeline: List[MappingSpec]):
        self.pipeline = pipeline

    def parse(self, record: dict) -> ParsingContext:
        context = ParsingContext()

        for spec in self.pipeline:
            mapper = MapperRegistry.get(spec.mapper)
            kwargs = {
                "record": {f: record.get(f) for f in spec.inputs.get("fields", [])},
                "specs": spec.outputs,
            }
            mapper_params = inspect.signature(mapper).parameters
            if "context" in mapper_params:
                kwargs["context"] = context
                kwargs["context_key"] = spec.inputs.get("context", [])

            new_features = mapper(**kwargs)

            if not isinstance(new_features, list):
                new_features = [new_features]

            for i, feature in enumerate(new_features):
                context.add_feature(spec.outputs[i], feature)

        return context
