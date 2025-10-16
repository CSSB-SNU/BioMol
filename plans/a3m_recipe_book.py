from biomol.io.instructions.a3m_instructions import (
    build_container,
    parse_headers,
    parse_sequence,
)
from biomol.io.recipe import RecipeBook

"""Build a a3m-specific Cooker.

This factory function constructs and returns a Cooker preconfigured
with a3m parsing recipes and instructions.
"""

a3m_recipe = RecipeBook()

a3m_recipe.add(
    targets=[
        ("parsed_sequences", str),
    ],
    instruction=parse_sequence(),
    inputs={
        "kwargs": {
            "raw_sequences": ("raw_sequences", str | None),
            "a3m_type": ("a3m_type", str | None),
        },
    },
)

a3m_recipe.add(
    targets=[
        ("parsed_headers", dict),
    ],
    instruction=parse_headers(),
    inputs={
        "kwargs": {
            "headers": ("headers", list[str] | None),
        },
    },
)

a3m_recipe.add(
    targets=[
        ("msa_container", dict),
    ],
    instruction=build_container(),
    inputs={
        "kwargs": {
            "sequences": ("parsed_sequences", dict | None),
            "headers": ("parsed_headers", dict | None),
        },
    },
)

RECIPE = a3m_recipe
TARGETS = ["msa_container"]
