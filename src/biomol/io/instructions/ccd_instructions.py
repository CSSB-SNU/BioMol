"""
Instruction Factories for BioMol Recipe Processing.

This module provides a set of "instruction factories" used by the RecipeBook and
Cooker systems. The functions defined here follow a functional programming paradigm,
specifically the "function factory" and "closure" patterns.

--------------------------------------------------------------------------------
How to Define a Custom Instruction
--------------------------------------------------------------------------------

Instead of defining a single function that takes all possible arguments (including
type information like `dtype`), you define a **higher-order function** (a factory)
that takes configuration parameters (like `dtype`) and returns a new, specialized
"worker" function.

### 📝 Template for a New Instruction Factory

Here is a simple, commented template for creating a new instruction factory.

# --- 1. Import necessary types ---
from typing import Callable, type, TypeVar
import numpy as np
from biomol.core.feature import NodeFeature # or EdgeFeature

# --- 2. Define your TypeVars if needed ---
NumericType = TypeVar("NumericType", bound=np.generic)

# --- 3. Define the Factory Function ---
# The factory takes configuration (like dtype) and returns a callable.
def your_new_instruction(
    *, dtype: type[NumericType], some_config_value: float
) -> Callable[..., NodeFeature]:
    \"\"\"This is the factory's docstring, explaining what it configures.\"\"\"

    # --- 4. Define the Inner "Worker" Function ---
    # This is the actual instruction that the Cooker will execute.
    # Note that it does NOT take `dtype` or `some_config_value` as arguments.
    # Its signature should match the inputs your recipe will provide.

    def _worker(
        data: list,
        *,
    ) -> NodeFeature:
        \"\"\"This is the worker's docstring, explaining the core logic.\"\"\"

        # --- 5. Implement the Logic ---
        # The worker uses the variables `dtype` and `some_config_value` from
        # the outer scope (this is called a "closure").
        processed_data = [dtype(x * some_config_value) for x in data]
        value = np.array(processed_data, dtype=dtype)

        return NodeFeature(value=value)

    # --- 6. Return the configured worker function ---
    return _worker

# --- 7. Call the factory to get the configured worker function ---
    instruction=your_new_instruction(dtype=np.float32, some_config_value=10.0),

"""

from collections.abc import Callable
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from biomol.core.feature import EdgeFeature, NodeFeature

InputType = TypeVar("InputType", str, int, float)
FeatureType = TypeVar("FeatureType")
NumericType = TypeVar("NumericType", int, float)


def identity_instruction(
    *,
    dtype: type[InputType],
) -> Callable[..., tuple[NodeFeature, NodeFeature] | NodeFeature]:
    """
    Return a configured instruction function that maps fields to node features.

    The returned function 'remembers' the dtype via closure.
    """

    def _worker(
        data: list[InputType] | NDArray,
        on_missing: dict[str, NumericType] | None = None,
    ) -> tuple[NodeFeature, NodeFeature] | NodeFeature:
        if on_missing:
            formatted_data = [
                dtype(datum) if datum not in on_missing else on_missing[datum]
                for datum in data
            ]
            mask = np.array([d not in on_missing for d in data], dtype=bool)
            mask_feature = NodeFeature(value=mask)
        else:
            formatted_data = [dtype(datum) for datum in data]

        value = np.array(formatted_data, dtype=dtype)
        data_feature = NodeFeature(value=value)

        return (data_feature, mask_feature) if on_missing else data_feature

    return _worker


def stack_instruction(
    *,
    dtype: type[NumericType],
) -> Callable[..., tuple[NodeFeature, NodeFeature]]:
    """Return a configured instruction that stacks multiple fields."""

    def _worker(
        *args: list[InputType] | NDArray,
        on_missing: dict[str, NumericType],
    ) -> tuple[NodeFeature, NodeFeature]:
        result_data, mask_data = [], []
        if not args:
            msg = "At least one field must be provided to stack."
            raise ValueError(msg)
        first_len = len(args[0])
        if not all(len(arg) == first_len for arg in args):
            msg = "All fields must have the same length."
            raise ValueError(msg)

        for data in args:
            formatted = [
                dtype(x) if x not in on_missing else on_missing[x] for x in data
            ]
            result_data.append(np.array(formatted, dtype=dtype))
            mask_data.append([x not in on_missing for x in data])

        stacked_value = np.stack(result_data, axis=-1)
        stacked_mask = np.all(np.array(mask_data, dtype=bool).T, axis=-1)

        return (
            NodeFeature(value=stacked_value),
            NodeFeature(value=stacked_mask),
        )

    return _worker


def bond_instruction(*, dtype: type[FeatureType]) -> Callable[..., EdgeFeature]:
    """Return a configured instruction that creates edge features."""

    def _worker(
        *args: list[InputType] | NDArray,
        src: list[InputType] | NDArray,
        dst: list[InputType] | NDArray,
        atom_id: NodeFeature,
    ) -> EdgeFeature:
        order = np.argsort(atom_id.value)
        src_indices = order[np.searchsorted(atom_id.value, np.array(src), sorter=order)]
        dst_indices = order[np.searchsorted(atom_id.value, np.array(dst), sorter=order)]

        values = [np.array([dtype(x) for x in data]) for data in args]
        if not values:
            msg = "At least one feature field (*args) must be provided."
            raise ValueError(msg)

        final_value = np.stack(values, axis=-1) if len(values) > 1 else values[0]

        return EdgeFeature(
            value=final_value,
            src_indices=src_indices.astype(int),
            dst_indices=dst_indices.astype(int),
        )

    return _worker
