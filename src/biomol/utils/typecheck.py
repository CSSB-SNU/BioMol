"""Type checking utilities for the biomol package."""

import os
from collections.abc import Callable

from beartype import beartype
from jaxtyping import jaxtyped

_SHOULD_TYPECHECK = os.environ.get("SHOULD_TYPECHECK", "false").lower() == "true"


def typecheck(cls_or_func: type | Callable) -> type | Callable:
    """Decorate a class or function with type checking.

    If the environment variable `SHOULD_TYPECHECK` is set to "true", the function will
    apply type checking using `beartype` and `jaxtyping`. Otherwise, it will return the
    class or function unchanged. We recommend setting this variable in testing only, as
    type checking can be expensive. See jaxtyping [documentation](https://docs.kidger.site/jaxtyping/api/runtime-type-checking/#jaxtyping.jaxtyped)
    for more details.
    """
    if _SHOULD_TYPECHECK:
        return jaxtyped(typechecker=beartype)(cls_or_func)
    return cls_or_func
