"""Exact inheritance helpers shared by static gameplay catalog builders.

Native logic overlays use ordinary replacement semantics plus two explicit
numeric operators in the current competitive ruleset: ``("%", value)`` and
``("+", value)``. Keeping this resolver in one module prevents CardSpec,
EntityArchetype, and MechanicProfile builders from interpreting the same
derived record differently.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

from native_runner.contracts import ContractError


_RELATIVE_OPERATORS = frozenset({"%", "+"})


def _relative_numeric_overlay(base: Any, overlay: Any) -> Any:
    if not (
        isinstance(overlay, Sequence)
        and not isinstance(overlay, (str, bytes, bytearray))
        and len(overlay) == 2
        and isinstance(overlay[0], str)
        and isinstance(overlay[1], (int, float))
        and not isinstance(overlay[1], bool)
    ):
        return overlay
    operator = overlay[0]
    # Two-item tuples also encode ordinary predicates such as ``("=", 3)``.
    # Only the two operators proved to be inheritance overlays are evaluated;
    # all other tuples remain exact opaque data for the action compiler.
    if operator not in _RELATIVE_OPERATORS:
        return overlay
    if isinstance(base, bool) or not isinstance(base, (int, float)):
        raise ContractError(f"native relative overlay {operator!r} requires a numeric base, got {base!r}")
    if not math.isfinite(float(base)) or not math.isfinite(float(overlay[1])):
        raise ContractError("native relative overlay operands must be finite")
    if operator == "+":
        result = float(base) + float(overlay[1])
    else:
        result = float(base) * float(overlay[1]) / 100.0
    if isinstance(base, int) and not isinstance(base, bool):
        return math.trunc(result)
    return result


def merge_effective_record(inherited: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Apply one native derived-record overlay to an inherited record.

    Empty strings, zeroes, ``False``, and empty collections are intentional
    native clears and therefore replace inherited values. Lists are replaced
    as complete values except for the two explicit numeric overlay operators.
    """

    result = dict(inherited)
    for key, value in overlay.items():
        inherited_value = result.get(key)
        if isinstance(value, Mapping) and isinstance(inherited_value, Mapping):
            result[key] = merge_effective_record(inherited_value, value)
        else:
            result[key] = _relative_numeric_overlay(inherited_value, value)
    return result
