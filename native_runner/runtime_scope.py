"""Compact competitive card scope and its ruleset binding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any

from .contracts import content_hash


NORMAL_MODE_PUBLICATION_SCOPE = "normal-mode-only"
NORMAL_MODE_CARD_SCOPE_VERSION = "normal-mode-card-scope.v1"
NORMAL_MODE_CASE_SCOPE_SHA256 = "7fdb0167bb462dc8bed3e3a72631bcbaeb61ecd267c0e8701c24e0bb5aa5f39b"
NORMAL_MODE_RUNTIME_POLICY_SCOPE_VERSION = "normal-mode-runtime-policy-scope.v2"
NORMAL_MODE_SOURCE_CARD_COUNT = 152
NORMAL_MODE_ELIGIBLE_CARD_COUNT = 122
NORMAL_MODE_INVENTORY_ROLE = "topology-capture-only"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class RuntimeScopeError(ValueError):
    """A runtime publication scope is malformed or not canonical."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeScopeError(f"{label} must be an object")
    return value


def _exact_fields(value: Any, *, expected: set[str], label: str) -> Mapping[str, Any]:
    result = _mapping(value, label)
    if set(result) != expected or any(not isinstance(key, str) for key in result):
        raise RuntimeScopeError(f"{label} fields are not exact")
    return result


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RuntimeScopeError(f"{label} must be an array")
    return value


def _int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RuntimeScopeError(f"{label} must be an integer >= {minimum}")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise RuntimeScopeError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _sorted_unique_positive_ints(value: Any, label: str) -> tuple[int, ...]:
    items = tuple(_sequence(value, label))
    if (
        any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in items)
        or tuple(sorted(items)) != items
        or len(set(items)) != len(items)
    ):
        raise RuntimeScopeError(f"{label} must be sorted unique positive integers")
    return items


def build_normal_mode_runtime_policy_scope(
    *, normal_mode_scope_version: str, eligible_card_ids: Sequence[int], case_scope_id: str
) -> dict[str, Any]:
    """Build the one publication scope shared by ruleset and evidence."""

    if normal_mode_scope_version != NORMAL_MODE_CARD_SCOPE_VERSION:
        raise RuntimeScopeError("normal-mode card scope version drifted")
    eligible = _sorted_unique_positive_ints(eligible_card_ids, "normal-mode eligible card IDs")
    if len(eligible) != NORMAL_MODE_ELIGIBLE_CARD_COUNT:
        raise RuntimeScopeError(f"normal-mode eligible card count must be {NORMAL_MODE_ELIGIBLE_CARD_COUNT}")
    if case_scope_id != NORMAL_MODE_CASE_SCOPE_SHA256:
        raise RuntimeScopeError("card support compatibility identity mismatch")
    _validate_eligible_partition(eligible)
    payload: dict[str, Any] = {
        "version": NORMAL_MODE_RUNTIME_POLICY_SCOPE_VERSION,
        "publication_scope": NORMAL_MODE_PUBLICATION_SCOPE,
        "normal_mode_scope_version": normal_mode_scope_version,
        "source_card_count": NORMAL_MODE_SOURCE_CARD_COUNT,
        "eligible_card_count": len(eligible),
        "eligible_card_ids": list(eligible),
        "eligible_card_ids_sha256": content_hash(list(eligible)),
        "inventory_role": NORMAL_MODE_INVENTORY_ROLE,
        "case_scope_id": case_scope_id,
    }
    payload["scope_id"] = content_hash(payload)
    return payload


def validate_normal_mode_runtime_policy_scope(value: Any) -> Mapping[str, Any]:
    """Validate a persisted normal-mode publication scope without defaults."""

    raw = _exact_fields(
        value,
        expected={
            "version",
            "publication_scope",
            "normal_mode_scope_version",
            "source_card_count",
            "eligible_card_count",
            "eligible_card_ids",
            "eligible_card_ids_sha256",
            "inventory_role",
            "case_scope_id",
            "scope_id",
        },
        label="normal-mode runtime policy scope",
    )
    if (
        raw["version"] != NORMAL_MODE_RUNTIME_POLICY_SCOPE_VERSION
        or raw["publication_scope"] != NORMAL_MODE_PUBLICATION_SCOPE
        or raw["normal_mode_scope_version"] != NORMAL_MODE_CARD_SCOPE_VERSION
        or raw["inventory_role"] != NORMAL_MODE_INVENTORY_ROLE
        or _int(raw["source_card_count"], "normal-mode source card count") != NORMAL_MODE_SOURCE_CARD_COUNT
        or _int(raw["eligible_card_count"], "normal-mode eligible card count") != NORMAL_MODE_ELIGIBLE_CARD_COUNT
    ):
        raise RuntimeScopeError("normal-mode runtime policy constants drifted")
    eligible = _sorted_unique_positive_ints(raw["eligible_card_ids"], "normal-mode eligible card IDs")
    if len(eligible) != NORMAL_MODE_ELIGIBLE_CARD_COUNT:
        raise RuntimeScopeError("normal-mode eligible card IDs are incomplete")
    if _sha256(raw["eligible_card_ids_sha256"], "normal-mode eligible card IDs SHA-256") != content_hash(
        list(eligible)
    ):
        raise RuntimeScopeError("normal-mode eligible card ID hash drifted")
    if raw["case_scope_id"] != NORMAL_MODE_CASE_SCOPE_SHA256:
        raise RuntimeScopeError("card support compatibility identity mismatch")
    _validate_eligible_partition(eligible)
    if _sha256(raw["scope_id"], "normal-mode runtime policy scope ID") != (
        content_hash({key: item for key, item in raw.items() if key != "scope_id"})
    ):
        raise RuntimeScopeError("normal-mode runtime policy scope ID integrity failure")
    return raw


def _validate_eligible_partition(eligible_card_ids: Sequence[int]) -> None:
    # Lazy import avoids a cycle while semantic contracts are initialized.
    from .semantic_subset import NORMAL_MODE_POLICY_CARD_IDS
    if frozenset(eligible_card_ids) != NORMAL_MODE_POLICY_CARD_IDS:
        raise RuntimeScopeError("card support does not match the eligible partition")
