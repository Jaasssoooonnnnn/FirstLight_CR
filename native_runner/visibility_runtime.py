"""Exact-build invisibility transitions and contextual target gates.

This module deliberately does not manufacture owner-relative visibility or a
timeless ``targetable`` bit.  It promotes only:

* the exact type-3 ``Invisible`` count boundary crossed by one native Buff
  apply/remove handler; and
* the actual invisibility gate reached by one of the two proven ordinary
  attack-acquisition loops.

Both resolvers fail closed when any pointer/identity/call-site attestation is
missing.  The shared native probe now exports the transition producer; the
contextual acquisition gate remains unavailable until it has a PC-relative-
safe native bridge.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from types import MappingProxyType
from typing import Mapping


VISIBILITY_RUNTIME_SCHEMA = "native-visibility-runtime.v1"
EXACT_LIBG_SHA256 = "110aa2b5cac391c498645e072b0d88729428c2c2845e7e2737ca8ee979059783"
EXACT_LIBG_BUILD_ID = "90e6f351f0dec4a28b494c3812d2629fd45d5a83"

ROYAL_GHOST_CARD_ID = 26_000_050
BUFF_COMPONENT_TYPE = 3
BUFF_COMPONENT_VTABLE_OFFSET = 0x018A1D58
BUFF_ASSET_VTABLE_OFFSET = 0x01882500
ATTACK_COMPONENT_TYPE = 0
ATTACK_COMPONENT_VTABLE_OFFSET = 0x018A1DD0

BUFF_APPLY_OFFSET = 0x00F5A2D4
BUFF_REMOVE_OFFSET = 0x00F5A578
PRIMARY_SELECTOR_OFFSET = 0x00F5D16C
PRIMARY_INVISIBILITY_GATE_OFFSET = 0x00F5D314
SECONDARY_SELECTOR_OFFSET = 0x00F5D7E0
SECONDARY_INVISIBILITY_GATE_OFFSET = 0x00F5DA94

EntityKey = tuple[int, int, int]


class VisibilityRuntimeError(ValueError):
    """Raised for malformed raw facts, not merely for missing proof."""


class ClosureStatus(str, Enum):
    EXACT_PATH_READY = "exact_path_ready"
    CONTEXTUAL_ONLY = "contextual_only"
    BLOCKED_NATIVE_PRODUCER = "blocked_native_producer"


class VisibilityPhase(str, Enum):
    VISIBLE = "visible"
    INVISIBLE = "invisible"


class VisibilityTransitionKind(str, Enum):
    BECAME_INVISIBLE = "became_invisible"
    BECAME_VISIBLE = "became_visible"


class OrdinaryTargetGateOutcome(str, Enum):
    PASSED_INVISIBILITY_GATE = "passed_invisibility_gate"
    REJECTED_INVISIBLE = "rejected_invisible"


def _nonnegative(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise VisibilityRuntimeError(f"{name} must be a non-negative integer")
    return value


def _optional_positive(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise VisibilityRuntimeError(f"{name} must be a positive integer")
    return value


def _owner(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        raise VisibilityRuntimeError(f"{name} must be owner 0 or 1")
    return value


def _entity_key(value: EntityKey, owner: int, name: str) -> EntityKey:
    if (
        not isinstance(value, tuple)
        or len(value) != 3
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value)
    ):
        raise VisibilityRuntimeError(f"{name} must contain three integers")
    if value[0] != owner:
        raise VisibilityRuntimeError(f"{name} owner does not match")
    return value


def _validate_raw(record, subjects: tuple[str, ...], *, positive_ids=False) -> None:
    """Validate each declared raw field once, keeping identity checks first."""
    _nonnegative(record.tick, "tick")
    owners = {subject: _owner(getattr(record, subject + "_owner"), subject + "_owner") for subject in subjects}
    for subject, owner in owners.items():
        name = subject + "_key"
        object.__setattr__(record, name, _entity_key(getattr(record, name), owner, name))
    identity_fields = {subject + "_owner" for subject in subjects} | {"tick"}
    for item in fields(record):
        value = getattr(record, item.name)
        if item.type == "int" and item.name not in identity_fields:
            _nonnegative(value, item.name)
        elif item.type == "int | None":
            if positive_ids:
                _optional_positive(value, item.name)
            elif value is not None:
                _nonnegative(value, item.name)
    if positive_ids and record.buff_global_id is not None and record.buff_global_id > 0xFFFFFFFF:
        raise VisibilityRuntimeError("buff_global_id exceeds uint32")
    for item in fields(record):
        if item.type == "bool" and not isinstance(getattr(record, item.name), bool):
            raise VisibilityRuntimeError(f"{item.name} must be boolean")


def _event_from_raw(event_type, raw, **derived):
    values = {
        item.name: getattr(raw, item.name)
        for item in fields(event_type)
        if item.name not in derived and hasattr(raw, item.name)
    }
    return event_type(**values, **derived)


def phase_from_invisible_count(invisible_count: int) -> VisibilityPhase:
    _nonnegative(invisible_count, "invisible_count")
    return VisibilityPhase.INVISIBLE if invisible_count > 0 else VisibilityPhase.VISIBLE


@dataclass(frozen=True, slots=True)
class VisibilityTransitionRaw:
    tick: int
    subject_key: EntityKey
    subject_owner: int
    hook_offset: int
    invisible_count_before: int
    invisible_count_after: int
    active_entry_count: int
    hook_attested: bool
    subject_validated: bool
    component_validated: bool
    component_type: int
    component_vtable_offset: int
    component_owner_matches_subject: bool
    entry_validated: bool
    entry_component_matches: bool
    buff_asset_validated: bool
    buff_asset_vtable_offset: int
    buff_invisible: bool
    card_id: int | None = None
    buff_global_id: int | None = None

    def __post_init__(self) -> None:
        _validate_raw(self, ("subject",), positive_ids=True)


@dataclass(frozen=True, slots=True)
class VisibilityTransitionEvent:
    tick: int
    subject_key: EntityKey
    subject_owner: int
    phase: VisibilityPhase
    transition: VisibilityTransitionKind
    invisible_count_before: int
    invisible_count_after: int
    hook_offset: int
    card_id: int | None
    buff_global_id: int | None
    scope: str = "native_invisibility_phase"
    evidence: str = "native_authoritative"
    complete: bool = True
    version: str = VISIBILITY_RUNTIME_SCHEMA

    @property
    def is_royal_ghost(self) -> bool:
        return self.card_id == ROYAL_GHOST_CARD_ID


def _exact_transition_identity(raw: VisibilityTransitionRaw) -> bool:
    return (
        raw.hook_attested
        and raw.subject_validated
        and raw.component_validated
        and raw.component_type == BUFF_COMPONENT_TYPE
        and raw.component_vtable_offset == BUFF_COMPONENT_VTABLE_OFFSET
        and raw.component_owner_matches_subject
        and raw.entry_validated
        and raw.entry_component_matches
        and raw.buff_asset_validated
        and raw.buff_asset_vtable_offset == BUFF_ASSET_VTABLE_OFFSET
    )


def resolve_visibility_transition(raw: VisibilityTransitionRaw) -> VisibilityTransitionEvent | None:
    """Promote only a 0 <-> positive boundary from the exact native handler.

    An overlapping invisible Buff changes 1 -> 2 or 2 -> 1 but does not change
    the subject's visibility phase and therefore emits no transition.
    """

    if not _exact_transition_identity(raw):
        return None
    before = raw.invisible_count_before
    after = raw.invisible_count_after

    if raw.hook_offset not in (BUFF_APPLY_OFFSET, BUFF_REMOVE_OFFSET):
        return None
    if not raw.buff_invisible:
        return None
    delta = 1 if raw.hook_offset == BUFF_APPLY_OFFSET else -1
    if after != before + delta or after > raw.active_entry_count:
        return None
    if delta == -1 and (before < 1 or before > raw.active_entry_count + 1):
        return None

    if before == 0 and after > 0:
        transition = VisibilityTransitionKind.BECAME_INVISIBLE
    elif before > 0 and after == 0:
        transition = VisibilityTransitionKind.BECAME_VISIBLE
    else:
        return None
    return _event_from_raw(
        VisibilityTransitionEvent, raw, phase=phase_from_invisible_count(after), transition=transition
    )


_ORDINARY_GATE_PATHS: frozenset[tuple[int, int]] = frozenset(
    {
        (PRIMARY_SELECTOR_OFFSET, PRIMARY_INVISIBILITY_GATE_OFFSET),
        (SECONDARY_SELECTOR_OFFSET, SECONDARY_INVISIBILITY_GATE_OFFSET),
    }
)


@dataclass(frozen=True, slots=True)
class OrdinaryTargetEligibilityRaw:
    tick: int
    attacker_key: EntityKey
    candidate_key: EntityKey
    attacker_owner: int
    candidate_owner: int
    selector_offset: int
    invisibility_gate_offset: int
    option: int
    gate_hook_attested: bool
    attacker_validated: bool
    candidate_validated: bool
    attack_component_validated: bool
    attack_component_type: int
    attack_component_vtable_offset: int
    component_owner_matches_attacker: bool
    candidate_filter_passed: bool
    pre_invisibility_filters_passed: bool
    reached_invisibility_gate: bool
    candidate_component_lookup_validated: bool
    candidate_buff_component_present: bool
    candidate_buff_component_vtable_offset: int | None
    invisible_count: int | None

    def __post_init__(self) -> None:
        _validate_raw(self, ("attacker", "candidate"))


@dataclass(frozen=True, slots=True)
class OrdinaryTargetEligibilityEvent:
    tick: int
    attacker_key: EntityKey
    candidate_key: EntityKey
    attacker_owner: int
    candidate_owner: int
    selector_offset: int
    invisibility_gate_offset: int
    invisible_count: int
    outcome: OrdinaryTargetGateOutcome
    scope: str = "ordinary_attack_acquisition.invisibility_gate"
    evidence: str = "native_authoritative"
    gate_decision_complete: bool = True
    final_selection_complete: bool = False
    targetable_by_owner: None = None
    version: str = VISIBILITY_RUNTIME_SCHEMA


def resolve_ordinary_target_eligibility(raw: OrdinaryTargetEligibilityRaw) -> OrdinaryTargetEligibilityEvent | None:
    """Resolve only the reached native invisibility gate, not final selection."""

    if not (
        raw.gate_hook_attested
        and raw.attacker_validated
        and raw.candidate_validated
        and raw.attack_component_validated
        and raw.attack_component_type == ATTACK_COMPONENT_TYPE
        and raw.attack_component_vtable_offset == ATTACK_COMPONENT_VTABLE_OFFSET
        and raw.component_owner_matches_attacker
        and raw.option == 1
        and raw.candidate_filter_passed
        and raw.pre_invisibility_filters_passed
        and raw.reached_invisibility_gate
        and raw.candidate_component_lookup_validated
        and (raw.selector_offset, raw.invisibility_gate_offset) in _ORDINARY_GATE_PATHS
    ):
        return None

    if raw.candidate_buff_component_present:
        if raw.candidate_buff_component_vtable_offset != BUFF_COMPONENT_VTABLE_OFFSET or raw.invisible_count is None:
            return None
        invisible_count = raw.invisible_count
    else:
        if raw.candidate_buff_component_vtable_offset is not None or raw.invisible_count is not None:
            return None
        invisible_count = 0

    outcome = (
        OrdinaryTargetGateOutcome.REJECTED_INVISIBLE
        if invisible_count > 0
        else OrdinaryTargetGateOutcome.PASSED_INVISIBILITY_GATE
    )
    return _event_from_raw(OrdinaryTargetEligibilityEvent, raw, invisible_count=invisible_count, outcome=outcome)


def resolve_public_by_owner_from_snapshot(*, observer_owner: int, subject_owner: int, invisible_count: int) -> None:
    """There is no proven owner-conditioned public/render producer."""

    del observer_owner, subject_owner, invisible_count
    return None


def resolve_targetable_by_owner_from_snapshot(*, observer_owner: int, subject_owner: int, invisible_count: int) -> None:
    """Target eligibility is call-contextual and cannot be snapshot-inferred."""

    del observer_owner, subject_owner, invisible_count
    return None


def closure_status() -> Mapping[str, ClosureStatus]:
    return MappingProxyType(
        {
            "entity.invisibility_phase_transition": ClosureStatus.EXACT_PATH_READY,
            "target_eligibility.ordinary_attack_invisibility_gate": ClosureStatus.CONTEXTUAL_ONLY,
            "target_eligibility.area_damage": ClosureStatus.CONTEXTUAL_ONLY,
            "entity.public_by_owner": ClosureStatus.BLOCKED_NATIVE_PRODUCER,
            "entity.targetable_by_owner": ClosureStatus.BLOCKED_NATIVE_PRODUCER,
        }
    )
