"""Strict typed contract for exact native special-movement telemetry.

The producer observes one deliberately narrow native edge: the ordinary
type-0 attack-component dash executor at its caller-filtered return address.
This module preserves that execution fact without promoting it to a visual
airborne/traversal phase or to a generic effect stage.
"""

from __future__ import annotations

from dataclasses import dataclass

from .runtime_schema import RuntimeSchema, same_entity_identity, wire_fields

SPECIAL_MOVEMENT_RUNTIME_SCHEMA = "native-special-movement-runtime.v1"
SPECIAL_MOVEMENT_RUNTIME_CAPACITY = 1024
SPECIAL_MOVEMENT_EXECUTE_HOOK_OFFSET = 0xF609DC
SPECIAL_MOVEMENT_EXECUTE_CALLER_OFFSET = 0xF61BF0
SPECIAL_MOVEMENT_OPERATION_FLAG = 1
SPECIAL_MOVEMENT_CAPABILITY_SOURCE = "caller-filtered-native-special-movement-hook-ring"
SPECIAL_MOVEMENT_CAPABILITY_VALIDATION = "sha-build-id-prologue-abi-type0-owner-target-identity"
_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1
_UINT32_MAX = (1 << 32) - 1


class SpecialMovementRuntimeError(ValueError):
    """Raised when special-movement telemetry violates its exact wire contract."""


@dataclass(frozen=True, slots=True)
class SpecialMovementEntityFactV1:
    """One fully validated native combat-entity identity and position."""

    native_object_id: int
    entity_key: tuple[int, int, int]
    owner: int
    object_index: int
    secondary_index: int
    card_id: int
    object_kind: int
    position: tuple[int, int]
    visibility_validated: bool
    invisible_count: int | None


@dataclass(frozen=True, slots=True)
class SpecialMovementCapabilityV1:
    status: str
    source: str
    validation: str
    confidence: str
    fail_closed: bool


@dataclass(frozen=True, slots=True)
class SpecialMovementEventV1:
    sequence: int
    tick: int
    kind: str
    mode: str
    stage: str
    hook_offset: int
    caller_offset: int
    source_before: SpecialMovementEntityFactV1
    source_after: SpecialMovementEntityFactV1
    target: SpecialMovementEntityFactV1
    requested_x: int
    requested_y: int
    target_radius: int
    operation_flag: int
    cooldown_before_ms: int
    cooldown_after_ms: int
    position_changed: bool
    complete_context: bool


@dataclass(frozen=True, slots=True)
class SpecialMovementRuntimeEnvelopeV1:
    generation: int
    state_epoch: int
    observation_tick: int
    capacity: int
    hook_set_attested: bool
    hook_set_installed: bool
    capability: SpecialMovementCapabilityV1
    epoch_first_sequence: int
    oldest_retained_sequence: int
    next_sequence: int
    overflow_count: int
    rejected_count: int
    sequence_gap_before_oldest: bool
    complete: bool
    events: tuple[SpecialMovementEventV1, ...]

    @property
    def capability_status(self) -> str:
        return self.capability.status


_schema = RuntimeSchema(
    error=SpecialMovementRuntimeError,
    label="specialMovementRuntime",
    version=SPECIAL_MOVEMENT_RUNTIME_SCHEMA,
    capacity=SPECIAL_MOVEMENT_RUNTIME_CAPACITY,
    capability_source=SPECIAL_MOVEMENT_CAPABILITY_SOURCE,
    capability_validation=SPECIAL_MOVEMENT_CAPABILITY_VALIDATION,
    capability_type=SpecialMovementCapabilityV1,
    entity_type=SpecialMovementEntityFactV1,
    envelope_type=SpecialMovementRuntimeEnvelopeV1,
    owner_bounds=(_INT32_MIN, _INT32_MAX),
    object_kind_min=-1,
)
_entity_fact = _schema.entity_fact
_same_entity_identity = same_entity_identity


def _event(
    value: object, *, index: int, expected_sequence: int, observation_tick: int, previous_tick: int | None
) -> SpecialMovementEventV1:
    label = f"specialMovementRuntime.events[{index}]"
    record = _schema.record(value, label, wire_fields(SpecialMovementEventV1), "v1 event")
    raw = record.raw
    sequence, tick = record.event_header(expected_sequence, observation_tick, previous_tick)
    if raw.get("kind") != "ordinary_dash_execute":
        raise SpecialMovementRuntimeError(f"{label}.kind is invalid")
    if raw.get("mode") != "dash":
        raise SpecialMovementRuntimeError(f"{label}.mode is invalid")
    if raw.get("stage") != "execute":
        raise SpecialMovementRuntimeError(f"{label}.stage is invalid")
    hook_offset = record.integer("hookOffset", 1, _UINT32_MAX)
    caller_offset = record.integer("callerOffset", 1, _UINT32_MAX)
    if hook_offset != SPECIAL_MOVEMENT_EXECUTE_HOOK_OFFSET:
        raise SpecialMovementRuntimeError(f"{label}.hookOffset is invalid")
    if caller_offset != SPECIAL_MOVEMENT_EXECUTE_CALLER_OFFSET:
        raise SpecialMovementRuntimeError(f"{label}.callerOffset is invalid")

    source_before = _entity_fact(raw.get("sourceBefore"), f"{label}.sourceBefore")
    source_after = _entity_fact(raw.get("sourceAfter"), f"{label}.sourceAfter")
    target = _entity_fact(raw.get("target"), f"{label}.target")
    if not _same_entity_identity(source_before, source_after):
        raise SpecialMovementRuntimeError(f"{label} source identity changed across the native call")

    requested_x = record.integer("requestedX", _INT32_MIN, _INT32_MAX)
    requested_y = record.integer("requestedY", _INT32_MIN, _INT32_MAX)
    target_radius = record.integer("targetRadius", 0, _INT32_MAX)
    operation_flag = record.integer("operationFlag", _INT32_MIN, _INT32_MAX)
    if operation_flag != SPECIAL_MOVEMENT_OPERATION_FLAG:
        raise SpecialMovementRuntimeError(f"{label}.operationFlag is invalid")
    cooldown_before_ms = record.integer("cooldownBeforeMs", 0, _INT32_MAX)
    cooldown_after_ms = record.integer("cooldownAfterMs", 0, _INT32_MAX)
    position_changed = record.boolean("positionChanged")
    if position_changed != (source_before.position != source_after.position):
        raise SpecialMovementRuntimeError(f"{label}.positionChanged contradicts source coordinates")
    if record.boolean("completeContext") is not True:
        raise SpecialMovementRuntimeError(f"{label} lacks complete native context")

    return SpecialMovementEventV1(
        sequence=sequence,
        tick=tick,
        kind="ordinary_dash_execute",
        mode="dash",
        stage="execute",
        hook_offset=hook_offset,
        caller_offset=caller_offset,
        source_before=source_before,
        source_after=source_after,
        target=target,
        requested_x=requested_x,
        requested_y=requested_y,
        target_radius=target_radius,
        operation_flag=operation_flag,
        cooldown_before_ms=cooldown_before_ms,
        cooldown_after_ms=cooldown_after_ms,
        position_changed=position_changed,
        complete_context=True,
    )


def parse_special_movement_runtime(
    value: object, *, generation: int, state_epoch: int, observation_tick: int
) -> SpecialMovementRuntimeEnvelopeV1:
    """Validate one same-observation native movement ring envelope."""
    return _schema.envelope(value, generation, state_epoch, observation_tick, _event)
