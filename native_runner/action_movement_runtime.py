"""Strict typed adapter for exact native action-owned movement records.

This module validates ``native-action-movement-runtime.v1`` independently of
ordinary dash and Buff/effect telemetry.  Its ``stage`` values name attested
native action boundaries only; they are never an ``EffectState.stage``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .runtime_schema import RuntimeSchema, same_entity_identity, wire_fields

ACTION_MOVEMENT_RUNTIME_SCHEMA = "native-action-movement-runtime.v1"
ACTION_MOVEMENT_RUNTIME_CAPACITY = 1_024
ACTION_MOVEMENT_CAPABILITY_SOURCE = "allowlisted-native-action-movement-hook-ring"
ACTION_MOVEMENT_CAPABILITY_VALIDATION = "sha-build-id-prologue-abi-action-name-global-id-class-vtable-source-identity"

GOLDEN_KNIGHT_HOP_HOOK_OFFSET = 0xF49D24
WARP_POSITION_COMMIT_HOOK_OFFSET = 0xE82A34
DASHING_ATTACK_CHAIN_DATA_VTABLE_OFFSET = 0x188DAF8
DASHING_ATTACK_CHAIN_RUNTIME_VTABLE_OFFSET = 0x189ECA0
WARP_CHARACTER_DATA_VTABLE_OFFSET = 0x1895120

_UINT32_MAX = (1 << 32) - 1
_INT32_MAX = (1 << 31) - 1


class ActionMovementRuntimeError(ValueError):
    """Raised when the native action-movement envelope is incoherent."""


@dataclass(frozen=True, slots=True)
class ActionMovementCapabilityV1:
    status: str
    source: str
    validation: str
    confidence: str
    fail_closed: bool


@dataclass(frozen=True, slots=True)
class ActionMovementEntityFactV1:
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
class ActionMovementEventV1:
    sequence: int
    tick: int
    kind: str
    mode: str
    stage: str
    hook_offset: int
    caller_offset: int
    action_data_global_id: int
    action_name: str
    action_class_vtable_offset: int
    runtime_class_vtable_offset: int | None
    source_before: ActionMovementEntityFactV1
    source_after: ActionMovementEntityFactV1
    target: ActionMovementEntityFactV1 | None
    chain_index: int | None
    position_changed: bool
    complete_context: bool


@dataclass(frozen=True, slots=True)
class ActionMovementRuntimeEnvelopeV1:
    generation: int
    state_epoch: int
    observation_tick: int
    capacity: int
    hook_set_attested: bool
    hook_set_installed: bool
    capability: ActionMovementCapabilityV1
    epoch_first_sequence: int
    oldest_retained_sequence: int
    next_sequence: int
    overflow_count: int
    rejected_count: int
    sequence_gap_before_oldest: bool
    complete: bool
    events: tuple[ActionMovementEventV1, ...]

    @property
    def capability_status(self) -> str:
        return self.capability.status


@dataclass(frozen=True, slots=True)
class _EventContract:
    mode: str
    stage: str
    hook_offset: int
    action_name: str
    action_class_vtable_offset: int
    runtime_class_vtable_offset: int | None
    source_card_ids: frozenset[int]
    target_required: bool
    chain_index_required: bool
    position_change_required: bool


_EVENT_CONTRACTS: Mapping[str, _EventContract] = {
    "golden_knight_chain_hop_launch": _EventContract(
        mode="dash_chain",
        stage="hop_launch",
        hook_offset=GOLDEN_KNIGHT_HOP_HOOK_OFFSET,
        action_name="GoldenKnight_Execute_Charge",
        action_class_vtable_offset=DASHING_ATTACK_CHAIN_DATA_VTABLE_OFFSET,
        runtime_class_vtable_offset=(DASHING_ATTACK_CHAIN_RUNTIME_VTABLE_OFFSET),
        source_card_ids=frozenset({26_000_074}),
        target_required=True,
        chain_index_required=True,
        position_change_required=False,
    ),
    "boss_bandit_warp_position_commit": _EventContract(
        mode="warp",
        stage="position_commit",
        hook_offset=WARP_POSITION_COMMIT_HOOK_OFFSET,
        action_name="BossBandit_ability_warp",
        action_class_vtable_offset=WARP_CHARACTER_DATA_VTABLE_OFFSET,
        runtime_class_vtable_offset=None,
        source_card_ids=frozenset({26_000_103}),
        target_required=False,
        chain_index_required=False,
        position_change_required=True,
    ),
    "elite_archer_warp_position_commit": _EventContract(
        mode="warp",
        stage="position_commit",
        hook_offset=WARP_POSITION_COMMIT_HOOK_OFFSET,
        action_name="EliteArcherHero_Ability_Warp",
        action_class_vtable_offset=WARP_CHARACTER_DATA_VTABLE_OFFSET,
        runtime_class_vtable_offset=None,
        source_card_ids=frozenset({26_000_062, 203_000_062}),
        target_required=False,
        chain_index_required=False,
        position_change_required=True,
    ),
}


_schema = RuntimeSchema(
    error=ActionMovementRuntimeError,
    label="actionMovementRuntime",
    version=ACTION_MOVEMENT_RUNTIME_SCHEMA,
    capacity=ACTION_MOVEMENT_RUNTIME_CAPACITY,
    capability_source=ACTION_MOVEMENT_CAPABILITY_SOURCE,
    capability_validation=ACTION_MOVEMENT_CAPABILITY_VALIDATION,
    capability_type=ActionMovementCapabilityV1,
    entity_type=ActionMovementEntityFactV1,
    envelope_type=ActionMovementRuntimeEnvelopeV1,
)
_entity_fact = _schema.entity_fact
_same_entity_identity = same_entity_identity


def _event(
    value: object, *, index: int, expected_sequence: int, observation_tick: int, previous_tick: int | None
) -> ActionMovementEventV1:
    label = f"actionMovementRuntime.events[{index}]"
    record = _schema.record(value, label, wire_fields(ActionMovementEventV1), "v1 event")
    raw = record.raw
    sequence, tick = record.event_header(expected_sequence, observation_tick, previous_tick)

    kind = raw.get("kind")
    if not isinstance(kind, str) or kind not in _EVENT_CONTRACTS:
        raise ActionMovementRuntimeError(f"{label}.kind is invalid")
    contract = _EVENT_CONTRACTS[kind]
    if raw.get("mode") != contract.mode:
        raise ActionMovementRuntimeError(f"{label}.mode is invalid")
    if raw.get("stage") != contract.stage:
        raise ActionMovementRuntimeError(f"{label}.stage is invalid")
    hook_offset = record.integer("hookOffset", 1, _UINT32_MAX)
    if hook_offset != contract.hook_offset:
        raise ActionMovementRuntimeError(f"{label}.hookOffset is invalid")
    caller_offset = record.integer("callerOffset", 1, _UINT32_MAX)
    action_data_global_id = record.integer("actionDataGlobalId", 1, _UINT32_MAX)
    if raw.get("actionName") != contract.action_name:
        raise ActionMovementRuntimeError(f"{label}.actionName is invalid")
    action_class_vtable_offset = record.integer("actionClassVtableOffset", 1, _UINT32_MAX)
    if action_class_vtable_offset != contract.action_class_vtable_offset:
        raise ActionMovementRuntimeError(f"{label}.actionClassVtableOffset is invalid")
    runtime_class_vtable_offset = record.optional_integer("runtimeClassVtableOffset", 1, _UINT32_MAX)
    if runtime_class_vtable_offset != contract.runtime_class_vtable_offset:
        raise ActionMovementRuntimeError(f"{label}.runtimeClassVtableOffset is invalid")

    source_before = _entity_fact(raw.get("sourceBefore"), f"{label}.sourceBefore")
    source_after = _entity_fact(raw.get("sourceAfter"), f"{label}.sourceAfter")
    if not _same_entity_identity(source_before, source_after):
        raise ActionMovementRuntimeError(f"{label} source identity changed across the native call")
    if source_before.card_id not in contract.source_card_ids:
        raise ActionMovementRuntimeError(f"{label} source card/form identity is invalid")

    target_raw = raw.get("target")
    target = None if target_raw is None else _entity_fact(target_raw, f"{label}.target")
    if contract.target_required != (target is not None):
        raise ActionMovementRuntimeError(f"{label}.target presence is invalid")
    if target is not None and (target.native_object_id == source_before.native_object_id):
        raise ActionMovementRuntimeError(f"{label}.target aliases the movement source")

    chain_index = record.optional_integer("chainIndex", 0, _INT32_MAX)
    if contract.chain_index_required != (chain_index is not None):
        raise ActionMovementRuntimeError(f"{label}.chainIndex presence is invalid")

    position_changed = record.boolean("positionChanged")
    if position_changed != (source_before.position != source_after.position):
        raise ActionMovementRuntimeError(f"{label}.positionChanged contradicts source coordinates")
    if contract.position_change_required and not position_changed:
        raise ActionMovementRuntimeError(f"{label} is not an executed position commit")
    if record.boolean("completeContext") is not True:
        raise ActionMovementRuntimeError(f"{label} lacks complete native context")

    return ActionMovementEventV1(
        sequence=sequence,
        tick=tick,
        kind=kind,
        mode=contract.mode,
        stage=contract.stage,
        hook_offset=hook_offset,
        caller_offset=caller_offset,
        action_data_global_id=action_data_global_id,
        action_name=contract.action_name,
        action_class_vtable_offset=action_class_vtable_offset,
        runtime_class_vtable_offset=runtime_class_vtable_offset,
        source_before=source_before,
        source_after=source_after,
        target=target,
        chain_index=chain_index,
        position_changed=position_changed,
        complete_context=True,
    )


def parse_action_movement_runtime(
    value: object, *, generation: int, state_epoch: int, observation_tick: int
) -> ActionMovementRuntimeEnvelopeV1:
    """Validate one same-observation native movement ring envelope."""
    return _schema.envelope(value, generation, state_epoch, observation_tick, _event)
