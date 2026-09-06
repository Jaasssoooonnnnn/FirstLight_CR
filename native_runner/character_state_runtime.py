"""Strict contract for raw allowlisted native character-state transitions.

The exact native setter is shared by many character lifecycle paths.  This
module validates its card-scoped pre/post facts while deliberately retaining
the engine's integer states and caller offsets without naming them as public
dash, jump, warp, or effect stages.
"""

from __future__ import annotations

from dataclasses import dataclass

from .runtime_schema import RuntimeSchema, same_entity_identity, wire_fields

CHARACTER_STATE_RUNTIME_SCHEMA = "native-character-state-runtime.v1"
CHARACTER_STATE_RUNTIME_CAPACITY = 2_048
CHARACTER_STATE_SETTER_HOOK_OFFSET = 0xF18654
CHARACTER_STATE_CAPABILITY_SOURCE = "allowlisted-native-character-state-hook-ring"
CHARACTER_STATE_CAPABILITY_VALIDATION = "sha-build-id-prologue-abi-kind5-card-identity-committed-state"
CHARACTER_STATE_MINIMUM = 0
CHARACTER_STATE_MAXIMUM = 16
CHARACTER_STATE_CARD_IDS = frozenset(
    {
        26_000_046,  # Bandit / Assassin
        26_000_055,  # Mega Knight
        26_000_062,  # Elite Archer base form
        203_000_062,  # Elite Archer hero form
        26_000_074,  # Golden Knight
        26_000_081,  # Super Hog Rider Terry
        26_000_093,  # Little Prince (causal champion)
        26_000_103,  # Boss Bandit
        26_000_115,  # Champion Guard spawned by Little Prince
    }
)
_INT32_MAX = (1 << 31) - 1
_UINT32_MAX = (1 << 32) - 1


class CharacterStateRuntimeError(ValueError):
    """Raised when native character-state telemetry violates its contract."""


@dataclass(frozen=True, slots=True)
class CharacterStateEntityFactV1:
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
class CharacterStateCapabilityV1:
    status: str
    source: str
    validation: str
    confidence: str
    fail_closed: bool


@dataclass(frozen=True, slots=True)
class CharacterStateTransitionV1:
    sequence: int
    tick: int
    kind: str
    hook_offset: int
    caller_offset: int
    previous_state: int
    requested_state: int
    committed_state: int
    source_before: CharacterStateEntityFactV1
    source_after: CharacterStateEntityFactV1
    position_changed: bool
    complete_context: bool


@dataclass(frozen=True, slots=True)
class CharacterStateRuntimeEnvelopeV1:
    generation: int
    state_epoch: int
    observation_tick: int
    capacity: int
    hook_set_attested: bool
    hook_set_installed: bool
    capability: CharacterStateCapabilityV1
    epoch_first_sequence: int
    oldest_retained_sequence: int
    next_sequence: int
    overflow_count: int
    rejected_count: int
    sequence_gap_before_oldest: bool
    complete: bool
    events: tuple[CharacterStateTransitionV1, ...]

    @property
    def capability_status(self) -> str:
        return self.capability.status


_schema = RuntimeSchema(
    error=CharacterStateRuntimeError,
    label="characterStateRuntime",
    version=CHARACTER_STATE_RUNTIME_SCHEMA,
    capacity=CHARACTER_STATE_RUNTIME_CAPACITY,
    capability_source=CHARACTER_STATE_CAPABILITY_SOURCE,
    capability_validation=CHARACTER_STATE_CAPABILITY_VALIDATION,
    capability_type=CharacterStateCapabilityV1,
    entity_type=CharacterStateEntityFactV1,
    envelope_type=CharacterStateRuntimeEnvelopeV1,
    character_cards=CHARACTER_STATE_CARD_IDS,
)
_entity_fact = _schema.entity_fact
_same_entity_identity = same_entity_identity


def _event(
    value: object, *, index: int, expected_sequence: int, observation_tick: int, previous_tick: int | None
) -> CharacterStateTransitionV1:
    label = f"characterStateRuntime.events[{index}]"
    record = _schema.record(value, label, wire_fields(CharacterStateTransitionV1), "v1 event")
    raw = record.raw
    sequence, tick = record.event_header(expected_sequence, observation_tick, previous_tick)
    if raw.get("kind") != "native_character_state_transition":
        raise CharacterStateRuntimeError(f"{label}.kind is invalid")
    hook_offset = record.integer("hookOffset", 1, _UINT32_MAX)
    if hook_offset != CHARACTER_STATE_SETTER_HOOK_OFFSET:
        raise CharacterStateRuntimeError(f"{label}.hookOffset is invalid")
    caller_offset = record.integer("callerOffset", 1, _UINT32_MAX)
    previous_state = record.integer("previousState", CHARACTER_STATE_MINIMUM, CHARACTER_STATE_MAXIMUM)
    requested_state = record.integer("requestedState", CHARACTER_STATE_MINIMUM, CHARACTER_STATE_MAXIMUM)
    committed_state = record.integer("committedState", CHARACTER_STATE_MINIMUM, CHARACTER_STATE_MAXIMUM)
    if requested_state != committed_state:
        raise CharacterStateRuntimeError(f"{label} requested state was not committed")
    if previous_state == committed_state:
        raise CharacterStateRuntimeError(f"{label} is not a transition")

    source_before = _entity_fact(raw.get("sourceBefore"), f"{label}.sourceBefore")
    source_after = _entity_fact(raw.get("sourceAfter"), f"{label}.sourceAfter")
    if not _same_entity_identity(source_before, source_after):
        raise CharacterStateRuntimeError(f"{label} source identity changed across the native call")
    position_changed = record.boolean("positionChanged")
    if position_changed != (source_before.position != source_after.position):
        raise CharacterStateRuntimeError(f"{label}.positionChanged contradicts source coordinates")
    if record.boolean("completeContext") is not True:
        raise CharacterStateRuntimeError(f"{label} lacks complete native context")
    return CharacterStateTransitionV1(
        sequence=sequence,
        tick=tick,
        kind="native_character_state_transition",
        hook_offset=hook_offset,
        caller_offset=caller_offset,
        previous_state=previous_state,
        requested_state=requested_state,
        committed_state=committed_state,
        source_before=source_before,
        source_after=source_after,
        position_changed=position_changed,
        complete_context=True,
    )


def parse_character_state_runtime(
    value: object, *, generation: int, state_epoch: int, observation_tick: int
) -> CharacterStateRuntimeEnvelopeV1:
    """Validate one same-observation native movement ring envelope."""
    return _schema.envelope(value, generation, state_epoch, observation_tick, _event)
