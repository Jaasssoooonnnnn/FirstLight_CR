"""Fail-closed binding of native rich telemetry to typed runtime contracts.

The native runner deliberately exposes rich telemetry in a separate envelope.
This module binds that envelope to one ordinary observation only after the
engine tick, generation, object slot, and stable entity tuple all agree.  It
does not classify native buff names; the native identity is retained while the
typed effect kind remains unknown until a content-addressed BUFF catalog join
is available.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
from typing import Any

from .action_movement_runtime import (
    ActionMovementRuntimeEnvelopeV1,
    ActionMovementRuntimeError,
    parse_action_movement_runtime,
)
from .character_state_runtime import (
    CharacterStateRuntimeEnvelopeV1,
    CharacterStateRuntimeError,
    parse_character_state_runtime,
)
from .contracts import (
    ABILITY_RUNTIME_STATE_FIELDS,
    ATTACK_STATE_FIELDS,
    CAPTURE_RUNTIME_STATE_FIELDS,
    THRESHOLD_RELOCATION_STATE_FIELDS,
    EFFECT_STATE_FIELDS,
    ENTITY_RESOURCE_STATE_FIELDS,
    ENTITY_RUNTIME_SEMANTIC_FIELDS,
    PERIODIC_ATTACK_MODIFIER_STATE_FIELDS,
    EVOLUTION_RUNTIME_STATE_FIELDS,
    PLAYER_RUNTIME_SEMANTIC_FIELDS,
    PROJECTILE_STATE_FIELDS,
    SHIELD_STATE_FIELDS,
    TOWER_RUNTIME_SEMANTIC_FIELDS,
    VISIBILITY_STATE_FIELDS,
    AbilityPhase,
    AbilityRuntimeStateV1,
    AbilitySpecV1,
    AttackStateV1,
    CardSpecV1,
    CaptureActionPhase,
    CaptureRuntimeStateV1,
    CaptureTargetPhase,
    CaptureTargetStateV1,
    EffectKind,
    EffectStateV1,
    EntityResourceStateV1,
    EvolutionPhase,
    EvolutionRuntimeStateV1,
    ProjectilePhase,
    ProjectileDragStage,
    ProjectileStateV1,
    PeriodicAttackModifierPhase,
    PeriodicAttackModifierStateV1,
    SemanticEvidenceLevel,
    SemanticProvenanceV1,
    ShieldStateV1,
    ThresholdRelocationPhase,
    ThresholdRelocationStateV1,
    VisibilityPhase,
    VisibilityStateV1,
)
from .effect_catalog import NativeEffectCatalogV1
from .normal_form_evidence import NORMAL_MODE_HERO_FORM_BINDINGS_BY_FORM_ID
from .phase_runtime import (
    ATTACK_SEQUENCE_DECAY_DURATION_MS,
    ATTACK_SEQUENCE_PROGRESS_LIMIT,
    PHASE_RUNTIME_SCHEMA,
    AttackRuntimeRaw,
    DamageRampIdentity,
    DamageRampSpec,
    DeploymentRuntimeRaw,
    DeploymentRuntimeV1,
    MovementRuntimeRaw,
    MovementRuntimeV1,
    PhaseHookEvent,
    PhaseHookKind,
    PhaseRuntimeError,
    PhaseRuntimeWindowV1,
    resolve_attack_runtime,
    resolve_deployment_runtime,
    resolve_movement_runtime,
)
from .special_movement_runtime import (
    SpecialMovementRuntimeEnvelopeV1,
    SpecialMovementRuntimeError,
    parse_special_movement_runtime,
)
from .visibility_runtime import VISIBILITY_RUNTIME_SCHEMA, VisibilityTransitionKind


RICH_TELEMETRY_SCHEMA = "native-rich-telemetry.v3"
COMBAT_EVENT_SCHEMA = "native-combat-events.v1"
NATIVE_OBJECT_ID_ENTITY_KEY_TAG = -2
MAX_COMBAT_EVENTS = 1024
MAX_ACTIVE_EFFECTS = 64
PHASE_OBJECT_SCHEMA = "native-phase-object.v1"
MAX_PHASE_EVENTS = 4096
MAX_VISIBILITY_EVENTS = 4096
REMAINING_RUNTIME_SCHEMA = "native-remaining-runtime.v1"
MAX_REMAINING_EVENTS = 4096
COMBAT_CONSUME_CARD_HOOK_OFFSET = 0xF38A68
EntityKey = tuple[int, int, int]
COMBAT_EVENT_KINDS = frozenset(
    {
        "damage",
        "death",
        "heal",
        "spawn",
        "projectile_spawn",
        "despawn",
        "shield_damage",
        "shield_break",
        "projectile_impact",
        "projectile_deflect",
        "projectile_expire",
        "projectile_terminal",
        "card_play",
    }
)
CARD_FORM_LABELS = ("BasicForm", "EvoForm", "HeroForm", "AutoChessForm", "FormEnd", "FlexSlot")
MIRROR_CARD_GLOBAL_ID = 28_000_006
ABILITY_BUTTON_STATE_LABELS = (
    "invalid/no match",
    "ChampionAbsent",
    "Ready",
    "ChampionDeploying",
    "LimitedAvailability",
    "ERR_START",
    "AllChargesConsumed",
    "ChampionPending",
    "OnCooldown",
    "NotEnoughElixir",
    "ChampionCasting",
    "Disabled",
    "TemporarilyUnavailable",
    "NoYetAvailable",
    "ERR_MAX",
)
ABILITY_QUEUEABLE_BUTTON_STATES = frozenset((2, 4))

# Exact-build Hero-form entity IDs do not reuse the base deck card ID.  Keep
# this bridge deliberately narrow and evidence-bound: it is used only after
# the runtime action name, configured cooldown, one live controller member,
# deck slot, and static ability record have already joined uniquely.
_VALIDATED_HERO_FORM_SOURCE_CARDS = {
    (binding.ability_id, binding.runtime_form_card_id): binding.source_card_id
    for binding in NORMAL_MODE_HERO_FORM_BINDINGS_BY_FORM_ID.values()
}
_REMAINING_EVENT_HOOKS: Mapping[str, frozenset[int]] = {
    "area_create": frozenset({0xF25B8C}),
    "area_expire": frozenset({0xF25D90}),
    "resource_delta": frozenset({0xF3BA4C, 0xF3B3D4}),
    "tower_aggro_acquire": frozenset({0xF5C894}),
    "tower_aggro_change": frozenset({0xF5C894}),
    "tower_aggro_lose": frozenset({0xF5C894}),
    "tower_activate": frozenset({0xF23110}),
    "lifetime_spawn": frozenset({0xF25B8C}),
    "lifetime_despawn": frozenset({0xF25D90}),
    "transform": frozenset({0xE58638}),
    "spawn_attach": frozenset({0xF19314}),
    "area_damage_eligibility": frozenset({0xF1CB90}),
    "area_relation_parent": frozenset({0xF25B8C}),
    "area_relation_follow": frozenset({0xF25B8C}),
    "area_relation_related": frozenset({0xF25B8C}),
    # Exact AEO-tick and ActionSpawnToLocation-mode-1 producers.
    "area_action_spawn": frozenset({0xF143A8, 0xE7B864}),
}
_VISIBILITY_EVENT_HOOKS: Mapping[VisibilityTransitionKind, int] = {
    VisibilityTransitionKind.BECAME_INVISIBLE: 0xF5A2D4,
    VisibilityTransitionKind.BECAME_VISIBLE: 0xF5A578,
}
_VISIBILITY_EVENT_CALLERS: Mapping[VisibilityTransitionKind, int] = {
    VisibilityTransitionKind.BECAME_INVISIBLE: 0xF59EAC,
    VisibilityTransitionKind.BECAME_VISIBLE: 0xF59308,
}


class RichTelemetryMergeError(ValueError):
    """Raised when ordinary and rich observations cannot be safely joined."""


@dataclass(frozen=True, slots=True)
class RuntimeEffectCatalog:
    """Pre-indexed, content-addressed view of the static BUFF catalog."""

    catalog_id: str
    card_logic_catalog_id: str
    definitions: Mapping[str, Any]
    definitions_by_global_id: Mapping[int, Any]

    @classmethod
    def from_catalog(cls, catalog: NativeEffectCatalogV1) -> "RuntimeEffectCatalog":
        return cls(
            catalog_id=catalog.catalog_id,
            card_logic_catalog_id=catalog.card_logic_catalog_id,
            definitions=catalog.by_name,
            definitions_by_global_id=catalog.by_global_id,
        )


@lru_cache(maxsize=1)
def load_default_effect_catalog() -> RuntimeEffectCatalog | None:
    """Load the newest immutable local catalog, or retain unknown semantics."""

    directory = Path(__file__).resolve().parent / "artifacts" / "effect-catalogs"
    candidates = sorted(directory.glob("effect-catalog-*.json"), key=lambda path: (path.stat().st_mtime_ns, path.name))
    if not candidates:
        return None
    try:
        catalog = NativeEffectCatalogV1.load(candidates[-1])
    except (OSError, KeyError, TypeError, ValueError):
        return None
    if candidates[-1].stem != f"effect-catalog-{catalog.catalog_id}":
        return None
    return RuntimeEffectCatalog.from_catalog(catalog)


def _integer(value: object, label: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RichTelemetryMergeError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise RichTelemetryMergeError(f"{label} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise RichTelemetryMergeError(f"{label} must be <= {maximum}")
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RichTelemetryMergeError(f"{label} must be a mapping")
    return value


def _record(value: object, label: str) -> "_Record":
    return _Record(_mapping(value, label), label)


def _items(value: object, label: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise RichTelemetryMergeError(f"{label} must be a sequence")
    return value


def _entity_key(value: object, label: str) -> EntityKey:
    parts = _items(value, label)
    if len(parts) != 3:
        raise RichTelemetryMergeError(f"{label} must contain three integers")
    return (_integer(parts[0], f"{label}[0]"), _integer(parts[1], f"{label}[1]"), _integer(parts[2], f"{label}[2]"))


def _expected_native_entity_key(
    *, owner: int, object_index: int, secondary_index: int, native_object_id: int
) -> EntityKey:
    del object_index, secondary_index
    return (owner, NATIVE_OBJECT_ID_ENTITY_KEY_TAG, native_object_id)


def _bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise RichTelemetryMergeError(f"{label} must be a boolean")
    return value


def _status_record(container: Mapping[str, Any], name: str, expected: str, *, label: str) -> None:
    record = _mapping(container.get(name), f"{label}.{name}")
    if record.get("status") != expected:
        raise RichTelemetryMergeError(f"{label}.{name}.status must be {expected!r}")


class _Record(Mapping[str, Any]):
    """A native mapping with a bound diagnostic path and typed field readers.

    The wire mapping is retained by reference: binding does not copy frames or
    construct intermediate dictionaries. Field ranges remain at each call site.
    """

    __slots__ = ("_raw", "_label", "get")

    def __init__(self, raw: Mapping[str, Any], label: str):
        self._raw = raw
        self._label = label
        self.get = raw.get

    def __getitem__(self, key: str) -> Any:
        return self._raw[key]

    def __iter__(self):
        return iter(self._raw)

    def __len__(self) -> int:
        return len(self._raw)

    def __contains__(self, key: object) -> bool:
        return key in self._raw

    def _read(self, parser, field: str, *, default=None, **bounds):
        return parser(self.get(field, default), f"{self._label}.{field}", **bounds)

    def integer(self, field: str, *, minimum: int | None = None, maximum: int | None = None, default=None) -> int:
        value = self.get(field, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise RichTelemetryMergeError(f"{self._label}.{field} must be an integer")
        if minimum is not None and value < minimum:
            raise RichTelemetryMergeError(f"{self._label}.{field} must be >= {minimum}")
        if maximum is not None and value > maximum:
            raise RichTelemetryMergeError(f"{self._label}.{field} must be <= {maximum}")
        return value

    def boolean(self, field: str) -> bool:
        value = self.get(field)
        if not isinstance(value, bool):
            raise RichTelemetryMergeError(f"{self._label}.{field} must be a boolean")
        return value

    def optional_integer(self, field: str, **bounds) -> int | None:
        return None if self.get(field) is None else self.integer(field, **bounds)

    def optional_nonnegative(self, field: str, **bounds) -> int | None:
        bounds.setdefault("minimum", 0)
        return self.optional_integer(field, **bounds)

    def tick(self, field: str) -> int | None:
        return self._read(_phase_optional_tick, field)

    def point(self, field: str) -> tuple[int, int] | None:
        return self._read(_combat_point, field)

    def entity_key(self, field: str) -> EntityKey:
        return self._read(_entity_key, field)

    def mapping(self, field: str) -> Mapping[str, Any]:
        return self._read(_mapping, field)

    def record(self, field: str) -> "_Record":
        return self._read(_record, field)

    def sequence(self, field: str) -> Sequence[object]:
        return self._read(_items, field)


@dataclass(frozen=True, slots=True)
class RichActiveEffect:
    """One native active-buff instance with its unclassified asset identity."""

    buff_global_id: int
    name: str
    remaining_ms: int
    source_entity_key: EntityKey | None
    source_entity_validated: bool


@dataclass(frozen=True, slots=True)
class RichProjectileTelemetry:
    projectile_data_global_id: int
    source_entity_key: EntityKey | None
    source_entity_validated: bool
    target_entity_key: EntityKey | None
    target_entity_validated: bool
    homing_target_entity_key: EntityKey | None
    homing_target_entity_validated: bool
    destination: tuple[int, int]
    terminal: bool
    native_phase: str
    drag_stage: str | None = None


@dataclass(frozen=True, slots=True)
class RichEntityResourceTelemetry:
    kind: str
    current_raw: int
    capacity_raw: int
    base_raw: int
    limit_raw: int
    normalized: float


@dataclass(frozen=True, slots=True)
class RichPeriodicAttackModifierTelemetry:
    phase: str
    period_attacks: int
    completed_attacks: int
    added_damage_raw: int
    linger_duration_ms: int
    linger_remaining_ms: int
    source_native_object_id: int
    source_entity_key: EntityKey | None
    source_resolved: bool


@dataclass(frozen=True, slots=True)
class RichCaptureTargetTelemetry:
    target_native_object_id: int
    target_entity_key: EntityKey | None
    target_resolved: bool
    phase: str
    elapsed_ms: int
    phase_budget_remaining_ms: int | None


@dataclass(frozen=True, slots=True)
class RichCaptureRuntimeTelemetry:
    phase: str
    configured_cooldown_ms: int
    cooldown_remaining_ms: int
    hit_frequency_ms: int
    hit_accumulator_ms: int
    targets: tuple[RichCaptureTargetTelemetry, ...]


@dataclass(frozen=True, slots=True)
class RichThresholdRelocationTelemetry:
    phase: str
    stage: int
    relocation_index: int
    thresholds_percent: tuple[int, ...]
    hide_duration_ms: int
    remaining_ms: int
    burrowed: bool


@dataclass(frozen=True, slots=True)
class RichAbilityRuntimeTelemetry:
    controller_slot: int
    action_data_global_id: int
    action_data_name: str
    selected_character_data_global_id: int
    remaining_cooldown_ms: int
    configured_cooldown_ms: int
    remaining_charges_raw: int
    max_charges: int
    button_state: int
    button_state_label: str
    available: bool
    champion_entity_keys: tuple[EntityKey, ...]


@dataclass(frozen=True, slots=True)
class RichEvolutionRuntimeTelemetry:
    deck_slot: int
    card_id: int
    base_spell_global_id: int
    evolvable: bool
    evolution_form_global_id: int | None
    progress: int
    cycle_required: int | None
    cycle_remaining: int | None
    ready: bool | None


@dataclass(frozen=True, slots=True)
class RichPlayerRuntimeTelemetry:
    owner: int
    owner_root_validated: bool
    owner_entity_key: EntityKey | None
    ability_runtime: tuple[RichAbilityRuntimeTelemetry, ...] | None
    evolution_runtime: tuple[RichEvolutionRuntimeTelemetry, ...] | None


@dataclass(frozen=True, slots=True)
class RichCombatEntityFact:
    validated: bool
    present: bool
    native_object_id: int | None
    entity_key: EntityKey | None
    owner: int | None
    object_index: int | None
    secondary_index: int | None
    card_id: int | None
    object_kind: int | None
    position: tuple[int, int] | None
    visibility_validated: bool
    invisible_count: int | None


@dataclass(frozen=True, slots=True)
class RichCombatDeploymentContext:
    deployment_sequence: int
    owner: int
    played_card_global_id: int
    effective_card_global_id: int | None
    card_parameter: int
    deck_slot: int
    cost: int
    form_code: int
    form_name: str


@dataclass(frozen=True, slots=True)
class RichCombatEventTelemetry:
    sequence: int
    tick: int
    generation: int
    state_epoch: int
    kind: str
    hook_offset: int
    caller_offset: int | None
    cause_sequence: int | None
    pool: str
    terminal_reason: str
    lethal: bool
    deployment_context: RichCombatDeploymentContext | None
    target: RichCombatEntityFact
    immediate_source: RichCombatEntityFact
    source: RichCombatEntityFact
    projectile: RichCombatEntityFact
    related: RichCombatEntityFact
    route_source_before: RichCombatEntityFact
    route_target_before: RichCombatEntityFact
    route_source_after: RichCombatEntityFact
    route_target_after: RichCombatEntityFact
    requested_amount: int | None
    actual_amount: int | None
    pre_hp: int | None
    post_hp: int | None
    pre_builtin_shield: int | None
    post_builtin_shield: int | None
    pre_buff_shield: int | None
    post_buff_shield: int | None
    destination_before: tuple[int, int] | None
    destination_after: tuple[int, int] | None

    @property
    def facts(self) -> tuple[RichCombatEntityFact, ...]:
        return (
            self.target,
            self.immediate_source,
            self.source,
            self.projectile,
            self.related,
            self.route_source_before,
            self.route_target_before,
            self.route_source_after,
            self.route_target_after,
        )


@dataclass(frozen=True, slots=True)
class RichCombatEventEnvelope:
    generation: int
    state_epoch: int
    observation_tick: int
    capacity: int
    hook_set_attested: bool
    hook_set_installed: bool
    capability_status: str
    epoch_first_sequence: int
    oldest_retained_sequence: int
    next_sequence: int
    overflow_count: int
    sequence_gap_before_oldest: bool
    rejected_capture_count: int
    complete: bool
    events: tuple[RichCombatEventTelemetry, ...]


@dataclass(frozen=True, slots=True)
class RichPhaseObjectTelemetry:
    """Exact-build snapshot facts used by the phase resolvers."""

    attack_validated: bool
    movement_validated: bool
    buffs_validated: bool
    attack_sequence_stage: int | None
    attack_timeline_ms: int | None
    load_remaining_ms: int | None
    deploy_remaining_ms: int | None
    deploy_previous_ms: int | None
    configured_deploy_time_ms: int | None
    hit_speed_ms: int | None
    attack_dash_time_ms: int | None
    base_movement_speed: int | None
    charge_speed_multiplier: int | None
    classic_charge_progress: int | None
    speed_positive_percent: int | None
    speed_negative_magnitude: int | None
    hit_speed_positive_percent: int | None
    hit_speed_negative_magnitude: int | None
    attack_step_input: int | None
    attack_step_output: int | None
    attack_step_tick: int | None
    movement_step_input: int | None
    movement_step_output: int | None
    movement_step_tick: int | None
    deploy_step_input: int | None
    deploy_step_output: int | None
    deploy_step_tick: int | None
    effective_movement_speed: int | None
    effective_movement_speed_tick: int | None
    movement_delta: int | None
    movement_delta_tick: int | None
    attack_sequence_progress_raw: int | None
    attack_sequence_progress_limit: int | None
    attack_sequence_decay_remaining_ms: int | None
    attack_sequence_decay_duration_ms: int | None


@dataclass(frozen=True, slots=True)
class RichPhaseRuntimeEnvelope:
    generation: int
    state_epoch: int
    observation_tick: int
    hook_set_attested: bool
    hook_set_installed: bool
    capability_status: str
    window: PhaseRuntimeWindowV1
    sequence_gap_before_oldest: bool
    events: tuple[PhaseHookEvent, ...]


@dataclass(frozen=True, slots=True)
class RichVisibilityRuntimeEvent:
    sequence: int
    tick: int
    kind: VisibilityTransitionKind
    hook_offset: int
    caller_offset: int
    subject: RichCombatEntityFact
    buff_global_id: int
    invisible_count_before: int
    invisible_count_after: int
    scope: str
    complete_context: bool


@dataclass(frozen=True, slots=True)
class RichVisibilityRuntimeEnvelope:
    generation: int
    state_epoch: int
    observation_tick: int
    capacity: int
    hook_set_attested: bool
    transition_hook_set_installed: bool
    contextual_gate_hook_set_installed: bool
    capability_status: str
    contextual_gate_capability_status: str
    epoch_first_sequence: int
    oldest_retained_sequence: int
    next_sequence: int
    overflow_count: int
    rejected_count: int
    sequence_gap_before_oldest: bool
    complete: bool
    public_by_owner_available: bool
    targetable_by_owner_available: bool
    events: tuple[RichVisibilityRuntimeEvent, ...]


@dataclass(frozen=True, slots=True)
class RichRemainingRuntimeEvent:
    sequence: int
    tick: int
    kind: str
    hook_offset: int
    caller_offset: int | None
    entity: RichCombatEntityFact
    source: RichCombatEntityFact
    target: RichCombatEntityFact
    target_before: RichCombatEntityFact
    target_after: RichCombatEntityFact
    object_kind: int | None
    runtime_vtable_offset: int | None
    data_before_global_id: int | None
    data_after_global_id: int | None
    expected_character_data_global_id: int | None
    expected_projectile_data_global_id: int | None
    configured_data_global_id: int | None
    resource_pre_fixed: int | None
    resource_post_fixed: int | None
    resource_actual_delta_fixed: int | None
    amount_argument: int | None
    configured_amount_argument: int | None
    resource_owner: int | None
    area_remaining_life_ms: int | None
    resource_cause: str
    transform_kind: str
    result: bool
    option: bool
    committed: bool
    reset_target: bool
    complete_context: bool

    @property
    def facts(self) -> tuple[RichCombatEntityFact, ...]:
        return (self.entity, self.source, self.target, self.target_before, self.target_after)


@dataclass(frozen=True, slots=True)
class RichRemainingRuntimeEnvelope:
    generation: int
    state_epoch: int
    observation_tick: int
    capacity: int
    hook_set_attested: bool
    hook_set_installed: bool
    capability_status: str
    epoch_first_sequence: int
    oldest_retained_sequence: int
    next_sequence: int
    overflow_count: int
    rejected_count: int
    sequence_gap_before_oldest: bool
    complete: bool
    public_by_owner_available: bool
    targetable_by_owner_available: bool
    events: tuple[RichRemainingRuntimeEvent, ...]


@dataclass(frozen=True, slots=True)
class RichDaggerDuchessRuntimeTelemetry:
    observed_tick: int
    charge_count: int
    max_charge_count: int
    recharge_elapsed_ms: int
    recharge_duration_ms: int


@dataclass(frozen=True, slots=True)
class RichRoyalChefRuntimeTelemetry:
    observed_tick: int
    start_delay_remaining_ms: int
    start_delay_duration_ms: int
    cooking_contribution: int
    contribution_needed: int
    throw_delay_remaining_ms: int | None
    target_native_object_id: int | None


RichTowerTroopRuntimeTelemetry = RichDaggerDuchessRuntimeTelemetry | RichRoyalChefRuntimeTelemetry


@dataclass(frozen=True, slots=True)
class RichTowerTroopRuntimeEnvelope:
    generation: int
    state_epoch: int
    observation_tick: int
    hook_set_attested: bool
    hook_set_installed: bool
    rejected_count: int
    complete: bool


@dataclass(frozen=True, slots=True)
class RichObjectTelemetry:
    slot: int
    native_object_id: int
    entity_key: EntityKey
    owner: int
    object_index: int
    secondary_index: int
    card_id: int
    data_global_id: int | None
    shield_current: int | None
    shield_maximum: int | None
    target_entity_key: EntityKey | None
    target_entity_validated: bool
    attack_sequence_stage: int | None
    active_effects: tuple[RichActiveEffect, ...] | None
    invisible_count: int | None
    visibility_state: str | None
    projectile: RichProjectileTelemetry | None
    entity_resource_runtime: RichEntityResourceTelemetry | None
    periodic_attack_modifier_runtime: RichPeriodicAttackModifierTelemetry | None
    capture_runtime: RichCaptureRuntimeTelemetry | None
    threshold_relocation_runtime: RichThresholdRelocationTelemetry | None
    phase_runtime: RichPhaseObjectTelemetry | None
    tower_troop_runtime: RichTowerTroopRuntimeTelemetry | None

    @property
    def invisible(self) -> bool:
        return self.invisible_count is not None and self.invisible_count > 0


@dataclass(frozen=True, slots=True)
class RichTelemetrySnapshot:
    tick: int
    generation: int
    state_epoch: int
    objects: tuple[RichObjectTelemetry | None, ...]
    objects_by_key: Mapping[EntityKey, RichObjectTelemetry]
    objects_by_native_object_id: Mapping[int, RichObjectTelemetry]
    players: Mapping[int, RichPlayerRuntimeTelemetry]
    combat_events: RichCombatEventEnvelope
    phase_runtime: RichPhaseRuntimeEnvelope | None
    visibility_runtime: RichVisibilityRuntimeEnvelope | None
    remaining_runtime: RichRemainingRuntimeEnvelope | None
    tower_troop_runtime: RichTowerTroopRuntimeEnvelope | None
    special_movement_runtime: SpecialMovementRuntimeEnvelopeV1 | None = None
    action_movement_runtime: ActionMovementRuntimeEnvelopeV1 | None = None
    character_state_runtime: CharacterStateRuntimeEnvelopeV1 | None = None

    def at_slot(self, slot: int) -> RichObjectTelemetry | None:
        if slot < 0 or slot >= len(self.objects):
            raise RichTelemetryMergeError(f"rich object slot {slot} is out of range")
        return self.objects[slot]


def _validate_envelope_identity(ordinary: Mapping[str, Any], rich: Mapping[str, Any]) -> tuple[int, int, int]:
    if rich.get("schema") != RICH_TELEMETRY_SCHEMA or rich.get("ok") is not True:
        raise RichTelemetryMergeError("rich telemetry schema/ok marker is invalid")
    ordinary_tick = _integer(ordinary.get("tick"), "ordinary.tick", minimum=0)
    rich_tick = _integer(rich.get("tick"), "rich.tick", minimum=0)
    ordinary_generation = _integer(ordinary.get("generation"), "ordinary.generation", minimum=0)
    rich_generation = _integer(rich.get("generation"), "rich.generation", minimum=0)
    if ordinary_tick != rich_tick:
        raise RichTelemetryMergeError(f"rich telemetry tick mismatch: ordinary={ordinary_tick}, rich={rich_tick}")
    if ordinary_generation != rich_generation:
        raise RichTelemetryMergeError(
            f"rich telemetry generation mismatch: ordinary={ordinary_generation}, rich={rich_generation}"
        )
    state_epoch = _integer(rich.get("stateEpoch"), "rich.stateEpoch", minimum=0)
    if ordinary.get("stateEpoch") is not None:
        ordinary_epoch = _integer(ordinary.get("stateEpoch"), "ordinary.stateEpoch", minimum=0)
        if ordinary_epoch != state_epoch:
            raise RichTelemetryMergeError(
                f"rich telemetry stateEpoch mismatch: ordinary={ordinary_epoch}, rich={state_epoch}"
            )
    return ordinary_tick, ordinary_generation, state_epoch


def _validate_field_provenance(rich: Mapping[str, Any]) -> None:
    provenance = _mapping(rich.get("provenance"), "rich.provenance")
    for name in (
        "nativeObjectId",
        "dataGlobalId",
        "targetEntityKey",
        "shield.current",
        "shield.max",
        "attackSequenceStage",
        "activeEffects",
        "activeEffects.buffGlobalId",
        "activeEffects.name",
        "activeEffects.remainingMs",
        "activeEffects.sourceEntityKey",
        "invisibleCount",
        "projectile.dragStage",
    ):
        _status_record(provenance, name, "authoritative", label="rich.provenance")
    _status_record(provenance, "entityKey", "derived", label="rich.provenance")
    _status_record(provenance, "activeEffects.sourceEntityValidated", "derived", label="rich.provenance")
    _status_record(provenance, "visibilityState", "derived", label="rich.provenance")
    for name in (
        "projectile",
        "projectile.projectileDataGlobalId",
        "projectile.sourceEntityKey",
        "projectile.targetEntityKey",
        "projectile.homingTargetEntityKey",
        "projectile.destination",
        "projectile.terminal",
        "projectile.nativePhase",
        "players.ownerRoot",
        "players.abilityRuntime",
        "players.abilityRuntime.buttonState",
        "players.abilityRuntime.cooldown",
        "players.abilityRuntime.charges",
        "players.evolutionRuntime",
        "players.evolutionRuntime.cycle",
    ):
        _status_record(provenance, name, "derived", label="rich.provenance")
    _status_record(provenance, "players.evolutionRuntime.playedForm", "unavailable", label="rich.provenance")
    if "towerTroopRuntime" in rich:
        _status_record(provenance, "towerTroopRuntime", "derived", label="rich.provenance")

    capabilities = _mapping(rich.get("capabilities"), "rich.capabilities")
    for name in ("targetEntity", "shield", "attackSequenceStage", "buffs", "invisibility"):
        _status_record(capabilities, name, "authoritative", label="rich.capabilities")
    for name in ("projectile", "abilityRuntime", "evolutionRuntime"):
        _status_record(capabilities, name, "derived", label="rich.capabilities")
    if "towerTroopRuntime" in rich:
        _status_record(capabilities, "towerTroopRuntime", "derived", label="rich.capabilities")
    if "phaseRuntime" in rich:
        for name in ("slow", "rage", "stun", "freeze", "attackPhase", "deployPhase", "chargeStage"):
            _status_record(capabilities, name, "derived", label="rich.capabilities")
    _status_record(capabilities, "impact", "unavailable", label="rich.capabilities")


def _combat_optional_integer(value: object, label: str, *, minimum: int = 0, maximum: int | None = None) -> int | None:
    if value is None:
        return None
    return _integer(value, label, minimum=minimum, maximum=maximum)


def _combat_point(value: object, label: str) -> tuple[int, int] | None:
    if value is None:
        return None
    parts = _items(value, label)
    if len(parts) != 2:
        raise RichTelemetryMergeError(f"{label} must contain two integers")
    return (_integer(parts[0], f"{label}[0]"), _integer(parts[1], f"{label}[1]"))


def _combat_fact(value: object, label: str) -> RichCombatEntityFact:
    fact = _record(value, label)
    required = {
        "validated",
        "present",
        "nativeObjectId",
        "entityKey",
        "owner",
        "objectIndex",
        "secondaryIndex",
        "cardId",
        "objectKind",
        "position",
        "visibilityValidated",
        "invisibleCount",
    }
    if set(fact) != required:
        raise RichTelemetryMergeError(f"{label} fields do not match the combat entity schema")
    validated = fact.boolean("validated")
    present = fact.boolean("present")
    visibility_validated = fact.boolean("visibilityValidated")
    if not present:
        if any(
            fact.get(name) is not None
            for name in (
                "nativeObjectId",
                "entityKey",
                "owner",
                "objectIndex",
                "secondaryIndex",
                "cardId",
                "objectKind",
                "position",
                "invisibleCount",
            )
        ):
            raise RichTelemetryMergeError(f"{label} exposes identity/state for an absent relation")
        if visibility_validated:
            raise RichTelemetryMergeError(f"{label} validates visibility for an absent relation")
        return RichCombatEntityFact(
            validated=validated,
            present=False,
            native_object_id=None,
            entity_key=None,
            owner=None,
            object_index=None,
            secondary_index=None,
            card_id=None,
            object_kind=None,
            position=None,
            visibility_validated=False,
            invisible_count=None,
        )
    if not validated:
        raise RichTelemetryMergeError(f"{label} exposes an unvalidated native entity")
    native_object_id = fact.integer("nativeObjectId", minimum=1)
    entity_key = fact.entity_key("entityKey")
    owner = fact.integer("owner")
    object_index = fact.integer("objectIndex")
    secondary_index = fact.integer("secondaryIndex")
    if entity_key != _expected_native_entity_key(
        owner=owner, object_index=object_index, secondary_index=secondary_index, native_object_id=native_object_id
    ):
        raise RichTelemetryMergeError(f"{label} entityKey does not match authoritative identity fields")
    card_id = fact.integer("cardId")
    object_kind = fact.integer("objectKind", minimum=-1)
    if object_kind > 31:
        raise RichTelemetryMergeError(f"{label}.objectKind exceeds the wire bound")
    position = fact.point("position")
    if position is None:
        raise RichTelemetryMergeError(f"{label} lacks a present position")
    invisible_count = fact.optional_nonnegative("invisibleCount")
    if visibility_validated != (invisible_count is not None):
        raise RichTelemetryMergeError(f"{label} visibility validation/value mismatch")
    return RichCombatEntityFact(
        validated=True,
        present=True,
        native_object_id=native_object_id,
        entity_key=entity_key,
        owner=owner,
        object_index=object_index,
        secondary_index=secondary_index,
        card_id=card_id,
        object_kind=object_kind,
        position=position,
        visibility_validated=visibility_validated,
        invisible_count=invisible_count,
    )


def _combat_deployment_context(value: object, label: str) -> RichCombatDeploymentContext | None:
    if value is None:
        return None
    raw = _record(value, label)
    required = {
        "deploymentSequence",
        "owner",
        "playedCardGlobalId",
        "effectiveCardGlobalId",
        "cardParameter",
        "deckSlot",
        "cost",
        "formCode",
        "formName",
        "consumeHookOffset",
    }
    if set(raw) != required:
        raise RichTelemetryMergeError(f"{label} fields do not match the v1 deployment schema")
    deployment_sequence = raw.integer("deploymentSequence", minimum=1)
    owner = raw.integer("owner", minimum=0)
    if owner > 1:
        raise RichTelemetryMergeError(f"{label}.owner is invalid")
    played_card_global_id = raw.integer("playedCardGlobalId", minimum=1, maximum=4294967295)
    effective_card_global_id = raw.optional_nonnegative("effectiveCardGlobalId", minimum=1, maximum=4294967295)
    card_parameter = raw.integer("cardParameter", minimum=0, maximum=4294967295)
    deck_slot = raw.integer("deckSlot", minimum=0)
    cost = raw.integer("cost", minimum=0)
    form_code = raw.integer("formCode", minimum=0)
    if deck_slot > 7 or cost > 15 or form_code >= len(CARD_FORM_LABELS):
        raise RichTelemetryMergeError(f"{label} packed descriptor field is out of range")
    if (
        ((card_parameter >> 22) & 0x3F) - 1 != deck_slot
        or card_parameter >> 28 != cost
        or card_parameter & 0xF != form_code
        or raw.get("formName") != CARD_FORM_LABELS[form_code]
        or raw.integer("consumeHookOffset", minimum=1) != COMBAT_CONSUME_CARD_HOOK_OFFSET
    ):
        raise RichTelemetryMergeError(f"{label} packed descriptor fields disagree")
    mirror_basic_form = (
        form_code == 0
        and played_card_global_id == MIRROR_CARD_GLOBAL_ID
        and effective_card_global_id is not None
        and effective_card_global_id != played_card_global_id
    )
    if (form_code == 0 and effective_card_global_id != played_card_global_id and not mirror_basic_form) or (
        form_code == 1 and effective_card_global_id is None
    ):
        raise RichTelemetryMergeError(f"{label} effective form identity is incomplete")
    return RichCombatDeploymentContext(
        deployment_sequence=deployment_sequence,
        owner=owner,
        played_card_global_id=played_card_global_id,
        effective_card_global_id=effective_card_global_id,
        card_parameter=card_parameter,
        deck_slot=deck_slot,
        cost=cost,
        form_code=form_code,
        form_name=CARD_FORM_LABELS[form_code],
    )


_EVENT_ENVELOPE_FIELDS = frozenset(
    {
        "ok",
        "schema",
        "generation",
        "stateEpoch",
        "observationTick",
        "capacity",
        "hookSetAttested",
        "hookSetInstalled",
        "capability",
        "epochFirstSequence",
        "oldestRetainedSequence",
        "nextSequence",
        "overflowCount",
        "sequenceGapBeforeOldest",
        "rejectedCount",
        "complete",
        "events",
    }
)


def _event_envelope(
    value: object,
    label: str,
    schema: str,
    capacity: int,
    *,
    generation: int,
    state_epoch: int,
    observation_tick: int,
    fields: Mapping[str, str | None] | None = None,
) -> _Record:
    """Bind the common identity and field boundary of native event rings."""
    envelope = _record(value, label)
    required = set(_EVENT_ENVELOPE_FIELDS)
    for old, new in (fields or {}).items():
        if old.startswith("+"):
            required.add(old[1:])
        else:
            required.remove(old)
            required.add(new)
    if set(envelope) - {"transmitFromSequence"} != required:
        raise RichTelemetryMergeError(f"{label} fields do not match the v1 schema")
    if envelope.get("ok") is not True or envelope.get("schema") != schema:
        raise RichTelemetryMergeError(f"{label} schema/ok marker is invalid")
    for name, expected, minimum in (
        ("generation", generation, None),
        ("stateEpoch", state_epoch, None),
        ("observationTick", observation_tick, -1),
    ):
        if envelope.integer(name, minimum=minimum) != expected:
            raise RichTelemetryMergeError(f"{label} observation identity is inconsistent")
    if envelope.integer("capacity", minimum=1) != capacity:
        raise RichTelemetryMergeError(f"{label} capacity is unexpected")
    return envelope


def _ring_counters(envelope: _Record) -> tuple[int, int, int, int, bool]:
    first = envelope.integer("epochFirstSequence", minimum=1)
    oldest = envelope.integer("oldestRetainedSequence", minimum=first)
    next_sequence = envelope.integer("nextSequence", minimum=oldest)
    overflow = envelope.integer("overflowCount", minimum=0)
    capacity = envelope["capacity"]
    if overflow != max(0, next_sequence - first - capacity):
        raise RichTelemetryMergeError(f"{envelope._label} overflow accounting is inconsistent")
    if oldest != max(first, next_sequence - capacity):
        raise RichTelemetryMergeError(f"{envelope._label} retained window is inconsistent")
    gap = envelope.boolean("sequenceGapBeforeOldest")
    if gap != (oldest > first):
        raise RichTelemetryMergeError(f"{envelope._label} retained-window gap is inconsistent")
    return first, oldest, next_sequence, overflow, gap


def _ring_records(
    envelope: _Record, first: int, oldest: int, next_sequence: int, minimum_sequence: int | None
) -> tuple[int, Sequence[object], int]:
    transmit_from = envelope.integer("transmitFromSequence", minimum=oldest, maximum=next_sequence, default=oldest)
    records = envelope.sequence("events")
    if len(records) != next_sequence - transmit_from or len(records) > envelope["capacity"]:
        raise RichTelemetryMergeError(f"{envelope._label} retained event range is incomplete")
    first_index = 0
    if minimum_sequence is not None and first <= minimum_sequence <= next_sequence:
        if transmit_from > max(minimum_sequence, oldest):
            raise RichTelemetryMergeError(f"{envelope._label} delta skipped the merge cursor")
        first_index = max(minimum_sequence, transmit_from) - transmit_from
    return transmit_from, records, first_index


def _combat_events(
    value: object, *, generation: int, state_epoch: int, observation_tick: int, minimum_sequence: int | None = None
) -> RichCombatEventEnvelope:
    envelope = _event_envelope(
        value,
        "rich.combatEvents",
        COMBAT_EVENT_SCHEMA,
        MAX_COMBAT_EVENTS,
        generation=generation,
        state_epoch=state_epoch,
        observation_tick=observation_tick,
        fields={"rejectedCount": "rejectedCaptureCount"},
    )
    capacity = envelope["capacity"]
    attested = envelope.boolean("hookSetAttested")
    installed = envelope.boolean("hookSetInstalled")
    if installed and not attested:
        raise RichTelemetryMergeError("rich.combatEvents installed hooks are not attested")
    complete = envelope.boolean("complete")
    rejected = envelope.integer("rejectedCaptureCount", minimum=0)
    if not complete or rejected:
        raise RichTelemetryMergeError("rich.combatEvents capture is incomplete or rejected a record")
    capability = envelope.mapping("capability")
    expected_status = "derived" if installed else "unavailable"
    if capability.get("status") != expected_status or capability.get("failClosed") is not True:
        raise RichTelemetryMergeError("rich.combatEvents capability contradicts hook state")
    for field in ("source", "validation", "confidence"):
        if not isinstance(capability.get(field), str) or not capability[field]:
            raise RichTelemetryMergeError(f"rich.combatEvents.capability.{field} is invalid")
    epoch_first, oldest, next_sequence, overflow, gap = _ring_counters(envelope)
    transmit_from, raw_events, first_index = _ring_records(
        envelope, epoch_first, oldest, next_sequence, minimum_sequence
    )
    if not installed and raw_events:
        raise RichTelemetryMergeError("unavailable rich.combatEvents exposed event records")

    fact_names = (
        "target",
        "immediateSource",
        "source",
        "projectile",
        "related",
        "routeSourceBefore",
        "routeTargetBefore",
        "routeSourceAfter",
        "routeTargetAfter",
    )
    amount_names = (
        "requestedAmount",
        "actualAmount",
        "preHp",
        "postHp",
        "preBuiltInShield",
        "postBuiltInShield",
        "preBuffShield",
        "postBuffShield",
    )
    event_fields = {
        "sequence",
        "tick",
        "generation",
        "stateEpoch",
        "kind",
        "hookOffset",
        "callerOffset",
        "causeSequence",
        "pool",
        "terminalReason",
        "lethal",
        "deploymentContext",
        *fact_names,
        *amount_names,
        "destinationBefore",
        "destinationAfter",
    }
    spawn_provenance_field = "spawnProvenance"
    parsed: list[RichCombatEventTelemetry] = []
    deployment_contexts: dict[int, RichCombatDeploymentContext] = {}
    card_play_deployments: set[int] = set()
    for index, raw_value in enumerate(raw_events[first_index:], start=first_index):
        label = f"rich.combatEvents.events[{index}]"
        raw = _record(raw_value, label)
        raw_fields = set(raw)
        if raw_fields != event_fields and raw_fields != event_fields | {spawn_provenance_field}:
            raise RichTelemetryMergeError(f"{label} fields do not match the v1 event schema")
        if spawn_provenance_field in raw and raw.get(spawn_provenance_field) is not None:
            provenance = raw.mapping(spawn_provenance_field)
            if set(provenance) != {"kind", "hookOffset", "sourceDataGlobalId"}:
                raise RichTelemetryMergeError(f"{label}.{spawn_provenance_field} fields are invalid")
        sequence = raw.integer("sequence", minimum=1)
        if sequence != transmit_from + index:
            raise RichTelemetryMergeError("rich.combatEvents sequences are not contiguous")
        event_tick = raw.integer("tick", minimum=-1)
        if observation_tick >= 0 and event_tick > observation_tick:
            raise RichTelemetryMergeError(f"{label} occurs after the observation")
        if raw.integer("generation") != generation:
            raise RichTelemetryMergeError(f"{label} generation mismatch")
        if raw.integer("stateEpoch") != state_epoch:
            raise RichTelemetryMergeError(f"{label} stateEpoch mismatch")
        kind = raw.get("kind")
        if kind not in COMBAT_EVENT_KINDS:
            raise RichTelemetryMergeError(f"{label}.kind is invalid")
        cause = raw.optional_nonnegative("causeSequence", minimum=1)
        if cause is not None and (cause < epoch_first or cause >= sequence):
            raise RichTelemetryMergeError(f"{label}.causeSequence is invalid")
        deployment = _combat_deployment_context(raw.get("deploymentContext"), f"{label}.deploymentContext")
        if deployment is not None:
            if kind not in {"spawn", "projectile_spawn", "card_play"}:
                raise RichTelemetryMergeError(f"{label} attaches deployment provenance to an incompatible event")
            prior_deployment = deployment_contexts.setdefault(deployment.deployment_sequence, deployment)
            if prior_deployment != deployment:
                raise RichTelemetryMergeError("rich.combatEvents deploymentSequence changed identity")
        if kind == "card_play":
            if deployment is None:
                raise RichTelemetryMergeError(f"{label} lacks its required deploymentContext")
            if deployment.deployment_sequence in card_play_deployments:
                raise RichTelemetryMergeError("rich.combatEvents repeats a card_play deploymentSequence")
            card_play_deployments.add(deployment.deployment_sequence)
            if raw.integer("hookOffset", minimum=1) != COMBAT_CONSUME_CARD_HOOK_OFFSET:
                raise RichTelemetryMergeError(f"{label}.hookOffset is invalid for card_play")
        facts = {name: _combat_fact(raw.get(name), f"{label}.{name}") for name in fact_names}
        target_required = kind in {
            "damage",
            "death",
            "heal",
            "spawn",
            "projectile_spawn",
            "despawn",
            "shield_damage",
            "shield_break",
            "projectile_impact",
        }
        projectile_required = kind in {
            "projectile_spawn",
            "projectile_impact",
            "projectile_deflect",
            "projectile_expire",
            "projectile_terminal",
        }
        if target_required and not facts["target"].present:
            raise RichTelemetryMergeError(f"{label} lacks its required target")
        if projectile_required and not facts["projectile"].present:
            raise RichTelemetryMergeError(f"{label} lacks its required projectile")
        if deployment is not None and facts["target"].owner in (0, 1) and facts["target"].owner != deployment.owner:
            raise RichTelemetryMergeError(f"{label} deployment owner contradicts the spawned target")
        amounts = {name: raw.optional_nonnegative(name) for name in amount_names}
        if kind in {"damage", "death", "heal", "shield_damage", "shield_break"} and (
            amounts["requestedAmount"] is None or amounts["actualAmount"] is None or amounts["actualAmount"] <= 0
        ):
            raise RichTelemetryMergeError(f"{label} lacks exact positive amount data")
        parsed.append(
            RichCombatEventTelemetry(
                sequence=sequence,
                tick=event_tick,
                generation=generation,
                state_epoch=state_epoch,
                kind=str(kind),
                hook_offset=raw.integer("hookOffset", minimum=1),
                caller_offset=raw.optional_nonnegative("callerOffset", minimum=1),
                cause_sequence=cause,
                pool=str(raw.get("pool")),
                terminal_reason=str(raw.get("terminalReason")),
                lethal=raw.boolean("lethal"),
                deployment_context=deployment,
                target=facts["target"],
                immediate_source=facts["immediateSource"],
                source=facts["source"],
                projectile=facts["projectile"],
                related=facts["related"],
                route_source_before=facts["routeSourceBefore"],
                route_target_before=facts["routeTargetBefore"],
                route_source_after=facts["routeSourceAfter"],
                route_target_after=facts["routeTargetAfter"],
                requested_amount=amounts["requestedAmount"],
                actual_amount=amounts["actualAmount"],
                pre_hp=amounts["preHp"],
                post_hp=amounts["postHp"],
                pre_builtin_shield=amounts["preBuiltInShield"],
                post_builtin_shield=amounts["postBuiltInShield"],
                pre_buff_shield=amounts["preBuffShield"],
                post_buff_shield=amounts["postBuffShield"],
                destination_before=raw.point("destinationBefore"),
                destination_after=raw.point("destinationAfter"),
            )
        )
    return RichCombatEventEnvelope(
        generation=generation,
        state_epoch=state_epoch,
        observation_tick=observation_tick,
        capacity=capacity,
        hook_set_attested=attested,
        hook_set_installed=installed,
        capability_status=expected_status,
        epoch_first_sequence=epoch_first,
        oldest_retained_sequence=oldest,
        next_sequence=next_sequence,
        overflow_count=overflow,
        sequence_gap_before_oldest=gap,
        rejected_capture_count=rejected,
        complete=complete,
        events=tuple(parsed),
    )


_PHASE_EVENT_HOOKS = {
    PhaseHookKind.ATTACK_START: 0xF23110,
    PhaseHookKind.ATTACK_RELEASE: 0xF5F100,
    PhaseHookKind.EFFECT_APPLY: 0xF5A2D4,
    PhaseHookKind.ATTACK_SCALE: 0xF5B4AC,
    PhaseHookKind.MOVEMENT_SCALE: 0xF5B3C8,
    PhaseHookKind.EFFECTIVE_MOVEMENT_SPEED: 0xF1D004,
    PhaseHookKind.MOVEMENT_DELTA: 0xF68808,
    PhaseHookKind.TARGET_RESET: 0xF5C894,
    PhaseHookKind.CLASSIC_CHARGE_READY: 0xF23110,
    PhaseHookKind.DEPLOY_SCALE: 0xF5B3C8,
}


def _phase_optional_integer(
    value: object, label: str, *, minimum: int | None = None, maximum: int | None = None
) -> int | None:
    if value is None:
        return None
    return _integer(value, label, minimum=minimum, maximum=maximum)


def _phase_optional_tick(value: object, label: str) -> int | None:
    tick = _phase_optional_integer(value, label, minimum=-1)
    return None if tick == -1 else tick


def _phase_object(value: object, label: str) -> RichPhaseObjectTelemetry | None:
    if value is None:
        return None
    raw = _record(value, label)
    required = {
        "schema",
        "attackValidated",
        "movementValidated",
        "buffsValidated",
        "attackSequenceStage",
        "attackTimelineMs",
        "loadRemainingMs",
        "deployRemainingMs",
        "deployPreviousMs",
        "configuredDeployTimeMs",
        "hitSpeedMs",
        "attackDashTimeMs",
        "baseMovementSpeed",
        "chargeSpeedMultiplier",
        "classicChargeProgress",
        "speedPositivePercent",
        "speedNegativeMagnitude",
        "hitSpeedPositivePercent",
        "hitSpeedNegativeMagnitude",
        "attackStepInput",
        "attackStepOutput",
        "attackStepTick",
        "movementStepInput",
        "movementStepOutput",
        "movementStepTick",
        "deployStepInput",
        "deployStepOutput",
        "deployStepTick",
        "effectiveMovementSpeed",
        "effectiveMovementSpeedTick",
        "movementDelta",
        "movementDeltaTick",
    }
    sequence_decay_fields = {
        "attackSequenceProgressRaw",
        "attackSequenceProgressLimit",
        "attackSequenceDecayRemainingMs",
        "attackSequenceDecayDurationMs",
    }
    raw_fields = set(raw)
    if (raw_fields != required and raw_fields != required | sequence_decay_fields) or raw.get(
        "schema"
    ) != PHASE_OBJECT_SCHEMA:
        raise RichTelemetryMergeError(f"{label} fields/schema do not match native-phase-object.v1")

    attack_validated = raw.boolean("attackValidated")
    movement_validated = raw.boolean("movementValidated")
    buffs_validated = raw.boolean("buffsValidated")
    stage = raw.optional_integer("attackSequenceStage", minimum=-1, maximum=255)
    if stage == -1:
        stage = None
    classic_charge = raw.optional_integer("classicChargeProgress", minimum=-1)
    if movement_validated != (classic_charge is not None):
        raise RichTelemetryMergeError(f"{label} movement validation/progress relationship is inconsistent")
    speed_positive = raw.optional_integer("speedPositivePercent", minimum=0)
    speed_negative = raw.optional_integer("speedNegativeMagnitude", minimum=0)
    hit_positive = raw.optional_integer("hitSpeedPositivePercent", minimum=0)
    hit_negative = raw.optional_integer("hitSpeedNegativeMagnitude", minimum=0)
    if buffs_validated != all(
        item is not None for item in (speed_positive, speed_negative, hit_positive, hit_negative)
    ):
        raise RichTelemetryMergeError(f"{label} Buff validation/extrema relationship is inconsistent")

    nonnegative_names = {
        "attack_timeline_ms": "attackTimelineMs",
        "load_remaining_ms": "loadRemainingMs",
        "deploy_remaining_ms": "deployRemainingMs",
        "deploy_previous_ms": "deployPreviousMs",
        "configured_deploy_time_ms": "configuredDeployTimeMs",
        "hit_speed_ms": "hitSpeedMs",
        "attack_dash_time_ms": "attackDashTimeMs",
        "base_movement_speed": "baseMovementSpeed",
        "charge_speed_multiplier": "chargeSpeedMultiplier",
        "attack_step_input": "attackStepInput",
        "attack_step_output": "attackStepOutput",
        "movement_step_input": "movementStepInput",
        "movement_step_output": "movementStepOutput",
        "deploy_step_input": "deployStepInput",
        "deploy_step_output": "deployStepOutput",
        "effective_movement_speed": "effectiveMovementSpeed",
    }
    values = {
        python_name: raw.optional_integer(wire_name, minimum=0) for python_name, wire_name in nonnegative_names.items()
    }
    if attack_validated and any(
        values[name] is None
        for name in (
            "attack_timeline_ms",
            "load_remaining_ms",
            "deploy_remaining_ms",
            "deploy_previous_ms",
            "configured_deploy_time_ms",
            "hit_speed_ms",
            "attack_dash_time_ms",
            "base_movement_speed",
            "charge_speed_multiplier",
        )
    ):
        raise RichTelemetryMergeError(f"{label} validated attack snapshot is incomplete")
    if not attack_validated and any(
        item is not None for item in (stage, values["attack_timeline_ms"], values["load_remaining_ms"])
    ):
        raise RichTelemetryMergeError(f"{label} exposes attack state without validation")

    progress: int | None = None
    progress_limit: int | None = None
    decay_remaining: int | None = None
    decay_duration: int | None = None
    if sequence_decay_fields <= raw_fields:
        progress = raw.optional_integer("attackSequenceProgressRaw", minimum=0, maximum=ATTACK_SEQUENCE_PROGRESS_LIMIT)
        progress_limit = raw.optional_integer("attackSequenceProgressLimit", minimum=1)
        decay_remaining = raw.optional_integer(
            "attackSequenceDecayRemainingMs", minimum=0, maximum=ATTACK_SEQUENCE_DECAY_DURATION_MS
        )
        decay_duration = raw.optional_integer("attackSequenceDecayDurationMs", minimum=1)
        if (
            (progress is None) != (progress_limit is None)
            or (decay_remaining is None) != (decay_duration is None)
            or progress_limit not in {None, ATTACK_SEQUENCE_PROGRESS_LIMIT}
            or decay_duration not in {None, ATTACK_SEQUENCE_DECAY_DURATION_MS}
            or (progress is not None and not attack_validated)
            or (decay_remaining is not None and not attack_validated)
        ):
            raise RichTelemetryMergeError(f"{label} attack sequence decay contract is inconsistent")
        if progress is not None:
            expected_stage = 0 if progress < 4 else 1 if progress < 9 else 2 if progress < 49 else 3
            if stage != expected_stage:
                raise RichTelemetryMergeError(f"{label} attack sequence progress/stage disagree")

    return RichPhaseObjectTelemetry(
        attack_validated=attack_validated,
        movement_validated=movement_validated,
        buffs_validated=buffs_validated,
        attack_sequence_stage=stage,
        classic_charge_progress=classic_charge,
        speed_positive_percent=speed_positive,
        speed_negative_magnitude=speed_negative,
        hit_speed_positive_percent=hit_positive,
        hit_speed_negative_magnitude=hit_negative,
        attack_step_tick=raw.tick("attackStepTick"),
        movement_step_tick=raw.tick("movementStepTick"),
        deploy_step_tick=raw.tick("deployStepTick"),
        effective_movement_speed_tick=raw.tick("effectiveMovementSpeedTick"),
        movement_delta=raw.optional_integer("movementDelta"),
        movement_delta_tick=raw.tick("movementDeltaTick"),
        attack_sequence_progress_raw=progress,
        attack_sequence_progress_limit=progress_limit,
        attack_sequence_decay_remaining_ms=decay_remaining,
        attack_sequence_decay_duration_ms=decay_duration,
        **values,
    )


def _phase_runtime_events(
    value: object, *, generation: int, state_epoch: int, observation_tick: int, minimum_sequence: int | None = None
) -> RichPhaseRuntimeEnvelope:
    envelope = _event_envelope(
        value,
        "rich.phaseRuntime",
        PHASE_RUNTIME_SCHEMA,
        MAX_PHASE_EVENTS,
        generation=generation,
        state_epoch=state_epoch,
        observation_tick=observation_tick,
    )
    capacity = envelope["capacity"]
    attested = envelope.boolean("hookSetAttested")
    installed = envelope.boolean("hookSetInstalled")
    if installed and not attested:
        raise RichTelemetryMergeError("rich.phaseRuntime hooks are installed without attestation")
    capability = envelope.mapping("capability")
    for field in ("source", "validation", "confidence"):
        if not isinstance(capability.get(field), str) or not capability[field]:
            raise RichTelemetryMergeError(f"rich.phaseRuntime.capability.{field} is invalid")
    if capability.get("failClosed") is not True:
        raise RichTelemetryMergeError("rich.phaseRuntime capability is not fail-closed")
    capability_status = capability.get("status")
    if capability_status not in {"derived", "unavailable"}:
        raise RichTelemetryMergeError("rich.phaseRuntime capability status is invalid")
    complete = envelope.boolean("complete")
    rejected = envelope.integer("rejectedCount", minimum=0)
    if complete and (not installed or not attested or capability_status != "derived" or rejected != 0):
        raise RichTelemetryMergeError("rich.phaseRuntime complete marker contradicts capture state")
    if capability_status == "derived" and (not installed or not attested):
        raise RichTelemetryMergeError("rich.phaseRuntime derived capability lacks installed hooks")
    if rejected and complete:
        raise RichTelemetryMergeError("rich.phaseRuntime rejected facts without failing closed")
    try:
        window = PhaseRuntimeWindowV1(
            capacity=capacity,
            epoch_first_sequence=envelope.integer("epochFirstSequence", minimum=1),
            oldest_retained_sequence=envelope.integer("oldestRetainedSequence", minimum=1),
            next_sequence=envelope.integer("nextSequence", minimum=1),
            overflow_count=envelope.integer("overflowCount", minimum=0),
            rejected_count=rejected,
            complete=complete,
        )
    except PhaseRuntimeError as error:
        raise RichTelemetryMergeError(str(error)) from error
    sequence_gap = envelope.boolean("sequenceGapBeforeOldest")
    if sequence_gap != (window.oldest_retained_sequence > window.epoch_first_sequence):
        raise RichTelemetryMergeError("rich.phaseRuntime retained-window gap marker is inconsistent")
    transmit_from, raw_events, first_index = _ring_records(
        envelope, window.epoch_first_sequence, window.oldest_retained_sequence, window.next_sequence, minimum_sequence
    )
    if capability_status == "unavailable" and raw_events:
        raise RichTelemetryMergeError("unavailable rich.phaseRuntime exposed event records")

    event_fields = {
        "sequence",
        "tick",
        "kind",
        "hookOffset",
        "callerOffset",
        "entity",
        "targetBefore",
        "targetAfter",
        "buffGlobalId",
        "buffRemainingMs",
        "speedMultiplier",
        "hitSpeedMultiplier",
        "inputStep",
        "outputStep",
        "timelineBefore",
        "timelineAfter",
        "classicChargeBefore",
        "classicChargeAfter",
        "success",
    }
    parsed: list[PhaseHookEvent] = []
    for index, value_item in enumerate(raw_events[first_index:], start=first_index):
        label = f"rich.phaseRuntime.events[{index}]"
        raw = _record(value_item, label)
        if set(raw) != event_fields:
            raise RichTelemetryMergeError(f"{label} fields do not match the v1 event schema")
        sequence = raw.integer("sequence", minimum=1)
        if sequence != transmit_from + index:
            raise RichTelemetryMergeError("rich.phaseRuntime event sequences are not contiguous")
        event_tick = raw.integer("tick", minimum=0)
        if event_tick > observation_tick:
            raise RichTelemetryMergeError(f"{label} occurs after the observation")
        try:
            kind = PhaseHookKind(raw.get("kind"))
        except ValueError as error:
            raise RichTelemetryMergeError(f"{label}.kind is invalid") from error
        hook_offset = raw.integer("hookOffset", minimum=1)
        if hook_offset != _PHASE_EVENT_HOOKS[kind]:
            raise RichTelemetryMergeError(f"{label}.hookOffset does not match its exact hook")
        entity = _combat_fact(raw.get("entity"), f"{label}.entity")
        before = _combat_fact(raw.get("targetBefore"), f"{label}.targetBefore")
        after = _combat_fact(raw.get("targetAfter"), f"{label}.targetAfter")
        if not entity.present or entity.entity_key is None:
            raise RichTelemetryMergeError(f"{label} lacks its emitting entity")
        try:
            parsed.append(
                PhaseHookEvent(
                    sequence=sequence,
                    tick=event_tick,
                    kind=kind,
                    entity_key=entity.entity_key,
                    target_key=(after.entity_key if after.present else before.entity_key if before.present else None),
                    hook_offset=hook_offset,
                    caller_offset=raw.optional_integer("callerOffset", minimum=1),
                    success=raw.boolean("success"),
                    buff_global_id=raw.optional_integer("buffGlobalId", minimum=1, maximum=4294967295),
                    buff_remaining_ms=raw.optional_integer("buffRemainingMs", minimum=-1),
                    speed_multiplier=raw.optional_integer("speedMultiplier"),
                    hit_speed_multiplier=raw.optional_integer("hitSpeedMultiplier"),
                    input_step=raw.optional_integer("inputStep"),
                    output_step=raw.optional_integer("outputStep", minimum=0),
                    timeline_before=raw.optional_integer("timelineBefore", minimum=0),
                    timeline_after=raw.optional_integer("timelineAfter", minimum=0),
                    classic_charge_before=raw.optional_integer("classicChargeBefore", minimum=-1),
                    classic_charge_after=raw.optional_integer("classicChargeAfter", minimum=-1),
                    target_present_before=before.present,
                    target_present_after=after.present,
                )
            )
        except PhaseRuntimeError as error:
            raise RichTelemetryMergeError(str(error)) from error
    return RichPhaseRuntimeEnvelope(
        generation=generation,
        state_epoch=state_epoch,
        observation_tick=observation_tick,
        hook_set_attested=attested,
        hook_set_installed=installed,
        capability_status=str(capability_status),
        window=window,
        sequence_gap_before_oldest=sequence_gap,
        events=tuple(parsed),
    )


def _visibility_runtime_events(
    value: object, *, generation: int, state_epoch: int, observation_tick: int, minimum_sequence: int | None = None
) -> RichVisibilityRuntimeEnvelope:
    envelope = _event_envelope(
        value,
        "rich.visibilityRuntime",
        VISIBILITY_RUNTIME_SCHEMA,
        MAX_VISIBILITY_EVENTS,
        generation=generation,
        state_epoch=state_epoch,
        observation_tick=observation_tick,
        fields={
            "hookSetInstalled": "transitionHookSetInstalled",
            "+contextualGateHookSetInstalled": None,
            "+contextualGateCapability": None,
            "+ownerRelativeVisibility": None,
        },
    )
    capacity = envelope["capacity"]
    attested = envelope.boolean("hookSetAttested")
    transition_installed = envelope.boolean("transitionHookSetInstalled")
    contextual_installed = envelope.boolean("contextualGateHookSetInstalled")
    if transition_installed and not attested:
        raise RichTelemetryMergeError("rich.visibilityRuntime transition hooks lack attestation")
    if contextual_installed:
        raise RichTelemetryMergeError("rich.visibilityRuntime contextual gate is unsupported by v1")

    capability = envelope.mapping("capability")
    contextual_capability = envelope.mapping("contextualGateCapability")
    for name, record in (("capability", capability), ("contextualGateCapability", contextual_capability)):
        for field in ("source", "validation", "confidence"):
            if not isinstance(record.get(field), str) or not record[field]:
                raise RichTelemetryMergeError(f"rich.visibilityRuntime.{name}.{field} is invalid")
        if record.get("failClosed") is not True:
            raise RichTelemetryMergeError(f"rich.visibilityRuntime.{name} is not fail-closed")
    capability_status = capability.get("status")
    if capability_status not in {"derived", "unavailable"}:
        raise RichTelemetryMergeError("rich.visibilityRuntime capability status is invalid")
    if (capability_status == "derived") != (transition_installed and attested):
        raise RichTelemetryMergeError("rich.visibilityRuntime capability/install state is inconsistent")
    contextual_status = contextual_capability.get("status")
    if contextual_status != "unavailable":
        raise RichTelemetryMergeError("rich.visibilityRuntime contextual gate must remain unavailable")

    visibility = envelope.record("ownerRelativeVisibility")
    if set(visibility) != {"publicByOwner", "targetableByOwner"}:
        raise RichTelemetryMergeError("rich.visibilityRuntime owner-relative visibility is invalid")
    availability: dict[str, bool] = {}
    for name in ("publicByOwner", "targetableByOwner"):
        record = visibility.mapping(name)
        if (
            set(record) != {"status", "reason"}
            or record.get("status") != "unavailable"
            or record.get("reason") != "no-exact-owner-conditioned-native-producer"
        ):
            raise RichTelemetryMergeError("rich.visibilityRuntime owner-relative claim is invalid")
        availability[name] = False

    epoch_first, oldest, next_sequence, overflow, sequence_gap = _ring_counters(envelope)
    rejected = envelope.integer("rejectedCount", minimum=0)
    complete = envelope.boolean("complete")
    if complete and (not transition_installed or not attested or capability_status != "derived" or rejected != 0):
        raise RichTelemetryMergeError("rich.visibilityRuntime complete marker contradicts capture state")
    if rejected and complete:
        raise RichTelemetryMergeError("rich.visibilityRuntime rejected facts without failing closed")
    transmit_from, raw_events, first_index = _ring_records(
        envelope, epoch_first, oldest, next_sequence, minimum_sequence
    )
    if capability_status == "unavailable" and raw_events:
        raise RichTelemetryMergeError("unavailable rich.visibilityRuntime exposed events")
    event_fields = {
        "sequence",
        "tick",
        "kind",
        "hookOffset",
        "callerOffset",
        "subject",
        "buffGlobalId",
        "invisibleCountBefore",
        "invisibleCountAfter",
        "scope",
        "completeContext",
    }
    parsed: list[RichVisibilityRuntimeEvent] = []
    for index, value_item in enumerate(raw_events[first_index:], start=first_index):
        label = f"rich.visibilityRuntime.events[{index}]"
        raw = _record(value_item, label)
        if set(raw) != event_fields:
            raise RichTelemetryMergeError(f"{label} fields do not match the v1 event schema")
        sequence = raw.integer("sequence", minimum=1)
        if sequence != transmit_from + index:
            raise RichTelemetryMergeError("rich.visibilityRuntime event sequences are not contiguous")
        tick = raw.integer("tick", minimum=0)
        if tick > observation_tick:
            raise RichTelemetryMergeError(f"{label} occurs after the observation")
        try:
            kind = VisibilityTransitionKind(raw.get("kind"))
        except ValueError as error:
            raise RichTelemetryMergeError(f"{label}.kind is invalid") from error
        hook_offset = raw.integer("hookOffset", minimum=1)
        caller_offset = raw.integer("callerOffset", minimum=1)
        if hook_offset != _VISIBILITY_EVENT_HOOKS[kind] or caller_offset != _VISIBILITY_EVENT_CALLERS[kind]:
            raise RichTelemetryMergeError(f"{label} hook/caller identity is invalid")
        subject = _combat_fact(raw.get("subject"), f"{label}.subject")
        if (
            not subject.present
            or subject.entity_key is None
            or not subject.visibility_validated
            or subject.invisible_count is None
        ):
            raise RichTelemetryMergeError(f"{label} lacks its validated visibility subject")
        buff_global_id = raw.integer("buffGlobalId", minimum=1, maximum=4294967295)
        before = raw.integer("invisibleCountBefore", minimum=0, maximum=MAX_ACTIVE_EFFECTS)
        after = raw.integer("invisibleCountAfter", minimum=0, maximum=MAX_ACTIVE_EFFECTS)
        expected_edge = (0, 1) if kind is VisibilityTransitionKind.BECAME_INVISIBLE else (1, 0)
        if (before, after) != expected_edge:
            raise RichTelemetryMergeError(f"{label} does not cross the exact visibility boundary")
        if subject.invisible_count != after:
            raise RichTelemetryMergeError(f"{label} subject count disagrees with the transition")
        if raw.get("scope") != "native_invisibility_phase":
            raise RichTelemetryMergeError(f"{label}.scope is invalid")
        complete_context = raw.boolean("completeContext")
        if not complete_context:
            raise RichTelemetryMergeError(f"{label} lacks complete context")
        parsed.append(
            RichVisibilityRuntimeEvent(
                sequence=sequence,
                tick=tick,
                kind=kind,
                hook_offset=hook_offset,
                caller_offset=caller_offset,
                subject=subject,
                buff_global_id=buff_global_id,
                invisible_count_before=before,
                invisible_count_after=after,
                scope="native_invisibility_phase",
                complete_context=True,
            )
        )
    return RichVisibilityRuntimeEnvelope(
        generation=generation,
        state_epoch=state_epoch,
        observation_tick=observation_tick,
        capacity=capacity,
        hook_set_attested=attested,
        transition_hook_set_installed=transition_installed,
        contextual_gate_hook_set_installed=contextual_installed,
        capability_status=str(capability_status),
        contextual_gate_capability_status=str(contextual_status),
        epoch_first_sequence=epoch_first,
        oldest_retained_sequence=oldest,
        next_sequence=next_sequence,
        overflow_count=overflow,
        rejected_count=rejected,
        sequence_gap_before_oldest=sequence_gap,
        complete=complete,
        public_by_owner_available=availability["publicByOwner"],
        targetable_by_owner_available=availability["targetableByOwner"],
        events=tuple(parsed),
    )


def _remaining_runtime_events(
    value: object, *, generation: int, state_epoch: int, observation_tick: int, minimum_sequence: int | None = None
) -> RichRemainingRuntimeEnvelope:
    envelope = _event_envelope(
        value,
        "rich.remainingRuntime",
        REMAINING_RUNTIME_SCHEMA,
        MAX_REMAINING_EVENTS,
        generation=generation,
        state_epoch=state_epoch,
        observation_tick=observation_tick,
        fields={"+ownerRelativeVisibility": None},
    )
    capacity = envelope["capacity"]
    attested = envelope.boolean("hookSetAttested")
    installed = envelope.boolean("hookSetInstalled")
    if installed and not attested:
        raise RichTelemetryMergeError("rich.remainingRuntime installed hooks are not attested")
    capability = envelope.mapping("capability")
    for field in ("source", "validation", "confidence"):
        if not isinstance(capability.get(field), str) or not capability[field]:
            raise RichTelemetryMergeError(f"rich.remainingRuntime.capability.{field} is invalid")
    if capability.get("failClosed") is not True:
        raise RichTelemetryMergeError("rich.remainingRuntime capability is not fail-closed")
    capability_status = capability.get("status")
    if capability_status not in {"derived", "unavailable"}:
        raise RichTelemetryMergeError("rich.remainingRuntime capability status is invalid")
    if capability_status == "derived" and (not installed or not attested):
        raise RichTelemetryMergeError("rich.remainingRuntime derived capability lacks installed hooks")

    visibility = envelope.record("ownerRelativeVisibility")
    if set(visibility) != {"publicByOwner", "targetableByOwner"}:
        raise RichTelemetryMergeError("rich.remainingRuntime owner-relative visibility is invalid")
    availability: dict[str, bool] = {}
    for name in ("publicByOwner", "targetableByOwner"):
        record = visibility.mapping(name)
        if (
            set(record) != {"status", "reason"}
            or record.get("status") != "unavailable"
            or record.get("reason") != "no-exact-owner-conditioned-native-producer"
        ):
            raise RichTelemetryMergeError(f"rich.remainingRuntime must fail closed for {name}")
        availability[name] = False

    epoch_first, oldest, next_sequence, overflow, sequence_gap = _ring_counters(envelope)
    rejected = envelope.integer("rejectedCount", minimum=0)
    complete = envelope.boolean("complete")
    if complete and (not installed or not attested or capability_status != "derived" or rejected != 0):
        raise RichTelemetryMergeError("rich.remainingRuntime complete marker contradicts capture state")
    transmit_from, raw_events, first_index = _ring_records(
        envelope, epoch_first, oldest, next_sequence, minimum_sequence
    )
    if capability_status == "unavailable" and raw_events:
        raise RichTelemetryMergeError("unavailable rich.remainingRuntime exposed event records")

    event_fields = {
        "sequence",
        "tick",
        "kind",
        "hookOffset",
        "callerOffset",
        "entity",
        "source",
        "target",
        "targetBefore",
        "targetAfter",
        "objectKind",
        "runtimeVtableOffset",
        "dataBeforeGlobalId",
        "dataAfterGlobalId",
        "expectedCharacterDataGlobalId",
        "expectedProjectileDataGlobalId",
        "configuredDataGlobalId",
        "resourcePreFixed",
        "resourcePostFixed",
        "resourceActualDeltaFixed",
        "amountArgument",
        "configuredAmountArgument",
        "resourceOwner",
        "areaRemainingLifeMs",
        "resourceCause",
        "transformKind",
        "result",
        "option",
        "committed",
        "resetTarget",
        "completeContext",
    }
    resource_edges = {
        "periodic": (0xF3BA4C, 0xF1ACD4, 10_000),
        "death_owner": (0xF3BA4C, 0xF63718, 10_000),
        "death_opponent": (0xF3B3D4, 0xF63834, 1),
    }
    parsed: list[RichRemainingRuntimeEvent] = []
    for index, raw_value in enumerate(raw_events[first_index:], start=first_index):
        label = f"rich.remainingRuntime.events[{index}]"
        raw = _record(raw_value, label)
        if set(raw) != event_fields:
            raise RichTelemetryMergeError(f"{label} fields do not match the v1 event schema")
        sequence = raw.integer("sequence", minimum=1)
        if sequence != transmit_from + index:
            raise RichTelemetryMergeError("rich.remainingRuntime sequences are not contiguous")
        tick = raw.integer("tick", minimum=0)
        if observation_tick >= 0 and tick > observation_tick:
            raise RichTelemetryMergeError(f"{label} occurs after the observation")
        kind = raw.get("kind")
        if kind not in _REMAINING_EVENT_HOOKS:
            raise RichTelemetryMergeError(f"{label}.kind is invalid")
        hook_offset = raw.integer("hookOffset", minimum=1)
        if hook_offset not in _REMAINING_EVENT_HOOKS[str(kind)]:
            raise RichTelemetryMergeError(f"{label}.hookOffset does not match its exact producer")
        caller_offset = raw.optional_nonnegative("callerOffset", minimum=1)
        facts = {
            name: _combat_fact(raw.get(wire), f"{label}.{wire}")
            for name, wire in (
                ("entity", "entity"),
                ("source", "source"),
                ("target", "target"),
                ("target_before", "targetBefore"),
                ("target_after", "targetAfter"),
            )
        }
        optional_ids = {
            name: raw.optional_nonnegative(wire, minimum=1)
            for name, wire in (
                ("data_before_global_id", "dataBeforeGlobalId"),
                ("data_after_global_id", "dataAfterGlobalId"),
                ("expected_character_data_global_id", "expectedCharacterDataGlobalId"),
                ("expected_projectile_data_global_id", "expectedProjectileDataGlobalId"),
                ("configured_data_global_id", "configuredDataGlobalId"),
            )
        }
        optional_values = {
            name: raw.optional_nonnegative(wire)
            for name, wire in (
                ("resource_pre_fixed", "resourcePreFixed"),
                ("resource_post_fixed", "resourcePostFixed"),
                ("resource_actual_delta_fixed", "resourceActualDeltaFixed"),
                ("amount_argument", "amountArgument"),
                ("configured_amount_argument", "configuredAmountArgument"),
                ("resource_owner", "resourceOwner"),
                ("area_remaining_life_ms", "areaRemainingLifeMs"),
            )
        }
        object_kind = raw.optional_nonnegative("objectKind", minimum=0)
        runtime_vtable_offset = raw.optional_nonnegative("runtimeVtableOffset", minimum=1)
        resource_cause = raw.get("resourceCause")
        transform_kind = raw.get("transformKind")
        if resource_cause not in {"none", *resource_edges}:
            raise RichTelemetryMergeError(f"{label}.resourceCause is invalid")
        if transform_kind not in {"none", "character", "projectile"}:
            raise RichTelemetryMergeError(f"{label}.transformKind is invalid")
        result = raw.boolean("result")
        option = raw.boolean("option")
        committed = raw.boolean("committed")
        reset_target = raw.boolean("resetTarget")
        complete_context = raw.boolean("completeContext")
        if not complete_context:
            raise RichTelemetryMergeError(f"{label} exposes an incomplete context")

        entity = facts["entity"]
        source = facts["source"]
        target = facts["target"]
        before = facts["target_before"]
        after = facts["target_after"]
        if kind == "resource_delta":
            edge = resource_edges.get(str(resource_cause))
            pre = optional_values["resource_pre_fixed"]
            post = optional_values["resource_post_fixed"]
            actual = optional_values["resource_actual_delta_fixed"]
            amount = optional_values["amount_argument"]
            configured = optional_values["configured_amount_argument"]
            if (
                edge is None
                or not source.present
                or hook_offset != edge[0]
                or caller_offset != edge[1]
                or pre is None
                or post is None
                or actual is None
                or amount is None
                or configured != amount
                or amount <= 0
                or pre < 0
                or post <= pre
                or actual != post - pre
                or actual > amount * edge[2]
                or optional_values["resource_owner"] not in {0, 1}
            ):
                raise RichTelemetryMergeError(f"{label} resource edge is invalid")
        elif resource_cause != "none":
            raise RichTelemetryMergeError(f"{label} has a stray resource cause")
        if kind == "transform":
            old_id = optional_ids["data_before_global_id"]
            new_id = optional_ids["data_after_global_id"]
            character_id = optional_ids["expected_character_data_global_id"]
            projectile_id = optional_ids["expected_projectile_data_global_id"]
            character_match = new_id is not None and new_id == character_id
            projectile_match = new_id is not None and new_id == projectile_id
            if (
                not entity.present
                or old_id is None
                or new_id is None
                or old_id == new_id
                or character_match == projectile_match
                or transform_kind != ("character" if character_match else "projectile")
                or optional_ids["configured_data_global_id"] != new_id
            ):
                raise RichTelemetryMergeError(f"{label} transform edge is invalid")
        elif transform_kind != "none":
            raise RichTelemetryMergeError(f"{label} has a stray transform kind")
        if kind in {"area_create", "area_expire"} and (
            not entity.present
            or object_kind != 3
            or runtime_vtable_offset != 0x189C2E8
            or (kind == "area_create" and not committed)
        ):
            raise RichTelemetryMergeError(f"{label} area lifecycle identity is invalid")
        if kind.startswith("area_relation_") and (
            not entity.present or not target.present or object_kind != 3 or runtime_vtable_offset != 0x189C2E8
        ):
            raise RichTelemetryMergeError(f"{label} area relation identity is invalid")
        if kind == "area_action_spawn":
            producer_identity_valid = (
                object_kind == 3 and source.object_kind == 3 and runtime_vtable_offset == 0x189C2E8
                if hook_offset == 0xF143A8
                else object_kind == source.object_kind and runtime_vtable_offset is not None
            )
            if (
                not entity.present
                or not source.present
                or not target.present
                or entity.native_object_id != target.native_object_id
                or source.native_object_id == target.native_object_id
                or not producer_identity_valid
                or caller_offset is not None
                or not committed
                or optional_ids["configured_data_global_id"] is None
                or optional_ids["data_after_global_id"] is None
            ):
                raise RichTelemetryMergeError(f"{label} exact producer-to-child edge is invalid")
        if kind.startswith("tower_aggro_"):
            expected_presence = {
                "tower_aggro_acquire": (False, True),
                "tower_aggro_change": (True, True),
                "tower_aggro_lose": (True, False),
            }[str(kind)]
            if not entity.present or (before.present, after.present) != expected_presence:
                raise RichTelemetryMergeError(f"{label} tower aggro edge is invalid")
        if kind == "tower_activate" and (not entity.present or caller_offset != 0xF58820 or not result):
            raise RichTelemetryMergeError(f"{label} tower activation edge is invalid")
        if kind == "spawn_attach" and (
            not entity.present
            or not source.present
            or not target.present
            or caller_offset != 0xF18FF4
            or not committed
            or optional_ids["configured_data_global_id"] != optional_ids["data_after_global_id"]
            or source.native_object_id == target.native_object_id
        ):
            raise RichTelemetryMergeError(f"{label} SpawnAttach edge is invalid")
        if kind == "area_damage_eligibility" and not entity.present:
            raise RichTelemetryMergeError(f"{label} area eligibility lacks its subject")

        parsed.append(
            RichRemainingRuntimeEvent(
                sequence=sequence,
                tick=tick,
                kind=str(kind),
                hook_offset=hook_offset,
                caller_offset=caller_offset,
                entity=entity,
                source=source,
                target=target,
                target_before=before,
                target_after=after,
                object_kind=object_kind,
                runtime_vtable_offset=runtime_vtable_offset,
                resource_cause=str(resource_cause),
                transform_kind=str(transform_kind),
                result=result,
                option=option,
                committed=committed,
                reset_target=reset_target,
                complete_context=complete_context,
                **optional_ids,
                **optional_values,
            )
        )
    return RichRemainingRuntimeEnvelope(
        generation=generation,
        state_epoch=state_epoch,
        observation_tick=observation_tick,
        capacity=capacity,
        hook_set_attested=attested,
        hook_set_installed=installed,
        capability_status=str(capability_status),
        epoch_first_sequence=epoch_first,
        oldest_retained_sequence=oldest,
        next_sequence=next_sequence,
        overflow_count=overflow,
        rejected_count=rejected,
        sequence_gap_before_oldest=sequence_gap,
        complete=complete,
        public_by_owner_available=availability["publicByOwner"],
        targetable_by_owner_available=availability["targetableByOwner"],
        events=tuple(parsed),
    )


def _active_effects(value: object, label: str) -> tuple[RichActiveEffect, ...] | None:
    if value is None:
        return None
    raw_effects = _items(value, label)
    effects: list[RichActiveEffect] = []
    for index, raw_effect in enumerate(raw_effects):
        effect_label = f"{label}[{index}]"
        effect = _record(raw_effect, effect_label)
        required = {"buffGlobalId", "name", "remainingMs", "sourceEntityKey", "sourceEntityValidated"}
        if set(effect) != required:
            raise RichTelemetryMergeError(f"{effect_label} fields do not match the v2 active-effect schema")
        global_id = effect.integer("buffGlobalId", minimum=1, maximum=4294967295)
        name = effect.get("name")
        if not isinstance(name, str):
            raise RichTelemetryMergeError(f"{effect_label}.name must be a string")
        encoded_name = name.encode("utf-8")
        if not 1 <= len(encoded_name) <= 128:
            raise RichTelemetryMergeError(f"{effect_label}.name must contain 1..128 UTF-8 bytes")
        remaining_ms = effect.integer("remainingMs", minimum=-1)
        source_raw = effect.get("sourceEntityKey")
        source_validated = effect.boolean("sourceEntityValidated")
        source_key = None if source_raw is None else _entity_key(source_raw, f"{effect_label}.sourceEntityKey")
        if source_key is not None and not source_validated:
            raise RichTelemetryMergeError(f"{effect_label} exposes an unvalidated source entity")
        effects.append(
            RichActiveEffect(
                buff_global_id=global_id,
                name=name,
                remaining_ms=remaining_ms,
                source_entity_key=source_key,
                source_entity_validated=source_validated,
            )
        )
    return tuple(effects)


def _projectile(value: object, label: str) -> RichProjectileTelemetry | None:
    if value is None:
        return None
    raw = _record(value, label)
    required = {
        "projectileDataGlobalId",
        "sourceEntityKey",
        "sourceEntityValidated",
        "targetEntityKey",
        "targetEntityValidated",
        "homingTargetEntityKey",
        "homingTargetEntityValidated",
        "destinationX",
        "destinationY",
        "terminal",
        "nativePhase",
        "dragStage",
    }
    if set(raw) != required:
        raise RichTelemetryMergeError(f"{label} fields do not match the v2 projectile schema")
    source_validated = raw.boolean("sourceEntityValidated")
    target_validated = raw.boolean("targetEntityValidated")
    homing_target_validated = raw.boolean("homingTargetEntityValidated")
    source_raw = raw.get("sourceEntityKey")
    target_raw = raw.get("targetEntityKey")
    homing_target_raw = raw.get("homingTargetEntityKey")
    source_key = None if source_raw is None else _entity_key(source_raw, f"{label}.sourceEntityKey")
    target_key = None if target_raw is None else _entity_key(target_raw, f"{label}.targetEntityKey")
    homing_target_key = (
        None if homing_target_raw is None else _entity_key(homing_target_raw, f"{label}.homingTargetEntityKey")
    )
    if source_key is not None and not source_validated:
        raise RichTelemetryMergeError(f"{label} exposes an unvalidated source")
    if target_key is not None and not target_validated:
        raise RichTelemetryMergeError(f"{label} exposes an unvalidated target")
    if homing_target_key is not None and not homing_target_validated:
        raise RichTelemetryMergeError(f"{label} exposes an unvalidated homing target")
    terminal = raw.boolean("terminal")
    expected_phase = "terminal_or_finished_processing" if terminal else "in_flight"
    if raw.get("nativePhase") != expected_phase:
        raise RichTelemetryMergeError(f"{label}.nativePhase does not match the terminal bit")
    drag_stage = raw.get("dragStage")
    if drag_stage not in {None, "outbound", "drag_back_active"}:
        raise RichTelemetryMergeError(f"{label}.dragStage is invalid")
    return RichProjectileTelemetry(
        projectile_data_global_id=raw.integer("projectileDataGlobalId", minimum=1, maximum=4294967295),
        source_entity_key=source_key,
        source_entity_validated=source_validated,
        target_entity_key=target_key,
        target_entity_validated=target_validated,
        homing_target_entity_key=homing_target_key,
        homing_target_entity_validated=homing_target_validated,
        destination=(raw.integer("destinationX"), raw.integer("destinationY")),
        terminal=terminal,
        native_phase=expected_phase,
        drag_stage=drag_stage,
    )


def _entity_resource_runtime(value: object, label: str) -> RichEntityResourceTelemetry | None:
    if value is None:
        return None
    raw = _record(value, label)
    required = {"kind", "currentRaw", "capacityRaw", "baseRaw", "limitRaw", "normalized", "status"}
    if set(raw) != required:
        raise RichTelemetryMergeError(f"{label} fields do not match the entity resource schema")
    if raw.get("kind") != "extra_spawn_accumulator":
        raise RichTelemetryMergeError(f"{label}.kind is invalid")
    if raw.get("status") != "authoritative":
        raise RichTelemetryMergeError(f"{label}.status must be authoritative")
    current = raw.integer("currentRaw", minimum=0)
    capacity = raw.integer("capacityRaw", minimum=1)
    base = raw.integer("baseRaw", minimum=1)
    limit = raw.integer("limitRaw", minimum=2)
    normalized_raw = raw.get("normalized")
    if isinstance(normalized_raw, bool) or not isinstance(normalized_raw, (int, float)):
        raise RichTelemetryMergeError(f"{label}.normalized must be numeric")
    normalized = float(normalized_raw)
    if (
        limit - base != capacity
        or current > capacity
        or not math.isfinite(normalized)
        or abs(normalized - current / capacity) > 1e-6
    ):
        raise RichTelemetryMergeError(f"{label} values are inconsistent")
    return RichEntityResourceTelemetry(
        kind="extra_spawn_accumulator",
        current_raw=current,
        capacity_raw=capacity,
        base_raw=base,
        limit_raw=limit,
        normalized=normalized,
    )


def _periodic_attack_modifier_runtime(value: object, label: str) -> RichPeriodicAttackModifierTelemetry | None:
    if value is None:
        return None
    raw = _record(value, label)
    required = {
        "schema",
        "actionDataGlobalId",
        "phase",
        "periodAttacks",
        "completedAttacks",
        "addedDamageRaw",
        "lingerDurationMs",
        "lingerRemainingMs",
        "sourceNativeObjectId",
        "sourceEntityKey",
        "sourceResolved",
        "status",
    }
    if set(raw) != required:
        raise RichTelemetryMergeError(f"{label} fields do not match the periodic attack modifier schema")
    if raw.get("schema") != "native-periodic-attack-modifier-runtime.v1":
        raise RichTelemetryMergeError(f"{label}.schema is invalid")
    if raw.get("status") != "authoritative":
        raise RichTelemetryMergeError(f"{label}.status must be authoritative")
    raw.integer("actionDataGlobalId", minimum=1, maximum=4294967295)
    period = raw.integer("periodAttacks", minimum=1)
    completed = raw.integer("completedAttacks", minimum=0)
    if completed >= period:
        raise RichTelemetryMergeError(f"{label}.completedAttacks is outside period")
    added_damage = raw.integer("addedDamageRaw", minimum=0)
    linger_duration = raw.integer("lingerDurationMs", minimum=1)
    linger_remaining = raw.integer("lingerRemainingMs", minimum=0, maximum=linger_duration)
    phase = raw.get("phase")
    if phase not in {"source_alive", "source_death_linger"}:
        raise RichTelemetryMergeError(f"{label}.phase is invalid")
    if (phase == "source_alive") != (linger_remaining == 0):
        raise RichTelemetryMergeError(f"{label} phase/linger timer is inconsistent")
    source_native_object_id = raw.integer("sourceNativeObjectId", minimum=1, maximum=4294967295)
    source_resolved = raw.boolean("sourceResolved")
    raw_source_key = raw.get("sourceEntityKey")
    if source_resolved != (raw_source_key is not None):
        raise RichTelemetryMergeError(f"{label} source resolution fields disagree")
    source_entity_key = _entity_key(raw_source_key, f"{label}.sourceEntityKey") if source_resolved else None
    if phase == "source_alive" and not source_resolved:
        raise RichTelemetryMergeError(f"{label} live source must resolve")
    return RichPeriodicAttackModifierTelemetry(
        phase=str(phase),
        period_attacks=period,
        completed_attacks=completed,
        added_damage_raw=added_damage,
        linger_duration_ms=linger_duration,
        linger_remaining_ms=linger_remaining,
        source_native_object_id=source_native_object_id,
        source_entity_key=source_entity_key,
        source_resolved=source_resolved,
    )


def _capture_runtime(value: object, label: str) -> RichCaptureRuntimeTelemetry | None:
    if value is None:
        return None
    raw = _record(value, label)
    required = {
        "schema",
        "actionDataGlobalId",
        "phase",
        "dragDelayMs",
        "grabPauseMs",
        "captureDragTimeMs",
        "configuredCooldownMs",
        "cooldownRemainingMs",
        "hitFrequencyMs",
        "hitAccumulatorMs",
        "completionResultCurrentUpdate",
        "firstCaptureHandled",
        "targets",
        "status",
    }
    if set(raw) != required:
        raise RichTelemetryMergeError(f"{label} fields do not match the capture runtime schema")
    if raw.get("schema") != "native-capture-runtime.v1":
        raise RichTelemetryMergeError(f"{label}.schema is invalid")
    if raw.get("status") != "authoritative":
        raise RichTelemetryMergeError(f"{label}.status must be authoritative")
    raw.integer("actionDataGlobalId", minimum=1, maximum=4294967295)
    drag_delay = raw.integer("dragDelayMs", minimum=0)
    grab_pause = raw.integer("grabPauseMs", minimum=0)
    capture_drag_time = raw.integer("captureDragTimeMs", minimum=0)
    configured_cooldown = raw.integer("configuredCooldownMs", minimum=0)
    cooldown_remaining = raw.integer("cooldownRemainingMs", minimum=0)
    hit_frequency = raw.integer("hitFrequencyMs", minimum=1)
    hit_accumulator = raw.integer("hitAccumulatorMs", minimum=0)
    if cooldown_remaining > configured_cooldown:
        raise RichTelemetryMergeError(f"{label} timers are inconsistent")
    raw.boolean("completionResultCurrentUpdate")
    raw.boolean("firstCaptureHandled")
    raw_targets = raw.get("targets")
    if not isinstance(raw_targets, list) or len(raw_targets) > 160:
        raise RichTelemetryMergeError(f"{label}.targets is invalid")
    phase = raw.get("phase")
    expected_phase = "active" if raw_targets else "release_cooldown" if cooldown_remaining > 0 else "idle"
    if phase != expected_phase:
        raise RichTelemetryMergeError(f"{label}.phase is inconsistent")

    timed_phases = {"acquired_delay", "grab_pause", "dragging"}
    untimed_phases = {"contained", "release_pending"}
    seen_native_ids: set[int] = set()
    targets: list[RichCaptureTargetTelemetry] = []
    for index, value in enumerate(raw_targets):
        target_label = f"{label}.targets[{index}]"
        target = _record(value, target_label)
        if set(target) != {
            "targetNativeObjectId",
            "targetEntityKey",
            "targetResolved",
            "phase",
            "elapsedMs",
            "phaseBudgetRemainingMs",
        }:
            raise RichTelemetryMergeError(f"{target_label} fields do not match the capture target schema")
        native_id = target.integer("targetNativeObjectId", minimum=1, maximum=4294967295)
        if native_id in seen_native_ids:
            raise RichTelemetryMergeError(f"{label}.targets contains a duplicate native identity")
        seen_native_ids.add(native_id)
        resolved = target.boolean("targetResolved")
        raw_key = target.get("targetEntityKey")
        if resolved != (raw_key is not None):
            raise RichTelemetryMergeError(f"{target_label} resolution fields disagree")
        target_key = _entity_key(raw_key, f"{target_label}.targetEntityKey") if resolved else None
        target_phase = target.get("phase")
        if target_phase not in timed_phases | untimed_phases:
            raise RichTelemetryMergeError(f"{target_label}.phase is invalid")
        if (target_phase == "release_pending") != (not resolved):
            raise RichTelemetryMergeError(f"{target_label} phase/resolution fields disagree")
        elapsed = target.integer("elapsedMs", minimum=0)
        remaining_raw = target.get("phaseBudgetRemainingMs")
        remaining: int | None = None
        if target_phase in timed_phases:
            remaining = _integer(remaining_raw, f"{target_label}.phaseBudgetRemainingMs", minimum=0)
            phase_boundary = {
                "acquired_delay": drag_delay,
                "grab_pause": drag_delay + grab_pause,
                "dragging": drag_delay + grab_pause + capture_drag_time,
            }[target_phase]
            expected_remaining = max(phase_boundary - elapsed, 0)
            if remaining != expected_remaining:
                raise RichTelemetryMergeError(f"{target_label} phase timer is inconsistent")
        elif remaining_raw is not None:
            raise RichTelemetryMergeError(f"{target_label}.phaseBudgetRemainingMs must be null")
        targets.append(
            RichCaptureTargetTelemetry(
                target_native_object_id=native_id,
                target_entity_key=target_key,
                target_resolved=resolved,
                phase=str(target_phase),
                elapsed_ms=elapsed,
                phase_budget_remaining_ms=remaining,
            )
        )
    return RichCaptureRuntimeTelemetry(
        phase=str(phase),
        configured_cooldown_ms=configured_cooldown,
        cooldown_remaining_ms=cooldown_remaining,
        hit_frequency_ms=hit_frequency,
        hit_accumulator_ms=hit_accumulator,
        targets=tuple(targets),
    )


def _threshold_relocation_runtime(value: object, label: str) -> RichThresholdRelocationTelemetry | None:
    if value is None:
        return None
    raw = _record(value, label)
    required = {
        "schema",
        "actionDataGlobalId",
        "phase",
        "stage",
        "relocationIndex",
        "hideDurationMs",
        "remainingMs",
        "burrowed",
        "thresholdsPercent",
        "status",
    }
    if set(raw) != required:
        raise RichTelemetryMergeError(f"{label} fields do not match the relocation runtime schema")
    if raw.get("schema") != "native-threshold-relocation-runtime.v1":
        raise RichTelemetryMergeError(f"{label}.schema is invalid")
    if raw.get("status") != "authoritative":
        raise RichTelemetryMergeError(f"{label}.status must be authoritative")
    raw.integer("actionDataGlobalId", minimum=1, maximum=4294967295)
    raw_thresholds = raw.get("thresholdsPercent")
    if not isinstance(raw_thresholds, list) or not 1 <= len(raw_thresholds) <= 16:
        raise RichTelemetryMergeError(f"{label}.thresholdsPercent is invalid")
    thresholds = tuple(
        _integer(item, f"{label}.thresholdsPercent[{index}]", minimum=0, maximum=100)
        for index, item in enumerate(raw_thresholds)
    )
    if any(right >= left for left, right in zip(thresholds, thresholds[1:])):
        raise RichTelemetryMergeError(f"{label}.thresholdsPercent must be strictly descending")
    terminal_stage = 2 * len(thresholds) + 1
    stage = raw.integer("stage", minimum=1, maximum=terminal_stage)
    relocation_index = raw.integer("relocationIndex", minimum=0, maximum=len(thresholds))
    if relocation_index != (stage - 1) // 2:
        raise RichTelemetryMergeError(f"{label}.relocationIndex is inconsistent")
    expected_phase = "exhausted" if stage == terminal_stage else "relocating" if stage % 2 == 0 else "waiting_threshold"
    if raw.get("phase") != expected_phase:
        raise RichTelemetryMergeError(f"{label}.phase is inconsistent")
    hide_duration = raw.integer("hideDurationMs", minimum=1)
    remaining = raw.integer("remainingMs", minimum=0)
    if remaining > hide_duration:
        raise RichTelemetryMergeError(f"{label}.remainingMs exceeds its duration")
    burrowed = raw.boolean("burrowed")
    if burrowed and (expected_phase != "relocating" or remaining == 0):
        raise RichTelemetryMergeError(f"{label}.burrowed is inconsistent")
    return RichThresholdRelocationTelemetry(
        phase=expected_phase,
        stage=stage,
        relocation_index=relocation_index,
        thresholds_percent=thresholds,
        hide_duration_ms=hide_duration,
        remaining_ms=remaining,
        burrowed=burrowed,
    )


def _tower_troop_runtime(value: object, label: str) -> RichTowerTroopRuntimeTelemetry | None:
    if value is None:
        return None
    raw = _record(value, label)
    common = {"schema", "kind", "observedTick"}
    if raw.get("schema") != "native-tower-troop-runtime.v1":
        raise RichTelemetryMergeError(f"{label}.schema is invalid")
    observed_tick = raw.integer("observedTick", minimum=0)
    kind = raw.get("kind")
    if kind == "dagger_duchess":
        required = {*common, "chargeCount", "maxChargeCount", "rechargeElapsedMs", "rechargeDurationMs"}
        if set(raw) != required:
            raise RichTelemetryMergeError(f"{label} fields do not match Dagger Duchess runtime v1")
        maximum = raw.integer("maxChargeCount", minimum=1, maximum=32)
        charge = raw.integer("chargeCount", minimum=0, maximum=maximum)
        duration = raw.integer("rechargeDurationMs", minimum=1, maximum=60000)
        elapsed = raw.integer("rechargeElapsedMs", minimum=0, maximum=duration)
        return RichDaggerDuchessRuntimeTelemetry(
            observed_tick=observed_tick,
            charge_count=charge,
            max_charge_count=maximum,
            recharge_elapsed_ms=elapsed,
            recharge_duration_ms=duration,
        )
    if kind == "royal_chef":
        required = {
            *common,
            "startDelayRemainingMs",
            "startDelayDurationMs",
            "cookingContribution",
            "contributionNeeded",
            "throwDelayRemainingMs",
            "targetNativeObjectId",
        }
        if set(raw) != required:
            raise RichTelemetryMergeError(f"{label} fields do not match Royal Chef runtime v1")
        start_duration = raw.integer("startDelayDurationMs", minimum=1, maximum=60000)
        start_remaining = raw.integer("startDelayRemainingMs", minimum=0, maximum=start_duration)
        contribution_needed = raw.integer("contributionNeeded", minimum=1, maximum=100000000)
        contribution = raw.integer("cookingContribution", minimum=0, maximum=100000000)
        throw_raw = raw.get("throwDelayRemainingMs")
        throw_delay = (
            None
            if throw_raw is None
            else _integer(throw_raw, f"{label}.throwDelayRemainingMs", minimum=-50, maximum=60_000)
        )
        if throw_delay is not None and throw_delay < 0 and throw_delay != -50:
            raise RichTelemetryMergeError(f"{label}.throwDelayRemainingMs has an invalid terminal value")
        target_raw = raw.get("targetNativeObjectId")
        target = (
            None
            if target_raw is None
            else _integer(target_raw, f"{label}.targetNativeObjectId", minimum=1, maximum=0xFFFFFFFF)
        )
        if throw_delay is None and target is not None:
            raise RichTelemetryMergeError(f"{label} cannot expose a target without a pending throw")
        if throw_delay == -50 and target is not None:
            raise RichTelemetryMergeError(f"{label} targetless release cannot expose a live target")
        return RichRoyalChefRuntimeTelemetry(
            observed_tick=observed_tick,
            start_delay_remaining_ms=start_remaining,
            start_delay_duration_ms=start_duration,
            cooking_contribution=contribution,
            contribution_needed=contribution_needed,
            throw_delay_remaining_ms=throw_delay,
            target_native_object_id=target,
        )
    raise RichTelemetryMergeError(f"{label}.kind is invalid")


def _tower_troop_runtime_envelope(
    value: object, *, generation: int, state_epoch: int, observation_tick: int
) -> RichTowerTroopRuntimeEnvelope:
    label = "rich.towerTroopRuntime"
    raw = _record(value, label)
    required = {
        "ok",
        "schema",
        "generation",
        "stateEpoch",
        "observationTick",
        "hookSetAttested",
        "hookSetInstalled",
        "rejectedCount",
        "complete",
    }
    if set(raw) != required or raw.get("ok") is not True or raw.get("schema") != "native-tower-troop-runtime.v1":
        raise RichTelemetryMergeError("rich.towerTroopRuntime fields/schema are invalid")
    if (
        raw.integer("generation", minimum=0) != generation
        or raw.integer("stateEpoch", minimum=0) != state_epoch
        or raw.integer("observationTick", minimum=0) != observation_tick
    ):
        raise RichTelemetryMergeError("rich.towerTroopRuntime identity does not match the observation")
    attested = raw.boolean("hookSetAttested")
    installed = raw.boolean("hookSetInstalled")
    rejected = raw.integer("rejectedCount", minimum=0)
    complete = raw.boolean("complete")
    if installed and not attested:
        raise RichTelemetryMergeError("rich.towerTroopRuntime cannot be installed without attestation")
    if complete != (installed and rejected == 0):
        raise RichTelemetryMergeError("rich.towerTroopRuntime completeness is inconsistent")
    return RichTowerTroopRuntimeEnvelope(
        generation=generation,
        state_epoch=state_epoch,
        observation_tick=observation_tick,
        hook_set_attested=attested,
        hook_set_installed=installed,
        rejected_count=rejected,
        complete=complete,
    )


def _rich_object(
    raw: Mapping[str, Any],
    *,
    slot: int,
    ordinary_native_object_id: int,
    ordinary_owner: int,
    ordinary_object_index: int,
    ordinary_secondary_index: int,
    ordinary_card_id: int,
) -> RichObjectTelemetry:
    required = {
        "slot",
        "nativeObjectId",
        "entityKey",
        "owner",
        "cardId",
        "dataGlobalId",
        "x",
        "y",
        "targetX",
        "targetY",
        "targetEntityKey",
        "targetEntityValidated",
        "objectIndex",
        "secondaryIndex",
        "hp",
        "maxHp",
        "shield",
        "attackSequenceStage",
        "invisibleCount",
        "visibilityState",
        "projectile",
        "entityResourceRuntime",
        "periodicAttackModifierRuntime",
        "captureRuntime",
        "thresholdRelocationRuntime",
        "activeEffects",
        "components",
    }
    if "phaseRuntime" in raw:
        required.add("phaseRuntime")
    if "towerTroopRuntime" in raw:
        required.add("towerTroopRuntime")
    if set(raw) != required:
        raise RichTelemetryMergeError(f"rich.objects[{slot}] fields do not match the v3 schema")
    native_object_id = _integer(
        raw.get("nativeObjectId"), f"rich.objects[{slot}].nativeObjectId", minimum=1, maximum=0xFFFFFFFF
    )
    if native_object_id != ordinary_native_object_id:
        raise RichTelemetryMergeError(f"ordinary/rich native object identity mismatch at slot {slot}")
    owner = _integer(raw.get("owner"), f"rich.objects[{slot}].owner")
    object_index = _integer(raw.get("objectIndex"), f"rich.objects[{slot}].objectIndex")
    secondary_index = _integer(raw.get("secondaryIndex"), f"rich.objects[{slot}].secondaryIndex")
    if owner != ordinary_owner or object_index != ordinary_object_index or secondary_index != ordinary_secondary_index:
        raise RichTelemetryMergeError(f"rich object authoritative identity fields mismatch at slot {slot}")
    rich_key = _entity_key(raw.get("entityKey"), f"rich.objects[{slot}].entityKey")
    expected_key = _expected_native_entity_key(
        owner=owner, object_index=object_index, secondary_index=secondary_index, native_object_id=native_object_id
    )
    if rich_key != expected_key:
        raise RichTelemetryMergeError(
            f"rich object effective identity mismatch at slot {slot}: expected={expected_key}, rich={rich_key}"
        )
    card_id = _integer(raw.get("cardId"), f"rich.objects[{slot}].cardId")
    if card_id != ordinary_card_id:
        raise RichTelemetryMergeError(f"rich object owner/card identity mismatch at slot {slot}")
    data_global_id = _integer(
        raw.get("dataGlobalId"), f"rich.objects[{slot}].dataGlobalId", minimum=1, maximum=0xFFFFFFFF
    )

    shield_raw = raw.get("shield")
    shield_current: int | None = None
    shield_maximum: int | None = None
    if shield_raw is not None:
        shield = _record(shield_raw, f"rich.objects[{slot}].shield")
        if shield.get("status") != "authoritative":
            raise RichTelemetryMergeError(f"rich.objects[{slot}].shield.status must be authoritative")
        shield_current = shield.integer("current", minimum=0)
        shield_maximum = shield.integer("max", minimum=0)
        if shield_current > shield_maximum:
            raise RichTelemetryMergeError(f"rich.objects[{slot}] shield exceeds its maximum")

    target_validated = _bool(raw.get("targetEntityValidated"), f"rich.objects[{slot}].targetEntityValidated")
    target_raw = raw.get("targetEntityKey")
    target_key = None if target_raw is None else _entity_key(target_raw, f"rich.objects[{slot}].targetEntityKey")
    if target_key is not None and not target_validated:
        raise RichTelemetryMergeError(f"rich.objects[{slot}] exposes an unvalidated target entity")

    stage_raw = raw.get("attackSequenceStage")
    stage = None if stage_raw is None else _integer(stage_raw, f"rich.objects[{slot}].attackSequenceStage", minimum=0)
    if stage is not None and stage > 255:
        raise RichTelemetryMergeError(f"rich.objects[{slot}].attackSequenceStage must be in 0..255")

    invisible_raw = raw.get("invisibleCount")
    invisible_count = (
        None if invisible_raw is None else _integer(invisible_raw, f"rich.objects[{slot}].invisibleCount", minimum=0)
    )
    visibility_raw = raw.get("visibilityState")
    if visibility_raw is not None and visibility_raw not in {"visible", "invisible"}:
        raise RichTelemetryMergeError(f"rich.objects[{slot}].visibilityState is invalid")
    visibility_state = visibility_raw if isinstance(visibility_raw, str) else None
    if (invisible_count is None) != (visibility_state is None):
        raise RichTelemetryMergeError(f"rich.objects[{slot}] has a partial invisibility state")
    if invisible_count is not None:
        expected_visibility = "invisible" if invisible_count > 0 else "visible"
        if visibility_state != expected_visibility:
            raise RichTelemetryMergeError(f"rich.objects[{slot}] invisibility count/state disagree")

    return RichObjectTelemetry(
        slot=slot,
        native_object_id=native_object_id,
        entity_key=rich_key,
        owner=owner,
        object_index=object_index,
        secondary_index=secondary_index,
        card_id=card_id,
        data_global_id=data_global_id,
        shield_current=shield_current,
        shield_maximum=shield_maximum,
        target_entity_key=target_key,
        target_entity_validated=target_validated,
        attack_sequence_stage=stage,
        active_effects=_active_effects(raw.get("activeEffects"), f"rich.objects[{slot}].activeEffects"),
        invisible_count=invisible_count,
        visibility_state=visibility_state,
        projectile=_projectile(raw.get("projectile"), f"rich.objects[{slot}].projectile"),
        entity_resource_runtime=_entity_resource_runtime(
            raw.get("entityResourceRuntime"), f"rich.objects[{slot}].entityResourceRuntime"
        ),
        periodic_attack_modifier_runtime=_periodic_attack_modifier_runtime(
            raw.get("periodicAttackModifierRuntime"), f"rich.objects[{slot}].periodicAttackModifierRuntime"
        ),
        capture_runtime=_capture_runtime(raw.get("captureRuntime"), f"rich.objects[{slot}].captureRuntime"),
        threshold_relocation_runtime=_threshold_relocation_runtime(
            raw.get("thresholdRelocationRuntime"), f"rich.objects[{slot}].thresholdRelocationRuntime"
        ),
        phase_runtime=_phase_object(raw.get("phaseRuntime"), f"rich.objects[{slot}].phaseRuntime"),
        tower_troop_runtime=_tower_troop_runtime(
            raw.get("towerTroopRuntime"), f"rich.objects[{slot}].towerTroopRuntime"
        ),
    )


def _rich_player(
    raw: Mapping[str, Any], *, owner: int, ordinary_player: Mapping[str, Any]
) -> RichPlayerRuntimeTelemetry:
    label = f"rich.players[{owner}]"
    required = {"owner", "ownerRootValidated", "ownerEntityKey", "abilityRuntime", "evolutionRuntime"}
    if set(raw) != required:
        raise RichTelemetryMergeError(f"{label} fields do not match the v2 player schema")
    if _integer(raw.get("owner"), f"{label}.owner") != owner:
        raise RichTelemetryMergeError("rich player records are not owner-aligned")
    owner_root_validated = _bool(raw.get("ownerRootValidated"), f"{label}.ownerRootValidated")
    owner_key_raw = raw.get("ownerEntityKey")
    owner_key = None if owner_key_raw is None else _entity_key(owner_key_raw, f"{label}.ownerEntityKey")
    if owner_key is not None and not owner_root_validated:
        raise RichTelemetryMergeError(f"{label} owner-root validation/key relationship is invalid")
    if owner_key is not None and owner_key[0] != owner:
        raise RichTelemetryMergeError(f"{label} owner-root identity belongs to another owner")

    ability_runtime: tuple[RichAbilityRuntimeTelemetry, ...] | None = None
    abilities_raw = raw.get("abilityRuntime")
    if abilities_raw is not None:
        if not owner_root_validated:
            raise RichTelemetryMergeError(f"{label} exposes abilities without a validated owner root")
        ability_values = _items(abilities_raw, f"{label}.abilityRuntime")
        if len(ability_values) > 2:
            raise RichTelemetryMergeError(f"{label}.abilityRuntime exceeds two controller slots")
        abilities: list[RichAbilityRuntimeTelemetry] = []
        previous_controller_slot = 0
        for index, value in enumerate(ability_values):
            ability_label = f"{label}.abilityRuntime[{index}]"
            ability = _record(value, ability_label)
            fields = {
                "controllerSlot",
                "actionDataGlobalId",
                "actionDataName",
                "selectedCharacterDataGlobalId",
                "remainingCooldownMs",
                "configuredCooldownMs",
                "remainingChargesRaw",
                "maxCharges",
                "buttonState",
                "buttonStateLabel",
                "available",
                "championEntityKeys",
            }
            if set(ability) != fields:
                raise RichTelemetryMergeError(f"{ability_label} fields do not match the v2 ability schema")
            controller_slot = ability.integer("controllerSlot", minimum=1)
            if controller_slot > 2 or controller_slot <= previous_controller_slot:
                raise RichTelemetryMergeError(f"{label} controller slots are invalid or not ordered")
            previous_controller_slot = controller_slot
            action_name = ability.get("actionDataName")
            if not isinstance(action_name, str):
                raise RichTelemetryMergeError(f"{ability_label}.actionDataName must be a string")
            encoded_name = action_name.encode("utf-8")
            if not 1 <= len(encoded_name) <= 128:
                raise RichTelemetryMergeError(f"{ability_label}.actionDataName must contain 1..128 UTF-8 bytes")
            configured_cooldown = ability.integer("configuredCooldownMs", minimum=0)
            remaining_cooldown = ability.integer("remainingCooldownMs", minimum=0)
            if remaining_cooldown > configured_cooldown:
                raise RichTelemetryMergeError(f"{ability_label} cooldown exceeds its configured duration")
            max_charges = ability.integer("maxCharges", minimum=0)
            remaining_charges = ability.integer("remainingChargesRaw", minimum=-1)
            if (max_charges > 0 and not 0 <= remaining_charges <= max_charges) or (
                max_charges <= 0 and remaining_charges != -1
            ):
                raise RichTelemetryMergeError(f"{ability_label} charge bounds are invalid")
            button_state = ability.integer("buttonState", minimum=0)
            if button_state >= len(ABILITY_BUTTON_STATE_LABELS):
                raise RichTelemetryMergeError(f"{ability_label}.buttonState is outside the exact enum")
            if ability.get("buttonStateLabel") != ABILITY_BUTTON_STATE_LABELS[button_state]:
                raise RichTelemetryMergeError(f"{ability_label}.buttonStateLabel is not exact")
            available = ability.boolean("available")
            if available != (button_state in ABILITY_QUEUEABLE_BUTTON_STATES):
                raise RichTelemetryMergeError(f"{ability_label}.available is not an exact queueable state")
            champion_values = ability.sequence("championEntityKeys")
            if len(champion_values) > 256:
                raise RichTelemetryMergeError(f"{ability_label}.championEntityKeys exceeds native capacity")
            champion_keys = tuple(
                _entity_key(value, f"{ability_label}.championEntityKeys[{key_index}]")
                for key_index, value in enumerate(champion_values)
            )
            if len(set(champion_keys)) != len(champion_keys):
                raise RichTelemetryMergeError(f"{ability_label} contains duplicate champion identities")
            if any(key[0] != owner for key in champion_keys):
                raise RichTelemetryMergeError(f"{ability_label} contains a champion owned by another player")
            abilities.append(
                RichAbilityRuntimeTelemetry(
                    controller_slot=controller_slot,
                    action_data_global_id=ability.integer("actionDataGlobalId", minimum=1),
                    action_data_name=action_name,
                    selected_character_data_global_id=ability.integer("selectedCharacterDataGlobalId", minimum=1),
                    remaining_cooldown_ms=remaining_cooldown,
                    configured_cooldown_ms=configured_cooldown,
                    remaining_charges_raw=remaining_charges,
                    max_charges=max_charges,
                    button_state=button_state,
                    button_state_label=ABILITY_BUTTON_STATE_LABELS[button_state],
                    available=available,
                    champion_entity_keys=champion_keys,
                )
            )
        ability_runtime = tuple(abilities)

    evolution_runtime: tuple[RichEvolutionRuntimeTelemetry, ...] | None = None
    evolution_raw = raw.get("evolutionRuntime")
    if evolution_raw is not None:
        if not owner_root_validated:
            raise RichTelemetryMergeError(f"{label} exposes evolution state without a validated owner root")
        evolution_values = _items(evolution_raw, f"{label}.evolutionRuntime")
        if len(evolution_values) > 8:
            raise RichTelemetryMergeError(f"{label}.evolutionRuntime exceeds eight deck slots")
        ordinary_deck_values = _items(ordinary_player.get("deck"), f"ordinary.players[{owner}].deck")
        if len(evolution_values) != len(ordinary_deck_values):
            raise RichTelemetryMergeError(f"{label}.evolutionRuntime does not cover the ordinary deck exactly")
        evolutions: list[RichEvolutionRuntimeTelemetry] = []
        for deck_slot, (value, ordinary_deck_value) in enumerate(
            zip(evolution_values, ordinary_deck_values, strict=True)
        ):
            slot_label = f"{label}.evolutionRuntime[{deck_slot}]"
            slot = _record(value, slot_label)
            fields = {
                "deckSlot",
                "cardId",
                "baseSpellGlobalId",
                "evolvable",
                "evolutionFormGlobalId",
                "progress",
                "cycleRequired",
                "cycleRemaining",
                "ready",
            }
            if set(slot) != fields:
                raise RichTelemetryMergeError(f"{slot_label} fields do not match the v2 evolution schema")
            if slot.integer("deckSlot") != deck_slot:
                raise RichTelemetryMergeError(f"{label} evolution deck slots are not contiguous")
            ordinary_deck = _record(ordinary_deck_value, f"ordinary.players[{owner}].deck[{deck_slot}]")
            if ordinary_deck.integer("deckSlot") != deck_slot:
                raise RichTelemetryMergeError(f"ordinary player {owner} deck slots are not contiguous")
            card_id = slot.integer("cardId", minimum=1)
            ordinary_card_id = ordinary_deck.integer("cardId", minimum=1)
            if card_id != ordinary_card_id or slot.integer("baseSpellGlobalId", minimum=1) != card_id:
                raise RichTelemetryMergeError(f"{slot_label} does not exactly join the ordinary deck card")
            evolvable = slot.boolean("evolvable")
            progress = slot.integer("progress", minimum=0)
            form_raw = slot.get("evolutionFormGlobalId")
            cycle_required_raw = slot.get("cycleRequired")
            cycle_remaining_raw = slot.get("cycleRemaining")
            ready_raw = slot.get("ready")
            if not evolvable:
                if (
                    progress != 0
                    or form_raw is not None
                    or cycle_required_raw is not None
                    or cycle_remaining_raw is not None
                    or ready_raw is not None
                ):
                    raise RichTelemetryMergeError(f"{slot_label} exposes evolution fields for a base-only card")
                evolution_form_id = None
                cycle_required = None
                cycle_remaining = None
                ready = None
            else:
                evolution_form_id = _integer(form_raw, f"{slot_label}.evolutionFormGlobalId", minimum=1)
                cycle_required = _integer(cycle_required_raw, f"{slot_label}.cycleRequired", minimum=1)
                cycle_remaining = _integer(cycle_remaining_raw, f"{slot_label}.cycleRemaining", minimum=0)
                ready = _bool(ready_raw, f"{slot_label}.ready")
                if (
                    progress > cycle_required
                    or cycle_remaining != cycle_required - progress
                    or ready != (progress >= cycle_required)
                ):
                    raise RichTelemetryMergeError(f"{slot_label} has inconsistent cycle arithmetic")
            evolutions.append(
                RichEvolutionRuntimeTelemetry(
                    deck_slot=deck_slot,
                    card_id=card_id,
                    base_spell_global_id=card_id,
                    evolvable=evolvable,
                    evolution_form_global_id=evolution_form_id,
                    progress=progress,
                    cycle_required=cycle_required,
                    cycle_remaining=cycle_remaining,
                    ready=ready,
                )
            )
        evolution_runtime = tuple(evolutions)

    return RichPlayerRuntimeTelemetry(
        owner=owner,
        owner_root_validated=owner_root_validated,
        owner_entity_key=owner_key,
        ability_runtime=ability_runtime,
        evolution_runtime=evolution_runtime,
    )


def bind_rich_telemetry(
    ordinary: Mapping[str, Any],
    rich: Mapping[str, Any],
    *,
    combat_event_floor: int | None = None,
    phase_event_floor: int | None = None,
    visibility_event_floor: int | None = None,
    remaining_event_floor: int | None = None,
) -> RichTelemetrySnapshot:
    """Validate and bind two same-state native observation envelopes."""

    tick, generation, state_epoch = _validate_envelope_identity(ordinary, rich)
    _validate_field_provenance(rich)
    combat_events = _combat_events(
        rich.get("combatEvents"),
        generation=generation,
        state_epoch=state_epoch,
        observation_tick=tick,
        minimum_sequence=combat_event_floor,
    )
    phase_runtime = (
        _phase_runtime_events(
            rich.get("phaseRuntime"),
            generation=generation,
            state_epoch=state_epoch,
            observation_tick=tick,
            minimum_sequence=phase_event_floor,
        )
        if "phaseRuntime" in rich
        else None
    )
    if "specialMovementRuntime" in rich:
        try:
            special_movement_runtime = parse_special_movement_runtime(
                rich.get("specialMovementRuntime"),
                generation=generation,
                state_epoch=state_epoch,
                observation_tick=tick,
            )
        except SpecialMovementRuntimeError as error:
            raise RichTelemetryMergeError(
                f"rich.specialMovementRuntime contradicts its exact contract: {error}"
            ) from error
    else:
        special_movement_runtime = None
    if "actionMovementRuntime" in rich:
        try:
            action_movement_runtime = parse_action_movement_runtime(
                rich.get("actionMovementRuntime"), generation=generation, state_epoch=state_epoch, observation_tick=tick
            )
        except ActionMovementRuntimeError as error:
            raise RichTelemetryMergeError(
                f"rich.actionMovementRuntime contradicts its exact contract: {error}"
            ) from error
    else:
        action_movement_runtime = None
    if "characterStateRuntime" in rich:
        try:
            character_state_runtime = parse_character_state_runtime(
                rich.get("characterStateRuntime"), generation=generation, state_epoch=state_epoch, observation_tick=tick
            )
        except CharacterStateRuntimeError as error:
            raise RichTelemetryMergeError(
                f"rich.characterStateRuntime contradicts its exact contract: {error}"
            ) from error
    else:
        character_state_runtime = None
    visibility_runtime = (
        _visibility_runtime_events(
            rich.get("visibilityRuntime"),
            generation=generation,
            state_epoch=state_epoch,
            observation_tick=tick,
            minimum_sequence=visibility_event_floor,
        )
        if "visibilityRuntime" in rich
        else None
    )
    remaining_runtime = (
        _remaining_runtime_events(
            rich.get("remainingRuntime"),
            generation=generation,
            state_epoch=state_epoch,
            observation_tick=tick,
            minimum_sequence=remaining_event_floor,
        )
        if "remainingRuntime" in rich
        else None
    )
    tower_troop_runtime = (
        _tower_troop_runtime_envelope(
            rich.get("towerTroopRuntime"), generation=generation, state_epoch=state_epoch, observation_tick=tick
        )
        if "towerTroopRuntime" in rich
        else None
    )
    ordinary_objects = _items(ordinary.get("objects"), "ordinary.objects")
    rich_objects = _items(rich.get("objects"), "rich.objects")
    ordinary_count = _integer(ordinary.get("count"), "ordinary.count", minimum=0)
    ordinary_returned = _integer(ordinary.get("returned"), "ordinary.returned", minimum=0)
    rich_count = _integer(rich.get("count"), "rich.count", minimum=0)
    rich_returned = _integer(rich.get("returned"), "rich.returned", minimum=0)
    if ordinary.get("truncated") is True or rich.get("truncated") is not False:
        raise RichTelemetryMergeError("truncated ordinary/rich object vectors cannot be identity-joined")
    if not (
        ordinary_count == ordinary_returned == len(ordinary_objects)
        and rich_count == rich_returned == len(rich_objects)
        and ordinary_count == rich_count
    ):
        raise RichTelemetryMergeError("ordinary/rich object vector count or returned length mismatch")

    bound: list[RichObjectTelemetry | None] = []
    by_key: dict[EntityKey, RichObjectTelemetry] = {}
    by_native_object_id: dict[int, RichObjectTelemetry] = {}
    for expected_slot, (ordinary_value, rich_value) in enumerate(zip(ordinary_objects, rich_objects, strict=True)):
        ordinary_object = _record(ordinary_value, f"ordinary.objects[{expected_slot}]")
        rich_object = _record(rich_value, f"rich.objects[{expected_slot}]")
        ordinary_slot = ordinary_object.integer("slot")
        rich_slot = rich_object.integer("slot")
        if ordinary_slot != expected_slot or rich_slot != expected_slot:
            raise RichTelemetryMergeError(f"ordinary/rich object slots are not aligned at index {expected_slot}")
        ordinary_null = ordinary_object.get("null") is True
        rich_null = rich_object.get("null") is True
        if ordinary_null != rich_null:
            raise RichTelemetryMergeError(f"ordinary/rich null-slot mismatch at slot {expected_slot}")
        if ordinary_null:
            bound.append(None)
            continue
        if ("phaseRuntime" in rich_object) != (phase_runtime is not None):
            raise RichTelemetryMergeError(
                f"rich.objects[{expected_slot}] phaseRuntime presence does not match the top-level envelope"
            )
        if ("towerTroopRuntime" in rich_object) != (tower_troop_runtime is not None):
            raise RichTelemetryMergeError(
                f"rich.objects[{expected_slot}] towerTroopRuntime presence does not match the top-level envelope"
            )
        ordinary_owner = ordinary_object.integer("owner")
        ordinary_native_object_id = ordinary_object.integer("nativeObjectId", minimum=1, maximum=4294967295)
        ordinary_object_index = ordinary_object.integer("objectIndex")
        ordinary_secondary_index = ordinary_object.integer("secondaryIndex")
        ordinary_card_id = ordinary_object.integer("cardId")
        item = _rich_object(
            rich_object,
            slot=expected_slot,
            ordinary_native_object_id=ordinary_native_object_id,
            ordinary_owner=ordinary_owner,
            ordinary_object_index=ordinary_object_index,
            ordinary_secondary_index=ordinary_secondary_index,
            ordinary_card_id=ordinary_card_id,
        )
        if item.native_object_id in by_native_object_id:
            raise RichTelemetryMergeError(f"duplicate rich native object identity {item.native_object_id}")
        if item.entity_key in by_key:
            raise RichTelemetryMergeError(f"duplicate rich entity identity {item.entity_key}")
        by_key[item.entity_key] = item
        by_native_object_id[item.native_object_id] = item
        bound.append(item)

    for item in by_key.values():
        if item.target_entity_key is not None and item.target_entity_key not in by_key:
            raise RichTelemetryMergeError(f"slot {item.slot} target identity is outside the current object vector")
        for effect in item.active_effects or ():
            if effect.source_entity_key is not None and effect.source_entity_key not in by_key:
                raise RichTelemetryMergeError(
                    f"slot {item.slot} effect source identity is outside the current object vector"
                )

        projectile = item.projectile
        if projectile is not None:
            for relation_name, relation_key in (
                ("source", projectile.source_entity_key),
                ("target", projectile.target_entity_key),
            ):
                if relation_key is not None and relation_key not in by_key:
                    raise RichTelemetryMergeError(
                        f"slot {item.slot} projectile {relation_name} identity is outside the current object vector"
                    )
        tower_runtime = item.tower_troop_runtime
        if tower_runtime is not None:
            if tower_troop_runtime is None or not tower_troop_runtime.complete or tower_runtime.observed_tick > tick:
                raise RichTelemetryMergeError(f"slot {item.slot} tower-troop runtime lacks a complete envelope")
            if isinstance(tower_runtime, RichRoyalChefRuntimeTelemetry):
                target_native_id = tower_runtime.target_native_object_id
                if target_native_id is not None and target_native_id not in by_native_object_id:
                    raise RichTelemetryMergeError(
                        f"slot {item.slot} Royal Chef target identity is outside the current object vector"
                    )

    ordinary_player_values = _items(ordinary.get("players"), "ordinary.players")
    rich_player_values = _items(rich.get("players"), "rich.players")
    if len(ordinary_player_values) != 2 or len(rich_player_values) != 2:
        raise RichTelemetryMergeError("ordinary/rich observations must contain exactly two players")
    ordinary_players: dict[int, Mapping[str, Any]] = {}
    for index, value in enumerate(ordinary_player_values):
        ordinary_player = _record(value, f"ordinary.players[{index}]")
        owner = ordinary_player.integer("owner")
        if owner not in (0, 1) or owner in ordinary_players:
            raise RichTelemetryMergeError("ordinary player owners must be exactly 0 and 1")
        ordinary_players[owner] = ordinary_player
    players: dict[int, RichPlayerRuntimeTelemetry] = {}
    for owner, value in enumerate(rich_player_values):
        player = _rich_player(
            _mapping(value, f"rich.players[{owner}]"), owner=owner, ordinary_player=ordinary_players[owner]
        )
        if player.owner_entity_key is not None and player.owner_entity_key not in by_key:
            raise RichTelemetryMergeError(f"rich player {owner} owner-root identity is outside the object vector")
        for ability in player.ability_runtime or ():
            for champion_key in ability.champion_entity_keys:
                if champion_key not in by_key:
                    raise RichTelemetryMergeError(f"rich player {owner} champion identity is outside the object vector")
        players[owner] = player

    return RichTelemetrySnapshot(
        tick=tick,
        generation=generation,
        state_epoch=state_epoch,
        objects=tuple(bound),
        objects_by_key=by_key,
        objects_by_native_object_id=by_native_object_id,
        players=players,
        combat_events=combat_events,
        phase_runtime=phase_runtime,
        visibility_runtime=visibility_runtime,
        remaining_runtime=remaining_runtime,
        tower_troop_runtime=tower_troop_runtime,
        special_movement_runtime=special_movement_runtime,
        action_movement_runtime=action_movement_runtime,
        character_state_runtime=character_state_runtime,
    )


def _field_provenance(
    fields: Sequence[str] | frozenset[str],
    evidence: Mapping[str, SemanticEvidenceLevel],
    sources: Mapping[str, tuple[str, ...]],
    *,
    observed_tick: int,
    notes: tuple[str, ...] = (),
) -> SemanticProvenanceV1:
    field_names = tuple(sorted(fields))
    return _cached_field_provenance(
        tuple((name, evidence.get(name, SemanticEvidenceLevel.UNKNOWN)) for name in field_names),
        tuple((name, sources[name]) for name in field_names if name in sources),
        observed_tick,
        notes,
    )


@lru_cache(maxsize=16_384)
def _cached_field_provenance(
    field_evidence: tuple[tuple[str, SemanticEvidenceLevel], ...],
    source_fields: tuple[tuple[str, tuple[str, ...]], ...],
    observed_tick: int,
    notes: tuple[str, ...],
) -> SemanticProvenanceV1:
    """Share identical immutable provenance records within replay hot paths."""

    return SemanticProvenanceV1(
        field_evidence=dict(field_evidence), source_fields=dict(source_fields), observed_tick=observed_tick, notes=notes
    )


@dataclass(frozen=True, slots=True)
class RuntimeProjection:
    shield: float | None
    shield_state: ShieldStateV1 | None
    effect_states: tuple[EffectStateV1, ...]
    attack_state: AttackStateV1 | None
    movement_runtime: MovementRuntimeV1 | None
    deployment_runtime: DeploymentRuntimeV1 | None
    projectile_state: ProjectileStateV1 | None
    visibility_state: VisibilityStateV1 | None
    resource_states: tuple[EntityResourceStateV1, ...]
    periodic_attack_modifier: PeriodicAttackModifierStateV1 | None
    capture_runtime: CaptureRuntimeStateV1 | None
    threshold_relocation_runtime: ThresholdRelocationStateV1 | None
    target_entity: int | None
    invisible_count: int | None
    domain_evidence: Mapping[str, SemanticEvidenceLevel]

    def entity_provenance(self, observed_tick: int) -> SemanticProvenanceV1:
        return _field_provenance(
            ENTITY_RUNTIME_SEMANTIC_FIELDS,
            self.domain_evidence,
            {
                "shield_state": ("objects[].shield",),
                "effect_states": ("objects[].activeEffects",),
                "attack_state": (
                    "objects[].targetEntityKey",
                    "objects[].attackSequenceStage",
                    "objects[].phaseRuntime",
                    "phaseRuntime.events[]",
                ),
                "movement_runtime": ("objects[].phaseRuntime",),
                "deployment_runtime": ("objects[].phaseRuntime",),
                "projectile_state": ("objects[].projectile",),
                "visibility_state": ("objects[].invisibleCount",),
                "resource_states": ("objects[].entityResourceRuntime",),
                "periodic_attack_modifier": ("objects[].periodicAttackModifierRuntime",),
                "capture_runtime": ("objects[].captureRuntime",),
                "threshold_relocation_runtime": ("objects[].thresholdRelocationRuntime",),
            },
            observed_tick=observed_tick,
        )

    def tower_provenance(self, observed_tick: int) -> SemanticProvenanceV1:
        return _field_provenance(
            TOWER_RUNTIME_SEMANTIC_FIELDS,
            self.domain_evidence,
            {
                "shield_state": ("objects[].shield",),
                "effect_states": ("objects[].activeEffects",),
                "attack_state": (
                    "objects[].targetEntityKey",
                    "objects[].attackSequenceStage",
                    "objects[].phaseRuntime",
                    "phaseRuntime.events[]",
                ),
                "visibility_state": ("objects[].invisibleCount",),
            },
            observed_tick=observed_tick,
        )


@dataclass(frozen=True, slots=True)
class PlayerRuntimeProjection:
    ability_runtime_states: tuple[AbilityRuntimeStateV1, ...]
    evolution_runtime_states: tuple[EvolutionRuntimeStateV1, ...]
    domain_evidence: Mapping[str, SemanticEvidenceLevel]

    def provenance(self, observed_tick: int) -> SemanticProvenanceV1:
        return _field_provenance(
            PLAYER_RUNTIME_SEMANTIC_FIELDS,
            self.domain_evidence,
            {
                "ability_runtime_states": ("players[].abilityRuntime",),
                "evolution_runtime_states": ("players[].evolutionRuntime",),
            },
            observed_tick=observed_tick,
            notes=(
                "player runtime telemetry is private to that owner in FAIR observations",
                "no ability-cast, projectile-impact, or evolution-play event is synthesized",
            ),
        )


def _ability_phase(button_state: int) -> AbilityPhase:
    return {
        1: AbilityPhase.UNAVAILABLE,
        2: AbilityPhase.READY,
        4: AbilityPhase.READY,
        6: AbilityPhase.EXHAUSTED,
        8: AbilityPhase.COOLDOWN,
        9: AbilityPhase.UNAVAILABLE,
        10: AbilityPhase.CASTING,
        11: AbilityPhase.UNAVAILABLE,
        12: AbilityPhase.UNAVAILABLE,
        13: AbilityPhase.UNAVAILABLE,
    }.get(button_state, AbilityPhase.UNKNOWN)


def normalized_ability_cooldown_ms(ability: AbilitySpecV1) -> int:
    """Return the exact native configured-cooldown representation.

    Runtime content uses an absent static ``Cooldown`` for charge-only,
    single-use abilities.  The native controller represents that same contract
    as ``configuredCooldownMs == 0``; explicit cooldowns remain exact integers.
    """

    return int(ability.cooldown_ms or 0)


def project_player_runtime(
    player: RichPlayerRuntimeTelemetry,
    *,
    ordinary_deck: Sequence[int],
    card_specs: Mapping[int, CardSpecV1],
    ability_specs: Mapping[str, AbilitySpecV1],
    canonical_ids: Mapping[EntityKey, int],
    rich_objects: Mapping[EntityKey, RichObjectTelemetry],
    visible_keys: frozenset[EntityKey],
    observed_tick: int,
) -> PlayerRuntimeProjection:
    """Join private player runtime to frozen specs without guessing identities."""

    deck = tuple(int(card_id) for card_id in ordinary_deck)
    abilities: list[AbilityRuntimeStateV1] = []
    for raw in player.ability_runtime or ():
        champion_keys = raw.champion_entity_keys
        champion_key = champion_keys[0] if len(champion_keys) == 1 else None
        champion_object = rich_objects.get(champion_key) if champion_key is not None else None
        live_source_card_id = (
            champion_object.card_id if champion_object is not None and champion_object.card_id > 0 else None
        )
        static_ability = ability_specs.get(raw.action_data_name)
        static_card = (
            card_specs.get(static_ability.source_card_id)
            if static_ability is not None and static_ability.source_card_id is not None
            else None
        )
        deck_slots = tuple(
            index
            for index, card_id in enumerate(deck)
            if static_ability is not None and card_id == static_ability.source_card_id
        )
        exact_catalog_join = bool(
            static_ability is not None
            and static_ability.source_card_id is not None
            and static_card is not None
            and static_card.ability_ids == (raw.action_data_name,)
            and len(deck_slots) == 1
            and normalized_ability_cooldown_ms(static_ability) == raw.configured_cooldown_ms
            and (
                static_ability.charges == raw.max_charges
                if static_ability.charges is not None
                else raw.max_charges == 0
            )
        )
        source_entity = None
        source_entity_evidence = SemanticEvidenceLevel.UNKNOWN
        join_notes: list[str] = []
        exact_source_join = bool(
            exact_catalog_join
            and champion_key is not None
            and (
                live_source_card_id == static_ability.source_card_id
                or _VALIDATED_HERO_FORM_SOURCE_CARDS.get((raw.action_data_name, live_source_card_id))
                == static_ability.source_card_id
            )
        )
        if exact_source_join and champion_key in visible_keys:
            source_entity = canonical_ids[champion_key]
            source_entity_evidence = SemanticEvidenceLevel.NATIVE_DERIVED
        elif exact_source_join:
            join_notes.append("ability source identity suppressed by FAIR visibility")
        elif exact_catalog_join:
            join_notes.append(
                "ability identity is uniquely catalog-joined but no unique matching live champion source exists"
            )
        else:
            join_notes.append(
                "ability ID/deck slot/source remain unknown because the exact "
                "action-name/card/entity/deck/static-cooldown join was not unique"
            )

        # AbilityRuntimeStateV1 requires a stable public ability_id.  The native
        # action global ID/name remain available in RichPlayerRuntimeTelemetry,
        # but no typed state is emitted until the runtime controller joins one
        # unique frozen ability/card/deck/source identity.
        if not exact_catalog_join:
            continue
        ability_id = raw.action_data_name
        unlimited_charges = raw.remaining_charges_raw == -1
        phase = _ability_phase(raw.button_state)
        phase_evidence = (
            SemanticEvidenceLevel.NATIVE_DERIVED if phase != AbilityPhase.UNKNOWN else SemanticEvidenceLevel.UNKNOWN
        )
        abilities.append(
            AbilityRuntimeStateV1(
                ability_id=ability_id,
                source_entity=source_entity,
                phase=phase,
                elixir_cost=(static_ability.elixir_cost if exact_catalog_join and static_ability is not None else None),
                cooldown_ms=raw.configured_cooldown_ms,
                remaining_cooldown_ms=raw.remaining_cooldown_ms,
                charges=(None if unlimited_charges else raw.remaining_charges_raw),
                available=raw.available,
                attributes={
                    "classification": (
                        "exact_catalog_runtime_join" if exact_catalog_join else "unknown_runtime_ability_identity"
                    ),
                    "native_action_data_global_id": raw.action_data_global_id,
                    "native_action_data_name": raw.action_data_name,
                    "selected_character_data_global_id": (raw.selected_character_data_global_id),
                    "controller_slot": raw.controller_slot,
                    "button_state": raw.button_state,
                    "button_state_label": raw.button_state_label,
                    "remaining_charges_raw": raw.remaining_charges_raw,
                    "max_charges": raw.max_charges,
                    "deck_slot": deck_slots[0] if exact_catalog_join else None,
                    "source_card_id": (
                        static_ability.source_card_id if exact_catalog_join and static_ability is not None else None
                    ),
                },
                provenance=_field_provenance(
                    ABILITY_RUNTIME_STATE_FIELDS,
                    {
                        "source_entity": source_entity_evidence,
                        "phase": phase_evidence,
                        "elixir_cost": (
                            SemanticEvidenceLevel.STATIC_DECLARED
                            if exact_catalog_join
                            and static_ability is not None
                            and static_ability.elixir_cost is not None
                            else SemanticEvidenceLevel.UNKNOWN
                        ),
                        "cooldown_ms": SemanticEvidenceLevel.NATIVE_DERIVED,
                        "remaining_cooldown_ms": (SemanticEvidenceLevel.NATIVE_DERIVED),
                        "charges": (
                            SemanticEvidenceLevel.NOT_APPLICABLE
                            if unlimited_charges
                            else SemanticEvidenceLevel.NATIVE_DERIVED
                        ),
                        "available": SemanticEvidenceLevel.NATIVE_DERIVED,
                        "cast_started_tick": SemanticEvidenceLevel.UNKNOWN,
                        "active_until_tick": SemanticEvidenceLevel.UNKNOWN,
                        "target_entity": SemanticEvidenceLevel.UNKNOWN,
                        "target_position": SemanticEvidenceLevel.UNKNOWN,
                    },
                    {
                        "source_entity": ("players[].abilityRuntime[].championEntityKeys",),
                        "phase": (
                            "players[].abilityRuntime[].buttonState",
                            "players[].abilityRuntime[].buttonStateLabel",
                        ),
                        "elixir_cost": ("AbilitySpecV1.elixir_cost",),
                        "cooldown_ms": ("players[].abilityRuntime[].configuredCooldownMs",),
                        "remaining_cooldown_ms": ("players[].abilityRuntime[].remainingCooldownMs",),
                        "charges": ("players[].abilityRuntime[].remainingChargesRaw",),
                        "available": ("players[].abilityRuntime[].buttonState",),
                    },
                    observed_tick=observed_tick,
                    notes=tuple(
                        (
                            *join_notes,
                            f"raw ability enum={raw.button_state}:{raw.button_state_label}",
                            "available is true only for exact queueable enums Ready=2 or LimitedAvailability=4",
                            "-1 charges means unlimited/not-applicable and is retained in attributes",
                        )
                    ),
                ),
            )
        )

    evolutions: list[EvolutionRuntimeStateV1] = []
    for raw in player.evolution_runtime or ():
        if not raw.evolvable:
            continue
        static_card = card_specs.get(raw.card_id)
        static_evolution = static_card.evolution if static_card is not None else None
        exact_catalog_join = bool(
            raw.deck_slot < len(deck)
            and deck[raw.deck_slot] == raw.card_id
            and static_card is not None
            and static_evolution is not None
            and static_evolution.base_form_id
            and static_evolution.evolution_form_id
            and static_evolution.cycle_required == raw.cycle_required
        )
        if exact_catalog_join:
            if raw.ready:
                phase = EvolutionPhase.READY
            elif raw.progress == 0:
                phase = EvolutionPhase.BASE
            else:
                phase = EvolutionPhase.CYCLING
        else:
            phase = EvolutionPhase.UNKNOWN
        evolutions.append(
            EvolutionRuntimeStateV1(
                card_id=raw.card_id,
                deck_slot=raw.deck_slot,
                phase=phase,
                base_form_id=(
                    static_evolution.base_form_id if exact_catalog_join and static_evolution is not None else None
                ),
                current_form_id=None,
                next_form_id=(
                    static_evolution.evolution_form_id if exact_catalog_join and static_evolution is not None else None
                ),
                cycle_required=raw.cycle_required,
                cycle_remaining=raw.cycle_remaining,
                ready=raw.ready,
                active=None,
                deployments_in_cycle=raw.progress,
                attributes={
                    "classification": (
                        "exact_catalog_runtime_join" if exact_catalog_join else "unknown_evolution_catalog_identity"
                    ),
                    "base_spell_global_id": raw.base_spell_global_id,
                    "evolution_form_global_id": raw.evolution_form_global_id,
                    "raw_progress": raw.progress,
                    "played_form": "unknown",
                    "per_entity_evolution_active": "unknown",
                },
                provenance=_field_provenance(
                    EVOLUTION_RUNTIME_STATE_FIELDS,
                    {
                        "deck_slot": SemanticEvidenceLevel.NATIVE_DERIVED,
                        "phase": (
                            SemanticEvidenceLevel.NATIVE_DERIVED
                            if exact_catalog_join
                            else SemanticEvidenceLevel.UNKNOWN
                        ),
                        "base_form_id": (
                            SemanticEvidenceLevel.STATIC_DECLARED
                            if exact_catalog_join
                            else SemanticEvidenceLevel.UNKNOWN
                        ),
                        "current_form_id": SemanticEvidenceLevel.UNKNOWN,
                        "next_form_id": (
                            SemanticEvidenceLevel.STATIC_DECLARED
                            if exact_catalog_join
                            else SemanticEvidenceLevel.UNKNOWN
                        ),
                        "cycle_required": SemanticEvidenceLevel.NATIVE_DERIVED,
                        "cycle_remaining": SemanticEvidenceLevel.NATIVE_DERIVED,
                        "ready": SemanticEvidenceLevel.NATIVE_DERIVED,
                        "active": SemanticEvidenceLevel.UNKNOWN,
                        "deployments_in_cycle": (SemanticEvidenceLevel.NATIVE_DERIVED),
                    },
                    {
                        "deck_slot": ("players[].evolutionRuntime[].deckSlot",),
                        "phase": ("players[].evolutionRuntime[].progress", "players[].evolutionRuntime[].ready"),
                        "base_form_id": ("CardSpecV1.evolution.base_form_id",),
                        "next_form_id": ("CardSpecV1.evolution.evolution_form_id",),
                        "cycle_required": ("players[].evolutionRuntime[].cycleRequired",),
                        "cycle_remaining": ("players[].evolutionRuntime[].cycleRemaining",),
                        "ready": ("players[].evolutionRuntime[].ready",),
                        "deployments_in_cycle": ("players[].evolutionRuntime[].progress",),
                    },
                    observed_tick=observed_tick,
                    notes=(
                        "snapshot slot state does not infer played form",
                        "exact consumed descriptors are carried separately by combatEvents spawn records",
                        "ready state never marks a deployed entity evolved",
                        "no evolution transition event is synthesized from snapshot deltas",
                    ),
                ),
            )
        )

    return PlayerRuntimeProjection(
        ability_runtime_states=tuple(abilities),
        evolution_runtime_states=tuple(evolutions),
        domain_evidence={
            "ability_runtime_states": (
                SemanticEvidenceLevel.NATIVE_DERIVED
                if player.ability_runtime is not None
                else SemanticEvidenceLevel.UNKNOWN
            ),
            "evolution_runtime_states": (
                SemanticEvidenceLevel.NATIVE_DERIVED
                if player.evolution_runtime is not None
                else SemanticEvidenceLevel.UNKNOWN
            ),
        },
    )


def _primary_effect_kind(tags: Sequence[str]) -> EffectKind:
    """Choose one conservative primary kind while retaining every static tag."""

    available = frozenset(tags)
    priority = (
        ("rage", EffectKind.RAGE),
        ("freeze", EffectKind.FREEZE),
        ("stun", EffectKind.STUN),
        ("root", EffectKind.ROOT),
        ("curse", EffectKind.CURSE),
        ("reflect", EffectKind.REFLECT),
        ("immunity", EffectKind.IMMUNITY),
        ("invisibility", EffectKind.INVISIBILITY),
        ("shield", EffectKind.SHIELD),
        ("charge", EffectKind.CHARGE),
        ("periodic_damage", EffectKind.DAMAGE_OVER_TIME),
        ("periodic_heal", EffectKind.HEAL_OVER_TIME),
        ("slow", EffectKind.SLOW),
        ("haste", EffectKind.HASTE),
        ("attraction", EffectKind.PULL),
        ("targeting_modifier", EffectKind.TARGETING_MODIFIER),
        ("periodic_spawn", EffectKind.SPAWN_MODIFIER),
        ("death_spawn", EffectKind.SPAWN_MODIFIER),
    )
    for tag, kind in priority:
        if tag in available:
            return kind
    if available and available != {"opaque_native_buff"}:
        return EffectKind.OTHER
    return EffectKind.UNKNOWN


def _effect_attributes(
    effect: RichActiveEffect, *, index: int, effect_catalog: RuntimeEffectCatalog | None
) -> tuple[EffectKind, SemanticEvidenceLevel, dict[str, Any], tuple[str, ...]]:
    attributes: dict[str, Any] = {
        "native_buff_global_id": effect.buff_global_id,
        "native_buff_name": effect.name,
        "native_remaining_ms": effect.remaining_ms,
        "native_effect_index": index,
        "native_identity_evidence": "native_authoritative",
        "non_expiring": effect.remaining_ms == -1,
    }
    if effect_catalog is None:
        attributes["classification"] = "catalog_unavailable"
        return (
            EffectKind.UNKNOWN,
            SemanticEvidenceLevel.UNKNOWN,
            attributes,
            ("static BUFF catalog unavailable; native identity retained", "effect kind and magnitude remain unknown"),
        )

    attributes.update(
        {
            "effect_catalog_id": effect_catalog.catalog_id,
            "card_logic_catalog_id": effect_catalog.card_logic_catalog_id,
            "catalog_join_key": "BUFF.buff_global_id+name",
        }
    )
    definition = effect_catalog.definitions.get(effect.name)
    definition_by_global_id = effect_catalog.definitions_by_global_id.get(effect.buff_global_id)
    if definition is None:
        attributes["classification"] = (
            "native_buff_name_global_id_mismatch" if definition_by_global_id is not None else "unknown_native_buff_name"
        )
        return (
            EffectKind.UNKNOWN,
            SemanticEvidenceLevel.UNKNOWN,
            attributes,
            ("native BUFF name is absent from the bound static catalog", "effect kind and magnitude remain unknown"),
        )

    expected_global_id = definition.buff_global_id
    if (
        expected_global_id != effect.buff_global_id
        or definition_by_global_id is None
        or definition_by_global_id.effect_name != effect.name
        or definition_by_global_id.node_id != definition.node_id
    ):
        attributes.update(
            {
                "classification": "native_buff_name_global_id_mismatch",
                "catalog_expected_buff_global_id": expected_global_id,
                "catalog_global_id_name": (
                    definition_by_global_id.effect_name if definition_by_global_id is not None else None
                ),
            }
        )
        return (
            EffectKind.UNKNOWN,
            SemanticEvidenceLevel.UNKNOWN,
            attributes,
            (
                "native BUFF name and global ID do not jointly match the static catalog",
                "effect kind and magnitude remain unknown",
            ),
        )
    tags = tuple(str(tag) for tag in definition.mechanic_tags)
    kind = _primary_effect_kind(tags)
    attributes.update(
        {
            "classification": "static_catalog_resolved",
            "effect_node_id": definition.node_id,
            "mechanic_tags": tags,
            "modifiers": dict(definition.modifiers),
            "lifecycle": dict(definition.lifecycle),
            "targeting": dict(definition.targeting),
            "spawn": dict(definition.spawn),
            "overrides": dict(definition.overrides),
            "unresolved_static_semantics": tuple(definition.unresolved_semantics),
            "static_semantics_evidence": "authoritative_static_card_logic",
            "primary_kind_rule": "native-effect-primary-kind.v1",
        }
    )
    return (
        kind,
        (SemanticEvidenceLevel.STATIC_DECLARED if kind != EffectKind.UNKNOWN else SemanticEvidenceLevel.UNKNOWN),
        attributes,
        (
            "runtime effect identity is native-authoritative",
            "kind and attributes are joined from the content-addressed static BUFF catalog",
            "mechanic_tags retain every catalog category; kind is only the primary category",
        ),
    )


def _project_projectile(
    item: RichObjectTelemetry,
    *,
    canonical_ids: Mapping[EntityKey, int],
    rich_objects: Mapping[EntityKey, RichObjectTelemetry],
    visible_keys: frozenset[EntityKey],
    observed_tick: int,
) -> ProjectileStateV1 | None:
    projectile = item.projectile
    if projectile is None:
        return None

    notes = [
        "kind-4 projectile layout is exact-build static-high and not yet named-live golden",
        "terminal_or_finished_processing never implies impact or expiry",
    ]
    source_entity: int | None = None
    source_card_id: int | None = None
    source_entity_evidence = SemanticEvidenceLevel.UNKNOWN
    source_card_evidence = SemanticEvidenceLevel.UNKNOWN
    if projectile.source_entity_validated:
        if projectile.source_entity_key is None:
            source_entity_evidence = SemanticEvidenceLevel.NATIVE_DERIVED
        elif projectile.source_entity_key in visible_keys:
            source_entity = canonical_ids[projectile.source_entity_key]
            source_entity_evidence = SemanticEvidenceLevel.NATIVE_DERIVED
            source_object = rich_objects[projectile.source_entity_key]
            if source_object.card_id > 0:
                source_card_id = source_object.card_id
                source_card_evidence = SemanticEvidenceLevel.NATIVE_DERIVED
        else:
            notes.append("projectile source identity suppressed by FAIR visibility")

    target_entity: int | None = None
    target_evidence = SemanticEvidenceLevel.UNKNOWN
    if projectile.target_entity_validated:
        if projectile.target_entity_key is None:
            target_evidence = SemanticEvidenceLevel.NATIVE_DERIVED
        elif projectile.target_entity_key in visible_keys:
            target_entity = canonical_ids[projectile.target_entity_key]
            target_evidence = SemanticEvidenceLevel.NATIVE_DERIVED
        else:
            notes.append("projectile target identity suppressed by FAIR visibility")

    homing: bool | None = None
    homing_evidence = SemanticEvidenceLevel.UNKNOWN
    if projectile.homing_target_entity_validated:
        if projectile.homing_target_entity_key is None:
            homing = False
            homing_evidence = SemanticEvidenceLevel.NATIVE_DERIVED
        elif projectile.homing_target_entity_key in visible_keys:
            homing = True
            homing_evidence = SemanticEvidenceLevel.NATIVE_DERIVED
        else:
            notes.append("projectile active homing state suppressed by FAIR visibility")

    in_flight = not projectile.terminal
    return ProjectileStateV1(
        projectile_id=(
            "projectile:"
            f"{item.entity_key[0]}:{item.entity_key[1]}:{item.entity_key[2]}:"
            f"{projectile.projectile_data_global_id}"
        ),
        phase=(ProjectilePhase.IN_FLIGHT if in_flight else ProjectilePhase.UNKNOWN),
        source_entity=source_entity,
        source_card_id=source_card_id,
        target_entity=target_entity,
        target_position=(float(projectile.destination[0]), float(projectile.destination[1])),
        homing=homing,
        drag_stage=(None if projectile.drag_stage is None else ProjectileDragStage(projectile.drag_stage)),
        attributes={
            "native_projectile_data_global_id": projectile.projectile_data_global_id,
            "native_terminal": projectile.terminal,
            "native_phase": projectile.native_phase,
            "native_drag_stage": projectile.drag_stage,
            "source_entity_validated": projectile.source_entity_validated,
            "target_entity_validated": projectile.target_entity_validated,
            "homing_target_entity_validated": (projectile.homing_target_entity_validated),
            "terminal_reason": "unknown",
        },
        provenance=_field_provenance(
            PROJECTILE_STATE_FIELDS,
            {
                "phase": (SemanticEvidenceLevel.NATIVE_DERIVED if in_flight else SemanticEvidenceLevel.UNKNOWN),
                "source_entity": source_entity_evidence,
                "source_card_id": source_card_evidence,
                "target_entity": target_evidence,
                "target_position": SemanticEvidenceLevel.NATIVE_DERIVED,
                "spawn_tick": SemanticEvidenceLevel.UNKNOWN,
                "expected_impact_tick": SemanticEvidenceLevel.UNKNOWN,
                "impact_tick": SemanticEvidenceLevel.UNKNOWN,
                "velocity": SemanticEvidenceLevel.UNKNOWN,
                "damage": SemanticEvidenceLevel.UNKNOWN,
                "radius_tiles": SemanticEvidenceLevel.UNKNOWN,
                "homing": homing_evidence,
                "deflected": SemanticEvidenceLevel.UNKNOWN,
                "drag_stage": (
                    SemanticEvidenceLevel.NATIVE_AUTHORITATIVE
                    if projectile.drag_stage is not None
                    else SemanticEvidenceLevel.UNKNOWN
                ),
            },
            {
                "phase": ("objects[].projectile.terminal", "objects[].projectile.nativePhase"),
                "source_entity": ("objects[].projectile.sourceEntityKey",),
                "source_card_id": ("objects[].projectile.sourceEntityKey", "objects[].cardId"),
                "target_entity": ("objects[].projectile.targetEntityKey",),
                "homing": (
                    "objects[].projectile.homingTargetEntityKey",
                    "objects[].projectile.homingTargetEntityValidated",
                ),
                "target_position": ("objects[].projectile.destinationX", "objects[].projectile.destinationY"),
                "drag_stage": ("objects[].projectile.dragStage",),
            },
            observed_tick=observed_tick,
            notes=tuple(notes),
        ),
    )


def _damage_ramp_spec(
    card_spec: CardSpecV1 | None, *, observed_stage: int | None, evolution_active: bool = False
) -> DamageRampSpec | None:
    """Join exact static stages for the three continuous-lock damage ramps."""

    if (
        card_spec is None
        or (card_spec.name == "InfernoDragon" and evolution_active)
        or (observed_stage is not None and observed_stage >= 3)
    ):
        return None
    identity = {
        "InfernoDragon": DamageRampIdentity.INFERNO_DRAGON,
        "InfernoTower": DamageRampIdentity.INFERNO_TOWER,
        "MightyMiner": DamageRampIdentity.MIGHTY_MINER,
    }.get(card_spec.name)
    if identity is None:
        return None
    definition = card_spec.attributes.get("resolved_summoned_form")
    if not isinstance(definition, Mapping):
        return None
    try:
        damage = tuple(int(definition[name]) for name in ("Damage", "VariableDamage2", "VariableDamage3"))
        first = int(definition["VariableDamageTime1"])
        second = int(definition["VariableDamageTime2"])
    except (KeyError, TypeError, ValueError):
        return None
    if first <= 0 or second <= 0 or any(item < 0 for item in damage):
        return None
    return DamageRampSpec(identity=identity, stage_durations_ms=(first, second, second), stage_damage=damage)


def project_runtime(
    item: RichObjectTelemetry,
    *,
    canonical_ids: Mapping[EntityKey, int],
    rich_objects: Mapping[EntityKey, RichObjectTelemetry],
    visible_keys: frozenset[EntityKey],
    observed_tick: int,
    effect_catalog: RuntimeEffectCatalog | None = None,
    phase_runtime: RichPhaseRuntimeEnvelope | None = None,
    phase_events: Sequence[PhaseHookEvent] | None = None,
    card_spec: CardSpecV1 | None = None,
    evolution_active: bool = False,
) -> RuntimeProjection:
    """Project one bound native object without exposing hidden entity links."""

    projectile_state = _project_projectile(
        item,
        canonical_ids=canonical_ids,
        rich_objects=rich_objects,
        visible_keys=visible_keys,
        observed_tick=observed_tick,
    )

    resource_states: tuple[EntityResourceStateV1, ...] = ()
    if item.entity_resource_runtime is not None:
        resource = item.entity_resource_runtime
        resource_states = (
            EntityResourceStateV1(
                kind=resource.kind,
                current_raw=resource.current_raw,
                capacity_raw=resource.capacity_raw,
                normalized=resource.normalized,
                provenance=_field_provenance(
                    ENTITY_RESOURCE_STATE_FIELDS,
                    {
                        field_name: SemanticEvidenceLevel.NATIVE_AUTHORITATIVE
                        for field_name in ENTITY_RESOURCE_STATE_FIELDS
                    },
                    {
                        "kind": ("objects[].entityResourceRuntime.kind",),
                        "current_raw": ("objects[].entityResourceRuntime.currentRaw",),
                        "capacity_raw": ("objects[].entityResourceRuntime.capacityRaw",),
                        "normalized": ("objects[].entityResourceRuntime.normalized",),
                    },
                    observed_tick=observed_tick,
                ),
            ),
        )

    periodic_attack_modifier: PeriodicAttackModifierStateV1 | None = None
    if item.periodic_attack_modifier_runtime is not None:
        raw_modifier = item.periodic_attack_modifier_runtime
        source_entity = (
            canonical_ids.get(raw_modifier.source_entity_key)
            if raw_modifier.source_entity_key in visible_keys
            else None
        )
        periodic_attack_modifier = PeriodicAttackModifierStateV1(
            phase=PeriodicAttackModifierPhase(raw_modifier.phase),
            period_attacks=raw_modifier.period_attacks,
            completed_attacks=raw_modifier.completed_attacks,
            source_entity=source_entity,
            linger_remaining_ms=raw_modifier.linger_remaining_ms,
            linger_duration_ms=raw_modifier.linger_duration_ms,
            provenance=_field_provenance(
                PERIODIC_ATTACK_MODIFIER_STATE_FIELDS,
                {
                    "phase": SemanticEvidenceLevel.NATIVE_DERIVED,
                    "period_attacks": SemanticEvidenceLevel.NATIVE_AUTHORITATIVE,
                    "completed_attacks": (SemanticEvidenceLevel.NATIVE_AUTHORITATIVE),
                    "source_entity": (
                        SemanticEvidenceLevel.NATIVE_DERIVED
                        if source_entity is not None
                        else SemanticEvidenceLevel.UNKNOWN
                    ),
                    "linger_remaining_ms": (SemanticEvidenceLevel.NATIVE_DERIVED),
                    "linger_duration_ms": (SemanticEvidenceLevel.NATIVE_AUTHORITATIVE),
                },
                {
                    "phase": ("objects[].periodicAttackModifierRuntime.phase",),
                    "period_attacks": ("objects[].periodicAttackModifierRuntime.periodAttacks",),
                    "completed_attacks": ("objects[].periodicAttackModifierRuntime.completedAttacks",),
                    "source_entity": ("objects[].periodicAttackModifierRuntime.sourceEntityKey",),
                    "linger_remaining_ms": ("objects[].periodicAttackModifierRuntime.lingerRemainingMs",),
                    "linger_duration_ms": ("objects[].periodicAttackModifierRuntime.lingerDurationMs",),
                },
                observed_tick=observed_tick,
            ),
        )

    capture_runtime: CaptureRuntimeStateV1 | None = None
    if item.capture_runtime is not None:
        raw_capture = item.capture_runtime
        capture_targets = tuple(
            CaptureTargetStateV1(
                phase=CaptureTargetPhase(target.phase),
                elapsed_ms=target.elapsed_ms,
                phase_budget_remaining_ms=target.phase_budget_remaining_ms,
                target_entity=(
                    canonical_ids.get(target.target_entity_key) if target.target_entity_key in visible_keys else None
                ),
            )
            for target in raw_capture.targets
        )
        capture_runtime = CaptureRuntimeStateV1(
            phase=CaptureActionPhase(raw_capture.phase),
            configured_cooldown_ms=raw_capture.configured_cooldown_ms,
            cooldown_remaining_ms=raw_capture.cooldown_remaining_ms,
            hit_frequency_ms=raw_capture.hit_frequency_ms,
            hit_accumulator_ms=raw_capture.hit_accumulator_ms,
            targets=capture_targets,
            provenance=_field_provenance(
                CAPTURE_RUNTIME_STATE_FIELDS,
                {
                    **{
                        field_name: SemanticEvidenceLevel.NATIVE_AUTHORITATIVE
                        for field_name in CAPTURE_RUNTIME_STATE_FIELDS
                        if field_name != "targets"
                    },
                    "targets": SemanticEvidenceLevel.NATIVE_DERIVED,
                },
                {
                    field_name: (f"objects[].captureRuntime.{field_name}",)
                    for field_name in CAPTURE_RUNTIME_STATE_FIELDS
                },
                observed_tick=observed_tick,
            ),
        )

    threshold_relocation_runtime: ThresholdRelocationStateV1 | None = None
    if item.threshold_relocation_runtime is not None:
        raw_relocation = item.threshold_relocation_runtime
        threshold_relocation_runtime = ThresholdRelocationStateV1(
            phase=ThresholdRelocationPhase(raw_relocation.phase),
            stage=raw_relocation.stage,
            relocation_index=raw_relocation.relocation_index,
            thresholds_percent=raw_relocation.thresholds_percent,
            hide_duration_ms=raw_relocation.hide_duration_ms,
            remaining_ms=raw_relocation.remaining_ms,
            burrowed=raw_relocation.burrowed,
            provenance=_field_provenance(
                THRESHOLD_RELOCATION_STATE_FIELDS,
                {
                    "phase": SemanticEvidenceLevel.NATIVE_DERIVED,
                    "stage": SemanticEvidenceLevel.NATIVE_AUTHORITATIVE,
                    "relocation_index": SemanticEvidenceLevel.NATIVE_DERIVED,
                    "thresholds_percent": (SemanticEvidenceLevel.NATIVE_AUTHORITATIVE),
                    "hide_duration_ms": (SemanticEvidenceLevel.NATIVE_AUTHORITATIVE),
                    "remaining_ms": SemanticEvidenceLevel.NATIVE_AUTHORITATIVE,
                    "burrowed": SemanticEvidenceLevel.NATIVE_DERIVED,
                },
                {
                    "phase": ("objects[].thresholdRelocationRuntime.phase",),
                    "stage": ("objects[].thresholdRelocationRuntime.stage",),
                    "relocation_index": ("objects[].thresholdRelocationRuntime.relocationIndex",),
                    "thresholds_percent": ("objects[].thresholdRelocationRuntime.thresholdsPercent",),
                    "hide_duration_ms": ("objects[].thresholdRelocationRuntime.hideDurationMs",),
                    "remaining_ms": ("objects[].thresholdRelocationRuntime.remainingMs",),
                    "burrowed": ("objects[].thresholdRelocationRuntime.burrowed",),
                },
                observed_tick=observed_tick,
            ),
        )

    shield_state: ShieldStateV1 | None = None
    shield_value: float | None = None
    if item.shield_current is not None and item.shield_maximum is not None:
        shield_value = float(item.shield_current)
        shield_state = ShieldStateV1(
            hitpoints=float(item.shield_current),
            max_hitpoints=float(item.shield_maximum),
            provenance=_field_provenance(
                SHIELD_STATE_FIELDS,
                {
                    "hitpoints": SemanticEvidenceLevel.NATIVE_AUTHORITATIVE,
                    "max_hitpoints": SemanticEvidenceLevel.NATIVE_AUTHORITATIVE,
                },
                {"hitpoints": ("objects[].shield.current",), "max_hitpoints": ("objects[].shield.max",)},
                observed_tick=observed_tick,
            ),
        )

    typed_effects: list[EffectStateV1] = []
    for index, effect in enumerate(item.active_effects or ()):
        source_entity: int | None = None
        source_owner: int | None = None
        source_card_id: int | None = None
        source_evidence = SemanticEvidenceLevel.UNKNOWN
        source_owner_evidence = SemanticEvidenceLevel.UNKNOWN
        source_card_evidence = SemanticEvidenceLevel.UNKNOWN
        source_key = effect.source_entity_key
        effect_notes: list[str] = []
        if source_key is None and effect.source_entity_validated:
            source_evidence = SemanticEvidenceLevel.NATIVE_AUTHORITATIVE
            effect_notes.append("validated native null source pointer")
        elif source_key is None:
            effect_notes.append(
                "sourceEntityKey=null is unavailable because the non-null native pointer "
                "did not resolve inside the bounded object vector"
            )
        elif source_key in visible_keys:
            source_entity = canonical_ids[source_key]
            source_owner = source_key[0] if source_key[0] in (0, 1) else None
            source_item = rich_objects[source_key]
            source_card_id = source_item.card_id if source_item.card_id > 0 else None
            source_evidence = SemanticEvidenceLevel.NATIVE_DERIVED
            source_owner_evidence = (
                SemanticEvidenceLevel.NATIVE_AUTHORITATIVE
                if source_owner is not None
                else SemanticEvidenceLevel.UNKNOWN
            )
            source_card_evidence = (
                SemanticEvidenceLevel.NATIVE_DERIVED if source_card_id is not None else SemanticEvidenceLevel.UNKNOWN
            )
        else:
            effect_notes.append("effect source identity suppressed by FAIR visibility")

        kind, kind_evidence, attributes, catalog_notes = _effect_attributes(
            effect, index=index, effect_catalog=effect_catalog
        )
        non_expiring = effect.remaining_ms == -1
        typed_effects.append(
            EffectStateV1(
                effect_id=f"buff:{effect.buff_global_id}",
                kind=kind,
                source_entity=source_entity,
                source_card_id=source_card_id,
                source_owner=source_owner,
                remaining_ms=None if non_expiring else effect.remaining_ms,
                active=True,
                attributes=attributes,
                provenance=_field_provenance(
                    EFFECT_STATE_FIELDS,
                    {
                        "kind": kind_evidence,
                        "source_entity": source_evidence,
                        "source_card_id": source_card_evidence,
                        "source_owner": source_owner_evidence,
                        "remaining_ms": (
                            SemanticEvidenceLevel.NOT_APPLICABLE
                            if non_expiring
                            else SemanticEvidenceLevel.NATIVE_AUTHORITATIVE
                        ),
                        "active": SemanticEvidenceLevel.NATIVE_AUTHORITATIVE,
                    },
                    {
                        "kind": ("static BUFF catalog mechanic_tags",),
                        "source_entity": (
                            "objects[].activeEffects[].sourceEntityKey",
                            "objects[].activeEffects[].sourceEntityValidated",
                        ),
                        "source_owner": ("objects[].activeEffects[].sourceEntityKey[0]",),
                        "remaining_ms": ("objects[].activeEffects[].remainingMs",),
                        "active": ("objects[].activeEffects[]",),
                    },
                    observed_tick=observed_tick,
                    notes=tuple((*effect_notes, *catalog_notes)),
                ),
            )
        )

    target_entity: int | None = None
    target_evidence = SemanticEvidenceLevel.UNKNOWN
    attack_notes: list[str] = []
    if item.target_entity_validated:
        if item.target_entity_key is None:
            target_evidence = SemanticEvidenceLevel.NATIVE_AUTHORITATIVE
        elif item.target_entity_key in visible_keys:
            target_entity = canonical_ids[item.target_entity_key]
            target_evidence = SemanticEvidenceLevel.NATIVE_DERIVED
        else:
            attack_notes.append("target entity identity suppressed by FAIR visibility")

    phase_snapshot = item.phase_runtime
    phase_hooks_ready = bool(
        phase_runtime is not None
        and phase_runtime.hook_set_attested
        and phase_runtime.hook_set_installed
        and phase_runtime.window.complete
        and phase_runtime.window.rejected_count == 0
    )
    resolver_events = tuple(
        phase_events if phase_events is not None else phase_runtime.events if phase_runtime is not None else ()
    )

    attack_state: AttackStateV1 | None = None
    movement_runtime: MovementRuntimeV1 | None = None
    deployment_runtime: DeploymentRuntimeV1 | None = None
    if phase_snapshot is not None:
        if (
            phase_snapshot.attack_sequence_stage is not None
            and item.attack_sequence_stage is not None
            and phase_snapshot.attack_sequence_stage != item.attack_sequence_stage
        ):
            raise RichTelemetryMergeError(f"slot {item.slot} phase/legacy attack stage snapshots disagree")
        phase_stage = (
            phase_snapshot.attack_sequence_stage
            if phase_snapshot.attack_sequence_stage is not None
            else item.attack_sequence_stage
        )
        target_snapshot_validated = item.target_entity_validated and (
            item.target_entity_key is None or item.target_entity_key in visible_keys
        )
        try:
            attack_state = resolve_attack_runtime(
                AttackRuntimeRaw(
                    entity_key=item.entity_key,
                    tick=observed_tick,
                    target_validated=target_snapshot_validated,
                    target_entity=target_entity,
                    attack_sequence_stage=phase_stage,
                    attack_timeline_ms=phase_snapshot.attack_timeline_ms,
                    load_remaining_ms=phase_snapshot.load_remaining_ms,
                    hit_speed_ms=phase_snapshot.hit_speed_ms,
                    attack_dash_time_ms=phase_snapshot.attack_dash_time_ms,
                    attack_step_native_ms=(
                        phase_snapshot.attack_step_output if phase_snapshot.attack_step_tick == observed_tick else None
                    ),
                    hook_set_attested=phase_hooks_ready,
                    attack_sequence_progress=(phase_snapshot.attack_sequence_progress_raw),
                    attack_sequence_progress_limit=(phase_snapshot.attack_sequence_progress_limit),
                    attack_sequence_decay_remaining_ms=(phase_snapshot.attack_sequence_decay_remaining_ms),
                    attack_sequence_decay_duration_ms=(phase_snapshot.attack_sequence_decay_duration_ms),
                    events=resolver_events if phase_hooks_ready else (),
                    damage_ramp_spec=_damage_ramp_spec(
                        card_spec, observed_stage=phase_stage, evolution_active=evolution_active
                    ),
                )
            ).state
            movement_runtime = resolve_movement_runtime(
                MovementRuntimeRaw(
                    entity_key=item.entity_key,
                    tick=observed_tick,
                    component_validated=phase_snapshot.movement_validated,
                    hook_set_attested=phase_hooks_ready,
                    movement_delta=(
                        phase_snapshot.movement_delta if phase_snapshot.movement_delta_tick == observed_tick else None
                    ),
                    effective_speed=(
                        phase_snapshot.effective_movement_speed
                        if phase_snapshot.effective_movement_speed_tick == observed_tick
                        else None
                    ),
                    speed_input=(
                        phase_snapshot.movement_step_input
                        if phase_snapshot.movement_step_tick == observed_tick
                        else None
                    ),
                    speed_after_effects=(
                        phase_snapshot.movement_step_output
                        if phase_snapshot.movement_step_tick == observed_tick
                        else None
                    ),
                    classic_charge_progress=(phase_snapshot.classic_charge_progress),
                    classic_charge_speed_multiplier=(phase_snapshot.charge_speed_multiplier),
                )
            )
            deploy_uses_scaled_step: bool | None = None
            if phase_snapshot.deploy_step_tick == observed_tick:
                deploy_uses_scaled_step = True
            elif (
                phase_snapshot.deploy_remaining_ms is not None
                and phase_snapshot.deploy_previous_ms is not None
                and (
                    phase_snapshot.deploy_remaining_ms == 0
                    or phase_snapshot.deploy_previous_ms - phase_snapshot.deploy_remaining_ms == 50
                )
            ):
                deploy_uses_scaled_step = False
            deployment_runtime = resolve_deployment_runtime(
                DeploymentRuntimeRaw(
                    tick=observed_tick,
                    component_validated=phase_snapshot.attack_validated,
                    remaining_native_ms=phase_snapshot.deploy_remaining_ms,
                    previous_remaining_native_ms=(phase_snapshot.deploy_previous_ms),
                    configured_deploy_time_ms=(phase_snapshot.configured_deploy_time_ms),
                    uses_effect_scaled_step=deploy_uses_scaled_step,
                    observed_step_native_ms=(
                        phase_snapshot.deploy_step_output if phase_snapshot.deploy_step_tick == observed_tick else None
                    ),
                )
            )
        except PhaseRuntimeError as error:
            raise RichTelemetryMergeError(
                f"slot {item.slot} phase runtime contradicts its exact contract: {error}"
            ) from error
    elif item.target_entity_validated or item.attack_sequence_stage is not None:
        stage_evidence = (
            SemanticEvidenceLevel.NATIVE_AUTHORITATIVE
            if item.attack_sequence_stage is not None
            else SemanticEvidenceLevel.UNKNOWN
        )
        attack_state = AttackStateV1(
            target_entity=target_entity,
            sequence_index=item.attack_sequence_stage,
            charge_stage=None,
            provenance=_field_provenance(
                ATTACK_STATE_FIELDS,
                {
                    "target_entity": target_evidence,
                    "sequence_index": stage_evidence,
                    "charge_stage": SemanticEvidenceLevel.UNKNOWN,
                },
                {"target_entity": ("objects[].targetEntityKey",), "sequence_index": ("objects[].attackSequenceStage",)},
                observed_tick=observed_tick,
                notes=tuple(
                    (*attack_notes, "attackSequenceStage is not chargeStage; native charge stage is unavailable")
                ),
            ),
        )

    visibility_state: VisibilityStateV1 | None = None
    if item.invisible_count is not None:
        visibility_state = VisibilityStateV1(
            phase=(VisibilityPhase.HIDDEN if item.invisible else VisibilityPhase.VISIBLE),
            provenance=_field_provenance(
                VISIBILITY_STATE_FIELDS,
                {"phase": SemanticEvidenceLevel.NATIVE_DERIVED},
                {"phase": ("objects[].invisibleCount", "objects[].visibilityState")},
                observed_tick=observed_tick,
                notes=(
                    f"native invisibleCount={item.invisible_count}",
                    "visible means the validated native invisibility counter is zero",
                    "phase describes native unit interaction, not player-visible entity existence",
                    "all deployed arena entities remain public in FAIR observations",
                ),
            ),
        )

    return RuntimeProjection(
        shield=shield_value,
        shield_state=shield_state,
        effect_states=tuple(typed_effects),
        attack_state=attack_state,
        movement_runtime=movement_runtime,
        deployment_runtime=deployment_runtime,
        projectile_state=projectile_state,
        visibility_state=visibility_state,
        resource_states=resource_states,
        periodic_attack_modifier=periodic_attack_modifier,
        capture_runtime=capture_runtime,
        threshold_relocation_runtime=threshold_relocation_runtime,
        target_entity=target_entity,
        invisible_count=item.invisible_count,
        domain_evidence={
            "shield_state": (
                SemanticEvidenceLevel.NATIVE_AUTHORITATIVE
                if shield_state is not None
                else SemanticEvidenceLevel.UNKNOWN
            ),
            "effect_states": (
                SemanticEvidenceLevel.NATIVE_AUTHORITATIVE
                if item.active_effects is not None
                else SemanticEvidenceLevel.UNKNOWN
            ),
            "attack_state": (
                SemanticEvidenceLevel.NATIVE_DERIVED if attack_state is not None else SemanticEvidenceLevel.UNKNOWN
            ),
            "movement_runtime": (
                SemanticEvidenceLevel.NATIVE_DERIVED if movement_runtime is not None else SemanticEvidenceLevel.UNKNOWN
            ),
            "deployment_runtime": (
                SemanticEvidenceLevel.NATIVE_DERIVED
                if deployment_runtime is not None
                else SemanticEvidenceLevel.UNKNOWN
            ),
            "projectile_state": (
                SemanticEvidenceLevel.NATIVE_DERIVED if projectile_state is not None else SemanticEvidenceLevel.UNKNOWN
            ),
            "visibility_state": (
                SemanticEvidenceLevel.NATIVE_DERIVED if visibility_state is not None else SemanticEvidenceLevel.UNKNOWN
            ),
            "resource_states": (
                SemanticEvidenceLevel.NATIVE_AUTHORITATIVE if resource_states else SemanticEvidenceLevel.UNKNOWN
            ),
            "periodic_attack_modifier": (
                SemanticEvidenceLevel.NATIVE_DERIVED
                if periodic_attack_modifier is not None
                else SemanticEvidenceLevel.UNKNOWN
            ),
            "capture_runtime": (
                SemanticEvidenceLevel.NATIVE_DERIVED if capture_runtime is not None else SemanticEvidenceLevel.UNKNOWN
            ),
            "threshold_relocation_runtime": (
                SemanticEvidenceLevel.NATIVE_DERIVED
                if threshold_relocation_runtime is not None
                else SemanticEvidenceLevel.UNKNOWN
            ),
        },
    )
