#!/usr/bin/env python3
"""Host API for headless and stock-rendered native Clash Royale matches.

Both modes run the real ARM64 battle engine inside the isolated stock client.
Headless mode owns a detached manager for exact, fast stepping; native-render
mode lets the client's own battle scene own and draw the manager in real time.
Actions and structured observations use the native command/state APIs in both
modes, without screenshot interpretation or ADB touch injection.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import socket
import threading
import time
from typing import Any, Iterable, Mapping


from .action_movement_runtime import ActionMovementRuntimeError, parse_action_movement_runtime
from .attestation import attestation_from_response, verify_runner_attestation
from .character_state_runtime import CharacterStateRuntimeError, parse_character_state_runtime
from .contracts import RunnerAttestationV1
from .match_factory import MatchConfig
from .special_movement_runtime import SpecialMovementRuntimeError, parse_special_movement_runtime


from .local_config import setting

DEFAULT_HOST = setting("CR_CONTROL_HOST", "127.0.0.1")
DEFAULT_PORT = int(setting("CR_CONTROL_PORT", "26789"))
MAX_TRANSITION_ACTIONS = 8
FIRST_PLAYABLE_TICK = 90
LIVE_COMMAND_AGE_TICKS = 20
LIVE_TOUCH_MAX_FUTURE_TICKS = 100
LIVE_TOUCH_MAX_STALE_TICKS = 20
LIVE_TOUCH_RETRY_LEAD_TICKS = 10
COMMAND_CONSUMPTION_STEPS = LIVE_COMMAND_AGE_TICKS + 1
NATIVE_RENDER_SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0)
RICH_TELEMETRY_SCHEMA = "native-rich-telemetry.v3"
COMBAT_EVENT_SCHEMA = "native-combat-events.v1"
PHASE_RUNTIME_SCHEMA = "native-phase-runtime.v1"
VISIBILITY_RUNTIME_SCHEMA = "native-visibility-runtime.v1"
REMAINING_RUNTIME_SCHEMA = "native-remaining-runtime.v1"
PHASE_OBJECT_SCHEMA = "native-phase-object.v1"
ATTACK_SEQUENCE_PROGRESS_LIMIT = 50
ATTACK_SEQUENCE_DECAY_DURATION_MS = 7_000
ATOMIC_OBSERVATION_SCHEMA = "native-observation-capture.v1"
RICH_TELEMETRY_STATUS_ENUM = ("authoritative", "derived", "pending", "unavailable")
MAX_OBSERVATION_OBJECTS = 256
MAX_RICH_OBJECTS = MAX_OBSERVATION_OBJECTS
NATIVE_OBJECT_ID_ENTITY_KEY_TAG = -2
MAX_RUNNER_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_NATIVE_OBJECT_SLOTS = 100_000
MAX_NATIVE_COMPONENT_SLOTS = 64
MAX_ACTIVE_EFFECTS = 64
MAX_ACTIVE_EFFECT_NAME_BYTES = 128
MAX_ATTACK_SEQUENCE_STAGE = 255
MAX_CHAMPION_CONTROLLERS = 2
MAX_DECK_SLOTS = 8
MAX_COMBAT_EVENTS = 1024
MAX_PHASE_EVENTS = 4096
MAX_VISIBILITY_EVENTS = 4096
MAX_REMAINING_EVENTS = 4096
PHASE_EVENT_HOOKS = {
    "attack_start": 0xF23110,
    "attack_release": 0xF5F100,
    "effect_apply": 0xF5A2D4,
    "attack_scale": 0xF5B4AC,
    "movement_scale": 0xF5B3C8,
    "effective_movement_speed": 0xF1D004,
    "movement_delta": 0xF68808,
    "target_reset": 0xF5C894,
    # Classic-charge readiness is observed by the generic action dispatcher;
    # the exact caller return address (0xF68958) disambiguates it from an
    # ordinary attack start.  It is not emitted by the movement-delta hook.
    "classic_charge_ready": 0xF23110,
    "deploy_scale": 0xF5B3C8,
}
VISIBILITY_EVENT_HOOKS = {"became_invisible": 0xF5A2D4, "became_visible": 0xF5A578}
VISIBILITY_EVENT_CALLERS = {"became_invisible": 0xF59EAC, "became_visible": 0xF59308}
REMAINING_EVENT_HOOKS = {
    "area_create": {0xF25B8C},
    "area_expire": {0xF25D90},
    "resource_delta": {0xF3BA4C, 0xF3B3D4},
    "tower_aggro_acquire": {0xF5C894},
    "tower_aggro_change": {0xF5C894},
    "tower_aggro_lose": {0xF5C894},
    "tower_activate": {0xF23110},
    "lifetime_spawn": {0xF25B8C},
    "lifetime_despawn": {0xF25D90},
    "transform": {0xE58638},
    "spawn_attach": {0xF19314},
    "area_damage_eligibility": {0xF1CB90},
    "area_relation_parent": {0xF25B8C},
    "area_relation_follow": {0xF25B8C},
    "area_relation_related": {0xF25B8C},
    # Both producers expose the same exact area->child edge.  The AEO tick
    # path owns area_tick_nested_spawn; direct ActionSpawnToLocation mode 1
    # owns action_spawn_to_location (used by Graveyard).
    "area_action_spawn": {0xF143A8, 0xE7B864},
}
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
COMBAT_EVENT_POOLS = frozenset({"none", "hitpoints", "built_in_shield", "buff_shield"})
COMBAT_TERMINAL_REASONS = frozenset({"none", "object_impact", "source_lost", "nonimpact_cancel", "unknown"})
COMBAT_CONSUME_CARD_HOOK_OFFSET = 0xF38A68
COMBAT_HOOK_OFFSETS = {
    "damage": {0xF642C4},
    "death": {0xF642C4},
    "heal": {0xF64D14},
    "spawn": {0xF25B8C},
    "projectile_spawn": {0xF25B8C},
    "despawn": {0xF25D90},
    "shield_damage": {0xF642C4},
    "shield_break": {0xF642C4},
    "projectile_impact": {0xF2981C},
    "projectile_deflect": {0xF1629C},
    "projectile_expire": {0xF2A11C, 0xF2A148},
    "projectile_terminal": {0xF28470},
    "card_play": {COMBAT_CONSUME_CARD_HOOK_OFFSET},
}
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
CARD_FORM_LABELS = ("BasicForm", "EvoForm", "HeroForm", "AutoChessForm", "FormEnd", "FlexSlot")


class RunnerError(RuntimeError):
    """Raised when the native runner rejects a request or is unavailable."""


@dataclass(frozen=True, slots=True)
class NativeCardDescriptorV1:
    """Exact public fields decoded from one native 20-byte card descriptor.

    The probe's native selection builder already validates the descriptor's
    data pointer and encoded deck slot.  This host-side view preserves the
    remaining packed command provenance, most importantly the consumed form.
    """

    packed: int
    deck_slot: int
    cost: int
    form_code: int
    form_name: str


def decode_native_card_parameter(
    card_parameter: object, *, expected_deck_slot: int | None = None, expected_cost: int | None = None
) -> NativeCardDescriptorV1:
    """Decode and fail-close one exact native card-selection packed word.

    Exact-build RE proves form in bits 0..3, deck slot plus one in bits
    22..27, and elixir cost in bits 28..31.  Other packed bits remain opaque
    and are deliberately retained without interpretation.
    """

    if isinstance(card_parameter, bool) or not isinstance(card_parameter, int) or not 0 <= card_parameter <= 0xFFFFFFFF:
        raise ValueError("card_parameter must be a uint32")
    form_code = card_parameter & 0xF
    if form_code >= len(CARD_FORM_LABELS):
        raise ValueError("card_parameter carries an unknown native form code")
    deck_slot = ((card_parameter >> 22) & 0x3F) - 1
    if not 0 <= deck_slot < MAX_DECK_SLOTS:
        raise ValueError("card_parameter carries an invalid native deck slot")
    cost = card_parameter >> 28
    if expected_deck_slot is not None and deck_slot != expected_deck_slot:
        raise ValueError("card_parameter deck slot does not match observation")
    if expected_cost is not None and cost != expected_cost:
        raise ValueError("card_parameter cost does not match observation")
    return NativeCardDescriptorV1(
        packed=card_parameter,
        deck_slot=deck_slot,
        cost=cost,
        form_code=form_code,
        form_name=CARD_FORM_LABELS[form_code],
    )


def _decode_observed_hand_card(card: Mapping[str, Any]) -> NativeCardDescriptorV1:
    try:
        deck_slot = _response_int(
            card.get("deckSlot"), "selected native hand deckSlot", minimum=0, maximum=MAX_DECK_SLOTS - 1
        )
        cost = _response_int(card.get("cost"), "selected native hand cost", minimum=0, maximum=15)
        return decode_native_card_parameter(card.get("cardParameter"), expected_deck_slot=deck_slot, expected_cost=cost)
    except ValueError as error:
        raise RunnerError("selected native hand card descriptor failed exact packed validation") from error


def _observed_command_card_id(card: Mapping[str, Any]) -> int:
    """Return the exact card ID carried in a native deploy command.

    Ordinary selections use their visible deck card ID. Mirror keeps the
    visible/root card as Mirror while the native selection's secondary data
    supplies the effective card expected in the command ``os`` field.
    """

    value = card.get("commandCardId", card.get("cardId"))
    return _response_int(value, "selected native hand commandCardId", minimum=1, maximum=0xFFFFFFFF)


def _response_int(value: object, label: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RunnerError(f"invalid {label} in native runner response")
    if minimum is not None and value < minimum:
        raise RunnerError(f"invalid {label} in native runner response")
    if maximum is not None and value > maximum:
        raise RunnerError(f"invalid {label} in native runner response")
    return value


def _response_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise RunnerError(f"invalid {label} in native runner response")
    return value


def _response_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RunnerError(f"invalid {label} in native runner response")
    return value


class _Fields:
    """Validate a mapping's fields while keeping their full protocol labels."""

    __slots__ = ("data", "label")

    def __init__(self, data: Mapping[str, Any], label: str):
        self.data = data
        self.label = label

    def integer(self, name, *, minimum=None, maximum=None, default=None):
        return _response_int(self.data.get(name, default), f"{self.label}.{name}", minimum=minimum, maximum=maximum)

    def optional_int(self, name, *, minimum=None, maximum=None, default=None):
        value = self.data.get(name, default)
        return None if value is None else _response_int(value, f"{self.label}.{name}", minimum=minimum, maximum=maximum)

    def boolean(self, name):
        return _response_bool(self.data.get(name), f"{self.label}.{name}")

    def mapping(self, name):
        return _response_mapping(self.data.get(name), f"{self.label}.{name}")

    def key(self, name):
        return _validate_entity_key(self.data.get(name), f"{self.label}.{name}")


def _validate_entity_key(value: object, label: str) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise RunnerError(f"invalid {label} in native runner response")
    return tuple(_response_int(part, f"{label}[{index}]") for index, part in enumerate(value))  # type: ignore[return-value]


def _expected_native_entity_key(
    *, owner: int, object_index: int, secondary_index: int, native_object_id: int
) -> tuple[int, int, int]:
    del object_index, secondary_index
    return (owner, NATIVE_OBJECT_ID_ENTITY_KEY_TAG, native_object_id)


def _validate_provenance_record(value: object, label: str) -> None:
    record = _response_mapping(value, label)
    record_check = _Fields(record, label)
    if record.get("status") not in RICH_TELEMETRY_STATUS_ENUM:
        raise RunnerError(f"invalid {label}.status in native runner response")
    for field in ("source", "validation", "confidence"):
        if not isinstance(record.get(field), str) or not record[field]:
            raise RunnerError(f"invalid {label}.{field} in native runner response")
    record_check.boolean("failClosed")


def _validate_optional_entity_relation(
    raw_key: object, raw_validated: object, label: str
) -> tuple[int, int, int] | None:
    validated = _response_bool(raw_validated, f"{label}Validated")
    if raw_key is None:
        return None
    key = _validate_entity_key(raw_key, label)
    if not validated:
        raise RunnerError(f"{label} exposed an unvalidated entity identity")
    return key


def _validate_projectile(value: object, label: str) -> tuple[tuple[str, tuple[int, int, int]], ...]:
    if value is None:
        return ()
    projectile = _response_mapping(value, label)
    projectile_check = _Fields(projectile, label)
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
    if set(projectile) != required:
        raise RunnerError(f"{label} fields do not match the v2 projectile schema")
    projectile_check.integer("projectileDataGlobalId", minimum=1, maximum=0xFFFFFFFF)
    source = _validate_optional_entity_relation(
        projectile.get("sourceEntityKey"), projectile.get("sourceEntityValidated"), f"{label}.sourceEntityKey"
    )
    target = _validate_optional_entity_relation(
        projectile.get("targetEntityKey"), projectile.get("targetEntityValidated"), f"{label}.targetEntityKey"
    )
    homing_target = _validate_optional_entity_relation(
        projectile.get("homingTargetEntityKey"),
        projectile.get("homingTargetEntityValidated"),
        f"{label}.homingTargetEntityKey",
    )
    projectile_check.integer("destinationX")
    projectile_check.integer("destinationY")
    terminal = projectile_check.boolean("terminal")
    expected_phase = "terminal_or_finished_processing" if terminal else "in_flight"
    if projectile.get("nativePhase") != expected_phase:
        raise RunnerError(f"{label}.nativePhase does not match its terminal bit")
    drag_stage = projectile.get("dragStage")
    if drag_stage not in {None, "outbound", "drag_back_active"}:
        raise RunnerError(f"{label}.dragStage is invalid")
    references: list[tuple[str, tuple[int, int, int]]] = []
    if source is not None:
        references.append((f"{label}.sourceEntityKey", source))
    if target is not None:
        references.append((f"{label}.targetEntityKey", target))
    if homing_target is not None:
        references.append((f"{label}.homingTargetEntityKey", homing_target))
    return tuple(references)


def _validate_entity_resource(value: object, label: str) -> None:
    if value is None:
        return
    resource = _response_mapping(value, label)
    resource_check = _Fields(resource, label)
    required = {"kind", "currentRaw", "capacityRaw", "baseRaw", "limitRaw", "normalized", "status"}
    if set(resource) != required:
        raise RunnerError(f"{label} fields do not match the resource schema")
    if resource.get("kind") != "extra_spawn_accumulator":
        raise RunnerError(f"{label}.kind is invalid")
    if resource.get("status") != "authoritative":
        raise RunnerError(f"{label}.status must be authoritative")
    current = resource_check.integer("currentRaw", minimum=0)
    capacity = resource_check.integer("capacityRaw", minimum=1)
    base = resource_check.integer("baseRaw", minimum=1)
    limit = resource_check.integer("limitRaw", minimum=2)
    normalized_raw = resource.get("normalized")
    if isinstance(normalized_raw, bool) or not isinstance(normalized_raw, (int, float)):
        raise RunnerError(f"{label}.normalized must be numeric")
    normalized = float(normalized_raw)
    if (
        limit - base != capacity
        or current > capacity
        or not math.isfinite(normalized)
        or abs(normalized - current / capacity) > 1e-6
    ):
        raise RunnerError(f"{label} values are inconsistent")


def _validate_periodic_attack_modifier_runtime(
    value: object, label: str
) -> tuple[tuple[str, tuple[int, int, int]], ...]:
    if value is None:
        return ()
    modifier = _response_mapping(value, label)
    modifier_check = _Fields(modifier, label)
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
    if set(modifier) != required:
        raise RunnerError(f"{label} fields do not match the modifier schema")
    if modifier.get("schema") != "native-periodic-attack-modifier-runtime.v1":
        raise RunnerError(f"{label}.schema is invalid")
    if modifier.get("status") != "authoritative":
        raise RunnerError(f"{label}.status must be authoritative")
    modifier_check.integer("actionDataGlobalId", minimum=1, maximum=0xFFFFFFFF)
    period = modifier_check.integer("periodAttacks", minimum=1)
    completed = modifier_check.integer("completedAttacks", minimum=0)
    if completed >= period:
        raise RunnerError(f"{label}.completedAttacks is outside period")
    modifier_check.integer("addedDamageRaw", minimum=0)
    linger_duration = modifier_check.integer("lingerDurationMs", minimum=1)
    linger_remaining = modifier_check.integer("lingerRemainingMs", minimum=0, maximum=linger_duration)
    phase = modifier.get("phase")
    if phase not in {"source_alive", "source_death_linger"}:
        raise RunnerError(f"{label}.phase is invalid")
    if (phase == "source_alive") != (linger_remaining == 0):
        raise RunnerError(f"{label} phase/linger timer is inconsistent")
    modifier_check.integer("sourceNativeObjectId", minimum=1, maximum=0xFFFFFFFF)
    source_resolved = modifier_check.boolean("sourceResolved")
    raw_source_key = modifier.get("sourceEntityKey")
    if source_resolved != (raw_source_key is not None):
        raise RunnerError(f"{label} source resolution fields disagree")
    if phase == "source_alive" and not source_resolved:
        raise RunnerError(f"{label} live source must resolve")
    if not source_resolved:
        return ()
    source_key = _validate_entity_key(raw_source_key, f"{label}.sourceEntityKey")
    return ((f"{label}.sourceEntityKey", source_key),)


def _validate_capture_runtime(value: object, label: str) -> tuple[tuple[str, tuple[int, int, int]], ...]:
    if value is None:
        return ()
    capture = _response_mapping(value, label)
    capture_check = _Fields(capture, label)
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
    if set(capture) != required:
        raise RunnerError(f"{label} fields do not match the capture schema")
    if capture.get("schema") != "native-capture-runtime.v1":
        raise RunnerError(f"{label}.schema is invalid")
    if capture.get("status") != "authoritative":
        raise RunnerError(f"{label}.status must be authoritative")
    capture_check.integer("actionDataGlobalId", minimum=1, maximum=0xFFFFFFFF)
    drag_delay = capture_check.integer("dragDelayMs", minimum=0)
    grab_pause = capture_check.integer("grabPauseMs", minimum=0)
    capture_drag_time = capture_check.integer("captureDragTimeMs", minimum=0)
    configured_cooldown = capture_check.integer("configuredCooldownMs", minimum=0)
    cooldown_remaining = capture_check.integer("cooldownRemainingMs", minimum=0)
    capture_check.integer("hitFrequencyMs", minimum=1)
    capture_check.integer("hitAccumulatorMs", minimum=0)
    if cooldown_remaining > configured_cooldown:
        raise RunnerError(f"{label} timers are inconsistent")
    capture_check.boolean("completionResultCurrentUpdate")
    capture_check.boolean("firstCaptureHandled")
    raw_targets = capture.get("targets")
    if not isinstance(raw_targets, list) or len(raw_targets) > 160:
        raise RunnerError(f"{label}.targets is invalid")
    action_phase = capture.get("phase")
    expected_action_phase = "active" if raw_targets else "release_cooldown" if cooldown_remaining > 0 else "idle"
    if action_phase != expected_action_phase:
        raise RunnerError(f"{label}.phase is inconsistent")

    phases_with_remaining = {"acquired_delay", "grab_pause", "dragging"}
    phases_without_remaining = {"contained", "release_pending"}
    seen_native_ids: set[int] = set()
    references: list[tuple[str, tuple[int, int, int]]] = []
    for index, value in enumerate(raw_targets):
        target_label = f"{label}.targets[{index}]"
        target = _response_mapping(value, target_label)
        target_check = _Fields(target, target_label)
        if set(target) != {
            "targetNativeObjectId",
            "targetEntityKey",
            "targetResolved",
            "phase",
            "elapsedMs",
            "phaseBudgetRemainingMs",
        }:
            raise RunnerError(f"{target_label} fields do not match the target schema")
        native_id = target_check.integer("targetNativeObjectId", minimum=1, maximum=0xFFFFFFFF)
        if native_id in seen_native_ids:
            raise RunnerError(f"{label}.targets contains a duplicate native identity")
        seen_native_ids.add(native_id)
        resolved = target_check.boolean("targetResolved")
        raw_key = target.get("targetEntityKey")
        if resolved != (raw_key is not None):
            raise RunnerError(f"{target_label} resolution fields disagree")
        if resolved:
            key = _validate_entity_key(raw_key, f"{target_label}.targetEntityKey")
            references.append((f"{target_label}.targetEntityKey", key))
        phase = target.get("phase")
        if phase not in phases_with_remaining | phases_without_remaining:
            raise RunnerError(f"{target_label}.phase is invalid")
        if (phase == "release_pending") != (not resolved):
            raise RunnerError(f"{target_label} phase/resolution fields disagree")
        elapsed = target_check.integer("elapsedMs", minimum=0)
        remaining_raw = target.get("phaseBudgetRemainingMs")
        if phase in phases_with_remaining:
            remaining = _response_int(remaining_raw, f"{target_label}.phaseBudgetRemainingMs", minimum=0)
            phase_boundary = {
                "acquired_delay": drag_delay,
                "grab_pause": drag_delay + grab_pause,
                "dragging": drag_delay + grab_pause + capture_drag_time,
            }[phase]
            expected_remaining = max(phase_boundary - elapsed, 0)
            if remaining != expected_remaining:
                raise RunnerError(f"{target_label} phase timer is inconsistent")
        elif remaining_raw is not None:
            raise RunnerError(f"{target_label}.phaseBudgetRemainingMs must be null")
    return tuple(references)


def _validate_threshold_relocation_runtime(value: object, label: str) -> None:
    if value is None:
        return
    relocation = _response_mapping(value, label)
    relocation_check = _Fields(relocation, label)
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
    if set(relocation) != required:
        raise RunnerError(f"{label} fields do not match the relocation schema")
    if relocation.get("schema") != "native-threshold-relocation-runtime.v1":
        raise RunnerError(f"{label}.schema is invalid")
    if relocation.get("status") != "authoritative":
        raise RunnerError(f"{label}.status must be authoritative")
    relocation_check.integer("actionDataGlobalId", minimum=1, maximum=0xFFFFFFFF)
    raw_thresholds = relocation.get("thresholdsPercent")
    if not isinstance(raw_thresholds, list) or not 1 <= len(raw_thresholds) <= 16:
        raise RunnerError(f"{label}.thresholdsPercent is invalid")
    thresholds = tuple(
        _response_int(item, f"{label}.thresholdsPercent[{index}]", minimum=0, maximum=100)
        for index, item in enumerate(raw_thresholds)
    )
    if any(right >= left for left, right in zip(thresholds, thresholds[1:])):
        raise RunnerError(f"{label}.thresholdsPercent must be strictly descending")
    stage = relocation_check.integer("stage", minimum=1, maximum=2 * len(thresholds) + 1)
    relocation_index = relocation_check.integer("relocationIndex", minimum=0, maximum=len(thresholds))
    if relocation_index != (stage - 1) // 2:
        raise RunnerError(f"{label}.relocationIndex is inconsistent")
    terminal_stage = 2 * len(thresholds) + 1
    expected_phase = "exhausted" if stage == terminal_stage else "relocating" if stage % 2 == 0 else "waiting_threshold"
    if relocation.get("phase") != expected_phase:
        raise RunnerError(f"{label}.phase is inconsistent")
    hide_duration = relocation_check.integer("hideDurationMs", minimum=1)
    remaining = relocation_check.integer("remainingMs", minimum=0)
    if remaining > hide_duration:
        raise RunnerError(f"{label}.remainingMs exceeds its duration")
    burrowed = relocation_check.boolean("burrowed")
    if burrowed and (expected_phase != "relocating" or remaining == 0):
        raise RunnerError(f"{label}.burrowed is inconsistent")


def _validate_combat_entity_fact(value: object, label: str) -> Mapping[str, Any]:
    fact = _response_mapping(value, label)
    fact_check = _Fields(fact, label)
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
        raise RunnerError(f"{label} fields do not match the combat entity schema")
    validated = fact_check.boolean("validated")
    present = fact_check.boolean("present")
    visibility_validated = fact_check.boolean("visibilityValidated")
    if not present:
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
        ):
            if fact.get(name) is not None:
                raise RunnerError(f"{label} exposes {name} for an absent relation")
        if visibility_validated:
            raise RunnerError(f"{label} validates visibility for an absent relation")
        return fact
    if not validated:
        raise RunnerError(f"{label} exposes an unvalidated native entity")
    native_object_id = fact_check.integer("nativeObjectId", minimum=1)
    key = fact_check.key("entityKey")
    owner = fact_check.integer("owner")
    object_index = fact_check.integer("objectIndex")
    secondary_index = fact_check.integer("secondaryIndex")
    if key != _expected_native_entity_key(
        owner=owner, object_index=object_index, secondary_index=secondary_index, native_object_id=native_object_id
    ):
        raise RunnerError(f"{label} entityKey does not match authoritative identity fields")
    fact_check.integer("cardId")
    fact_check.integer("objectKind", minimum=-1, maximum=31)
    position = fact.get("position")
    if not isinstance(position, list) or len(position) != 2:
        raise RunnerError(f"{label}.position is invalid")
    for index, coordinate in enumerate(position):
        _response_int(coordinate, f"{label}.position[{index}]")
    invisible = fact.get("invisibleCount")
    if visibility_validated:
        _response_int(invisible, f"{label}.invisibleCount", minimum=0, maximum=MAX_ACTIVE_EFFECTS)
    elif invisible is not None:
        raise RunnerError(f"{label} exposes unvalidated invisibility")
    return fact


_EVENT_ENVELOPE_FIELDS = {
    "ok",
    "schema",
    "generation",
    "stateEpoch",
    "observationTick",
    "capacity",
    "hookSetAttested",
    "capability",
    "epochFirstSequence",
    "oldestRetainedSequence",
    "nextSequence",
    "overflowCount",
    "sequenceGapBeforeOldest",
    "complete",
    "events",
}


def _check_event_header(check, schema, capacity, generation, state_epoch, tick):
    label, envelope = check.label, check.data
    if envelope.get("ok") is not True or envelope.get("schema") != schema:
        raise RunnerError(f"{label} schema/ok marker is invalid")
    for name, expected, minimum in (
        ("generation", generation, None),
        ("stateEpoch", state_epoch, None),
        ("observationTick", tick, -1),
    ):
        if check.integer(name, minimum=minimum) != expected:
            if label == "combatEvents":
                raise RunnerError(f"{label} {name} does not match rich telemetry")
            raise RunnerError(f"{label} observation identity is inconsistent")
    if check.integer("capacity", minimum=1) != capacity:
        raise RunnerError(f"unexpected {label} capacity")


def _check_event_window(check, capacity):
    """Verify ring retention and delta boundaries before any event is consumed."""
    label = check.label
    epoch_first = check.integer("epochFirstSequence", minimum=1)
    oldest = check.integer("oldestRetainedSequence", minimum=epoch_first)
    next_sequence = check.integer("nextSequence", minimum=oldest)
    transmit_from = check.integer("transmitFromSequence", minimum=oldest, maximum=next_sequence, default=oldest)
    overflow = check.integer("overflowCount", minimum=0)
    if overflow != max(0, next_sequence - epoch_first - capacity):
        raise RunnerError(f"{label} overflow accounting is inconsistent")
    expected_oldest = max(epoch_first, next_sequence - capacity)
    if label != "combatEvents" and oldest != expected_oldest:
        raise RunnerError(f"{label} retained window is inconsistent")
    gap = check.boolean("sequenceGapBeforeOldest")
    if label == "combatEvents":
        if gap != (oldest > epoch_first) or oldest != expected_oldest:
            raise RunnerError(f"{label} retained-sequence window is inconsistent")
    elif gap != (oldest > epoch_first):
        raise RunnerError(f"{label} retention-gap marker is inconsistent")
    return epoch_first, oldest, next_sequence, transmit_from


def _check_event_list(check, capacity, next_sequence, transmit_from, unavailable):
    label = check.label
    events = check.data.get("events")
    if not isinstance(events, list) or len(events) > capacity:
        raise RunnerError(f"{label}.events is invalid")
    if len(events) != next_sequence - transmit_from:
        description = {"combatEvents": "transmitted sequence range", "phaseRuntime": "transmitted event range"}.get(
            label, "retained event range"
        )
        raise RunnerError(f"{label} {description} is incomplete")
    if unavailable and events:
        detail = "capability exposed events" if label == "combatEvents" else "exposed events"
        raise RunnerError(f"unavailable {label} {detail}")
    return events


def _validate_combat_events(
    value: object, *, generation: int, state_epoch: int, observation_tick: int
) -> Mapping[str, Any]:
    envelope = _response_mapping(value, "combatEvents")
    envelope_check = _Fields(envelope, "combatEvents")
    required = _EVENT_ENVELOPE_FIELDS | {"rejectedCaptureCount", "hookSetInstalled"}
    if set(envelope) not in (required, required | {"transmitFromSequence"}):
        raise RunnerError("combatEvents fields do not match the v1 schema")
    _check_event_header(
        envelope_check, COMBAT_EVENT_SCHEMA, MAX_COMBAT_EVENTS, generation, state_epoch, observation_tick
    )
    attested = envelope_check.boolean("hookSetAttested")
    installed = envelope_check.boolean("hookSetInstalled")
    complete = envelope_check.boolean("complete")
    if installed and not attested:
        raise RunnerError("combatEvents hooks are installed without binary attestation")
    if not complete:
        raise RunnerError("combatEvents capture is incomplete")
    capability = envelope_check.mapping("capability")
    _validate_provenance_record(capability, "combatEvents.capability")
    expected_status = "derived" if installed else "unavailable"
    if capability.get("status") != expected_status:
        raise RunnerError("combatEvents capability status does not match hook state")

    epoch_first, oldest, next_sequence, transmit_from = _check_event_window(envelope_check, MAX_COMBAT_EVENTS)
    rejected = envelope_check.integer("rejectedCaptureCount", minimum=0)
    if rejected != 0:
        raise RunnerError("combatEvents dropped a native capture and is incomplete")
    events = _check_event_list(envelope_check, MAX_COMBAT_EVENTS, next_sequence, transmit_from, not installed)

    fact_fields = (
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
    amount_fields = (
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
        *fact_fields,
        *amount_fields,
        "destinationBefore",
        "destinationAfter",
    }
    spawn_provenance_field = "spawnProvenance"
    required_target = {
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
    required_projectile = {
        "projectile_spawn",
        "projectile_impact",
        "projectile_deflect",
        "projectile_expire",
        "projectile_terminal",
    }
    parsed_events: dict[int, tuple[Mapping[str, Any], Mapping[str, Mapping[str, Any]]]] = {}
    deployment_contexts: dict[int, Mapping[str, Any]] = {}
    card_play_deployments: set[int] = set()

    def present_fact(fact: Mapping[str, Any]) -> bool:
        return fact.get("validated") is True and fact.get("present") is True

    def same_native_entity(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        return present_fact(left) and present_fact(right) and left.get("nativeObjectId") == right.get("nativeObjectId")

    for index, raw_event in enumerate(events):
        label = f"combatEvents.events[{index}]"
        event = _response_mapping(raw_event, label)
        event_check = _Fields(event, label)
        # The producer field was added without changing the ring schema name
        # so older saved diagnostics remain readable. A live upgraded probe
        # always emits it (null when no exact producer scope exists).
        event_keys = set(event)
        if event_keys != event_fields and event_keys != event_fields | {spawn_provenance_field}:
            raise RunnerError(f"{label} fields do not match the v1 event schema")
        sequence = event_check.integer("sequence")
        if sequence != transmit_from + index:
            raise RunnerError("combatEvents sequences are not contiguous")
        tick = event_check.integer("tick", minimum=-1)
        if observation_tick >= 0 and tick > observation_tick:
            raise RunnerError(f"{label} occurs after its observation")
        if event_check.integer("generation") != generation:
            raise RunnerError(f"{label} generation mismatch")
        if event_check.integer("stateEpoch") != state_epoch:
            raise RunnerError(f"{label} stateEpoch mismatch")
        kind = event.get("kind")
        if kind not in COMBAT_EVENT_KINDS:
            raise RunnerError(f"{label}.kind is invalid")
        hook_offset = event_check.integer("hookOffset", minimum=1)
        if hook_offset not in COMBAT_HOOK_OFFSETS[str(kind)]:
            raise RunnerError(f"{label}.hookOffset is invalid for {kind}")
        caller_offset = event_check.optional_int("callerOffset", minimum=1)
        allowed_callers = {
            # 0xF12D54 is the exact return after the action-vtable +0x118
            # dispatch in function 0xF12C44.  Native BarbLogHero healing uses
            # this generic action path; retaining the single return address
            # keeps arbitrary heal callers fail-closed.
            "heal": {0xF12D54, 0xF20DF0, 0xF2B374, 0xF319A4},
            "spawn": {0xF2FFFC},
            "projectile_spawn": {0xF2FFFC},
            "projectile_expire": {0xF2A120, 0xF2A14C},
            "projectile_terminal": None,
        }
        if caller_offset is not None:
            allowed = allowed_callers.get(str(kind))
            if allowed is not None and caller_offset not in allowed:
                raise RunnerError(f"{label}.callerOffset is invalid for {kind}")
            if kind not in allowed_callers:
                raise RunnerError(f"{label} exposes an unexpected callerOffset")
        cause = event_check.optional_int("causeSequence", minimum=1)
        if cause is not None and (cause < epoch_first or cause >= sequence):
            raise RunnerError(f"{label}.causeSequence is outside the earlier epoch")
        pool = event.get("pool")
        if pool not in COMBAT_EVENT_POOLS:
            raise RunnerError(f"{label}.pool is invalid")
        reason = event.get("terminalReason")
        if reason not in COMBAT_TERMINAL_REASONS:
            raise RunnerError(f"{label}.terminalReason is invalid")
        lethal = event_check.boolean("lethal")
        raw_deployment = event.get("deploymentContext")
        deployment: Mapping[str, Any] | None = None
        if raw_deployment is not None:
            deployment = _response_mapping(raw_deployment, f"{label}.deploymentContext")
            deployment_fields = {
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
            if set(deployment) != deployment_fields:
                raise RunnerError(f"{label}.deploymentContext fields do not match the v1 schema")
            deployment_sequence = _response_int(
                deployment.get("deploymentSequence"), f"{label}.deploymentContext.deploymentSequence", minimum=1
            )
            owner = _response_int(deployment.get("owner"), f"{label}.deploymentContext.owner", minimum=0, maximum=1)
            played_card_global_id = _response_int(
                deployment.get("playedCardGlobalId"),
                f"{label}.deploymentContext.playedCardGlobalId",
                minimum=1,
                maximum=0xFFFFFFFF,
            )
            effective_card_global_id = deployment.get("effectiveCardGlobalId")
            if effective_card_global_id is not None:
                effective_card_global_id = _response_int(
                    effective_card_global_id,
                    f"{label}.deploymentContext.effectiveCardGlobalId",
                    minimum=1,
                    maximum=0xFFFFFFFF,
                )
            card_parameter = _response_int(
                deployment.get("cardParameter"),
                f"{label}.deploymentContext.cardParameter",
                minimum=0,
                maximum=0xFFFFFFFF,
            )
            deck_slot = _response_int(
                deployment.get("deckSlot"), f"{label}.deploymentContext.deckSlot", minimum=0, maximum=MAX_DECK_SLOTS - 1
            )
            cost = _response_int(deployment.get("cost"), f"{label}.deploymentContext.cost", minimum=0, maximum=15)
            form_code = _response_int(
                deployment.get("formCode"),
                f"{label}.deploymentContext.formCode",
                minimum=0,
                maximum=len(CARD_FORM_LABELS) - 1,
            )
            try:
                descriptor = decode_native_card_parameter(
                    card_parameter, expected_deck_slot=deck_slot, expected_cost=cost
                )
            except ValueError as error:
                raise RunnerError(f"{label}.deploymentContext packed card descriptor is invalid") from error
            if (
                descriptor.form_code != form_code
                or deployment.get("formName") != descriptor.form_name
                or _response_int(
                    deployment.get("consumeHookOffset"), f"{label}.deploymentContext.consumeHookOffset", minimum=1
                )
                != COMBAT_CONSUME_CARD_HOOK_OFFSET
            ):
                raise RunnerError(f"{label}.deploymentContext descriptor fields disagree")
            mirror_basic_form = (
                form_code == 0
                and played_card_global_id == MIRROR_CARD_GLOBAL_ID
                and effective_card_global_id is not None
                and effective_card_global_id != played_card_global_id
            )
            if (form_code == 0 and effective_card_global_id != played_card_global_id and not mirror_basic_form) or (
                form_code == 1 and effective_card_global_id is None
            ):
                raise RunnerError(f"{label}.deploymentContext effective form identity is incomplete")
            prior_deployment = deployment_contexts.setdefault(deployment_sequence, deployment)
            if prior_deployment != deployment:
                raise RunnerError("combatEvents deploymentSequence changed descriptor identity")
            if kind not in {"spawn", "projectile_spawn", "card_play"}:
                raise RunnerError(f"{label} attaches deployment provenance to an incompatible event")
        if kind == "card_play":
            if deployment is None:
                raise RunnerError(f"{label} lacks its required deploymentContext")
            deployment_sequence = _response_int(
                deployment.get("deploymentSequence"), f"{label}.deploymentContext.deploymentSequence", minimum=1
            )
            if deployment_sequence in card_play_deployments:
                raise RunnerError("combatEvents repeats a card_play deploymentSequence")
            card_play_deployments.add(deployment_sequence)
        facts = {name: _validate_combat_entity_fact(event.get(name), f"{label}.{name}") for name in fact_fields}
        raw_spawn_provenance = event.get(spawn_provenance_field)
        if raw_spawn_provenance is not None:
            provenance = _response_mapping(raw_spawn_provenance, f"{label}.spawnProvenance")
            if set(provenance) != {"kind", "hookOffset", "sourceDataGlobalId"}:
                raise RunnerError(f"{label}.spawnProvenance fields do not match the v1 schema")
            provenance_kind = provenance.get("kind")
            expected_provenance_hook = {"area_tick_nested_spawn": 0xF143A8, "action_spawn_to_location": 0xE7B864}.get(
                provenance_kind
            )
            if expected_provenance_hook is None:
                raise RunnerError(f"{label}.spawnProvenance.kind is invalid")
            if (
                _response_int(provenance.get("hookOffset"), f"{label}.spawnProvenance.hookOffset", minimum=1)
                != expected_provenance_hook
            ):
                raise RunnerError(f"{label}.spawnProvenance.hookOffset is invalid")
            _response_int(
                provenance.get("sourceDataGlobalId"),
                f"{label}.spawnProvenance.sourceDataGlobalId",
                minimum=1,
                maximum=0xFFFFFFFF,
            )
            if (
                kind not in {"spawn", "projectile_spawn"}
                or cause is not None
                or not same_native_entity(facts["source"], facts["immediateSource"])
                or not present_fact(facts["source"])
                or same_native_entity(facts["source"], facts["target"])
                or (kind == "spawn" and present_fact(facts["projectile"]))
            ):
                raise RunnerError(f"{label}.spawnProvenance lacks an exact producer-to-child edge")
        if kind in required_target and not present_fact(facts["target"]):
            raise RunnerError(f"{label} lacks its required target identity")
        if kind in required_projectile and not present_fact(facts["projectile"]):
            raise RunnerError(f"{label} lacks its required projectile identity")
        if deployment is not None and facts["target"].get("owner") in (0, 1) and facts["target"].get("owner") != owner:
            raise RunnerError(f"{label}.deploymentContext owner contradicts the spawned target")
        # Each fact has already proved its tagged key from its capture-time
        # owner and nativeObjectId in _validate_combat_entity_fact.  Do not
        # impose a second epoch-wide nativeObjectId -> entityKey bijection:
        # owner is mutable for native team-switch effects (for example
        # SuperEliteArcherCharm), so the same live object legitimately changes
        # from [0, -2, id] to [1, -2, id].  The probe only claims current-vector
        # uniqueness for nativeObjectId, not immutability across retained
        # historical events.
        amounts = {name: event_check.optional_int(name, minimum=0) for name in amount_fields}
        amount_kinds = {"damage", "death", "heal", "shield_damage", "shield_break"}
        if kind in amount_kinds:
            if amounts["actualAmount"] is None or amounts["actualAmount"] <= 0:
                raise RunnerError(f"{label} lacks a positive exact amount")
            if amounts["requestedAmount"] is None:
                raise RunnerError(f"{label} lacks its requested amount")
        elif any(value is not None for value in amounts.values()):
            raise RunnerError(f"{label} exposes amounts for a non-amount event")

        if kind in {"damage", "death", "shield_damage", "shield_break"}:
            if pool not in {"hitpoints", "built_in_shield", "buff_shield"}:
                raise RunnerError(f"{label} has an invalid damage pool")
            if any(amounts[name] is None for name in ("preHp", "postHp", "preBuiltInShield", "postBuiltInShield")):
                raise RunnerError(f"{label} has an incomplete HP/shield transition")
            if pool == "hitpoints":
                if amounts["preHp"] - amounts["postHp"] != amounts["actualAmount"]:
                    raise RunnerError(f"{label} HP delta does not match actualAmount")
            elif pool == "built_in_shield":
                if amounts["preBuiltInShield"] - amounts["postBuiltInShield"] != amounts["actualAmount"]:
                    raise RunnerError(f"{label} built-in shield delta does not match actualAmount")
            else:
                if amounts["preBuffShield"] is None or amounts["postBuffShield"] is None:
                    raise RunnerError(f"{label} has an incomplete Buff shield delta")
                if amounts["preBuffShield"] - amounts["postBuffShield"] != amounts["actualAmount"]:
                    raise RunnerError(f"{label} Buff shield delta does not match actualAmount")
            if lethal and (pool != "hitpoints" or amounts["postHp"] != 0):
                raise RunnerError(f"{label} lethal flag contradicts its HP transition")
        elif kind == "heal":
            if pool not in {"hitpoints", "built_in_shield"}:
                raise RunnerError(f"{label} has an invalid healing pool")
            if any(amounts[name] is None for name in ("preHp", "postHp", "preBuiltInShield", "postBuiltInShield")):
                raise RunnerError(f"{label} has an incomplete healing transition")
            delta = (
                amounts["postHp"] - amounts["preHp"]
                if pool == "hitpoints"
                else amounts["postBuiltInShield"] - amounts["preBuiltInShield"]
            )
            if delta != amounts["actualAmount"]:
                raise RunnerError(f"{label} healing delta does not match actualAmount")

        if kind == "death":
            if not lethal or cause is None:
                raise RunnerError(f"{label} is not linked to a lethal damage event")
        elif kind in {"shield_damage", "shield_break"}:
            if lethal or pool not in {"built_in_shield", "buff_shield"} or cause is None:
                raise RunnerError(f"{label} has invalid shield causality")
            if kind == "shield_break":
                post_field = "postBuiltInShield" if pool == "built_in_shield" else "postBuffShield"
                if amounts[post_field] != 0:
                    raise RunnerError(f"{label} does not end at zero shield")
        elif kind != "damage" and lethal:
            raise RunnerError(f"{label} exposes lethal outside damage/death")

        if kind in {"spawn", "projectile_spawn", "projectile_impact", "projectile_deflect"} and (
            pool != "none" or reason != "none"
        ):
            raise RunnerError(f"{label} has an impossible pool/terminal reason")
        if kind == "despawn" and pool != "none":
            raise RunnerError(f"{label} assigns a pool to removal")
        if kind == "projectile_spawn" and not same_native_entity(facts["target"], facts["projectile"]):
            raise RunnerError(f"{label} projectile spawn identities disagree")
        for destination_name in ("destinationBefore", "destinationAfter"):
            destination = event.get(destination_name)
            if destination is None:
                continue
            if not isinstance(destination, list) or len(destination) != 2:
                raise RunnerError(f"{label}.{destination_name} is invalid")
            for coordinate_index, coordinate in enumerate(destination):
                _response_int(coordinate, f"{label}.{destination_name}[{coordinate_index}]")
        if kind == "projectile_expire" and reason not in {"source_lost", "nonimpact_cancel"}:
            raise RunnerError(f"{label} names an unproved projectile expiry")
        if kind == "projectile_expire" and pool != "none":
            raise RunnerError(f"{label} assigns a pool to projectile expiry")
        if kind == "projectile_terminal" and reason != "unknown":
            raise RunnerError(f"{label} assigns a reason to an unknown terminal")
        if kind == "projectile_terminal" and pool != "none":
            raise RunnerError(f"{label} assigns a pool to projectile terminal")
        if kind not in {"death", "shield_damage", "shield_break", "despawn"} and cause is not None:
            raise RunnerError(f"{label} exposes unexpected causality")
        parsed_events[sequence] = (event, facts)

    expected_cause_kinds = {
        "death": {"damage"},
        "shield_damage": {"damage"},
        "shield_break": {"shield_damage"},
        "despawn": {"death", "projectile_impact", "projectile_expire", "projectile_terminal"},
    }
    for sequence, (event, facts) in parsed_events.items():
        cause = event.get("causeSequence")
        if cause is None or cause < oldest:
            continue
        cause_event, cause_facts = parsed_events[int(cause)]
        if cause_event.get("kind") not in expected_cause_kinds[str(event["kind"])]:
            raise RunnerError(f"combatEvents event {sequence} has an incompatible cause kind")
        subject = (
            facts["projectile"] if event["kind"] == "despawn" and present_fact(facts["projectile"]) else facts["target"]
        )
        cause_subject = (
            cause_facts["projectile"]
            if cause_event["kind"] in {"projectile_impact", "projectile_expire", "projectile_terminal"}
            else cause_facts["target"]
        )
        if not same_native_entity(subject, cause_subject):
            raise RunnerError(f"combatEvents event {sequence} cause identity does not match")
    return envelope


def _validate_phase_object(value: object, label: str, *, observation_tick: int) -> None:
    if value is None:
        return
    phase = _response_mapping(value, label)
    phase_check = _Fields(phase, label)
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
    phase_fields = set(phase)
    if (phase_fields != required and phase_fields != required | sequence_decay_fields) or phase.get(
        "schema"
    ) != PHASE_OBJECT_SCHEMA:
        raise RunnerError(f"{label} does not match native-phase-object.v1")
    attack_validated = phase_check.boolean("attackValidated")
    movement_validated = phase_check.boolean("movementValidated")
    buffs_validated = phase_check.boolean("buffsValidated")
    stage = phase_check.optional_int("attackSequenceStage", minimum=-1, maximum=MAX_ATTACK_SEQUENCE_STAGE)
    nonnegative_fields = (
        "attackTimelineMs",
        "loadRemainingMs",
        "deployRemainingMs",
        "deployPreviousMs",
        "configuredDeployTimeMs",
        "hitSpeedMs",
        "attackDashTimeMs",
        "baseMovementSpeed",
        "chargeSpeedMultiplier",
        "attackStepInput",
        "attackStepOutput",
        "movementStepInput",
        "movementStepOutput",
        "deployStepInput",
        "deployStepOutput",
        "effectiveMovementSpeed",
    )
    parsed = {name: phase_check.optional_int(name, minimum=0) for name in nonnegative_fields}
    if attack_validated:
        if any(
            parsed[name] is None
            for name in (
                "attackTimelineMs",
                "loadRemainingMs",
                "deployRemainingMs",
                "deployPreviousMs",
                "configuredDeployTimeMs",
                "hitSpeedMs",
                "attackDashTimeMs",
                "baseMovementSpeed",
                "chargeSpeedMultiplier",
            )
        ):
            raise RunnerError(f"{label} validated attack snapshot is incomplete")
    elif stage is not None or any(parsed[name] is not None for name in ("attackTimelineMs", "loadRemainingMs")):
        raise RunnerError(f"{label} exposes unvalidated attack state")
    charge = phase_check.optional_int("classicChargeProgress", minimum=-1)
    if movement_validated != (charge is not None):
        raise RunnerError(f"{label} movement validation is inconsistent")
    if sequence_decay_fields <= set(phase):
        progress = phase_check.optional_int(
            "attackSequenceProgressRaw", minimum=0, maximum=ATTACK_SEQUENCE_PROGRESS_LIMIT
        )
        progress_limit = phase_check.optional_int("attackSequenceProgressLimit", minimum=1)
        decay_remaining = phase_check.optional_int(
            "attackSequenceDecayRemainingMs", minimum=0, maximum=ATTACK_SEQUENCE_DECAY_DURATION_MS
        )
        decay_duration = phase_check.optional_int("attackSequenceDecayDurationMs", minimum=1)
        if (
            (progress is None) != (progress_limit is None)
            or (decay_remaining is None) != (decay_duration is None)
            or progress_limit not in {None, ATTACK_SEQUENCE_PROGRESS_LIMIT}
            or decay_duration not in {None, ATTACK_SEQUENCE_DECAY_DURATION_MS}
            or (progress is not None and not attack_validated)
            or (decay_remaining is not None and not attack_validated)
        ):
            raise RunnerError(f"{label} attack sequence decay contract is inconsistent")
        if progress is not None:
            expected_stage = 0 if progress < 4 else 1 if progress < 9 else 2 if progress < 49 else 3
            if stage != expected_stage:
                raise RunnerError(f"{label} attack sequence progress/stage disagree")
    extrema = tuple(
        phase_check.optional_int(name, minimum=0)
        for name in (
            "speedPositivePercent",
            "speedNegativeMagnitude",
            "hitSpeedPositivePercent",
            "hitSpeedNegativeMagnitude",
        )
    )
    if buffs_validated != all(item is not None for item in extrema):
        raise RunnerError(f"{label} Buff extrema validation is inconsistent")
    phase_check.optional_int("movementDelta")
    for value_name, tick_name in (
        ("attackStepOutput", "attackStepTick"),
        ("movementStepOutput", "movementStepTick"),
        ("deployStepOutput", "deployStepTick"),
        ("effectiveMovementSpeed", "effectiveMovementSpeedTick"),
        ("movementDelta", "movementDeltaTick"),
    ):
        tick = phase_check.integer(tick_name, minimum=-1)
        if observation_tick >= 0 and tick > observation_tick:
            raise RunnerError(f"{label}.{tick_name} is after the observation")
        if (phase.get(value_name) is None) != (tick == -1):
            raise RunnerError(f"{label} cached {value_name}/{tick_name} relationship is invalid")


def _validate_phase_runtime(
    value: object, *, generation: int, state_epoch: int, observation_tick: int, minimum_sequence: int | None = None
) -> Mapping[str, Any]:
    envelope = _response_mapping(value, "phaseRuntime")
    envelope_check = _Fields(envelope, "phaseRuntime")
    required = _EVENT_ENVELOPE_FIELDS | {"hookSetInstalled", "rejectedCount"}
    if set(envelope) not in (required, required | {"transmitFromSequence"}):
        raise RunnerError("phaseRuntime fields do not match the v1 schema")
    _check_event_header(
        envelope_check, PHASE_RUNTIME_SCHEMA, MAX_PHASE_EVENTS, generation, state_epoch, observation_tick
    )
    attested = envelope_check.boolean("hookSetAttested")
    installed = envelope_check.boolean("hookSetInstalled")
    if installed and not attested:
        raise RunnerError("phaseRuntime hooks are installed without attestation")
    capability = envelope_check.mapping("capability")
    _validate_provenance_record(capability, "phaseRuntime.capability")
    status = capability.get("status")
    if status not in {"derived", "unavailable"}:
        raise RunnerError("phaseRuntime capability status is invalid")
    if status == "derived" and (not installed or not attested):
        raise RunnerError("phaseRuntime derived capability lacks installed hooks")
    epoch_first, oldest, next_sequence, transmit_from = _check_event_window(envelope_check, MAX_PHASE_EVENTS)
    rejected = envelope_check.integer("rejectedCount", minimum=0)
    complete = envelope_check.boolean("complete")
    if complete and (status != "derived" or not installed or not attested or rejected != 0):
        raise RunnerError("phaseRuntime complete marker contradicts capture state")
    if rejected and complete:
        raise RunnerError("phaseRuntime rejected a fact without failing closed")
    events = _check_event_list(envelope_check, MAX_PHASE_EVENTS, next_sequence, transmit_from, status == "unavailable")

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
    if minimum_sequence is not None:
        if not epoch_first <= minimum_sequence <= next_sequence:
            raise RunnerError("phaseRuntime validation cursor is outside the epoch")
        if transmit_from > max(minimum_sequence, oldest):
            raise RunnerError("phaseRuntime delta skipped the validation cursor")
    for index, event_value in enumerate(events):
        label = f"phaseRuntime.events[{index}]"
        event = _response_mapping(event_value, label)
        event_check = _Fields(event, label)
        if set(event) != event_fields:
            raise RunnerError(f"{label} fields do not match the v1 schema")
        if event_check.integer("sequence", minimum=1) != transmit_from + index:
            raise RunnerError("phaseRuntime event sequences are not contiguous")
        event_tick = event_check.integer("tick", minimum=0)
        if observation_tick >= 0 and event_tick > observation_tick:
            raise RunnerError(f"{label} occurs after the observation")
        kind = event.get("kind")
        if kind not in PHASE_EVENT_HOOKS:
            raise RunnerError(f"{label}.kind is invalid")
        if event_check.integer("hookOffset", minimum=1) != PHASE_EVENT_HOOKS[str(kind)]:
            raise RunnerError(f"{label}.hookOffset is invalid")
        entity = _validate_combat_entity_fact(event.get("entity"), f"{label}.entity")
        if entity.get("present") is not True:
            raise RunnerError(f"{label} lacks its emitting entity")
        _validate_combat_entity_fact(event.get("targetBefore"), f"{label}.targetBefore")
        _validate_combat_entity_fact(event.get("targetAfter"), f"{label}.targetAfter")
        event_check.optional_int("callerOffset", minimum=1)
        event_check.optional_int("buffGlobalId", minimum=1, maximum=0xFFFFFFFF)
        event_check.optional_int("buffRemainingMs", minimum=-1)
        event_check.optional_int("speedMultiplier")
        event_check.optional_int("hitSpeedMultiplier")
        event_check.optional_int("inputStep")
        for name in ("outputStep", "timelineBefore", "timelineAfter"):
            event_check.optional_int(name, minimum=0)
        for name in ("classicChargeBefore", "classicChargeAfter"):
            event_check.optional_int(name, minimum=-1)
        event_check.boolean("success")
    return envelope


def _validate_special_movement_runtime(
    value: object, *, generation: int, state_epoch: int, observation_tick: int
) -> Mapping[str, Any]:
    try:
        parse_special_movement_runtime(
            value, generation=generation, state_epoch=state_epoch, observation_tick=observation_tick
        )
    except SpecialMovementRuntimeError as error:
        raise RunnerError(f"invalid specialMovementRuntime in native runner response: {error}") from error
    return _response_mapping(value, "specialMovementRuntime")


def _validate_action_movement_runtime(
    value: object, *, generation: int, state_epoch: int, observation_tick: int
) -> Mapping[str, Any]:
    try:
        parse_action_movement_runtime(
            value, generation=generation, state_epoch=state_epoch, observation_tick=observation_tick
        )
    except ActionMovementRuntimeError as error:
        raise RunnerError(f"invalid actionMovementRuntime in native runner response: {error}") from error
    return _response_mapping(value, "actionMovementRuntime")


def _validate_character_state_runtime(
    value: object, *, generation: int, state_epoch: int, observation_tick: int
) -> Mapping[str, Any]:
    try:
        parse_character_state_runtime(
            value, generation=generation, state_epoch=state_epoch, observation_tick=observation_tick
        )
    except CharacterStateRuntimeError as error:
        raise RunnerError(f"invalid characterStateRuntime in native runner response: {error}") from error
    return _response_mapping(value, "characterStateRuntime")


def _validate_visibility_runtime(
    value: object, *, generation: int, state_epoch: int, observation_tick: int
) -> Mapping[str, Any]:
    envelope = _response_mapping(value, "visibilityRuntime")
    envelope_check = _Fields(envelope, "visibilityRuntime")
    required = _EVENT_ENVELOPE_FIELDS | {
        "ownerRelativeVisibility",
        "transitionHookSetInstalled",
        "contextualGateHookSetInstalled",
        "contextualGateCapability",
        "rejectedCount",
    }
    if set(envelope) not in (required, required | {"transmitFromSequence"}):
        raise RunnerError("visibilityRuntime fields do not match the v1 schema")
    _check_event_header(
        envelope_check, VISIBILITY_RUNTIME_SCHEMA, MAX_VISIBILITY_EVENTS, generation, state_epoch, observation_tick
    )
    attested = envelope_check.boolean("hookSetAttested")
    transition_installed = envelope_check.boolean("transitionHookSetInstalled")
    contextual_installed = envelope_check.boolean("contextualGateHookSetInstalled")
    if transition_installed and not attested:
        raise RunnerError("visibilityRuntime transition hooks lack attestation")
    if contextual_installed:
        raise RunnerError("visibilityRuntime contextual gate is unsupported by v1")
    capability = envelope_check.mapping("capability")
    _validate_provenance_record(capability, "visibilityRuntime.capability")
    status = capability.get("status")
    if status not in {"derived", "unavailable"}:
        raise RunnerError("visibilityRuntime capability status is invalid")
    if (status == "derived") != (transition_installed and attested):
        raise RunnerError("visibilityRuntime capability/install state is inconsistent")
    contextual_capability = envelope_check.mapping("contextualGateCapability")
    _validate_provenance_record(contextual_capability, "visibilityRuntime.contextualGateCapability")
    if contextual_capability.get("status") != "unavailable":
        raise RunnerError("visibilityRuntime contextual gate must remain unavailable")

    visibility = envelope_check.mapping("ownerRelativeVisibility")
    if set(visibility) != {"publicByOwner", "targetableByOwner"}:
        raise RunnerError("visibilityRuntime owner-relative visibility fields are invalid")
    for name in ("publicByOwner", "targetableByOwner"):
        record = _response_mapping(visibility.get(name), f"visibilityRuntime.ownerRelativeVisibility.{name}")
        if (
            set(record) != {"status", "reason"}
            or record.get("status") != "unavailable"
            or record.get("reason") != "no-exact-owner-conditioned-native-producer"
        ):
            raise RunnerError("visibilityRuntime owner-relative visibility claim is invalid")

    epoch_first, oldest, next_sequence, transmit_from = _check_event_window(envelope_check, MAX_VISIBILITY_EVENTS)
    rejected = envelope_check.integer("rejectedCount", minimum=0)
    complete = envelope_check.boolean("complete")
    if complete and (status != "derived" or not transition_installed or not attested or rejected != 0):
        raise RunnerError("visibilityRuntime complete marker contradicts capture state")
    if rejected and complete:
        raise RunnerError("visibilityRuntime rejected a fact without failing closed")
    events = _check_event_list(
        envelope_check, MAX_VISIBILITY_EVENTS, next_sequence, transmit_from, status == "unavailable"
    )

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
    for index, event_value in enumerate(events):
        label = f"visibilityRuntime.events[{index}]"
        event = _response_mapping(event_value, label)
        event_check = _Fields(event, label)
        if set(event) != event_fields:
            raise RunnerError(f"{label} fields do not match the v1 schema")
        if event_check.integer("sequence", minimum=1) != transmit_from + index:
            raise RunnerError("visibilityRuntime event sequences are not contiguous")
        event_tick = event_check.integer("tick", minimum=0)
        if observation_tick >= 0 and event_tick > observation_tick:
            raise RunnerError(f"{label} occurs after the observation")
        kind = event.get("kind")
        if kind not in VISIBILITY_EVENT_HOOKS:
            raise RunnerError(f"{label}.kind is invalid")
        kind_label = str(kind)
        if (
            event_check.integer("hookOffset", minimum=1) != VISIBILITY_EVENT_HOOKS[kind_label]
            or event_check.integer("callerOffset", minimum=1) != VISIBILITY_EVENT_CALLERS[kind_label]
        ):
            raise RunnerError(f"{label} hook/caller identity is invalid")
        subject = _validate_combat_entity_fact(event.get("subject"), f"{label}.subject")
        if subject.get("present") is not True or subject.get("visibilityValidated") is not True:
            raise RunnerError(f"{label} lacks its validated visibility subject")
        event_check.integer("buffGlobalId", minimum=1, maximum=0xFFFFFFFF)
        before = event_check.integer("invisibleCountBefore", minimum=0, maximum=MAX_ACTIVE_EFFECTS)
        after = event_check.integer("invisibleCountAfter", minimum=0, maximum=MAX_ACTIVE_EFFECTS)
        expected_edge = (0, 1) if kind_label == "became_invisible" else (1, 0)
        if (before, after) != expected_edge:
            raise RunnerError(f"{label} does not cross the exact visibility boundary")
        if subject.get("invisibleCount") != after:
            raise RunnerError(f"{label} subject count disagrees with the transition")
        if event.get("scope") != "native_invisibility_phase":
            raise RunnerError(f"{label}.scope is invalid")
        if event_check.boolean("completeContext") is not True:
            raise RunnerError(f"{label} lacks complete context")
    return envelope


def _validate_remaining_runtime(
    value: object, *, generation: int, state_epoch: int, observation_tick: int
) -> Mapping[str, Any]:
    envelope = _response_mapping(value, "remainingRuntime")
    envelope_check = _Fields(envelope, "remainingRuntime")
    required = _EVENT_ENVELOPE_FIELDS | {"ownerRelativeVisibility", "hookSetInstalled", "rejectedCount"}
    envelope_keys = set(envelope) - {"transmitFromSequence"}
    if envelope_keys != required:
        raise RunnerError("remainingRuntime fields do not match the v1 schema")
    _check_event_header(
        envelope_check, REMAINING_RUNTIME_SCHEMA, MAX_REMAINING_EVENTS, generation, state_epoch, observation_tick
    )
    attested = envelope_check.boolean("hookSetAttested")
    installed = envelope_check.boolean("hookSetInstalled")
    if installed and not attested:
        raise RunnerError("remainingRuntime hooks are installed without attestation")
    capability = envelope_check.mapping("capability")
    _validate_provenance_record(capability, "remainingRuntime.capability")
    status = capability.get("status")
    if status not in {"derived", "unavailable"}:
        raise RunnerError("remainingRuntime capability status is invalid")
    if status == "derived" and (not installed or not attested):
        raise RunnerError("remainingRuntime derived capability lacks installed hooks")

    visibility = envelope_check.mapping("ownerRelativeVisibility")
    if set(visibility) != {"publicByOwner", "targetableByOwner"}:
        raise RunnerError("remainingRuntime owner-relative visibility fields are invalid")
    for name in ("publicByOwner", "targetableByOwner"):
        record = _response_mapping(visibility.get(name), f"remainingRuntime.ownerRelativeVisibility.{name}")
        if (
            set(record) != {"status", "reason"}
            or record.get("status") != "unavailable"
            or record.get("reason") != ("no-exact-owner-conditioned-native-producer")
        ):
            raise RunnerError(f"remainingRuntime must fail closed for {name}")

    epoch_first, oldest, next_sequence, transmit_from = _check_event_window(envelope_check, MAX_REMAINING_EVENTS)
    rejected = envelope_check.integer("rejectedCount", minimum=0)
    complete = envelope_check.boolean("complete")
    if complete and (status != "derived" or not installed or not attested or rejected != 0):
        raise RunnerError("remainingRuntime complete marker contradicts capture state")

    events = _check_event_list(
        envelope_check, MAX_REMAINING_EVENTS, next_sequence, transmit_from, status == "unavailable"
    )

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

    def present(fact: Mapping[str, Any]) -> bool:
        return fact.get("validated") is True and fact.get("present") is True

    for index, event_value in enumerate(events):
        label = f"remainingRuntime.events[{index}]"
        event = _response_mapping(event_value, label)
        event_check = _Fields(event, label)
        if set(event) != event_fields:
            raise RunnerError(f"{label} fields do not match the v1 schema")
        if event_check.integer("sequence", minimum=1) != transmit_from + index:
            raise RunnerError("remainingRuntime event sequences are not contiguous")
        event_tick = event_check.integer("tick", minimum=0)
        if observation_tick >= 0 and event_tick > observation_tick:
            raise RunnerError(f"{label} occurs after the observation")
        kind = event.get("kind")
        if kind not in REMAINING_EVENT_HOOKS:
            raise RunnerError(f"{label}.kind is invalid")
        hook_offset = event_check.integer("hookOffset", minimum=1)
        if hook_offset not in REMAINING_EVENT_HOOKS[str(kind)]:
            raise RunnerError(f"{label}.hookOffset is invalid")
        caller_offset = event_check.optional_int("callerOffset", minimum=1)
        facts = {
            name: _validate_combat_entity_fact(event.get(name), f"{label}.{name}")
            for name in ("entity", "source", "target", "targetBefore", "targetAfter")
        }
        object_kind = event_check.optional_int("objectKind", minimum=0)
        vtable_offset = event_check.optional_int("runtimeVtableOffset", minimum=1)
        global_ids = {
            name: event_check.optional_int(name, minimum=1)
            for name in (
                "dataBeforeGlobalId",
                "dataAfterGlobalId",
                "expectedCharacterDataGlobalId",
                "expectedProjectileDataGlobalId",
                "configuredDataGlobalId",
            )
        }
        optional_ints = {
            name: event_check.optional_int(name)
            for name in (
                "resourcePreFixed",
                "resourcePostFixed",
                "resourceActualDeltaFixed",
                "amountArgument",
                "configuredAmountArgument",
                "resourceOwner",
                "areaRemainingLifeMs",
            )
        }
        result = event_check.boolean("result")
        event_check.boolean("option")
        committed = event_check.boolean("committed")
        event_check.boolean("resetTarget")
        if not event_check.boolean("completeContext"):
            raise RunnerError(f"{label} exposes an incomplete native context")

        resource_cause = event.get("resourceCause")
        transform_kind = event.get("transformKind")
        if resource_cause not in {"none", *resource_edges}:
            raise RunnerError(f"{label}.resourceCause is invalid")
        if transform_kind not in {"none", "character", "projectile"}:
            raise RunnerError(f"{label}.transformKind is invalid")

        if kind == "resource_delta":
            if not present(facts["source"]) or resource_cause not in resource_edges:
                raise RunnerError(f"{label} lacks its exact resource source")
            expected_hook, expected_caller, scale = resource_edges[str(resource_cause)]
            if hook_offset != expected_hook or caller_offset != expected_caller:
                raise RunnerError(f"{label} resource producer edge is invalid")
            pre = optional_ints["resourcePreFixed"]
            post = optional_ints["resourcePostFixed"]
            actual = optional_ints["resourceActualDeltaFixed"]
            amount = optional_ints["amountArgument"]
            configured = optional_ints["configuredAmountArgument"]
            owner = optional_ints["resourceOwner"]
            if (
                pre is None
                or post is None
                or actual is None
                or amount is None
                or configured is None
                or owner not in {0, 1}
                or pre < 0
                or post <= pre
                or actual != post - pre
                or amount <= 0
                or configured != amount
                or actual > amount * scale
            ):
                raise RunnerError(f"{label} resource arithmetic is invalid")
        elif resource_cause != "none":
            raise RunnerError(f"{label} has a stray resource cause")

        if kind == "transform":
            before = global_ids["dataBeforeGlobalId"]
            after = global_ids["dataAfterGlobalId"]
            expected_character = global_ids["expectedCharacterDataGlobalId"]
            expected_projectile = global_ids["expectedProjectileDataGlobalId"]
            matches = (
                after is not None and after == expected_character,
                after is not None and after == expected_projectile,
            )
            if (
                not present(facts["entity"])
                or caller_offset is not None
                or before is None
                or after is None
                or before == after
                or matches[0] == matches[1]
                or transform_kind != ("character" if matches[0] else "projectile")
                or global_ids["configuredDataGlobalId"] != after
            ):
                raise RunnerError(f"{label} transform transition is invalid")
        elif transform_kind != "none":
            raise RunnerError(f"{label} has a stray transform kind")

        if kind in {"area_create", "area_expire"}:
            if (
                not present(facts["entity"])
                or object_kind != 3
                or vtable_offset != 0x189C2E8
                or (kind == "area_create" and not committed)
                or caller_offset is not None
            ):
                raise RunnerError(f"{label} area lifecycle identity is invalid")
        elif kind in {"area_relation_parent", "area_relation_follow", "area_relation_related"}:
            if (
                not present(facts["entity"])
                or not present(facts["target"])
                or object_kind != 3
                or vtable_offset != 0x189C2E8
                or caller_offset is not None
            ):
                raise RunnerError(f"{label} area relation is invalid")
        elif kind == "area_action_spawn":
            source_kind = facts["source"].get("objectKind")
            producer_identity_valid = (
                object_kind == 3 and source_kind == 3 and vtable_offset == 0x189C2E8
                if hook_offset == 0xF143A8
                else object_kind == source_kind and vtable_offset is not None
            )
            if (
                not present(facts["entity"])
                or not present(facts["source"])
                or not present(facts["target"])
                or facts["entity"].get("nativeObjectId") != facts["target"].get("nativeObjectId")
                or facts["source"].get("nativeObjectId") == facts["target"].get("nativeObjectId")
                or not producer_identity_valid
                or caller_offset is not None
                or not committed
                or global_ids["configuredDataGlobalId"] is None
                or global_ids["dataAfterGlobalId"] is None
            ):
                raise RunnerError(f"{label} exact producer-to-child edge is invalid")
        elif kind in {"lifetime_spawn", "lifetime_despawn"}:
            if (
                not present(facts["entity"])
                or caller_offset is not None
                or (kind == "lifetime_spawn" and not committed)
            ):
                raise RunnerError(f"{label} lifetime edge is invalid")
        elif str(kind).startswith("tower_aggro_"):
            before_present = present(facts["targetBefore"])
            after_present = present(facts["targetAfter"])
            expected_presence = {
                "tower_aggro_acquire": (False, True),
                "tower_aggro_change": (True, True),
                "tower_aggro_lose": (True, False),
            }[str(kind)]
            if (
                not present(facts["entity"])
                or (before_present, after_present) != expected_presence
                or caller_offset is not None
            ):
                raise RunnerError(f"{label} tower aggro edge is invalid")
        elif kind == "tower_activate":
            if not present(facts["entity"]) or caller_offset != 0xF58820 or not result:
                raise RunnerError(f"{label} tower activation edge is invalid")
        elif kind == "spawn_attach":
            if (
                not present(facts["entity"])
                or not present(facts["source"])
                or not present(facts["target"])
                or caller_offset != 0xF18FF4
                or not committed
                or global_ids["configuredDataGlobalId"] != global_ids["dataAfterGlobalId"]
                or facts["source"].get("nativeObjectId") == facts["target"].get("nativeObjectId")
            ):
                raise RunnerError(f"{label} SpawnAttach edge is invalid")
        elif kind == "area_damage_eligibility":
            if not present(facts["entity"]) or caller_offset is not None:
                raise RunnerError(f"{label} area eligibility context is invalid")
    return envelope


def _validate_player_runtime(value: object, expected_owner: int) -> tuple[tuple[str, tuple[int, int, int]], ...]:
    label = f"players[{expected_owner}]"
    player = _response_mapping(value, label)
    player_check = _Fields(player, label)
    required = {"owner", "ownerRootValidated", "ownerEntityKey", "abilityRuntime", "evolutionRuntime"}
    if set(player) != required:
        raise RunnerError(f"{label} fields do not match the v2 player schema")
    if player_check.integer("owner") != expected_owner:
        raise RunnerError("rich telemetry players are not ordered by owner")
    owner_root_valid = player_check.boolean("ownerRootValidated")
    owner_key_raw = player.get("ownerEntityKey")
    owner_key = None if owner_key_raw is None else _validate_entity_key(owner_key_raw, f"{label}.ownerEntityKey")
    if owner_key is not None and not owner_root_valid:
        raise RunnerError(f"{label} owner-root validation/key relationship is invalid")
    if owner_key is not None and owner_key[0] != expected_owner:
        raise RunnerError(f"{label} owner-root identity belongs to another owner")

    references: list[tuple[str, tuple[int, int, int]]] = []
    if owner_key is not None:
        references.append((f"{label}.ownerEntityKey", owner_key))

    raw_abilities = player.get("abilityRuntime")
    if raw_abilities is None:
        if not owner_root_valid:
            pass
    else:
        if not owner_root_valid or not isinstance(raw_abilities, list):
            raise RunnerError(f"{label}.abilityRuntime is invalid")
        if len(raw_abilities) > MAX_CHAMPION_CONTROLLERS:
            raise RunnerError(f"{label}.abilityRuntime exceeds controller capacity")
        previous_slot = 0
        for index, raw_ability in enumerate(raw_abilities):
            ability_label = f"{label}.abilityRuntime[{index}]"
            ability = _response_mapping(raw_ability, ability_label)
            ability_check = _Fields(ability, ability_label)
            ability_fields = {
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
            if set(ability) != ability_fields:
                raise RunnerError(f"{ability_label} fields do not match the v2 ability schema")
            controller_slot = ability_check.integer("controllerSlot", minimum=1, maximum=MAX_CHAMPION_CONTROLLERS)
            if controller_slot <= previous_slot:
                raise RunnerError(f"{label} controller slots are not strictly ordered")
            previous_slot = controller_slot
            ability_check.integer("actionDataGlobalId", minimum=1)
            action_name = ability.get("actionDataName")
            if not isinstance(action_name, str):
                raise RunnerError(f"{ability_label}.actionDataName is invalid")
            try:
                action_name_bytes = action_name.encode("utf-8")
            except UnicodeEncodeError as error:
                raise RunnerError(f"{ability_label}.actionDataName is invalid") from error
            if not 1 <= len(action_name_bytes) <= MAX_ACTIVE_EFFECT_NAME_BYTES:
                raise RunnerError(f"{ability_label}.actionDataName is invalid")
            ability_check.integer("selectedCharacterDataGlobalId", minimum=1)
            configured = ability_check.integer("configuredCooldownMs", minimum=0)
            remaining = ability_check.integer("remainingCooldownMs", minimum=0)
            if remaining > configured:
                raise RunnerError(f"{ability_label} cooldown exceeds configured value")
            max_charges = ability_check.integer("maxCharges", minimum=0)
            remaining_charges = ability_check.integer("remainingChargesRaw", minimum=-1)
            if (max_charges > 0 and not 0 <= remaining_charges <= max_charges) or (
                max_charges <= 0 and remaining_charges != -1
            ):
                raise RunnerError(f"{ability_label} charge bounds are invalid")
            button_state = ability_check.integer("buttonState", minimum=0, maximum=len(ABILITY_BUTTON_STATE_LABELS) - 1)
            if ability.get("buttonStateLabel") != ABILITY_BUTTON_STATE_LABELS[button_state]:
                raise RunnerError(f"{ability_label}.buttonStateLabel is invalid")
            if ability_check.boolean("available") != (button_state in ABILITY_QUEUEABLE_BUTTON_STATES):
                raise RunnerError(f"{ability_label}.available is not an exact queueable state")
            champion_keys = ability.get("championEntityKeys")
            if not isinstance(champion_keys, list) or len(champion_keys) > MAX_RICH_OBJECTS:
                raise RunnerError(f"{ability_label}.championEntityKeys is invalid")
            seen_champion_keys: set[tuple[int, int, int]] = set()
            for champion_index, raw_key in enumerate(champion_keys):
                champion_label = f"{ability_label}.championEntityKeys[{champion_index}]"
                champion_key = _validate_entity_key(raw_key, champion_label)
                if champion_key[0] != expected_owner:
                    raise RunnerError(f"{ability_label} champion identity belongs to another owner")
                if champion_key in seen_champion_keys:
                    raise RunnerError(f"{ability_label} has duplicate champion keys")
                seen_champion_keys.add(champion_key)
                references.append((champion_label, champion_key))

    raw_evolution = player.get("evolutionRuntime")
    if raw_evolution is None:
        return tuple(references)
    if not owner_root_valid or not isinstance(raw_evolution, list):
        raise RunnerError(f"{label}.evolutionRuntime is invalid")
    if len(raw_evolution) > MAX_DECK_SLOTS:
        raise RunnerError(f"{label}.evolutionRuntime exceeds deck capacity")
    for expected_slot, raw_slot in enumerate(raw_evolution):
        slot_label = f"{label}.evolutionRuntime[{expected_slot}]"
        slot = _response_mapping(raw_slot, slot_label)
        slot_check = _Fields(slot, slot_label)
        slot_fields = {
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
        if set(slot) != slot_fields:
            raise RunnerError(f"{slot_label} fields do not match the v2 evolution schema")
        if slot_check.integer("deckSlot") != expected_slot:
            raise RunnerError(f"{label} evolution deck slots are not contiguous")
        card_id = slot_check.integer("cardId", minimum=1)
        if slot_check.integer("baseSpellGlobalId", minimum=1) != card_id:
            raise RunnerError(f"{slot_label} base-spell identity does not match cardId")
        evolvable = slot_check.boolean("evolvable")
        progress = slot_check.integer("progress", minimum=0)
        if not evolvable:
            if (
                progress != 0
                or slot.get("evolutionFormGlobalId") is not None
                or slot.get("cycleRequired") is not None
                or slot.get("cycleRemaining") is not None
                or slot.get("ready") is not None
            ):
                raise RunnerError(f"{slot_label} exposed evolution state for a base-only card")
            continue
        slot_check.integer("evolutionFormGlobalId", minimum=1)
        cycle_required = slot_check.integer("cycleRequired", minimum=1)
        cycle_remaining = slot_check.integer("cycleRemaining", minimum=0)
        if progress > cycle_required or cycle_remaining != cycle_required - progress:
            raise RunnerError(f"{slot_label} evolution cycle arithmetic is invalid")
        if slot_check.boolean("ready") != (progress >= cycle_required):
            raise RunnerError(f"{slot_label}.ready does not match its cycle")
    return tuple(references)


def _validate_rich_observation(result: dict[str, Any], *, phase_event_floor: int | None = None) -> dict[str, Any]:
    top_level_fields = {
        "ok",
        "schema",
        "statusEnum",
        "generation",
        "stateEpoch",
        "tick",
        "provenance",
        "capabilities",
        "count",
        "objects",
        "players",
        "combatEvents",
        "returned",
        "truncated",
    }
    optional_runtimes = {
        "phaseRuntime",
        "specialMovementRuntime",
        "actionMovementRuntime",
        "characterStateRuntime",
        "visibilityRuntime",
        "remainingRuntime",
        "towerTroopRuntime",
    }
    has_phase_runtime = "phaseRuntime" in result
    top_level_fields |= optional_runtimes.intersection(result)
    has_tower_troop_runtime = "towerTroopRuntime" in result
    if set(result) != top_level_fields:
        raise RunnerError("rich telemetry fields do not match the v3 schema")
    if result.get("ok") is not True:
        raise RunnerError("native rich telemetry is not successful")
    if result.get("schema") != RICH_TELEMETRY_SCHEMA:
        raise RunnerError(f"unexpected rich telemetry schema: {result.get('schema')!r}")
    if result.get("statusEnum") != list(RICH_TELEMETRY_STATUS_ENUM):
        raise RunnerError("invalid rich telemetry statusEnum")
    identity = {
        "generation": _response_int(result.get("generation"), "generation", minimum=0),
        "state_epoch": _response_int(result.get("stateEpoch"), "stateEpoch", minimum=0),
        "observation_tick": _response_int(result.get("tick"), "tick", minimum=-1),
    }

    provenance = _response_mapping(result.get("provenance"), "provenance")
    capabilities = _response_mapping(result.get("capabilities"), "capabilities")
    if not provenance or not capabilities:
        raise RunnerError("rich telemetry provenance and capabilities must be nonempty")
    for name, record in provenance.items():
        _validate_provenance_record(record, f"provenance.{name}")
    for name, record in capabilities.items():
        _validate_provenance_record(record, f"capabilities.{name}")
    required_capabilities = {
        "entityCore",
        "targetCoordinates",
        "targetEntity",
        "hitpoints",
        "componentInventory",
        "shield",
        "attackSequenceStage",
        "buffs",
        "debuffs",
        "slow",
        "rage",
        "stun",
        "freeze",
        "attackPhase",
        "deployPhase",
        "chargeStage",
        "sourceEntity",
        "projectile",
        "impact",
        "visibility",
        "invisibility",
        "abilityRuntime",
        "evolutionRuntime",
    }
    if not required_capabilities.issubset(capabilities):
        missing = sorted(required_capabilities.difference(capabilities))
        raise RunnerError(f"rich telemetry is missing capabilities: {missing}")
    expected_capability_status = {
        "entityCore": "authoritative",
        "targetCoordinates": "authoritative",
        "targetEntity": "authoritative",
        "hitpoints": "authoritative",
        "componentInventory": "authoritative",
        "shield": "authoritative",
        "attackSequenceStage": "authoritative",
        "buffs": "authoritative",
        "debuffs": "unavailable",
        "slow": "derived" if has_phase_runtime else "unavailable",
        "rage": "derived" if has_phase_runtime else "unavailable",
        "stun": "derived" if has_phase_runtime else "unavailable",
        "freeze": "derived" if has_phase_runtime else "unavailable",
        "attackPhase": "derived" if has_phase_runtime else "unavailable",
        "deployPhase": "derived" if has_phase_runtime else "unavailable",
        "chargeStage": "derived" if has_phase_runtime else "unavailable",
        "sourceEntity": "unavailable",
        "projectile": "derived",
        "impact": "unavailable",
        "visibility": "unavailable",
        "invisibility": "authoritative",
        "abilityRuntime": "derived",
        "evolutionRuntime": "derived",
    }
    for name, expected_status in expected_capability_status.items():
        if _response_mapping(capabilities[name], f"capabilities.{name}").get("status") != expected_status:
            raise RunnerError(f"unexpected capabilities.{name}.status in native runner response")
    expected_provenance_status = {
        "nativeObjectId": "authoritative",
        "dataGlobalId": "authoritative",
        "targetEntityKey": "authoritative",
        "targetEntityValidated": "derived",
        "shield.current": "authoritative",
        "shield.max": "authoritative",
        "attackSequenceStage": "authoritative",
        "activeEffects": "authoritative",
        "activeEffects.buffGlobalId": "authoritative",
        "activeEffects.name": "authoritative",
        "activeEffects.remainingMs": "authoritative",
        "activeEffects.sourceEntityKey": "authoritative",
        "activeEffects.sourceEntityValidated": "derived",
        "invisibleCount": "authoritative",
        "visibilityState": "derived",
        "components.slots.vtableLibgOffset": "authoritative",
        "components.slots.typeGetterLibgOffset": "authoritative",
        "components.slots.engineType": "authoritative",
        "projectile": "derived",
        "projectile.projectileDataGlobalId": "derived",
        "projectile.sourceEntityKey": "derived",
        "projectile.targetEntityKey": "derived",
        "projectile.homingTargetEntityKey": "derived",
        "projectile.destination": "derived",
        "projectile.terminal": "derived",
        "projectile.nativePhase": "derived",
        "projectile.dragStage": "authoritative",
        "entityResourceRuntime": "authoritative",
        "periodicAttackModifierRuntime": "authoritative",
        "captureRuntime": "authoritative",
        "thresholdRelocationRuntime": "authoritative",
        "players.ownerRoot": "derived",
        "players.abilityRuntime": "derived",
        "players.abilityRuntime.buttonState": "derived",
        "players.abilityRuntime.cooldown": "derived",
        "players.abilityRuntime.charges": "derived",
        "players.evolutionRuntime": "derived",
        "players.evolutionRuntime.cycle": "derived",
        "players.evolutionRuntime.playedForm": "unavailable",
    }
    for name, expected_status in expected_provenance_status.items():
        if (
            name not in provenance
            or _response_mapping(provenance[name], f"provenance.{name}").get("status") != expected_status
        ):
            raise RunnerError(f"unexpected or missing provenance.{name}.status in native runner response")

    count = _response_int(result.get("count"), "count", minimum=0)
    returned = _response_int(result.get("returned"), "returned", minimum=0, maximum=MAX_RICH_OBJECTS)
    truncated = _response_bool(result.get("truncated"), "truncated")
    objects = result.get("objects")
    if not isinstance(objects, list) or len(objects) != returned or returned > count:
        raise RunnerError("invalid rich telemetry objects/returned/count relationship")
    if not truncated and returned != count:
        raise RunnerError("non-truncated rich telemetry omitted objects")

    entity_keys: set[tuple[int, int, int]] = set()
    native_object_ids: set[int] = set()
    entity_references: list[tuple[str, tuple[int, int, int]]] = []
    for expected_slot, raw_object in enumerate(objects):
        object_label = f"objects[{expected_slot}]"
        entity = _response_mapping(raw_object, object_label)
        entity_check = _Fields(entity, object_label)
        slot = entity_check.integer("slot")
        if slot != expected_slot:
            raise RunnerError("rich telemetry object slots are not contiguous")
        if entity.get("null") is True:
            if set(entity) != {"slot", "null"}:
                raise RunnerError(f"{object_label} null slot contains live fields")
            continue

        live_object_fields = {
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
        if has_phase_runtime:
            live_object_fields.add("phaseRuntime")
        if has_tower_troop_runtime:
            live_object_fields.add("towerTroopRuntime")
        if set(entity) != live_object_fields:
            raise RunnerError(f"{object_label} fields do not match the v3 schema")

        owner = entity_check.integer("owner")
        native_object_id = entity_check.integer("nativeObjectId", minimum=1, maximum=0xFFFFFFFF)
        if native_object_id in native_object_ids:
            raise RunnerError(f"{object_label} duplicates a native object identity")
        native_object_ids.add(native_object_id)
        entity_check.integer("dataGlobalId", minimum=1, maximum=0xFFFFFFFF)
        object_index = entity_check.integer("objectIndex")
        secondary_index = entity_check.integer("secondaryIndex")
        entity_key = entity_check.key("entityKey")
        if entity_key != _expected_native_entity_key(
            owner=owner, object_index=object_index, secondary_index=secondary_index, native_object_id=native_object_id
        ):
            raise RunnerError(f"{object_label}.entityKey does not match source fields")
        if entity_key in entity_keys:
            raise RunnerError(f"{object_label} duplicates an entity identity")
        entity_keys.add(entity_key)
        for field in ("cardId", "x", "y", "targetX", "targetY"):
            entity_check.integer(field)

        target_validated = entity_check.boolean("targetEntityValidated")
        target_key = entity.get("targetEntityKey")
        if target_key is not None:
            validated_target_key = _validate_entity_key(target_key, f"{object_label}.targetEntityKey")
            if not target_validated:
                raise RunnerError(f"{object_label} returned an unvalidated target entity")
            entity_references.append((f"{object_label}.targetEntityKey", validated_target_key))

        hp = entity.get("hp")
        maximum_hp = entity.get("maxHp")
        if (hp is None) != (maximum_hp is None):
            raise RunnerError(f"{object_label} has a partial hitpoint pair")
        if hp is not None:
            _response_int(hp, f"{object_label}.hp")
            _response_int(maximum_hp, f"{object_label}.maxHp", minimum=0)

        if "shield" not in entity:
            raise RunnerError(f"{object_label} is missing shield telemetry")
        shield = entity.get("shield")
        if shield is not None:
            shield_label = f"{object_label}.shield"
            shield_record = _response_mapping(shield, shield_label)
            shield_record_check = _Fields(shield_record, shield_label)
            if set(shield_record) != {"current", "max", "status"}:
                raise RunnerError(f"{object_label}.shield fields are invalid")
            shield_current = shield_record_check.integer("current", minimum=0)
            shield_maximum = shield_record_check.integer("max", minimum=0)
            if shield_current > shield_maximum:
                raise RunnerError(f"{object_label} has shield above its maximum")
            if shield_record.get("status") != "authoritative":
                raise RunnerError(f"unexpected {object_label}.shield.status in native runner response")

        if "attackSequenceStage" not in entity:
            raise RunnerError(f"{object_label} is missing attackSequenceStage")
        attack_sequence_stage = entity.get("attackSequenceStage")
        if attack_sequence_stage is not None:
            _response_int(
                attack_sequence_stage,
                f"{object_label}.attackSequenceStage",
                minimum=0,
                maximum=MAX_ATTACK_SEQUENCE_STAGE,
            )

        for field in ("activeEffects", "invisibleCount", "visibilityState"):
            if field not in entity:
                raise RunnerError(f"{object_label} is missing {field}")
        active_effects = entity.get("activeEffects")
        if active_effects is not None:
            if not isinstance(active_effects, list) or len(active_effects) > MAX_ACTIVE_EFFECTS:
                raise RunnerError(f"{object_label}.activeEffects is invalid")
            for effect_index, raw_effect in enumerate(active_effects):
                effect_label = f"{object_label}.activeEffects[{effect_index}]"
                effect = _response_mapping(raw_effect, effect_label)
                effect_check = _Fields(effect, effect_label)
                required_effect_fields = {
                    "buffGlobalId",
                    "name",
                    "remainingMs",
                    "sourceEntityKey",
                    "sourceEntityValidated",
                }
                if set(effect) != required_effect_fields:
                    raise RunnerError(f"{object_label}.activeEffects[{effect_index}] is incomplete")
                effect_check.integer("buffGlobalId", minimum=1, maximum=0xFFFFFFFF)
                name = effect.get("name")
                if not isinstance(name, str):
                    raise RunnerError(f"invalid {effect_label}.name in native runner response")
                try:
                    encoded_name = name.encode("utf-8")
                except UnicodeEncodeError as error:
                    raise RunnerError(f"invalid {effect_label}.name in native runner response") from error
                if not 1 <= len(encoded_name) <= MAX_ACTIVE_EFFECT_NAME_BYTES:
                    raise RunnerError(f"invalid {effect_label}.name in native runner response")
                effect_check.integer("remainingMs", minimum=-1)
                source_entity_key = effect.get("sourceEntityKey")
                source_entity_validated = effect_check.boolean("sourceEntityValidated")
                if source_entity_key is not None:
                    validated_source_key = _validate_entity_key(source_entity_key, f"{effect_label}.sourceEntityKey")
                    entity_references.append((f"{effect_label}.sourceEntityKey", validated_source_key))
                    if not source_entity_validated:
                        raise RunnerError(
                            f"{object_label}.activeEffects[{effect_index}] exposed an unvalidated source entity"
                        )

        invisible_count = entity.get("invisibleCount")
        visibility_state = entity.get("visibilityState")
        if invisible_count is None:
            if visibility_state is not None:
                raise RunnerError(f"{object_label} exposed visibilityState without invisibleCount")
        else:
            validated_invisible_count = _response_int(
                invisible_count, f"{object_label}.invisibleCount", minimum=0, maximum=MAX_ACTIVE_EFFECTS
            )
            expected_visibility_state = "invisible" if validated_invisible_count > 0 else "visible"
            if visibility_state != expected_visibility_state:
                raise RunnerError(f"{object_label}.visibilityState does not match invisibleCount")
            if isinstance(active_effects, list) and validated_invisible_count > len(active_effects):
                raise RunnerError(f"{object_label}.invisibleCount exceeds activeEffects")

        if "projectile" not in entity:
            raise RunnerError(f"{object_label} is missing projectile telemetry")
        entity_references.extend(_validate_projectile(entity.get("projectile"), f"{object_label}.projectile"))
        _validate_threshold_relocation_runtime(
            entity.get("thresholdRelocationRuntime"), f"{object_label}.thresholdRelocationRuntime"
        )
        _validate_entity_resource(entity.get("entityResourceRuntime"), f"{object_label}.entityResourceRuntime")
        entity_references.extend(
            _validate_periodic_attack_modifier_runtime(
                entity.get("periodicAttackModifierRuntime"), f"{object_label}.periodicAttackModifierRuntime"
            )
        )
        entity_references.extend(
            _validate_capture_runtime(entity.get("captureRuntime"), f"{object_label}.captureRuntime")
        )
        if has_phase_runtime:
            _validate_phase_object(
                entity.get("phaseRuntime"),
                f"{object_label}.phaseRuntime",
                observation_tick=identity["observation_tick"],
            )

        components_label = f"{object_label}.components"
        components = _response_mapping(entity.get("components"), components_label)
        components_check = _Fields(components, components_label)
        if set(components) != {"valid", "count", "capacity", "slots"}:
            raise RunnerError(f"{object_label}.components fields are invalid")
        container_valid = components_check.boolean("valid")
        component_count = components_check.integer("count")
        component_capacity = components_check.integer("capacity")
        component_slots = components.get("slots")
        if not isinstance(component_slots, list):
            raise RunnerError(f"{object_label}.components.slots is invalid")
        if container_valid:
            if not (
                0 <= component_count <= component_capacity <= MAX_NATIVE_COMPONENT_SLOTS
                and len(component_slots) == component_count
            ):
                raise RunnerError(f"{object_label} has invalid component bounds")
        elif component_slots:
            raise RunnerError(f"{object_label} exposed slots from an invalid container")

        for expected_component_slot, raw_component in enumerate(component_slots):
            component_label = f"{object_label}.components.slots[{expected_component_slot}]"
            component = _response_mapping(raw_component, component_label)
            component_check = _Fields(component, component_label)
            component_fields = {
                "slot",
                "present",
                "ownerBacklinkValid",
                "vtableLibgOffset",
                "typeGetterLibgOffset",
                "engineType",
                "typeValid",
                "slotMatchesType",
            }
            if set(component) != component_fields:
                raise RunnerError(f"{object_label} component fields do not match the v2 schema")
            component_slot = component_check.integer("slot")
            if component_slot != expected_component_slot:
                raise RunnerError(f"{object_label} component slots are not contiguous")
            present = component_check.boolean("present")
            backlink = component.get("ownerBacklinkValid")
            if present:
                _response_bool(backlink, f"{component_label}.ownerBacklinkValid")
            elif backlink is not None:
                raise RunnerError(f"{object_label} absent component has a backlink result")

            type_valid = component_check.boolean("typeValid")
            vtable_offset = component.get("vtableLibgOffset")
            getter_offset = component.get("typeGetterLibgOffset")
            engine_type = component.get("engineType")
            slot_matches_type = component.get("slotMatchesType")
            for offset_value, offset_label in (
                (vtable_offset, "vtableLibgOffset"),
                (getter_offset, "typeGetterLibgOffset"),
            ):
                if offset_value is not None:
                    _response_int(offset_value, f"{component_label}.{offset_label}", minimum=0)
            if type_valid:
                _response_int(engine_type, f"{component_label}.engineType", minimum=0, maximum=31)
                _response_bool(slot_matches_type, f"{component_label}.slotMatchesType")
                if vtable_offset is None or getter_offset is None:
                    raise RunnerError(f"{object_label} valid component type lacks offsets")
            elif engine_type is not None or slot_matches_type is not None:
                raise RunnerError(f"{object_label} exposed an invalid component type")

    players = result.get("players")
    if not isinstance(players, list) or len(players) != 2:
        raise RunnerError("rich telemetry must expose exactly two player runtime records")
    for owner, player in enumerate(players):
        entity_references.extend(_validate_player_runtime(player, owner))
    for label, entity_key in entity_references:
        if entity_key not in entity_keys:
            raise RunnerError(f"{label} is outside the current bounded object vector")
    for name, validate in (
        ("combatEvents", _validate_combat_events),
        ("phaseRuntime", _validate_phase_runtime),
        ("specialMovementRuntime", _validate_special_movement_runtime),
        ("actionMovementRuntime", _validate_action_movement_runtime),
        ("characterStateRuntime", _validate_character_state_runtime),
        ("visibilityRuntime", _validate_visibility_runtime),
        ("remainingRuntime", _validate_remaining_runtime),
    ):
        if name in result:
            options = {"minimum_sequence": phase_event_floor} if name == "phaseRuntime" else {}
            validate(result.get(name), **identity, **options)
    return result


def _validate_atomic_observation(result: dict[str, Any], *, phase_event_floor: int | None = None) -> dict[str, Any]:
    """Validate one complete, same-manager ordinary + rich capture."""

    if result.get("schema") != ATOMIC_OBSERVATION_SCHEMA:
        raise RunnerError(f"unexpected atomic observation schema: {result.get('schema')!r}")
    if result.get("ok") is not True:
        raise RunnerError("native atomic observation is not successful")
    if result.get("atomic") is not True:
        raise RunnerError("native observation capture is not marked atomic")
    if _response_int(result.get("maxObjects"), "maxObjects", minimum=1) != MAX_OBSERVATION_OBJECTS:
        raise RunnerError("unexpected atomic observation object bound")
    identity = (
        _response_int(result.get("tick"), "tick", minimum=-1),
        _response_int(result.get("generation"), "generation", minimum=0),
        _response_int(result.get("stateEpoch"), "stateEpoch", minimum=0),
    )

    ordinary = _response_mapping(result.get("ordinary"), "ordinary")
    ordinary_check = _Fields(ordinary, "ordinary")
    rich = _response_mapping(result.get("rich"), "rich")
    for envelope, label in ((ordinary, "ordinary"), (rich, "rich")):
        if envelope.get("ok") is not True:
            raise RunnerError(f"atomic {label} envelope is not successful")
        nested_identity = (
            _response_int(envelope.get("tick"), f"{label}.tick", minimum=-1),
            _response_int(envelope.get("generation"), f"{label}.generation", minimum=0),
            _response_int(envelope.get("stateEpoch"), f"{label}.stateEpoch", minimum=0),
        )
        if nested_identity != identity:
            raise RunnerError(f"atomic {label} identity does not match capture identity")

    ordinary_count = ordinary_check.integer("count", minimum=0, maximum=MAX_OBSERVATION_OBJECTS)
    ordinary_returned = ordinary_check.integer("returned", minimum=0, maximum=MAX_OBSERVATION_OBJECTS)
    ordinary_truncated = ordinary_check.boolean("truncated")
    ordinary_objects = ordinary.get("objects")
    if (
        ordinary_truncated
        or ordinary_returned != ordinary_count
        or not isinstance(ordinary_objects, list)
        or len(ordinary_objects) != ordinary_returned
    ):
        raise RunnerError("atomic ordinary observation is incomplete")
    for expected_slot, raw_object in enumerate(ordinary_objects):
        item = _response_mapping(raw_object, f"ordinary.objects[{expected_slot}]")
        if _response_int(item.get("slot"), f"ordinary.objects[{expected_slot}].slot") != expected_slot:
            raise RunnerError("atomic ordinary object slots are not contiguous")

    validated_rich = _validate_rich_observation(dict(rich), phase_event_floor=phase_event_floor)
    if validated_rich["truncated"]:
        raise RunnerError("atomic rich observation is incomplete")
    if validated_rich["count"] != ordinary_count or validated_rich["returned"] != ordinary_returned:
        raise RunnerError("atomic ordinary/rich object counts do not match")
    return result


@dataclass(frozen=True, slots=True)
class DeployAction:
    """Low-level replay command. Prefer :class:`HandAction` for training."""

    player_id: int
    card_id: int
    card_parameter: int
    x: int
    y: int


@dataclass(frozen=True, slots=True)
class HandAction:
    """Play the card currently visible in one native hand slot."""

    owner: int
    hand_index: int
    x: int
    y: int


@dataclass(frozen=True, slots=True)
class AbilityAction:
    """Activate the ready ability of one exact live champion entity.

    The tagged, vector-unique native-object key is accepted here instead of
    the private ``cgid`` process value. Legacy nonnegative raw tuples remain
    accepted for persisted callers. The probe resolves either representation
    against the current bounded object graph and never exposes ``cgid``.
    """

    owner: int
    object_index: int
    secondary_index: int


class NativeClashEnv:
    """Synchronous two-sided environment backed by the native battle engine."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, timeout: float = 10.0) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        # A transition may inspect the current tick and then submit one atomic
        # native play request.  Keep both operations serialized while allowing
        # the helper to call the ordinary request path recursively.
        self._lock = threading.RLock()
        self._phase_validation_identity: tuple[int, int] | None = None
        self._phase_validation_next_sequence: int | None = None
        self._persistent_transport_requested = False
        self._persistent_transport_supported: bool | None = None
        self._persistent_connection: socket.socket | None = None
        self._persistent_reader: Any | None = None

    @staticmethod
    def _read_only_request(command: str) -> bool:
        return command in {
            "status",
            "multi-status",
            "observe",
            "observe-rich",
            "observe-atomic",
            "render status",
            "touch status",
            "attest",
        }

    def _close_persistent_transport_locked(self) -> None:
        reader = self._persistent_reader
        connection = self._persistent_connection
        self._persistent_reader = None
        self._persistent_connection = None
        if reader is not None:
            try:
                reader.close()
            except OSError:
                pass
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass

    def close_transport(self) -> None:
        """Close an optional persistent control session without engine state."""

        with self._lock:
            self._close_persistent_transport_locked()

    def _ensure_persistent_transport_locked(self) -> None:
        if self._persistent_connection is not None:
            return
        if self._persistent_transport_supported is False:
            raise RunnerError("persistent native control session is unsupported")
        connection = socket.create_connection((self.host, self.port), timeout=self.timeout)
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        reader = connection.makefile("rb")
        try:
            connection.sendall(b"session-v1\n")
            payload = reader.readline(4097)
            if len(payload) > 4096 or not payload.endswith(b"\n"):
                raise RunnerError("invalid persistent native session handshake")
            result = json.loads(payload.decode("utf-8"))
            if not result.get("ok") or result.get("session") != "control-session.v1":
                self._persistent_transport_supported = False
                raise RunnerError("persistent native control session is unsupported")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, RunnerError):
            try:
                reader.close()
            finally:
                connection.close()
            raise
        self._persistent_connection = connection
        self._persistent_reader = reader
        self._persistent_transport_supported = True
        return

    def _request(self, command: str, *, _read_retry_count: int = 2) -> dict[str, Any]:
        with self._lock:
            try:
                if self._persistent_transport_requested and self._persistent_transport_supported is not False:
                    self._ensure_persistent_transport_locked()
                    connection = self._persistent_connection
                    reader = self._persistent_reader
                    if connection is None or reader is None:
                        raise RunnerError("persistent native control session has no socket")
                    connection.sendall(command.encode("utf-8") + b"\n")
                    payload = reader.readline(MAX_RUNNER_RESPONSE_BYTES + 2)
                    if not payload.endswith(b"\n"):
                        raise RunnerError("persistent native control session ended mid-response")
                    response_bytes = len(payload)
                    if response_bytes > MAX_RUNNER_RESPONSE_BYTES:
                        raise RunnerError("native runner response exceeds the 16 MiB host bound")
                else:
                    connection = socket.create_connection((self.host, self.port), timeout=self.timeout)
                    with connection:
                        connection.sendall(command.encode("utf-8") + b"\n")
                        chunks: list[bytes] = []
                        response_bytes = 0
                        while True:
                            chunk = connection.recv(65536)
                            if not chunk:
                                break
                            response_bytes += len(chunk)
                            if response_bytes > MAX_RUNNER_RESPONSE_BYTES:
                                raise RunnerError("native runner response exceeds the 16 MiB host bound")
                            chunks.append(chunk)
                    payload = b"".join(chunks)
            except (OSError, TimeoutError, RunnerError) as error:
                self._close_persistent_transport_locked()
                if self._persistent_transport_supported is False:
                    return self._request(command, _read_retry_count=_read_retry_count)
                if self._read_only_request(command) and _read_retry_count > 0:
                    time.sleep(0.002)
                    return self._request(command, _read_retry_count=_read_retry_count - 1)
                if isinstance(error, RunnerError):
                    raise
                raise RunnerError(f"native runner transport failed for {command!r}") from error
        try:
            result = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            if self._read_only_request(command) and _read_retry_count > 0:
                time.sleep(0.002)
                return self._request(command, _read_retry_count=_read_retry_count - 1)
            raise RunnerError(f"invalid native runner response to {command!r}") from error
        if not result.get("ok"):
            error_text = str(result.get("error", result))
            diagnostics = ", ".join(
                f"{key}={result[key]}" for key in ("stage", "bytesUsed", "capacity") if key in result
            )
            if diagnostics:
                error_text = f"{error_text} ({diagnostics})"
            raise RunnerError(error_text)
        return result

    def status(self) -> dict[str, Any]:
        return self._request("status")

    def stop_resident_mode(self) -> dict[str, Any]:
        """Destroy all resident slots and restore legacy singleton commands."""

        return self._request("multi-stop")



    def attest(self, ruleset_manifest: Any | None = None) -> RunnerAttestationV1:
        """Read the live process identity and optionally bind it to a ruleset.

        This attests the files opened/mapped by the Android process. It never
        substitutes hashes from the host workspace for missing runner fields.
        """

        result = attestation_from_response(self._request("attest"))
        if ruleset_manifest is not None:
            result = verify_runner_attestation(result, ruleset_manifest)
        return result

    def set_rendering(self, enabled: bool) -> dict[str, Any]:
        """Enable or suppress native EGL frame submission for this process."""

        return self._request("render on" if enabled else "render off")

    def render_status(self) -> dict[str, Any]:
        return self._request("render status")

    def wait_ready(self, timeout: float = 15.0) -> dict[str, Any]:
        """Wait until the cold runner can accept a match configuration.

        A fresh app process intentionally has no manager yet. ``create_match``
        constructs it, so readiness here means the injected native runtime is
        listening rather than that a replay-created manager already exists.
        """

        deadline = time.monotonic() + timeout
        while True:
            status = self.status()
            if status.get("coldReady") or status.get("ready"):
                return status
            if time.monotonic() >= deadline:
                raise RunnerError("native cold runner did not become ready")
            time.sleep(0.05)

    def configure_replay(self, replay: str | Mapping[str, Any]) -> dict[str, Any]:
        """Install a fresh native battle schema and synchronously reset.

        Recorded command/event streams are expected to be empty. Prefer
        :meth:`create_match`, which enforces that invariant.
        """

        payload = replay if isinstance(replay, str) else json.dumps(replay, ensure_ascii=False, separators=(",", ":"))
        if not payload.startswith("{") or "\n" in payload or "\r" in payload:
            raise ValueError("replay configuration must be one compact JSON object")
        if len(payload.encode("utf-8")) >= 64 * 1024:
            raise ValueError("replay configuration exceeds the native 64 KiB limit")
        return self._request(f"configure {payload}")

    def configure_native_render(self, replay: str | Mapping[str, Any], *, wait_timeout: float = 15.0) -> dict[str, Any]:
        """Create a real stock battle scene directly from local match JSON.

        This uses the client's ReplayBattleController only as a native scene
        owner. The command/event arrays remain empty, so no recorded replay is
        played and neither login nor Battle Log navigation is required.
        """

        payload = replay if isinstance(replay, str) else json.dumps(replay, ensure_ascii=False, separators=(",", ":"))
        if not payload.startswith("{") or "\n" in payload or "\r" in payload:
            raise ValueError("native-render configuration must be one compact JSON object")
        if len(payload.encode("utf-8")) >= 64 * 1024:
            raise ValueError("native-render configuration exceeds the native 64 KiB limit")
        accepted = self._request(f"configure-native {payload}")
        sequence = int(accepted["sequence"])
        deadline = time.monotonic() + wait_timeout
        while True:
            status = self.status()
            if (
                status.get("mode") == "native-render"
                and status.get("nativeRenderReady")
                and int(status.get("nativeRenderLoaded", 0)) >= sequence
            ):
                return status
            if time.monotonic() >= deadline:
                raise RunnerError(f"native renderer did not load configuration {sequence}")
            time.sleep(0.05)

    def create_match(self, config: MatchConfig, *, template_path: str | Path | None = None) -> dict[str, Any]:
        """Construct a new match with caller-selected decks, rules, and seed."""

        payload = config.to_json() if template_path is None else config.to_json(template_path)
        self.configure_replay(payload)
        observation = self.observe()
        observed_decks = tuple(
            tuple(card["cardId"] for card in player["deck"])
            for player in sorted(observation["players"], key=lambda item: item["owner"])
        )
        expected_decks = (config.deck0, config.deck1)
        if observed_decks != expected_decks:
            raise RunnerError(f"native factory deck mismatch: expected {expected_decks}, got {observed_decks}")
        return observation

    def create_native_match(
        self, config: MatchConfig, *, template_path: str | Path | None = None, wait_timeout: float = 15.0
    ) -> dict[str, Any]:
        """Start a caller-configured match in Null's Royale's real renderer."""

        payload = config.to_json() if template_path is None else config.to_json(template_path)
        self.configure_native_render(payload, wait_timeout=wait_timeout)
        observation = self.observe()
        observed_decks = tuple(
            tuple(card["cardId"] for card in player["deck"])
            for player in sorted(observation["players"], key=lambda item: item["owner"])
        )
        expected_decks = (config.deck0, config.deck1)
        if observed_decks != expected_decks:
            raise RunnerError(f"native-render deck mismatch: expected {expected_decks}, got {observed_decks}")
        return observation

    def step(self, ticks: int = 1) -> dict[str, Any]:
        if not 1 <= ticks <= 1_000_000:
            raise ValueError("ticks must be in 1..1_000_000")
        return self._request(f"step {ticks}")

    def advance_native_render(self, ticks: int) -> dict[str, Any]:
        """Run an exact number of stock-renderer logic ticks, then pause."""

        if isinstance(ticks, bool) or not isinstance(ticks, int):
            raise TypeError("ticks must be an integer")
        if not 1 <= ticks <= 1_000_000:
            raise ValueError("ticks must be in 1..1_000_000")
        result = self._request(f"advance-native {ticks}")
        if (
            result.get("mode") != "native-render"
            or result.get("paused") is not True
            or int(result.get("requestedTicks", -1)) != ticks
        ):
            raise RunnerError("native synchronized advance returned an invalid receipt")
        return result

    def observe(self) -> dict[str, Any]:
        return self._request("observe")

    def observe_rich(self) -> dict[str, Any]:
        """Return validated native component/runtime telemetry.

        This is a separate infrastructure envelope and does not broaden the
        ordinary model observation contract. Pending or unavailable fields
        remain explicitly fail-closed in the returned capability profile.
        """

        return _validate_rich_observation(self._request("observe-rich"))

    def observe_atomic(self) -> dict[str, Any]:
        """Capture complete ordinary and rich envelopes at one native state.

        Both nested objects are encoded by the probe while it retains the same
        manager capture lock.  Tick, generation, stateEpoch, object counts, and
        non-truncation are revalidated here before either view is returned.
        """

        return self._validate_atomic_result(self._request("observe-atomic"))

    def _validate_atomic_result(self, result: Mapping[str, Any]) -> dict[str, Any]:
        identity = (result.get("generation"), result.get("stateEpoch"))
        phase_event_floor = (
            self._phase_validation_next_sequence if identity == self._phase_validation_identity else None
        )
        validated = _validate_atomic_observation(result, phase_event_floor=phase_event_floor)
        rich = validated.get("rich")
        phase_runtime = rich.get("phaseRuntime") if isinstance(rich, Mapping) else None
        if isinstance(phase_runtime, Mapping):
            self._phase_validation_identity = (int(validated["generation"]), int(validated["stateEpoch"]))
            self._phase_validation_next_sequence = int(phase_runtime["nextSequence"])
        else:
            self._phase_validation_identity = None
            self._phase_validation_next_sequence = None
        return validated

    def set_speed(self, multiplier: float) -> dict[str, Any]:
        """Set stock native-render logical speed without changing render FPS.

        Supported multipliers are exact and intentionally bounded so the
        renderer and its stock-owned manager remain synchronized.
        """

        if isinstance(multiplier, bool) or not isinstance(multiplier, (int, float)):
            raise ValueError("native-render speed must be one of 0.25, 0.5, 1, 2, 4")
        value = float(multiplier)
        if value not in NATIVE_RENDER_SPEEDS:
            raise ValueError("native-render speed must be one of 0.25, 0.5, 1, 2, 4")
        return self._request(f"speed {value:g}")

    def touch_status(self) -> dict[str, Any]:
        """Return the direct native-screen card-selection state."""

        return self._request("touch status")

    def pause(self) -> dict[str, Any]:
        """Pause native-render logical advancement for synchronized capture."""

        return self._request("pause")

    def resume(self) -> dict[str, Any]:
        """Resume a renderer paused by :meth:`pause`."""

        return self._request("resume")

    @staticmethod
    def _snapshot_handle_value(handle: int | Mapping[str, Any]) -> int:
        value: Any = handle.get("handle") if isinstance(handle, Mapping) else handle
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= (1 << 64) - 1:
            raise ValueError("snapshot handle must be a positive uint64")
        return value

    def create_snapshot(self) -> dict[str, Any]:
        """Retain a lossless native state snapshot inside the probe process.

        The returned handle is valid only for the current app process and
        headless manager generation. ``reset``/``create_match`` invalidates it;
        call :meth:`release_snapshot` when a branch is no longer needed.
        """

        return self._request("snapshot-create")

    def restore(self, handle: int | Mapping[str, Any]) -> dict[str, Any]:
        """Restore one process-local snapshot in place and return observation."""

        snapshot_handle = self._snapshot_handle_value(handle)
        with self._lock:
            receipt = self._request(f"restore {snapshot_handle}")
            observation = self.observe()
        observation["restore"] = receipt
        return observation

    def release_snapshot(self, handle: int | Mapping[str, Any]) -> dict[str, Any]:
        """Release one process-local native snapshot handle."""

        snapshot_handle = self._snapshot_handle_value(handle)
        return self._request(f"release-snapshot {snapshot_handle}")

    def inject_command(self, command: str | Mapping[str, Any]) -> dict[str, Any]:
        """Queue one exact native logic command without advancing the engine."""

        payload = (
            command if isinstance(command, str) else json.dumps(command, ensure_ascii=False, separators=(",", ":"))
        )
        if not payload.startswith("{") or "\n" in payload or "\r" in payload:
            raise ValueError("native command must be one compact JSON object")
        if len(payload.encode("utf-8")) >= 2048:
            raise ValueError("native command exceeds the probe's 2 KiB limit")
        return self._request(f"inject {payload}")

    def queue_hand_action_next_tick(self, action: HandAction) -> dict[str, Any]:
        """Queue a model play for the next tick without a live-render clock race."""

        with self._lock:
            self.pause()
            try:
                return self.queue_hand_action_at(action, execute_in_ticks=1)
            finally:
                self.resume()

    def queue_hand_action_at(
        self,
        action: HandAction,
        *,
        execute_tick: int | None = None,
        execute_in_ticks: int | None = None,
        clamp_late: bool = False,
    ) -> dict[str, Any]:
        """Queue a hand action for an exact observable application tick.

        Exactly one of ``execute_tick`` and ``execute_in_ticks`` may be given.
        Unlike :meth:`deploy`, these values describe the intended execution
        tick rather than the start of the native 20-tick command-age interval.
        The method does not advance time, so both owners can be
        queued from the same pre-step state before one shared ``step`` call.
        ``clamp_late`` keeps live policy commands valid if inference finishes
        too close to or after their observation-anchored deadline.
        """

        if action.owner not in (0, 1):
            raise ValueError("owner must be 0 or 1")
        if action.hand_index < 0:
            raise ValueError("hand_index must be non-negative")
        if execute_tick is not None and execute_in_ticks is not None:
            raise ValueError("pass execute_tick or execute_in_ticks, not both")
        if execute_in_ticks is not None and execute_in_ticks < 1:
            raise ValueError("execute_in_ticks must be positive")

        with self._lock:
            before = self.observe()
            current_tick = int(before["tick"])
            applied_tick = int(execute_tick) if execute_tick is not None else current_tick + int(execute_in_ticks or 1)
            if clamp_late and execute_tick is not None and applied_tick <= current_tick + 2:
                applied_tick = current_tick + LIVE_COMMAND_AGE_TICKS
            applied_tick = max(FIRST_PLAYABLE_TICK + 1, applied_tick)
            # The engine consumes a command while advancing from t2 to the
            # following state.  Therefore a state observed at applied_tick was
            # produced from a command whose age boundary is applied_tick - 1.
            boundary = applied_tick - 1
            try:
                player = next(item for item in before["players"] if item["owner"] == action.owner)
                card = next(item for item in player["hand"] if item["handIndex"] == action.hand_index)
            except StopIteration as error:
                raise ValueError("selected native hand slot is not available") from error
            if card.get("cardParameter") is None:
                raise RunnerError("selected native hand card is not encoded")
            descriptor = _decode_observed_hand_card(card)

            account_id = int(player["accountId"])
            command = {
                "ct": 86,
                "c": {
                    "t": boundary - LIVE_COMMAND_AGE_TICKS,
                    "t2": boundary,
                    "idHi": (account_id >> 32) & 0xFFFFFFFF,
                    "idLo": account_id & 0xFFFFFFFF,
                    "px": action.x,
                    "py": action.y,
                    "sid": -1,
                    "sel": {"os": _observed_command_card_id(card), "pd": int(card["cardParameter"])},
                },
            }
            result = self.inject_command(command)
            queued_tick_value = result.get("tick")
            if (
                queued_tick_value is None
                and result.get("mode") == "resident-headless"
                and result.get("injected") is True
            ):
                # Resident injection is synchronous and cannot advance this
                # controlled slot.  Older probes acknowledged the successful
                # queue operation without echoing its tick, so preserve exact
                # reset-and-replay compatibility while newer probes return the
                # tick explicitly.
                queued_tick_value = current_tick
            queued_tick = _response_int(queued_tick_value, "hand queued tick", minimum=0)
            if queued_tick >= applied_tick:
                raise RunnerError("native hand action was not queued before its absolute execute tick")
            return {
                **result,
                "owner": action.owner,
                "handIndex": action.hand_index,
                "cardId": int(card["cardId"]),
                "commandCardId": _observed_command_card_id(card),
                "cardParameter": descriptor.packed,
                "deckSlot": descriptor.deck_slot,
                "cost": descriptor.cost,
                "formCode": descriptor.form_code,
                "formName": descriptor.form_name,
                "queuedAtTick": queued_tick,
                "commandAgeBoundaryTick": boundary,
                "executeTick": applied_tick,
            }

    def queue_ability_action_at(self, action: AbilityAction, *, execute_in_ticks: int = 1) -> dict[str, Any]:
        """Queue an exact ready Champion ability at a future native tick."""

        if action.owner not in (0, 1):
            raise ValueError("owner must be 0 or 1")
        tagged_native_id = (
            action.object_index == NATIVE_OBJECT_ID_ENTITY_KEY_TAG and 1 <= action.secondary_index <= 0xFFFFFFFF
        )
        legacy_raw_tuple = action.object_index >= 0 and action.secondary_index >= 0
        if not tagged_native_id and not legacy_raw_tuple:
            raise ValueError("ability source identity is not a valid entity key")
        if not 1 <= execute_in_ticks <= 0xFFFF:
            raise ValueError("ability execute offset must be in 1..65535")
        result = self._request(
            f"activate-ability {action.owner} {action.object_index} {action.secondary_index} {execute_in_ticks}"
        )
        queued_tick = _response_int(result.get("tick"), "ability queued tick", minimum=0)
        return {
            **result,
            "owner": action.owner,
            "sourceEntityKey": [action.owner, action.object_index, action.secondary_index],
            "queuedAtTick": queued_tick,
            "executeTick": queued_tick + execute_in_ticks,
        }

    def clear_replay_schedule(self) -> dict[str, Any]:
        """Discard probe-owned semantic actions for the current renderer."""

        return self._request("replay-schedule-clear")

    def schedule_replay_card_at_tick(
        self, *, owner: int, card_id: int, x: int, y: int, execute_tick: int
    ) -> dict[str, Any]:
        """Resolve and queue one card inside native at ``execute_tick - 1``.

        This registers only the public card identity and target.  The probe's
        per-logic-tick hook reads the live hand/form at the exact command
        boundary, so rendered playback does not need to pause for telemetry.
        """

        if owner not in (0, 1):
            raise ValueError("owner must be 0 or 1")
        if card_id <= 0:
            raise ValueError("card_id must be positive")
        if not 0 <= x < 18_000 or not 0 <= y < 32_000:
            raise ValueError("replay card target is outside the native arena")
        if not 1 <= execute_tick <= 0x7FFFFFFF:
            raise ValueError("replay card execute tick must be in 1..2147483647")
        result = self._request(f"replay-schedule-card {owner} {card_id} {x} {y} {execute_tick}")
        sequence = _response_int(result.get("sequence"), "scheduled replay card sequence", minimum=1)
        registered_tick = _response_int(
            result.get("registeredAtTick"), "scheduled replay card registration tick", minimum=0
        )
        return {
            **result,
            "sequence": sequence,
            "owner": owner,
            "cardId": card_id,
            "registeredAtTick": registered_tick,
            "executeTick": execute_tick,
        }

    def schedule_replay_ability_at_tick(
        self, *, owner: int, ability_name_hints: Iterable[str] = (), execute_tick: int
    ) -> dict[str, Any]:
        """Resolve the unique queueable Champion in native at the exact boundary."""

        if owner not in (0, 1):
            raise ValueError("owner must be 0 or 1")
        if not 1 <= execute_tick <= 0x7FFFFFFF:
            raise ValueError("replay ability execute tick must be in 1..2147483647")
        normalized_hints = tuple(
            dict.fromkeys(
                "".join(character.casefold() for character in str(value) if character.isascii() and character.isalnum())
                for value in ability_name_hints
            )
        )
        hints = tuple(value for value in normalized_hints if value)
        encoded_hints = ",".join(hints) if hints else "-"
        if len(encoded_hints) > 260:
            raise ValueError("replay ability name hints are too long")
        result = self._request(f"replay-schedule-ability {owner} {encoded_hints} {execute_tick}")
        sequence = _response_int(result.get("sequence"), "scheduled replay ability sequence", minimum=1)
        registered_tick = _response_int(
            result.get("registeredAtTick"), "scheduled replay ability registration tick", minimum=0
        )
        return {
            **result,
            "sequence": sequence,
            "owner": owner,
            "abilityNameHints": hints,
            "registeredAtTick": registered_tick,
            "executeTick": execute_tick,
        }

    def replay_schedule_status(self, sequence: int) -> dict[str, Any]:
        """Read the boundary-time result of one probe-owned replay action."""

        if sequence < 1:
            raise ValueError("scheduled replay action sequence must be positive")
        result = self._request(f"replay-schedule-status {sequence}")
        state = str(result.get("state", ""))
        if state not in {"pending", "succeeded", "failed"}:
            raise RunnerError("native returned an invalid replay schedule state")
        return {
            **result,
            "sequence": _response_int(result.get("sequence"), "scheduled replay action sequence", minimum=1),
            "state": state,
            "executeTick": _response_int(result.get("executeTick"), "scheduled replay action execute tick", minimum=1),
        }

    def activate_ability(self, action: AbilityAction) -> dict[str, Any]:
        """Public alias for exact Champion activation."""

        return self.queue_ability_action_at(action)

    def deploy(self, action: DeployAction, delay_ticks: int = 1) -> dict[str, Any]:
        """Queue one already-encoded native command without advancing time."""

        if delay_ticks < 1:
            raise ValueError("delay_ticks must be positive")
        return self._request(
            f"deploy {action.player_id} {action.card_id} {action.card_parameter} {action.x} {action.y} {delay_ticks}"
        )

    def transition(
        self, actions: Iterable[HandAction] = (), *, delay_ticks: int | None = None, advance_ticks: int | None = None
    ) -> dict[str, Any]:
        """Play native hand slots, advance ticks, and return structured state.

        ``delay_ticks=None`` schedules decisions at the first legal native
        battle tick (or the next tick once play has started).
        ``advance_ticks=None`` asks the probe to cross the engine's command
        consumption boundary itself. Callers do not need to know the opening
        boundary, native live-command aging rule, account IDs, card IDs, costs,
        or packed card parameters. Both owners' actions are resolved from the
        same pre-step native state.
        """

        queued = tuple(actions)
        if len(queued) > MAX_TRANSITION_ACTIONS:
            raise ValueError(f"a transition supports at most {MAX_TRANSITION_ACTIONS} actions")
        if delay_ticks is not None and delay_ticks < 1:
            raise ValueError("delay_ticks must be positive or None")
        requested_advance = 0 if advance_ticks is None else advance_ticks
        if not 0 <= requested_advance <= 1_000_000:
            raise ValueError("advance_ticks must be None or in 0..1_000_000")
        for action in queued:
            if action.owner not in (0, 1):
                raise ValueError("owner must be 0 or 1")
            if action.hand_index < 0:
                raise ValueError("hand_index must be non-negative")

        with self._lock:
            if delay_ticks is None:
                current_tick = int(self.observe()["tick"])
                wire_delay = max(1, FIRST_PLAYABLE_TICK - current_tick)
            else:
                wire_delay = delay_ticks
            wire_advance = requested_advance
            if delay_ticks is None and queued and advance_ticks is not None:
                # A fixed frame skip must still reach the automatically chosen
                # legal command timestamp on the opening decision.  Later
                # decisions retain their requested frame skip (normally 22).
                wire_advance = max(wire_advance, wire_delay + COMMAND_CONSUMPTION_STEPS)
            fields = ["play", str(wire_delay), str(wire_advance), str(len(queued))]
            for action in queued:
                fields.extend((str(action.owner), str(action.hand_index), str(action.x), str(action.y)))
            return self._request(" ".join(fields))

    def play_immediate(self, action: HandAction) -> dict[str, Any]:
        """Consume one hand action on the next legal native tick.

        Live commands normally carry a 20-tick ``t``/``t2`` age and the
        ordinary transition API advances across that boundary.  For an
        interactive viewer, preserve the native age interval but backdate it
        to the current consumption boundary, then advance exactly one tick.
        """

        if action.owner not in (0, 1):
            raise ValueError("owner must be 0 or 1")
        if action.hand_index < 0:
            raise ValueError("hand_index must be non-negative")

        if self.status().get("mode") == "native-render":
            return self.transition((action,), delay_ticks=1, advance_ticks=0)

        with self._lock:
            before = self.observe()
            if int(before.get("queuedCommands", 0)) > 0:
                raise RunnerError("native command queue is not empty")
            try:
                player = next(item for item in before["players"] if item["owner"] == action.owner)
                card = next(item for item in player["hand"] if item["handIndex"] == action.hand_index)
            except StopIteration as error:
                raise ValueError("selected native hand slot is not available") from error
            if card.get("cardParameter") is None:
                raise RunnerError("selected native hand card is not encoded")
            _decode_observed_hand_card(card)

            account_id = int(player["accountId"])
            consumption_boundary = max(FIRST_PLAYABLE_TICK, int(before["tick"]))
            command = {
                "ct": 86,
                "c": {
                    "t": consumption_boundary - LIVE_COMMAND_AGE_TICKS,
                    "t2": consumption_boundary,
                    "idHi": (account_id >> 32) & 0xFFFFFFFF,
                    "idLo": account_id & 0xFFFFFFFF,
                    "px": action.x,
                    "py": action.y,
                    "sid": -1,
                    "sel": {"os": _observed_command_card_id(card), "pd": int(card["cardParameter"])},
                },
            }
            self._request("inject " + json.dumps(command, ensure_ascii=False, separators=(",", ":")))
            for recovery_tick in range(COMMAND_CONSUMPTION_STEPS):
                try:
                    self.step(1)
                    break
                except RunnerError as error:
                    if str(error) != "native hand card is invalid" or recovery_tick + 1 == COMMAND_CONSUMPTION_STEPS:
                        raise
                    # The action was already consumed while producing the
                    # failed step response. Some native cycle/form transitions
                    # expose a bounded descriptor sentinel while the hand
                    # rotation settles. Advance without re-injecting the
                    # command; every response must either validate or repeat
                    # the one exact recoverable error.
            else:
                raise AssertionError("bounded native hand step recovery did not terminate")
            for recovery_tick in range(COMMAND_CONSUMPTION_STEPS + 1):
                try:
                    return self.observe()
                except RunnerError as error:
                    if str(error) != "native hand card is invalid" or recovery_tick == COMMAND_CONSUMPTION_STEPS:
                        raise
                try:
                    self.step(1)
                except RunnerError as error:
                    if str(error) != "native hand card is invalid":
                        raise
            raise AssertionError("bounded native hand observation recovery did not terminate")




class ResidentNativeClashEnv(NativeClashEnv):
    """One resident match slot on a shared semantic-headless endpoint.

    Multiple instances may share the same host and port as long as each uses a
    distinct ``env_id``.  The probe keeps every native GameStateManager alive
    and serializes all libg work through its one resident execution lane.
    """

    # Rich hook rings are process-local.  The probe starts a clean telemetry
    # epoch whenever it rebinds the execution lane to another resident slot.
    # BattleEnv uses this explicit capability marker to distinguish that clean
    # boundary from an actual overwritten-event gap.
    resident_event_rings_rebind_on_switch = True

    def __init__(self, env_id: int, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, timeout: float = 10.0) -> None:
        if not 0 <= env_id < 16:
            raise ValueError("resident env_id must be in 0..15")
        super().__init__(host=host, port=port, timeout=timeout)
        self.env_id = env_id
        self._resident_closed = False

    def _raw_request(self, command: str, *, _read_retry_count: int = 2) -> dict[str, Any]:
        return super()._request(command, _read_retry_count=_read_retry_count)

    def _request(self, command: str, *, _read_retry_count: int = 2) -> dict[str, Any]:
        if self._resident_closed:
            raise RunnerError("resident native environment is closed")
        # Attestation and touch/render diagnostics describe the one process,
        # not an individual resident match.
        if (
            command == "attest"
            or command.startswith("touch ")
            or command.startswith("render ")
            or command.startswith(f"env {self.env_id} ")
        ):
            raw_command = command
        else:
            raw_command = f"env {self.env_id} {command}"
        return self._raw_request(raw_command, _read_retry_count=_read_retry_count)

    def wait_ready(self, timeout: float = 15.0) -> dict[str, Any]:
        """Wait for the shared probe process; a slot becomes ready on configure."""

        deadline = time.monotonic() + timeout
        while True:
            status = self._raw_request("multi-status")
            if status.get("ok"):
                return status
            if time.monotonic() >= deadline:
                raise RunnerError("resident native runner did not become ready")
            time.sleep(0.05)

    def resident_status(self) -> dict[str, Any]:
        return self._raw_request("multi-status")

    def resident_performance(self) -> dict[str, Any]:
        """Return cumulative native resident hot-path timing counters."""

        return self._raw_request("multi-perf")

    def reset_resident_performance(self) -> dict[str, Any]:
        """Reset native resident timing without changing any match state."""

        return self._raw_request("multi-perf-reset")

    def close(self) -> None:
        if self._resident_closed:
            return
        try:
            self._raw_request(f"env {self.env_id} close")
        finally:
            self._resident_closed = True
