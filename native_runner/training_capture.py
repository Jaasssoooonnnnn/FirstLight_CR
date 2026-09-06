"""Compact native capture used only by the semantic training data plane."""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Any, Sequence

from .rich_telemetry_adapter import (
    RichAbilityRuntimeTelemetry,
    RichActiveEffect,
    RichCombatDeploymentContext,
    RichCombatEntityFact,
    RichCombatEventEnvelope,
    RichCombatEventTelemetry,
    RichDaggerDuchessRuntimeTelemetry,
    RichEvolutionRuntimeTelemetry,
    RichObjectTelemetry,
    RichPhaseObjectTelemetry,
    RichPhaseRuntimeEnvelope,
    RichPlayerRuntimeTelemetry,
    RichProjectileTelemetry,
    RichRoyalChefRuntimeTelemetry,
    RichTelemetrySnapshot,
    RichTowerTroopRuntimeEnvelope,
)
from .phase_runtime import PhaseHookEvent, PhaseHookKind, PhaseRuntimeWindowV1


_MAGIC = b"CRTF"
_VERSION = 4
_NONE_I32 = -(2**31)
_HEADER = struct.Struct("<4sHHQQQiiiiiHH")
_PLAYER = struct.Struct("<iiiIIH6Bi")
_HAND_CARD = struct.Struct("<iiiiIi")
_DECK_CARD = struct.Struct("<ii")
_ABILITY = struct.Struct("<8iHH")
_EVOLUTION = struct.Struct("<9i")
_OBJECT = struct.Struct("<IiI15i")
_UINT32 = struct.Struct("<I")
_PROJECTILE = struct.Struct("<I7i")
_PHASE = struct.Struct("<I28i")
_EFFECT = struct.Struct("<IiiBH")
_COMBAT_ENVELOPE = struct.Struct("<IQQQQQH")
_COMBAT_EVENT = struct.Struct("<QiBBBBHQQQ8i")
_COMBAT_DEPLOYMENT = struct.Struct("<B3xQiIIIiii")
_COMBAT_FACT = struct.Struct("<II8i")
_PHASE_ENVELOPE = struct.Struct("<IQQQQQH")
_PHASE_EVENT = struct.Struct("<QiBBHQQiIiI10i")
_TOWER_RUNTIME = struct.Struct("<B3x10iI")
_TOWER_ENVELOPE = struct.Struct("<IQ")

_FLAG_ENDED = 1 << 0
_FLAG_FINALIZED = 1 << 1
_FLAG_TRUNCATED = 1 << 2

_PLAYER_OWNER_ROOT_VALID = 1 << 0
_PLAYER_OWNER_ROOT_PRESENT = 1 << 1
_PLAYER_ABILITY_VALID = 1 << 2
_PLAYER_EVOLUTION_VALID = 1 << 3

_OBJECT_NULL = 1 << 0
_OBJECT_HP = 1 << 1
_OBJECT_SHIELD = 1 << 2
_OBJECT_TARGET_VALIDATED = 1 << 3
_OBJECT_TARGET_PRESENT = 1 << 4
_OBJECT_PROJECTILE = 1 << 5
_OBJECT_PHASE = 1 << 6
_OBJECT_EFFECTS_VALID = 1 << 7
_OBJECT_INVISIBLE_VALID = 1 << 8
_OBJECT_DATA_GLOBAL_ID = 1 << 9
_OBJECT_TOWER_RUNTIME = 1 << 10

_ABILITY_BUTTON_LABELS = (
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
_PHASE_HOOK_KINDS = (
    None,
    PhaseHookKind.ATTACK_START,
    PhaseHookKind.ATTACK_RELEASE,
    PhaseHookKind.EFFECT_APPLY,
    PhaseHookKind.ATTACK_SCALE,
    PhaseHookKind.MOVEMENT_SCALE,
    PhaseHookKind.EFFECTIVE_MOVEMENT_SPEED,
    PhaseHookKind.MOVEMENT_DELTA,
    PhaseHookKind.TARGET_RESET,
    PhaseHookKind.CLASSIC_CHARGE_READY,
    PhaseHookKind.DEPLOY_SCALE,
)
_COMBAT_KINDS = (
    None,
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
)
_COMBAT_POOLS = ("none", "hitpoints", "built_in_shield", "buff_shield")
_COMBAT_TERMINAL_REASONS = ("none", "object_impact", "source_lost", "nonimpact_cancel", "unknown")
_CARD_FORM_LABELS = ("BasicForm", "EvoForm", "HeroForm", "AutoChessForm", "FormEnd", "FlexSlot")
_MIRROR_CARD_GLOBAL_ID = 28_000_006
_COMBAT_CAPACITY = 1024
_COMBAT_CONSUME_CARD_HOOK_OFFSET = 0xF38A68


class TrainingCaptureError(ValueError):
    """The compact capture is truncated or contradicts its wire contract."""


@dataclass(frozen=True, slots=True)
class TrainingAtomicCaptureV1:
    ordinary: dict[str, Any]
    snapshot: RichTelemetrySnapshot


class _Reader:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.offset = 0

    def unpack(self, schema: struct.Struct, label: str) -> tuple[Any, ...]:
        end = self.offset + schema.size
        if end > len(self.payload):
            raise TrainingCaptureError(f"{label} exceeds compact capture")
        result = schema.unpack_from(self.payload, self.offset)
        self.offset = end
        return result

    def bytes(self, size: int, label: str) -> bytes:
        if size < 0 or self.offset + size > len(self.payload):
            raise TrainingCaptureError(f"{label} exceeds compact capture")
        result = self.payload[self.offset : self.offset + size]
        self.offset += size
        return result


def _optional(value: int) -> int | None:
    return None if value == _NONE_I32 else value


def _phase_optional(value: int, *, tick: bool = False) -> int | None:
    if value == _NONE_I32 or (tick and value == -1):
        return None
    return value


def _key(objects: Sequence[RichObjectTelemetry | None], slot: int, label: str) -> tuple[int, int, int] | None:
    if slot == -1:
        return None
    if not 0 <= slot < len(objects):
        raise TrainingCaptureError(f"{label} object slot is out of range")
    item = objects[slot]
    if item is None:
        raise TrainingCaptureError(f"{label} references a null object slot")
    return item.entity_key


def _empty_combat_fact(*, validated: bool = False) -> RichCombatEntityFact:
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


def _combat_fact(reader: _Reader, label: str) -> RichCombatEntityFact:
    (
        flags,
        native_object_id,
        owner,
        object_index,
        secondary_index,
        card_id,
        object_kind,
        position_x,
        position_y,
        invisible_count,
    ) = reader.unpack(_COMBAT_FACT, label)
    if flags & ~0x7:
        raise TrainingCaptureError(f"{label} flags are invalid")
    validated = bool(flags & 0x1)
    present = bool(flags & 0x2)
    visibility_validated = bool(flags & 0x4)
    if not present:
        if visibility_validated:
            raise TrainingCaptureError(f"{label} validates visibility for an absent relation")
        return _empty_combat_fact(validated=validated)
    if not validated or native_object_id == 0:
        raise TrainingCaptureError(f"{label} exposes an invalid native entity")
    if not -1 <= object_kind <= 31 or (visibility_validated and invisible_count < 0):
        raise TrainingCaptureError(f"{label} state is invalid")
    return RichCombatEntityFact(
        validated=True,
        present=True,
        native_object_id=native_object_id,
        entity_key=(owner, -2, native_object_id),
        owner=owner,
        object_index=object_index,
        secondary_index=secondary_index,
        card_id=card_id,
        object_kind=object_kind,
        position=(position_x, position_y),
        visibility_validated=visibility_validated,
        invisible_count=(invisible_count if visibility_validated else None),
    )


def decode_training_capture(payload: bytes) -> TrainingAtomicCaptureV1:
    """Decode the current native V4 capture without a rich JSON intermediate."""

    reader = _Reader(payload)
    (
        magic,
        version,
        flags,
        generation,
        state_epoch,
        steps,
        tick,
        world_result_raw,
        crowns_zero,
        crowns_one,
        queued_commands,
        player_count,
        object_count,
    ) = reader.unpack(_HEADER, "header")
    if magic != _MAGIC or version != _VERSION:
        raise TrainingCaptureError(f"unsupported training capture magic/version: {magic!r}/{version}")
    if flags & ~(_FLAG_ENDED | _FLAG_FINALIZED | _FLAG_TRUNCATED):
        raise TrainingCaptureError("training capture contains unknown flags")
    if player_count != 2 or object_count > 256:
        raise TrainingCaptureError("training capture counts are invalid")
    if tick < 0:
        raise TrainingCaptureError("training capture tick is invalid")

    ordinary_players: list[dict[str, Any]] = []
    rich_player_payloads: list[
        tuple[
            int, int, tuple[RichAbilityRuntimeTelemetry, ...] | None, tuple[RichEvolutionRuntimeTelemetry, ...] | None
        ]
    ] = []
    for player_index in range(player_count):
        (
            owner,
            elixir_raw,
            crowns_raw,
            account_id_high,
            account_id_low,
            player_flags,
            hand_count,
            cycle_count,
            deck_count,
            ability_count,
            evolution_count,
            reserved,
            owner_root_slot,
        ) = reader.unpack(_PLAYER, f"player[{player_index}]")
        if (
            owner != player_index
            or reserved != 0
            or hand_count > 4
            or cycle_count > 8
            or deck_count > 8
            or ability_count > 2
            or evolution_count > 8
            or player_flags
            & ~(_PLAYER_OWNER_ROOT_VALID | _PLAYER_OWNER_ROOT_PRESENT | _PLAYER_ABILITY_VALID | _PLAYER_EVOLUTION_VALID)
        ):
            raise TrainingCaptureError(f"player[{player_index}] header is invalid")

        hand: list[dict[str, Any]] = []
        for index in range(hand_count):
            (hand_index, deck_slot, card_id, command_card_id, card_parameter, cost) = reader.unpack(
                _HAND_CARD, f"player[{owner}].hand[{index}]"
            )
            if hand_index < 0 or deck_slot < 0 or card_id <= 0 or command_card_id <= 0:
                raise TrainingCaptureError("compact hand card is invalid")
            item: dict[str, Any] = {"handIndex": hand_index, "deckSlot": deck_slot, "cardId": card_id}
            if card_parameter != 0xFFFFFFFF and cost >= 0:
                item.update({"commandCardId": command_card_id, "cardParameter": card_parameter, "cost": cost})
            else:
                item.update({"cardParameter": None, "cost": None})
            hand.append(item)

        cycle: list[dict[str, int]] = []
        for index in range(cycle_count):
            deck_slot, card_id = reader.unpack(_DECK_CARD, f"player[{owner}].cycle[{index}]")
            if deck_slot < 0 or card_id <= 0:
                raise TrainingCaptureError("compact cycle card is invalid")
            cycle.append({"cycleIndex": index, "deckSlot": deck_slot, "cardId": card_id})

        deck: list[dict[str, int]] = []
        for index in range(deck_count):
            deck_slot, card_id = reader.unpack(_DECK_CARD, f"player[{owner}].deck[{index}]")
            if deck_slot != index or card_id <= 0:
                raise TrainingCaptureError("compact deck card is invalid")
            deck.append({"deckSlot": deck_slot, "cardId": card_id})

        abilities: list[RichAbilityRuntimeTelemetry] = []
        for index in range(ability_count):
            (
                controller_slot,
                action_data_global_id,
                selected_character_global_id,
                remaining_cooldown_ms,
                configured_cooldown_ms,
                remaining_charges_raw,
                max_charges,
                button_state,
                name_size,
                champion_count,
            ) = reader.unpack(_ABILITY, f"player[{owner}].ability[{index}]")
            if (
                not 1 <= controller_slot <= 2
                or action_data_global_id <= 0
                or selected_character_global_id <= 0
                or remaining_cooldown_ms < 0
                or configured_cooldown_ms < remaining_cooldown_ms
                or not 1 <= name_size <= 128
                or champion_count > 256
                or not 0 <= button_state < len(_ABILITY_BUTTON_LABELS)
            ):
                raise TrainingCaptureError("compact ability runtime is invalid")
            try:
                action_name = reader.bytes(name_size, f"player[{owner}].ability[{index}].name").decode("utf-8")
            except UnicodeDecodeError as error:
                raise TrainingCaptureError("compact ability name is not UTF-8") from error
            champion_slots = tuple(
                reader.unpack(struct.Struct("<i"), f"player[{owner}].ability[{index}].champion[{champion_index}]")[0]
                for champion_index in range(champion_count)
            )
            abilities.append(
                RichAbilityRuntimeTelemetry(
                    controller_slot=controller_slot,
                    action_data_global_id=action_data_global_id,
                    action_data_name=action_name,
                    selected_character_data_global_id=(selected_character_global_id),
                    remaining_cooldown_ms=remaining_cooldown_ms,
                    configured_cooldown_ms=configured_cooldown_ms,
                    remaining_charges_raw=remaining_charges_raw,
                    max_charges=max_charges,
                    button_state=button_state,
                    button_state_label=_ABILITY_BUTTON_LABELS[button_state],
                    available=button_state in {2, 4},
                    # Resolved after the object table has been decoded.
                    champion_entity_keys=champion_slots,  # type: ignore[arg-type]
                )
            )

        evolutions: list[RichEvolutionRuntimeTelemetry] = []
        for index in range(evolution_count):
            (
                deck_slot,
                card_id,
                base_spell_global_id,
                evolvable_raw,
                evolution_form_global_id,
                progress,
                cycle_required,
                cycle_remaining,
                ready_raw,
            ) = reader.unpack(_EVOLUTION, f"player[{owner}].evolution[{index}]")
            if (
                deck_slot != index
                or card_id <= 0
                or base_spell_global_id != card_id
                or evolvable_raw not in (0, 1)
                or ready_raw not in (_NONE_I32, 0, 1)
            ):
                raise TrainingCaptureError("compact evolution runtime is invalid")
            evolutions.append(
                RichEvolutionRuntimeTelemetry(
                    deck_slot=deck_slot,
                    card_id=card_id,
                    base_spell_global_id=base_spell_global_id,
                    evolvable=bool(evolvable_raw),
                    evolution_form_global_id=_optional(evolution_form_global_id),
                    progress=progress,
                    cycle_required=_optional(cycle_required),
                    cycle_remaining=_optional(cycle_remaining),
                    ready=(None if ready_raw == _NONE_I32 else bool(ready_raw)),
                )
            )

        account_id = (account_id_high << 32) | account_id_low
        ordinary_players.append(
            {
                "owner": owner,
                "accountId": account_id,
                "accountIdHigh": account_id_high,
                "accountIdLow": account_id_low,
                "elixirRaw": elixir_raw,
                "elixir": elixir_raw / 10_000.0,
                "crownsRaw": crowns_raw,
                "hand": hand,
                "cycle": cycle,
                "nextCard": ({"deckSlot": cycle[0]["deckSlot"], "cardId": cycle[0]["cardId"]} if cycle else None),
                "deck": deck,
            }
        )
        rich_player_payloads.append(
            (
                player_flags,
                owner_root_slot,
                (tuple(abilities) if player_flags & _PLAYER_ABILITY_VALID else None),
                (tuple(evolutions) if player_flags & _PLAYER_EVOLUTION_VALID else None),
            )
        )

    ordinary_objects: list[dict[str, Any]] = []
    objects: list[RichObjectTelemetry | None] = []
    pending_targets: list[int] = []
    pending_projectiles: list[tuple[int, tuple[int, int, int], tuple[int, int], int, bool] | None] = []
    pending_effects: list[tuple[tuple[int, int, bool, str, int], ...] | None] = []
    for expected_slot in range(object_count):
        (
            object_flags,
            slot,
            native_object_id,
            owner,
            card_id,
            x,
            y,
            target_x,
            target_y,
            object_index,
            secondary_index,
            hitpoints,
            max_hitpoints,
            shield,
            max_shield,
            attack_stage,
            invisible_count,
            target_slot,
        ) = reader.unpack(_OBJECT, f"object[{expected_slot}]")
        if slot != expected_slot or object_flags & ~(
            _OBJECT_NULL
            | _OBJECT_HP
            | _OBJECT_SHIELD
            | _OBJECT_TARGET_VALIDATED
            | _OBJECT_TARGET_PRESENT
            | _OBJECT_PROJECTILE
            | _OBJECT_PHASE
            | _OBJECT_EFFECTS_VALID
            | _OBJECT_INVISIBLE_VALID
            | _OBJECT_DATA_GLOBAL_ID
            | _OBJECT_TOWER_RUNTIME
        ):
            raise TrainingCaptureError(f"object[{expected_slot}] header is invalid")
        if object_flags & _OBJECT_NULL:
            if object_flags != _OBJECT_NULL:
                raise TrainingCaptureError(f"object[{expected_slot}] null row carries live fields")
            ordinary_objects.append({"slot": slot, "null": True})
            objects.append(None)
            pending_targets.append(-1)
            pending_projectiles.append(None)
            pending_effects.append(None)
            continue
        if native_object_id == 0:
            raise TrainingCaptureError("compact live object has zero identity")
        if not object_flags & _OBJECT_DATA_GLOBAL_ID:
            raise TrainingCaptureError("compact live object lacks exact LogicData identity")
        data_global_id = None
        if object_flags & _OBJECT_DATA_GLOBAL_ID:
            data_global_id = int(reader.unpack(_UINT32, f"object[{slot}].data_global_id")[0])
            if data_global_id <= 0:
                raise TrainingCaptureError("compact live object has invalid LogicData identity")
        entity_key = (owner, -2, native_object_id)
        ordinary_item: dict[str, Any] = {
            "slot": slot,
            "nativeObjectId": native_object_id,
            "owner": owner,
            "cardId": card_id,
            "x": x,
            "y": y,
            "targetX": target_x,
            "targetY": target_y,
            "objectIndex": object_index,
            "secondaryIndex": secondary_index,
            "hp": (hitpoints if object_flags & _OBJECT_HP else None),
            "maxHp": (max_hitpoints if object_flags & _OBJECT_HP else None),
        }
        ordinary_objects.append(ordinary_item)

        projectile_pending = None
        if object_flags & _OBJECT_PROJECTILE:
            (
                projectile_data_global_id,
                source_slot,
                projectile_target_slot,
                homing_slot,
                destination_x,
                destination_y,
                validation_mask,
                terminal_raw,
            ) = reader.unpack(_PROJECTILE, f"object[{slot}].projectile")
            if projectile_data_global_id <= 0 or validation_mask & ~0x7 or terminal_raw not in (0, 1):
                raise TrainingCaptureError("compact projectile is invalid")
            projectile_pending = (
                projectile_data_global_id,
                (source_slot, projectile_target_slot, homing_slot),
                (destination_x, destination_y),
                validation_mask,
                bool(terminal_raw),
            )

        phase = None
        if object_flags & _OBJECT_PHASE:
            phase_values = reader.unpack(_PHASE, f"object[{slot}].phase")
            phase_flags = int(phase_values[0])
            if phase_flags & ~0x7:
                raise TrainingCaptureError("compact phase flags are invalid")
            values = tuple(int(value) for value in phase_values[1:])
            phase = RichPhaseObjectTelemetry(
                attack_validated=bool(phase_flags & 0x1),
                movement_validated=bool(phase_flags & 0x2),
                buffs_validated=bool(phase_flags & 0x4),
                attack_sequence_stage=(None if values[0] in (_NONE_I32, -1) else values[0]),
                attack_timeline_ms=_optional(values[1]),
                load_remaining_ms=_optional(values[2]),
                deploy_remaining_ms=_optional(values[3]),
                deploy_previous_ms=_optional(values[4]),
                configured_deploy_time_ms=_optional(values[5]),
                hit_speed_ms=_optional(values[6]),
                attack_dash_time_ms=_optional(values[7]),
                base_movement_speed=_optional(values[8]),
                charge_speed_multiplier=_optional(values[9]),
                classic_charge_progress=_optional(values[10]),
                speed_positive_percent=_optional(values[11]),
                speed_negative_magnitude=_optional(values[12]),
                hit_speed_positive_percent=_optional(values[13]),
                hit_speed_negative_magnitude=_optional(values[14]),
                attack_step_input=_optional(values[15]),
                attack_step_output=_optional(values[16]),
                attack_step_tick=_phase_optional(values[17], tick=True),
                movement_step_input=_optional(values[18]),
                movement_step_output=_optional(values[19]),
                movement_step_tick=_phase_optional(values[20], tick=True),
                deploy_step_input=_optional(values[21]),
                deploy_step_output=_optional(values[22]),
                deploy_step_tick=_phase_optional(values[23], tick=True),
                effective_movement_speed=_optional(values[24]),
                effective_movement_speed_tick=_phase_optional(values[25], tick=True),
                movement_delta=_optional(values[26]),
                movement_delta_tick=_phase_optional(values[27], tick=True),
                attack_sequence_progress_raw=None,
                attack_sequence_progress_limit=None,
                attack_sequence_decay_remaining_ms=None,
                attack_sequence_decay_duration_ms=None,
            )

        tower_runtime = None
        if object_flags & _OBJECT_TOWER_RUNTIME:
            (
                tower_kind,
                observed_tick,
                charge_count,
                max_charge_count,
                recharge_elapsed_ms,
                recharge_duration_ms,
                start_delay_remaining_ms,
                start_delay_duration_ms,
                cooking_contribution,
                contribution_needed,
                throw_delay_remaining_ms,
                target_native_object_id,
            ) = reader.unpack(_TOWER_RUNTIME, f"object[{slot}].tower_runtime")
            if not 0 <= observed_tick <= tick:
                raise TrainingCaptureError("compact tower-troop observed tick is invalid")
            if tower_kind == 1:
                if (
                    not 1 <= max_charge_count <= 32
                    or not 0 <= charge_count <= max_charge_count
                    or not 1 <= recharge_duration_ms <= 60_000
                    or not 0 <= recharge_elapsed_ms <= recharge_duration_ms
                    or any(
                        value != 0
                        for value in (
                            start_delay_remaining_ms,
                            start_delay_duration_ms,
                            cooking_contribution,
                            contribution_needed,
                            target_native_object_id,
                        )
                    )
                    or throw_delay_remaining_ms != -1
                ):
                    raise TrainingCaptureError("compact Dagger Duchess runtime is invalid")
                tower_runtime = RichDaggerDuchessRuntimeTelemetry(
                    observed_tick=observed_tick,
                    charge_count=charge_count,
                    max_charge_count=max_charge_count,
                    recharge_elapsed_ms=recharge_elapsed_ms,
                    recharge_duration_ms=recharge_duration_ms,
                )
            elif tower_kind == 2:
                if (
                    any(
                        value != 0
                        for value in (charge_count, max_charge_count, recharge_elapsed_ms, recharge_duration_ms)
                    )
                    or not 1 <= start_delay_duration_ms <= 60_000
                    or not 0 <= start_delay_remaining_ms <= start_delay_duration_ms
                    or not 0 <= cooking_contribution <= 100_000_000
                    or not 1 <= contribution_needed <= 100_000_000
                    or (throw_delay_remaining_ms not in {-50, -1} and throw_delay_remaining_ms < 0)
                    or (throw_delay_remaining_ms in {-50, -1} and target_native_object_id != 0)
                ):
                    raise TrainingCaptureError("compact Royal Chef runtime is invalid")
                tower_runtime = RichRoyalChefRuntimeTelemetry(
                    observed_tick=observed_tick,
                    start_delay_remaining_ms=start_delay_remaining_ms,
                    start_delay_duration_ms=start_delay_duration_ms,
                    cooking_contribution=cooking_contribution,
                    contribution_needed=contribution_needed,
                    throw_delay_remaining_ms=(None if throw_delay_remaining_ms == -1 else throw_delay_remaining_ms),
                    target_native_object_id=(None if target_native_object_id == 0 else target_native_object_id),
                )
            else:
                raise TrainingCaptureError("compact tower-troop runtime kind is invalid")

        effect_count = reader.unpack(struct.Struct("<H"), f"object[{slot}].effect_count")[0]
        if effect_count > 64:
            raise TrainingCaptureError("compact effect count is invalid")
        effects_pending: list[tuple[int, int, bool, str]] = []
        for effect_index in range(effect_count):
            (buff_global_id, remaining_ms, source_slot, source_validated, name_size) = reader.unpack(
                _EFFECT, f"object[{slot}].effect[{effect_index}]"
            )
            if buff_global_id == 0 or remaining_ms < -1 or source_validated not in (0, 1) or not 1 <= name_size <= 128:
                raise TrainingCaptureError("compact active effect is invalid")
            try:
                name = reader.bytes(name_size, f"object[{slot}].effect[{effect_index}].name").decode("utf-8")
            except UnicodeDecodeError as error:
                raise TrainingCaptureError("compact effect name is not UTF-8") from error
            effects_pending.append((buff_global_id, remaining_ms, bool(source_validated), name, source_slot))

        objects.append(
            RichObjectTelemetry(
                slot=slot,
                native_object_id=native_object_id,
                entity_key=entity_key,
                owner=owner,
                object_index=object_index,
                secondary_index=secondary_index,
                card_id=card_id,
                data_global_id=data_global_id,
                shield_current=(shield if object_flags & _OBJECT_SHIELD else None),
                shield_maximum=(max_shield if object_flags & _OBJECT_SHIELD else None),
                target_entity_key=None,
                target_entity_validated=bool(object_flags & _OBJECT_TARGET_VALIDATED),
                attack_sequence_stage=(_optional(attack_stage) if attack_stage != -1 else None),
                active_effects=(() if object_flags & _OBJECT_EFFECTS_VALID else None),
                invisible_count=(invisible_count if object_flags & _OBJECT_INVISIBLE_VALID else None),
                visibility_state=(
                    ("invisible" if invisible_count > 0 else "visible")
                    if object_flags & _OBJECT_INVISIBLE_VALID
                    else None
                ),
                projectile=None,
                entity_resource_runtime=None,
                periodic_attack_modifier_runtime=None,
                capture_runtime=None,
                threshold_relocation_runtime=None,
                phase_runtime=phase,
                tower_troop_runtime=tower_runtime,
            )
        )
        pending_targets.append(target_slot)
        pending_projectiles.append(projectile_pending)
        pending_effects.append(tuple(effects_pending))

    objects_by_key = {item.entity_key: item for item in objects if item is not None}
    if len(objects_by_key) != sum(item is not None for item in objects):
        raise TrainingCaptureError("compact object identities are not unique")

    resolved_objects: list[RichObjectTelemetry | None] = []
    for slot, item in enumerate(objects):
        if item is None:
            resolved_objects.append(None)
            continue
        target = _key(objects, pending_targets[slot], f"object[{slot}].target")
        projectile_payload = pending_projectiles[slot]
        projectile = None
        if projectile_payload is not None:
            (data_id, reference_slots, destination, validation_mask, terminal) = projectile_payload
            source_key, target_key, homing_key = (
                _key(objects, reference_slot, f"object[{slot}].projectile") for reference_slot in reference_slots
            )
            projectile = RichProjectileTelemetry(
                projectile_data_global_id=data_id,
                source_entity_key=source_key,
                source_entity_validated=bool(validation_mask & 0x1),
                target_entity_key=target_key,
                target_entity_validated=bool(validation_mask & 0x2),
                homing_target_entity_key=homing_key,
                homing_target_entity_validated=bool(validation_mask & 0x4),
                destination=destination,
                terminal=terminal,
                native_phase=("terminal_or_finished_processing" if terminal else "in_flight"),
            )
        effects_payload = pending_effects[slot]
        effects = None
        if effects_payload is not None:
            effects = tuple(
                RichActiveEffect(
                    buff_global_id=buff_id,
                    name=name,
                    remaining_ms=remaining_ms,
                    source_entity_key=_key(objects, source_slot, f"object[{slot}].effect"),
                    source_entity_validated=source_validated,
                )
                for (buff_id, remaining_ms, source_validated, name, source_slot) in effects_payload
            )
        resolved_objects.append(
            RichObjectTelemetry(
                slot=item.slot,
                native_object_id=item.native_object_id,
                entity_key=item.entity_key,
                owner=item.owner,
                object_index=item.object_index,
                secondary_index=item.secondary_index,
                card_id=item.card_id,
                data_global_id=item.data_global_id,
                shield_current=item.shield_current,
                shield_maximum=item.shield_maximum,
                target_entity_key=target,
                target_entity_validated=item.target_entity_validated,
                attack_sequence_stage=item.attack_sequence_stage,
                active_effects=effects,
                invisible_count=item.invisible_count,
                visibility_state=item.visibility_state,
                projectile=projectile,
                entity_resource_runtime=item.entity_resource_runtime,
                periodic_attack_modifier_runtime=(item.periodic_attack_modifier_runtime),
                capture_runtime=item.capture_runtime,
                threshold_relocation_runtime=(item.threshold_relocation_runtime),
                phase_runtime=item.phase_runtime,
                tower_troop_runtime=item.tower_troop_runtime,
            )
        )

    resolved_by_key = {item.entity_key: item for item in resolved_objects if item is not None}
    resolved_by_native_object_id = {item.native_object_id: item for item in resolved_objects if item is not None}
    if len(resolved_by_native_object_id) != len(resolved_by_key):
        raise TrainingCaptureError("compact native object identities are not unique")
    rich_players: dict[int, RichPlayerRuntimeTelemetry] = {}
    for owner, (player_flags, owner_root_slot, abilities, evolutions) in enumerate(rich_player_payloads):
        resolved_abilities = None
        if abilities is not None:
            resolved_abilities = tuple(
                RichAbilityRuntimeTelemetry(
                    controller_slot=ability.controller_slot,
                    action_data_global_id=ability.action_data_global_id,
                    action_data_name=ability.action_data_name,
                    selected_character_data_global_id=(ability.selected_character_data_global_id),
                    remaining_cooldown_ms=ability.remaining_cooldown_ms,
                    configured_cooldown_ms=ability.configured_cooldown_ms,
                    remaining_charges_raw=ability.remaining_charges_raw,
                    max_charges=ability.max_charges,
                    button_state=ability.button_state,
                    button_state_label=ability.button_state_label,
                    available=ability.available,
                    champion_entity_keys=tuple(
                        key
                        for slot in ability.champion_entity_keys
                        if (key := _key(resolved_objects, int(slot), f"player[{owner}].ability")) is not None
                    ),
                )
                for ability in abilities
            )
        rich_players[owner] = RichPlayerRuntimeTelemetry(
            owner=owner,
            owner_root_validated=bool(player_flags & _PLAYER_OWNER_ROOT_VALID),
            owner_entity_key=(
                _key(resolved_objects, owner_root_slot, f"player[{owner}].owner_root")
                if player_flags & _PLAYER_OWNER_ROOT_PRESENT
                else None
            ),
            ability_runtime=resolved_abilities,
            evolution_runtime=evolutions,
        )

    (
        combat_flags,
        combat_epoch_first,
        combat_oldest,
        combat_next,
        combat_overflow,
        combat_rejected,
        combat_event_count,
    ) = reader.unpack(_COMBAT_ENVELOPE, "combat_envelope")
    if (
        combat_flags != 0x7
        or combat_epoch_first < 1
        or combat_oldest < combat_epoch_first
        or combat_next < combat_oldest
        or combat_rejected != 0
        or combat_event_count != combat_next - combat_oldest
        or combat_event_count > _COMBAT_CAPACITY
        or combat_overflow != max(0, combat_next - combat_epoch_first - _COMBAT_CAPACITY)
        or combat_oldest != max(combat_epoch_first, combat_next - _COMBAT_CAPACITY)
    ):
        raise TrainingCaptureError("compact combat envelope is invalid")

    combat_events: list[RichCombatEventTelemetry] = []
    deployment_contexts: dict[int, RichCombatDeploymentContext] = {}
    card_play_deployments: set[int] = set()
    for event_index in range(combat_event_count):
        (
            sequence,
            event_tick,
            kind_raw,
            pool_raw,
            terminal_raw,
            event_flags,
            reserved,
            hook_offset,
            caller_offset_raw,
            cause_sequence_raw,
            requested_amount_raw,
            actual_amount_raw,
            pre_hp_raw,
            post_hp_raw,
            pre_builtin_shield_raw,
            post_builtin_shield_raw,
            pre_buff_shield_raw,
            post_buff_shield_raw,
        ) = reader.unpack(_COMBAT_EVENT, f"combat_event[{event_index}]")
        if (
            sequence != combat_oldest + event_index
            or event_tick < -1
            or event_tick > tick
            or not 1 <= kind_raw < len(_COMBAT_KINDS)
            or not 0 <= pool_raw < len(_COMBAT_POOLS)
            or not 0 <= terminal_raw < len(_COMBAT_TERMINAL_REASONS)
            or event_flags & ~0x7
            or reserved != 0
            or hook_offset == 0
            or bool(event_flags & 0x2) != (caller_offset_raw != 0)
            or bool(event_flags & 0x4) != (cause_sequence_raw != 0)
            or (cause_sequence_raw != 0 and not (combat_epoch_first <= cause_sequence_raw < sequence))
        ):
            raise TrainingCaptureError(f"compact combat event[{event_index}] is invalid")
        kind = _COMBAT_KINDS[kind_raw]
        if kind is None:
            raise TrainingCaptureError("compact combat kind is invalid")

        (
            deployment_validated,
            deployment_sequence,
            deployment_owner,
            played_card_global_id,
            effective_card_global_id_raw,
            card_parameter,
            deck_slot,
            cost,
            form_code,
        ) = reader.unpack(_COMBAT_DEPLOYMENT, f"combat_event[{event_index}].deployment")
        if deployment_validated not in (0, 1):
            raise TrainingCaptureError(f"compact combat event[{event_index}] deployment is invalid")
        deployment = None
        if deployment_validated:
            if (
                deployment_sequence < 1
                or deployment_owner not in (0, 1)
                or played_card_global_id == 0
                or not 0 <= deck_slot <= 7
                or not 0 <= cost <= 15
                or not 0 <= form_code < len(_CARD_FORM_LABELS)
                or ((card_parameter >> 22) & 0x3F) - 1 != deck_slot
                or card_parameter >> 28 != cost
                or card_parameter & 0xF != form_code
                or (
                    form_code == 0
                    and effective_card_global_id_raw != played_card_global_id
                    and not (played_card_global_id == _MIRROR_CARD_GLOBAL_ID and effective_card_global_id_raw != 0)
                )
                or (form_code == 1 and effective_card_global_id_raw == 0)
                or kind not in {"spawn", "projectile_spawn", "card_play"}
            ):
                raise TrainingCaptureError(f"compact combat event[{event_index}] deployment descriptor is invalid")
            deployment = RichCombatDeploymentContext(
                deployment_sequence=deployment_sequence,
                owner=deployment_owner,
                played_card_global_id=played_card_global_id,
                effective_card_global_id=(effective_card_global_id_raw or None),
                card_parameter=card_parameter,
                deck_slot=deck_slot,
                cost=cost,
                form_code=form_code,
                form_name=_CARD_FORM_LABELS[form_code],
            )
            prior_deployment = deployment_contexts.setdefault(deployment_sequence, deployment)
            if prior_deployment != deployment:
                raise TrainingCaptureError("compact combat deployment sequence changed identity")
        elif kind == "card_play":
            raise TrainingCaptureError(f"compact combat event[{event_index}] card play lacks deployment provenance")
        if kind == "card_play":
            if hook_offset != _COMBAT_CONSUME_CARD_HOOK_OFFSET or deployment_sequence in card_play_deployments:
                raise TrainingCaptureError(f"compact combat event[{event_index}] card play is invalid")
            card_play_deployments.add(deployment_sequence)

        target = _combat_fact(reader, f"combat_event[{event_index}].target")
        immediate_source = _combat_fact(reader, f"combat_event[{event_index}].immediate_source")
        source = _combat_fact(reader, f"combat_event[{event_index}].source")
        projectile = _combat_fact(reader, f"combat_event[{event_index}].projectile")
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
        if (
            (target_required and not target.present)
            or (projectile_required and not projectile.present)
            or (deployment is not None and target.owner in (0, 1) and target.owner != deployment.owner)
        ):
            raise TrainingCaptureError(f"compact combat event[{event_index}] relations are invalid")

        amount_values = tuple(
            _optional(value)
            for value in (
                requested_amount_raw,
                actual_amount_raw,
                pre_hp_raw,
                post_hp_raw,
                pre_builtin_shield_raw,
                post_builtin_shield_raw,
                pre_buff_shield_raw,
                post_buff_shield_raw,
            )
        )
        if any(value is not None and value < 0 for value in amount_values):
            raise TrainingCaptureError(f"compact combat event[{event_index}] amount is invalid")
        (
            requested_amount,
            actual_amount,
            pre_hp,
            post_hp,
            pre_builtin_shield,
            post_builtin_shield,
            pre_buff_shield,
            post_buff_shield,
        ) = amount_values
        if kind in {"damage", "death", "heal", "shield_damage", "shield_break"} and (
            requested_amount is None or actual_amount is None or actual_amount <= 0
        ):
            raise TrainingCaptureError(f"compact combat event[{event_index}] lacks exact amount")
        absent = _empty_combat_fact()
        combat_events.append(
            RichCombatEventTelemetry(
                sequence=sequence,
                tick=event_tick,
                generation=generation,
                state_epoch=state_epoch,
                kind=kind,
                hook_offset=hook_offset,
                caller_offset=(caller_offset_raw if event_flags & 0x2 else None),
                cause_sequence=(cause_sequence_raw if event_flags & 0x4 else None),
                pool=_COMBAT_POOLS[pool_raw],
                terminal_reason=_COMBAT_TERMINAL_REASONS[terminal_raw],
                lethal=bool(event_flags & 0x1),
                deployment_context=deployment,
                target=target,
                immediate_source=immediate_source,
                source=source,
                projectile=projectile,
                related=absent,
                route_source_before=absent,
                route_target_before=absent,
                route_source_after=absent,
                route_target_after=absent,
                requested_amount=requested_amount,
                actual_amount=actual_amount,
                pre_hp=pre_hp,
                post_hp=post_hp,
                pre_builtin_shield=pre_builtin_shield,
                post_builtin_shield=post_builtin_shield,
                pre_buff_shield=pre_buff_shield,
                post_buff_shield=post_buff_shield,
                destination_before=None,
                destination_after=None,
            )
        )
    combat_envelope = RichCombatEventEnvelope(
        generation=generation,
        state_epoch=state_epoch,
        observation_tick=tick,
        capacity=_COMBAT_CAPACITY,
        hook_set_attested=True,
        hook_set_installed=True,
        capability_status="derived",
        epoch_first_sequence=combat_epoch_first,
        oldest_retained_sequence=combat_oldest,
        next_sequence=combat_next,
        overflow_count=combat_overflow,
        sequence_gap_before_oldest=(combat_oldest > combat_epoch_first),
        rejected_capture_count=combat_rejected,
        complete=True,
        events=tuple(combat_events),
    )

    (
        phase_flags,
        epoch_first_sequence,
        oldest_retained_sequence,
        next_sequence,
        overflow_count,
        rejected_count,
        phase_event_count,
    ) = reader.unpack(_PHASE_ENVELOPE, "phase_envelope")
    if (
        phase_flags & ~0x7
        or not phase_flags & 0x1
        or not phase_flags & 0x2
        or not phase_flags & 0x4
        or phase_event_count != next_sequence - oldest_retained_sequence
    ):
        raise TrainingCaptureError("compact phase envelope is invalid")
    phase_events: list[PhaseHookEvent] = []
    for event_index in range(phase_event_count):
        (
            sequence,
            event_tick,
            kind_raw,
            event_flags,
            reserved,
            hook_offset,
            caller_offset,
            entity_owner,
            entity_native_object_id,
            target_owner,
            target_native_object_id,
            buff_global_id,
            buff_remaining_ms,
            speed_multiplier,
            hit_speed_multiplier,
            input_step,
            output_step,
            timeline_before,
            timeline_after,
            classic_charge_before,
            classic_charge_after,
        ) = reader.unpack(_PHASE_EVENT, f"phase_event[{event_index}]")
        if (
            sequence != oldest_retained_sequence + event_index
            or event_tick < 0
            or event_tick > tick
            or not 1 <= kind_raw < len(_PHASE_HOOK_KINDS)
            or event_flags & ~0xF
            or reserved != 0
            or hook_offset == 0
            or entity_native_object_id == 0
        ):
            raise TrainingCaptureError("compact phase event is invalid")
        target_before = bool(event_flags & 0x2)
        target_after = bool(event_flags & 0x4)
        target_key = (target_owner, -2, target_native_object_id) if target_before or target_after else None
        phase_events.append(
            PhaseHookEvent(
                sequence=sequence,
                tick=event_tick,
                kind=_PHASE_HOOK_KINDS[kind_raw],
                entity_key=(entity_owner, -2, entity_native_object_id),
                target_key=target_key,
                hook_offset=hook_offset,
                caller_offset=(caller_offset if event_flags & 0x8 else None),
                success=bool(event_flags & 0x1),
                buff_global_id=_optional(buff_global_id),
                buff_remaining_ms=_optional(buff_remaining_ms),
                speed_multiplier=_optional(speed_multiplier),
                hit_speed_multiplier=_optional(hit_speed_multiplier),
                input_step=_optional(input_step),
                output_step=_optional(output_step),
                timeline_before=_optional(timeline_before),
                timeline_after=_optional(timeline_after),
                classic_charge_before=_optional(classic_charge_before),
                classic_charge_after=_optional(classic_charge_after),
                target_present_before=target_before,
                target_present_after=target_after,
            )
        )
    phase_window = PhaseRuntimeWindowV1(
        capacity=4096,
        epoch_first_sequence=epoch_first_sequence,
        oldest_retained_sequence=oldest_retained_sequence,
        next_sequence=next_sequence,
        overflow_count=overflow_count,
        rejected_count=rejected_count,
        complete=True,
    )
    phase_runtime = RichPhaseRuntimeEnvelope(
        generation=generation,
        state_epoch=state_epoch,
        observation_tick=tick,
        hook_set_attested=True,
        hook_set_installed=True,
        capability_status="derived",
        window=phase_window,
        sequence_gap_before_oldest=(oldest_retained_sequence > epoch_first_sequence),
        events=tuple(phase_events),
    )

    tower_troop_runtime = None
    tower_flags, tower_rejected = reader.unpack(_TOWER_ENVELOPE, "tower_troop_envelope")
    if tower_flags != 0x7 or tower_rejected != 0:
        raise TrainingCaptureError("compact tower-troop envelope is invalid")
    tower_troop_runtime = RichTowerTroopRuntimeEnvelope(
        generation=generation,
        state_epoch=state_epoch,
        observation_tick=tick,
        hook_set_attested=True,
        hook_set_installed=True,
        rejected_count=0,
        complete=True,
    )

    if reader.offset != len(payload):
        raise TrainingCaptureError("compact capture contains trailing bytes")

    finalized = bool(flags & _FLAG_FINALIZED)
    winner = world_result_raw if finalized and world_result_raw in (0, 1) else None
    ordinary = {
        "ok": True,
        "generation": generation,
        "stateEpoch": state_epoch,
        "tick": tick,
        "steps": steps,
        "ended": bool(flags & _FLAG_ENDED),
        "finalized": finalized,
        "worldResult": world_result_raw if finalized else None,
        "worldResultRaw": world_result_raw,
        "winner": winner,
        "crownsRaw": [crowns_zero, crowns_one],
        "snapshotHandles": 0,
        "count": object_count,
        "queuedCommands": queued_commands,
        "players": ordinary_players,
        "objects": ordinary_objects,
        "returned": object_count,
        "truncated": bool(flags & _FLAG_TRUNCATED),
    }
    snapshot = RichTelemetrySnapshot(
        tick=tick,
        generation=generation,
        state_epoch=state_epoch,
        objects=tuple(resolved_objects),
        objects_by_key=resolved_by_key,
        objects_by_native_object_id=resolved_by_native_object_id,
        players=rich_players,
        combat_events=combat_envelope,
        phase_runtime=phase_runtime,
        visibility_runtime=None,
        remaining_runtime=None,
        tower_troop_runtime=tower_troop_runtime,
    )
    return TrainingAtomicCaptureV1(ordinary=ordinary, snapshot=snapshot)
