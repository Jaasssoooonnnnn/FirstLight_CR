from __future__ import annotations

from native_runner.tests.asset_helpers import user_apk_bytes

import hashlib
import struct

import pytest

from native_runner.contracts import AttackPhase, SemanticEvidenceLevel
from native_runner.phase_runtime import (
    EXACT_LIBG_SHA256,
    AttackRuntimeRaw,
    ClassicChargePhase,
    DamageRampIdentity,
    DamageRampSpec,
    DeploymentPhase,
    DeploymentRuntimeRaw,
    MovementPhase,
    MovementRuntimeRaw,
    PhaseHookEvent,
    PhaseHookKind,
    PhaseRuntimeError,
    PhaseRuntimeWindowV1,
    classic_charge_increment,
    classic_charge_phase,
    combine_effect_multipliers,
    correlate_attack_interrupts,
    resolve_attack_runtime,
    resolve_damage_ramp,
    resolve_deployment_runtime,
    resolve_movement_runtime,
    scale_effect_value,
    stage_from_durations,
)


def _event(
    sequence: int,
    tick: int,
    kind: PhaseHookKind,
    **values: object,
) -> PhaseHookEvent:
    return PhaseHookEvent(
        sequence=sequence,
        tick=tick,
        kind=kind,
        entity_key=(0, 7, 0),
        **values,
    )


def test_exact_build_and_hook_fingerprints_match_libg() -> None:
    data = user_apk_bytes("lib/arm64-v8a/libg.so")
    assert hashlib.sha256(data).hexdigest() == EXACT_LIBG_SHA256
    expected = {
        0xF23110: "420300b4fd7bbda9f65701a9f44f02a9",
        0xF5F100: "ff8305d1fd7b10a9fc6f11a9fa6712a9",
        0xF5C894: "fd7bbba9f90b00f9f85f02a9f65703a9",
        0xF5A2D4: "ffc301d1fd7b01a9fc6f02a9fa6703a9",
        0xF5B3C8: "fd7bbca9f85f01a9f65702a9f44f03a9",
        0xF5B4AC: "fd7bbca9f85f01a9f65702a9f44f03a9",
        0xF1D004: "fd7bbda9f50b00f9f44f02a9fd030091",
        0xF68808: "fd7bbca9f70b00f9f65702a9f44f03a9",
        0xF609DC: "ffc301d1fd7b01a9fc6f02a9fa6703a9",
    }
    for offset, fingerprint in expected.items():
        assert data[offset : offset + 16].hex() == fingerprint

    # Both ordinary attack starts and classic-charge-ready dispatches enter
    # through the exact action-dispatch hook.  Their attested caller return
    # addresses distinguish the two phase-event kinds.
    from native_runner.cr_native_env import PHASE_EVENT_HOOKS

    assert PHASE_EVENT_HOOKS["attack_start"] == 0xF23110
    assert PHASE_EVENT_HOOKS["classic_charge_ready"] == 0xF23110


def test_host_ring_cursor_only_fails_for_an_unconsumed_rollover_gap() -> None:
    window = PhaseRuntimeWindowV1(
        capacity=4_096,
        epoch_first_sequence=10,
        oldest_retained_sequence=20,
        next_sequence=4_116,
        overflow_count=10,
        rejected_count=0,
        complete=True,
    )
    assert window.has_unconsumed_gap(19) is True
    assert window.usable_from(19) is False
    assert window.has_unconsumed_gap(20) is False
    assert window.usable_from(20) is True
    assert window.has_unconsumed_gap(4_116) is False
    assert (
        PhaseRuntimeWindowV1.from_mapping(window.to_mapping())
        == window
    )

    with pytest.raises(PhaseRuntimeError, match="rejecting native facts"):
        PhaseRuntimeWindowV1(
            capacity=4_096,
            epoch_first_sequence=10,
            oldest_retained_sequence=10,
            next_sequence=20,
            overflow_count=0,
            rejected_count=1,
            complete=True,
        )


def test_ice_and_rage_consumers_use_native_extrema_not_addition() -> None:
    ice = combine_effect_multipliers([-30])
    rage = combine_effect_multipliers([130])
    together = combine_effect_multipliers([-30, 130])

    # f5b3c8 movement consumer.
    assert scale_effect_value(100, ice) == 70
    assert scale_effect_value(100, rage) == 130
    assert scale_effect_value(100, together) == 91

    # f5b4ac attack-timeline consumer uses the same extrema algorithm but a
    # distinct HitSpeedMultiplier field and a 50-ms baseline here.
    assert scale_effect_value(50, ice) == 35
    assert scale_effect_value(50, rage) == 65
    assert scale_effect_value(50, together) == 45
    assert scale_effect_value(50, [-100]) == 0


def test_classic_movement_charge_is_not_inferno_damage_ramp() -> None:
    assert classic_charge_phase(-1) == ClassicChargePhase.UNAVAILABLE
    assert classic_charge_phase(9_999) == ClassicChargePhase.ACCUMULATING
    assert classic_charge_phase(10_000) == ClassicChargePhase.READY
    assert classic_charge_increment(250, 250) == 1_000
    assert classic_charge_increment(-10, 250) == 0
    with pytest.raises(PhaseRuntimeError):
        classic_charge_phase(-2)


@pytest.mark.parametrize(
    "identity",
    [
        DamageRampIdentity.INFERNO_DRAGON,
        DamageRampIdentity.INFERNO_TOWER,
        DamageRampIdentity.MIGHTY_MINER,
    ],
)
def test_inferno_damage_ramp_stage_and_timer(identity: DamageRampIdentity) -> None:
    spec = DamageRampSpec(
        identity=identity,
        stage_durations_ms=(2_000, 2_000, 2_000),
        stage_damage=(29, 99, 349),
    )
    assert stage_from_durations(1_999, spec.stage_durations_ms) == 0
    assert stage_from_durations(2_000, spec.stage_durations_ms) == 1
    assert stage_from_durations(4_000, spec.stage_durations_ms) == 2

    ramp = resolve_damage_ramp(
        raw_stage=1,
        timeline_ms=2_500,
        target_locked=True,
        attack_step_native_ms=35,
        spec=spec,
    )
    assert ramp is not None
    assert ramp.identity == identity
    assert ramp.stage == 1
    assert ramp.stage_elapsed_native_ms == 500
    assert ramp.next_stage_remaining_native_ms == 1_500
    assert ramp.next_stage_remaining_wall_ms == 2_150
    assert ramp.damage == 99
    assert ramp.damage_multiplier == pytest.approx(99 / 29)
    assert ramp.locked is True


def test_attack_projection_withholds_transiently_incoherent_damage_ramp() -> None:
    projection = resolve_attack_runtime(
        AttackRuntimeRaw(
            entity_key=(0, 7, 0),
            tick=10,
            target_validated=True,
            target_entity=42,
            attack_sequence_stage=1,
            attack_timeline_ms=0,
            load_remaining_ms=0,
            hit_speed_ms=1_000,
            attack_dash_time_ms=0,
            attack_step_native_ms=None,
            hook_set_attested=True,
            damage_ramp_spec=DamageRampSpec(
                identity=DamageRampIdentity.MIGHTY_MINER,
                stage_durations_ms=(2_000, 2_000, 2_000),
                stage_damage=(17, 80, 160),
            ),
        )
    )

    assert projection.damage_ramp is None
    assert projection.state.sequence_index == 1
    assert projection.state.charge_stage is None
    assert projection.state.charge_elapsed_ms is None
    assert projection.state.damage_multiplier is None
    assert "derived charge fields were withheld" in " ".join(
        projection.state.provenance.notes
    )


def test_generic_attack_sequence_is_not_promoted_to_charge_stage() -> None:
    spec = DamageRampSpec(
        identity=DamageRampIdentity.OTHER,
        stage_durations_ms=(1_000, 1_000),
        stage_damage=(10, 20),
    )
    assert (
        resolve_damage_ramp(
            raw_stage=1,
            timeline_ms=1_500,
            target_locked=True,
            attack_step_native_ms=50,
            spec=spec,
        )
        is None
    )


def test_electrical_full_stop_interrupt_requires_exact_consumer_confirmation() -> None:
    events = (
        _event(1, 100, PhaseHookKind.ATTACK_START),
        _event(
            2,
            101,
            PhaseHookKind.EFFECT_APPLY,
            buff_global_id=9000009,
            buff_remaining_ms=500,
            hit_speed_multiplier=-100,
            speed_multiplier=-100,
        ),
        _event(
            3,
            101,
            PhaseHookKind.ATTACK_SCALE,
            input_step=50,
            output_step=0,
        ),
        _event(
            4,
            101,
            PhaseHookKind.TARGET_RESET,
            timeline_before=350,
            timeline_after=0,
            target_present_before=True,
            target_present_after=False,
        ),
    )
    interrupts = correlate_attack_interrupts(events)
    assert len(interrupts) == 1
    assert interrupts[0].effect_global_id == 9000009
    assert interrupts[0].timeline_pause_confirmed is True
    assert interrupts[0].destructive_reset_confirmed is True

    raw = AttackRuntimeRaw(
        entity_key=(0, 7, 0),
        tick=101,
        target_validated=True,
        target_entity=None,
        attack_sequence_stage=0,
        attack_timeline_ms=0,
        load_remaining_ms=0,
        hit_speed_ms=1_000,
        attack_dash_time_ms=0,
        attack_step_native_ms=0,
        hook_set_attested=True,
        events=events,
    )
    projection = resolve_attack_runtime(raw)
    assert projection.state.phase == AttackPhase.INTERRUPTED
    assert projection.state.interrupted is True
    assert projection.interrupt is not None
    assert (
        projection.state.provenance.field_evidence["phase"]
        == SemanticEvidenceLevel.NATIVE_DERIVED
    )
    assert "destructive timeline reset" in " ".join(
        projection.state.provenance.notes
    )


def test_ice_slow_does_not_become_electrical_interrupt() -> None:
    events = (
        _event(1, 10, PhaseHookKind.ATTACK_START),
        _event(
            2,
            11,
            PhaseHookKind.EFFECT_APPLY,
            buff_global_id=9_000_003,
            hit_speed_multiplier=-30,
            speed_multiplier=-30,
        ),
        _event(
            3,
            11,
            PhaseHookKind.ATTACK_SCALE,
            input_step=50,
            output_step=35,
        ),
    )
    assert correlate_attack_interrupts(events) == ()


def test_attack_windup_release_timer_and_load_gate_are_separate() -> None:
    windup = resolve_attack_runtime(
        AttackRuntimeRaw(
            entity_key=(0, 7, 0),
            tick=10,
            target_validated=True,
            target_entity=42,
            attack_sequence_stage=0,
            attack_timeline_ms=250,
            load_remaining_ms=0,
            hit_speed_ms=1_000,
            attack_dash_time_ms=100,
            attack_step_native_ms=65,
            hook_set_attested=True,
            events=(_event(1, 9, PhaseHookKind.ATTACK_START),),
        )
    )
    assert windup.state.phase == AttackPhase.WINDUP
    assert windup.timing.release_remaining_native_ms == 650
    assert windup.timing.release_remaining_wall_ms == 500
    assert windup.state.phase_remaining_ms == 500

    cooldown = resolve_attack_runtime(
        AttackRuntimeRaw(
            entity_key=(0, 7, 0),
            tick=20,
            target_validated=True,
            target_entity=42,
            attack_sequence_stage=0,
            attack_timeline_ms=0,
            load_remaining_ms=125,
            hit_speed_ms=1_000,
            attack_dash_time_ms=0,
            attack_step_native_ms=50,
            hook_set_attested=True,
        )
    )
    assert cooldown.state.phase == AttackPhase.COOLDOWN
    assert cooldown.timing.load_remaining_native_ms == 125
    assert cooldown.timing.load_remaining_wall_ms == 150
    assert cooldown.state.cooldown_remaining_ms == 150


def test_snapshot_only_ambiguous_attack_phase_fails_closed() -> None:
    projection = resolve_attack_runtime(
        AttackRuntimeRaw(
            entity_key=(0, 7, 0),
            tick=10,
            target_validated=True,
            target_entity=42,
            attack_sequence_stage=0,
            attack_timeline_ms=500,
            load_remaining_ms=0,
            hit_speed_ms=1_000,
            attack_dash_time_ms=0,
            attack_step_native_ms=50,
            hook_set_attested=True,
        )
    )
    assert projection.state.phase == AttackPhase.UNKNOWN
    assert (
        projection.state.provenance.field_evidence["phase"]
        == SemanticEvidenceLevel.UNKNOWN
    )


def test_attack_sequence_progress_and_decay_remain_distinct_runtime_state() -> None:
    projection = resolve_attack_runtime(
        AttackRuntimeRaw(
            entity_key=(0, 7, 0),
            tick=10,
            target_validated=True,
            target_entity=42,
            attack_sequence_stage=2,
            attack_timeline_ms=500,
            load_remaining_ms=0,
            hit_speed_ms=1_000,
            attack_dash_time_ms=0,
            attack_step_native_ms=50,
            hook_set_attested=True,
            attack_sequence_progress=12,
            attack_sequence_progress_limit=50,
            attack_sequence_decay_remaining_ms=3_500,
            attack_sequence_decay_duration_ms=7_000,
        )
    )

    assert projection.state.sequence_index == 2
    assert projection.state.sequence_progress == 12
    assert projection.state.sequence_progress_limit == 50
    assert projection.state.sequence_decay_remaining_ms == 3_500
    assert projection.state.sequence_decay_duration_ms == 7_000
    assert (
        projection.state.provenance.field_evidence["sequence_progress"]
        == SemanticEvidenceLevel.NATIVE_AUTHORITATIVE
    )


def test_movement_runtime_uses_delta_and_keeps_classic_charge_separate() -> None:
    state = resolve_movement_runtime(
        MovementRuntimeRaw(
            entity_key=(1, 8, 0),
            tick=30,
            component_validated=True,
            hook_set_attested=True,
            movement_delta=80,
            effective_speed=195,
            speed_input=150,
            speed_after_effects=105,
            classic_charge_progress=10_000,
            classic_charge_speed_multiplier=185,
        )
    )
    assert state.phase == MovementPhase.MOVING
    assert state.effective_speed == 195
    assert state.effect_scaled_speed == 105
    assert state.classic_charge_phase == ClassicChargePhase.READY
    assert state.classic_charge_speed_multiplier == 185
    assert state.complete is True


def test_phase_runtime_accepts_only_the_exact_native_id_entity_key_tag() -> None:
    raw = MovementRuntimeRaw(
        entity_key=(0, -2, 5_000_000),
        tick=0,
        component_validated=False,
        hook_set_attested=True,
        movement_delta=None,
        effective_speed=None,
        speed_input=None,
        speed_after_effects=None,
        classic_charge_progress=None,
        classic_charge_speed_multiplier=None,
    )
    assert raw.entity_key == (0, -2, 5_000_000)
    assert resolve_movement_runtime(raw).phase == MovementPhase.UNAVAILABLE

    with pytest.raises(PhaseRuntimeError, match="exact native-ID tag"):
        MovementRuntimeRaw(
            entity_key=(0, -1, 5_000_000),
            tick=0,
            component_validated=False,
            hook_set_attested=True,
            movement_delta=None,
            effective_speed=None,
            speed_input=None,
            speed_after_effects=None,
            classic_charge_progress=None,
            classic_charge_speed_multiplier=None,
        )


def test_deployment_gate_uses_live_remaining_and_fails_closed_on_unknown_step() -> None:
    normal = resolve_deployment_runtime(
        DeploymentRuntimeRaw(
            tick=1,
            component_validated=True,
            remaining_native_ms=975,
            previous_remaining_native_ms=1_000,
            configured_deploy_time_ms=1_000,
            uses_effect_scaled_step=False,
            observed_step_native_ms=None,
        )
    )
    assert normal.phase == DeploymentPhase.DEPLOYING
    assert normal.remaining_wall_ms == 1_000
    assert normal.observed_step_native_ms == 50
    assert normal.complete is True

    special_unknown = resolve_deployment_runtime(
        DeploymentRuntimeRaw(
            tick=1,
            component_validated=True,
            remaining_native_ms=975,
            previous_remaining_native_ms=1_000,
            configured_deploy_time_ms=1_000,
            uses_effect_scaled_step=True,
            observed_step_native_ms=None,
        )
    )
    assert special_unknown.phase == DeploymentPhase.DEPLOYING
    assert special_unknown.remaining_wall_ms is None
    assert special_unknown.complete is False

    special_raged = resolve_deployment_runtime(
        DeploymentRuntimeRaw(
            tick=1,
            component_validated=True,
            remaining_native_ms=975,
            previous_remaining_native_ms=1_000,
            configured_deploy_time_ms=1_000,
            uses_effect_scaled_step=True,
            observed_step_native_ms=65,
        )
    )
    assert special_raged.remaining_wall_ms == 750
    assert special_raged.complete is True


def test_native_stage_function_body_uses_entry_duration_at_plus_a0() -> None:
    # d99404 loops over the AttackSequence entries and subtracts entry+0xa0.
    words = [
        struct.unpack_from("<I", user_apk_bytes("lib/arm64-v8a/libg.so"), offset)[0]
        for offset in range(0xD99458, 0xD99478, 4)
    ]
    assert 0xB940A16B in words  # ldr w11, [x11,#0xa0]
