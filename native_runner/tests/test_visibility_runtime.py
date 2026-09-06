from __future__ import annotations

from native_runner.tests.asset_helpers import user_apk_bytes

import hashlib
import struct

from native_runner.visibility_runtime import (
    ATTACK_COMPONENT_TYPE,
    ATTACK_COMPONENT_VTABLE_OFFSET,
    BUFF_APPLY_OFFSET,
    BUFF_ASSET_VTABLE_OFFSET,
    BUFF_COMPONENT_TYPE,
    BUFF_COMPONENT_VTABLE_OFFSET,
    BUFF_REMOVE_OFFSET,
    EXACT_LIBG_SHA256,
    PRIMARY_INVISIBILITY_GATE_OFFSET,
    PRIMARY_SELECTOR_OFFSET,
    ROYAL_GHOST_CARD_ID,
    SECONDARY_INVISIBILITY_GATE_OFFSET,
    SECONDARY_SELECTOR_OFFSET,
    ClosureStatus,
    OrdinaryTargetEligibilityRaw,
    OrdinaryTargetGateOutcome,
    VisibilityPhase,
    VisibilityTransitionKind,
    VisibilityTransitionRaw,
    closure_status,
    resolve_ordinary_target_eligibility,
    resolve_public_by_owner_from_snapshot,
    resolve_targetable_by_owner_from_snapshot,
    resolve_visibility_transition,
)


def _transition_raw(**changes: object) -> VisibilityTransitionRaw:
    values: dict[str, object] = {
        "tick": 100,
        "subject_key": (0, 77, 0),
        "subject_owner": 0,
        "hook_offset": BUFF_APPLY_OFFSET,
        "invisible_count_before": 0,
        "invisible_count_after": 1,
        "active_entry_count": 1,
        "hook_attested": True,
        "subject_validated": True,
        "component_validated": True,
        "component_type": BUFF_COMPONENT_TYPE,
        "component_vtable_offset": BUFF_COMPONENT_VTABLE_OFFSET,
        "component_owner_matches_subject": True,
        "entry_validated": True,
        "entry_component_matches": True,
        "buff_asset_validated": True,
        "buff_asset_vtable_offset": BUFF_ASSET_VTABLE_OFFSET,
        "buff_invisible": True,
        "card_id": ROYAL_GHOST_CARD_ID,
        "buff_global_id": 9_000_001,
    }
    values.update(changes)
    return VisibilityTransitionRaw(**values)  # type: ignore[arg-type]


def _target_raw(**changes: object) -> OrdinaryTargetEligibilityRaw:
    values: dict[str, object] = {
        "tick": 200,
        "attacker_key": (0, 12, 0),
        "candidate_key": (1, 34, 0),
        "attacker_owner": 0,
        "candidate_owner": 1,
        "selector_offset": PRIMARY_SELECTOR_OFFSET,
        "invisibility_gate_offset": PRIMARY_INVISIBILITY_GATE_OFFSET,
        "option": 1,
        "gate_hook_attested": True,
        "attacker_validated": True,
        "candidate_validated": True,
        "attack_component_validated": True,
        "attack_component_type": ATTACK_COMPONENT_TYPE,
        "attack_component_vtable_offset": ATTACK_COMPONENT_VTABLE_OFFSET,
        "component_owner_matches_attacker": True,
        "candidate_filter_passed": True,
        "pre_invisibility_filters_passed": True,
        "reached_invisibility_gate": True,
        "candidate_component_lookup_validated": True,
        "candidate_buff_component_present": True,
        "candidate_buff_component_vtable_offset": BUFF_COMPONENT_VTABLE_OFFSET,
        "invisible_count": 1,
    }
    values.update(changes)
    return OrdinaryTargetEligibilityRaw(**values)  # type: ignore[arg-type]


def test_exact_build_fingerprints_and_vtables() -> None:
    data = user_apk_bytes("lib/arm64-v8a/libg.so")
    assert hashlib.sha256(data).hexdigest() == EXACT_LIBG_SHA256
    expected = {
        0xF5A2D4: "ffc301d1fd7b01a9fc6f02a9fa6703a9",
        0xF59EA8: "0b010094dc0000b49f0318eb80000054",
        0xF5A578: "ff4302d1fd7b03a9fc6f04a9fa6705a9",
        0xF59304: "9d040094683241398802003469224b29",
        0xF1C248: "fd7bbea9f30b00f9fd030091f30300aa",
        0xF1D700: "fd7bbfa9fd030091d73f0094800000b4",
        0x83E7B0: "a6761b94480b40f9290080522901200a",
        0x850E1C: "0b2d1b944001003648e19a526884a772",
        0xDE2970: "fd7bbda9f65701a9f44f02a9fd030091",
        0xDE29B4: "e00314aaf603042a342b05941f00156b",
        0xDE2C54: "68aa413988000034e00314aaa8ea0494",
        0xB25568: "695645b9087d42b93f01086b81010054",
        0xB25770: "21beffd021a03591e0630091a2028052",
        0xDAD198: "e1a9fff0213c1991e04300919b751794",
        0xDAD1EC: "ebe3009408000012e043009168660c39",
        0xF5CBE8: "fd7bbca9f85f01a9f65702a9f44f03a9",
        0xF5D16C: "ff4303d1fd7b07a9fc6f08a9fa6709a9",
        0xF5D2F8: "3cfeff97a0fd0736e00319aad840ff97",
        0xF5D314: "083440b91f010071acfcff54a8435fb8",
        0xF5D7E0: "ff0303d1fd7b06a9fc6f07a9fa6708a9",
        0xF5DA4C: "67fcff97e0fe0736880340f9e0031caa",
        0xF5DA94: "083440b91f0100718cfcff545f070071",
        0xF1CB90: "08c040f9680000b4e0030012c0035fd6",
        0xF1A46C: "682640f9084941f9281000b4681e41b9",
        0xF1930C: "01ac01b9c0035fd6ff4304d1fd7b0ba9",
    }
    for offset, fingerprint in expected.items():
        assert data[offset : offset + 16].hex() == fingerprint

    # Android RELA is at a fixed exact-build offset.  Verify the Buff asset
    # identity and the character +0x148 contextual area predicate.
    relocations: dict[int, int] = {}
    for position in range(0x73B8, 0x73B8 + 2_791_824, 24):
        relocation_offset, _info, addend = struct.unpack_from(
            "<QQq", data, position
        )
        if relocation_offset in (0x1882500, 0x189C4E8 + 0x148):
            relocations[relocation_offset] = addend
    assert relocations == {
        0x1882500: 0xD9614C,
        0x189C4E8 + 0x148: 0xF1CB90,
    }


def test_phase_edge_emits_only_zero_to_positive_and_positive_to_zero() -> None:
    became_invisible = resolve_visibility_transition(_transition_raw())
    assert became_invisible is not None
    assert became_invisible.phase is VisibilityPhase.INVISIBLE
    assert (
        became_invisible.transition
        is VisibilityTransitionKind.BECAME_INVISIBLE
    )
    assert became_invisible.is_royal_ghost is True

    became_visible = resolve_visibility_transition(
        _transition_raw(
            hook_offset=BUFF_REMOVE_OFFSET,
            invisible_count_before=1,
            invisible_count_after=0,
            active_entry_count=0,
        )
    )
    assert became_visible is not None
    assert became_visible.phase is VisibilityPhase.VISIBLE
    assert became_visible.transition is VisibilityTransitionKind.BECAME_VISIBLE

    assert (
        resolve_visibility_transition(
            _transition_raw(
                invisible_count_before=1,
                invisible_count_after=2,
                active_entry_count=2,
            )
        )
        is None
    )
    assert (
        resolve_visibility_transition(
            _transition_raw(
                hook_offset=BUFF_REMOVE_OFFSET,
                invisible_count_before=2,
                invisible_count_after=1,
                active_entry_count=1,
            )
        )
        is None
    )


def test_visibility_transition_rejects_wrong_identity_or_count_delta() -> None:
    assert (
        resolve_visibility_transition(
            _transition_raw(component_owner_matches_subject=False)
        )
        is None
    )
    assert (
        resolve_visibility_transition(
            _transition_raw(invisible_count_after=2, active_entry_count=2)
        )
        is None
    )
    assert (
        resolve_visibility_transition(
            _transition_raw(
                buff_invisible=False,
                invisible_count_after=0,
            )
        )
        is None
    )


def test_contextual_ordinary_gate_rejects_invisible_candidate() -> None:
    event = resolve_ordinary_target_eligibility(_target_raw())
    assert event is not None
    assert event.outcome is OrdinaryTargetGateOutcome.REJECTED_INVISIBLE
    assert event.attacker_owner == 0
    assert event.candidate_owner == 1
    assert event.scope == "ordinary_attack_acquisition.invisibility_gate"
    assert event.gate_decision_complete is True
    assert event.final_selection_complete is False
    assert event.targetable_by_owner is None


def test_contextual_gate_can_pass_without_claiming_final_targetability() -> None:
    absent_component = resolve_ordinary_target_eligibility(
        _target_raw(
            selector_offset=SECONDARY_SELECTOR_OFFSET,
            invisibility_gate_offset=SECONDARY_INVISIBILITY_GATE_OFFSET,
            candidate_buff_component_present=False,
            candidate_buff_component_vtable_offset=None,
            invisible_count=None,
        )
    )
    assert absent_component is not None
    assert (
        absent_component.outcome
        is OrdinaryTargetGateOutcome.PASSED_INVISIBILITY_GATE
    )
    assert absent_component.invisible_count == 0
    assert absent_component.final_selection_complete is False

    assert (
        resolve_ordinary_target_eligibility(
            _target_raw(
                selector_offset=PRIMARY_SELECTOR_OFFSET,
                invisibility_gate_offset=SECONDARY_INVISIBILITY_GATE_OFFSET,
            )
        )
        is None
    )
    assert (
        resolve_ordinary_target_eligibility(
            _target_raw(pre_invisibility_filters_passed=False)
        )
        is None
    )


def test_owner_public_and_snapshot_targetable_remain_unavailable() -> None:
    assert (
        resolve_public_by_owner_from_snapshot(
            observer_owner=0,
            subject_owner=1,
            invisible_count=0,
        )
        is None
    )
    assert (
        resolve_public_by_owner_from_snapshot(
            observer_owner=1,
            subject_owner=1,
            invisible_count=1,
        )
        is None
    )
    assert (
        resolve_targetable_by_owner_from_snapshot(
            observer_owner=0,
            subject_owner=1,
            invisible_count=1,
        )
        is None
    )
    status = closure_status()
    assert (
        status["entity.invisibility_phase_transition"]
        is ClosureStatus.EXACT_PATH_READY
    )
    assert (
        status["target_eligibility.ordinary_attack_invisibility_gate"]
        is ClosureStatus.CONTEXTUAL_ONLY
    )
    assert (
        status["entity.public_by_owner"]
        is ClosureStatus.BLOCKED_NATIVE_PRODUCER
    )
    assert (
        status["entity.targetable_by_owner"]
        is ClosureStatus.BLOCKED_NATIVE_PRODUCER
    )
