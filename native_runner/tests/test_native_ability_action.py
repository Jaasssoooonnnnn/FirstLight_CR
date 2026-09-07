from __future__ import annotations

from native_runner.paths import WORKSPACE_ROOT, PACKAGE_ROOT

from typing import Any

import pytest

from native_runner.cr_native_env import (
    AbilityAction,
    NativeClashEnv,
)
from native_runner.match_factory import MatchConfig
from native_runner.tests.cpp_source import cpp_function


WORKSPACE = WORKSPACE_ROOT


class _StubNativeClashEnv(NativeClashEnv):
    def __init__(self) -> None:
        super().__init__()
        self.commands: list[str] = []

    def _request(self, command: str) -> dict[str, Any]:
        self.commands.append(command)
        return {
            "ok": True,
            "generation": 4,
            "sequence": 9,
            "tick": 420,
            "registeredAtTick": 420,
        }


def test_host_ability_api_passes_only_stable_entity_key() -> None:
    env = _StubNativeClashEnv()

    result = env.queue_ability_action_at(AbilityAction(0, 17, 3))

    assert env.commands == ["activate-ability 0 17 3 1"]
    assert result["sourceEntityKey"] == [0, 17, 3]
    assert result["queuedAtTick"] == 420
    assert result["executeTick"] == 421
    assert "cgid" not in env.commands[0]


def test_host_ability_api_accepts_tagged_native_object_identity() -> None:
    env = _StubNativeClashEnv()

    result = env.queue_ability_action_at(AbilityAction(0, -2, 5_000_006))

    assert env.commands == ["activate-ability 0 -2 5000006 1"]
    assert result["sourceEntityKey"] == [0, -2, 5_000_006]


@pytest.mark.parametrize(
    ("action", "ticks", "message"),
    (
        (AbilityAction(2, 17, 3), 1, "owner"),
        (AbilityAction(0, -1, 3), 1, "identity"),
        (AbilityAction(0, 17, -1), 1, "identity"),
        (AbilityAction(0, 17, 3), 0, "1..65535"),
        (AbilityAction(0, 17, 3), 65_536, "1..65535"),
    ),
)
def test_host_ability_api_fails_closed(
    action: AbilityAction,
    ticks: int,
    message: str,
) -> None:
    env = _StubNativeClashEnv()

    with pytest.raises(ValueError, match=message):
        env.queue_ability_action_at(action, execute_in_ticks=ticks)

    assert env.commands == []


def test_host_ability_api_schedules_a_future_tick() -> None:
    env = _StubNativeClashEnv()

    result = env.queue_ability_action_at(
        AbilityAction(0, 17, 3),
        execute_in_ticks=7,
    )

    assert env.commands == ["activate-ability 0 17 3 7"]
    assert result["queuedAtTick"] == 420
    assert result["executeTick"] == 427
















def test_host_replay_ability_registration_defers_identity_resolution() -> None:
    env = _StubNativeClashEnv()

    result = env.schedule_replay_ability_at_tick(
        owner=1,
        ability_name_hints=("Wizard Hero", "wizard-hero"),
        execute_tick=512,
    )

    assert env.commands == ["replay-schedule-ability 1 wizardhero 512"]
    assert result["sequence"] == 9
    assert result["registeredAtTick"] == 420
    assert result["abilityNameHints"] == ("wizardhero",)


def test_host_replay_schedule_status_keeps_boundary_failure_explicit() -> None:
    class _StatusNative(_StubNativeClashEnv):
        def _request(self, command: str) -> dict[str, Any]:
            self.commands.append(command)
            return {
                "ok": True,
                "sequence": 9,
                "state": "failed",
                "error": "ability-unavailable",
                "queuedAtTick": 511,
                "executeTick": 512,
            }

    env = _StatusNative()

    result = env.replay_schedule_status(9)

    assert env.commands == ["replay-schedule-status 9"]
    assert result["state"] == "failed"
    assert result["error"] == "ability-unavailable"


def test_host_replay_schedule_status_keeps_true_ambiguity_explicit() -> None:
    class _StatusNative(_StubNativeClashEnv):
        def _request(self, command: str) -> dict[str, Any]:
            self.commands.append(command)
            return {
                "ok": True,
                "sequence": 9,
                "state": "failed",
                "error": "ability-ambiguous",
                "queuedAtTick": 511,
                "executeTick": 512,
            }

    env = _StatusNative()

    result = env.replay_schedule_status(9)

    assert env.commands == ["replay-schedule-status 9"]
    assert result["state"] == "failed"
    assert result["error"] == "ability-ambiguous"




def test_probe_resolves_private_cgid_from_one_ready_live_champion() -> None:
    source = (PACKAGE_ROOT / "probe" / "cr_replay_probe.cpp").read_text(
        encoding="utf-8"
    )

    assert "resolve_ready_ability_request" in source
    assert "candidate.owner != owner" in source
    assert "ability_button_state_is_queueable" in source
    assert "return value == 2 || value == 4" in source
    assert "ability.remaining_cooldown_ms != 0" in source
    assert "ability.remaining_charges_raw == 0" in source
    assert "matches == 0" in source
    assert "matches > 1" in source
    assert "AbilityResolutionStatus::Ambiguous" in source
    assert 'return "ability-ambiguous"' in source
    assert "could not resolve one ready native champion ability source" in source
    assert "read_object_field<std::int32_t>(champion, 0x08)" in source
    assert '"{\\"ct\\":2,\\"c\\":{\\"t\\":%d,\\"t2\\":%d' in source
    assert "command_tick + 1" in source
    assert '"activate-ability "' in source
    assert '"activate-ability-at "' not in source
    assert '"activate-unique-ability-at "' not in source
    ability_handler = source.split(
        'if (std::strncmp(command, "activate-ability ", 17) == 0)',
        maxsplit=1,
    )[1].split(
        'if (std::strncmp(command, "inject ", 7) == 0)',
        maxsplit=1,
    )[0]
    assert "command_tick + delay - 2" in ability_handler
    assert "? target_execute_tick - 1" not in ability_handler
    assert "kUniqueReadyAbilityEntityKeyTag" in source






def test_probe_ready_ability_resolution_skips_only_proven_absent_peer() -> None:
    source = (PACKAGE_ROOT / "probe" / "cr_replay_probe.cpp").read_text(
        encoding="utf-8"
    )
    absent = source.split(
        "bool champion_controller_is_exact_absent(",
        maxsplit=1,
    )[1].split("bool read_evolution_slot(", maxsplit=1)[0]
    resolver = source.split(
        "bool resolve_ready_ability_request(",
        maxsplit=1,
    )[1].split("void cancel_pending_live_work_locked()", maxsplit=1)[0]
    invalid_controller = resolver.split(
        "if (!read_champion_controller(",
        maxsplit=1,
    )[1].split("if (select_unique_ready", maxsplit=1)[0]

    assert "read_native_player_runtime" not in resolver
    assert "champion_controller_is_exact_empty(owner_root, controller)" in resolver
    assert "champion_controller_is_exact_absent(owner_root, controller)" in resolver
    assert invalid_controller.index("champion_controller_is_exact_absent") < (
        invalid_controller.index("return false;")
    )
    assert "read_object_field<std::int32_t>(controller, 0x98) != 1" in absent
    assert "remaining_charges != -1" in absent
    assert "remaining_charges < 0 || remaining_charges > max_charges" in absent
    assert "max_charges <= 0" in absent
    assert "champions.count == 0" in absent


def test_probe_semantic_ability_prefers_latest_queueable_carrier() -> None:
    source = (PACKAGE_ROOT / "probe" / "cr_replay_probe.cpp").read_text(
        encoding="utf-8"
    )
    resolver = source.split(
        "bool resolve_ready_ability_request(",
        maxsplit=1,
    )[1].split("void cancel_pending_live_work_locked()", maxsplit=1)[0]

    queueability_check = resolver.index(
        "if (!ability_button_state_is_queueable(ability.button_state)"
    )
    newest_selection = resolver.index(
        "candidate.native_object_id > selected_native_object_id"
    )

    assert queueability_check < newest_selection
    assert "ability.remaining_cooldown_ms != 0" in resolver
    assert "ability.remaining_charges_raw == 0" in resolver
    assert "selected_native_object_id = candidate.native_object_id" in resolver
    assert "!select_unique_ready && matches > 1" in resolver
    assert "select_unique_ready && latest_candidate_ambiguous" in resolver


def test_probe_resolves_semantic_replay_actions_in_each_logic_step() -> None:
    source = (PACKAGE_ROOT / "probe" / "cr_replay_probe.cpp").read_text(
        encoding="utf-8"
    )
    step_hook = cpp_function(source, "game_state_step_hook")
    scheduler = cpp_function(source, "process_scheduled_replay_actions")

    assert step_hook.index("process_scheduled_replay_actions(manager)") < step_hook.index(
        "g_original_game_state_step(manager)"
    )
    assert "scheduled.target_execute_tick - 1" in scheduler
    assert "resolve_card_deploy_request" in scheduler
    assert "resolve_scheduled_replay_ability" in scheduler
    assert "g_native_render_paused" not in scheduler
    assert "g_native_render_speed_quarters" not in scheduler


def test_resident_ability_receipt_carries_the_queue_tick() -> None:
    source = (
        PACKAGE_ROOT / "probe" / "resident_multimatch.inc"
    ).read_text(encoding="utf-8")

    ability_handler = source.split(
        'if (std::strncmp(command, "activate-ability ", 17) == 0)',
        maxsplit=1,
    )[1]
    assert '"\\"tick\\":%d,\\"executeTick\\":%d,\\"injected\\":%s}"' in ability_handler
    assert "g_last_controlled_tick" in ability_handler
    assert "g_last_controlled_tick + delay - 2" in ability_handler
    assert "receipt.execute_tick - 2" in ability_handler
    assert "g_last_controlled_tick + delay - 1" not in ability_handler
    assert "action.execute_offset > ticks[index]" not in ability_handler


def test_resident_card_injection_receipt_carries_the_queue_tick() -> None:
    source = (
        PACKAGE_ROOT / "probe" / "resident_multimatch.inc"
    ).read_text(encoding="utf-8")

    injection_handler = source.split(
        'if (std::strncmp(command, "inject ", 7) == 0)',
        maxsplit=1,
    )[1].split(
        'if (std::strncmp(command, "play ", 5) == 0)',
        maxsplit=1,
    )[0]
    assert '"\\\"tick\\\":%d,\\\"injected\\\":true}"' in injection_handler
    assert "slot_id" in injection_handler
    assert "g_last_controlled_tick" in injection_handler


def test_match_config_emits_exact_form_availability_bits_only_when_set() -> None:
    default_payload = MatchConfig().to_replay_dict()
    assert all(set(item) == {"d"} for item in default_payload["battle"]["deck0"]["sp"])

    configured = MatchConfig(
        deck0_form_availability=(1, 2, 3, 0, 0, 0, 0, 0),
        deck1_form_availability=(0, 0, 0, 0, 0, 0, 0, 1),
    ).to_replay_dict()
    assert configured["battle"]["deck0"]["sp"][:4] == [
        {"d": MatchConfig().deck0[0], "el": 1},
        {"d": MatchConfig().deck0[1], "el": 2},
        {"d": MatchConfig().deck0[2], "el": 3},
        {"d": MatchConfig().deck0[3]},
    ]
    assert configured["battle"]["deck1"]["sp"][7]["el"] == 1


def test_match_config_emits_king_tower_level_for_both_native_owner_paths() -> None:
    configured = MatchConfig(king_tower_level=11).to_replay_dict()

    assert [configured["battle"][f"avatar{owner}"]["expLevel"] for owner in (0, 1)] == [
        11,
        11,
    ]
    assert [configured["battle"]["hbd"][owner]["kt"] for owner in (0, 1)] == [
        11,
        11,
    ]


def test_match_config_emits_independent_tower_troop_support_cards() -> None:
    configured = MatchConfig(
        tower_troop0_id=159_000_001,
        tower_troop1_id=159_000_003,
    ).to_replay_dict()

    assert configured["battle"]["deck0"]["sc"][0]["d"] == 159_000_001
    assert configured["battle"]["deck1"]["sc"][0]["d"] == 159_000_003


@pytest.mark.parametrize("tower_troop_id", (True, 0, 158_999_999, 160_000_000))
def test_match_config_rejects_non_support_card_tower_ids(
    tower_troop_id: object,
) -> None:
    with pytest.raises(ValueError, match="tower_troop0_id"):
        MatchConfig(tower_troop0_id=tower_troop_id)  # type: ignore[arg-type]


@pytest.mark.parametrize("level", (True, 0, 11.5, 17))
def test_match_config_rejects_invalid_king_tower_level(level: object) -> None:
    with pytest.raises(ValueError, match="king_tower_level"):
        MatchConfig(king_tower_level=level)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "masks",
    (
        (0,) * 7,
        (0,) * 7 + (4,),
        (0,) * 7 + (True,),
    ),
)
def test_match_config_rejects_invalid_form_availability(masks: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="form_availability"):
        MatchConfig(deck0_form_availability=masks)
