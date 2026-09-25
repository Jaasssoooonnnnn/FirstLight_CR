from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from native_runner import royaleapi_replay
from native_runner import user_interface as interface


def _quiet_gpu() -> interface.GpuStatus:
    return interface.GpuStatus(
        available=True,
        utilization_percent=20.0,
        utilization_samples=(18.0, 22.0),
        memory_used_mib=2_000.0,
        memory_total_mib=8_000.0,
        memory_used_percent=25.0,
        detail="ok",
    )


class _FakeVar:
    def __init__(self, value: str = "") -> None:
        self.value = value

    def get(self) -> str:
        return self.value

    def set(self, value: str) -> None:
        self.value = value


class _EmptyTree:
    def get_children(self) -> tuple[object, ...]:
        return ()

    def delete(self, *_items: object) -> None:
        return None


def test_manual_match_passes_selected_tower_troops_to_native_replay(monkeypatch) -> None:
    seeds = iter((12345, 67890))
    monkeypatch.setattr(interface, "_new_match_seed", lambda: next(seeds))
    app = object.__new__(interface.CRHarnessInterface)
    preset = interface.MATCH_PRESETS[0]
    app.deck0_editor = SimpleNamespace(values=lambda: (preset.deck0, preset.forms0), tower_troop_id=lambda: 159_000_001)
    app.deck1_editor = SimpleNamespace(values=lambda: (preset.deck1, preset.forms1), tower_troop_id=lambda: 159_000_002)

    config = app._match_config()

    assert config.deck0 == preset.deck0
    assert config.deck1 == preset.deck1
    assert config.seed == 12345
    assert app._match_config().seed == 67890
    assert (config.level_cap, config.minimum_card_level, config.king_tower_level) == (11, 11, 11)
    assert (config.owner0_name, config.owner1_name) == ("PEKKA-11-A", "PEKKA-11-B")
    assert (config.tower_troop0_id, config.tower_troop1_id) == (159_000_001, 159_000_002)
    replay = config.to_replay_dict()
    assert replay["battle"]["deck0"]["sc"][0]["d"] == 159_000_001
    assert replay["battle"]["deck1"]["sc"][0]["d"] == 159_000_002


def test_model_deck_role_check_blocks_three_evolutions_before_start(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    app = object.__new__(interface.CRHarnessInterface)
    app.busy = False
    app.model_checkpoint_var = _FakeVar(str(checkpoint))
    app.model_deck0_editor = SimpleNamespace(values=lambda: (
        (26000021, 26000030, 26000014, 26000038, 28000011, 27000000, 28000000, 26000010),
        (0, 1, 2, 0, 0, 1, 0, 1),
    ))
    app.model_deck1_editor = SimpleNamespace(values=lambda: (
        interface.PEKKA_BRIDGE_SPAM_DECK, interface.PEKKA_BRIDGE_SPAM_FORMS,
    ))
    errors: list[BaseException] = []
    app._show_error = errors.append
    app._confirm_replace_owned_session = lambda _kind: pytest.fail("session changed before role validation")

    app.start_model_match()

    assert len(errors) == 1
    assert "3 张觉醒、1 张英雄" in str(errors[0])
    assert "改为“基础”" in str(errors[0])
    assert interface.validate_model_deck_roles((0, 0, 2, 0, 0, 1, 0, 1)) is None


def test_model_match_launches_deterministic_argmax(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(interface, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(interface, "_new_match_seed", lambda: 23456)
    app = object.__new__(interface.CRHarnessInterface)
    app.busy = False
    app.owns_native_session = True
    app.model_checkpoint_var = _FakeVar(str(checkpoint))
    app.model_status_var = _FakeVar()
    app.model_deck0_editor = SimpleNamespace(values=lambda: (
        interface.PEKKA_BRIDGE_SPAM_DECK, interface.PEKKA_BRIDGE_SPAM_FORMS,
    ), tower_troop_id=lambda: 159_000_004)
    app.model_deck1_editor = SimpleNamespace(values=app.model_deck0_editor.values, tower_troop_id=lambda: 159_000_001)
    app._confirm_replace_owned_session = lambda _kind: True
    app._check_operation_conflicts = lambda: True
    app.stop_replay = lambda: None
    app._terminate_overlay = lambda: None
    app._set_model_active = lambda _active: None
    captured: dict[str, object] = {}

    def launch(module, _log_path, **options):
        captured.update(module=module, **options)
        return SimpleNamespace(poll=lambda: 0, returncode=0), None

    app._launch_python = launch
    app._background = lambda _name, run, _success, _failed: run()

    app.start_model_match()

    assert captured["module"] == "native_runner.training.v4.offline_agent"
    assert "deterministic" in captured and captured["deterministic"] is None
    assert app.model_status_var.get() == "正在准备离线 VM 和模型…"
    config = json.loads((app.model_artifact_dir / "match.json").read_text(encoding="utf-8"))
    assert config["seed"] == 23456
    assert (config["tower_troop0_id"], config["tower_troop1_id"]) == (159_000_004, 159_000_001)


def test_ai_duel_launches_both_checkpoints_and_decks(monkeypatch, tmp_path: Path) -> None:
    checkpoints = (tmp_path / "top.pt", tmp_path / "bottom.pt")
    for checkpoint in checkpoints:
        checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(interface, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(interface, "_new_match_seed", lambda: 34567)
    app = object.__new__(interface.CRHarnessInterface)
    app.busy = False
    app.owns_native_session = True
    app.duel_checkpoint0_var = _FakeVar(str(checkpoints[0]))
    app.duel_checkpoint1_var = _FakeVar(str(checkpoints[1]))
    app.duel_deck0_editor = SimpleNamespace(values=lambda: (
        interface.PEKKA_BRIDGE_SPAM_DECK, interface.PEKKA_BRIDGE_SPAM_FORMS,
    ), tower_troop_id=lambda: 159_000_002)
    app.duel_deck1_editor = SimpleNamespace(values=app.duel_deck0_editor.values, tower_troop_id=lambda: 159_000_004)
    app.duel_status_var = _FakeVar()
    app._confirm_replace_owned_session = lambda _kind: True
    app._check_operation_conflicts = lambda: True
    app.stop_replay = lambda: None
    app._terminate_overlay = lambda: None
    app._set_model_active = lambda _active: None
    captured: dict[str, object] = {}

    def launch(module, _log_path, **options):
        captured.update(module=module, **options)
        return SimpleNamespace(poll=lambda: 0, returncode=0), None

    app._launch_python = launch
    app._background = lambda _name, run, _success, _failed: run()

    app.start_ai_duel()

    assert captured["module"] == "native_runner.training.v4.offline_duel"
    assert (captured["checkpoint_0"], captured["checkpoint_1"]) == checkpoints
    config = json.loads((app.model_artifact_dir / "match.json").read_text(encoding="utf-8"))
    assert config["seed"] == 34567
    assert config["deck0"] == list(interface.PEKKA_BRIDGE_SPAM_DECK)
    assert config["deck1"] == list(interface.PEKKA_BRIDGE_SPAM_DECK)
    assert (config["tower_troop0_id"], config["tower_troop1_id"]) == (159_000_002, 159_000_004)
    assert app.model_match_kind == "duel"


def test_force_ai_play_sends_request_to_chosen_side(tmp_path: Path) -> None:
    app = object.__new__(interface.CRHarnessInterface)
    app.model_match_kind = "duel"
    app.model_active = True
    app.model_artifact_dir = tmp_path
    app.model_process = SimpleNamespace(poll=lambda: None)

    app.force_ai_play(1)

    assert (tmp_path / "force-1").exists()
    assert not (tmp_path / "force-0").exists()


def test_custom_collected_replay_directory_is_exclusive(
    monkeypatch,
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "merged"
    dataset_root.mkdir()
    app = object.__new__(interface.CRHarnessInterface)
    app.collected_dataset_var = _FakeVar(str(dataset_root))
    app.collected_limit_var = _FakeVar("500")
    app.collected_query_var = _FakeVar("")
    app.collected_replay_status_var = _FakeVar()
    app.collected_replay_tree = _EmptyTree()
    app.collected_replay_entries = {}
    captured: dict[str, object] = {}

    def fake_list_collected_replays(**kwargs: object) -> tuple[object, ...]:
        captured.update(kwargs)
        return ()

    monkeypatch.setattr(
        royaleapi_replay,
        "list_collected_replays",
        fake_list_collected_replays,
    )

    app.refresh_collected_replays()

    assert captured["dataset_root"] == dataset_root.resolve()
    assert captured["limit"] == 500
    assert captured["query"] == ""
    assert captured["personal_dataset_roots"] == ()
    assert "显示 0 条采集回放" in app.collected_replay_status_var.get()


def test_resource_warnings_include_peak_gpu_vram_memory_and_training() -> None:
    warnings = interface.resource_warnings(
        interface.MemoryStatus(
            used_percent=91.0,
            available_gib=2.0,
            total_gib=32.0,
        ),
        interface.GpuStatus(
            available=True,
            utilization_percent=69.0,
            utilization_samples=(58.0, 80.0),
            memory_used_mib=7_400.0,
            memory_total_mib=8_000.0,
            memory_used_percent=92.5,
            detail="ok",
        ),
        (
            interface.ProcessStatus(
                pid=1234,
                command_line=(
                    "python -m "
                    "native_runner.training.future_training_entrypoint"
                ),
            ),
        ),
    )

    assert len(warnings) == 4
    assert "内存" in warnings[0]
    assert "80%" in warnings[1]
    assert "显存" in warnings[2]
    assert "PID 1234" in warnings[3]


def test_resource_warnings_stay_empty_below_thresholds() -> None:
    assert interface.resource_warnings(
        interface.MemoryStatus(
            used_percent=40.0,
            available_gib=12.0,
            total_gib=32.0,
        ),
        _quiet_gpu(),
        (),
    ) == ()


def test_runtime_conflicts_fail_closed_without_mutating_processes(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        interface,
        "query_training_processes",
        lambda: (
            interface.ProcessStatus(
                pid=55,
                command_line=(
                    "python -m "
                    "native_runner.training.future_training_entrypoint"
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        interface,
        "query_established_native_clients",
        lambda: (),
    )
    monkeypatch.setattr(
        interface,
        "_probe_native",
        lambda command: (
            {"occupied": 8}
            if command == "multi-status"
            else {"mode": "native-render"}
        ),
    )

    conflicts = interface.runtime_conflicts(
        allow_owned_native_render=False,
    )

    assert not any("PID 55" in item for item in conflicts)
    assert any("8 个 resident" in item for item in conflicts)
    assert any("不属于本界面" in item for item in conflicts)


def test_replay_runs_include_current_runs_replay_directory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "native_runner"
    workspace_root = tmp_path
    replay_directory = (
        workspace_root / "runs" / "pekka_bridge_spam_v1" / "replays"
    )
    replay_directory.mkdir(parents=True)
    (
        replay_directory
        / "match-000000000512-seed-1-episode-test.crr.json.gz"
    ).touch()
    package_root.mkdir(exist_ok=True)
    monkeypatch.setattr(interface, "PACKAGE_ROOT", package_root)
    monkeypatch.setattr(interface, "WORKSPACE_ROOT", workspace_root)

    runs = interface.discover_replay_runs()

    assert runs == (
        interface.ReplayRun(
            label="正式训练 · pekka_bridge_spam_v1",
            directory=replay_directory.resolve(),
        ),
    )


def test_replay_runs_include_nested_formal_run_directory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "native_runner"
    workspace_root = tmp_path
    replay_directory = (
        workspace_root
        / "runs"
        / "pekka_bridge_spam_reward_v3"
        / "formal-20260802-005535"
        / "replays"
    )
    replay_directory.mkdir(parents=True)
    (
        replay_directory
        / "match-000000000512-seed-1-episode-test.crr.json.gz"
    ).touch()
    package_root.mkdir(exist_ok=True)
    monkeypatch.setattr(interface, "PACKAGE_ROOT", package_root)
    monkeypatch.setattr(interface, "WORKSPACE_ROOT", workspace_root)

    runs = interface.discover_replay_runs()

    assert runs == (
        interface.ReplayRun(
            label=(
                "正式训练 · pekka_bridge_spam_reward_v3 / "
                "formal-20260802-005535"
            ),
            directory=replay_directory.resolve(),
        ),
    )


def test_offline_launcher_receives_exact_interactive_worker(
    monkeypatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(interface, "_endpoint_is_cold_ready", lambda: False)

    def run_hidden(arguments, *, timeout, cwd=interface.WORKSPACE_ROOT):
        del timeout, cwd
        calls.append(tuple(str(item) for item in arguments))
        return subprocess.CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(interface, "_run_hidden", run_hidden)

    interface.ensure_offline_runner(lambda _message: None)

    command = calls[0]
    assert command[command.index("-VmIndex") + 1] == str(interface.VM_INDEX)
    assert command[command.index("-Serial") + 1] == interface.ADB_SERIAL
    assert command[command.index("-ControlPort") + 1] == str(interface.CONTROL_PORT)


def test_force_restart_does_not_reuse_a_cold_ready_endpoint(
    monkeypatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(interface, "_endpoint_is_cold_ready", lambda: True)

    def run_hidden(arguments, *, timeout, cwd=interface.WORKSPACE_ROOT):
        del timeout, cwd
        calls.append(tuple(str(item) for item in arguments))
        return subprocess.CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(interface, "_run_hidden", run_hidden)

    interface.ensure_offline_runner(
        lambda _message: None,
        force_restart=True,
    )

    assert len(calls) == 1
    assert "start_offline.ps1" in calls[0][calls[0].index("-File") + 1]


def _configuration_timeout(
    *,
    sequence: int,
    processed: int,
    loaded: int,
) -> royaleapi_replay.NativeRenderConfigurationLoadError:
    return royaleapi_replay.NativeRenderConfigurationLoadError(
        "02LY92UU8GJQ",
        sequence,
        status={
            "mode": "native-render",
            "nativeRenderReady": False,
            "nativeRenderSequence": sequence,
            "nativeRenderProcessed": processed,
            "nativeRenderLoaded": loaded,
        },
    )


def test_replay_calibration_restarts_dedicated_runner_exactly_once(
    monkeypatch,
) -> None:
    prepared = SimpleNamespace(replay_tag="02LY92UU8GJQ")
    natives = iter((object(), object()))
    made: list[object] = []
    restarts: list[str] = []
    progress: list[str] = []
    calls = 0

    def make_native() -> object:
        native = next(natives)
        made.append(native)
        return native

    def calibrate(value, native, *, on_attempt):
        nonlocal calls
        assert value is prepared
        assert native is made[-1]
        on_attempt("calibration-attempt")
        calls += 1
        if calls == 1:
            raise _configuration_timeout(sequence=6, processed=1, loaded=0)
        return "calibrated"

    monkeypatch.setattr(
        royaleapi_replay,
        "calibrate_collected_replay_deal",
        calibrate,
    )

    result, native = interface._calibrate_collected_replay_with_one_restart(
        prepared,
        make_native=make_native,
        restart_runner=lambda: restarts.append("restart"),
        progress=progress.append,
    )

    assert result == "calibrated"
    assert native is made[1]
    assert calls == 2
    assert restarts == ["restart"]
    assert progress.count("calibration-attempt") == 2
    assert any("sequence=6/processed=1/loaded=0" in item for item in progress)


def test_replay_calibration_stops_after_second_load_timeout(
    monkeypatch,
) -> None:
    prepared = SimpleNamespace(replay_tag="02LY92UU8GJQ")
    made: list[object] = []
    restarts: list[str] = []
    errors = iter(
        (
            _configuration_timeout(sequence=6, processed=1, loaded=0),
            _configuration_timeout(sequence=1, processed=1, loaded=0),
        )
    )

    def make_native() -> object:
        native = object()
        made.append(native)
        return native

    def calibrate(_value, _native, *, on_attempt):
        del on_attempt
        raise next(errors)

    monkeypatch.setattr(
        royaleapi_replay,
        "calibrate_collected_replay_deal",
        calibrate,
    )

    with pytest.raises(royaleapi_replay.RoyaleAPIReplayError) as raised:
        interface._calibrate_collected_replay_with_one_restart(
            prepared,
            make_native=make_native,
            restart_runner=lambda: restarts.append("restart"),
            progress=lambda _message: None,
        )

    assert len(made) == 2
    assert restarts == ["restart"]
    message = str(raised.value)
    assert "after exactly one dedicated runner restart" in message
    assert "sequence=6/processed=1/loaded=0" in message
    assert "sequence=1/processed=1/loaded=0" in message


def test_replay_calibration_does_not_restart_for_a_deal_error(
    monkeypatch,
) -> None:
    prepared = SimpleNamespace(replay_tag="02LY92UU8GJQ")
    made: list[object] = []
    restarts: list[str] = []

    def make_native() -> object:
        native = object()
        made.append(native)
        return native

    def calibrate(_value, _native, *, on_attempt):
        del on_attempt
        raise royaleapi_replay.RoyaleAPIReplayError(
            "observed native 4+4 is genuinely incompatible"
        )

    monkeypatch.setattr(
        royaleapi_replay,
        "calibrate_collected_replay_deal",
        calibrate,
    )

    with pytest.raises(
        royaleapi_replay.RoyaleAPIReplayError,
        match="genuinely incompatible",
    ):
        interface._calibrate_collected_replay_with_one_restart(
            prepared,
            make_native=make_native,
            restart_runner=lambda: restarts.append("restart"),
            progress=lambda _message: None,
        )

    assert len(made) == 1
    assert restarts == []


def test_owned_native_render_is_not_reported_as_foreign(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        interface,
        "query_training_processes",
        lambda: (),
    )
    monkeypatch.setattr(
        interface,
        "query_established_native_clients",
        lambda: (),
    )
    monkeypatch.setattr(
        interface,
        "_probe_native",
        lambda command: (
            {"occupied": 0}
            if command == "multi-status"
            else {"mode": "native-render"}
        ),
    )

    assert interface.runtime_conflicts(
        allow_owned_native_render=True,
    ) == ()


def test_speed_xbow_preset_cards_and_requested_forms_exist() -> None:
    options = {
        option.card_id: option
        for option in interface.load_card_options()
    }

    assert set(interface.SPEED_XBOW_DECK) <= options.keys()
    for card_id, form_mask in zip(
        interface.SPEED_XBOW_DECK,
        interface.SPEED_XBOW_FORMS,
        strict=True,
    ):
        requested_form = interface.MASK_TO_FORM[form_mask]
        assert requested_form in options[card_id].forms


def test_gui_preflight_defers_vm_detection(monkeypatch) -> None:
    monkeypatch.setattr(
        interface,
        "query_vm_status",
        lambda **_kwargs: pytest.fail("startup must not inspect the VM"),
    )
    monkeypatch.setattr(
        interface,
        "query_memory_status",
        lambda: interface.MemoryStatus(40.0, 12.0, 32.0),
    )
    monkeypatch.setattr(interface, "query_gpu_status", _quiet_gpu)
    monkeypatch.setattr(interface, "query_training_processes", lambda: ())

    report = interface.collect_preflight_report(check_vm=False)

    assert not report.vm.found
    assert report.vm.detail == "离线 VM 将在实际操作时检测"


def test_model_checkpoint_discovery_prefers_newest(
    monkeypatch,
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoints" / "run"
    checkpoint_root.mkdir(parents=True)
    older = checkpoint_root / "checkpoint-step-00000001.pt"
    newer = checkpoint_root / "checkpoint-step-00000002.pt"
    older.write_bytes(b"old")
    newer.write_bytes(b"new")
    older.touch()
    newer.touch()
    older_time = older.stat().st_mtime_ns - 1_000_000_000
    import os

    os.utime(older, ns=(older_time, older_time))
    monkeypatch.setattr(interface, "REPOSITORY_ROOT", tmp_path)

    assert interface.discover_model_checkpoints() == (
        newer.resolve(),
        older.resolve(),
    )

    specialist = checkpoint_root / "hog-specialist.pt"
    specialist.write_bytes(b"specialist")
    assert specialist.resolve() in interface.discover_model_checkpoints()


def test_hidden_launcher_does_not_wait_for_descendant_output_handles() -> None:
    import os
    import signal
    import sys
    import time

    script = (
        "import subprocess,sys; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(10)'],"
        f"close_fds=False,creationflags={interface.CREATE_NO_WINDOW}); "
        "print(child.pid,flush=True)"
    )
    started = time.monotonic()
    result = interface._run_hidden((sys.executable, "-c", script), timeout=2.0)
    try:
        assert result.returncode == 0
        assert time.monotonic() - started < 2.0
    finally:
        os.kill(int(result.stdout.strip()), signal.SIGTERM)


def test_every_selectable_card_has_game_chinese_and_english_labels() -> None:
    cards = interface.load_card_options()
    assert len(cards) == 122
    for card in cards:
        chinese, english = interface.CARD_NAMES[card.card_id]
        assert any("\u4e00" <= char <= "\u9fff" for char in chinese)
        assert english
        assert card.display == f"{chinese} / {english}  [{card.card_id}]"
    assert interface.CARD_NAMES[26000013] == ("炸弹兵", "Bomber")
    assert interface.CARD_NAMES[26000045] == ("飞斧屠夫", "Executioner")


def test_saved_decks_round_trip_keeps_card_order_and_forms(tmp_path: Path) -> None:
    path = tmp_path / "custom_decks.json"
    saved = interface.SavedDeck("我的皮卡牌组", interface.PEKKA_BRIDGE_SPAM_DECK, interface.PEKKA_BRIDGE_SPAM_FORMS, 159_000_004)

    assert interface.load_saved_decks(path) == {}
    interface.write_saved_decks(path, {saved.name: saved})

    assert interface.load_saved_decks(path) == {saved.name: saved}
    assert "我的皮卡牌组" in path.read_text(encoding="utf-8")


def test_saved_decks_without_tower_troop_still_use_princess_tower(tmp_path: Path) -> None:
    path = tmp_path / "custom_decks.json"
    path.write_text(json.dumps({"version": 1, "decks": [{
        "name": "old deck", "deck": list(interface.PEKKA_BRIDGE_SPAM_DECK),
        "forms": list(interface.PEKKA_BRIDGE_SPAM_FORMS),
    }]}), encoding="utf-8")

    assert interface.load_saved_decks(path)["old deck"].tower_troop_id == 159_000_000


def test_saved_decks_reject_invalid_file_without_replacing_it(tmp_path: Path) -> None:
    path = tmp_path / "custom_decks.json"
    path.write_text('{"version": 1, "decks": [{"name": "坏牌组", "deck": [], "forms": []}]}', encoding="utf-8")

    with pytest.raises(ValueError, match="卡牌或形态不正确"):
        interface.load_saved_decks(path)

    assert path.read_text(encoding="utf-8").startswith('{"version": 1')
