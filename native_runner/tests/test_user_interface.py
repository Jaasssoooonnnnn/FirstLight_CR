from __future__ import annotations

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
    app.collected_dataset_note_var = _FakeVar()
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
    assert app.collected_dataset_note_var.get() == "仅当前目录"
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
