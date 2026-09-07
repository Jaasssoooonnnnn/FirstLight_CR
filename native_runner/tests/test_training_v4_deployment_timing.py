from native_runner.training.v4.deployment_timing import DeploymentTimingRecorder


def event(kind, tick, seq, deployment="1:7"):
    return dict(
        kind=kind,
        tick=tick,
        nativeEventId=f"1:{seq}",
        deploymentId=deployment,
        cardId=26000010,
        nativeObjectId=123,
        objectKind=1,
    )


def test_delayed_spawn_uses_native_tick_and_deduplicates_multi_unit_card():
    logs = []
    recorder = DeploymentTimingRecorder(logs.append)
    recorder.submit(3, 100, [{"kind": "play_card", "cardId": 26000010}])
    play = event("card_play", 106, 1)
    recorder.observe([play], 110)
    spawn = event("spawn", 108, 2)
    recorder.observe([play, spawn, event("spawn", 109, 3)], 115)
    recorder.observe([spawn], 120)
    rows = [r for r in logs if r["event"] == "deployment_spawn_latency"]
    assert len(rows) == 1
    assert rows[0]["latencyMs"] == 400
    assert rows[0]["spawnTick"] == 108


def test_ambiguous_card_match_does_not_invent_latency():
    logs = []
    recorder = DeploymentTimingRecorder(logs.append)
    for tick in (100, 105):
        recorder.submit(tick, tick, [{"kind": "play_card", "cardId": 26000010}])
    recorder.observe([event("card_play", 110, 1), event("spawn", 111, 2)], 115)
    recorder.finish()
    assert not any(r["event"] == "deployment_spawn_latency" for r in logs)
    assert sum(r["event"] == "deployment_timing_unresolved" for r in logs) == 2


def test_spawn_cannot_match_different_deployment_and_missing_spawn_is_reported():
    logs = []
    recorder = DeploymentTimingRecorder(logs.append)
    recorder.submit(1, 100, [{"kind": "play_card", "cardId": 26000010}])
    recorder.observe(
        [event("card_play", 101, 1), event("spawn", 102, 2, deployment="1:8")], 105
    )
    recorder.finish()
    assert not any(r["event"] == "deployment_spawn_latency" for r in logs)
    assert logs[-1]["reason"] == "no_native_spawn"
