from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest

from native_runner.cr_native_env import RunnerError
from native_runner.match_factory import MatchConfig
from native_runner.native_overlay import NativeOverlayApp, Selection
from native_runner.training.v4 import offline_agent


def test_interface_match_json_round_trips_through_cli(monkeypatch, tmp_path):
    match = MatchConfig(seed=37, king_tower_level=11)
    config = tmp_path / "match.json"
    config.write_text(json.dumps(asdict(match)), encoding="utf-8")
    artifacts = tmp_path / "run"
    artifacts.mkdir()

    def run(checkpoint, restored, **options):
        assert checkpoint == "model.pt"
        assert restored == match
        assert options["actor_owner"] == 1
        assert options["sample"] is False
        assert not options["should_stop"]()
        options["on_ready"]({"mode": "native-render"})
        (artifacts / "stop").touch()
        assert options["should_stop"]()
        return {"stopped": True}

    monkeypatch.setattr(offline_agent, "run_offline_match", run)
    assert (
        offline_agent.main(
            [
                "--checkpoint",
                "model.pt",
                "--match-config",
                str(config),
                "--actor-owner",
                "1",
                "--deterministic",
                "--artifact-dir",
                str(artifacts),
            ]
        )
        == 0
    )
    assert json.loads((artifacts / "ready.json").read_text())["mode"] == "native-render"


@pytest.mark.parametrize("owner", [0, 1])
@pytest.mark.parametrize("hand_temporarily_empty", [False, True])
def test_model_overlay_uses_manual_play_path_for_human_side(owner, hand_temporarily_empty):
    queued = []
    card = {"handIndex": 0, "cardId": 26000000, "cost": 3}
    app = object.__new__(NativeOverlayApp)
    app.owner = owner
    app.card_specs = {}
    app.env = SimpleNamespace(
        observe=lambda: {
            "tick": 1000,
            "players": [
                {
                    "owner": owner,
                    "hand": (
                        [] if queued and hand_temporarily_empty else [{**card, "cardId": 26000001} if queued else card]
                    ),
                    "elixirRaw": 100000,
                }
            ]
        },
        queue_hand_action_at=lambda action, **kwargs: queued.append((action, kwargs)),
    )
    selection = Selection(owner, 0, 0, card["cardId"], 3, "Knight")
    assert app._run_play(selection, 3000, 6000).outcome == "accepted"
    assert queued[0][0].owner == owner
    assert queued[0][1] == {"execute_tick": 1021, "clamp_late": True}
    with pytest.raises(RunnerError):
        app._run_play(Selection(1 - owner, 0, 0, card["cardId"], 3, "Knight"), 3000, 6000)
    app.request_pause_toggle()
    app.request_speed(2)
    assert len(queued) == 1


def test_model_cards_preserve_native_command_age_and_policy_delay():
    from native_runner.contracts import ActionKind, ActionV1, TargetKind

    action = ActionV1(
        owner=0,
        kind=ActionKind.PLAY_CARD,
        hand_slot=0,
        card_id=26000000,
        target_kind=TargetKind.GRID,
        target_grid=(3, 6),
        execute_offset_ticks=4,
    )
    rendered = offline_agent._rendered_action(action)
    assert rendered.execute_offset_ticks == 24
    assert rendered.metadata["native_command_age_ticks"] == 20
    assert rendered.card_id == action.card_id
    assert rendered.target_grid == action.target_grid
    assert offline_agent._rendered_action(
        ActionV1.play(0, 0, (3, 6), execute_offset_ticks=1)
    ).execute_offset_ticks == 21
    wait = ActionV1.wait(0, ticks=5)
    assert offline_agent._rendered_action(wait) is wait


def test_model_inference_keeps_native_renderer_running(monkeypatch, tmp_path):
    calls = []

    class Native:
        running = False

        def set_speed(self, speed):
            assert speed == 1.0

        def resume(self):
            self.running = True
            calls.append("resume")

        def pause(self):
            self.running = False
            calls.append("pause")

        def close_transport(self):
            pass

    native = Native()
    observations = {
        owner: SimpleNamespace(tick=90, players=[SimpleNamespace(owner=owner, elixir_exact=6.0)], events=())
        for owner in (0, 1)
    }

    class Environment:
        def reset(self, *, match_config, options):
            assert not options.get("synchronize_native_render_steps", False)
            return observations, {}

        def step(self, commands, *, record_trace):
            assert native.running
            return observations, {}, {"__all__": True}, {}, {}

    def decide(observation):
        assert native.running
        calls.append("infer")
        return SimpleNamespace(decoded=SimpleNamespace(actions=()), inference_ms=1)

    monkeypatch.setattr(offline_agent, "NativeClashEnv", lambda *_a, **_kw: native)
    monkeypatch.setattr(offline_agent, "BattleEnvV1", lambda **_kw: Environment())
    monkeypatch.setattr(
        offline_agent,
        "load_policy_v4",
        lambda *_a, **_kw: SimpleNamespace(
            model=None, checkpoint_path="test.pt", checkpoint_id="test", checkpoint_sha256="test"
        ),
    )
    monkeypatch.setattr(
        offline_agent,
        "production_semantic_bundle",
        lambda: SimpleNamespace(native_card_catalog=None, native_effect_catalog=None, semantic_subset_contract=None),
    )
    monkeypatch.setattr(offline_agent.RuntimeEffectCatalog, "from_catalog", lambda _: None)
    monkeypatch.setattr(
        offline_agent,
        "build_policy_session_v4",
        lambda *_a, **_kw: SimpleNamespace(
            start_episode=lambda *_a, **_kw: None, end_episode=lambda: None, decide=decide
        ),
    )

    result = offline_agent.run_offline_match("test.pt", MatchConfig(), actor_owner=0, artifact_dir=tmp_path)
    assert result["terminated"]
    assert calls == ["resume", "infer", "pause"]
