from __future__ import annotations

import json
from types import SimpleNamespace

from native_runner.contracts import ActionKind, ActionV1, TargetKind
from native_runner.match_factory import MatchConfig
from native_runner.training.v4 import offline_duel


def test_duel_infers_both_owners_and_consumes_force_request(monkeypatch, tmp_path) -> None:
    calls: list[tuple[int, bool]] = []
    observations = {
        owner: SimpleNamespace(owner=owner, tick=90, players=[SimpleNamespace(owner=owner, elixir_exact=6.0)])
        for owner in (0, 1)
    }

    class Native:
        public_card_play_events_from_combat_ring = False

        def set_speed(self, speed):
            assert speed == 1.0

        def resume(self):
            pass

        def pause(self):
            pass

        def close_transport(self):
            pass

    class Environment:
        def reset(self, *, match_config, options):
            assert options["render_mode"] == "native-render"
            return observations, {}

        def step(self, commands, *, record_trace):
            assert commands[0][0].kind == ActionKind.WAIT
            assert commands[1][0].kind == ActionKind.PLAY_CARD
            return observations, {}, {"__all__": True}, {}, {}

    def session(_environment, _model, *, actor_owner, **_options):
        def decide(_observation, *, force_act):
            calls.append((actor_owner, force_act))
            actions = (
                (
                    ActionV1(
                        owner=actor_owner,
                        kind=ActionKind.PLAY_CARD,
                        hand_slot=0,
                        card_id=26000000,
                        target_kind=TargetKind.GRID,
                        target_grid=(3, 6),
                    ),
                )
                if force_act
                else ()
            )
            return SimpleNamespace(decoded=SimpleNamespace(actions=actions), inference_ms=1.0)

        return SimpleNamespace(start_episode=lambda *_args, **_kwargs: None, end_episode=lambda: None, decide=decide)

    monkeypatch.setattr(offline_duel, "NativeClashEnv", lambda *_args, **_kwargs: Native())
    monkeypatch.setattr(offline_duel, "BattleEnvV1", lambda **_kwargs: Environment())
    monkeypatch.setattr(
        offline_duel,
        "load_policy_v4",
        lambda path, **_kwargs: SimpleNamespace(
            model=path, checkpoint_path=path, checkpoint_id=path, checkpoint_sha256=path
        ),
    )
    monkeypatch.setattr(
        offline_duel,
        "production_semantic_bundle",
        lambda: SimpleNamespace(native_card_catalog=None, native_effect_catalog=None, semantic_subset_contract=None),
    )
    monkeypatch.setattr(offline_duel.RuntimeEffectCatalog, "from_catalog", lambda _catalog: None)
    monkeypatch.setattr(offline_duel, "build_policy_session_v4", session)
    (tmp_path / "force-1").touch()

    result = offline_duel.run_ai_duel(("top.pt", "bottom.pt"), MatchConfig(), artifact_dir=tmp_path)

    assert calls == [(0, False), (1, True)]
    assert result["forced_plays"] == [0, 1]
    assert not (tmp_path / "force-1").exists()
    assert json.loads((tmp_path / "identity.json").read_text(encoding="utf-8"))["checkpoints"][1]["path"] == "bottom.pt"
