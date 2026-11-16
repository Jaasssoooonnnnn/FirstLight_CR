from types import SimpleNamespace

import pytest

from native_runner.competitive_data import load_competitive_data
from native_runner.cr_native_env import NativeClashEnv, RunnerError
from native_runner.duel_modes import DUEL_MODES, resolve_duel_mode, validate_mode_towers
from native_runner.match_factory import MatchConfig
from native_runner.training.v4.factory import deck_roles_from_form_availability


def test_duel_modes_use_eight_card_rules_instead_of_random_or_event_decks():
    modes = load_competitive_data("runtime")["game_modes"]
    assert modes[str(DUEL_MODES[1].game_mode)] == ("7xElixir_Friendly", "7xElixir")
    assert modes[str(DUEL_MODES[2].game_mode)] == ("RampUpElixir_Friendly", "RampUp")
    assert DUEL_MODES[3].game_mode == DUEL_MODES[0].game_mode
    assert DUEL_MODES[1].random_decks and DUEL_MODES[3].evolutions
    assert resolve_duel_mode("7倍圣水无限火力 (7x Elixir)") == DUEL_MODES[1]


@pytest.mark.parametrize("name", ["圣水桶模式 (Elixir Barrel)", DUEL_MODES[4].name, "typo"])
def test_unverified_and_unknown_modes_fail_before_starting(name):
    with pytest.raises(ValueError):
        resolve_duel_mode(name)


def test_extended_deck_roles_report_the_model_projection():
    deck = MatchConfig().deck0
    with pytest.warns(RuntimeWarning, match="role projection"):
        roles = deck_roles_from_form_availability(deck, (1,) * 8)
    assert sum(evo for _, evo in roles.values()) == 2
    assert len(roles) == 8


@pytest.mark.parametrize("error", ["active object graph is not ready", "native battle result is invalid"])
def test_initial_observation_retries_only_a_transient_graph(monkeypatch, error):
    match = MatchConfig()
    observation = {"players": [
        {"owner": o, "deck": [{"cardId": c} for c in deck]}
        for o, deck in enumerate((match.deck0, match.deck1))
    ]}
    calls = []

    def observe():
        calls.append(1)
        if len(calls) == 1:
            raise RunnerError(error)
        return observation

    fake = SimpleNamespace(configure_native_render=lambda *a, **k: None, observe=observe)
    monkeypatch.setattr("native_runner.cr_native_env.time.sleep", lambda _: None)
    if error == "active object graph is not ready":
        assert NativeClashEnv.create_native_match(fake, match) == observation
        assert len(calls) == 2
    else:
        with pytest.raises(RunnerError, match=error):
            NativeClashEnv.create_native_match(fake, match)
        assert len(calls) == 1


def test_initial_observation_wait_is_bounded(monkeypatch):
    fake = SimpleNamespace(
        configure_native_render=lambda *a, **k: None,
        observe=lambda: (_ for _ in ()).throw(RunnerError("active object graph is not ready")),
    )
    times = iter((0.0, 2.0))
    monkeypatch.setattr("native_runner.cr_native_env.time.monotonic", lambda: next(times))
    with pytest.raises(RunnerError, match="not ready"):
        NativeClashEnv.create_native_match(fake, MatchConfig(), wait_timeout=1.0)


def test_unverified_chef_is_rejected_instead_of_replaced():
    with pytest.raises(ValueError, match="皇家大厨"):
        validate_mode_towers(DUEL_MODES[1], 159000004)
    validate_mode_towers(DUEL_MODES[0], 159000004)
    validate_mode_towers(DUEL_MODES[1], 159000000, 159000001)
