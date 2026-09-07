from types import SimpleNamespace

import pytest

from native_runner.contracts import ActionKind
from native_runner.training.v4.async_cluster_self_play import (
    _first_successful_card_tick_v4,
    _newly_wasted_elixir_v4,
)


def event(*, tick=200, owner=0, event_type="action_executed",
          kind=ActionKind.PLAY_CARD.value):
    return SimpleNamespace(tick=tick, owner=owner, event_type=event_type,
                           data={"kind": kind})


def test_first_successful_card_uses_attested_execution_not_request_or_ability():
    rows = [event(tick=100, event_type="action_requested"),
            event(tick=110, kind=ActionKind.ACTIVATE_ABILITY.value),
            event(tick=130, owner=1), event(tick=140), event(tick=150)]
    assert _first_successful_card_tick_v4(rows, owner=0, current=None) == 140
    assert _first_successful_card_tick_v4(rows, owner=0, current=120) == 120
    assert _first_successful_card_tick_v4([], owner=0, current=None) is None


@pytest.mark.parametrize("before,after,generated,expected", [
    (10.0, 10.0, .1, .1),
    (9.95, 10.0, .1, .05),
    (9.0, 9.1, .1, 0.0),
    (10.0, 9.5, .1, 0.0),
])
def test_newly_wasted_elixir_matches_transition(before, after, generated, expected):
    assert _newly_wasted_elixir_v4(
        before_elixir=before, after_elixir=after, generated_elixir=generated
    ) == pytest.approx(expected)


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_invalid_wasted_elixir_inputs_rejected(value):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _newly_wasted_elixir_v4(
            before_elixir=value, after_elixir=10.0, generated_elixir=.1
        )
