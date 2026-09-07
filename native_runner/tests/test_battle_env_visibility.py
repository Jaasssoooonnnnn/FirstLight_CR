from __future__ import annotations

import pytest

from native_runner.battle_env import BattleEnvError
from native_runner.card_specs import build_card_catalog
from native_runner.match_factory import MatchConfig
from native_runner.semantic_subset import SEMANTIC_BASELINE_DECK
from native_runner.tests.test_battle_env_rich_telemetry import (
    _RichNativeStub,
    _environment,
    _ordinary,
    _rich,
)


ROYAL_GHOST = 26000050
UNSUPPORTED_SUPER_WITCH = 26000066


class _NativeStub(_RichNativeStub):
    def __init__(self) -> None:
        ordinary = _ordinary()
        super().__init__(ordinary, _rich(ordinary))
        self.create_calls = 0

    def create_match(self, config: MatchConfig) -> dict[str, object]:
        self.create_calls += 1
        return super().create_match(config)

    def create_native_match(self, config: MatchConfig) -> dict[str, object]:
        return self.create_match(config)


@pytest.fixture(scope="module")
def catalog():
    return build_card_catalog()


def test_hidden_capable_card_is_not_excluded_as_player_private(catalog) -> None:
    ghost = next(item for item in catalog.specs if item.card_id == ROYAL_GHOST)

    from native_runner.semantic_subset import exclusion_reasons

    assert "owner_relative_visibility_unavailable" not in exclusion_reasons(ghost)


def test_unknown_card_fails_before_native_mutation(catalog) -> None:
    native = _NativeStub()
    env = _environment(native, catalog)
    deck = (29999999, *SEMANTIC_BASELINE_DECK[:7])

    with pytest.raises(BattleEnvError, match="unknown IDs"):
        env.reset(match_config=MatchConfig(
            deck0=deck, deck1=SEMANTIC_BASELINE_DECK
        ))

    assert native.create_calls == 0


def test_native_render_path_is_gated_before_renderer_mutation(catalog) -> None:
    native = _NativeStub()
    env = _environment(native, catalog)
    deck = (UNSUPPORTED_SUPER_WITCH, *SEMANTIC_BASELINE_DECK[:7])

    with pytest.raises(BattleEnvError, match="semantic supported-subset"):
        env.reset(
            match_config=MatchConfig(deck0=deck, deck1=SEMANTIC_BASELINE_DECK),
            options={"render_mode": "native-render"},
        )

    assert native.create_calls == 0


def test_reviewed_semantic_deck_passes_gate_and_reports_contract_id(catalog) -> None:
    native = _NativeStub()
    env = _environment(native, catalog)

    _observations, info = env.reset(match_config=MatchConfig(
        deck0=SEMANTIC_BASELINE_DECK,
        deck1=SEMANTIC_BASELINE_DECK,
    ))

    assert native.create_calls == 1
    assert info["semantic_subset_id"] == env.semantic_subset_contract.subset_id
