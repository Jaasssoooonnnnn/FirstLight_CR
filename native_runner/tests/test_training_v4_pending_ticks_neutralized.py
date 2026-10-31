"""``pending_ticks_remaining_over_20`` is inert by default (tensorizer and cache).

Live receipts report the touch tick as the execution tick, so the feature is
always 0 in live play while IL caches and PPO self-play saw a decaying
countdown.  ``ModelConfigV4.neutralize_pending_ticks_remaining`` (default on)
zeroes that own-card column everywhere without changing the tensor width.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from native_runner.contracts import ActionKind, ActionV1, PendingActionV1
from native_runner.match_factory import STANDARD_DECK
from native_runner.tests.test_training_v4_cache import _result
from native_runner.tests.test_training_v4_learning import _wait_actions
from native_runner.tests.test_training_v4_model import _batch
from native_runner.tests.test_training_v4_tensorizer import _observation, _semantic_catalog, _tensorizer
from native_runner.training.v4.cache import (
    ILCacheChunkRefV1,
    assert_il_sequences_exact,
    load_il_cache_shard,
    load_il_cache_shards,
    write_il_cache_shard,
)
from native_runner.training.v4.config import (
    ABLATION_FIELD_NAMES,
    CARD_RUNTIME_FEATURE_NAMES,
    PENDING_TICKS_REMAINING_FEATURE_INDEX,
    ModelConfigV4,
)
from native_runner.training.v4.imitation import ILSequenceV4

INDEX = {name: position for position, name in enumerate(CARD_RUNTIME_FEATURE_NAMES)}
PENDING = PENDING_TICKS_REMAINING_FEATURE_INDEX
OTHER_PENDING = tuple(
    INDEX[name]
    for name in (
        "pending_execution",
        "pending_target_x_over_board_width",
        "pending_target_y_over_board_height",
        "ability_pending",
    )
)
FLAG_OFF = ModelConfigV4(neutralize_pending_ticks_remaining=False)


def test_flag_contract() -> None:
    assert PENDING == 17
    assert CARD_RUNTIME_FEATURE_NAMES[PENDING] == "pending_ticks_remaining_over_20"
    assert len(CARD_RUNTIME_FEATURE_NAMES) == ModelConfigV4().card_runtime_feature_dim == 21
    assert ModelConfigV4().neutralize_pending_ticks_remaining is True
    # Observation-contract flag, not an architecture ablation.
    assert "neutralize_pending_ticks_remaining" not in ABLATION_FIELD_NAMES
    assert FLAG_OFF.active_ablations() == {}
    with pytest.raises(ValueError, match="must be bool"):
        ModelConfigV4(neutralize_pending_ticks_remaining=1)  # type: ignore[arg-type]


# --------------------------------------------------------------------- tensorizer


def _in_flight_features(config: ModelConfigV4 | None) -> tuple[torch.Tensor, int, int, torch.Tensor]:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    tensorizer = _tensorizer(specs, catalog, abilities, archetypes, config=config)
    card = ActionV1.play(owner=0, hand_slot=1, grid=(8, 10), card_id=STANDARD_DECK[1], action_id="card")
    ability = ActionV1(owner=0, kind=ActionKind.ACTIVATE_ABILITY, source_entity=501, action_id="ability")
    observation = replace(
        _observation(skeleton_count=0),
        pending_actions=(
            PendingActionV1(action=card, requested_tick=15, expected_execution_tick=36),
            PendingActionV1(action=ability, requested_tick=18, expected_execution_tick=38),
        ),
    )
    batch = tensorizer.tensorize(observation)
    own = batch.own_cards
    rows = {int(own.card_vocab_id[0, row]): row for row in range(8)}
    card_row = rows[tensorizer._vocab_id(STANDARD_DECK[1], context="test")]
    ability_row = rows[tensorizer._vocab_id(STANDARD_DECK[0], context="test")]
    return own.runtime_features[0], card_row, ability_row, batch.opponent_cards.runtime_features


def test_tensorizer_zeroes_pending_ticks_by_default_and_keeps_real_value_when_off() -> None:
    neutral, card_row, ability_row, neutral_opponent = _in_flight_features(None)
    real, real_card_row, real_ability_row, real_opponent = _in_flight_features(FLAG_OFF)
    assert (card_row, ability_row) == (real_card_row, real_ability_row)

    # Flag off: the decaying countdown the IL/PPO paths used to see.
    assert float(real[card_row, PENDING]) == pytest.approx(16 / 20)
    assert float(real[ability_row, PENDING]) == pytest.approx(18 / 20)
    # Default: the column is inert for every own card.
    assert int(torch.count_nonzero(neutral[:, PENDING])) == 0
    assert tuple(neutral.shape) == tuple(real.shape) == (8, 21)

    # (c) every other column, including the other pending ones, is unchanged.
    keep = [column for column in range(neutral.shape[1]) if column != PENDING]
    assert torch.equal(neutral[:, keep], real[:, keep])
    assert float(neutral[card_row, INDEX["pending_execution"]]) == 1.0
    assert float(neutral[ability_row, INDEX["ability_pending"]]) == 1.0
    assert int(torch.count_nonzero(neutral[:, list(OTHER_PENDING[1:3])])) > 0
    assert torch.equal(neutral_opponent, real_opponent)


# -------------------------------------------------------------------------- cache


def _pending_sequence(sequence_id: str, steps: int, *, pending_ticks: float) -> ILSequenceV4:
    batch = _batch(batch_size=1)
    runtime = batch.own_cards.runtime_features.clone()
    runtime[0, 2, INDEX["pending_execution"]] = 1.0
    runtime[0, 2, PENDING] = pending_ticks
    runtime[0, 2, INDEX["pending_target_x_over_board_width"]] = 0.5
    runtime[0, 2, INDEX["pending_target_y_over_board_height"]] = 0.25
    runtime[0, 5, INDEX["ability_pending"]] = 1.0
    runtime[0, 5, PENDING] = pending_ticks / 2
    runtime[0, 5, 0] = 1.0
    batch = replace(batch, own_cards=replace(batch.own_cards, runtime_features=runtime))
    batch = batch.to_storage(float_dtype=torch.float16)
    return ILSequenceV4(
        observations=(batch,) * steps,
        actions=(_wait_actions(1),) * steps,
        episode_start=torch.tensor([[step == 0] for step in range(steps)], dtype=torch.bool),
        valid_mask=torch.ones(steps, 1, dtype=torch.bool),
        returns=torch.arange(steps, dtype=torch.float32)[:, None],
        gate_loss_mask=None,
        value_loss_mask=torch.ones(steps, 1, dtype=torch.bool),
        sequence_id=sequence_id,
    )


def _write_shard(tmp_path: Path, *, pending_ticks: float) -> tuple[Path, tuple[ILSequenceV4, ...]]:
    sequences = (
        _pending_sequence("p:owner-0", 3, pending_ticks=pending_ticks),
        _pending_sequence("p:owner-1", 2, pending_ticks=pending_ticks),
    )
    data_path = tmp_path / "shard-00000.cril.zst"
    write_il_cache_shard(data_path, (_result("p", *sequences),), shard_id="shard-00000", contract={})
    return data_path.with_suffix(".json"), sequences


def _with_column_zeroed(sequence: ILSequenceV4) -> ILSequenceV4:
    observations = []
    for observation in sequence.observations:
        runtime = observation.own_cards.runtime_features.clone()
        runtime[..., PENDING] = 0
        observations.append(replace(observation, own_cards=replace(observation.own_cards, runtime_features=runtime)))
    return replace(sequence, observations=tuple(observations))


def test_cache_round_trip_zeroes_legacy_pending_ticks_by_default(tmp_path: Path) -> None:
    summary_path, sequences = _write_shard(tmp_path, pending_ticks=0.75)
    stored = sequences[0].observations[0].own_cards.runtime_features
    assert int(torch.count_nonzero(stored[..., PENDING])) == 2  # legacy nonzero cache

    shard = load_il_cache_shard(summary_path)
    assert shard.pending_ticks_neutralized
    assert int(torch.count_nonzero(shard.observations.own_cards.runtime_features[..., PENDING])) == 0
    for index, expected in enumerate(sequences):
        # Only column 17 differs from what was written; (c) other pending columns intact.
        assert_il_sequences_exact(_with_column_zeroed(expected), shard.sequence(index))

    loaded = shard.load_batch(
        (
            ILCacheChunkRefV1(sequence_index=0, start=0, steps=3),
            ILCacheChunkRefV1(sequence_index=1, start=0, steps=2),
        ),
        time_steps=3,
    )
    for observation in loaded.sequence.observations:
        runtime = observation.own_cards.runtime_features
        assert int(torch.count_nonzero(runtime[..., PENDING])) == 0
        assert float(runtime[0, 2, INDEX["pending_execution"]]) == 1.0
        assert float(runtime[0, 2, INDEX["pending_target_x_over_board_width"]]) == 0.5
        assert float(runtime[0, 2, INDEX["pending_target_y_over_board_height"]]) == 0.25
        assert float(runtime[0, 5, INDEX["ability_pending"]]) == 1.0
        assert float(runtime[0, 5, 0]) == 1.0

    # The multi-shard loader also defaults to the neutralising config.
    (multi,) = load_il_cache_shards([summary_path])
    assert torch.equal(multi.observations.own_cards.runtime_features, shard.observations.own_cards.runtime_features)


def test_cache_round_trip_keeps_pending_ticks_when_flag_off(tmp_path: Path) -> None:
    summary_path, sequences = _write_shard(tmp_path, pending_ticks=0.75)

    shard = load_il_cache_shard(summary_path, config=FLAG_OFF)
    assert not shard.pending_ticks_neutralized
    for index, expected in enumerate(sequences):
        assert_il_sequences_exact(expected, shard.sequence(index))
    (multi,) = load_il_cache_shards([summary_path], config=FLAG_OFF, workers=2)
    assert float(multi.observations.own_cards.runtime_features[0, 2, PENDING]) == 0.75


def test_cache_neutralisation_is_a_no_op_on_already_zero_column(tmp_path: Path) -> None:
    summary_path, sequences = _write_shard(tmp_path, pending_ticks=0.0)

    neutral = load_il_cache_shard(summary_path)
    untouched = load_il_cache_shard(summary_path, config=FLAG_OFF)
    assert not neutral.pending_ticks_neutralized
    assert torch.equal(
        neutral.observations.own_cards.runtime_features, untouched.observations.own_cards.runtime_features
    )
    for index, expected in enumerate(sequences):
        assert_il_sequences_exact(expected, neutral.sequence(index))
