"""C1 full fix: cards affordable within the 200 ms decision window are candidates.

One rule everywhere: a decision at native tick d may touch micro-action i at
delay bin k_i (k native ticks); it is legal when
``cost_i <= min(10, e0 + regen(d) * k_i) - sum_{j<i} cost_j`` with
``e0 = max(0, raw - reserved)``.  BattleEnvV1 (mask + compound validation),
the tensorizer, ShadowCandidateLegality (sampling, teacher forcing, decoding),
expert alignment, checkpoint loading and live inference must agree.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from native_runner.battle_env import (
    BattleEnvError,
    TOUCH_WINDOW_DELAY_BINS,
    _earliest_affordable_bin,
    _touch_window_elixir_raw,
)
from native_runner.contracts import ActionKind, ActionV1, TargetKind
from native_runner.tests.test_battle_env import FakeNativeEnv, _card, _make_env, _reset, _test_catalog
from native_runner.tests.test_training_v4_delay_aware_legality import EXPENSIVE, _with_elixir
from native_runner.tests.test_training_v4_model import _batch, _model, _with_candidate_mask
from native_runner.training.v4 import policy_session
from native_runner.training.v4.checkpoint import (
    LEGACY_FIELD_DEFAULTS,
    load_actor_critic_checkpoint,
    model_config_from_contract,
    save_actor_critic_checkpoint,
)
from native_runner.training.v4.config import ModelConfigV4
from native_runner.training.v4.decoding import (
    DELAY_AWARE_LEGALITY_DEFAULT,
    ShadowCandidateLegality,
    decode_action_sequence_v4,
    elixir_multiplier_from_match_scalars,
)
from native_runner.training.v4.expert import (
    ExpertActionAlignmentError,
    TimedExpertActionV4,
    build_expert_action_batch,
)
from native_runner.training.v4.factory import build_episode_tensorizer_v4, build_production_model_v4
from native_runner.training.v4.tensors import ActionSequenceV4, CANDIDATE_DEPLOY, GATE_ACT, GATE_WAIT

# FakeNativeEnv semantic baseline deck: hand slots 0..3 cost 3, 2, 5, 3.
NEAR_MISS_SLOT = 2
NEAR_MISS_COST = 5
TWO_X_TICK = 2_500  # ELIXIR_SEGMENTS: 2x from native tick 2400.


class _WindowNativeEnv(FakeNativeEnv):
    """Consistent four-card hand + cycle with configurable elixir and start tick."""

    def __init__(self, *, elixir_raw: int, tick: int = 90) -> None:
        super().__init__()
        self.elixir_raw = elixir_raw
        self.start_tick = tick

    def _player(self, owner: int, deck: tuple[int, ...]) -> dict[str, Any]:  # type: ignore[override]
        catalog = _test_catalog()
        hand = [
            _card(slot, deck[slot], int(catalog.by_id[deck[slot]].elixir_cost or 0), deck_slot=slot)
            for slot in range(4)
        ]
        return {
            "owner": owner,
            "accountId": 70_000_000 + owner,
            "elixirRaw": self.elixir_raw if owner == 0 else 50_000,
            "hand": hand,
            "nextCard": _card(4, deck[4], int(catalog.by_id[deck[4]].elixir_cost or 0), deck_slot=4),
            "deck": [{"cardId": card_id} for card_id in deck],
            "cycle": [{"cardId": card_id} for card_id in deck[4:]],
        }

    def _initial_state(self, match: Any) -> dict[str, Any]:
        state = super()._initial_state(match)
        state["tick"] = self.start_tick
        return state


def _env(elixir_raw: int, *, tick: int = 90):
    env, native = _make_env(_WindowNativeEnv(elixir_raw=elixir_raw, tick=tick))
    observations, _info = _reset(env)
    return env, native, observations[0]


def _play(slot: int, *, delay_bin: int | None = None, offset: int = 1, grid: tuple[int, int] = (8, 10)) -> ActionV1:
    metadata = {} if delay_bin is None else {"policy_delay_offset_bin": delay_bin, "base_latency_ticks": 1}
    return ActionV1.play(
        owner=0,
        hand_slot=slot,
        grid=grid,
        execute_offset_ticks=offset if delay_bin is None else 1 + delay_bin,
        next_decision_ticks=5,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def test_window_regeneration_uses_native_segment_rates() -> None:
    one_x = _touch_window_elixir_raw(90, 49_700)
    assert len(one_x) == TOUCH_WINDOW_DELAY_BINS
    assert one_x[1] - one_x[0] == pytest.approx(100_000 * 50 / 28_000)
    two_x = _touch_window_elixir_raw(TWO_X_TICK, 49_700)
    assert two_x[1] - two_x[0] == pytest.approx(100_000 * 50 / 14_000)
    three_x = _touch_window_elixir_raw(5_000, 49_700)
    assert three_x[1] - three_x[0] == pytest.approx(100_000 * 50 / 9_300)
    # The raw bar caps at 10 but never drops below the starting value.
    assert _touch_window_elixir_raw(90, 99_950)[-1] == 100_000.0
    # Integral costs are exact at bin 0 (half-raw tolerance only).
    assert _earliest_affordable_bin(30_000, _touch_window_elixir_raw(90, 30_000)) == 0
    assert _earliest_affordable_bin(30_001, _touch_window_elixir_raw(90, 30_000)) == 1


@pytest.mark.parametrize(("tick", "expected_bin"), [(90, 2), (TWO_X_TICK, 1)])
def test_near_miss_card_is_a_candidate_with_its_earliest_bin(tick: int, expected_bin: int) -> None:
    # 0.03 elixir short of a 5-cost card: 1/56 per tick at 1x, 1/28 at 2x.
    _env_, _native, observation = _env(49_700, tick=tick)
    mask = observation.action_mask
    assert mask.hand_slots == (True, True, True, True)
    assert mask.reasons["slots"][str(NEAR_MISS_SLOT)] == "legal"
    bins = mask.reasons["affordable_from_bin"]["slots"]
    assert bins[str(NEAR_MISS_SLOT)] == expected_bin
    assert [bins[str(slot)] for slot in (0, 1, 3)] == [0, 0, 0]
    assert mask.reasons["effective_elixir"] == pytest.approx(4.97)
    # WAIT_UNTIL_AFFORDABLE keeps the decision-tick meaning.
    assert mask.kinds[ActionKind.WAIT_UNTIL_AFFORDABLE.value] is True


@pytest.mark.parametrize("tick", [90, TWO_X_TICK, 5_000])
def test_card_half_an_elixir_short_is_not_a_candidate(tick: int) -> None:
    _env_, _native, observation = _env(45_000, tick=tick)
    mask = observation.action_mask
    assert not mask.hand_slots[NEAR_MISS_SLOT]
    assert mask.reasons["slots"][str(NEAR_MISS_SLOT)] == "insufficient_elixir"
    assert mask.reasons["affordable_from_bin"]["slots"][str(NEAR_MISS_SLOT)] is None


def test_single_near_miss_play_is_validated_at_its_touch_bin() -> None:
    env, _native, _observation = _env(49_700)
    raw = env._raw
    for delay_bin in (0, 1):
        with pytest.raises(BattleEnvError, match="overspends elixir"):
            env._validate_compound_actions(0, (_play(NEAR_MISS_SLOT, delay_bin=delay_bin),), raw)
    for delay_bin in (2, 3, 4):
        env._validate_compound_actions(0, (_play(NEAR_MISS_SLOT, delay_bin=delay_bin),), raw)
    # Without policy metadata the bin is execute_offset_ticks - 1.
    with pytest.raises(BattleEnvError, match="overspends elixir"):
        env._validate_compound_actions(0, (_play(NEAR_MISS_SLOT, offset=2),), raw)
    env._validate_compound_actions(0, (_play(NEAR_MISS_SLOT, offset=3),), raw)


def test_compound_actions_are_validated_sequentially_in_touch_order() -> None:
    # e0 = 6.97 at 1x; slot 1 (2) + slot 2 (5) = 7 needs the second touch at
    # bin >= 2 (6.97 + 2/56 - 2 >= 5), whichever card is touched first.
    env, _native, _observation = _env(69_700)
    raw = env._raw
    env._validate_compound_actions(0, (_play(1, delay_bin=0), _play(NEAR_MISS_SLOT, delay_bin=2)), raw)
    env._validate_compound_actions(0, (_play(1, delay_bin=2), _play(NEAR_MISS_SLOT, delay_bin=2)), raw)
    env._validate_compound_actions(0, (_play(NEAR_MISS_SLOT, delay_bin=0), _play(1, delay_bin=4)), raw)
    with pytest.raises(BattleEnvError, match="overspends elixir"):
        env._validate_compound_actions(0, (_play(1, delay_bin=0), _play(NEAR_MISS_SLOT, delay_bin=1)), raw)
    with pytest.raises(BattleEnvError, match="overspends elixir"):
        env._validate_compound_actions(0, (_play(NEAR_MISS_SLOT, delay_bin=0), _play(1, delay_bin=1)), raw)
    # Submission order does not matter; touch order does.
    with pytest.raises(BattleEnvError, match="overspends elixir"):
        env._validate_compound_actions(0, (_play(NEAR_MISS_SLOT, delay_bin=1), _play(1, delay_bin=0)), raw)
    # 3 + 5 = 8 > 6.97 + 4/56 at every bin.
    with pytest.raises(BattleEnvError, match="overspends elixir"):
        env._validate_compound_actions(0, (_play(0, delay_bin=4), _play(NEAR_MISS_SLOT, delay_bin=4)), raw)


def test_validated_near_miss_play_is_queued_and_reserved() -> None:
    env, native, _observation = _env(49_700)
    play = replace(_play(NEAR_MISS_SLOT, delay_bin=2), next_decision_ticks=1)
    observations, *_rest = env.step({0: play, 1: ActionV1.wait(1, ticks=1)})
    assert len(native.queued) == 1
    assert native.queued[0]["execute_tick"] == 90 + 3  # touch offset 1 + bin 2 (no simulated delay in the fake)
    mask = observations[0].action_mask
    assert mask.reasons["slots"][str(NEAR_MISS_SLOT)] == "pending_execution"
    assert mask.reasons["reserved_elixir"] == 5.0


# ---------------------------------------------------------------------------
# Tensorizer + decoder parity
# ---------------------------------------------------------------------------


def _tensorized(elixir_raw: int, *, tick: int = 90):
    env, native, observation = _env(elixir_raw, tick=tick)
    tensorizer = build_episode_tensorizer_v4(env.episode_config, actor_owner=0)
    tensorizer.start_episode(observation, initial_elixir={0: elixir_raw / 10_000.0, 1: 5.0})
    return env, native, observation, tensorizer, tensorizer.tensorize(observation)


@pytest.mark.parametrize("tick", [90, TWO_X_TICK, 5_000])
@pytest.mark.parametrize(
    "elixir_raw", [0, 1_000, 19_800, 29_700, 29_999, 30_000, 48_000, 49_300, 49_700, 49_950, 70_000]
)
def test_tensorizer_rows_and_decoder_bins_match_the_environment(tick: int, elixir_raw: int) -> None:
    _env_, _native, observation, tensorizer, batch = _tensorized(elixir_raw, tick=tick)
    candidates = batch.candidates
    reasons = observation.action_mask.reasons
    assert float(candidates.elixir[0]) == pytest.approx(reasons["effective_elixir"], abs=1e-6)
    assert float(batch.match_scalars[0, 1]) * 3.0 == pytest.approx(observation.time.elixir_multiplier, abs=1e-6)
    shadow = ShadowCandidateLegality(
        candidates, tensorizer.config, elixir_multiplier=elixir_multiplier_from_match_scalars(batch.match_scalars)
    )
    assert shadow.delay_aware
    affordable = shadow.affordable_at_bins()[0]
    env_bins = reasons["affordable_from_bin"]["slots"]
    emitted = set()
    for row in candidates.mask[0].nonzero(as_tuple=False).flatten().tolist():
        assert int(candidates.variant[0, row]) == CANDIDATE_DEPLOY
        slot = int(candidates.native_hand_slot[0, row])
        emitted.add(slot)
        expected = env_bins[str(slot)]
        assert expected is not None
        decoder_bins = affordable[row].tolist()
        assert decoder_bins.index(True) == expected
        assert decoder_bins == [index >= expected for index in range(TOUCH_WINDOW_DELAY_BINS)]
    assert emitted == {slot for slot, legal in enumerate(observation.action_mask.hand_slots) if legal}
    assert emitted == {int(slot) for slot, value in env_bins.items() if value is not None}
    assert bool(shadow.candidate_mask()[0, candidates.mask[0]].all())


# ---------------------------------------------------------------------------
# Expert alignment / teacher forcing
# ---------------------------------------------------------------------------


def _expert_play(card_id: int, *, tick: int, grid: tuple[int, int] = (8, 10)) -> TimedExpertActionV4:
    action = ActionV1(
        owner=0,
        kind=ActionKind.PLAY_CARD,
        hand_slot=NEAR_MISS_SLOT,
        card_id=card_id,
        target_kind=TargetKind.GRID,
        target_grid=grid,
        metadata={"source_command_tick": tick},
    )
    return TimedExpertActionV4(source_tick=tick, action=action, source_index=0)


def test_expert_touch_at_bin_two_on_a_near_miss_card_is_teacher_forced() -> None:
    _env_, _native, observation, tensorizer, batch = _tensorized(49_700)
    card_id = int(observation.players[0].hand[NEAR_MISS_SLOT])
    decision = observation.tick
    sequence = build_expert_action_batch(
        batch,
        ((_expert_play(card_id, tick=decision + 2),),),
        decision_ticks_by_row=(decision,),
        perspectives=(tensorizer.perspective,),
        config=tensorizer.config,
    )
    assert sequence.gate.tolist() == [GATE_ACT]
    assert sequence.delay_offset_bin.tolist() == [[2, -1]]
    with pytest.raises(ExpertActionAlignmentError, match="unaffordable at its touch delay bin"):
        build_expert_action_batch(
            batch,
            ((_expert_play(card_id, tick=decision + 1),),),
            decision_ticks_by_row=(decision,),
            perspectives=(tensorizer.perspective,),
            config=tensorizer.config,
        )
    model = build_production_model_v4(device="cpu").eval()
    with torch.no_grad():
        output = model.evaluate_actions(batch, model.initial_state(1), sequence)
    assert torch.isfinite(output.log_prob).all()
    assert output.action_components is not None
    assert output.action_components.delay_offset_legal_mask[0, 0].tolist() == [False, False, True, True, True]


def test_native_decoder_checks_the_touch_bin_and_the_environment_accepts_it() -> None:
    env, _native, observation, tensorizer, batch = _tensorized(49_700)
    candidates = batch.candidates
    row = int(((candidates.native_hand_slot[0] == NEAR_MISS_SLOT) & candidates.mask[0]).nonzero()[0, 0])

    def sequence(delay_bin: int) -> ActionSequenceV4:
        return ActionSequenceV4(
            gate=torch.tensor([GATE_ACT]),
            micro_action_count=torch.tensor([1]),
            candidate_index=torch.tensor([[row, -1]]),
            candidate_uid=torch.tensor([[int(candidates.uid[0, row]), -1]]),
            target_cell=torch.tensor([[10 * 18 + 8, -1]]),
            delay_offset_bin=torch.tensor([[delay_bin, -1]]),
        )

    def decode(delay_bin: int, config: ModelConfigV4):
        return decode_action_sequence_v4(
            sequence(delay_bin),
            candidates,
            row=0,
            observation=observation,
            catalog=tensorizer.catalog,
            deck=tensorizer.deck,
            card_costs=tensorizer.card_costs,
            ability_id_by_vocab_id=tensorizer.ability_id_by_vocab_id,
            config=config,
        )

    with pytest.raises(ValueError, match="sampled delay bin is unaffordable"):
        decode(1, tensorizer.config)
    decoded = decode(2, tensorizer.config)
    assert decoded.actions[0].metadata["policy_delay_offset_bin"] == 2
    assert decoded.actions[0].execute_offset_ticks == 3
    env._validate_compound_actions(0, decoded.actions, env._raw)
    with pytest.raises(ValueError, match="illegal under V4 shadow state"):
        decode(2, replace(tensorizer.config, delay_aware_legality=False))


# ---------------------------------------------------------------------------
# Sampling (PPO self-play) never leaves the environment's legal set
# ---------------------------------------------------------------------------


def test_sampled_compound_actions_are_affordable_and_pass_the_environment() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_production_model_v4(device=device).eval()
    with torch.no_grad():
        model.gate_head[-1].bias.copy_(torch.tensor([-20.0, 20.0]))
        model.continue_head[-1].bias.copy_(torch.tensor([0.0, 0.0]))
        delay_head = model.delay_offset_head[-1]
        assert isinstance(delay_head, torch.nn.Linear)
        delay_head.weight.zero_()
        delay_head.bias.zero_()
    total = 0
    acted_late = 0
    scenarios = ((49_700, 90), (69_700, 90), (79_700, TWO_X_TICK), (49_000, 5_000))
    samples = 250
    for elixir_raw, tick in scenarios:
        env, _native, observation, tensorizer, batch = _tensorized(elixir_raw, tick=tick)
        repeated = batch.index_select(torch.zeros(samples, dtype=torch.long))
        torch.manual_seed(elixir_raw + tick)
        with torch.no_grad():
            output = model.sample_for_ppo_rollout(
                repeated.to_model_input(device), model.initial_state(samples, device=device)
            )
        actions = output.actions.to(torch.device("cpu"))
        actions.validate(tensorizer.config, candidate_count=int(repeated.candidates.mask.shape[1]))
        shadow = ShadowCandidateLegality(
            repeated.candidates,
            tensorizer.config,
            elixir_multiplier=elixir_multiplier_from_match_scalars(repeated.match_scalars),
        )
        for step in range(2):
            active = actions.micro_action_count.cpu() > step
            index = actions.candidate_index[:, step].cpu().clamp_min(0)
            delay = actions.delay_offset_bin[:, step].cpu().clamp_min(0)
            pairs = shadow.affordable_at_bins()[torch.arange(samples), index, delay]
            assert bool((pairs | ~active).all()), "sampled an unaffordable (candidate, bin) pair"
            ordered = shadow.delay_offset_mask(step=step, candidate_index=index)[torch.arange(samples), delay]
            assert bool((ordered | ~active).all())
            shadow.apply(active, index, actions.target_cell[:, step].cpu(), delay, step=step)
        for row in range(samples):
            if int(actions.gate[row]) != GATE_ACT:
                continue
            # Every row is a copy of the single tensorized observation.
            decoded = decode_action_sequence_v4(
                actions.index_select(torch.tensor([row])),
                batch.candidates,
                row=0,
                observation=observation,
                catalog=tensorizer.catalog,
                deck=tensorizer.deck,
                card_costs=tensorizer.card_costs,
                ability_id_by_vocab_id=tensorizer.ability_id_by_vocab_id,
                horizontal_mirror=tensorizer.perspective.horizontal_mirror,
                hand_slot_permutation=tensorizer.perspective.hand_slot_permutation,
                config=tensorizer.config,
                base_latency_ticks=1,
                validate=False,
            )
            env._validate_compound_actions(0, decoded.actions, env._raw)
            acted_late += any(int(item.metadata["policy_delay_offset_bin"]) >= 1 for item in decoded.actions)
            total += 1
    assert total >= 900
    assert acted_late > 0


# ---------------------------------------------------------------------------
# Config, checkpoints and live inference
# ---------------------------------------------------------------------------


def test_config_flag_drives_the_shadow_and_the_module_default() -> None:
    assert ModelConfigV4().delay_aware_legality is True
    assert DELAY_AWARE_LEGALITY_DEFAULT is True
    with pytest.raises(ValueError, match="must be bool"):
        ModelConfigV4(delay_aware_legality=1)  # type: ignore[arg-type]
    short = _with_elixir(_batch(batch_size=1), 9.0 - 0.03)
    legacy = replace(ModelConfigV4(), delay_aware_legality=False)
    assert ShadowCandidateLegality(short.candidates, ModelConfigV4()).delay_aware
    assert not ShadowCandidateLegality(short.candidates, legacy).delay_aware
    assert ShadowCandidateLegality(short.candidates, legacy, delay_aware=True).delay_aware


def test_legacy_checkpoint_without_the_field_keeps_the_decision_tick_rule(tmp_path: Path) -> None:
    assert LEGACY_FIELD_DEFAULTS == {"delay_aware_legality": False}
    source = tmp_path / "current.pt"
    save_actor_critic_checkpoint(source, _model(), training_stage="imitation", gamma_per_decision=0.99)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    assert payload["contract"]["model_config"]["delay_aware_legality"] is True
    payload["contract"]["model_config"].pop("delay_aware_legality")
    legacy_path = tmp_path / "legacy.pt"
    torch.save(payload, legacy_path)

    assert model_config_from_contract(payload["contract"]).delay_aware_legality is False
    model = _model()
    assert model.config.delay_aware_legality is True
    load_actor_critic_checkpoint(legacy_path, model, map_location="cpu")
    assert model.config.delay_aware_legality is False
    # The loaded model samples/teacher-forces with the old rule.
    short = _with_elixir(_batch(batch_size=1), 9.0 - 0.03)
    mask = torch.zeros_like(short.candidates.mask)
    mask[0, EXPENSIVE] = True
    only_expensive = replace(short, candidates=_with_candidate_mask(short.candidates, mask))
    with torch.no_grad():
        model.gate_head[-1].bias.copy_(torch.tensor([-20.0, 20.0]))
        assert model.eval().act(only_expensive, model.initial_state(1)).actions.gate.tolist() == [GATE_WAIT]

    # A current checkpoint (field present) loads into a legacy-configured model
    # and the model adopts the checkpoint's rule; the state dict stays strict.
    legacy_model = _model()
    legacy_model.config = replace(legacy_model.config, delay_aware_legality=False)
    load_actor_critic_checkpoint(source, legacy_model, map_location="cpu")
    assert legacy_model.config.delay_aware_legality is True
    # Architecture fields stay strict.
    payload["contract"]["model_config"]["lstm_hidden_dim"] = 1024
    torch.save(payload, legacy_path)
    with pytest.raises(ValueError, match=r"model_config\.lstm_hidden_dim"):
        load_actor_critic_checkpoint(legacy_path, _model(), map_location="cpu")


def _only(candidate_elixir: float, cost: float):
    batch = _with_elixir(_batch(batch_size=1), candidate_elixir, costs={EXPENSIVE: cost})
    mask = torch.zeros_like(batch.candidates.mask)
    mask[0, EXPENSIVE] = True
    return replace(batch, candidates=_with_candidate_mask(batch.candidates, mask))


def test_live_delay_cap_is_part_of_shadow_legality() -> None:
    model = _model().eval()
    with torch.no_grad():
        model.gate_head[-1].bias.copy_(torch.tensor([-20.0, 20.0]))
        delay_head = model.delay_offset_head[-1]
        assert isinstance(delay_head, torch.nn.Linear)
        delay_head.weight.zero_()
        delay_head.bias.copy_(torch.tensor([0.0, 0.0, 0.0, 0.0, 20.0]))  # prefer bin 4
    bin_two = _only(9.0 - 0.03, 9.0)  # affordable from bin 2 at 1x
    bin_one = _only(9.0 - 0.015, 9.0)  # affordable from bin 1 at 1x
    assert model.act(bin_two, model.initial_state(1)).actions.gate.tolist() == [GATE_ACT]

    model.set_inference_delay_offset_max_ms(50)
    allowed = ShadowCandidateLegality(
        bin_two.candidates,
        model.config,
        elixir_multiplier=elixir_multiplier_from_match_scalars(bin_two.match_scalars),
        allowed_delay_bins=model._inference_delay_offset_mask,
    )
    assert not bool(allowed.candidate_mask()[0, EXPENSIVE])
    assert model.act(bin_two, model.initial_state(1)).actions.gate.tolist() == [GATE_WAIT]
    chosen = model.act(bin_one, model.initial_state(1)).actions
    assert chosen.gate.tolist() == [GATE_ACT]
    assert chosen.delay_offset_bin.tolist() == [[1, -1]]
    # Teacher forcing / PPO evaluation is scored without the inference cap.
    with torch.no_grad():
        output = model.evaluate_actions(bin_two, model.initial_state(1), _forced_late())
    assert torch.isfinite(output.log_prob).all()


def _forced_late():
    from native_runner.tests.test_training_v4_delay_aware_legality import _forced

    return _forced(EXPENSIVE, uid=104, target=6 * 18 + 5, delay=4)


def test_loader_applies_the_recorded_delay_cap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-step-00000001.pt"
    checkpoint.write_bytes(b"x")
    for extra, expected in (({"inference_delay_offset_max_ms": 50}, [True, True, False, False, False]), ({}, [True] * 5)):
        model = _model()
        payload = {
            "contract": {},
            "extra": extra,
            "checkpoint_id": "recorded-cap",
            "update_step": 1,
            "training_stage": "imitation",
        }
        monkeypatch.setattr(policy_session, "build_production_model_v4", lambda *, device, config, _model=model: _model)
        monkeypatch.setattr(policy_session, "model_config_from_contract", lambda contract: None)
        monkeypatch.setattr(policy_session, "read_actor_critic_checkpoint", lambda *args, _p=payload, **kw: dict(_p))
        monkeypatch.setattr(policy_session, "restore_actor_critic_checkpoint", lambda payload, model, **kw: payload)
        loaded = policy_session.load_policy_v4(checkpoint, device="cpu")
        assert loaded.model._inference_delay_offset_mask.tolist() == expected
