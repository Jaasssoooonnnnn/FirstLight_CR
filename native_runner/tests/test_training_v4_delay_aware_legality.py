from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from native_runner.battle_env import ELIXIR_SEGMENTS
from native_runner.contracts import ActionKind, ActionV1, TargetKind
from native_runner.perspective import PerspectiveTransformV1
from native_runner.tests.test_training_v4_model import _batch, _model, _with_candidate_mask
from native_runner.training.v4.config import ModelConfigV4
from native_runner.training.v4.decoding import (
    DELAY_AWARE_LEGALITY_DEFAULT,
    ELIXIR_FULL_BAR_MS_BY_MULTIPLIER,
    ELIXIR_MULTIPLIER_SCALAR_INDEX,
    ShadowCandidateLegality,
    elixir_multiplier_from_match_scalars,
)
from native_runner.training.v4.expert import (
    ExpertActionAlignmentError,
    TimedExpertActionV4,
    build_expert_action_batch,
)
from native_runner.training.v4.tensors import ActionSequenceV4, GATE_ACT, UniversalSemanticBatchV4

EXPENSIVE = 3  # _batch candidate row 3: DEPLOY, cost 9, GRID placement everywhere.


def _with_elixir(
    batch: UniversalSemanticBatchV4, elixir: float, *, multiplier: float = 1.0, costs: dict[int, float] | None = None
) -> UniversalSemanticBatchV4:
    match_scalars = batch.match_scalars.clone()
    match_scalars[:, ELIXIR_MULTIPLIER_SCALAR_INDEX] = multiplier / 3.0
    cost = batch.candidates.cost.clone()
    for row, value in (costs or {}).items():
        cost[0, row] = value
    candidates = replace(batch.candidates, elixir=torch.full_like(batch.candidates.elixir, elixir), cost=cost)
    return replace(batch, match_scalars=match_scalars, candidates=candidates)


def _shadow(batch: UniversalSemanticBatchV4, *, delay_aware: bool = True) -> ShadowCandidateLegality:
    return ShadowCandidateLegality(
        batch.candidates,
        ModelConfigV4(),
        elixir_multiplier=elixir_multiplier_from_match_scalars(batch.match_scalars),
        delay_aware=delay_aware,
    )


def _bins(shadow: ShadowCandidateLegality, candidate: int, *, step: int = 0) -> list[bool]:
    return shadow.delay_offset_mask(step=step, candidate_index=torch.tensor([candidate]))[0].tolist()


def test_regen_table_mirrors_native_elixir_segments() -> None:
    assert tuple(segment[2] for segment in ELIXIR_SEGMENTS) == tuple(
        full_bar for _level, full_bar in ELIXIR_FULL_BAR_MS_BY_MULTIPLIER
    )


def test_delay_aware_legality_is_on_by_default_and_legacy_config_keeps_the_old_rule() -> None:
    # C1 full fix: the default follows ModelConfigV4.delay_aware_legality.
    assert DELAY_AWARE_LEGALITY_DEFAULT is True
    assert ModelConfigV4().delay_aware_legality is True
    short = _with_elixir(_batch(batch_size=1), 9.0 - 0.015)
    multiplier = elixir_multiplier_from_match_scalars(short.match_scalars)
    assert ShadowCandidateLegality(short.candidates, ModelConfigV4(), elixir_multiplier=multiplier).delay_aware
    legacy_config = replace(ModelConfigV4(), delay_aware_legality=False)
    legacy = ShadowCandidateLegality(short.candidates, legacy_config, elixir_multiplier=multiplier)
    assert not legacy.delay_aware
    assert not bool(legacy.candidate_mask()[0, EXPENSIVE])
    assert _bins(legacy, 0) == [True] * 5
    # Teacher forcing under the legacy config rejects a label that is legal
    # only after regeneration.
    model = _model().eval()
    model.config = legacy_config
    with pytest.raises(ValueError, match="stored candidate is illegal under shadow state"):
        model.evaluate_actions(short, model.initial_state(1), _forced(EXPENSIVE, uid=104, target=6 * 18 + 5, delay=4))
    # Expert alignment under the legacy config rejects the near-miss touch.
    card_id = int(short.candidates.native_visible_card_id[0, EXPENSIVE])
    with pytest.raises(ExpertActionAlignmentError, match="illegal under V4 shadow state"):
        build_expert_action_batch(
            short,
            ((_play(card_id, hand_slot=3, target=(5, 6), tick=134),),),
            decision_ticks_by_row=(130,),
            perspectives=(PerspectiveTransformV1(actor_owner=0),),
            config=legacy_config,
        )


def test_near_miss_candidate_is_legal_only_at_later_bins() -> None:
    slightly_short = _with_elixir(_batch(batch_size=1), 9.0 - 0.015)
    shadow = _shadow(slightly_short)
    assert bool(shadow.candidate_mask()[0, EXPENSIVE])
    assert _bins(shadow, EXPENSIVE) == [False, True, True, True, True]
    assert not bool(_shadow(slightly_short, delay_aware=False).candidate_mask()[0, EXPENSIVE])

    short = _with_elixir(_batch(batch_size=1), 9.0 - 0.03)
    shadow = _shadow(short)
    assert bool(shadow.candidate_mask()[0, EXPENSIVE])
    assert _bins(shadow, EXPENSIVE) == [False, False, True, True, True]
    # The cheap candidates stay affordable at every bin.
    assert _bins(shadow, 0) == [True] * 5
    assert shadow.affordable_at_bins()[0, EXPENSIVE].tolist() == [False, False, True, True, True]


def test_triple_elixir_shrinks_the_bin_threshold() -> None:
    short = _with_elixir(_batch(batch_size=1), 9.0 - 0.03, multiplier=3.0)
    assert _bins(_shadow(short), EXPENSIVE) == [False, True, True, True, True]
    # fp16 storage of 2/3 still snaps to the native 2x rate (0.0357/tick).
    double = _with_elixir(_batch(batch_size=1), 9.0 - 0.03, multiplier=2.0).to_storage(float_dtype=torch.float16)
    assert _bins(_shadow(double), EXPENSIVE) == [False, True, True, True, True]


@pytest.mark.parametrize("multiplier", [1.0, 3.0])
def test_candidate_far_short_stays_illegal(multiplier: float) -> None:
    far = _with_elixir(_batch(batch_size=1), 8.5, multiplier=multiplier)
    shadow = _shadow(far)
    assert not bool(shadow.candidate_mask()[0, EXPENSIVE])
    assert not bool(shadow.affordable_at_bins()[0, EXPENSIVE].any())


def test_second_action_orders_bins_and_spends_first_cost_at_its_touch() -> None:
    config = ModelConfigV4()
    batch = _with_elixir(_batch(batch_size=1), 6.97)
    active = torch.tensor([True])
    # First action: cost 3 at bin 1 -> 6.97 + 1/56 - 3 = 3.9879, so cost 4 needs bin >= 2.
    shadow = _shadow(batch)
    shadow.apply(active, torch.tensor([0]), torch.tensor([0]), torch.tensor([1]), step=0)
    assert torch.allclose(shadow.remaining_elixir, torch.tensor([3.97]))  # model feature: unchanged semantics
    assert bool(shadow.candidate_mask()[0, 1])
    assert _bins(shadow, 1, step=1) == [False, False, True, True, True]
    legacy = _shadow(batch, delay_aware=False)
    legacy.apply(active, torch.tensor([0]), torch.tensor([0]), torch.tensor([1]), step=0)
    assert not bool(legacy.candidate_mask()[0, 1])
    # A later first touch leaves only order-preserving bins.
    later = _shadow(batch)
    later.apply(active, torch.tensor([0]), torch.tensor([0]), torch.tensor([3]), step=0)
    assert _bins(later, 1, step=1) == [False, False, False, True, True]
    assert len(config.delay_offset_ms) == 5


def test_second_action_respects_the_ten_elixir_cap() -> None:
    # 9.99 + 4/56 caps at 10; the first action (cost 1) stays reserved: 9.0 < 9.05.
    batch = _with_elixir(_batch(batch_size=1), 9.99, costs={EXPENSIVE: 9.05})
    shadow = _shadow(batch)
    shadow.apply(torch.tensor([True]), torch.tensor([2]), torch.tensor([-1]), torch.tensor([4]), step=0)
    assert torch.allclose(shadow.spent_elixir, torch.tensor([1.0]))
    assert torch.allclose(shadow.elixir_at_bins()[0, -1], torch.tensor(9.0))
    assert not bool(shadow.candidate_mask()[0, EXPENSIVE])
    # Touching the first action earlier does not help: the touched command is
    # reserved (executed 20 ticks later), so the raw bar still caps at 10 and
    # the second action sees min(10, 9.99 + k/56) - 1 <= 9.0 at every bin.
    early = _shadow(batch)
    early.apply(torch.tensor([True]), torch.tensor([2]), torch.tensor([-1]), torch.tensor([0]), step=0)
    assert not bool(early.candidate_mask()[0, EXPENSIVE])
    # Below the cap regeneration accrues for the reserved spend as well.
    uncapped = _shadow(_with_elixir(_batch(batch_size=1), 9.90, costs={EXPENSIVE: 8.93}))
    uncapped.apply(torch.tensor([True]), torch.tensor([2]), torch.tensor([-1]), torch.tensor([0]), step=0)
    assert bool(uncapped.candidate_mask()[0, EXPENSIVE])
    assert _bins(uncapped, EXPENSIVE, step=1) == [False, False, True, True, True]


def _forced(candidate: int, *, uid: int, target: int, delay: int) -> ActionSequenceV4:
    return ActionSequenceV4(
        gate=torch.tensor([GATE_ACT]),
        micro_action_count=torch.tensor([1]),
        candidate_index=torch.tensor([[candidate, -1]]),
        candidate_uid=torch.tensor([[uid, -1]]),
        target_cell=torch.tensor([[target, -1]]),
        delay_offset_bin=torch.tensor([[delay, -1]]),
    )


def test_teacher_forcing_accepts_labels_legal_only_at_later_bins() -> None:
    model = _model().eval()
    batch = _with_elixir(_batch(batch_size=1), 9.0 - 0.03)
    state = model.initial_state(1)
    with torch.no_grad():
        output = model.evaluate_actions(batch, state, _forced(EXPENSIVE, uid=104, target=6 * 18 + 5, delay=2))
    assert output.action_components is not None
    assert output.action_components.delay_offset_legal_mask[0, 0].tolist() == [False, False, True, True, True]
    assert torch.isfinite(output.log_prob).all()
    with pytest.raises(ValueError, match="stored delay offsets must preserve action order"):
        model.evaluate_actions(batch, state, _forced(EXPENSIVE, uid=104, target=6 * 18 + 5, delay=1))
    far = _with_elixir(_batch(batch_size=1), 8.5)
    with pytest.raises(ValueError, match="stored candidate is illegal under shadow state"):
        model.evaluate_actions(far, state, _forced(EXPENSIVE, uid=104, target=6 * 18 + 5, delay=4))


@pytest.mark.parametrize("enabled", [False, True])
def test_act_never_samples_an_unaffordable_bin(enabled: bool) -> None:
    model = _model().eval()
    model.config = replace(model.config, delay_aware_legality=enabled)
    with torch.no_grad():
        model.gate_head[-1].bias.copy_(torch.tensor([-20.0, 20.0]))
        model.continue_head[-1].bias.copy_(torch.tensor([20.0, -20.0]))
        delay_head = model.delay_offset_head[-1]
        assert isinstance(delay_head, torch.nn.Linear)
        delay_head.weight.zero_()
        delay_head.bias.copy_(torch.tensor([20.0, 10.0, 0.0, 0.0, 0.0]))  # prefer bin 0
    # Only the near-miss candidate exists.  Legacy config (decision-tick
    # rule): no legal candidate, so the model WAITs.  Delay-aware (default):
    # it may ACT, but only at a bin where regeneration has covered the cost.
    batch = _with_elixir(_batch(batch_size=1), 9.0 - 0.03)
    mask = torch.zeros_like(batch.candidates.mask)
    mask[0, EXPENSIVE] = True
    only_expensive = replace(batch, candidates=_with_candidate_mask(batch.candidates, mask))
    chosen = model.act(only_expensive, model.initial_state(1)).actions
    if enabled:
        assert chosen.gate.tolist() == [GATE_ACT]
        assert chosen.delay_offset_bin.tolist() == [[2, -1]]
    else:
        assert chosen.gate.tolist() == [0]
    with torch.no_grad():
        model.delay_offset_head[-1].bias.zero_()

    config = ModelConfigV4()
    for seed in range(24):
        torch.manual_seed(seed)
        elixir = 3.95 + 0.01 * (seed % 10)
        sampled_batch = _with_elixir(_batch(batch_size=1), elixir)
        with torch.no_grad():
            model.continue_head[-1].bias.copy_(torch.tensor([0.0, 0.0]))
        output = model.sample_for_ppo_rollout(sampled_batch, model.initial_state(1))
        actions = output.actions
        shadow = _shadow(sampled_batch)
        for step in range(int(actions.micro_action_count[0])):
            index = actions.candidate_index[:, step]
            delay = actions.delay_offset_bin[:, step]
            assert bool(shadow.affordable_at_bins()[0, int(index[0]), int(delay[0])])
            assert bool(shadow.delay_offset_mask(step=step, candidate_index=index)[0, int(delay[0])])
            shadow.apply(torch.tensor([True]), index, actions.target_cell[:, step], delay, step=step)
        assert config.max_micro_actions == 2


def _play(card_id: int, *, hand_slot: int, target: tuple[int, int], tick: int) -> TimedExpertActionV4:
    action = ActionV1(
        owner=0,
        kind=ActionKind.PLAY_CARD,
        hand_slot=hand_slot,
        card_id=card_id,
        target_kind=TargetKind.GRID,
        target_grid=target,
        metadata={"source_command_tick": tick},
    )
    return TimedExpertActionV4(source_tick=tick, action=action, source_index=0)


def test_expert_alignment_accepts_near_miss_touch_at_its_bin() -> None:
    batch = _with_elixir(_batch(batch_size=1), 9.0 - 0.03)
    card_id = int(batch.candidates.native_visible_card_id[0, EXPENSIVE])
    perspective = (PerspectiveTransformV1(actor_owner=0),)

    sequence = build_expert_action_batch(
        batch,
        ((_play(card_id, hand_slot=3, target=(5, 6), tick=132),),),
        decision_ticks_by_row=(130,),
        perspectives=perspective,
        delay_aware=True,
    )
    assert sequence.gate.tolist() == [GATE_ACT]
    assert sequence.candidate_uid.tolist() == [[104, -1]]
    assert sequence.target_cell.tolist() == [[6 * 18 + 5, -1]]
    assert sequence.delay_offset_bin.tolist() == [[2, -1]]

    with pytest.raises(ExpertActionAlignmentError, match="unaffordable at its touch delay bin"):
        build_expert_action_batch(
            batch,
            ((_play(card_id, hand_slot=3, target=(5, 6), tick=131),),),
            decision_ticks_by_row=(130,),
            perspectives=perspective,
            delay_aware=True,
        )
    with pytest.raises(ExpertActionAlignmentError, match="illegal under V4 shadow state"):
        build_expert_action_batch(
            batch,
            ((_play(card_id, hand_slot=3, target=(5, 6), tick=132),),),
            decision_ticks_by_row=(130,),
            perspectives=perspective,
            delay_aware=False,
        )
    # Hand membership (slot and card) is unchanged.
    with pytest.raises(ExpertActionAlignmentError, match="matched 0 legal candidates"):
        build_expert_action_batch(
            batch,
            ((_play(card_id, hand_slot=2, target=(5, 6), tick=132),),),
            decision_ticks_by_row=(130,),
            perspectives=perspective,
            delay_aware=True,
        )
