from dataclasses import fields, is_dataclass, replace

import pytest
import torch

from native_runner.contracts import ActionKind, ActionV1, TargetKind
from native_runner.perspective import PerspectiveTransformV1
from native_runner.tests.test_training_v4_model import _batch, _model
from native_runner.training.v4.tensors import ActionEvaluationComponentsV4, ActionSequenceV4, GATE_ACT, GATE_WAIT, TARGET_ENTITY
from native_runner.training.v4.imitation import ILSequenceV4, collate_il_sequences, collate_padded_il_sequences, imitation_loss, slice_il_sequence
from native_runner.training.v4.config import ModelConfigV4
from native_runner.training.v4.ppo import PPOConfigV4, StoredRolloutV4
from native_runner.training.v4.learning import RecurrentEvaluationV4, discounted_returns, evaluate_recurrent_sequence, generalized_advantage_estimate, trusted_decision_mask
from native_runner.training.v4.expert import TimedExpertActionV4, build_expert_action_batch, screen_expert_timeline
from native_runner.training.v4.checkpoint import load_actor_critic_checkpoint, save_actor_critic_checkpoint
from native_runner.training.v4.dataset import screen_replay_payload
from native_runner.training.v4.train_imitation_cache import _DistributedImitationModule
from native_runner.training.v4.components import SpatialScatterEncoder
from native_runner.training.v4.policy_session import _DeterministicCudaGraphV4
from native_runner.training.v4.distributed_ppo import (
    _balanced_lane_permutation,
    distributed_ppo_update_v4,
)
from native_runner.training.v4.async_cluster_self_play import (
    _RowPackedTensorRecordV4,
    _StochasticCudaGraphV4,
    _action_to_wire,
    _actions_from_packed,
    _pack_action_batch_to_cpu,
    _persistent_rollout_graph_v4,
)
from native_runner.training.v4.imitation import (
    _delay_neighbor_smoothed_nll,
    _masked_gate_mean,
)
from native_runner.training.v4.tensors import (
    concatenate_padded_tensor_records,
    concatenate_tensor_records,
)


def _clone_tensor_tree(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if is_dataclass(value):
        return type(value)(
            **{
                item.name: _clone_tensor_tree(getattr(value, item.name))
                for item in fields(value)
            }
        )
    return value


def _assert_tensor_tree_equal(actual: object, expected: object) -> None:
    if isinstance(actual, torch.Tensor):
        assert isinstance(expected, torch.Tensor)
        if actual.is_floating_point():
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        else:
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        return
    if is_dataclass(actual):
        assert type(expected) is type(actual)
        for item in fields(actual):
            _assert_tensor_tree_equal(
                getattr(actual, item.name),
                getattr(expected, item.name),
            )
        return
    assert actual == expected


def _wait_actions(batch_size: int) -> ActionSequenceV4:
    return ActionSequenceV4(
        gate=torch.full((batch_size,), GATE_WAIT, dtype=torch.long),
        micro_action_count=torch.zeros(batch_size, dtype=torch.long),
        candidate_index=torch.full((batch_size, 2), -1, dtype=torch.long),
        candidate_uid=torch.full((batch_size, 2), -1, dtype=torch.long),
        target_cell=torch.full((batch_size, 2), -1, dtype=torch.long),
        delay_offset_bin=torch.full((batch_size, 2), -1, dtype=torch.long),
    )


def _one_action(batch_size: int = 2) -> ActionSequenceV4:
    result = _wait_actions(batch_size)
    result.gate[0] = GATE_ACT
    result.micro_action_count[0] = 1
    result.candidate_index[0, 0] = 0
    result.candidate_uid[0, 0] = 101
    result.target_cell[0, 0] = 0
    result.delay_offset_bin[0, 0] = 0
    return result


def test_il_loss_normalizes_each_head_by_its_own_population() -> None:
    batch = _batch(batch_size=2)
    actions = (_one_action(), _wait_actions(2))
    sequence = ILSequenceV4(
        observations=(batch, batch),
        actions=actions,
        episode_start=torch.tensor([[True, True], [False, False]]),
        valid_mask=torch.ones(2, 2, dtype=torch.bool),
        returns=torch.zeros(2, 2),
    )
    components = ActionEvaluationComponentsV4(
        gate_log_prob=torch.full((2, 2), -1.0),
        gate_entropy=torch.zeros(2, 2),
        candidate_log_prob=torch.full((2, 2, 2), -2.0),
        candidate_entropy=torch.zeros(2, 2, 2),
        target_log_prob=torch.full((2, 2, 2), -3.0),
        target_entropy=torch.zeros(2, 2, 2),
        delay_offset_log_prob=torch.full((2, 2, 2), -4.0),
        delay_offset_log_probs=torch.full((2, 2, 2, 5), -4.0),
        delay_offset_legal_mask=torch.ones(2, 2, 2, 5, dtype=torch.bool),
        delay_offset_entropy=torch.zeros(2, 2, 2),
        continue_log_prob=torch.full((2, 2), -5.0),
        continue_entropy=torch.zeros(2, 2),
    )
    state = _model().initial_state(2)
    evaluation = RecurrentEvaluationV4(
        log_prob=torch.zeros(2, 2),
        entropy=torch.zeros(2, 2),
        value=torch.zeros(2, 2),
        components=components,
        final_state=state,
    )

    losses = imitation_loss(
        sequence,
        evaluation,
        gate_act_weight=1.0,
        delay_neighbor_weight=0.0,
    )

    assert losses.gate.item() == pytest.approx(1.0)
    assert losses.candidate.item() == pytest.approx(2.0)
    assert losses.target.item() == pytest.approx(3.0)
    assert losses.delay.item() == pytest.approx(4.0)
    assert losses.continue_action.item() == pytest.approx(5.0)
    assert losses.gate_count.item() == 4
    assert losses.candidate_count.item() == 1
    assert losses.target_count.item() == 1
    assert losses.delay_count.item() == 1
    assert losses.continue_count.item() == 1


def test_gate_act_class_weight_counts_positive_rows_eight_times() -> None:
    loss, count = _masked_gate_mean(
        torch.tensor([2.0, 1.0, 1.0, 1.0]),
        torch.ones(4, dtype=torch.bool),
        torch.tensor([GATE_ACT, GATE_WAIT, GATE_WAIT, GATE_WAIT]),
        act_weight=8.0,
    )

    assert loss.item() == pytest.approx((8.0 * 2.0 + 3.0) / 4.0)
    assert count.item() == 4


@pytest.mark.parametrize(
    ("target_bin", "legal", "expected"),
    (
        (2, (True, True, True, True, True), 1.6),
        (0, (True, True, True, True, True), 4.25),
        (2, (False, False, True, True, True), 1.5),
    ),
)
def test_delay_neighbor_smoothing_respects_boundaries_and_ordering(
    target_bin: int,
    legal: tuple[bool, ...],
    expected: float,
) -> None:
    losses = _delay_neighbor_smoothed_nll(
        -torch.tensor([[5.0, 2.0, 1.0, 3.0, 7.0]]),
        torch.tensor([legal]),
        torch.tensor([target_bin]),
        neighbor_weight=0.2,
    )

    assert losses.item() == pytest.approx(expected)


def test_recurrent_evaluation_carries_state_instead_of_resetting_frames() -> None:
    torch.manual_seed(3)
    model = _model().eval()
    batch = _batch(batch_size=1)
    actions = _wait_actions(1)
    starts = torch.tensor([[True], [False]])

    with torch.no_grad():
        stacked = evaluate_recurrent_sequence(
            model,
            (batch, batch),
            (actions, actions),
            starts,
            gate_temperature=1.0,
            action_temperature=1.0,
        )
        first = model.evaluate_actions(
            batch,
            model.initial_state(1),
            actions,
            episode_start=starts[0],
            gate_temperature=1.0,
            action_temperature=1.0,
        )
        second = model.evaluate_actions(
            batch,
            first.next_state,
            actions,
            episode_start=starts[1],
            gate_temperature=1.0,
            action_temperature=1.0,
        )

    torch.testing.assert_close(stacked.value, torch.stack((first.value, second.value)))
    torch.testing.assert_close(stacked.final_state.hidden, second.next_state.hidden)


def test_recurrent_preencoding_matches_stepwise_observation_encoding() -> None:
    torch.manual_seed(31)
    model = _model().eval()
    batch = _batch(batch_size=2)
    actions = _wait_actions(2)
    starts = torch.tensor([[True, True], [False, False], [False, False]])

    with torch.no_grad():
        stepwise = evaluate_recurrent_sequence(
            model,
            (batch, batch, batch),
            (actions, actions, actions),
            starts,
        )
        preencoded = evaluate_recurrent_sequence(
            model,
            (batch, batch, batch),
            (actions, actions, actions),
            starts,
            preencode_observations=True,
        )

    torch.testing.assert_close(preencoded.log_prob, stepwise.log_prob)
    torch.testing.assert_close(preencoded.value, stepwise.value)
    torch.testing.assert_close(
        preencoded.final_state.hidden,
        stepwise.final_state.hidden,
    )


def test_fused_ppo_sequence_matches_stepwise_recurrent_evaluation() -> None:
    torch.manual_seed(37)
    model = _model().eval()
    batch = _batch(batch_size=2)
    actions = _wait_actions(2)
    starts = torch.tensor([[True, True], [False, False], [False, False]])
    flat_batch = concatenate_padded_tensor_records((batch, batch, batch))
    flat_actions = concatenate_tensor_records((actions, actions, actions))

    with torch.no_grad():
        encoded = model.encode_observation(flat_batch)
        fused_log_prob, fused_entropy, fused_value, fused_state = (
            model.evaluate_encoded_action_sequence(
                flat_batch,
                encoded,
                model.initial_state(2),
                flat_actions,
                starts,
                time_steps=3,
            )
        )
        stepwise = evaluate_recurrent_sequence(
            model,
            (batch, batch, batch),
            (actions, actions, actions),
            starts,
            encoded_observations=tuple(
                encoded.narrow_batch(step * 2, 2) for step in range(3)
            ),
        )

    torch.testing.assert_close(fused_log_prob, stepwise.log_prob)
    torch.testing.assert_close(fused_entropy, stepwise.entropy)
    torch.testing.assert_close(fused_value, stepwise.value)
    torch.testing.assert_close(fused_state.hidden, stepwise.final_state.hidden)
    torch.testing.assert_close(fused_state.cell, stepwise.final_state.cell)


@pytest.mark.parametrize(
    "actions",
    (_wait_actions(2), _one_action(2)),
    ids=("all-wait", "mixed-act-wait"),
)
def test_fused_ppo_sequence_preserves_training_gradients(
    actions: ActionSequenceV4,
) -> None:
    torch.manual_seed(41)
    stepwise_model = _model().train()
    fused_model = _model().train()
    fused_model.load_state_dict(stepwise_model.state_dict())
    batch = _batch(batch_size=2)
    starts = torch.tensor([[True, True], [False, False]])

    stepwise = evaluate_recurrent_sequence(
        stepwise_model,
        (batch, batch),
        (actions, actions),
        starts,
        preencode_observations=True,
    )
    stepwise_loss = (
        stepwise.log_prob.sum()
        + stepwise.value.sum()
        + 0.01 * stepwise.entropy.sum()
    )
    stepwise_loss.backward()

    flat_batch = concatenate_padded_tensor_records((batch, batch))
    flat_actions = concatenate_tensor_records((actions, actions))
    encoded = fused_model.encode_observation(flat_batch)
    fused_log_prob, fused_entropy, fused_value, _state = (
        fused_model.evaluate_encoded_action_sequence(
            flat_batch,
            encoded,
            fused_model.initial_state(2),
            flat_actions,
            starts,
            time_steps=2,
        )
    )
    fused_loss = (
        fused_log_prob.sum()
        + fused_value.sum()
        + 0.01 * fused_entropy.sum()
    )
    fused_loss.backward()

    for (stepwise_name, stepwise_parameter), (fused_name, fused_parameter) in zip(
        stepwise_model.named_parameters(),
        fused_model.named_parameters(),
        strict=True,
    ):
        assert fused_name == stepwise_name
        if stepwise_parameter.grad is None or fused_parameter.grad is None:
            assert stepwise_parameter.grad is None
            assert fused_parameter.grad is None
            continue
        torch.testing.assert_close(
            fused_parameter.grad,
            stepwise_parameter.grad,
            rtol=2e-4,
            atol=2e-5,
        )


def test_il_collation_and_chunks_preserve_chronology_and_masks() -> None:
    batch = _batch(batch_size=1)
    first = ILSequenceV4(
        observations=(batch, batch, batch),
        actions=(_wait_actions(1),) * 3,
        episode_start=torch.tensor([[True], [False], [False]]),
        valid_mask=torch.ones(3, 1, dtype=torch.bool),
        returns=torch.tensor([[1.0], [2.0], [3.0]]),
        sequence_id="first",
    )
    second = replace(
        first,
        returns=torch.tensor([[4.0], [5.0], [6.0]]),
        gate_loss_mask=torch.tensor([[True], [False], [True]]),
        sequence_id="second",
    )

    collated = collate_il_sequences((first, second))
    chunks = (slice_il_sequence(collated, 0, 2), slice_il_sequence(collated, 2, 3))

    assert collated.batch_size == 2
    assert collated.returns.tolist() == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    assert collated.gate_loss_mask is not None
    assert collated.gate_loss_mask.tolist() == [
        [True, True],
        [True, False],
        [True, True],
    ]
    assert [chunk.time_steps for chunk in chunks] == [2, 1]
    assert chunks[1].episode_start.tolist() == [[False, False]]


def test_padded_il_collation_masks_variable_length_tail() -> None:
    batch = _batch(batch_size=1)
    long = ILSequenceV4(
        observations=(batch, batch, batch),
        actions=(_wait_actions(1),) * 3,
        episode_start=torch.tensor([[True], [False], [False]]),
        valid_mask=torch.ones(3, 1, dtype=torch.bool),
        returns=torch.tensor([[1.0], [2.0], [3.0]]),
        value_loss_mask=torch.ones(3, 1, dtype=torch.bool),
        sequence_id="long",
    )
    short = ILSequenceV4(
        observations=(batch, batch),
        actions=(_wait_actions(1),) * 2,
        episode_start=torch.tensor([[True], [False]]),
        valid_mask=torch.ones(2, 1, dtype=torch.bool),
        returns=torch.tensor([[4.0], [5.0]]),
        gate_loss_mask=torch.tensor([[True], [False]]),
        value_loss_mask=torch.ones(2, 1, dtype=torch.bool),
        sequence_id="short",
    )

    collated = collate_padded_il_sequences((long, short))

    assert collated.time_steps == 3
    assert collated.batch_size == 2
    assert collated.valid_mask.tolist() == [
        [True, True],
        [True, True],
        [True, False],
    ]
    assert collated.episode_start.tolist() == [
        [True, True],
        [False, False],
        [False, False],
    ]
    assert collated.returns.tolist() == [[1.0, 4.0], [2.0, 5.0], [3.0, 0.0]]
    assert collated.gate_loss_mask is not None
    assert collated.gate_loss_mask.tolist() == [
        [True, True],
        [True, False],
        [True, False],
    ]
    assert collated.value_loss_mask is not None
    assert collated.value_loss_mask.tolist() == [
        [True, True],
        [True, True],
        [True, False],
    ]


def test_returns_and_gae_distinguish_terminal_from_time_limit() -> None:
    rewards = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    terminated = torch.tensor([[False, False], [True, False]])
    truncated = torch.tensor([[False, False], [False, True]])
    truncation_value = torch.tensor([[0.0, 0.0], [0.0, 10.0]])

    returns = discounted_returns(
        rewards,
        terminated,
        truncated,
        gamma=0.5,
        truncation_bootstrap_value=truncation_value,
    )

    torch.testing.assert_close(returns[:, 0], torch.tensor([2.0, 2.0]))
    torch.testing.assert_close(returns[:, 1], torch.tensor([4.5, 7.0]))
    advantages, targets = generalized_advantage_estimate(
        rewards,
        torch.zeros_like(rewards),
        torch.tensor([[0.0, 0.0], [99.0, 10.0]]),
        terminated,
        truncated,
        gamma=0.5,
        gae_lambda=1.0,
    )
    torch.testing.assert_close(advantages[:, 0], torch.tensor([2.0, 2.0]))
    torch.testing.assert_close(advantages[:, 1], torch.tensor([4.5, 7.0]))
    torch.testing.assert_close(targets, advantages)


def test_failure_cutoff_drops_complete_last_ten_seconds() -> None:
    ticks = torch.tensor([[100], [105], [110], [115]])
    mask = trusted_decision_mask(ticks, torch.tensor([315]))
    assert mask[:, 0].tolist() == [True, True, True, False]


def test_expert_preflight_counts_and_rejects_overflow_window() -> None:
    actions = tuple(
        TimedExpertActionV4(
            source_tick=130 + index,
            source_index=index,
            action=ActionV1.play(0, index % 4, (1 + index, 2), card_id=100 + index),
        )
        for index in range(3)
    )
    report = screen_expert_timeline(actions)
    assert report.overflow_window_count == 1
    assert report.maximum_actions_in_window == 3
    assert not report.accepted


def test_expert_alignment_uses_candidate_uid_grid_and_delay() -> None:
    batch = _batch(batch_size=1)
    card_id = int(batch.candidates.native_visible_card_id[0, 0])
    play = ActionV1(
        owner=0,
        kind=ActionKind.PLAY_CARD,
        hand_slot=0,
        card_id=card_id,
        target_kind=TargetKind.GRID,
        target_grid=(3, 4),
    )
    ability = ActionV1(
        owner=0,
        kind=ActionKind.ACTIVATE_ABILITY,
        source_entity=501,
    )
    sequence = build_expert_action_batch(
        batch,
        ((
            TimedExpertActionV4(132, play, 0),
            TimedExpertActionV4(133, ability, 1),
        ),),
        decision_ticks_by_row=(130,),
        perspectives=(PerspectiveTransformV1(actor_owner=0),),
    )
    assert sequence.gate.item() == GATE_ACT
    assert sequence.micro_action_count.item() == 2
    assert sequence.candidate_uid.tolist() == [[101, 103]]
    assert sequence.target_cell.tolist() == [[4 * 18 + 3, -1]]
    assert sequence.delay_offset_bin.tolist() == [[2, 3]]


def test_dataset_preflight_keeps_true_side_and_natural_windows() -> None:
    payload = {
        "replay_tag": "abc",
        "events": [
            {
                "kind": "play_card",
                "side": "team",
                "replay_tick_20hz": 130,
                "source_fields": {"data_i": 1},
            },
            {
                "kind": "activate_ability",
                "side": "opponent",
                "replay_tick_20hz": 134,
                "source_fields": {"data_i": 1},
            },
        ],
    }
    report = screen_replay_payload(payload)
    assert report.accepted
    assert report.data_i == 1


def test_current_v4_contract_rejects_unimplemented_entity_target() -> None:
    model = _model().eval()
    batch = _batch(batch_size=1)
    modes = batch.candidates.target_mode.clone()
    modes[0, 0] = TARGET_ENTITY
    candidates = replace(batch.candidates, target_mode=modes)

    with pytest.raises(ValueError, match="only NONE and GRID"):
        model.act(replace(batch, candidates=candidates), model.initial_state(1))


def test_ppo_stored_rollout_reuses_behavior_observations() -> None:
    torch.manual_seed(5)
    model = _model().eval()
    batch = _batch(batch_size=1)
    actions = _wait_actions(1)
    starts = torch.tensor([[True]])
    initial_state = model.initial_state(1)
    with torch.no_grad():
        behavior = evaluate_recurrent_sequence(
            model,
            (batch,),
            (actions,),
            starts,
            initial_state=initial_state,
        )
    rollout = StoredRolloutV4(
        observations=(batch,),
        actions=(actions,),
        episode_start=starts,
        valid_mask=torch.ones(1, 1, dtype=torch.bool),
        behavior_log_prob=behavior.log_prob,
        behavior_value=behavior.value,
        next_value=torch.zeros(1, 1),
        rewards=torch.ones(1, 1),
        terminated=torch.ones(1, 1, dtype=torch.bool),
        truncated=torch.zeros(1, 1, dtype=torch.bool),
        initial_state=initial_state,
        gate_temperature=1.0,
        action_temperature=1.0,
        continue_temperature=1.0,
        policy_state_id="fixture-policy",
    )

    metrics = distributed_ppo_update_v4(
        model, torch.optim.AdamW(model.parameters(), lr=1e-4), rollout,
        epochs=1, lane_minibatch_size=1, sequence_chunk_steps=1,
        config=PPOConfigV4(entropy_coefficient=0.0), autocast_dtype=None,
    )

    assert metrics[0]["approx_kl"] == pytest.approx(0.0, abs=1e-6)
    assert torch.isfinite(torch.tensor(metrics[0]["total"]))


def test_ppo_rejects_continue_temperature_or_policy_identity_drift() -> None:
    model = _model().eval()
    batch = _batch(batch_size=1)
    actions = _wait_actions(1)
    starts = torch.tensor([[True]])
    initial_state = model.initial_state(1)
    with torch.no_grad():
        behavior = evaluate_recurrent_sequence(
            model,
            (batch,),
            (actions,),
            starts,
            initial_state=initial_state,
        )
    rollout = StoredRolloutV4(
        observations=(batch,),
        actions=(actions,),
        episode_start=starts,
        valid_mask=torch.ones(1, 1, dtype=torch.bool),
        behavior_log_prob=behavior.log_prob,
        behavior_value=behavior.value,
        next_value=torch.zeros(1, 1),
        rewards=torch.ones(1, 1),
        terminated=torch.ones(1, 1, dtype=torch.bool),
        truncated=torch.zeros(1, 1, dtype=torch.bool),
        initial_state=initial_state,
        gate_temperature=1.0,
        action_temperature=1.0,
        continue_temperature=1.0,
        policy_state_id="behavior-a",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    with pytest.raises(ValueError, match="rollout behavior identity differs"):
        distributed_ppo_update_v4(
            model, optimizer, rollout, epochs=1, lane_minibatch_size=1,
            sequence_chunk_steps=1, expected_policy_state_id="behavior-b",
        )
    model.set_ppo_continue_temperature(2.0)
    with pytest.raises(ValueError, match="continue temperature changed"):
        distributed_ppo_update_v4(
            model, optimizer, rollout, epochs=1, lane_minibatch_size=1,
            sequence_chunk_steps=1,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_stochastic_rollout_cuda_graph_advances_rng() -> None:
    model = _model().cuda().eval()
    model.set_ppo_gate_temperature(100.0)
    batch = _batch(batch_size=16)
    state = model.initial_state(16, device="cuda")
    starts = torch.zeros(16, dtype=torch.bool, device="cuda")
    graph = _StochasticCudaGraphV4(torch.device("cuda"))

    gates = []
    for _ in range(16):
        output = graph.run(model, batch, state, starts)
        gates.append(output.actions.gate.cpu().clone())
    torch.cuda.synchronize()

    assert torch.unique(torch.stack(gates), dim=0).shape[0] > 1


def test_distributed_il_forward_accepts_packed_sequence() -> None:
    torch.manual_seed(7)
    model = _model()
    packed = _batch(batch_size=1).to_storage(float_dtype=torch.float16)
    sequence = ILSequenceV4(
        observations=(packed,),
        actions=(_wait_actions(1),),
        episode_start=torch.tensor([[True]]),
        valid_mask=torch.ones(1, 1, dtype=torch.bool),
        returns=torch.zeros(1, 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    sequence = sequence.to("cpu")
    evaluation = _DistributedImitationModule(model)(
        sequence, model.initial_state(1), validate=False, preencode_observations=True,
    )
    losses = imitation_loss(sequence, evaluation)
    losses.total.backward()
    optimizer.step()

    assert torch.isfinite(losses.total)
    assert evaluation.final_state.hidden.dtype == torch.float32


def test_actor_critic_checkpoint_restores_value_head_and_temperatures(tmp_path) -> None:
    model = _model()
    model.set_ppo_gate_temperature(0.5)
    model.set_ppo_action_temperature(0.7)
    model.set_ppo_continue_temperature(2.0)
    value_name, value_parameter = next(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("value_head")
    )
    expected_value = value_parameter.detach().clone()
    path = tmp_path / "actor-critic.pt"
    save_actor_critic_checkpoint(
        path,
        model,
        training_stage="il",
        gamma_per_decision=0.99,
    )
    with torch.no_grad():
        value_parameter.add_(1.0)
    model.set_ppo_gate_temperature(1.0)
    model.set_ppo_action_temperature(1.0)
    model.set_ppo_continue_temperature(1.0)

    payload = load_actor_critic_checkpoint(path, model, map_location="cpu")

    restored = dict(model.named_parameters())[value_name]
    torch.testing.assert_close(restored, expected_value)
    assert model.ppo_gate_temperature == pytest.approx(0.5)
    assert model.ppo_action_temperature == pytest.approx(0.7)
    assert model.ppo_continue_temperature == pytest.approx(2.0)
    assert payload["training_stage"] == "il"


def test_il_checkpoint_loads_into_a_ppo_update(tmp_path) -> None:
    torch.manual_seed(9)
    batch = _batch(batch_size=1).to_storage(float_dtype=torch.float16)
    expert = _one_action(batch_size=1)
    sequence = ILSequenceV4(
        observations=(batch,),
        actions=(expert,),
        episode_start=torch.ones(1, 1, dtype=torch.bool),
        valid_mask=torch.ones(1, 1, dtype=torch.bool),
        returns=torch.ones(1, 1),
    )
    il_model = _model()
    il_optimizer = torch.optim.AdamW(il_model.parameters(), lr=1e-4)
    sequence = sequence.to("cpu")
    il_evaluation = _DistributedImitationModule(il_model)(
        sequence, il_model.initial_state(1), validate=False, preencode_observations=True,
    )
    il_loss = imitation_loss(sequence, il_evaluation)
    il_loss.total.backward()
    il_optimizer.step()
    assert il_loss.value_count.item() == 1
    assert il_loss.candidate_count.item() == 1

    checkpoint_path = tmp_path / "il.pt"
    checkpoint_id = save_actor_critic_checkpoint(
        checkpoint_path,
        il_model,
        optimizer=il_optimizer,
        update_step=1,
        training_stage="il",
        gamma_per_decision=0.99,
    )
    ppo_model = _model()
    payload = load_actor_critic_checkpoint(
        checkpoint_path,
        ppo_model,
        map_location="cpu",
    )
    assert payload["checkpoint_id"] == checkpoint_id

    model_batch = batch.to_model_input("cpu")
    initial_state = ppo_model.initial_state(1)
    starts = torch.ones(1, 1, dtype=torch.bool)
    with torch.no_grad():
        behavior = evaluate_recurrent_sequence(
            ppo_model,
            (model_batch,),
            (expert,),
            starts,
            initial_state=initial_state,
        )
    rollout = StoredRolloutV4(
        observations=(batch,),
        actions=(expert,),
        episode_start=starts,
        valid_mask=torch.ones(1, 1, dtype=torch.bool),
        behavior_log_prob=behavior.log_prob,
        behavior_value=behavior.value,
        next_value=torch.zeros(1, 1),
        rewards=torch.ones(1, 1),
        terminated=torch.ones(1, 1, dtype=torch.bool),
        truncated=torch.zeros(1, 1, dtype=torch.bool),
        initial_state=initial_state,
        gate_temperature=ppo_model.ppo_gate_temperature,
        action_temperature=ppo_model.ppo_action_temperature,
        continue_temperature=ppo_model.ppo_continue_temperature,
        policy_state_id=checkpoint_id,
    )
    ppo_optimizer = torch.optim.AdamW(ppo_model.parameters(), lr=1e-4)
    value_parameter = next(
        parameter
        for name, parameter in ppo_model.named_parameters()
        if name.startswith("value_head")
    )
    before = value_parameter.detach().clone()

    metrics = distributed_ppo_update_v4(
        ppo_model,
        ppo_optimizer,
        rollout,
        epochs=1,
        lane_minibatch_size=1,
        sequence_chunk_steps=1,
        validate=False,
    )

    assert len(metrics) == 1
    assert metrics[0]["valid_count"] == 1
    assert not torch.equal(value_parameter.detach(), before)


def test_world_one_chunked_distributed_ppo_updates_value_head() -> None:
    torch.manual_seed(91)
    model = _model()
    batch = _batch(batch_size=1).to_storage(float_dtype=torch.float16)
    action = _one_action(batch_size=1)
    initial_state = model.initial_state(1)
    starts = torch.ones(1, 1, dtype=torch.bool)
    with torch.no_grad():
        behavior = evaluate_recurrent_sequence(
            model,
            (batch.to_model_input("cpu"),),
            (action,),
            starts,
            initial_state=initial_state,
        )
    rollout = StoredRolloutV4(
        observations=(batch,),
        actions=(action,),
        episode_start=starts,
        valid_mask=torch.ones(1, 1, dtype=torch.bool),
        behavior_log_prob=behavior.log_prob,
        behavior_value=behavior.value,
        next_value=torch.zeros(1, 1),
        rewards=torch.ones(1, 1),
        terminated=torch.ones(1, 1, dtype=torch.bool),
        truncated=torch.zeros(1, 1, dtype=torch.bool),
        initial_state=initial_state,
        gate_temperature=model.ppo_gate_temperature,
        action_temperature=model.ppo_action_temperature,
        continue_temperature=model.ppo_continue_temperature,
        policy_state_id="distributed-behavior",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    value_parameter = next(
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("value_head")
    )
    before = value_parameter.detach().clone()

    metrics = distributed_ppo_update_v4(
        model,
        optimizer,
        rollout,
        epochs=1,
        lane_minibatch_size=1,
        sequence_chunk_steps=1,
        expected_policy_state_id="distributed-behavior",
        validate=False,
    )

    assert len(metrics) == 1
    assert metrics[0]["valid_count"] == 1
    assert not torch.equal(value_parameter.detach(), before)


def test_partial_distributed_minibatches_select_only_current_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(93)
    model = _model().eval()
    batch = _batch(batch_size=2).to_storage(float_dtype=torch.float16)
    actions = _wait_actions(2)
    starts = torch.ones(1, 2, dtype=torch.bool)
    initial_state = model.initial_state(2)
    with torch.no_grad():
        behavior = evaluate_recurrent_sequence(
            model,
            (batch.to_model_input("cpu"),),
            (actions,),
            starts,
            initial_state=initial_state,
        )
    rollout = StoredRolloutV4(
        observations=(batch,),
        actions=(actions,),
        episode_start=starts,
        valid_mask=torch.ones(1, 2, dtype=torch.bool),
        behavior_log_prob=behavior.log_prob,
        behavior_value=behavior.value,
        next_value=torch.zeros(1, 2),
        rewards=torch.ones(1, 2),
        terminated=torch.ones(1, 2, dtype=torch.bool),
        truncated=torch.zeros(1, 2, dtype=torch.bool),
        initial_state=initial_state,
        gate_temperature=model.ppo_gate_temperature,
        action_temperature=model.ppo_action_temperature,
        continue_temperature=model.ppo_continue_temperature,
        policy_state_id="distributed-lazy-lanes",
    )

    def reject_whole_rollout_copy(
        _self: StoredRolloutV4,
        _indices: torch.Tensor,
    ) -> StoredRolloutV4:
        raise AssertionError("partial minibatch copied the whole rollout")

    monkeypatch.setattr(
        StoredRolloutV4,
        "select_lanes",
        reject_whole_rollout_copy,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    metrics = distributed_ppo_update_v4(
        model,
        optimizer,
        rollout,
        epochs=1,
        lane_minibatch_size=1,
        sequence_chunk_steps=1,
        config=PPOConfigV4(entropy_coefficient=0.0, target_kl=None),
        expected_policy_state_id="distributed-lazy-lanes",
        validate=False,
    )

    assert len(metrics) == 2
    assert [row["valid_count"] for row in metrics] == [1.0, 1.0]


def test_distributed_ppo_microbatches_preserve_one_optimizer_update() -> None:
    torch.manual_seed(94)
    full_model = _model().eval()
    micro_model = _model().eval()
    micro_model.load_state_dict(full_model.state_dict(), strict=True)
    batch = _batch(batch_size=2).to_storage(float_dtype=torch.float16)
    actions = _wait_actions(2)
    starts = torch.ones(1, 2, dtype=torch.bool)
    initial_state = full_model.initial_state(2)
    with torch.no_grad():
        behavior = evaluate_recurrent_sequence(
            full_model,
            (batch.to_model_input("cpu"),),
            (actions,),
            starts,
            initial_state=initial_state,
        )
    rollout = StoredRolloutV4(
        observations=(batch,),
        actions=(actions,),
        episode_start=starts,
        valid_mask=torch.ones(1, 2, dtype=torch.bool),
        behavior_log_prob=behavior.log_prob,
        behavior_value=behavior.value,
        next_value=torch.zeros(1, 2),
        rewards=torch.tensor([[1.0, -0.25]]),
        terminated=torch.ones(1, 2, dtype=torch.bool),
        truncated=torch.zeros(1, 2, dtype=torch.bool),
        initial_state=initial_state,
        gate_temperature=full_model.ppo_gate_temperature,
        action_temperature=full_model.ppo_action_temperature,
        continue_temperature=full_model.ppo_continue_temperature,
        policy_state_id="distributed-microbatch-equivalence",
    )
    config = PPOConfigV4(
        entropy_coefficient=0.0,
        max_grad_norm=None,
        target_kl=None,
    )
    full_optimizer = torch.optim.AdamW(full_model.parameters(), lr=1e-4)
    micro_optimizer = torch.optim.AdamW(micro_model.parameters(), lr=1e-4)

    full_metrics = distributed_ppo_update_v4(
        full_model,
        full_optimizer,
        rollout,
        epochs=1,
        lane_minibatch_size=2,
        lane_microbatch_size=2,
        sequence_chunk_steps=1,
        config=config,
        autocast_dtype=None,
        expected_policy_state_id="distributed-microbatch-equivalence",
        validate=False,
    )
    micro_metrics = distributed_ppo_update_v4(
        micro_model,
        micro_optimizer,
        rollout,
        epochs=1,
        lane_minibatch_size=2,
        lane_microbatch_size=1,
        sequence_chunk_steps=1,
        config=config,
        autocast_dtype=None,
        expected_policy_state_id="distributed-microbatch-equivalence",
        validate=False,
    )

    assert len(full_metrics) == len(micro_metrics) == 1
    for key in (
        "total",
        "policy",
        "value",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "valid_count",
    ):
        assert micro_metrics[0][key] == pytest.approx(
            full_metrics[0][key],
            rel=5e-4,
            abs=5e-5,
        )
    for full_parameter, micro_parameter in zip(
        full_model.parameters(),
        micro_model.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(
            micro_parameter,
            full_parameter,
            rtol=5e-4,
            atol=5e-5,
        )


def test_lane_permutation_balances_valid_frames_across_minibatches() -> None:
    valid_mask = torch.arange(8)[:, None] < torch.tensor([8, 7, 6, 5, 4, 3])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(17)

    permutation = _balanced_lane_permutation(
        valid_mask,
        minibatch_size=2,
        generator=generator,
    )

    assert sorted(permutation.tolist()) == list(range(6))
    lengths = valid_mask.sum(dim=0)
    loads = [
        int(lengths.index_select(0, permutation[start : start + 2]).sum())
        for start in range(0, 6, 2)
    ]
    assert loads == [11, 11, 11]


def test_packed_action_transfer_preserves_storage_and_wire_rows() -> None:
    actions = _one_action(batch_size=2)

    stored, packed = _pack_action_batch_to_cpu(actions, 2)

    _assert_tensor_tree_equal(stored, actions.to_storage("cpu", float_dtype=None))
    unpacked = _actions_from_packed((3, 7), packed)
    assert tuple(_action_to_wire(unpacked[row]) for row in (3, 7)) == tuple(
        _action_to_wire(stored.narrow_batch(row, 1)) for row in range(2)
    )
    assert not torch.is_inference(stored.gate)


def test_row_packed_tensor_record_preserves_select_and_scatter() -> None:
    batch = _batch(batch_size=3).to_storage("cpu", float_dtype=torch.float16)

    packed = _RowPackedTensorRecordV4(batch)
    selected = packed.index_select(torch.tensor([2, 0], dtype=torch.long))
    destination = packed.empty_rows(4)
    destination.index_copy_(torch.tensor([1, 3], dtype=torch.long), selected)

    _assert_tensor_tree_equal(packed.record, batch)
    _assert_tensor_tree_equal(
        selected.record,
        batch.index_select(torch.tensor([2, 0], dtype=torch.long)),
    )
    _assert_tensor_tree_equal(
        destination.narrow_rows(1, 1).record,
        batch.narrow_batch(2, 1),
    )
    _assert_tensor_tree_equal(
        destination.narrow_rows(3, 1).record,
        batch.narrow_batch(0, 1),
    )


def test_rollout_cuda_graph_cache_survives_parameter_updates() -> None:
    model = _model()
    device = torch.device("cpu")

    first = _persistent_rollout_graph_v4(model, device, 16)
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    second = _persistent_rollout_graph_v4(model, device, 16)

    assert second is first
    assert _persistent_rollout_graph_v4(model, device, 24) is not first


def test_frozen_rollout_cuda_graph_cache_falls_back_on_new_capacity() -> None:
    model = _model()
    device = torch.device("cpu")
    cached = _persistent_rollout_graph_v4(model, device, 16)
    model.__dict__["_ppo_rollout_cuda_graph_cache_frozen_v4"] = True

    assert _persistent_rollout_graph_v4(model, device, 16) is cached
    assert _persistent_rollout_graph_v4(model, device, 24) is None


def test_chunked_distributed_ppo_drops_fully_padded_tail_lanes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(92)
    model = _model()
    batch = _batch(batch_size=2).to_storage(float_dtype=torch.float16)
    action = _one_action(batch_size=2)
    initial_state = model.initial_state(2)
    starts = torch.tensor([[True, True], [False, False]])
    with torch.no_grad():
        behavior = evaluate_recurrent_sequence(
            model,
            (batch.to_model_input("cpu"), batch.to_model_input("cpu")),
            (action, action),
            starts,
            initial_state=initial_state,
        )
    rollout = StoredRolloutV4(
        observations=(batch, batch),
        actions=(action, action),
        episode_start=starts,
        valid_mask=torch.tensor([[True, True], [True, False]]),
        behavior_log_prob=behavior.log_prob,
        behavior_value=behavior.value,
        next_value=torch.zeros(2, 2),
        rewards=torch.ones(2, 2),
        terminated=torch.tensor([[False, False], [True, False]]),
        truncated=torch.zeros(2, 2, dtype=torch.bool),
        initial_state=initial_state,
        gate_temperature=model.ppo_gate_temperature,
        action_temperature=model.ppo_action_temperature,
        continue_temperature=model.ppo_continue_temperature,
        policy_state_id="distributed-padded-tail",
    )
    encoded_batch_sizes: list[int] = []
    original_encode = model.encode_observation

    def record_encode(batch_to_encode: object, *args: object, **kwargs: object) -> object:
        encoded_batch_sizes.append(batch_to_encode.batch_size)  # type: ignore[attr-defined]
        return original_encode(batch_to_encode, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(model, "encode_observation", record_encode)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    metrics = distributed_ppo_update_v4(
        model,
        optimizer,
        rollout,
        epochs=1,
        lane_minibatch_size=2,
        sequence_chunk_steps=1,
        config=PPOConfigV4(entropy_coefficient=0.0),
        expected_policy_state_id="distributed-padded-tail",
        validate=False,
    )

    assert encoded_batch_sizes == [2, 1]
    assert metrics[0]["valid_count"] == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA autocast required")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_spatial_scatter_autocast_keeps_scatter_dtype(dtype: torch.dtype) -> None:
    config = ModelConfigV4()
    encoder = SpatialScatterEncoder(config).cuda().eval()
    child = torch.randn(1, 2, config.child_dim, device="cuda")
    position = torch.tensor([[[4.25, 8.5], [6.5, 9.25]]], device="cuda")
    extent = torch.cat((position, position), dim=-1)
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        result = encoder.scatter(
            child,
            position,
            extent,
            torch.zeros(1, 2, device="cuda"),
            torch.ones(1, 2, dtype=torch.bool, device="cuda"),
        )
    assert result.dtype == dtype
    assert torch.isfinite(result).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph required")
def test_deterministic_cuda_graph_matches_eager_across_variable_sets() -> None:
    torch.manual_seed(20260819)
    device = torch.device("cuda")
    model = _model().to(device).eval()
    first_batch = _batch(batch_size=1)
    initial_state = model.initial_state(1, device=device)
    first_start = torch.tensor([True], dtype=torch.bool, device=device)
    runner = _DeterministicCudaGraphV4(device)

    with torch.inference_mode():
        eager_first = model.act(
            first_batch.to_model_input(device),
            initial_state,
            episode_start=first_start,
            validate=False,
        )
        expected_first = _clone_tensor_tree(eager_first)
        graphed_first = runner.run(
            model,
            first_batch,
            initial_state,
            first_start,
        )
        torch.cuda.synchronize(device)
        _assert_tensor_tree_equal(graphed_first, expected_first)

        second_batch = _batch(batch_size=1)
        second_batch.match_scalars[:, 0] = 0.75
        second_batch = replace(
            second_batch,
            active_effects=type(second_batch.active_effects)(
                effect_vocab_id=torch.tensor([[1]]),
                parent_type=torch.tensor([[0]]),
                parent_index=torch.tensor([[0]]),
                source_owner_type=torch.tensor([[1]]),
                runtime_features=torch.zeros(
                    1,
                    1,
                    second_batch.active_effects.runtime_features.shape[-1],
                ),
                mask=torch.tensor([[True]]),
            ),
            relation_edges=type(second_batch.relation_edges)(
                source=torch.cat(
                    (
                        second_batch.relation_edges.source,
                        second_batch.relation_edges.source[:, :1],
                    ),
                    dim=1,
                ),
                target=torch.cat(
                    (
                        second_batch.relation_edges.target,
                        second_batch.relation_edges.target[:, :1],
                    ),
                    dim=1,
                ),
                relation_type=torch.cat(
                    (
                        second_batch.relation_edges.relation_type,
                        second_batch.relation_edges.relation_type[:, :1],
                    ),
                    dim=1,
                ),
                mask=torch.ones(1, 4, dtype=torch.bool),
            ),
        )
        second_start = torch.tensor([False], dtype=torch.bool, device=device)
        eager_second = model.act(
            second_batch.to_model_input(device),
            eager_first.next_state,
            episode_start=second_start,
            validate=False,
        )
        expected_second = _clone_tensor_tree(eager_second)
        graphed_second = runner.run(
            model,
            second_batch,
            eager_first.next_state,
            second_start,
        )
        torch.cuda.synchronize(device)
        _assert_tensor_tree_equal(graphed_second, expected_second)
