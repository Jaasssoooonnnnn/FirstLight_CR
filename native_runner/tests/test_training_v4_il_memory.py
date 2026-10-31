"""IL training-step memory controls: lane micro-batching and activation checkpointing.

Both must leave the math unchanged.  Micro-batching re-associates the
per-head sums (and the weight-gradient reductions over the batch), so it is
compared at fp32 reduction-order tolerance; checkpointing recomputes the same
kernels and is compared at 1e-6.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from native_runner.tests.test_training_v4_model import _model
from native_runner.tests.test_training_v4_trainer_options import _write_cache, single_process_group  # noqa: F401
from native_runner.training.v4 import components
from native_runner.training.v4.cache import load_il_cache_shards
from native_runner.training.v4.components import ActivationCheckpointingV4
from native_runner.training.v4.imitation import ILLossWeightsV4, ILSequenceV4, imitation_loss
from native_runner.training.v4.learning import evaluate_recurrent_sequence
from native_runner.training.v4.model import UniversalCardPolicyV4
from native_runner.training.v4.train_imitation_cache import (
    _LOSS_HEADS,
    ReplayWeightsV1,
    _DistributedImitationModule,
    _masked_chunk,
    build_training_plan,
    chunk_gradients,
    chunk_loss_scale,
    lane_slices,
    loss_populations,
    slice_sequence_lanes,
)


TIME_STEPS = 3


def _fixture(tmp_path: Path, *, weighted: bool):
    index_path, tags_path = _write_cache(tmp_path, contract={})
    plans, _metadata = build_training_plan([index_path], tags_path, manifest_workers=1)
    shards = load_il_cache_shards([plan.summary_path for plan in plans])
    # Unequal lane weights exercise the weighted populations (frame_weight).
    replay_weights = ReplayWeightsV1({("a", 0): 2.0, ("b", None): 0.5, ("c", 1): 0.0}) if weighted else None
    max_length = max(length for plan in plans for length in plan.sequence_lengths)
    chunk_count = -(-max_length // TIME_STEPS)
    chunks = [
        _masked_chunk(
            shards,
            plans,
            start=index * TIME_STEPS,
            time_steps=TIME_STEPS,
            device=torch.device("cpu"),
            dummy=False,
            replay_weights=replay_weights,
        )
        for index in range(chunk_count)
    ]
    return chunks, sum(plan.owner_frames for plan in plans)


def _train(
    chunks: list[tuple[ILSequenceV4, int]],
    group_owner_frames: int,
    *,
    micro_batches: int,
    fused: bool,
    checkpointing: bool | tuple[str, ...] = False,
    optimizer_name: str = "adamw",
):
    """Mimic one rank of ``train_imitation_cache.main`` (DDP over a world-1 gloo group)."""

    torch.manual_seed(3)
    policy = _model()
    policy.set_activation_checkpointing(checkpointing)
    # The fixture catalogs leave some encoders unused; production DDP keeps the default.
    ddp = DistributedDataParallel(_DistributedImitationModule(policy), find_unused_parameters=True)
    optimizer = (
        torch.optim.AdamW(policy.parameters(), lr=3e-5, weight_decay=1e-2)
        if optimizer_name == "adamw"
        else torch.optim.SGD(policy.parameters(), lr=1e-3)
    )
    state = policy.initial_state(chunks[0][0].batch_size)
    rows = []
    for sequence, frames in chunks:
        scale = chunk_loss_scale(
            "group-expected", chunk_frames=frames, group_owner_frames=group_owner_frames, chunk_count=len(chunks)
        )
        optimizer.zero_grad(set_to_none=True)
        state, metrics = chunk_gradients(
            ddp,
            sequence,
            state,
            weights=ILLossWeightsV4(),
            world_size=1,
            loss_scale=scale,
            micro_batches=micro_batches,
            validate=True,
            preencode_observations=True,
            fused_sequence=fused,
        )
        gradients = {name: parameter.grad.clone() for name, parameter in policy.named_parameters() if parameter.grad is not None}
        nn.utils.clip_grad_norm_(ddp.parameters(), 1.0)
        before = {name: parameter.detach().clone() for name, parameter in policy.named_parameters()}
        optimizer.step()
        updates = {name: parameter.detach() - before[name] for name, parameter in policy.named_parameters()}
        rows.append({"metrics": metrics, "gradients": gradients, "updates": updates, "state": state})
    return rows


def _max_difference(expected: dict[str, torch.Tensor], actual: dict[str, torch.Tensor]) -> float:
    assert expected.keys() == actual.keys()
    return max(float((expected[name] - actual[name]).abs().max()) for name in expected)


def _assert_metrics_close(expected: dict[str, float], actual: dict[str, float]) -> None:
    assert expected.keys() == actual.keys()
    for name, value in expected.items():
        assert actual[name] == pytest.approx(value, rel=1e-6, abs=1e-6), name


# ---------------------------------------------------------------------------
# Lane slicing and populations
# ---------------------------------------------------------------------------


def test_lane_slices_are_contiguous_and_balanced() -> None:
    assert lane_slices(6, 1) == ((0, 6),)
    assert lane_slices(6, 4) == ((0, 2), (2, 4), (4, 5), (5, 6))
    assert lane_slices(512, 2) == ((0, 256), (256, 512))
    assert lane_slices(2, 4) == ((0, 1), (1, 2))  # a dummy tail rank has two lanes
    with pytest.raises(ValueError):
        lane_slices(0, 2)
    with pytest.raises(ValueError):
        lane_slices(4, 0)


def test_slice_sequence_lanes_selects_every_field(tmp_path: Path) -> None:
    (sequence, _frames), *_rest = _fixture(tmp_path, weighted=True)[0]
    part = slice_sequence_lanes(sequence, 2, 5)
    part.validate()
    assert (part.time_steps, part.batch_size) == (TIME_STEPS, 3)
    for name in ("episode_start", "valid_mask", "returns", "gate_loss_mask", "value_loss_mask", "frame_weight"):
        full = getattr(sequence, name)
        assert full is not None
        assert torch.equal(getattr(part, name), full[:, 2:5])
    assert torch.equal(part.observations[1].match_scalars, sequence.observations[1].match_scalars[2:5])
    assert torch.equal(part.actions[2].candidate_index, sequence.actions[2].candidate_index[2:5])
    with pytest.raises(ValueError):
        slice_sequence_lanes(sequence, 3, 3)


@pytest.mark.parametrize("weighted", [False, True])
def test_loss_populations_match_imitation_loss(tmp_path: Path, weighted: bool) -> None:
    chunks, _owner_frames = _fixture(tmp_path, weighted=weighted)
    torch.manual_seed(3)
    model = _model().eval()
    for sequence, _frames in chunks:
        with torch.no_grad():
            evaluation = evaluate_recurrent_sequence(model, sequence.observations, sequence.actions, sequence.episode_start)
        losses = imitation_loss(sequence, evaluation)
        counts, weighted_counts = loss_populations(sequence)
        for index, (_loss, count_name, weighted_name) in enumerate(_LOSS_HEADS):
            assert float(counts[index]) == float(getattr(losses, count_name)), count_name
            assert float(weighted_counts[index]) == pytest.approx(float(getattr(losses, weighted_name))), weighted_name


# ---------------------------------------------------------------------------
# Micro-batched gradient accumulation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("fused", [True, False])
def test_micro_batches_match_single_pass(
    tmp_path: Path, single_process_group: None, fused: bool, weighted: bool  # noqa: F811
) -> None:
    """Two TBPTT chunks with carried LSTM state: metrics, state, gradients and updates agree."""

    chunks, owner_frames = _fixture(tmp_path, weighted=weighted)
    assert len(chunks) == 2 and chunks[0][0].batch_size == 6
    # SGD keeps each update proportional to its gradient, so chunk 2 runs on
    # parameters that agree to ~1e-9 and the carried state can be compared.
    reference = _train(chunks, owner_frames, micro_batches=1, fused=fused, optimizer_name="sgd")
    for micro_batches in (2, 4):
        rows = _train(chunks, owner_frames, micro_batches=micro_batches, fused=fused, optimizer_name="sgd")
        for expected, actual in zip(reference, rows, strict=True):
            _assert_metrics_close(expected["metrics"], actual["metrics"])
            for name in ("hidden", "cell"):
                torch.testing.assert_close(
                    getattr(actual["state"], name), getattr(expected["state"], name), rtol=0.0, atol=1e-6
                )
            # Same gradient up to fp32 reduction order: within 1e-6 of the largest entry.
            largest = max(float(gradient.abs().max()) for gradient in expected["gradients"].values())
            assert _max_difference(expected["gradients"], actual["gradients"]) <= 1e-6 * max(1.0, largest)
            assert _max_difference(expected["updates"], actual["updates"]) <= 1e-6


@pytest.mark.parametrize("weighted", [False, True])
def test_micro_batches_adamw_step_matches_single_pass(
    tmp_path: Path, single_process_group: None, weighted: bool  # noqa: F811
) -> None:
    """One step with the trainer's AdamW (lr 3e-5, weight decay 1e-2)."""

    chunks, owner_frames = _fixture(tmp_path, weighted=weighted)
    reference = _train(chunks[:1], owner_frames, micro_batches=1, fused=True)
    for micro_batches in (2, 4):
        rows = _train(chunks[:1], owner_frames, micro_batches=micro_batches, fused=True)
        _assert_metrics_close(reference[0]["metrics"], rows[0]["metrics"])
        # Adam divides each entry by its own magnitude, so a parameter whose
        # true gradient is zero (the candidate-key bias, which the pointer
        # softmax ignores, has |grad| ~5e-9 of pure rounding noise) turns that
        # noise into an O(lr) step.  Bound the difference by lr / 10.
        assert _max_difference(reference[0]["updates"], rows[0]["updates"]) <= 3e-6


@pytest.mark.parametrize("micro_batches", [1, 2])
def test_chunk_gradients_reject_non_finite_loss(
    tmp_path: Path, single_process_group: None, micro_batches: int  # noqa: F811
) -> None:
    chunks, _owner_frames = _fixture(tmp_path, weighted=False)
    sequence, _frames = chunks[0]
    torch.manual_seed(3)
    policy = _model()
    with torch.no_grad():
        policy.value_head[-1].bias.fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="non-finite distributed IL loss at chunk=0"):
        chunk_gradients(
            _DistributedImitationModule(policy),
            sequence,
            policy.initial_state(6),
            weights=ILLossWeightsV4(),
            world_size=1,
            micro_batches=micro_batches,
            context="chunk=0",
        )


# ---------------------------------------------------------------------------
# Activation checkpointing
# ---------------------------------------------------------------------------


def _forward_backward(model: UniversalCardPolicyV4, sequence: ILSequenceV4, *, fused: bool):
    model.zero_grad(set_to_none=True)
    evaluation = evaluate_recurrent_sequence(
        model,
        sequence.observations,
        sequence.actions,
        sequence.episode_start,
        preencode_observations=True,
        fused=fused,
    )
    losses = imitation_loss(sequence, evaluation)
    losses.total.backward()
    gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters() if parameter.grad is not None}
    return evaluation, losses.total.detach(), gradients


@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize(
    "scopes", [("pair-bias",), ("attention",), ("spatial",), ("attention", "spatial"), ("pair-bias", "spatial"), True]
)
def test_activation_checkpointing_matches_plain_forward_and_backward(
    tmp_path: Path, fused: bool, scopes: bool | tuple[str, ...]
) -> None:
    (sequence, _frames), *_rest = _fixture(tmp_path, weighted=True)[0]
    torch.manual_seed(3)
    model = _model()
    plain_evaluation, plain_loss, plain_gradients = _forward_backward(model, sequence, fused=fused)
    model.set_activation_checkpointing(scopes)
    assert model.activation_checkpointing
    evaluation, loss, gradients = _forward_backward(model, sequence, fused=fused)
    torch.testing.assert_close(loss, plain_loss, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(evaluation.value, plain_evaluation.value, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(evaluation.log_prob, plain_evaluation.log_prob, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(evaluation.final_state.hidden, plain_evaluation.final_state.hidden, rtol=0.0, atol=1e-6)
    assert gradients.keys() == plain_gradients.keys()
    assert _max_difference(plain_gradients, gradients) <= 1e-6


def test_activation_checkpointing_combines_with_micro_batches(
    tmp_path: Path, single_process_group: None  # noqa: F811
) -> None:
    chunks, owner_frames = _fixture(tmp_path, weighted=True)
    plain = _train(chunks, owner_frames, micro_batches=2, fused=True)
    checkpointed = _train(chunks, owner_frames, micro_batches=2, fused=True, checkpointing=True)
    for expected, actual in zip(plain, checkpointed, strict=True):
        _assert_metrics_close(expected["metrics"], actual["metrics"])
        assert _max_difference(expected["gradients"], actual["gradients"]) <= 1e-6
        assert _max_difference(expected["updates"], actual["updates"]) <= 1e-6


def test_activation_checkpointing_is_inert_without_autograd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (sequence, _frames), *_rest = _fixture(tmp_path, weighted=False)[0]
    torch.manual_seed(3)
    model = _model()
    model.set_activation_checkpointing(("pair-bias", "attention", "spatial"))
    calls = []
    original = components.checkpointed

    def counting(function, *arguments):
        calls.append(function)
        return original(function, *arguments)

    monkeypatch.setattr(components, "checkpointed", counting)
    for context in (torch.no_grad, torch.inference_mode):
        with context():
            evaluate_recurrent_sequence(
                model, sequence.observations, sequence.actions, sequence.episode_start, preencode_observations=True
            )
    assert calls == []
    _forward_backward(model, sequence, fused=True)
    # transformer_layers attention blocks + spatial_res_blocks residual blocks.
    assert len(calls) == model.config.transformer_layers + model.config.spatial_res_blocks


def test_activation_checkpointing_is_inert_during_cuda_graph_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with torch.enable_grad():
        assert not ActivationCheckpointingV4.recording(SimpleNamespace(is_cuda=True))  # type: ignore[arg-type]
        assert ActivationCheckpointingV4.recording(torch.zeros(1))
    with torch.no_grad():
        assert not ActivationCheckpointingV4.recording(torch.zeros(1))


def test_activation_checkpointing_is_runtime_only() -> None:
    torch.manual_seed(3)
    model = _model()
    keys = tuple(model.state_dict())
    assert model.activation_checkpointing == ()
    model.set_activation_checkpointing(True)
    assert model.activation_checkpointing == ("attention", "spatial")
    assert all(block.checkpointing is model._activation_checkpointing for block in model.relation_transformer.blocks)
    assert model.spatial_encoder.checkpointing is model._activation_checkpointing
    assert tuple(model.state_dict()) == keys
    model.set_activation_checkpointing(["pair-bias"])
    assert model.activation_checkpointing == ("pair-bias",)
    model.set_activation_checkpointing(False)
    assert model.activation_checkpointing == ()
    with pytest.raises(ValueError, match="scopes"):
        model.set_activation_checkpointing(("ffn",))
    with pytest.raises(TypeError):
        model.set_activation_checkpointing("attention")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Held-out evaluation micro-batches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fused", [True, False])
def test_heldout_evaluation_micro_batches_give_the_same_rows(tmp_path: Path, fused: bool) -> None:
    from native_runner.tests.test_training_v4_evaluate_imitation_cache import (
        _evaluate,
        _save_checkpoint,
        _without_timing,
        _write_fixture,
    )

    index_path, tags_path, _lanes = _write_fixture(tmp_path)
    checkpoints = [tmp_path / "checkpoint-a.pt", tmp_path / "checkpoint-b.pt"]
    for seed, path in enumerate(checkpoints):
        _save_checkpoint(path, 11 + seed, update_step=7 + seed)
    # Two shards per batch: six lanes, so four slices are uneven (2, 2, 1, 1).
    reference = _evaluate(index_path, tags_path, checkpoints, shards_per_batch=2, fused_sequence=fused)
    for micro_batches in (2, 4):
        rows = _evaluate(
            index_path, tags_path, checkpoints, shards_per_batch=2, fused_sequence=fused, micro_batches=micro_batches
        )
        for expected, actual in zip(reference, rows, strict=True):
            expected = _without_timing(expected)
            actual = _without_timing(actual)
            assert actual.pop("micro_batches") == micro_batches
            assert expected.pop("micro_batches") == 1
            assert actual.keys() == expected.keys()
            for key, value in expected.items():
                if isinstance(value, float):
                    assert actual[key] == pytest.approx(value, rel=1e-6, abs=1e-6), key
                else:
                    assert actual[key] == value, key
