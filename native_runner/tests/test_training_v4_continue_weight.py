"""Opt-in class weight for SECOND labels in the IL ``continue`` head.

``ILLossWeightsV4.continue_second_weight`` (CLI ``--continue-second-weight``)
multiplies the per-label weight of ACT frames whose continue label is SECOND
(``micro_action_count == 2``) inside the continue head's weighted mean,
numerator and denominator, composed multiplicatively with ``frame_weight``:

    continue = sum(act * fw * cw * nll) / sum(act * fw * cw),  cw = W on SECOND, 1 on STOP

``continue_count`` stays the unweighted ACT population.
"""

from __future__ import annotations

from dataclasses import fields, replace
import inspect
from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from native_runner.tests.test_training_v4_il_memory import _fixture
from native_runner.tests.test_training_v4_model import _model
from native_runner.tests.test_training_v4_trainer_options import (  # noqa: F401
    _evaluate_chunk,
    _whole_chunk,
    single_process_group,
)
from native_runner.training.v4 import train_imitation_cache
from native_runner.training.v4.imitation import (
    ILLossWeightsV4,
    ILSequenceV4,
    _weighted_masked_class_mean,
    _weighted_masked_mean,
    continue_label_class_weight,
    imitation_loss,
)
from native_runner.training.v4.learning import evaluate_recurrent_sequence
from native_runner.training.v4.tensors import GATE_ACT
from native_runner.training.v4.train_imitation_cache import (
    _LOSS_HEADS,
    _DistributedImitationModule,
    _distributed_loss,
    _nonnegative_finite_float,
    chunk_gradients,
    chunk_loss_scale,
    loss_populations,
)


_OTHER_HEADS = ("gate", "candidate", "target", "delay", "value")


def _labels(sequence: ILSequenceV4) -> tuple[torch.Tensor, torch.Tensor]:
    """``(act_mask, second_mask)`` over ``[T,B]``."""

    gates = torch.stack([item.gate for item in sequence.actions])
    micro_counts = torch.stack([item.micro_action_count for item in sequence.actions])
    act = sequence.valid_mask & (gates == GATE_ACT)
    return act, act & (micro_counts == 2)


def _lane_weights(sequence: ILSequenceV4, lane_weights: list[float]) -> ILSequenceV4:
    frame_weight = torch.tensor(lane_weights, dtype=torch.float32)[None, :].expand(sequence.valid_mask.shape)
    return replace(sequence, frame_weight=frame_weight.contiguous())


def _evaluate_with_grad(sequence: ILSequenceV4, *, seed: int = 3):
    torch.manual_seed(seed)
    model = _model().eval()
    evaluation = evaluate_recurrent_sequence(
        model,
        sequence.observations,
        sequence.actions,
        sequence.episode_start,
        gate_temperature=1.0,
        action_temperature=1.0,
        continue_temperature=1.0,
    )
    return model, evaluation


def _gradients(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.grad.clone() for name, parameter in model.named_parameters() if parameter.grad is not None}


# ---------------------------------------------------------------------------
# Loss formula
# ---------------------------------------------------------------------------


def test_class_weight_tensor_and_validation() -> None:
    micro_counts = torch.tensor([[0, 1], [2, 1]])
    assert continue_label_class_weight(micro_counts, 1.0) is None
    torch.testing.assert_close(
        continue_label_class_weight(micro_counts, 7.5), torch.tensor([[1.0, 1.0], [7.5, 1.0]]), rtol=0.0, atol=0.0
    )
    for bad in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            continue_label_class_weight(micro_counts, bad)
        with pytest.raises(ValueError):
            ILLossWeightsV4(continue_second_weight=bad)


def test_class_weighted_mean_by_hand() -> None:
    """Tiny synthetic [T=2,B=3] batch with known labels and frame weights."""

    values = torch.tensor([[0.5, 2.0, 1.0], [4.0, 0.25, 3.0]])
    act = torch.tensor([[True, True, False], [True, True, True]])
    micro_counts = torch.tensor([[1, 2, 0], [2, 1, 1]])  # SECOND at (0,1) and (1,0)
    frame_weight = torch.tensor([[1.0, 0.5, 9.0], [2.0, 1.0, 0.0]])
    w = 10.0
    class_weight = continue_label_class_weight(micro_counts, w)

    mean, count, weighted = _weighted_masked_class_mean(values, act, None, class_weight)
    # STOP: 0.5, 0.25, 3.0 (weight 1); SECOND: 2.0, 4.0 (weight 10).
    assert float(count) == 5.0
    assert float(weighted) == pytest.approx(3.0 + 2.0 * w)
    assert float(mean) == pytest.approx((0.5 + 0.25 + 3.0 + w * (2.0 + 4.0)) / (3.0 + 2.0 * w), rel=1e-7)

    mean, count, weighted = _weighted_masked_class_mean(values, act, frame_weight, class_weight)
    numerator = 1.0 * 0.5 + 0.5 * w * 2.0 + 2.0 * w * 4.0 + 1.0 * 0.25 + 0.0 * 3.0
    denominator = 1.0 + 0.5 * w + 2.0 * w + 1.0 + 0.0
    assert float(count) == 5.0
    assert float(weighted) == pytest.approx(denominator)
    assert float(mean) == pytest.approx(numerator / denominator, rel=1e-7)

    # W = 0 with only SECOND labels: an empty weighted population gives 0, not NaN.
    only_second = torch.tensor([[False, True, False], [True, False, False]])
    mean, _count, weighted = _weighted_masked_class_mean(
        values, only_second, None, continue_label_class_weight(micro_counts, 0.0)
    )
    assert float(weighted) == 0.0 and float(mean) == 0.0


@pytest.mark.parametrize("replay_weighted", [False, True])
def test_weight_one_is_bit_identical_to_unweighted_loss_and_gradients(tmp_path: Path, replay_weighted: bool) -> None:
    sequence, _frames = _whole_chunk(tmp_path)
    if replay_weighted:
        sequence = _lane_weights(sequence, [2.0, 0.5, 1.0, 0.0, 1.5, 3.0])
    act, second = _labels(sequence)
    assert int(second.sum()) == 2  # the fixture has two GRID_FIRST + BUILDING_SECOND frames

    model, evaluation = _evaluate_with_grad(sequence)
    losses = imitation_loss(sequence, evaluation, weights=ILLossWeightsV4(continue_second_weight=1.0))
    # The pre-change continue term, verbatim.
    old_continue, old_count, old_weighted = _weighted_masked_mean(
        -evaluation.components.continue_log_prob, act, sequence.frame_weight
    )
    assert torch.equal(losses.continue_action, old_continue)
    assert torch.equal(losses.continue_count, old_count)
    assert torch.equal(losses.continue_weighted_count, old_weighted)
    old_total = (
        losses.gate + losses.candidate + losses.target + losses.delay + old_continue + losses.value
    )
    assert torch.equal(losses.total, old_total)
    default = imitation_loss(sequence, evaluation)
    for item in fields(default):
        assert torch.equal(getattr(default, item.name), getattr(losses, item.name)), item.name

    losses.total.backward(retain_graph=True)
    new_gradients = _gradients(model)
    model.zero_grad(set_to_none=True)
    old_total.backward()
    old_gradients = _gradients(model)
    assert new_gradients.keys() == old_gradients.keys()
    for name, gradient in new_gradients.items():
        assert torch.equal(gradient, old_gradients[name]), name

    assert float(losses.continue_second_count) == 2.0
    fw = torch.ones_like(sequence.valid_mask, dtype=torch.float32) if sequence.frame_weight is None else sequence.frame_weight
    assert float(losses.continue_second_weighted_count) == pytest.approx(float((fw * second).sum()))


@pytest.mark.parametrize("w", [0.0, 0.25, 50.0])
def test_continue_term_equals_hand_weighted_mean(tmp_path: Path, w: float) -> None:
    sequence, _frames = _whole_chunk(tmp_path)
    evaluation = _evaluate_chunk(sequence)
    act, second = _labels(sequence)
    plain = imitation_loss(sequence, evaluation)
    losses = imitation_loss(sequence, evaluation, weights=ILLossWeightsV4(continue_second_weight=w))

    nll = -evaluation.components.continue_log_prob
    weight = act.to(torch.float32) * torch.where(second, w, 1.0)
    expected = (nll * weight).sum() / weight.sum()
    torch.testing.assert_close(losses.continue_action, expected, rtol=1e-6, atol=0.0)
    assert float(losses.continue_weighted_count) == pytest.approx(float(act.sum() - second.sum()) + w * 2.0)
    # The logged unweighted population is unchanged by the class weight.
    assert torch.equal(losses.continue_count, plain.continue_count)
    assert float(losses.continue_second_count) == 2.0
    assert float(losses.continue_second_weighted_count) == pytest.approx(2.0 * w)
    # Every other head is untouched, bit for bit.
    for head in _OTHER_HEADS:
        assert torch.equal(getattr(losses, head), getattr(plain, head)), head
    torch.testing.assert_close(
        losses.total, plain.total - plain.continue_action + losses.continue_action, rtol=1e-6, atol=1e-6
    )
    if w != 1.0:
        assert float(losses.continue_action) != pytest.approx(float(plain.continue_action), rel=1e-6)


def test_class_weight_composes_multiplicatively_with_frame_weight(tmp_path: Path) -> None:
    sequence, _frames = _whole_chunk(tmp_path)
    evaluation = _evaluate_chunk(sequence)
    act, second = _labels(sequence)
    w = 12.0
    weighted = _lane_weights(sequence, [2.0, 0.5, 0.25, 1.0, 3.0, 1.0])
    losses = imitation_loss(weighted, evaluation, weights=ILLossWeightsV4(continue_second_weight=w))

    nll = -evaluation.components.continue_log_prob
    total_weight = act.to(torch.float32) * weighted.frame_weight * torch.where(second, w, 1.0)
    torch.testing.assert_close(losses.continue_action, (nll * total_weight).sum() / total_weight.sum(), rtol=1e-6, atol=0.0)
    assert float(losses.continue_weighted_count) == pytest.approx(float(total_weight.sum()))
    assert float(losses.continue_second_weighted_count) == pytest.approx(float((total_weight * second).sum()))

    # Same as folding W into the frame weight of SECOND frames (continue is per frame).
    folded = replace(weighted, frame_weight=weighted.frame_weight * torch.where(second, w, 1.0))
    folded_losses = imitation_loss(folded, evaluation)
    torch.testing.assert_close(losses.continue_action, folded_losses.continue_action, rtol=1e-6, atol=0.0)
    torch.testing.assert_close(losses.continue_weighted_count, folded_losses.continue_weighted_count)

    # A zero replay weight removes a SECOND frame whatever W is.
    second_lanes = second.any(dim=0).nonzero().flatten().tolist()
    lane_weights = [1.0] * sequence.batch_size
    for lane in second_lanes:
        lane_weights[lane] = 0.0
    dropped = _lane_weights(sequence, lane_weights)
    for value in (1.0, w):
        dropped_losses = imitation_loss(dropped, evaluation, weights=ILLossWeightsV4(continue_second_weight=value))
        assert float(dropped_losses.continue_second_weighted_count) == 0.0
    torch.testing.assert_close(
        imitation_loss(dropped, evaluation, weights=ILLossWeightsV4(continue_second_weight=w)).continue_action,
        imitation_loss(dropped, evaluation).continue_action,
        rtol=0.0,
        atol=0.0,
    )


# ---------------------------------------------------------------------------
# Trainer: populations, distributed loss, micro-batches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("weighted", [False, True])
def test_loss_populations_follow_the_class_weight(tmp_path: Path, weighted: bool) -> None:
    chunks, _owner_frames = _fixture(tmp_path, weighted=weighted)
    torch.manual_seed(3)
    model = _model().eval()
    w = 30.0
    for sequence, _frames in chunks:
        with torch.no_grad():
            evaluation = evaluate_recurrent_sequence(model, sequence.observations, sequence.actions, sequence.episode_start)
        losses = imitation_loss(sequence, evaluation, weights=ILLossWeightsV4(continue_second_weight=w))
        counts, weighted_counts = loss_populations(sequence, continue_second_weight=w)
        for index, (_loss, count_name, weighted_name) in enumerate(_LOSS_HEADS):
            assert float(counts[index]) == float(getattr(losses, count_name)), count_name
            assert float(weighted_counts[index]) == pytest.approx(float(getattr(losses, weighted_name))), weighted_name


def test_distributed_loss_under_class_weight(tmp_path: Path, single_process_group: None) -> None:  # noqa: F811
    sequence, _frames = _whole_chunk(tmp_path)
    evaluation = _evaluate_chunk(sequence)
    weighted = _lane_weights(sequence, [2.0, 0.5, 1.0, 1.0, 1.0, 1.0])
    plain_weights = ILLossWeightsV4()
    plain_total, plain_metrics = _distributed_loss(imitation_loss(weighted, evaluation), plain_weights, world_size=1)

    weights = ILLossWeightsV4(continue_second_weight=20.0)
    losses = imitation_loss(weighted, evaluation, weights=weights)
    total, metrics = _distributed_loss(losses, weights, world_size=1)
    torch.testing.assert_close(total, losses.total)
    assert metrics["continue_action"] == pytest.approx(float(losses.continue_action), rel=1e-6)
    assert metrics["continue_count"] == plain_metrics["continue_count"]
    assert metrics["continue_second_count"] == plain_metrics["continue_second_count"] == 2.0
    assert metrics["continue_second_weighted_fraction"] == pytest.approx(
        float(losses.continue_second_weighted_count) / float(losses.continue_weighted_count)
    )
    assert metrics["continue_second_weighted_fraction"] > plain_metrics["continue_second_weighted_fraction"] > 0.0
    for head in _OTHER_HEADS:
        assert metrics[head] == plain_metrics[head], head
    assert not torch.equal(total, plain_total)


def _train(chunks, owner_frames: int, *, micro_batches: int, fused: bool, weights: ILLossWeightsV4):
    torch.manual_seed(3)
    policy = _model()
    ddp = DistributedDataParallel(_DistributedImitationModule(policy), find_unused_parameters=True)
    optimizer = torch.optim.SGD(policy.parameters(), lr=1e-3)
    state = policy.initial_state(chunks[0][0].batch_size)
    rows = []
    for sequence, frames in chunks:
        scale = chunk_loss_scale(
            "group-expected", chunk_frames=frames, group_owner_frames=owner_frames, chunk_count=len(chunks)
        )
        optimizer.zero_grad(set_to_none=True)
        state, metrics = chunk_gradients(
            ddp,
            sequence,
            state,
            weights=weights,
            world_size=1,
            loss_scale=scale,
            micro_batches=micro_batches,
            validate=True,
            preencode_observations=True,
            fused_sequence=fused,
        )
        rows.append({"metrics": metrics, "gradients": _gradients(policy), "state": state})
        optimizer.step()
    return rows


@pytest.mark.parametrize("fused", [True, False])
def test_micro_batches_agree_under_class_weight(
    tmp_path: Path, single_process_group: None, fused: bool  # noqa: F811
) -> None:
    chunks, owner_frames = _fixture(tmp_path, weighted=True)
    weights = ILLossWeightsV4(continue_second_weight=25.0)
    assert sum(int(_labels(sequence)[1].sum()) for sequence, _frames in chunks) == 2
    reference = _train(chunks, owner_frames, micro_batches=1, fused=fused, weights=weights)
    plain = _train(chunks, owner_frames, micro_batches=1, fused=fused, weights=ILLossWeightsV4())
    rows = _train(chunks, owner_frames, micro_batches=2, fused=fused, weights=weights)
    saw_second = False
    for expected, actual, unweighted in zip(reference, rows, plain, strict=True):
        assert expected["metrics"].keys() == actual["metrics"].keys()
        for name, value in expected["metrics"].items():
            assert actual["metrics"][name] == pytest.approx(value, rel=1e-6, abs=1e-6), name
        for name in ("hidden", "cell"):
            torch.testing.assert_close(getattr(actual["state"], name), getattr(expected["state"], name), rtol=0.0, atol=1e-6)
        largest = max(float(gradient.abs().max()) for gradient in expected["gradients"].values())
        difference = max(
            float((expected["gradients"][name] - actual["gradients"][name]).abs().max()) for name in expected["gradients"]
        )
        assert difference <= 1e-6 * max(1.0, largest)
        assert expected["metrics"]["continue_count"] == unweighted["metrics"]["continue_count"]
        if expected["metrics"]["continue_second_count"] > 0.0:
            saw_second = True
            assert expected["metrics"]["continue_action"] != pytest.approx(
                unweighted["metrics"]["continue_action"], rel=1e-6
            )
    assert saw_second


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_nonnegative_finite_float_parser() -> None:
    import argparse

    assert _nonnegative_finite_float("1") == 1.0
    assert _nonnegative_finite_float("0") == 0.0
    assert _nonnegative_finite_float("1427.5") == 1427.5
    for bad in ("-0.5", "nan", "inf", "-inf", "abc", ""):
        with pytest.raises(argparse.ArgumentTypeError):
            _nonnegative_finite_float(bad)


_REQUIRED = ["--index", "index.json", "--train-tags", "tags.txt", "--output-dir", "out"]


def test_cli_help_lists_continue_second_weight(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["train_imitation_cache", "--help"])
    with pytest.raises(SystemExit) as raised:
        train_imitation_cache.main()
    assert raised.value.code == 0
    assert "--continue-second-weight" in capsys.readouterr().out


@pytest.mark.parametrize("bad", ["-1", "nan", "inf", "abc"])
def test_cli_rejects_invalid_continue_second_weight(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], bad: str
) -> None:
    monkeypatch.setattr(sys, "argv", ["train_imitation_cache", *_REQUIRED, f"--continue-second-weight={bad}"])
    with pytest.raises(SystemExit) as raised:
        train_imitation_cache.main()
    assert raised.value.code == 2
    assert "--continue-second-weight" in capsys.readouterr().err


def test_cli_passes_weight_to_loss_startup_and_checkpoint() -> None:
    source = inspect.getsource(train_imitation_cache.main)
    assert "ILLossWeightsV4(delay=args.delay_weight, continue_second_weight=args.continue_second_weight)" in source
    # Logged in the startup event and the checkpoint extras.
    assert source.count('"continue_second_weight": args.continue_second_weight') == 2
