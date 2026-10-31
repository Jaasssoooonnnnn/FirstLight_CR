"""Fused T*B imitation evaluation equals the stepped TBPTT decoder.

The fused path (``evaluate_recurrent_sequence(..., fused=True)``) runs one
``torch._VF.lstm`` over the chunk and decodes only ACT rows.  Every component
matches the stepped path except WAIT rows' unmasked delay log-probs / legal
mask and downstream argmax, which the fused path leaves at the decoder's
initial zeros / -1; those entries are masked by every consumer, so losses,
gradients and held-out statistics are compared without exception.
"""

from __future__ import annotations

from dataclasses import fields, replace
import gc
import os
from pathlib import Path

import pytest
import torch

from native_runner.tests.test_training_v4_cache import _result, _sequence
from native_runner.tests.test_training_v4_evaluate_imitation_cache import (
    _evaluate as _evaluate_heldout,
    _save_checkpoint,
    _without_timing,
    _write_fixture,
)
from native_runner.tests.test_training_v4_model import _model
from native_runner.training.v4.cache import load_il_cache_shard, write_il_cache_shard
from native_runner.training.v4.evaluate_imitation_cache import chunk_statistics
from native_runner.training.v4.imitation import (
    ILSequenceV4,
    collate_padded_il_sequences,
    imitation_loss,
    slice_il_sequence,
)
from native_runner.training.v4.learning import (
    RecurrentEvaluationV4,
    _evaluate_fused_sequence,
    evaluate_recurrent_sequence,
)
from native_runner.training.v4.model import UniversalCardPolicyV4
from native_runner.training.v4.tensors import GATE_ACT, RecurrentPolicyStateV4, concatenate_padded_tensor_records
from native_runner.training.v4.train_imitation_cache import _DistributedImitationModule

ATOL = 1e-5
GRAD_RTOL = 1e-4
# WAIT-row fields the fused path leaves at the dense decoder's initial values.
WAIT_ROW_UNMASKED_COMPONENTS = ("delay_offset_log_probs", "delay_offset_legal_mask")
WAIT_ROW_UNMASKED_ARGMAX = ("candidate_argmax", "target_argmax", "delay_argmax", "continue_argmax")

REAL_CACHE = Path(
    os.environ.get(
        "CR_FUSED_IL_REAL_CACHE",
        "real-il-cache-not-configured",
    )
)


def _evaluate(
    model: UniversalCardPolicyV4,
    sequence: ILSequenceV4,
    state: RecurrentPolicyStateV4 | None,
    *,
    fused: bool,
    collect_argmax: bool = True,
    validate: bool = True,
) -> RecurrentEvaluationV4:
    return evaluate_recurrent_sequence(
        model,
        sequence.observations,
        sequence.actions,
        sequence.episode_start,
        initial_state=state,
        gate_temperature=1.0,
        action_temperature=1.0,
        continue_temperature=1.0,
        validate=validate,
        preencode_observations=True,
        collect_argmax=collect_argmax,
        fused=fused,
    )


def _close(actual: torch.Tensor, expected: torch.Tensor, label: str, *, atol: float = ATOL) -> float:
    assert actual.shape == expected.shape, label
    assert actual.dtype == expected.dtype, label
    if not actual.is_floating_point():
        assert torch.equal(actual, expected), label
        return 0.0
    torch.testing.assert_close(actual, expected, rtol=ATOL, atol=atol, msg=label)
    return float((actual.double() - expected.double()).abs().max()) if actual.numel() else 0.0


def assert_evaluations_match(
    stepped: RecurrentEvaluationV4, fused: RecurrentEvaluationV4, sequence: ILSequenceV4
) -> dict[str, float]:
    """Compare every output; return the max absolute difference per field."""

    report: dict[str, float] = {}
    for name in ("log_prob", "entropy", "value"):
        report[name] = _close(getattr(fused, name), getattr(stepped, name), name)
    for name in ("hidden", "cell"):
        report[f"final_state.{name}"] = _close(
            getattr(fused.final_state, name), getattr(stepped.final_state, name), name
        )
    gates = torch.stack([item.gate for item in sequence.actions])
    act = gates == GATE_ACT
    for item in fields(stepped.components):
        expected = getattr(stepped.components, item.name)
        actual = getattr(fused.components, item.name)
        label = f"components.{item.name}"
        if item.name in WAIT_ROW_UNMASKED_COMPONENTS:
            report[label] = _close(actual[act], expected[act], label)
            assert not torch.any(actual[~act].to(torch.bool)), f"{label} must stay zero on WAIT rows"
        else:
            report[label] = _close(actual, expected, label)
    assert (stepped.argmax is None) == (fused.argmax is None)
    if stepped.argmax is not None and fused.argmax is not None:
        for item in fields(stepped.argmax):
            expected = getattr(stepped.argmax, item.name)
            actual = getattr(fused.argmax, item.name)
            label = f"argmax.{item.name}"
            if item.name in WAIT_ROW_UNMASKED_ARGMAX:
                _close(actual[act], expected[act], label)
                initial = 0 if item.name == "continue_argmax" else -1
                assert torch.all(actual[~act] == initial), f"{label} must stay {initial} on WAIT rows"
            else:
                _close(actual, expected, label)
    return report


def assert_losses_match(stepped: RecurrentEvaluationV4, fused: RecurrentEvaluationV4, sequence: ILSequenceV4) -> None:
    stepped_loss = imitation_loss(sequence, stepped, gate_act_weight=8.0, delay_neighbor_weight=0.2)
    fused_loss = imitation_loss(sequence, fused, gate_act_weight=8.0, delay_neighbor_weight=0.2)
    for item in fields(stepped_loss):
        _close(getattr(fused_loss, item.name).detach(), getattr(stepped_loss, item.name).detach(), item.name)


def _loss_gradients(
    model: UniversalCardPolicyV4, sequence: ILSequenceV4, state: RecurrentPolicyStateV4 | None, *, fused: bool
) -> tuple[RecurrentEvaluationV4, dict[str, torch.Tensor | None]]:
    model.zero_grad(set_to_none=True)
    evaluation = _evaluate(model, sequence, state, fused=fused, collect_argmax=False)
    losses = imitation_loss(sequence, evaluation, gate_act_weight=8.0, delay_neighbor_weight=0.2)
    losses.total.backward()
    gradients = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }
    model.zero_grad(set_to_none=True)
    return evaluation, gradients


def assert_gradients_match(
    model: UniversalCardPolicyV4, sequence: ILSequenceV4, state: RecurrentPolicyStateV4 | None
) -> float:
    """Return the worst per-parameter relative gradient error."""

    _stepped, stepped_grads = _loss_gradients(model, sequence, state, fused=False)
    _fused, fused_grads = _loss_gradients(model, sequence, state, fused=True)
    worst = 0.0
    total_difference = 0.0
    total_norm = 0.0
    for name, expected in stepped_grads.items():
        actual = fused_grads[name]
        assert (actual is None) == (expected is None), name
        if actual is None or expected is None:
            continue
        difference = float((actual.double() - expected.double()).norm())
        norm = float(expected.double().norm())
        total_difference += difference**2
        total_norm += norm**2
        # Relative to the parameter's own gradient, with an absolute floor for
        # gradients that are zero (or rounding noise) in both paths.
        assert difference <= GRAD_RTOL * norm + 1e-7, f"{name}: |diff| {difference:.3e} vs |grad| {norm:.3e}"
        if norm > 1e-6:
            worst = max(worst, difference / norm)
    assert total_difference**0.5 <= GRAD_RTOL * total_norm**0.5
    return worst


def _mixed_sequence(tmp_path: Path) -> ILSequenceV4:
    """Tiny written-and-reloaded cache with WAIT, one- and two-micro ACT rows."""

    index_path, _tags_path, _lanes = _write_fixture(tmp_path)
    shards = [load_il_cache_shard(index_path.parent / f"shard-{index:05d}.cril.json") for index in (0, 1)]
    stored = [shards[0].sequence(0), shards[0].sequence(1)]
    stored += [shards[1].sequence(index) for index in range(4)]
    return collate_padded_il_sequences(stored).to("cpu")


def _wait_sequence(tmp_path: Path) -> ILSequenceV4:
    """The all-WAIT tiny cache of test_training_v4_cache."""

    data_path = tmp_path / "wait-shard-00000.cril.zst"
    write_il_cache_shard(
        data_path,
        (_result("w", _sequence("w:owner-0", [0.1, 0.2, 0.3]), _sequence("w:owner-1", [-0.1, 0.0, 0.5])),),
        shard_id="wait-shard-00000",
        contract={},
    )
    shard = load_il_cache_shard(data_path.with_suffix(".json"))
    return collate_padded_il_sequences([shard.sequence(0), shard.sequence(1)]).to("cpu")


def _small_model(seed: int) -> UniversalCardPolicyV4:
    torch.manual_seed(seed)
    return _model()


def test_fused_il_matches_stepped_on_mixed_tiny_cache(tmp_path: Path) -> None:
    model = _small_model(17).eval()
    sequence = _mixed_sequence(tmp_path)
    gates = torch.stack([item.gate for item in sequence.actions])
    counts = torch.stack([item.micro_action_count for item in sequence.actions])
    assert torch.any(gates == GATE_ACT) and torch.any(counts == 2) and torch.any(gates != GATE_ACT)

    with torch.no_grad():
        stepped = _evaluate(model, sequence, None, fused=False)
        fused = _evaluate(model, sequence, None, fused=True)
    assert_evaluations_match(stepped, fused, sequence)
    assert_losses_match(stepped, fused, sequence)
    statistics_stepped = chunk_statistics(sequence, stepped, board_width=18, owner_frames=sequence.time_steps)
    statistics_fused = chunk_statistics(sequence, fused, board_width=18, owner_frames=sequence.time_steps)
    torch.testing.assert_close(statistics_fused, statistics_stepped, rtol=ATOL, atol=ATOL)

    # A TBPTT continuation chunk: nonzero carried state and no reset at step 0.
    first = slice_il_sequence(sequence, 0, 2)
    second = slice_il_sequence(sequence, 2, sequence.time_steps)
    assert not torch.any(second.episode_start)
    with torch.no_grad():
        carried = _evaluate(model, first, None, fused=False).final_state
        stepped = _evaluate(model, second, carried, fused=False)
        fused = _evaluate(model, second, carried, fused=True)
    assert_evaluations_match(stepped, fused, second)
    assert_losses_match(stepped, fused, second)


@pytest.mark.parametrize("validate", (True, False))
def test_fused_il_gradients_match_stepped(tmp_path: Path, validate: bool) -> None:
    model = _small_model(23).train()
    sequence = _mixed_sequence(tmp_path)
    stepped = _evaluate(model, sequence, None, fused=False, validate=validate)
    fused = _evaluate(model, sequence, None, fused=True, validate=validate)
    assert_evaluations_match(stepped, fused, sequence)
    assert_losses_match(stepped, fused, sequence)
    assert_gradients_match(model, sequence, None)


@pytest.mark.parametrize(
    "ablation", ("ablate_lstm", "ablate_spatial_memory", "ablate_decoder_cross_attention")
)
def test_fused_il_matches_stepped_under_ablations(tmp_path: Path, ablation: str) -> None:
    from native_runner.tests.test_training_v4_ablations import _build
    from native_runner.training.v4.config import ModelConfigV4

    model = _build(ModelConfigV4(**{ablation: True}), seed=43).train()
    sequence = _mixed_sequence(tmp_path)
    stepped = _evaluate(model, sequence, None, fused=False)
    fused = _evaluate(model, sequence, None, fused=True)
    assert_evaluations_match(stepped, fused, sequence)
    assert_losses_match(stepped, fused, sequence)
    assert_gradients_match(model, sequence, None)


def test_fused_il_all_wait_cache_keeps_every_decoder_parameter_in_the_graph(tmp_path: Path) -> None:
    model = _small_model(29).train()
    sequence = _wait_sequence(tmp_path)
    assert not any(torch.any(item.gate == GATE_ACT) for item in sequence.actions)
    with torch.no_grad():
        stepped = _evaluate(model, sequence, None, fused=False)
        fused = _evaluate(model, sequence, None, fused=True)
    assert_evaluations_match(stepped, fused, sequence)
    assert_losses_match(stepped, fused, sequence)
    # Same parameters receive (zero) gradients as in the dense decoder, so a
    # DDP rank with an all-WAIT (or dummy) chunk still reduces every bucket.
    assert_gradients_match(model, sequence, None)


def test_fused_il_falls_back_to_stepped_on_mid_chunk_reset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _small_model(31).eval()
    sequence = _mixed_sequence(tmp_path)
    episode_start = sequence.episode_start.clone()
    episode_start[2, 1] = True
    reset = replace(sequence, episode_start=episode_start)

    with torch.no_grad():
        stepped = _evaluate(model, reset, None, fused=False)

        def forbidden(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("fused sequence evaluation must not run across a reset")

        monkeypatch.setattr(model, "evaluate_encoded_action_sequence", forbidden)
        fallback = _evaluate(model, reset, None, fused=True)
        monkeypatch.undo()
        # Without the fallback the fused kernel itself refuses the chunk.
        flat = concatenate_padded_tensor_records(tuple(reset.observations))
        with pytest.raises(ValueError, match="mid-lane resets"):
            _evaluate_fused_sequence(
                model,
                flat,  # type: ignore[arg-type]
                model.encode_observation(flat),  # type: ignore[arg-type]
                reset.actions,
                reset.episode_start,
                model.initial_state(reset.batch_size),
                gate_temperature=1.0,
                action_temperature=1.0,
                continue_temperature=1.0,
                validate=True,
                collect_argmax=True,
            )
    # The fallback is the stepped path itself: bit-identical outputs.
    for name in ("log_prob", "entropy", "value"):
        assert torch.equal(getattr(fallback, name), getattr(stepped, name))
    for item in fields(stepped.components):
        assert torch.equal(getattr(fallback.components, item.name), getattr(stepped.components, item.name))
    assert fallback.argmax is not None and stepped.argmax is not None
    for item in fields(stepped.argmax):
        assert torch.equal(getattr(fallback.argmax, item.name), getattr(stepped.argmax, item.name))


def test_fused_il_without_preencoding_uses_stepped_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _small_model(37).eval()
    sequence = _mixed_sequence(tmp_path)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("fused evaluation requires preencoded observations")

    monkeypatch.setattr(model, "evaluate_encoded_action_sequence", forbidden)
    with torch.no_grad():
        evaluation = evaluate_recurrent_sequence(
            model, sequence.observations, sequence.actions, sequence.episode_start, fused=True
        )
    assert tuple(evaluation.log_prob.shape) == (sequence.time_steps, sequence.batch_size)


def test_distributed_il_module_forwards_fused_flag(tmp_path: Path) -> None:
    model = _small_model(41).eval()
    sequence = _mixed_sequence(tmp_path)
    module = _DistributedImitationModule(model)
    with torch.no_grad():
        stepped = module(sequence, model.initial_state(sequence.batch_size), validate=True, preencode_observations=True)
        fused = module(
            sequence,
            model.initial_state(sequence.batch_size),
            validate=True,
            preencode_observations=True,
            fused_sequence=True,
        )
    assert stepped.argmax is None and fused.argmax is None
    assert_evaluations_match(stepped, fused, sequence)
    assert_losses_match(stepped, fused, sequence)


def test_heldout_evaluator_fused_rows_match_stepped_rows(tmp_path: Path) -> None:
    index_path, tags_path, _lanes = _write_fixture(tmp_path)
    checkpoint = tmp_path / "checkpoint.pt"
    _save_checkpoint(checkpoint, 13, update_step=3)
    rows = {}
    for fused in (False, True):
        (row,) = _evaluate_heldout(index_path, tags_path, [checkpoint], time_steps=3, fused_sequence=fused)
        rows[fused] = _without_timing(row)
    assert rows[True].keys() == rows[False].keys()
    for name, expected in rows[False].items():
        actual = rows[True][name]
        if isinstance(expected, float):
            assert actual == pytest.approx(expected, abs=ATOL), name
        else:
            assert actual == expected, name


def _real_plans(cache_dir: Path, *, shards: int, replays_per_shard: int) -> list[object]:
    from native_runner.training.v4 import train_imitation_cache as trainer
    from native_runner.training.v4.cache import load_il_cache_manifest

    plans = []
    for path in sorted(cache_dir.glob("part-*.cril.json"))[:shards]:
        manifest = load_il_cache_manifest(path)
        tags = {item["replay_tag"] for item in manifest["sequences"]}  # type: ignore[index, union-attr]
        units = sorted(trainer._units_from_manifest(manifest, tags), key=lambda unit: unit.sequence_indices)
        plans.append(trainer.ShardTrainingPlanV1(summary_path=path.resolve(), units=tuple(units[:replays_per_shard])))
    return plans


def _load_real_shards(plans: list[object]) -> tuple[list[object], tuple[object, ...]]:
    """Inflate shards one at a time, keeping as many as host memory allows.

    A 1.8 GB container transiently needs about three times its size while
    safetensors deserializes it, which can exceed a busy workstation's commit
    headroom.  That is an environment limit, not an equivalence failure: the
    test continues with the shards already loaded and skips only when not even
    one shard fits.
    """

    loaded = []
    for plan in plans:
        gc.collect()
        try:
            loaded.append(load_il_cache_shard(plan.summary_path))  # type: ignore[attr-defined]
        except BaseException as error:  # pyo3's PanicException derives from BaseException
            if not (isinstance(error, MemoryError) or type(error).__name__ == "PanicException"):
                raise
            if not loaded:
                pytest.skip(f"not enough host memory to inflate a real cache shard: {error!r}")
            break
    return plans[: len(loaded)], tuple(loaded)


@pytest.mark.skipif(not REAL_CACHE.is_dir(), reason="real IL cache shards are not available on this machine")
def test_fused_il_matches_stepped_on_real_shards_with_production_model() -> None:
    from native_runner.training.v4.factory import build_production_model_v4
    from native_runner.training.v4.train_imitation_cache import _masked_chunk

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    previous_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        plans, loaded = _load_real_shards(_real_plans(REAL_CACHE, shards=2, replays_per_shard=2))
        torch.manual_seed(0)
        model = build_production_model_v4(device=device).train()
        state: RecurrentPolicyStateV4 | None = None
        act_rows = two_micro_rows = 0
        for start in (0, 32, 64):
            sequence, _frames = _masked_chunk(
                loaded, plans, start=start, time_steps=32, device=device, dummy=False  # type: ignore[arg-type]
            )
            gates = torch.stack([item.gate for item in sequence.actions])
            counts = torch.stack([item.micro_action_count for item in sequence.actions])
            act_rows += int((gates == GATE_ACT).sum())
            two_micro_rows += int((counts == 2).sum())
            with torch.no_grad():
                stepped = _evaluate(model, sequence, state, fused=False)
                fused = _evaluate(model, sequence, state, fused=True)
            assert_evaluations_match(stepped, fused, sequence)
            assert_losses_match(stepped, fused, sequence)
            torch.testing.assert_close(
                chunk_statistics(sequence, fused, board_width=18, owner_frames=1),
                chunk_statistics(sequence, stepped, board_width=18, owner_frames=1),
                rtol=ATOL,
                atol=ATOL,
            )
            assert_gradients_match(model, sequence, state)
            state = stepped.final_state
        assert act_rows > 0
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous_tf32
