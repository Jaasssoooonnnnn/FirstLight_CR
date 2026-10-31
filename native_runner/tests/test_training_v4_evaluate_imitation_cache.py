from __future__ import annotations

from dataclasses import fields, replace
import json
import math
from pathlib import Path

import pytest
import torch

from native_runner.tests.test_training_v4_cache import _result
from native_runner.tests.test_training_v4_learning import _wait_actions
from native_runner.tests.test_training_v4_model import _batch, _model
from native_runner.training.v4.cache import load_il_cache_shard, write_il_cache_index, write_il_cache_shard
from native_runner.training.v4.checkpoint import load_actor_critic_checkpoint, save_actor_critic_checkpoint
from native_runner.training.v4 import model as model_module
from native_runner.training.v4.evaluate_imitation_cache import (
    _SUM_FIELDS,
    _SUM_INDEX,
    ENTROPY_HEADS,
    DistributionRecorderV1,
    balanced_partitions,
    evaluate_checkpoints,
    head_masks,
    summarize_statistics,
    top1_log_probs,
)
from native_runner.training.v4.imitation import ILSequenceV4, collate_padded_il_sequences, imitation_loss
from native_runner.training.v4.learning import evaluate_recurrent_sequence
from native_runner.training.v4.tensors import GATE_ACT, ActionSequenceV4
from native_runner.training.v4.train_imitation_cache import ReplayUnitV1, ShardTrainingPlanV1


def _act(*micro: tuple[int, int, int, int]) -> ActionSequenceV4:
    """Build one ACT row from ``(candidate_index, uid, target_cell, delay_bin)``."""

    result = _wait_actions(1)
    result.gate[0] = GATE_ACT
    result.micro_action_count[0] = len(micro)
    for step, (index, uid, cell, delay) in enumerate(micro):
        result.candidate_index[0, step] = index
        result.candidate_uid[0, step] = uid
        result.target_cell[0, step] = cell
        result.delay_offset_bin[0, step] = delay
    return result


WAIT = "wait"
GRID_ONE = (0, 101, 5 * 18 + 4, 1)
GRID_FIRST = (0, 101, 5 * 18 + 4, 0)
BUILDING_SECOND = (1, 102, 20 * 18 + 10, 2)
ABILITY = (2, 103, -1, 0)


def _action(spec: object) -> ActionSequenceV4:
    if spec == WAIT:
        return _wait_actions(1)
    return _act(*spec)  # type: ignore[misc]


def _lane(
    sequence_id: str,
    specs: list[object],
    *,
    valid: list[bool] | None = None,
    gate_loss: list[bool] | None = None,
) -> ILSequenceV4:
    steps = len(specs)
    observation = _batch(batch_size=1).to_storage(float_dtype=torch.float16)
    valid_mask = torch.tensor([[item] for item in (valid or [True] * steps)], dtype=torch.bool)
    return ILSequenceV4(
        observations=(observation,) * steps,
        actions=tuple(_action(spec) for spec in specs),
        episode_start=torch.tensor([[step == 0] for step in range(steps)], dtype=torch.bool),
        valid_mask=valid_mask,
        returns=torch.linspace(-0.5, 1.5, steps)[:, None],
        gate_loss_mask=(None if gate_loss is None else torch.tensor([[item] for item in gate_loss], dtype=torch.bool)),
        value_loss_mask=valid_mask.clone(),
        sequence_id=sequence_id,
    )


def _fixture_lanes() -> dict[str, tuple[ILSequenceV4, ILSequenceV4]]:
    return {
        "a": (
            _lane("a:owner-0", [WAIT, (GRID_ONE,), WAIT, (GRID_FIRST, BUILDING_SECOND), (ABILITY,)]),
            _lane("a:owner-1", [(GRID_ONE,), WAIT, WAIT], gate_loss=[True, False, True]),
        ),
        "b": (
            _lane("b:owner-0", [WAIT, (ABILITY,), (GRID_FIRST, BUILDING_SECOND), WAIT]),
            _lane("b:owner-1", [WAIT, (GRID_ONE,), WAIT, WAIT], valid=[True, True, True, False]),
        ),
        "c": (
            _lane("c:owner-0", [WAIT, (GRID_ONE,)]),
            _lane("c:owner-1", [WAIT, WAIT]),
        ),
    }


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, tuple[ILSequenceV4, ILSequenceV4]]]:
    lanes = _fixture_lanes()
    directory = tmp_path / "cache"
    directory.mkdir()
    summaries = []
    for shard_index, tags in enumerate((("a",), ("b", "c"))):
        data_path = directory / f"shard-{shard_index:05d}.cril.zst"
        summary = write_il_cache_shard(
            data_path,
            tuple(_result(tag, *lanes[tag]) for tag in tags),
            shard_id=f"shard-{shard_index:05d}",
            contract={},
        )
        summary["summary_file"] = data_path.with_suffix(".json").name
        summaries.append(summary)
    index_path = directory / "index.json"
    write_il_cache_index(index_path, summaries, contract={})
    tags_path = tmp_path / "validation-replay-tags.txt"
    tags_path.write_text("a\nb\n", encoding="utf-8")
    return index_path, tags_path, lanes


def _save_checkpoint(path: Path, seed: int, *, update_step: int) -> None:
    torch.manual_seed(seed)
    model = _model()
    save_actor_critic_checkpoint(
        path,
        model,
        update_step=update_step,
        training_stage="imitation",
        gamma_per_decision=0.99,
        extra={"next_epoch": 2},
    )


def _evaluate(index_path: Path, tags_path: Path, checkpoints: list[Path], **kwargs: object) -> list[dict[str, object]]:
    options: dict[str, object] = {
        "device": "cpu",
        "time_steps": 2,
        "shards_per_batch": 1,
        "load_workers": 1,
        "manifest_workers": 1,
        "autocast": "none",
        "validate": True,
        "model": _model(),
    }
    options.update(kwargs)
    return evaluate_checkpoints([index_path], tags_path, checkpoints, **options)  # type: ignore[arg-type]


_TIMING_FIELDS = {"checkpoint_eval_s", "cache_load_s", "elapsed_s"}


def _without_timing(row: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in row.items() if key not in _TIMING_FIELDS}


def test_heldout_evaluation_matches_unweighted_imitation_loss(tmp_path: Path) -> None:
    index_path, tags_path, lanes = _write_fixture(tmp_path)
    checkpoint = tmp_path / "checkpoint-a.pt"
    _save_checkpoint(checkpoint, 11, update_step=7)
    output = tmp_path / "eval" / "metrics.jsonl"
    output.parent.mkdir()

    (row,) = _evaluate(index_path, tags_path, [checkpoint], output_path=output)

    selected = [lane for tag in ("a", "b") for lane in lanes[tag]]
    assert row["checkpoint"] == str(checkpoint.resolve())
    assert row["update_step"] == 7
    assert row["training_stage"] == "imitation"
    assert row["next_epoch"] == 2
    assert row["replays"] == 2 and row["lanes"] == 4 and row["world_size"] == 1
    assert row["owner_frames"] == sum(lane.time_steps for lane in selected) == 16
    assert row["valid_frames"] == sum(int(lane.valid_mask.sum()) for lane in selected) == 15
    assert row["gate_count"] == 14
    assert row["gate_act_count"] == 7 and row["gate_wait_count"] == 7
    assert row["candidate_count"] == row["delay_count"] == 9
    assert row["target_count"] == 7
    assert row["continue_count"] == 7
    assert row["value_count"] == 15
    assert row["gate_act_rate_expert"] == pytest.approx(0.5)
    for name, value in row.items():
        if isinstance(value, float):
            assert math.isfinite(value), name
    for name in ("candidate_top1", "target_exact", "target_within1", "delay_top1", "continue_top1", "gate_act_top1"):
        assert 0.0 <= float(row[name]) <= 1.0  # type: ignore[arg-type]
    assert float(row["target_within1"]) >= float(row["target_exact"])  # type: ignore[arg-type]
    assert row["decision_nll"] == pytest.approx(float(row["candidate_nll"]) + float(row["target_nll"]))  # type: ignore[arg-type]
    written = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert written == [json.loads(json.dumps(row, sort_keys=True))]

    # Whole-sequence reference: one recurrent pass over every lane at once.
    model = _model()
    load_actor_critic_checkpoint(checkpoint, model, map_location="cpu")
    model.eval()
    shards = {
        tag: load_il_cache_shard(index_path.parent / f"shard-{index:05d}.cril.json")
        for index, tag in enumerate(("a", "b"))
    }
    stored = [shards[tag].sequence(owner) for tag in ("a", "b") for owner in (0, 1)]
    sequence = collate_padded_il_sequences(stored).to("cpu")
    with torch.no_grad():
        evaluation = evaluate_recurrent_sequence(
            model,
            sequence.observations,
            sequence.actions,
            sequence.episode_start,
            gate_temperature=1.0,
            action_temperature=1.0,
            continue_temperature=1.0,
        )
        losses = imitation_loss(sequence, evaluation, gate_act_weight=1.0, delay_neighbor_weight=0.0)
    assert losses.candidate_count.item() == row["candidate_count"]
    assert float(row["candidate_nll"]) == pytest.approx(losses.candidate.item(), abs=1e-5)  # type: ignore[arg-type]
    assert float(row["target_nll"]) == pytest.approx(losses.target.item(), abs=1e-5)  # type: ignore[arg-type]
    assert float(row["gate_nll_all"]) == pytest.approx(losses.gate.item(), abs=1e-5)  # type: ignore[arg-type]
    assert float(row["delay_nll"]) == pytest.approx(losses.delay.item(), abs=1e-5)  # type: ignore[arg-type]
    assert float(row["continue_nll"]) == pytest.approx(losses.continue_action.item(), abs=1e-5)  # type: ignore[arg-type]
    assert float(row["value_huber"]) == pytest.approx(losses.value.item(), abs=1e-5)  # type: ignore[arg-type]


def test_heldout_evaluation_is_deterministic_and_swaps_checkpoints(tmp_path: Path) -> None:
    index_path, tags_path, _lanes = _write_fixture(tmp_path)
    first = tmp_path / "checkpoint-a.pt"
    second = tmp_path / "checkpoint-b.pt"
    _save_checkpoint(first, 11, update_step=7)
    _save_checkpoint(second, 23, update_step=9)

    (alone_first,) = _evaluate(index_path, tags_path, [first])
    (again_first,) = _evaluate(index_path, tags_path, [first])
    (alone_second,) = _evaluate(index_path, tags_path, [second])
    joint = _evaluate(index_path, tags_path, [first, second], shards_per_batch=2, time_steps=3)

    assert _without_timing(alone_first) == _without_timing(again_first)
    assert alone_first["candidate_nll"] != alone_second["candidate_nll"]
    for alone, combined in zip((alone_first, alone_second), joint, strict=True):
        assert combined["checkpoint_id"] == alone["checkpoint_id"]
        for name, value in alone.items():
            if name in _TIMING_FIELDS or name in {"time_steps", "shards_per_batch"}:
                continue
            if isinstance(value, float):
                assert combined[name] == pytest.approx(value, abs=1e-5), name
            else:
                assert combined[name] == value, name


def test_heldout_evaluation_rejects_bad_arguments(tmp_path: Path) -> None:
    index_path, tags_path, _lanes = _write_fixture(tmp_path)
    checkpoint = tmp_path / "checkpoint-a.pt"
    _save_checkpoint(checkpoint, 11, update_step=7)

    with pytest.raises(ValueError, match="unique"):
        _evaluate(index_path, tags_path, [checkpoint, checkpoint])
    with pytest.raises(ValueError, match="positive"):
        _evaluate(index_path, tags_path, [checkpoint], time_steps=0)
    with pytest.raises(ValueError, match="autocast"):
        _evaluate(index_path, tags_path, [checkpoint], autocast="int8")


def test_balanced_partitions_assign_each_shard_once_by_owner_frames(tmp_path: Path) -> None:
    plans = tuple(
        ShardTrainingPlanV1(
            summary_path=tmp_path / f"{length}.json",
            units=(ReplayUnitV1(replay_tag=str(length), sequence_indices=(0, 1), sequence_lengths=(length, length)),),
        )
        for length in (9, 7, 5, 3, 1)
    )

    partitions = balanced_partitions(plans, 2)

    assert sorted(plan.summary_path for part in partitions for plan in part) == sorted(
        plan.summary_path for plan in plans
    )
    assert [sum(plan.owner_frames for plan in part) for part in partitions] == [26, 24]


def _two_action_batch() -> tuple[torch.nn.Module, object, ActionSequenceV4]:
    torch.manual_seed(5)
    model = _model().eval()
    batch = _batch(batch_size=1)
    return model, batch, _act(GRID_FIRST, BUILDING_SECOND)


def _assert_tree_equal(actual: object, expected: object) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert torch.equal(actual, expected)
        return
    for item in fields(expected):  # type: ignore[arg-type]
        _assert_tree_equal(getattr(actual, item.name), getattr(expected, item.name))


def test_collect_argmax_false_leaves_outputs_unchanged() -> None:
    model, batch, actions = _two_action_batch()
    state = model.initial_state(1)

    with torch.no_grad():
        plain = model.evaluate_actions(batch, state, actions, gate_temperature=1.0, action_temperature=1.0)
        collected = model.evaluate_actions(
            batch, state, actions, gate_temperature=1.0, action_temperature=1.0, collect_argmax=True
        )
        sequence_plain = evaluate_recurrent_sequence(
            model, (batch, batch), (actions, actions), torch.tensor([[True], [False]]), preencode_observations=True
        )
        sequence_collected = evaluate_recurrent_sequence(
            model,
            (batch, batch),
            (actions, actions),
            torch.tensor([[True], [False]]),
            preencode_observations=True,
            collect_argmax=True,
        )

    assert plain.argmax is None
    assert collected.argmax is not None
    for name in ("actions", "log_prob", "entropy", "value", "next_state", "action_components"):
        _assert_tree_equal(getattr(collected, name), getattr(plain, name))
    assert sequence_plain.argmax is None
    assert sequence_collected.argmax is not None
    assert tuple(sequence_collected.argmax.candidate_argmax.shape) == (2, 1, 2)
    for name in ("log_prob", "entropy", "value", "components", "final_state"):
        _assert_tree_equal(getattr(sequence_collected, name), getattr(sequence_plain, name))


def test_collected_argmax_matches_greedy_decode_under_its_own_prefix() -> None:
    model, batch, _actions = _two_action_batch()
    model.set_inference_delay_offset_max_ms(None)
    state = model.initial_state(1)

    with torch.no_grad():
        greedy_gate = model.act(batch, state).actions.gate
        greedy = model.act_after_preselected_act(batch, model.forward(batch, state)).actions
        forced = model.evaluate_actions(
            batch, state, greedy, gate_temperature=1.0, action_temperature=1.0, collect_argmax=True
        )

    argmax = forced.argmax
    assert argmax is not None
    assert torch.equal(argmax.gate_argmax, greedy_gate)
    assert torch.equal(argmax.continue_argmax, (greedy.micro_action_count == 2).to(torch.long))
    for step in range(int(greedy.micro_action_count.item())):
        assert torch.equal(argmax.candidate_argmax[:, step], greedy.candidate_index[:, step])
        assert torch.equal(argmax.delay_argmax[:, step], greedy.delay_offset_bin[:, step])
        if int(greedy.target_cell[0, step]) >= 0:
            assert torch.equal(argmax.target_argmax[:, step], greedy.target_cell[:, step])


# --- Illegal supervised labels -------------------------------------------------
#
# The fixture model has zero match scalars, so elixir never regenerates inside
# the delay window and its legality is the decision-tick rule of a legacy
# checkpoint.  Candidates: 0 (cost 3, GRID), 1 (cost 4, building GRID),
# 2 (ability, cost 1), 3 (cost 9).


def _observation(*, elixir: float = 10.0, blocked_cell: tuple[int, int] | None = None) -> object:
    batch = _batch(batch_size=1)
    placement = batch.candidates.placement.clone()
    if blocked_cell is not None:
        y, x = blocked_cell
        placement[0, 0, y, x] = False
    candidates = replace(
        batch.candidates,
        elixir=torch.tensor([elixir], dtype=batch.candidates.elixir.dtype),
        placement=placement,
    )
    return replace(batch, candidates=candidates).to_storage(float_dtype=torch.float16)


def _custom_lane(sequence_id: str, steps: list[tuple[object, object]]) -> ILSequenceV4:
    count = len(steps)
    valid_mask = torch.ones(count, 1, dtype=torch.bool)
    return ILSequenceV4(
        observations=tuple(observation for _spec, observation in steps),  # type: ignore[misc]
        actions=tuple(_action(spec) for spec, _observation in steps),
        episode_start=torch.tensor([[step == 0] for step in range(count)], dtype=torch.bool),
        valid_mask=valid_mask,
        returns=torch.linspace(-0.5, 1.5, count)[:, None],
        gate_loss_mask=None,
        value_loss_mask=valid_mask.clone(),
        sequence_id=sequence_id,
    )


def _illegal_steps() -> list[tuple[object, object]]:
    """One illegal label per head, built from the model's own legality rule."""

    return [
        (WAIT, _observation()),
        # Candidate 0 costs 3 > 2 elixir (candidate 2 stays legal, so the gate is legal).
        ((GRID_ONE,), _observation(elixir=2.0)),
        # Candidate 0 cannot be placed on the expert cell (row 5, column 4).
        ((GRID_ONE,), _observation(blocked_cell=(5, 4))),
        # The second micro-action's delay bin precedes the first's.
        (((0, 101, 5 * 18 + 4, 3), (1, 102, 20 * 18 + 10, 1)), _observation()),
        # After candidate 0 only 0.5 elixir remains: SECOND (continue) and candidate 1 are illegal.
        ((GRID_FIRST, BUILDING_SECOND), _observation(elixir=3.5)),
        # Nothing is affordable: gate ACT is illegal, and so is the ability candidate.
        ((ABILITY,), _observation(elixir=0.5)),
    ]


_EXPECTED_ILLEGAL = {"gate": 1, "candidate": 3, "target": 1, "delay": 1, "continue": 1}
_HEAD_COUNTS = {
    "gate": "gate_count",
    "candidate": "candidate_count",
    "target": "target_count",
    "delay": "delay_count",
    "continue": "continue_count",
}
_NEW_FIELDS = {"illegal_label_policy"} | {
    f"{head}_illegal_label_{suffix}" for head in _EXPECTED_ILLEGAL for suffix in ("count", "frac")
}


def _write_lanes(tmp_path: Path, lanes: dict[str, tuple[ILSequenceV4, ILSequenceV4]]) -> tuple[Path, Path]:
    directory = tmp_path / "cache"
    directory.mkdir()
    summaries = []
    for shard_index, tag in enumerate(sorted(lanes)):
        data_path = directory / f"shard-{shard_index:05d}.cril.zst"
        summary = write_il_cache_shard(
            data_path, (_result(tag, *lanes[tag]),), shard_id=f"shard-{shard_index:05d}", contract={}
        )
        summary["summary_file"] = data_path.with_suffix(".json").name
        summaries.append(summary)
    index_path = directory / "index.json"
    write_il_cache_index(index_path, summaries, contract={})
    tags_path = tmp_path / "tags.txt"
    tags_path.write_text("".join(f"{tag}\n" for tag in sorted(lanes)), encoding="utf-8")
    return index_path, tags_path


def _illegal_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    lanes = {
        "x": (
            _custom_lane("x:owner-0", _illegal_steps()),
            _custom_lane("x:owner-1", [(WAIT, _observation()), ((GRID_ONE,), _observation())]),
        ),
        "y": (
            _custom_lane("y:owner-0", [((GRID_FIRST, BUILDING_SECOND), _observation()), (WAIT, _observation())]),
            _custom_lane("y:owner-1", [(WAIT, _observation()), ((ABILITY,), _observation())]),
        ),
    }
    index_path, tags_path = _write_lanes(tmp_path, lanes)
    checkpoint = tmp_path / "checkpoint.pt"
    _save_checkpoint(checkpoint, 11, update_step=3)
    return index_path, tags_path, checkpoint


def _existing_fields(row: dict[str, object]) -> str:
    """The row as the evaluator wrote it before the policy existed (timings aside)."""

    return json.dumps(
        {key: value for key, value in row.items() if key not in _TIMING_FIELDS | _NEW_FIELDS}, sort_keys=True
    )


def test_illegal_label_counts_and_skip_policy_exclude_each_head(tmp_path: Path) -> None:
    index_path, tags_path, checkpoint = _illegal_fixture(tmp_path)

    (counted,) = _evaluate(index_path, tags_path, [checkpoint], validate=False, illegal_label_policy="count")
    (skipped,) = _evaluate(index_path, tags_path, [checkpoint], validate=False)

    assert skipped["illegal_label_policy"] == "skip" and counted["illegal_label_policy"] == "count"
    for head, expected in _EXPECTED_ILLEGAL.items():
        count_name = _HEAD_COUNTS[head]
        supervised = int(counted[count_name])  # type: ignore[call-overload]
        assert counted[f"{head}_illegal_label_count"] == skipped[f"{head}_illegal_label_count"] == expected, head
        # ``count`` keeps the illegal labels in the population, ``skip`` drops exactly them.
        assert skipped[count_name] == supervised - expected, head
        assert counted[f"{head}_illegal_label_frac"] == pytest.approx(expected / supervised), head
        assert skipped[f"{head}_illegal_label_frac"] == pytest.approx(expected / supervised), head
    assert skipped["gate_act_count"] == int(counted["gate_act_count"]) - 1  # type: ignore[call-overload]
    assert skipped["gate_wait_count"] == counted["gate_wait_count"]
    # The value head has no legality mask and is untouched.
    assert skipped["value_count"] == counted["value_count"]
    assert skipped["value_huber"] == counted["value_huber"]

    # ``count`` is the historical arithmetic: the masked-logit sentinel (~3.4e38)
    # dominates every head with an illegal label, and the delay head's
    # renormalized NLL is +inf, reported as such.
    for name in ("gate_nll_all", "gate_nll_act", "candidate_nll", "target_nll", "continue_nll", "decision_nll"):
        assert float(counted[name]) > 1e30, name  # type: ignore[arg-type]
    assert counted["delay_nll"] == math.inf
    # ``skip`` scores the remaining labels with plain, finite likelihoods.
    for name in ("gate_nll_all", "gate_nll_act", "candidate_nll", "target_nll", "delay_nll", "continue_nll"):
        assert 0.0 <= float(skipped[name]) < 50.0, name  # type: ignore[arg-type]
    assert skipped["decision_nll"] == pytest.approx(
        float(skipped["candidate_nll"]) + float(skipped["target_nll"])  # type: ignore[arg-type]
    )
    for name in ("candidate_top1", "target_exact", "delay_top1", "continue_top1", "gate_act_top1"):
        assert 0.0 <= float(skipped[name]) <= 1.0, name  # type: ignore[arg-type]


def test_skip_policy_matches_legal_labels_scored_alone(tmp_path: Path) -> None:
    """Under ``skip`` an illegal candidate label contributes exactly nothing to its head."""

    clean = tmp_path / "clean"
    dirty = tmp_path / "dirty"
    clean.mkdir()
    dirty.mkdir()
    base = [(WAIT, _observation()), ((GRID_ONE,), _observation())]
    # One extra trailing frame whose candidate label is illegal (its other labels are legal).
    tail = ((GRID_ONE,), _observation(elixir=2.0))
    clean_index, clean_tags = _write_lanes(clean, {"z": (_custom_lane("z:0", base), _custom_lane("z:1", base))})
    dirty_index, dirty_tags = _write_lanes(
        dirty, {"z": (_custom_lane("z:0", [*base, tail]), _custom_lane("z:1", base))}
    )
    checkpoint = tmp_path / "checkpoint.pt"
    _save_checkpoint(checkpoint, 11, update_step=3)

    (reference,) = _evaluate(clean_index, clean_tags, [checkpoint], validate=False)
    (skipped,) = _evaluate(dirty_index, dirty_tags, [checkpoint], validate=False)

    assert skipped["candidate_illegal_label_count"] == 1
    assert skipped["candidate_count"] == reference["candidate_count"]
    assert skipped["candidate_nll"] == pytest.approx(reference["candidate_nll"], abs=1e-6)
    assert skipped["candidate_top1"] == reference["candidate_top1"]
    # The other heads score the extra frame normally.
    assert skipped["target_count"] == int(reference["target_count"]) + 1  # type: ignore[call-overload]
    assert skipped["target_illegal_label_count"] == 0


def test_error_policy_raises_with_illegal_label_counts(tmp_path: Path) -> None:
    index_path, tags_path, checkpoint = _illegal_fixture(tmp_path)
    output = tmp_path / "metrics.jsonl"

    with pytest.raises(ValueError, match="illegal under the checkpoint's own legality mask") as caught:
        _evaluate(
            index_path, tags_path, [checkpoint], validate=False, illegal_label_policy="error", output_path=output
        )

    message = str(caught.value)
    assert str(checkpoint.resolve()) in message
    for head, expected in _EXPECTED_ILLEGAL.items():
        assert f"{head}={expected}/" in message, head
    assert not output.exists()


def test_policies_leave_rows_unchanged_when_no_label_is_illegal(tmp_path: Path) -> None:
    index_path, tags_path, _lanes = _write_fixture(tmp_path)
    checkpoint = tmp_path / "checkpoint-a.pt"
    _save_checkpoint(checkpoint, 11, update_step=7)

    (default,) = _evaluate(index_path, tags_path, [checkpoint])
    rows = {
        policy: _evaluate(index_path, tags_path, [checkpoint], illegal_label_policy=policy)[0]
        for policy in ("skip", "count", "error")
    }

    assert default["illegal_label_policy"] == "skip"
    # ``count`` is the pre-policy arithmetic, so every pre-existing field is byte-identical.
    before = _existing_fields(rows["count"])
    assert _existing_fields(default) == _existing_fields(rows["skip"]) == _existing_fields(rows["error"]) == before
    for row in (default, *rows.values()):
        assert set(row) - set(json.loads(_existing_fields(row))) - _TIMING_FIELDS == _NEW_FIELDS
        for head in _EXPECTED_ILLEGAL:
            assert row[f"{head}_illegal_label_count"] == 0
            assert row[f"{head}_illegal_label_frac"] == 0.0


def test_summarize_reports_non_finite_head_metrics_and_rejects_bad_counts() -> None:
    sums = [0.0] * len(_SUM_FIELDS)
    for name, value in {
        "owner_frames": 4.0,
        "valid_frames": 4.0,
        "candidate_count": 2.0,
        "candidate_nll_sum": 1.5,
        "target_count": 2.0,
        "target_nll_sum": math.inf,
        "delay_count": 2.0,
        "delay_nll_sum": math.nan,
        "candidate_illegal_label_count": 1.0,
    }.items():
        sums[_SUM_INDEX[name]] = value

    skipped = summarize_statistics(sums)
    counted = summarize_statistics(sums, illegal_label_policy="count")

    assert skipped["candidate_nll"] == 0.75
    assert skipped["target_nll"] == math.inf and skipped["decision_nll"] == math.inf
    assert math.isnan(float(skipped["delay_nll"]))  # type: ignore[arg-type]
    assert '"target_nll": Infinity' in json.dumps(skipped, sort_keys=True)
    assert skipped["candidate_illegal_label_frac"] == pytest.approx(1 / 3)
    assert counted["candidate_illegal_label_frac"] == pytest.approx(1 / 2)
    assert skipped["gate_illegal_label_frac"] is None

    bad = list(sums)
    bad[_SUM_INDEX["candidate_count"]] = math.nan
    with pytest.raises(FloatingPointError, match="count"):
        summarize_statistics(bad)
    with pytest.raises(ValueError, match="illegal_label_policy"):
        summarize_statistics(sums, illegal_label_policy="drop")


def test_heldout_evaluation_rejects_unknown_illegal_label_policy(tmp_path: Path) -> None:
    index_path, tags_path, _lanes = _write_fixture(tmp_path)
    checkpoint = tmp_path / "checkpoint-a.pt"
    _save_checkpoint(checkpoint, 11, update_step=7)

    with pytest.raises(ValueError, match="illegal_label_policy"):
        _evaluate(index_path, tags_path, [checkpoint], illegal_label_policy="ignore")


# --- Opt-in sharpness report (--report-entropy) --------------------------------

_SHARPNESS_SUFFIXES = ("_entropy", "_top1_prob", "_top1_gt09_frac")


def _sharpness_fields(row: dict[str, object]) -> set[str]:
    return {name for name in row if name.endswith(_SHARPNESS_SUFFIXES) or name.startswith("gate_entropy_")} | (
        {"report_entropy"} & set(row)
    )


def _whole_sequence(index_path: Path):  # type: ignore[no-untyped-def]
    shards = {
        tag: load_il_cache_shard(index_path.parent / f"shard-{index:05d}.cril.json")
        for index, tag in enumerate(("a", "b"))
    }
    stored = [shards[tag].sequence(owner) for tag in ("a", "b") for owner in (0, 1)]
    return collate_padded_il_sequences(stored).to("cpu")


def _binary_entropy(probability: torch.Tensor) -> torch.Tensor:
    p = probability.double().clamp(1e-12, 1.0)
    q = (1.0 - probability.double()).clamp(1e-12, 1.0)
    return -(p * p.log() + q * q.log()) * (probability < 1.0)


def test_report_entropy_is_off_by_default_and_leaves_rows_unchanged(tmp_path: Path) -> None:
    index_path, tags_path, _lanes = _write_fixture(tmp_path)
    checkpoint = tmp_path / "checkpoint-a.pt"
    _save_checkpoint(checkpoint, 11, update_step=7)

    (plain,) = _evaluate(index_path, tags_path, [checkpoint])
    (reported,) = _evaluate(index_path, tags_path, [checkpoint], report_entropy=True)

    assert not _sharpness_fields(plain)
    added = set(reported).difference(plain)
    assert added == _sharpness_fields(reported)
    assert reported["report_entropy"] is True
    for head in ENTROPY_HEADS:
        assert {f"{head}_entropy", f"{head}_top1_prob", f"{head}_top1_gt09_frac"} <= added
    for name, value in _without_timing(plain).items():
        if isinstance(value, float):
            assert reported[name] == pytest.approx(value, abs=1e-6), name
        else:
            assert reported[name] == value, name
    assert model_module._distribution.__name__ == "_distribution"


def test_report_entropy_means_match_teacher_forced_components(tmp_path: Path) -> None:
    index_path, tags_path, _lanes = _write_fixture(tmp_path)
    checkpoint = tmp_path / "checkpoint-a.pt"
    _save_checkpoint(checkpoint, 11, update_step=7)

    (row,) = _evaluate(index_path, tags_path, [checkpoint], report_entropy=True, time_steps=3, shards_per_batch=2)

    model = _model()
    load_actor_critic_checkpoint(checkpoint, model, map_location="cpu")
    model.eval()
    sequence = _whole_sequence(index_path)
    with torch.no_grad():
        evaluation = evaluate_recurrent_sequence(
            model,
            sequence.observations,
            sequence.actions,
            sequence.episode_start,
            gate_temperature=1.0,
            action_temperature=1.0,
            continue_temperature=1.0,
        )
    masks = head_masks(sequence)
    components = evaluation.components
    for head, entropy, mask in (
        ("gate", components.gate_entropy, masks.gate),
        ("candidate", components.candidate_entropy, masks.micro),
        ("target", components.target_entropy, masks.grid),
        ("delay", components.delay_offset_entropy, masks.micro),
        ("continue", components.continue_entropy, masks.act),
    ):
        expected = float(entropy[mask].double().mean())
        assert float(row[f"{head}_entropy"]) == pytest.approx(expected, abs=1e-5), head  # type: ignore[arg-type]
        assert 0.0 < float(row[f"{head}_top1_prob"]) <= 1.0  # type: ignore[arg-type]
        assert 0.0 <= float(row[f"{head}_top1_gt09_frac"]) <= 1.0  # type: ignore[arg-type]
    assert float(row["gate_entropy_act"]) == pytest.approx(  # type: ignore[arg-type]
        float(components.gate_entropy[masks.gate_act].double().mean()), abs=1e-5
    )
    assert float(row["gate_entropy_wait"]) == pytest.approx(  # type: ignore[arg-type]
        float(components.gate_entropy[masks.gate_wait].double().mean()), abs=1e-5
    )
    assert float(row["gate_entropy"]) <= math.log(2.0) + 1e-6  # type: ignore[arg-type]
    assert float(row["continue_entropy"]) <= math.log(2.0) + 1e-6  # type: ignore[arg-type]


def test_top1_log_probs_align_with_every_head(tmp_path: Path) -> None:
    index_path, _tags_path, _lanes = _write_fixture(tmp_path)
    torch.manual_seed(3)
    model = _model().eval()
    sequence = _whole_sequence(index_path)
    original = model_module._distribution

    for decode_mask in (None, sequence.valid_mask):
        with torch.no_grad(), DistributionRecorderV1() as recorder:
            evaluation = evaluate_recurrent_sequence(
                model,
                sequence.observations,
                sequence.actions,
                sequence.episode_start,
                gate_temperature=1.0,
                action_temperature=1.0,
                continue_temperature=1.0,
                preencode_observations=True,
                fused=True,
                decode_mask=decode_mask,
            )
        assert model_module._distribution is original
        top1 = top1_log_probs(recorder.calls, sequence, evaluation)
        masks = head_masks(sequence)
        components = evaluation.components

        # Binary heads: the top-1 probability determines the entropy exactly.
        gate_p = top1.gate.exp()
        assert torch.all(gate_p >= 0.5 - 1e-6)
        assert torch.allclose(
            _binary_entropy(gate_p)[masks.gate], components.gate_entropy.double()[masks.gate], atol=1e-4
        )
        continue_p = top1.continue_.exp()
        assert torch.allclose(
            _binary_entropy(continue_p)[masks.act], components.continue_entropy.double()[masks.act], atol=1e-4
        )
        # Categorical heads: max p >= exp(-H) and >= the expert label's probability.
        for top, entropy, log_prob, mask in (
            (top1.candidate, components.candidate_entropy, components.candidate_log_prob, masks.micro),
            (top1.target, components.target_entropy, components.target_log_prob, masks.grid),
        ):
            assert bool(mask.any())
            probability = top.exp()[mask]
            assert torch.all(probability <= 1.0 + 1e-6)
            assert torch.all(probability >= torch.exp(-entropy[mask].float()) - 1e-5)
            assert torch.all(probability >= log_prob[mask].float().exp() - 1e-5)


def test_top1_log_probs_rejects_unrecognized_layouts(tmp_path: Path) -> None:
    index_path, tags_path, _lanes = _write_fixture(tmp_path)
    torch.manual_seed(3)
    model = _model().eval()
    sequence = _whole_sequence(index_path)
    with torch.no_grad(), DistributionRecorderV1() as recorder:
        evaluation = evaluate_recurrent_sequence(
            model,
            sequence.observations,
            sequence.actions,
            sequence.episode_start,
            preencode_observations=True,
            fused=True,
        )
    with pytest.raises(RuntimeError, match="categoricals"):
        top1_log_probs(recorder.calls[:5], sequence, evaluation)
    with pytest.raises(RuntimeError, match="no categorical"):
        top1_log_probs([], sequence, evaluation)
    with pytest.raises(RuntimeError, match="nested"):
        with DistributionRecorderV1(), DistributionRecorderV1():
            pass
    assert model_module._distribution.__name__ == "_distribution"

    checkpoint = tmp_path / "checkpoint-a.pt"
    _save_checkpoint(checkpoint, 11, update_step=7)
    with pytest.raises(ValueError, match="fused_sequence"):
        _evaluate(index_path, tags_path, [checkpoint], report_entropy=True, fused_sequence=False)
