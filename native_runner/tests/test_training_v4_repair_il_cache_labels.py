from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import pytest
import torch

from native_runner.contracts import ActionKind, ActionV1, TargetKind
from native_runner.tests.test_training_v4_cache import _result
from native_runner.tests.test_training_v4_learning import _wait_actions
from native_runner.tests.test_training_v4_model import _batch
from native_runner.training.v4.cache import load_il_cache_shard, write_il_cache_shard
from native_runner.training.v4.decoding import ELIXIR_MULTIPLIER_SCALAR_INDEX
from native_runner.training.v4.expert import TimedExpertActionV4
from native_runner.training.v4.imitation import ILSequenceV4
from native_runner.training.v4.repair_il_cache_labels import (
    LABEL_REPAIR_SCHEMA,
    read_raw_shard,
    repair_shard,
)
from native_runner.training.v4.tensors import ActionSequenceV4, GATE_ACT, GATE_WAIT

EXPENSIVE = 3  # _batch candidate row 3: DEPLOY, cost 9, uid 104, legal everywhere.


def _frame(elixir: float) -> Any:
    batch = _batch(batch_size=1)
    match_scalars = batch.match_scalars.clone()
    match_scalars[:, ELIXIR_MULTIPLIER_SCALAR_INDEX] = 1.0 / 3.0
    candidates = replace(batch.candidates, elixir=torch.tensor([elixir]))
    return replace(batch, match_scalars=match_scalars, candidates=candidates).to_storage(float_dtype=torch.float16)


def _act(candidate: int, uid: int, target: int, delay: int) -> ActionSequenceV4:
    return replace(
        _wait_actions(1),
        gate=torch.tensor([GATE_ACT]),
        micro_action_count=torch.tensor([1]),
        candidate_index=torch.tensor([[candidate, -1]]),
        candidate_uid=torch.tensor([[uid, -1]]),
        target_cell=torch.tensor([[target, -1]]),
        delay_offset_bin=torch.tensor([[delay, -1]]),
    )


def _lane(sequence_id: str, elixirs: list[float], actions: list[ActionSequenceV4], gate_loss: list[bool]) -> ILSequenceV4:
    steps = len(elixirs)
    return ILSequenceV4(
        observations=tuple(_frame(value) for value in elixirs),
        actions=tuple(actions),
        episode_start=torch.tensor([[step == 0] for step in range(steps)], dtype=torch.bool),
        valid_mask=torch.ones(steps, 1, dtype=torch.bool),
        returns=torch.zeros(steps, 1),
        gate_loss_mask=torch.tensor(gate_loss, dtype=torch.bool)[:, None],
        value_loss_mask=torch.ones(steps, 1, dtype=torch.bool),
        sequence_id=sequence_id,
    )


def _play(owner: int, card_id: int, *, slot: int, target: tuple[int, int], tick: int, index: int) -> TimedExpertActionV4:
    action = ActionV1(
        owner=owner,
        kind=ActionKind.PLAY_CARD,
        hand_slot=slot,
        card_id=card_id,
        target_kind=TargetKind.GRID,
        target_grid=target,
        metadata={"source_command_tick": tick, "source_index": index},
    )
    return TimedExpertActionV4(source_tick=tick, action=action, source_index=index)


@pytest.fixture()
def source_shard(tmp_path: Path) -> tuple[Path, tuple[TimedExpertActionV4, ...]]:
    batch = _batch(batch_size=1)
    cheap_card = int(batch.candidates.native_visible_card_id[0, 0])
    expensive_card = int(batch.candidates.native_visible_card_id[0, EXPENSIVE])
    wait = _wait_actions(1)
    # Owner 0: frame 0 is a supervised play (touch d+1); frame 1 is the
    # alignment-dropped near miss (cost 9, elixir 8.97 at d=95, touch 97).
    owner0 = _lane(
        "a:owner-0",
        [10.0, 9.0 - 0.03, 10.0],
        [_act(0, 101, 4 * 18 + 3, 1), wait, wait],
        [True, False, True],
    )
    # Owner 1: frame 1 is dropped 0.5 elixir short; it must stay dropped.
    owner1 = _lane("a:owner-1", [10.0, 8.5, 10.0], [_act(0, 101, 4 * 18 + 3, 1), wait, wait], [True, False, True])
    source_dir = tmp_path / "node-0"
    data_path = source_dir / "part-00-shard-000000.cril.zst"
    write_il_cache_shard(data_path, (_result("a", owner0, owner1),), shard_id="part-00-shard-000000", contract={"t": 1})
    actions = (
        _play(0, cheap_card, slot=0, target=(3, 4), tick=91, index=0),
        _play(0, expensive_card, slot=3, target=(5, 6), tick=97, index=1),
        _play(1, cheap_card, slot=0, target=(14, 27), tick=91, index=2),
        _play(1, expensive_card, slot=3, target=(5, 6), tick=99, index=3),
    )
    return data_path.with_suffix(".json"), actions


def _run(summary: Path, output: Path, actions: tuple[TimedExpertActionV4, ...], **kwargs: Any) -> dict[str, Any]:
    payloads: dict[str, Mapping[str, Any]] = {"a": {"replay_tag": "a"}}

    def expert_actions(payload: Mapping[str, Any]) -> tuple[TimedExpertActionV4, ...]:
        assert payload["replay_tag"] == "a"
        return actions

    return repair_shard(summary, output, payload_for_tag=payloads.__getitem__, expert_actions=expert_actions, **kwargs)


def test_repair_restores_near_miss_label_and_nothing_else(
    tmp_path: Path, source_shard: tuple[Path, tuple[TimedExpertActionV4, ...]]
) -> None:
    summary, actions = source_shard
    source_bytes = {path.name: path.read_bytes() for path in summary.parent.iterdir()}
    output = tmp_path / "out" / "node-0" / summary.name

    report = _run(summary, output, actions, verify=True)

    assert report["repaired_frames"] == 1
    assert report["attempted_frames"] == 2
    assert report["per_reason_counts"] == {"illegal_under_shadow_state": 1, "repaired": 1}
    assert report["verified"] is True
    # Source shards are never touched.
    assert {path.name: path.read_bytes() for path in summary.parent.iterdir()} == source_bytes

    repaired = load_il_cache_shard(output)
    original = load_il_cache_shard(summary)
    assert repaired.actions.gate.tolist() == [GATE_ACT, GATE_ACT, GATE_WAIT, GATE_ACT, GATE_WAIT, GATE_WAIT]
    assert repaired.actions.candidate_uid[1].tolist() == [104, -1]
    assert repaired.actions.candidate_index[1].tolist() == [EXPENSIVE, -1]
    assert repaired.actions.target_cell[1].tolist() == [6 * 18 + 5, -1]
    assert repaired.actions.delay_offset_bin[1].tolist() == [2, -1]
    assert repaired.gate_loss_mask[:, 0].tolist() == [True, True, True, True, False, True]
    previous = repaired.observations.previous_action
    assert previous.gate[2].item() == GATE_ACT
    assert previous.micro_action_count[2].item() == 1
    assert previous.action_mask[2].tolist() == [True, False]
    assert previous.delay_offset_bin[2].tolist() == [2, -1]
    assert previous.target_cell[2].tolist() == [6 * 18 + 5, -1]
    assert previous.visible_card_vocab_id[2, 0] == original.observations.candidates.visible_card_vocab_id[1, EXPENSIVE]
    manifest = repaired.manifest["label_repair"]
    assert manifest["schema"] == LABEL_REPAIR_SCHEMA
    assert manifest["repaired_frames"] == 1 and manifest["attempted_frames"] == 2
    assert set(manifest["source_sha256"]) == {"repair_il_cache_labels.py", "decoding.py", "expert.py"}
    assert repaired.summary["label_repair"] == {
        "schema": LABEL_REPAIR_SCHEMA,
        "repaired_frames": 1,
        "attempted_frames": 2,
        "per_reason_counts": {"illegal_under_shadow_state": 1, "repaired": 1},
    }

    # Everything outside the repaired action row, its gate bit and the next
    # frame's previous-action record is bit-identical.
    source_raw = read_raw_shard(summary).tensors
    output_raw = read_raw_shard(output).tensors
    assert source_raw.keys() == output_raw.keys()
    changed = {name for name in source_raw if not torch.equal(source_raw[name], output_raw[name])}
    assert changed <= {
        "gate_loss_mask",
        *(f"actions.{name}" for name in ("gate", "micro_action_count", "candidate_index", "candidate_uid", "target_cell", "delay_offset_bin")),
        *(f"observations.previous_action.{name}" for name in ("gate", "micro_action_count", "visible_card_vocab_id", "effective_card_vocab_id", "target_cell", "delay_offset_bin", "action_mask")),
    }
    for name in changed:
        rows = (source_raw[name] != output_raw[name]).reshape(source_raw[name].shape[0], -1).any(dim=1)
        assert rows.nonzero().flatten().tolist() in ([1], [2], [1, 2])


def test_repair_is_idempotent_and_dry_run_writes_nothing(
    tmp_path: Path, source_shard: tuple[Path, tuple[TimedExpertActionV4, ...]]
) -> None:
    summary, actions = source_shard
    first = tmp_path / "first" / "node-0" / summary.name
    second = tmp_path / "second" / "node-0" / summary.name
    dry = tmp_path / "dry" / "node-0" / summary.name

    _run(summary, first, actions)
    report = _run(first, second, actions, verify=True)
    dry_report = _run(summary, dry, actions, dry_run=True)

    assert report["repaired_frames"] == 0 and report["copied_unchanged"] is True
    assert report["per_reason_counts"] == {"illegal_under_shadow_state": 1}
    for name in (summary.name, summary.with_suffix(".zst").name):
        assert (first.parent / name).read_bytes() == (second.parent / name).read_bytes()
    assert dry_report["repaired_frames"] == 1
    assert not dry.parent.exists()


def test_repair_refuses_source_directory_and_reports_missing_candidates(
    tmp_path: Path, source_shard: tuple[Path, tuple[TimedExpertActionV4, ...]]
) -> None:
    summary, actions = source_shard
    with pytest.raises(ValueError, match="source directory"):
        _run(summary, summary.parent / "copy.cril.json", actions)
    # A card that is not among the stored candidates (the native mask omitted
    # it) cannot be repaired offline.
    absent = (actions[0], replace(actions[1], action=replace(actions[1].action, card_id=7777)), *actions[2:])
    report = _run(summary, tmp_path / "absent" / "node-0" / summary.name, absent)
    assert report["repaired_frames"] == 0
    assert report["per_reason_counts"] == {"candidate_absent_other": 1, "illegal_under_shadow_state": 1}
    # The card in hand under another slot than the expert's: the simulated
    # hand diverged, so the label is not restored.
    moved = (actions[0], replace(actions[1], action=replace(actions[1].action, hand_slot=2)), *actions[2:])
    report = _run(summary, tmp_path / "moved" / "node-0" / summary.name, moved)
    assert report["per_reason_counts"] == {"hand_slot_mismatch": 1, "illegal_under_shadow_state": 1}
    # A lane whose stored ACT labels disagree with the source is skipped whole.
    shifted = (replace(actions[0], source_tick=96), *actions[1:])
    report = _run(summary, tmp_path / "shifted" / "node-0" / summary.name, shifted)
    assert report["per_reason_counts"] == {"illegal_under_shadow_state": 1, "lane_timeline_mismatch": 1}
    # A supervised play whose source slot differs from the stored one means
    # the uncalibrated slots cannot be trusted for that lane.
    uncalibrated = (replace(actions[0], action=replace(actions[0].action, hand_slot=2)), *actions[1:])
    report = _run(summary, tmp_path / "uncalibrated" / "node-0" / summary.name, uncalibrated)
    assert report["per_reason_counts"] == {"illegal_under_shadow_state": 1, "lane_hand_slots_uncalibrated": 1}


@pytest.mark.parametrize(
    ("touch_tick", "reason"),
    [(97, "candidate_absent_affordable_at_touch"), (96, "candidate_absent_unaffordable_at_touch")],
)
def test_unaffordable_card_missing_from_stored_candidates_is_reported_not_repaired(
    tmp_path: Path, touch_tick: int, reason: str
) -> None:
    # This is the dominant production case: the native action mask omits a
    # hand slot it cannot afford at d, so the candidate row was never stored.
    from native_runner.tests.test_training_v4_model import _with_candidate_mask

    batch = _batch(batch_size=1)
    cheap_card = int(batch.candidates.native_visible_card_id[0, 0])
    expensive_card = int(batch.candidates.native_visible_card_id[0, EXPENSIVE])
    wait = _wait_actions(1)
    owner0 = _lane("c:owner-0", [10.0, 9.0 - 0.03, 10.0], [_act(0, 101, 4 * 18 + 3, 1), wait, wait], [True, False, True])
    hidden = owner0.observations[1]
    mask = hidden.candidates.mask.clone()
    mask[0, EXPENSIVE] = False
    owner0 = replace(
        owner0,
        observations=(
            owner0.observations[0],
            replace(hidden, candidates=_with_candidate_mask(hidden.candidates, mask)),
            owner0.observations[2],
        ),
    )
    owner1 = _lane("c:owner-1", [10.0, 10.0, 10.0], [wait, wait, wait], [True, True, True])
    data_path = tmp_path / "node-0" / "part-00-shard-000002.cril.zst"
    write_il_cache_shard(data_path, (_result("c", owner0, owner1),), shard_id="part-00-shard-000002", contract={})
    actions = (
        _play(0, cheap_card, slot=0, target=(3, 4), tick=91, index=0),
        _play(0, expensive_card, slot=3, target=(5, 6), tick=touch_tick, index=1),
    )

    output = tmp_path / "out" / "node-0" / data_path.with_suffix(".json").name
    report = repair_shard(
        data_path.with_suffix(".json"),
        output,
        payload_for_tag=lambda tag: {"replay_tag": tag},
        expert_actions=lambda payload: actions,
        verify=True,
    )
    assert report["repaired_frames"] == 0
    assert report["per_reason_counts"] == {reason: 1}
    # Nothing repaired: the container is copied verbatim; only the summary
    # records the pass, and a rerun reproduces both files exactly.
    assert report["copied_unchanged"] is True
    assert (output.parent / data_path.name).read_bytes() == data_path.read_bytes()
    assert load_il_cache_shard(output).summary["label_repair"]["per_reason_counts"] == {reason: 1}
    rerun = tmp_path / "rerun" / "node-0" / output.name
    repair_shard(output, rerun, payload_for_tag=lambda tag: {"replay_tag": tag}, expert_actions=lambda payload: actions)
    assert rerun.read_bytes() == output.read_bytes()
    assert (rerun.parent / data_path.name).read_bytes() == data_path.read_bytes()


def test_repair_needs_a_supervised_card_play_to_trust_lane_slots(tmp_path: Path) -> None:
    wait = _wait_actions(1)
    expensive_card = int(_batch(batch_size=1).candidates.native_visible_card_id[0, EXPENSIVE])
    owner0 = _lane("b:owner-0", [10.0, 9.0 - 0.03, 10.0], [wait, wait, wait], [True, False, True])
    owner1 = _lane("b:owner-1", [10.0, 10.0, 10.0], [wait, wait, wait], [True, True, True])
    data_path = tmp_path / "node-0" / "part-00-shard-000001.cril.zst"
    write_il_cache_shard(data_path, (_result("b", owner0, owner1),), shard_id="part-00-shard-000001", contract={})
    actions = (_play(0, expensive_card, slot=3, target=(5, 6), tick=97, index=0),)

    def expert_actions(payload: Mapping[str, Any]) -> tuple[TimedExpertActionV4, ...]:
        return actions

    report = repair_shard(
        data_path.with_suffix(".json"),
        tmp_path / "out" / "node-0" / data_path.with_suffix(".json").name,
        payload_for_tag=lambda tag: {"replay_tag": tag},
        expert_actions=expert_actions,
    )
    assert report["per_reason_counts"] == {"lane_hand_slots_unverified": 1}
