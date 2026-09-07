from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import native_runner.training.v4.cache_builder as cache_builder
from native_runner.tests.test_training_v4_learning import _wait_actions
from native_runner.tests.test_training_v4_model import _batch
from native_runner.training.v4.cache import (
    ILCacheChunkRefV1,
    _sanitize_legacy_illegal_expert_targets,
    _upgrade_legacy_bridge_placement,
    _upgrade_legacy_pocket_placement,
    assert_il_sequences_exact,
    collate_loaded_il_batches,
    load_il_cache_index,
    load_il_cache_manifest,
    load_il_cache_shard,
    write_il_cache_index,
    write_il_cache_shard,
)
from native_runner.training.v4.imitation import (
    ILSequenceV4,
    collate_padded_il_sequences,
)
from native_runner.training.v4.producer import ReplayProductionResultV4
from native_runner.training.v4.train_imitation_cache import (
    ReplayUnitV1,
    ShardTrainingPlanV1,
    _epoch_groups,
    build_training_plan,
)


def test_cache_source_hashes_bind_native_assembly() -> None:
    package_root = Path(cache_builder.__file__).resolve().parents[2]
    hashes = cache_builder._source_hashes()
    for relative in ("probe/native_call_arm64.S", "probe/remaining_runtime_arm64.S"):
        expected = hashlib.sha256((package_root / relative).read_bytes()).hexdigest()
        assert hashes[relative] == expected


def test_cache_partition_counts_rejected_source_rows_before_partitioning(tmp_path: Path) -> None:
    arrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    replay_root = tmp_path / "replays"
    replay_root.mkdir()
    payloads = [
        {"replay_tag": "empty", "events": []},
        {"replay_tag": "accepted", "events": [{
            "kind": "play_card", "replay_tick_20hz": 130,
            "side": "team", "source_fields": {"data_i": 0},
        }]},
        {"replay_tag": "empty-second", "events": []},
    ]
    parquet.write_table(arrow.table({
        "replay_tag": [payload["replay_tag"] for payload in payloads],
        "payload_json": [json.dumps(payload) for payload in payloads],
    }), replay_root / "part-000000.parquet")

    selected, labels = cache_builder._select_partition(
        (tmp_path,), source_count=3, partition_index=0, partition_count=2,
    )

    assert [index for index, _ in selected] == [0, 2]
    assert [row["replay_tag"] for row in labels] == ["empty", "empty-second"]
    assert [row["preflight_rejection_reason"] for row in labels] == [
        "no_expert_actions", "no_expert_actions",
    ]


def _sequence(
    sequence_id: str,
    returns: list[float],
    *,
    gate_loss_mask: torch.Tensor | None = None,
) -> ILSequenceV4:
    batch = _batch(batch_size=1).to_storage(float_dtype=torch.float16)
    steps = len(returns)
    return ILSequenceV4(
        observations=(batch,) * steps,
        actions=(_wait_actions(1),) * steps,
        episode_start=torch.tensor(
            [[step == 0] for step in range(steps)], dtype=torch.bool
        ),
        valid_mask=torch.ones(steps, 1, dtype=torch.bool),
        returns=torch.tensor(returns, dtype=torch.float32)[:, None],
        gate_loss_mask=gate_loss_mask,
        value_loss_mask=torch.ones(steps, 1, dtype=torch.bool),
        sequence_id=sequence_id,
    )


def _result(
    replay_tag: str,
    owner0: ILSequenceV4,
    owner1: ILSequenceV4,
) -> ReplayProductionResultV4:
    return ReplayProductionResultV4(
        replay_tag=replay_tag,
        sequences=(owner0, owner1),
        completed=True,
        failure_reason=None,
        first_untrusted_tick=None,
        simulated_end_tick=100,
        source_end_tick=100,
        expert_action_count=2,
        executed_expert_action_count=2,
        winner_matches=True,
        crowns_match=True,
    )


def test_legacy_cache_pocket_migration_restores_center_seam() -> None:
    batch = _batch(batch_size=1)
    placement = torch.zeros_like(batch.candidates.placement)
    placement[:, 0, 17:23, :7] = True
    placement[:, 1, 17:23, 11:] = True
    runtime = batch.candidates.runtime_features.clone()
    candidates = replace(
        batch.candidates,
        placement=placement,
        runtime_features=runtime,
    )
    migrated = replace(batch, candidates=candidates)

    _upgrade_legacy_pocket_placement(migrated)

    assert bool(migrated.candidates.placement[0, 0, 18, 8])
    assert not bool(migrated.candidates.placement[0, 0, 18, 9])
    assert bool(migrated.candidates.placement[0, 1, 18, 9])
    assert not bool(migrated.candidates.placement[0, 1, 18, 8])
    expected = migrated.candidates.placement.to(torch.float32).mean(dim=(-2, -1))
    torch.testing.assert_close(
        migrated.candidates.runtime_features[:, :, 1],
        expected,
    )


def test_legacy_cache_migration_drops_unrepresentable_target_without_leakage() -> None:
    batch = _batch(batch_size=1)
    placement = batch.candidates.placement.clone()
    placement[0, 1, 5, 6] = False
    batch = replace(
        batch,
        candidates=replace(batch.candidates, placement=placement),
    )
    actions = replace(
        _wait_actions(1),
        gate=torch.tensor([1]),
        micro_action_count=torch.tensor([1]),
        candidate_index=torch.tensor([[1, -1]]),
        candidate_uid=torch.tensor([[102, -1]]),
        target_cell=torch.tensor([[5 * 18 + 6, -1]]),
        delay_offset_bin=torch.tensor([[3, -1]]),
    )

    gate_loss_mask = torch.ones(1, 1, dtype=torch.bool)
    changed = _sanitize_legacy_illegal_expert_targets(
        batch,
        actions,
        gate_loss_mask,
    )

    assert int(changed.sum()) == 1
    assert not bool(batch.candidates.placement[0, 1, 5, 6])
    assert actions.gate.tolist() == [0]
    assert actions.micro_action_count.tolist() == [0]
    assert actions.target_cell.tolist() == [[-1, -1]]
    assert gate_loss_mask.tolist() == [[False]]


def test_legacy_cache_bridge_migration_adds_only_troop_bridge_cells() -> None:
    batch = _batch(batch_size=1)
    placement = batch.candidates.placement.clone()
    placement[:, :, 15:17] = False
    batch = replace(
        batch,
        candidates=replace(batch.candidates, placement=placement),
    )

    _upgrade_legacy_bridge_placement(batch)

    assert bool(batch.candidates.placement[0, 0, 15, 2])
    assert not bool(batch.candidates.placement[0, 0, 15, 4])
    assert not bool(batch.candidates.placement[0, 1, 15, 2])


def test_cache_round_trip_preserves_every_sequence_tensor(tmp_path: Path) -> None:
    sequences = (
        _sequence("a:owner-0", [1.0, 2.0, 3.0]),
        _sequence(
            "a:owner-1",
            [4.0, 5.0, 6.0],
            gate_loss_mask=torch.tensor([[True], [False], [True]]),
        ),
    )
    data_path = tmp_path / "shard-00000.cril.zst"
    summary = write_il_cache_shard(
        data_path,
        (_result("a", *sequences),),
        shard_id="shard-00000",
        contract={"tensorizer": "test"},
    )

    shard = load_il_cache_shard(data_path.with_suffix(".json"))

    assert summary["owner_frames"] == 6
    assert summary["completed_replays"] == 1
    assert summary["failed_replays"] == 0
    assert summary["winner_mismatches"] == 0
    assert summary["crowns_mismatches"] == 0
    assert shard.owner_frames == 6
    assert shard.manifest["contract"] == {"tensorizer": "test"}
    assert load_il_cache_manifest(data_path.with_suffix(".json")) == shard.manifest
    for index, expected in enumerate(sequences):
        assert_il_sequences_exact(expected, shard.sequence(index))


def test_cache_batch_load_matches_existing_padded_collation(tmp_path: Path) -> None:
    long = _sequence("long", [1.0, 2.0, 3.0])
    short = _sequence(
        "short",
        [4.0, 5.0],
        gate_loss_mask=torch.tensor([[True], [False]]),
    )
    result = _result("mixed", long, short)
    data_path = tmp_path / "shard-00000.cril.zst"
    write_il_cache_shard(
        data_path,
        (result,),
        shard_id="shard-00000",
        contract={},
    )
    shard = load_il_cache_shard(data_path.with_suffix(".json"))

    loaded = shard.load_batch(
        (
            ILCacheChunkRefV1(sequence_index=0, start=0, steps=3),
            ILCacheChunkRefV1(sequence_index=1, start=0, steps=2),
        ),
        time_steps=3,
    )
    expected = collate_padded_il_sequences((long, short))

    assert loaded.owner_frames == 5
    assert_il_sequences_exact(expected, loaded.sequence)


def test_cross_shard_cache_batch_collation_preserves_lane_order(
    tmp_path: Path,
) -> None:
    paths = []
    for shard_index, value in enumerate((1.0, 3.0)):
        owner0 = _sequence(f"s{shard_index}:owner-0", [value, value + 1.0])
        owner1 = replace(owner0, sequence_id=f"s{shard_index}:owner-1")
        data_path = tmp_path / f"shard-{shard_index:05d}.cril.zst"
        write_il_cache_shard(
            data_path,
            (_result(f"s{shard_index}", owner0, owner1),),
            shard_id=f"shard-{shard_index:05d}",
            contract={},
        )
        paths.append(data_path.with_suffix(".json"))
    shards = tuple(load_il_cache_shard(path) for path in paths)
    batches = tuple(
        shard.load_batch(
            (ILCacheChunkRefV1(sequence_index=0, start=0, steps=2),),
            time_steps=2,
        )
        for shard in shards
    )

    loaded = collate_loaded_il_batches(batches)

    assert loaded.owner_frames == 4
    assert loaded.sequence.batch_size == 2
    torch.testing.assert_close(
        loaded.sequence.returns,
        torch.tensor([[1.0, 3.0], [2.0, 4.0]]),
    )


def test_cache_detects_compressed_corruption(tmp_path: Path) -> None:
    sequence = _sequence("a", [1.0])
    data_path = tmp_path / "shard-00000.cril.zst"
    write_il_cache_shard(
        data_path,
        (_result("a", sequence, replace(sequence, sequence_id="b")),),
        shard_id="shard-00000",
        contract={},
    )
    payload = bytearray(data_path.read_bytes())
    payload[-1] ^= 0xFF
    data_path.write_bytes(payload)

    with pytest.raises(ValueError, match="compressed SHA-256"):
        load_il_cache_shard(data_path.with_suffix(".json"))


def test_cache_rejects_nonfinite_values_after_fp16_packing(tmp_path: Path) -> None:
    sequence = _sequence("bad", [1.0])
    observation = sequence.observations[0]
    match_scalars = observation.match_scalars.clone()
    match_scalars[0, 0] = float("inf")
    bad = replace(
        sequence,
        observations=(replace(observation, match_scalars=match_scalars),),
    )

    with pytest.raises(
        ValueError,
        match=r"observations\.match_scalars contains 1 non-finite",
    ):
        write_il_cache_shard(
            tmp_path / "shard-00000.cril.zst",
            (_result("bad", bad, replace(sequence, sequence_id="other")),),
            shard_id="shard-00000",
            contract={},
        )


def test_cache_index_is_canonical_and_detects_metadata_changes(tmp_path: Path) -> None:
    summary = {
        "replay_results": 1,
        "sequences": 2,
        "owner_frames": 6,
        "match_decision_frames": 3,
        "compressed_bytes": 10,
        "container_bytes": 20,
    }
    index_path = tmp_path / "index.json"
    written = write_il_cache_index(
        index_path,
        (summary,),
        contract={"runtime": "test"},
    )

    assert load_il_cache_index(index_path) == written
    assert written["completed_replays"] == 0
    index_path.write_text(
        index_path.read_text(encoding="utf-8").replace(
            '"owner_frames":6', '"owner_frames":7'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="index SHA-256"):
        load_il_cache_index(index_path)


def test_cache_training_plan_uses_exact_split_and_later_recovery(
    tmp_path: Path,
) -> None:
    indexes = []
    for directory_name, tags in (
        ("base", ("a", "b")),
        ("recovery", ("a",)),
    ):
        directory = tmp_path / directory_name
        directory.mkdir()
        summaries = []
        for shard_index, replay_tag in enumerate(tags):
            owner0 = _sequence(f"{replay_tag}:owner-0", [float(shard_index + 1)])
            owner1 = replace(owner0, sequence_id=f"{replay_tag}:owner-1")
            data_path = directory / f"shard-{shard_index:05d}.cril.zst"
            summary = write_il_cache_shard(
                data_path,
                (_result(replay_tag, owner0, owner1),),
                shard_id=f"{directory_name}-{shard_index}",
                contract={},
            )
            summary["summary_file"] = data_path.with_suffix(".json").name
            summaries.append(summary)
        index_path = directory / "index.json"
        write_il_cache_index(index_path, summaries, contract={})
        indexes.append(index_path)
    tags_path = tmp_path / "train-replay-tags.txt"
    tags_path.write_text("a\nb\n", encoding="utf-8")

    plans, metadata = build_training_plan(
        indexes,
        tags_path,
        manifest_workers=2,
    )

    locations = {
        unit.replay_tag: plan.summary_path.parent.name
        for plan in plans
        for unit in plan.units
    }
    assert locations == {"a": "recovery", "b": "base"}
    assert metadata["selected_replays"] == 2
    assert metadata["selected_owner_sequences"] == 4
    assert metadata["overridden_replay_copies"] == 1


def test_cache_training_epoch_groups_bucket_sequence_lengths(tmp_path: Path) -> None:
    plans = tuple(
        ShardTrainingPlanV1(
            summary_path=tmp_path / f"{length}.json",
            units=(
                ReplayUnitV1(
                    replay_tag=str(length),
                    sequence_indices=(0, 1),
                    sequence_lengths=(length, length),
                ),
            ),
        )
        for length in range(1, 9)
    )

    groups = _epoch_groups(plans, seed=7, epoch=0, group_size=2)

    assert {plan.summary_path for group in groups for plan in group} == {
        plan.summary_path for plan in plans
    }
    assert all(
        max(max(plan.sequence_lengths) for plan in group)
        - min(max(plan.sequence_lengths) for plan in group)
        == 1
        for group in groups
    )


def test_cache_worker_calibrates_before_observation_extraction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    del tmp_path
    events: list[str] = []

    class _Native:
        def stop_resident_mode(self) -> None:
            events.append("stop-resident")

        def create_match(self, config: object) -> object:
            assert config == "deal-config"
            events.append("headless-create")
            return {"deal": "observed"}

    native_endpoint = _Native()
    raw = SimpleNamespace(replay_tag="raw")
    calibrated = SimpleNamespace(
        replay_tag="calibrated",
        replay=SimpleNamespace(episode_config=object()),
    )

    class _ResidentNative:
        def close(self) -> None:
            events.append("close")

    environment = SimpleNamespace(native=_ResidentNative())
    coordinator = object()
    tensorizers = (object(), object())
    produced = SimpleNamespace(sequences=(object(),))

    monkeypatch.setattr(cache_builder, "prepare_collected_replay", lambda _p: raw)
    monkeypatch.setattr(cache_builder, "replay_expert_actions", lambda _replay: ())

    def calibrate(prepared: object, selected_native: object) -> object:
        assert prepared is raw
        assert selected_native.create_native_match("deal-config") == {
            "deal": "observed"
        }
        assert selected_native.pause()["mode"] == "headless"
        events.append("calibrate")
        return calibrated

    def build_environments(
        prepared: object,
        *,
        env_ids: object,
        **_kwargs: object,
    ) -> object:
        assert prepared == (calibrated,)
        assert env_ids == (0,)
        events.append("environment")
        return (environment,), coordinator

    def build_tensorizers(episode: object) -> tuple[object, object]:
        assert episode is calibrated.replay.episode_config
        events.append("tensorizers")
        return tensorizers

    def produce(
        selected_environments: object,
        prepared: object,
        selected_tensorizers: object,
        **kwargs: object,
    ) -> object:
        assert selected_environments == (environment,)
        assert prepared == (calibrated,)
        assert selected_tensorizers == (tensorizers,)
        assert kwargs["max_frames"] is None
        assert kwargs["batch_coordinator"] is coordinator
        events.append("observation")
        return (produced,)

    monkeypatch.setattr(cache_builder, "calibrate_collected_replay_deal", calibrate)
    monkeypatch.setattr(
        cache_builder,
        "build_resident_collected_replay_batch_v4",
        build_environments,
    )
    monkeypatch.setattr(
        cache_builder, "build_episode_tensorizers_v4", build_tensorizers
    )
    monkeypatch.setattr(cache_builder, "produce_il_replay_batch", produce)

    results = cache_builder._convert_batch(
        native=native_endpoint,
        payload_batch=((4, {"replay": 1}),),
        host="127.0.0.1",
        port=25000,
        gamma=0.99,
        max_frames=None,
        timeout=12.0,
        resident_slots=1,
        capture_mode="zlib-json",
    )

    assert results == (produced,)
    assert events == [
        "stop-resident",
        "headless-create",
        "calibrate",
        "tensorizers",
        "environment",
        "observation",
        "close",
    ]
