"""Dense subset repack: writer-identical shards, verification, planner equivalence, real-cache subset."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import random
import struct

import pytest
import torch

from native_runner.tests.test_training_v4_cache import _result
from native_runner.tests.test_training_v4_evaluate_imitation_cache import (
    _evaluate,
    _fixture_lanes,
    _save_checkpoint,
    _without_timing,
)
from native_runner.training.v4 import repack_il_cache_subset as repack_module
from native_runner.training.v4.cache import (
    _decode_il_cache_shard,
    _pad_dynamic_observation,
    _tensor_from_spec,
    assert_il_sequences_exact,
    load_il_cache_index,
    load_il_cache_manifest,
    load_il_cache_shard,
    write_il_cache_index,
    write_il_cache_shard,
)
from native_runner.training.v4.repack_il_cache_subset import assign_replays, repack, safetensors_layout, verify
from native_runner.training.v4.train_imitation_cache import build_training_plan

_CONTRACT = {"gamma_per_decision": 0.99, "fixture": "repack"}


def _lanes():
    """Fixture lanes; replay ``a`` is widened so its source shard has larger dynamic capacities."""

    lanes = dict(_fixture_lanes())
    lanes["a"] = tuple(
        replace(
            lane,
            observations=tuple(
                _pad_dynamic_observation(
                    observation,
                    active_effect_count=int(observation.active_effects.mask.shape[1]) + 2,
                    relation_edge_count=int(observation.relation_edges.mask.shape[1]) + 3,
                    candidate_count=int(observation.candidates.mask.shape[1]) + 1,
                )
                for observation in lane.observations
            ),
        )
        for lane in lanes["a"]
    )
    return lanes


def _results():
    lanes = _lanes()
    results = {tag: _result(tag, *lanes[tag]) for tag in lanes}
    # Varied per-replay metadata so the recomputed shard aggregates are exercised.
    results["b"] = replace(results["b"], completed=False, failure_reason="unknown engine stop", winner_matches=False)
    results["c"] = replace(results["c"], crowns_match=None, expert_action_count=5)
    return results


def _write_source(tmp_path: Path, layout=(("a", "c"), ("b",))) -> tuple[Path, Path]:
    results = _results()
    directory = tmp_path / "source" / "node-0"
    directory.mkdir(parents=True)
    summaries = []
    for index, tags in enumerate(layout):
        data_path = directory / f"part-{index:05d}.cril.zst"
        summary = write_il_cache_shard(
            data_path, tuple(results[tag] for tag in tags), shard_id=f"part-{index:05d}", contract=_CONTRACT
        )
        summaries.append({**summary, "summary_file": data_path.with_suffix(".json").name, "host": "h", "port": 1})
    index_path = directory / "index.json"
    write_il_cache_index(index_path, summaries, contract={**_CONTRACT, "selected_replays": 3})
    tags_path = tmp_path / "tags.txt"
    tags_path.write_text("a\nb\nc\n", encoding="utf-8")
    return index_path, tags_path


def _container(summary_path: Path) -> bytes:
    import zstandard

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    compressed = (summary_path.parent / summary["data_file"]).read_bytes()
    return zstandard.ZstdDecompressor().stream_reader(compressed).read()


def test_safetensors_layout_matches_the_library() -> None:
    from safetensors.torch import save

    tensors = {
        "z": torch.zeros(0, 4),
        "b": torch.tensor([True, False, True]),
        "a": torch.arange(4, dtype=torch.float16).view(2, 2),
        "c": torch.tensor([7], dtype=torch.int64),
        "d": torch.arange(5, dtype=torch.int32),
        "e": torch.tensor([1, 2], dtype=torch.uint8),
        "g": torch.ones(3, dtype=torch.bfloat16),
        "h": torch.ones(1, dtype=torch.int16),
        "i": torch.ones(1, dtype=torch.int8),
        "j": torch.ones(2, dtype=torch.float64),
        "f": torch.ones(2, dtype=torch.float32),
    }
    header, order = safetensors_layout(tensors)
    payload = header + b"".join(
        tensors[name].contiguous().view(-1).view(torch.uint8).numpy().tobytes() for name in order
    )
    assert payload == save(tensors)
    assert struct.unpack("<Q", header[:8])[0] % 8 == 0


def test_assignment_is_sorted_seeded_and_chunked() -> None:
    tags = [f"t{index:02d}" for index in range(37)]
    shards = assign_replays(reversed(tags), replays_per_shard=16, seed=5)
    expected = sorted(tags)
    random.Random(5).shuffle(expected)
    assert [len(shard) for shard in shards] == [16, 16, 5]
    assert [tag for shard in shards for tag in shard] == expected
    assert assign_replays(tags, replays_per_shard=16, seed=5) == shards
    assert assign_replays(tags, replays_per_shard=16, seed=6) != shards


@pytest.mark.parametrize("replays_per_shard", (3, 2))
def test_dense_shards_are_byte_identical_to_the_writer(tmp_path: Path, replays_per_shard: int) -> None:
    index_path, tags_path = _write_source(tmp_path)
    record = repack([index_path], tags_path, tmp_path / "dense", replays_per_shard=replays_per_shard, seed=3)
    results = _results()
    dense_index = load_il_cache_index(record["dense_index"])
    assert len(record["assignment"]) == (1 if replays_per_shard == 3 else 2)
    assert dense_index["contract"] == {**_CONTRACT, "selected_replays": 3}
    for position, tags in enumerate(record["assignment"]):
        shard_id = f"dense-shard-{position:06d}"
        dense_summary = tmp_path / "dense" / "node-0" / f"{shard_id}.cril.json"
        reference_path = tmp_path / "reference" / f"{shard_id}.cril.zst"
        reference = write_il_cache_shard(
            reference_path,
            tuple(results[tag] for tag in tags),
            shard_id=shard_id,
            contract=_CONTRACT,
            compression_level=-5,
        )
        actual = json.loads(dense_summary.read_text(encoding="utf-8"))
        assert _container(dense_summary) == _container(reference_path.with_suffix(".json"))
        for key, value in reference.items():
            if key not in ("compressed_sha256", "compressed_bytes", "compression_ratio"):
                assert actual[key] == value, key
        row = dense_index["shards"][position]
        assert (row["summary_file"], row["host"], row["port"]) == (f"{shard_id}.cril.json", "dense-repack", 0)
    assert not (tmp_path / "dense" / ".spill").exists()


def test_verify_checks_sequences_planner_and_detects_changes(tmp_path: Path) -> None:
    index_path, tags_path = _write_source(tmp_path)
    output = tmp_path / "dense"
    repack([index_path], tags_path, output, replays_per_shard=2, seed=1)
    report = verify(output)
    assert report["ok"] and report["replays_checked"] == 3 and report["dense_shards"] == 2

    source_plans, source_meta = build_training_plan([index_path], tags_path, manifest_workers=1)
    dense_plans, dense_meta = build_training_plan([output / "node-0" / "index.json"], tags_path, manifest_workers=1)
    assert source_meta["selected_owner_frames"] == dense_meta["selected_owner_frames"]
    assert source_meta["cache_gamma_per_decision"].popitem()[1] == dense_meta["cache_gamma_per_decision"].popitem()[1]

    def sequences(plans):
        out = {}
        for plan in plans:
            shard = load_il_cache_shard(plan.summary_path)
            for unit in plan.units:
                out[unit.replay_tag] = tuple(shard.sequence(index) for index in unit.sequence_indices)
        return out

    left, right = sequences(source_plans), sequences(dense_plans)
    assert set(left) == set(right) == {"a", "b", "c"}
    for tag in left:
        for expected, actual in zip(left[tag], right[tag], strict=True):
            assert_il_sequences_exact(expected, actual)

    record_path = output / "repack.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["replays"]["b"]["digest"] = "0" * 64
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(AssertionError, match="repack record digest"):
        verify(output)


def test_evaluator_rows_match_between_source_and_dense(tmp_path: Path) -> None:
    index_path, tags_path = _write_source(tmp_path)
    output = tmp_path / "dense"
    repack([index_path], tags_path, output, replays_per_shard=3, seed=0)
    checkpoint = tmp_path / "checkpoint.pt"
    _save_checkpoint(checkpoint, 9, update_step=1)
    options = {"shards_per_batch": 8}
    (source_row,) = _evaluate(index_path, tags_path, [checkpoint], **options)
    (dense_row,) = _evaluate(output / "node-0" / "index.json", tags_path, [checkpoint], **options)
    source_row, dense_row = _without_timing(source_row), _without_timing(dense_row)
    for key in ("cache_indexes", "cache_index_sha256", "shards"):
        source_row.pop(key), dense_row.pop(key)
    assert source_row.keys() == dense_row.keys()
    for key, value in source_row.items():
        if isinstance(value, float):
            assert dense_row[key] == pytest.approx(value, rel=1e-6, abs=1e-9), key
        else:
            assert dense_row[key] == value, key


def test_refuses_existing_output_and_unknown_tags(tmp_path: Path) -> None:
    index_path, tags_path = _write_source(tmp_path)
    output = tmp_path / "dense"
    repack([index_path], tags_path, output, replays_per_shard=3)
    with pytest.raises(FileExistsError):
        repack([index_path], tags_path, output, replays_per_shard=3)
    missing = tmp_path / "missing.txt"
    missing.write_text("a\nzz\n", encoding="utf-8")
    with pytest.raises(ValueError, match="absent from caches"):
        repack([index_path], missing, tmp_path / "other")


_CACHE10 = Path(
    os.environ.get(
        "IL_CACHE10_DIR",
        "real-il-cache-not-configured",
    )
)


@pytest.mark.skipif(not (_CACHE10 / "index10.json").exists(), reason="local real cache (cache10) not available")
def test_real_cache_subset_with_short_last_shard(tmp_path: Path) -> None:
    index_path = _CACHE10 / "index10.json"
    tags = sorted(
        {
            row["replay_tag"]
            for summary in sorted(_CACHE10.glob("part-*.cril.json"))[:3]
            for row in load_il_cache_manifest(summary)["sequences"]
        }
    )
    random.Random(2).shuffle(tags)
    tags_path = tmp_path / "tags.txt"
    tags_path.write_text("".join(f"{tag}\n" for tag in sorted(tags[:21])), encoding="utf-8")
    output = tmp_path / "dense"
    record = repack([index_path], tags_path, output, replays_per_shard=16, seed=11, workers=2)
    assert [len(shard) for shard in record["assignment"]] == [16, 5]
    report = verify(output, workers=2)
    assert report["ok"] and report["replays_checked"] == 21 and report["source_shards"] == 3

    # Raw bytes of two replays against the source shard, frame by frame.
    dense_plans, _ = build_training_plan([output / "node-0" / "index.json"], tags_path, manifest_workers=2)
    source_plans, _ = build_training_plan([index_path], tags_path, manifest_workers=2)
    source_by_tag = {unit.replay_tag: plan for plan in source_plans for unit in plan.units}
    plan = dense_plans[-1]
    decoded = _decode_il_cache_shard(plan.summary_path, verify=True)
    dense = {spec.name: _tensor_from_spec(decoded.buffer, spec) for spec in decoded.specs}
    dense_rows = repack_module._replay_descriptors(decoded.manifest)
    for unit in plan.units[:2]:
        source_decoded = _decode_il_cache_shard(source_by_tag[unit.replay_tag].summary_path, verify=True)
        source = {spec.name: _tensor_from_spec(source_decoded.buffer, spec) for spec in source_decoded.specs}
        source_rows = repack_module._replay_descriptors(source_decoded.manifest)
        left, _ = repack_module.extract_replay(source, source_rows[unit.replay_tag])
        right, _ = repack_module.extract_replay(dense, dense_rows[unit.replay_tag])
        assert left.keys() == right.keys()
        for name in left:
            assert left[name].dtype == right[name].dtype and left[name].equal(right[name]), name


def _write_two_nodes(tmp_path: Path, *, manifest_override: dict, index_override: dict) -> tuple[list[Path], Path]:
    """node-0 holds (a, c), node-1 holds (b); node-1's contracts get the overrides."""

    results = _results()
    indexes = []
    for node, tags, manifest_extra, index_extra in (
        (0, ("a", "c"), {"partition_index": 0, "partition_replay_count": 2}, {"selected_replays": 2}),
        (1, ("b",), {"partition_index": 1, "partition_replay_count": 1, **manifest_override}, {"selected_replays": 1}),
    ):
        directory = tmp_path / "source" / f"node-{node}"
        directory.mkdir(parents=True)
        contract = {**_CONTRACT, **manifest_extra}
        data_path = directory / f"part-{node:02d}-shard-000000.cril.zst"
        summary = write_il_cache_shard(
            data_path, tuple(results[tag] for tag in tags), shard_id=data_path.stem, contract=contract
        )
        row = {**summary, "summary_file": data_path.with_suffix(".json").name, "host": "h", "port": 1}
        index_contract = {**contract, **index_extra, **(index_override if node == 1 else {})}
        write_il_cache_index(directory / "index.json", [row], contract=index_contract)
        indexes.append(directory / "index.json")
    tags_path = tmp_path / "tags.txt"
    tags_path.write_text("a\nb\nc\n", encoding="utf-8")
    return indexes, tags_path


def test_partition_fields_that_vary_per_node_are_tolerated_and_recorded(tmp_path: Path) -> None:
    indexes, tags_path = _write_two_nodes(tmp_path, manifest_override={}, index_override={})
    output = tmp_path / "dense"
    record = repack(indexes, tags_path, output, replays_per_shard=3, seed=0)
    assert verify(output)["ok"]
    variations = record["source_manifest_contract_variations"]
    assert set(variations) == {"partition_index", "partition_replay_count"}
    assert {json.loads(value) for value in variations["partition_index"]} == {0, 1}
    assert set(record["source_index_contract_variations"]) == {
        "partition_index",
        "partition_replay_count",
        "selected_replays",
    }
    (sources,) = record["dense_shard_contract_sources"].values()
    assert sources["partition_index"] == ["0", "1"]
    # The dense shard carries the first source's contract; the index the first index's.
    first = load_il_cache_manifest(Path(record["source_manifest_contract_first_source"]))["contract"]
    dense_summary = output / "node-0" / "dense-shard-000000.cril.json"
    assert load_il_cache_manifest(dense_summary)["contract"] == first
    assert (
        load_il_cache_index(output / "node-0" / "index.json")["contract"] == load_il_cache_index(indexes[0])["contract"]
    )


@pytest.mark.parametrize(
    ("manifest_override", "index_override", "key"),
    (
        ({"gamma_per_decision": 0.995}, {}, "gamma_per_decision"),
        ({}, {"gamma_per_decision": 0.995}, "gamma_per_decision"),
        ({"storage_float_dtype": "float32"}, {}, "storage_float_dtype"),
    ),
)
def test_semantic_contract_differences_fail_before_the_source_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, manifest_override: dict, index_override: dict, key: str
) -> None:
    indexes, tags_path = _write_two_nodes(tmp_path, manifest_override=manifest_override, index_override=index_override)

    def source_pass_must_not_run(*_args, **_kwargs):
        raise AssertionError("the source pass started before the contract check")

    monkeypatch.setattr(repack_module, "source_task", source_pass_must_not_run)
    output = tmp_path / "dense"
    with pytest.raises(ValueError, match=key) as error:
        repack(indexes, tags_path, output, replays_per_shard=3)
    assert "0.99" in str(error.value) or "float32" in str(error.value)
    assert "partition_index" not in str(error.value).split("(tolerated")[0]
    assert not (output / "node-0").exists()


def test_extra_tolerated_contract_keys_can_be_named(tmp_path: Path) -> None:
    indexes, tags_path = _write_two_nodes(tmp_path, manifest_override={"fixture": "node-1 note"}, index_override={})
    with pytest.raises(ValueError, match="fixture"):
        repack(indexes, tags_path, tmp_path / "strict", replays_per_shard=3)
    record = repack(indexes, tags_path, tmp_path / "lenient", replays_per_shard=3, tolerate_contract_keys=["fixture"])
    assert "fixture" in record["source_manifest_contract_variations"]
    assert verify(tmp_path / "lenient")["ok"]


def test_check_contracts_digests_long_values_and_lists_examples() -> None:
    long_list = [{"replay_tag": f"t{index}"} for index in range(100)]
    contracts = {
        "node-0": {"gamma_per_decision": 0.99, "selected_replays": long_list, "partition_index": 0},
        "node-1": {"gamma_per_decision": 0.99, "selected_replays": long_list[:50], "partition_index": 1},
    }
    first, variations = repack_module.check_contracts(
        contracts, tolerated=repack_module.TOLERATED_CONTRACT_KEYS, what="test"
    )
    assert first == contracts["node-0"]
    assert all(value.startswith("sha256:") for value in variations["selected_replays"])
    assert variations["partition_index"] == {"0": ["node-0"], "1": ["node-1"]}
    contracts["node-1"]["gamma_per_decision"] = 0.995
    with pytest.raises(ValueError, match=r"gamma_per_decision: 0.99 in 1 \(node-0\); 0.995 in 1 \(node-1\)"):
        repack_module.check_contracts(contracts, tolerated=repack_module.TOLERATED_CONTRACT_KEYS, what="test")
