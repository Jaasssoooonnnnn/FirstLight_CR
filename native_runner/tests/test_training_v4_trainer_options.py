from __future__ import annotations

from dataclasses import asdict, fields, replace
import inspect
from pathlib import Path
import typing

import pytest
import torch

from native_runner.tests.test_training_v4_cache import _result
from native_runner.tests.test_training_v4_evaluate_imitation_cache import (
    _evaluate,
    _fixture_lanes,
    _save_checkpoint,
)
from native_runner.tests.test_training_v4_model import _model
from native_runner.training.v4.cache import load_il_cache_shards, write_il_cache_index, write_il_cache_shard
from native_runner.training.v4.checkpoint import (
    load_actor_critic_checkpoint,
    model_config_from_contract,
    save_actor_critic_checkpoint,
)
from native_runner.training.v4.config import ModelConfigV4
from native_runner.training.v4.evaluate_imitation_cache import _model_for_config
from native_runner.training.v4.factory import build_production_model_v4, model_config_from_overrides
from native_runner.training.v4.imitation import (
    ILLossWeightsV4,
    ILSequenceV4,
    collate_padded_il_sequences,
    imitation_loss,
    slice_il_sequence,
)
from native_runner.training.v4.learning import evaluate_recurrent_sequence
from native_runner.training.v4.train_imitation_cache import (
    ReplayWeightsV1,
    _distributed_loss,
    _masked_chunk,
    build_training_plan,
    chunk_loss_scale,
    load_model_config_overrides,
    load_replay_weights,
    replay_weight_summary,
    require_cache_gamma,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_cache(
    tmp_path: Path, *, contract: dict[str, object], tags: tuple[str, ...] = ("a", "b", "c"), name: str = "cache"
) -> tuple[Path, Path]:
    """Two shards: (a) and (b, c); the tag file selects ``tags``."""

    lanes = _fixture_lanes()
    directory = tmp_path / name
    directory.mkdir()
    summaries = []
    for shard_index, shard_tags in enumerate((("a",), ("b", "c"))):
        data_path = directory / f"shard-{shard_index:05d}.cril.zst"
        summary = write_il_cache_shard(
            data_path,
            tuple(_result(tag, *lanes[tag]) for tag in shard_tags),
            shard_id=f"{name}-{shard_index:05d}",
            contract=contract,
        )
        summary["summary_file"] = data_path.with_suffix(".json").name
        summaries.append(summary)
    index_path = directory / "index.json"
    write_il_cache_index(index_path, summaries, contract=contract)
    tags_path = tmp_path / f"{name}-tags.txt"
    tags_path.write_text("".join(f"{tag}\n" for tag in tags), encoding="utf-8")
    return index_path, tags_path


@pytest.fixture
def single_process_group(tmp_path: Path):
    import torch.distributed as dist

    if dist.is_initialized():
        pytest.skip("a process group is already initialized")
    dist.init_process_group(
        "gloo", init_method=(tmp_path / "process-group").resolve().as_uri(), rank=0, world_size=1
    )
    try:
        yield
    finally:
        dist.destroy_process_group()


def _whole_chunk(tmp_path: Path, **chunk_options: object) -> tuple[ILSequenceV4, int]:
    index_path, tags_path = _write_cache(tmp_path, contract={})
    plans, _metadata = build_training_plan([index_path], tags_path, manifest_workers=1)
    shards = load_il_cache_shards([plan.summary_path for plan in plans])
    return _masked_chunk(
        shards, plans, start=0, time_steps=5, device=torch.device("cpu"), dummy=False, **chunk_options  # type: ignore[arg-type]
    )


def _evaluate_chunk(sequence: ILSequenceV4, *, seed: int = 3):
    torch.manual_seed(seed)
    model = _model().eval()
    with torch.no_grad():
        return evaluate_recurrent_sequence(
            model,
            sequence.observations,
            sequence.actions,
            sequence.episode_start,
            gate_temperature=1.0,
            action_temperature=1.0,
            continue_temperature=1.0,
        )


_HEADS = ("gate", "candidate", "target", "delay", "continue_action", "value")
_COUNTS = ("gate_count", "candidate_count", "target_count", "delay_count", "continue_count", "value_count")
_WEIGHTED = tuple(name.replace("_count", "_weighted_count") for name in _COUNTS)


# ---------------------------------------------------------------------------
# M1: checkpoint contract tolerance
# ---------------------------------------------------------------------------


def _rewrite_contract(tmp_path: Path, mutate) -> Path:
    torch.manual_seed(4)
    source = tmp_path / "source.pt"
    save_actor_critic_checkpoint(source, _model(), training_stage="imitation", gamma_per_decision=0.99)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    mutate(payload["contract"])
    rewritten = tmp_path / "rewritten.pt"
    torch.save(payload, rewritten)
    return rewritten


def test_checkpoint_missing_model_config_key_is_filled_with_current_default(tmp_path: Path) -> None:
    path = _rewrite_contract(tmp_path, lambda contract: contract["model_config"].pop("initial_second_probability"))
    stored = torch.load(path, map_location="cpu", weights_only=False)
    assert "initial_second_probability" not in stored["contract"]["model_config"]

    model = _model()
    payload = load_actor_critic_checkpoint(path, model, map_location="cpu")

    assert payload["checkpoint_id"] == stored["checkpoint_id"]
    for name, value in model.state_dict().items():
        assert torch.equal(value, stored["model_state_dict"][name]), name
    assert model_config_from_contract(stored["contract"]) == ModelConfigV4()


def test_checkpoint_unknown_model_config_key_is_rejected(tmp_path: Path) -> None:
    path = _rewrite_contract(tmp_path, lambda contract: contract["model_config"].update(future_switch=True))
    with pytest.raises(ValueError, match="future_switch"):
        load_actor_critic_checkpoint(path, _model(), map_location="cpu")
    with pytest.raises(ValueError, match="future_switch"):
        model_config_from_contract(torch.load(path, map_location="cpu", weights_only=False)["contract"])


def test_checkpoint_differing_model_config_key_is_rejected(tmp_path: Path) -> None:
    path = _rewrite_contract(tmp_path, lambda contract: contract["model_config"].update(lstm_hidden_dim=1024))
    with pytest.raises(ValueError, match=r"model_config\.lstm_hidden_dim"):
        load_actor_critic_checkpoint(path, _model(), map_location="cpu")


def test_checkpoint_other_contract_fields_stay_strict(tmp_path: Path) -> None:
    def mutate(contract: dict) -> None:
        contract["model_config"].pop("initial_second_probability")
        contract["catalog_ids"]["card"] = "another-catalog"

    path = _rewrite_contract(tmp_path, mutate)
    with pytest.raises(ValueError, match="catalog_ids"):
        load_actor_critic_checkpoint(path, _model(), map_location="cpu")


def test_model_config_from_contract_restores_non_default_config() -> None:
    config = replace(ModelConfigV4(), lstm_hidden_dim=256)
    stored = asdict(config)
    stored["delay_offset_ms"] = list(stored["delay_offset_ms"])
    del stored["initial_act_probability"]
    assert model_config_from_contract({"model_config": stored}) == config
    with pytest.raises(ValueError, match="model_config"):
        model_config_from_contract({})


# ---------------------------------------------------------------------------
# M5: model config overrides
# ---------------------------------------------------------------------------


def test_model_config_overrides_are_type_checked() -> None:
    assert model_config_from_overrides({}) == ModelConfigV4()
    config = model_config_from_overrides(
        {"lstm_hidden_dim": 1024, "initial_act_probability": 0.2, "delay_offset_ms": [0, 50, 100, 150, 200]}
    )
    assert config == replace(ModelConfigV4(), lstm_hidden_dim=1024, initial_act_probability=0.2)
    assert isinstance(model_config_from_overrides({"spatial_res_blocks": 2}).spatial_res_blocks, int)

    with pytest.raises(ValueError, match="unknown ModelConfigV4 override fields: \\['not_a_field'\\]"):
        model_config_from_overrides({"not_a_field": 1})
    for bad in (
        {"lstm_hidden_dim": 1.5},
        {"lstm_hidden_dim": True},
        {"lstm_hidden_dim": "512"},
        {"initial_act_probability": "0.2"},
        {"initial_act_probability": float("nan")},
        {"delay_offset_ms": 0},
        {"delay_offset_ms": [0, 50.0, 100, 150, 200]},
    ):
        with pytest.raises(ValueError, match=next(iter(bad))):
            model_config_from_overrides(bad)
    # Values pass type checks but still go through ModelConfigV4 validation.
    with pytest.raises(ValueError):
        model_config_from_overrides({"lstm_hidden_dim": 0})


def test_model_config_overrides_check_bool_and_str_fields() -> None:
    annotations = typing.get_type_hints(ModelConfigV4)
    bool_fields = [item.name for item in fields(ModelConfigV4) if annotations[item.name] is bool]
    str_fields = [item.name for item in fields(ModelConfigV4) if annotations[item.name] is str]
    for name in bool_fields:
        with pytest.raises(ValueError, match=name):
            model_config_from_overrides({name: 1})
    for name in str_fields:
        with pytest.raises(ValueError, match=name):
            model_config_from_overrides({name: 1})


def test_model_config_json_inline_and_file(tmp_path: Path) -> None:
    assert load_model_config_overrides('{"lstm_hidden_dim": 1024}') == {"lstm_hidden_dim": 1024}
    path = tmp_path / "overrides.json"
    path.write_text('{"decoder_hidden_dim": 256}', encoding="utf-8")
    assert load_model_config_overrides(f"@{path}") == {"decoder_hidden_dim": 256}
    with pytest.raises(ValueError, match="JSON object"):
        load_model_config_overrides("[1, 2]")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_model_config_overrides("{lstm_hidden_dim: 1}")


def test_build_production_model_accepts_config() -> None:
    parameters = inspect.signature(build_production_model_v4).parameters
    assert parameters["config"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["config"].default is None


def test_evaluation_rebuilds_model_from_each_checkpoint_config(tmp_path: Path) -> None:
    index_path, tags_path = _write_cache(tmp_path, contract={}, tags=("a", "b"))
    default_checkpoint = tmp_path / "default.pt"
    _save_checkpoint(default_checkpoint, 11, update_step=7)
    small_checkpoint = tmp_path / "small-lstm.pt"
    small_config = replace(ModelConfigV4(), lstm_hidden_dim=256)
    torch.manual_seed(23)
    small_model = _model_for_config(small_config, template=_model(), device=torch.device("cpu"))
    save_actor_critic_checkpoint(
        small_checkpoint, small_model, update_step=9, training_stage="imitation", gamma_per_decision=0.99
    )

    (alone_default,) = _evaluate(index_path, tags_path, [default_checkpoint])
    (alone_small,) = _evaluate(index_path, tags_path, [small_checkpoint])
    joint = _evaluate(index_path, tags_path, [small_checkpoint, default_checkpoint])

    assert alone_small["model_config"]["lstm_hidden_dim"] == 256  # type: ignore[index]
    assert alone_default["model_config"] == asdict(ModelConfigV4())
    for alone, combined in zip((alone_small, alone_default), joint, strict=True):
        assert combined["checkpoint_id"] == alone["checkpoint_id"]
        for name, value in alone.items():
            if name in {"checkpoint_eval_s", "cache_load_s", "elapsed_s"}:
                continue
            if isinstance(value, float):
                assert combined[name] == pytest.approx(value, abs=1e-5), name
            else:
                assert combined[name] == value, name


# ---------------------------------------------------------------------------
# S5: replay weights
# ---------------------------------------------------------------------------


def _weights_file(tmp_path: Path, rows: list[str], *, header: str = "replay_tag\towner\tweight") -> Path:
    path = tmp_path / "weights.tsv"
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


def test_replay_weight_file_parsing(tmp_path: Path) -> None:
    weights = load_replay_weights(_weights_file(tmp_path, ["a\t0\t2", "b\t*\t0.5"]), strict=False)
    assert weights.lane_weight("a", 0) == 2.0
    assert weights.lane_weight("a", 1) == 1.0
    assert weights.lane_weight("b", 0) == weights.lane_weight("b", 1) == 0.5
    assert weights.lane_weight("zzz", 1) == 1.0
    strict = replace(weights, strict=True)
    with pytest.raises(ValueError, match="replay a owner 1"):
        strict.lane_weight("a", 1)

    for rows, message in (
        (["a\t0\t-1"], ">= 0"),
        (["a\t0\tinf"], ">= 0"),
        (["a\t0\tx"], "not a number"),
        (["a\t2\t1"], "owner must be 0, 1 or"),
        (["a\t0"], "malformed"),
        (["a\t0\t1", "a\t0\t2"], "duplicates"),
        (["a\t*\t1", "a\t1\t2"], "both a \\* row"),
    ):
        with pytest.raises(ValueError, match=message):
            load_replay_weights(_weights_file(tmp_path, rows), strict=False)
    with pytest.raises(ValueError, match="header"):
        load_replay_weights(_weights_file(tmp_path, ["a\t0\t1"], header="tag\towner\tweight"), strict=False)


def test_replay_weights_build_lane_frame_weights_with_zero_padding(tmp_path: Path) -> None:
    index_path, tags_path = _write_cache(tmp_path, contract={})
    plans, _metadata = build_training_plan([index_path], tags_path, manifest_workers=1)
    shards = load_il_cache_shards([plan.summary_path for plan in plans])
    weights = ReplayWeightsV1(weights={("a", 0): 2.0, ("b", None): 0.5, ("c", 1): 3.0})

    # Lane order: a0, a1 (shard 0); b0, b1, c0, c1 (shard 1).  Lengths 5, 3, 4, 4, 2, 2.
    sequence, frames = _masked_chunk(
        shards, plans, start=3, time_steps=3, device=torch.device("cpu"), dummy=False, replay_weights=weights
    )
    assert frames == 2 + 0 + 1 + 1 + 0 + 0
    assert sequence.frame_weight is not None and sequence.frame_weight.dtype == torch.float32
    expected = torch.tensor(
        [
            [2.0, 0.0, 0.5, 0.5, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    assert torch.equal(sequence.frame_weight, expected)

    unweighted, _frames = _masked_chunk(shards, plans, start=3, time_steps=3, device=torch.device("cpu"), dummy=False)
    assert unweighted.frame_weight is None

    summary = replay_weight_summary(weights, plans)
    assert summary["replay_weight_explicit_lanes"] == 4
    assert summary["replay_weight_default_lanes"] == 2
    assert summary["replay_weight_rows_outside_split"] == 0
    with pytest.raises(ValueError, match="replay a owner 1"):
        replay_weight_summary(replace(weights, strict=True), plans)
    with pytest.raises(ValueError, match="replay a owner 1"):
        _masked_chunk(
            shards,
            plans,
            start=0,
            time_steps=3,
            device=torch.device("cpu"),
            dummy=False,
            replay_weights=replace(weights, strict=True),
        )


def test_frame_weight_follows_slicing_collation_and_validation(tmp_path: Path) -> None:
    sequence, _frames = _whole_chunk(tmp_path)
    weighted = replace(sequence, frame_weight=torch.full(sequence.valid_mask.shape, 0.5))
    weighted.validate()
    sliced = slice_il_sequence(weighted, 1, 3)
    assert sliced.frame_weight is not None and torch.equal(sliced.frame_weight, weighted.frame_weight[1:3])
    moved = weighted.to("cpu")
    assert moved.frame_weight is not None and torch.equal(moved.frame_weight, weighted.frame_weight)

    short = slice_il_sequence(sequence, 0, 2)
    collated = collate_padded_il_sequences((weighted, short))
    assert collated.frame_weight is not None
    assert torch.equal(collated.frame_weight[:, :6], weighted.frame_weight)
    assert torch.equal(collated.frame_weight[:2, 6:], torch.ones(2, 6))
    assert torch.equal(collated.frame_weight[2:, 6:], torch.zeros(3, 6))
    assert collate_padded_il_sequences((sequence, short)).frame_weight is None

    for bad in (
        torch.full(sequence.valid_mask.shape, -1.0),
        torch.full(sequence.valid_mask.shape, float("nan")),
        torch.ones(sequence.valid_mask.shape, dtype=torch.float64),
        torch.ones(1, 6),
    ):
        with pytest.raises(ValueError, match="frame_weight"):
            replace(sequence, frame_weight=bad).validate()


def test_uniform_frame_weight_reproduces_unweighted_loss_bit_for_bit(tmp_path: Path) -> None:
    sequence, _frames = _whole_chunk(tmp_path)
    evaluation = _evaluate_chunk(sequence)
    plain = imitation_loss(sequence, evaluation)
    ones = imitation_loss(replace(sequence, frame_weight=torch.ones(sequence.valid_mask.shape)), evaluation)
    for item in fields(plain):
        assert torch.equal(getattr(ones, item.name), getattr(plain, item.name)), item.name
    for count, weighted in zip(_COUNTS, _WEIGHTED, strict=True):
        assert torch.equal(getattr(plain, weighted), getattr(plain, count))


def test_doubling_one_lane_reweights_only_heads_that_contain_it(tmp_path: Path) -> None:
    sequence, _frames = _whole_chunk(tmp_path)
    evaluation = _evaluate_chunk(sequence)
    lane = 5  # c:owner-1 = [WAIT, WAIT]: gate and value frames only.
    shape = sequence.valid_mask.shape

    def loss_with(lane_weights: list[float]):
        frame_weight = torch.tensor(lane_weights, dtype=torch.float32)[None, :].expand(shape).contiguous()
        return imitation_loss(replace(sequence, frame_weight=frame_weight), evaluation)

    plain = imitation_loss(sequence, evaluation)
    doubled = loss_with([1.0] * lane + [2.0])
    rest = loss_with([1.0] * lane + [0.0])
    only = loss_with([0.0] * lane + [1.0])

    for head, count, weighted in zip(_HEADS, _COUNTS, _WEIGHTED, strict=True):
        assert torch.equal(getattr(doubled, count), getattr(plain, count)), count
        lane_weight = float(getattr(only, weighted))
        if lane_weight == 0.0:
            assert torch.equal(getattr(doubled, head), getattr(plain, head)), head
            continue
        rest_weight = float(getattr(rest, weighted))
        expected = (float(getattr(rest, head)) * rest_weight + 2.0 * float(getattr(only, head)) * lane_weight) / (
            rest_weight + 2.0 * lane_weight
        )
        assert float(getattr(doubled, weighted)) == pytest.approx(rest_weight + 2.0 * lane_weight)
        assert float(getattr(doubled, head)) == pytest.approx(expected, rel=1e-6), head
        assert float(getattr(doubled, head)) != pytest.approx(float(getattr(plain, head)), rel=1e-9), head
    assert float(only.candidate_weighted_count) == 0.0
    assert float(only.gate_weighted_count) == 2.0 and float(only.value_weighted_count) == 2.0

    # Value head by hand: weighted Huber mean over valid frames.
    raw = torch.nn.functional.smooth_l1_loss(evaluation.value, sequence.returns, reduction="none")
    weight = sequence.value_loss_mask.to(torch.float32) * torch.tensor([1.0] * lane + [2.0])[None, :]
    torch.testing.assert_close(doubled.value, (raw * weight).sum() / weight.sum())


def test_distributed_loss_uses_weighted_population(tmp_path: Path, single_process_group: None) -> None:
    sequence, _frames = _whole_chunk(tmp_path)
    evaluation = _evaluate_chunk(sequence)
    weights = ILLossWeightsV4()
    plain_total, plain_metrics = _distributed_loss(imitation_loss(sequence, evaluation), weights, world_size=1)
    ones = replace(sequence, frame_weight=torch.ones(sequence.valid_mask.shape))
    ones_total, ones_metrics = _distributed_loss(imitation_loss(ones, evaluation), weights, world_size=1)
    assert torch.equal(ones_total, plain_total)
    assert ones_metrics == plain_metrics

    doubled = replace(sequence, frame_weight=torch.tensor([1.0] * 5 + [2.0])[None, :].expand(5, 6).contiguous())
    losses = imitation_loss(doubled, evaluation)
    total, metrics = _distributed_loss(losses, weights, world_size=1)
    torch.testing.assert_close(total, losses.total)
    assert metrics["gate"] == pytest.approx(float(losses.gate))
    assert metrics["gate_count"] == plain_metrics["gate_count"]


# ---------------------------------------------------------------------------
# S4: chunk normalization
# ---------------------------------------------------------------------------


def test_chunk_loss_scale_averages_to_one_over_a_group() -> None:
    frames = (12, 4)
    scales = [
        chunk_loss_scale("group-expected", chunk_frames=value, group_owner_frames=16, chunk_count=2)
        for value in frames
    ]
    assert scales == [1.5, 0.5]
    assert sum(scales) == len(frames)
    assert chunk_loss_scale("per-chunk", chunk_frames=4, group_owner_frames=16, chunk_count=2) == 1.0
    with pytest.raises(ValueError, match="chunk normalization"):
        chunk_loss_scale("per-frame", chunk_frames=4, group_owner_frames=16, chunk_count=2)
    with pytest.raises(ValueError, match="positive"):
        chunk_loss_scale("group-expected", chunk_frames=4, group_owner_frames=0, chunk_count=2)


def test_group_expected_normalization_rescales_each_chunk(tmp_path: Path, single_process_group: None) -> None:
    index_path, tags_path = _write_cache(tmp_path, contract={}, tags=("a", "b"))
    plans, _metadata = build_training_plan([index_path], tags_path, manifest_workers=1)
    shards = load_il_cache_shards([plan.summary_path for plan in plans])
    group_owner_frames = sum(plan.owner_frames for plan in plans)
    max_length = max(length for plan in plans for length in plan.sequence_lengths)
    time_steps = 3
    chunk_count = -(-max_length // time_steps)
    assert (group_owner_frames, chunk_count) == (16, 2)

    torch.manual_seed(3)
    model = _model().eval()
    state = model.initial_state(4)
    weights = ILLossWeightsV4()
    scales = []
    for chunk_index in range(chunk_count):
        sequence, frames = _masked_chunk(
            shards, plans, start=chunk_index * time_steps, time_steps=time_steps, device=torch.device("cpu"), dummy=False
        )
        with torch.no_grad():
            evaluation = evaluate_recurrent_sequence(
                model, sequence.observations, sequence.actions, sequence.episode_start, initial_state=state
            )
        state = evaluation.final_state
        losses = imitation_loss(sequence, evaluation)
        per_chunk, per_chunk_metrics = _distributed_loss(
            losses,
            weights,
            world_size=1,
            loss_scale=chunk_loss_scale(
                "per-chunk", chunk_frames=frames, group_owner_frames=group_owner_frames, chunk_count=chunk_count
            ),
        )
        scale = chunk_loss_scale(
            "group-expected", chunk_frames=frames, group_owner_frames=group_owner_frames, chunk_count=chunk_count
        )
        group_expected, group_expected_metrics = _distributed_loss(losses, weights, world_size=1, loss_scale=scale)
        scales.append(scale)
        assert frames == (12, 4)[chunk_index]
        torch.testing.assert_close(group_expected, per_chunk * scale)
        assert group_expected_metrics.pop("chunk_loss_scale") == scale
        assert per_chunk_metrics.pop("chunk_loss_scale") == 1.0
        assert group_expected_metrics == per_chunk_metrics
    assert scales == [1.5, 0.5]


# ---------------------------------------------------------------------------
# A-7: gamma consistency
# ---------------------------------------------------------------------------


def test_trainer_gamma_must_match_cache_contract(tmp_path: Path) -> None:
    index_path, tags_path = _write_cache(tmp_path, contract={"gamma_per_decision": 0.99}, tags=("a",))
    _plans, metadata = build_training_plan([index_path], tags_path, manifest_workers=1)
    assert metadata["cache_gamma_per_decision"] == {str(index_path): 0.99}
    assert require_cache_gamma(metadata, 0.99) == 0.99
    with pytest.raises(ValueError, match="differs from cache gamma_per_decision"):
        require_cache_gamma(metadata, 0.995)

    legacy_index, legacy_tags = _write_cache(tmp_path, contract={}, tags=("b",), name="legacy")
    _plans, legacy_metadata = build_training_plan(
        [index_path, legacy_index], legacy_tags, manifest_workers=1
    )
    assert legacy_metadata["cache_gamma_per_decision"] == {str(index_path): 0.99, str(legacy_index): None}
    with pytest.raises(ValueError, match="no gamma_per_decision"):
        require_cache_gamma(legacy_metadata, 0.99)
