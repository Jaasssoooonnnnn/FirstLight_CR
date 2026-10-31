"""``delay_policy="earliest"``: play each micro-action at its first legal delay bin.

The IL delay head only learned the (noise) prior of the expert's touch phase,
so inference can ignore it and touch at the earliest bin that the shadow's
delay-aware, sequential, order-preserving legality allows.  ``"sample"`` (the
default) must stay bit-identical to the historical decoder.
"""

from __future__ import annotations

from dataclasses import fields, replace
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from native_runner.tests.test_training_v4_c1_window_affordability import TWO_X_TICK, _tensorized
from native_runner.tests.test_training_v4_delay_aware_legality import EXPENSIVE, _with_elixir
from native_runner.tests.test_training_v4_model import _batch, _model, _with_candidate_mask
from native_runner.training.v4 import async_cluster_self_play as collector
from native_runner.training.v4 import evaluate_head_to_head as h2h
from native_runner.training.v4 import policy_session
from native_runner.training.v4 import make_head_to_head_schedule as schedule_module
from native_runner.training.v4.decoding import (
    DELAY_POLICIES,
    ShadowCandidateLegality,
    decode_action_sequence_v4,
    earliest_delay_offset_bin,
    elixir_multiplier_from_match_scalars,
    validate_delay_policy,
)
from native_runner.training.v4.factory import build_production_model_v4
from native_runner.training.v4.tensors import GATE_ACT, GATE_WAIT


ABILITY = 2  # _batch candidate row 2: target-free ability, cost 1.
CHEAP = 0  # _batch candidate row 0: DEPLOY, cost 3, GRID.


def _only(batch: Any, rows: tuple[int, ...]) -> Any:
    mask = torch.zeros_like(batch.candidates.mask)
    for row in rows:
        mask[0, row] = True
    return replace(batch, candidates=_with_candidate_mask(batch.candidates, mask))


def _rigged(*, delay_bias: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 20.0), second: float = -20.0) -> Any:
    """Always ACT; the delay head strongly prefers ``delay_bias``'s argmax."""

    torch.manual_seed(3)
    model = _model().eval()
    with torch.no_grad():
        model.gate_head[-1].bias.copy_(torch.tensor([-20.0, 20.0]))
        model.continue_head[-1].weight.zero_()
        model.continue_head[-1].bias.copy_(torch.tensor([0.0, second]))
        delay_head = model.delay_offset_head[-1]
        assert isinstance(delay_head, torch.nn.Linear)
        delay_head.weight.zero_()
        delay_head.bias.copy_(torch.tensor(delay_bias))
    return model


def _act(model: Any, batch: Any) -> Any:
    with torch.no_grad():
        return model.act(batch, model.initial_state(batch.batch_size))


def _sample(model: Any, batch: Any, rows: int, *, seed: int = 11) -> tuple[Any, Any]:
    repeated = batch.index_select(torch.zeros(rows, dtype=torch.long))
    torch.manual_seed(seed)
    with torch.no_grad():
        return repeated, model.sample_for_ppo_rollout(repeated, model.initial_state(rows))


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


def test_earliest_bin_is_the_first_legal_bin_and_empty_rows_fall_back_to_zero() -> None:
    mask = torch.tensor(
        [
            [True, True, True, True, True],
            [False, False, True, True, True],
            [False, True, False, True, False],
            [False, False, False, False, True],
            [False, False, False, False, False],
        ]
    )
    assert earliest_delay_offset_bin(mask).tolist() == [0, 2, 1, 4, 0]
    with pytest.raises(ValueError):
        earliest_delay_offset_bin(mask.long())
    assert DELAY_POLICIES == ("sample", "earliest")
    assert validate_delay_policy("earliest") == "earliest"
    for bad in ("argmax", "", None, 0):
        with pytest.raises(ValueError):
            validate_delay_policy(bad)  # type: ignore[arg-type]
    model = _model()
    assert model.delay_policy == "sample"
    with pytest.raises(ValueError):
        model.set_delay_policy("first")
    model.set_delay_policy("earliest")
    assert model.delay_policy == "earliest"
    # Runtime-only: nothing new in the state dict.
    assert set(model.state_dict()) == set(_model().state_dict())


# (a) all bins legal -> bin 0 even though the head prefers bin 4.
def test_earliest_picks_bin_zero_when_every_bin_is_legal() -> None:
    model = _rigged()
    batch = _only(_batch(batch_size=1), (EXPENSIVE,))  # 10 elixir, cost 9
    assert _act(model, batch).actions.delay_offset_bin.tolist() == [[4, -1]]
    model.set_delay_policy("earliest")
    assert _act(model, batch).actions.delay_offset_bin.tolist() == [[0, -1]]
    _rows, output = _sample(model, batch, 64)
    assert output.actions.delay_offset_bin[:, 0].tolist() == [0] * 64


# (b) window-affordable candidate -> its first affordable bin.
@pytest.mark.parametrize(
    ("elixir", "multiplier", "expected"),
    ((9.0 - 0.015, 1.0, 1), (9.0 - 0.03, 1.0, 2), (9.0 - 0.06, 1.0, 4), (9.0 - 0.03, 2.0, 1), (9.0 - 0.05, 3.0, 1)),
)
def test_earliest_picks_the_first_affordable_bin_of_a_window_affordable_candidate(
    elixir: float, multiplier: float, expected: int
) -> None:
    batch = _only(_with_elixir(_batch(batch_size=1), elixir, multiplier=multiplier), (EXPENSIVE,))
    shadow = ShadowCandidateLegality(
        batch.candidates, _model().config, elixir_multiplier=elixir_multiplier_from_match_scalars(batch.match_scalars)
    )
    bins = shadow.delay_offset_mask(step=0, candidate_index=torch.tensor([EXPENSIVE]))[0].tolist()
    assert bins.index(True) == expected
    for bias in ((20.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 20.0)):
        model = _rigged(delay_bias=bias)
        model.set_delay_policy("earliest")
        assert _act(model, batch).actions.delay_offset_bin.tolist() == [[expected, -1]]
        _rows, output = _sample(model, batch, 32)
        assert output.actions.delay_offset_bin[:, 0].tolist() == [expected] * 32


def test_earliest_respects_the_inference_delay_cap() -> None:
    model = _rigged()
    model.set_delay_policy("earliest")
    model.set_inference_delay_offset_max_ms(50)
    bin_one = _only(_with_elixir(_batch(batch_size=1), 9.0 - 0.015), (EXPENSIVE,))
    bin_two = _only(_with_elixir(_batch(batch_size=1), 9.0 - 0.03), (EXPENSIVE,))
    assert _act(model, bin_one).actions.delay_offset_bin.tolist() == [[1, -1]]
    # Affordable only beyond the cap: not a candidate at all (unchanged rule).
    assert _act(model, bin_two).actions.gate.tolist() == [GATE_WAIT]


# (c) second micro-action: sequential elixir legality and action order.
def test_second_micro_action_gets_its_sequential_earliest_bin() -> None:
    # e0 = 9.97 at 1x: the 9-cost card alone is affordable at bin 0, but after
    # the 1-cost ability (reserved) the pair needs 10 -> bin 2, either order.
    batch = _only(_with_elixir(_batch(batch_size=1), 9.97), (ABILITY, EXPENSIVE))
    model = _rigged(second=20.0)
    head = _act(model, batch).actions
    assert head.micro_action_count.tolist() == [2] and head.delay_offset_bin.tolist() == [[4, 4]]
    model.set_delay_policy("earliest")
    chosen = _act(model, batch).actions
    assert chosen.micro_action_count.tolist() == [2]
    assert sorted(chosen.candidate_index[0].tolist()) == [ABILITY, EXPENSIVE]
    assert chosen.delay_offset_bin.tolist() == [[0, 2]]
    rows, output = _sample(model, batch, 64)
    assert output.actions.micro_action_count.tolist() == [2] * 64
    assert output.actions.delay_offset_bin.tolist() == [[0, 2]] * 64
    # Order preservation (second >= first) is implied here: elixir only
    # grows across the window and the second touch adds cost, so no SECOND
    # candidate can have a sequential earliest bin before the first touch.
    # Independent replay: every pair is legal, the first bin is the
    # standalone earliest and the second is the earliest after the spend.
    shadow = ShadowCandidateLegality(
        rows.candidates, model.config, elixir_multiplier=elixir_multiplier_from_match_scalars(rows.match_scalars)
    )
    alone = shadow.affordable_at_bins()[:, EXPENSIVE].clone()
    assert alone[0].tolist() == [True] * 5
    active = torch.ones(64, dtype=torch.bool)
    for step in range(2):
        index = output.actions.candidate_index[:, step]
        delay = output.actions.delay_offset_bin[:, step]
        legal = shadow.delay_offset_mask(step=step, candidate_index=index)
        assert bool(legal[torch.arange(64), delay].all())
        assert torch.equal(delay, legal.long().argmax(dim=-1))
        shadow.apply(active, index, output.actions.target_cell[:, step], delay, step=step)


# (d) legacy (decision-tick legality) checkpoints always get bin 0.
def test_legacy_legality_checkpoints_always_get_bin_zero() -> None:
    batch = _only(_batch(batch_size=1), (ABILITY, CHEAP))  # 10 elixir, costs 1 + 3
    model = _rigged(second=20.0)
    model.config = replace(model.config, delay_aware_legality=False)
    assert _act(model, batch).actions.delay_offset_bin.tolist() == [[4, 4]]
    model.set_delay_policy("earliest")
    assert _act(model, batch).actions.delay_offset_bin.tolist() == [[0, 0]]
    _rows, output = _sample(model, batch, 32)
    assert output.actions.delay_offset_bin.tolist() == [[0, 0]] * 32
    model.set_inference_delay_offset_max_ms(50)
    assert _act(model, batch).actions.delay_offset_bin.tolist() == [[0, 0]]
    # A near-miss card stays illegal under the legacy rule (no regeneration).
    short = _only(_with_elixir(_batch(batch_size=1), 9.0 - 0.015), (EXPENSIVE,))
    assert _act(model, short).actions.gate.tolist() == [GATE_WAIT]


# Environment oracle: decoded earliest actions pass BattleEnvV1 and sit at the
# environment's own earliest touch bins.
def test_earliest_compound_actions_match_the_environment_touch_bins() -> None:
    model = build_production_model_v4(device=torch.device("cpu")).eval()
    with torch.no_grad():
        model.gate_head[-1].bias.copy_(torch.tensor([-20.0, 20.0]))
        model.continue_head[-1].bias.copy_(torch.tensor([0.0, 0.0]))
    model.set_delay_policy("earliest")
    late = 0
    seconds = 0
    for elixir_raw, tick in ((49_700, 90), (69_700, 90), (79_700, TWO_X_TICK), (49_000, 5_000)):
        env, _native, observation, tensorizer, batch = _tensorized(elixir_raw, tick=tick)
        rows, output = _sample(model, batch, 60, seed=elixir_raw + tick)
        env_bins = observation.action_mask.reasons["affordable_from_bin"]["slots"]
        for row in range(60):
            if int(output.actions.gate[row]) != GATE_ACT:
                continue
            decoded = decode_action_sequence_v4(
                output.actions.index_select(torch.tensor([row])),
                batch.candidates,
                row=0,
                observation=observation,
                catalog=tensorizer.catalog,
                deck=tensorizer.deck,
                card_costs=tensorizer.card_costs,
                ability_id_by_vocab_id=tensorizer.ability_id_by_vocab_id,
                horizontal_mirror=tensorizer.perspective.horizontal_mirror,
                hand_slot_permutation=tensorizer.perspective.hand_slot_permutation,
                config=tensorizer.config,
                base_latency_ticks=1,
                validate=False,
            )
            env._validate_compound_actions(0, decoded.actions, env._raw)
            first, *rest = decoded.actions
            first_bin = int(first.metadata["policy_delay_offset_bin"])
            assert first_bin == env_bins[str(first.hand_slot)]
            late += first_bin > 0
            if rest:
                seconds += 1
                second_bin = int(rest[0].metadata["policy_delay_offset_bin"])
                # Oracle: the environment rejects the pair one bin earlier.
                assert second_bin >= first_bin
                if second_bin > first_bin:
                    earlier = replace(
                        rest[0], metadata={**dict(rest[0].metadata), "policy_delay_offset_bin": second_bin - 1}
                    )
                    with pytest.raises(Exception, match="overspends elixir"):
                        env._validate_compound_actions(0, (first, earlier), env._raw)
    assert late > 0 and seconds > 0


# ---------------------------------------------------------------------------
# Log-prob bookkeeping (PPO)
# ---------------------------------------------------------------------------


def test_earliest_excludes_delay_terms_and_ppo_evaluation_matches_the_behaviour() -> None:
    batch = _only(_with_elixir(_batch(batch_size=1), 9.97), (CHEAP, ABILITY, EXPENSIVE))
    model = _rigged(second=0.0)
    with torch.no_grad():
        model.delay_offset_head[-1].bias.copy_(torch.tensor([0.3, -0.2, 0.1, 0.0, 0.4]))
        model.gate_head[-1].bias.copy_(torch.tensor([0.0, 0.0]))
    model.set_delay_policy("earliest")
    rows, behaviour = _sample(model, batch, 48)
    components = behaviour.action_components
    assert components is not None
    assert torch.count_nonzero(components.delay_offset_log_prob) == 0
    assert torch.count_nonzero(components.delay_offset_entropy) == 0
    expected = (
        components.gate_log_prob
        + components.candidate_log_prob.sum(dim=-1)
        + components.target_log_prob.sum(dim=-1)
        + components.continue_log_prob
    )
    assert torch.equal(behaviour.log_prob, expected)
    # The head is still evaluated (diagnostics), only not scored.
    acting = behaviour.actions.gate == GATE_ACT
    assert bool(acting.any()) and bool(components.delay_offset_legal_mask[acting, 0].any(dim=-1).all())
    with torch.no_grad():
        learner = model.evaluate_actions(rows, model.initial_state(48), behaviour.actions)
        encoded = model.encode_observation(rows)
        sequence = model.evaluate_encoded_action_sequence(
            rows,
            encoded,
            model.initial_state(48),
            behaviour.actions,
            torch.zeros(1, 48, dtype=torch.bool),
            time_steps=1,
            return_log_prob_components=True,
        )
    torch.testing.assert_close(learner.log_prob, behaviour.log_prob, rtol=0, atol=1e-6)
    torch.testing.assert_close(learner.entropy, behaviour.entropy, rtol=0, atol=1e-6)
    torch.testing.assert_close(sequence[0].flatten(), behaviour.log_prob, rtol=0, atol=1e-5)
    assert torch.count_nonzero(sequence[4][..., 3]) == 0  # delay column of the [T,B,5] components
    # The same actions scored with the head differ by exactly the delay terms.
    model.set_delay_policy("sample")
    with torch.no_grad():
        scored = model.evaluate_actions(rows, model.initial_state(48), behaviour.actions)
    assert scored.action_components is not None
    torch.testing.assert_close(
        scored.log_prob - scored.action_components.delay_offset_log_prob.sum(dim=-1), learner.log_prob
    )
    assert bool((scored.action_components.delay_offset_log_prob[acting, 0] < 0).all())


def test_earliest_learner_rejects_stored_delays_that_are_not_earliest() -> None:
    from native_runner.tests.test_training_v4_delay_aware_legality import _forced

    model = _model().eval()
    model.set_delay_policy("earliest")
    batch = _with_elixir(_batch(batch_size=1), 9.0 - 0.03)  # EXPENSIVE first legal at bin 2
    with torch.no_grad():
        model.evaluate_actions(batch, model.initial_state(1), _forced(EXPENSIVE, uid=104, target=6 * 18 + 5, delay=2))
        with pytest.raises(ValueError, match="not the earliest legal bin"):
            model.evaluate_actions(
                batch, model.initial_state(1), _forced(EXPENSIVE, uid=104, target=6 * 18 + 5, delay=4)
            )


# (f) ``sample`` stays byte-for-byte identical.
def _flatten(output: Any) -> list[torch.Tensor]:
    result = []
    for name in ("actions", "action_components"):
        record = getattr(output, name)
        result.extend(getattr(record, item.name) for item in fields(record))
    result.extend((output.log_prob, output.entropy, output.value))
    return result


def test_sample_policy_is_bit_identical_to_the_default_decoder() -> None:
    batch = _only(_with_elixir(_batch(batch_size=1), 9.97), (CHEAP, ABILITY, EXPENSIVE))
    reference = _rigged(second=0.0)
    toggled = _rigged(second=0.0)
    for model in (reference, toggled):
        with torch.no_grad():
            model.delay_offset_head[-1].bias.copy_(torch.tensor([0.3, -0.2, 0.1, 0.0, 0.4]))
            model.gate_head[-1].bias.copy_(torch.tensor([0.0, 0.0]))
    toggled.set_delay_policy("earliest")
    _sample(toggled, batch, 8)
    toggled.set_delay_policy("sample")
    for seed in (1, 2, 3):
        _rows, expected = _sample(reference, batch, 96, seed=seed)
        expected_rng = torch.get_rng_state()
        _rows, actual = _sample(toggled, batch, 96, seed=seed)
        assert torch.equal(torch.get_rng_state(), expected_rng)
        for left, right in zip(_flatten(expected), _flatten(actual), strict=True):
            assert left.dtype == right.dtype and torch.equal(left, right)
        # The head is really used: bins beyond the earliest occur.
        assert bool((actual.actions.delay_offset_bin[:, 0] > 2).any())


# ---------------------------------------------------------------------------
# Live, collector and head-to-head plumbing
# ---------------------------------------------------------------------------


def test_live_loader_sets_the_delay_policy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-step-00000001.pt"
    checkpoint.write_bytes(b"x")
    # The loader builds the model from the checkpoint's own contract
    # (architecture overrides such as strong-v1's CNN 128 / LSTM 768).
    payload = {
        "contract": {"model_config": {"spatial_hidden_channels": 128, "lstm_hidden_dim": 768, "lstm_input_dim": 768}},
        "checkpoint_id": "x",
        "update_step": 1,
        "training_stage": "imitation",
    }
    built_configs: list[object] = []
    for kwargs, expected in (({}, "sample"), ({"delay_policy": "earliest"}, "earliest")):
        model = _model()

        def _build(*, device, config=None, _model=model):
            built_configs.append(config)
            return _model

        monkeypatch.setattr(policy_session, "build_production_model_v4", _build)
        monkeypatch.setattr(policy_session, "read_actor_critic_checkpoint", lambda *args, **kw: dict(payload))
        monkeypatch.setattr(policy_session, "restore_actor_critic_checkpoint", lambda payload, model, **kw: payload)
        loaded = policy_session.load_policy_v4(checkpoint, device="cpu", **kwargs)
        assert loaded.model.delay_policy == expected
    assert [(c.spatial_hidden_channels, c.lstm_hidden_dim, c.lstm_input_dim) for c in built_configs] == [(128, 768, 768)] * 2
    with pytest.raises(ValueError):
        policy_session.load_policy_v4(checkpoint, device="cpu", delay_policy="late")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph capture needs CUDA")
def test_earliest_runs_inside_the_live_cuda_graph() -> None:
    device = torch.device("cuda")
    batch = _only(_with_elixir(_batch(batch_size=1), 9.0 - 0.03), (EXPENSIVE,))
    model = _rigged().to(device)
    model.set_delay_policy("earliest")
    graph = policy_session._DeterministicCudaGraphV4(device)
    state = model.initial_state(1, device=device)
    start = torch.ones(1, dtype=torch.bool, device=device)
    for _ in range(2):
        output = graph.run(model, batch, state, start)
        assert output.actions.delay_offset_bin.tolist() == [[2, -1]]
        assert torch.count_nonzero(output.action_components.delay_offset_log_prob) == 0


def test_rollout_graph_cache_separates_delay_policies() -> None:
    model = _model()
    device = torch.device("cpu")
    default = collector._persistent_rollout_graph_v4(model, device, 4)
    assert collector._persistent_rollout_graph_v4(model, device, 4) is default
    model.set_delay_policy("earliest")
    earliest = collector._persistent_rollout_graph_v4(model, device, 4)
    assert earliest is not None and earliest is not default
    model.set_delay_policy("sample")
    assert collector._persistent_rollout_graph_v4(model, device, 4) is default


# (e) spec round trip through the schedule / driver.
def test_model_spec_parses_the_delay_policy() -> None:
    spec = h2h.parse_model_spec_v4(r"A2=D:\ckpt\a.pt:delay_policy=earliest:gate_logit_offset=-1.5:mode=argmax")
    assert (spec.name, spec.path, spec.delay_policy) == ("A2", Path(r"D:\ckpt\a.pt"), "earliest")
    assert (spec.gate_logit_offset, spec.mode) == (-1.5, "argmax")
    assert h2h.parse_model_spec_v4("A=/a.pt").delay_policy == "sample"
    assert h2h.parse_model_spec_v4("A=/a.pt:delay_policy=sample").delay_policy == "sample"
    for bad in ("A=/a.pt:delay_policy=late", "A=/a.pt:delay_policy=earliest:delay_policy=sample"):
        with pytest.raises(ValueError):
            h2h.parse_model_spec_v4(bad)
    with pytest.raises(ValueError):
        h2h.ModelSpecV4(name="A", path=Path("/a.pt"), delay_policy="argmax")
    model = _model()
    assert h2h.behavior_key_v4("id", 0.0, "argmax", model) == "id|gate_logit_offset=0.0|mode=argmax|argmax"
    model.set_delay_policy("earliest")
    assert h2h.behavior_key_v4("id", 0.0, "argmax", model).endswith("|argmax|delay_policy=earliest")


def test_driver_plays_one_checkpoint_against_itself_with_both_delay_policies(monkeypatch, tmp_path) -> None:
    from native_runner.tests.test_training_v4_head_to_head import _fake_report, _topology
    from native_runner.tests.test_training_v4_ppo_matchmaking import _manifest

    manifest = _manifest()
    pool = tmp_path / "pool.json"
    pool.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(schedule_module, "DeckPoolManifestV4", SimpleNamespace(load=lambda _path: manifest))
    monkeypatch.setattr(h2h, "DeckPoolManifestV4", SimpleNamespace(load=lambda _path: manifest))
    schedule = tmp_path / "schedule.jsonl"
    schedule_module.main(["--deck-pool", str(pool), "--output", str(schedule), "--configurations", "4", "--seed", "2"])
    topology = tmp_path / "topology.json"
    topology.write_text(json.dumps(_topology(3)), encoding="utf-8")
    checkpoint = tmp_path / "A.pt"
    checkpoint.write_bytes(b"A")
    built: list[Any] = []

    def fake_build(path: Path, device: torch.device) -> tuple[Any, dict[str, Any]]:
        assert Path(path) == checkpoint
        built.append(_model())
        return built[-1], {"checkpoint_id": "id-A", "update_step": 0, "training_stage": "test"}

    seen: list[tuple[str, dict[str, str]]] = []

    def fake_collect(specs, assignments, current_model, anchor_model, history_models, **kwargs):
        assert anchor_model is current_model
        seen.append((current_model.delay_policy, {name: model.delay_policy for name, model in history_models.items()}))
        callback = kwargs["segment_callback"]
        reports = [_fake_report(item, seat0_wins=item.matchup.sequence % 2 == 0) for item in assignments]
        return callback(SimpleNamespace(reports=tuple(reports), timing={"total_seconds": 1.0}))

    monkeypatch.setattr(h2h, "build_model_from_checkpoint_v4", fake_build)
    monkeypatch.setattr(h2h, "collect_async_cluster_wave_v4", fake_collect)
    monkeypatch.setattr(torch, "set_num_threads", lambda _value: None)
    output = tmp_path / "run"
    args = [
        "--topology", str(topology), "--deck-pool", str(pool), "--schedule", str(schedule),
        "--model", f"A={checkpoint}", "--model", f"AE={checkpoint}:delay_policy=earliest",
        "--pairs", "A:AE", "--games-per-pair", "8", "--mode", "sample",
        "--output-dir", str(output), "--device", "cpu",
    ]  # fmt: skip
    h2h.main(args)
    assert len(built) == 2 and built[0] is not built[1]
    assert sorted(model.delay_policy for model in built) == ["earliest", "sample"]
    # Four waves (3 engines x 1 lane): each model is current twice.
    assert len(seen) == 4
    assert {(current, tuple(history.items())) for current, history in seen} == {
        ("earliest", (("A", "sample"),)),
        ("sample", (("AE", "earliest"),)),
    }
    identity = json.loads((output / "identity.json").read_text(encoding="utf-8"))
    models = identity["run_key"]["models"]
    assert "delay_policy" not in models["A"] and models["AE"]["delay_policy"] == "earliest"
    assert models["A"]["checkpoint_id"] == models["AE"]["checkpoint_id"] == "id-A"
    assert models["AE"]["behavior"] == models["A"]["behavior"] + "|delay_policy=earliest"
    rows = [json.loads(line) for line in (output / "games.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 8
    for row in rows:
        for seat in (0, 1):
            behavior = row[f"seat{seat}_behavior"]
            assert behavior.endswith("|delay_policy=earliest") == (row[f"seat{seat}_model"] == "AE")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["complete"] is True and summary["pairs"]["A:AE"]["games"] == 8
    # The identity is part of the resume key: flipping the policy is a different run.
    flipped = [f"A={checkpoint}:delay_policy=earliest" if item == f"A={checkpoint}" else item for item in args]
    with pytest.raises(ValueError, match="resume identity differs"):
        h2h.main([*flipped, "--resume"])
