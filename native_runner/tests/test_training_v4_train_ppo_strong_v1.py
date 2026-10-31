"""strong-v1 PPO trainer plumbing: contract model build, delay policy,
gate correction, opponent shares, critic warmup default, resume relaxation, topology,
king-reward wiring and expert BC weights."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import inspect
import math
import random

import pytest
import torch

from native_runner.tests.test_training_v4_model import _ability_catalog, _catalog
from native_runner.tests.test_training_v4_ppo_matchmaking import _manifest
from native_runner.training.v4 import train_ppo_self_play_cluster as trainer
from native_runner.training.v4.async_cluster_self_play import collect_async_cluster_wave_v4
from native_runner.training.v4.catalog import EntityArchetypeCatalogV1
from native_runner.training.v4.checkpoint import save_actor_critic_checkpoint
from native_runner.training.v4.config import ModelConfigV4
from native_runner.training.v4.league import PolicyLeagueV4
from native_runner.training.v4.matchmaking import (
    POLICY_CURRENT,
    POLICY_HISTORY,
    POLICY_IL,
    DeckMatchmakerV4,
    MatchmakerConfigV4,
)
from native_runner.training.v4.mechanics import EffectSemanticCatalogV1, MechanicProfileCatalogV1
from native_runner.training.v4.model import UniversalCardPolicyV4
from native_runner.training.v4.ppo_anchor import GATE_LOGIT_OFFSET_EXTRA_KEY
from native_runner.training.v4.ppo_expert_bc import expert_bc_contract_v4
from native_runner.training.v4.tensors import GATE_ACT

# A non-default (strong-v1-like, but small) architecture.
SMALL_STRONG = {"spatial_hidden_channels": 32, "lstm_hidden_dim": 192, "lstm_input_dim": 192}


def _fixture_model(config: ModelConfigV4 | None = None) -> UniversalCardPolicyV4:
    catalog = _catalog()
    abilities = _ability_catalog(catalog)
    archetypes = EntityArchetypeCatalogV1(catalog.raw_card_ids, ("form:fixture_a", "form:fixture_b"))
    return UniversalCardPolicyV4(
        catalog,
        ability_catalog=abilities,
        entity_archetype_catalog=archetypes,
        effect_catalog=EffectSemanticCatalogV1.empty(catalog.raw_card_ids),
        mechanic_profile_catalog=MechanicProfileCatalogV1.empty(
            card_catalog=catalog, ability_catalog=abilities, entity_archetype_catalog=archetypes
        ),
        config=config,
    )


@pytest.fixture
def contract_builder(monkeypatch: pytest.MonkeyPatch) -> list[ModelConfigV4 | None]:
    built: list[ModelConfigV4 | None] = []

    def build(*, device=None, config=None):
        built.append(config)
        model = _fixture_model(config)
        return model if device is None else model.to(device)

    monkeypatch.setattr(trainer, "build_production_model_v4", build)
    return built


def _il_checkpoint(tmp_path, *, extra=None, name="strong-v1.pt"):
    torch.manual_seed(5)
    model = _fixture_model(ModelConfigV4(**SMALL_STRONG))
    path = tmp_path / name
    save_actor_critic_checkpoint(
        path,
        model,
        training_stage="imitation",
        gamma_per_decision=0.99,
        extra={"gate_act_weight": 8.0, "continue_second_weight": 32.0} if extra is None else extra,
    )
    return path, model


def _parse(*extra: str):
    return trainer.build_argument_parser().parse_args(
        ["--deck-pool", "pool.json", "--checkpoint", "il.pt", "--output-dir", "out", "--engine-ready", "t.json", *extra]
    )


# ---------------------------------------------------------------------------
# §2.1 contract build, §2.2 delay policy, §2.4 gate correction
# ---------------------------------------------------------------------------


def test_frozen_model_is_built_from_the_checkpoint_contract(tmp_path, contract_builder) -> None:
    path, source = _il_checkpoint(tmp_path)
    model, payload = trainer._load_frozen_model(
        path, device=torch.device("cpu"), delay_policy="earliest", gate_logit_offset="auto"
    )
    assert contract_builder and contract_builder[-1] is not None
    assert (model.config.spatial_hidden_channels, model.config.lstm_hidden_dim) == (32, 192)
    assert asdict(model.config) == asdict(source.config)
    assert model.delay_policy == "earliest"
    assert model.ppo_rollout_sampling is False  # frozen opponent: argmax
    assert not any(parameter.requires_grad for parameter in model.parameters())
    expected_bias = source.gate_head[-1].bias.detach().clone()
    expected_bias[GATE_ACT] -= math.log(8.0)
    torch.testing.assert_close(model.gate_head[-1].bias.detach(), expected_bias)
    for name, tensor in source.state_dict().items():
        if name != "gate_head.3.bias" and not name.startswith("gate_head") and tensor.is_floating_point():
            assert torch.equal(model.state_dict()[name], tensor), name
    assert payload["extra"]["gate_act_weight"] == 8.0


def test_learner_resume_restores_optimizer_and_never_reapplies_the_gate_offset(tmp_path, contract_builder) -> None:
    path, _source = _il_checkpoint(tmp_path)
    learner, _payload, optimizer = trainer._load_contract_model(
        path, device=torch.device("cpu"), delay_policy="earliest", gate_logit_offset="auto"
    )
    assert optimizer is None
    applied = trainer._gate_logit_offset_of(learner)
    assert applied == pytest.approx(-math.log(8.0))
    adam = torch.optim.AdamW(learner.parameters(), lr=1e-3)
    learner.gate_head[-1].bias.sum().backward()
    adam.step()
    snapshot = tmp_path / "ppo-strict.pt"
    save_actor_critic_checkpoint(
        snapshot,
        learner,
        optimizer=adam,
        update_step=3,
        training_stage="ppo",
        gamma_per_decision=0.9997,
        extra={GATE_LOGIT_OFFSET_EXTRA_KEY: applied, "gate_act_weight": 8.0},
    )
    resumed, payload, resumed_optimizer = trainer._load_contract_model(
        snapshot,
        device=torch.device("cpu"),
        delay_policy="earliest",
        gate_logit_offset="auto",
        optimizer_factory=lambda model: torch.optim.AdamW(model.parameters(), lr=1e-3),
    )
    assert torch.equal(resumed.gate_head[-1].bias, learner.gate_head[-1].bias)  # not shifted twice
    assert trainer._gate_logit_offset_of(resumed) == pytest.approx(applied)
    assert resumed_optimizer is not None and resumed_optimizer.state_dict()["state"]
    assert payload["update_step"] == 3 and resumed.delay_policy == "earliest"


def test_history_models_get_the_delay_policy(tmp_path, contract_builder, monkeypatch) -> None:
    from native_runner.training.v4.cluster_self_play import ClusterMatchAssignmentV4
    from native_runner.training.v4.league import LeagueEntryV4
    from native_runner.training.v4 import policy_session

    path, _ = _il_checkpoint(tmp_path)
    payload = torch.load(path, weights_only=False)
    entry = LeagueEntryV4(checkpoint_id=payload["checkpoint_id"], checkpoint_path=str(path), update_step=0)

    class Assignment:
        opponent = entry

    # Opponents now use the same builder as the live UI, independently of the
    # training reference builder patched by contract_builder.
    monkeypatch.setattr(policy_session, "build_production_model_v4", trainer.build_production_model_v4)
    models = trainer._load_history_models(
        (Assignment(),),  # type: ignore[arg-type]
        {},
        device=torch.device("cpu"),
    )
    (model,) = models.values()
    assert model.delay_policy == "earliest" and model.config.lstm_hidden_dim == 192
    assert ClusterMatchAssignmentV4  # imported for type parity


# ---------------------------------------------------------------------------
# CLI: temperatures, warmups, anchor, BC weights
# ---------------------------------------------------------------------------


def test_cli_defaults_for_the_strong_v1_recipe() -> None:
    args = _parse()
    assert args.environment_warmup_ticks == 130
    assert args.opening_force_after_seconds == 15
    assert args.critic_warmup_iters == 10
    assert args.delay_policy == "sample" and args.anchor_checkpoint is None
    assert args.king_activation_penalty == 0.1 and args.king_activation_scope == "any"
    assert args.king_shaping_mode == "princess_gated" and args.king_potential_weight == 3.0
    assert args.fireball_king_activation_penalty == 0.0
    assert trainer._resolve_temperatures(args, anchor_enabled=False) == (0.2, 0.2, 2.0)
    anchored = _parse("--anchor-checkpoint", "il.pt", "--delay-policy", "earliest")
    assert trainer._resolve_temperatures(anchored, anchor_enabled=True) == (1.0, 1.0, 1.0)
    assert trainer._resolve_temperatures(
        _parse("--anchor-checkpoint", "il.pt", "--gate-temperature-start", "0.3",
               "--gate-temperature-end", "0.3"), anchor_enabled=True
    ) == (0.3, 0.3, 1.0)
    config = trainer._anchor_kl_config(anchored)
    assert config is not None
    assert config.initial_coefficient == 0.05 and config.initial_continue_coefficient == pytest.approx(0.25)
    assert config.continue_mode == "tied" and config.adaptive
    assert config.no_decrease_updates == 20
    assert trainer._anchor_kl_config(_parse("--anchor-checkpoint", "il.pt", "--anchor-kl-no-decrease-updates", "0")).no_decrease_updates == 0
    assert trainer._anchor_kl_config(args) is None


def test_expert_bc_weights_follow_the_strong_v1_objective() -> None:
    earliest = trainer._expert_bc_loss_weights(_parse("--delay-policy", "earliest"))
    assert earliest.delay == 0.0 and earliest.continue_second_weight == 32.0 and earliest.value == 0.0
    legacy = trainer._expert_bc_loss_weights(_parse())
    assert legacy.delay == 1.0 and legacy.continue_second_weight == 1.0

    class Sampler:
        def contract(self):
            return {"schema": "v4-ppo-expert-bc.v2"}

    assert "loss_weights" not in expert_bc_contract_v4(Sampler(), loss_weights=legacy)
    assert expert_bc_contract_v4(Sampler(), loss_weights=earliest)["loss_weights"]["delay"] == 0.0


# ---------------------------------------------------------------------------
# §2.5 opponents
# ---------------------------------------------------------------------------


def test_opponent_mix_arguments_are_split() -> None:
    mix = trainer._resolve_opponent_mix(_parse("--self-play-percent", "30", "--history-percent", "20"))
    assert mix == {"current_il": 50, "self_play": 30, "fixed_opponent": 0, "history": 20}
    mix = trainer._resolve_opponent_mix(
        _parse(
            "--self-play-percent", "30", "--history-percent", "10",
            "--fixed-opponent-percent", "10", "--fixed-opponent-checkpoint", "x.pt",
        )
    )
    assert mix == {"current_il": 50, "self_play": 30, "fixed_opponent": 10, "history": 10}
    with pytest.raises(ValueError, match="exceed 100"):
        trainer._resolve_opponent_mix(_parse("--self-play-percent", "90", "--history-percent", "20"))


def _league(tmp_path, ids):
    league = PolicyLeagueV4(anchor_checkpoint_id="anchor-id", anchor_checkpoint_path=tmp_path / "anchor.pt")
    for index, checkpoint_id in enumerate(ids):
        league.add_snapshot(checkpoint_id, tmp_path / f"{checkpoint_id}.pt", update_step=index + 1)
    return league


def test_fixed_opponents_keep_pfsp_snapshots_sampling(tmp_path) -> None:
    manifest = _manifest()
    config = MatchmakerConfigV4(current_current=30, current_il=40, current_history=30)
    snapshots = _league(tmp_path, ("snap-1", "snap-2"))
    fixed = _league(tmp_path, ("fixed-a",))
    shards = trainer._balanced_assignments(
        DeckMatchmakerV4(manifest, seed=97, config=config),
        snapshots,
        random.Random(101),
        match_capacities=(1000,),
        rich_capacities=(0,),
        special_towers_require_rich=False,
        fixed_league=fixed,
        fixed_opponent_percent=10,
        history_percent=20,
    )
    rows = shards[0]
    opponents = Counter(row.opponent.checkpoint_id for row in rows if row.opponent is not None)
    policies = Counter(row.matchup.policy_matchup for row in rows)
    assert policies[POLICY_HISTORY] == 300 and policies[POLICY_CURRENT] == 300 and policies[POLICY_IL] == 400
    assert opponents["fixed-a"] > 0 and opponents["snap-1"] + opponents["snap-2"] > 0
    fixed_share = opponents["fixed-a"] / policies[POLICY_HISTORY]
    assert abs(fixed_share - 1 / 3) < 0.1


def test_snapshot_share_falls_back_to_il_before_the_first_snapshot(tmp_path) -> None:
    manifest = _manifest()
    config = MatchmakerConfigV4(current_current=50, current_il=30, current_history=20)
    matchmaker = DeckMatchmakerV4(manifest, seed=5, config=config)
    shards = trainer._balanced_assignments(
        matchmaker,
        _league(tmp_path, ()),
        random.Random(7),
        match_capacities=(100,),
        rich_capacities=(0,),
        special_towers_require_rich=False,
        fixed_league=_league(tmp_path, ()),
        fixed_opponent_percent=0,
        history_percent=20,
    )
    policies = Counter(row.matchup.policy_matchup for row in shards[0])
    assert policies[POLICY_HISTORY] == 0 and policies[POLICY_IL] == 50 and policies[POLICY_CURRENT] == 50
    # The first snapshot restores the normal 30 IL / 50 self-play / 20 history mix.
    shards = trainer._balanced_assignments(
        matchmaker,
        _league(tmp_path, ("snap-10",)),
        random.Random(7),
        match_capacities=(100,),
        rich_capacities=(0,),
        special_towers_require_rich=False,
        fixed_league=_league(tmp_path, ()),
        fixed_opponent_percent=0,
        history_percent=20,
    )
    assert Counter(row.matchup.policy_matchup for row in shards[0]) == {
        POLICY_IL: 30, POLICY_CURRENT: 50, POLICY_HISTORY: 20,
    }
    # Fixed opponents remain distinct; only the empty snapshot share becomes IL.
    config = MatchmakerConfigV4(current_current=30, current_il=40, current_history=30)
    shards = trainer._balanced_assignments(
        DeckMatchmakerV4(manifest, seed=5, config=config),
        _league(tmp_path, ()),
        random.Random(7),
        match_capacities=(200,),
        rich_capacities=(0,),
        special_towers_require_rich=False,
        fixed_league=_league(tmp_path, ("fixed-a",)),
        fixed_opponent_percent=10,
        history_percent=20,
    )
    rows = shards[0]
    policies = Counter(row.matchup.policy_matchup for row in rows)
    assert policies[POLICY_IL] > 80
    assert all(row.opponent.checkpoint_id == "fixed-a" for row in rows if row.matchup.policy_matchup == POLICY_HISTORY)
    assert policies[POLICY_HISTORY] > 0 and policies[POLICY_CURRENT] == 60


# ---------------------------------------------------------------------------
# §2.7 resume relaxation
# ---------------------------------------------------------------------------


def _contract(**overrides):
    contract = {
        "schema": trainer.RESUME_CONTRACT_SCHEMA,
        "learner_contract": {"model_config": {"lstm_hidden_dim": 768}},
        "delay_policy": "earliest",
        "gate_logit_offset": -math.log(8.0),
        "behavior_temperatures": {"action": 1.0, "continue": 1.0},
        "gate_temperature_schedule": (1.0, 1.0, 1),
        "deck_pool": "/pool.json",
        "engine_count": 55,
        "resident_slots_per_engine": 1,
        "world_size": 4,
        "environment_warmup_ticks": 90,
        "decision_ticks": 5,
        "battle_levels": (11, 16),
        "battle_level_split": "exact",
        "seed": 1,
        "learner_deck_id": None,
        "fixed_learner_opponent_mix": False,
        "focus_tail_current_focus_percent": 75,
        "expert_bc": None,
        "shaping_beta": 0.05,
        "reward_parameters": {"king_activation_penalty": 0.1},
        "il_anchor_kl": {"initial_coefficient": 0.05},
        "learning_rate": 1e-5,
    }
    contract.update(overrides)
    return contract


def test_resume_allows_reward_anchor_and_engine_count_changes() -> None:
    changes = trainer._check_resume_contract(
        _contract(),
        _contract(
            shaping_beta=0.1,
            reward_parameters={"king_activation_penalty": 0.0},
            il_anchor_kl={"initial_coefficient": 0.1},
            engine_count=108,
            learning_rate=2e-5,
        ),
    )
    assert set(changes) == {"shaping_beta", "reward_parameters", "il_anchor_kl", "engine_count", "learning_rate"}
    assert trainer._check_resume_contract(_contract(), _contract()) == {}


@pytest.mark.parametrize(
    "override",
    (
        {"learner_contract": {"model_config": {"lstm_hidden_dim": 512}}},
        {"delay_policy": "sample"},
        {"gate_logit_offset": 0.0},
        {"environment_warmup_ticks": 130},
        {"deck_pool": "/other.json"},
        {"world_size": 8},
        {"resident_slots_per_engine": 8},
    ),
)
def test_resume_rejects_architecture_and_data_contract_changes(override) -> None:
    with pytest.raises(ValueError, match="architecture/data contract changed"):
        trainer._check_resume_contract(_contract(), _contract(**override))


def test_resume_rejects_pre_v2_contracts() -> None:
    with pytest.raises(ValueError, match="schema"):
        trainer._check_resume_contract({"schema": "v4-ppo-strict-resume.v1"}, _contract())


# ---------------------------------------------------------------------------
# Topology and king reward wiring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lanes", (1, 8))
def test_single_slot_topology_is_accepted(monkeypatch, lanes) -> None:
    ready = {
        "schema": "v4-ppo-multinode-topology.v1",
        "trainer_slurm_job_id": "1001",
        "trainer_host": "gpu-a",
        "gpus": 4,
        "engine_count": 55,
        "resident_slots_per_engine": lanes,
        "decision_ticks": 5,
    }
    monkeypatch.setenv("SLURM_JOB_ID", "1001")
    monkeypatch.setattr(trainer.socket, "gethostname", lambda: "gpu-a")
    monkeypatch.setattr(trainer.torch.cuda, "device_count", lambda: 4)
    assert trainer._validate_topology(ready, world_size=4) == (55, lanes)
    with pytest.raises(RuntimeError, match="lanes"):
        trainer._validate_topology({**ready, "resident_slots_per_engine": 0}, world_size=4)


def test_king_reward_parameters_reach_the_collector() -> None:
    parameters = trainer._king_reward_parameters(_parse())
    assert parameters == {
        "king_activation_penalty": 0.1,
        "king_activation_scope": "any",
        "king_shaping_mode": "princess_gated",
        "king_potential_weight": 3.0,
    }
    accepted = inspect.signature(collect_async_cluster_wave_v4).parameters
    assert set(parameters) <= set(accepted)
    source = inspect.getsource(trainer.main)
    assert "**_king_reward_parameters(args)" in source
    assert "validate_king_reward_parameters_v4(" in source


def test_king_activation_report_fields_are_aggregated() -> None:
    reports = [
        {
            "current_owners": (0,),
            "king_activations": (1, 0),
            "king_activation_tick": (900, None),
            "king_activation_source_unattributed": (0, 0),
            "king_activation_reward_total": (-0.1, 0.1),
            "king_activation_details": ({"owner": 0, "source_family": "spell"},),
        },
        {
            "current_owners": (1,),
            "king_activations": (1, 0),
            "king_activation_tick": (1200, None),
            "king_activation_source_unattributed": (1, 0),
            "king_activation_reward_total": (-0.1, 0.1),
            "king_activation_details": ({"owner": 0, "source_family": "troop"},),
        },
        {"current_owners": (0,)},  # old worker without the fields
    ]
    metrics = trainer._king_activation_batch_metrics(reports)
    assert metrics["batch_performance/king_activation_rate"] == 0.5
    assert metrics["batch_performance/king_activated_against_rate"] == 0.5
    assert metrics["batch_performance/king_activation_reward_mean"] == pytest.approx(0.0)
    assert metrics["batch_performance/king_activation_tick_mean"] == 900.0
    assert metrics["batch_performance/king_activation_source/spell"] == 1
    assert metrics["batch_performance/king_activation_source/troop"] == 0
    assert trainer._king_activation_batch_metrics([{"current_owners": (0,)}]) == {}


# ---------------------------------------------------------------------------
# phase.json for the GPU filler
# ---------------------------------------------------------------------------


def test_phase_file_is_written_atomically(tmp_path) -> None:
    import json
    import time

    path = tmp_path / "phase.json"
    before = time.time()
    trainer._write_phase(path, "rollout", 3)
    first = json.loads(path.read_text(encoding="utf-8"))
    assert first["phase"] == "rollout" and first["iteration"] == 3 and first["ts"] >= before
    for phase in ("update", "critic_warmup", "idle"):
        trainer._write_phase(path, phase, 4)
        assert json.loads(path.read_text(encoding="utf-8"))["phase"] == phase
    assert sorted(item.name for item in tmp_path.iterdir()) == ["phase.json"]  # no temporary left
    with pytest.raises(ValueError, match="phase"):
        trainer._write_phase(path, "training", 5)
    assert json.loads(path.read_text(encoding="utf-8"))["phase"] == "idle"


def test_trainer_publishes_every_phase_around_collection_and_update() -> None:
    source = inspect.getsource(trainer.main)
    assert 'publish_phase("rollout", initial_update_step + 1)' in source
    assert 'publish_phase("critic_warmup" if update_step <= args.critic_warmup_iters else "update", update_step)' in source
    assert source.count('publish_phase("rollout", update_step + 1)') == 2
    assert '_write_phase(phase_path, "idle", phase_iteration)' in source
    # rollout is announced before the collector is (re)started, update right after it publishes.
    assert source.index('publish_phase("rollout", initial_update_step + 1)') < source.index(
        "start_segment_session(actor_state_id)"
    )
    assert source.index("collected = next_segment_collection()") < source.index('else "update", update_step)')


# ---------------------------------------------------------------------------
# --ddp-backend gloo: several ranks per GPU
# ---------------------------------------------------------------------------


def test_gloo_backend_stacks_ranks_on_gpus_and_nccl_keeps_one_per_gpu() -> None:
    assert [trainer._rank_cuda_device_index(rank, backend="gloo", device_count=2) for rank in range(8)] == [
        0, 1, 0, 1, 0, 1, 0, 1
    ]
    assert trainer._rank_cuda_device_index(1, backend="nccl", device_count=2) == 1
    with pytest.raises(ValueError, match="gloo"):
        trainer._rank_cuda_device_index(2, backend="nccl", device_count=2)
    assert _parse().ddp_backend == "nccl"
    assert _parse("--ddp-backend", "gloo").ddp_backend == "gloo"


def test_topology_accepts_eight_gloo_ranks_on_two_gpus(monkeypatch) -> None:
    ready = {
        "schema": "v4-ppo-multinode-topology.v1",
        "trainer_slurm_job_id": "1002",
        "trainer_host": "gb001",
        "gpus": 8,
        "engine_count": 697,
        "resident_slots_per_engine": 1,
        "decision_ticks": 5,
    }
    monkeypatch.setenv("SLURM_JOB_ID", "1002")
    monkeypatch.setattr(trainer.socket, "gethostname", lambda: "gb001")
    monkeypatch.setattr(trainer.torch.cuda, "device_count", lambda: 2)
    assert trainer._validate_topology(ready, world_size=8, backend="gloo") == (697, 1)
    with pytest.raises(RuntimeError, match="visible_gpu_count"):
        trainer._validate_topology(ready, world_size=8, backend="nccl")
    with pytest.raises(RuntimeError, match="visible_gpu_count"):
        trainer._validate_topology({**ready, "gpus": 7}, world_size=7, backend="gloo")
    declared = {**ready, "cuda_devices": 2, "rank_cuda_device": [0, 0, 0, 0, 1, 1, 1, 1]}
    assert trainer._validate_topology(declared, world_size=8, backend="gloo") == (697, 1)
    with pytest.raises(RuntimeError, match="visible_gpu_count"):
        trainer._validate_topology({**declared, "cuda_devices": 4}, world_size=8, backend="gloo")
    with pytest.raises(RuntimeError, match="rank_cuda_device"):
        trainer._validate_topology({**declared, "rank_cuda_device": [0] * 8}, world_size=8, backend="gloo")


def test_rank_device_mapping_honors_declared_topology() -> None:
    mapping = [0, 0, 0, 0, 1, 1, 1, 1]
    assert [trainer._rank_cuda_device_index(r, backend="gloo", device_count=2,
            rank_cuda_device=mapping) for r in range(8)] == mapping
    with pytest.raises(ValueError, match="unavailable"):
        trainer._rank_cuda_device_index(0, backend="gloo", device_count=2, rank_cuda_device=[2])
    with pytest.raises(ValueError, match="gloo"):
        trainer._rank_cuda_device_index(0, backend="nccl", device_count=2, rank_cuda_device=mapping)
