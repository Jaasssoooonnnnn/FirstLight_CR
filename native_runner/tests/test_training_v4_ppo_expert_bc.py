from datetime import timedelta
import importlib.util
import json
from pathlib import Path
import random
from types import SimpleNamespace

import pytest
import torch

from native_runner.tests.test_training_v4_model import _batch, _model
from native_runner.training.v4.imitation import ILSequenceV4
from native_runner.training.v4.learning import trusted_decision_mask
from native_runner.training.v4.tensors import ActionSequenceV4, GATE_ACT, GATE_WAIT
from native_runner.training.v4.ppo_expert_bc import (
    ExpertBatchItemV4, HogExpertSamplerV4, _reduce_actor_gradients, _probe_log_probs,
    distributed_expert_bc_update_v4, head_counts, selected_window_mask, cached_sequence_quality,
    expert_hog_probability_sums,
    hog_gate_mask, hog_gate_nll_sum, expert_bc_contract_v4,
)


def _wait_actions(batch_size):
    return ActionSequenceV4(
        gate=torch.full((batch_size,), GATE_WAIT, dtype=torch.long),
        micro_action_count=torch.zeros(batch_size, dtype=torch.long),
        candidate_index=torch.full((batch_size, 2), -1, dtype=torch.long),
        candidate_uid=torch.full((batch_size, 2), -1, dtype=torch.long),
        target_cell=torch.full((batch_size, 2), -1, dtype=torch.long),
        delay_offset_bin=torch.full((batch_size, 2), -1, dtype=torch.long))


def _one_action(batch_size):
    action = _wait_actions(batch_size)
    action.gate[0] = GATE_ACT
    action.micro_action_count[0] = 1
    action.candidate_index[0, 0] = 0
    action.candidate_uid[0, 0] = 101
    action.target_cell[0, 0] = 0
    action.delay_offset_bin[0, 0] = 0
    return action


def _selector():
    path = Path(__file__).parents[1] / "tools/experiments/build_hog_expert_manifest.py"
    spec = importlib.util.spec_from_file_location("hog_selector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _payload(data_i=0, result="victory", own_hp=10000, other_hp=10300):
    sel = _selector()
    player = dict(deck=[dict(card_key=k) for k in sel.EXACT_DECK],
                  tower_card=dict(card_key="tower-princess"),
                  final_tower_hitpoints=dict(total=own_hp))
    other = dict(deck=[dict(card_key="knight")]*8,
                 final_tower_hitpoints=dict(total=other_hp))
    return dict(battle=dict(result=result, team=dict(players=[player]), opponent=dict(players=[other])),
                events=[dict(kind="play_card", card_key="hog-rider", side="team",
                             replay_tick_20hz=220, source_fields=dict(data_i=data_i))])


@pytest.mark.parametrize("data_i,owner", [(0, 1), (1, 0)])
def test_source_team_native_owner_mapping(data_i, owner):
    rows = _selector().select_sides(_payload(data_i), "test")
    assert len(rows) == 1 and rows[0]["owner"] == owner


@pytest.mark.parametrize("data_i,owner", [(0, 0), (1, 1)])
def test_source_opponent_native_owner_mapping(data_i, owner):
    p = _payload(data_i, result="defeat")
    p["battle"]["team"], p["battle"]["opponent"] = p["battle"]["opponent"], p["battle"]["team"]
    p["events"][0]["side"] = "opponent"
    row = _selector().select_sides(p, "test")[0]
    assert row["owner"] == owner and row["outcome"] == "win"


def test_exact_forms_and_win_close_loss_filter():
    sel = _selector()
    p = _payload(result="defeat")
    row = sel.select_sides(p, "test")[0]
    assert row["outcome"] == "close_loss" and row["weight"] == .25
    assert sel.select_sides(_payload(result="defeat", own_hp=5000), "test") == []
    assert sel.select_sides(_payload(result="draw"), "test") == []
    p["battle"]["team"]["players"][0]["deck"] = [dict(card_key=k.replace("cannon-ev1", "cannon")) for k in sel.EXACT_DECK]
    assert sel.select_sides(p, "test") == []
    p = _payload(result="defeat")
    del p["battle"]["team"]["players"][0]["final_tower_hitpoints"]
    assert sel.select_sides(p, "test") == []


@pytest.mark.parametrize("reason,status", [
    ("terminal fidelity mismatch: winner", "trusted_prefix"),
    ("battle ended before all expert actions executed", "trusted_prefix"),
    ("RuntimeError: rejected action", "excluded"),
])
def test_original_failure_policy(reason, status):
    r = dict(completed=False, failure_reason=reason, decision_frame_count=200,
             simulated_end_tick=1100, first_untrusted_tick=1100)
    assert _selector().choose_cache_result(r) == status
    assert _selector().choose_cache_result(dict(r, simulated_end_tick=799)) == "excluded"
    assert _selector().choose_cache_result(dict(r, decision_frame_count=0)) == "excluded"


def test_windows_preserve_failure_tail_and_mask_holes():
    ticks = torch.arange(90, 2090, 5)[:, None]
    valid = trusted_decision_mask(ticks, torch.tensor([1800]))
    valid[5] = False
    selected = selected_window_mask(valid, [20, 220, 340], window_steps=128, rng=random.Random(7))
    assert not (selected & ~valid).any()
    assert selected[:128].sum() == 127
    assert selected[220] and not selected[340]
    assert not selected[ticks[:, 0] + 5 > 1600].any()


def test_global_head_populations_and_no_value():
    batch = _batch(batch_size=1)
    seq = ILSequenceV4((batch, batch, batch), (_one_action(1), _wait_actions(1), _one_action(1)),
                       torch.tensor([[True], [False], [False]]),
                       torch.tensor([[True], [True], [False]]), torch.zeros(3, 1),
                       gate_loss_mask=torch.tensor([[True], [False], [False]]))
    assert head_counts(seq).tolist() == [1, 1, 1, 1, 1]


def test_stateless_sampler_resume_does_not_touch_global_rng(tmp_path):
    rows = [dict(replay_tag=str(i), owner=i % 2, outcome="win", weight=1.0,
                 cache_status="completed", cached_hog_count=2,
                 cached_episode_start=True, cached_natural_time=True) for i in range(20)]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(dict(schema="v4-hog-expert-selection.v1", rows=rows,
                                   cache_labels_verified=True, discard_tail_ticks=200)))
    first = HogExpertSamplerV4(path, seed=3)
    resumed = HogExpertSamplerV4(path, seed=3)
    state = random.getstate()
    assert first.rows_for_update(29, 3, 8) == resumed.rows_for_update(29, 3, 8)
    assert first.rows_for_update(29, 3, 8) != first.rows_for_update(30, 3, 8)
    assert state == random.getstate()
    assert first.contract() == resumed.contract()


def test_sampler_rejects_source_only_manifest_and_zero_cached_hogs(tmp_path):
    path=tmp_path/"unsafe.json"
    row=dict(replay_tag="02RY9PCCCJ9R", owner=1, outcome="win", weight=1.,
             cache_status="trusted_prefix", cached_hog_count=0,
             cached_episode_start=True, cached_natural_time=True)
    payload=dict(schema="v4-hog-expert-selection.v1", rows=[row], discard_tail_ticks=200)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="actual cached labels"):
        HogExpertSamplerV4(path,seed=1)
    payload["cache_labels_verified"]=True
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="audited Hog labels"):
        HogExpertSamplerV4(path,seed=1)


def test_cached_quality_counts_only_the_owner_and_trusted_hog_actions():
    actions=_wait_actions(5)
    actions.gate[:]=GATE_ACT
    actions.micro_action_count[:]=1
    actions.candidate_index[:,0]=0
    valid=torch.tensor([[True],[True],[False],[True],[True]])
    native_ids=torch.full((5,1),26000021,dtype=torch.long)
    shard=SimpleNamespace(descriptors=[SimpleNamespace(offset=1,length=3,natural_time=True)],
                          actions=actions,valid_mask=valid,gate_loss_mask=valid.clone(),
                          episode_start=torch.tensor([[False],[True],[False],[False],[False]]),
                          observations=SimpleNamespace(candidates=SimpleNamespace(native_visible_card_id=native_ids)))
    q=cached_sequence_quality(shard,0)
    assert q["cached_hog_count"]==2 and q["cached_hog_frames"]==[0,2]
    assert q["cached_valid_frames"]==2 and q["cached_episode_start"]
    native_ids[1]=26000030
    native_ids[3]=26000030
    assert cached_sequence_quality(shard,0)["cached_hog_count"]==0


def test_hog_probability_telemetry_uses_rollout_temperature_without_gradient():
    actions=(_one_action(1),_one_action(1),_one_action(1))
    obs=SimpleNamespace(candidates=SimpleNamespace(native_visible_card_id=torch.tensor([[26000021]])))
    sequence=SimpleNamespace(observations=(obs,obs,obs),actions=actions,
                             valid_mask=torch.tensor([[True],[True],[False]]))
    gate=torch.tensor([[.2],[.8],[.99]],dtype=torch.float64,requires_grad=True)
    card=torch.tensor([[[.5,1.]],[[.25,1.]],[[1.,1.]]],dtype=torch.float64,requires_grad=True)
    evaluation=SimpleNamespace(components=SimpleNamespace(gate_log_prob=gate.log(),candidate_log_prob=card.log()))
    result=expert_hog_probability_sums(sequence,evaluation,rollout_gate_temperature=.2)
    expected_gate=torch.sigmoid(torch.logit(gate.detach()[:2])/.2).flatten()
    assert not result.requires_grad and gate.grad is None and card.grad is None
    assert result[0]==2 and result[1]==1 and result[3]==.75 and result[5]==1
    torch.testing.assert_close(result[2],expected_gate.sum())
    torch.testing.assert_close(result[4],(expected_gate*torch.tensor([.5,.25])).sum())


def test_bc_does_not_move_unused_critic_adam_momentum():
    actor = torch.nn.Parameter(torch.tensor(1.0))
    critic = torch.nn.Parameter(torch.tensor(2.0))
    opt = torch.optim.AdamW([actor, critic], lr=.01, weight_decay=0)
    (actor + critic).backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    before = critic.detach().clone()
    actor.square().backward()
    _reduce_actor_gradients([actor, critic])
    assert critic.grad is None
    opt.step()
    torch.testing.assert_close(critic, before, rtol=0, atol=0)


@pytest.mark.parametrize("hog_gate_weight,hog_gate_scope,marker", [
    (0.0, "all", None), (0.5, "all", None),
    (0.5, "proactive-first", 0), (0.5, "proactive-first", 4), (0.5, "proactive-first", None)])
@pytest.mark.parametrize("gradient_scope", ["full", "gate-head-only"])
def test_bc_real_model_step_carries_burnin_and_leaves_value_head(monkeypatch, hog_gate_weight, hog_gate_scope, marker, gradient_scope):
    import native_runner.training.v4.ppo_expert_bc as bc
    model = _model()
    batch = _batch(batch_size=1)
    batch.candidates.native_visible_card_id[0, 0] = 26000021
    valid = torch.tensor([[True], [True], [False], [False], [True], [True]])
    seq = ILSequenceV4((batch,)*6, (_one_action(1), _wait_actions(1))*3,
                       torch.tensor([[True], [False], [False], [False], [False], [False]]),
                       valid, torch.full((6, 1), 1000.0), gate_loss_mask=valid,
                       value_loss_mask=torch.zeros_like(valid))
    metadata = {"outcome": "win", "proactive_first_hog_frame": marker}
    sampler = SimpleNamespace(sample=lambda *args: [ExpertBatchItemV4(seq, metadata, 2, 2)])
    original = bc.evaluate_recurrent_sequence
    calls = []
    def traced(*args, **kwargs):
        calls.append((torch.is_grad_enabled(), kwargs["initial_state"]))
        return original(*args, **kwargs)
    monkeypatch.setattr(bc, "evaluate_recurrent_sequence", traced)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    metrics = distributed_expert_bc_update_v4(model, opt, sampler, update_step=1,
                                              sequence_chunk_steps=2, validate=True,
                                              hog_gate_weight=hog_gate_weight, hog_gate_scope=hog_gate_scope,
                                              hog_gate_gradient_scope=gradient_scope)
    assert [c[0] for c in calls] == [True, False, True]
    assert calls[0][1] is None and all(c[1] is not None for c in calls[1:])
    assert metrics["bc_optimizer_steps"] == 1 and metrics["bc_supervised_frames"] == 4
    assert metrics["bc_candidate_count"] == 2
    if hog_gate_weight:
        count = 2 if hog_gate_scope == "all" else int(marker is not None)
        assert metrics["bc_hog_gate_count"] == count
        assert (metrics["bc_hog_gate_loss"] > 0) == (count > 0)
        if hog_gate_scope == "proactive-first":
            assert metrics["bc_proactive_hog_gate_empty_update"] == int(count == 0)
        if gradient_scope == "gate-head-only":
            assert metrics["bc_hog_gate_head_only"] == 1
    assert any(not torch.equal(before[n], p) for n, p in model.named_parameters())
    value_names = [n for n, _ in model.named_parameters() if n.startswith("value_head.")]
    assert value_names
    for n, p in model.named_parameters():
        if n in value_names:
            torch.testing.assert_close(p, before[n], rtol=0, atol=0)


def test_policy_probe_uses_rollout_temperatures_and_preserves_masks():
    model = _model().eval()
    batch = _batch(batch_size=1)
    rollout = SimpleNamespace(time_steps=2, batch_size=1, observations=(batch, batch),
                              actions=(_one_action(1), _wait_actions(1)),
                              episode_start=torch.tensor([[True], [False]]),
                              valid_mask=torch.ones(2, 1, dtype=torch.bool),
                              forced_gate_mask=torch.tensor([[True], [False]]),
                              initial_state=model.initial_state(1), gate_temperature=.2,
                              action_temperature=1., continue_temperature=5.)
    logp, valid = _probe_log_probs(model, rollout, device=torch.device("cpu"), validate=True)
    assert torch.isfinite(logp).all()
    assert valid.tolist() == [[False], [True]]
    assert rollout.valid_mask.all()


def test_hog_gate_supervision_preserves_all_original_masks_and_counts_frames_once():
    obs = SimpleNamespace(candidates=SimpleNamespace(native_visible_card_id=torch.tensor([[26000021,26000030]])))
    actions = [_one_action(1) for _ in range(6)]
    actions[3] = _wait_actions(1)
    actions[4].candidate_index[0,0] = 1
    actions[5].micro_action_count[0] = 2
    actions[5].candidate_index[0,1] = 0
    seq = SimpleNamespace(observations=(obs,)*6, actions=tuple(actions),
                          valid_mask=torch.tensor([[True],[True],[False],[True],[True],[True]]),
                          gate_loss_mask=torch.tensor([[True],[False],[True],[True],[True],[True]]))
    assert hog_gate_mask(seq).flatten().tolist() == [True,False,False,False,False,True]
    logp = torch.tensor([[-1.],[-2.],[-3.],[-4.],[-5.],[-6.]], requires_grad=True)
    ev = SimpleNamespace(components=SimpleNamespace(gate_log_prob=logp))
    total, count = hog_gate_nll_sum(seq,ev)
    assert total == 7 and count == 2
    total.backward()
    assert logp.grad.flatten().tolist() == [-1,0,0,0,0,-1]


def test_optional_hog_gate_contract_is_strict_and_default_compatible():
    sampler=SimpleNamespace(contract=lambda:dict(schema="v4-ppo-expert-bc.v2",seed=7))
    old={**sampler.contract(),"coefficient":.2}
    assert expert_bc_contract_v4(sampler) == old
    new=expert_bc_contract_v4(sampler,hog_gate_weight=.5)
    assert new != old and new["schema"] == "v4-ppo-expert-bc.v3"
    assert new["hog_gate_supervision"]["weight"] == .5
    for bad in (-1,float("nan"),float("inf")):
        with pytest.raises(ValueError):
            expert_bc_contract_v4(sampler,hog_gate_weight=bad)


def test_proactive_first_gate_only_uses_absolute_first_frame_and_original_masks():
    obs = SimpleNamespace(candidates=SimpleNamespace(native_visible_card_id=torch.tensor([[26000021]])))
    actions = [_one_action(1) for _ in range(6)]
    actions[3] = _wait_actions(1)
    seq = SimpleNamespace(observations=(obs,)*6, actions=tuple(actions),
                          valid_mask=torch.tensor([[True],[True],[False],[True],[True],[True]]),
                          gate_loss_mask=torch.tensor([[True],[False],[True],[True],[True],[True]]))
    before = (seq.valid_mask.clone(), seq.gate_loss_mask.clone())
    for marker, count in [(None,0), (48,1), (49,0), (50,0), (51,0), (52,1), (53,1), (54,0)]:
        meta = dict(proactive_first_hog_frame=marker)
        mask = hog_gate_mask(seq, scope="proactive-first", metadata=meta, frame_offset=48)
        assert int(mask.sum()) == count
        if count: assert bool(mask[marker-48,0])
    logp = torch.full((6,1), -2., requires_grad=True)
    ev = SimpleNamespace(components=SimpleNamespace(gate_log_prob=logp))
    nll, count = hog_gate_nll_sum(seq, ev, scope="proactive-first",
                                metadata=dict(proactive_first_hog_frame=52), frame_offset=48)
    assert nll == 2 and count == 1
    nll.backward()
    assert logp.grad.flatten().tolist() == [0,0,0,0,-1,0]
    assert torch.equal(seq.valid_mask, before[0]) and torch.equal(seq.gate_loss_mask, before[1])
    for meta in (None, {}, dict(proactive_first_hog_frame=True), dict(proactive_first_hog_frame=-1)):
        with pytest.raises(ValueError): hog_gate_mask(seq, scope="proactive-first", metadata=meta)


def test_proactive_contract_requires_provenance_and_preserves_zero_weight_contract():
    sampler = SimpleNamespace(contract=lambda: dict(schema="v4-ppo-expert-bc.v2", seed=7))
    assert expert_bc_contract_v4(sampler, hog_gate_scope="proactive-first") == expert_bc_contract_v4(sampler)
    with pytest.raises(ValueError):
        expert_bc_contract_v4(sampler, hog_gate_weight=.5, hog_gate_scope="proactive-first")
    sampler.proactive_opening_contract = dict(schema="v4-hog-proactive-opening-gate.v1", source_context_sha256="test")
    new = expert_bc_contract_v4(sampler, hog_gate_weight=.5, hog_gate_scope="proactive-first")
    assert new["schema"] == "v4-ppo-expert-bc.v4"
    assert new["hog_gate_supervision"]["scope"] == "proactive-first"
    assert new["hog_gate_supervision"]["annotation"] == sampler.proactive_opening_contract
    assert new != expert_bc_contract_v4(sampler, hog_gate_weight=.5)


def test_sampler_proactive_annotations_fail_closed(tmp_path):
    row = dict(replay_tag="first", owner=1, outcome="win", weight=1., cache_status="completed",
               cached_hog_count=2, cached_hog_frames=[33,100], cached_episode_start=True,
               cached_natural_time=True, sequence_length=200, source_hog_ticks=[300,635],
               proactive_first_hog_frame=33, proactive_first_enemy_tick=301)
    payload = dict(schema="v4-hog-expert-selection.v1", discard_tail_ticks=200, cache_labels_verified=True,
                   rows=[row], proactive_opening_gate=dict(schema="v4-hog-proactive-opening-gate.v1",
                   primary_cache_start_tick=130, decision_stride_ticks=5, eligible_sides=1))
    path = tmp_path/"manifest.json"
    path.write_text(json.dumps(payload))
    assert HogExpertSamplerV4(path, seed=1).rows[0]["proactive_first_hog_frame"] == 33
    for key, value in [("proactive_first_hog_frame", 34), ("proactive_first_hog_frame", True),
                       ("proactive_first_enemy_tick", 300), ("proactive_first_enemy_tick", 299)]:
        bad = json.loads(json.dumps(payload)); bad["rows"][0][key] = value
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError): HogExpertSamplerV4(path, seed=1)
    bad = json.loads(json.dumps(payload)); bad["proactive_opening_gate"]["eligible_sides"] = 2
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError): HogExpertSamplerV4(path, seed=1)


def _partitioned_expert_items(rank, empty):
    batch = _batch(batch_size=1)
    batch.candidates.native_visible_card_id[0,0] = 26000021
    valid = torch.tensor([[True],[True],[False],[False],[True],[True]])
    seq = ILSequenceV4((batch,)*6, (_one_action(1), _wait_actions(1))*3,
                      torch.tensor([[True],[False],[False],[False],[False],[False]]),
                      valid, torch.zeros(6,1), gate_loss_mask=valid,
                      value_loss_mask=torch.zeros_like(valid))
    return [ExpertBatchItemV4(seq, dict(outcome="win", proactive_first_hog_frame=(
            None if empty or rank == 1 else frame)), 2, 2) for frame in (0,4)]


def _run_partitioned_bc(rank, world, init_path, output_dir, empty, gradient_scope):
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=Path(init_path).resolve().as_uri(),
        rank=rank, world_size=world, timeout=timedelta(seconds=30),
    )
    torch.manual_seed(731)
    model = _model()
    opt = torch.optim.SGD(model.parameters(), lr=.001)
    sampler = SimpleNamespace(sample=lambda *unused: _partitioned_expert_items(rank, empty))
    metrics = distributed_expert_bc_update_v4(model, opt, sampler, update_step=1,
        sequence_chunk_steps=2, hog_gate_weight=.5, hog_gate_scope="proactive-first", validate=True,
        hog_gate_gradient_scope=gradient_scope)
    torch.save(dict(state=model.state_dict(), metrics=metrics), Path(output_dir)/f"rank{rank}.pt")
    dist.destroy_process_group()


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("gradient_scope", ["full", "gate-head-only"])
def test_proactive_global_normalization_matches_serial_with_empty_local_ranks(tmp_path, empty, gradient_scope):
    import torch.multiprocessing as mp
    torch.manual_seed(731)
    model = _model()
    before = {n:p.detach().clone() for n,p in model.named_parameters()}
    opt = torch.optim.SGD(model.parameters(), lr=.001)
    sampler = SimpleNamespace(sample=lambda *unused: _partitioned_expert_items(0,empty)+_partitioned_expert_items(1,empty))
    metrics = distributed_expert_bc_update_v4(model, opt, sampler, update_step=1,
        sequence_chunk_steps=2, hog_gate_weight=.5, hog_gate_scope="proactive-first", validate=True,
        hog_gate_gradient_scope=gradient_scope)
    assert metrics["bc_hog_gate_count"] == (0 if empty else 2)
    assert metrics["bc_proactive_hog_gate_empty_update"] == int(empty)
    mp.spawn(_run_partitioned_bc, args=(2,str(tmp_path/"gloo-init"),str(tmp_path),empty,gradient_scope), nprocs=2, join=True)
    for rank in range(2):
        actual = torch.load(tmp_path/f"rank{rank}.pt", weights_only=True)
        for name, expected in model.state_dict().items():
            torch.testing.assert_close(actual["state"][name], expected, rtol=1e-5, atol=2e-7)
        for key in ("bc_loss", "bc_hog_gate_loss", "bc_gate_loss", "bc_candidate_loss", "bc_hog_gate_count"):
            assert actual["metrics"][key] == pytest.approx(metrics[key], rel=1e-6, abs=1e-7)
    for name, parameter in model.named_parameters():
        if name.startswith("value_head."):
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
    assert any(not torch.equal(p,before[n]) for n,p in model.named_parameters())


def test_legacy_all_hog_gate_scope_still_rejects_empty_auxiliary_population():
    model = _model()
    items = _partitioned_expert_items(0, True)
    for item in items:
        for observation in item.sequence.observations:
            observation.candidates.native_visible_card_id[0,0] = 26000030
    sampler = SimpleNamespace(sample=lambda *unused: items)
    with pytest.raises(RuntimeError, match="trusted gate-supervised Hog"):
        distributed_expert_bc_update_v4(model, torch.optim.SGD(model.parameters(), lr=.001),
            sampler, update_step=1, sequence_chunk_steps=2, hog_gate_weight=.5)


def test_gate_only_auxiliary_changes_only_gate_raw_gradients_but_keeps_normal_bc():
    torch.manual_seed(99)
    items = _partitioned_expert_items(0, False)
    prototype = _model()
    # A fresh toy gate can have zero output weights: ensure a nonzero Jacobian
    # into its input so the full-gradient comparison can detect backbone effects.
    with torch.no_grad():
        prototype.gate_head[-1].weight.normal_(std=.05)
    initial = {k:v.detach().clone() for k,v in prototype.state_dict().items()}
    def run(weight, scope):
        model = _model()
        model.load_state_dict(initial)
        opt = torch.optim.SGD(model.parameters(), lr=.001)
        sampler = SimpleNamespace(sample=lambda *unused: items)
        metrics = distributed_expert_bc_update_v4(model, opt, sampler, update_step=1,
            sequence_chunk_steps=2, hog_gate_weight=weight, hog_gate_scope="proactive-first",
            hog_gate_gradient_scope=scope, max_grad_norm=1e6, validate=True)
        return model.state_dict(), metrics
    # Disable clipping ONLY in this raw-gradient routing test. Production retains
    # shared clipping and can rescale normal gradients when the gate norm changes.
    base, _ = run(0., "full")
    head, head_metrics = run(.5, "gate-head-only")
    full, full_metrics = run(.5, "full")
    nongate = [n for n in initial if not n.startswith("gate_head.")]
    for name in nongate:
        torch.testing.assert_close(head[name], base[name], rtol=0, atol=0)
    assert any(not torch.equal(head[n],base[n]) for n in initial if n.startswith("gate_head."))
    assert any(not torch.equal(full[n],base[n]) for n in nongate)
    assert any(not torch.equal(head[n],initial[n]) for n in nongate), "normal BC was accidentally restricted"
    for name in initial:
        if name.startswith("value_head."):
            torch.testing.assert_close(head[name], initial[name], rtol=0, atol=0)
    assert head_metrics["bc_optimizer_steps"] == full_metrics["bc_optimizer_steps"] == 1
    assert head_metrics["bc_hog_gate_count"] == full_metrics["bc_hog_gate_count"] == 2
    assert head_metrics["bc_loss"] == pytest.approx(full_metrics["bc_loss"], rel=1e-7)


def test_gate_only_auxiliary_contract_is_strict_and_legacy_defaults_unchanged():
    sampler = SimpleNamespace(contract=lambda:dict(schema="v4-ppo-expert-bc.v2", seed=7),
        proactive_opening_contract=dict(schema="v4-hog-proactive-opening-gate.v1", source_context_sha256="context"))
    assert expert_bc_contract_v4(sampler, hog_gate_gradient_scope="gate-head-only") == expert_bc_contract_v4(sampler)
    for scope, schema in (("all", "v4-ppo-expert-bc.v3"), ("proactive-first", "v4-ppo-expert-bc.v4")):
        old = expert_bc_contract_v4(sampler, hog_gate_weight=.5, hog_gate_scope=scope)
        explicit = expert_bc_contract_v4(sampler, hog_gate_weight=.5, hog_gate_scope=scope,
                                       hog_gate_gradient_scope="full")
        new = expert_bc_contract_v4(sampler, hog_gate_weight=.5, hog_gate_scope=scope,
                                  hog_gate_gradient_scope="gate-head-only")
        assert old == explicit and old["schema"] == schema
        assert new["schema"] == "v4-ppo-expert-bc.v5" and new != old
        assert new["hog_gate_supervision"]["gradient_scope"] == "gate-head-only"
        assert new["hog_gate_supervision"]["normal_bc_gradient_scope"] == "full"
    with pytest.raises(ValueError):
        expert_bc_contract_v4(sampler, hog_gate_gradient_scope="typo")


def test_gate_only_auxiliary_requires_trainable_gate_parameters():
    model = _model()
    for parameter in model.gate_head.parameters():
        parameter.requires_grad_(False)
    with pytest.raises(ValueError, match="trainable gate_head"):
        distributed_expert_bc_update_v4(model, torch.optim.SGD(model.parameters(), lr=.001),
            None, update_step=1, sequence_chunk_steps=2, hog_gate_weight=.5,
            hog_gate_gradient_scope="gate-head-only")
