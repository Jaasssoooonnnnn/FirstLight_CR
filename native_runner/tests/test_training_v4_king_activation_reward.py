from types import SimpleNamespace as NS
from dataclasses import replace

import pytest

from native_runner.contracts import TowerStateV1
from native_runner.training.v4.king_activation_reward import (
    sleeping_king_damage_v4, fireball_activation_evidence_v4,
    apply_fireball_king_activation_penalty_v4,
)


def board(owner=1):
    return (TowerStateV1(entity_id=owner*3, owner=owner, tower_kind="king",
                         position=(8500, 29500 if owner else 2500),
                         hitpoints=4000, max_hitpoints=4000),
            TowerStateV1(entity_id=owner*3+1, owner=owner, tower_kind="princess",
                         position=(3500, 25500), hitpoints=3000,
                         max_hitpoints=3000, tower_troop_id=159000000),
            TowerStateV1(entity_id=owner*3+2, owner=owner, tower_kind="princess",
                         position=(13500, 25500), hitpoints=3000,
                         max_hitpoints=3000, tower_troop_id=159000000))


def damage(king, owner, *, card=28000000, tick=405, pre=4000, post=3700):
    absent=NS(present=False, validated=False, owner=None, card_id=None)
    source=NS(present=True, validated=True, owner=owner, card_id=card)
    return NS(kind="damage", tick=tick, sequence=17, pre_hp=pre, post_hp=post,
              deployment_context=NS(played_card_global_id=card, owner=owner),
              immediate_source=source, source=source, projectile=absent,
              target=NS(present=True, validated=True, owner=king.owner, position=king.position))


@pytest.mark.parametrize("owner", [0,1])
def test_fireball_splash_on_sleeping_enemy_king_is_penalized(owner):
    b=board(1-owner); a=(replace(b[0],hitpoints=3700),*b[1:])
    king=sleeping_king_damage_v4(b,a,owner)
    assert king==b[0]
    # No cast-position requirement: damaging a king while aiming at a nearby
    # Mother Witch has exactly the same causal Fireball hit event.
    evidence=fireball_activation_evidence_v4(king,[damage(king,owner)],owner=owner,
        before_tick=400,after_tick=405,canonical_tick=lambda t:t)
    assert evidence['source_card_id']==28000000 and evidence['owner']==owner


@pytest.mark.parametrize("mode", ["already_damaged","already_activated","side_lost_before","side_lost_now","lethal","no_damage"])
def test_does_not_penalize_normal_attack_or_natural_activation(mode):
    b=board();a=(replace(b[0],hitpoints=3700),*b[1:])
    if mode=='already_damaged':b=(replace(b[0],hitpoints=3900),*b[1:])
    if mode=='already_activated':b=(replace(b[0],status=('activated',)),*b[1:])
    if mode=='side_lost_before':b=(b[0],replace(b[1],hitpoints=0),b[2])
    if mode=='side_lost_now':a=(a[0],replace(a[1],hitpoints=0),a[2])
    if mode=='lethal':a=(replace(a[0],hitpoints=0),*a[1:])
    if mode=='no_damage':a=b
    assert sleeping_king_damage_v4(b,a,0) is None


@pytest.mark.parametrize("mode", ["other_card","wrong_owner","old_event","late_event","second_hit","wrong_tower"])
def test_only_actual_first_fireball_hp_damage_is_attributed(mode):
    k=board()[0]; e=damage(k,0)
    if mode=='other_card':e=damage(k,0,card=26000021)
    if mode=='wrong_owner':e=damage(k,1)
    if mode=='old_event':e.tick=400
    if mode=='late_event':e.tick=406
    if mode=='second_hit':e.pre_hp=3900
    if mode=='wrong_tower':e.target.position=(3500,25500)
    assert fireball_activation_evidence_v4(k,[e],owner=0,before_tick=400,after_tick=405,canonical_tick=lambda t:t) is None


def test_actual_reward_deduplicates_across_observations_and_segments():
    b=board();a=(replace(b[0],hitpoints=3700),*b[1:]);ev=damage(b[0],0)
    env=NS(raw_observation={},_rich_snapshot_for=lambda raw:NS(combat_events=NS(events=[ev])),_canonical_tick=lambda t:t)
    runtime=NS(assignment=NS(matchup=NS(current_owners=(0,))),environment=env,
        seen_king_activation_owners=set(),sleeping_enemy_king_damage_count=(0,0),
        king_activation_not_attributed_to_fireball=(0,0),fireball_king_activations=(0,0),
        fireball_king_activation_penalty_total=(0.,0.),fireball_king_activation_details=[])
    before={0:NS(tick=400,towers=b)};after={0:NS(tick=405,towers=a)}
    reward=[.02,0.]
    apply_fireball_king_activation_penalty_v4(runtime,before,after,reward,.5)
    assert reward==pytest.approx([-.48,0.])
    apply_fireball_king_activation_penalty_v4(runtime,before,after,reward,.5)
    assert reward==pytest.approx([-.48,0.])
    assert runtime.fireball_king_activations==(1,0)
    assert runtime.fireball_king_activation_penalty_total==(.5,0.)


def test_remote_worker_forwards_new_penalty(monkeypatch):
    from native_runner.training.v4 import remote_ppo_worker_server as server
    from native_runner.training.v4.async_cluster_self_play import AsyncResidentEngineSpecV4
    got={}
    monkeypatch.setattr(server,'_prepare_worker_process',lambda _:None)
    monkeypatch.setattr(server,'_engine_worker',lambda *args,**kw:got.update(kw))
    server._run_worker(None,dict(spec=object.__new__(AsyncResidentEngineSpecV4),assignments=(),
        gamma_per_decision=.9997,shaping_beta=.15,mutual_elixir_overflow_penalty=.03,
        mutual_elixir_overflow_grace=2.,unilateral_elixir_overflow_penalty=.15,
        elixir_overflow_step_penalty_cap=2.4,policy_state_id='test',validate_tensors=False,
        max_decision_steps=1204,fireball_king_activation_penalty=.5),-1)
    assert got=={'fireball_king_activation_penalty':.5}


def test_personal_four_times_and_unilateral_eight_times_before_cap():
    from native_runner.training.v4.async_cluster_self_play import _elixir_overflow_penalty_transition_v4 as f
    def step(p,u,cap):return f(personal_wasted_elixir=3.,personal_accrued_cost=p,
        unilateral_wasted_elixir=2.,unilateral_accrued_cost=4*u,before_elixir=10.,
        after_elixir=10.,generated_elixir=.1,both_full=False,personal_coefficient=p,
        personal_grace_elixir=2.,unilateral_coefficient=u,step_penalty_cap=cap)
    old=step(.0075,.01875,.3);new=step(.03,.15,2.4)
    assert new[4]==pytest.approx(4*old[4])
    assert new[5]==pytest.approx(8*old[5])
