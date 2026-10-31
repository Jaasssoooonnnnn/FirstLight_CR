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


# --------------------------------------------------------------------------- generalized activation

from native_runner.training.v4.king_activation_reward import (  # noqa: E402
    DEFAULT_KING_ACTIVATION_PENALTY,
    DEFAULT_KING_ACTIVATION_SCOPE,
    apply_king_activation_reward_v4,
    configure_king_tower_shaping_v4,
    king_activation_source_v4,
    validate_king_reward_parameters_v4,
)

ARROWS, HOG, CANNON = 28000001, 26000021, 27000000


def _runtime(events, current_owners=(0,)):
    env = NS(raw_observation={}, _rich_snapshot_for=lambda raw: NS(combat_events=NS(events=list(events))),
             _canonical_tick=lambda t: t)
    return NS(assignment=NS(matchup=NS(current_owners=current_owners)), environment=env,
              king_activation_seen_owners=set(), king_activations=(0, 0), king_activation_tick=(None, None),
              king_activation_source_card_id=(None, None), king_activation_source_unattributed=(0, 0),
              king_activation_reward_total=(0., 0.), king_activation_details=[])


def _transition(attacker, *, king_hp=3700):
    """Both seats' views of one step in which ``attacker`` hits the sleeping enemy King."""
    b = board(1 - attacker) + board(attacker)
    victim_king = b[0]
    a = (replace(victim_king, hitpoints=king_hp), *b[1:])
    before = {o: NS(tick=400, towers=b) for o in (0, 1)}
    after = {o: NS(tick=405, towers=a) for o in (0, 1)}
    return before, after, victim_king


def test_defaults_match_plan():
    assert DEFAULT_KING_ACTIVATION_PENALTY == 0.1 and DEFAULT_KING_ACTIVATION_SCOPE == "any"


@pytest.mark.parametrize("attacker", [0, 1])
@pytest.mark.parametrize("current_owners", [(0,), (1,), (0, 1)])
def test_activation_is_zero_sum_for_either_side_and_any_learner_seat(attacker, current_owners):
    before, after, king = _transition(attacker)
    runtime = _runtime([damage(king, attacker, card=HOG)], current_owners)
    reward = {0: .01, 1: -.01}
    apply_king_activation_reward_v4(runtime, before, after, reward, .1, "any")
    victim = 1 - attacker
    base = {0: .01, 1: -.01}
    assert reward[attacker] == pytest.approx(base[attacker] - .1)
    assert reward[victim] == pytest.approx(base[victim] + .1)
    assert reward[0] + reward[1] == pytest.approx(0.)
    expected_counts = [0, 0]
    expected_counts[attacker] = 1
    assert runtime.king_activations == tuple(expected_counts)
    totals = runtime.king_activation_reward_total
    assert totals[attacker] == pytest.approx(-.1) and totals[victim] == pytest.approx(.1)
    detail, = runtime.king_activation_details
    assert (detail["owner"], detail["victim_owner"], detail["tick"]) == (attacker, victim, 405)
    assert (detail["source_card_id"], detail["source_family"], detail["penalized"]) == (HOG, "troop", True)


def test_activation_triggers_only_once_per_king_across_steps_and_segments():
    before, after, king = _transition(0)
    runtime = _runtime([damage(king, 0, card=ARROWS)])
    reward = {0: 0., 1: 0.}
    for _ in range(3):
        apply_king_activation_reward_v4(runtime, before, after, reward, .1, "any")
    assert reward == {0: pytest.approx(-.1), 1: pytest.approx(.1)}
    # A later hit on the (now damaged) King is not a first activation either.
    later_before = {o: NS(tick=405, towers=after[o].towers) for o in (0, 1)}
    later_after = {o: NS(tick=410, towers=(replace(after[o].towers[0], hitpoints=3000), *after[o].towers[1:]))
                   for o in (0, 1)}
    runtime.king_activation_seen_owners.clear()
    apply_king_activation_reward_v4(runtime, later_before, later_after, reward, .1, "any")
    assert runtime.king_activations == (1, 0) and len(runtime.king_activation_details) == 1


@pytest.mark.parametrize("card,family", [(28000000, "spell"), (ARROWS, "spell"), (HOG, "troop"), (CANNON, "building")])
def test_source_card_and_family_are_identified(card, family):
    before, after, king = _transition(0)
    runtime = _runtime([damage(king, 0, card=card)])
    apply_king_activation_reward_v4(runtime, before, after, {0: 0., 1: 0.}, 0., "any")
    assert runtime.king_activation_source_card_id == (card, None)
    assert runtime.king_activation_details[0]["source_family"] == family


def test_source_falls_back_to_validated_immediate_source_and_tolerates_float_positions():
    king = board()[0]
    event = damage(king, 0, card=HOG)
    event.deployment_context = None
    event.target.position = (8500.4, 29499.6)
    evidence = king_activation_source_v4(king, [event], owner=0, before_tick=400, after_tick=405,
                                         canonical_tick=lambda t: t)
    assert evidence["source_card_id"] == HOG and evidence["source_family"] == "troop"


@pytest.mark.parametrize("scope,card,penalized", [
    ("spell", ARROWS, True), ("spell", HOG, False), ("spell", CANNON, False),
    ("fireball", 28000000, True), ("fireball", ARROWS, False),
    ("any", HOG, True), ("any", CANNON, True),
])
def test_scope_controls_reward_but_never_statistics(scope, card, penalized):
    before, after, king = _transition(0)
    runtime = _runtime([damage(king, 0, card=card)])
    reward = {0: 0., 1: 0.}
    apply_king_activation_reward_v4(runtime, before, after, reward, .1, scope)
    assert runtime.king_activations == (1, 0)
    assert runtime.king_activation_details[0]["penalized"] is penalized
    assert reward[0] == pytest.approx(-.1 if penalized else 0.)
    assert reward[1] == pytest.approx(.1 if penalized else 0.)


@pytest.mark.parametrize("events", ["none", "enemy_source", "wrong_tower"])
def test_unattributed_source_still_counts_and_any_scope_still_penalizes(events):
    before, after, king = _transition(0)
    if events == "none":
        evs = []
    elif events == "enemy_source":
        evs = [damage(king, 1, card=HOG)]
    else:
        evs = [damage(king, 0, card=HOG)]
        evs[0].target.position = (3500, 25500)
    runtime = _runtime(evs)
    reward = {0: 0., 1: 0.}
    apply_king_activation_reward_v4(runtime, before, after, reward, .1, "any")
    assert runtime.king_activations == (1, 0)
    assert runtime.king_activation_source_unattributed == (1, 0)
    assert runtime.king_activation_source_card_id == (None, None)
    assert runtime.king_activation_details[0]["source_family"] == "unknown"
    assert runtime.king_activation_details[0]["tick"] == 405
    assert reward == {0: pytest.approx(-.1), 1: pytest.approx(.1)}
    runtime = _runtime(evs)
    reward = {0: 0., 1: 0.}
    apply_king_activation_reward_v4(runtime, before, after, reward, .1, "spell")
    assert reward == {0: 0., 1: 0.} and runtime.king_activations == (1, 0)


def test_zero_coefficient_records_statistics_without_touching_reward():
    before, after, king = _transition(1)
    runtime = _runtime([damage(king, 1, card=ARROWS, tick=403)])
    reward = {0: .25, 1: -.25}
    apply_king_activation_reward_v4(runtime, before, after, reward, 0., "any")
    assert reward == {0: .25, 1: -.25}
    assert runtime.king_activations == (0, 1)
    assert runtime.king_activation_tick == (None, 403)
    assert runtime.king_activation_source_card_id == (None, ARROWS)
    assert runtime.king_activation_reward_total == (0., 0.)
    assert runtime.king_activation_details[0]["penalized"] is False


def test_natural_activation_and_normal_attacks_are_not_activations():
    b = board(1) + board(0)
    a = (replace(b[0], hitpoints=3700), replace(b[1], hitpoints=0), *b[2:])
    runtime = _runtime([])
    apply_king_activation_reward_v4(runtime, {o: NS(tick=400, towers=b) for o in (0, 1)},
                                    {o: NS(tick=405, towers=a) for o in (0, 1)}, {0: 0., 1: 0.}, .1, "any")
    assert runtime.king_activations == (0, 0) and runtime.king_activation_details == []


def test_generalized_and_legacy_fireball_paths_keep_separate_state():
    before, after, king = _transition(0)
    runtime = _runtime([damage(king, 0)])
    runtime.seen_king_activation_owners = set()
    runtime.sleeping_enemy_king_damage_count = (0, 0)
    runtime.king_activation_not_attributed_to_fireball = (0, 0)
    runtime.fireball_king_activations = (0, 0)
    runtime.fireball_king_activation_penalty_total = (0., 0.)
    runtime.fireball_king_activation_details = []
    reward = {0: 0., 1: 0.}
    apply_fireball_king_activation_penalty_v4(runtime, before, after, reward, .5)
    apply_king_activation_reward_v4(runtime, before, after, reward, .1, "any")
    assert runtime.fireball_king_activations == (1, 0) and runtime.king_activations == (1, 0)
    assert reward == {0: pytest.approx(-.6), 1: pytest.approx(.1)}


@pytest.mark.parametrize("kwargs", [
    dict(king_activation_penalty=-.1), dict(king_activation_penalty=float("nan")),
    dict(king_activation_scope="troop"), dict(king_shaping_mode="awake_only"),
    dict(king_potential_weight=float("inf")), dict(king_potential_weight=-1.),
])
def test_invalid_king_reward_parameters_are_rejected(kwargs):
    good = dict(king_activation_penalty=.1, king_activation_scope="any", king_shaping_mode="legacy",
                king_potential_weight=3.)
    validate_king_reward_parameters_v4(**good)
    with pytest.raises(ValueError):
        validate_king_reward_parameters_v4(**{**good, **kwargs})
    before, after, _ = _transition(0)
    if "king_activation_penalty" in kwargs or "king_activation_scope" in kwargs:
        with pytest.raises(ValueError):
            apply_king_activation_reward_v4(_runtime([]), before, after, {0: 0., 1: 0.},
                                            kwargs.get("king_activation_penalty", .1),
                                            kwargs.get("king_activation_scope", "any"))


def test_configure_helper_applies_to_engine_environments_and_refuses_silent_drop():
    seen = []
    engine = NS(environments=(NS(configure_tower_shaping=lambda **kw: seen.append(kw)),) * 2)
    configure_king_tower_shaping_v4(engine, king_shaping_mode="princess_gated", king_potential_weight=3.)
    assert seen == [dict(king_shaping_mode="princess_gated", king_potential_weight=3.)] * 2
    configure_king_tower_shaping_v4(NS(), king_shaping_mode="legacy", king_potential_weight=3.)
    with pytest.raises(RuntimeError):
        configure_king_tower_shaping_v4(NS(), king_shaping_mode="princess_gated", king_potential_weight=3.)


@pytest.mark.parametrize("extra", [
    dict(king_activation_penalty=0., king_activation_scope="any", king_shaping_mode="legacy",
         king_potential_weight=3.),
    dict(king_activation_penalty=.1, king_activation_scope="spell", king_shaping_mode="princess_gated",
         king_potential_weight=1.6),
])
def test_remote_worker_forwards_king_reward_keys_including_explicit_defaults(monkeypatch, extra):
    from native_runner.training.v4 import remote_ppo_worker_server as server
    from native_runner.training.v4.async_cluster_self_play import AsyncResidentEngineSpecV4
    got = {}
    monkeypatch.setattr(server, "_prepare_worker_process", lambda _: None)
    monkeypatch.setattr(server, "_engine_worker", lambda *args, **kw: got.update(kw))
    server._run_worker(None, dict(spec=object.__new__(AsyncResidentEngineSpecV4), assignments=(),
        gamma_per_decision=.9997, shaping_beta=.05, mutual_elixir_overflow_penalty=0.,
        mutual_elixir_overflow_grace=4., unilateral_elixir_overflow_penalty=0.,
        elixir_overflow_step_penalty_cap=.1, policy_state_id="test", validate_tensors=False,
        max_decision_steps=1204, **extra), -1)
    assert got == extra


def test_collector_entry_points_accept_king_reward_keys_with_neutral_defaults():
    import inspect
    from native_runner.training.v4 import async_cluster_self_play as collector
    names = ("king_activation_penalty", "king_activation_scope", "king_shaping_mode", "king_potential_weight")
    for function in (collector._engine_worker, collector.collect_async_cluster_wave_v4):
        params = inspect.signature(function).parameters
        assert tuple(params[name].default for name in names) == (0.0, "any", "legacy", 3.0)


def test_worker_terminal_report_carries_king_activation_statistics(monkeypatch):
    from native_runner.tests.test_training_v4_head_to_head import _done_report, run_fake_worker
    connection, _state = run_fake_worker(monkeypatch, end_tick=150)
    report = _done_report(connection)
    assert report["king_activations"] == (0, 0)
    assert report["king_activation_tick"] == (None, None)
    assert report["king_activation_source_card_id"] == (None, None)
    assert report["king_activation_source_unattributed"] == (0, 0)
    assert report["king_activation_reward_total"] == (0., 0.)
    assert report["king_activation_details"] == ()
    assert report["king_shaping_mode"] == "legacy"


def test_match_runtime_carries_king_activation_fields():
    from native_runner.training.v4.cluster_self_play import _MatchRuntimeV4
    assert {"king_activation_seen_owners", "king_activations", "king_activation_tick",
            "king_activation_source_card_id", "king_activation_source_unattributed",
            "king_activation_reward_total", "king_activation_details"} <= set(_MatchRuntimeV4.__dataclass_fields__)
