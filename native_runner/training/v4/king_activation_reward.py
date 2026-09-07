"""Reward-only attribution of Fireball damage to a sleeping enemy King Tower.

Native combat facts are used only for reward; no oracle facts enter policy input.
Tower.active means alive, NOT awake. Sleeping requires full HP, no activation
status, and both Princess Towers still standing.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

FIREBALL_CARD_ID = 28_000_000


def sleeping_king_damage_v4(before: Sequence[Any], after: Sequence[Any], owner: int):
    """Return the enemy king on a nonlethal first-damage transition."""
    enemy = 1 - owner
    kings = [t for t in before if t.owner == enemy and t.tower_kind == "king"]
    if len(kings) != 1:
        return None
    king = kings[0]
    if "activated" in king.status or not math.isclose(king.hitpoints, king.max_hitpoints, abs_tol=1e-6):
        return None
    # Exclude natural activation through a lost side tower, including a loss
    # inside this observation interval, and exclude lethal King Tower damage.
    for towers in (before, after):
        side = [t for t in towers if t.owner == enemy and t.tower_kind != "king"]
        if len(side) != 2 or any(t.hitpoints <= 0 for t in side):
            return None
    new = next((t for t in after if t.entity_id == king.entity_id), None)
    if new is None or not 0 < new.hitpoints < king.hitpoints:
        return None
    return king


def fireball_activation_evidence_v4(
    king: Any,
    events: Sequence[Any],
    *,
    owner: int,
    before_tick: int,
    after_tick: int,
    canonical_tick: Callable[[int], int],
) -> dict[str, Any] | None:
    """Use the first HP-loss event, not proximity or merely a Fireball cast.

    The splash center need not be the king: native damage attribution also covers
    Fireballs aimed at an adjacent troop. If source attribution is unavailable,
    return None and let the caller record the unattributed transition.
    """
    for event in sorted(events, key=lambda e: (e.tick, e.sequence)):
        tick = canonical_tick(event.tick)
        target = event.target
        if not (
            event.kind == "damage"
            and before_tick < tick <= after_tick
            and target.present
            and target.validated
            and target.owner == king.owner
            and target.position is not None
            and tuple(target.position) == tuple(king.position)
            and event.pre_hp is not None
            and event.post_hp is not None
            and math.isclose(event.pre_hp, king.max_hitpoints, abs_tol=1e-6)
            and 0 < event.post_hp < event.pre_hp
        ):
            continue
        deployment = event.deployment_context
        if deployment is not None:
            card = deployment.played_card_global_id
            source_owner = deployment.owner
        else:
            sources = (event.immediate_source, event.source, event.projectile)
            source = next(
                (
                    s
                    for s in sources
                    if s.present and s.validated and s.owner == owner and s.card_id == FIREBALL_CARD_ID
                ),
                None,
            )
            card = None if source is None else source.card_id
            source_owner = None if source is None else source.owner
        if card != FIREBALL_CARD_ID or source_owner != owner:
            return None
        return {
            "owner": owner,
            "enemy_owner": king.owner,
            "king_entity_id": king.entity_id,
            "tick": tick,
            "native_sequence": event.sequence,
            "source_card_id": card,
            "pre_hp": event.pre_hp,
            "post_hp": event.post_hp,
        }
    return None


def apply_fireball_king_activation_penalty_v4(
    runtime: Any, before_observations: Any, after_observations: Any, adjusted_reward: list[float], coefficient: float
) -> None:
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("fireball activation penalty must be finite and nonnegative")
    if coefficient == 0:
        return
    for owner in runtime.assignment.matchup.current_owners:
        before, after = before_observations[owner], after_observations[owner]
        king = sleeping_king_damage_v4(before.towers, after.towers, owner)
        if king is None or owner in runtime.seen_king_activation_owners:
            continue
        runtime.seen_king_activation_owners.add(owner)
        counts = list(runtime.sleeping_enemy_king_damage_count)
        counts[owner] += 1
        runtime.sleeping_enemy_king_damage_count = tuple(counts)
        env = runtime.environment
        snapshot = env._rich_snapshot_for(env.raw_observation)
        events = () if snapshot is None else snapshot.combat_events.events
        evidence = fireball_activation_evidence_v4(
            king,
            events,
            owner=owner,
            before_tick=before.tick,
            after_tick=after.tick,
            canonical_tick=env._canonical_tick,
        )
        if evidence is None:
            missing = list(runtime.king_activation_not_attributed_to_fireball)
            missing[owner] += 1
            runtime.king_activation_not_attributed_to_fireball = tuple(missing)
            continue
        counts = list(runtime.fireball_king_activations)
        totals = list(runtime.fireball_king_activation_penalty_total)
        counts[owner] += 1
        totals[owner] += coefficient
        runtime.fireball_king_activations = tuple(counts)
        runtime.fireball_king_activation_penalty_total = tuple(totals)
        runtime.fireball_king_activation_details.append({**evidence, "penalty": coefficient})
        adjusted_reward[owner] -= coefficient
