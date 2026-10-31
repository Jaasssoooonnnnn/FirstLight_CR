"""Reward-only attribution of first damage to a sleeping enemy King Tower.

Native combat facts are used only for reward; no oracle facts enter policy input.
Tower.active means alive, NOT awake. Sleeping requires full HP, no activation
status, and both Princess Towers still standing.

Two entry points share the detector:

* ``apply_fireball_king_activation_penalty_v4`` -- legacy, Fireball only,
  one-sided, skipped entirely at coefficient 0 (kept unchanged for old runs).
* ``apply_king_activation_reward_v4`` -- the first
  activation of a sleeping enemy King by ANY own damage source costs the
  activator ``-c`` and pays the victim ``+c`` (zero-sum, once per King).
  Detection and statistics run at ``c == 0`` as well.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

FIREBALL_CARD_ID = 28_000_000
# global id // 1_000_000: 26 troop, 27 building, 28 spell.
CARD_FAMILY_NAMES = {26: "troop", 27: "building", 28: "spell"}
SPELL_CARD_FAMILY = 28
KING_ACTIVATION_SCOPES = ("any", "spell", "fireball")
# CLI default for the generalized activation reward.
DEFAULT_KING_ACTIVATION_PENALTY = 0.1
DEFAULT_KING_ACTIVATION_SCOPE = "any"
# Tower coordinates are world units (1 tile = 1000); combat targets may come
# back as floats, so the generalized King match tolerates sub-unit rounding.
KING_POSITION_TOLERANCE = 1.0


def sleeping_king_damage_v4(before: Sequence[Any], after: Sequence[Any], owner: int):
    """Return the enemy king on a nonlethal first-damage transition."""
    enemy = 1 - owner
    kings = [t for t in before if t.owner == enemy and t.tower_kind == "king"]
    if len(kings) != 1:
        return None
    king = kings[0]
    status = getattr(king, "status", ())
    if "activated" in status or not math.isclose(king.hitpoints, king.max_hitpoints, abs_tol=1e-6):
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
    found = _first_king_hp_loss_event(
        king, events, before_tick=before_tick, after_tick=after_tick, canonical_tick=canonical_tick, tolerance=None
    )
    if found is not None:
        event, tick = found
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


def _position_matches(target: Sequence[float], king: Sequence[float], tolerance: float | None) -> bool:
    if tolerance is None:
        return tuple(target) == tuple(king)
    return len(target) == len(king) and all(abs(float(a) - float(b)) <= tolerance for a, b in zip(target, king))


def _first_king_hp_loss_event(
    king: Any,
    events: Sequence[Any],
    *,
    before_tick: int,
    after_tick: int,
    canonical_tick: Callable[[int], int],
    tolerance: float | None,
) -> tuple[Any, int] | None:
    """The first native damage event in the window that took this King off full HP."""
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
            and _position_matches(target.position, king.position, tolerance)
            and event.pre_hp is not None
            and event.post_hp is not None
            and math.isclose(event.pre_hp, king.max_hitpoints, abs_tol=1e-6)
            and 0 < event.post_hp < event.pre_hp
        ):
            continue
        return event, tick
    return None


def card_family_v4(card_id: int | None) -> str:
    if card_id is None:
        return "unknown"
    return CARD_FAMILY_NAMES.get(int(card_id) // 1_000_000, "other")


def king_activation_source_v4(
    king: Any,
    events: Sequence[Any],
    *,
    owner: int,
    before_tick: int,
    after_tick: int,
    canonical_tick: Callable[[int], int],
) -> dict[str, Any] | None:
    """Name the own card behind the first HP loss of a sleeping enemy King.

    Returns None when no matching damage event exists (combat hook absent or
    event outside the window).  When the event exists but its source cannot be
    tied to ``owner``, ``source_card_id`` is None.
    """
    found = _first_king_hp_loss_event(
        king,
        events,
        before_tick=before_tick,
        after_tick=after_tick,
        canonical_tick=canonical_tick,
        tolerance=KING_POSITION_TOLERANCE,
    )
    if found is None:
        return None
    event, tick = found
    card: int | None = None
    deployment = event.deployment_context
    if deployment is not None and deployment.owner == owner and deployment.played_card_global_id is not None:
        card = int(deployment.played_card_global_id)
    else:
        for source in (event.immediate_source, event.source, event.projectile):
            if source.present and source.validated and source.owner == owner and source.card_id is not None:
                card = int(source.card_id)
                break
    return {
        "tick": tick,
        "native_sequence": event.sequence,
        "source_card_id": card,
        "source_family": card_family_v4(card),
        "pre_hp": event.pre_hp,
        "post_hp": event.post_hp,
    }


def _scope_accepts(scope: str, card_id: int | None) -> bool:
    if scope == "any":
        return True
    if card_id is None:
        return False
    if scope == "spell":
        return int(card_id) // 1_000_000 == SPELL_CARD_FAMILY
    return int(card_id) == FIREBALL_CARD_ID


def validate_king_reward_parameters_v4(
    *, king_activation_penalty: float, king_activation_scope: str, king_shaping_mode: str, king_potential_weight: float
) -> None:
    """One validation boundary for trainer, collector and remote worker."""
    from ...battle_env import KING_SHAPING_MODES

    if not math.isfinite(king_activation_penalty) or king_activation_penalty < 0.0:
        raise ValueError("king activation penalty must be finite and nonnegative")
    if king_activation_scope not in KING_ACTIVATION_SCOPES:
        raise ValueError(f"king activation scope must be one of {KING_ACTIVATION_SCOPES}")
    if king_shaping_mode not in KING_SHAPING_MODES:
        raise ValueError(f"king shaping mode must be one of {KING_SHAPING_MODES}")
    if not math.isfinite(king_potential_weight) or king_potential_weight < 0.0:
        raise ValueError("king potential weight must be finite and nonnegative")


def configure_king_tower_shaping_v4(engine: Any, *, king_shaping_mode: str, king_potential_weight: float) -> None:
    """Apply the hash-neutral King shaping knobs to every BattleEnvV1 of an engine.

    Must run before the engine's environments are reset.  An engine without an
    ``environments`` attribute is accepted only with the default knobs, so a
    non-default mode can never be silently dropped by a different engine type.
    """
    from ...battle_env import DEFAULT_KING_POTENTIAL_WEIGHT

    environments = getattr(engine, "environments", None)
    if environments is None:
        if king_shaping_mode != "legacy" or float(king_potential_weight) != DEFAULT_KING_POTENTIAL_WEIGHT:
            raise RuntimeError("engine exposes no environments; cannot apply non-default King tower shaping")
        return
    for environment in environments:
        environment.configure_tower_shaping(
            king_shaping_mode=king_shaping_mode, king_potential_weight=king_potential_weight
        )


def apply_king_activation_reward_v4(
    runtime: Any,
    before_observations: Any,
    after_observations: Any,
    adjusted_reward: Any,
    coefficient: float = 0.0,
    scope: str = DEFAULT_KING_ACTIVATION_SCOPE,
) -> None:
    """Zero-sum, once-per-King reward for waking a sleeping enemy King.

    Both owners are examined regardless of which seats are learners, so the
    victim's ``+c`` exists even against a frozen opponent.  Statistics are
    recorded for every detected activation independent of ``coefficient`` and
    ``scope``; rewards move only when ``coefficient > 0`` and the source passes
    ``scope``.  Damage to a King can only come from its opponent, so the
    activator is known even when the combat hook cannot name the source card.
    """
    if not math.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("king activation penalty must be finite and nonnegative")
    if scope not in KING_ACTIVATION_SCOPES:
        raise ValueError(f"king activation scope must be one of {KING_ACTIVATION_SCOPES}")
    for owner in (0, 1):
        if owner in runtime.king_activation_seen_owners:
            continue
        before, after = before_observations[owner], after_observations[owner]
        king = sleeping_king_damage_v4(before.towers, after.towers, owner)
        if king is None:
            continue
        runtime.king_activation_seen_owners.add(owner)
        victim = 1 - owner
        env = runtime.environment
        snapshot = env._rich_snapshot_for(env.raw_observation)
        events = () if snapshot is None else snapshot.combat_events.events
        source = king_activation_source_v4(
            king,
            events,
            owner=owner,
            before_tick=int(before.tick),
            after_tick=int(after.tick),
            canonical_tick=env._canonical_tick,
        )
        card = None if source is None else source["source_card_id"]
        penalized = coefficient > 0.0 and _scope_accepts(scope, card)
        applied = float(coefficient) if penalized else 0.0
        tick = int(after.tick) if source is None else int(source["tick"])
        counts = list(runtime.king_activations)
        counts[owner] += 1
        runtime.king_activations = (counts[0], counts[1])
        ticks = list(runtime.king_activation_tick)
        ticks[owner] = tick
        runtime.king_activation_tick = (ticks[0], ticks[1])
        cards = list(runtime.king_activation_source_card_id)
        cards[owner] = card
        runtime.king_activation_source_card_id = (cards[0], cards[1])
        if card is None:
            missing = list(runtime.king_activation_source_unattributed)
            missing[owner] += 1
            runtime.king_activation_source_unattributed = (missing[0], missing[1])
        totals = list(runtime.king_activation_reward_total)
        totals[owner] -= applied
        totals[victim] += applied
        runtime.king_activation_reward_total = (totals[0], totals[1])
        runtime.king_activation_details.append(
            {
                "owner": owner,
                "victim_owner": victim,
                "king_entity_id": king.entity_id,
                "tick": tick,
                "observation_tick": int(after.tick),
                "native_sequence": None if source is None else source["native_sequence"],
                "source_card_id": card,
                "source_family": card_family_v4(card),
                "combat_event_found": source is not None,
                "scope": scope,
                "coefficient": float(coefficient),
                "penalized": penalized,
                "activator_reward": -applied,
            }
        )
        if penalized:
            adjusted_reward[owner] = float(adjusted_reward[owner]) - applied
            adjusted_reward[victim] = float(adjusted_reward[victim]) + applied


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
