"""Environment and episode construction for exact-deck V4 PPO matches."""

from __future__ import annotations

import math
from typing import Mapping

from functools import lru_cache

from ...arena import PLACEMENT_MASK_VERSION
from ...battle_env import BattleEnvV1
from ...contracts import EnvironmentConfigV1, EpisodeConfigV1, content_hash
from .factory import production_semantic_bundle
from .matchmaking import ScheduledMatchupV4


PPO_RULESET_SCHEMA_V4 = "v4-ppo-self-play-ruleset.v4"
PPO_DECISION_TICKS = 5


def ppo_environment_config_v4(
    *, gamma_per_decision: float = 0.9997, shaping_beta: float = 0.05, max_battle_ticks: int | None = None
) -> EnvironmentConfigV1:
    if not math.isfinite(gamma_per_decision) or not 0.0 < gamma_per_decision < 1.0:
        raise ValueError("PPO gamma must be finite and in (0,1)")
    if not math.isfinite(shaping_beta) or shaping_beta < 0.0:
        raise ValueError("PPO shaping beta must be finite and non-negative")
    tau_ticks = PPO_DECISION_TICKS / -math.log(gamma_per_decision)
    kwargs = {"shaping_beta": float(shaping_beta), "shaping_tau_ticks": float(tau_ticks)}
    if max_battle_ticks is not None:
        kwargs["max_battle_ticks"] = int(max_battle_ticks)
    return EnvironmentConfigV1(**kwargs)


@lru_cache(maxsize=16)
def ppo_ruleset_id_v4(environment: EnvironmentConfigV1) -> str:
    bundle = production_semantic_bundle()
    return content_hash(
        {
            "schema": PPO_RULESET_SCHEMA_V4,
            "environment": environment.to_dict(),
            "placement_mask_version": PLACEMENT_MASK_VERSION,
            "card_catalog_id": bundle.card_catalog.catalog_id,
            "ability_catalog_id": bundle.ability_catalog.catalog_id,
            "entity_archetype_catalog_id": (bundle.entity_archetype_catalog.catalog_id),
            "effect_catalog_id": bundle.effect_catalog.catalog_id,
            "mechanic_profile_catalog_id": (bundle.mechanic_profile_catalog.catalog_id),
        }
    )


def build_ppo_environment_v4(native: object, environment: EnvironmentConfigV1) -> BattleEnvV1:
    bundle = production_semantic_bundle()
    return BattleEnvV1(
        native=native,  # type: ignore[arg-type]
        card_catalog=bundle.native_card_catalog,
        semantic_subset_contract=bundle.semantic_subset_contract,
        ruleset_id=ppo_ruleset_id_v4(environment),
        warmup_ticks=environment.warmup_ticks,
        max_battle_ticks=environment.max_battle_ticks,
        entity_limit=environment.entity_limit,
        event_limit=environment.event_limit,
        event_window_ticks=environment.event_window_ticks,
        shaping_beta=environment.shaping_beta,
        shaping_tau_ticks=environment.shaping_tau_ticks,
        include_native_digest=environment.include_native_digest,
    )


def episode_for_matchup_v4(matchup: ScheduledMatchupV4, environment: EnvironmentConfigV1) -> EpisodeConfigV1:
    layout0, layout1 = matchup.runtime_layouts()
    return EpisodeConfigV1(
        ruleset_id=ppo_ruleset_id_v4(environment),
        deck0=layout0.card_ids,
        deck1=layout1.card_ids,
        seed=matchup.seed,
        decision_hz=4.0,
        event_driven_decisions=False,
        environment=environment,
        tags={
            "level_cap": matchup.battle_level,
            "minimum_card_level": matchup.battle_level,
            "king_tower_level": matchup.battle_level,
            "deck0_form_availability": layout0.form_availability,
            "deck1_form_availability": layout1.form_availability,
            "tower_troop0_id": layout0.tower_troop_id,
            "tower_troop1_id": layout1.tower_troop_id,
            "owner0_name": f"PPO-{matchup.policy_matchup}-0",
            "owner1_name": f"PPO-{matchup.policy_matchup}-1",
            "end_tick": environment.max_battle_ticks,
            "ppo_match_sequence": matchup.sequence,
            "ppo_match_category": matchup.category,
            "ppo_deck0_id": matchup.deck0.deck_id,
            "ppo_deck1_id": matchup.deck1.deck_id,
        },
    )


def validate_reward_parameters_v4(parameters: Mapping[str, object]) -> None:
    """One validation boundary shared by the trainer, collector, and worker."""

    def require_finite(names: tuple[str, ...], message: str) -> None:
        if any(not math.isfinite(parameters[name]) or parameters[name] < 0.0 for name in names):
            raise ValueError(message)

    require_finite(("fireball_king_activation_penalty",), "fireball activation penalty must be finite and nonnegative")
    require_finite(
        (
            "mutual_elixir_overflow_penalty",
            "mutual_elixir_overflow_grace",
            "unilateral_elixir_overflow_penalty",
            "elixir_overflow_step_penalty_cap",
        ),
        "elixir overflow reward parameters must be finite",
    )
    if parameters["elixir_overflow_step_penalty_cap"] <= 0.0:
        raise ValueError("elixir overflow step penalty cap must be positive")
    require_finite(
        ("hog_deploy_reward", "hog_deploy_reward_episode_cap"),
        "hog deployment reward parameters must be finite and nonnegative",
    )
    if parameters["hog_deploy_opening_reward"] is not None:
        require_finite(("hog_deploy_opening_reward",), "hog opening reward must be finite and nonnegative")
    require_finite(
        (
            "first_hog_timing_reward_max",
            "first_hog_timing_start_seconds",
            "first_hog_timing_deadline_seconds",
            "first_hog_missed_deadline_penalty",
            "first_hog_deferral_penalty",
            "first_hog_deferral_penalty_episode_cap",
        ),
        "first-Hog reward parameters must be finite and nonnegative",
    )
    if parameters["first_hog_timing_deadline_seconds"] <= parameters["first_hog_timing_start_seconds"]:
        raise ValueError("first-Hog deadline must be after its reward start")
    if parameters["hog_deploy_opening_reward"] is not None and any(
        parameters[name] > 0.0
        for name in ("first_hog_timing_reward_max", "first_hog_missed_deadline_penalty", "first_hog_deferral_penalty")
    ):
        raise ValueError("legacy flat Hog window cannot mix with first-Hog shaping")
