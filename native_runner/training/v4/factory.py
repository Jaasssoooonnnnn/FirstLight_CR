"""Production catalog/model/tensorizer construction for V4 training."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Mapping, Sequence

import torch

from ...card_specs import CardSpecCatalog, build_card_catalog
from ...card_logic import build_static_card_logic_catalog
from ...competitive_data import competitive_data_initialization
from ...battle_env import BattleEnvV1
from ...cr_native_env import DEFAULT_HOST, DEFAULT_PORT, ResidentNativeClashEnv
from ...contracts import AbilitySpecV1, CardSpecV1, EpisodeConfigV1, content_hash
from ...effect_catalog import NativeEffectCatalogV1, build_effect_catalog
from ...projectile_catalog import build_projectile_catalog
from ...resident_batch_vector import ResidentBatchCoordinatorV1, ResidentBatchNativeProxyV1
from ...semantic_subset import SemanticSupportedSubsetV1, build_semantic_supported_subset
from ..card_features import HERO_FORM_TO_BASE_CARD
from ..tracking import DeterministicPublicTracker
from .catalog import AbilityCatalogV1, CardCatalogV1, EntityArchetypeCatalogV1, normal_mode_card_specs
from .mechanics import EffectSemanticCatalogV1, MechanicProfileCatalogV1
from .model import UniversalCardPolicyV4
from .tensorizer import UniversalObservationTensorizerV4

if TYPE_CHECKING:
    from ...royaleapi_replay import PreparedCollectedReplay


@dataclass(frozen=True, slots=True)
class ProductionSemanticBundleV4:
    native_card_catalog: CardSpecCatalog
    semantic_subset_contract: SemanticSupportedSubsetV1
    card_specs: Mapping[int, CardSpecV1]
    ability_specs: Mapping[str, AbilitySpecV1]
    card_catalog: CardCatalogV1
    ability_catalog: AbilityCatalogV1
    entity_archetype_catalog: EntityArchetypeCatalogV1
    native_effect_catalog: NativeEffectCatalogV1
    effect_catalog: EffectSemanticCatalogV1
    mechanic_profile_catalog: MechanicProfileCatalogV1


@lru_cache(maxsize=1)
@competitive_data_initialization()
def production_semantic_bundle() -> ProductionSemanticBundleV4:
    native_cards = build_card_catalog()
    semantic_subset = build_semantic_supported_subset(native_cards)
    static_logic = build_static_card_logic_catalog(card_catalog=native_cards)
    projectiles = build_projectile_catalog(static_logic)
    native_effects = build_effect_catalog(static_logic)
    specs = normal_mode_card_specs(native_cards.by_id, require_frozen_baseline=True)
    cards = CardCatalogV1.from_card_spec_catalog(native_cards, static_logic=static_logic)
    abilities = AbilityCatalogV1.from_card_spec_catalog(native_cards)
    archetypes = EntityArchetypeCatalogV1.from_card_specs(
        specs, static_logic=static_logic, projectile_catalog=projectiles
    )
    effects = EffectSemanticCatalogV1.from_native_catalog(
        specs, static_logic=static_logic, native_effect_catalog=native_effects
    )
    mechanics = MechanicProfileCatalogV1.from_compiled()
    return ProductionSemanticBundleV4(
        native_card_catalog=native_cards,
        semantic_subset_contract=semantic_subset,
        card_specs=specs,
        ability_specs={item.ability_id: item for item in native_cards.abilities},
        card_catalog=cards,
        ability_catalog=abilities,
        entity_archetype_catalog=archetypes,
        native_effect_catalog=native_effects,
        effect_catalog=effects,
        mechanic_profile_catalog=mechanics,
    )


def build_production_model_v4(*, device: torch.device | str | None = None) -> UniversalCardPolicyV4:
    bundle = production_semantic_bundle()
    model = UniversalCardPolicyV4(
        bundle.card_catalog,
        ability_catalog=bundle.ability_catalog,
        entity_archetype_catalog=bundle.entity_archetype_catalog,
        effect_catalog=bundle.effect_catalog,
        mechanic_profile_catalog=bundle.mechanic_profile_catalog,
    )
    return model if device is None else model.to(device)


def deck_roles_from_form_availability(
    deck: Sequence[int], form_availability: Sequence[int]
) -> dict[int, tuple[bool, bool]]:
    """Map native hero/elite and evolution form bits to V4 deck roles."""

    if len(deck) != 8 or len(form_availability) != 8:
        raise ValueError("deck and form availability must each contain eight rows")
    return {
        int(card_id): (bool(int(mask) & 0x2), bool(int(mask) & 0x1))
        for card_id, mask in zip(deck, form_availability, strict=True)
    }


def _public_tracker_v4(
    bundle: ProductionSemanticBundleV4,
    decks: tuple[tuple[int, ...], tuple[int, ...]],
    roles: tuple[dict[int, tuple[bool, bool]], dict[int, tuple[bool, bool]]],
) -> DeterministicPublicTracker:
    card_costs = {
        card_id: float(bundle.card_specs[card_id].elixir_cost)
        for deck in decks
        for card_id in deck
        if bundle.card_specs[card_id].elixir_cost is not None
    }
    if set(card_costs) != set(decks[0]).union(decks[1]):
        raise ValueError("every tracked deck card needs an exact elixir cost")
    ability_costs: dict[int, dict[int, float]] = {0: {}, 1: {}}
    ability_cooldowns: dict[int, dict[int, int]] = {0: {}, 1: {}}
    ability_charges: dict[int, dict[int, int]] = {0: {}, 1: {}}
    evolution_cycles: dict[int, dict[int, int]] = {0: {}, 1: {}}
    for owner in (0, 1):
        for card_id in decks[owner]:
            elite, evolution = roles[owner][card_id]
            spec = bundle.card_specs[card_id]
            if elite:
                if len(spec.ability_ids) != 1:
                    raise ValueError("an enabled hero card needs exactly one ability contract")
                ability = bundle.ability_specs[spec.ability_ids[0]]
                if ability.elixir_cost is None or ability.charges is None:
                    raise ValueError("enabled ability cost/charges must be exact")
                cooldown_ms = int(ability.cooldown_ms or 0)
                if cooldown_ms == 0:
                    if ability.charges != 1:
                        raise ValueError("zero-cooldown ability must be single-charge")
                    # A consumed single charge is exhausted before this
                    # positive sentinel can be consulted by the tracker.
                    cooldown_ms = 50
                ability_costs[owner][card_id] = float(ability.elixir_cost)
                ability_cooldowns[owner][card_id] = cooldown_ms
                ability_charges[owner][card_id] = int(ability.charges)
            if evolution:
                cycle_required = None if spec.evolution is None else spec.evolution.cycle_required
                if cycle_required is None or cycle_required <= 0:
                    raise ValueError("an enabled evolution card needs an exact positive cycle")
                evolution_cycles[owner][card_id] = int(cycle_required)

    fallback_cost = {owner: next(iter(ability_costs[owner].values()), 0.0) for owner in (0, 1)}
    fallback_cooldown = {owner: next(iter(ability_cooldowns[owner].values()), 50) for owner in (0, 1)}
    return DeterministicPublicTracker(
        decks={0: decks[0], 1: decks[1]},
        card_costs=card_costs,
        ability_cost_by_owner=fallback_cost,
        ability_cooldown_ms_by_owner=fallback_cooldown,
        evolution_cycle_required={
            card_id: required for values in evolution_cycles.values() for card_id, required in values.items()
        },
        evolution_cycle_required_by_owner=evolution_cycles,
        ability_cost_by_owner_card=ability_costs,
        ability_cooldown_ms_by_owner_card=ability_cooldowns,
        ability_max_charges_by_owner_card=ability_charges,
        ability_form_to_base_card=HERO_FORM_TO_BASE_CARD,
    )


def build_episode_tensorizers_v4(
    episode: EpisodeConfigV1, *, horizontal_mirrors: tuple[bool, bool] = (False, False)
) -> tuple[UniversalObservationTensorizerV4, UniversalObservationTensorizerV4]:
    return tuple(
        build_episode_tensorizer_v4(episode, actor_owner=owner, horizontal_mirror=horizontal_mirrors[owner])
        for owner in (0, 1)
    )  # type: ignore[return-value]


def build_episode_tensorizer_v4(
    episode: EpisodeConfigV1, *, actor_owner: int, horizontal_mirror: bool = False
) -> UniversalObservationTensorizerV4:
    """Build one actor view while retaining both decks in its public tracker.

    Offline training builds both actor views and therefore requires exact role
    masks for both decks.  A live FAIR actor knows its own equipped forms but
    deliberately carries the catalog-supported opponent superset only inside
    the private public-event resolver.  Constructing the unused opponent view
    would incorrectly apply the one-deck equipment limit to that superset.
    """

    if actor_owner not in (0, 1):
        raise ValueError("actor_owner must be 0 or 1")
    bundle = production_semantic_bundle()
    decks = (episode.deck0, episode.deck1)
    forms = tuple(
        tuple(int(value) for value in episode.tags.get(f"deck{owner}_form_availability", (0,) * 8)) for owner in (0, 1)
    )
    roles = tuple(deck_roles_from_form_availability(decks[owner], forms[owner]) for owner in (0, 1))
    return UniversalObservationTensorizerV4(
        actor_owner=actor_owner,
        card_catalog=bundle.card_catalog,
        ability_catalog=bundle.ability_catalog,
        entity_archetype_catalog=bundle.entity_archetype_catalog,
        effect_catalog=bundle.effect_catalog,
        card_specs=bundle.card_specs,
        deck=decks[actor_owner],
        tracker=_public_tracker_v4(bundle, decks, roles),
        deck_roles=roles[actor_owner],
        horizontal_mirror=horizontal_mirror,
        reject_unknown_public_semantics=True,
    )


def build_collected_replay_environment_v4(
    prepared: PreparedCollectedReplay, *, native: object | None = None
) -> BattleEnvV1:
    """Build a BattleEnv whose time limit matches this source replay exactly."""

    source = prepared.replay.episode_config.environment
    execution_ruleset_id = content_hash(
        {
            "source_ruleset_id": prepared.replay.ruleset_id,
            "environment": {**source.to_dict(), "max_battle_ticks": prepared.replay.end_native_tick},
        }
    )
    bundle = production_semantic_bundle()
    return BattleEnvV1(
        native=native,  # type: ignore[arg-type]
        card_catalog=bundle.native_card_catalog,
        semantic_subset_contract=bundle.semantic_subset_contract,
        ruleset_id=execution_ruleset_id,
        warmup_ticks=source.warmup_ticks,
        max_battle_ticks=prepared.replay.end_native_tick,
        entity_limit=source.entity_limit,
        event_limit=source.event_limit,
        event_window_ticks=source.event_window_ticks,
        shaping_beta=source.shaping_beta,
        shaping_tau_ticks=source.shaping_tau_ticks,
        include_native_digest=source.include_native_digest,
    )


def build_resident_training_proxies_v4(
    env_ids: Sequence[int],
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    timeout: float = 30.0,
    capture_mode: str = "json",
) -> tuple[tuple[ResidentBatchNativeProxyV1, ...], ResidentBatchCoordinatorV1]:
    """Build one shared native coordinator for configurable resident slots."""

    normalized = tuple(int(env_id) for env_id in env_ids)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("resident training env IDs must be non-empty and unique")
    if capture_mode not in {"zlib-json", "compact", "json"}:
        raise ValueError("capture_mode must be zlib-json, compact, or json")
    natives = tuple(ResidentNativeClashEnv(env_id, host, port, timeout=timeout) for env_id in normalized)
    coordinator = ResidentBatchCoordinatorV1(
        natives,
        timeout=timeout,
        training_capture=capture_mode == "compact",
        compressed_json=capture_mode in {"compact", "zlib-json"},
        # BattleEnv still performs its typed fail-closed binding.  Avoid only
        # the duplicate transport-layer JSON validation in the hot loop.
        validate_json_capture=False,
    )
    return (tuple(ResidentBatchNativeProxyV1(native, coordinator) for native in natives), coordinator)


def build_resident_collected_replay_batch_v4(
    prepared_replays: Sequence[PreparedCollectedReplay],
    *,
    env_ids: Sequence[int],
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    timeout: float = 30.0,
    capture_mode: str = "json",
) -> tuple[tuple[BattleEnvV1, ...], ResidentBatchCoordinatorV1]:
    """Build replay environments sharing one configurable-slot engine path."""

    if not prepared_replays or len(prepared_replays) != len(env_ids):
        raise ValueError("prepared replays and env IDs must be non-empty and aligned")
    proxies, coordinator = build_resident_training_proxies_v4(
        env_ids, host=host, port=port, timeout=timeout, capture_mode=capture_mode
    )
    environments = tuple(
        build_collected_replay_environment_v4(prepared, native=proxy)
        for prepared, proxy in zip(prepared_replays, proxies, strict=True)
    )
    return environments, coordinator
