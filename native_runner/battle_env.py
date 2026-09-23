"""Canonical two-player environment on top of the exact native v15 engine.

``NativeClashEnv`` is intentionally a thin engine binding.  This module is the
model-facing protocol: versioned observations/actions, fair-state filtering,
dynamic masks, event history, pending execution timestamps, terminal rewards,
and a PettingZoo-style parallel step signature.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math
import time
import uuid
from typing import Any

from .arena import (
    OccupiedFootprintV1,
    building_footprints_overlap,
    card_placement_legal_world,
    card_placement_mask,
    cell_to_world,
    native_building_footprint,
    uses_entity_deployment_center,
)
from .building_placement_profiles import building_placement_profile
from .card_specs import CardSpecCatalog, build_card_catalog, sha256_file
from .attestation import RunnerAttestationError, verify_runner_attestation
from .contracts import (
    ActionKind,
    ActionMaskV1,
    ActionV1,
    AbilityPhase,
    AbilityRuntimeStateV1,
    CausalGroupKind,
    CausalGroupRefV1,
    COMBAT_EVENT_FIELDS,
    COMPETITIVE_TOWER_TROOP_IDS,
    CombatEventKind,
    CombatEventV1,
    DaggerDuchessRuntimeStateV1,
    EntityStateV1,
    EVOLUTION_RUNTIME_STATE_FIELDS,
    EnvironmentConfigV1,
    EpisodeConfigV1,
    EventV1,
    EvolutionPhase,
    EvolutionRuntimeStateV1,
    ObservationTier,
    ObservationV1,
    OpponentBeliefV1,
    PendingActionV1,
    PlayerStateV1,
    ROYAL_CHEF_TOWER_TROOP_ID,
    RoyalChefRuntimeStateV1,
    RunnerAttestationV1,
    SemanticEvidenceLevel,
    SemanticProvenanceV1,
    TargetKind,
    TerminalV1,
    TimeStateV1,
    TOWER_RUNTIME_SEMANTIC_FIELDS,
    TowerStateV1,
    content_hash,
    frozen_mapping,
)
from .cr_native_env import (
    FIRST_PLAYABLE_TICK,
    AbilityAction,
    HandAction,
    NativeClashEnv,
    RunnerError,
)
from .match_factory import (
    DEFAULT_TEMPLATE_PATH,
    NATIVE_GAMEPLAY_END_TICK,
    NATIVE_MATCH_END_TICK,
    PRINCESS_TOWER_TROOP_ID,
    MatchConfig,
)
from .normal_form_evidence import NORMAL_MODE_HERO_FORM_TO_BASE_CARD
from .phase_runtime import PhaseHookEvent
from .rich_telemetry_adapter import (
    EntityKey,
    RichObjectTelemetry,
    RichDaggerDuchessRuntimeTelemetry,
    RichCombatDeploymentContext,
    RichCombatEntityFact,
    RichCombatEventTelemetry,
    RichRemainingRuntimeEvent,
    RichVisibilityRuntimeEvent,
    RichPlayerRuntimeTelemetry,
    RichRoyalChefRuntimeTelemetry,
    RichTelemetryMergeError,
    RichTelemetrySnapshot,
    RuntimeEffectCatalog,
    bind_rich_telemetry,
    load_default_effect_catalog,
    normalized_ability_cooldown_ms,
    project_player_runtime,
    project_runtime,
)
from .ruleset import RulesetManifestV1, build_ruleset_manifest
from .semantic_subset import (
    SEMANTIC_BASELINE_DECK,
    SemanticSubsetError,
    SemanticSupportedSubsetV1,
    build_semantic_supported_subset,
)
from .timeline import GameModeTimelineV1, load_game_mode_timeline


TICKS_PER_SECOND = 20
TICK_MS = 50
_ACTION_MASK_CACHE_LIMIT = 64
STANDARD_MATCH_TICKS = NATIVE_MATCH_END_TICK
STANDARD_GAME_MODE = 72_000_006
STANDARD_ARENA = 54_000_001
CLASSIC_CHALLENGE_GAME_MODE = 72_000_009
CLASSIC_CHALLENGE_ARENA = 54_000_036
# Keep the producer wider than V4's 160-child memory so semantic priority,
# child overflow, and full spatial aggregates are computed after observation.
ENTITY_LIMIT = 256
EVENT_LIMIT = 256
EVENT_WINDOW_TICKS = 45 * TICKS_PER_SECOND
# Default battle_timelines.csv: duration seconds and milliseconds required to
# fill the complete ten-elixir bar.  These are not a cosmetic multiplier: they
# are the exact production schedule used for affordability waits and beliefs.
ELIXIR_SEGMENTS: tuple[tuple[int, int | None, int], ...] = (
    (0, 2400, 28_000),
    (2400, 4800, 14_000),
    (4800, 6000, 9_300),
)

# entity_id, owner, kind, x, y
TOWER_LAYOUT: tuple[tuple[int, int, str, int, int], ...] = (
    (0, 0, "king", 9000, 3000),
    (1, 0, "princess_left", 3500, 6500),
    (2, 0, "princess_right", 14500, 6500),
    (3, 1, "king", 9000, 29000),
    (4, 1, "princess_left", 3500, 25500),
    (5, 1, "princess_right", 14500, 25500),
)
TOWER_BY_POSITION = {(x, y): (entity_id, owner, kind) for entity_id, owner, kind, x, y in TOWER_LAYOUT}
MIRROR_CARD_ID = 28_000_006
ABILITY_ATTESTATION_MIN_GRACE_TICKS = 40
ABILITY_ATTESTATION_CAST_MARGIN_TICKS = 20
_ABILITY_PENDING_STATUSES = frozenset({"queued", "awaiting_execution"})


def _ability_activation_evidence(before: tuple[int, int, int], after: tuple[int, int, int]) -> list[str]:
    """Use identical native controller edges for receipts and public casts."""
    old_button, old_cooldown, old_charges = before
    button, cooldown, charges = after
    return [
        name
        for name, occurred in (
            ("cooldown_started", cooldown > old_cooldown),
            ("charge_consumed", old_charges >= 0 and 0 <= charges < old_charges),
            ("ready_to_casting_state", old_button in {2, 4} and button in {7, 10}),
        )
        if occurred
    ]


def _fair_snapshot_damage_contract(
    *, tick: int, target_entity: int, target_card_id: int | None, position: tuple[float, float], amount: float
) -> tuple[CombatEventV1, SemanticProvenanceV1]:
    """Project a public HP delta without inventing its attacker."""

    evidence = {name: SemanticEvidenceLevel.UNKNOWN for name in COMBAT_EVENT_FIELDS}
    evidence.update(
        {
            "kind": SemanticEvidenceLevel.NATIVE_DERIVED,
            "target_entity": SemanticEvidenceLevel.NATIVE_DERIVED,
            "position": SemanticEvidenceLevel.NATIVE_DERIVED,
            "amount": SemanticEvidenceLevel.NATIVE_DERIVED,
        }
    )
    source_fields: dict[str, tuple[str, ...]] = {
        "kind": ("successive public hitpoint snapshots",),
        "target_entity": ("canonical public entity identity",),
        "position": ("public entity position",),
        "amount": ("successive public hitpoint snapshots",),
    }
    if target_card_id is not None:
        evidence["target_card_id"] = SemanticEvidenceLevel.NATIVE_DERIVED
        source_fields["target_card_id"] = ("public entity cardId",)
    combat = CombatEventV1(
        kind=CombatEventKind.DAMAGE,
        target_entity=target_entity,
        target_card_id=target_card_id,
        position=position,
        amount=amount,
        provenance=SemanticProvenanceV1(
            field_evidence=evidence,
            source_fields=source_fields,
            observed_tick=tick,
            notes=("snapshot delta identifies the victim and amount only; source intentionally remains unknown",),
        ),
    )
    return combat, SemanticProvenanceV1(
        field_evidence={"combat": SemanticEvidenceLevel.NATIVE_DERIVED},
        source_fields={"combat": ("successive public hitpoint snapshots",)},
        observed_tick=tick,
        notes=("FAIR-safe snapshot damage projection",),
    )


class BattleEnvError(RuntimeError):
    """Raised when a model-facing request violates the environment contract."""


@dataclass(slots=True)
class _PendingRecord:
    action: ActionV1
    card_id: int
    effective_card_id: int
    native_effective_card_id: int
    requested_native_tick: int
    expected_native_tick: int
    hand_slot: int
    card_parameter: int
    deck_slot: int
    cost: int
    form_code: int
    native_form_code: int
    form_name: str
    requested_at_ms: int
    generated_at_ms: int
    inference_end_ms: int
    status: str = "queued"
    server_apply_ms: int | None = None


@dataclass(slots=True)
class _PendingAbilityRecord:
    action: ActionV1
    ability_id: str
    source_card_id: int
    source_entity_key: EntityKey
    controller_slot: int
    action_data_global_id: int
    before_button_state: int
    before_remaining_cooldown_ms: int
    before_remaining_charges_raw: int
    before_configured_cooldown_ms: int
    before_max_charges: int
    requested_native_tick: int
    expected_native_tick: int
    attestation_deadline_native_tick: int
    cost_raw: int
    requested_at_ms: int
    generated_at_ms: int
    inference_end_ms: int
    status: str = "queued"
    server_apply_ms: int | None = None


@dataclass(frozen=True, slots=True)
class _RichObservationContext:
    snapshot: RichTelemetrySnapshot
    canonical_ids: Mapping[EntityKey, int]
    visible_keys: frozenset[EntityKey]

    def item_for_raw(self, item: Mapping[str, Any]) -> RichObjectTelemetry:
        try:
            slot = int(item["slot"])
        except (KeyError, TypeError, ValueError) as error:
            raise BattleEnvError("ordinary native object has no valid slot") from error
        rich_item = self.snapshot.at_slot(slot)
        if rich_item is None:
            raise BattleEnvError(f"ordinary object slot {slot} matched a rich null slot")
        return rich_item


@dataclass(frozen=True, slots=True)
class _CombatEvolutionEntity:
    deployment: RichCombatDeploymentContext
    observed_tick: int


@dataclass(frozen=True, slots=True)
class _NativePublicCardPlay:
    sequence: int
    state_epoch: int
    tick: int
    deployment: RichCombatDeploymentContext
    position: tuple[float, float] | None


@dataclass(frozen=True, slots=True)
class ReplayOperationV1:
    """One deterministic operation retained for recorded UI replay."""

    actions: tuple[ActionV1, ...]
    advance_ticks: int
    start_native_tick: int
    end_native_tick: int
    requested_advance_ticks: int | None = None


def _player(raw: Mapping[str, Any], owner: int) -> Mapping[str, Any]:
    try:
        return next(item for item in raw["players"] if int(item["owner"]) == owner)
    except (KeyError, StopIteration) as error:
        raise BattleEnvError(f"native observation has no owner {owner}") from error


def _diagnostic_hand_snapshot(player: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        {
            "hand_index": int(item["handIndex"]),
            "card_id": int(item["cardId"]),
            "card_parameter": int(item.get("cardParameter", -1)),
            "deck_slot": int(item.get("deckSlot", -1)),
            "cost": (None if item.get("cost") is None else int(item["cost"])),
        }
        for item in sorted(player.get("hand", ()), key=lambda value: int(value["handIndex"]))
    )


def _entity_id(item: Mapping[str, Any]) -> int:
    native_object_id = item.get("nativeObjectId")
    if isinstance(native_object_id, int) and not isinstance(native_object_id, bool) and native_object_id > 0:
        return 1_000_000_000 + native_object_id
    # Backward compatibility for persisted pre-nativeObjectId fixtures only.
    object_index = int(item.get("objectIndex", -1))
    secondary = int(item.get("secondaryIndex", -1))
    if object_index >= 0 and secondary >= 0:
        return 10_000 + object_index * 1_000 + secondary
    return 1_000_000 + int(item.get("slot", 0))


def _elixir_raw_per_tick(native_tick: int) -> float:
    tick = max(0, int(native_tick))
    for start, end, full_bar_ms in ELIXIR_SEGMENTS:
        if tick >= start and (end is None or tick < end):
            return 100_000.0 * TICK_MS / full_bar_ms
    # Finalization normally ends the match at this point.  Retain the final
    # declared rate if the engine needs a few tiebreak ticks.
    return 100_000.0 * TICK_MS / ELIXIR_SEGMENTS[-1][2]


def _elixir_generated_raw(start_tick: int, end_tick: int) -> float:
    cursor = max(0, int(start_tick))
    stop = max(cursor, int(end_tick))
    generated = 0.0
    while cursor < stop:
        boundary = stop
        for start, end, _ in ELIXIR_SEGMENTS:
            if cursor >= start and (end is None or cursor < end):
                boundary = min(stop, end if end is not None else stop)
                break
        generated += (boundary - cursor) * _elixir_raw_per_tick(cursor)
        cursor = boundary
    return generated


class BattleEnvV1:
    """Parallel, two-sided, canonical battle environment.

    ``reset`` and ``step`` follow the PettingZoo parallel convention without
    importing PettingZoo.  Observations are immutable :class:`ObservationV1`
    contracts.  The native manager remains the sole source of game dynamics.
    """

    possible_agents = (0, 1)

    def __init__(
        self,
        native: NativeClashEnv | None = None,
        *,
        card_catalog: CardSpecCatalog | None = None,
        semantic_subset_contract: SemanticSupportedSubsetV1 | None = None,
        effect_catalog: RuntimeEffectCatalog | None = None,
        ruleset_manifest: RulesetManifestV1 | None = None,
        ruleset_id: str | None = None,
        mode_timeline: GameModeTimelineV1 | None = None,
        warmup_ticks: int = FIRST_PLAYABLE_TICK,
        max_battle_ticks: int = STANDARD_MATCH_TICKS,
        entity_limit: int = ENTITY_LIMIT,
        event_limit: int = EVENT_LIMIT,
        event_window_ticks: int = EVENT_WINDOW_TICKS,
        shaping_beta: float = 0.0,
        shaping_tau_ticks: float = 1200.0,
        include_native_digest: bool = True,
    ) -> None:
        environment_config = EnvironmentConfigV1(
            warmup_ticks=warmup_ticks,
            max_battle_ticks=max_battle_ticks,
            entity_limit=entity_limit,
            event_limit=event_limit,
            event_window_ticks=event_window_ticks,
            shaping_beta=shaping_beta,
            shaping_tau_ticks=shaping_tau_ticks,
            include_native_digest=include_native_digest,
        )
        self.native = native or NativeClashEnv()
        verified_timeline = load_game_mode_timeline(STANDARD_GAME_MODE)
        self.mode_timeline = mode_timeline or verified_timeline
        if self.mode_timeline.game_mode_id != STANDARD_GAME_MODE:
            raise ValueError(
                "BattleEnvV1 currently guarantees canonical masks/towers only "
                f"for standard game mode {STANDARD_GAME_MODE}"
            )
        if (
            self.mode_timeline.game_modes_sha256 != verified_timeline.game_modes_sha256
            or self.mode_timeline.timeline.source_sha256 != verified_timeline.timeline.source_sha256
        ):
            raise ValueError(
                "an injected mode timeline must match the content-addressed standard timeline in this workspace"
            )
        self.card_catalog = card_catalog or build_card_catalog()
        # Catalog digests are immutable for the lifetime of an environment.
        # Keep them out of the 10 Hz observation hot path.
        self._card_specs_hash = self.card_catalog.specs_hash
        self.effect_catalog = effect_catalog or load_default_effect_catalog()
        self.card_specs = {item.card_id: item for item in self.card_catalog.specs}
        self.ability_specs = {item.ability_id: item for item in self.card_catalog.abilities}
        self.semantic_subset_contract = semantic_subset_contract or build_semantic_supported_subset(self.card_catalog)
        try:
            self.semantic_subset_contract.verify_catalog(self.card_catalog)
        except SemanticSubsetError as error:
            raise ValueError(f"invalid semantic supported-subset contract: {error}") from error
        self._semantic_subset_binding = self.semantic_subset_contract.binding()
        self._policy_form_masks = {
            int(item.card_id): int(item.allowed_form_availability)
            for item in self.semantic_subset_contract.allowed_cards
        }
        self._explicit_hero_form_base_ids = frozenset(
            int(card_id) for card_id in NORMAL_MODE_HERO_FORM_TO_BASE_CARD.values()
        )
        if ruleset_manifest is not None and ruleset_id is not None:
            if ruleset_manifest.ruleset_id != ruleset_id:
                raise ValueError("ruleset_manifest and ruleset_id disagree")
        # Production episodes never receive a placeholder patch identity.
        # Tests and remote workers can inject a previously verified ID to avoid
        # re-hashing the local asset pack for every environment instance.
        self.ruleset_manifest = ruleset_manifest
        if ruleset_manifest is None and ruleset_id is None:
            self.ruleset_manifest = build_ruleset_manifest(
                catalog=self.card_catalog, environment_config=environment_config
            )
        self.ruleset_id = self.ruleset_manifest.ruleset_id if self.ruleset_manifest is not None else str(ruleset_id)
        if self.ruleset_manifest is not None:
            manifest_environment = EnvironmentConfigV1.from_mapping(self.ruleset_manifest.config.get("environment"))
            if manifest_environment != environment_config:
                raise ValueError("BattleEnv environment knobs do not match the ruleset manifest")
            manifest_timeline = self.ruleset_manifest.config.get("effective_timeline", {})
            if (
                manifest_timeline.get("game_modes_sha256") != self.mode_timeline.game_modes_sha256
                or manifest_timeline.get("timeline_sha256") != self.mode_timeline.timeline.source_sha256
            ):
                raise ValueError("BattleEnv timeline does not match the ruleset manifest")
            subset_binding = self.ruleset_manifest.config.get("semantic_supported_subset")
            if not isinstance(subset_binding, Mapping):
                raise ValueError("ruleset manifest does not bind the semantic supported subset")
            if dict(subset_binding) != self._semantic_subset_binding:
                raise ValueError("BattleEnv semantic supported subset does not match the ruleset manifest")
        self.environment_config = environment_config
        self.warmup_ticks = warmup_ticks
        self.max_battle_ticks = max_battle_ticks
        self.entity_limit = entity_limit
        self.event_limit = event_limit
        self.event_window_ticks = event_window_ticks
        self.shaping_beta = float(shaping_beta)
        self.shaping_tau_ticks = float(shaping_tau_ticks)
        self.include_native_digest = include_native_digest

        self.agents = list(self.possible_agents)
        self.episode_config: EpisodeConfigV1 | None = None
        self.match_config: MatchConfig | None = None
        self.episode_id: str | None = None
        self.runner_attestation: RunnerAttestationV1 | None = None
        self._episode_start_native_tick = 0
        self._raw: dict[str, Any] | None = None
        self._rich_cache_identity: tuple[int, int, int | None] | None = None
        self._rich_cache: RichTelemetrySnapshot | None = None
        self._fair_context_cache_identity: tuple[int, int, int | None] | None = None
        self._fair_context_cache: _RichObservationContext | None = None
        self._fair_scene_cache_identity: tuple[int, int, int | None] | None = None
        self._fair_scene_cache: (
            tuple[_RichObservationContext | None, tuple[TowerStateV1, ...], tuple[EntityStateV1, ...], int] | None
        ) = None
        self._initial_tower_hp: dict[int, float] = {}
        self._tower_object_keys: dict[int, int] = {}
        self._activated_king_tower_ids: set[int] = set()
        self._events: deque[EventV1] = deque(maxlen=event_limit)
        self._event_archive: dict[str, dict[str, int]] = {}
        self._pending: list[_PendingRecord] = []
        self._pending_abilities: list[_PendingAbilityRecord] = []
        self._revealed: dict[int, set[int]] = {0: set(), 1: set()}
        self._belief_elixir: dict[int, tuple[float, float]] = {0: (6.0, 6.0), 1: (6.0, 6.0)}
        self._entity_birth: dict[int, int] = {}
        self._previous_entities: dict[int, dict[str, Any]] = {}
        self._previous_rich_objects: dict[int, RichObjectTelemetry] = {}
        self._previous_player_runtime: dict[int, RichPlayerRuntimeTelemetry] = {}
        self._combat_event_identity: tuple[int, int] | None = None
        self._combat_next_sequence: int | None = None
        self._public_card_play_event_identity: tuple[int, int] | None = None
        self._public_card_play_next_sequence: int | None = None
        self._combat_native_entity_ids: dict[int, int] = {}
        self._combat_evolution_entities: dict[int, _CombatEvolutionEntity] = {}
        self._causal_group_by_entity: dict[int, CausalGroupRefV1] = {}
        self._combat_event_kind_by_sequence: dict[int, str] = {}
        self._phase_event_identity: tuple[int, int] | None = None
        self._phase_next_sequence: int | None = None
        self._phase_events_by_entity: dict[EntityKey, list[PhaseHookEvent]] = {}
        self._visibility_event_identity: tuple[int, int] | None = None
        self._visibility_next_sequence: int | None = None
        self._remaining_event_identity: tuple[int, int] | None = None
        self._remaining_next_sequence: int | None = None
        self._remaining_rejected_count_baseline: int | None = None
        self._allow_initial_remaining_runtime_rejections = False
        self._synchronize_native_render_steps = False
        self._entity_velocity: dict[int, tuple[float, float]] = {}
        self._previous_towers: dict[int, TowerStateV1] = {}
        self._last_potential = {0: 0.0, 1: 0.0}
        self._last_rewards = {0: 0.0, 1: 0.0}
        self._terminated = False
        self._truncated = False
        self._trace: list[ReplayOperationV1] = []
        self._decision_deadline_native: dict[int, int] = {0: 0, 1: 0}
        self._last_advance_event = False
        self._placement_mask_cache: dict[tuple[Any, ...], Mapping[str, Any]] = {}
        self._runtime_placement_cache: dict[tuple[Any, ...], tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
        self._ordinary_tower_cache: dict[tuple[int, int, int | None], tuple[TowerStateV1, ...]] = {}
        self._action_mask_cache: dict[tuple[Any, ...], ActionMaskV1] = {}
        self._action_player_runtime_cache: dict[tuple[int, int, int | None, int], Any] = {}
        self._public_metadata_static: Mapping[str, Any] | None = None

    @property
    def trace(self) -> tuple[ReplayOperationV1, ...]:
        return tuple(self._trace)

    def deployment_timing_events(self, owner: int) -> list[dict[str, Any]]:
        """Diagnostic native timestamps; never projected into policy observations."""
        snapshot = self._rich_snapshot_for(self.raw_observation)
        if snapshot is None:
            return []
        result = []
        for event in snapshot.combat_events.events:
            deployment = event.deployment_context
            if deployment is None or deployment.owner != owner:
                continue
            if event.kind not in {"card_play", "spawn", "projectile_spawn"}:
                continue
            if event.kind == "spawn" and event.target.object_kind == 4:
                continue
            subject = event.projectile if event.kind == "projectile_spawn" else event.target
            result.append(
                {
                    "nativeEventId": f"{event.state_epoch}:{event.sequence}",
                    "deploymentId": f"{event.state_epoch}:{deployment.deployment_sequence}",
                    "kind": event.kind,
                    "tick": self._canonical_tick(event.tick),
                    "cardId": deployment.played_card_global_id,
                    "nativeObjectId": subject.native_object_id,
                    "objectKind": subject.object_kind,
                }
            )
        return result

    @property
    def raw_observation(self) -> Mapping[str, Any]:
        if self._raw is None:
            raise BattleEnvError("environment has not been reset")
        return self._raw

    @property
    def awaiting_ability_action_ids(self) -> tuple[str, ...]:
        """Actor-private ability commands still awaiting native apply evidence."""

        return tuple(
            sorted(
                item.action.action_id for item in self._pending_abilities if item.status in _ABILITY_PENDING_STATUSES
            )
        )

    @property
    def next_ability_attestation_deadline_native_tick(self) -> int | None:
        deadlines = tuple(
            item.attestation_deadline_native_tick
            for item in self._pending_abilities
            if item.status in _ABILITY_PENDING_STATUSES
        )
        return min(deadlines) if deadlines else None

    def close(self) -> None:
        """Standard-env compatibility; the shared native service stays alive."""

        self.agents = []

    def _verify_runner_attestation(self) -> RunnerAttestationV1 | None:
        """Bind the live endpoint to the manifest before native mutation."""

        if self.ruleset_manifest is None:
            # Explicit IDs remain useful for isolated tests and imported remote
            # fixtures, but they are not a claim about a live native process.
            self.runner_attestation = None
            return None
        if not hasattr(self.native, "attest"):
            raise BattleEnvError("native endpoint has no RunnerAttestationV1 API; refusing a manifested episode")
        try:
            reported = self.native.attest()  # type: ignore[attr-defined]
            attestation = verify_runner_attestation(reported, self.ruleset_manifest)
        except (RunnerAttestationError, RunnerError, TypeError, ValueError) as error:
            raise BattleEnvError(f"native runner attestation failed: {error}") from error
        self.runner_attestation = attestation
        return attestation

    def prepare_runner_attestation(self) -> RunnerAttestationV1 | None:
        """Precompute the immutable live-runner identity before an episode."""

        return self._verify_runner_attestation()

    def _episode_from_match(self, match: MatchConfig, render_mode: str) -> EpisodeConfigV1:
        return EpisodeConfigV1(
            ruleset_id=self.ruleset_id,
            deck0=match.deck0,
            deck1=match.deck1,
            seed=match.seed,
            game_mode=match.game_mode,
            arena=match.arena,
            render_mode=render_mode,
            environment=self.environment_config,
            tags={
                "location": match.location,
                "level_cap": match.level_cap,
                "minimum_card_level": match.minimum_card_level,
                "deck0_form_availability": match.deck0_form_availability,
                "deck1_form_availability": match.deck1_form_availability,
                "tower_troop0_id": match.tower_troop0_id,
                "tower_troop1_id": match.tower_troop1_id,
                "owner0_name": match.owner0_name,
                "owner1_name": match.owner1_name,
                "end_tick": match.end_tick,
                **({"king_tower_level": match.king_tower_level} if match.king_tower_level is not None else {}),
            },
        )

    def reset(
        self,
        config: EpisodeConfigV1 | Mapping[str, Any] | None = None,
        *,
        match_config: MatchConfig | None = None,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[dict[int, ObservationV1], dict[str, Any]]:
        """Create a fresh empty-command match and return both fair views."""

        if self.ruleset_manifest is not None:
            from .paths import WORKSPACE_ROOT
            from .ruleset import _relative

            relative = _relative(DEFAULT_TEMPLATE_PATH, WORKSPACE_ROOT)
            expected = self.ruleset_manifest.files.get("data", {}).get(relative)
            if expected is None:
                raise BattleEnvError("ruleset manifest does not bind the native match schema template")
            actual = sha256_file(DEFAULT_TEMPLATE_PATH)
            if actual != expected:
                raise BattleEnvError("native match schema template changed after ruleset construction")

        opts = dict(options or {})
        if config is not None and not isinstance(config, EpisodeConfigV1):
            config = EpisodeConfigV1.from_mapping(config)
        if config is not None and match_config is not None:
            raise ValueError("pass config or match_config, not both")

        render_mode = str(opts.get("render_mode", config.render_mode if config else "headless"))
        if config is not None and config.observation_tier != ObservationTier.FAIR:
            raise BattleEnvError("BattleEnvV1 only supports semantic-fair episodes")
        pause_native_render = opts.get("pause_native_render", False)
        suppress_native_render = opts.get("suppress_native_render", False)
        defer_headless_warmup = opts.get("defer_headless_warmup", False)
        headless_initial_step_ticks = opts.get("headless_initial_step_ticks", 0)
        allow_initial_remaining_runtime_rejections = opts.get("allow_initial_remaining_runtime_rejections", False)
        synchronize_native_render_steps = opts.get("synchronize_native_render_steps", False)
        allow_verified_standard_layout_alias = opts.get("allow_verified_standard_layout_alias", False)
        native_render_speed = opts.get("native_render_speed")
        for name, value in (
            ("pause_native_render", pause_native_render),
            ("suppress_native_render", suppress_native_render),
            ("defer_headless_warmup", defer_headless_warmup),
            ("allow_initial_remaining_runtime_rejections", allow_initial_remaining_runtime_rejections),
            ("synchronize_native_render_steps", synchronize_native_render_steps),
            ("allow_verified_standard_layout_alias", allow_verified_standard_layout_alias),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be boolean")
        if render_mode != "native-render" and (
            pause_native_render
            or suppress_native_render
            or synchronize_native_render_steps
            or native_render_speed is not None
        ):
            raise ValueError("native render control options require native-render mode")
        if render_mode != "headless" and defer_headless_warmup:
            raise ValueError("deferred headless warmup requires headless mode")
        if (
            isinstance(headless_initial_step_ticks, bool)
            or not isinstance(headless_initial_step_ticks, int)
            or not 0 <= headless_initial_step_ticks <= 1
        ):
            raise ValueError("headless_initial_step_ticks must be 0 or 1")
        if render_mode != "headless" and headless_initial_step_ticks:
            raise ValueError("headless initial stepping requires headless mode")
        if headless_initial_step_ticks and not defer_headless_warmup:
            raise ValueError("headless initial stepping requires deferred headless warmup")
        if native_render_speed is not None and (
            isinstance(native_render_speed, bool) or not isinstance(native_render_speed, (int, float))
        ):
            raise TypeError("native_render_speed must be numeric")
        self._allow_initial_remaining_runtime_rejections = allow_initial_remaining_runtime_rejections
        self._synchronize_native_render_steps = synchronize_native_render_steps
        if config is not None:
            if config.ruleset_id != self.ruleset_id:
                raise BattleEnvError(f"episode ruleset {config.ruleset_id!r} does not match {self.ruleset_id!r}")
            if config.environment != self.environment_config:
                raise BattleEnvError("episode environment config does not match this BattleEnv/ruleset")
            chosen_seed = config.seed if seed is None else int(seed)
            match = MatchConfig(
                deck0=config.deck0,
                deck1=config.deck1,
                seed=chosen_seed,
                game_mode=config.game_mode,
                arena=config.arena,
                location=int(config.tags.get("location", 15000199)),
                level_cap=int(config.tags.get("level_cap", 0)),
                minimum_card_level=int(config.tags.get("minimum_card_level", 0)),
                deck0_form_availability=tuple(
                    int(value) for value in config.tags.get("deck0_form_availability", (0,) * 8)
                ),
                deck1_form_availability=tuple(
                    int(value) for value in config.tags.get("deck1_form_availability", (0,) * 8)
                ),
                tower_troop0_id=int(config.tags.get("tower_troop0_id", PRINCESS_TOWER_TROOP_ID)),
                tower_troop1_id=int(config.tags.get("tower_troop1_id", PRINCESS_TOWER_TROOP_ID)),
                king_tower_level=(
                    int(config.tags["king_tower_level"]) if config.tags.get("king_tower_level") is not None else None
                ),
                owner0_name=str(config.tags.get("owner0_name", "Policy-0")),
                owner1_name=str(config.tags.get("owner1_name", "Policy-1")),
                end_tick=int(config.tags.get("end_tick", 7200)),
            )
            episode = replace(
                config,
                seed=chosen_seed,
                render_mode=render_mode,
                tags={
                    **dict(config.tags),
                    "tower_troop0_id": match.tower_troop0_id,
                    "tower_troop1_id": match.tower_troop1_id,
                },
            )
        else:
            match = (
                match_config
                or self.match_config
                or MatchConfig(deck0=SEMANTIC_BASELINE_DECK, deck1=SEMANTIC_BASELINE_DECK, seed=seed or 1)
            )
            if seed is not None and match.seed != int(seed):
                match = replace(match, seed=int(seed))
            episode = self._episode_from_match(match, render_mode)

        if "decision_ticks" in opts:
            decision_ticks = int(opts["decision_ticks"])
            if decision_ticks < 1:
                raise ValueError("decision_ticks must be positive")
            episode = replace(episode, decision_hz=TICKS_PER_SECOND / decision_ticks)
        if "event_driven_decisions" in opts:
            event_driven_decisions = opts["event_driven_decisions"]
            if not isinstance(event_driven_decisions, bool):
                raise TypeError("event_driven_decisions must be boolean")
            episode = replace(episode, event_driven_decisions=event_driven_decisions)

        standard_presentation = match.game_mode == self.mode_timeline.game_mode_id and match.arena == STANDARD_ARENA
        verified_challenge_alias = bool(
            allow_verified_standard_layout_alias
            and match.game_mode == CLASSIC_CHALLENGE_GAME_MODE
            and match.arena == CLASSIC_CHALLENGE_ARENA
            and episode.tags.get("native_arena_profile") == "classic-legendary-arena.v1"
        )
        if not (standard_presentation or verified_challenge_alias) or episode.map_id != "standard-1v1":
            raise BattleEnvError(
                "BattleEnvV1 fails closed outside verified standard-layout 1v1 "
                "presentations; use NativeClashEnv directly until that map has "
                "its own layout and masks"
            )
        if int(match.end_tick) != self.max_battle_ticks:
            raise BattleEnvError(
                "native match end_tick and environment max_battle_ticks "
                f"must match exactly: {match.end_tick} != "
                f"{self.max_battle_ticks}"
            )
        for owner, tower_troop_id in ((0, match.tower_troop0_id), (1, match.tower_troop1_id)):
            if tower_troop_id not in COMPETITIVE_TOWER_TROOP_IDS:
                raise BattleEnvError(f"owner {owner} tower troop is not in the current competitive tower troop pool")

        unknown_cards = sorted(set((*match.deck0, *match.deck1)).difference(self.card_specs))
        if unknown_cards:
            raise BattleEnvError(
                "canonical BattleEnvV1 requires a content-addressed CardSpec for "
                f"every deck card; unknown IDs: {unknown_cards}. Use NativeClashEnv "
                "directly for low-level compatibility probing."
            )

        try:
            self.semantic_subset_contract.assert_deck(
                match.deck0, form_availability=match.deck0_form_availability, label="deck0"
            )
            self.semantic_subset_contract.assert_deck(
                match.deck1, form_availability=match.deck1_form_availability, label="deck1"
            )
        except SemanticSubsetError as error:
            raise BattleEnvError(str(error)) from error

        # Final pre-mutation gate: a host-side manifest cannot identify the
        # process currently listening on the endpoint. Resident training workers
        # may reuse their previously verified immutable identity; a cached
        # result is required before reuse.
        reuse_runner_attestation = opts.get("reuse_runner_attestation", False)
        if not isinstance(reuse_runner_attestation, bool):
            raise TypeError("reuse_runner_attestation must be boolean")
        if reuse_runner_attestation:
            if self.ruleset_manifest is not None and self.runner_attestation is None:
                raise BattleEnvError("runner attestation reuse was requested before preparation")
        else:
            self._verify_runner_attestation()

        if render_mode == "native-render":
            raw = self.native.create_native_match(match)
            if pause_native_render or synchronize_native_render_steps:
                self.native.pause()
            if suppress_native_render:
                self.native.set_rendering(False)
            if native_render_speed is not None:
                self.native.set_speed(float(native_render_speed))
        elif render_mode == "headless":
            raw = self.native.create_match(match)
            if headless_initial_step_ticks:
                self.native.step(headless_initial_step_ticks)
                raw = self.native.observe()
            if self.warmup_ticks and not defer_headless_warmup:
                self.native.step(self.warmup_ticks)
                raw = self.native.observe()
        else:
            raise ValueError("render_mode must be headless or native-render")

        raw, prefetched_rich = self._atomic_observation_pair()

        self.match_config = match
        self.episode_config = episode
        self.episode_id = str(uuid.uuid4())
        self._public_metadata_static = None
        # Native tick zero is the episode clock.  The headless warm-up is part
        # of the real elixir/timeline state and must not be subtracted away.
        self._episode_start_native_tick = 0
        self._raw = dict(raw)
        self._invalidate_rich_cache()
        self._ordinary_tower_cache.clear()
        self._action_mask_cache.clear()
        self._action_player_runtime_cache.clear()
        if prefetched_rich is not None:
            self._prime_rich_cache(self._raw, prefetched_rich)
        self._events.clear()
        self._event_archive.clear()
        self._pending.clear()
        self._pending_abilities.clear()
        self._revealed = {0: set(), 1: set()}
        # Initial elixir is deterministic public ruleset state.  Seeding the
        # belief from it avoids silently assuming six elixir after warm-up.
        self._belief_elixir = {
            owner: (
                float(_player(self._raw, owner).get("elixirRaw", 0)) / 10_000.0,
                float(_player(self._raw, owner).get("elixirRaw", 0)) / 10_000.0,
            )
            for owner in self.possible_agents
        }
        self._entity_birth.clear()
        self._previous_entities.clear()
        self._previous_rich_objects.clear()
        self._previous_player_runtime.clear()
        self._combat_event_identity = None
        self._combat_next_sequence = None
        self._public_card_play_event_identity = None
        self._public_card_play_next_sequence = None
        self._combat_native_entity_ids.clear()
        self._combat_evolution_entities.clear()
        self._causal_group_by_entity.clear()
        self._combat_event_kind_by_sequence.clear()
        self._phase_event_identity = None
        self._phase_next_sequence = None
        self._phase_events_by_entity.clear()
        self._visibility_event_identity = None
        self._visibility_next_sequence = None
        self._remaining_event_identity = None
        self._remaining_next_sequence = None
        self._remaining_rejected_count_baseline = None
        self._entity_velocity.clear()
        self._previous_towers.clear()
        self._terminated = False
        self._truncated = False
        self._trace.clear()
        self._decision_deadline_native = {owner: int(raw["tick"]) for owner in self.possible_agents}
        self._last_advance_event = False
        self._tower_object_keys = self._bind_tower_object_keys(self._raw)
        self._activated_king_tower_ids.clear()
        self._initial_tower_hp.clear()
        initial_towers = self._extract_towers(self._raw)
        self._initial_tower_hp = {tower.entity_id: tower.max_hitpoints for tower in initial_towers}
        self._previous_towers = {tower.entity_id: tower for tower in initial_towers}
        initial_rich_context = self._rich_context(self._raw)
        self._previous_entities = self._raw_entity_map(self._raw, rich_context=initial_rich_context)
        self._prime_combat_event_cursor(initial_rich_context.snapshot)
        self._prime_phase_event_cursor(initial_rich_context.snapshot)
        self._prime_visibility_event_cursor(initial_rich_context.snapshot)
        self._prime_remaining_event_cursor(initial_rich_context.snapshot)
        self._previous_rich_objects = {
            canonical_id: initial_rich_context.snapshot.objects_by_key[key]
            for key, canonical_id in initial_rich_context.canonical_ids.items()
            if key in initial_rich_context.visible_keys
        }
        self._previous_player_runtime = dict(initial_rich_context.snapshot.players)
        initial_public_card_plays = self._consume_native_public_card_plays(initial_rich_context)
        self._append_native_public_card_plays(
            initial_public_card_plays,
            resolved_pending=(),
            update_belief=False,
            form_aliases=self._native_form_aliases(initial_rich_context),
        )
        for entity_id in self._previous_entities:
            self._entity_birth[entity_id] = int(self._raw["tick"])
        self._last_potential = {owner: self._potential(owner, initial_towers) for owner in self.possible_agents}
        self._last_rewards = {0: 0.0, 1: 0.0}

        observations = self.observe_all(ObservationTier.FAIR)
        info = {
            "episode_id": self.episode_id,
            "ruleset_id": self.ruleset_id,
            "native_tick": int(raw["tick"]),
            "seed": match.seed,
            "render_mode": render_mode,
            "empty_recorded_commands": True,
            "observation_version": ObservationV1.VERSION,
            "action_version": ActionV1.VERSION,
            "environment_config": self.environment_config.to_dict(),
            "semantic_subset_id": self.semantic_subset_contract.subset_id,
            "semantic_subset_criteria_version": (self.semantic_subset_contract.criteria_version),
            "runner_attestation_digest": (
                self.runner_attestation.attestation_digest if self.runner_attestation is not None else None
            ),
        }
        return observations, info

    def _invalidate_rich_cache(self) -> None:
        self._rich_cache_identity = None
        self._rich_cache = None
        self._fair_context_cache_identity = None
        self._fair_context_cache = None
        self._fair_scene_cache_identity = None
        self._fair_scene_cache = None

    @staticmethod
    def _rich_identity(raw: Mapping[str, Any]) -> tuple[int, int, int | None]:
        try:
            return (
                int(raw["tick"]),
                int(raw["generation"]),
                int(raw["stateEpoch"]) if raw.get("stateEpoch") is not None else None,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise BattleEnvError("ordinary observation lacks tick/generation identity for rich merge") from error

    def _atomic_observation_pair(self) -> tuple[dict[str, Any], RichTelemetrySnapshot]:
        try:
            capture = self.native.observe_atomic()
        except RunnerError as error:
            raise BattleEnvError(f"native atomic telemetry failed closed: {error}") from error
        direct_ordinary = getattr(capture, "ordinary", None)
        direct_snapshot = getattr(capture, "snapshot", None)
        if isinstance(direct_ordinary, Mapping) and isinstance(direct_snapshot, RichTelemetrySnapshot):
            return dict(direct_ordinary), direct_snapshot
        if not isinstance(capture, Mapping):
            raise BattleEnvError("native atomic telemetry response is not a mapping")
        ordinary = capture.get("ordinary")
        rich = capture.get("rich")
        if not isinstance(ordinary, Mapping) or not isinstance(rich, Mapping):
            raise BattleEnvError("native atomic telemetry lacks ordinary/rich envelopes")
        raw = dict(ordinary)
        capture_identity = (int(raw.get("generation", -1)), int(raw.get("stateEpoch", -1)))
        combat_floors = tuple(
            value
            for identity, value in (
                (self._combat_event_identity, self._combat_next_sequence),
                (self._public_card_play_event_identity, self._public_card_play_next_sequence),
            )
            if identity == capture_identity and value is not None
        )
        try:
            snapshot = bind_rich_telemetry(
                raw,
                rich,
                combat_event_floor=(min(combat_floors) if combat_floors else None),
                phase_event_floor=(
                    self._phase_next_sequence if self._phase_event_identity == capture_identity else None
                ),
                visibility_event_floor=(
                    self._visibility_next_sequence if self._visibility_event_identity == capture_identity else None
                ),
                remaining_event_floor=(
                    self._remaining_next_sequence if self._remaining_event_identity == capture_identity else None
                ),
            )
        except RichTelemetryMergeError as error:
            raise BattleEnvError(f"native atomic telemetry merge failed: {error}") from error
        return raw, snapshot

    def _prime_rich_cache(self, raw: Mapping[str, Any], snapshot: RichTelemetrySnapshot) -> None:
        self._rich_cache_identity = self._rich_identity(raw)
        self._rich_cache = snapshot

    def _rich_snapshot_for(self, raw: Mapping[str, Any]) -> RichTelemetrySnapshot:
        identity = self._rich_identity(raw)
        if self._rich_cache_identity == identity and self._rich_cache is not None:
            return self._rich_cache
        atomic_raw, snapshot = self._atomic_observation_pair()
        atomic_identity = self._rich_identity(atomic_raw)
        if atomic_identity != identity:
            raise BattleEnvError(
                "native atomic telemetry does not match the requested cached "
                f"observation: requested={identity}, captured={atomic_identity}"
            )
        self._prime_rich_cache(raw, snapshot)
        return snapshot

    def _bind_tower_object_keys(self, raw: Mapping[str, Any]) -> dict[int, int]:
        """Bind each fixed tower to its episode-stable native identity."""

        objects = tuple(item for item in raw.get("objects", ()) if not item.get("null"))
        bound: dict[int, int] = {}
        for entity_id, owner, _kind, x, y in TOWER_LAYOUT:
            candidates = tuple(
                item
                for item in objects
                if (int(item.get("x", -1)), int(item.get("y", -1))) == (x, y)
                and int(item.get("owner", -1)) == owner
                and item.get("maxHp") is not None
                and float(item["maxHp"]) > 0.0
            )
            if len(candidates) != 1:
                raise BattleEnvError(
                    f"initial native state does not contain exactly one tower at {(x, y)} for owner {owner}"
                )
            bound[entity_id] = _entity_id(candidates[0])
        if len(set(bound.values())) != len(TOWER_LAYOUT):
            raise BattleEnvError("initial fixed towers do not have unique native identities")
        return bound

    def _tower_identity_for_item(self, item: Mapping[str, Any]) -> tuple[int, int, str, int, int] | None:
        object_key = _entity_id(item)
        tower_object_keys = getattr(self, "_tower_object_keys", {})
        identities = tuple(identity for identity in TOWER_LAYOUT if tower_object_keys.get(identity[0]) == object_key)
        if not identities:
            return None
        if len(identities) != 1:
            raise BattleEnvError("one native object identity is bound to multiple towers")
        entity_id, owner, kind, x, y = identities[0]
        if (
            int(item.get("owner", -1)) != owner
            or (int(item.get("x", -1)), int(item.get("y", -1))) != (x, y)
            or item.get("maxHp") is None
            or float(item["maxHp"]) <= 0.0
        ):
            raise BattleEnvError(
                f"bound native tower identity changed metadata: entity_id={entity_id}, object_key={object_key}"
            )
        return entity_id, owner, kind, x, y

    def _bound_tower_items(self, raw: Mapping[str, Any]) -> dict[int, Mapping[str, Any] | None]:
        expected_ids = {item[0] for item in TOWER_LAYOUT}
        if set(self._tower_object_keys) != expected_ids:
            raise BattleEnvError("fixed native tower identities are not initialized")
        result: dict[int, Mapping[str, Any] | None] = {entity_id: None for entity_id in expected_ids}
        for item in raw.get("objects", ()):
            if item.get("null"):
                continue
            identity = self._tower_identity_for_item(item)
            if identity is None:
                continue
            entity_id = identity[0]
            if result[entity_id] is not None:
                raise BattleEnvError(f"native capture repeated tower identity {entity_id}")
            result[entity_id] = item
        return result

    def _is_bound_tower_item(self, item: Mapping[str, Any]) -> bool:
        return self._tower_identity_for_item(item) is not None

    def _rich_context(self, raw: Mapping[str, Any]) -> _RichObservationContext:
        identity = self._rich_identity(raw)
        if self._fair_context_cache_identity == identity and self._fair_context_cache is not None:
            return self._fair_context_cache
        snapshot = self._rich_snapshot_for(raw)
        canonical_ids: dict[EntityKey, int] = {}
        fixed_tower_keys: set[EntityKey] = set()
        for raw_item in raw.get("objects", ()):
            if raw_item.get("null"):
                continue
            item = snapshot.at_slot(int(raw_item["slot"]))
            if item is None:
                raise BattleEnvError("rich merge produced a null slot for a live object")
            tower_identity = self._tower_identity_for_item(raw_item)
            canonical_id = (
                int(tower_identity[0])
                if tower_identity is not None
                # nativeObjectId is validated unique by the rich adapter.
                # The former object_index * 1000 + secondary_index encoding
                # collided as soon as a secondary index reached 1000.
                else 1_000_000_000 + item.native_object_id
            )
            if item.entity_key in canonical_ids:
                raise BattleEnvError(f"duplicate canonical rich identity {item.entity_key}")
            if canonical_id in canonical_ids.values():
                raise BattleEnvError(f"duplicate canonical entity id {canonical_id} during rich merge")
            canonical_ids[item.entity_key] = canonical_id
            if tower_identity is not None:
                fixed_tower_keys.add(item.entity_key)

        owner_root_keys = frozenset(
            player.owner_entity_key
            for player in snapshot.players.values()
            if player.owner_root_validated and player.owner_entity_key is not None
        )
        # Native invisibility/submerge state controls unit acquisition and
        # interaction; it does not hide the arena entity from either player.
        # Ghost has a public translucent render, Tesla has a public submerged
        # render, and Suspicious Bush has a public bush render.  Player-root
        # objects remain private implementation state.
        private_owner_root_keys = owner_root_keys.difference(fixed_tower_keys)
        visible_keys = frozenset(canonical_ids).difference(private_owner_root_keys)
        result = _RichObservationContext(snapshot=snapshot, canonical_ids=canonical_ids, visible_keys=visible_keys)
        # Both FAIR actors see the same public arena objects. Actor-private
        # state is projected later by _players, action_mask and _events_for.
        self._fair_context_cache_identity = identity
        self._fair_context_cache = result
        return result

    def _visible_entity_count(self, raw: Mapping[str, Any], context: _RichObservationContext) -> int:
        count = 0
        for item in raw.get("objects", ()):
            if item.get("null"):
                continue
            if self._is_bound_tower_item(item):
                continue
            rich_item = context.item_for_raw(item)
            if rich_item.entity_key in context.visible_keys:
                count += 1
        return count

    def _raw_entity_map(
        self, raw: Mapping[str, Any], *, rich_context: _RichObservationContext | None = None
    ) -> dict[int, dict[str, Any]]:
        result: dict[int, dict[str, Any]] = {}
        owner_root_keys = (
            frozenset(
                player.owner_entity_key
                for player in rich_context.snapshot.players.values()
                if player.owner_root_validated and player.owner_entity_key is not None
            )
            if rich_context is not None
            else frozenset()
        )
        for item in raw.get("objects", ()):
            if item.get("null"):
                continue
            if self._is_bound_tower_item(item):
                continue
            if rich_context is not None:
                rich_item = rich_context.item_for_raw(item)
                if rich_item.entity_key in owner_root_keys:
                    continue
                result[rich_context.canonical_ids[rich_item.entity_key]] = dict(item)
            else:
                result[_entity_id(item)] = dict(item)
        return result

    def _configured_tower_troop_id(self, owner: int) -> int:
        if self.episode_config is None:
            raise BattleEnvError("tower troop identity requires an episode config")
        tag = f"tower_troop{owner}_id"
        raw_id = self.episode_config.tags.get(tag)
        if isinstance(raw_id, bool) or not isinstance(raw_id, int):
            raise BattleEnvError(f"episode tag {tag} is not an exact integer ID")
        if raw_id not in COMPETITIVE_TOWER_TROOP_IDS:
            raise BattleEnvError(f"episode tag {tag} is not a current competitive tower troop")
        return raw_id

    @staticmethod
    def _tower_troop_id(configured_id: int, tower_kind: str) -> int | None:
        if configured_id == ROYAL_CHEF_TOWER_TROOP_ID:
            return ROYAL_CHEF_TOWER_TROOP_ID if tower_kind == "king" else PRINCESS_TOWER_TROOP_ID
        return None if tower_kind == "king" else configured_id

    def _extract_towers(
        self, raw: Mapping[str, Any], *, rich_context: _RichObservationContext | None = None
    ) -> tuple[TowerStateV1, ...]:
        cache_identity: tuple[int, int, int | None] | None = None
        if rich_context is None:
            cache_identity = self._rich_identity(raw)
            cached = self._ordinary_tower_cache.get(cache_identity)
            if cached is not None:
                return cached
        tower_items = self._bound_tower_items(raw)
        configured_tower_troops = {owner: self._configured_tower_troop_id(owner) for owner in (0, 1)}
        surviving_side_towers = {
            owner: sum(
                1
                for entity_id, layout_owner, kind, _x, _y in TOWER_LAYOUT
                if layout_owner == owner
                and kind != "king"
                and tower_items[entity_id] is not None
                and float(tower_items[entity_id]["hp"]) > 0
            )
            for owner in (0, 1)
        }
        if rich_context is not None and any(
            tower_troop_id in {159_000_002, 159_000_004} for tower_troop_id in configured_tower_troops.values()
        ):
            tower_runtime_envelope = rich_context.snapshot.tower_troop_runtime
            if tower_runtime_envelope is None or not tower_runtime_envelope.complete:
                raise BattleEnvError("special Tower Troop match lacks complete native runtime telemetry")
        result: list[TowerStateV1] = []
        for entity_id, owner, kind, x, y in TOWER_LAYOUT:
            item = tower_items[entity_id]
            known_max = self._initial_tower_hp.get(entity_id)
            max_hp = float(item["maxHp"]) if item is not None else float(known_max or 1.0)
            hp = max(0.0, float(item["hp"])) if item is not None else 0.0
            runtime_fields: dict[str, Any] = {}
            tower_runtime_state: DaggerDuchessRuntimeStateV1 | RoyalChefRuntimeStateV1 | None = None
            tower_troop_id = self._tower_troop_id(configured_tower_troops[owner], kind)
            tower_active = item is not None and hp > 0
            tower_runtime_relevant = tower_active and (
                (kind != "king" and tower_troop_id == 159_000_002)
                or (kind == "king" and configured_tower_troops[owner] == ROYAL_CHEF_TOWER_TROOP_ID)
            )
            if rich_context is not None and item is not None:
                rich_item = rich_context.item_for_raw(item)
                if rich_item.entity_key not in rich_context.visible_keys:
                    raise BattleEnvError("FAIR rich telemetry marked a fixed arena tower invisible")
                runtime = project_runtime(
                    rich_item,
                    canonical_ids=rich_context.canonical_ids,
                    rich_objects=rich_context.snapshot.objects_by_key,
                    visible_keys=rich_context.visible_keys,
                    observed_tick=self._canonical_tick(int(raw["tick"])),
                    effect_catalog=self.effect_catalog,
                    phase_runtime=rich_context.snapshot.phase_runtime,
                    phase_events=self._phase_events_by_entity.get(rich_item.entity_key, ()),
                    card_spec=self.card_specs.get(rich_item.card_id),
                )
                tower_runtime = rich_item.tower_troop_runtime
                if isinstance(tower_runtime, RichDaggerDuchessRuntimeTelemetry):
                    if kind == "king" or tower_troop_id != 159_000_002:
                        raise BattleEnvError("Dagger Duchess runtime is attached to the wrong tower")
                    tower_runtime_state = DaggerDuchessRuntimeStateV1(
                        charge_count=tower_runtime.charge_count,
                        max_charge_count=tower_runtime.max_charge_count,
                        recharge_elapsed_ms=tower_runtime.recharge_elapsed_ms,
                        recharge_duration_ms=tower_runtime.recharge_duration_ms,
                    )
                elif isinstance(tower_runtime, RichRoyalChefRuntimeTelemetry):
                    if kind != "king" or configured_tower_troops[owner] != ROYAL_CHEF_TOWER_TROOP_ID:
                        raise BattleEnvError("Royal Chef runtime is attached to the wrong tower")
                    chef_target: int | None = None
                    if tower_runtime.target_native_object_id is not None:
                        target_item = rich_context.snapshot.objects_by_native_object_id.get(
                            tower_runtime.target_native_object_id
                        )
                        # The native Chef keeps the selected object ID while
                        # its throw delay is pending even when that object has
                        # already died and left the live object map.  The
                        # eventual pancake is deliberately wasted; expose the
                        # pending throw with no target instead of inventing an
                        # entity or treating the dead reference as envelope
                        # corruption.
                        if target_item is not None:
                            if target_item.entity_key not in rich_context.visible_keys:
                                raise BattleEnvError("Royal Chef target is outside FAIR arena visibility")
                            chef_target = rich_context.canonical_ids[target_item.entity_key]
                    tower_runtime_state = RoyalChefRuntimeStateV1(
                        start_delay_remaining_ms=(tower_runtime.start_delay_remaining_ms),
                        start_delay_duration_ms=(tower_runtime.start_delay_duration_ms),
                        cooking_contribution=tower_runtime.cooking_contribution,
                        contribution_needed=tower_runtime.contribution_needed,
                        surviving_side_towers=surviving_side_towers[owner],
                        throw_delay_remaining_ms=(tower_runtime.throw_delay_remaining_ms),
                        target_entity=chef_target,
                    )
                elif tower_runtime_relevant:
                    raise BattleEnvError("active special Tower Troop lacks its native runtime snapshot")
                tower_provenance = runtime.tower_provenance(self._canonical_tick(int(raw["tick"])))
                tower_runtime_evidence = (
                    SemanticEvidenceLevel.NATIVE_DERIVED
                    if tower_runtime_state is not None
                    else SemanticEvidenceLevel.NOT_APPLICABLE
                    if not tower_runtime_relevant
                    else SemanticEvidenceLevel.UNKNOWN
                )
                tower_provenance = SemanticProvenanceV1(
                    field_evidence={**tower_provenance.field_evidence, "tower_troop_runtime": tower_runtime_evidence},
                    source_fields={
                        **tower_provenance.source_fields,
                        **(
                            {"tower_troop_runtime": ("objects[].towerTroopRuntime", "towerTroopRuntime")}
                            if tower_runtime_state is not None
                            else {}
                        ),
                    },
                    observed_tick=tower_provenance.observed_tick,
                    notes=tower_provenance.notes,
                )
                runtime_fields = {
                    "shield": runtime.shield or 0.0,
                    "visible_target": runtime.target_entity,
                    "shield_state": runtime.shield_state,
                    "effect_states": runtime.effect_states,
                    "attack_state": runtime.attack_state,
                    "visibility_state": runtime.visibility_state,
                    "tower_troop_runtime": tower_runtime_state,
                    "runtime_provenance": tower_provenance,
                }
            elif not tower_runtime_relevant:
                runtime_fields["runtime_provenance"] = SemanticProvenanceV1(
                    field_evidence={
                        name: (
                            SemanticEvidenceLevel.NOT_APPLICABLE
                            if name == "tower_troop_runtime"
                            else SemanticEvidenceLevel.UNKNOWN
                        )
                        for name in TOWER_RUNTIME_SEMANTIC_FIELDS
                    },
                    observed_tick=self._canonical_tick(int(raw["tick"])),
                )
            result.append(
                TowerStateV1(
                    entity_id=entity_id,
                    owner=owner,
                    tower_kind=kind,
                    position=(float(x), float(y)),
                    hitpoints=min(hp, max_hp),
                    max_hitpoints=max_hp,
                    tower_troop_id=tower_troop_id,
                    active=tower_active,
                    status=(("activated",) if entity_id in self._activated_king_tower_ids else ()),
                    **runtime_fields,
                )
            )
        towers = tuple(result)
        if cache_identity is not None:
            if len(self._ordinary_tower_cache) >= 4:
                self._ordinary_tower_cache.clear()
            self._ordinary_tower_cache[cache_identity] = towers
        return towers

    @staticmethod
    def _crowns(towers: Sequence[TowerStateV1], raw: Mapping[str, Any] | None = None) -> dict[int, int]:
        if raw is not None:
            exposed: dict[int, int] = {}
            for player in raw.get("players", ()):
                crowns_value = player.get("crownsRaw", player.get("crowns"))
                if player.get("owner") in (0, 1) and crowns_value is not None:
                    exposed[int(player["owner"])] = int(crowns_value)
            top_level = raw.get("crownsRaw")
            if set(exposed) != {0, 1} and isinstance(top_level, Sequence) and len(top_level) == 2:
                exposed = {0: int(top_level[0]), 1: int(top_level[1])}
            if set(exposed) == {0, 1} and all(0 <= value <= 3 for value in exposed.values()):
                return exposed
        result = {0: 0, 1: 0}
        for victim in (0, 1):
            king = next(item for item in towers if item.owner == victim and item.tower_kind == "king")
            attacker = 1 - victim
            if not king.active:
                result[attacker] = 3
            else:
                result[attacker] = sum(
                    1
                    for item in towers
                    if item.owner == victim and item.tower_kind.startswith("princess") and not item.active
                )
        return result

    def _terminal(self, raw: Mapping[str, Any], towers: Sequence[TowerStateV1]) -> TerminalV1:
        kings = {item.owner: item for item in towers if item.tower_kind == "king"}
        dead_kings = [owner for owner, tower in kings.items() if not tower.active]
        finalized = raw.get("finalized", raw.get("finalizedEnded"))
        ended = (bool(finalized) if finalized is not None else bool(raw.get("ended"))) or bool(dead_kings)
        if not ended:
            return TerminalV1()
        winner: int | None = None
        reason = "engine_ended"
        result_owner = raw.get("winner", raw.get("resultOwner"))
        if result_owner in (0, 1):
            winner = int(result_owner)
            reason = "native_final_result"
        elif len(dead_kings) == 1:
            winner = 1 - dead_kings[0]
            reason = "king_tower_destroyed"
        else:
            crowns = self._crowns(towers, raw)
            if crowns[0] != crowns[1]:
                winner = 0 if crowns[0] > crowns[1] else 1
                reason = "crown_advantage"
            else:
                # The engine normally resolves tiebreak by reducing a tower to
                # zero. If this build stops without doing so, preserve a draw
                # instead of inventing a winner from hidden state.
                reason = "draw_or_unexposed_tiebreak"
        results = (0.0, 0.0) if winner is None else ((1.0, -1.0) if winner == 0 else (-1.0, 1.0))
        return TerminalV1(
            ended=True,
            winner=winner,
            result_by_owner=results,
            reason=reason,
            terminal_tick=self._canonical_tick(int(raw["tick"])),
        )

    def _canonical_tick(self, native_tick: int) -> int:
        return max(0, int(native_tick) - self._episode_start_native_tick)

    def _destroyed_enemy_lanes(self, owner: int, towers: Sequence[TowerStateV1]) -> tuple[str, ...]:
        opponent = 1 - owner
        lanes: list[str] = []
        for tower in towers:
            if tower.owner != opponent or tower.active:
                continue
            if tower.tower_kind == "princess_left":
                lanes.append("left")
            elif tower.tower_kind == "princess_right":
                lanes.append("right")
        return tuple(lanes)

    def _assert_policy_ability_enabled(self, owner: int, state: AbilityRuntimeStateV1) -> None:
        """Bind one typed ability to its static card and episode deck slot."""

        spec = self.ability_specs.get(state.ability_id)
        source_card_id = state.attributes.get("source_card_id")
        if spec is None or type(source_card_id) is not int or spec.source_card_id != source_card_id:
            raise BattleEnvError("native ability runtime has no exact policy source-card contract")
        card_spec = self.card_specs.get(source_card_id)
        if (
            card_spec is None
            or state.ability_id not in card_spec.ability_ids
            or not self._policy_form_masks.get(source_card_id, 0) & 2
        ):
            raise BattleEnvError(
                f"native ability runtime source is not policy-ready: card={source_card_id}, ability={state.ability_id}"
            )
        episode_form_mask = self._episode_form_mask(owner, source_card_id)
        if source_card_id in self._explicit_hero_form_base_ids and not episode_form_mask & 2:
            raise BattleEnvError(
                "native explicit-Hero ability is not enabled by the episode "
                f"deck contract: owner={owner}, card={source_card_id}, "
                f"enabled={episode_form_mask}"
            )

    def _ability_action_candidates(
        self, owner: int, source: Mapping[str, Any], *, rich_context: _RichObservationContext | None = None
    ) -> tuple[dict[int, tuple[AbilityRuntimeStateV1, EntityKey]], Mapping[str, Any]]:
        """Return only exact, currently legal native type-2 activations."""

        native_command_available = callable(getattr(self.native, "queue_ability_action_at", None))
        context = rich_context or self._rich_context(source)
        if context is None:
            return {}, {"status": "unavailable", "reason": "rich_runtime_missing"}
        raw_player = _player(source, owner)
        rich_player = context.snapshot.players.get(owner)
        if rich_player is None:
            return {}, {"status": "unavailable", "reason": "owner_runtime_missing"}
        runtime = self._action_player_runtime(owner, source, context)
        if runtime is None:
            return {}, {"status": "unavailable", "reason": "owner_runtime_missing"}
        if not native_command_available:
            return {}, {"status": "unavailable", "reason": "native_command_api_missing"}
        key_by_canonical_id = {canonical_id: entity_key for entity_key, canonical_id in context.canonical_ids.items()}
        elixir = float(int(raw_player.get("elixirRaw", 0))) / 10_000.0
        candidates: dict[int, tuple[AbilityRuntimeStateV1, EntityKey]] = {}
        reasons: dict[str, str] = {}
        for state in runtime.ability_runtime_states:
            source_entity = state.source_entity
            reason = "legal"
            spec = self.ability_specs.get(state.ability_id)
            entity_key = key_by_canonical_id.get(source_entity) if source_entity is not None else None
            if spec is not None:
                self._assert_policy_ability_enabled(owner, state)
            if source_entity is None or entity_key is None:
                reason = "exact_source_unavailable"
            elif spec is None:
                reason = "ability_spec_unavailable"
            elif spec.target_schema.allowed != (TargetKind.NONE,):
                # ct=2 has no target fields. Never silently discard a target
                # required by a future or differently encoded ability.
                reason = "native_command_has_no_target_payload"
            elif state.phase != AbilityPhase.READY or state.available is not True:
                reason = f"native_phase_{state.phase.value}"
            elif state.remaining_cooldown_ms != 0:
                reason = "cooldown_nonzero"
            elif state.charges is not None and state.charges <= 0:
                reason = "charges_exhausted"
            elif state.elixir_cost is None:
                reason = "elixir_cost_unknown"
            elif state.elixir_cost > elixir:
                reason = "insufficient_elixir"
            elif source_entity in candidates:
                reason = "duplicate_source_identity"
                candidates.pop(source_entity, None)
            else:
                candidates[source_entity] = (state, entity_key)
            reasons[str(source_entity)] = reason
        return candidates, {
            "status": "legal" if candidates else "masked",
            "command": "native_ct2_T_Tplus1",
            "sources": reasons,
        }

    def _action_player_runtime(
        self, owner: int, source: Mapping[str, Any], context: _RichObservationContext
    ) -> Any | None:
        """Project one owner's runtime once per atomic observation identity.

        The semantic training observer and the dynamic action mask both need
        this exact projection.  Keeping it keyed by the native identity avoids
        repeating the Python-heavy join while preserving per-owner privacy.
        """

        key = (*self._rich_identity(source), int(owner))
        cached = self._action_player_runtime_cache.get(key)
        if cached is not None:
            return cached
        rich_player = context.snapshot.players.get(owner)
        if rich_player is None:
            return None
        raw_player = _player(source, owner)
        runtime = project_player_runtime(
            rich_player,
            ordinary_deck=tuple(int(card["cardId"]) for card in raw_player.get("deck", ())),
            card_specs=self.card_specs,
            ability_specs=self.ability_specs,
            canonical_ids=context.canonical_ids,
            rich_objects=context.snapshot.objects_by_key,
            visible_keys=context.visible_keys,
            observed_tick=self._canonical_tick(int(source["tick"])),
        )
        for ability_state in runtime.ability_runtime_states:
            self._assert_policy_ability_enabled(owner, ability_state)
        if len(self._action_player_runtime_cache) >= 128:
            self._action_player_runtime_cache.clear()
        self._action_player_runtime_cache[key] = runtime
        return runtime

    def _episode_form_mask(self, owner: int, card_id: int) -> int:
        """Return the exact per-slot form mask bound to the active episode."""

        if owner not in (0, 1):
            raise BattleEnvError("native policy identity has an invalid owner")
        if self.match_config is None:
            raise BattleEnvError("native policy identity has no active episode deck contract")
        deck = self.match_config.deck0 if owner == 0 else self.match_config.deck1
        forms = self.match_config.deck0_form_availability if owner == 0 else self.match_config.deck1_form_availability
        matching_slots = tuple(slot for slot, deck_card_id in enumerate(deck) if int(deck_card_id) == int(card_id))
        if len(matching_slots) != 1:
            raise BattleEnvError(
                f"native policy identity is not uniquely present in the owner {owner} episode deck: {card_id}"
            )
        return int(forms[matching_slots[0]])

    def _policy_effective_identity(
        self,
        native_effective_card_id: int,
        native_form_code: int,
        *,
        form_aliases: Mapping[int, int] | None = None,
        owner: int | None = None,
    ) -> tuple[int, int]:
        """Split native command identity from the canonical policy identity."""

        if native_form_code not in {0, 1, 2}:
            raise BattleEnvError("native hand selection has invalid effective cost/form")
        aliases = dict(NORMAL_MODE_HERO_FORM_TO_BASE_CARD)
        for raw_form_card_id, raw_base_card_id in (form_aliases or {}).items():
            form_card_id = int(raw_form_card_id)
            base_card_id = int(raw_base_card_id)
            previous = aliases.get(form_card_id)
            if previous is not None and previous != base_card_id:
                raise BattleEnvError("native form card maps to multiple base card identities")
            aliases[form_card_id] = base_card_id
        effective_card_id = aliases.get(native_effective_card_id, native_effective_card_id)
        if native_effective_card_id in NORMAL_MODE_HERO_FORM_TO_BASE_CARD:
            if native_form_code not in {0, 2}:
                raise BattleEnvError("native Hero form identity has a conflicting native form code")
            form_code = 2
        elif native_effective_card_id in aliases and native_effective_card_id != effective_card_id:
            if native_form_code not in {0, 1}:
                raise BattleEnvError("native evolution form identity has a conflicting native form code")
            form_code = 1
        else:
            form_code = native_form_code
        if owner is not None and owner not in (0, 1):
            raise BattleEnvError("native policy identity has an invalid owner")
        episode_form_mask = self._episode_form_mask(owner, effective_card_id) if owner is not None else None
        required_form_bit = {1: 1, 2: 2}.get(form_code, 0)
        if required_form_bit:
            spec = self.card_specs.get(effective_card_id)
            supported_form_mask = self._policy_form_masks.get(effective_card_id, 0)
            exact_static_form = bool(
                spec is not None
                and supported_form_mask & required_form_bit
                and ((form_code == 1 and spec.evolution is not None) or (form_code == 2 and bool(spec.ability_ids)))
            )
            if not exact_static_form:
                raise BattleEnvError(
                    "native effective card identity exposes an unsupported "
                    f"policy form: card={effective_card_id}, form={form_code}"
                )
            if episode_form_mask is not None and not episode_form_mask & required_form_bit:
                raise BattleEnvError(
                    "native policy form is not enabled by the episode deck "
                    f"contract: owner={owner}, card={effective_card_id}, "
                    f"form={form_code}, enabled={episode_form_mask}"
                )
        return effective_card_id, form_code

    def _hand_runtime_contract(
        self, card: Mapping[str, Any], *, form_aliases: Mapping[int, int] | None = None, owner: int | None = None
    ) -> Mapping[str, int | float]:
        """Project one private native selection without losing Mirror state."""

        if any(
            isinstance(card.get(field), bool) or not isinstance(card.get(field), int)
            for field in ("cardId", "handIndex", "cost")
        ):
            raise BattleEnvError("native hand selection lacks exact identity/slot/cost")
        try:
            visible_card_id = int(card["cardId"])
            hand_slot = int(card["handIndex"])
            effective_cost = float(card["cost"])
        except (KeyError, TypeError, ValueError) as error:
            raise BattleEnvError("native hand selection lacks exact identity/slot/cost") from error
        for field in ("commandCardId", "formCode", "cardParameter"):
            value = card.get(field)
            if value is not None and type(value) is not int:
                raise BattleEnvError(f"native hand selection has non-integer {field}")
        raw_form = card.get("formCode")
        if raw_form is None and card.get("cardParameter") is not None:
            raw_form = int(card["cardParameter"]) & 0xF
        native_form_code = int(raw_form or 0)
        if (
            raw_form is not None
            and card.get("cardParameter") is not None
            and native_form_code != int(card["cardParameter"]) & 0xF
        ):
            raise BattleEnvError("native hand formCode contradicts cardParameter")
        raw_effective = card.get("commandCardId")
        native_effective_card_id = visible_card_id if raw_effective is None else int(raw_effective)
        effective_card_id, form_code = self._policy_effective_identity(
            native_effective_card_id, native_form_code, form_aliases=form_aliases, owner=owner
        )
        if owner is not None:
            self._episode_form_mask(owner, visible_card_id)
        if visible_card_id == MIRROR_CARD_ID:
            if raw_effective is None:
                raise BattleEnvError("Mirror hand selection has no exact effective card identity")
            if effective_card_id == MIRROR_CARD_ID or effective_card_id not in self.card_specs:
                raise BattleEnvError("Mirror hand selection has an invalid effective card identity")
            source_cost = self.card_specs[effective_card_id].elixir_cost
            if source_cost is None or not math.isclose(effective_cost, float(source_cost) + 1.0, abs_tol=1e-6):
                raise BattleEnvError("Mirror hand selection cost does not equal source cost plus one")
        else:
            if effective_card_id != visible_card_id:
                raise BattleEnvError("ordinary hand selection changed effective card identity")
        if not 0.0 <= effective_cost <= 15.0:
            raise BattleEnvError("native hand selection has invalid effective cost/form")
        return frozen_mapping(
            {
                "hand_slot": hand_slot,
                "visible_card_id": visible_card_id,
                "effective_card_id": effective_card_id,
                "native_effective_card_id": native_effective_card_id,
                "effective_cost": effective_cost,
                "form_code": form_code,
                "native_form_code": native_form_code,
            }
        )

    def _native_form_aliases(self, rich_context: _RichObservationContext) -> Mapping[int, int]:
        """Join exact runtime form globals to normal-mode base card IDs."""

        aliases: dict[int, int] = dict(NORMAL_MODE_HERO_FORM_TO_BASE_CARD)
        for player in rich_context.snapshot.players.values():
            for evolution in player.evolution_runtime or ():
                form_id = evolution.evolution_form_global_id
                if form_id is None:
                    continue
                form_card_id = int(form_id)
                base_card_id = int(evolution.card_id)
                spec = self.card_specs.get(base_card_id)
                if spec is None or spec.evolution is None:
                    raise BattleEnvError("native evolution form has no policy base-card contract")
                previous = aliases.get(form_card_id)
                if previous is not None and previous != base_card_id:
                    raise BattleEnvError("native form card maps to multiple base card identities")
                aliases[form_card_id] = base_card_id
        return aliases

    @staticmethod
    def _active_tower_footprints(towers: Sequence[TowerStateV1]) -> tuple[OccupiedFootprintV1, ...]:
        return tuple(
            OccupiedFootprintV1(
                float(tower.position[0]),
                float(tower.position[1]),
                4 if tower.tower_kind == "king" else 3,
                4 if tower.tower_kind == "king" else 3,
                f"tower:{tower.entity_id}:{tower.tower_kind}",
            )
            for tower in towers
            if tower.active
        )

    def _occupied_building_footprints(
        self, raw: Mapping[str, Any], rich_context: _RichObservationContext | None = None
    ) -> tuple[tuple[OccupiedFootprintV1, ...], tuple[int, ...]]:
        """Project live native building roots into exact occupied rectangles."""

        occupied: dict[tuple[int, int, int, int], OccupiedFootprintV1] = {}
        unknown: set[int] = set()
        evolution_aliases: dict[int, int] = {}
        if rich_context is not None:
            for player in rich_context.snapshot.players.values():
                for evolution in player.evolution_runtime or ():
                    form_id = evolution.evolution_form_global_id
                    if form_id is None:
                        continue
                    existing = evolution_aliases.get(int(form_id))
                    if existing is not None and existing != int(evolution.card_id):
                        raise BattleEnvError("native evolution form maps to multiple base cards")
                    evolution_aliases[int(form_id)] = int(evolution.card_id)
        for item in raw.get("objects", ()):
            if item.get("null") or float(item.get("hp", 0.0) or 0.0) <= 0.0:
                continue
            position = (int(item.get("x", -1)), int(item.get("y", -1)))
            if self._is_bound_tower_item(item):
                continue
            raw_card_id = int(item.get("cardId", 0) or 0)
            resolved_card_id = evolution_aliases.get(raw_card_id, raw_card_id)
            spec = self.card_specs.get(resolved_card_id)
            if spec is None or spec.kind.value != "building":
                continue
            placement_profile = building_placement_profile(resolved_card_id)
            if (
                placement_profile is not None
                and placement_profile.deploy_target_ready
                and not placement_profile.stationary_collision_rectangle
            ):
                continue
            footprint = native_building_footprint(spec)
            if footprint is None:
                unknown.add(resolved_card_id)
                continue
            signature = (position[0], position[1], footprint.width_tiles, footprint.height_tiles)
            occupied[signature] = OccupiedFootprintV1(
                float(position[0]),
                float(position[1]),
                footprint.width_tiles,
                footprint.height_tiles,
                (f"building:{resolved_card_id}:{int(item.get('objectIndex', item.get('slot', -1)))}"),
            )
        for pending in self._pending:
            if pending.status != "queued":
                continue
            spec = self.card_specs.get(pending.effective_card_id)
            if spec is None or spec.kind.value != "building":
                continue
            placement_profile = building_placement_profile(pending.effective_card_id)
            if (
                placement_profile is not None
                and placement_profile.deploy_target_ready
                and not placement_profile.stationary_collision_rectangle
            ):
                continue
            form = "evolution" if int(pending.form_code) == 1 else "base"
            footprint = native_building_footprint(spec, form=form)
            if footprint is None:
                unknown.add(pending.effective_card_id)
                continue
            center_x, center_y = self._world_target(pending.action)
            signature = (center_x, center_y, footprint.width_tiles, footprint.height_tiles)
            occupied[signature] = OccupiedFootprintV1(
                float(center_x),
                float(center_y),
                footprint.width_tiles,
                footprint.height_tiles,
                (f"pending-building:{pending.effective_card_id}:{pending.action.action_id}"),
            )
        return (tuple(occupied[key] for key in sorted(occupied)), tuple(sorted(unknown)))

    def _cached_placement_entry(
        self,
        *,
        card_id: int,
        owner: int,
        form: str,
        destroyed_enemy_princess_lanes: Sequence[str],
        active_tower_footprints: Sequence[OccupiedFootprintV1],
        occupied_building_footprints: Sequence[OccupiedFootprintV1],
        unknown_occupied_building_ids: Sequence[int],
    ) -> Mapping[str, Any]:
        lanes = tuple(sorted(set(destroyed_enemy_princess_lanes)))
        placement_profile = building_placement_profile(card_id)
        is_stationary_building = bool(
            self.card_specs[card_id].kind.value == "building"
            and placement_profile is not None
            and placement_profile.stationary_collision_rectangle
        )
        uses_tower_rectangles = uses_entity_deployment_center(self.card_specs[card_id])
        tower_signature = (
            tuple(
                (item.center_x_units, item.center_y_units, item.width_tiles, item.height_tiles)
                for item in active_tower_footprints
            )
            if uses_tower_rectangles
            else ()
        )
        occupied_signature = (
            tuple(
                (item.center_x_units, item.center_y_units, item.width_tiles, item.height_tiles)
                for item in occupied_building_footprints
            )
            if is_stationary_building
            else ()
        )
        unknown_signature = tuple(sorted(set(unknown_occupied_building_ids))) if is_stationary_building else ()
        key = (card_id, owner, form, lanes, tower_signature, occupied_signature, unknown_signature)
        cached = self._placement_mask_cache.get(key)
        if cached is not None:
            return cached
        artifact = card_placement_mask(
            self.card_specs[card_id],
            owner,
            ruleset_id=self.ruleset_id,
            form=form,
            destroyed_enemy_princess_lanes=lanes,
            active_tower_footprints=active_tower_footprints,
            occupied_building_footprints=occupied_building_footprints,
        )
        rows = artifact.rows
        accuracy = artifact.accuracy.value
        reasons = artifact.reasons
        mask_id = artifact.mask_id
        if is_stationary_building and unknown_signature:
            rows = tuple(tuple(False for _ in range(18)) for _ in range(32))
            accuracy = "blocked"
            reasons = tuple(
                sorted(
                    {
                        *artifact.reasons,
                        "unverified_live_building_obstacle:" + ",".join(str(value) for value in unknown_signature),
                    }
                )
            )
            mask_id = content_hash(
                {"base_mask_id": artifact.mask_id, "rows": rows, "accuracy": accuracy, "reasons": reasons}
            )
        entry = frozen_mapping(
            {
                "card_id": card_id,
                "shape": (32, 18),
                "row_major": rows,
                "placement_rule": artifact.rule.value,
                "accuracy": accuracy,
                "mask_id": mask_id,
                "algorithm": artifact.algorithm,
                "collision_radius_units": artifact.collision_radius_units,
                "footprint_width_tiles": artifact.footprint_width_tiles,
                "footprint_height_tiles": artifact.footprint_height_tiles,
                "model_subcell_offset": artifact.model_subcell_offset,
                "form": artifact.form,
                "reasons": reasons,
            }
        )
        if len(self._placement_mask_cache) >= 4096:
            self._placement_mask_cache.clear()
        self._placement_mask_cache[key] = entry
        return entry

    def _cached_runtime_placement_entry(
        self,
        base: Mapping[str, Any],
        *,
        visible_card_id: int,
        effective_card_id: int,
        native_effective_card_id: int,
        effective_cost: float,
        form_code: int,
        native_form_code: int,
    ) -> Mapping[str, Any]:
        key = (
            id(base),
            visible_card_id,
            effective_card_id,
            native_effective_card_id,
            effective_cost,
            form_code,
            native_form_code,
        )
        cached = self._runtime_placement_cache.get(key)
        if cached is not None and cached[0] is base:
            return cached[1]
        result = frozen_mapping(
            {
                **dict(base),
                "visible_card_id": visible_card_id,
                "effective_card_id": effective_card_id,
                "native_effective_card_id": native_effective_card_id,
                "effective_cost": effective_cost,
                "form_code": form_code,
                "native_form_code": native_form_code,
            }
        )
        if len(self._runtime_placement_cache) >= 4096:
            self._runtime_placement_cache.clear()
        # Retaining the base object makes the identity key immune to id reuse.
        self._runtime_placement_cache[key] = (base, result)
        return result

    def action_mask(self, owner: int, raw: Mapping[str, Any] | None = None) -> ActionMaskV1:
        source = raw or self.raw_observation
        deadline = self._decision_deadline_native.get(owner, int(source.get("tick", 0)))
        identity = self._rich_identity(source)
        pending_cards = tuple(item for item in self._pending if item.status == "queued" and item.action.owner == owner)
        pending_abilities = tuple(
            item
            for item in self._pending_abilities
            if item.status in _ABILITY_PENDING_STATUSES and item.action.owner == owner
        )
        pending_signature = (
            tuple(
                (item.action.action_id, item.hand_slot, item.card_id, item.expected_native_tick, item.cost)
                for item in pending_cards
            ),
            tuple(
                (item.action.action_id, item.action.source_entity, item.expected_native_tick, item.cost_raw)
                for item in pending_abilities
            ),
        )
        cache_key = (*identity, int(owner), int(deadline), pending_signature)
        cached = self._action_mask_cache.get(cache_key)
        if cached is not None:
            return cached
        player = _player(source, owner)
        elixir_raw = int(player.get("elixirRaw", 0))
        reserved_elixir_raw = sum(item.cost * 10_000 for item in pending_cards) + sum(
            item.cost_raw for item in pending_abilities
        )
        effective_elixir_raw = max(0, elixir_raw - reserved_elixir_raw)
        if int(source.get("tick", 0)) < deadline:
            result = ActionMaskV1(
                kinds={
                    ActionKind.WAIT.value: True,
                    ActionKind.WAIT_UNTIL_AFFORDABLE.value: False,
                    ActionKind.PLAY_CARD.value: False,
                    ActionKind.ACTIVATE_ABILITY.value: False,
                },
                reasons={
                    "decision_ready": False,
                    "next_decision_tick": self._canonical_tick(deadline),
                    "raw_elixir": elixir_raw / 10_000.0,
                    "reserved_elixir": reserved_elixir_raw / 10_000.0,
                    "effective_elixir": effective_elixir_raw / 10_000.0,
                    "pending_action_count": (len(pending_cards) + len(pending_abilities)),
                },
            )
            if len(self._action_mask_cache) >= _ACTION_MASK_CACHE_LIMIT:
                self._action_mask_cache.clear()
            self._action_mask_cache[cache_key] = result
            return result
        rich_context = self._rich_context(source)
        form_aliases = self._native_form_aliases(rich_context)
        towers = self._extract_towers(source)
        lanes = self._destroyed_enemy_lanes(owner, towers)
        active_tower_footprints = self._active_tower_footprints(towers)
        (occupied_building_footprints, unknown_occupied_building_ids) = self._occupied_building_footprints(
            source, rich_context
        )
        hand_by_index = {int(card["handIndex"]): card for card in player.get("hand", ())}
        pending_hand_slots = {item.hand_slot for item in pending_cards}
        slots: list[bool] = []
        masks: dict[str, Any] = {}
        reasons: dict[str, Any] = {}
        has_unaffordable = False
        for slot in range(4):
            card = hand_by_index.get(slot)
            legal = False
            if card is None:
                reasons[str(slot)] = "not_in_hand"
            elif slot in pending_hand_slots:
                reasons[str(slot)] = "pending_execution"
            elif card.get("cardParameter") is None or card.get("cost") is None:
                reasons[str(slot)] = "native_selection_unavailable"
            elif int(card["cost"]) * 10_000 > effective_elixir_raw:
                reasons[str(slot)] = "insufficient_elixir"
                has_unaffordable = True
            else:
                legal = True
                reasons[str(slot)] = "legal"
            if card is not None:
                runtime = self._hand_runtime_contract(card, form_aliases=form_aliases, owner=owner)
                visible_card_id = int(runtime["visible_card_id"])
                effective_card_id = int(runtime["effective_card_id"])
                spec = self.card_specs.get(effective_card_id)
                if spec is None:
                    legal = False
                    reasons[str(slot)] = "card_spec_unavailable"
                    rows = tuple(tuple(False for _ in range(18)) for _ in range(32))
                    masks[str(slot)] = {
                        "card_id": effective_card_id,
                        "visible_card_id": visible_card_id,
                        "effective_card_id": effective_card_id,
                        "native_effective_card_id": int(runtime["native_effective_card_id"]),
                        "effective_cost": float(runtime["effective_cost"]),
                        "form_code": int(runtime["form_code"]),
                        "native_form_code": int(runtime["native_form_code"]),
                        "shape": [32, 18],
                        "row_major": [list(row) for row in rows],
                        "placement_rule": "unknown",
                        "accuracy": "blocked",
                        "reasons": ["card_spec_unavailable"],
                    }
                    slots.append(legal)
                    continue
                form = "evolution" if int(runtime["form_code"]) == 1 else "base"
                entry = self._cached_placement_entry(
                    card_id=effective_card_id,
                    owner=owner,
                    form=form,
                    destroyed_enemy_princess_lanes=lanes,
                    active_tower_footprints=active_tower_footprints,
                    occupied_building_footprints=(occupied_building_footprints),
                    unknown_occupied_building_ids=(unknown_occupied_building_ids),
                )
                entry = self._cached_runtime_placement_entry(
                    entry,
                    visible_card_id=visible_card_id,
                    effective_card_id=effective_card_id,
                    native_effective_card_id=int(runtime["native_effective_card_id"]),
                    effective_cost=float(runtime["effective_cost"]),
                    form_code=int(runtime["form_code"]),
                    native_form_code=int(runtime["native_form_code"]),
                )
                if not any(any(row) for row in entry["row_major"]):
                    legal = False
                    reasons[str(slot)] = "no_legal_placement"
                masks[str(slot)] = entry
            slots.append(legal)
        ability_candidates, ability_reasons = self._ability_action_candidates(owner, source, rich_context=rich_context)
        pending_ability_sources = {item.action.source_entity for item in pending_abilities}
        ability_candidates = {
            source_entity: candidate
            for source_entity, candidate in ability_candidates.items()
            if source_entity not in pending_ability_sources
            and candidate[0].elixir_cost is not None
            and int(round(float(candidate[0].elixir_cost) * 10_000)) <= effective_elixir_raw
        }
        if pending_ability_sources:
            ability_reasons = {
                **dict(ability_reasons),
                "pending_sources": tuple(
                    sorted(int(source_entity) for source_entity in pending_ability_sources if source_entity is not None)
                ),
            }
        result = ActionMaskV1(
            kinds={
                ActionKind.WAIT.value: True,
                ActionKind.WAIT_UNTIL_AFFORDABLE.value: has_unaffordable,
                ActionKind.PLAY_CARD.value: any(slots),
                ActionKind.ACTIVATE_ABILITY.value: bool(ability_candidates),
            },
            hand_slots=tuple(slots),
            placement_masks=masks,
            ability_sources=tuple(sorted(ability_candidates)),
            target_entities=(),
            reasons={
                "decision_ready": True,
                "slots": reasons,
                "ability": ability_reasons,
                "raw_elixir": elixir_raw / 10_000.0,
                "reserved_elixir": reserved_elixir_raw / 10_000.0,
                "effective_elixir": effective_elixir_raw / 10_000.0,
                "pending_action_count": (len(pending_cards) + len(pending_abilities)),
                "territory_source": "v15 standard rules + native-probed terrain",
                "post_tower_pockets": "conservative captured-command bounds",
                "building_placement": ("native tile footprint + anchor lattice + live rectangles"),
            },
        )
        if len(self._action_mask_cache) >= _ACTION_MASK_CACHE_LIMIT:
            self._action_mask_cache.clear()
        self._action_mask_cache[cache_key] = result
        return result

    def _validate_action(self, action: ActionV1, raw: Mapping[str, Any], *, mask: ActionMaskV1 | None = None) -> None:
        mask = mask or self.action_mask(action.owner, raw)
        if action.kind in {ActionKind.PLAY_CARD, ActionKind.ACTIVATE_ABILITY} and action.execute_offset_ticks > 0xFFFF:
            raise BattleEnvError("native action execute offset exceeds 65535 ticks")
        if not bool(mask.kinds.get(action.kind.value, False)):
            raise BattleEnvError(f"{action.kind.value} is masked for owner {action.owner}")
        if action.kind == ActionKind.WAIT:
            return
        if action.kind == ActionKind.WAIT_UNTIL_AFFORDABLE:
            assert action.hand_slot is not None
            player = _player(raw, action.owner)
            if not any(
                int(item["handIndex"]) == action.hand_slot and item.get("cost") is not None
                for item in player.get("hand", ())
            ):
                raise BattleEnvError(f"wait-until-affordable slot {action.hand_slot} is unavailable")
            return
        if action.kind == ActionKind.ACTIVATE_ABILITY:
            if action.target_kind != TargetKind.NONE:
                raise BattleEnvError("native v15 ct=2 ability activation has no target payload")
            assert action.source_entity is not None
            candidates, _reasons = self._ability_action_candidates(action.owner, raw)
            candidate = candidates.get(action.source_entity)
            if candidate is None:
                raise BattleEnvError(f"ability source {action.source_entity} is not currently legal")
            state, _entity_key = candidate
            if action.ability_id is not None and action.ability_id != state.ability_id:
                raise BattleEnvError(
                    f"ability source {action.source_entity} exposes {state.ability_id!r}, "
                    f"not requested {action.ability_id!r}"
                )
            return
        assert action.hand_slot is not None
        if not mask.hand_slots[action.hand_slot]:
            raise BattleEnvError(f"hand slot {action.hand_slot} is masked")
        player = _player(raw, action.owner)
        selected = next(item for item in player.get("hand", ()) if int(item["handIndex"]) == action.hand_slot)
        if action.card_id is not None and int(selected["cardId"]) != action.card_id:
            raise BattleEnvError(
                f"hand slot {action.hand_slot} contains {selected['cardId']}, not requested card {action.card_id}"
            )
        if action.target_kind not in {TargetKind.GRID, TargetKind.AREA_CENTER, TargetKind.DIRECTION}:
            raise BattleEnvError("v15 card plays currently require a grid target")
        assert action.target_grid is not None
        entry = mask.placement_masks[str(action.hand_slot)]
        x, y = action.target_grid
        replay_compatibility = bool(getattr(self, "_replay_render_compatibility", False))
        if not replay_compatibility and not bool(entry["row_major"][y][x]):
            raise BattleEnvError(f"grid target {(x, y)} is illegal for hand slot {action.hand_slot}")
        card_id = int(selected["cardId"])
        effective_card_id = int(entry.get("effective_card_id", card_id))
        if int(entry.get("visible_card_id", card_id)) != card_id:
            raise BattleEnvError("native placement contract changed visible card identity")
        spec = self.card_specs.get(effective_card_id)
        if spec is None:
            raise BattleEnvError(f"effective card {effective_card_id} has no bound CardSpec")
        towers = self._extract_towers(raw)
        lanes = self._destroyed_enemy_lanes(action.owner, towers)
        world_target = self._world_target(action)
        form = "evolution" if int(entry.get("form_code", 0)) == 1 else "base"
        active_tower_footprints = self._active_tower_footprints(towers)
        placement_rich_context = self._rich_context(raw)
        (occupied_building_footprints, unknown_occupied_building_ids) = self._occupied_building_footprints(
            raw, placement_rich_context
        )
        placement_profile = building_placement_profile(effective_card_id) if spec.kind.value == "building" else None
        if placement_profile is not None and placement_profile.stationary_collision_rectangle:
            model_offset = entry.get("model_subcell_offset")
            if (
                not isinstance(model_offset, Sequence)
                or isinstance(model_offset, (str, bytes))
                or len(model_offset) != 2
            ):
                raise BattleEnvError(f"building {card_id} has no exact native anchor offset")
            sign = 1.0 if action.owner == 0 else -1.0
            expected_offset = (float(model_offset[0]) * sign, float(model_offset[1]) * sign)
            actual_offset = action.subcell_offset or (0.0, 0.0)
            if not replay_compatibility and any(
                not math.isclose(float(actual), expected, abs_tol=1e-6)
                for actual, expected in zip(actual_offset, expected_offset, strict=True)
            ):
                raise BattleEnvError(
                    f"building {effective_card_id} requires native subcell offset "
                    f"{expected_offset}, got {tuple(actual_offset)}"
                )
            if not replay_compatibility and unknown_occupied_building_ids:
                raise BattleEnvError(
                    "building placement is blocked by an unverified live "
                    f"building footprint: {unknown_occupied_building_ids}"
                )
        if not replay_compatibility and not card_placement_legal_world(
            spec,
            action.owner,
            world_target,
            form=form,
            destroyed_enemy_princess_lanes=lanes,
            active_tower_footprints=active_tower_footprints,
            occupied_building_footprints=occupied_building_footprints,
        ):
            raise BattleEnvError(f"final world target {world_target} is illegal after subcell offset")

    def _validate_compound_actions(self, owner: int, actions: Sequence[ActionV1], raw: Mapping[str, Any]) -> None:
        if not 1 <= len(actions) <= 3:
            raise BattleEnvError("one decision must contain one to three actions")
        if any(action.owner != owner for action in actions):
            raise BattleEnvError("compound decision contains another owner's action")
        waits = [action for action in actions if action.kind in {ActionKind.WAIT, ActionKind.WAIT_UNTIL_AFFORDABLE}]
        if waits and len(actions) != 1:
            raise BattleEnvError("WAIT cannot be mixed with compound commands")
        mask = self.action_mask(owner, raw)
        for action in actions:
            self._validate_action(action, raw, mask=mask)
        if waits:
            return

        player = _player(raw, owner)
        hand = {int(item["handIndex"]): item for item in player.get("hand", ())}
        used_slots: set[int] = set()
        ability_count = 0
        total_cost_raw = 0
        planned_buildings: list[OccupiedFootprintV1] = []
        for action in actions:
            if action.kind == ActionKind.ACTIVATE_ABILITY:
                ability_count += 1
                assert action.source_entity is not None
                candidates, _reasons = self._ability_action_candidates(owner, raw)
                state, _key = candidates[action.source_entity]
                if state.elixir_cost is None:
                    raise BattleEnvError("compound ability has no exact native elixir cost")
                total_cost_raw += int(round(float(state.elixir_cost) * 10_000))
                continue
            if action.kind != ActionKind.PLAY_CARD:
                continue
            assert action.hand_slot is not None
            if action.hand_slot in used_slots:
                raise BattleEnvError(f"compound decision repeats hand slot {action.hand_slot}")
            used_slots.add(action.hand_slot)
            selected = hand[action.hand_slot]
            if selected.get("cost") is None:
                raise BattleEnvError("compound card has no exact native cost")
            total_cost_raw += int(selected["cost"]) * 10_000
            card_id = int(selected["cardId"])
            entry = mask.placement_masks[str(action.hand_slot)]
            effective_card_id = int(entry.get("effective_card_id", card_id))
            spec = self.card_specs[effective_card_id]
            placement_profile = building_placement_profile(effective_card_id) if spec.kind.value == "building" else None
            if placement_profile is not None and placement_profile.stationary_collision_rectangle:
                x, y = self._world_target(action)
                width = int(entry.get("footprint_width_tiles") or 0)
                height = int(entry.get("footprint_height_tiles") or 0)
                if width <= 0 or height <= 0:
                    raise BattleEnvError(f"building {effective_card_id} has no exact native footprint")
                candidate = OccupiedFootprintV1(float(x), float(y), width, height, f"compound:{action.action_id}")
                for prior in planned_buildings:
                    if building_footprints_overlap(candidate, prior):
                        raise BattleEnvError("planned compound buildings overlap")
                planned_buildings.append(candidate)
        if ability_count > 1:
            raise BattleEnvError("compound decision activates the ability twice")
        reserved_cost_raw = sum(
            item.cost * 10_000 for item in self._pending if item.status == "queued" and item.action.owner == owner
        ) + sum(
            item.cost_raw
            for item in self._pending_abilities
            if item.status in _ABILITY_PENDING_STATUSES and item.action.owner == owner
        )
        if total_cost_raw > max(0, int(player.get("elixirRaw", 0)) - reserved_cost_raw):
            raise BattleEnvError("compound decision overspends elixir")

    def _world_target(self, action: ActionV1) -> tuple[int, int]:
        assert action.target_grid is not None
        x, y = cell_to_world(action.target_grid)
        if action.subcell_offset is not None:
            x += int(round(action.subcell_offset[0] * 1000))
            y += int(round(action.subcell_offset[1] * 1000))
        return min(max(x, 0), 17_999), min(max(y, 0), 31_999)

    def _ticks_until_affordable(self, action: ActionV1, raw: Mapping[str, Any]) -> int:
        assert action.hand_slot is not None
        player = _player(raw, action.owner)
        card = next((item for item in player.get("hand", ()) if int(item["handIndex"]) == action.hand_slot), None)
        if card is None or card.get("cost") is None:
            raise BattleEnvError("wait_until_affordable source slot is unavailable")
        deficit = max(0, int(card["cost"]) * 10_000 - int(player.get("elixirRaw", 0)))
        if deficit == 0:
            return 1
        return self.mode_timeline.timeline.ticks_for_raw(int(raw["tick"]), deficit)

    def _advance_headless(
        self, before: Mapping[str, Any], ticks: int
    ) -> tuple[Mapping[str, Any], RichTelemetrySnapshot]:
        # Global early wakes can disclose private opponent transition timing.
        # Production FAIR episodes use fixed, atomic decision boundaries.
        self._last_advance_event = False
        self.native.step(ticks)
        return self._atomic_observation_pair()

    def _advance_synchronized_native_render(
        self, before: Mapping[str, Any], ticks: int
    ) -> tuple[Mapping[str, Any], RichTelemetrySnapshot | None]:
        if not self.episode_config or self.episode_config.event_driven_decisions:
            raise BattleEnvError("synchronized native-render steps require fixed decisions")
        advance = getattr(self.native, "advance_native_render", None)
        if not callable(advance):
            raise BattleEnvError("native endpoint lacks synchronized renderer advancement")
        target_tick = int(before["tick"]) + ticks
        receipt = advance(ticks)
        observed_tick = int(receipt.get("tick", -1))
        ended = bool(receipt.get("ended", False))
        if not ended and observed_tick != target_tick:
            raise RunnerError(
                "synchronized native-render decision boundary drifted: "
                f"expected {target_tick}, stopped at {observed_tick}"
            )
        after, prefetched_rich = self._atomic_observation_pair()
        after_tick = int(after.get("tick", -1))
        after_terminal = bool(after.get("finalized", after.get("finalizedEnded", after.get("ended", False))))
        if not after_terminal and after_tick != target_tick:
            raise RunnerError(
                f"synchronized native-render atomic capture drifted: expected {target_tick}, observed {after_tick}"
            )
        return after, prefetched_rich

    def step(
        self,
        actions: Mapping[int, ActionV1 | Mapping[str, Any] | Sequence[ActionV1 | Mapping[str, Any]]],
        *,
        record_trace: bool = True,
        observation_owners: Sequence[int] | None = None,
        policy_features_only: bool = False,
    ) -> tuple[dict[int, ObservationV1], dict[int, float], dict[Any, bool], dict[Any, bool], dict[Any, Any]]:
        if self._raw is None or self.episode_config is None:
            raise BattleEnvError("reset must be called before step")
        if self._terminated or self._truncated:
            raise BattleEnvError("episode is already finished; call reset")
        if observation_owners is None:
            output_owners = self.possible_agents
        else:
            output_owners = tuple(int(owner) for owner in observation_owners)
            if (
                not output_owners
                or len(set(output_owners)) != len(output_owners)
                or any(owner not in self.possible_agents for owner in output_owners)
            ):
                raise ValueError("observation_owners must contain unique owners from {0,1}")

        before_native_tick = int(self._raw["tick"])
        normalized: dict[int, tuple[ActionV1, ...]] = {}
        acting_owners: set[int] = set()
        for owner in self.possible_agents:
            value = actions.get(owner)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, Mapping)):
                supplied_group = tuple(
                    item if isinstance(item, ActionV1) else ActionV1.from_mapping(item) for item in value
                )
            elif value is None:
                supplied_group = ()
            else:
                supplied_group = (value if isinstance(value, ActionV1) else ActionV1.from_mapping(value),)
            ready = before_native_tick >= self._decision_deadline_native.get(owner, before_native_tick)
            if not ready:
                if supplied_group:
                    if any(item.kind != ActionKind.WAIT for item in supplied_group):
                        raise BattleEnvError(
                            f"owner {owner} cannot act before native tick {self._decision_deadline_native[owner]}"
                        )
                group = (
                    ActionV1.wait(
                        owner,
                        ticks=max(1, self._decision_deadline_native[owner] - before_native_tick),
                        metadata={"synthetic_until_existing_deadline": True},
                    ),
                )
            else:
                group = supplied_group or (
                    ActionV1.wait(owner, ticks=max(1, round(TICKS_PER_SECOND / self.episode_config.decision_hz))),
                )
                acting_owners.add(owner)
            if ready:
                self._validate_compound_actions(owner, group, self._raw)
            normalized[owner] = group

        before = self._raw
        diagnostic_action_masks = {
            owner: self.action_mask(owner, before)
            for owner in acting_owners
            if any(action.kind == ActionKind.PLAY_CARD for action in normalized[owner])
        }
        before_towers = self._extract_towers(before)
        now_ms = int(time.time() * 1000)
        for owner in self.possible_agents:
            if owner not in acting_owners:
                continue
            for compound_index, action in enumerate(normalized[owner]):
                if action.metadata.get("synthetic_policy_sleep") is True:
                    continue
                self._emit_event(
                    tick=self._canonical_tick(before_native_tick),
                    event_type="action_requested",
                    owner=action.owner,
                    card_id=action.card_id,
                    position=(
                        tuple(map(float, self._world_target(action))) if action.kind == ActionKind.PLAY_CARD else None
                    ),
                    data={
                        "action_id": action.action_id,
                        "kind": action.kind.value,
                        "compound_index": compound_index,
                        "compound_size": len(normalized[owner]),
                        "private_to": action.owner,
                    },
                )

        for owner in self.possible_agents:
            if owner not in acting_owners:
                continue
            delays = (
                self._ticks_until_affordable(action, before)
                if action.kind == ActionKind.WAIT_UNTIL_AFFORDABLE
                else action.next_decision_ticks
                for action in normalized[owner]
            )
            self._decision_deadline_native[owner] = before_native_tick + max(delays)
        # Wake at the first independent actor deadline.  Existing longer waits
        # remain scheduled and are not silently replaced on that wake-up.
        advance_ticks = max(1, min(self._decision_deadline_native.values()) - before_native_tick)

        for owner in self.possible_agents:
            if owner not in acting_owners:
                continue
            for compound_index, action in enumerate(normalized[owner]):
                if action.kind == ActionKind.ACTIVATE_ABILITY:
                    assert action.source_entity is not None
                    candidates, _reasons = self._ability_action_candidates(action.owner, before)
                    state, entity_key = candidates[action.source_entity]
                    runtime_identity_fields = {
                        "source_card_id": state.attributes.get("source_card_id"),
                        "controller_slot": state.attributes.get("controller_slot"),
                        "action_data_global_id": state.attributes.get("native_action_data_global_id"),
                        "button_state": state.attributes.get("button_state"),
                        "remaining_charges_raw": state.attributes.get("remaining_charges_raw"),
                        "configured_cooldown_ms": state.cooldown_ms,
                        "max_charges": state.attributes.get("max_charges"),
                    }
                    if (
                        any(
                            isinstance(value, bool) or not isinstance(value, int)
                            for value in runtime_identity_fields.values()
                        )
                        or state.remaining_cooldown_ms != 0
                    ):
                        raise BattleEnvError("legal ability source lacks an exact pre-command runtime identity")
                    execute_in = max(1, action.execute_offset_ticks or 1)
                    ability_action = AbilityAction(*entity_key)
                    ability_spec = self.ability_specs[state.ability_id]
                    expected_max_charges = int(ability_spec.charges or 0)
                    if (
                        int(runtime_identity_fields["configured_cooldown_ms"])
                        != normalized_ability_cooldown_ms(ability_spec)
                        or int(runtime_identity_fields["max_charges"]) != expected_max_charges
                    ):
                        raise BattleEnvError("legal ability source drifted from its static cooldown/charge contract")
                    try:
                        queued = self.native.queue_ability_action_at(ability_action, execute_in_ticks=execute_in)
                    except RunnerError as error:
                        if str(error) != "could not resolve one ready native champion ability source":
                            raise
                        # A live renderer can advance after the policy observation:
                        # a previously ready source may be gone at command time.
                        self._emit_event(
                            tick=self._canonical_tick(before_native_tick),
                            event_type="action_rejected",
                            owner=action.owner,
                            entity_id=action.source_entity,
                            data={
                                "action_id": action.action_id,
                                "kind": ActionKind.ACTIVATE_ABILITY.value,
                                "ability_id": state.ability_id,
                                "reason": "native_ability_source_no_longer_ready",
                                "compound_index": compound_index,
                                "compound_size": len(normalized[owner]),
                                "private_to": action.owner,
                            },
                        )
                        continue
                    expected = int(queued["executeTick"])
                    cast_ticks = int(math.ceil(float(ability_spec.cast_time_ms or 0) / TICK_MS))
                    attestation_grace_ticks = max(
                        ABILITY_ATTESTATION_MIN_GRACE_TICKS, cast_ticks + ABILITY_ATTESTATION_CAST_MARGIN_TICKS
                    )
                    self._pending_abilities.append(
                        _PendingAbilityRecord(
                            action=action,
                            ability_id=state.ability_id,
                            source_card_id=int(runtime_identity_fields["source_card_id"]),
                            source_entity_key=entity_key,
                            controller_slot=int(runtime_identity_fields["controller_slot"]),
                            action_data_global_id=int(runtime_identity_fields["action_data_global_id"]),
                            before_button_state=int(runtime_identity_fields["button_state"]),
                            before_remaining_cooldown_ms=0,
                            before_remaining_charges_raw=int(runtime_identity_fields["remaining_charges_raw"]),
                            before_configured_cooldown_ms=int(runtime_identity_fields["configured_cooldown_ms"]),
                            before_max_charges=int(runtime_identity_fields["max_charges"]),
                            requested_native_tick=before_native_tick,
                            expected_native_tick=expected,
                            attestation_deadline_native_tick=(expected + attestation_grace_ticks),
                            cost_raw=int(round(float(state.elixir_cost or 0.0) * 10_000)),
                            requested_at_ms=(action.requested_at_ms or now_ms),
                            generated_at_ms=(action.generated_at_ms or now_ms),
                            inference_end_ms=(action.generated_at_ms or now_ms),
                        )
                    )
                    self._emit_event(
                        tick=self._canonical_tick(before_native_tick),
                        event_type="ability_command_queued",
                        owner=action.owner,
                        entity_id=action.source_entity,
                        data={
                            "action_id": action.action_id,
                            "ability_id": state.ability_id,
                            "compound_index": compound_index,
                            "compound_size": len(normalized[owner]),
                            "expected_execution_tick": self._canonical_tick(expected),
                            "execute_offset_ticks": execute_in,
                            "end_to_end_matched": queued.get("endToEndMatched"),
                            "end_to_end_us": queued.get("endToEndUs"),
                            "timing": dict(action.metadata),
                            "private_to": action.owner,
                        },
                    )
                    continue
                if action.kind != ActionKind.PLAY_CARD:
                    continue
                assert action.hand_slot is not None
                player = _player(before, action.owner)
                card = next(item for item in player["hand"] if int(item["handIndex"]) == action.hand_slot)
                card_id = int(card["cardId"])
                x, y = self._world_target(action)
                execute_in = max(1, action.execute_offset_ticks or 1)
                hand_action = HandAction(action.owner, action.hand_slot, x, y)
                if action.metadata.get("native_command_age_ticks") is not None:
                    # The renderer continues during inference.  Policy offsets
                    # describe time since the decision observation, so do not
                    # restart the clock when the native queue is reached.
                    queued = self.native.queue_hand_action_at(
                        hand_action,
                        execute_tick=before_native_tick + execute_in,
                        clamp_late=True,
                    )
                else:
                    queued = self.native.queue_hand_action_at(hand_action, execute_in_ticks=execute_in)
                if bool(queued.get("terminalCanceled")):
                    self._emit_event(
                        tick=self._canonical_tick(int(queued["executeTick"])),
                        event_type="action_canceled_terminal",
                        owner=action.owner,
                        card_id=card_id,
                        data={
                            "action_id": action.action_id,
                            "compound_index": compound_index,
                            "compound_size": len(normalized[owner]),
                            "private_to": action.owner,
                        },
                    )
                    continue
                action_rich_context = self._rich_context(before)
                runtime = self._hand_runtime_contract(
                    card, form_aliases=self._native_form_aliases(action_rich_context), owner=action.owner
                )
                if (
                    int(queued["cardId"]) != int(runtime["visible_card_id"])
                    or int(queued["commandCardId"]) != int(runtime["native_effective_card_id"])
                    or float(queued["cost"]) != float(runtime["effective_cost"])
                    or int(queued["formCode"]) != int(runtime["native_form_code"])
                ):
                    raise BattleEnvError("native queued hand receipt changed the validated runtime card contract")
                expected = int(queued["executeTick"])
                placement_entry = diagnostic_action_masks[action.owner].placement_masks[str(action.hand_slot)]
                assert action.target_grid is not None
                target_grid_x, target_grid_y = action.target_grid
                diagnostic_action = replace(
                    action,
                    metadata={
                        **dict(action.metadata),
                        "rejection_diagnostic": {
                            "compound_index": compound_index,
                            "compound_size": len(normalized[owner]),
                            "target_grid": action.target_grid,
                            "target_world": (x, y),
                            "subcell_offset": action.subcell_offset,
                            "mask_id": placement_entry.get("mask_id"),
                            "placement_rule": placement_entry.get("placement_rule"),
                            "mask_accuracy": placement_entry.get("accuracy"),
                            "mask_algorithm": placement_entry.get("algorithm"),
                            "mask_reasons": placement_entry.get("reasons", ()),
                            "mask_target_legal": bool(placement_entry["row_major"][target_grid_y][target_grid_x]),
                            "source_native_hand_slot": placement_entry.get("source_native_hand_slot"),
                            "before_elixir_raw": int(player.get("elixirRaw", -1)),
                            "before_hand": _diagnostic_hand_snapshot(player),
                            "queue_receipt": {
                                "async": bool(queued.get("async")),
                                "sequence": queued.get("sequence"),
                                "execute_tick": expected,
                                "card_id": int(queued["cardId"]),
                                "command_card_id": int(queued["commandCardId"]),
                                "card_parameter": int(queued["cardParameter"]),
                                "deck_slot": int(queued["deckSlot"]),
                                "cost": int(queued["cost"]),
                                "form_code": int(queued["formCode"]),
                            },
                        },
                    },
                )
                self._pending.append(
                    _PendingRecord(
                        action=diagnostic_action,
                        card_id=card_id,
                        effective_card_id=int(runtime["effective_card_id"]),
                        native_effective_card_id=int(runtime["native_effective_card_id"]),
                        requested_native_tick=before_native_tick,
                        expected_native_tick=expected,
                        hand_slot=action.hand_slot,
                        card_parameter=int(queued["cardParameter"]),
                        deck_slot=int(queued["deckSlot"]),
                        cost=int(queued["cost"]),
                        form_code=int(runtime["form_code"]),
                        native_form_code=int(runtime["native_form_code"]),
                        form_name=str(queued["formName"]),
                        requested_at_ms=action.requested_at_ms or now_ms,
                        generated_at_ms=action.generated_at_ms or now_ms,
                        inference_end_ms=action.generated_at_ms or now_ms,
                    )
                )
                self._emit_event(
                    tick=self._canonical_tick(before_native_tick),
                    event_type="card_form_command_queued",
                    owner=action.owner,
                    card_id=card_id,
                    data={
                        "action_id": action.action_id,
                        "compound_index": compound_index,
                        "compound_size": len(normalized[owner]),
                        "card_parameter": int(queued["cardParameter"]),
                        "deck_slot": int(queued["deckSlot"]),
                        "cost": int(queued["cost"]),
                        "visible_card_id": card_id,
                        "effective_card_id": int(runtime["effective_card_id"]),
                        "native_effective_card_id": int(queued["commandCardId"]),
                        "effective_cost": float(queued["cost"]),
                        "form_code": int(queued["formCode"]),
                        "policy_form_code": int(runtime["form_code"]),
                        "native_form_code": int(runtime["native_form_code"]),
                        "form_name": str(queued["formName"]),
                        "expected_execution_tick": self._canonical_tick(expected),
                        "execute_offset_ticks": execute_in,
                        "end_to_end_matched": queued.get("endToEndMatched"),
                        "end_to_end_us": queued.get("endToEndUs"),
                        "timing": dict(action.metadata),
                        "private_to": action.owner,
                    },
                )

        mode = self.episode_config.render_mode
        prefetched_rich: RichTelemetrySnapshot | None = None
        if mode == "headless":
            after, prefetched_rich = self._advance_headless(before, advance_ticks)
        else:
            self._last_advance_event = False
            target_tick = before_native_tick + advance_ticks
            wait_timeout = max(2.0, advance_ticks / TICKS_PER_SECOND + 2.0)
            if self._synchronize_native_render_steps:
                after, prefetched_rich = self._advance_synchronized_native_render(before, advance_ticks)
            else:
                deadline = time.monotonic() + wait_timeout
                after = before
                while int(after.get("tick", -1)) < target_tick:
                    if bool(after.get("finalized", after.get("finalizedEnded", after.get("ended", False)))):
                        break
                    if time.monotonic() >= deadline:
                        raise RunnerError("native-render decision interval timed out")
                    time.sleep(0.01)
                    after = self.native.observe()

        if prefetched_rich is None:
            after, prefetched_rich = self._atomic_observation_pair()

        self._raw = dict(after)
        self._invalidate_rich_cache()
        if prefetched_rich is not None:
            self._prime_rich_cache(self._raw, prefetched_rich)
        after_rich_context = self._rich_context(after)
        native_public_card_plays = self._consume_native_public_card_plays(after_rich_context)
        self._consume_native_phase_events(after_rich_context)
        exact_native_visibility = self._consume_native_visibility_events(after_rich_context)
        self._consume_native_remaining_events(after_rich_context)
        self._consume_native_combat_events(after_rich_context)
        after_native_tick = int(after["tick"])
        if self._last_advance_event:
            self._decision_deadline_native = {owner: after_native_tick for owner in self.possible_agents}
        self._update_beliefs(before_native_tick, after_native_tick)
        resolved_pending = self._resolve_pending(
            after, int(time.time() * 1000), native_public_card_plays=native_public_card_plays
        )
        self._append_native_public_card_plays(
            native_public_card_plays,
            resolved_pending=resolved_pending,
            update_belief=True,
            form_aliases=self._native_form_aliases(after_rich_context),
        )
        self._resolve_pending_abilities(after, int(time.time() * 1000), rich_context=after_rich_context)
        self._derive_events(
            before,
            after,
            rich_context=after_rich_context,
            exact_native_visibility=exact_native_visibility,
            resolved_pending=resolved_pending,
        )
        after_towers = self._extract_towers(after)
        terminal = self._terminal(after, after_towers)
        self._terminated = terminal.ended
        self._truncated = not self._terminated and self._canonical_tick(after_native_tick) >= self.max_battle_ticks
        self._last_rewards = self._rewards(
            before_towers, after_towers, terminal, max(1, after_native_tick - before_native_tick)
        )
        if record_trace:
            self._trace.append(
                ReplayOperationV1(
                    actions=tuple(action for owner in self.possible_agents for action in normalized[owner]),
                    advance_ticks=after_native_tick - before_native_tick,
                    start_native_tick=before_native_tick,
                    end_native_tick=after_native_tick,
                    requested_advance_ticks=advance_ticks,
                )
            )

        observations = {
            owner: self.observe(ObservationTier.FAIR, owner=owner, policy_features_only=policy_features_only)
            for owner in output_owners
        }
        terminations: dict[Any, bool] = {0: self._terminated, 1: self._terminated, "__all__": self._terminated}
        truncations: dict[Any, bool] = {0: self._truncated, 1: self._truncated, "__all__": self._truncated}
        if policy_features_only:
            infos: dict[Any, Any] = {}
        else:
            infos = {
                owner: {
                    "episode_id": self.episode_id,
                    "ruleset_id": self.ruleset_id,
                    "native_tick": after_native_tick,
                    "terminal_reason": terminal.reason,
                    "pending_count": sum(
                        1 for item in self._pending if item.status == "queued" and item.action.owner == owner
                    )
                    + sum(
                        1
                        for item in self._pending_abilities
                        if item.status in _ABILITY_PENDING_STATUSES and item.action.owner == owner
                    ),
                    "decision_ready": after_native_tick >= self._decision_deadline_native[owner],
                    "next_decision_tick": self._canonical_tick(self._decision_deadline_native[owner]),
                }
                for owner in self.possible_agents
            }
            infos["__all__"] = {"trace_length": len(self._trace), "event_wake": self._last_advance_event}
        return observations, dict(self._last_rewards), terminations, truncations, infos

    def step_actor(
        self,
        actor_owner: int,
        actions: Mapping[int, ActionV1 | Mapping[str, Any] | Sequence[ActionV1 | Mapping[str, Any]]],
        *,
        record_trace: bool = True,
    ) -> tuple[ObservationV1, float, bool, bool, Mapping[str, Any]]:
        """Advance one match but materialize only the selected actor's view.

        Native dynamics, public-event tracking, rewards, and terminal state
        remain two-sided.  This method only avoids constructing the unused
        opponent ObservationV1/Raster at the model boundary.
        """

        if actor_owner not in self.possible_agents:
            raise ValueError("actor_owner must be 0 or 1")
        observations, rewards, terminated, truncated, infos = self.step(
            actions, record_trace=record_trace, observation_owners=(actor_owner,)
        )
        return (
            observations[actor_owner],
            rewards[actor_owner],
            terminated[actor_owner],
            truncated[actor_owner],
            infos[actor_owner],
        )

    def _update_beliefs(self, before_tick: int, after_tick: int) -> None:
        generated = self.mode_timeline.timeline.generated_raw(before_tick, after_tick) / 10_000.0
        for owner, (low, high) in tuple(self._belief_elixir.items()):
            self._belief_elixir[owner] = (min(10.0, low + generated), min(10.0, high + generated))

    @staticmethod
    def _event_archive_key(event: EventV1) -> str:
        if event.combat is not None and event.combat.visible_by_owner:
            visible_zero = event.combat.visible_by_owner.get("0") is True
            visible_one = event.combat.visible_by_owner.get("1") is True
            if visible_zero and visible_one:
                scope = "public"
            elif visible_zero:
                scope = "owner-0"
            elif visible_one:
                scope = "owner-1"
            else:
                scope = "oracle"
            return f"{scope}:{event.event_type}"
        if event.data.get("oracle_only") is True:
            return f"oracle:{event.event_type}"
        private_to = event.data.get("private_to")
        scope = "public" if private_to is None else f"owner-{int(private_to)}"
        return f"{scope}:{event.event_type}"

    def _archive_event(self, event: EventV1) -> None:
        key = self._event_archive_key(event)
        record = self._event_archive.setdefault(key, {"count": 0, "last_tick": event.tick})
        record["count"] += 1
        record["last_tick"] = max(record["last_tick"], event.tick)

    def _append_event(self, event: EventV1) -> None:
        if len(self._events) == self.event_limit:
            self._archive_event(self._events[0])
        self._events.append(event)

    def _emit_event(self, **fields: Any) -> None:
        """Create and retain an event through the same archive boundary."""
        self._append_event(EventV1(**fields))

    def _prime_combat_event_cursor(self, snapshot: RichTelemetrySnapshot) -> None:
        envelope = snapshot.combat_events
        self._combat_event_identity = (envelope.generation, envelope.state_epoch)
        # Reset establishes the episode baseline. Initial tower/object spawns
        # are intentionally not replayed into the post-reset event stream.
        self._combat_next_sequence = envelope.next_sequence
        self._combat_native_entity_ids.clear()
        self._combat_evolution_entities.clear()
        self._causal_group_by_entity.clear()
        self._combat_event_kind_by_sequence.clear()

    def _consume_native_public_card_plays(
        self, rich_context: _RichObservationContext
    ) -> tuple[_NativePublicCardPlay, ...]:
        """Consume exact public card plays independently of combat projection.

        Model actions already have action-executed events. Human inputs from
        the overlay enter the native queue directly, so the consume-card hook
        supplies their public plays, including spells without durable entities.
        """

        if not getattr(self.native, "public_card_play_events_from_combat_ring", False):
            return ()
        envelope = rich_context.snapshot.combat_events
        if (
            not envelope.complete
            or not envelope.hook_set_attested
            or not envelope.hook_set_installed
            or envelope.rejected_capture_count
        ):
            raise BattleEnvError("public card-play reconstruction requires a complete attested combat ring")

        identity = (envelope.generation, envelope.state_epoch)
        if self._public_card_play_event_identity != identity:
            if envelope.oldest_retained_sequence != envelope.epoch_first_sequence:
                raise BattleEnvError("native card-play history was already overwritten at episode start")
            self._public_card_play_event_identity = identity
            self._public_card_play_next_sequence = envelope.epoch_first_sequence
        expected = self._checked_ring_sequence(envelope, self._public_card_play_next_sequence, label="public card-play")

        positions: dict[int, tuple[int, tuple[float, float]]] = {}
        for event in envelope.events:
            deployment = event.deployment_context
            if deployment is None:
                continue
            priority = 0
            position: tuple[float, float] | None = None
            if event.kind == "projectile_spawn" and event.destination_after is not None:
                priority = 2
                position = tuple(map(float, event.destination_after))
            elif event.kind == "spawn" and event.target.position is not None:
                priority = 1
                position = tuple(map(float, event.target.position))
            if position is None:
                continue
            prior = positions.get(deployment.deployment_sequence)
            if prior is None or priority > prior[0]:
                positions[deployment.deployment_sequence] = (priority, position)

        plays: list[_NativePublicCardPlay] = []
        for event in envelope.events:
            if event.sequence < expected or event.kind != "card_play":
                continue
            if event.tick < 0 or event.deployment_context is None:
                raise BattleEnvError("native card_play lacks a valid tick/deployment context")
            deployment = event.deployment_context
            positioned = positions.get(deployment.deployment_sequence)
            plays.append(
                _NativePublicCardPlay(
                    sequence=event.sequence,
                    state_epoch=event.state_epoch,
                    tick=event.tick,
                    deployment=deployment,
                    position=positioned[1] if positioned is not None else None,
                )
            )
        self._public_card_play_next_sequence = envelope.next_sequence
        return tuple(plays)

    def _append_native_public_card_plays(
        self,
        plays: Sequence[_NativePublicCardPlay],
        *,
        resolved_pending: Sequence[_PendingRecord],
        update_belief: bool,
        form_aliases: Mapping[int, int] | None = None,
    ) -> None:
        """Append unmatched native plays without leaking deck-slot internals."""

        matched_pending: set[int] = set()
        for play in plays:
            deployment = play.deployment
            owner = int(deployment.owner)
            card_id = int(deployment.played_card_global_id)
            # The public receipt identifies the selected deck card separately
            # from Mirror's effective source.  Both identities must belong to
            # the acting owner's episode contract.
            self._episode_form_mask(owner, card_id)
            native_effective_card_id = int(deployment.effective_card_global_id or card_id)
            native_form_code = int(deployment.form_code)
            effective_card_id, form_code = self._policy_effective_identity(
                native_effective_card_id, native_form_code, form_aliases=form_aliases, owner=owner
            )
            nearby = [
                item
                for item in resolved_pending
                if id(item) not in matched_pending
                and item.action.owner == owner
                and item.card_id == card_id
                and abs(item.expected_native_tick - play.tick) <= TICKS_PER_SECOND
            ]
            if nearby:
                matched = min(
                    nearby, key=lambda item: abs(item.expected_native_tick - play.tick)
                )
                if matched.status != "executed":
                    raise BattleEnvError("native card_play contradicts a rejected local action")
                # The private queue can expose a runtime form global with
                # native form 0 while the public deployment ring exposes the
                # same Hero as its base global with policy form 2.  The queue
                # receipt was already checked byte-for-byte; join the public
                # receipt through the exact canonical alias contract.
                if (
                    matched.effective_card_id != effective_card_id
                    or matched.cost != int(deployment.cost)
                    or matched.form_code != form_code
                ):
                    raise BattleEnvError("native card_play contradicts the queued effective card contract")
                matched_pending.add(id(matched))
                continue

            spec = self.card_specs.get(card_id)
            effective_spec = self.card_specs.get(effective_card_id)
            if spec is None or effective_spec is None:
                raise BattleEnvError(f"native public play has no deterministic CardSpec: {card_id}")
            if card_id == MIRROR_CARD_ID and effective_card_id == card_id:
                raise BattleEnvError("native public Mirror play has no effective card identity")
            self._revealed[owner].add(card_id)
            if update_belief:
                low, high = self._belief_elixir[owner]
                cost = float(deployment.cost)
                self._belief_elixir[owner] = (max(0.0, low - cost), max(0.0, high - cost))
            self._emit_event(
                tick=self._canonical_tick(play.tick),
                event_type="action_executed",
                owner=owner,
                card_id=card_id,
                position=play.position,
                data={
                    "native_sequence": play.sequence,
                    "native_event_id": (f"{play.state_epoch}:{play.sequence}"),
                    "visible_card_id": card_id,
                    "effective_card_id": effective_card_id,
                    "native_effective_card_id": native_effective_card_id,
                    "effective_cost": float(deployment.cost),
                    "form_code": form_code,
                    "native_form_code": native_form_code,
                    "deployment_sequence": (deployment.deployment_sequence),
                    "played_form": deployment.form_name,
                    "source": "native_card_play",
                },
            )

    def _prime_phase_event_cursor(self, snapshot: RichTelemetrySnapshot) -> None:
        envelope = snapshot.phase_runtime
        self._phase_events_by_entity.clear()
        if envelope is None:
            self._phase_event_identity = None
            self._phase_next_sequence = None
            return
        if envelope.window.rejected_count:
            raise BattleEnvError("native phase runtime rejected capture facts at episode reset")
        self._phase_event_identity = (envelope.generation, envelope.state_epoch)
        # Reset is the explicit consumer baseline; pre-reset retained records
        # are neither replayed nor treated as a gap.
        self._phase_next_sequence = envelope.window.next_sequence

    def _prime_visibility_event_cursor(self, snapshot: RichTelemetrySnapshot) -> None:
        envelope = snapshot.visibility_runtime
        if envelope is None:
            self._visibility_event_identity = None
            self._visibility_next_sequence = None
            return
        if envelope.rejected_count:
            raise BattleEnvError("native visibility runtime rejected capture facts at episode reset")
        self._visibility_event_identity = (envelope.generation, envelope.state_epoch)
        self._visibility_next_sequence = envelope.next_sequence

    @staticmethod
    def _is_initial_tower_aggro_bootstrap_rejection(envelope: Any) -> bool:
        """Recognize the engine's pre-consumer tower bootstrap calls.

        The stock native renderer initializes every tower's attack component
        before BattleEnv can establish its event cursor.  The hook records the
        six exact null-to-tower target transitions and also increments the
        cumulative rejection counter once per earlier, not-yet-bound call.
        A cold headless match followed by the one-tick renderer-alignment step
        repeats that same six-transition pass once after rebuilding the world.
        Current object state is captured atomically, so only this fully typed
        tick-zero pattern, with one or two identical passes, is safe to treat
        as an initial consumer baseline.
        """

        if (
            envelope.rejected_count != 6
            or envelope.complete
            or not 0 <= envelope.observation_tick <= FIRST_PLAYABLE_TICK
        ):
            return False
        bootstrap = tuple(event for event in envelope.events if event.kind == "tower_aggro_acquire" and event.tick == 0)
        if len(bootstrap) not in {6, 12}:
            return False
        signatures: dict[tuple[int, int], int] = {}
        for event in bootstrap:
            entity = event.entity
            target = event.target_after
            if (
                event.committed
                or not event.complete_context
                or not entity.validated
                or not entity.present
                or entity.native_object_id is None
                or entity.object_kind != 5
                or entity.card_id != -1
                or entity.owner not in (0, 1)
                or not event.target_before.validated
                or event.target_before.present
                or not target.validated
                or not target.present
                or target.native_object_id is None
                or target.object_kind != 5
                or target.card_id != -1
                or target.owner not in (0, 1)
                or target.owner == entity.owner
                or event.target != target
            ):
                return False
            signature = (entity.native_object_id, target.native_object_id)
            signatures[signature] = signatures.get(signature, 0) + 1
        repeats = len(bootstrap) // 6
        return len(signatures) == 6 and all(count == repeats for count in signatures.values())

    def _prime_remaining_event_cursor(self, snapshot: RichTelemetrySnapshot) -> None:
        envelope = snapshot.remaining_runtime
        if envelope is None:
            self._remaining_event_identity = None
            self._remaining_next_sequence = None
            self._remaining_rejected_count_baseline = None
            return
        bootstrap_rejection = bool(
            self._allow_initial_remaining_runtime_rejections
            and self._is_initial_tower_aggro_bootstrap_rejection(envelope)
        )
        if envelope.rejected_count and not bootstrap_rejection:
            raise BattleEnvError("native remaining runtime rejected capture facts at episode reset")
        self._remaining_event_identity = (envelope.generation, envelope.state_epoch)
        self._remaining_next_sequence = envelope.next_sequence
        self._remaining_rejected_count_baseline = envelope.rejected_count

    def _resume_native_ring_cursor(
        self,
        *,
        expected: int,
        epoch_first_sequence: int,
        oldest_retained_sequence: int,
        overflow_count: int,
        label: str,
    ) -> int:
        """Return the first event to consume without hiding a real ring gap.

        Resident matches share process-local telemetry rings.  Switching the
        single native execution lane to a different slot starts a clean ring
        epoch before that slot advances.  Ring sequences remain process-global,
        while match generation/stateEpoch remain slot-local, so the clean
        rebind cannot be detected from the match identity alone.

        A clean resident boundary is unambiguous: no event was dropped,
        ``oldest`` still equals ``epochFirst``, and overflow is zero.  Any other
        cursor gap remains a hard error.
        """

        if expected >= oldest_retained_sequence:
            return expected
        clean_resident_rebind = (
            bool(getattr(self.native, "resident_event_rings_rebind_on_switch", False))
            and oldest_retained_sequence == epoch_first_sequence
            and overflow_count == 0
        )
        if clean_resident_rebind:
            return epoch_first_sequence
        raise BattleEnvError(f"native {label} event ring overwrote unseen sequence records")

    def _consume_native_phase_events(self, rich_context: _RichObservationContext) -> None:
        self._consume_native_runtime_events(rich_context, "phase")

    def _consume_native_visibility_events(self, rich_context: _RichObservationContext) -> bool:
        return self._consume_native_runtime_events(rich_context, "visibility")

    def _append_native_visibility_event(
        self, event: RichVisibilityRuntimeEvent, rich_context: _RichObservationContext, state_epoch: int
    ) -> None:
        subject_id = self._combat_entity_id(event.subject, rich_context)
        if subject_id is None:
            raise BattleEnvError("present visibility-runtime subject lacks canonical identity")
        became_invisible = event.kind.value == "became_invisible"
        self._emit_event(
            tick=self._canonical_tick(event.tick),
            event_type="visibility_change",
            owner=(event.subject.owner if event.subject.owner in (0, 1) else None),
            entity_id=subject_id,
            card_id=(
                event.subject.card_id if event.subject.card_id is not None and event.subject.card_id > 0 else None
            ),
            position=event.subject.position,
            data={
                "native_sequence": event.sequence,
                "native_event_id": (f"visibility:{state_epoch}:{event.sequence}"),
                # Owner-relative visibility remains unavailable. Exact
                # phase edges are retained for oracle consumers only.
                "oracle_only": True,
                "from": ("visible" if became_invisible else "hidden"),
                "to": ("hidden" if became_invisible else "visible"),
                "native_invisible_count": (event.invisible_count_after),
                "native_invisible_count_before": (event.invisible_count_before),
                "buff_global_id": event.buff_global_id,
                "hook_offset": event.hook_offset,
                "caller_offset": event.caller_offset,
                "scope": event.scope,
            },
        )

    def _consume_native_remaining_events(self, rich_context: _RichObservationContext) -> None:
        self._consume_native_runtime_events(rich_context, "remaining")

    def _consume_native_runtime_events(self, rich_context: _RichObservationContext, kind: str) -> bool:
        """Consume one native ring with shared gap and capability checks.

        Remaining-runtime bootstrap rejections and phase history retain their own
        epoch baselines; all rings advance only after every unseen event is handled.
        """
        envelope = getattr(rich_context.snapshot, f"{kind}_runtime")
        if envelope is None:
            return False
        window = envelope.window if kind == "phase" else envelope
        identity = (envelope.generation, envelope.state_epoch)
        identity_attr = f"_{kind}_event_identity"
        sequence_attr = f"_{kind}_next_sequence"
        changed_epoch = getattr(self, identity_attr) != identity
        rejected_baseline = 0
        if kind == "remaining":
            if changed_epoch:
                bootstrap = (
                    self._allow_initial_remaining_runtime_rejections
                    and self._is_initial_tower_aggro_bootstrap_rejection(envelope)
                )
                if envelope.rejected_count and not bootstrap:
                    raise BattleEnvError("native remaining runtime rejected capture facts")
                self._remaining_rejected_count_baseline = envelope.rejected_count
            rejected_baseline = self._remaining_rejected_count_baseline or 0
            if window.rejected_count < rejected_baseline:
                raise BattleEnvError("native remaining runtime rejected count regressed")
        if window.rejected_count > rejected_baseline:
            raise BattleEnvError(f"native {kind} runtime rejected capture facts")
        if changed_epoch:
            setattr(self, identity_attr, identity)
            setattr(self, sequence_attr, window.epoch_first_sequence)
            if kind == "phase":
                self._phase_events_by_entity.clear()
        expected = self._checked_ring_sequence(window, getattr(self, sequence_attr), label=kind)
        installed = envelope.transition_hook_set_installed if kind == "visibility" else envelope.hook_set_installed
        if not installed or envelope.capability_status == "unavailable":
            if envelope.events:
                raise BattleEnvError(f"unavailable native {kind} capability exposed records")
            setattr(self, sequence_attr, window.next_sequence)
            return False
        allowed_bootstrap = (
            kind == "remaining"
            and rejected_baseline
            and self._allow_initial_remaining_runtime_rejections
            and window.rejected_count == rejected_baseline
        )
        if not window.complete and not allowed_bootstrap:
            raise BattleEnvError(f"native {kind} runtime capture is incomplete")
        for event in envelope.events:
            if event.sequence < expected:
                continue
            if kind == "phase":
                canonical = replace(event, tick=self._canonical_tick(event.tick))
                self._phase_events_by_entity.setdefault(canonical.entity_key, []).append(canonical)
            elif kind == "visibility":
                self._append_native_visibility_event(event, rich_context, envelope.state_epoch)
            else:
                self._append_native_remaining_event(event, rich_context, envelope.state_epoch)
        setattr(self, sequence_attr, window.next_sequence)
        return True

    def _checked_ring_sequence(self, window: Any, expected: int | None, *, label: str) -> int:
        expected = self._resume_native_ring_cursor(
            expected=window.epoch_first_sequence if expected is None else expected,
            epoch_first_sequence=window.epoch_first_sequence,
            oldest_retained_sequence=window.oldest_retained_sequence,
            overflow_count=window.overflow_count,
            label=label,
        )
        if expected > window.next_sequence:
            noun = "" if label == "public card-play" else " event"
            raise BattleEnvError(f"native {label}{noun} sequence regressed")
        return expected

    def _append_native_remaining_event(
        self, event: RichRemainingRuntimeEvent, rich_context: _RichObservationContext, state_epoch: int
    ) -> None:
        identities: dict[str, int] = {}
        for name, fact in (
            ("entity", event.entity),
            ("source_entity", event.source),
            ("target_entity", event.target),
            ("target_before", event.target_before),
            ("target_after", event.target_after),
        ):
            canonical = self._combat_entity_id(fact, rich_context)
            if fact.present:
                if canonical is None:
                    raise BattleEnvError("present remaining-runtime fact lacks canonical identity")
                identities[name] = canonical
        subject = event.entity if event.entity.present else event.source if event.source.present else event.target
        subject_id = self._combat_entity_id(subject, rich_context)
        if event.kind == "tower_activate":
            king_identity = next(
                (
                    (entity_id, owner)
                    for entity_id, owner, kind, _x, _y in TOWER_LAYOUT
                    if entity_id == subject_id and kind == "king"
                ),
                None,
            )
            if king_identity is None or not event.entity.present or event.entity.owner != king_identity[1]:
                raise BattleEnvError("native tower_activate did not bind to its fixed king tower")
            self._activated_king_tower_ids.add(king_identity[0])
            self._ordinary_tower_cache.clear()
        data: dict[str, Any] = {
            "native_sequence": event.sequence,
            "native_event_id": (f"remaining:{state_epoch}:{event.sequence}"),
            # Owner-relative public/targetable predicates remain unavailable.
            # Publishing these causal facts to FAIR observations would leak
            # hidden identity, so the production boundary is oracle-only.
            "oracle_only": True,
            "hook_offset": event.hook_offset,
            "resource_cause": event.resource_cause,
            "transform_kind": event.transform_kind,
            "result": event.result,
            "option": event.option,
            "committed": event.committed,
            "reset_target": event.reset_target,
            **identities,
        }
        optional_values = {
            "caller_offset": event.caller_offset,
            "object_kind": event.object_kind,
            "runtime_vtable_offset": event.runtime_vtable_offset,
            "data_before_global_id": event.data_before_global_id,
            "data_after_global_id": event.data_after_global_id,
            "expected_character_data_global_id": (event.expected_character_data_global_id),
            "expected_projectile_data_global_id": (event.expected_projectile_data_global_id),
            "configured_data_global_id": event.configured_data_global_id,
            "resource_pre_fixed": event.resource_pre_fixed,
            "resource_post_fixed": event.resource_post_fixed,
            "resource_actual_delta_fixed": (event.resource_actual_delta_fixed),
            "amount_argument": event.amount_argument,
            "configured_amount_argument": (event.configured_amount_argument),
            "resource_owner": event.resource_owner,
            "area_remaining_life_ms": event.area_remaining_life_ms,
        }
        data.update({name: value for name, value in optional_values.items() if value is not None})
        owner = event.resource_owner if event.kind == "resource_delta" else subject.owner
        self._emit_event(
            tick=self._canonical_tick(event.tick),
            event_type=f"runtime_{event.kind}",
            owner=owner if owner in (0, 1) else None,
            entity_id=subject_id,
            card_id=(
                subject.card_id if subject.present and subject.card_id is not None and subject.card_id > 0 else None
            ),
            position=subject.position if subject.present else None,
            data=data,
        )

    def _combat_entity_id(self, fact: RichCombatEntityFact, rich_context: _RichObservationContext) -> int | None:
        if not fact.present or fact.native_object_id is None or fact.entity_key is None:
            return None
        known = self._combat_native_entity_ids.get(fact.native_object_id)
        if known is not None:
            return known
        canonical = rich_context.canonical_ids.get(fact.entity_key)
        if canonical is None:
            for entity_id, raw in self._previous_entities.items():
                if raw.get("nativeObjectId") == fact.native_object_id:
                    canonical = entity_id
                    break
                key = (int(raw.get("owner", -1)), int(raw.get("objectIndex", -1)), int(raw.get("secondaryIndex", -1)))
                if key == fact.entity_key:
                    canonical = entity_id
                    break
        if canonical is None and fact.position is not None:
            tower = TOWER_BY_POSITION.get(fact.position)
            if (
                tower is not None
                and tower[1] == fact.owner
                and fact.object_kind != 4
                and (fact.card_id is None or fact.card_id <= 0)
            ):
                canonical = int(tower[0])
        if canonical is None:
            _, object_index, secondary_index = fact.entity_key
            canonical = (
                10_000 + object_index * 1_000 + max(0, secondary_index)
                if object_index >= 0
                else 2_000_000_000 + fact.native_object_id
            )
        conflicting_native_id = next(
            (
                native_id
                for native_id, entity_id in self._combat_native_entity_ids.items()
                if entity_id == canonical and native_id != fact.native_object_id
            ),
            None,
        )
        if conflicting_native_id is not None:
            raise BattleEnvError("native combat identities collide on one canonical entity id")
        self._combat_native_entity_ids[fact.native_object_id] = canonical
        return canonical

    def _normal_mode_causal_source_card_id(
        self, event: RichCombatEventTelemetry, *, parent_entity_id: int | None
    ) -> int | None:
        """Resolve a public playable source card without exposing form handles."""

        candidates: list[int | None] = []
        deployment = event.deployment_context
        if deployment is not None:
            candidates.append(
                deployment.effective_card_global_id
                if deployment.played_card_global_id == MIRROR_CARD_ID
                else deployment.played_card_global_id
            )
        if parent_entity_id is not None:
            parent = self._causal_group_by_entity.get(parent_entity_id)
            if parent is not None:
                candidates.append(parent.source_card_id)
        candidates.extend(
            (event.immediate_source.card_id, event.source.card_id, event.target.card_id, event.projectile.card_id)
        )
        for value in candidates:
            if value is None:
                continue
            card_id = int(value)
            spec = self.card_specs.get(card_id)
            if spec is not None and (
                spec.attributes.get("NotVisible") is not True and spec.attributes.get("NotInUse") is not True
            ):
                return card_id
        return None

    def _causal_parent_entity_id(
        self, event: RichCombatEventTelemetry, rich_context: _RichObservationContext
    ) -> int | None:
        subject_ids = {
            entity_id
            for fact in (event.target, event.projectile)
            if (entity_id := self._combat_entity_id(fact, rich_context)) is not None
        }
        for fact in (event.immediate_source, event.source, event.related):
            entity_id = self._combat_entity_id(fact, rich_context)
            if entity_id is not None and entity_id not in subject_ids:
                return entity_id
        return None

    def _bind_causal_group(self, entity_id: int, reference: CausalGroupRefV1) -> None:
        prior = self._causal_group_by_entity.get(entity_id)
        if prior is not None and prior != reference:
            raise BattleEnvError(f"entity {entity_id} acquired conflicting public causal roots")
        self._causal_group_by_entity[entity_id] = reference

    def _record_public_causal_group(
        self, event: RichCombatEventTelemetry, rich_context: _RichObservationContext
    ) -> None:
        """Bind exact native public event handles to durable arena entities."""

        if event.kind not in {"spawn", "projectile_spawn", "projectile_deflect"}:
            return
        if event.kind == "spawn" and event.target.object_kind == 4:
            # The dedicated release hook below owns projectile identity.
            return
        scope = f"{event.generation}:{event.state_epoch}"
        if event.kind == "projectile_deflect":
            entity_id = self._combat_entity_id(event.projectile, rich_context)
            if entity_id is None:
                raise BattleEnvError("native projectile deflect lacks a canonical projectile")
            prior = self._causal_group_by_entity.get(entity_id)
            if prior is None or prior.kind != CausalGroupKind.VOLLEY:
                raise BattleEnvError("native projectile deflect lacks its exact release root")
            deflector_entity_id = self._combat_entity_id(event.source, rich_context)
            self._causal_group_by_entity[entity_id] = CausalGroupRefV1(
                kind=CausalGroupKind.VOLLEY,
                handle=f"volley:{scope}:deflect:{event.sequence}",
                source_card_id=prior.source_card_id,
                parent_entity_id=deflector_entity_id,
            )
            return
        parent_entity_id = self._causal_parent_entity_id(event, rich_context)
        source_card_id = self._normal_mode_causal_source_card_id(event, parent_entity_id=parent_entity_id)
        deployment = event.deployment_context
        if event.kind == "projectile_spawn":
            entity_id = self._combat_entity_id(event.projectile, rich_context)
            if entity_id is None:
                entity_id = self._combat_entity_id(event.target, rich_context)
            if entity_id is None:
                return
            trigger = (
                f"deploy:{deployment.deployment_sequence}"
                if deployment is not None
                else (
                    f"cause:{event.cause_sequence}" if event.cause_sequence is not None else f"event:{event.sequence}"
                )
            )
            self._bind_causal_group(
                entity_id,
                CausalGroupRefV1(
                    kind=CausalGroupKind.VOLLEY,
                    handle=f"volley:{scope}:{trigger}",
                    source_card_id=source_card_id,
                    parent_entity_id=parent_entity_id,
                ),
            )
            return

        entity_id = self._combat_entity_id(event.target, rich_context)
        if entity_id is None:
            return
        parent_reference = self._causal_group_by_entity.get(parent_entity_id) if parent_entity_id is not None else None
        if (
            deployment is None
            and event.cause_sequence is None
            and parent_reference is not None
            and parent_reference.kind == CausalGroupKind.PERSISTENT_EFFECT
        ):
            self._bind_causal_group(
                entity_id,
                CausalGroupRefV1(
                    kind=parent_reference.kind,
                    handle=parent_reference.handle,
                    source_card_id=(parent_reference.source_card_id or source_card_id),
                    parent_entity_id=parent_entity_id,
                ),
            )
            return
        cause_kind = (
            self._combat_event_kind_by_sequence.get(event.cause_sequence) if event.cause_sequence is not None else None
        )
        runtime_spawn = (
            event.cause_sequence is not None and cause_kind not in {"card_play", "deploy_requested", "deploy_executed"}
        ) or (deployment is None and parent_entity_id is not None and parent_entity_id in self._causal_group_by_entity)
        if event.target.object_kind == 3 and deployment is not None and not runtime_spawn:
            kind = CausalGroupKind.PERSISTENT_EFFECT
            handle = f"effect:{scope}:{deployment.deployment_sequence}"
            parent = None
        elif deployment is not None and not runtime_spawn:
            kind = CausalGroupKind.DEPLOYMENT
            handle = f"deploy:{scope}:{deployment.deployment_sequence}"
            parent = None
        else:
            kind = CausalGroupKind.SPAWN_WAVE
            trigger = f"cause:{event.cause_sequence}" if event.cause_sequence is not None else f"event:{event.sequence}"
            handle = f"spawn:{scope}:{trigger}"
            parent = parent_entity_id
        self._bind_causal_group(
            entity_id,
            CausalGroupRefV1(kind=kind, handle=handle, source_card_id=source_card_id, parent_entity_id=parent),
        )

    def _consume_native_combat_events(self, rich_context: _RichObservationContext) -> None:
        """Consume exact native records for causal and evolution identities.

        FAIR combat observations use public snapshot deltas. The native ring
        remains mandatory for sequence checks, public groups and entity forms.
        """

        envelope = rich_context.snapshot.combat_events
        identity = (envelope.generation, envelope.state_epoch)
        if self._combat_event_identity != identity:
            if envelope.oldest_retained_sequence != envelope.epoch_first_sequence:
                raise BattleEnvError("native combat event history was already overwritten at epoch attach")
            self._combat_event_identity = identity
            self._combat_next_sequence = envelope.epoch_first_sequence
            self._combat_native_entity_ids.clear()
            self._combat_evolution_entities.clear()
            self._causal_group_by_entity.clear()
            self._combat_event_kind_by_sequence.clear()
        expected = self._checked_ring_sequence(envelope, self._combat_next_sequence, label="combat")
        if not envelope.hook_set_installed:
            if envelope.events:
                raise BattleEnvError("unavailable native combat capability exposed records")
            self._combat_next_sequence = envelope.next_sequence
            return

        unseen = tuple(event for event in envelope.events if event.sequence >= expected)
        for event in unseen:
            if event.tick < 0:
                raise BattleEnvError("post-reset native combat event has no valid engine tick")
            self._combat_event_kind_by_sequence[event.sequence] = event.kind
            if event.kind == "card_play":
                # The public-play cursor above owns this non-combat semantic
                # event. It is intentionally not projected into CombatEventV1.
                continue
            for fact in event.facts:
                entity_id = self._combat_entity_id(fact, rich_context)
                if fact.present and fact.native_object_id is not None:
                    if entity_id is None:
                        raise BattleEnvError("present native combat fact lacks a canonical identity")
            self._record_public_causal_group(event, rich_context)
            target_entity_id = self._combat_entity_id(event.target, rich_context)
            if event.kind == "spawn" and target_entity_id is not None:
                deployment = event.deployment_context
                if deployment is not None and deployment.form_code == 1 and event.target.owner == deployment.owner:
                    prior_evolution = self._combat_evolution_entities.get(target_entity_id)
                    current_evolution = _CombatEvolutionEntity(
                        deployment=deployment, observed_tick=self._canonical_tick(event.tick)
                    )
                    if prior_evolution is not None and prior_evolution != current_evolution:
                        raise BattleEnvError("one native entity acquired conflicting evolution deployments")
                    self._combat_evolution_entities[target_entity_id] = current_evolution
                else:
                    self._combat_evolution_entities.pop(target_entity_id, None)
            elif event.kind == "despawn" and target_entity_id is not None:
                self._combat_evolution_entities.pop(target_entity_id, None)
                self._causal_group_by_entity.pop(target_entity_id, None)
        self._combat_next_sequence = envelope.next_sequence
        return

    @staticmethod
    def _effect_instance_key(effect: Any) -> tuple[Any, ...]:
        return (effect.buff_global_id, effect.name, effect.source_entity_key, effect.source_entity_validated)

    def _append_rich_entity_transition_events(
        self, *, tick: int, rich_context: _RichObservationContext, exact_native_visibility: bool = False
    ) -> None:
        current = {
            canonical_id: rich_context.snapshot.objects_by_key[key]
            for key, canonical_id in rich_context.canonical_ids.items()
            if key in rich_context.visible_keys
        }
        for entity_id in sorted(set(self._previous_rich_objects) & set(current)):
            old = self._previous_rich_objects[entity_id]
            new = current[entity_id]
            common = {
                "tick": tick,
                "owner": new.owner if new.owner in (0, 1) else None,
                "entity_id": entity_id,
                "card_id": new.card_id if new.card_id > 0 else None,
            }

            if old.shield_current is not None and new.shield_current is not None:
                shield_delta = new.shield_current - old.shield_current
                if shield_delta:
                    self._emit_event(
                        event_type="shield_damage" if shield_delta < 0 else "shield_gain",
                        data={"amount": abs(shield_delta), "remaining_shield": new.shield_current},
                        **common,
                    )
                    if old.shield_current > 0 and new.shield_current == 0:
                        self._emit_event(event_type="shield_break", **common)

            if old.active_effects is not None and new.active_effects is not None:
                old_groups: dict[tuple[Any, ...], list[Any]] = {}
                new_groups: dict[tuple[Any, ...], list[Any]] = {}
                for effect in old.active_effects:
                    old_groups.setdefault(self._effect_instance_key(effect), []).append(effect)
                for effect in new.active_effects:
                    new_groups.setdefault(self._effect_instance_key(effect), []).append(effect)
                for key in sorted(set(old_groups) | set(new_groups), key=repr):
                    old_items = old_groups.get(key, [])
                    new_items = new_groups.get(key, [])
                    representative = (new_items or old_items)[0]
                    effect_data: dict[str, Any] = {
                        "buff_global_id": representative.buff_global_id,
                        "buff_name": representative.name,
                    }
                    source_key = representative.source_entity_key
                    if representative.source_entity_validated and source_key in rich_context.canonical_ids:
                        source_id = rich_context.canonical_ids[source_key]
                        effect_data["source_entity"] = source_id
                    count_delta = len(new_items) - len(old_items)
                    if count_delta:
                        event_type = "effect_remove"
                        data = {**effect_data, "count_delta": abs(count_delta)}
                        if count_delta > 0:
                            event_type = "effect_apply" if not old_items else "effect_stack"
                            data["remaining_ms"] = representative.remaining_ms
                        self._emit_event(event_type=event_type, data=data, **common)
                    for old_effect, new_effect in zip(old_items, new_items):
                        if old_effect.remaining_ms >= 0 and new_effect.remaining_ms > old_effect.remaining_ms:
                            self._emit_event(
                                event_type="effect_refresh",
                                data={
                                    **effect_data,
                                    "old_remaining_ms": old_effect.remaining_ms,
                                    "remaining_ms": new_effect.remaining_ms,
                                },
                                **common,
                            )

            if (
                not exact_native_visibility
                and old.invisible_count is not None
                and new.invisible_count is not None
                and old.invisible != new.invisible
            ):
                # A public entity disappearing or reappearing is itself a
                # public transition. Keep the event public even though the
                # entity becomes private after entering hidden state.
                self._emit_event(
                    event_type="visibility_change",
                    data={
                        "from": "hidden" if old.invisible else "visible",
                        "to": "hidden" if new.invisible else "visible",
                        "native_invisible_count": new.invisible_count,
                    },
                    **common,
                )

            if old.target_entity_validated and new.target_entity_validated:
                old_target = old.target_entity_key
                new_target = new.target_entity_key
                if old_target != new_target:
                    event_type = (
                        "target_acquire"
                        if old_target is None
                        else "target_lose"
                        if new_target is None
                        else "target_change"
                    )
                    target_data: dict[str, Any] = {}
                    if new_target in rich_context.canonical_ids:
                        target_id = rich_context.canonical_ids[new_target]
                        target_data["target_entity"] = target_id
                    self._emit_event(event_type=event_type, data=target_data, **common)

            if (
                old.attack_sequence_stage is not None
                and new.attack_sequence_stage is not None
                and old.attack_sequence_stage != new.attack_sequence_stage
            ):
                self._emit_event(
                    event_type="attack_sequence_change",
                    data={"from": old.attack_sequence_stage, "to": new.attack_sequence_stage},
                    **common,
                )

            if (
                old.projectile is not None
                and new.projectile is not None
                and not old.projectile.terminal
                and new.projectile.terminal
            ):
                self._emit_event(
                    event_type="projectile_terminal",
                    data={
                        "projectile_data_global_id": (new.projectile.projectile_data_global_id),
                        "terminal_reason": "unknown",
                    },
                    **common,
                )

        self._previous_rich_objects = current

    def _append_player_runtime_transition_events(
        self, *, tick: int, rich_context: _RichObservationContext, resolved_pending: Sequence[_PendingRecord] = ()
    ) -> None:
        current = dict(rich_context.snapshot.players)
        for owner in sorted(set(self._previous_player_runtime) & set(current)):
            old = self._previous_player_runtime[owner]
            new = current[owner]
            if old.ability_runtime is not None and new.ability_runtime is not None:
                old_abilities = {
                    (item.controller_slot, item.action_data_global_id): item for item in old.ability_runtime
                }
                new_abilities = {
                    (item.controller_slot, item.action_data_global_id): item for item in new.ability_runtime
                }
                for key in sorted(set(old_abilities) & set(new_abilities)):
                    old_ability = old_abilities[key]
                    new_ability = new_abilities[key]
                    before = (
                        old_ability.button_state,
                        old_ability.remaining_cooldown_ms,
                        old_ability.remaining_charges_raw,
                    )
                    after = (
                        new_ability.button_state,
                        new_ability.remaining_cooldown_ms,
                        new_ability.remaining_charges_raw,
                    )
                    if before != after:
                        activation_evidence = _ability_activation_evidence(before, after)
                        source_keys = new_ability.champion_entity_keys or old_ability.champion_entity_keys
                        public_source_keys = tuple(
                            source_key
                            for source_key in source_keys
                            if source_key in rich_context.visible_keys and source_key in rich_context.canonical_ids
                        )
                        if activation_evidence and len(public_source_keys) == 1:
                            source_key = public_source_keys[0]
                            source_entity = rich_context.canonical_ids[source_key]
                            source = rich_context.snapshot.objects_by_key.get(source_key)
                            source_card_id = source.card_id if source is not None and source.card_id > 0 else None
                            # Ability controller state is owner-private, but a
                            # cast by a uniquely joined visible arena unit is a
                            # public battle fact. Publish only the fact of the
                            # activation; action IDs, charges, cooldowns, and
                            # controller slots remain private.
                            self._emit_event(
                                tick=tick,
                                event_type="ability_activation",
                                owner=owner,
                                entity_id=source_entity,
                                card_id=source_card_id,
                                data={
                                    "ability_id": (new_ability.action_data_name),
                                    "source": ("visible_runtime_transition"),
                                    "activation_evidence": tuple(activation_evidence),
                                },
                            )
                        self._emit_event(
                            tick=tick,
                            event_type="ability_state_change",
                            owner=owner,
                            data={
                                "ability_id": new_ability.action_data_name,
                                "from_button_state": old_ability.button_state_label,
                                "to_button_state": new_ability.button_state_label,
                                "remaining_cooldown_ms": (new_ability.remaining_cooldown_ms),
                                "remaining_charges_raw": (new_ability.remaining_charges_raw),
                                "private_to": owner,
                            },
                        )
            if old.evolution_runtime is not None and new.evolution_runtime is not None:
                old_slots = {(item.deck_slot, item.card_id): item for item in old.evolution_runtime}
                new_slots = {(item.deck_slot, item.card_id): item for item in new.evolution_runtime}
                for key in sorted(set(old_slots) & set(new_slots)):
                    old_slot = old_slots[key]
                    new_slot = new_slots[key]
                    evolved_commands = tuple(
                        item
                        for item in resolved_pending
                        if item.status == "executed"
                        and item.action.owner == owner
                        and item.deck_slot == new_slot.deck_slot
                        and item.card_id == new_slot.card_id
                        and item.form_code == 1
                    )
                    exact_evolution_play = (
                        old_slot.ready is True
                        and new_slot.ready is False
                        and new_slot.progress < old_slot.progress
                        and len(evolved_commands) == 1
                    )
                    if exact_evolution_play:
                        command = evolved_commands[0]
                        self._emit_event(
                            tick=tick,
                            event_type="evolution_played",
                            owner=owner,
                            card_id=new_slot.card_id,
                            data={
                                "action_id": command.action.action_id,
                                "deck_slot": new_slot.deck_slot,
                                "played_form": "EvoForm",
                                "form_code": command.form_code,
                                "form_name": command.form_name,
                                "private_to": owner,
                            },
                        )
                    if old_slot.progress != new_slot.progress or old_slot.ready != new_slot.ready:
                        self._emit_event(
                            tick=tick,
                            event_type=(
                                "evolution_ready"
                                if old_slot.ready is False and new_slot.ready is True
                                else "evolution_progress_change"
                            ),
                            owner=owner,
                            card_id=new_slot.card_id,
                            data={
                                "deck_slot": new_slot.deck_slot,
                                "from_progress": old_slot.progress,
                                "progress": new_slot.progress,
                                "cycle_required": new_slot.cycle_required,
                                "ready": new_slot.ready,
                                "played_form": "unknown",
                                "private_to": owner,
                            },
                        )
        self._previous_player_runtime = current

    def _derive_events(
        self,
        before: Mapping[str, Any],
        after: Mapping[str, Any],
        *,
        rich_context: _RichObservationContext,
        exact_native_visibility: bool = False,
        resolved_pending: Sequence[_PendingRecord] = (),
    ) -> None:
        tick = self._canonical_tick(int(after["tick"]))
        delta_ticks = max(1, int(after["tick"]) - int(before["tick"]))
        previous = self._previous_entities
        current = self._raw_entity_map(after, rich_context=rich_context)
        velocities: dict[int, tuple[float, float]] = {}
        for entity_id, item in current.items():
            card_id = int(item.get("cardId", -1))
            owner = item.get("owner")
            if entity_id not in previous:
                self._entity_birth[entity_id] = int(after["tick"])
                self._emit_event(
                    tick=tick,
                    event_type="spawn",
                    owner=owner if owner in (0, 1) else None,
                    entity_id=entity_id,
                    card_id=card_id if card_id > 0 else None,
                    position=(float(item.get("x", 0)), float(item.get("y", 0))),
                )
            else:
                velocities[entity_id] = (
                    (float(item.get("x", 0)) - float(previous[entity_id].get("x", 0))) / delta_ticks,
                    (float(item.get("y", 0)) - float(previous[entity_id].get("y", 0))) / delta_ticks,
                )
                old_hp, new_hp = previous[entity_id].get("hp"), item.get("hp")
                if old_hp is not None and new_hp is not None and int(new_hp) < int(old_hp):
                    position = (float(item.get("x", 0)), float(item.get("y", 0)))
                    victim_card_id = card_id if card_id > 0 else None
                    combat, combat_provenance = _fair_snapshot_damage_contract(
                        tick=tick,
                        target_entity=entity_id,
                        target_card_id=victim_card_id,
                        position=position,
                        amount=float(int(old_hp) - int(new_hp)),
                    )
                    self._emit_event(
                        tick=tick,
                        event_type="damage",
                        owner=owner if owner in (0, 1) else None,
                        entity_id=entity_id,
                        card_id=victim_card_id,
                        position=position,
                        data={"amount": int(old_hp) - int(new_hp), "remaining_hp": int(new_hp)},
                        combat=combat,
                        runtime_provenance=combat_provenance,
                    )
            # Entity births are public board facts, not proof that their card
            # was selected from the owner's deck: buildings, deaths, and
            # abilities can spawn ordinary units.  Deck reveals come only
            # from exact native card-play receipts or attested local actions.
        for entity_id, item in previous.items():
            if entity_id not in current:
                card_id = int(item.get("cardId", -1))
                owner = item.get("owner")
                self._emit_event(
                    tick=tick,
                    event_type="death_or_despawn",
                    owner=owner if owner in (0, 1) else None,
                    entity_id=entity_id,
                    card_id=card_id if card_id > 0 else None,
                    position=(float(item.get("x", 0)), float(item.get("y", 0))),
                )
        current_towers = {item.entity_id: item for item in self._extract_towers(after)}
        for entity_id, tower in current_towers.items():
            old = self._previous_towers.get(entity_id)
            if old is not None and tower.hitpoints < old.hitpoints:
                typed_fields: dict[str, Any] = {}
                combat, combat_provenance = _fair_snapshot_damage_contract(
                    tick=tick,
                    target_entity=entity_id,
                    target_card_id=None,
                    position=tower.position,
                    amount=old.hitpoints - tower.hitpoints,
                )
                typed_fields = {"combat": combat, "runtime_provenance": combat_provenance}
                self._emit_event(
                    tick=tick,
                    event_type="tower_damage",
                    owner=tower.owner,
                    entity_id=entity_id,
                    position=tower.position,
                    data={"amount": old.hitpoints - tower.hitpoints, "remaining_hp": tower.hitpoints},
                    **typed_fields,
                )
            if old is not None and old.active and not tower.active:
                self._emit_event(
                    tick=tick,
                    event_type="tower_destroyed",
                    owner=tower.owner,
                    entity_id=entity_id,
                    position=tower.position,
                    data={"tower_kind": tower.tower_kind},
                )
        self._append_rich_entity_transition_events(
            tick=tick, rich_context=rich_context, exact_native_visibility=exact_native_visibility
        )
        self._append_player_runtime_transition_events(
            tick=tick, rich_context=rich_context, resolved_pending=resolved_pending
        )
        self._entity_velocity = velocities
        self._previous_entities = current
        self._previous_towers = current_towers

    def _resolve_pending(
        self,
        raw: Mapping[str, Any],
        observed_ms: int,
        *,
        native_public_card_plays: Sequence[_NativePublicCardPlay] = (),
    ) -> tuple[_PendingRecord, ...]:
        native_tick = int(raw["tick"])
        resolved: list[_PendingRecord] = []
        for item in self._pending:
            if item.status != "queued":
                continue
            resolution_tick = item.expected_native_tick
            if native_tick < item.expected_native_tick:
                continue
            player = _player(raw, item.action.owner)
            same_card = any(
                int(card["handIndex"]) == item.hand_slot
                and int(card["cardId"]) == item.card_id
                and int(card.get("cardParameter", -1)) == item.card_parameter
                and int(card.get("deckSlot", -1)) == item.deck_slot
                for card in player.get("hand", ())
            )
            item.status = "rejected" if same_card else "executed"
            item.server_apply_ms = observed_ms
            resolved.append(item)
            spec = self.card_specs.get(item.card_id)
            publicly_observable = False
            if item.status == "executed":
                # A played card is public to both players. Native invisibility,
                # submerge, and disguise affect interaction/render phase, not
                # whether the opponent learns that the card was played.
                publicly_observable = spec is not None
                if publicly_observable:
                    self._revealed[item.action.owner].add(item.card_id)
                    cost = float(item.cost)
                    low, high = self._belief_elixir[item.action.owner]
                    self._belief_elixir[item.action.owner] = (max(0.0, low - cost), max(0.0, high - cost))
            event_data = {
                "action_id": item.action.action_id,
                "kind": ActionKind.PLAY_CARD.value,
                "requested_tick": self._canonical_tick(item.requested_native_tick),
                "expected_execution_tick": self._canonical_tick(item.expected_native_tick),
                "visible_card_id": item.card_id,
                "effective_card_id": item.effective_card_id,
                "native_effective_card_id": item.native_effective_card_id,
                "effective_cost": float(item.cost),
                "form_code": item.form_code,
                "native_form_code": item.native_form_code,
            }
            if item.status == "rejected":
                stored_diagnostic = item.action.metadata.get("rejection_diagnostic")
                diagnostic = dict(stored_diagnostic) if isinstance(stored_diagnostic, Mapping) else {}
                nearby_native_plays = []
                for play in native_public_card_plays:
                    deployment = play.deployment
                    if (
                        deployment.owner != item.action.owner
                        or deployment.played_card_global_id != item.card_id
                        or abs(item.expected_native_tick - play.tick) > TICKS_PER_SECOND
                    ):
                        continue
                    nearby_native_plays.append(
                        {
                            "sequence": play.sequence,
                            "tick": self._canonical_tick(play.tick),
                            "played_card_id": deployment.played_card_global_id,
                            "effective_card_id": (deployment.effective_card_global_id),
                            "card_parameter": deployment.card_parameter,
                            "deck_slot": deployment.deck_slot,
                            "cost": deployment.cost,
                            "form_code": deployment.form_code,
                            "position": play.position,
                        }
                    )
                diagnostic.update(
                    {
                        "resolution_mode": "resident_hand_slot_attestation",
                        "resolved_native_tick": resolution_tick,
                        "observed_native_tick": native_tick,
                        "same_card_still_in_slot": same_card,
                        "after_elixir_raw": int(player.get("elixirRaw", -1)),
                        "after_hand": _diagnostic_hand_snapshot(player),
                        "native_card_play_candidates": tuple(nearby_native_plays),
                    }
                )
                event_data["rejection_diagnostic"] = diagnostic
            if not publicly_observable:
                event_data["private_to"] = item.action.owner
            event_type = "action_canceled_terminal" if item.status == "canceled_terminal" else f"action_{item.status}"
            self._emit_event(
                tick=self._canonical_tick(resolution_tick),
                event_type=event_type,
                owner=item.action.owner,
                card_id=item.card_id,
                position=tuple(map(float, self._world_target(item.action))),
                data=event_data,
            )
        return tuple(resolved)

    def _resolve_pending_abilities(
        self, raw: Mapping[str, Any], observed_ms: int, *, rich_context: _RichObservationContext
    ) -> tuple[_PendingAbilityRecord, ...]:
        native_tick = int(raw["tick"])
        resolved: list[_PendingAbilityRecord] = []
        for item in self._pending_abilities:
            if item.status not in _ABILITY_PENDING_STATUSES or native_tick < item.expected_native_tick:
                continue
            evidence: list[str] = []
            pending_reason: str
            rich_player = rich_context.snapshot.players.get(item.action.owner)
            if rich_player is None or rich_player.ability_runtime is None:
                pending_reason = "ability_runtime_missing_during_attestation"
                matches = ()
            else:
                ability_spec = self.ability_specs[item.ability_id]
                expected_max_charges = int(ability_spec.charges or 0)
                expected_cooldown_ms = normalized_ability_cooldown_ms(ability_spec)
                matches = tuple(
                    ability
                    for ability in rich_player.ability_runtime
                    if ability.controller_slot == item.controller_slot
                    and ability.action_data_global_id == item.action_data_global_id
                    and ability.action_data_name == item.ability_id
                    and ability.configured_cooldown_ms == item.before_configured_cooldown_ms
                    and ability.configured_cooldown_ms == expected_cooldown_ms
                    and ability.max_charges == item.before_max_charges
                    and ability.max_charges == expected_max_charges
                )
                if len(matches) != 1:
                    pending_reason = "ability_runtime_identity_not_unique"
                else:
                    pending_reason = "ability_runtime_has_no_execution_edge"
            if len(matches) == 1:
                after = matches[0]
                evidence = _ability_activation_evidence(
                    (item.before_button_state, item.before_remaining_cooldown_ms, item.before_remaining_charges_raw),
                    (after.button_state, after.remaining_cooldown_ms, after.remaining_charges_raw),
                )

            if not evidence and native_tick < item.attestation_deadline_native_tick:
                item.status = "awaiting_execution"
                continue

            item.status = "executed" if evidence else "unattested"
            if evidence:
                item.server_apply_ms = observed_ms
            resolved.append(item)
            event_data = {
                "action_id": item.action.action_id,
                "kind": ActionKind.ACTIVATE_ABILITY.value,
                "ability_id": item.ability_id,
                "source_card_id": item.source_card_id,
                "requested_tick": self._canonical_tick(item.requested_native_tick),
                "expected_execution_tick": self._canonical_tick(item.expected_native_tick),
                "observed_execution_tick": self._canonical_tick(native_tick),
                "private_to": item.action.owner,
            }
            if item.status == "executed":
                executed_data = {
                    **event_data,
                    "ability_execution_evidence": tuple(evidence),
                    "fair_ability_activation_exact": True,
                }
                self._emit_event(
                    tick=self._canonical_tick(native_tick),
                    event_type="action_executed",
                    owner=item.action.owner,
                    entity_id=item.action.source_entity,
                    data=executed_data,
                )
                self._emit_event(
                    tick=self._canonical_tick(native_tick),
                    event_type="runtime_ability_activation",
                    owner=item.action.owner,
                    entity_id=item.action.source_entity,
                    card_id=item.source_card_id,
                    data=executed_data,
                )
            else:
                self._emit_event(
                    tick=self._canonical_tick(native_tick),
                    event_type="action_unattested",
                    owner=item.action.owner,
                    entity_id=item.action.source_entity,
                    data={
                        **event_data,
                        "reason": pending_reason,
                        "attestation_deadline_tick": self._canonical_tick(item.attestation_deadline_native_tick),
                    },
                )
        return tuple(resolved)

    def _potential(self, owner: int, towers: Sequence[TowerStateV1]) -> float:
        crowns = self._crowns(towers)
        value = float(crowns[owner] - crowns[1 - owner])
        for tower in towers:
            initial = self._initial_tower_hp.get(tower.entity_id, tower.max_hitpoints)
            loss = 1.0 - tower.hitpoints / max(1.0, initial)
            weight = 3.0 if tower.tower_kind == "king" else 1.0
            value += weight * loss * (1.0 if tower.owner != owner else -1.0)
        return value

    def _rewards(
        self, before: Sequence[TowerStateV1], after: Sequence[TowerStateV1], terminal: TerminalV1, delta_ticks: int
    ) -> dict[int, float]:
        result = {0: 0.0, 1: 0.0}
        gamma = math.exp(-delta_ticks / self.shaping_tau_ticks)
        for owner in self.possible_agents:
            # Potential-based shaping preserves the original win/loss task
            # only when terminal states have zero potential.  Carrying the
            # crown/tower potential into the terminal reward adds a separate
            # bonus for fast 3-crowns and changes the optimal policy.
            current = 0.0 if terminal.ended else self._potential(owner, after)
            shaping = self.shaping_beta * (gamma * current - self._last_potential[owner])
            result[owner] = float(terminal.result_by_owner[owner]) + shaping
            self._last_potential[owner] = current
        # Floating-point operations are made explicitly zero-sum.
        antisymmetric = (result[0] - result[1]) / 2.0
        return {0: antisymmetric, 1: -antisymmetric}

    def _combat_evolution_state(self, entity_id: int) -> EvolutionRuntimeStateV1 | None:
        linked = self._combat_evolution_entities.get(entity_id)
        if linked is None:
            return None
        deployment = linked.deployment
        if deployment.form_code != 1 or deployment.form_name != "EvoForm":
            raise BattleEnvError("native evolved entity carries a non-evolution deployment")
        card = self.card_specs.get(deployment.played_card_global_id)
        evolution = card.evolution if card is not None else None
        current_form_id = (
            evolution.evolution_form_id
            if evolution is not None and evolution.evolution_form_id
            else deployment.form_name
        )
        base_form_id = evolution.base_form_id if evolution is not None and evolution.base_form_id else None
        evidence = {name: SemanticEvidenceLevel.UNKNOWN for name in EVOLUTION_RUNTIME_STATE_FIELDS}
        evidence.update(
            {
                "phase": SemanticEvidenceLevel.NATIVE_DERIVED,
                "current_form_id": SemanticEvidenceLevel.NATIVE_DERIVED,
                "active": SemanticEvidenceLevel.NATIVE_DERIVED,
            }
        )
        sources: dict[str, tuple[str, ...]] = {
            "phase": (
                "combatEvents.events[].deploymentContext.formCode",
                "combatEvents.events[].target.nativeObjectId",
            ),
            "current_form_id": ("combatEvents.events[].deploymentContext.formCode",),
            "active": (
                "combatEvents.events[].deploymentContext.formCode",
                "combatEvents.events[].target.nativeObjectId",
            ),
        }
        if base_form_id is not None:
            evidence["base_form_id"] = SemanticEvidenceLevel.STATIC_DECLARED
            sources["base_form_id"] = ("CardSpecV1.evolution.base_form_id",)
        return EvolutionRuntimeStateV1(
            card_id=deployment.played_card_global_id,
            phase=EvolutionPhase.EVOLVED,
            base_form_id=base_form_id,
            current_form_id=current_form_id,
            active=True,
            attributes={
                "classification": "exact_consumed_descriptor_spawn_link",
                "deployment_sequence": deployment.deployment_sequence,
                "native_form": deployment.form_name,
                "owner_private_descriptor_fields_withheld": True,
            },
            provenance=SemanticProvenanceV1(
                field_evidence=evidence,
                source_fields=sources,
                observed_tick=linked.observed_tick,
                notes=(
                    "libg+0xf38a68 TLS deployment context linked directly to the spawn hook",
                    "no tick or position association was used",
                    "deck slot, packed card parameter, and cost remain owner-private",
                ),
            ),
        )

    def _entities(self, raw: Mapping[str, Any], *, rich_context: _RichObservationContext) -> tuple[EntityStateV1, ...]:
        native_tick = int(raw["tick"])
        observed_tick = self._canonical_tick(native_tick)
        items: list[EntityStateV1] = []
        for entity_id, item in sorted(self._raw_entity_map(raw, rich_context=rich_context).items()):
            rich_item = rich_context.item_for_raw(item)
            if rich_item.entity_key not in rich_context.visible_keys:
                continue
            evolution_state = self._combat_evolution_state(entity_id)
            runtime = project_runtime(
                rich_item,
                canonical_ids=rich_context.canonical_ids,
                rich_objects=rich_context.snapshot.objects_by_key,
                visible_keys=rich_context.visible_keys,
                observed_tick=observed_tick,
                effect_catalog=self.effect_catalog,
                phase_runtime=rich_context.snapshot.phase_runtime,
                phase_events=self._phase_events_by_entity.get(rich_item.entity_key, ()),
                card_spec=self.card_specs.get(rich_item.card_id),
                evolution_active=evolution_state is not None,
            )
            card_id_raw = int(item.get("cardId", -1))
            spec = self.card_specs.get(card_id_raw)
            kind = (
                "projectile"
                if rich_item.projectile is not None
                else (spec.kind.value if spec is not None else ("effect" if card_id_raw <= 0 else "unknown"))
            )
            velocity = self._entity_velocity.get(entity_id)
            if runtime.projectile_state is not None and velocity is not None:
                projectile = runtime.projectile_state
                projectile_provenance = projectile.provenance
                runtime = replace(
                    runtime,
                    projectile_state=replace(
                        projectile,
                        velocity=velocity,
                        provenance=SemanticProvenanceV1(
                            field_evidence={
                                **dict(projectile_provenance.field_evidence),
                                "velocity": (SemanticEvidenceLevel.NATIVE_DERIVED),
                            },
                            source_fields={
                                **dict(projectile_provenance.source_fields),
                                "velocity": (
                                    "ObservationV1.objects[].x",
                                    "ObservationV1.objects[].y",
                                    "ObservationV1.tick",
                                ),
                            },
                            observed_tick=observed_tick,
                            notes=(
                                *projectile_provenance.notes,
                                "velocity is the exact native position delta divided by the observed native tick delta",
                            ),
                        ),
                    ),
                )
            birth = self._entity_birth.get(entity_id, native_tick)
            runtime_fields: dict[str, Any] = {}
            runtime_provenance = runtime.entity_provenance(observed_tick)
            if evolution_state is not None:
                runtime_provenance = SemanticProvenanceV1(
                    field_evidence={
                        **dict(runtime_provenance.field_evidence),
                        "evolution_state": (SemanticEvidenceLevel.NATIVE_DERIVED),
                    },
                    source_fields={
                        **dict(runtime_provenance.source_fields),
                        "evolution_state": (
                            "combatEvents.events[].deploymentContext",
                            "combatEvents.events[].target.nativeObjectId",
                        ),
                    },
                    observed_tick=observed_tick,
                    notes=(*runtime_provenance.notes, "evolution state comes from exact dynamic deployment provenance"),
                )
            runtime_fields = {
                "shield": runtime.shield,
                "visible_target": runtime.target_entity,
                "shield_state": runtime.shield_state,
                "effect_states": runtime.effect_states,
                "attack_state": runtime.attack_state,
                "movement_runtime": (
                    runtime.movement_runtime.to_mapping() if runtime.movement_runtime is not None else None
                ),
                "deployment_runtime": (
                    runtime.deployment_runtime.to_mapping() if runtime.deployment_runtime is not None else None
                ),
                "projectile_state": runtime.projectile_state,
                "visibility_state": runtime.visibility_state,
                "resource_states": runtime.resource_states,
                "periodic_attack_modifier": (runtime.periodic_attack_modifier),
                "capture_runtime": runtime.capture_runtime,
                "threshold_relocation_runtime": (runtime.threshold_relocation_runtime),
                "evolution_state": evolution_state,
                "runtime_provenance": runtime_provenance,
            }
            items.append(
                EntityStateV1(
                    entity_id=entity_id,
                    owner=item.get("owner") if item.get("owner") in (0, 1) else None,
                    card_id=card_id_raw if card_id_raw > 0 else None,
                    entity_kind=kind,
                    position=(float(item.get("x", 0)), float(item.get("y", 0))),
                    native_data_global_id=rich_item.data_global_id,
                    velocity=velocity,
                    hitpoints=float(item["hp"]) if item.get("hp") is not None else None,
                    max_hitpoints=float(item["maxHp"]) if item.get("maxHp") is not None else None,
                    age_ms=max(0, native_tick - birth) * TICK_MS,
                    causal_group=self._causal_group_by_entity.get(entity_id),
                    **runtime_fields,
                )
            )
        return tuple(items[: self.entity_limit])

    def _players(
        self,
        raw: Mapping[str, Any],
        towers: Sequence[TowerStateV1],
        *,
        actor_owner: int | None,
        rich_context: _RichObservationContext,
    ) -> tuple[PlayerStateV1, ...]:
        crowns = self._crowns(towers, raw)
        result: list[PlayerStateV1] = []
        form_aliases = self._native_form_aliases(rich_context)
        for native_player in sorted(raw["players"], key=lambda item: int(item["owner"])):
            owner = int(native_player["owner"])
            private = owner == actor_owner
            exact = float(native_player["elixirRaw"]) / 10_000.0 if private else None
            player_metadata: dict[str, Any] = {}
            if private:
                player_metadata["hand_slot_count"] = len(native_player.get("hand", ()))
                player_metadata["hand_slot_by_card"] = {
                    str(int(card["cardId"])): int(card["handIndex"]) for card in native_player.get("hand", ())
                }
                player_metadata["hand_runtime_by_slot"] = {
                    str(int(card["handIndex"])): dict(
                        self._hand_runtime_contract(card, form_aliases=form_aliases, owner=owner)
                    )
                    for card in native_player.get("hand", ())
                }
                player_metadata["native_hand_capacity"] = 4
                raw_elixir = int(native_player.get("elixirRaw", 0))
                if raw_elixir >= 100_000:
                    player_metadata["time_to_next_elixir_ms"] = None
                    player_metadata["time_to_full_elixir_ms"] = 0
                else:
                    next_integer_raw = min(100_000, (raw_elixir // 10_000 + 1) * 10_000)
                    player_metadata["time_to_next_elixir_ms"] = (
                        self.mode_timeline.timeline.ticks_for_raw(int(raw["tick"]), next_integer_raw - raw_elixir)
                        * TICK_MS
                    )
                    player_metadata["time_to_full_elixir_ms"] = (
                        self.mode_timeline.timeline.ticks_for_raw(int(raw["tick"]), 100_000 - raw_elixir) * TICK_MS
                    )
                player_metadata["near_elixir_overflow"] = raw_elixir >= 90_000
            runtime_fields: dict[str, Any] = {}
            if private:
                runtime = self._action_player_runtime(owner, raw, rich_context)
                if runtime is None:
                    raise BattleEnvError("FAIR player observation lacks native runtime")
                runtime_fields = {
                    "ability_runtime_states": runtime.ability_runtime_states,
                    "evolution_runtime_states": runtime.evolution_runtime_states,
                    "runtime_provenance": runtime.provenance(self._canonical_tick(int(raw["tick"]))),
                }
            result.append(
                PlayerStateV1(
                    owner=owner,
                    crowns=crowns[owner],
                    tower_ids=tuple(item.entity_id for item in towers if item.owner == owner),
                    elixir_exact=exact,
                    elixir_visible=exact,
                    hand=tuple(
                        int(card["cardId"])
                        for card in sorted(native_player.get("hand", ()), key=lambda item: int(item["handIndex"]))
                    )
                    if private
                    else (),
                    next_card=(
                        int(native_player["nextCard"]["cardId"])
                        if private and native_player.get("nextCard") is not None
                        else None
                    ),
                    deck=tuple(int(card["cardId"]) for card in native_player.get("deck", ())) if private else (),
                    cycle=tuple(int(card["cardId"]) for card in native_player.get("cycle", ())) if private else (),
                    revealed_cards=tuple(sorted(self._revealed[owner])),
                    evolution_state=native_player.get("evolutionState", {}) if private else {},
                    ability_state=native_player.get("abilityState", {}) if private else {},
                    private_state_visible=private,
                    metadata=player_metadata,
                    **runtime_fields,
                )
            )
        return tuple(result)

    def _belief(self, actor_owner: int, tick: int) -> OpponentBeliefV1:
        opponent = 1 - actor_owner
        low, high = self._belief_elixir[opponent]
        bins = [0.0] * 11
        lower_bin = min(10, max(0, int(math.floor(low))))
        upper_bin = min(10, max(lower_bin, int(math.ceil(high))))
        probability = 1.0 / (upper_bin - lower_bin + 1)
        for index in range(lower_bin, upper_bin + 1):
            bins[index] = probability
        return OpponentBeliefV1(
            opponent_owner=opponent,
            deck_probabilities={str(card_id): 1.0 for card_id in sorted(self._revealed[opponent])},
            hand_hypotheses=(),
            next_card_probabilities={},
            elixir_probabilities=tuple(bins),
            updated_tick=tick,
        )

    @staticmethod
    def _event_visible_to(event: EventV1, owner: int | None, hidden_entity_ids: frozenset[int] = frozenset()) -> bool:
        if event.combat is not None and event.combat.visible_by_owner:
            return owner in (0, 1) and event.combat.visible_by_owner.get(str(owner)) is True
        if event.data.get("oracle_only") is True:
            return False
        # Visibility is fixed at event creation via data.private_to. Filtering
        # by the entity's *current* hidden state would make old public events
        # disappear and later reappear, itself a visibility side channel.
        private_to = event.data.get("private_to")
        return private_to is None or int(private_to) == owner

    def _events_for(
        self, owner: int | None, tick: int, hidden_entity_ids: frozenset[int] = frozenset()
    ) -> tuple[EventV1, ...]:
        cutoff = max(0, tick - self.event_window_ticks)
        result: list[EventV1] = []
        for event in self._events:
            if event.tick < cutoff or not self._event_visible_to(event, owner, hidden_entity_ids):
                continue
            result.append(event)
        return tuple(result)

    def _event_summary_for(
        self, owner: int | None, tick: int, hidden_entity_ids: frozenset[int] = frozenset()
    ) -> Mapping[str, Any]:
        cutoff = max(0, tick - self.event_window_ticks)
        counts: dict[str, int] = {}
        last_tick: dict[str, int] = {}
        # Hidden-entity events receive a permanent private scope when they are
        # created, before they can enter the identity-free archive.
        for key, record in self._event_archive.items():
            scope, event_type = key.split(":", 1)
            if scope not in {"public", f"owner-{owner}"}:
                continue
            counts[event_type] = counts.get(event_type, 0) + int(record["count"])
            last_tick[event_type] = max(last_tick.get(event_type, 0), int(record["last_tick"]))
        for event in self._events:
            if event.tick >= cutoff or not self._event_visible_to(event, owner, hidden_entity_ids):
                continue
            counts[event.event_type] = counts.get(event.event_type, 0) + 1
            last_tick[event.event_type] = max(last_tick.get(event.event_type, 0), event.tick)
        return {
            "window_ticks": self.event_window_ticks,
            "window_ms": self.event_window_ticks * TICK_MS,
            "earlier_counts": counts,
            "earlier_last_tick": last_tick,
        }

    def _pending_for(self, owner: int | None) -> tuple[PendingActionV1, ...]:
        return tuple(
            PendingActionV1(
                action=item.action,
                requested_tick=self._canonical_tick(item.requested_native_tick),
                expected_execution_tick=self._canonical_tick(item.expected_native_tick),
                generated_at_ms=item.generated_at_ms,
                requested_at_ms=item.requested_at_ms,
                inference_end_ms=item.inference_end_ms,
                server_apply_ms=item.server_apply_ms,
                status=item.status,
            )
            for records, statuses in ((self._pending, {"queued"}), (self._pending_abilities, _ABILITY_PENDING_STATUSES))
            for item in records
            if item.status in statuses and item.action.owner == owner
        )

    def _static_public_observation_metadata(self) -> Mapping[str, Any]:
        cached = self._public_metadata_static
        if cached is not None:
            return cached
        if self.episode_config is None:
            raise BattleEnvError("reset must be called before observe")
        cached = frozen_mapping(
            {
                "entity_limit": self.entity_limit,
                "runner_attestation_digest": (
                    self.runner_attestation.attestation_digest if self.runner_attestation is not None else None
                ),
                "event_limit": self.event_limit,
                "map_id": self.episode_config.map_id,
                "card_specs_hash": self._card_specs_hash,
                "native_effect_catalog_id": (
                    self.effect_catalog.catalog_id if self.effect_catalog is not None else None
                ),
                "timeline": {
                    "game_mode_id": self.mode_timeline.game_mode_id,
                    "game_mode_name": self.mode_timeline.game_mode_name,
                    "name": self.mode_timeline.timeline.name,
                    "source_sha256": self.mode_timeline.timeline.source_sha256,
                    "maximum_tick": self.mode_timeline.timeline.maximum_tick,
                },
                "capabilities": {
                    "exact_engine_dynamics": True,
                    "exact_base_deployment_mask": True,
                    "ability_activation": bool(hasattr(self.native, "activate_ability")),
                    "native_snapshot_restore": bool(
                        hasattr(self.native, "create_snapshot") and hasattr(self.native, "restore")
                    ),
                    "native_rich_telemetry": True,
                },
            }
        )
        self._public_metadata_static = cached
        return cached

    def observe(
        self,
        tier: ObservationTier | str = ObservationTier.FAIR,
        *,
        owner: int | None = None,
        policy_features_only: bool = False,
    ) -> ObservationV1:
        if self._raw is None:
            raise BattleEnvError("reset must be called before observe")
        actual_tier = tier if isinstance(tier, ObservationTier) else ObservationTier(str(tier))
        if actual_tier != ObservationTier.FAIR:
            raise BattleEnvError("BattleEnvV1 only emits semantic-fair observations")
        if owner not in self.possible_agents:
            raise ValueError("FAIR observations require owner 0 or 1")
        raw = self._raw
        native_tick = int(raw["tick"])
        tick = self._canonical_tick(native_tick)
        phase, multiplier = self.mode_timeline.timeline.phase(native_tick)
        scene_identity = self._rich_identity(raw)
        if self._fair_scene_cache_identity == scene_identity and self._fair_scene_cache is not None:
            rich_context, towers, entities, raw_entity_count = self._fair_scene_cache
        else:
            rich_context = self._rich_context(raw)
            towers = self._extract_towers(raw, rich_context=rich_context)
            entities = self._entities(raw, rich_context=rich_context)
            raw_entity_count = self._visible_entity_count(raw, rich_context)
            self._fair_scene_cache_identity = scene_identity
            self._fair_scene_cache = (rich_context, towers, entities, raw_entity_count)
        visible_events = self._events_for(owner, tick)
        terminal = self._terminal(raw, towers)
        action_mask = self.action_mask(owner, raw) if owner is not None else ActionMaskV1()
        # The model-facing clock ends with the five-minute gameplay timeline.
        # ``max_battle_ticks`` is only the later native-finalization watchdog.
        remaining = max(0, NATIVE_GAMEPLAY_END_TICK - tick) * TICK_MS
        captured_ms = None if policy_features_only else int(time.time() * 1000)
        if policy_features_only:
            public_metadata: dict[str, Any] = {"entity_overflow_count": max(0, raw_entity_count - self.entity_limit)}
        else:
            public_metadata = {
                **self._static_public_observation_metadata(),
                "entity_count_before_limit": raw_entity_count,
                "entity_overflow_count": max(0, raw_entity_count - self.entity_limit),
            }
            public_metadata["event_history"] = frozen_mapping(
                {**self._event_summary_for(owner, tick), "returned": len(visible_events), "limit": self.event_limit}
            )
        frozen_public_metadata = frozen_mapping(public_metadata)
        return ObservationV1(
            tier=actual_tier,
            owner=owner,
            tick=tick,
            time=TimeStateV1(
                elapsed_ms=tick * TICK_MS,
                remaining_ms=remaining,
                server_time_ms=native_tick * TICK_MS,
                capture_time_ms=captured_ms,
                available_time_ms=captured_ms,
                elixir_multiplier=multiplier,
                tick_ms=TICK_MS,
            ),
            phase=phase,
            players=self._players(raw, towers, actor_owner=owner, rich_context=rich_context),
            towers=towers,
            entities=entities,
            events=visible_events,
            opponent_belief=(None if policy_features_only or owner is None else self._belief(owner, tick)),
            action_mask=action_mask,
            pending_actions=(() if policy_features_only else self._pending_for(owner)),
            terminal=terminal,
            native_digest=None,
            ruleset_id=None if policy_features_only else self.ruleset_id,
            episode_id=self.episode_id,
            raster=None,
            truncated=(self._truncated or bool(raw.get("truncated")) or raw_entity_count > self.entity_limit),
            metadata=frozen_public_metadata,
        )

    def observe_all(self, tier: ObservationTier | str = ObservationTier.FAIR) -> dict[int, ObservationV1]:
        actual = tier if isinstance(tier, ObservationTier) else ObservationTier(str(tier))
        return {owner: self.observe(actual, owner=owner) for owner in self.possible_agents}
