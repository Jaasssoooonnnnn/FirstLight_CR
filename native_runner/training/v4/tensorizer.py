"""Live FAIR ``ObservationV1`` to ``UniversalSemanticBatchV4`` conversion.

This is intentionally a live-only tensorizer.  The durable/stored batch
format is a separate migration boundary and is not defined here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from ...arena import WORLD_HEIGHT, WORLD_WIDTH, native_building_footprint
from ...building_placement_profiles import building_placement_profile
from ...contracts import (
    AbilityRuntimeStateV1,
    AbilityPhase,
    AttackPhase,
    CardKind,
    CardSpecV1,
    CaptureTargetPhase,
    CausalGroupKind,
    CombatEventKind,
    DaggerDuchessRuntimeStateV1,
    EntityStateV1,
    EffectKind,
    ObservationTier,
    ObservationV1,
    PeriodicAttackModifierPhase,
    ProjectilePhase,
    ProjectileDragStage,
    RoyalChefRuntimeStateV1,
    TargetKind,
    ThresholdRelocationPhase,
    VisibilityPhase,
)
from ...perspective import PerspectiveTransformV1
from ..card_features import ENTITY_FORM_TO_BASE_CARD
from .catalog import (
    AbilityCatalogV1,
    CardCatalogV1,
    EntityArchetypeCatalogV1,
    UNKNOWN_CARD_VOCAB_ID,
    UNKNOWN_ENTITY_ARCHETYPE_VOCAB_ID,
)
from .config import ModelConfigV4, UNIVERSAL_OBSERVATION_VERSION
from .decoding import candidate_uid_v4
from .grouping import CausalGroupV4, build_causal_groups
from .native_actions import MIRROR_CARD_ID, ability_runtime_contract, mirror_play_runtime_contract
from .mechanics import (
    ACTIVE_EFFECT_REMAINING_CAP_MS,
    ACTIVE_EFFECT_RUNTIME_FEATURE_NAMES,
    EffectSemanticCatalogV1,
    UNKNOWN_EFFECT_VOCAB_ID,
)
from .tensors import (
    ActiveEffectSetV4,
    ActionCandidatesV4,
    ActionSequenceV4,
    BattleGroupSetV4,
    CANDIDATE_ABILITY,
    CANDIDATE_DEPLOY,
    CardSetV4,
    EventSetV4,
    EFFECT_PARENT_CHILD,
    EFFECT_PARENT_TOWER,
    GATE_WAIT,
    PreviousActionV4,
    RelationEdgesV4,
    TARGET_GRID,
    TARGET_NONE,
    TowerSetV4,
    UniversalSemanticBatchV4,
)

if TYPE_CHECKING:
    from ..tracking import DeterministicPublicTracker


OWNER_SELF = 0
OWNER_ENEMY = 1
OWNER_NEUTRAL = 2

FORM_NORMAL = 0
FORM_EVOLUTION = 1
FORM_HERO = 2
FORM_HERO_ACTIVE = 3

# Policy cards occupy the 26/27/28-million GlobalID namespaces. Native
# CharacterData, tower data, and internal forms use other namespaces. Unknown
# values outside the policy namespaces are entity identities, not evidence
# that an off-catalog card was played.
POLICY_CARD_GLOBAL_ID_MIN = 26_000_000
POLICY_CARD_GLOBAL_ID_MAX_EXCLUSIVE = 29_000_000

GROUP_TYPE = {"deployment": 0, "spawn_wave": 1, "volley": 2, "persistent_effect": 3}
CHILD_TYPE = {
    "troop": 0,
    "building": 1,
    "hero": 2,
    "projectile": 3,
    "area": 4,
    "effect": 5,
    "character": 6,
    "unknown": 7,
}
TOWER_TYPE = {"king": 0, "princess_left": 1, "princess_right": 2}
TOWER_TROOP_TYPE = {
    159_000_000: 1,  # Tower Princess
    159_000_001: 2,  # Cannoneer
    159_000_002: 3,  # Dagger Duchess
    159_000_004: 4,  # Royal Chef
}
EVENT_TYPE = {
    CombatEventKind.SPAWN.value: 1,
    CombatEventKind.DESPAWN.value: 2,
    CombatEventKind.DEATH.value: 3,
    CombatEventKind.DAMAGE.value: 4,
    CombatEventKind.HEAL.value: 5,
    CombatEventKind.SHIELD_GAIN.value: 6,
    CombatEventKind.SHIELD_DAMAGE.value: 7,
    CombatEventKind.SHIELD_BREAK.value: 8,
    CombatEventKind.EFFECT_APPLY.value: 9,
    CombatEventKind.EFFECT_REFRESH.value: 10,
    CombatEventKind.EFFECT_STACK.value: 11,
    CombatEventKind.EFFECT_REMOVE.value: 12,
    CombatEventKind.TARGET_ACQUIRE.value: 13,
    CombatEventKind.TARGET_CHANGE.value: 14,
    CombatEventKind.TARGET_LOSE.value: 15,
    CombatEventKind.ATTACK_START.value: 16,
    CombatEventKind.ATTACK_RELEASE.value: 17,
    CombatEventKind.ATTACK_HIT.value: 18,
    CombatEventKind.ATTACK_INTERRUPT.value: 19,
    CombatEventKind.PROJECTILE_SPAWN.value: 20,
    CombatEventKind.PROJECTILE_IMPACT.value: 21,
    CombatEventKind.PROJECTILE_EXPIRE.value: 22,
    CombatEventKind.PROJECTILE_DEFLECT.value: 23,
    CombatEventKind.TRANSFORM.value: 24,
    CombatEventKind.VISIBILITY_CHANGE.value: 25,
    CombatEventKind.ABILITY_START.value: 26,
    CombatEventKind.ABILITY_ACTIVATE.value: 27,
    CombatEventKind.ABILITY_END.value: 28,
    CombatEventKind.EVOLUTION_READY.value: 29,
    CombatEventKind.EVOLUTION_DEPLOY.value: 30,
    CombatEventKind.AREA_CREATE.value: 31,
    CombatEventKind.AREA_EXPIRE.value: 32,
    CombatEventKind.TOWER_AGGRO_CHANGE.value: 33,
    CombatEventKind.TOWER_ACTIVATE.value: 34,
    "tower_damage": 35,
    "tower_destroyed": 36,
    "attack_sequence_change": 37,
    "ability_state_change": 38,
}

EVENT_TYPE_ALIASES = {
    "death_or_despawn": CombatEventKind.DEATH.value,
    "effect_add": CombatEventKind.EFFECT_APPLY.value,
    "projectile_terminal": CombatEventKind.PROJECTILE_EXPIRE.value,
    "evolution_played": CombatEventKind.EVOLUTION_DEPLOY.value,
    "ability_activation": CombatEventKind.ABILITY_ACTIVATE.value,
    "runtime_ability_activation": CombatEventKind.ABILITY_ACTIVATE.value,
}

IGNORED_EVENT_TYPES = frozenset(
    {
        CombatEventKind.DEPLOY_REQUESTED.value,
        CombatEventKind.DEPLOY_EXECUTED.value,
        "action_requested",
        "action_executed",
        # Rejections remain fail-closed rollout audit data. They are not a
        # checkpoint vocabulary event and must not crash tensorization before
        # the collector can reject the affected wave with its native receipt.
        "action_rejected",
        "action_canceled_terminal",
        "action_unattested",
        "ability_command_queued",
        "card_form_command_queued",
        # Current evolution progress is already present in the private card
        # runtime features; this bookkeeping edge adds no policy information.
        "evolution_progress_change",
    }
)

REL_TARGETS = 1
REL_TARGETED_BY = 2
REL_SOURCE_OF = 3
REL_SOURCED_BY = 4
REL_TARGETING_MODIFIER_OF = 5
REL_TARGETING_MODIFIED_BY = 6
REL_CARD_REPRESENTS_GROUP = 7
REL_SPAWNED_BY_GROUP = 8
REL_DERIVED_FROM_VOLLEY = 9
REL_TOWER_TROOP_SUPPORTS = 10
REL_TOWER_TROOP_SUPPORTED_BY = 11
REL_CAPTURES = 12
REL_CAPTURED_BY = 13

_EMPTY_ACTIVE_EFFECTS = ActiveEffectSetV4(
    effect_vocab_id=torch.zeros(1, 0, dtype=torch.long),
    parent_type=torch.zeros(1, 0, dtype=torch.long),
    parent_index=torch.full((1, 0), -1, dtype=torch.long),
    source_owner_type=torch.zeros(1, 0, dtype=torch.long),
    runtime_features=torch.zeros(1, 0, len(ACTIVE_EFFECT_RUNTIME_FEATURE_NAMES)),
    mask=torch.zeros(1, 0, dtype=torch.bool),
)
_EMPTY_CANDIDATE_PLACEMENT = torch.zeros(32, 18, dtype=torch.bool)


def _finite(value: object, scale: float = 1.0) -> float:
    if value is None or isinstance(value, bool):
        return 0.0
    try:
        result = float(value) / scale
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _enum_text(value: object) -> str:
    return str(getattr(value, "value", value))


class _NumpyTensorArena:
    """Serve non-overlapping tensor views from three zeroed dtype buffers."""

    __slots__ = ("_bool_buffer", "_bool_offset", "_float_buffer", "_float_offset", "_int_buffer", "_int_offset")

    def __init__(self, *, float_count: int = 0, int_count: int = 0, bool_count: int = 0) -> None:
        self._float_buffer = np.zeros(float_count, dtype=np.float32)
        self._int_buffer = np.zeros(int_count, dtype=np.int64)
        self._bool_buffer = np.zeros(bool_count, dtype=np.bool_)
        self._float_offset = 0
        self._int_offset = 0
        self._bool_offset = 0

    @staticmethod
    def _take(buffer: np.ndarray, start: int, shape: tuple[int, ...]) -> tuple[np.ndarray, int]:
        end = start + math.prod(shape)
        if end > buffer.size:
            raise RuntimeError("NumPy tensor arena capacity exhausted")
        return buffer[start:end].reshape(shape), end

    def floats(self, *shape: int) -> np.ndarray:
        result, self._float_offset = self._take(self._float_buffer, self._float_offset, shape)
        return result

    def ints(self, *shape: int) -> np.ndarray:
        result, self._int_offset = self._take(self._int_buffer, self._int_offset, shape)
        return result

    def bools(self, *shape: int) -> np.ndarray:
        result, self._bool_offset = self._take(self._bool_buffer, self._bool_offset, shape)
        return result


class UniversalObservationTensorizerV4:
    """Build one batch row from the current actor-fair observation."""

    SCHEMA_VERSION = UNIVERSAL_OBSERVATION_VERSION

    def __init__(
        self,
        *,
        actor_owner: int,
        card_catalog: CardCatalogV1,
        ability_catalog: AbilityCatalogV1,
        entity_archetype_catalog: EntityArchetypeCatalogV1,
        effect_catalog: EffectSemanticCatalogV1,
        card_specs: Mapping[int, CardSpecV1],
        deck: Sequence[int],
        tracker: "DeterministicPublicTracker | None" = None,
        deck_roles: Mapping[int, tuple[bool, bool]],
        horizontal_mirror: bool = False,
        config: ModelConfigV4 | None = None,
        entity_form_to_base_card: Mapping[int, int] = ENTITY_FORM_TO_BASE_CARD,
        reject_unknown_public_semantics: bool = False,
    ) -> None:
        if actor_owner not in (0, 1):
            raise ValueError("actor_owner must be 0 or 1")
        if not isinstance(horizontal_mirror, bool):
            raise TypeError("horizontal_mirror must be boolean")
        if not isinstance(reject_unknown_public_semantics, bool):
            raise TypeError("reject_unknown_public_semantics must be boolean")
        self.actor_owner = int(actor_owner)
        self.reject_unknown_public_semantics = reject_unknown_public_semantics
        self.catalog = card_catalog
        self.card_specs = {int(card_id): spec for card_id, spec in card_specs.items()}
        self.deck = tuple(int(card_id) for card_id in deck)
        self.config = config or ModelConfigV4()
        if len(self.deck) != 8 or len(set(self.deck)) != 8:
            raise ValueError("V4 tensorizer requires eight unique deck cards")
        missing = set(self.deck).difference(self.card_specs)
        if missing:
            raise ValueError(f"card specs are missing actor deck cards: {sorted(missing)}")
        if tuple(sorted(self.card_specs)) != self.catalog.raw_card_ids:
            raise ValueError("CardCatalog and supplied CardSpec scopes differ")
        self.ability_catalog = ability_catalog
        self.entity_archetype_catalog = entity_archetype_catalog
        self.effect_catalog = effect_catalog
        if self.ability_catalog.card_scope != self.catalog.raw_card_ids:
            raise ValueError("AbilityCatalog and CardCatalog scopes differ")
        if self.entity_archetype_catalog.card_scope != self.catalog.raw_card_ids:
            raise ValueError("EntityArchetypeCatalog and CardCatalog scopes differ")
        if self.effect_catalog.card_scope != self.catalog.raw_card_ids:
            raise ValueError("EffectSemanticCatalog and CardCatalog scopes differ")
        if self.ability_catalog.vocab_size > self.config.ability_vocab_size:
            raise ValueError("ability catalog exceeds ModelConfigV4 capacity")
        if self.entity_archetype_catalog.vocab_size > self.config.child_archetype_count:
            raise ValueError("entity archetype catalog exceeds ModelConfigV4 capacity")
        self.tracker = tracker
        self.deck_roles = {int(card_id): (bool(value[0]), bool(value[1])) for card_id, value in deck_roles.items()}
        if set(self.deck_roles) != set(self.deck):
            raise ValueError("deck roles must classify every actor deck card exactly")
        elite_count = sum(role[0] for role in self.deck_roles.values())
        evolution_count = sum(role[1] for role in self.deck_roles.values())
        if any(elite and evolution for elite, evolution in self.deck_roles.values()):
            raise ValueError("one deck card cannot occupy both role types")
        if elite_count > 2 or evolution_count > 2 or elite_count + evolution_count > 3:
            raise ValueError("deck role counts violate the V4 match contract")
        self.entity_form_to_base_card = {int(form): int(base) for form, base in entity_form_to_base_card.items()}
        self.card_costs = {
            card_id: float(spec.elixir_cost)
            for card_id, spec in self.card_specs.items()
            if spec.elixir_cost is not None
        }
        if set(self.deck).difference(self.card_costs):
            raise ValueError("every actor deck card needs an exact elixir cost")
        self.ability_ids_by_card = {
            int(card_id): frozenset(str(value) for value in spec.ability_ids)
            for card_id, spec in self.card_specs.items()
            if spec.ability_ids
        }
        self.perspective = PerspectiveTransformV1(actor_owner=self.actor_owner, horizontal_mirror=horizontal_mirror)
        self._opponent_revealed: set[int] = set()
        self._opponent_evolution_revealed: set[int] = set()
        self._ability_state_by_source_entity: dict[int, AbilityRuntimeStateV1] = {}
        self._previous_action = self._empty_previous_action()
        self._episode_id: str | None = None
        self._episode_start_tick: int | None = None
        self._last_tensorized_tick: int | None = None
        self._placement_tensor_cache: dict[int, tuple[object, Tensor]] = {}
        self._frame_tile_position_cache: dict[tuple[float, float], tuple[float, float]] = {}
        self._frame_base_card_id_cache: dict[int, int | None] = {}
        self._frame_spatial_extent_cache: dict[int, tuple[float, float, float, float]] = {}
        self._frame_spatial_radius_cache: dict[int, float] = {}
        self._frame_archetype_id_cache: dict[int, int] = {}
        self._frame_child_kind_cache: dict[int, str] = {}

    @property
    def ability_id_by_vocab_id(self) -> dict[int, str]:
        return {index + 1: ability_id for index, ability_id in enumerate(self.ability_catalog.ability_ids)}

    def _clear_episode_state(self) -> None:
        self._opponent_revealed.clear()
        self._opponent_evolution_revealed.clear()
        self._ability_state_by_source_entity.clear()
        self._previous_action = self._empty_previous_action()

    def _validate_observation_identity(self, observation: ObservationV1) -> None:
        if observation.tier == ObservationTier.ORACLE:
            raise ValueError("V4 policy tensorizer accepts FAIR actor observations only")
        if observation.owner != self.actor_owner:
            raise ValueError("observation owner does not match V4 tensorizer")
        if not observation.episode_id:
            raise ValueError("V4 tensorizer requires a non-empty episode_id")

    def start_episode(self, observation: ObservationV1, *, initial_elixir: Mapping[int, float] | None = None) -> None:
        """Reset all state and bind this tensorizer to one episode."""

        self._validate_observation_identity(observation)
        if self.tracker is not None:
            if initial_elixir is None:
                raise ValueError("tracker-backed V4 episodes require exact initial elixir for both owners")
            self.tracker.reset(tick=observation.tick, initial_elixir=initial_elixir)
        self._clear_episode_state()
        self._episode_id = str(observation.episode_id)
        self._episode_start_tick = int(observation.tick)
        self._last_tensorized_tick = None

    def reset(self, observation: ObservationV1, *, initial_elixir: Mapping[int, float] | None = None) -> None:
        """Alias for starting a freshly reset episode."""

        self.start_episode(observation, initial_elixir=initial_elixir)

    def end_episode(self) -> None:
        """Clear episode-local memory and make further steps fail closed."""

        if self.tracker is not None:
            self.tracker.end_episode()
        self._clear_episode_state()
        self._episode_id = None
        self._episode_start_tick = None
        self._last_tensorized_tick = None

    def _validate_episode_step(self, observation: ObservationV1) -> None:
        self._validate_observation_identity(observation)
        if self._episode_id is None or self._episode_start_tick is None:
            raise RuntimeError("start_episode must be called before V4 tensorization")
        if observation.episode_id != self._episode_id:
            raise ValueError("observation episode_id does not match the active V4 episode")
        minimum_tick = self._episode_start_tick if self._last_tensorized_tick is None else self._last_tensorized_tick
        if observation.tick < minimum_tick:
            raise ValueError("V4 episode observations cannot move backwards")

    def _empty_previous_action(self) -> PreviousActionV4:
        return PreviousActionV4(
            gate=torch.tensor([GATE_WAIT], dtype=torch.long),
            micro_action_count=torch.zeros(1, dtype=torch.long),
            variant=torch.zeros(1, 2, dtype=torch.long),
            visible_card_vocab_id=torch.zeros(1, 2, dtype=torch.long),
            effective_card_vocab_id=torch.zeros(1, 2, dtype=torch.long),
            effective_form=torch.zeros(1, 2, dtype=torch.long),
            ability_vocab_id=torch.zeros(1, 2, dtype=torch.long),
            target_cell=torch.full((1, 2), -1, dtype=torch.long),
            delay_offset_bin=torch.full((1, 2), -1, dtype=torch.long),
            source_position=torch.zeros(1, 2, 2),
            source_mask=torch.zeros(1, 2, dtype=torch.bool),
            action_mask=torch.zeros(1, 2, dtype=torch.bool),
        )

    def record_action(
        self, sequence: ActionSequenceV4, batch: UniversalSemanticBatchV4, *, row: int = 0, validate: bool = True
    ) -> None:
        """Record the sampled semantic action for the next observation row."""

        candidates = batch.candidates
        if validate:
            sequence.validate(self.config, candidate_count=int(candidates.mask.shape[1]))
        count = int(sequence.micro_action_count[row].item())
        previous = self._empty_previous_action()
        previous.gate = sequence.gate[row : row + 1].detach().cpu().clone()
        previous.micro_action_count = sequence.micro_action_count[row : row + 1].detach().cpu().clone()
        for step in range(count):
            uid = int(sequence.candidate_uid[row, step].item())
            matches = (candidates.mask[row] & (candidates.uid[row] == uid)).nonzero(as_tuple=False).flatten()
            if matches.numel() != 1:
                raise ValueError("previous action candidate UID is missing or duplicated")
            candidate_row = int(matches[0].item())
            for name in (
                "variant",
                "visible_card_vocab_id",
                "effective_card_vocab_id",
                "effective_form",
                "ability_vocab_id",
            ):
                getattr(previous, name)[0, step] = getattr(candidates, name)[row, candidate_row].cpu()
            if int(candidates.variant[row, candidate_row]) == CANDIDATE_ABILITY:
                child_row = int(candidates.source_child_index[row, candidate_row].item())
                group_row = int(candidates.source_group_index[row, candidate_row].item())
                if child_row >= 0:
                    previous.source_position[0, step] = batch.groups.child_position[row, child_row].detach().cpu()
                elif group_row >= 0:
                    previous.source_position[0, step] = batch.groups.position[row, group_row].detach().cpu()
                else:
                    raise ValueError("Ability candidate lacks a retained source")
                previous.source_mask[0, step] = True
            previous.target_cell[0, step] = sequence.target_cell[row, step].cpu()
            previous.delay_offset_bin[0, step] = sequence.delay_offset_bin[row, step].cpu()
            previous.action_mask[0, step] = True
        self._previous_action = previous

    def _owner_type(self, owner: int | None) -> int:
        if owner is None:
            return OWNER_NEUTRAL
        return OWNER_SELF if owner == self.actor_owner else OWNER_ENEMY

    def _tile_position(self, position: Sequence[float | int]) -> tuple[float, float]:
        cache_key = (float(position[0]), float(position[1]))
        cached = self._frame_tile_position_cache.get(cache_key)
        if cached is not None:
            return cached
        world_x = float(WORLD_WIDTH) - cache_key[0] if self.perspective.flip_x else cache_key[0]
        world_y = float(WORLD_HEIGHT) - cache_key[1] if self.perspective.flip_y else cache_key[1]
        x = world_x / (float(WORLD_WIDTH) / self.config.board_width)
        y = world_y / (float(WORLD_HEIGHT) / self.config.board_height)
        result = (min(max(x, 0.0), self.config.board_width - 1e-4), min(max(y, 0.0), self.config.board_height - 1e-4))
        self._frame_tile_position_cache[cache_key] = result
        return result

    def _model_tower_kind(self, tower_kind: str) -> str:
        if not self.perspective.flip_x:
            return tower_kind
        if tower_kind == "princess_left":
            return "princess_right"
        if tower_kind == "princess_right":
            return "princess_left"
        return tower_kind

    def _model_velocity(self, velocity: Sequence[float | int]) -> tuple[float, float]:
        return (
            -float(velocity[0]) if self.perspective.flip_x else float(velocity[0]),
            -float(velocity[1]) if self.perspective.flip_y else float(velocity[1]),
        )

    def _normalize_public_card_id(self, card_id: int | None) -> int | None:
        if card_id is None:
            return None
        raw_card_id = int(card_id)
        base_card_id = int(self.entity_form_to_base_card.get(raw_card_id, raw_card_id))
        if base_card_id == raw_card_id and not (
            POLICY_CARD_GLOBAL_ID_MIN <= raw_card_id < POLICY_CARD_GLOBAL_ID_MAX_EXCLUSIVE
        ):
            return None
        return base_card_id

    def _base_card_id(self, entity: EntityStateV1) -> int | None:
        cache_key = int(entity.entity_id)
        if cache_key in self._frame_base_card_id_cache:
            return self._frame_base_card_id_cache[cache_key]
        if entity.evolution_state is not None:
            result = self._normalize_public_card_id(entity.evolution_state.card_id)
        elif entity.projectile_state is not None and entity.projectile_state.source_card_id is not None:
            result = self._normalize_public_card_id(entity.projectile_state.source_card_id)
        else:
            result = self._normalize_public_card_id(entity.card_id)
        self._frame_base_card_id_cache[cache_key] = result
        return result

    def _entity_source_card_id(self, entity: EntityStateV1) -> int | None:
        reference = entity.causal_group
        card_id = (
            reference.source_card_id
            if reference is not None and reference.source_card_id is not None
            else self._base_card_id(entity)
        )
        return self._normalize_public_card_id(card_id)

    def _entity_is_persistent_area_root(self, entity: EntityStateV1) -> bool:
        kind = self._entity_child_kind(entity)
        if kind in {"area", "effect"}:
            return True
        reference = entity.causal_group
        return (
            kind == CardKind.SPELL.value
            and reference is not None
            and reference.kind == CausalGroupKind.PERSISTENT_EFFECT
        )

    def _entity_spatial_extent(self, entity: EntityStateV1) -> tuple[float, float, float, float]:
        """Return only source-backed live geometry, never a guessed radius."""

        cache_key = int(entity.entity_id)
        cached = self._frame_spatial_extent_cache.get(cache_key)
        if cached is not None:
            return cached
        x, y = self._tile_position(entity.position)
        if self._entity_child_kind(entity) == "building":
            entity_card_id = self._base_card_id(entity)
            entity_spec = self.card_specs.get(entity_card_id) if entity_card_id is not None else None
            footprint = native_building_footprint(entity_spec) if entity_spec is not None else None
            if footprint is not None:
                half_width = float(footprint.width_tiles) / 2.0
                half_height = float(footprint.height_tiles) / 2.0
                result = (x - half_width, y - half_height, x + half_width, y + half_height)
            else:
                result = None
        else:
            result = None
        if result is None:
            exact_radius = self._entity_spatial_radius(entity)
            if exact_radius > 0:
                result = (x - exact_radius, y - exact_radius, x + exact_radius, y + exact_radius)
            else:
                result = (x, y, x, y)
        self._frame_spatial_extent_cache[cache_key] = result
        return result

    def _entity_spatial_radius(self, entity: EntityStateV1) -> float:
        cache_key = int(entity.entity_id)
        cached = self._frame_spatial_radius_cache.get(cache_key)
        if cached is not None:
            return cached
        if not self._entity_is_persistent_area_root(entity):
            result = 0.0
        else:
            archetype = self.entity_archetype_catalog.metadata_for_vocab_id(self._entity_archetype_id(entity))
            result = None
        if result is None and (archetype.child_kind == "area" and archetype.projectile_radius_known):
            result = float(archetype.projectile_radius_tiles)
        if result is None:
            card_id = self._entity_source_card_id(entity)
            spec = self.card_specs.get(card_id) if card_id is not None else None
            # CardSpec radius has several source meanings for troops/buildings.
            # Only a persistent Spell root makes it an unambiguous area circle.
            radius = spec.radius_tiles if spec is not None else None
            if (
                spec is None
                or spec.kind != CardKind.SPELL
                or radius is None
                or not math.isfinite(float(radius))
                or radius <= 0
            ):
                result = 0.0
            else:
                result = float(radius)
        self._frame_spatial_radius_cache[cache_key] = result
        return result

    def _extent_cell_ranges(self, extent: tuple[float, float, float, float]) -> tuple[range, range]:
        """Select policy cells whose centers lie inside an exact rectangle."""

        left, bottom, right, top = extent
        start_x = max(0, math.ceil(left - 0.5))
        stop_x = min(self.config.board_width, math.ceil(right - 0.5))
        start_y = max(0, math.ceil(bottom - 0.5))
        stop_y = min(self.config.board_height, math.ceil(top - 0.5))
        return range(start_x, max(start_x, stop_x)), range(start_y, max(start_y, stop_y))

    def _vocab_id(self, card_id: int | None, *, context: str) -> int:
        if card_id is None:
            return self.catalog.vocab_id(None)
        return self.catalog.require_vocab_id(int(card_id), context=context)

    def _runtime_form(self, entity: EntityStateV1) -> int:
        if entity.evolution_state is not None and entity.evolution_state.active is True:
            return FORM_EVOLUTION
        if entity.entity_kind == "hero":
            active = any(_enum_text(state.phase) == "active" for state in entity.ability_states)
            runtime = self._ability_state_by_source_entity.get(int(entity.entity_id))
            active = active or (runtime is not None and runtime.phase == AbilityPhase.ACTIVE)
            return FORM_HERO_ACTIVE if active else FORM_HERO
        return FORM_NORMAL

    def _card_runtime_features(self, observation: ObservationV1, *, owner: int, card_id: int) -> list[float]:
        features = [0.0] * self.config.card_runtime_feature_dim
        player = next(item for item in observation.players if item.owner == owner)
        features[0] = float(card_id in player.hand)
        features[1] = float(player.next_card == card_id)
        features[2] = float(card_id in player.revealed_cards)
        evolution = next((item for item in player.evolution_runtime_states if int(item.card_id) == card_id), None)
        public_evolution = None
        if self.tracker is not None:
            try:
                public = self.tracker.card_state(owner, card_id)
                public_evolution = public
                if owner != self.actor_owner:
                    features[6] = _finite(getattr(public, "availability", 0), 3.0)
                    features[7] = _finite(getattr(public, "cycle_distance", 0), 4.0)
                    features[8] = float(getattr(public, "initial_state_known", False))
            except (KeyError, ValueError):
                pass
        if evolution is not None:
            features[3] = 1.0
            features[4] = float(evolution.ready is True)
            features[5] = _finite(evolution.cycle_remaining, 4.0)
        elif (
            owner != self.actor_owner and card_id in self._opponent_evolution_revealed and public_evolution is not None
        ):
            features[3] = 1.0
            features[4] = float(public_evolution.evolution_ready)
            features[5] = _finite(public_evolution.evolution_cycle_remaining, 4.0)
        if owner == self.actor_owner:
            ability_ids = self.ability_ids_by_card.get(card_id, frozenset())
            runtime = next((state for state in player.ability_runtime_states if state.ability_id in ability_ids), None)
            if runtime is not None:
                features[9] = 1.0
                features[10] = float(runtime.available is True)
                features[11] = _finite(runtime.remaining_cooldown_ms, 30_000.0)
                features[12] = _finite(runtime.charges, 4.0)
                features[13] = float(runtime.phase == AbilityPhase.CASTING)
                features[14] = float(runtime.phase == AbilityPhase.ACTIVE)
        elif self.tracker is not None and self.ability_ids_by_card.get(card_id):
            public_ability = self.tracker.ability_state(owner, tick=observation.tick, card_id=card_id)
            features[9] = 1.0
            features[10] = float(public_ability.ready)
            features[11] = _finite(public_ability.cooldown_remaining_ms, 30_000.0)
            features[12] = _finite(public_ability.remaining_charges, 4.0)
        return features

    def _own_card_runtime_form(self, observation: ObservationV1, card_id: int) -> int:
        player = next(item for item in observation.players if item.owner == self.actor_owner)
        if card_id in player.hand:
            hand_slot_by_card = player.metadata.get("hand_slot_by_card")
            if not isinstance(hand_slot_by_card, Mapping):
                raise ValueError("actor hand has no exact model-slot mapping")
            slot = hand_slot_by_card.get(str(card_id), hand_slot_by_card.get(card_id))
            if slot is None:
                raise ValueError("actor hand card has no exact model slot")
            runtime_by_slot = player.metadata.get("hand_runtime_by_slot")
            if not isinstance(runtime_by_slot, Mapping):
                raise ValueError("actor hand has no exact runtime mapping")
            runtime = runtime_by_slot.get(str(int(slot)), runtime_by_slot.get(int(slot)))
            if not isinstance(runtime, Mapping) or type(runtime.get("form_code")) is not int:
                raise ValueError("actor hand card has no exact runtime form")
            form_code = int(runtime["form_code"])
            if not 0 <= form_code < self.config.form_type_count:
                raise ValueError("actor hand runtime form is outside the V4 vocabulary")
            return form_code
        evolution = next((state for state in player.evolution_runtime_states if int(state.card_id) == card_id), None)
        if evolution is not None and evolution.ready is True:
            return FORM_EVOLUTION
        return FORM_NORMAL

    def _remember_public_evolution_forms(self, observation: ObservationV1) -> None:
        """Persist only forms that have already appeared in public events."""

        for event in observation.events:
            if not event.visible or event.owner == self.actor_owner:
                continue
            combat = event.combat
            card_id = combat.source_card_id if combat is not None else event.card_id
            form_code = event.data.get("form_code")
            is_evolution = (
                (combat is not None and combat.evolution_form_id is not None)
                or form_code == FORM_EVOLUTION
                or event.event_type in {CombatEventKind.EVOLUTION_DEPLOY.value, "evolution_played"}
            )
            if card_id is None or not is_evolution:
                continue
            base_card_id = self.entity_form_to_base_card.get(int(card_id), int(card_id))
            if self.catalog.vocab_id(base_card_id) != UNKNOWN_CARD_VOCAB_ID:
                self._opponent_evolution_revealed.add(base_card_id)

    def _card_sets(self, observation: ObservationV1) -> tuple[CardSetV4, CardSetV4, dict[int, int]]:
        own_order = tuple(sorted(self.deck, key=lambda card_id: self._vocab_id(card_id, context="actor deck card")))
        card_to_row = {card_id: row for row, card_id in enumerate(own_order)}
        own_roles = [self.deck_roles[card_id] for card_id in own_order]
        own = CardSetV4(
            card_vocab_id=torch.tensor(
                [[self._vocab_id(card_id, context="actor deck card") for card_id in own_order]], dtype=torch.long
            ),
            runtime_form=torch.tensor(
                [[self._own_card_runtime_form(observation, card_id) for card_id in own_order]], dtype=torch.long
            ),
            role_bits=torch.tensor([own_roles], dtype=torch.bool),
            runtime_features=torch.tensor(
                [
                    [
                        self._card_runtime_features(observation, owner=self.actor_owner, card_id=card_id)
                        for card_id in own_order
                    ]
                ],
                dtype=torch.float32,
            ),
            mask=torch.ones(1, 8, dtype=torch.bool),
        )

        enemy = 1 - self.actor_owner
        enemy_player = next(item for item in observation.players if item.owner == enemy)
        self._opponent_revealed.update(int(value) for value in enemy_player.revealed_cards)
        if self.tracker is not None:
            self._opponent_revealed.update(int(value) for value in self.tracker.revealed_cards(enemy))
        unknown_reveals = sorted(
            card_id for card_id in self._opponent_revealed if self.catalog.vocab_id(card_id) == UNKNOWN_CARD_VOCAB_ID
        )
        if unknown_reveals:
            raise ValueError(f"global card catalog is missing public opponent cards: {unknown_reveals}")
        if len(self._opponent_revealed) > 8:
            raise ValueError("opponent publicly revealed more than eight cards")
        visible = tuple(
            sorted(
                self._opponent_revealed, key=lambda card_id: self._vocab_id(card_id, context="public opponent card")
            )[:8]
        )
        opponent_ids = [self._vocab_id(card_id, context="public opponent card") for card_id in visible]
        opponent_ids.extend([0] * (8 - len(opponent_ids)))
        unknown_count = 8 - len(visible)
        opponent_ids.append(UNKNOWN_CARD_VOCAB_ID if unknown_count > 0 else 0)
        opponent_mask = [True] * len(visible) + [False] * unknown_count + [unknown_count > 0]
        opponent_runtime = [
            self._card_runtime_features(observation, owner=enemy, card_id=card_id) for card_id in visible
        ]
        opponent_runtime.extend([[0.0] * self.config.card_runtime_feature_dim for _ in range(8 - len(visible))])
        unknown = [0.0] * self.config.card_runtime_feature_dim
        unknown[15] = unknown_count / 8.0
        opponent_runtime.append(unknown)
        opponent = CardSetV4(
            card_vocab_id=torch.tensor([opponent_ids], dtype=torch.long),
            runtime_form=torch.zeros(1, 9, dtype=torch.long),
            role_bits=torch.zeros(1, 9, 2, dtype=torch.bool),
            runtime_features=torch.tensor([opponent_runtime], dtype=torch.float32),
            mask=torch.tensor([opponent_mask], dtype=torch.bool),
        )
        return own, opponent, card_to_row

    def _towers(self, observation: ObservationV1) -> tuple[TowerSetV4, dict[int, int]]:
        towers = sorted(
            observation.towers,
            key=lambda item: (
                item.owner != self.actor_owner,
                TOWER_TYPE.get(self._model_tower_kind(item.tower_kind), len(TOWER_TYPE)),
                item.entity_id,
            ),
        )
        if len(towers) > self.config.max_towers:
            raise ValueError("observation exceeds the six-tower V4 contract")
        count = self.config.max_towers
        type_rows: list[int] = []
        tower_troop_rows: list[int] = []
        owner_rows: list[int] = []
        feature_rows: list[list[float]] = []
        position_rows: list[tuple[float, float]] = []
        extent_rows: list[tuple[float, float, float, float]] = []
        index_by_entity: dict[int, int] = {}
        for row, tower in enumerate(towers):
            position = self._tile_position(tower.position)
            tower_kind = self._model_tower_kind(tower.tower_kind)
            try:
                tower_type = TOWER_TYPE[tower_kind]
            except KeyError as error:
                raise ValueError("tower kind is outside standard competitive 1v1") from error
            tower_troop = 0
            if tower.tower_troop_id is not None:
                try:
                    tower_troop = TOWER_TROOP_TYPE[int(tower.tower_troop_id)]
                except KeyError as error:
                    raise ValueError("FAIR tower state contains a non-competitive Tower Troop") from error
            half_extent = 2.0 if tower_kind == "king" else 1.5
            extent = (
                position[0] - half_extent,
                position[1] - half_extent,
                position[0] + half_extent,
                position[1] + half_extent,
            )
            feature = [0.0] * self.config.tower_feature_dim
            feature[0] = _finite(tower.hitpoints, tower.max_hitpoints)
            feature[1] = _finite(tower.shield, tower.max_hitpoints)
            feature[2] = float(tower.active)
            feature[3] = float(tower.visible_target is not None)
            feature[4] = float("activated" in tower.status)
            feature[5] = _finite(tower.hitpoints, 5000.0)
            feature[6] = _finite(tower.max_hitpoints, 5000.0)
            if tower.attack_state is not None:
                attack = tower.attack_state
                feature[7] = _finite(attack.cooldown_remaining_ms, 5000.0)
                feature[8] = _finite(attack.phase_remaining_ms, 5000.0)
                feature[9] = float(attack.phase == AttackPhase.WINDUP)
                feature[10] = float(attack.phase in {AttackPhase.RELEASE, AttackPhase.CHANNEL})
                feature[11] = float(attack.phase == AttackPhase.COOLDOWN)
                feature[12] = float(attack.phase == AttackPhase.INTERRUPTED or attack.interrupted is True)
                feature[13] = _finite(attack.sequence_index, 10.0)
                feature[14] = _finite(attack.damage_multiplier, 5.0)
            tower_runtime = tower.tower_troop_runtime
            if isinstance(tower_runtime, DaggerDuchessRuntimeStateV1):
                feature[15] = _finite(tower_runtime.charge_count, tower_runtime.max_charge_count)
                feature[16] = _finite(tower_runtime.recharge_elapsed_ms, tower_runtime.recharge_duration_ms)
            elif isinstance(tower_runtime, RoyalChefRuntimeStateV1):
                feature[17] = _finite(tower_runtime.start_delay_remaining_ms, tower_runtime.start_delay_duration_ms)
                feature[18] = min(1.0, _finite(tower_runtime.cooking_contribution, tower_runtime.contribution_needed))
                feature[19] = tower_runtime.surviving_side_towers / 2.0
                feature[20] = float(tower_runtime.throw_delay_remaining_ms is not None)
            type_rows.append(tower_type)
            tower_troop_rows.append(tower_troop)
            owner_rows.append(self._owner_type(tower.owner))
            feature_rows.append(feature)
            position_rows.append(position)
            extent_rows.append(extent)
            index_by_entity[int(tower.entity_id)] = row

        padding = count - len(towers)
        type_rows.extend([0] * padding)
        tower_troop_rows.extend([0] * padding)
        owner_rows.extend([OWNER_NEUTRAL] * padding)
        feature_rows.extend([[0.0] * self.config.tower_feature_dim for _ in range(padding)])
        position_rows.extend([(0.0, 0.0)] * padding)
        extent_rows.extend([(0.0, 0.0, 0.0, 0.0)] * padding)
        mask_rows = [True] * len(towers) + [False] * padding
        return TowerSetV4(
            torch.tensor([type_rows], dtype=torch.long),
            torch.tensor([tower_troop_rows], dtype=torch.long),
            torch.tensor([owner_rows], dtype=torch.long),
            torch.tensor([feature_rows], dtype=torch.float32),
            torch.tensor([position_rows], dtype=torch.float32),
            torch.tensor([extent_rows], dtype=torch.float32),
            torch.tensor([mask_rows], dtype=torch.bool),
        ), index_by_entity

    def _group_priority(
        self, group: CausalGroupV4, entity_by_id: Mapping[int, EntityStateV1], ability_sources: set[int]
    ) -> tuple[object, ...]:
        members = [entity_by_id[value] for value in group.child_entity_ids]
        relation_ids = {
            int(value)
            for entity in members
            for value in (entity.visible_target, entity.source_entity)
            if value is not None
        }
        return (
            not any(entity.entity_id in ability_sources for entity in members),
            not bool(relation_ids),
            not any(entity.entity_kind in {"hero", "building"} for entity in members),
            not any(entity.projectile_state is not None for entity in members),
            min(group.child_entity_ids),
        )

    def _child_priority(
        self, entity: EntityStateV1, *, ability_sources: set[int], relation_ids: set[int]
    ) -> tuple[object, ...]:
        projectile = entity.projectile_state
        return (
            entity.entity_id not in ability_sources,
            entity.entity_id not in relation_ids,
            entity.entity_kind not in {"hero", "building"},
            not (projectile is not None and projectile.expected_impact_tick is not None),
            -_finite(entity.hitpoints),
            entity.entity_id,
        )

    def _groups(self, observation: ObservationV1) -> tuple[BattleGroupSetV4, dict[int, int], dict[int, int], int, int]:
        entity_by_id = {int(entity.entity_id): entity for entity in observation.entities}
        ability_sources = {int(value) for value in observation.action_mask.ability_sources}
        raw_groups = build_causal_groups(observation.entities)
        ordered = sorted(raw_groups, key=lambda group: self._group_priority(group, entity_by_id, ability_sources))
        ability_group_count = sum(
            any(value in ability_sources for value in group.child_entity_ids) for group in ordered
        )
        if ability_group_count > self.config.max_battle_groups:
            raise ValueError("battle-group capacity cannot retain every ability source")
        retained = ordered[: self.config.max_battle_groups]
        group_overflow = len(raw_groups) - len(retained)
        group_index_by_entity = {
            entity_id: group_row for group_row, group in enumerate(retained) for entity_id in group.child_entity_ids
        }

        relation_ids = {
            int(value)
            for entity in observation.entities
            for value in (entity.visible_target, entity.source_entity)
            if value is not None
        }
        child_candidates = [
            (group_index_by_entity[entity.entity_id], entity)
            for entity in observation.entities
            if entity.entity_id in group_index_by_entity
        ]
        child_candidates.sort(
            key=lambda item: self._child_priority(item[1], ability_sources=ability_sources, relation_ids=relation_ids)
        )
        required_children = sum(entity.entity_id in ability_sources for _, entity in child_candidates)
        if required_children > self.config.max_children_total:
            raise ValueError("child capacity cannot retain every ability source")
        retained_children = child_candidates[: self.config.max_children_total]
        retained_child_ids = {entity.entity_id for _, entity in retained_children}
        child_overflow = len(child_candidates) - len(retained_children)

        group_count = self.config.max_battle_groups
        child_count = self.config.max_children_total
        arena = _NumpyTensorArena(
            float_count=(
                group_count * (self.config.group_feature_dim + 2 + 4)
                + child_count * (self.config.child_feature_dim + 2 + 4 + 1)
            ),
            int_count=group_count * 4 + child_count * 3,
            bool_count=group_count + child_count,
        )
        source_card = arena.ints(1, group_count)
        group_type = arena.ints(1, group_count)
        owner_type = arena.ints(1, group_count)
        owner_type.fill(OWNER_NEUTRAL)
        runtime_form = arena.ints(1, group_count)
        features = arena.floats(1, group_count, self.config.group_feature_dim)
        positions = arena.floats(1, group_count, 2)
        extents = arena.floats(1, group_count, 4)
        group_mask = arena.bools(1, group_count)

        for group_row, group in enumerate(retained):
            members = [entity_by_id[value] for value in group.child_entity_ids]
            child_positions = [self._tile_position(entity.position) for entity in members]
            member_extents = [self._entity_spatial_extent(entity) for entity in members]
            center_x = sum(value[0] for value in child_positions) / len(child_positions)
            center_y = sum(value[1] for value in child_positions) / len(child_positions)
            base_card_id = group.source_card_id
            if base_card_id is not None:
                base_card_id = self._normalize_public_card_id(base_card_id)
            if base_card_id is None:
                base_card_id = next(
                    (self._base_card_id(entity) for entity in members if self._base_card_id(entity) is not None), None
                )
            source_card[0, group_row] = self._vocab_id(base_card_id, context="public causal-group source card")
            group_type[0, group_row] = GROUP_TYPE[group.key.kind.value]
            owner_type[0, group_row] = self._owner_type(group.owner)
            runtime_form[0, group_row] = max(self._runtime_form(entity) for entity in members)
            positions[0, group_row] = (center_x, center_y)
            extents[0, group_row] = (
                min(value[0] for value in member_extents),
                min(value[1] for value in member_extents),
                max(value[2] for value in member_extents),
                max(value[3] for value in member_extents),
            )
            retained_count = sum(entity.entity_id in retained_child_ids for entity in members)
            hitpoints = [_finite(entity.hitpoints) for entity in members]
            ratios = [_finite(entity.hitpoints, float(entity.max_hitpoints or 1.0)) for entity in members]
            shields = [_finite(entity.shield) for entity in members]
            features[0, group_row, 0] = len(members) / 16.0
            features[0, group_row, 1] = (len(members) - retained_count) / 16.0
            features[0, group_row, 2] = sum(hitpoints) / 10_000.0
            features[0, group_row, 3] = sum(shields) / 10_000.0
            features[0, group_row, 4] = sum(ratios) / len(ratios)
            features[0, group_row, 5] = min(ratios)
            features[0, group_row, 6] = max(ratios)
            features[0, group_row, 7] = max(value[2] for value in member_extents) - min(
                value[0] for value in member_extents
            )
            features[0, group_row, 8] = max(value[3] for value in member_extents) - min(
                value[1] for value in member_extents
            )
            features[0, group_row, 9] = float(any(entity.entity_id in ability_sources for entity in members))
            features[0, group_row, 10] = float(any(entity.visible_target is not None for entity in members))
            velocities: list[tuple[float, float]] = []
            for entity in members:
                if entity.velocity is None:
                    continue
                model_velocity = self._model_velocity(entity.velocity)
                vx = model_velocity[0] / (WORLD_WIDTH / self.config.board_width)
                vy = model_velocity[1] / (WORLD_HEIGHT / self.config.board_height)
                velocities.append((vx, vy))
            if velocities:
                mean_vx = sum(value[0] for value in velocities) / len(velocities)
                mean_vy = sum(value[1] for value in velocities) / len(velocities)
                velocity_dispersion = sum(
                    (value[0] - mean_vx) ** 2 + (value[1] - mean_vy) ** 2 for value in velocities
                ) / len(velocities)
                features[0, group_row, 11] = _finite(mean_vx, 10.0)
                features[0, group_row, 12] = _finite(mean_vy, 10.0)
                features[0, group_row, 13] = _finite(math.hypot(mean_vx, mean_vy), 10.0)
                features[0, group_row, 14] = _finite(velocity_dispersion, 100.0)
            offsets = [(value[0] - center_x, value[1] - center_y) for value in child_positions]
            features[0, group_row, 15] = sum(value[0] ** 2 for value in offsets) / (
                len(offsets) * self.config.board_width**2
            )
            features[0, group_row, 16] = sum(value[1] ** 2 for value in offsets) / (
                len(offsets) * self.config.board_height**2
            )
            features[0, group_row, 17] = sum(value[0] * value[1] for value in offsets) / (
                len(offsets) * self.config.board_width * self.config.board_height
            )
            projectile_entities = [entity for entity in members if entity.projectile_state is not None]
            known_projectile_damage: list[float] = []
            for entity in projectile_entities:
                projectile = entity.projectile_state
                if projectile is not None and projectile.damage is not None:
                    known_projectile_damage.append(float(projectile.damage))
                    continue
                static_damage, static_known = self._entity_static_damage(entity)
                if static_known:
                    known_projectile_damage.append(static_damage)
            features[0, group_row, 18] = len(projectile_entities) / 16.0
            features[0, group_row, 19] = sum(known_projectile_damage) / 5_000.0
            features[0, group_row, 20] = (
                len(known_projectile_damage) / len(projectile_entities) if projectile_entities else 0.0
            )
            features[0, group_row, 21] = float(bool(group.parent_entity_ids))
            group_mask[0, group_row] = True

        child_archetype = arena.ints(1, child_count)
        child_group = arena.ints(1, child_count)
        child_type = arena.ints(1, child_count)
        child_features = arena.floats(1, child_count, self.config.child_feature_dim)
        child_positions = arena.floats(1, child_count, 2)
        child_extents = arena.floats(1, child_count, 4)
        child_radii = arena.floats(1, child_count)
        child_mask = arena.bools(1, child_count)
        child_index_by_entity: dict[int, int] = {}
        for child_row, (group_row, entity) in enumerate(retained_children):
            archetype_id = self._entity_archetype_id(entity)
            archetype = self.entity_archetype_catalog.metadata_for_vocab_id(archetype_id)
            child_archetype[0, child_row] = archetype_id
            child_group[0, child_row] = group_row
            kind = self._entity_child_kind(entity)
            if self.reject_unknown_public_semantics and kind not in CHILD_TYPE:
                raise ValueError(
                    "public entity has an unrecognized child kind: "
                    f"entity_id={entity.entity_id}, entity_kind={entity.entity_kind!r}, "
                    f"resolved_kind={kind!r}, native_data_global_id="
                    f"{entity.native_data_global_id!r}"
                )
            if self.reject_unknown_public_semantics and kind == "unknown":
                raise ValueError(
                    "public entity resolved to UNKNOWN child kind: "
                    f"entity_id={entity.entity_id}, entity_kind={entity.entity_kind!r}, "
                    f"native_data_global_id={entity.native_data_global_id!r}"
                )
            child_type[0, child_row] = CHILD_TYPE.get(kind, CHILD_TYPE["unknown"])
            position = self._tile_position(entity.position)
            child_positions[0, child_row] = position
            child_extents[0, child_row] = self._entity_spatial_extent(entity)
            child_radii[0, child_row] = self._entity_spatial_radius(entity)
            dynamic = child_features[0, child_row]
            dynamic[0] = _finite(entity.hitpoints, entity.max_hitpoints or 1.0)
            dynamic[1] = _finite(entity.hitpoints, 10_000.0)
            dynamic[2] = _finite(entity.max_hitpoints, 10_000.0)
            dynamic[3] = _finite(entity.shield, 10_000.0)
            dynamic[4] = _finite(entity.age_ms, 60_000.0)
            dynamic[5] = float(entity.visible_target is not None)
            dynamic[6] = float(entity.entity_id in ability_sources)
            dynamic[7] = _finite(position[0], self.config.board_width)
            dynamic[8] = _finite(position[1], self.config.board_height)
            if entity.velocity is not None:
                model_velocity = self._model_velocity(entity.velocity)
                velocity = (
                    model_velocity[0] / (WORLD_WIDTH / self.config.board_width),
                    model_velocity[1] / (WORLD_HEIGHT / self.config.board_height),
                )
                dynamic[9] = _finite(velocity[0], 10.0)
                dynamic[10] = _finite(velocity[1], 10.0)
            if entity.attack_state is not None:
                attack = entity.attack_state
                dynamic[11] = _finite(attack.cooldown_remaining_ms, 5000.0)
                dynamic[12] = _finite(attack.phase_remaining_ms, 5000.0)
                dynamic[16] = float(attack.phase == AttackPhase.WINDUP)
                dynamic[17] = float(attack.phase in {AttackPhase.RELEASE, AttackPhase.CHANNEL})
                dynamic[18] = float(attack.phase == AttackPhase.COOLDOWN)
                dynamic[19] = float(attack.phase == AttackPhase.CHARGING)
                dynamic[20] = float(attack.phase == AttackPhase.INTERRUPTED or attack.interrupted is True)
                dynamic[21] = _finite(attack.sequence_index, 10.0)
                dynamic[22] = _finite(attack.charge_stage, 10.0)
                dynamic[23] = _finite(attack.charge_elapsed_ms, 5000.0)
                dynamic[24] = _finite(attack.damage_multiplier, 5.0)
                dynamic[25] = float(attack.locked is True)
                if attack.sequence_progress is not None:
                    dynamic[64] = _finite(attack.sequence_progress, attack.sequence_progress_limit)
                    dynamic[65] = 1.0
                if attack.sequence_decay_remaining_ms is not None:
                    dynamic[66] = _finite(attack.sequence_decay_remaining_ms, attack.sequence_decay_duration_ms)
                    dynamic[67] = 1.0
            deployment = entity.deployment_runtime
            if deployment is not None:
                dynamic[26] = float(deployment.get("phase") == "deploying")
                dynamic[27] = _finite(deployment.get("remaining_native_ms"), 5000.0)
            movement = entity.movement_runtime
            if movement is not None:
                dynamic[28] = _finite(movement.get("effective_speed"), 1000.0)
                dynamic[29] = float(movement.get("classic_charge_phase") == "ready")
                dynamic[35] = _finite(movement.get("classic_charge_progress"), 10_000.0)
            if entity.visibility_state is not None:
                visibility = entity.visibility_state
                dynamic[30] = float(visibility.phase in {VisibilityPhase.HIDDEN, VisibilityPhase.BURROWED})
                dynamic[31] = _finite(visibility.transition_remaining_ms, 5000.0)
                targetable = visibility.targetable_by_owner.get(str(self.actor_owner))
                if isinstance(targetable, bool):
                    dynamic[36] = float(targetable)
                    dynamic[37] = 1.0
                if isinstance(visibility.area_damage_eligible, bool):
                    dynamic[38] = float(visibility.area_damage_eligible)
                    dynamic[39] = 1.0
            projectile = entity.projectile_state
            if projectile is not None:
                damage = projectile.damage
                damage_known = damage is not None
                if not damage_known and archetype.static_damage_basis_known:
                    damage = archetype.static_damage_basis
                    damage_known = True
                radius = projectile.radius_tiles
                radius_known = radius is not None
                if not radius_known and archetype.projectile_radius_known:
                    radius = archetype.projectile_radius_tiles
                    radius_known = True
                dynamic[13] = _finite(damage, 5000.0)
                dynamic[14] = _finite(radius, 5.0)
                dynamic[15] = float(projectile.homing is True)
                dynamic[32] = float(projectile.phase == ProjectilePhase.IN_FLIGHT)
                dynamic[33] = float(damage_known)
                dynamic[34] = float(radius_known)
                if projectile.target_position is not None:
                    target_position = self._tile_position(projectile.target_position)
                    dynamic[40] = _finite(target_position[0], self.config.board_width)
                    dynamic[41] = _finite(target_position[1], self.config.board_height)
                    dynamic[42] = 1.0
                if projectile.drag_stage is not None:
                    dynamic[43] = float(projectile.drag_stage == ProjectileDragStage.DRAG_BACK_ACTIVE)
                    dynamic[44] = 1.0
            for resource in entity.resource_states:
                if resource.kind == "extra_spawn_accumulator":
                    dynamic[45] = float(resource.normalized)
                    dynamic[46] = 1.0
                    break
            capture = entity.capture_runtime
            if capture is not None:
                dynamic[47] = 1.0
                phase_indices = {
                    CaptureTargetPhase.ACQUIRED_DELAY: 48,
                    CaptureTargetPhase.GRAB_PAUSE: 49,
                    CaptureTargetPhase.DRAGGING: 50,
                    CaptureTargetPhase.CONTAINED: 51,
                    CaptureTargetPhase.RELEASE_PENDING: 52,
                }
                remaining_values: list[int] = []
                for target in capture.targets:
                    dynamic[phase_indices[target.phase]] = 1.0
                    if target.phase_budget_remaining_ms is not None:
                        remaining_values.append(target.phase_budget_remaining_ms)
                if remaining_values:
                    dynamic[53] = _finite(min(remaining_values), 5000.0)
                dynamic[54] = min(_finite(capture.hit_accumulator_ms, capture.hit_frequency_ms), 1.0)
                if capture.configured_cooldown_ms > 0:
                    dynamic[55] = _finite(capture.cooldown_remaining_ms, capture.configured_cooldown_ms)
            relocation = entity.threshold_relocation_runtime
            if relocation is not None:
                dynamic[56] = 1.0
                dynamic[57] = float(relocation.phase == ThresholdRelocationPhase.WAITING_THRESHOLD)
                dynamic[58] = float(relocation.phase == ThresholdRelocationPhase.RELOCATING)
                dynamic[59] = float(relocation.phase == ThresholdRelocationPhase.EXHAUSTED)
                dynamic[60] = float(relocation.burrowed)
                dynamic[61] = _finite(relocation.remaining_ms, relocation.hide_duration_ms)
                dynamic[62] = _finite(relocation.relocation_index, relocation.threshold_count)
                if relocation.relocation_index < relocation.threshold_count:
                    dynamic[63] = _finite(relocation.thresholds_percent[relocation.relocation_index], 100.0)
            modifier = entity.periodic_attack_modifier
            if modifier is not None:
                dynamic[68] = 1.0
                dynamic[69] = _finite(modifier.completed_attacks, modifier.period_attacks)
                dynamic[70] = float(modifier.phase == PeriodicAttackModifierPhase.SOURCE_DEATH_LINGER)
                dynamic[71] = _finite(modifier.linger_remaining_ms, modifier.linger_duration_ms)
            child_mask[0, child_row] = True
            child_index_by_entity[int(entity.entity_id)] = child_row

        groups = BattleGroupSetV4(
            source_card_vocab_id=torch.from_numpy(source_card),
            group_type=torch.from_numpy(group_type),
            owner_type=torch.from_numpy(owner_type),
            runtime_form=torch.from_numpy(runtime_form),
            features=torch.from_numpy(features),
            position=torch.from_numpy(positions),
            extent=torch.from_numpy(extents),
            mask=torch.from_numpy(group_mask),
            child_archetype_id=torch.from_numpy(child_archetype),
            child_group_index=torch.from_numpy(child_group),
            child_type=torch.from_numpy(child_type),
            child_features=torch.from_numpy(child_features),
            child_position=torch.from_numpy(child_positions),
            child_extent=torch.from_numpy(child_extents),
            child_radius=torch.from_numpy(child_radii),
            child_mask=torch.from_numpy(child_mask),
        )
        return (groups, group_index_by_entity, child_index_by_entity, group_overflow, child_overflow)

    def _active_effects(
        self,
        observation: ObservationV1,
        *,
        child_index_by_entity: Mapping[int, int],
        tower_index_by_entity: Mapping[int, int],
    ) -> ActiveEffectSetV4:
        rows: list[tuple[int, int, int, int, tuple[float, ...]]] = []

        def append_effects(effects: Sequence[object], *, parent_type: int, parent_index: int) -> None:
            for effect in sorted(
                effects,
                key=lambda item: (
                    str(getattr(item, "effect_id", "")),
                    int(getattr(item, "source_entity", -1) or -1),
                    int(getattr(item, "started_tick", -1) or -1),
                ),
            ):
                if getattr(effect, "active", None) is False:
                    continue
                remaining = getattr(effect, "remaining_ms", None)
                stacks = getattr(effect, "stacks", None)
                magnitude = getattr(effect, "magnitude", None)
                remaining_known = isinstance(remaining, (int, float)) and not isinstance(remaining, bool)
                stacks_known = isinstance(stacks, int) and not isinstance(stacks, bool)
                magnitude_known = (
                    isinstance(magnitude, (int, float))
                    and not isinstance(magnitude, bool)
                    and math.isfinite(float(magnitude))
                )
                runtime = (
                    min(_finite(remaining, 10_000.0), ACTIVE_EFFECT_REMAINING_CAP_MS / 10_000.0),
                    float(remaining_known),
                    float(remaining is None and getattr(effect, "end_tick", None) is None),
                    _finite(stacks, 10.0),
                    float(stacks_known),
                    _finite(magnitude),
                    float(magnitude_known),
                )
                effect_vocab_id = self.effect_catalog.runtime_vocab_id(effect)
                if self.reject_unknown_public_semantics and effect_vocab_id == UNKNOWN_EFFECT_VOCAB_ID:
                    attributes = getattr(effect, "attributes", {})
                    native_buff_global_id = (
                        attributes.get("native_buff_global_id") if isinstance(attributes, Mapping) else None
                    )
                    raise ValueError(
                        "active public effect resolved to UNKNOWN: "
                        f"effect_id={getattr(effect, 'effect_id', None)!r}, "
                        f"native_buff_global_id={native_buff_global_id!r}, "
                        f"parent_type={parent_type}, parent_index={parent_index}"
                    )
                rows.append(
                    (
                        effect_vocab_id,
                        parent_type,
                        parent_index,
                        self._owner_type(getattr(effect, "source_owner", None)),
                        runtime,
                    )
                )

        for tower in observation.towers:
            row = tower_index_by_entity.get(int(tower.entity_id))
            if row is not None:
                append_effects(tower.effect_states, parent_type=EFFECT_PARENT_TOWER, parent_index=row)
        for entity in observation.entities:
            row = child_index_by_entity.get(int(entity.entity_id))
            if row is not None:
                append_effects(entity.effect_states, parent_type=EFFECT_PARENT_CHILD, parent_index=row)
        rows.sort(key=lambda item: (item[1], item[2], item[0], item[3], item[4]))
        count = len(rows)
        if count == 0:
            return _EMPTY_ACTIVE_EFFECTS
        return ActiveEffectSetV4(
            effect_vocab_id=torch.tensor([[value[0] for value in rows]], dtype=torch.long),
            parent_type=torch.tensor([[value[1] for value in rows]], dtype=torch.long),
            parent_index=torch.tensor([[value[2] for value in rows]], dtype=torch.long),
            source_owner_type=torch.tensor([[value[3] for value in rows]], dtype=torch.long),
            runtime_features=torch.tensor([[value[4] for value in rows]], dtype=torch.float32),
            mask=torch.ones(1, count, dtype=torch.bool),
        )

    def _entity_archetype_id(self, entity: EntityStateV1) -> int:
        cache_key = int(entity.entity_id)
        cached = self._frame_archetype_id_cache.get(cache_key)
        if cached is not None:
            return cached
        projectile = entity.projectile_state
        global_id: object = None
        if projectile is not None:
            global_id = projectile.attributes.get("native_projectile_data_global_id")
        if isinstance(global_id, bool) or not isinstance(global_id, int):
            global_id = entity.native_data_global_id
        if isinstance(global_id, int) and not isinstance(global_id, bool):
            # Native EntityKind is presentation telemetry and is not a reliable
            # LogicData namespace discriminator: live CharacterData and
            # AreaEffectData objects can both arrive labelled as another kind.
            result = self.entity_archetype_catalog.runtime_global_vocab_id(global_id)
        else:
            evolution = entity.evolution_state
            if evolution is not None and evolution.current_form_id:
                result = self.entity_archetype_catalog.form_vocab_id(evolution.current_form_id)
            else:
                result = UNKNOWN_ENTITY_ARCHETYPE_VOCAB_ID
        if self.reject_unknown_public_semantics and result == UNKNOWN_ENTITY_ARCHETYPE_VOCAB_ID:
            projectile_global_id = (
                projectile.attributes.get("native_projectile_data_global_id") if projectile is not None else None
            )
            raise ValueError(
                "public entity resolved to UNKNOWN archetype: "
                f"entity_id={entity.entity_id}, entity_kind={entity.entity_kind!r}, "
                f"card_id={entity.card_id!r}, native_data_global_id="
                f"{entity.native_data_global_id!r}, projectile_global_id="
                f"{projectile_global_id!r}"
            )
        self._frame_archetype_id_cache[cache_key] = result
        return result

    def _entity_static_damage(self, entity: EntityStateV1) -> tuple[float, bool]:
        metadata = self.entity_archetype_catalog.metadata_for_vocab_id(self._entity_archetype_id(entity))
        return metadata.static_damage_basis, metadata.static_damage_basis_known

    def _entity_child_kind(self, entity: EntityStateV1) -> str:
        cache_key = int(entity.entity_id)
        cached = self._frame_child_kind_cache.get(cache_key)
        if cached is not None:
            return cached
        metadata = self.entity_archetype_catalog.metadata_for_vocab_id(self._entity_archetype_id(entity))
        if not metadata.child_kind_known:
            result = entity.entity_kind
        elif metadata.child_kind in {"building", "projectile", "area"}:
            result = metadata.child_kind
        elif metadata.child_kind == "character" and entity.entity_kind not in {"troop", "hero"}:
            result = "character"
        else:
            result = entity.entity_kind
        self._frame_child_kind_cache[cache_key] = result
        return result

    def _events(
        self,
        observation: ObservationV1,
        *,
        group_index_by_entity: Mapping[int, int],
        tower_token_by_entity: Mapping[int, int],
    ) -> tuple[EventSetV4, int, int]:
        entity_by_id = {int(entity.entity_id): entity for entity in observation.entities}
        group_token_start = self.config.max_towers + self.config.max_own_cards + self.config.max_opponent_cards

        def causal_address(entity_id: int | None) -> tuple[str, str] | None:
            if entity_id is None:
                return None
            entity = entity_by_id.get(int(entity_id))
            if entity is None or entity.causal_group is None:
                return ("entity", str(int(entity_id)))
            return (entity.causal_group.kind.value, entity.causal_group.handle)

        def target_address(entity_id: int | None) -> tuple[str, int] | None:
            if entity_id is None:
                return None
            target_id = int(entity_id)
            if target_id in tower_token_by_entity:
                return ("token", tower_token_by_entity[target_id])
            if target_id in group_index_by_entity:
                return ("token", group_token_start + group_index_by_entity[target_id])
            return ("entity", target_id)

        grouped: dict[tuple[object, ...], list[object]] = {}
        affected_target_events = {
            CombatEventKind.SHIELD_GAIN.value,
            CombatEventKind.SHIELD_DAMAGE.value,
            CombatEventKind.SHIELD_BREAK.value,
            CombatEventKind.EFFECT_APPLY.value,
            CombatEventKind.EFFECT_REFRESH.value,
            CombatEventKind.EFFECT_STACK.value,
            CombatEventKind.EFFECT_REMOVE.value,
            CombatEventKind.VISIBILITY_CHANGE.value,
        }
        targeting_events = {
            CombatEventKind.TARGET_ACQUIRE.value,
            CombatEventKind.TARGET_CHANGE.value,
            CombatEventKind.TARGET_LOSE.value,
        }
        interval_start_tick = observation.tick - self.config.decision_ticks
        for event in observation.events:
            if event.tick > observation.tick:
                raise ValueError("observation contains a future event")
            if event.tick <= interval_start_tick:
                continue
            if not event.visible:
                continue
            raw_event_type = str(event.event_type)
            if raw_event_type in IGNORED_EVENT_TYPES:
                continue
            canonical_event_type = EVENT_TYPE_ALIASES.get(raw_event_type, raw_event_type)
            if canonical_event_type not in EVENT_TYPE:
                raise ValueError(f"public event type {raw_event_type!r} is outside the V4 catalog")
            combat = event.combat
            # A typed combat payload is authoritative even when its source is
            # unknown.  In particular, FAIR snapshot damage identifies the
            # victim in ``event.entity_id`` and must not relabel it as attacker.
            if combat is not None:
                source_entity = combat.source_entity
                target_entity = combat.target_entity
                card_id = combat.source_card_id
            elif canonical_event_type in affected_target_events:
                raw_source = event.data.get("source_entity")
                source_entity = (
                    int(raw_source) if isinstance(raw_source, int) and not isinstance(raw_source, bool) else None
                )
                target_entity = event.entity_id
                source_state = entity_by_id.get(source_entity) if source_entity is not None else None
                card_id = self._base_card_id(source_state) if source_state is not None else None
            else:
                source_entity = event.entity_id
                raw_target = event.data.get("target_entity")
                target_entity = (
                    int(raw_target)
                    if canonical_event_type in targeting_events
                    and isinstance(raw_target, int)
                    and not isinstance(raw_target, bool)
                    else None
                )
                card_id = event.card_id
            source_state = next(
                (
                    entity_by_id[int(entity_id)]
                    for entity_id in (source_entity, event.entity_id)
                    if entity_id is not None and int(entity_id) in entity_by_id
                ),
                None,
            )
            if (
                card_id is not None
                and self.catalog.vocab_id(int(card_id)) == UNKNOWN_CARD_VOCAB_ID
                and any(
                    entity_id is not None and int(entity_id) in tower_token_by_entity
                    for entity_id in (source_entity, event.entity_id)
                )
            ):
                # Native tower events expose the tower CharacterData ID in
                # ``card_id``.  The source is already represented by its tower
                # token; that internal ID is not an off-scope playable card.
                card_id = None
            if (
                card_id is not None
                and not (POLICY_CARD_GLOBAL_ID_MIN <= int(card_id) < POLICY_CARD_GLOBAL_ID_MAX_EXCLUSIVE)
                and source_state is not None
            ):
                inherited = self._entity_source_card_id(source_state)
                if inherited is not None and self.catalog.vocab_id(inherited) != UNKNOWN_CARD_VOCAB_ID:
                    card_id = inherited
            if card_id is not None and self.catalog.vocab_id(int(card_id)) == UNKNOWN_CARD_VOCAB_ID:
                normalized = self._normalize_public_card_id(int(card_id))
                if normalized is not None and self.catalog.vocab_id(normalized) != UNKNOWN_CARD_VOCAB_ID:
                    card_id = normalized
                if self.catalog.vocab_id(int(card_id)) == UNKNOWN_CARD_VOCAB_ID:
                    normalized = self._entity_source_card_id(source_state) if source_state is not None else None
                    if normalized is not None and self.catalog.vocab_id(normalized) != UNKNOWN_CARD_VOCAB_ID:
                        card_id = normalized
                    elif not (POLICY_CARD_GLOBAL_ID_MIN <= int(card_id) < POLICY_CARD_GLOBAL_ID_MAX_EXCLUSIVE):
                        card_id = None
            card_vocab_id = self._vocab_id(card_id, context=f"public event {event.event_type} source card")
            source_form = FORM_NORMAL
            form_known = False
            raw_form_code = event.data.get("form_code")
            if type(raw_form_code) is int and not (0 <= int(raw_form_code) < self.config.form_type_count):
                raise ValueError("public event form is outside the V4 vocabulary")
            if combat is not None and combat.evolution_form_id:
                source_form = FORM_EVOLUTION
                form_known = True
            elif type(raw_form_code) is int:
                source_form = int(raw_form_code)
                form_known = True
            elif source_entity is not None:
                source_state = entity_by_id.get(int(source_entity))
                if source_state is not None:
                    source_form = self._runtime_form(source_state)
                    form_known = True
            source_group_index = group_index_by_entity.get(int(source_entity), -1) if source_entity is not None else -1
            target = target_address(target_entity)
            target_token_index = int(target[1]) if target is not None and target[0] == "token" else -1
            action_identity = event.data.get("action_id")
            if action_identity is None and combat is not None:
                action_identity = combat.attributes.get("cause_event_id")
            if action_identity is None and combat is not None:
                action_identity = combat.attributes.get("deployment_sequence")
            key = (
                observation.tick,
                canonical_event_type,
                self._owner_type(event.owner),
                card_vocab_id,
                source_form,
                form_known,
                causal_address(source_entity),
                target,
                action_identity,
            )
            grouped.setdefault(key, []).append(
                (event, source_entity, target_entity, source_group_index, target_token_index)
            )
        ordered = sorted(
            grouped.items(),
            key=lambda item: (
                max(value[0].tick for value in item[1]),
                str(item[0][1]),
                int(item[0][2]),
                int(item[0][3]),
                int(item[1][0][3]),
                int(item[1][0][4]),
            ),
            reverse=True,
        )
        selected = ordered[: self.config.max_recent_events]
        overflow = len(ordered) - len(selected)
        count = self.config.max_recent_events
        arena = _NumpyTensorArena(
            float_count=count * self.config.event_feature_dim, int_count=count * 6, bool_count=count
        )
        event_type = arena.ints(1, count)
        owner_type = arena.ints(1, count)
        owner_type.fill(OWNER_NEUTRAL)
        source_card = arena.ints(1, count)
        source_form = arena.ints(1, count)
        source_group = arena.ints(1, count)
        source_group.fill(-1)
        target_token = arena.ints(1, count)
        target_token.fill(-1)
        features = arena.floats(1, count, self.config.event_feature_dim)
        mask = arena.bools(1, count)
        for row, (key, values) in enumerate(selected):
            events = [value[0] for value in values]
            latest = max(event.tick for event in events)
            earliest = min(event.tick for event in events)
            amounts = [
                float(event.combat.amount)
                for event in events
                if event.combat is not None and event.combat.amount is not None
            ]
            if not amounts and str(key[1]) in affected_target_events:
                amounts = [
                    float(event.data["amount"])
                    for event in events
                    if isinstance(event.data.get("amount"), (int, float))
                    and not isinstance(event.data.get("amount"), bool)
                ]
            event_positions = [
                event.position if event.position is not None else event.combat.position
                for event in events
                if event.position is not None or (event.combat is not None and event.combat.position is not None)
            ]
            positions = [self._tile_position(position) for position in event_positions if position is not None]
            event_type[0, row] = EVENT_TYPE[str(key[1])]
            owner_type[0, row] = int(key[2])
            source_card[0, row] = int(key[3])
            source_form[0, row] = int(key[4])
            source_group[0, row] = int(values[0][3])
            target_token[0, row] = int(values[0][4])
            features[0, row, 0] = max(0, observation.tick - latest) / self.config.decision_ticks
            features[0, row, 1] = sum(amounts) / 5_000.0
            features[0, row, 2] = float(any(value[1] is not None for value in values))
            features[0, row, 3] = float(any(value[2] is not None for value in values))
            features[0, row, 4] = len(events) / 16.0
            features[0, row, 5] = max(amounts, default=0.0) / 5_000.0
            if positions:
                features[0, row, 6] = (
                    sum(position[0] for position in positions) / len(positions) / self.config.board_width
                )
                features[0, row, 7] = (
                    sum(position[1] for position in positions) / len(positions) / self.config.board_height
                )
            features[0, row, 8] = len(positions) / len(events)
            features[0, row, 9] = len(amounts) / len(events)
            features[0, row, 10] = (latest - earliest) / self.config.decision_ticks
            features[0, row, 11] = float(bool(key[5]))
            mask[0, row] = True
        return (
            EventSetV4(
                torch.from_numpy(event_type),
                torch.from_numpy(owner_type),
                torch.from_numpy(source_card),
                torch.from_numpy(source_form),
                torch.from_numpy(source_group),
                torch.from_numpy(target_token),
                torch.from_numpy(features),
                torch.from_numpy(mask),
            ),
            len(selected),
            overflow,
        )

    def _placement_rows(self, entry: Mapping[str, object]) -> Tensor:
        rows = entry.get("row_major")
        cache_key = id(rows)
        cached = self._placement_tensor_cache.get(cache_key)
        if cached is not None and cached[0] is rows:
            return cached[1]
        if (
            not isinstance(rows, Sequence)
            or isinstance(rows, (str, bytes, bytearray))
            or len(rows) != 32
            or any(
                not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)) or len(row) != 18
                for row in rows
            )
        ):
            raise ValueError("candidate placement must be row-major [32][18]")
        result = torch.tensor(rows, dtype=torch.bool)
        if len(self._placement_tensor_cache) >= 4096:
            self._placement_tensor_cache.clear()
        self._placement_tensor_cache[cache_key] = (rows, result)
        return result

    def _candidates(
        self,
        observation: ObservationV1,
        *,
        card_to_row: Mapping[int, int],
        group_index_by_entity: Mapping[int, int],
        child_index_by_entity: Mapping[int, int],
    ) -> tuple[ActionCandidatesV4, int]:
        player = next(item for item in observation.players if item.owner == self.actor_owner)
        candidate_rows: list[dict[str, object]] = []
        raw_hand_slots = player.metadata.get("hand_slot_by_card")
        if not isinstance(raw_hand_slots, Mapping):
            raise ValueError("actor hand has no exact model-slot mapping")
        hand_slot_by_card = {int(card_id): int(slot) for card_id, slot in raw_hand_slots.items()}
        if set(hand_slot_by_card) != set(map(int, player.hand)):
            raise ValueError("actor hand and hand-slot mapping disagree")
        for card_id in player.hand:
            model_slot = hand_slot_by_card[int(card_id)]
            if not 0 <= model_slot < len(observation.action_mask.hand_slots):
                raise ValueError("actor hand slot is outside 0..3")
            if not observation.action_mask.hand_slots[model_slot]:
                continue
            entry = observation.action_mask.placement_masks.get(str(model_slot))
            if not isinstance(entry, Mapping):
                raise ValueError("legal hand slot has no exact placement entry")
            native_slot = int(entry.get("source_native_hand_slot", model_slot))
            effective_card_id = int(card_id)
            raw_effective_cost = entry.get("effective_cost")
            if (
                isinstance(raw_effective_cost, bool)
                or not isinstance(raw_effective_cost, (int, float))
                or not math.isfinite(float(raw_effective_cost))
                or not 0.0 <= float(raw_effective_cost) <= 15.0
            ):
                raise ValueError("legal hand slot has no exact runtime cost")
            effective_cost = float(raw_effective_cost)
            if type(entry.get("form_code")) is not int:
                raise ValueError("legal hand slot has no exact runtime form")
            effective_form = int(entry["form_code"])
            if not 0 <= effective_form < self.config.form_type_count:
                raise ValueError("candidate runtime form is outside the V4 vocabulary")
            if int(card_id) == MIRROR_CARD_ID:
                effective_card_id, effective_cost, effective_form = mirror_play_runtime_contract(
                    player=player,
                    hand_slot=native_slot,
                    deck=self.deck,
                    card_costs=self.card_costs,
                    placement_entry=entry,
                )
            elif effective_form != self._own_card_runtime_form(observation, int(card_id)):
                raise ValueError("card token and candidate runtime forms disagree")
            placement = self._placement_rows(entry)
            profile = building_placement_profile(effective_card_id)
            is_building = bool(
                self.card_specs[effective_card_id].kind == CardKind.BUILDING
                and profile is not None
                and profile.stationary_collision_rectangle
            )
            width = _finite(entry.get("footprint_width_tiles"))
            height = _finite(entry.get("footprint_height_tiles"))
            offset = entry.get("model_subcell_offset")
            offset_x = 0.0
            offset_y = 0.0
            if isinstance(offset, Sequence) and not isinstance(offset, (str, bytes, bytearray)) and len(offset) == 2:
                offset_x = float(offset[0])
                offset_y = float(offset[1])
            if is_building and (width <= 0.0 or height <= 0.0):
                raise ValueError("stationary building candidate lacks an exact footprint")
            visible_vocab = self._vocab_id(int(card_id), context="legal deployment visible card")
            effective_vocab = self._vocab_id(effective_card_id, context="legal deployment effective card")
            candidate_rows.append(
                {
                    "uid": candidate_uid_v4(
                        variant=CANDIDATE_DEPLOY,
                        visible_card_vocab_id=visible_vocab,
                        effective_card_vocab_id=effective_vocab,
                        native_hand_slot=native_slot,
                    ),
                    "variant": CANDIDATE_DEPLOY,
                    "visible_card_vocab_id": visible_vocab,
                    "effective_card_vocab_id": effective_vocab,
                    "effective_form": effective_form,
                    "own_card_row": card_to_row[int(card_id)],
                    "source_group_index": -1,
                    "source_child_index": -1,
                    "ability_vocab_id": 0,
                    "cost": effective_cost,
                    "target_mode": TARGET_GRID,
                    "placement": placement,
                    "is_building": is_building,
                    "building_half_width": width / 2.0 if is_building else 0.0,
                    "building_half_height": height / 2.0 if is_building else 0.0,
                    "building_offset_x": offset_x,
                    "building_offset_y": offset_y,
                    "exclusion_group_id": 1000 + native_slot,
                    "native_hand_slot": native_slot,
                    "native_source_entity": -1,
                    "native_visible_card_id": int(card_id),
                }
            )

        for source_entity in observation.action_mask.ability_sources:
            source = int(source_entity)
            runtime_contract = ability_runtime_contract(observation, source_entity=source, deck=self.deck)
            if runtime_contract.target_mode != TargetKind.NONE:
                raise ValueError("current native abilities must be target-free")
            ability_vocab = self.ability_catalog.vocab_id(runtime_contract.ability_id)
            entity = next(item for item in observation.entities if item.entity_id == source)
            visible_vocab = self._vocab_id(runtime_contract.source_card_id, context="legal ability source card")
            candidate_rows.append(
                {
                    "uid": candidate_uid_v4(
                        variant=CANDIDATE_ABILITY,
                        visible_card_vocab_id=visible_vocab,
                        effective_card_vocab_id=visible_vocab,
                        native_source_entity=source,
                        ability_vocab_id=ability_vocab,
                    ),
                    "variant": CANDIDATE_ABILITY,
                    "visible_card_vocab_id": visible_vocab,
                    "effective_card_vocab_id": visible_vocab,
                    "effective_form": self._runtime_form(entity),
                    "own_card_row": card_to_row[runtime_contract.source_card_id],
                    "source_group_index": group_index_by_entity.get(source, -1),
                    "source_child_index": child_index_by_entity.get(source, -1),
                    "ability_vocab_id": ability_vocab,
                    "cost": runtime_contract.cost,
                    "target_mode": TARGET_NONE,
                    "placement": _EMPTY_CANDIDATE_PLACEMENT,
                    "is_building": False,
                    "building_half_width": 0.0,
                    "building_half_height": 0.0,
                    "building_offset_x": 0.0,
                    "building_offset_y": 0.0,
                    "exclusion_group_id": 2000 + runtime_contract.cooldown_group_id,
                    "native_hand_slot": -1,
                    "native_source_entity": source,
                    "native_visible_card_id": -1,
                }
            )
        if len(candidate_rows) > self.config.max_action_candidates:
            raise ValueError("live legal candidate set exceeds V4 capacity")

        count = self.config.max_action_candidates
        arena = _NumpyTensorArena(
            float_count=count * (self.config.candidate_runtime_feature_dim + 5),
            int_count=count * 14,
            bool_count=count * (2 + 32 * 18),
        )
        specifications = (
            ("mask", arena.bools, (), 0),
            ("uid", arena.ints, (), 0),
            ("variant", arena.ints, (), 0),
            ("visible_card_vocab_id", arena.ints, (), 0),
            ("effective_card_vocab_id", arena.ints, (), 0),
            ("effective_form", arena.ints, (), 0),
            ("own_card_row", arena.ints, (), -1),
            ("source_group_index", arena.ints, (), -1),
            ("source_child_index", arena.ints, (), -1),
            ("ability_vocab_id", arena.ints, (), 0),
            ("cost", arena.floats, (), 0),
            ("runtime_features", arena.floats, (self.config.candidate_runtime_feature_dim,), 0),
            ("target_mode", arena.ints, (), 0),
            ("placement", arena.bools, (32, 18), 0),
            ("is_building", arena.bools, (), 0),
            ("building_half_width", arena.floats, (), 0),
            ("building_half_height", arena.floats, (), 0),
            ("building_offset_x", arena.floats, (), 0),
            ("building_offset_y", arena.floats, (), 0),
            ("exclusion_group_id", arena.ints, (), -1),
            ("native_hand_slot", arena.ints, (), -1),
            ("native_source_entity", arena.ints, (), -1),
            ("native_visible_card_id", arena.ints, (), -1),
        )
        arrays = {}
        for name, allocate, tail, padding in specifications:
            value = allocate(1, count, *tail)
            if padding:
                value.fill(padding)
            arrays[name] = value
        scalar_fields = {
            name: (array, {"i": int, "f": float, "b": bool}[array.dtype.kind])
            for name, array in arrays.items()
            if name not in {"mask", "placement", "runtime_features"}
        }
        for row, values in enumerate(candidate_rows):
            arrays["mask"][0, row] = True
            for name, (array, convert) in scalar_fields.items():
                array[0, row] = convert(values[name])
            placement = values["placement"]
            if not isinstance(placement, Tensor):
                raise TypeError("candidate placement must be a Tensor")
            placement_array = placement.numpy()
            arrays["placement"][0, row] = placement_array
            runtime = arrays["runtime_features"][0, row]
            runtime[0] = float(values["cost"]) / 10.0
            runtime[1] = float(placement_array.mean()) if int(values["target_mode"]) == TARGET_GRID else 1.0
        effective_elixir = observation.action_mask.reasons.get("effective_elixir")
        if effective_elixir is None or not math.isfinite(float(effective_elixir)):
            raise ValueError("action mask lacks exact effective elixir")
        return (
            ActionCandidatesV4(
                elixir=torch.tensor([float(effective_elixir)]),
                **{name: torch.from_numpy(array) for name, array in arrays.items()},
            ),
            len(candidate_rows),
        )

    def _relations(
        self,
        observation: ObservationV1,
        *,
        group_index_by_entity: Mapping[int, int],
        tower_token_by_entity: Mapping[int, int],
        own_cards: CardSetV4,
        opponent_cards: CardSetV4,
        groups: BattleGroupSetV4,
    ) -> RelationEdgesV4:
        own_start = self.config.max_towers
        opponent_start = own_start + self.config.max_own_cards
        group_start = self.config.max_towers + self.config.max_own_cards + self.config.max_opponent_cards
        edges: set[tuple[int, int, int]] = set()

        def token(entity_id: int) -> int | None:
            if entity_id in tower_token_by_entity:
                return tower_token_by_entity[entity_id]
            if entity_id in group_index_by_entity:
                return group_start + group_index_by_entity[entity_id]
            return None

        card_tokens: dict[tuple[int, int], int] = {}
        own_mask = own_cards.mask[0].numpy()
        own_card_ids = own_cards.card_vocab_id[0].numpy()
        for raw_row in np.flatnonzero(own_mask):
            row = int(raw_row)
            card_tokens[(OWNER_SELF, int(own_card_ids[row]))] = own_start + row
        opponent_mask = opponent_cards.mask[0].numpy()
        opponent_card_ids = opponent_cards.card_vocab_id[0].numpy()
        for raw_row in np.flatnonzero(opponent_mask):
            row = int(raw_row)
            card_tokens[(OWNER_ENEMY, int(opponent_card_ids[row]))] = opponent_start + row

        for entity in observation.entities:
            source = token(int(entity.entity_id))
            if source is None:
                continue
            target_ids = {
                int(value)
                for value in (
                    entity.visible_target,
                    (entity.attack_state.target_entity if entity.attack_state is not None else None),
                    (entity.projectile_state.target_entity if entity.projectile_state is not None else None),
                )
                if value is not None
            }
            for target_id in target_ids:
                target = token(target_id)
                if target is not None and target != source:
                    edges.add((source, target, REL_TARGETS))
                    edges.add((target, source, REL_TARGETED_BY))
            if entity.capture_runtime is not None:
                for capture_target in entity.capture_runtime.targets:
                    if capture_target.target_entity is None:
                        continue
                    target = token(int(capture_target.target_entity))
                    if target is not None and target != source:
                        edges.add((source, target, REL_CAPTURES))
                        edges.add((target, source, REL_CAPTURED_BY))
            source_ids = {
                int(value)
                for value in (
                    entity.source_entity,
                    (entity.projectile_state.source_entity if entity.projectile_state is not None else None),
                    (
                        entity.periodic_attack_modifier.source_entity
                        if entity.periodic_attack_modifier is not None
                        else None
                    ),
                    *(effect.source_entity for effect in entity.effect_states),
                )
                if value is not None
            }
            for source_id in source_ids:
                parent = token(source_id)
                if parent is not None and parent != source:
                    edges.add((parent, source, REL_SOURCE_OF))
                    edges.add((source, parent, REL_SOURCED_BY))
            for effect in entity.effect_states:
                if effect.kind != EffectKind.TARGETING_MODIFIER or effect.source_entity is None:
                    continue
                modifier = token(int(effect.source_entity))
                if modifier is not None and modifier != source:
                    edges.add((modifier, source, REL_TARGETING_MODIFIER_OF))
                    edges.add((source, modifier, REL_TARGETING_MODIFIED_BY))

            reference = entity.causal_group
            if reference is not None and reference.parent_entity_id is not None:
                child_group = group_index_by_entity.get(int(entity.entity_id))
                parent_group = group_index_by_entity.get(int(reference.parent_entity_id))
                if child_group is not None and parent_group is not None and child_group != parent_group:
                    relation = (
                        REL_DERIVED_FROM_VOLLEY if reference.kind == CausalGroupKind.VOLLEY else REL_SPAWNED_BY_GROUP
                    )
                    edges.add((group_start + child_group, group_start + parent_group, relation))

        for tower in observation.towers:
            source = token(int(tower.entity_id))
            if source is None:
                continue
            target_ids = {
                int(value)
                for value in (
                    tower.visible_target,
                    (tower.attack_state.target_entity if tower.attack_state is not None else None),
                )
                if value is not None
            }
            for target_id in target_ids:
                target = token(target_id)
                if target is not None and target != source:
                    edges.add((source, target, REL_TARGETS))
                    edges.add((target, source, REL_TARGETED_BY))
            tower_runtime = tower.tower_troop_runtime
            if isinstance(tower_runtime, RoyalChefRuntimeStateV1) and tower_runtime.target_entity is not None:
                support_target = token(int(tower_runtime.target_entity))
                if support_target is not None and support_target != source:
                    edges.add((source, support_target, REL_TOWER_TROOP_SUPPORTS))
                    edges.add((support_target, source, REL_TOWER_TROOP_SUPPORTED_BY))

        group_mask = groups.mask[0].numpy()
        group_owners = groups.owner_type[0].numpy()
        group_source_cards = groups.source_card_vocab_id[0].numpy()
        for raw_group_row in np.flatnonzero(group_mask):
            group_row = int(raw_group_row)
            key = (int(group_owners[group_row]), int(group_source_cards[group_row]))
            card_token = card_tokens.get(key)
            if card_token is not None:
                edges.add((card_token, group_start + group_row, REL_CARD_REPRESENTS_GROUP))

        ordered = sorted(edges)
        count = max(1, len(ordered))
        arena = _NumpyTensorArena(int_count=count * 3, bool_count=count)
        source = arena.ints(1, count)
        target = arena.ints(1, count)
        relation = arena.ints(1, count)
        mask = arena.bools(1, count)
        for row, (edge_source, edge_target, edge_type) in enumerate(ordered):
            source[0, row] = edge_source
            target[0, row] = edge_target
            relation[0, row] = edge_type
            mask[0, row] = True
        return RelationEdgesV4(
            torch.from_numpy(source), torch.from_numpy(target), torch.from_numpy(relation), torch.from_numpy(mask)
        )

    def _entity_is_airborne(self, entity: EntityStateV1) -> bool:
        archetype = self.entity_archetype_catalog.metadata_for_vocab_id(self._entity_archetype_id(entity))
        return archetype.is_airborne if archetype.is_airborne_known else False

    def _entity_threat(self, entity: EntityStateV1) -> float:
        if entity.projectile_state is not None and entity.projectile_state.damage is not None:
            return float(entity.projectile_state.damage)
        damage, known = self._entity_static_damage(entity)
        return damage if known else 0.0

    def _spatial_planes(self, observation: ObservationV1) -> Tensor:
        planes = np.zeros(
            (1, self.config.explicit_spatial_channels, self.config.board_height, self.config.board_width),
            dtype=np.float32,
        )
        for entity in observation.entities:
            if entity.owner is None:
                continue
            kind = self._entity_child_kind(entity)
            if self._entity_is_persistent_area_root(entity):
                continue
            x, y = self._tile_position(entity.position)
            cell_x = min(self.config.board_width - 1, max(0, int(x)))
            cell_y = min(self.config.board_height - 1, max(0, int(y)))
            base = 0 if entity.owner == self.actor_owner else 7
            projectile = kind == "projectile"
            building = kind == "building"
            category = 3 if projectile else 2 if building else 1 if self._entity_is_airborne(entity) else 0
            footprint = self._entity_spatial_extent(entity) if building else (x, y, x, y)
            footprint_x, footprint_y = self._extent_cell_ranges(footprint)
            if building and footprint_x and footprint_y:
                for occupied_y in footprint_y:
                    for occupied_x in footprint_x:
                        planes[0, base + category, occupied_y, occupied_x] += 1.0
            else:
                planes[0, base + category, cell_y, cell_x] += 1.0
            planes[0, base + 4, cell_y, cell_x] += _finite(entity.hitpoints, 10_000.0)
            planes[0, base + 5, cell_y, cell_x] += _finite(entity.shield, 10_000.0)
            planes[0, base + 6, cell_y, cell_x] += self._entity_threat(entity) / 5_000.0
        return torch.from_numpy(planes)

    def _scalars(
        self,
        observation: ObservationV1,
        *,
        group_overflow: int,
        child_overflow: int,
        event_count: int,
        event_overflow: int,
        candidate_count: int,
    ) -> Tensor:
        result = np.zeros((1, self.config.generic_scalar_dim), dtype=np.float32)
        players = {player.owner: player for player in observation.players}
        own = players[self.actor_owner]
        enemy = players[1 - self.actor_owner]
        result[0, 0] = _finite(observation.time.remaining_ms, 300_000.0)
        result[0, 1] = _finite(observation.time.elixir_multiplier, 3.0)
        result[0, 2] = float(observation.phase in {"overtime", "sudden_death"})
        result[0, 3] = float(observation.phase == "tiebreak")
        result[0, 4] = _finite(own.elixir_exact, 10.0)
        result[0, 5] = own.crowns / 3.0
        result[0, 6] = enemy.crowns / 3.0
        result[0, 7] = len(observation.entities) / self.config.max_children_total
        result[0, 8] = event_count / self.config.max_recent_events
        result[0, 9] = group_overflow / self.config.max_battle_groups
        result[0, 10] = child_overflow / self.config.max_children_total
        producer_overflow = observation.metadata.get("entity_overflow_count", 0)
        result[0, 11] = _finite(producer_overflow, self.config.max_children_total)
        result[0, 12] = event_overflow / self.config.max_recent_events
        result[0, 13] = candidate_count / self.config.max_action_candidates
        result[0, 14] = _finite(observation.action_mask.reasons.get("reserved_elixir"), 10.0)
        # Without a tracker the only sound public bound is the full [0, 10]
        # interval.  Encoding it as (upper=1, uncertainty=1) keeps unknown
        # distinct from a tracker-proven exact zero without widening the schema.
        result[0, 15] = 1.0
        result[0, 16] = 1.0
        if self.tracker is not None:
            enemy_low, enemy_high = self.tracker.elixir_interval(1 - self.actor_owner)
            result[0, 15] = enemy_high / 10.0
            result[0, 16] = max(0.0, enemy_high - enemy_low) / 10.0
        # Native owner 0 is absolute True Red and owner 1 is True Blue.
        # This episode-static identity intentionally survives perspective and
        # horizontal-mirror normalization as an explicit model input.
        result[0, 17] = float(self.actor_owner == 0)
        return torch.from_numpy(result)

    def tensorize(self, observation: ObservationV1, *, validate: bool = True) -> UniversalSemanticBatchV4:
        self._validate_episode_step(observation)
        if self.tracker is not None:
            self.tracker.update(observation)
            self.tracker.reconcile_owner_private_state(observation)
        actor = next(player for player in observation.players if player.owner == self.actor_owner)
        self._ability_state_by_source_entity = {
            int(state.source_entity): state for state in actor.ability_runtime_states if state.source_entity is not None
        }
        model_observation = self.perspective.observation_policy_metadata_to_model(observation)
        self._frame_tile_position_cache.clear()
        self._frame_base_card_id_cache.clear()
        self._frame_spatial_extent_cache.clear()
        self._frame_spatial_radius_cache.clear()
        self._frame_archetype_id_cache.clear()
        self._frame_child_kind_cache.clear()
        self._remember_public_evolution_forms(model_observation)
        own_cards, opponent_cards, card_to_row = self._card_sets(model_observation)
        towers, tower_token_by_entity = self._towers(model_observation)
        (groups, group_index_by_entity, child_index_by_entity, group_overflow, child_overflow) = self._groups(
            model_observation
        )
        active_effects = self._active_effects(
            model_observation, child_index_by_entity=child_index_by_entity, tower_index_by_entity=tower_token_by_entity
        )
        events, event_count, event_overflow = self._events(
            model_observation, group_index_by_entity=group_index_by_entity, tower_token_by_entity=tower_token_by_entity
        )
        candidates, candidate_count = self._candidates(
            model_observation,
            card_to_row=card_to_row,
            group_index_by_entity=group_index_by_entity,
            child_index_by_entity=child_index_by_entity,
        )
        relations = self._relations(
            model_observation,
            group_index_by_entity=group_index_by_entity,
            tower_token_by_entity=tower_token_by_entity,
            own_cards=own_cards,
            opponent_cards=opponent_cards,
            groups=groups,
        )
        batch = UniversalSemanticBatchV4(
            match_scalars=self._scalars(
                model_observation,
                group_overflow=group_overflow,
                child_overflow=child_overflow,
                event_count=event_count,
                event_overflow=event_overflow,
                candidate_count=candidate_count,
            ),
            own_cards=own_cards,
            opponent_cards=opponent_cards,
            towers=towers,
            groups=groups,
            active_effects=active_effects,
            events=events,
            relation_edges=relations,
            explicit_spatial_planes=self._spatial_planes(model_observation),
            # Episode updates replace this record wholesale; they never mutate
            # its tensors.  The existing record is therefore already an
            # immutable snapshot for this batch and needs no per-field clone.
            previous_action=self._previous_action,
            candidates=candidates,
        )
        if validate:
            batch.validate(
                self.config,
                card_vocab_size=self.catalog.vocab_size,
                ability_vocab_size=self.ability_catalog.vocab_size,
                entity_archetype_vocab_size=(self.entity_archetype_catalog.vocab_size),
                effect_vocab_size=self.effect_catalog.vocab_size,
            )
        self._last_tensorized_tick = int(observation.tick)
        return batch
