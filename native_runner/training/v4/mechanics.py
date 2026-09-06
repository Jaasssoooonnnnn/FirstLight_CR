"""Reviewed competitive mechanics facts and their unchanged V4 tensor encoding.

Operation paths and conditions are categorical model features, never executable
rules. Their original ordering remains part of the checkpoint contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor

from ...card_logic import StaticCardLogicCatalogV1
from ...contracts import CardSpecV1, content_hash
from ...effect_catalog import NativeEffectCatalogV1
from ..card_features import _compiled_fields, _require_compiled_source, _require_compiled_specs
from .catalog import PAD_ENTITY_ARCHETYPE_VOCAB_ID, AbilityCatalogV1, CardCatalogV1, EntityArchetypeCatalogV1

PAD_EFFECT_VOCAB_ID = 0


UNKNOWN_EFFECT_VOCAB_ID = 1


FIRST_REAL_EFFECT_VOCAB_ID = 2


PAD_MECHANIC_PROFILE_ID = 0


EFFECT_TAG_NAMES = (
    "slow",
    "haste",
    "rage",
    "stun",
    "freeze",
    "movement_lock",
    "attack_lock",
    "spawn_lock",
    "invisibility",
    "minimum_hp",
    "periodic_damage",
    "periodic_heal",
    "damage_multiplier",
    "damage_reduction",
    "damage_immunity",
    "pushback_immunity",
    "attraction",
    "target_reset",
    "target_lock_control",
    "charge",
    "curse",
    "periodic_spawn",
    "clone",
    "projectile_override",
    "remove_on_attack",
    "on_start_action",
    "on_remove_action",
    "stackable",
    "opaque_native_buff",
)


EFFECT_NUMERIC_FEATURE_NAMES = (
    "move_speed_delta",
    "hit_speed_delta",
    "spawn_speed_delta",
    "damage_multiplier_delta",
    "damage_reduction_ratio",
    "damage_per_second_over_1000",
    "damage_per_second_known",
    "heal_per_second_over_1000",
    "attraction_over_500",
    "push_speed_over_500",
    "crown_tower_damage_delta",
    "override_charge_range_over_10000",
    "allowed_overheal_ratio",
    "building_damage_ratio",
    "character_crown_tower_damage_ratio",
)


EFFECT_SEMANTIC_FEATURE_NAMES = EFFECT_TAG_NAMES + EFFECT_NUMERIC_FEATURE_NAMES


ACTIVE_EFFECT_REMAINING_CAP_MS = 300_000


ACTIVE_EFFECT_RUNTIME_FEATURE_NAMES = (
    "remaining_over_10000ms_capped_at_300000ms",
    "remaining_known",
    "non_expiring",
    "stacks_over_10",
    "stacks_known",
    "magnitude",
    "magnitude_known",
)


MECHANIC_TAG_NAMES = (
    "grounding",
    "pull",
    "knock_up",
    "knockback",
    "taunt",
    "target_reset",
    "retarget",
    "reflect",
    "parry",
    "damage_multiplier",
    "charge",
    "dash",
    "jump",
    "warp",
    "recoil",
    "revive",
    "spawn",
    "death_spawn",
    "periodic_spawn",
    "clone",
    "splash",
    "multi_projectile",
    "pierce",
    "chain",
    "bounce_return",
    "persistent_area",
    "periodic_damage",
    "periodic_heal",
    "heal",
    "transform",
    "projectile_override",
    "active_ability",
    "ammo",
    "target_lock",
    "damage_ramp",
    "attack_sequence",
    "variable_damage_stage",
    "periodic_support",
    "projectile",
    "area_effect",
    "attachment",
    "scheduled_spawn",
    "self_destruct",
    "non_deflectable",
    "damage_immunity",
    "pushback_immunity",
    "target_seeking",
    "non_expiring",
    "capture",
    "visibility_transition",
    "movement_modifier",
    "phase_change",
    "ability_progress",
    "effect_immunity",
)


MECHANIC_NUMERIC_FEATURE_NAMES = (
    "count_or_resource_over_10",
    "duration_over_10000ms",
    "interval_over_5000ms",
    "extent_over_10000_units",
    "range_min_over_10000_units",
    "range_max_over_10000_units",
    "amount_over_1000",
    "speed_over_1000",
    "acceleration_over_1000",
    "angle_over_360deg",
    "ratio_over_100",
    "interaction_scalar_over_1000",
    "offset_x_over_10000_units",
    "offset_y_over_10000_units",
    "ordinal_over_10",
)


MECHANIC_GUARD_LIMIT = 4


@dataclass(frozen=True, slots=True)
class EffectSemanticCatalogV1:
    """Source-release Buff identities and shared exact semantic rows."""

    card_scope: tuple[int, ...]
    buff_global_ids: tuple[int, ...]
    effect_names: tuple[str, ...]
    semantic_features: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        scope = tuple(int(value) for value in self.card_scope)
        ids = tuple(int(value) for value in self.buff_global_ids)
        names = tuple(str(value) for value in self.effect_names)
        rows = tuple(tuple(float(value) for value in row) for row in self.semantic_features)
        if tuple(sorted(set(scope))) != scope:
            raise ValueError("Effect catalog card scope must be unique and sorted")
        if len(ids) != len(names) or len(ids) != len(rows):
            raise ValueError("Effect catalog identities and feature rows differ")
        if tuple(sorted(set(ids))) != ids or any(value <= 0 for value in ids):
            raise ValueError("Effect Buff global IDs must be positive, unique, and sorted")
        if len(set(names)) != len(names) or any(not value for value in names):
            raise ValueError("Effect names must be non-empty and unique")
        if any(len(row) != len(EFFECT_SEMANTIC_FEATURE_NAMES) for row in rows):
            raise ValueError("Effect semantic rows do not match their named schema")
        if any(not math.isfinite(value) for row in rows for value in row):
            raise ValueError("Effect semantic rows must be finite")
        object.__setattr__(self, "card_scope", scope)
        object.__setattr__(self, "buff_global_ids", ids)
        object.__setattr__(self, "effect_names", names)
        object.__setattr__(self, "semantic_features", rows)

    @classmethod
    def from_native_catalog(
        cls,
        card_specs: Mapping[int, CardSpecV1],
        *,
        static_logic: StaticCardLogicCatalogV1,
        native_effect_catalog: NativeEffectCatalogV1,
    ) -> "EffectSemanticCatalogV1":
        _require_compiled_specs(card_specs, complete=True)
        _require_compiled_source("card_logic", static_logic)
        _require_compiled_source("effects", native_effect_catalog)
        return cls(**_compiled_fields("effect_catalog"))

    @classmethod
    def empty(cls, card_scope: Sequence[int]) -> "EffectSemanticCatalogV1":
        return cls(tuple(sorted(int(value) for value in card_scope)), (), (), ())

    @property
    def vocab_size(self) -> int:
        return FIRST_REAL_EFFECT_VOCAB_ID + len(self.buff_global_ids)

    @property
    def semantic_dim(self) -> int:
        return len(EFFECT_SEMANTIC_FEATURE_NAMES)

    @property
    def catalog_id(self) -> str:
        return content_hash(
            {
                "version": "effect-semantic-catalog.v1",
                "feature_names": EFFECT_SEMANTIC_FEATURE_NAMES,
                "card_scope": self.card_scope,
                "buff_global_ids": self.buff_global_ids,
                "effect_names": self.effect_names,
                "semantic_features": self.semantic_features,
            }
        )

    def vocab_id(self, buff_global_id: int) -> int:
        try:
            return FIRST_REAL_EFFECT_VOCAB_ID + self.buff_global_ids.index(int(buff_global_id))
        except ValueError:
            return UNKNOWN_EFFECT_VOCAB_ID

    def runtime_vocab_id(self, effect: object) -> int:
        attributes = getattr(effect, "attributes", {})
        global_id = attributes.get("native_buff_global_id") if isinstance(attributes, Mapping) else None
        if global_id is None:
            raw_id = str(getattr(effect, "effect_id", ""))
            if raw_id.startswith("buff:"):
                try:
                    global_id = int(raw_id.split(":", 1)[1])
                except ValueError:
                    return UNKNOWN_EFFECT_VOCAB_ID
        if isinstance(global_id, bool) or not isinstance(global_id, int):
            return UNKNOWN_EFFECT_VOCAB_ID
        return self.vocab_id(global_id)

    def feature_tensor(self, *, dtype: torch.dtype = torch.float32, device: torch.device | str | None = None) -> Tensor:
        rows = ((0.0,) * self.semantic_dim,) + ((0.0,) * self.semantic_dim,) + self.semantic_features
        return torch.tensor(rows, dtype=dtype, device=device)


@dataclass(frozen=True, slots=True)
class MechanicOperationV1:
    trigger: str
    operation: str
    target: str
    mechanic_tags: tuple[str, ...] = ()
    effect_vocab_id: int = PAD_EFFECT_VOCAB_ID
    produced_archetype_id: int = PAD_ENTITY_ARCHETYPE_VOCAB_ID
    numeric_features: tuple[float, ...] = (0.0,) * len(MECHANIC_NUMERIC_FEATURE_NAMES)
    source_archetype: str = "unknown"
    control_path: str = "root"
    condition: str = "always"
    amount_basis: str = "none"
    guard_archetypes: tuple[str, ...] = ()
    guard_polarities: tuple[int, ...] = ()
    guard_numeric: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        if not self.trigger or not self.operation or not self.target:
            raise ValueError("mechanic operation categorical fields cannot be empty")
        if self.effect_vocab_id < 0 or self.produced_archetype_id < 0:
            raise ValueError("mechanic operation vocabulary IDs cannot be negative")
        if not all(
            isinstance(value, str) and value
            for value in (self.source_archetype, self.control_path, self.condition, self.amount_basis)
        ):
            raise ValueError("mechanic operation context fields cannot be empty")
        tags = tuple(sorted(set(str(value) for value in self.mechanic_tags)))
        if not set(tags).issubset(MECHANIC_TAG_NAMES):
            raise ValueError("mechanic operation contains an unknown semantic tag")
        numeric = tuple(float(value) for value in self.numeric_features)
        if len(numeric) != len(MECHANIC_NUMERIC_FEATURE_NAMES) or any(not math.isfinite(value) for value in numeric):
            raise ValueError("mechanic operation numeric row is invalid")
        guard_archetypes = tuple(str(value) for value in self.guard_archetypes)
        guard_polarities = tuple(int(value) for value in self.guard_polarities)
        guard_numeric = tuple((float(value[0]), float(value[1])) for value in self.guard_numeric)
        if not (len(guard_archetypes) == len(guard_polarities) == len(guard_numeric) <= MECHANIC_GUARD_LIMIT):
            raise ValueError("mechanic guard rows do not align")
        if any(not value for value in guard_archetypes) or any(value not in {-1, 1} for value in guard_polarities):
            raise ValueError("mechanic guard categoricals are invalid")
        if any(not math.isfinite(number) for pair in guard_numeric for number in pair):
            raise ValueError("mechanic guard numeric values must be finite")
        object.__setattr__(self, "mechanic_tags", tags)
        object.__setattr__(self, "numeric_features", numeric)
        object.__setattr__(self, "guard_archetypes", guard_archetypes)
        object.__setattr__(self, "guard_polarities", guard_polarities)
        object.__setattr__(self, "guard_numeric", guard_numeric)


@dataclass(frozen=True, slots=True)
class MechanicProfileCatalogV1:
    """Structured operation lists addressed by form, Ability, and archetype."""

    card_scope: tuple[int, ...]
    profile_keys: tuple[str, ...]
    profile_operations: tuple[tuple[MechanicOperationV1, ...], ...]
    card_form_profile_ids: tuple[tuple[int, int, int, int], ...]
    ability_profile_ids: tuple[int, ...]
    archetype_profile_ids: tuple[int, ...]
    tower_troop_profile_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        scope = tuple(int(value) for value in self.card_scope)
        keys = tuple(str(value) for value in self.profile_keys)
        operations = tuple(tuple(row) for row in self.profile_operations)
        if tuple(sorted(set(scope))) != scope:
            raise ValueError("MechanicProfile card scope must be unique and sorted")
        if len(keys) != len(operations) or any(not value for value in keys):
            raise ValueError("Mechanic profile keys and operation rows differ")
        if len(set(keys)) != len(keys):
            raise ValueError("Mechanic profile keys must be unique")
        profile_count = 1 + len(keys)
        mappings = (value for row in self.card_form_profile_ids for value in row)
        mappings = (*mappings, *self.ability_profile_ids, *self.archetype_profile_ids, *self.tower_troop_profile_ids)
        if any(not 0 <= int(value) < profile_count for value in mappings):
            raise ValueError("Mechanic profile mapping is outside the catalog")
        if any(len(row) != 4 for row in self.card_form_profile_ids):
            raise ValueError("Card mechanic profiles require four runtime forms")
        object.__setattr__(self, "card_scope", scope)
        object.__setattr__(self, "profile_keys", keys)
        object.__setattr__(self, "profile_operations", operations)

    @classmethod
    def from_compiled(cls) -> "MechanicProfileCatalogV1":
        """Load the fixed catalog after verifying its source and data identities."""
        data = _compiled_fields("mechanic_profile_catalog")
        data["profile_operations"] = tuple(
            tuple(MechanicOperationV1(**operation) for operation in row) for row in data["profile_operations"]
        )
        return cls(**data)

    @classmethod
    def empty(
        cls,
        *,
        card_catalog: CardCatalogV1,
        ability_catalog: AbilityCatalogV1,
        entity_archetype_catalog: EntityArchetypeCatalogV1,
    ) -> "MechanicProfileCatalogV1":
        return cls(
            card_catalog.raw_card_ids,
            (),
            (),
            tuple((0, 0, 0, 0) for _ in range(card_catalog.vocab_size)),
            tuple(0 for _ in range(ability_catalog.vocab_size)),
            tuple(0 for _ in range(entity_archetype_catalog.vocab_size)),
            (0, 0, 0, 0, 0),
        )

    @property
    def profile_count(self) -> int:
        return 1 + len(self.profile_keys)

    @property
    def catalog_id(self) -> str:
        return content_hash(
            {
                "version": "mechanic-profile-catalog.v1",
                "effect_tags": EFFECT_TAG_NAMES,
                "mechanic_tags": MECHANIC_TAG_NAMES,
                "numeric_features": MECHANIC_NUMERIC_FEATURE_NAMES,
                "guard_limit": MECHANIC_GUARD_LIMIT,
                "card_scope": self.card_scope,
                "profile_keys": self.profile_keys,
                "profile_operations": self.profile_operations,
                "card_form_profile_ids": self.card_form_profile_ids,
                "ability_profile_ids": self.ability_profile_ids,
                "archetype_profile_ids": self.archetype_profile_ids,
                "tower_troop_profile_ids": self.tower_troop_profile_ids,
            }
        )

    def tensor_tables(self) -> dict[str, Tensor | tuple[str, ...]]:
        flat = [
            (profile_index, operation)
            for profile_index, operations in enumerate(self.profile_operations, start=1)
            for operation in operations
        ]
        trigger_keys = tuple(sorted({operation.trigger for _, operation in flat}))
        operation_keys = tuple(sorted({operation.operation for _, operation in flat}))
        target_keys = tuple(sorted({operation.target for _, operation in flat}))
        source_keys = tuple(sorted({operation.source_archetype for _, operation in flat}))
        control_path_keys = tuple(sorted({operation.control_path for _, operation in flat}))
        condition_keys = tuple(sorted({operation.condition for _, operation in flat}))
        amount_basis_keys = tuple(sorted({operation.amount_basis for _, operation in flat}))
        guard_keys = ("<pad>",) + tuple(
            sorted({guard for _, operation in flat for guard in operation.guard_archetypes})
        )
        trigger_id = {key: index for index, key in enumerate(trigger_keys)}
        operation_id = {key: index for index, key in enumerate(operation_keys)}
        target_id = {key: index for index, key in enumerate(target_keys)}
        source_id = {key: index for index, key in enumerate(source_keys)}
        control_path_id = {key: index for index, key in enumerate(control_path_keys)}
        condition_id = {key: index for index, key in enumerate(condition_keys)}
        amount_basis_id = {key: index for index, key in enumerate(amount_basis_keys)}
        guard_id = {key: index for index, key in enumerate(guard_keys)}

        def padded_guards(operation: MechanicOperationV1) -> tuple[int, ...]:
            values = tuple(guard_id[value] for value in operation.guard_archetypes)
            return values + (0,) * (MECHANIC_GUARD_LIMIT - len(values))

        def padded_polarities(operation: MechanicOperationV1) -> tuple[float, ...]:
            values = tuple(float(value) for value in operation.guard_polarities)
            return values + (0.0,) * (MECHANIC_GUARD_LIMIT - len(values))

        def scaled_guard_number(value: float) -> float:
            return math.copysign(math.log1p(abs(float(value))) / math.log1p(10_000.0), float(value))

        def padded_guard_numeric(operation: MechanicOperationV1) -> tuple[tuple[float, float], ...]:
            values = tuple(
                (scaled_guard_number(left), scaled_guard_number(right)) for left, right in operation.guard_numeric
            )
            return values + ((0.0, 0.0),) * (MECHANIC_GUARD_LIMIT - len(values))

        return {
            "trigger_keys": trigger_keys,
            "operation_keys": operation_keys,
            "target_keys": target_keys,
            "source_keys": source_keys,
            "control_path_keys": control_path_keys,
            "condition_keys": condition_keys,
            "amount_basis_keys": amount_basis_keys,
            "guard_keys": guard_keys,
            "profile_id": torch.tensor([profile for profile, _ in flat], dtype=torch.long),
            "trigger_id": torch.tensor([trigger_id[operation.trigger] for _, operation in flat], dtype=torch.long),
            "operation_id": torch.tensor(
                [operation_id[operation.operation] for _, operation in flat], dtype=torch.long
            ),
            "target_id": torch.tensor([target_id[operation.target] for _, operation in flat], dtype=torch.long),
            "source_id": torch.tensor(
                [source_id[operation.source_archetype] for _, operation in flat], dtype=torch.long
            ),
            "control_path_id": torch.tensor(
                [control_path_id[operation.control_path] for _, operation in flat], dtype=torch.long
            ),
            "condition_id": torch.tensor(
                [condition_id[operation.condition] for _, operation in flat], dtype=torch.long
            ),
            "amount_basis_id": torch.tensor(
                [amount_basis_id[operation.amount_basis] for _, operation in flat], dtype=torch.long
            ),
            "guard_archetype_id": torch.tensor(
                [padded_guards(operation) for _, operation in flat], dtype=torch.long
            ).reshape(len(flat), MECHANIC_GUARD_LIMIT),
            "guard_polarity": torch.tensor(
                [padded_polarities(operation) for _, operation in flat], dtype=torch.float32
            ).reshape(len(flat), MECHANIC_GUARD_LIMIT),
            "guard_numeric": torch.tensor(
                [padded_guard_numeric(operation) for _, operation in flat], dtype=torch.float32
            ).reshape(len(flat), MECHANIC_GUARD_LIMIT, 2),
            "guard_mask": torch.tensor(
                [
                    [index < len(operation.guard_archetypes) for index in range(MECHANIC_GUARD_LIMIT)]
                    for _, operation in flat
                ],
                dtype=torch.bool,
            ).reshape(len(flat), MECHANIC_GUARD_LIMIT),
            "tags": torch.tensor(
                [[float(name in operation.mechanic_tags) for name in MECHANIC_TAG_NAMES] for _, operation in flat],
                dtype=torch.float32,
            ).reshape(len(flat), len(MECHANIC_TAG_NAMES)),
            "effect_vocab_id": torch.tensor([operation.effect_vocab_id for _, operation in flat], dtype=torch.long),
            "produced_archetype_id": torch.tensor(
                [operation.produced_archetype_id for _, operation in flat], dtype=torch.long
            ),
            "numeric": torch.tensor([operation.numeric_features for _, operation in flat], dtype=torch.float32).reshape(
                len(flat), len(MECHANIC_NUMERIC_FEATURE_NAMES)
            ),
        }
