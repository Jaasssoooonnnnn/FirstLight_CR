"""Versioned, immutable wire contracts for the native battle infrastructure.

The native probe intentionally exposes a compact engine-shaped dictionary.  The
contracts in this module are the stable boundary used by data collection,
training, evaluation, and renderer capture.  They are deliberately independent
of NumPy/Gym so a trajectory remains readable with only the Python standard
library installed.

V1 uses content-addressable canonical JSON.  Unknown game mechanics are never
silently guessed: CardSpecV1 records them in ``unknown_fields`` and keeps the
uninterpreted public configuration in ``attributes`` for later schema upgrades.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from functools import cache
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, ClassVar, TypeVar

from .match_factory import NATIVE_MATCH_END_TICK


OBSERVATION_VERSION = "observation.v1"
ACTION_VERSION = "action.v1"
CARD_SPEC_VERSION = "card-spec.v1"
ABILITY_SPEC_VERSION = "ability-spec.v1"
LATENCY_CONFIG_VERSION = "latency-config.v1"
ENVIRONMENT_CONFIG_VERSION = "environment-config.v1"
EPISODE_CONFIG_VERSION = "episode-config.v1"
RASTER_VERSION = "raster.v2"
RUNNER_ATTESTATION_VERSION = "runner-attestation.v1"
RUNNER_PROTOCOL_VERSION = "cr-native-control.v1"
SEMANTIC_PROVENANCE_VERSION = "semantic-provenance.v1"
SHIELD_STATE_VERSION = "shield-state.v1"
EFFECT_STATE_VERSION = "effect-state.v1"
ATTACK_STATE_VERSION = "attack-state.v1"
PROJECTILE_STATE_VERSION = "projectile-state.v1"
ENTITY_RESOURCE_STATE_VERSION = "entity-resource-state.v1"
PERIODIC_ATTACK_MODIFIER_STATE_VERSION = "periodic-attack-modifier-state.v1"
CAPTURE_TARGET_STATE_VERSION = "capture-target-state.v1"
CAPTURE_RUNTIME_STATE_VERSION = "capture-runtime-state.v1"
THRESHOLD_RELOCATION_STATE_VERSION = "threshold-relocation-state.v1"
VISIBILITY_STATE_VERSION = "visibility-state.v1"
ABILITY_RUNTIME_STATE_VERSION = "ability-runtime-state.v1"
EVOLUTION_RUNTIME_STATE_VERSION = "evolution-runtime-state.v1"
COMBAT_EVENT_VERSION = "combat-event.v1"
CAUSAL_GROUP_REF_VERSION = "causal-group-ref.v1"
DAGGER_DUCHESS_RUNTIME_VERSION = "dagger-duchess-runtime.v1"
ROYAL_CHEF_RUNTIME_VERSION = "royal-chef-runtime.v1"

BOARD_WIDTH = 18
BOARD_HEIGHT = 32
DEFAULT_TICK_MS = 50
COMPETITIVE_TOWER_TROOP_IDS = frozenset(
    {
        159_000_000,  # Tower Princess
        159_000_001,  # Cannoneer
        159_000_002,  # Dagger Duchess
        159_000_004,  # Royal Chef; 159_000_003 is an unused internal row
    }
)
ROYAL_CHEF_TOWER_TROOP_ID = 159_000_004

RASTER_V2_CHANNELS: tuple[str, ...] = (
    "own_unit_density",
    "enemy_unit_density",
    "own_unit_hp_ratio",
    "enemy_unit_hp_ratio",
    "own_unit_shield_ratio",
    "enemy_unit_shield_ratio",
    "own_troop_density",
    "enemy_troop_density",
    "own_building_density",
    "enemy_building_density",
    "own_other_density",
    "enemy_other_density",
    "neutral_density",
    "own_tower_hp_ratio",
    "enemy_tower_hp_ratio",
    "legal_deployment_union",
    "hand_slot_0_legal",
    "hand_slot_1_legal",
    "hand_slot_2_legal",
    "hand_slot_3_legal",
)
RASTER_V2_BINARY_CHANNELS = frozenset(
    {"legal_deployment_union", "hand_slot_0_legal", "hand_slot_1_legal", "hand_slot_2_legal", "hand_slot_3_legal"}
)


class ContractError(ValueError):
    """Raised when a versioned contract is malformed or leaks private state."""


class FrozenMapping(Mapping[str, Any]):
    """Small recursively immutable mapping with deterministic equality/hash."""

    __slots__ = ("_data", "_hash")

    def __init__(self, value: Mapping[str, Any] | None = None) -> None:
        data = {str(key): _freeze(item) for key, item in (value or {}).items()}
        self._data = MappingProxyType(data)
        self._hash: int | None = None

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"FrozenMapping({dict(self._data)!r})"

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(canonical_json(self))
        return self._hash

    def __copy__(self) -> "FrozenMapping":
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> "FrozenMapping":
        # FrozenMapping recursively freezes every child at construction time,
        # so sharing it across an in-process snapshot is safe.  Returning the
        # same object also prevents copy.deepcopy from descending into the
        # intentionally non-pickleable MappingProxyType backing store.
        memo[id(self)] = self
        return self


def _freeze(value: Any) -> Any:
    if isinstance(value, FrozenMapping):
        return value
    if isinstance(value, Mapping):
        return FrozenMapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze(item) for item in value), key=repr))
    return value


def frozen_mapping(value: Mapping[str, Any] | None = None) -> FrozenMapping:
    return value if isinstance(value, FrozenMapping) else FrozenMapping(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _jsonable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_jsonable(item) for item in sorted(value, key=repr)]
    if isinstance(value, float) and not math.isfinite(value):
        raise ContractError("NaN and infinity are forbidden in persisted contracts")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ContractError(f"unsupported contract value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Return the byte-stable UTF-8 JSON representation used for hashes."""

    return json.dumps(_jsonable(value), ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class ContractMixin:
    """Common stable serialization for all public contracts."""

    VERSION: ClassVar[str]

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)

    def to_json(self, *, pretty: bool = False) -> str:
        if not pretty:
            return canonical_json(self)
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)

    def content_hash(self) -> str:
        return content_hash(self)


_RUNNER_HOOKS = ("game_state_load", "game_state_step", "native_replay_runner", "render_gate", "replay_json_setter")
_RUNNER_CAPABILITIES = ("headless", "native_render", "snapshot_restore", "structured_observation", "two_sided_actions")


def _lower_hex(value: str, length: int) -> bool:
    return (
        len(value) == length and value == value.lower() and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True, slots=True)
class RunnerAttestationV1(ContractMixin):
    """Identity of the actual native process serving one runner endpoint.

    The digest covers only immutable process/content identity. Readiness and the
    mapped-content check remain explicit runtime gates, so polling them cannot
    silently change the identity assigned to an episode.
    """

    protocol_version: str
    package_name: str
    abi: str
    probe_sha256: str
    libg_sha256: str
    libg_build_id: str
    content_fingerprint_sha256: str
    content_assets_sha256: str
    content_version: str
    content_manifest_sha1: str
    hook_install_ok: bool
    hooks: Mapping[str, bool]
    capabilities: Mapping[str, bool]
    content_ready: bool
    content_assets_mapped: bool
    production_ready: bool
    attestation_digest: str
    version: str = field(default=RUNNER_ATTESTATION_VERSION, init=False)
    VERSION: ClassVar[str] = RUNNER_ATTESTATION_VERSION

    def __post_init__(self) -> None:
        if self.protocol_version != RUNNER_PROTOCOL_VERSION:
            raise ContractError(f"unsupported runner protocol: {self.protocol_version!r}")
        if not self.package_name or not self.abi:
            raise ContractError("runner package_name and abi must not be empty")
        for name in ("probe_sha256", "libg_sha256", "content_fingerprint_sha256", "content_assets_sha256"):
            value = str(getattr(self, name))
            if value and not _lower_hex(value, 64):
                raise ContractError(f"{name} must be an empty value or lowercase SHA-256")
        if self.libg_build_id and not _lower_hex(self.libg_build_id, 40):
            raise ContractError("libg_build_id must be an empty value or 20-byte lowercase hex")
        if self.content_manifest_sha1 and not _lower_hex(self.content_manifest_sha1, 40):
            raise ContractError("content_manifest_sha1 must be an empty value or lowercase SHA-1")
        normalized_hooks = {str(key): bool(value) for key, value in self.hooks.items()}
        normalized_capabilities = {str(key): bool(value) for key, value in self.capabilities.items()}
        if set(normalized_hooks) != set(_RUNNER_HOOKS):
            raise ContractError("runner hook attestation must contain exactly: " + ", ".join(_RUNNER_HOOKS))
        if set(normalized_capabilities) != set(_RUNNER_CAPABILITIES):
            raise ContractError("runner capabilities must contain exactly: " + ", ".join(_RUNNER_CAPABILITIES))
        object.__setattr__(self, "hooks", frozen_mapping(normalized_hooks))
        object.__setattr__(self, "capabilities", frozen_mapping(normalized_capabilities))

        expected_production_ready = (
            self.content_ready
            and self.content_assets_mapped
            and self.hook_install_ok
            and all(normalized_hooks.values())
            and all(normalized_capabilities.values())
            and all(
                bool(getattr(self, name))
                for name in (
                    "probe_sha256",
                    "libg_sha256",
                    "libg_build_id",
                    "content_fingerprint_sha256",
                    "content_assets_sha256",
                    "content_version",
                    "content_manifest_sha1",
                )
            )
        )
        if self.production_ready != expected_production_ready:
            raise ContractError("production_ready disagrees with runner identity/readiness gates")
        expected_digest = self.calculate_attestation_digest()
        if not _lower_hex(self.attestation_digest, 64):
            raise ContractError("attestation_digest must be a lowercase SHA-256")
        if self.attestation_digest != expected_digest:
            raise ContractError(
                f"runner attestation digest mismatch: expected {expected_digest}, got {self.attestation_digest}"
            )

    def identity_dict(self) -> dict[str, Any]:
        return {
            "abi": self.abi,
            "capabilities": dict(self.capabilities),
            "content_assets_sha256": self.content_assets_sha256,
            "content_fingerprint_sha256": self.content_fingerprint_sha256,
            "content_manifest_sha1": self.content_manifest_sha1,
            "content_version": self.content_version,
            "hook_install_ok": self.hook_install_ok,
            "hooks": dict(self.hooks),
            "libg_build_id": self.libg_build_id,
            "libg_sha256": self.libg_sha256,
            "package_name": self.package_name,
            "probe_sha256": self.probe_sha256,
            "protocol_version": self.protocol_version,
            "version": self.version,
        }

    def calculate_attestation_digest(self) -> str:
        return content_hash(self.identity_dict())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RunnerAttestationV1":
        _version(value, RUNNER_ATTESTATION_VERSION)
        data = dict(value)
        data.pop("version", None)
        return cls(**data)


E = TypeVar("E", bound=Enum)


def _enum(enum_type: type[E], value: E | str, label: str) -> E:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except ValueError as error:
        allowed = ", ".join(item.value for item in enum_type)
        raise ContractError(f"{label} must be one of: {allowed}") from error


def _version(mapping: Mapping[str, Any], expected: str) -> None:
    actual = mapping.get("version")
    if actual != expected:
        raise ContractError(f"expected contract {expected!r}, got {actual!r}")


def _strict_contract_mapping(
    value: Mapping[str, Any], *, expected_version: str, allowed_fields: Iterable[str], label: str
) -> dict[str, Any]:
    """Validate a canonical nested contract without silently dropping fields."""

    _version(value, expected_version)
    allowed = set(allowed_fields)
    unknown = sorted(set(value).difference(allowed))
    if unknown:
        raise ContractError(f"unknown {label} fields: {', '.join(unknown)}")
    data = dict(value)
    data.pop("version", None)
    return data


@cache
def _contract_field_names(cls: type) -> frozenset[str]:
    return frozenset(item.name for item in fields(cls))


class StrictContractMixin(ContractMixin):
    """Use the dataclass declaration as the canonical wire field inventory."""

    STRICT_LABEL: ClassVar[str]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]):
        return cls(
            **_strict_contract_mapping(
                value, expected_version=cls.VERSION, allowed_fields=_contract_field_names(cls), label=cls.STRICT_LABEL
            )
        )


def _set_contract(instance: Any, name: str, contract: type) -> None:
    value = getattr(instance, name)
    if value is not None and not isinstance(value, contract):
        object.__setattr__(instance, name, contract.from_mapping(value))


def _set_contracts(instance: Any, name: str, contract: type) -> None:
    object.__setattr__(
        instance,
        name,
        tuple(item if isinstance(item, contract) else contract.from_mapping(item) for item in getattr(instance, name)),
    )


def _optional_contract(contract: type, value: Any) -> Any:
    return None if value is None else contract.from_mapping(value)


def _contract_sequence(contract: type, values: Iterable[Any]) -> tuple:
    return tuple(contract.from_mapping(value) for value in values)


def _set_provenance(instance: Any, name: str, required: Iterable[str], label: str) -> None:
    object.__setattr__(instance, name, _semantic_provenance(getattr(instance, name), required, label=label))


def _nonnegative(instance: Any, names: str, prefix: str = "") -> None:
    for name in names.split():
        value = getattr(instance, name)
        if value is not None and value < 0:
            raise ContractError(f"{prefix}{name} must be non-negative")


def _int_tuple(value: Iterable[Any] | None) -> tuple[int, ...]:
    return tuple(int(item) for item in (value or ()))


def _pair(value: Sequence[Any] | None, *, cast: type = float) -> tuple[Any, Any] | None:
    if value is None:
        return None
    if len(value) != 2:
        raise ContractError("coordinate values must contain exactly two items")
    return (cast(value[0]), cast(value[1]))


class ObservationTier(str, Enum):
    ORACLE = "oracle"
    FAIR = "fair"
    HUMAN = "human"
    RGB = "rgb"


class ActionKind(str, Enum):
    WAIT = "wait"
    WAIT_UNTIL_AFFORDABLE = "wait_until_affordable"
    PLAY_CARD = "play_card"
    ACTIVATE_ABILITY = "activate_ability"


class TargetKind(str, Enum):
    NONE = "none"
    GRID = "grid"
    FRIENDLY_ENTITY = "friendly_entity"
    ENEMY_ENTITY = "enemy_entity"
    DIRECTION = "direction"
    AREA_CENTER = "area_center"


class CardKind(str, Enum):
    TROOP = "troop"
    SPELL = "spell"
    BUILDING = "building"
    HERO = "hero"
    UNKNOWN = "unknown"


class SemanticEvidenceLevel(str, Enum):
    """How one runtime field was obtained.

    Only ``native_authoritative`` is strong enough for native golden tests.
    Derived/static values remain useful training features, but callers must opt
    into them explicitly.  ``unknown`` is never coerced to a gameplay value.
    """

    NATIVE_AUTHORITATIVE = "native_authoritative"
    NATIVE_DERIVED = "native_derived"
    STATIC_DECLARED = "static_declared"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class EffectKind(str, Enum):
    BUFF = "buff"
    DEBUFF = "debuff"
    DAMAGE_OVER_TIME = "damage_over_time"
    HEAL_OVER_TIME = "heal_over_time"
    STUN = "stun"
    FREEZE = "freeze"
    SLOW = "slow"
    HASTE = "haste"
    RAGE = "rage"
    ROOT = "root"
    KNOCKBACK = "knockback"
    PULL = "pull"
    SILENCE = "silence"
    INTERRUPT = "interrupt"
    INVISIBILITY = "invisibility"
    REVEAL = "reveal"
    UNTARGETABLE = "untargetable"
    SHIELD = "shield"
    IMMUNITY = "immunity"
    REFLECT = "reflect"
    CHARGE = "charge"
    DAMAGE_RAMP = "damage_ramp"
    MARK = "mark"
    CURSE = "curse"
    TRANSFORM = "transform"
    SPAWN_MODIFIER = "spawn_modifier"
    TARGETING_MODIFIER = "targeting_modifier"
    MOVEMENT_MODIFIER = "movement_modifier"
    OTHER = "other"
    UNKNOWN = "unknown"


class AttackPhase(str, Enum):
    IDLE = "idle"
    ACQUIRING = "acquiring"
    WINDUP = "windup"
    RELEASE = "release"
    BACKSWING = "backswing"
    COOLDOWN = "cooldown"
    CHANNEL = "channel"
    CHARGING = "charging"
    INTERRUPTED = "interrupted"
    UNKNOWN = "unknown"


class ProjectilePhase(str, Enum):
    SPAWNED = "spawned"
    IN_FLIGHT = "in_flight"
    IMPACTED = "impacted"
    EXPIRED = "expired"
    DEFLECTED = "deflected"
    UNKNOWN = "unknown"


class CaptureActionPhase(str, Enum):
    IDLE = "idle"
    ACTIVE = "active"
    RELEASE_COOLDOWN = "release_cooldown"


class CaptureTargetPhase(str, Enum):
    ACQUIRED_DELAY = "acquired_delay"
    GRAB_PAUSE = "grab_pause"
    DRAGGING = "dragging"
    CONTAINED = "contained"
    RELEASE_PENDING = "release_pending"


class ThresholdRelocationPhase(str, Enum):
    WAITING_THRESHOLD = "waiting_threshold"
    RELOCATING = "relocating"
    EXHAUSTED = "exhausted"


class PeriodicAttackModifierPhase(str, Enum):
    SOURCE_ALIVE = "source_alive"
    SOURCE_DEATH_LINGER = "source_death_linger"


class ProjectileDragStage(str, Enum):
    OUTBOUND = "outbound"
    DRAG_BACK_ACTIVE = "drag_back_active"


class VisibilityPhase(str, Enum):
    VISIBLE = "visible"
    HIDING = "hiding"
    HIDDEN = "hidden"
    REVEALING = "revealing"
    BURROWED = "burrowed"
    UNKNOWN = "unknown"


class AbilityPhase(str, Enum):
    UNAVAILABLE = "unavailable"
    READY = "ready"
    CASTING = "casting"
    ACTIVE = "active"
    COOLDOWN = "cooldown"
    EXHAUSTED = "exhausted"
    INTERRUPTED = "interrupted"
    UNKNOWN = "unknown"


class EvolutionPhase(str, Enum):
    BASE = "base"
    CYCLING = "cycling"
    READY = "ready"
    TRANSFORMING = "transforming"
    EVOLVED = "evolved"
    UNKNOWN = "unknown"


class CombatEventKind(str, Enum):
    DEPLOY_REQUESTED = "deploy_requested"
    DEPLOY_EXECUTED = "deploy_executed"
    SPAWN = "spawn"
    DESPAWN = "despawn"
    DEATH = "death"
    DAMAGE = "damage"
    HEAL = "heal"
    SHIELD_GAIN = "shield_gain"
    SHIELD_DAMAGE = "shield_damage"
    SHIELD_BREAK = "shield_break"
    EFFECT_APPLY = "effect_apply"
    EFFECT_REFRESH = "effect_refresh"
    EFFECT_STACK = "effect_stack"
    EFFECT_REMOVE = "effect_remove"
    TARGET_ACQUIRE = "target_acquire"
    TARGET_CHANGE = "target_change"
    TARGET_LOSE = "target_lose"
    ATTACK_START = "attack_start"
    ATTACK_RELEASE = "attack_release"
    ATTACK_HIT = "attack_hit"
    ATTACK_INTERRUPT = "attack_interrupt"
    PROJECTILE_SPAWN = "projectile_spawn"
    PROJECTILE_IMPACT = "projectile_impact"
    PROJECTILE_EXPIRE = "projectile_expire"
    PROJECTILE_DEFLECT = "projectile_deflect"
    TRANSFORM = "transform"
    VISIBILITY_CHANGE = "visibility_change"
    ABILITY_START = "ability_start"
    ABILITY_ACTIVATE = "ability_activate"
    ABILITY_END = "ability_end"
    EVOLUTION_READY = "evolution_ready"
    EVOLUTION_DEPLOY = "evolution_deploy"
    AREA_CREATE = "area_create"
    AREA_EXPIRE = "area_expire"
    TOWER_AGGRO_CHANGE = "tower_aggro_change"
    TOWER_ACTIVATE = "tower_activate"
    UNKNOWN = "unknown"


class CausalGroupKind(str, Enum):
    """Public causal roots used only to address battle groups."""

    DEPLOYMENT = "deployment"
    SPAWN_WAVE = "spawn_wave"
    VOLLEY = "volley"
    PERSISTENT_EFFECT = "persistent_effect"


@dataclass(frozen=True, slots=True)
class TargetSchemaV1(ContractMixin):
    allowed: tuple[TargetKind, ...] = (TargetKind.GRID,)
    placement_mask_key: str = "default"
    min_range_tiles: float | None = None
    max_range_tiles: float | None = None
    requires_visible_target: bool = True
    VERSION: ClassVar[str] = "target-schema.v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed", tuple(_enum(TargetKind, item, "target kind") for item in self.allowed))
        if not self.allowed:
            raise ContractError("target schema must allow at least one target kind")
        if len(set(self.allowed)) != len(self.allowed):
            raise ContractError("target schema contains duplicate target kinds")
        if self.min_range_tiles is not None and self.min_range_tiles < 0:
            raise ContractError("min_range_tiles must be non-negative")
        if self.max_range_tiles is not None and self.max_range_tiles < 0:
            raise ContractError("max_range_tiles must be non-negative")
        if (
            self.min_range_tiles is not None
            and self.max_range_tiles is not None
            and self.min_range_tiles > self.max_range_tiles
        ):
            raise ContractError("min_range_tiles cannot exceed max_range_tiles")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TargetSchemaV1":
        return cls(
            allowed=tuple(value.get("allowed", (TargetKind.GRID.value,))),
            placement_mask_key=str(value.get("placement_mask_key", "default")),
            min_range_tiles=value.get("min_range_tiles"),
            max_range_tiles=value.get("max_range_tiles"),
            requires_visible_target=bool(value.get("requires_visible_target", True)),
        )


@dataclass(frozen=True, slots=True)
class MechanicOpV1(ContractMixin):
    trigger: str
    effect: str
    condition: str | None = None
    target_selector: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=FrozenMapping)
    custom_op: str | None = None
    VERSION: ClassVar[str] = "mechanic-op.v1"

    def __post_init__(self) -> None:
        if not self.trigger or not self.effect:
            raise ContractError("mechanic trigger and effect must not be empty")
        object.__setattr__(self, "parameters", frozen_mapping(self.parameters))
        if self.effect == "CustomOp" and not self.custom_op:
            raise ContractError("CustomOp mechanics require custom_op")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MechanicOpV1":
        return cls(
            trigger=str(value["trigger"]),
            effect=str(value["effect"]),
            condition=value.get("condition"),
            target_selector=value.get("target_selector"),
            parameters=value.get("parameters", {}),
            custom_op=value.get("custom_op"),
        )


@dataclass(frozen=True, slots=True)
class EvolutionSpecV1(ContractMixin):
    base_form_id: str | None = None
    evolution_form_id: str | None = None
    cycle_required: int | None = None
    effects: tuple[MechanicOpV1, ...] = ()
    VERSION: ClassVar[str] = "evolution-spec.v1"

    def __post_init__(self) -> None:
        if self.cycle_required is not None and self.cycle_required < 0:
            raise ContractError("cycle_required must be non-negative")
        _set_contracts(self, "effects", MechanicOpV1)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvolutionSpecV1":
        return cls(
            base_form_id=value.get("base_form_id"),
            evolution_form_id=value.get("evolution_form_id"),
            cycle_required=value.get("cycle_required"),
            effects=tuple(value.get("effects", ())),
        )


@dataclass(frozen=True, slots=True)
class AbilitySpecV1(ContractMixin):
    ability_id: str
    source_card_id: int | None = None
    source_entity: int | None = None
    elixir_cost: float | None = None
    cooldown_ms: int | None = None
    remaining_cooldown_ms: int | None = None
    charges: int | None = None
    availability: bool | None = None
    cast_time_ms: int | None = None
    target_schema: TargetSchemaV1 = field(default_factory=lambda: TargetSchemaV1((TargetKind.NONE,)))
    effect_graph: tuple[MechanicOpV1, ...] = ()
    activation_condition: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=FrozenMapping)
    source_records: tuple[str, ...] = ()
    unknown_fields: tuple[str, ...] = ()
    version: str = field(default=ABILITY_SPEC_VERSION, init=False)
    VERSION: ClassVar[str] = ABILITY_SPEC_VERSION

    def __post_init__(self) -> None:
        if not self.ability_id:
            raise ContractError("ability_id must not be empty")
        if self.source_card_id is not None and self.source_card_id <= 0:
            raise ContractError("source_card_id must be positive")
        if self.source_entity is not None and self.source_entity < 0:
            raise ContractError("source_entity must be non-negative")
        _nonnegative(self, "elixir_cost cooldown_ms remaining_cooldown_ms charges cast_time_ms", "")
        if not isinstance(self.target_schema, TargetSchemaV1):
            object.__setattr__(self, "target_schema", TargetSchemaV1.from_mapping(self.target_schema))
        _set_contracts(self, "effect_graph", MechanicOpV1)
        object.__setattr__(self, "attributes", frozen_mapping(self.attributes))
        object.__setattr__(self, "source_records", tuple(str(item) for item in self.source_records))
        object.__setattr__(self, "unknown_fields", tuple(sorted(set(str(item) for item in self.unknown_fields))))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AbilitySpecV1":
        _version(value, ABILITY_SPEC_VERSION)
        return cls(
            ability_id=str(value["ability_id"]),
            source_card_id=value.get("source_card_id"),
            source_entity=value.get("source_entity"),
            elixir_cost=value.get("elixir_cost"),
            cooldown_ms=value.get("cooldown_ms", value.get("cooldown")),
            remaining_cooldown_ms=value.get("remaining_cooldown_ms", value.get("remaining_cooldown")),
            charges=value.get("charges"),
            availability=value.get("availability"),
            cast_time_ms=value.get("cast_time_ms", value.get("cast_time")),
            target_schema=TargetSchemaV1.from_mapping(value.get("target_schema", {"allowed": ["none"]})),
            effect_graph=tuple(value.get("effect_graph", ())),
            activation_condition=value.get("activation_condition"),
            attributes=value.get("attributes", {}),
            source_records=tuple(value.get("source_records", ())),
            unknown_fields=tuple(value.get("unknown_fields", ())),
        )


@dataclass(frozen=True, slots=True)
class CardSpecV1(ContractMixin):
    card_id: int
    name: str
    kind: CardKind
    elixir_cost: float | None = None
    hitpoints: float | None = None
    damage: float | None = None
    hit_speed_ms: int | None = None
    range_tiles: float | None = None
    move_speed: float | None = None
    deploy_time_ms: int | None = None
    count: int | None = None
    radius_tiles: float | None = None
    projectile_speed: float | None = None
    formation: str | None = None
    knockback: float | None = None
    duration_ms: int | None = None
    shield_hitpoints: float | None = None
    target_schema: TargetSchemaV1 = field(default_factory=TargetSchemaV1)
    mechanics: tuple[MechanicOpV1, ...] = ()
    summoned_cards: tuple[int, ...] = ()
    summoned_forms: tuple[str, ...] = ()
    evolution: EvolutionSpecV1 | None = None
    ability_ids: tuple[str, ...] = ()
    numeric_features: Mapping[str, float | int | None] = field(default_factory=FrozenMapping)
    categorical_features: Mapping[str, Any] = field(default_factory=FrozenMapping)
    attributes: Mapping[str, Any] = field(default_factory=FrozenMapping)
    source_records: tuple[str, ...] = ()
    unknown_fields: tuple[str, ...] = ()
    residual_key: str | None = None
    residual_init: float = 0.0
    version: str = field(default=CARD_SPEC_VERSION, init=False)
    VERSION: ClassVar[str] = CARD_SPEC_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum(CardKind, self.kind, "card kind"))
        if self.card_id <= 0:
            raise ContractError("card_id must be positive")
        if not self.name:
            raise ContractError("card name must not be empty")
        _nonnegative(
            self,
            "elixir_cost hitpoints damage hit_speed_ms range_tiles move_speed deploy_time_ms count radius_tiles projectile_speed duration_ms shield_hitpoints",
            "",
        )
        if not isinstance(self.target_schema, TargetSchemaV1):
            object.__setattr__(self, "target_schema", TargetSchemaV1.from_mapping(self.target_schema))
        _set_contracts(self, "mechanics", MechanicOpV1)
        object.__setattr__(self, "summoned_cards", _int_tuple(self.summoned_cards))
        if any(item <= 0 for item in self.summoned_cards):
            raise ContractError("summoned card IDs must be positive")
        object.__setattr__(self, "summoned_forms", tuple(str(item) for item in self.summoned_forms))
        _set_contract(self, "evolution", EvolutionSpecV1)
        object.__setattr__(self, "ability_ids", tuple(str(item) for item in self.ability_ids))
        object.__setattr__(self, "numeric_features", frozen_mapping(self.numeric_features))
        object.__setattr__(self, "categorical_features", frozen_mapping(self.categorical_features))
        object.__setattr__(self, "attributes", frozen_mapping(self.attributes))
        object.__setattr__(self, "source_records", tuple(str(item) for item in self.source_records))
        object.__setattr__(self, "unknown_fields", tuple(sorted(set(str(item) for item in self.unknown_fields))))
        object.__setattr__(self, "residual_key", self.residual_key or f"card:{self.card_id}")
        if self.residual_init != 0.0:
            raise ContractError("V1 unseen-card residuals must initialize to zero")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CardSpecV1":
        _version(value, CARD_SPEC_VERSION)
        kwargs = dict(value)
        kwargs.pop("version", None)
        kwargs["target_schema"] = TargetSchemaV1.from_mapping(kwargs.get("target_schema", {}))
        if kwargs.get("evolution") is not None:
            kwargs["evolution"] = EvolutionSpecV1.from_mapping(kwargs["evolution"])
        kwargs["mechanics"] = tuple(MechanicOpV1.from_mapping(item) for item in kwargs.get("mechanics", ()))
        return cls(**kwargs)


SHIELD_STATE_FIELDS = frozenset({"hitpoints", "max_hitpoints", "kind", "source_entity", "broken", "broken_tick"})
EFFECT_STATE_FIELDS = frozenset(
    {
        "kind",
        "source_entity",
        "source_card_id",
        "source_owner",
        "started_tick",
        "end_tick",
        "remaining_ms",
        "stacks",
        "magnitude",
        "stage",
        "active",
    }
)
ATTACK_STATE_FIELDS = frozenset(
    {
        "phase",
        "target_entity",
        "target_position",
        "phase_started_tick",
        "phase_remaining_ms",
        "cooldown_remaining_ms",
        "sequence_index",
        "sequence_progress",
        "sequence_progress_limit",
        "sequence_decay_remaining_ms",
        "sequence_decay_duration_ms",
        "charge_stage",
        "charge_elapsed_ms",
        "damage_multiplier",
        "locked",
        "interrupted",
    }
)
PROJECTILE_STATE_FIELDS = frozenset(
    {
        "phase",
        "source_entity",
        "source_card_id",
        "target_entity",
        "target_position",
        "spawn_tick",
        "expected_impact_tick",
        "impact_tick",
        "velocity",
        "damage",
        "radius_tiles",
        "homing",
        "deflected",
        "drag_stage",
    }
)
ENTITY_RESOURCE_STATE_FIELDS = frozenset({"kind", "current_raw", "capacity_raw", "normalized"})
PERIODIC_ATTACK_MODIFIER_STATE_FIELDS = frozenset(
    {"phase", "period_attacks", "completed_attacks", "source_entity", "linger_remaining_ms", "linger_duration_ms"}
)
CAPTURE_RUNTIME_STATE_FIELDS = frozenset(
    {"phase", "configured_cooldown_ms", "cooldown_remaining_ms", "hit_frequency_ms", "hit_accumulator_ms", "targets"}
)
THRESHOLD_RELOCATION_STATE_FIELDS = frozenset(
    {"phase", "stage", "relocation_index", "thresholds_percent", "hide_duration_ms", "remaining_ms", "burrowed"}
)
VISIBILITY_STATE_FIELDS = frozenset(
    {"phase", "public_by_owner", "targetable_by_owner", "area_damage_eligible", "transition_remaining_ms"}
)
ABILITY_RUNTIME_STATE_FIELDS = frozenset(
    {
        "source_entity",
        "phase",
        "elixir_cost",
        "cooldown_ms",
        "remaining_cooldown_ms",
        "charges",
        "available",
        "cast_started_tick",
        "active_until_tick",
        "target_entity",
        "target_position",
    }
)
EVOLUTION_RUNTIME_STATE_FIELDS = frozenset(
    {
        "deck_slot",
        "phase",
        "base_form_id",
        "current_form_id",
        "next_form_id",
        "cycle_required",
        "cycle_remaining",
        "ready",
        "active",
        "deployments_in_cycle",
    }
)
COMBAT_EVENT_FIELDS = frozenset(
    {
        "kind",
        "source_entity",
        "source_card_id",
        "target_entity",
        "target_card_id",
        "position",
        "amount",
        "effect_id",
        "projectile_id",
        "ability_id",
        "evolution_form_id",
        "visible_by_owner",
    }
)
ENTITY_RUNTIME_SEMANTIC_FIELDS = frozenset(
    {
        "shield_state",
        "effect_states",
        "attack_state",
        "projectile_state",
        "visibility_state",
        "ability_states",
        "evolution_state",
        "movement_runtime",
        "deployment_runtime",
        "resource_states",
        "capture_runtime",
        "threshold_relocation_runtime",
        "periodic_attack_modifier",
    }
)
PLAYER_RUNTIME_SEMANTIC_FIELDS = frozenset({"ability_runtime_states", "evolution_runtime_states"})
TOWER_RUNTIME_SEMANTIC_FIELDS = frozenset(
    {"shield_state", "effect_states", "attack_state", "visibility_state", "tower_troop_runtime"}
)
EVENT_RUNTIME_SEMANTIC_FIELDS = frozenset({"combat"})


@dataclass(frozen=True, slots=True)
class SemanticProvenanceV1(StrictContractMixin):
    """Per-field evidence and completeness for typed runtime state.

    Every enclosing state validates this mapping against its complete field
    universe.  Consequently an omitted native field is persisted as
    ``unknown`` instead of acquiring a dangerous default value.
    """

    field_evidence: Mapping[str, SemanticEvidenceLevel]
    source_fields: Mapping[str, tuple[str, ...]] = field(default_factory=FrozenMapping)
    observed_tick: int | None = None
    notes: tuple[str, ...] = ()
    version: str = field(default=SEMANTIC_PROVENANCE_VERSION, init=False)
    VERSION: ClassVar[str] = SEMANTIC_PROVENANCE_VERSION

    def __post_init__(self) -> None:
        evidence = {
            str(name): _enum(SemanticEvidenceLevel, value, f"semantic evidence for {name}")
            for name, value in self.field_evidence.items()
        }
        if not evidence or any(not name for name in evidence):
            raise ContractError("semantic provenance requires named fields")
        sources = {str(name): tuple(str(item) for item in values) for name, values in self.source_fields.items()}
        if not set(sources).issubset(evidence):
            raise ContractError("semantic provenance source_fields reference unknown fields")
        if self.observed_tick is not None and self.observed_tick < 0:
            raise ContractError("semantic provenance observed_tick must be non-negative")
        object.__setattr__(self, "field_evidence", frozen_mapping(evidence))
        object.__setattr__(self, "source_fields", frozen_mapping(sources))
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @property
    def unknown_fields(self) -> tuple[str, ...]:
        return tuple(
            sorted(name for name, evidence in self.field_evidence.items() if evidence == SemanticEvidenceLevel.UNKNOWN)
        )

    def validate_fields(self, expected: Iterable[str], *, label: str) -> None:
        expected_set = {str(item) for item in expected}
        actual = set(self.field_evidence)
        if actual != expected_set:
            missing = sorted(expected_set.difference(actual))
            extra = sorted(actual.difference(expected_set))
            raise ContractError(f"{label} provenance must cover every field exactly; missing={missing}, extra={extra}")

    def assert_known(
        self,
        fields_required: Iterable[str],
        *,
        label: str,
        require_native: bool = False,
        require_authoritative: bool = False,
    ) -> None:
        required = tuple(str(item) for item in fields_required)
        allowed = {
            SemanticEvidenceLevel.NATIVE_AUTHORITATIVE,
            SemanticEvidenceLevel.NATIVE_DERIVED,
            SemanticEvidenceLevel.STATIC_DECLARED,
        }
        if require_native:
            allowed.discard(SemanticEvidenceLevel.STATIC_DECLARED)
        if require_authoritative:
            allowed = {SemanticEvidenceLevel.NATIVE_AUTHORITATIVE}
        unavailable = sorted(
            name for name in required if self.field_evidence.get(name, SemanticEvidenceLevel.UNKNOWN) not in allowed
        )
        if unavailable:
            raise ContractError(
                f"{label} fails closed; required semantic fields unavailable: " + ", ".join(unavailable)
            )

    @classmethod
    def unknown_all(cls, fields_required: Iterable[str]) -> "SemanticProvenanceV1":
        return cls({str(name): SemanticEvidenceLevel.UNKNOWN for name in sorted(fields_required)})

    STRICT_LABEL: ClassVar[str] = "semantic provenance"


def _semantic_provenance(
    value: SemanticProvenanceV1 | Mapping[str, Any], expected: Iterable[str], *, label: str
) -> SemanticProvenanceV1:
    result = value if isinstance(value, SemanticProvenanceV1) else SemanticProvenanceV1.from_mapping(value)
    result.validate_fields(expected, label=label)
    return result


def _mapping_provenance(value: Any, expected: Iterable[str]) -> SemanticProvenanceV1:
    return SemanticProvenanceV1.unknown_all(expected) if value is None else SemanticProvenanceV1.from_mapping(value)


_ABSENT_EVIDENCE = frozenset({SemanticEvidenceLevel.UNKNOWN, SemanticEvidenceLevel.NOT_APPLICABLE})
_POSITIVE_EVIDENCE = frozenset(
    {
        SemanticEvidenceLevel.NATIVE_AUTHORITATIVE,
        SemanticEvidenceLevel.NATIVE_DERIVED,
        SemanticEvidenceLevel.STATIC_DECLARED,
    }
)


def _require_presence_evidence(
    present: bool, evidence: SemanticEvidenceLevel, present_error: str, absent_error: str | None = None
) -> None:
    if present and evidence in _ABSENT_EVIDENCE:
        raise ContractError(present_error)
    if not present and absent_error and evidence in _POSITIVE_EVIDENCE:
        raise ContractError(absent_error)


def _validate_domains(instance: Any, label: str, *, singular: str = "", plural: str = "") -> None:
    for names, collection in ((singular, False), (plural, True)):
        for name in names.split():
            state = getattr(instance, name)
            present = bool(state) if collection else state is not None
            evidence = instance.runtime_provenance.field_evidence[name]
            if present and evidence in _ABSENT_EVIDENCE:
                raise ContractError(f"{label} {name} carries data without positive provenance")
            if not collection and not present and evidence in _POSITIVE_EVIDENCE:
                raise ContractError(f"{label} {name} provenance claims data but state is absent")


def _owner_visibility_mapping(value: Mapping[str, bool | None] | None, *, label: str) -> FrozenMapping:
    raw = value or {}
    if set(str(key) for key in raw) not in (set(), {"0", "1"}):
        raise ContractError(f"{label} must be empty or cover owners 0 and 1 exactly")
    normalized: dict[str, bool | None] = {}
    for key, item in raw.items():
        if item is not None and not isinstance(item, bool):
            raise ContractError(f"{label} values must be bool or null")
        normalized[str(key)] = item
    return frozen_mapping(normalized)


@dataclass(frozen=True, slots=True)
class ShieldStateV1(StrictContractMixin):
    hitpoints: float | None = None
    max_hitpoints: float | None = None
    kind: str | None = None
    source_entity: int | None = None
    broken: bool | None = None
    broken_tick: int | None = None
    provenance: SemanticProvenanceV1 = field(
        default_factory=lambda: SemanticProvenanceV1.unknown_all(SHIELD_STATE_FIELDS)
    )
    version: str = field(default=SHIELD_STATE_VERSION, init=False)
    VERSION: ClassVar[str] = SHIELD_STATE_VERSION

    def __post_init__(self) -> None:
        _nonnegative(self, "hitpoints max_hitpoints source_entity broken_tick", "shield ")
        if self.hitpoints is not None and self.max_hitpoints is not None and self.hitpoints > self.max_hitpoints:
            raise ContractError("shield hitpoints cannot exceed max_hitpoints")
        _set_provenance(self, "provenance", SHIELD_STATE_FIELDS, "shield state")

    STRICT_LABEL: ClassVar[str] = "shield state"


@dataclass(frozen=True, slots=True)
class EffectStateV1(StrictContractMixin):
    effect_id: str
    kind: EffectKind = EffectKind.UNKNOWN
    source_entity: int | None = None
    source_card_id: int | None = None
    source_owner: int | None = None
    started_tick: int | None = None
    end_tick: int | None = None
    remaining_ms: int | None = None
    stacks: int | None = None
    magnitude: float | None = None
    stage: str | None = None
    active: bool | None = None
    attributes: Mapping[str, Any] = field(default_factory=FrozenMapping)
    provenance: SemanticProvenanceV1 = field(
        default_factory=lambda: SemanticProvenanceV1.unknown_all(EFFECT_STATE_FIELDS)
    )
    version: str = field(default=EFFECT_STATE_VERSION, init=False)
    VERSION: ClassVar[str] = EFFECT_STATE_VERSION

    def __post_init__(self) -> None:
        if not self.effect_id:
            raise ContractError("effect_id must not be empty")
        object.__setattr__(self, "kind", _enum(EffectKind, self.kind, "effect kind"))
        if self.source_owner is not None and self.source_owner not in (0, 1):
            raise ContractError("effect source_owner must be 0, 1, or null")
        _nonnegative(self, "source_entity source_card_id started_tick end_tick remaining_ms stacks", "effect ")
        if self.started_tick is not None and self.end_tick is not None and self.end_tick < self.started_tick:
            raise ContractError("effect end_tick cannot precede started_tick")
        object.__setattr__(self, "attributes", frozen_mapping(self.attributes))
        _set_provenance(self, "provenance", EFFECT_STATE_FIELDS, "effect state")

    STRICT_LABEL: ClassVar[str] = "effect state"


@dataclass(frozen=True, slots=True)
class AttackStateV1(StrictContractMixin):
    phase: AttackPhase = AttackPhase.UNKNOWN
    target_entity: int | None = None
    target_position: tuple[float, float] | None = None
    phase_started_tick: int | None = None
    phase_remaining_ms: int | None = None
    cooldown_remaining_ms: int | None = None
    sequence_index: int | None = None
    sequence_progress: int | None = None
    sequence_progress_limit: int | None = None
    sequence_decay_remaining_ms: int | None = None
    sequence_decay_duration_ms: int | None = None
    charge_stage: int | None = None
    charge_elapsed_ms: int | None = None
    damage_multiplier: float | None = None
    locked: bool | None = None
    interrupted: bool | None = None
    provenance: SemanticProvenanceV1 = field(
        default_factory=lambda: SemanticProvenanceV1.unknown_all(ATTACK_STATE_FIELDS)
    )
    version: str = field(default=ATTACK_STATE_VERSION, init=False)
    VERSION: ClassVar[str] = ATTACK_STATE_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase", _enum(AttackPhase, self.phase, "attack phase"))
        object.__setattr__(self, "target_position", _pair(self.target_position, cast=float))
        _nonnegative(
            self,
            "target_entity phase_started_tick phase_remaining_ms cooldown_remaining_ms sequence_index sequence_progress sequence_progress_limit sequence_decay_remaining_ms sequence_decay_duration_ms charge_stage charge_elapsed_ms damage_multiplier",
            "attack ",
        )
        if (self.sequence_progress is None) != (self.sequence_progress_limit is None):
            raise ContractError("attack sequence progress pair is incomplete")
        if self.sequence_progress_limit is not None and (
            self.sequence_progress_limit <= 0 or self.sequence_progress > self.sequence_progress_limit
        ):
            raise ContractError("attack sequence progress is outside its limit")
        if (self.sequence_decay_remaining_ms is None) != (self.sequence_decay_duration_ms is None):
            raise ContractError("attack sequence decay pair is incomplete")
        if self.sequence_decay_duration_ms is not None and (
            self.sequence_decay_duration_ms <= 0 or self.sequence_decay_remaining_ms > self.sequence_decay_duration_ms
        ):
            raise ContractError("attack sequence decay is outside its duration")
        _set_provenance(self, "provenance", ATTACK_STATE_FIELDS, "attack state")

    STRICT_LABEL: ClassVar[str] = "attack state"


@dataclass(frozen=True, slots=True)
class ProjectileStateV1(StrictContractMixin):
    projectile_id: str
    phase: ProjectilePhase = ProjectilePhase.UNKNOWN
    source_entity: int | None = None
    source_card_id: int | None = None
    target_entity: int | None = None
    target_position: tuple[float, float] | None = None
    spawn_tick: int | None = None
    expected_impact_tick: int | None = None
    impact_tick: int | None = None
    velocity: tuple[float, float] | None = None
    damage: float | None = None
    radius_tiles: float | None = None
    homing: bool | None = None
    deflected: bool | None = None
    drag_stage: ProjectileDragStage | None = None
    attributes: Mapping[str, Any] = field(default_factory=FrozenMapping)
    provenance: SemanticProvenanceV1 = field(
        default_factory=lambda: SemanticProvenanceV1.unknown_all(PROJECTILE_STATE_FIELDS)
    )
    version: str = field(default=PROJECTILE_STATE_VERSION, init=False)
    VERSION: ClassVar[str] = PROJECTILE_STATE_VERSION

    def __post_init__(self) -> None:
        if not self.projectile_id:
            raise ContractError("projectile_id must not be empty")
        object.__setattr__(self, "phase", _enum(ProjectilePhase, self.phase, "projectile phase"))
        object.__setattr__(self, "target_position", _pair(self.target_position, cast=float))
        object.__setattr__(self, "velocity", _pair(self.velocity, cast=float))
        if self.drag_stage is not None:
            object.__setattr__(self, "drag_stage", _enum(ProjectileDragStage, self.drag_stage, "projectile drag stage"))
        _nonnegative(
            self,
            "source_entity source_card_id target_entity spawn_tick expected_impact_tick impact_tick damage radius_tiles",
            "projectile ",
        )
        object.__setattr__(self, "attributes", frozen_mapping(self.attributes))
        _set_provenance(self, "provenance", PROJECTILE_STATE_FIELDS, "projectile state")

    STRICT_LABEL: ClassVar[str] = "projectile state"


@dataclass(frozen=True, slots=True)
class EntityResourceStateV1(StrictContractMixin):
    kind: str
    current_raw: int
    capacity_raw: int
    normalized: float
    provenance: SemanticProvenanceV1 = field(
        default_factory=lambda: SemanticProvenanceV1.unknown_all(ENTITY_RESOURCE_STATE_FIELDS)
    )
    version: str = field(default=ENTITY_RESOURCE_STATE_VERSION, init=False)
    VERSION: ClassVar[str] = ENTITY_RESOURCE_STATE_VERSION

    def __post_init__(self) -> None:
        if self.kind != "extra_spawn_accumulator":
            raise ContractError("entity resource kind is invalid")
        if self.capacity_raw <= 0:
            raise ContractError("entity resource capacity_raw must be positive")
        if self.current_raw < 0 or self.current_raw > self.capacity_raw:
            raise ContractError("entity resource current_raw is outside capacity")
        expected = self.current_raw / self.capacity_raw
        if not math.isfinite(self.normalized) or not math.isclose(self.normalized, expected, rel_tol=0.0, abs_tol=1e-6):
            raise ContractError("entity resource normalized value is inconsistent")
        _set_provenance(self, "provenance", ENTITY_RESOURCE_STATE_FIELDS, "entity resource state")

    STRICT_LABEL: ClassVar[str] = "entity resource state"


@dataclass(frozen=True, slots=True)
class PeriodicAttackModifierStateV1(StrictContractMixin):
    phase: PeriodicAttackModifierPhase
    period_attacks: int
    completed_attacks: int
    source_entity: int | None
    linger_remaining_ms: int
    linger_duration_ms: int
    provenance: SemanticProvenanceV1 = field(
        default_factory=lambda: SemanticProvenanceV1.unknown_all(PERIODIC_ATTACK_MODIFIER_STATE_FIELDS)
    )
    version: str = field(default=PERIODIC_ATTACK_MODIFIER_STATE_VERSION, init=False)
    VERSION: ClassVar[str] = PERIODIC_ATTACK_MODIFIER_STATE_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "phase", _enum(PeriodicAttackModifierPhase, self.phase, "periodic attack modifier phase")
        )
        if self.period_attacks <= 0:
            raise ContractError("periodic attack modifier period must be positive")
        if not 0 <= self.completed_attacks < self.period_attacks:
            raise ContractError("periodic attack modifier counter is outside its period")
        if self.source_entity is not None and self.source_entity < 0:
            raise ContractError("periodic attack modifier source entity must be non-negative")
        if self.linger_duration_ms <= 0:
            raise ContractError("periodic attack modifier linger duration must be positive")
        if not 0 <= self.linger_remaining_ms <= self.linger_duration_ms:
            raise ContractError("periodic attack modifier linger time is outside its duration")
        if self.phase == PeriodicAttackModifierPhase.SOURCE_ALIVE and self.linger_remaining_ms != 0:
            raise ContractError("source-alive periodic attack modifier cannot carry linger time")
        if self.phase == PeriodicAttackModifierPhase.SOURCE_DEATH_LINGER and self.linger_remaining_ms <= 0:
            raise ContractError("source-death periodic attack modifier requires positive linger time")
        _set_provenance(self, "provenance", PERIODIC_ATTACK_MODIFIER_STATE_FIELDS, "periodic attack modifier state")

    STRICT_LABEL: ClassVar[str] = "periodic attack modifier state"


@dataclass(frozen=True, slots=True)
class CaptureTargetStateV1(StrictContractMixin):
    phase: CaptureTargetPhase
    elapsed_ms: int
    phase_budget_remaining_ms: int | None = None
    target_entity: int | None = None
    version: str = field(default=CAPTURE_TARGET_STATE_VERSION, init=False)
    VERSION: ClassVar[str] = CAPTURE_TARGET_STATE_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase", _enum(CaptureTargetPhase, self.phase, "capture target phase"))
        if self.elapsed_ms < 0:
            raise ContractError("capture target elapsed_ms must be non-negative")
        if self.phase_budget_remaining_ms is not None and self.phase_budget_remaining_ms < 0:
            raise ContractError("capture target phase_budget_remaining_ms must be non-negative")
        if self.target_entity is not None and self.target_entity < 0:
            raise ContractError("capture target entity must be non-negative")
        timed_phases = {CaptureTargetPhase.ACQUIRED_DELAY, CaptureTargetPhase.GRAB_PAUSE, CaptureTargetPhase.DRAGGING}
        if (self.phase in timed_phases) != (self.phase_budget_remaining_ms is not None):
            raise ContractError("capture target phase_budget_remaining_ms does not match its phase")

    STRICT_LABEL: ClassVar[str] = "capture target state"


@dataclass(frozen=True, slots=True)
class CaptureRuntimeStateV1(ContractMixin):
    phase: CaptureActionPhase
    configured_cooldown_ms: int
    cooldown_remaining_ms: int
    hit_frequency_ms: int
    hit_accumulator_ms: int
    targets: tuple[CaptureTargetStateV1, ...] = ()
    provenance: SemanticProvenanceV1 = field(
        default_factory=lambda: SemanticProvenanceV1.unknown_all(CAPTURE_RUNTIME_STATE_FIELDS)
    )
    version: str = field(default=CAPTURE_RUNTIME_STATE_VERSION, init=False)
    VERSION: ClassVar[str] = CAPTURE_RUNTIME_STATE_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase", _enum(CaptureActionPhase, self.phase, "capture action phase"))
        _set_contracts(self, "targets", CaptureTargetStateV1)
        if self.configured_cooldown_ms < 0:
            raise ContractError("capture configured cooldown must be non-negative")
        if not 0 <= self.cooldown_remaining_ms <= self.configured_cooldown_ms:
            raise ContractError("capture cooldown remaining is outside its duration")
        if self.hit_frequency_ms <= 0:
            raise ContractError("capture hit frequency must be positive")
        if self.hit_accumulator_ms < 0:
            raise ContractError("capture hit accumulator must be non-negative")
        expected_phase = (
            CaptureActionPhase.ACTIVE
            if self.targets
            else CaptureActionPhase.RELEASE_COOLDOWN
            if self.cooldown_remaining_ms > 0
            else CaptureActionPhase.IDLE
        )
        if self.phase != expected_phase:
            raise ContractError("capture action phase is inconsistent")
        _set_provenance(self, "provenance", CAPTURE_RUNTIME_STATE_FIELDS, "capture runtime state")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CaptureRuntimeStateV1":
        data = _strict_contract_mapping(
            value,
            expected_version=cls.VERSION,
            allowed_fields={"version", *CAPTURE_RUNTIME_STATE_FIELDS, "provenance"},
            label="capture runtime state",
        )
        data["targets"] = tuple(CaptureTargetStateV1.from_mapping(item) for item in data.get("targets", ()))
        return cls(**data)


@dataclass(frozen=True, slots=True)
class ThresholdRelocationStateV1(StrictContractMixin):
    phase: ThresholdRelocationPhase
    stage: int
    relocation_index: int
    thresholds_percent: tuple[int, ...]
    hide_duration_ms: int
    remaining_ms: int
    burrowed: bool
    provenance: SemanticProvenanceV1 = field(
        default_factory=lambda: SemanticProvenanceV1.unknown_all(THRESHOLD_RELOCATION_STATE_FIELDS)
    )
    version: str = field(default=THRESHOLD_RELOCATION_STATE_VERSION, init=False)
    VERSION: ClassVar[str] = THRESHOLD_RELOCATION_STATE_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase", _enum(ThresholdRelocationPhase, self.phase, "threshold relocation phase"))
        thresholds = tuple(self.thresholds_percent)
        if any(type(value) is not int for value in thresholds):
            raise ContractError("relocation thresholds must be exact integers")
        object.__setattr__(self, "thresholds_percent", thresholds)
        if not self.thresholds_percent:
            raise ContractError("relocation thresholds must not be empty")
        if any(not 0 <= value <= 100 for value in self.thresholds_percent):
            raise ContractError("relocation threshold is outside 0..100 percent")
        if any(right >= left for left, right in zip(self.thresholds_percent, self.thresholds_percent[1:])):
            raise ContractError("relocation thresholds must be strictly descending")
        terminal_stage = 2 * self.threshold_count + 1
        if not 1 <= self.stage <= terminal_stage:
            raise ContractError("relocation stage is outside its threshold graph")
        if self.relocation_index != (self.stage - 1) // 2:
            raise ContractError("relocation index is inconsistent with its stage")
        expected_phase = (
            ThresholdRelocationPhase.EXHAUSTED
            if self.stage == terminal_stage
            else ThresholdRelocationPhase.RELOCATING
            if self.stage % 2 == 0
            else ThresholdRelocationPhase.WAITING_THRESHOLD
        )
        if self.phase != expected_phase:
            raise ContractError("relocation phase is inconsistent with its stage")
        if self.hide_duration_ms <= 0:
            raise ContractError("relocation hide duration must be positive")
        if not 0 <= self.remaining_ms <= self.hide_duration_ms:
            raise ContractError("relocation remaining time is outside its duration")
        if self.burrowed and (self.phase != ThresholdRelocationPhase.RELOCATING or self.remaining_ms == 0):
            raise ContractError("relocation burrowed state is inconsistent")
        _set_provenance(self, "provenance", THRESHOLD_RELOCATION_STATE_FIELDS, "threshold relocation state")

    @property
    def threshold_count(self) -> int:
        return len(self.thresholds_percent)

    STRICT_LABEL: ClassVar[str] = "threshold relocation state"


@dataclass(frozen=True, slots=True)
class VisibilityStateV1(StrictContractMixin):
    phase: VisibilityPhase = VisibilityPhase.UNKNOWN
    public_by_owner: Mapping[str, bool | None] = field(default_factory=FrozenMapping)
    targetable_by_owner: Mapping[str, bool | None] = field(default_factory=FrozenMapping)
    area_damage_eligible: bool | None = None
    transition_remaining_ms: int | None = None
    provenance: SemanticProvenanceV1 = field(
        default_factory=lambda: SemanticProvenanceV1.unknown_all(VISIBILITY_STATE_FIELDS)
    )
    version: str = field(default=VISIBILITY_STATE_VERSION, init=False)
    VERSION: ClassVar[str] = VISIBILITY_STATE_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase", _enum(VisibilityPhase, self.phase, "visibility phase"))
        object.__setattr__(
            self, "public_by_owner", _owner_visibility_mapping(self.public_by_owner, label="public_by_owner")
        )
        object.__setattr__(
            self,
            "targetable_by_owner",
            _owner_visibility_mapping(self.targetable_by_owner, label="targetable_by_owner"),
        )
        if self.transition_remaining_ms is not None and self.transition_remaining_ms < 0:
            raise ContractError("visibility transition_remaining_ms must be non-negative")
        _set_provenance(self, "provenance", VISIBILITY_STATE_FIELDS, "visibility state")

    def assert_public_for(self, owner: int) -> None:
        if owner not in (0, 1):
            raise ContractError("visibility owner must be 0 or 1")
        self.provenance.assert_known(
            ("public_by_owner",), label="visibility state", require_native=True, require_authoritative=True
        )
        if self.public_by_owner.get(str(owner)) is not True:
            raise ContractError(f"entity is not authoritatively public to owner {owner}")

    STRICT_LABEL: ClassVar[str] = "visibility state"


@dataclass(frozen=True, slots=True)
class AbilityRuntimeStateV1(StrictContractMixin):
    ability_id: str
    source_entity: int | None = None
    phase: AbilityPhase = AbilityPhase.UNKNOWN
    elixir_cost: float | None = None
    cooldown_ms: int | None = None
    remaining_cooldown_ms: int | None = None
    charges: int | None = None
    available: bool | None = None
    cast_started_tick: int | None = None
    active_until_tick: int | None = None
    target_entity: int | None = None
    target_position: tuple[float, float] | None = None
    attributes: Mapping[str, Any] = field(default_factory=FrozenMapping)
    provenance: SemanticProvenanceV1 = field(
        default_factory=lambda: SemanticProvenanceV1.unknown_all(ABILITY_RUNTIME_STATE_FIELDS)
    )
    version: str = field(default=ABILITY_RUNTIME_STATE_VERSION, init=False)
    VERSION: ClassVar[str] = ABILITY_RUNTIME_STATE_VERSION

    def __post_init__(self) -> None:
        if not self.ability_id:
            raise ContractError("runtime ability_id must not be empty")
        object.__setattr__(self, "phase", _enum(AbilityPhase, self.phase, "ability phase"))
        object.__setattr__(self, "target_position", _pair(self.target_position, cast=float))
        _nonnegative(
            self,
            "source_entity elixir_cost cooldown_ms remaining_cooldown_ms charges cast_started_tick active_until_tick target_entity",
            "ability ",
        )
        object.__setattr__(self, "attributes", frozen_mapping(self.attributes))
        _set_provenance(self, "provenance", ABILITY_RUNTIME_STATE_FIELDS, "ability runtime state")

    STRICT_LABEL: ClassVar[str] = "ability runtime state"


@dataclass(frozen=True, slots=True)
class EvolutionRuntimeStateV1(StrictContractMixin):
    card_id: int
    deck_slot: int | None = None
    phase: EvolutionPhase = EvolutionPhase.UNKNOWN
    base_form_id: str | None = None
    current_form_id: str | None = None
    next_form_id: str | None = None
    cycle_required: int | None = None
    cycle_remaining: int | None = None
    ready: bool | None = None
    active: bool | None = None
    deployments_in_cycle: int | None = None
    attributes: Mapping[str, Any] = field(default_factory=FrozenMapping)
    provenance: SemanticProvenanceV1 = field(
        default_factory=lambda: SemanticProvenanceV1.unknown_all(EVOLUTION_RUNTIME_STATE_FIELDS)
    )
    version: str = field(default=EVOLUTION_RUNTIME_STATE_VERSION, init=False)
    VERSION: ClassVar[str] = EVOLUTION_RUNTIME_STATE_VERSION

    def __post_init__(self) -> None:
        if self.card_id <= 0:
            raise ContractError("evolution runtime card_id must be positive")
        object.__setattr__(self, "phase", _enum(EvolutionPhase, self.phase, "evolution phase"))
        _nonnegative(self, "deck_slot cycle_required cycle_remaining deployments_in_cycle", "evolution ")
        if self.deck_slot is not None and self.deck_slot > 7:
            raise ContractError("evolution deck_slot must be in 0..7")
        object.__setattr__(self, "attributes", frozen_mapping(self.attributes))
        _set_provenance(self, "provenance", EVOLUTION_RUNTIME_STATE_FIELDS, "evolution runtime state")

    STRICT_LABEL: ClassVar[str] = "evolution runtime state"


@dataclass(frozen=True, slots=True)
class CombatEventV1(StrictContractMixin):
    """Typed public combat semantics.

    ``target_entity`` is the affected object.  A producer must leave source
    fields unknown unless it has positive public evidence for the attacker;
    an enclosing :class:`EventV1.entity_id` is not source evidence.
    """

    kind: CombatEventKind = CombatEventKind.UNKNOWN
    source_entity: int | None = None
    source_card_id: int | None = None
    target_entity: int | None = None
    target_card_id: int | None = None
    position: tuple[float, float] | None = None
    amount: float | None = None
    effect_id: str | None = None
    projectile_id: str | None = None
    ability_id: str | None = None
    evolution_form_id: str | None = None
    visible_by_owner: Mapping[str, bool | None] = field(default_factory=FrozenMapping)
    attributes: Mapping[str, Any] = field(default_factory=FrozenMapping)
    provenance: SemanticProvenanceV1 = field(
        default_factory=lambda: SemanticProvenanceV1.unknown_all(COMBAT_EVENT_FIELDS)
    )
    version: str = field(default=COMBAT_EVENT_VERSION, init=False)
    VERSION: ClassVar[str] = COMBAT_EVENT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum(CombatEventKind, self.kind, "combat event kind"))
        object.__setattr__(self, "position", _pair(self.position, cast=float))
        _nonnegative(self, "source_entity source_card_id target_entity target_card_id amount", "combat event ")
        object.__setattr__(
            self,
            "visible_by_owner",
            _owner_visibility_mapping(self.visible_by_owner, label="combat event visible_by_owner"),
        )
        object.__setattr__(self, "attributes", frozen_mapping(self.attributes))
        _set_provenance(self, "provenance", COMBAT_EVENT_FIELDS, "combat event")

    STRICT_LABEL: ClassVar[str] = "combat event"


@dataclass(frozen=True, slots=True)
class ActionV1(ContractMixin):
    owner: int
    kind: ActionKind
    hand_slot: int | None = None
    card_id: int | None = None
    source_entity: int | None = None
    ability_id: str | None = None
    target_kind: TargetKind = TargetKind.NONE
    target_grid: tuple[int, int] | None = None
    target_entity: int | None = None
    subcell_offset: tuple[float, float] | None = None
    execute_offset_ticks: int = 0
    next_decision_ticks: int = 1
    requested_at_ms: int | None = None
    generated_at_ms: int | None = None
    action_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=FrozenMapping)
    version: str = field(default=ACTION_VERSION, init=False)
    VERSION: ClassVar[str] = ACTION_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum(ActionKind, self.kind, "action kind"))
        object.__setattr__(self, "target_kind", _enum(TargetKind, self.target_kind, "target kind"))
        if self.owner not in (0, 1):
            raise ContractError("action owner must be 0 or 1")
        if self.execute_offset_ticks < 0:
            raise ContractError("execute_offset_ticks must be non-negative")
        if self.next_decision_ticks < 1:
            raise ContractError("next_decision_ticks must be positive")
        if self.hand_slot is not None and not 0 <= self.hand_slot < 4:
            raise ContractError("hand_slot must be in 0..3")
        if self.card_id is not None and self.card_id <= 0:
            raise ContractError("card_id must be positive")
        if self.source_entity is not None and self.source_entity < 0:
            raise ContractError("source_entity must be non-negative")
        if self.target_entity is not None and self.target_entity < 0:
            raise ContractError("target_entity must be non-negative")
        grid = _pair(self.target_grid, cast=int)
        offset = _pair(self.subcell_offset, cast=float)
        object.__setattr__(self, "target_grid", grid)
        object.__setattr__(self, "subcell_offset", offset)
        if grid is not None and not (0 <= grid[0] < BOARD_WIDTH and 0 <= grid[1] < BOARD_HEIGHT):
            raise ContractError(f"target_grid must be inside {BOARD_WIDTH}x{BOARD_HEIGHT}")
        if offset is not None and any(not -0.5 <= item <= 0.5 for item in offset):
            raise ContractError("subcell_offset values must be in [-0.5, 0.5]")
        grid_targets = {TargetKind.GRID, TargetKind.AREA_CENTER, TargetKind.DIRECTION}
        entity_targets = {TargetKind.FRIENDLY_ENTITY, TargetKind.ENEMY_ENTITY}
        if self.target_kind in grid_targets and grid is None:
            raise ContractError(f"{self.target_kind.value} actions require target_grid")
        if self.target_kind in entity_targets and self.target_entity is None:
            raise ContractError(f"{self.target_kind.value} actions require target_entity")
        if self.target_kind == TargetKind.NONE and (grid is not None or self.target_entity is not None):
            raise ContractError("target_kind=none cannot carry a target")
        if self.kind == ActionKind.PLAY_CARD and self.hand_slot is None:
            raise ContractError("play_card requires hand_slot")
        if self.kind == ActionKind.WAIT_UNTIL_AFFORDABLE and self.hand_slot is None:
            raise ContractError("wait_until_affordable requires hand_slot")
        if self.kind == ActionKind.ACTIVATE_ABILITY and self.source_entity is None:
            raise ContractError("activate_ability requires source_entity")
        if self.kind == ActionKind.WAIT and any(
            item is not None for item in (self.hand_slot, self.card_id, self.source_entity, self.ability_id)
        ):
            raise ContractError("wait actions cannot carry a source")
        object.__setattr__(self, "metadata", frozen_mapping(self.metadata))

    @classmethod
    def wait(cls, owner: int, ticks: int = 1, **kwargs: Any) -> "ActionV1":
        return cls(owner=owner, kind=ActionKind.WAIT, next_decision_ticks=ticks, **kwargs)

    @classmethod
    def play(cls, owner: int, hand_slot: int, grid: tuple[int, int], **kwargs: Any) -> "ActionV1":
        return cls(
            owner=owner,
            kind=ActionKind.PLAY_CARD,
            hand_slot=hand_slot,
            target_kind=TargetKind.GRID,
            target_grid=grid,
            **kwargs,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ActionV1":
        _version(value, ACTION_VERSION)
        target = value.get("target")
        target_grid = value.get("target_grid")
        target_entity = value.get("target_entity", value.get("target_entity_id"))
        if isinstance(target, Mapping):
            target_grid = target_grid if target_grid is not None else target.get("grid")
            target_entity = target_entity if target_entity is not None else target.get("entity")
        return cls(
            owner=int(value["owner"]),
            kind=value.get("kind", value.get("candidate", ActionKind.WAIT.value)),
            hand_slot=value.get("hand_slot"),
            card_id=value.get("card_id"),
            source_entity=value.get("source_entity", value.get("source_entity_id")),
            ability_id=value.get("ability_id"),
            target_kind=value.get("target_kind", TargetKind.NONE.value),
            target_grid=target_grid,
            target_entity=target_entity,
            subcell_offset=value.get("subcell_offset", value.get("target_offset")),
            execute_offset_ticks=int(value.get("execute_offset_ticks", 0)),
            next_decision_ticks=int(value.get("next_decision_ticks", 1)),
            requested_at_ms=value.get("requested_at_ms"),
            generated_at_ms=value.get("generated_at_ms"),
            action_id=value.get("action_id"),
            metadata=value.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class TimeStateV1(ContractMixin):
    elapsed_ms: int = 0
    remaining_ms: int | None = None
    server_time_ms: int | None = None
    capture_time_ms: int | None = None
    available_time_ms: int | None = None
    elixir_multiplier: float = 1.0
    tick_ms: int = DEFAULT_TICK_MS
    VERSION: ClassVar[str] = "time-state.v1"

    def __post_init__(self) -> None:
        _nonnegative(self, "elapsed_ms remaining_ms server_time_ms capture_time_ms available_time_ms", "")
        if self.tick_ms <= 0:
            raise ContractError("tick_ms must be positive")
        if self.elixir_multiplier <= 0:
            raise ContractError("elixir_multiplier must be positive")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None, *, tick: int = 0) -> "TimeStateV1":
        data = value or {}
        tick_ms = int(data.get("tick_ms", DEFAULT_TICK_MS))
        return cls(
            elapsed_ms=int(data.get("elapsed_ms", tick * tick_ms)),
            remaining_ms=data.get("remaining_ms"),
            server_time_ms=data.get("server_time_ms"),
            capture_time_ms=data.get("capture_time_ms", data.get("observation_time_ms")),
            available_time_ms=data.get("available_time_ms"),
            elixir_multiplier=float(data.get("elixir_multiplier", 1.0)),
            tick_ms=tick_ms,
        )


def _card_id(value: Any) -> int:
    if isinstance(value, Mapping):
        value = value.get("card_id", value.get("cardId", value.get("id", 0)))
    return int(value)


@dataclass(frozen=True, slots=True)
class PlayerStateV1(ContractMixin):
    owner: int
    crowns: int = 0
    tower_ids: tuple[int, ...] = ()
    elixir_exact: float | None = None
    elixir_visible: float | None = None
    hand: tuple[int, ...] = ()
    next_card: int | None = None
    deck: tuple[int, ...] = ()
    cycle: tuple[int, ...] = ()
    revealed_cards: tuple[int, ...] = ()
    evolution_state: Mapping[str, Any] = field(default_factory=FrozenMapping)
    ability_state: Mapping[str, Any] = field(default_factory=FrozenMapping)
    private_state_visible: bool = False
    metadata: Mapping[str, Any] = field(default_factory=FrozenMapping)
    ability_runtime_states: tuple[AbilityRuntimeStateV1, ...] = ()
    evolution_runtime_states: tuple[EvolutionRuntimeStateV1, ...] = ()
    runtime_provenance: SemanticProvenanceV1 = field(
        default_factory=lambda: SemanticProvenanceV1.unknown_all(PLAYER_RUNTIME_SEMANTIC_FIELDS)
    )
    VERSION: ClassVar[str] = "player-state.v1"

    def __post_init__(self) -> None:
        if self.owner not in (0, 1):
            raise ContractError("player owner must be 0 or 1")
        if not 0 <= self.crowns <= 3:
            raise ContractError("crowns must be in 0..3")
        for label, value in (("elixir_exact", self.elixir_exact), ("elixir_visible", self.elixir_visible)):
            if value is not None and not 0 <= value <= 10.0001:
                raise ContractError(f"{label} must be in 0..10")
        for label, cards in (
            ("hand", self.hand),
            ("deck", self.deck),
            ("cycle", self.cycle),
            ("revealed_cards", self.revealed_cards),
        ):
            normalized = _int_tuple(cards)
            if any(card <= 0 for card in normalized):
                raise ContractError(f"{label} contains an invalid card ID")
            object.__setattr__(self, label, normalized)
        if self.hand and len(self.hand) > 4:
            raise ContractError("hand cannot contain more than four cards")
        if self.deck and len(self.deck) > 8:
            raise ContractError("deck cannot contain more than eight cards")
        if self.next_card is not None and self.next_card <= 0:
            raise ContractError("next_card must be positive")
        object.__setattr__(self, "tower_ids", _int_tuple(self.tower_ids))
        object.__setattr__(self, "evolution_state", frozen_mapping(self.evolution_state))
        object.__setattr__(self, "ability_state", frozen_mapping(self.ability_state))
        object.__setattr__(self, "metadata", frozen_mapping(self.metadata))
        _set_contracts(self, "ability_runtime_states", AbilityRuntimeStateV1)
        _set_contracts(self, "evolution_runtime_states", EvolutionRuntimeStateV1)
        _set_provenance(self, "runtime_provenance", PLAYER_RUNTIME_SEMANTIC_FIELDS, "player runtime state")
        _validate_domains(self, "player", plural="ability_runtime_states evolution_runtime_states")
        if not self.private_state_visible and (
            self.elixir_exact is not None or self.hand or self.next_card is not None or self.deck or self.cycle
        ):
            raise ContractError("private_state_visible=false cannot carry exact elixir/hand/deck/cycle")
        if not self.private_state_visible and (self.ability_runtime_states or self.evolution_runtime_states):
            raise ContractError("private_state_visible=false cannot carry exact ability/evolution runtime")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, private_default: bool = False) -> "PlayerStateV1":
        private = bool(value.get("private_state_visible", value.get("privateStateVisible", private_default)))
        raw_elixir = value.get("elixir_exact", value.get("elixirRaw"))
        if raw_elixir is not None and "elixirRaw" in value and float(raw_elixir) > 10:
            # The v15 probe exposes fixed-point 1/10000 elixir (60000 == 6.0).
            raw_elixir = float(raw_elixir) / 10000.0
        visible_elixir = value.get("elixir_visible")
        if visible_elixir is None and private:
            # Probe ``elixir`` is derived from the exact internal fixed-point
            # value.  It is safe for the owner/oracle, never for an opponent in
            # a fair observation unless a wrapper supplies elixir_visible.
            visible_elixir = value.get("elixir")
        return cls(
            owner=int(value["owner"]),
            crowns=int(value.get("crowns", 0)),
            tower_ids=_int_tuple(value.get("tower_ids", ())),
            elixir_exact=raw_elixir if private else None,
            elixir_visible=visible_elixir,
            hand=tuple(_card_id(item) for item in value.get("hand", ())) if private else (),
            next_card=_card_id(value.get("next_card", value.get("nextCard")))
            if private and value.get("next_card", value.get("nextCard")) is not None
            else None,
            deck=tuple(_card_id(item) for item in value.get("deck", ())) if private else (),
            cycle=tuple(_card_id(item) for item in value.get("cycle", ())) if private else (),
            revealed_cards=tuple(_card_id(item) for item in value.get("revealed_cards", ())),
            evolution_state=value.get("evolution_state", {}),
            ability_state=value.get("ability_state", {}),
            private_state_visible=private,
            metadata=value.get("metadata", {}),
            ability_runtime_states=tuple(
                AbilityRuntimeStateV1.from_mapping(item) for item in value.get("ability_runtime_states", ())
            ),
            evolution_runtime_states=tuple(
                EvolutionRuntimeStateV1.from_mapping(item) for item in value.get("evolution_runtime_states", ())
            ),
            runtime_provenance=(_mapping_provenance(value.get("runtime_provenance"), PLAYER_RUNTIME_SEMANTIC_FIELDS)),
        )


@dataclass(frozen=True, slots=True)
class DaggerDuchessRuntimeStateV1(ContractMixin):
    charge_count: int
    max_charge_count: int
    recharge_elapsed_ms: int
    recharge_duration_ms: int
    version: str = field(default=DAGGER_DUCHESS_RUNTIME_VERSION, init=False)
    VERSION: ClassVar[str] = DAGGER_DUCHESS_RUNTIME_VERSION

    def __post_init__(self) -> None:
        if (
            self.max_charge_count <= 0
            or not 0 <= self.charge_count <= self.max_charge_count
            or self.recharge_duration_ms <= 0
            or not 0 <= self.recharge_elapsed_ms <= self.recharge_duration_ms
        ):
            raise ContractError("invalid Dagger Duchess charge runtime")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DaggerDuchessRuntimeStateV1":
        _version(value, DAGGER_DUCHESS_RUNTIME_VERSION)
        return cls(
            charge_count=int(value["charge_count"]),
            max_charge_count=int(value["max_charge_count"]),
            recharge_elapsed_ms=int(value["recharge_elapsed_ms"]),
            recharge_duration_ms=int(value["recharge_duration_ms"]),
        )


@dataclass(frozen=True, slots=True)
class RoyalChefRuntimeStateV1(ContractMixin):
    start_delay_remaining_ms: int
    start_delay_duration_ms: int
    cooking_contribution: int
    contribution_needed: int
    surviving_side_towers: int
    throw_delay_remaining_ms: int | None = None
    target_entity: int | None = None
    version: str = field(default=ROYAL_CHEF_RUNTIME_VERSION, init=False)
    VERSION: ClassVar[str] = ROYAL_CHEF_RUNTIME_VERSION

    def __post_init__(self) -> None:
        if (
            self.start_delay_duration_ms <= 0
            or not 0 <= self.start_delay_remaining_ms <= self.start_delay_duration_ms
            or self.cooking_contribution < 0
            or self.contribution_needed <= 0
            or self.surviving_side_towers not in (0, 1, 2)
        ):
            raise ContractError("invalid Royal Chef cooking runtime")
        if self.throw_delay_remaining_ms is None and self.target_entity is not None:
            raise ContractError("Royal Chef cannot expose a target without a pending throw")
        if self.throw_delay_remaining_ms is not None and (
            self.throw_delay_remaining_ms < -50
            or (self.throw_delay_remaining_ms < 0 and self.throw_delay_remaining_ms != -50)
            or (self.target_entity is not None and self.target_entity < 0)
            or (self.throw_delay_remaining_ms == -50 and self.target_entity is not None)
        ):
            raise ContractError("invalid Royal Chef pending throw runtime")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RoyalChefRuntimeStateV1":
        _version(value, ROYAL_CHEF_RUNTIME_VERSION)
        return cls(
            start_delay_remaining_ms=int(value["start_delay_remaining_ms"]),
            start_delay_duration_ms=int(value["start_delay_duration_ms"]),
            cooking_contribution=int(value["cooking_contribution"]),
            contribution_needed=int(value["contribution_needed"]),
            surviving_side_towers=int(value["surviving_side_towers"]),
            throw_delay_remaining_ms=(
                int(value["throw_delay_remaining_ms"]) if value.get("throw_delay_remaining_ms") is not None else None
            ),
            target_entity=(int(value["target_entity"]) if value.get("target_entity") is not None else None),
        )


TowerTroopRuntimeStateV1 = DaggerDuchessRuntimeStateV1 | RoyalChefRuntimeStateV1


@dataclass(frozen=True, slots=True)
class TowerStateV1(ContractMixin):
    entity_id: int
    owner: int
    tower_kind: str
    position: tuple[float, float]
    hitpoints: float
    max_hitpoints: float
    tower_troop_id: int | None = None
    shield: float = 0.0
    active: bool = True
    visible_target: int | None = None
    status: tuple[str, ...] = ()
    internal: Mapping[str, Any] = field(default_factory=FrozenMapping)
    shield_state: ShieldStateV1 | None = None
    effect_states: tuple[EffectStateV1, ...] = ()
    attack_state: AttackStateV1 | None = None
    visibility_state: VisibilityStateV1 | None = None
    tower_troop_runtime: TowerTroopRuntimeStateV1 | None = None
    runtime_provenance: SemanticProvenanceV1 = field(
        default_factory=lambda: SemanticProvenanceV1.unknown_all(TOWER_RUNTIME_SEMANTIC_FIELDS)
    )
    VERSION: ClassVar[str] = "tower-state.v1"

    def __post_init__(self) -> None:
        if self.entity_id < 0 or self.owner not in (0, 1):
            raise ContractError("invalid tower identity")
        object.__setattr__(self, "position", _pair(self.position, cast=float))
        if self.hitpoints < 0 or self.max_hitpoints <= 0 or self.hitpoints > self.max_hitpoints:
            raise ContractError("invalid tower hitpoints")
        if self.shield < 0:
            raise ContractError("tower shield must be non-negative")
        if self.tower_kind == "king" and self.tower_troop_id not in (None, ROYAL_CHEF_TOWER_TROOP_ID):
            raise ContractError("king tower only accepts the Royal Chef identity")
        if self.tower_kind != "king" and self.tower_troop_id is None:
            raise ContractError("side tower requires an exact tower troop ID")
        if self.tower_kind != "king" and self.tower_troop_id == ROYAL_CHEF_TOWER_TROOP_ID:
            raise ContractError("Royal Chef identity belongs to the king tower")
        if self.tower_troop_id is not None and self.tower_troop_id not in COMPETITIVE_TOWER_TROOP_IDS:
            raise ContractError("tower troop ID is not in the current competitive pool")
        object.__setattr__(self, "status", tuple(str(item) for item in self.status))
        object.__setattr__(self, "internal", frozen_mapping(self.internal))
        _set_contract(self, "shield_state", ShieldStateV1)
        _set_contracts(self, "effect_states", EffectStateV1)
        _set_contract(self, "attack_state", AttackStateV1)
        _set_contract(self, "visibility_state", VisibilityStateV1)
        tower_runtime = self.tower_troop_runtime
        if tower_runtime is not None and not isinstance(
            tower_runtime, (DaggerDuchessRuntimeStateV1, RoyalChefRuntimeStateV1)
        ):
            if not isinstance(tower_runtime, Mapping):
                raise ContractError("tower troop runtime must be a V1 mapping")
            version = tower_runtime.get("version")
            if version == DAGGER_DUCHESS_RUNTIME_VERSION:
                tower_runtime = DaggerDuchessRuntimeStateV1.from_mapping(tower_runtime)
            elif version == ROYAL_CHEF_RUNTIME_VERSION:
                tower_runtime = RoyalChefRuntimeStateV1.from_mapping(tower_runtime)
            else:
                raise ContractError("unknown tower troop runtime version")
            object.__setattr__(self, "tower_troop_runtime", tower_runtime)
        if isinstance(tower_runtime, DaggerDuchessRuntimeStateV1) and (
            self.tower_kind == "king" or self.tower_troop_id != 159_000_002
        ):
            raise ContractError("Dagger Duchess runtime requires a Dagger Duchess side tower")
        if isinstance(tower_runtime, RoyalChefRuntimeStateV1) and (
            self.tower_kind != "king" or self.tower_troop_id != ROYAL_CHEF_TOWER_TROOP_ID
        ):
            raise ContractError("Royal Chef cooking runtime belongs to the king tower")
        _set_provenance(self, "runtime_provenance", TOWER_RUNTIME_SEMANTIC_FIELDS, "tower runtime state")
        _validate_domains(self, "tower", singular="shield_state attack_state visibility_state")
        _require_presence_evidence(
            bool(self.effect_states),
            self.runtime_provenance.field_evidence["effect_states"],
            "tower effect_states carry data without positive provenance",
        )
        _require_presence_evidence(
            tower_runtime is not None,
            self.runtime_provenance.field_evidence["tower_troop_runtime"],
            "tower troop runtime carries data without positive provenance",
            "tower troop runtime provenance claims absent data",
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TowerStateV1":
        return cls(
            entity_id=int(value.get("entity_id", value.get("id", 0))),
            owner=int(value["owner"]),
            tower_kind=str(value.get("tower_kind", value.get("kind", "unknown"))),
            position=value.get("position", (value.get("x", 0), value.get("y", 0))),
            hitpoints=float(value.get("hitpoints", value.get("hp", 0))),
            max_hitpoints=float(value.get("max_hitpoints", value.get("maxHp", 1))),
            tower_troop_id=(int(value["tower_troop_id"]) if value.get("tower_troop_id") is not None else None),
            shield=float(value.get("shield", 0)),
            active=bool(value.get("active", True)),
            visible_target=value.get("visible_target", value.get("target")),
            status=tuple(value.get("status", ())),
            internal=value.get("internal", {}),
            shield_state=(_optional_contract(ShieldStateV1, value.get("shield_state"))),
            effect_states=tuple(EffectStateV1.from_mapping(item) for item in value.get("effect_states", ())),
            attack_state=(_optional_contract(AttackStateV1, value.get("attack_state"))),
            visibility_state=(_optional_contract(VisibilityStateV1, value.get("visibility_state"))),
            tower_troop_runtime=value.get("tower_troop_runtime"),
            runtime_provenance=(_mapping_provenance(value.get("runtime_provenance"), TOWER_RUNTIME_SEMANTIC_FIELDS)),
        )


def _runtime_extension_mapping(
    value: Mapping[str, Any] | None, *, label: str, version: str, fields: frozenset[str], phases: frozenset[str]
) -> FrozenMapping | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"version", *fields}:
        raise ContractError(f"{label} fields do not match {version}")
    if value.get("version") != version or value.get("phase") not in phases:
        raise ContractError(f"{label} version/phase is invalid")
    observed_tick = value.get("observed_tick")
    if isinstance(observed_tick, bool) or not isinstance(observed_tick, int) or observed_tick < 0:
        raise ContractError(f"{label} observed_tick is invalid")
    if not isinstance(value.get("complete"), bool):
        raise ContractError(f"{label} complete marker is invalid")
    for name, number in value.items():
        if name in {"version", "phase", "classic_charge_phase", "complete"}:
            continue
        if number is not None and (isinstance(number, bool) or not isinstance(number, int)):
            raise ContractError(f"{label} {name} must be an integer or null")
    return frozen_mapping(value)


@dataclass(frozen=True, slots=True)
class CausalGroupRefV1(ContractMixin):
    """Opaque public address shared by children of one causal root.

    ``handle`` is deliberately an addressing field, not a model feature.  Card
    and parent identities describe the group but never participate in group
    equality.
    """

    kind: CausalGroupKind
    handle: str
    source_card_id: int | None = None
    parent_entity_id: int | None = None
    version: str = field(default=CAUSAL_GROUP_REF_VERSION, init=False)
    VERSION: ClassVar[str] = CAUSAL_GROUP_REF_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum(CausalGroupKind, self.kind, "causal group kind"))
        handle = str(self.handle)
        if not handle:
            raise ContractError("causal group handle must not be empty")
        object.__setattr__(self, "handle", handle)
        if self.source_card_id is not None and (
            isinstance(self.source_card_id, bool)
            or not isinstance(self.source_card_id, int)
            or self.source_card_id <= 0
        ):
            raise ContractError("causal group source_card_id must be positive")
        if self.parent_entity_id is not None and (
            isinstance(self.parent_entity_id, bool)
            or not isinstance(self.parent_entity_id, int)
            or self.parent_entity_id < 0
        ):
            raise ContractError("causal group parent_entity_id must be non-negative")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CausalGroupRefV1":
        _version(value, CAUSAL_GROUP_REF_VERSION)
        return cls(
            kind=value["kind"],
            handle=str(value["handle"]),
            source_card_id=value.get("source_card_id"),
            parent_entity_id=value.get("parent_entity_id"),
        )


@dataclass(frozen=True, slots=True)
class EntityStateV1(ContractMixin):
    entity_id: int
    owner: int | None
    card_id: int | None
    entity_kind: str
    position: tuple[float, float]
    velocity: tuple[float, float] | None = None
    hitpoints: float | None = None
    max_hitpoints: float | None = None
    shield: float | None = None
    age_ms: int | None = None
    visible_target: int | None = None
    movement_target: tuple[float, float] | None = None
    attack_phase: str | None = None
    status: tuple[str, ...] = ()
    effects: tuple[str, ...] = ()
    source_entity: int | None = None
    causal_group: CausalGroupRefV1 | None = None
    visible: bool | None = None
    native_data_global_id: int | None = None
    internal: Mapping[str, Any] = field(default_factory=FrozenMapping)
    shield_state: ShieldStateV1 | None = None
    effect_states: tuple[EffectStateV1, ...] = ()
    attack_state: AttackStateV1 | None = None
    movement_runtime: Mapping[str, Any] | None = None
    deployment_runtime: Mapping[str, Any] | None = None
    projectile_state: ProjectileStateV1 | None = None
    visibility_state: VisibilityStateV1 | None = None
    ability_states: tuple[AbilityRuntimeStateV1, ...] = ()
    evolution_state: EvolutionRuntimeStateV1 | None = None
    resource_states: tuple[EntityResourceStateV1, ...] = ()
    capture_runtime: CaptureRuntimeStateV1 | None = None
    threshold_relocation_runtime: ThresholdRelocationStateV1 | None = None
    periodic_attack_modifier: PeriodicAttackModifierStateV1 | None = None
    runtime_provenance: SemanticProvenanceV1 = field(
        default_factory=lambda: SemanticProvenanceV1.unknown_all(ENTITY_RUNTIME_SEMANTIC_FIELDS)
    )
    VERSION: ClassVar[str] = "entity-state.v1"

    def __post_init__(self) -> None:
        if self.entity_id < 0:
            raise ContractError("entity_id must be non-negative")
        if self.owner is not None and self.owner not in (0, 1):
            raise ContractError("entity owner must be 0, 1, or null")
        if self.card_id is not None and self.card_id <= 0:
            raise ContractError("entity card_id must be positive")
        if self.native_data_global_id is not None and (
            isinstance(self.native_data_global_id, bool)
            or not isinstance(self.native_data_global_id, int)
            or self.native_data_global_id <= 0
        ):
            raise ContractError("entity native_data_global_id must be positive")
        object.__setattr__(self, "position", _pair(self.position, cast=float))
        object.__setattr__(self, "velocity", _pair(self.velocity, cast=float))
        object.__setattr__(self, "movement_target", _pair(self.movement_target, cast=float))
        _nonnegative(self, "hitpoints max_hitpoints shield age_ms", "entity ")
        # Native maxHp is the character's static maximum field.  Mechanics
        # such as Evolution Witch overheal can legitimately retain current HP
        # above it, and that ratio is model-visible information rather than an
        # invalid observation.  Non-negativity remains enforced above.
        object.__setattr__(self, "status", tuple(str(item) for item in self.status))
        object.__setattr__(self, "effects", tuple(str(item) for item in self.effects))
        _set_contract(self, "causal_group", CausalGroupRefV1)
        object.__setattr__(self, "internal", frozen_mapping(self.internal))
        _set_contract(self, "shield_state", ShieldStateV1)
        _set_contracts(self, "effect_states", EffectStateV1)
        _set_contract(self, "attack_state", AttackStateV1)
        object.__setattr__(
            self,
            "movement_runtime",
            _runtime_extension_mapping(
                self.movement_runtime,
                label="entity movement runtime",
                version="movement-runtime.v1",
                fields=frozenset(
                    {
                        "phase",
                        "effective_speed",
                        "effect_scaled_speed",
                        "movement_delta",
                        "classic_charge_phase",
                        "classic_charge_progress",
                        "classic_charge_speed_multiplier",
                        "observed_tick",
                        "complete",
                    }
                ),
                phases=frozenset({"unavailable", "stationary", "moving", "unknown"}),
            ),
        )
        if self.movement_runtime is not None and self.movement_runtime.get("classic_charge_phase") not in {
            "unavailable",
            "accumulating",
            "ready",
        }:
            raise ContractError("entity movement runtime classic_charge_phase is invalid")
        object.__setattr__(
            self,
            "deployment_runtime",
            _runtime_extension_mapping(
                self.deployment_runtime,
                label="entity deployment runtime",
                version="deployment-runtime.v1",
                fields=frozenset(
                    {
                        "phase",
                        "remaining_native_ms",
                        "remaining_wall_ms",
                        "configured_deploy_time_ms",
                        "observed_step_native_ms",
                        "observed_tick",
                        "complete",
                    }
                ),
                phases=frozenset({"deploying", "active", "unknown"}),
            ),
        )
        _set_contract(self, "projectile_state", ProjectileStateV1)
        _set_contract(self, "visibility_state", VisibilityStateV1)
        _set_contracts(self, "ability_states", AbilityRuntimeStateV1)
        _set_contract(self, "evolution_state", EvolutionRuntimeStateV1)
        _set_contracts(self, "resource_states", EntityResourceStateV1)
        resource_kinds = [item.kind for item in self.resource_states]
        if len(resource_kinds) != len(set(resource_kinds)):
            raise ContractError("entity resource state kinds must be unique")
        _set_contract(self, "capture_runtime", CaptureRuntimeStateV1)
        _set_contract(self, "threshold_relocation_runtime", ThresholdRelocationStateV1)
        _set_contract(self, "periodic_attack_modifier", PeriodicAttackModifierStateV1)
        provenance_value = self.runtime_provenance
        if isinstance(provenance_value, SemanticProvenanceV1):
            extension_fields = {
                "movement_runtime",
                "deployment_runtime",
                "resource_states",
                "capture_runtime",
                "threshold_relocation_runtime",
                "periodic_attack_modifier",
            }
            provided_fields = set(provenance_value.field_evidence)
            missing_extensions = ENTITY_RUNTIME_SEMANTIC_FIELDS.difference(provided_fields)
            if (
                provided_fields <= ENTITY_RUNTIME_SEMANTIC_FIELDS
                and missing_extensions
                and missing_extensions <= extension_fields
            ):
                provenance_value = SemanticProvenanceV1(
                    field_evidence={
                        **dict(provenance_value.field_evidence),
                        **{field_name: SemanticEvidenceLevel.UNKNOWN for field_name in missing_extensions},
                    },
                    source_fields=dict(provenance_value.source_fields),
                    observed_tick=provenance_value.observed_tick,
                    notes=provenance_value.notes,
                )
        runtime_provenance = _semantic_provenance(
            provenance_value, ENTITY_RUNTIME_SEMANTIC_FIELDS, label="entity runtime state"
        )
        object.__setattr__(self, "runtime_provenance", runtime_provenance)
        _validate_domains(
            self,
            "entity",
            singular=(
                "shield_state attack_state movement_runtime deployment_runtime "
                "projectile_state visibility_state evolution_state capture_runtime "
                "threshold_relocation_runtime periodic_attack_modifier"
            ),
            plural="effect_states ability_states resource_states",
        )
        if self.shield is not None and self.shield_state is not None:
            if self.shield_state.hitpoints is not None and self.shield != self.shield_state.hitpoints:
                raise ContractError("legacy shield conflicts with typed shield_state")

    def assert_runtime_semantics(self, required_domains: Iterable[str], *, require_authoritative: bool = False) -> None:
        """Fail closed unless the requested typed runtime domains are present."""

        self.runtime_provenance.assert_known(
            required_domains,
            label=f"entity {self.entity_id} runtime state",
            require_native=True,
            require_authoritative=require_authoritative,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EntityStateV1":
        return cls(
            entity_id=int(value.get("entity_id", value.get("id", value.get("slot", 0)))),
            owner=value.get("owner"),
            card_id=value.get("card_id", value.get("cardId")),
            entity_kind=str(value.get("entity_kind", value.get("kind", value.get("type", "unknown")))),
            position=value.get("position", (value.get("x", 0), value.get("y", 0))),
            velocity=value.get("velocity"),
            hitpoints=value.get("hitpoints", value.get("hp")),
            max_hitpoints=value.get("max_hitpoints", value.get("maxHp")),
            shield=value.get("shield"),
            age_ms=value.get("age_ms"),
            visible_target=value.get("visible_target", value.get("target")),
            movement_target=value.get("movement_target"),
            attack_phase=value.get("attack_phase"),
            status=tuple(value.get("status", ())),
            effects=tuple(value.get("effects", ())),
            source_entity=value.get("source_entity"),
            causal_group=(_optional_contract(CausalGroupRefV1, value.get("causal_group"))),
            visible=value.get("visible"),
            native_data_global_id=value.get("native_data_global_id", value.get("dataGlobalId")),
            internal=value.get("internal", {}),
            shield_state=(_optional_contract(ShieldStateV1, value.get("shield_state"))),
            effect_states=tuple(EffectStateV1.from_mapping(item) for item in value.get("effect_states", ())),
            attack_state=(_optional_contract(AttackStateV1, value.get("attack_state"))),
            movement_runtime=value.get("movement_runtime"),
            deployment_runtime=value.get("deployment_runtime"),
            projectile_state=(_optional_contract(ProjectileStateV1, value.get("projectile_state"))),
            visibility_state=(_optional_contract(VisibilityStateV1, value.get("visibility_state"))),
            ability_states=tuple(AbilityRuntimeStateV1.from_mapping(item) for item in value.get("ability_states", ())),
            evolution_state=(_optional_contract(EvolutionRuntimeStateV1, value.get("evolution_state"))),
            resource_states=tuple(
                EntityResourceStateV1.from_mapping(item) for item in value.get("resource_states", ())
            ),
            capture_runtime=(_optional_contract(CaptureRuntimeStateV1, value.get("capture_runtime"))),
            threshold_relocation_runtime=(
                _optional_contract(ThresholdRelocationStateV1, value.get("threshold_relocation_runtime"))
            ),
            periodic_attack_modifier=(
                _optional_contract(PeriodicAttackModifierStateV1, value.get("periodic_attack_modifier"))
            ),
            runtime_provenance=(_mapping_provenance(value.get("runtime_provenance"), ENTITY_RUNTIME_SEMANTIC_FIELDS)),
        )


@dataclass(frozen=True, slots=True)
class EventV1(ContractMixin):
    tick: int
    event_type: str
    owner: int | None = None
    entity_id: int | None = None
    card_id: int | None = None
    position: tuple[float, float] | None = None
    visible: bool = True
    data: Mapping[str, Any] = field(default_factory=FrozenMapping)
    combat: CombatEventV1 | None = None
    runtime_provenance: SemanticProvenanceV1 = field(
        default_factory=lambda: SemanticProvenanceV1.unknown_all(EVENT_RUNTIME_SEMANTIC_FIELDS)
    )
    VERSION: ClassVar[str] = "event.v1"

    def __post_init__(self) -> None:
        if self.tick < 0 or not self.event_type:
            raise ContractError("event tick/type is invalid")
        if self.owner is not None and self.owner not in (0, 1):
            raise ContractError("event owner must be 0, 1, or null")
        object.__setattr__(self, "position", _pair(self.position, cast=float))
        object.__setattr__(self, "data", frozen_mapping(self.data))
        _set_contract(self, "combat", CombatEventV1)
        _set_provenance(self, "runtime_provenance", EVENT_RUNTIME_SEMANTIC_FIELDS, "event runtime state")
        _require_presence_evidence(
            self.combat is not None,
            self.runtime_provenance.field_evidence["combat"],
            "event combat data requires positive provenance",
            "event combat provenance claims absent data",
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EventV1":
        return cls(
            tick=int(value["tick"]),
            event_type=str(value.get("event_type", value.get("type", "unknown"))),
            owner=value.get("owner"),
            entity_id=value.get("entity_id"),
            card_id=value.get("card_id", value.get("cardId")),
            position=value.get("position"),
            visible=bool(value.get("visible", True)),
            data=value.get("data", {}),
            combat=(_optional_contract(CombatEventV1, value.get("combat"))),
            runtime_provenance=(_mapping_provenance(value.get("runtime_provenance"), EVENT_RUNTIME_SEMANTIC_FIELDS)),
        )


@dataclass(frozen=True, slots=True)
class OpponentBeliefV1(ContractMixin):
    opponent_owner: int
    deck_probabilities: Mapping[str, float] = field(default_factory=FrozenMapping)
    hand_hypotheses: tuple[Mapping[str, Any], ...] = ()
    next_card_probabilities: Mapping[str, float] = field(default_factory=FrozenMapping)
    elixir_probabilities: tuple[float, ...] = ()
    evolution_hypotheses: Mapping[str, Any] = field(default_factory=FrozenMapping)
    ability_hypotheses: Mapping[str, Any] = field(default_factory=FrozenMapping)
    updated_tick: int = 0
    VERSION: ClassVar[str] = "opponent-belief.v1"

    def __post_init__(self) -> None:
        if self.opponent_owner not in (0, 1) or self.updated_tick < 0:
            raise ContractError("invalid opponent belief identity/tick")
        object.__setattr__(self, "deck_probabilities", frozen_mapping(self.deck_probabilities))
        object.__setattr__(self, "hand_hypotheses", tuple(frozen_mapping(item) for item in self.hand_hypotheses))
        object.__setattr__(self, "next_card_probabilities", frozen_mapping(self.next_card_probabilities))
        probs = tuple(float(item) for item in self.elixir_probabilities)
        if probs and (
            len(probs) != 11 or any(item < 0 for item in probs) or not math.isclose(sum(probs), 1.0, abs_tol=1e-5)
        ):
            raise ContractError("elixir_probabilities must be an empty vector or 11 probabilities summing to one")
        object.__setattr__(self, "elixir_probabilities", probs)
        object.__setattr__(self, "evolution_hypotheses", frozen_mapping(self.evolution_hypotheses))
        object.__setattr__(self, "ability_hypotheses", frozen_mapping(self.ability_hypotheses))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OpponentBeliefV1":
        return cls(
            opponent_owner=int(value["opponent_owner"]),
            deck_probabilities=value.get("deck_probabilities", {}),
            hand_hypotheses=tuple(value.get("hand_hypotheses", ())),
            next_card_probabilities=value.get("next_card_probabilities", {}),
            elixir_probabilities=tuple(value.get("elixir_probabilities", ())),
            evolution_hypotheses=value.get("evolution_hypotheses", {}),
            ability_hypotheses=value.get("ability_hypotheses", {}),
            updated_tick=int(value.get("updated_tick", 0)),
        )


@dataclass(frozen=True, slots=True)
class ActionMaskV1(ContractMixin):
    kinds: Mapping[str, bool] = field(default_factory=lambda: FrozenMapping({ActionKind.WAIT.value: True}))
    hand_slots: tuple[bool, bool, bool, bool] = (False, False, False, False)
    placement_masks: Mapping[str, Any] = field(default_factory=FrozenMapping)
    ability_sources: tuple[int, ...] = ()
    target_entities: tuple[int, ...] = ()
    reasons: Mapping[str, Any] = field(default_factory=FrozenMapping)
    VERSION: ClassVar[str] = "action-mask.v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "kinds", frozen_mapping(self.kinds))
        hand = tuple(bool(item) for item in self.hand_slots)
        if len(hand) != 4:
            raise ContractError("hand_slots mask must have exactly four entries")
        object.__setattr__(self, "hand_slots", hand)
        object.__setattr__(self, "placement_masks", frozen_mapping(self.placement_masks))
        object.__setattr__(self, "ability_sources", _int_tuple(self.ability_sources))
        object.__setattr__(self, "target_entities", _int_tuple(self.target_entities))
        object.__setattr__(self, "reasons", frozen_mapping(self.reasons))
        if not bool(self.kinds.get(ActionKind.WAIT.value, False)):
            raise ContractError("WAIT must always remain legal")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "ActionMaskV1":
        data = value or {}
        return cls(
            kinds=data.get("kinds", {ActionKind.WAIT.value: True}),
            hand_slots=tuple(data.get("hand_slots", (False, False, False, False))),
            placement_masks=data.get("placement_masks", {}),
            ability_sources=tuple(data.get("ability_sources", ())),
            target_entities=tuple(data.get("target_entities", ())),
            reasons=data.get("reasons", {}),
        )


@dataclass(frozen=True, slots=True)
class RasterV2(ContractMixin):
    """Owner-relative 32x18 semantic plane with action-legality channels.

    RasterV2 is built only from the already filtered entities/towers supplied
    to an actor observation.  It never reads the raw native object graph.  The
    five legality planes are derived from the same ``ActionMaskV1`` returned in
    that observation and are bound by ``source_action_mask_hash``.
    """

    owner: int
    channels: Mapping[str, Any]
    channel_availability: Mapping[str, bool]
    source_action_mask_version: str
    source_action_mask_hash: str
    shape: tuple[int, int] = (BOARD_HEIGHT, BOARD_WIDTH)
    channel_order: tuple[str, ...] = RASTER_V2_CHANNELS
    spatial_frame: str = "native-arena-row-major"
    perspective_transform_hash: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=FrozenMapping)
    version: str = field(default=RASTER_VERSION, init=False)
    VERSION: ClassVar[str] = RASTER_VERSION

    def __post_init__(self) -> None:
        if self.owner not in (0, 1):
            raise ContractError("RasterV2 owner must be 0 or 1")
        shape = tuple(int(item) for item in self.shape)
        if shape != (BOARD_HEIGHT, BOARD_WIDTH):
            raise ContractError(f"RasterV2 shape must be {(BOARD_HEIGHT, BOARD_WIDTH)}")
        order = tuple(str(item) for item in self.channel_order)
        if order != RASTER_V2_CHANNELS:
            raise ContractError("RasterV2 channel_order must match the V2 schema")
        if set(self.channels) != set(order):
            missing = sorted(set(order) - set(self.channels))
            extra = sorted(set(self.channels) - set(order))
            raise ContractError(f"RasterV2 channels mismatch; missing={missing}, extra={extra}")
        normalized: dict[str, tuple[tuple[float, ...], ...]] = {}
        for name in order:
            raw_rows = self.channels[name]
            if (
                not isinstance(raw_rows, Sequence)
                or isinstance(raw_rows, (str, bytes, bytearray))
                or len(raw_rows) != BOARD_HEIGHT
            ):
                raise ContractError(f"RasterV2 channel {name!r} must have {BOARD_HEIGHT} rows")
            rows: list[tuple[float, ...]] = []
            for raw_row in raw_rows:
                if (
                    not isinstance(raw_row, Sequence)
                    or isinstance(raw_row, (str, bytes, bytearray))
                    or len(raw_row) != BOARD_WIDTH
                ):
                    raise ContractError(f"RasterV2 channel {name!r} rows must have {BOARD_WIDTH} values")
                row = tuple(float(value) for value in raw_row)
                if any(not math.isfinite(value) or value < 0.0 for value in row):
                    raise ContractError(f"RasterV2 channel {name!r} values must be finite and non-negative")
                if name in RASTER_V2_BINARY_CHANNELS and any(value not in (0.0, 1.0) for value in row):
                    raise ContractError(f"RasterV2 mask channel {name!r} must be binary")
                rows.append(row)
            normalized[name] = tuple(rows)
        availability = {str(name): bool(value) for name, value in self.channel_availability.items()}
        if set(availability) != set(order):
            raise ContractError("RasterV2 channel_availability must cover every channel exactly")
        if self.source_action_mask_version != ActionMaskV1.VERSION:
            raise ContractError(f"RasterV2 requires {ActionMaskV1.VERSION}, got {self.source_action_mask_version!r}")
        for label, digest in (
            ("source_action_mask_hash", self.source_action_mask_hash),
            ("perspective_transform_hash", self.perspective_transform_hash),
        ):
            if digest is not None and (
                len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ContractError(f"RasterV2 {label} must be a lowercase SHA-256")
        if self.spatial_frame not in {"native-arena-row-major", "owner-model-row-major"}:
            raise ContractError("RasterV2 has an unsupported spatial_frame")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "channel_order", order)
        object.__setattr__(self, "channels", frozen_mapping(normalized))
        object.__setattr__(self, "channel_availability", frozen_mapping(availability))
        object.__setattr__(self, "metadata", frozen_mapping(self.metadata))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RasterV2":
        _version(value, RASTER_VERSION)
        allowed = {
            "version",
            "owner",
            "channels",
            "channel_availability",
            "source_action_mask_version",
            "source_action_mask_hash",
            "shape",
            "channel_order",
            "spatial_frame",
            "perspective_transform_hash",
            "metadata",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ContractError(f"unknown RasterV2 fields: {', '.join(unknown)}")
        return cls(
            owner=int(value["owner"]),
            channels=value["channels"],
            channel_availability=value["channel_availability"],
            source_action_mask_version=str(value["source_action_mask_version"]),
            source_action_mask_hash=str(value["source_action_mask_hash"]),
            shape=tuple(value.get("shape", ())),
            channel_order=tuple(value.get("channel_order", ())),
            spatial_frame=str(value.get("spatial_frame", "")),
            perspective_transform_hash=value.get("perspective_transform_hash"),
            metadata=value.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class PendingActionV1(ContractMixin):
    action: ActionV1
    requested_tick: int
    expected_execution_tick: int
    generated_at_ms: int | None = None
    requested_at_ms: int | None = None
    inference_end_ms: int | None = None
    server_apply_ms: int | None = None
    status: str = "queued"
    VERSION: ClassVar[str] = "pending-action.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.action, ActionV1):
            object.__setattr__(self, "action", ActionV1.from_mapping(self.action))
        if self.requested_tick < 0 or self.expected_execution_tick < self.requested_tick:
            raise ContractError("pending action execution precedes request")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PendingActionV1":
        return cls(
            action=ActionV1.from_mapping(value["action"]),
            requested_tick=int(value["requested_tick"]),
            expected_execution_tick=int(value["expected_execution_tick"]),
            generated_at_ms=value.get("generated_at_ms"),
            requested_at_ms=value.get("requested_at_ms"),
            inference_end_ms=value.get("inference_end_ms"),
            server_apply_ms=value.get("server_apply_ms"),
            status=str(value.get("status", "queued")),
        )


@dataclass(frozen=True, slots=True)
class TerminalV1(ContractMixin):
    ended: bool = False
    winner: int | None = None
    result_by_owner: tuple[float, float] = (0.0, 0.0)
    reason: str | None = None
    terminal_tick: int | None = None
    VERSION: ClassVar[str] = "terminal.v1"

    def __post_init__(self) -> None:
        if self.winner is not None and self.winner not in (0, 1):
            raise ContractError("winner must be 0, 1, or null")
        results = tuple(float(item) for item in self.result_by_owner)
        if len(results) != 2 or any(item not in (-1.0, 0.0, 1.0) for item in results):
            raise ContractError("result_by_owner must contain two values from {-1,0,1}")
        if not math.isclose(results[0], -results[1], abs_tol=1e-9):
            raise ContractError("terminal result must be zero-sum")
        object.__setattr__(self, "result_by_owner", results)
        if not self.ended and (self.winner is not None or results != (0.0, 0.0) or self.terminal_tick is not None):
            raise ContractError("non-terminal state cannot carry a result/winner/terminal_tick")
        if self.ended and self.terminal_tick is None:
            raise ContractError("terminal state requires terminal_tick")
        if self.winner is not None and results[self.winner] != 1.0:
            raise ContractError("winner must have result +1")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | bool | None, *, tick: int = 0) -> "TerminalV1":
        if value is None or value is False:
            return cls()
        if value is True:
            return cls(ended=True, terminal_tick=tick)
        ended = bool(value.get("ended", value.get("done", False)))
        winner = value.get("winner")
        results = value.get("result_by_owner")
        if results is None and ended and winner in (0, 1):
            results = (1.0, -1.0) if winner == 0 else (-1.0, 1.0)
        return cls(
            ended=ended,
            winner=winner,
            result_by_owner=tuple(results or (0.0, 0.0)),
            reason=value.get("reason"),
            terminal_tick=value.get("terminal_tick", tick if ended else None),
        )


@dataclass(frozen=True, slots=True)
class ObservationV1(ContractMixin):
    tier: ObservationTier
    tick: int
    owner: int | None = None
    time: TimeStateV1 = field(default_factory=TimeStateV1)
    phase: str = "unknown"
    players: tuple[PlayerStateV1, ...] = ()
    towers: tuple[TowerStateV1, ...] = ()
    entities: tuple[EntityStateV1, ...] = ()
    events: tuple[EventV1, ...] = ()
    opponent_belief: OpponentBeliefV1 | None = None
    action_mask: ActionMaskV1 = field(default_factory=ActionMaskV1)
    pending_actions: tuple[PendingActionV1, ...] = ()
    terminal: TerminalV1 = field(default_factory=TerminalV1)
    native_digest: str | None = None
    ruleset_id: str | None = None
    episode_id: str | None = None
    raster: RasterV2 | None = None
    truncated: bool = False
    metadata: Mapping[str, Any] = field(default_factory=FrozenMapping)
    version: str = field(default=OBSERVATION_VERSION, init=False)
    VERSION: ClassVar[str] = OBSERVATION_VERSION

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ObservationV1":
        _version(value, OBSERVATION_VERSION)
        data = dict(value)
        data.pop("version", None)
        return cls(**data)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tier", _enum(ObservationTier, self.tier, "observation tier"))
        if self.tick < 0:
            raise ContractError("observation tick must be non-negative")
        if self.owner is not None and self.owner not in (0, 1):
            raise ContractError("observation owner must be 0, 1, or null")
        if self.tier != ObservationTier.ORACLE and self.owner is None:
            raise ContractError("fair/human/RGB observations require an actor owner")
        if not isinstance(self.time, TimeStateV1):
            object.__setattr__(self, "time", TimeStateV1.from_mapping(self.time, tick=self.tick))
        for label, contract_type in (
            ("players", PlayerStateV1),
            ("towers", TowerStateV1),
            ("entities", EntityStateV1),
            ("events", EventV1),
            ("pending_actions", PendingActionV1),
        ):
            factory = contract_type.from_mapping
            normalized = tuple(
                item if isinstance(item, contract_type) else factory(item) for item in getattr(self, label)
            )
            object.__setattr__(self, label, normalized)
        if len({player.owner for player in self.players}) != len(self.players):
            raise ContractError("observation contains duplicate player owners")
        _set_contract(self, "opponent_belief", OpponentBeliefV1)
        if not isinstance(self.action_mask, ActionMaskV1):
            object.__setattr__(self, "action_mask", ActionMaskV1.from_mapping(self.action_mask))
        if not isinstance(self.terminal, TerminalV1):
            object.__setattr__(self, "terminal", TerminalV1.from_mapping(self.terminal, tick=self.tick))
        _set_contract(self, "raster", RasterV2)
        object.__setattr__(self, "metadata", frozen_mapping(self.metadata))
        if self.tier != ObservationTier.ORACLE:
            if self.native_digest is not None:
                raise ContractError("native_digest is privileged and forbidden in fair/human/RGB observations")
            for player in self.players:
                if player.owner != self.owner and player.private_state_visible:
                    raise ContractError("opponent private state leaked into actor observation")
            if any(entity.internal for entity in self.entities) or any(tower.internal for tower in self.towers):
                raise ContractError("internal engine fields leaked into actor observation")
            if any(pending.action.owner != self.owner for pending in self.pending_actions):
                raise ContractError("opponent pending command leaked into actor observation")


def _range_pair(value: Sequence[int] | int, label: str) -> tuple[int, int]:
    if isinstance(value, int):
        result = (value, value)
    else:
        if len(value) != 2:
            raise ContractError(f"{label} must be an integer or [min,max]")
        result = (int(value[0]), int(value[1]))
    if result[0] < 0 or result[1] < result[0]:
        raise ContractError(f"invalid {label} range")
    return result


@dataclass(frozen=True, slots=True)
class LatencyConfigV1(ContractMixin):
    observation_delay_ms: tuple[int, int] = (0, 0)
    inference_delay_ms: tuple[int, int] = (0, 0)
    network_jitter_ms: tuple[int, int] = (0, 0)
    execution_delay_ms: tuple[int, int] = (0, 0)
    frame_drop_probability: float = 0.0
    command_drop_probability: float = 0.0
    timestamp_quantization_ms: int = 1
    min_reaction_ms: int = 0
    min_touch_interval_ms: int = 0
    command_queue_limit: int = 32
    delay_bins_ms: tuple[int, ...] = (100, 150, 200, 300, 450, 700, 1000, 1500)
    seed_offset: int = 0
    version: str = field(default=LATENCY_CONFIG_VERSION, init=False)
    VERSION: ClassVar[str] = LATENCY_CONFIG_VERSION

    def __post_init__(self) -> None:
        for label in ("observation_delay_ms", "inference_delay_ms", "network_jitter_ms", "execution_delay_ms"):
            object.__setattr__(self, label, _range_pair(getattr(self, label), label))
        for label, probability in (
            ("frame_drop_probability", self.frame_drop_probability),
            ("command_drop_probability", self.command_drop_probability),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ContractError(f"{label} must be in [0,1]")
        for label, number in (
            ("timestamp_quantization_ms", self.timestamp_quantization_ms),
            ("min_reaction_ms", self.min_reaction_ms),
            ("min_touch_interval_ms", self.min_touch_interval_ms),
            ("command_queue_limit", self.command_queue_limit),
        ):
            if number < 0:
                raise ContractError(f"{label} must be non-negative")
        if self.timestamp_quantization_ms == 0 or self.command_queue_limit == 0:
            raise ContractError("timestamp_quantization_ms and command_queue_limit must be positive")
        bins = tuple(int(item) for item in self.delay_bins_ms)
        if any(item <= 0 for item in bins) or tuple(sorted(set(bins))) != bins:
            raise ContractError("delay_bins_ms must be unique, positive, and sorted")
        object.__setattr__(self, "delay_bins_ms", bins)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "LatencyConfigV1":
        if value is None:
            return cls()
        data = dict(value or {})
        _version(data, LATENCY_CONFIG_VERSION)
        data.pop("version", None)
        return cls(**data)


@dataclass(frozen=True, slots=True)
class EnvironmentConfigV1(ContractMixin):
    """All Python-side knobs that change the MDP, reward, or observation."""

    warmup_ticks: int = 130
    max_battle_ticks: int = NATIVE_MATCH_END_TICK
    entity_limit: int = 256
    event_limit: int = 256
    event_window_ticks: int = 900
    shaping_beta: float = 0.0
    shaping_tau_ticks: float = 1200.0
    include_native_digest: bool = True
    version: str = field(default=ENVIRONMENT_CONFIG_VERSION, init=False)
    VERSION: ClassVar[str] = ENVIRONMENT_CONFIG_VERSION

    def __post_init__(self) -> None:
        if self.warmup_ticks < 0 or self.max_battle_ticks < 1:
            raise ContractError("warmup_ticks/max_battle_ticks are invalid")
        if not 1 <= self.entity_limit <= 256:
            raise ContractError("entity_limit must be in [1,256]")
        if self.event_limit < 1 or self.event_window_ticks < 1:
            raise ContractError("event limits must be positive")
        if self.shaping_beta < 0 or self.shaping_tau_ticks <= 0:
            raise ContractError("reward shaping parameters are invalid")
        if not isinstance(self.include_native_digest, bool):
            raise ContractError("include_native_digest must be boolean")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "EnvironmentConfigV1":
        if value is None:
            return cls()
        data = dict(value or {})
        _version(data, ENVIRONMENT_CONFIG_VERSION)
        data.pop("version", None)
        return cls(**data)


@dataclass(frozen=True, slots=True)
class EpisodeConfigV1(ContractMixin):
    ruleset_id: str
    deck0: tuple[int, ...]
    deck1: tuple[int, ...]
    seed: int
    observation_tier: ObservationTier = ObservationTier.FAIR
    game_mode: int = 72000006
    arena: int = 54000001
    map_id: str = "standard-1v1"
    tick_ms: int = DEFAULT_TICK_MS
    decision_hz: float = 10.0
    event_driven_decisions: bool = True
    latency: LatencyConfigV1 = field(default_factory=LatencyConfigV1)
    environment: EnvironmentConfigV1 = field(default_factory=EnvironmentConfigV1)
    policy_hashes: Mapping[str, str] = field(default_factory=FrozenMapping)
    render_mode: str = "headless"
    deterministic: bool = True
    tags: Mapping[str, Any] = field(default_factory=FrozenMapping)
    version: str = field(default=EPISODE_CONFIG_VERSION, init=False)
    VERSION: ClassVar[str] = EPISODE_CONFIG_VERSION

    def __post_init__(self) -> None:
        if not self.ruleset_id:
            raise ContractError("ruleset_id must not be empty")
        object.__setattr__(self, "observation_tier", _enum(ObservationTier, self.observation_tier, "observation tier"))
        for label in ("deck0", "deck1"):
            deck = _int_tuple(getattr(self, label))
            if len(deck) != 8 or any(card <= 0 for card in deck):
                raise ContractError(f"{label} must contain exactly eight positive card IDs")
            if len(set(deck)) != len(deck):
                raise ContractError(f"{label} must not contain duplicate cards")
            object.__setattr__(self, label, deck)
        if not -(1 << 31) <= int(self.seed) < (1 << 32):
            raise ContractError("seed must fit the native 32-bit field")
        if self.game_mode <= 0 or self.arena <= 0 or self.tick_ms <= 0 or self.decision_hz <= 0:
            raise ContractError("game_mode/arena/tick_ms/decision_hz must be positive")
        if not isinstance(self.latency, LatencyConfigV1):
            object.__setattr__(self, "latency", LatencyConfigV1.from_mapping(self.latency))
        if not isinstance(self.environment, EnvironmentConfigV1):
            object.__setattr__(self, "environment", EnvironmentConfigV1.from_mapping(self.environment))
        object.__setattr__(self, "policy_hashes", frozen_mapping(self.policy_hashes))
        object.__setattr__(self, "tags", frozen_mapping(self.tags))
        if self.render_mode not in {"headless", "native-render"}:
            raise ContractError("render_mode must be headless or native-render")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EpisodeConfigV1":
        _version(value, EPISODE_CONFIG_VERSION)
        data = dict(value)
        data.pop("version", None)
        data["latency"] = LatencyConfigV1.from_mapping(data.get("latency"))
        data["environment"] = EnvironmentConfigV1.from_mapping(data.get("environment"))
        return cls(**data)


def contract_schema_descriptor() -> dict[str, Any]:
    """Return a stable field/type descriptor for manifest compatibility gates."""

    public = (
        ActionV1,
        ObservationV1,
        CardSpecV1,
        AbilitySpecV1,
        LatencyConfigV1,
        EnvironmentConfigV1,
        EpisodeConfigV1,
        RunnerAttestationV1,
        PlayerStateV1,
        TowerStateV1,
        EntityStateV1,
        EventV1,
        PendingActionV1,
        TerminalV1,
        ActionMaskV1,
        RasterV2,
        OpponentBeliefV1,
        TimeStateV1,
        TargetSchemaV1,
        MechanicOpV1,
        EvolutionSpecV1,
        SemanticProvenanceV1,
        ShieldStateV1,
        EffectStateV1,
        AttackStateV1,
        ProjectileStateV1,
        EntityResourceStateV1,
        CaptureTargetStateV1,
        PeriodicAttackModifierStateV1,
        CaptureRuntimeStateV1,
        VisibilityStateV1,
        ThresholdRelocationStateV1,
        AbilityRuntimeStateV1,
        EvolutionRuntimeStateV1,
        CombatEventV1,
        CausalGroupRefV1,
        DaggerDuchessRuntimeStateV1,
        RoyalChefRuntimeStateV1,
    )
    return {
        "versions": {
            "observation": OBSERVATION_VERSION,
            "action": ACTION_VERSION,
            "card_spec": CARD_SPEC_VERSION,
            "ability_spec": ABILITY_SPEC_VERSION,
            "latency": LATENCY_CONFIG_VERSION,
            "environment": ENVIRONMENT_CONFIG_VERSION,
            "episode": EPISODE_CONFIG_VERSION,
            "runner_attestation": RUNNER_ATTESTATION_VERSION,
            "opponent_belief": OpponentBeliefV1.VERSION,
            "target_schema": TargetSchemaV1.VERSION,
            "mechanic_op": MechanicOpV1.VERSION,
            "evolution_spec": EvolutionSpecV1.VERSION,
            "semantic_provenance": SEMANTIC_PROVENANCE_VERSION,
            "shield_state": SHIELD_STATE_VERSION,
            "effect_state": EFFECT_STATE_VERSION,
            "attack_state": ATTACK_STATE_VERSION,
            "projectile_state": PROJECTILE_STATE_VERSION,
            "entity_resource_state": ENTITY_RESOURCE_STATE_VERSION,
            "periodic_attack_modifier_state": (PERIODIC_ATTACK_MODIFIER_STATE_VERSION),
            "capture_target_state": CAPTURE_TARGET_STATE_VERSION,
            "capture_runtime_state": CAPTURE_RUNTIME_STATE_VERSION,
            "threshold_relocation_state": THRESHOLD_RELOCATION_STATE_VERSION,
            "visibility_state": VISIBILITY_STATE_VERSION,
            "ability_runtime_state": ABILITY_RUNTIME_STATE_VERSION,
            "evolution_runtime_state": EVOLUTION_RUNTIME_STATE_VERSION,
            "combat_event": COMBAT_EVENT_VERSION,
            "causal_group_ref": CAUSAL_GROUP_REF_VERSION,
            "dagger_duchess_runtime": DAGGER_DUCHESS_RUNTIME_VERSION,
            "royal_chef_runtime": ROYAL_CHEF_RUNTIME_VERSION,
            "raster": RASTER_VERSION,
        },
        "contracts": {contract.__name__: [item.name for item in fields(contract)] for contract in public},
        "board": {"width": BOARD_WIDTH, "height": BOARD_HEIGHT, "default_tick_ms": DEFAULT_TICK_MS},
    }
