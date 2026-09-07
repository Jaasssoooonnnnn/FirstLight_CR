"""Stable competitive V4 vocabularies and exact runtime identity lookup."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping, Sequence

import torch
from torch import Tensor

from ...card_logic import StaticCardLogicCatalogV1, build_static_card_logic_catalog
from ...card_specs import CardSpecCatalog
from ...contracts import AbilitySpecV1, CardSpecV1, content_hash
from ..card_features import (
    CARD_FEATURE_SCHEMA_VERSION,
    CARD_MECHANIC_FEATURE_NAMES,
    CARD_STATIC_FEATURE_NAMES,
    CardFeatureRow,
    _compiled_fields,
    _require_compiled_abilities,
    _require_compiled_source,
    _require_compiled_specs,
    build_card_feature_rows,
    pack_mechanic_features,
    pack_static_features,
)
from .config import ABILITY_FEATURE_NAMES, UNIVERSAL_CARD_CATALOG_VERSION, ModelConfigV4

if TYPE_CHECKING:
    from ...card_logic import StaticCardLogicCatalogV1
    from ...projectile_catalog import NativeProjectileCatalogV1


PAD_CARD_VOCAB_ID = 0


UNKNOWN_CARD_VOCAB_ID = 1


FIRST_REAL_CARD_VOCAB_ID = 2


PAD_ABILITY_VOCAB_ID = 0


PAD_ENTITY_ARCHETYPE_VOCAB_ID = 0


UNKNOWN_ENTITY_ARCHETYPE_VOCAB_ID = 1


FIRST_REAL_ENTITY_ARCHETYPE_VOCAB_ID = 2


NORMAL_MODE_SOURCE_CARD_COUNT = 152


NORMAL_MODE_POLICY_CARD_COUNT = 122


ENTITY_ARCHETYPE_STATIC_DAMAGE_LEVEL = 11


@dataclass(frozen=True, slots=True)
class EntityArchetypeMetadataV1:
    """Minimal exact static semantics for one EntityArchetype row.

    ``static_damage_basis`` is an unnormalized tournament-level-11 damage
    value, projected with the exact carrier rarity curve in ``static_logic``.
    A false ``*_known`` flag means that the exact static record does not
    establish that value; no parent-card fallback is used.
    """

    archetype_key: str
    child_kind: str = "unknown"
    child_kind_known: bool = False
    is_airborne: bool = False
    is_airborne_known: bool = False
    static_damage_basis: float = 0.0
    static_damage_basis_known: bool = False
    projectile_radius_tiles: float = 0.0
    projectile_radius_known: bool = False
    lifetime_ms: float = 0.0
    lifetime_known: bool = False


def _unknown_archetype_metadata(key: str) -> EntityArchetypeMetadataV1:
    return EntityArchetypeMetadataV1(archetype_key=str(key))


def normal_mode_card_specs(
    card_specs: Mapping[int, CardSpecV1], *, require_frozen_baseline: bool = False
) -> dict[int, CardSpecV1]:
    """Return only cards admitted to ordinary competitive matches.

    The native catalog intentionally retains hidden/event definitions.  V4 is
    a frozen competitive policy baseline, so its production builder also
    checks the reviewed 152 -> 122 partition instead of silently changing the
    vocabulary when the native source drifts.
    """

    scoped = {
        int(card_id): spec
        for card_id, spec in card_specs.items()
        if spec.attributes.get("NotVisible") is not True and spec.attributes.get("NotInUse") is not True
    }
    if require_frozen_baseline:
        from ...semantic_subset import NORMAL_MODE_POLICY_CARD_IDS, NORMAL_MODE_POLICY_EXCLUDED_CARD_IDS

        source_ids = frozenset(int(card_id) for card_id in card_specs)
        if (
            len(card_specs) != NORMAL_MODE_SOURCE_CARD_COUNT
            or len(scoped) != NORMAL_MODE_POLICY_CARD_COUNT
            or frozenset(scoped) != NORMAL_MODE_POLICY_CARD_IDS
            or source_ids != NORMAL_MODE_POLICY_CARD_IDS | NORMAL_MODE_POLICY_EXCLUDED_CARD_IDS
        ):
            raise ValueError(
                "V4 normal-mode card partition drifted: "
                f"source={len(card_specs)}, eligible={len(scoped)}, "
                f"expected={NORMAL_MODE_SOURCE_CARD_COUNT}/"
                f"{NORMAL_MODE_POLICY_CARD_COUNT}"
            )
    return scoped


@dataclass(frozen=True, slots=True)
class AbilityCatalogV1:
    """Full competitive Ability ordering plus its exact static feature rows."""

    card_scope: tuple[int, ...]
    ability_ids: tuple[str, ...]
    static_features: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        card_scope = tuple(int(value) for value in self.card_scope)
        if tuple(sorted(set(card_scope))) != card_scope:
            raise ValueError("AbilityCatalog card scope must be unique and sorted")
        normalized = tuple(str(value) for value in self.ability_ids)
        if any(not value for value in normalized):
            raise ValueError("ability IDs must not be empty")
        if tuple(sorted(set(normalized))) != normalized:
            raise ValueError("ability IDs must be unique and sorted")
        if len(self.static_features) != len(normalized):
            raise ValueError("AbilityCatalog feature rows do not match its ordering")
        if any(len(row) != len(ABILITY_FEATURE_NAMES) for row in self.static_features):
            raise ValueError("AbilityCatalog static rows do not match their named schema")
        if any(not math.isfinite(value) for row in self.static_features for value in row):
            raise ValueError("AbilityCatalog static rows must be finite")
        object.__setattr__(self, "card_scope", card_scope)
        object.__setattr__(self, "ability_ids", normalized)
        object.__setattr__(
            self, "static_features", tuple(tuple(float(value) for value in row) for row in self.static_features)
        )

    @classmethod
    def from_specs(
        cls, card_specs: Mapping[int, CardSpecV1], ability_specs: Sequence[AbilitySpecV1]
    ) -> "AbilityCatalogV1":
        _require_compiled_specs(card_specs)
        _require_compiled_abilities(card_specs, ability_specs)
        data = _compiled_fields("ability_catalog")
        rows = dict(zip(data["ability_ids"], data["static_features"], strict=True))
        identities = tuple(sorted({name for card in card_specs.values() for name in card.ability_ids}))
        return cls(tuple(sorted(card_specs)), identities, tuple(rows[name] for name in identities))

    @classmethod
    def from_card_spec_catalog(cls, card_catalog: CardSpecCatalog) -> "AbilityCatalogV1":
        return cls.from_specs(
            normal_mode_card_specs(card_catalog.by_id, require_frozen_baseline=True), card_catalog.abilities
        )

    @classmethod
    def empty(cls, card_scope: Sequence[int]) -> "AbilityCatalogV1":
        """Build an explicit no-Ability catalog for synthetic card fixtures."""

        return cls(tuple(sorted(int(value) for value in card_scope)), (), ())

    @property
    def vocab_size(self) -> int:
        return 1 + len(self.ability_ids)

    @property
    def catalog_id(self) -> str:
        return content_hash(
            {
                "feature_names": ABILITY_FEATURE_NAMES,
                "card_scope": self.card_scope,
                "ability_ids": self.ability_ids,
                "static_features": self.static_features,
            }
        )

    def vocab_id(self, ability_id: str) -> int:
        try:
            return 1 + self.ability_ids.index(str(ability_id))
        except ValueError as error:
            raise ValueError(f"unknown ability ID {ability_id!r}") from error

    def ability_id(self, vocab_id: int) -> str | None:
        if int(vocab_id) == PAD_ABILITY_VOCAB_ID:
            return None
        index = int(vocab_id) - 1
        if not 0 <= index < len(self.ability_ids):
            raise ValueError("ability vocab ID is outside this catalog")
        return self.ability_ids[index]

    def feature_tensor(
        self, config: ModelConfigV4, *, dtype: torch.dtype = torch.float32, device: torch.device | str | None = None
    ) -> Tensor:
        if self.vocab_size > config.ability_vocab_size:
            raise ValueError("AbilityCatalog exceeds ModelConfigV4 capacity")
        rows = (
            ((0.0,) * config.ability_feature_dim,)
            + self.static_features
            + ((0.0,) * config.ability_feature_dim,) * (config.ability_vocab_size - self.vocab_size)
        )
        return torch.tensor(rows, dtype=dtype, device=device)


@dataclass(frozen=True, slots=True)
class EntityArchetypeCatalogV1:
    """Canonical semantics plus exact live LogicData runtime aliases."""

    card_scope: tuple[int, ...]
    archetype_keys: tuple[str, ...]
    character_global_ids: tuple[tuple[int, str], ...] = ()
    archetype_metadata: tuple[EntityArchetypeMetadataV1, ...] = ()
    archetype_node_ids: tuple[str, ...] = ()
    archetype_key_aliases: tuple[tuple[str, str], ...] = ()
    archetype_node_aliases: tuple[tuple[str, str], ...] = ()
    _vocab_by_key: Mapping[str, int] = field(init=False, repr=False, compare=False)
    _vocab_by_node_id: Mapping[str, int] = field(init=False, repr=False, compare=False)
    _character_vocab_by_global_id: Mapping[int, int] = field(init=False, repr=False, compare=False)
    _runtime_vocab_by_global_id: Mapping[int, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        card_scope = tuple(int(value) for value in self.card_scope)
        keys = tuple(str(value) for value in self.archetype_keys)
        character_global_ids = tuple((int(global_id), str(form_id)) for global_id, form_id in self.character_global_ids)
        metadata = tuple(self.archetype_metadata) or tuple(_unknown_archetype_metadata(key) for key in keys)
        node_ids = tuple(str(value) for value in self.archetype_node_ids) or tuple("" for _ in keys)
        key_aliases = tuple((str(alias), str(canonical)) for alias, canonical in self.archetype_key_aliases)
        node_aliases = tuple((str(alias), str(canonical)) for alias, canonical in self.archetype_node_aliases)
        if tuple(sorted(set(card_scope))) != card_scope:
            raise ValueError("archetype card scope must be unique and sorted")
        if tuple(sorted(set(keys))) != keys or any(not value for value in keys):
            raise ValueError("archetype keys must be non-empty, unique, and sorted")
        if tuple(sorted(character_global_ids)) != character_global_ids:
            raise ValueError("Character global IDs must be sorted")
        if len({value[0] for value in character_global_ids}) != len(character_global_ids) or len(
            {value[1] for value in character_global_ids}
        ) != len(character_global_ids):
            raise ValueError("Character global IDs and form identities must be unique")
        addressable_keys = set(keys) | {alias for alias, _canonical in key_aliases}
        if any(
            global_id <= 0 or global_id > 0xFFFFFFFF or not form_id or f"form:{form_id}" not in addressable_keys
            for global_id, form_id in character_global_ids
        ):
            raise ValueError("Character global-ID mapping is outside the catalog")
        if tuple(row.archetype_key for row in metadata) != keys:
            raise ValueError("archetype metadata must align with archetype keys")
        if len(node_ids) != len(keys):
            raise ValueError("archetype node identities must align with keys")
        materialized_nodes = tuple(value for value in node_ids if value)
        if len(set(materialized_nodes)) != len(materialized_nodes):
            raise ValueError("materialized archetype node identities must be unique")
        for label, aliases, occupied in (
            ("archetype key", key_aliases, set(keys)),
            ("archetype node", node_aliases, set(materialized_nodes)),
        ):
            if tuple(sorted(aliases)) != aliases:
                raise ValueError(f"{label} aliases must be sorted")
            alias_names = tuple(alias for alias, _canonical in aliases)
            if (
                any(not alias or not canonical for alias, canonical in aliases)
                or len(set(alias_names)) != len(alias_names)
                or set(alias_names).intersection(occupied)
                or any(canonical not in keys for _alias, canonical in aliases)
            ):
                raise ValueError(f"{label} aliases must be unique and reference canonical keys")
        if any(
            not math.isfinite(value)
            for row in metadata
            for value in (row.static_damage_basis, row.projectile_radius_tiles, row.lifetime_ms)
        ):
            raise ValueError("archetype metadata numeric values must be finite")
        object.__setattr__(self, "card_scope", card_scope)
        object.__setattr__(self, "archetype_keys", keys)
        object.__setattr__(self, "character_global_ids", character_global_ids)
        object.__setattr__(self, "archetype_metadata", metadata)
        object.__setattr__(self, "archetype_node_ids", node_ids)
        object.__setattr__(self, "archetype_key_aliases", key_aliases)
        object.__setattr__(self, "archetype_node_aliases", node_aliases)
        vocab_by_key = {key: FIRST_REAL_ENTITY_ARCHETYPE_VOCAB_ID + index for index, key in enumerate(keys)}
        vocab_by_key.update((alias, vocab_by_key[canonical]) for alias, canonical in key_aliases)
        vocab_by_node_id = {
            node_id: FIRST_REAL_ENTITY_ARCHETYPE_VOCAB_ID + index for index, node_id in enumerate(node_ids) if node_id
        }
        vocab_by_node_id.update((alias, vocab_by_key[canonical]) for alias, canonical in node_aliases)
        character_vocab_by_global_id = {
            global_id: vocab_by_key[f"form:{form_id}"] for global_id, form_id in character_global_ids
        }
        runtime_vocab_by_global_id = dict(character_vocab_by_global_id)
        for key, vocab_id in vocab_by_key.items():
            namespace, raw_identity = key.split(":", 1)
            if namespace not in {"projectile", "area"}:
                continue
            global_id = int(raw_identity)
            existing = runtime_vocab_by_global_id.get(global_id)
            if existing is not None and existing != vocab_id:
                raise ValueError("one runtime LogicData global ID maps to multiple EntityArchetypes")
            runtime_vocab_by_global_id[global_id] = vocab_id
        object.__setattr__(self, "_vocab_by_key", MappingProxyType(vocab_by_key))
        object.__setattr__(self, "_vocab_by_node_id", MappingProxyType(vocab_by_node_id))
        object.__setattr__(self, "_character_vocab_by_global_id", MappingProxyType(character_vocab_by_global_id))
        object.__setattr__(self, "_runtime_vocab_by_global_id", MappingProxyType(runtime_vocab_by_global_id))

    @classmethod
    def from_card_specs(
        cls,
        card_specs: Mapping[int, CardSpecV1],
        *,
        static_logic: "StaticCardLogicCatalogV1 | None" = None,
        projectile_catalog: "NativeProjectileCatalogV1 | None" = None,
        character_global_ids: Mapping[str, int] | None = None,
    ) -> "EntityArchetypeCatalogV1":
        _require_compiled_specs(card_specs, complete=True)
        _require_compiled_source("card_logic", static_logic)
        _require_compiled_source("projectiles", projectile_catalog)
        data = _compiled_fields("entity_archetype_catalog")
        if character_global_ids is not None:
            expected = {name: value for value, name in data["character_global_ids"]}
            if any(character_global_ids.get(name) != value for name, value in expected.items()):
                raise ValueError("Character global IDs differ from the compiled competitive release")
        data["archetype_metadata"] = tuple(EntityArchetypeMetadataV1(**row) for row in data["archetype_metadata"])
        return cls(**data)

    @classmethod
    def from_card_spec_catalog(
        cls,
        card_catalog: CardSpecCatalog,
        *,
        static_logic: "StaticCardLogicCatalogV1",
        projectile_catalog: "NativeProjectileCatalogV1",
    ) -> "EntityArchetypeCatalogV1":
        return cls.from_card_specs(
            normal_mode_card_specs(card_catalog.by_id, require_frozen_baseline=True),
            static_logic=static_logic,
            projectile_catalog=projectile_catalog,
        )

    @property
    def vocab_size(self) -> int:
        return FIRST_REAL_ENTITY_ARCHETYPE_VOCAB_ID + len(self.archetype_keys)

    @property
    def catalog_id(self) -> str:
        return content_hash(
            {
                "version": "entity-archetype-catalog.v4",
                "card_scope": self.card_scope,
                "archetype_keys": self.archetype_keys,
                "character_global_ids": self.character_global_ids,
                "archetype_node_ids": self.archetype_node_ids,
                "archetype_key_aliases": self.archetype_key_aliases,
                "archetype_node_aliases": self.archetype_node_aliases,
                "metadata": tuple(
                    (
                        row.archetype_key,
                        row.child_kind,
                        row.child_kind_known,
                        row.is_airborne,
                        row.is_airborne_known,
                        row.static_damage_basis,
                        row.static_damage_basis_known,
                        row.projectile_radius_tiles,
                        row.projectile_radius_known,
                        row.lifetime_ms,
                        row.lifetime_known,
                    )
                    for row in self.archetype_metadata
                ),
            }
        )

    def vocab_id(self, key: str) -> int:
        return self._vocab_by_key.get(str(key), UNKNOWN_ENTITY_ARCHETYPE_VOCAB_ID)

    def form_vocab_id(self, form_id: str) -> int:
        return self.vocab_id(f"form:{form_id}")

    def projectile_vocab_id(self, projectile_global_id: int) -> int:
        return self.vocab_id(f"projectile:{int(projectile_global_id)}")

    def area_vocab_id(self, area_global_id: int) -> int:
        return self.vocab_id(f"area:{int(area_global_id)}")

    def node_vocab_id(self, node_id: str) -> int:
        return self._vocab_by_node_id.get(str(node_id), UNKNOWN_ENTITY_ARCHETYPE_VOCAB_ID)

    def character_vocab_id(self, character_global_id: int) -> int:
        return self._character_vocab_by_global_id.get(int(character_global_id), UNKNOWN_ENTITY_ARCHETYPE_VOCAB_ID)

    def runtime_global_vocab_id(self, data_global_id: int) -> int:
        """Resolve an exact live LogicData identity across runtime namespaces."""

        return self._runtime_vocab_by_global_id.get(int(data_global_id), UNKNOWN_ENTITY_ARCHETYPE_VOCAB_ID)

    def metadata_for_vocab_id(self, vocab_id: int) -> EntityArchetypeMetadataV1:
        if int(vocab_id) in {PAD_ENTITY_ARCHETYPE_VOCAB_ID, UNKNOWN_ENTITY_ARCHETYPE_VOCAB_ID}:
            return _unknown_archetype_metadata("unknown")
        index = int(vocab_id) - FIRST_REAL_ENTITY_ARCHETYPE_VOCAB_ID
        if not 0 <= index < len(self.archetype_metadata):
            raise ValueError("entity archetype vocab ID is outside this catalog")
        return self.archetype_metadata[index]


@dataclass(frozen=True, slots=True)
class CardCatalogV1:
    """Canonical raw-card ordering plus deterministic semantic features.

    Real cards are sorted by their positive raw native card ID.  Therefore the
    same card set always produces the same vocabulary rows without depending
    on deck order or checkpoint metadata.  PAD and UNKNOWN occupy rows 0 and 1.
    """

    raw_card_ids: tuple[int, ...]
    static_features: tuple[tuple[float, ...], ...]
    mechanic_features: tuple[tuple[float, ...], ...]
    version: str = UNIVERSAL_CARD_CATALOG_VERSION

    def __post_init__(self) -> None:
        if self.version != UNIVERSAL_CARD_CATALOG_VERSION:
            raise ValueError("unsupported universal card catalog version")
        if any(
            isinstance(card_id, bool) or not isinstance(card_id, int) or card_id <= 0 for card_id in self.raw_card_ids
        ):
            raise ValueError("raw card IDs must be positive integers")
        if tuple(sorted(set(self.raw_card_ids))) != self.raw_card_ids:
            raise ValueError("raw card IDs must be unique and sorted")
        expected = len(self.raw_card_ids)
        if not len(self.static_features) == len(self.mechanic_features) == expected:
            raise ValueError("catalog feature rows do not match card ordering")
        for label, rows in (("static", self.static_features), ("mechanic", self.mechanic_features)):
            widths = {len(row) for row in rows}
            if len(widths) > 1:
                raise ValueError(f"catalog {label} rows have inconsistent widths")
            if any(not math.isfinite(value) for row in rows for value in row):
                raise ValueError(f"catalog {label} rows must be finite")
        if self.static_features and self.static_dim != len(CARD_STATIC_FEATURE_NAMES):
            raise ValueError("catalog static rows do not match their named schema")
        if self.mechanic_features and self.mechanic_dim != len(CARD_MECHANIC_FEATURE_NAMES):
            raise ValueError("catalog mechanic rows do not match their named schema")

    @classmethod
    def from_feature_rows(
        cls,
        card_specs: Mapping[int, CardSpecV1],
        feature_rows: Mapping[int, CardFeatureRow],
        *,
        config: ModelConfigV4 | None = None,
    ) -> "CardCatalogV1":
        """Build the canonical catalog from the existing exact card features."""

        actual = config or ModelConfigV4()
        raw_card_ids = tuple(sorted(int(card_id) for card_id in card_specs))
        if set(raw_card_ids) != set(int(card_id) for card_id in feature_rows):
            missing_rows = sorted(set(raw_card_ids).difference(feature_rows))
            extra_rows = sorted(set(feature_rows).difference(raw_card_ids))
            raise ValueError(f"card specs and feature rows differ: missing={missing_rows}, extra={extra_rows}")
        static: list[tuple[float, ...]] = []
        mechanics: list[tuple[float, ...]] = []
        for card_id in raw_card_ids:
            spec = card_specs[card_id]
            row = feature_rows[card_id]
            static.append(tuple(pack_static_features(spec, row, width=actual.card_static_dim)))
            mechanic = tuple(pack_mechanic_features(spec, row, width=actual.card_mechanic_dim))
            mechanics.append(mechanic)
        return cls(raw_card_ids=raw_card_ids, static_features=tuple(static), mechanic_features=tuple(mechanics))

    @classmethod
    def from_card_spec_catalog(
        cls,
        card_catalog: CardSpecCatalog,
        *,
        static_logic: StaticCardLogicCatalogV1 | None = None,
        config: ModelConfigV4 | None = None,
    ) -> "CardCatalogV1":
        """Build every global row from exact static and Ability evidence."""

        exact_logic = static_logic or build_static_card_logic_catalog(card_catalog=card_catalog)
        scoped_specs = normal_mode_card_specs(card_catalog.by_id, require_frozen_baseline=True)
        feature_rows = build_card_feature_rows(
            scoped_specs, static_logic=exact_logic, ability_specs=card_catalog.abilities
        )
        return cls.from_feature_rows(scoped_specs, feature_rows, config=config)

    @classmethod
    def from_raw_ids(
        cls,
        raw_card_ids: Sequence[int],
        *,
        static_dim: int = len(CARD_STATIC_FEATURE_NAMES),
        mechanic_dim: int = len(CARD_MECHANIC_FEATURE_NAMES),
    ) -> "CardCatalogV1":
        """Create a zero-feature catalog for tests and explicit fixtures."""

        ordered = tuple(sorted({int(card_id) for card_id in raw_card_ids}))
        return cls(
            raw_card_ids=ordered,
            static_features=tuple((0.0,) * static_dim for _ in ordered),
            mechanic_features=tuple((0.0,) * mechanic_dim for _ in ordered),
        )

    @property
    def vocab_size(self) -> int:
        return FIRST_REAL_CARD_VOCAB_ID + len(self.raw_card_ids)

    @property
    def catalog_id(self) -> str:
        """Content identity for audit logs; checkpoints need not own ordering."""

        return content_hash(
            {
                "version": self.version,
                "feature_schema_version": CARD_FEATURE_SCHEMA_VERSION,
                "static_feature_names": CARD_STATIC_FEATURE_NAMES,
                "mechanic_feature_names": CARD_MECHANIC_FEATURE_NAMES,
                "raw_card_ids": self.raw_card_ids,
                "static_features": self.static_features,
                "mechanic_features": self.mechanic_features,
            }
        )

    @property
    def static_dim(self) -> int:
        return len(self.static_features[0]) if self.static_features else 0

    @property
    def mechanic_dim(self) -> int:
        return len(self.mechanic_features[0]) if self.mechanic_features else 0

    def vocab_id(self, raw_card_id: int | None) -> int:
        if raw_card_id is None:
            return PAD_CARD_VOCAB_ID
        try:
            return FIRST_REAL_CARD_VOCAB_ID + self.raw_card_ids.index(int(raw_card_id))
        except ValueError:
            return UNKNOWN_CARD_VOCAB_ID

    def require_vocab_id(self, raw_card_id: int, *, context: str = "card") -> int:
        """Map one concrete public card ID and reject off-scope content."""

        vocab_id = self.vocab_id(raw_card_id)
        if vocab_id == UNKNOWN_CARD_VOCAB_ID:
            raise ValueError(
                f"{context} {int(raw_card_id)} is outside this V4 catalog "
                "(the production baseline is the frozen 122-card normal-mode set)"
            )
        return vocab_id

    def raw_card_id(self, vocab_id: int) -> int | None:
        if vocab_id in {PAD_CARD_VOCAB_ID, UNKNOWN_CARD_VOCAB_ID}:
            return None
        index = int(vocab_id) - FIRST_REAL_CARD_VOCAB_ID
        if not 0 <= index < len(self.raw_card_ids):
            raise ValueError("card vocab ID is outside this catalog")
        return self.raw_card_ids[index]

    def feature_tensors(
        self, *, dtype: torch.dtype = torch.float32, device: torch.device | str | None = None
    ) -> tuple[Tensor, Tensor]:
        """Return PAD/UNKNOWN-prefixed catalog tensors in canonical order."""

        static_dim = self.static_dim
        mechanic_dim = self.mechanic_dim
        static_rows = ((0.0,) * static_dim,) * 2 + self.static_features
        mechanic_rows = ((0.0,) * mechanic_dim,) * 2 + self.mechanic_features
        return (
            torch.tensor(static_rows, dtype=dtype, device=device),
            torch.tensor(mechanic_rows, dtype=dtype, device=device),
        )
