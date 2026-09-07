"""Validated runtime identity joins from pinned competitive resource facts."""

from __future__ import annotations
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, ClassVar, Mapping
from .card_logic import StaticCardLogicCatalogV1
from .competitive_data import load_competitive_data
from .contracts import ContractError, ContractMixin, content_hash, frozen_mapping

EFFECT_CATALOG_VERSION = "native-effect-catalog.v3"


EFFECT_CATALOG_CRITERIA_VERSION = "native-effect-semantics.2026-08-18.v6"


RUNTIME_EFFECT_RESOLUTION_VERSION = "native-runtime-effect-resolution.v3"


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


_UINT32_MAX = 0xFFFFFFFF


_BUFF_GLOBAL_ID_TYPE = "Buff"


_BUFF_GLOBAL_ID_PREFIX = _BUFF_GLOBAL_ID_TYPE


_GLOBAL_IDS_OLDFORMAT_RELATIVE = Path("assets/csv_logic/global_ids_oldformat.csv")


_GLOBAL_ID_STRATEGIES = frozenset({"explicit_oldformat_registry", "fnv1a_type_name"})


_FNV1A_OFFSET_BASIS = 0x811C9DC5


_FNV1A_PRIME = 0x01000193


def _fnv1a32(value: str) -> int:
    result = _FNV1A_OFFSET_BASIS
    for byte in value.encode("utf-8"):
        result = ((result ^ byte) * _FNV1A_PRIME) & _UINT32_MAX
    return result


def _effect_global_id_key(effect_name: str) -> str:
    return f"{_BUFF_GLOBAL_ID_PREFIX}{effect_name}"


@dataclass(frozen=True, slots=True)
class NativeEffectDefinitionV1(ContractMixin):
    effect_name: str
    node_id: str
    buff_global_id: int
    global_id_strategy: str
    global_id_key: str
    mechanic_tags: tuple[str, ...]
    mechanic_evidence: Mapping[str, tuple[str, ...]]
    application_sources: tuple[Mapping[str, Any], ...]
    modifiers: Mapping[str, Any]
    lifecycle: Mapping[str, Any]
    targeting: Mapping[str, Any]
    spawn: Mapping[str, Any]
    overrides: Mapping[str, Any]
    raw_record: Mapping[str, Any]
    resolved_record: Mapping[str, Any]
    inheritance_chain: tuple[str, ...]
    unresolved_semantics: tuple[str, ...]
    source_records: tuple[str, ...]
    version: str = field(default=EFFECT_CATALOG_VERSION, init=False)
    VERSION: ClassVar[str] = EFFECT_CATALOG_VERSION

    def __post_init__(self) -> None:
        if not self.effect_name or self.node_id != f"BUFF.{self.effect_name}":
            raise ContractError("native effect definition has an invalid identity")
        if (
            isinstance(self.buff_global_id, bool)
            or not isinstance(self.buff_global_id, int)
            or not 1 <= self.buff_global_id <= _UINT32_MAX
        ):
            raise ContractError("native effect definition has an invalid uint32 Buff global ID")
        if self.global_id_strategy not in _GLOBAL_ID_STRATEGIES:
            raise ContractError("native effect definition has an invalid ID strategy")
        if self.global_id_key != _effect_global_id_key(self.effect_name):
            raise ContractError("native effect definition has an invalid ID hash key")
        if self.global_id_strategy == "fnv1a_type_name" and self.buff_global_id != _fnv1a32(self.global_id_key):
            raise ContractError(
                f"native effect {self.effect_name!r} Buff global ID does not match exact FNV-1a type+name hash"
            )
        if not self.mechanic_tags:
            raise ContractError("native effect definition must have a semantic tag")
        object.__setattr__(self, "mechanic_tags", tuple(sorted(set(self.mechanic_tags))))
        normalized_evidence = {
            str(tag): tuple(sorted(set(str(item) for item in items))) for tag, items in self.mechanic_evidence.items()
        }
        if set(normalized_evidence) != set(self.mechanic_tags) or any(
            not items for items in normalized_evidence.values()
        ):
            raise ContractError("native effect mechanic tags require non-empty exact evidence")
        object.__setattr__(self, "mechanic_evidence", frozen_mapping(normalized_evidence))
        applications: list[Mapping[str, Any]] = []
        for application in self.application_sources:
            if not isinstance(application, Mapping) or not {
                "source_node_id",
                "field_path",
                "application_kind",
                "parameters",
                "parameter_provenance",
                "source_records",
            }.issubset(application):
                raise ContractError("native effect has an invalid application source")
            applications.append(frozen_mapping(application))
        object.__setattr__(
            self,
            "application_sources",
            tuple(sorted(applications, key=lambda item: (str(item["source_node_id"]), str(item["field_path"])))),
        )
        for field_name in (
            "modifiers",
            "lifecycle",
            "targeting",
            "spawn",
            "overrides",
            "raw_record",
            "resolved_record",
        ):
            object.__setattr__(self, field_name, frozen_mapping(getattr(self, field_name)))
        object.__setattr__(self, "inheritance_chain", tuple(self.inheritance_chain))
        object.__setattr__(self, "unresolved_semantics", tuple(sorted(set(self.unresolved_semantics))))
        object.__setattr__(self, "source_records", tuple(sorted(set(self.source_records))))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "NativeEffectDefinitionV1":
        if value.get("version") != cls.VERSION:
            raise ContractError(f"unsupported native effect definition: {value.get('version')}")
        return cls(
            effect_name=str(value["effect_name"]),
            node_id=str(value["node_id"]),
            buff_global_id=value["buff_global_id"],
            global_id_strategy=str(value["global_id_strategy"]),
            global_id_key=str(value["global_id_key"]),
            mechanic_tags=tuple(value.get("mechanic_tags", ())),
            mechanic_evidence=value.get("mechanic_evidence", {}),
            application_sources=tuple(value.get("application_sources", ())),
            modifiers=value.get("modifiers", {}),
            lifecycle=value.get("lifecycle", {}),
            targeting=value.get("targeting", {}),
            spawn=value.get("spawn", {}),
            overrides=value.get("overrides", {}),
            raw_record=value.get("raw_record", {}),
            resolved_record=value.get("resolved_record", {}),
            inheritance_chain=tuple(value.get("inheritance_chain", ())),
            unresolved_semantics=tuple(value.get("unresolved_semantics", ())),
            source_records=tuple(value.get("source_records", ())),
        )


@dataclass(frozen=True, slots=True)
class NativeEffectCatalogV1(ContractMixin):
    card_logic_catalog_id: str
    criteria_version: str
    global_id_registry: Mapping[str, Any]
    effects: tuple[NativeEffectDefinitionV1, ...]
    version: str = field(default=EFFECT_CATALOG_VERSION, init=False)
    VERSION: ClassVar[str] = EFFECT_CATALOG_VERSION

    def __post_init__(self) -> None:
        if not _SHA256.fullmatch(self.card_logic_catalog_id):
            raise ContractError("effect catalog requires a card-logic SHA-256")
        if self.criteria_version != EFFECT_CATALOG_CRITERIA_VERSION:
            raise ContractError("unsupported effect catalog criteria")
        ordered = tuple(sorted(self.effects, key=lambda item: item.effect_name))
        if len({item.effect_name for item in ordered}) != len(ordered):
            raise ContractError("effect catalog contains duplicate names")
        registry = frozen_mapping(self.global_id_registry)
        entries = registry.get("buff_entries")
        sources = registry.get("entry_source_records")
        if not isinstance(entries, Mapping) or not isinstance(sources, Mapping):
            raise ContractError("effect catalog has an invalid Buff ID registry")
        source_record = registry.get("source_record")
        if (
            registry.get("registry_kind") != "global_ids_oldformat.csv"
            or registry.get("global_id_type") != _BUFF_GLOBAL_ID_TYPE
            or not isinstance(source_record, str)
            or not source_record.replace("\\", "/").endswith(_GLOBAL_IDS_OLDFORMAT_RELATIVE.as_posix())
        ):
            raise ContractError("effect catalog Buff registry identity is invalid")
        for digest_name in ("raw_sha256", "decoded_sha256"):
            digest = registry.get(digest_name)
            if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
                raise ContractError(f"effect catalog Buff registry {digest_name} is invalid")
        if registry.get("explicit_buff_count") != len(entries):
            raise ContractError("effect catalog Buff ID registry count is invalid")
        if set(entries) != set(sources):
            raise ContractError("effect catalog Buff registry provenance is incomplete")
        entry_ids: list[int] = []
        for name, global_id in entries.items():
            if (
                not isinstance(name, str)
                or not name
                or isinstance(global_id, bool)
                or not isinstance(global_id, int)
                or not 1 <= global_id <= _UINT32_MAX
                or not str(sources[name]).startswith(f"{source_record}#named-row=")
            ):
                raise ContractError("effect catalog Buff registry entry is invalid")
            entry_ids.append(global_id)
        if len(set(entry_ids)) != len(entry_ids):
            raise ContractError("effect catalog Buff registry IDs are not unique")
        if registry.get("fallback_prefix") != _BUFF_GLOBAL_ID_PREFIX:
            raise ContractError("effect catalog Buff hash prefix is invalid")
        if (
            registry.get("fallback_offset_basis") != _FNV1A_OFFSET_BASIS
            or registry.get("fallback_prime") != _FNV1A_PRIME
        ):
            raise ContractError("effect catalog Buff hash parameters are invalid")
        for item in ordered:
            explicit = entries.get(item.effect_name)
            if explicit is not None:
                if (
                    item.global_id_strategy != "explicit_oldformat_registry"
                    or item.buff_global_id != explicit
                    or item.effect_name not in sources
                ):
                    raise ContractError(
                        f"native effect {item.effect_name!r} conflicts with explicit old-format Buff registry"
                    )
            elif item.global_id_strategy != "fnv1a_type_name":
                raise ContractError(f"native effect {item.effect_name!r} must use exact FNV-1a type+name fallback")
        unknown_entries = sorted(set(entries).difference(item.effect_name for item in ordered))
        if unknown_entries:
            raise ContractError(
                "Buff ID registry contains definitions absent from the effect catalog: " + ", ".join(unknown_entries)
            )
        global_ids = tuple(item.buff_global_id for item in ordered)
        if len(set(global_ids)) != len(global_ids):
            duplicates = sorted(global_id for global_id in set(global_ids) if global_ids.count(global_id) > 1)
            raise ContractError(
                "effect catalog contains duplicate Buff global IDs: "
                + ", ".join(str(global_id) for global_id in duplicates)
            )
        object.__setattr__(self, "global_id_registry", registry)
        object.__setattr__(self, "effects", ordered)

    @property
    def catalog_id(self) -> str:
        return content_hash(self)

    @property
    def by_name(self) -> dict[str, NativeEffectDefinitionV1]:
        return {item.effect_name: item for item in self.effects}

    @property
    def by_global_id(self) -> dict[int, NativeEffectDefinitionV1]:
        return {item.buff_global_id: item for item in self.effects}

    @property
    def summary(self) -> dict[str, Any]:
        all_tags = sorted({tag for item in self.effects for tag in item.mechanic_tags})
        return {
            "catalog_id": self.catalog_id,
            "card_logic_catalog_id": self.card_logic_catalog_id,
            "effect_count": len(self.effects),
            "effects_with_buff_global_id": len(self.by_global_id),
            "effects_without_buff_global_id": 0,
            "explicit_oldformat_global_id_count": sum(
                item.global_id_strategy == "explicit_oldformat_registry" for item in self.effects
            ),
            "fnv1a_global_id_count": sum(item.global_id_strategy == "fnv1a_type_name" for item in self.effects),
            "global_ids_with_high_bit_set": sum(item.buff_global_id > 0x7FFFFFFF for item in self.effects),
            "mechanic_tag_count": len(all_tags),
            "mechanic_tags": all_tags,
            "effects_with_unresolved_semantics": sum(bool(item.unresolved_semantics) for item in self.effects),
        }

    def resolve_runtime_effect(self, value: Mapping[str, Any]) -> dict[str, Any]:
        """Join one validated native activeEffect without inventing semantics."""

        name = value.get("name")
        global_id = value.get("buffGlobalId")
        remaining_ms = value.get("remainingMs")
        source_key = value.get("sourceEntityKey")
        if not isinstance(name, str) or not name:
            raise ContractError("runtime effect name is invalid")
        if isinstance(global_id, bool) or not isinstance(global_id, int) or not 1 <= global_id <= _UINT32_MAX:
            raise ContractError("runtime effect global ID is invalid")
        if isinstance(remaining_ms, bool) or not isinstance(remaining_ms, int) or remaining_ms < -1:
            raise ContractError("runtime effect remaining duration is invalid")
        definition_by_name = self.by_name.get(name)
        definition_by_global_id = self.by_global_id.get(global_id)
        identity_mismatch = (definition_by_name is not None and definition_by_name.buff_global_id != global_id) or (
            definition_by_global_id is not None and definition_by_global_id.effect_name != name
        )
        definition = None if identity_mismatch else definition_by_name
        if definition is None:
            definition_status = "identity_mismatch" if identity_mismatch else "unresolved"
        else:
            definition_status = "resolved"
        result: dict[str, Any] = {
            "version": RUNTIME_EFFECT_RESOLUTION_VERSION,
            "runtimeIdentity": {
                "name": name,
                "buffGlobalId": global_id,
                "remainingMs": remaining_ms,
                "sourceEntityKey": source_key,
            },
            "definitionStatus": definition_status,
            "effectCatalogId": self.catalog_id,
            "cardLogicCatalogId": self.card_logic_catalog_id,
            "joinKey": "BUFF.buff_global_id+name",
            "provenance": {
                "runtimeIdentity": "authoritative_native_type3_component",
                "staticIdentity": (
                    definition.global_id_strategy
                    if definition is not None
                    else ("unavailable_identity_mismatch" if identity_mismatch else "unavailable_unknown_buff")
                ),
                "staticSemantics": (
                    "authoritative_static_card_logic"
                    if definition is not None
                    else (
                        "unavailable_runtime_static_identity_mismatch"
                        if identity_mismatch
                        else "unavailable_unknown_buff_name"
                    )
                ),
            },
        }
        if definition is not None:
            result.update(
                {
                    "effectNodeId": definition.node_id,
                    "mechanicTags": list(definition.mechanic_tags),
                    "modifiers": definition.to_dict()["modifiers"],
                    "lifecycle": definition.to_dict()["lifecycle"],
                    "targeting": definition.to_dict()["targeting"],
                    "spawn": definition.to_dict()["spawn"],
                    "overrides": definition.to_dict()["overrides"],
                    "applicationSources": definition.to_dict()["application_sources"],
                    "mechanicEvidence": definition.to_dict()["mechanic_evidence"],
                    "unresolvedSemantics": list(definition.unresolved_semantics),
                }
            )
        return result

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_json(pretty=True) + "\n"
        if destination.exists():
            if destination.read_text(encoding="utf-8") != payload:
                raise FileExistsError(f"immutable effect catalog path contains different bytes: {destination}")
            return destination
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(destination)
        return destination

    def save_content_addressed(self, directory: str | Path) -> Path:
        return self.save(Path(directory) / f"effect-catalog-{self.catalog_id}.json")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "NativeEffectCatalogV1":
        if value.get("version") != cls.VERSION:
            raise ContractError(f"unsupported native effect catalog: {value.get('version')}")
        effects = value.get("effects", ())
        if not isinstance(effects, (list, tuple)):
            raise ContractError("native effect catalog effects must be a sequence")
        return cls(
            card_logic_catalog_id=str(value["card_logic_catalog_id"]),
            criteria_version=str(value["criteria_version"]),
            global_id_registry=value.get("global_id_registry", {}),
            effects=tuple(NativeEffectDefinitionV1.from_mapping(item) for item in effects),
        )

    @classmethod
    def load(cls, path: str | Path) -> "NativeEffectCatalogV1":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))


def build_effect_catalog(
    card_logic: StaticCardLogicCatalogV1, workspace_root: str | Path | None = None
) -> NativeEffectCatalogV1:
    data = load_competitive_data("effects", workspace_root)
    if card_logic.catalog_id != data["card_logic_catalog_id"]:
        raise ContractError("static logic differs from the pinned competitive release")
    return NativeEffectCatalogV1.from_mapping(data)
