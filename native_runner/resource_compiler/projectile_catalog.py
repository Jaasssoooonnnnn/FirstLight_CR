"""Content-addressed static semantics for native projectile identities.

The native projectile snapshot exposes ``LogicProjectileData+0x40`` as
``projectileDataGlobalId``.  This module joins that uint32 identity to the
lossless ``PROJECTILE`` nodes in :mod:`native_runner.card_logic` using the
exact old-format global-ID registry and the native type+name FNV-1a fallback.

Static ``HomingTime > 0`` proves that a projectile is configured to home.  It
does not prove that a particular live projectile currently has a secondary
tracking target; that runtime state remains explicitly unavailable here.
"""

from __future__ import annotations

from native_runner.paths import WORKSPACE_ROOT

import csv
from dataclasses import dataclass, field
import hashlib
import io
import json
from pathlib import Path
import re
from typing import Any, ClassVar, Mapping

from native_runner.resource_compiler.card_logic import StaticCardLogicCatalogV1
from native_runner.resource_compiler.card_specs import sha256_file
from native_runner.contracts import ContractError, ContractMixin, content_hash, frozen_mapping

try:
    from sc_compression import decompress as _sc_decompress
except ImportError:  # pragma: no cover - minimal actor hosts fail closed below.
    _sc_decompress = None


PROJECTILE_CATALOG_VERSION = "native-projectile-catalog.v1"
PROJECTILE_CATALOG_CRITERIA_VERSION = "native-projectile-semantics.2026-07-24.v1"
RUNTIME_PROJECTILE_RESOLUTION_VERSION = "native-runtime-projectile-resolution.v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UINT32_MAX = 0xFFFFFFFF
_PROJECTILE_GLOBAL_ID_TYPE = "Projectile"
_GLOBAL_IDS_OLDFORMAT_RELATIVE = Path("assets/csv_logic/global_ids_oldformat.csv")
_GLOBAL_ID_STRATEGIES = frozenset({"explicit_oldformat_registry", "fnv1a_type_name"})
_FNV1A_OFFSET_BASIS = 0x811C9DC5
_FNV1A_PRIME = 0x01000193


def _fnv1a32(value: str) -> int:
    result = _FNV1A_OFFSET_BASIS
    for byte in value.encode("utf-8"):
        result ^= byte
        result = (result * _FNV1A_PRIME) & _UINT32_MAX
    return result


def logic_data_global_id(data_type: str, name: str) -> int:
    """Resolve the engine's stable fallback identity for one LogicData row."""

    if not data_type or not name:
        raise ValueError("LogicData type and name must not be empty")
    return _fnv1a32(f"{data_type}{name}")


def _global_id_key(projectile_name: str) -> str:
    return f"{_PROJECTILE_GLOBAL_ID_TYPE}{projectile_name}"


def _relative_source(path: Path, workspace: Path) -> str:
    resolved = path.resolve()
    return resolved.relative_to(workspace).as_posix() if resolved.is_relative_to(workspace) else resolved.as_posix()


def _read_global_id_registry(
    card_logic: StaticCardLogicCatalogV1, workspace: Path
) -> tuple[Path, bytes, list[dict[str, str]], str]:
    release = Path(card_logic.source_release)
    release_root = release if release.is_absolute() else workspace / release
    path = release_root / _GLOBAL_IDS_OLDFORMAT_RELATIVE
    plain: bytes | None = None
    if path.is_file():
        raw = path.read_bytes()
        if raw.lstrip(b"\xef\xbb\xbf\r\n\t ").startswith(b'"ID","Name","Type"'):
            plain = raw
        elif _sc_decompress is not None:
            try:
                decoded, _signature, _version = _sc_decompress(raw)
            except Exception:
                decoded = b""
            if decoded.lstrip(b"\xef\xbb\xbf\r\n\t ").startswith(b'"ID","Name","Type"'):
                plain = decoded
    if plain is None:
        raise ContractError(f"exact LogicData global-ID registry is missing or undecodable: {path}")
    with io.StringIO(plain.decode("utf-8-sig"), newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or rows[0].get("ID") != "int":
        raise ContractError("global_ids_oldformat.csv is missing its SC type row")
    return path, plain, rows, _relative_source(path, workspace)


def _global_id_entries(
    rows: list[dict[str, str]], *, data_type: str, source: str
) -> tuple[dict[str, int], dict[str, str]]:
    entries: dict[str, int] = {}
    sources: dict[str, str] = {}
    ids: dict[int, str] = {}
    for row_index, row in enumerate(rows[1:]):
        if row.get("Type") != data_type:
            continue
        name = str(row.get("Name", "")).strip()
        try:
            global_id = int(str(row.get("ID", "")))
        except ValueError as error:
            raise ContractError(f"invalid {data_type} global ID at {source}#named-row={row_index}") from error
        if not name or not 1 <= global_id <= _UINT32_MAX:
            raise ContractError(f"invalid {data_type} registry entry at {source}#named-row={row_index}")
        if name in entries:
            raise ContractError(f"duplicate {data_type} registry name: {name}")
        if global_id in ids:
            raise ContractError(f"duplicate {data_type} registry ID {global_id}: {ids[global_id]}, {name}")
        entries[name] = global_id
        ids[global_id] = name
        sources[name] = f"{source}#named-row={row_index}"
    return entries, sources


def load_logic_data_global_ids(
    card_logic: StaticCardLogicCatalogV1, *, data_type: str, workspace_root: str | Path | None = None
) -> dict[str, int]:
    """Load exact registered IDs for one public LogicData type."""

    workspace = Path(workspace_root or WORKSPACE_ROOT).resolve()
    _path, _plain, rows, source = _read_global_id_registry(card_logic, workspace)
    entries, _sources = _global_id_entries(rows, data_type=data_type, source=source)
    return entries


def _load_global_id_registry(card_logic: StaticCardLogicCatalogV1, workspace: Path) -> Mapping[str, Any]:
    path, plain, rows, relative = _read_global_id_registry(card_logic, workspace)
    entries, sources = _global_id_entries(rows, data_type=_PROJECTILE_GLOBAL_ID_TYPE, source=relative)

    return {
        "registry_kind": "global_ids_oldformat.csv",
        "global_id_type": _PROJECTILE_GLOBAL_ID_TYPE,
        "source_record": relative,
        "raw_sha256": sha256_file(path),
        "decoded_sha256": hashlib.sha256(plain).hexdigest(),
        "explicit_projectile_count": len(entries),
        "projectile_entries": entries,
        "entry_source_records": sources,
        "fallback_algorithm": "fnv1a32_utf8_type_plus_name",
        "fallback_prefix": _PROJECTILE_GLOBAL_ID_TYPE,
        "fallback_offset_basis": _FNV1A_OFFSET_BASIS,
        "fallback_prime": _FNV1A_PRIME,
        "native_evidence": (
            "libg.arm64-v15.535.13@0xd7401c:name_hash_registry_path",
            "libg.arm64-v15.535.13@0xd740a4:offset_basis=0x811c9dc5",
            "libg.arm64-v15.535.13@0xd740c4:prime=0x01000193",
            "libg.arm64-v15.535.13@0xdc2560:LogicData_type=0x0a",
        ),
    }


def _resolve_inherited_record(
    name: str, records: Mapping[str, Mapping[str, Any]], *, chain: tuple[str, ...] = ()
) -> tuple[dict[str, Any], tuple[str, ...], tuple[str, ...]]:
    if name in chain:
        return dict(records[name]), chain + (name,), (f"cyclic_base:{name}",)
    record = dict(records[name])
    base = record.get("Base")
    if not isinstance(base, str) or not base:
        return record, (name,), ()
    if base.startswith("PROJECTILE."):
        base = base.split(".", 1)[1]
    if base not in records:
        return record, (name,), (f"missing_base:{base}",)
    inherited, inherited_chain, unresolved = _resolve_inherited_record(base, records, chain=chain + (name,))
    inherited.update(record)
    return inherited, inherited_chain + (name,), unresolved


def _optional_nonnegative_number(value: Any, label: str) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ContractError(f"{label} must be a non-negative number or null")
    return value


@dataclass(frozen=True, slots=True)
class NativeProjectileDefinitionV1(ContractMixin):
    projectile_name: str
    node_id: str
    projectile_global_id: int
    global_id_strategy: str
    global_id_key: str
    configured_homing: bool
    homing_time_ms: int | float | None
    homing_min_distance: int | float | None
    raw_record: Mapping[str, Any]
    resolved_record: Mapping[str, Any]
    inheritance_chain: tuple[str, ...]
    unresolved_semantics: tuple[str, ...]
    source_records: tuple[str, ...]
    version: str = field(default=PROJECTILE_CATALOG_VERSION, init=False)
    VERSION: ClassVar[str] = PROJECTILE_CATALOG_VERSION

    def __post_init__(self) -> None:
        if not self.projectile_name or self.node_id != f"PROJECTILE.{self.projectile_name}":
            raise ContractError("native projectile definition has an invalid identity")
        if (
            isinstance(self.projectile_global_id, bool)
            or not isinstance(self.projectile_global_id, int)
            or not 1 <= self.projectile_global_id <= _UINT32_MAX
        ):
            raise ContractError("native projectile definition has an invalid uint32 ID")
        if self.global_id_strategy not in _GLOBAL_ID_STRATEGIES:
            raise ContractError("native projectile definition has an invalid ID strategy")
        if self.global_id_key != _global_id_key(self.projectile_name):
            raise ContractError("native projectile definition has an invalid ID hash key")
        if self.global_id_strategy == "fnv1a_type_name" and self.projectile_global_id != _fnv1a32(self.global_id_key):
            raise ContractError(
                f"native projectile {self.projectile_name!r} global ID does not match exact FNV-1a type+name hash"
            )
        homing_time = _optional_nonnegative_number(self.homing_time_ms, "homing_time_ms")
        homing_min_distance = _optional_nonnegative_number(self.homing_min_distance, "homing_min_distance")
        if self.configured_homing != bool(homing_time is not None and homing_time > 0):
            raise ContractError("configured_homing must be derived exactly from HomingTime > 0")
        object.__setattr__(self, "homing_time_ms", homing_time)
        object.__setattr__(self, "homing_min_distance", homing_min_distance)
        object.__setattr__(self, "raw_record", frozen_mapping(self.raw_record))
        object.__setattr__(self, "resolved_record", frozen_mapping(self.resolved_record))
        object.__setattr__(self, "inheritance_chain", tuple(self.inheritance_chain))
        object.__setattr__(self, "unresolved_semantics", tuple(sorted(set(self.unresolved_semantics))))
        object.__setattr__(self, "source_records", tuple(sorted(set(self.source_records))))
        if not self.source_records:
            raise ContractError("native projectile definition requires source records")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "NativeProjectileDefinitionV1":
        if value.get("version") != cls.VERSION:
            raise ContractError(f"unsupported native projectile definition: {value.get('version')}")
        return cls(
            projectile_name=str(value["projectile_name"]),
            node_id=str(value["node_id"]),
            projectile_global_id=value["projectile_global_id"],
            global_id_strategy=str(value["global_id_strategy"]),
            global_id_key=str(value["global_id_key"]),
            configured_homing=value["configured_homing"],
            homing_time_ms=value.get("homing_time_ms"),
            homing_min_distance=value.get("homing_min_distance"),
            raw_record=value.get("raw_record", {}),
            resolved_record=value.get("resolved_record", {}),
            inheritance_chain=tuple(value.get("inheritance_chain", ())),
            unresolved_semantics=tuple(value.get("unresolved_semantics", ())),
            source_records=tuple(value.get("source_records", ())),
        )


@dataclass(frozen=True, slots=True)
class NativeProjectileCatalogV1(ContractMixin):
    card_logic_catalog_id: str
    criteria_version: str
    global_id_registry: Mapping[str, Any]
    projectiles: tuple[NativeProjectileDefinitionV1, ...]
    version: str = field(default=PROJECTILE_CATALOG_VERSION, init=False)
    VERSION: ClassVar[str] = PROJECTILE_CATALOG_VERSION

    def __post_init__(self) -> None:
        if not _SHA256.fullmatch(self.card_logic_catalog_id):
            raise ContractError("projectile catalog requires a card-logic SHA-256")
        if self.criteria_version != PROJECTILE_CATALOG_CRITERIA_VERSION:
            raise ContractError("unsupported projectile catalog criteria")
        ordered = tuple(sorted(self.projectiles, key=lambda item: item.projectile_name))
        if len({item.projectile_name for item in ordered}) != len(ordered):
            raise ContractError("projectile catalog contains duplicate names")
        global_ids = tuple(item.projectile_global_id for item in ordered)
        if len(set(global_ids)) != len(global_ids):
            raise ContractError("projectile catalog contains duplicate global IDs")

        registry = frozen_mapping(self.global_id_registry)
        entries = registry.get("projectile_entries")
        sources = registry.get("entry_source_records")
        if not isinstance(entries, Mapping) or not isinstance(sources, Mapping):
            raise ContractError("projectile catalog has an invalid ID registry")
        if (
            registry.get("registry_kind") != "global_ids_oldformat.csv"
            or registry.get("global_id_type") != _PROJECTILE_GLOBAL_ID_TYPE
            or registry.get("explicit_projectile_count") != len(entries)
            or set(entries) != set(sources)
            or registry.get("fallback_prefix") != _PROJECTILE_GLOBAL_ID_TYPE
            or registry.get("fallback_offset_basis") != _FNV1A_OFFSET_BASIS
            or registry.get("fallback_prime") != _FNV1A_PRIME
        ):
            raise ContractError("projectile catalog registry identity is invalid")
        for digest_name in ("raw_sha256", "decoded_sha256"):
            digest = registry.get(digest_name)
            if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
                raise ContractError(f"projectile catalog registry {digest_name} is invalid")
        for item in ordered:
            explicit = entries.get(item.projectile_name)
            if explicit is None:
                if item.global_id_strategy != "fnv1a_type_name":
                    raise ContractError("unregistered projectile must use FNV-1a ID")
            elif item.global_id_strategy != "explicit_oldformat_registry" or item.projectile_global_id != explicit:
                raise ContractError("registered projectile global ID does not match")
        object.__setattr__(self, "global_id_registry", registry)
        object.__setattr__(self, "projectiles", ordered)

    @property
    def catalog_id(self) -> str:
        return content_hash(self)

    @property
    def by_name(self) -> dict[str, NativeProjectileDefinitionV1]:
        return {item.projectile_name: item for item in self.projectiles}

    @property
    def by_global_id(self) -> dict[int, NativeProjectileDefinitionV1]:
        return {item.projectile_global_id: item for item in self.projectiles}

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "catalog_id": self.catalog_id,
            "card_logic_catalog_id": self.card_logic_catalog_id,
            "projectile_count": len(self.projectiles),
            "explicit_oldformat_global_id_count": sum(
                item.global_id_strategy == "explicit_oldformat_registry" for item in self.projectiles
            ),
            "fnv1a_global_id_count": sum(item.global_id_strategy == "fnv1a_type_name" for item in self.projectiles),
            "configured_homing_count": sum(item.configured_homing for item in self.projectiles),
            "runtime_active_homing_available": False,
            "projectiles_with_unresolved_semantics": sum(bool(item.unresolved_semantics) for item in self.projectiles),
        }

    def resolve_runtime_projectile(self, value: Mapping[str, Any]) -> dict[str, Any]:
        """Join one validated native projectile without inventing active homing."""

        global_id = value.get("projectileDataGlobalId")
        if isinstance(global_id, bool) or not isinstance(global_id, int) or not 1 <= global_id <= _UINT32_MAX:
            raise ContractError("runtime projectile global ID is invalid")
        definition = self.by_global_id.get(global_id)
        result: dict[str, Any] = {
            "version": RUNTIME_PROJECTILE_RESOLUTION_VERSION,
            "runtimeIdentity": {"projectileDataGlobalId": global_id},
            "definitionStatus": "resolved" if definition is not None else "unresolved",
            "projectileCatalogId": self.catalog_id,
            "cardLogicCatalogId": self.card_logic_catalog_id,
            "joinKey": "PROJECTILE.projectile_global_id",
            "provenance": {
                "runtimeIdentity": "authoritative_native_LogicProjectileData+0x40",
                "staticSemantics": (
                    "authoritative_static_card_logic"
                    if definition is not None
                    else "unavailable_unknown_projectile_global_id"
                ),
                "activeHoming": "unavailable_secondary_homing_target_not_exposed",
            },
            "activeHoming": None,
        }
        if definition is not None:
            result.update(
                {
                    "projectileName": definition.projectile_name,
                    "projectileNodeId": definition.node_id,
                    "configuredHoming": definition.configured_homing,
                    "homingTimeMs": definition.homing_time_ms,
                    "homingMinDistance": definition.homing_min_distance,
                    "resolvedRecord": definition.to_dict()["resolved_record"],
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
                raise FileExistsError(f"immutable projectile catalog path contains different bytes: {destination}")
            return destination
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(destination)
        return destination

    def save_content_addressed(self, directory: str | Path) -> Path:
        return self.save(Path(directory) / f"projectile-catalog-{self.catalog_id}.json")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "NativeProjectileCatalogV1":
        if value.get("version") != cls.VERSION:
            raise ContractError(f"unsupported native projectile catalog: {value.get('version')}")
        projectiles = value.get("projectiles", ())
        if not isinstance(projectiles, (list, tuple)):
            raise ContractError("native projectile catalog projectiles must be a sequence")
        return cls(
            card_logic_catalog_id=str(value["card_logic_catalog_id"]),
            criteria_version=str(value["criteria_version"]),
            global_id_registry=value.get("global_id_registry", {}),
            projectiles=tuple(NativeProjectileDefinitionV1.from_mapping(item) for item in projectiles),
        )

    @classmethod
    def load(cls, path: str | Path) -> "NativeProjectileCatalogV1":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))


def build_projectile_catalog(
    card_logic: StaticCardLogicCatalogV1, workspace_root: str | Path | None = None
) -> NativeProjectileCatalogV1:
    workspace = Path(workspace_root or WORKSPACE_ROOT).resolve()
    registry = _load_global_id_registry(card_logic, workspace)
    explicit_ids = registry["projectile_entries"]
    if not isinstance(explicit_ids, Mapping):
        raise ContractError("Projectile global-ID registry entries are invalid")
    nodes = {str(node["name"]): node for node in card_logic.nodes.values() if node.get("namespace") == "PROJECTILE"}
    records = {name: dict(node.get("merged_record", {})) for name, node in nodes.items()}
    projectiles: list[NativeProjectileDefinitionV1] = []
    for name, node in sorted(nodes.items()):
        raw_record = records[name]
        resolved, inheritance, unresolved = _resolve_inherited_record(name, records)
        source_records = tuple(
            str(fragment["source_record"])
            for fragment in node.get("fragments", ())
            if isinstance(fragment, Mapping) and fragment.get("source_record")
        )
        global_id_key = _global_id_key(name)
        explicit_global_id = explicit_ids.get(name)
        if explicit_global_id is None:
            projectile_global_id = logic_data_global_id(_PROJECTILE_GLOBAL_ID_TYPE, name)
            strategy = "fnv1a_type_name"
        else:
            projectile_global_id = int(explicit_global_id)
            strategy = "explicit_oldformat_registry"
        homing_time = resolved.get("HomingTime")
        homing_min_distance = resolved.get("HomingMinDistance")
        projectiles.append(
            NativeProjectileDefinitionV1(
                projectile_name=name,
                node_id=str(node["node_id"]),
                projectile_global_id=projectile_global_id,
                global_id_strategy=strategy,
                global_id_key=global_id_key,
                configured_homing=isinstance(homing_time, (int, float))
                and not isinstance(homing_time, bool)
                and homing_time > 0,
                homing_time_ms=homing_time,
                homing_min_distance=homing_min_distance,
                raw_record=raw_record,
                resolved_record=resolved,
                inheritance_chain=inheritance,
                unresolved_semantics=unresolved,
                source_records=source_records,
            )
        )
    return NativeProjectileCatalogV1(
        card_logic_catalog_id=card_logic.catalog_id,
        criteria_version=PROJECTILE_CATALOG_CRITERIA_VERSION,
        global_id_registry=registry,
        projectiles=tuple(projectiles),
    )
