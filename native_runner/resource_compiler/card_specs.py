"""Generate versioned CardSpecV1 records from locally supplied game configuration.

Supercell asset tables in an APK/update are often SC-compressed even though
their names end in ``.csv`` or ``.toml``.  This loader verifies plaintext and,
when the repository's sc-compression dependency is available, decodes assets
in memory without changing the originals.  Unavailable fields remain explicit
unknowns instead of being filled from stale third-party values.
"""

from __future__ import annotations

from native_runner.paths import WORKSPACE_ROOT

from collections.abc import Iterable, Mapping
import csv
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import re
import tomllib
from typing import Any

try:  # Repository tooling dependency; plaintext captures remain a fallback.
    from sc_compression import decompress as _sc_decompress
except ImportError:  # pragma: no cover - exercised on minimal actor hosts.
    _sc_decompress = None

from native_runner.contracts import (
    AbilitySpecV1,
    CardKind,
    CardSpecV1,
    ContractError,
    ContractMixin,
    EvolutionSpecV1,
    FrozenMapping,
    MechanicOpV1,
    SemanticEvidenceLevel,
    TargetKind,
    TargetSchemaV1,
    content_hash,
    frozen_mapping,
)

from native_runner.normal_form_evidence import (
    NORMAL_MODE_DIRECT_HERO_BINDINGS_BY_CARD_ID,
    NORMAL_MODE_HERO_FORM_BINDINGS_BY_CARD_ID,
)

CATALOG_VERSION = "card-spec-catalog.v1"
FIELD_APPLICABILITY_VERSION = "card-spec-field-applicability.v3"

# Exact-build native loader defaults for LogicCharacterData integer fields.
# The ARM64 v15.535.13 loader passes zero to its integer reader for ``Speed``
# at 0xd9ac48..0xd9ac5c (data+0x410; copied to effective data+0x414 at
# 0xd97df4..0xd97dfc) and for ``Hitpoints`` at 0xd9bc90..0xd9bca4
# (data+0x7a8).  Keep the raw decoded definition untouched; these defaults are
# used only when an actual entity definition exists and the compact field is
# applicable to that card kind.
_NATIVE_CHARACTER_DEFAULT_RELEASE = "nr_15.535.13_release_ff1c6c29"
_NATIVE_CHARACTER_DEFAULT_BINARY_RELATIVE = Path("lib/arm64-v8a/libg.so")
_NATIVE_CHARACTER_DEFAULT_BINARY_SHA256 = "110aa2b5cac391c498645e072b0d88729428c2c2845e7e2737ca8ee979059783"
_NATIVE_CHARACTER_INT_DEFAULTS: Mapping[str, int] = {"Hitpoints": 0, "Speed": 0}
_NATIVE_CHARACTER_DEFAULT_EVIDENCE: Mapping[str, tuple[str, ...]] = {
    "Hitpoints": (
        "native_loader_default:LogicCharacterData.Hitpoints=0",
        f"source_release:{_NATIVE_CHARACTER_DEFAULT_RELEASE}",
        f"libg.arm64.sha256:{_NATIVE_CHARACTER_DEFAULT_BINARY_SHA256}",
        "libg.arm64-v15.535.13@0xd9bc90:integer_default=0->data+0x7a8",
    ),
    "Speed": (
        "native_loader_default:LogicCharacterData.Speed=0",
        f"source_release:{_NATIVE_CHARACTER_DEFAULT_RELEASE}",
        f"libg.arm64.sha256:{_NATIVE_CHARACTER_DEFAULT_BINARY_SHA256}",
        "libg.arm64-v15.535.13@0xd9ac48:integer_default=0->data+0x410",
        "libg.arm64-v15.535.13@0xd97df4:data+0x410->effective+0x414",
    ),
}

# ``SemanticEvidenceLevel`` is shared with live telemetry and intentionally
# stays small.  These two values are static-projection decisions, not runtime
# evidence levels: the lossless source is present, but the compact CardSpec
# scalar is either a compound mechanic or has no proved one-scalar projection.
COMPOUND_DECLARED = "compound_declared"
RAW_STATIC_ONLY = "raw_static_only"

_HEAL_SPIRIT_CARD_ID = 28_000_016

_DEFINITION_NAMESPACES = (
    "CHARACTER",
    "BUILDING",
    "PROJECTILE",
    "ABILITY",
    "EXT",
    "ACTION",
    "AEO",
    "SPELL_CHARACTER",
    "SPELL_BUILDING",
    "SPELL_OTHER",
    "SPELL_HERO",
    "SPELL_EVOLVED",
)

# Aggregate TOMLs use ``[Name]`` while per-card TOMLs use
# ``[NAMESPACE.Name]``.  Both encodings feed the same native registries.
_TOML_DEFAULT_NAMESPACE: Mapping[str, str] = {
    "actions.toml": "ACTION",
    "area_effect_objects.toml": "AEO",
    "area_effect_objects_evo.toml": "AEO",
    "buildings.toml": "BUILDING",
    "buildings_evo.toml": "BUILDING",
    "characters.toml": "CHARACTER",
    "characters_evo.toml": "CHARACTER",
    "projectiles.toml": "PROJECTILE",
    "projectiles_evo.toml": "PROJECTILE",
    "spells_buildings.toml": "SPELL_BUILDING",
    "spells_characters.toml": "SPELL_CHARACTER",
    "spells_evolved.toml": "SPELL_EVOLVED",
    "spells_heroes.toml": "SPELL_HERO",
    "spells_other.toml": "SPELL_OTHER",
}

_CARD_DEFINITION_NAMESPACE: Mapping[str, str] = {
    "spells_characters.csv": "SPELL_CHARACTER",
    "spells_buildings.csv": "SPELL_BUILDING",
    "spells_other.csv": "SPELL_OTHER",
    "spells_heroes.csv": "SPELL_HERO",
}

_ACTION_REFERENCE_FIELDS = frozenset(
    {
        "ActionOnCapturedObject",
        "ActionOnCooldownReady",
        "ActionOnFlyHeightReached",
        "ActionOnGround",
        "ActionOnLanding",
        "ActionOnSelfWhenTriggered",
        "ActionOnSelfWhenTriggeredLeft",
        "ActionOnSelfWhenTriggeredRight",
        "ActionOnStartDescending",
        "ActionOnTargetReached",
        "ActionOnTargets",
        "ActionToGetDataFrom",
        "ActionToGetTargetFrom",
        "ActionToRunIfNoMatch",
        "ActionToRunOnSpawned",
        "ActionToRun",
        "ActionToTakeDataFrom",
        "ActionToExecute",
        "ActionWhenUnitBuffed",
        "DoAttackAction",
        "HasTargetOnDeployAction",
        "HideAction",
        "HideActions",
        "InstigatorAction",
        "NextAction",
        "NoTargetOnDeployAction",
        "OnActivationAction",
        "OnAfterDashAction",
        "OnAttackAction",
        "OnDamageTakenAction",
        "OnExecuteAction",
        "OnFalseAction",
        "OnHitAction",
        "OnKillAction",
        "OnKilledAction",
        "OnChainBegan",
        "OnSpawnAction",
        "OnStartAction",
        "OnStartingAction",
        "OnStartingAttackAction",
        "OnTrueAction",
        "PostSpawnAction",
        "SelfAction",
        "ShieldLostAction",
        "SubActions",
        "SummonActionData",
        "SuccessAction",
        "SuccessActionOnInstigator",
        "FailureAction",
        "FailureActionOnInstigator",
        "FirstAppearAction",
        "TetherHitAction",
        "OnTetherActivationActionOnConnectedUnit",
        "WarpAction",
    }
)
CARD_TABLES: tuple[tuple[str, int, CardKind], ...] = (
    ("spells_characters.csv", 26_000_000, CardKind.TROOP),
    ("spells_buildings.csv", 27_000_000, CardKind.BUILDING),
    ("spells_other.csv", 28_000_000, CardKind.SPELL),
    ("spells_heroes.csv", 29_000_000, CardKind.HERO),
)

_INT = re.compile(r"^[+-]?\d+$")
_FLOAT = re.compile(r"^[+-]?(?:\d+\.\d*|\d*\.\d+)(?:[eE][+-]?\d+)?$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=16)
def _cached_file_identity_sha256(name: str, size: int, modified_ns: int) -> str:
    del size, modified_ns
    return sha256_file(Path(name))


def _matches_native_character_default_build(workspace: Path, source: Path, source_release: str) -> bool:
    if source_release != _NATIVE_CHARACTER_DEFAULT_RELEASE:
        return False
    release_roots = tuple(ancestor for ancestor in (source, *source.parents) if ancestor.name == source_release)
    candidates = release_roots or (workspace / source_release,)
    for release_root in dict.fromkeys(candidates):
        binary = release_root / _NATIVE_CHARACTER_DEFAULT_BINARY_RELATIVE
        if not binary.is_file():
            continue
        stat = binary.stat()
        actual = _cached_file_identity_sha256(str(binary.resolve()), stat.st_size, stat.st_mtime_ns)
        return actual == _NATIVE_CHARACTER_DEFAULT_BINARY_SHA256
    return False


def _natural_version(path: Path) -> tuple[int, ...]:
    values = re.findall(r"\d+", path.as_posix())
    return tuple(int(item) for item in values[-5:])


def _looks_like_text_asset(value: bytes) -> bool:
    prefix = value[:256]
    if b"\x00" in prefix:
        return False
    stripped = prefix.lstrip(b"\xef\xbb\xbf\r\n\t ")
    return stripped.startswith((b'"Name"', b"Name,", b"["))


@lru_cache(maxsize=512)
def _asset_bytes_cached(name: str, size: int, modified_ns: int) -> bytes | None:
    raw = Path(name).read_bytes()
    if _looks_like_text_asset(raw):
        return raw
    if _sc_decompress is None:
        return None
    try:
        plain, _signature, _version = _sc_decompress(raw)
    except Exception:
        return None
    return plain if _looks_like_text_asset(plain) else None


def _asset_bytes(path: Path) -> bytes | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    stat = path.stat()
    return _asset_bytes_cached(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def _is_plaintext(path: Path) -> bool:
    """Compatibility name: true when an asset yields verified plaintext."""

    return _asset_bytes(path) is not None


def _source_candidates(workspace_root: Path, explicit: Iterable[str | Path]) -> list[Path]:
    candidates: list[Path] = []
    for raw in explicit:
        root = Path(raw).resolve()
        for candidate in (root, root / "assets" / "csv_logic", root / "csv_logic"):
            if candidate not in candidates:
                candidates.append(candidate)
    releases = sorted(workspace_root.glob("nr_*_release_*"), key=_natural_version, reverse=True)
    candidates.extend(
        item / "assets" / "csv_logic" for item in releases if item / "assets" / "csv_logic" not in candidates
    )
    capture_root = workspace_root / "captures" / "decompressed_assets"
    if capture_root.is_dir():
        discovered = sorted(
            (item / "assets" / "csv_logic" for item in capture_root.iterdir() if item.is_dir()),
            key=_natural_version,
            reverse=True,
        )
        candidates.extend(item for item in discovered if item not in candidates)
    # A caller can still explicitly select the historical server tables.  They
    # are not an automatic fallback because that would label old stats current.
    return candidates


def discover_card_source(workspace_root: str | Path, source_roots: Iterable[str | Path] = ()) -> Path:
    root = Path(workspace_root).resolve()
    checked: list[str] = []
    for candidate in _source_candidates(root, source_roots):
        checked.append(str(candidate))
        if _is_plaintext(candidate / "spells_characters.csv") and _is_plaintext(candidate / "spells_other.csv"):
            return candidate
    raise FileNotFoundError("no decompressed card tables found; checked: " + ", ".join(checked))


def _scalar(value: str | None) -> Any:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if _INT.fullmatch(text):
        return int(text)
    if _FLOAT.fullmatch(text):
        return float(text)
    return text


def _read_sc_csv(path: Path) -> tuple[dict[str, str], list[dict[str, Any]]]:
    plain = _asset_bytes(path)
    if plain is None:
        raise ContractError(f"asset table is not decompressed plaintext: {path}")
    with io.StringIO(plain.decode("utf-8-sig"), newline="") as stream:
        reader = csv.DictReader(stream)
        raw_rows = list(reader)
    if not raw_rows:
        return {}, []
    type_row = raw_rows[0]
    # SC tables use an object-name continuation convention.  Only named rows
    # allocate data-table IDs; this is why RoyalGiant is 26000024 rather than
    # 26000025 in the current capture.
    rows = [
        {key: _scalar(value) for key, value in row.items() if key is not None and _scalar(value) is not None}
        for row in raw_rows[1:]
        if row.get("Name", "").strip()
    ]
    return {key: str(value) for key, value in type_row.items() if key is not None and value}, rows


def _load_named_table(path: Path) -> dict[str, dict[str, Any]]:
    if not _is_plaintext(path):
        return {}
    _, rows = _read_sc_csv(path)
    return {str(row["Name"]): row for row in rows if row.get("Name")}


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_definition_fragments(
    source: Path, *, replacement_root: Path | None = None
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[tuple[str, str], tuple[Path, ...]]]:
    """Load exact TOML definitions grouped by normalized gameplay namespace.

    The v15 CSVs intentionally contain mostly names.  Numeric definitions live
    in per-mechanic TOMLs such as ``characters/knight.toml``.  We merge only
    explicit tables from those files; no gameplay value is inferred.
    """

    definitions: dict[str, dict[str, dict[str, Any]]] = {namespace: {} for namespace in _DEFINITION_NAMESPACES}
    provenance: dict[tuple[str, str], list[Path]] = {}
    replacement_paths = (
        {
            path.relative_to(replacement_root).as_posix().casefold()
            for path in replacement_root.rglob("*.toml")
            if _is_plaintext(path)
        }
        if replacement_root is not None and replacement_root.is_dir()
        else set()
    )
    candidates = sorted(
        path
        for path in set(source.rglob("*.toml"))
        if path.relative_to(source).as_posix().casefold() not in replacement_paths
    )
    for path in candidates:
        try:
            plain = _asset_bytes(path)
            if plain is None:
                continue
            document = tomllib.loads(plain.decode("utf-8-sig").rstrip("\x00\r\n\t "))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
            raise ContractError(f"invalid decompressed definition TOML {path}: {error}") from error
        for group in definitions:
            records = document.get(group, {})
            if not isinstance(records, Mapping):
                continue
            for name, raw_record in records.items():
                if not isinstance(raw_record, Mapping):
                    continue
                key = str(name)
                definitions[group][key] = _deep_merge(definitions[group].get(key, {}), raw_record)
                provenance.setdefault((group, key), []).append(path)
        default_namespace = _TOML_DEFAULT_NAMESPACE.get(path.name.lower())
        if default_namespace is not None:
            for name, raw_record in document.items():
                if name in definitions or not isinstance(raw_record, Mapping):
                    continue
                key = str(name)
                definitions[default_namespace][key] = _deep_merge(
                    definitions[default_namespace].get(key, {}), raw_record
                )
                provenance.setdefault((default_namespace, key), []).append(path)
    return definitions, {key: tuple(value) for key, value in provenance.items()}


def _same_definition_sources(
    provenance: Mapping[tuple[str, str], tuple[Path, ...]], left: tuple[str, str], right: tuple[str, str]
) -> bool:
    left_sources = {path.resolve() for path in provenance.get(left, ())}
    right_sources = {path.resolve() for path in provenance.get(right, ())}
    return bool(left_sources) and left_sources == right_sources


def _resolve_definition_inheritance(
    definitions: Mapping[str, Mapping[str, Mapping[str, Any]]], provenance: Mapping[tuple[str, str], tuple[Path, ...]]
) -> dict[str, dict[str, dict[str, Any]]]:
    """Apply explicit Base links plus proved native semantic aliases.

    TOML ``CHARACTER`` is the semantic LogicCharacterData namespace.  The
    decoded registry keeps physical ``BUILDING`` sections separate, so a
    qualified ``CHARACTER.X`` base can name the unique ``BUILDING.X`` record
    when that record explicitly declares ``IsBuilding=true``.  Derived EXT
    records can likewise be named through their semantic base namespace.
    """

    resolved: dict[str, dict[str, dict[str, Any]]] = {namespace: {} for namespace in definitions}
    active: set[tuple[str, str]] = set()

    def resolve(namespace: str, name: str) -> dict[str, Any]:
        key = (namespace, name)
        if name in resolved.get(namespace, {}):
            return resolved[namespace][name]
        raw = definitions.get(namespace, {}).get(name, {})
        if key in active or not raw:
            return dict(raw)
        active.add(key)
        base_namespace = namespace
        base_name: str | None = None
        base = raw.get("Base")
        if isinstance(base, str) and base:
            if "." in base:
                candidate_namespace, candidate_name = base.split(".", 1)
                if candidate_namespace in definitions:
                    base_namespace, base_name = candidate_namespace, candidate_name
            else:
                base_name = base
        if base_name not in definitions.get(base_namespace, {}):
            extension = definitions.get("EXT", {}).get(base_name or "", {})
            extension_base = extension.get("Base") if isinstance(extension, Mapping) else None
            if isinstance(extension_base, str) and extension_base.startswith(f"{base_namespace}."):
                base_namespace = "EXT"
            elif base_namespace == "CHARACTER":
                building = definitions.get("BUILDING", {}).get(base_name or "", {})
                if isinstance(building, Mapping) and building.get("IsBuilding") is True:
                    base_namespace = "BUILDING"
        inherited = resolve(base_namespace, base_name) if base_name in definitions.get(base_namespace, {}) else {}
        value = _deep_merge(inherited, raw)
        # A same-file ``CHARACTER.Name`` partial table can append nested data
        # to the derived ``EXT.Name`` object.  EliteArcherHero is the sole
        # active-source instance: its partial table contains only
        # AttackSequenceList while EXT supplies Base and identity fields.
        if namespace == "EXT" and isinstance(base, str) and "." in base:
            semantic_namespace = base.split(".", 1)[0]
            overlay = definitions.get(semantic_namespace, {}).get(name, {})
            if (
                isinstance(overlay, Mapping)
                and overlay
                and "Base" not in overlay
                and "Name" not in overlay
                and _same_definition_sources(provenance, ("EXT", name), (semantic_namespace, name))
            ):
                value = _deep_merge(value, overlay)
        active.remove(key)
        resolved[namespace][name] = value
        return value

    for namespace, records in definitions.items():
        for name in records:
            resolve(namespace, name)
    return resolved


def _number(row: Mapping[str, Any], *keys: str) -> float | int | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
    return None


def _milliseconds(value: Any, *, seconds: bool = False) -> int | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return max(0, int(round(float(value) * (1000 if seconds else 1))))


def _game_units(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return float(value) / 1000.0


def _declared_field_evidence(label: str, record: Mapping[str, Any], keys: Iterable[str]) -> list[str]:
    return [f"{label}.{key}" for key in keys if record.get(key) is not None]


def _field_applicability(
    *,
    kind: CardKind,
    row: Mapping[str, Any],
    unit: Mapping[str, Any],
    projectile: Mapping[str, Any],
    area_effect: Mapping[str, Any],
    standard: Mapping[str, float | int | None],
    summoned_names: tuple[str, ...] | None = None,
    count_projection_status: str | None = None,
    count_projection_evidence: tuple[str, ...] = (),
) -> dict[str, dict[str, Any]]:
    """Classify every normalized field without treating absence as unknown.

    ``CardSpecV1`` keeps nullable scalar fields for wire compatibility.  This
    sidecar records why a null means either *not applicable* or a genuine
    extraction gap.  Applicability is based on explicit type/reference and
    mechanic fields in the decoded records plus exact-build native loader
    defaults; card names are never used.  The lossless graph performs a
    second, wider closure check in :mod:`native_runner.semantic_coverage`.
    """

    entity_kind = kind in {CardKind.TROOP, CardKind.BUILDING, CardKind.HERO}
    if summoned_names is None:
        summoned_names = tuple(
            str(row[key])
            for key in ("SummonCharacter", "SummonCharacterSecond")
            if isinstance(row.get(key), str) and row.get(key)
        ) + tuple(str(item) for item in row.get("SummonCharactersList", ()) if isinstance(item, str) and item)
    multiple_summoned_forms = len(summoned_names) > 1
    unit_expected = entity_kind or bool(summoned_names)
    unit_available = bool(unit)
    projectile_name = row.get("Projectile") or unit.get("Projectile")
    projectile_expected = isinstance(projectile_name, str) and bool(projectile_name)
    attack_declared = any(
        unit.get(key) not in (None, False, "")
        for key in ("Damage", "DamageSpecial", "Projectile", "CustomFirstProjectile", "AttackSequenceList")
    )
    area_declared = any(
        record.get(key) not in (None, False, 0, "")
        for record, keys in (
            (row, ("Radius", "AreaEffectObject")),
            (unit, ("AreaDamageRadius", "DeathDamageRadius", "DashRadius", "ReflectedAttackRadius")),
            (projectile, ("Radius", "AoeToAir", "AoeToGround")),
            (area_effect, ("Radius", "Damage", "Buff")),
        )
        for key in keys
    )
    knockback_declared = any(
        record.get(key) not in (None, False, 0, "")
        for record, keys in (
            (row, ("Pushback",)),
            (unit, ("AttackPushBack", "DeathPushBack", "DashPushBack", "DashingPushback", "SpawnPushback")),
            (projectile, ("Pushback",)),
        )
        for key in keys
    )
    duration_declared = any(
        record.get(key) not in (None, False, 0, "")
        for record, keys in ((row, ("DurationSeconds", "AreaEffectObject")), (unit, ("LifeTime", "UpTimeMs")))
        for key in keys
    )
    shield_declared = any(
        unit.get(key) not in (None, False, 0, "")
        for key in (
            "ShieldHitpoints",
            "ShieldLostAction",
            "ShieldLostEffect",
            "BlueShieldExportName",
            "RedShieldExportName",
        )
    )

    declared_evidence: dict[str, list[str]] = {
        "elixir_cost": _declared_field_evidence("root", row, ("ManaCost", "DarkElixirCost")),
        "hitpoints": _declared_field_evidence("unit", unit, ("Hitpoints",)),
        "damage": [
            *_declared_field_evidence("unit", unit, ("Damage",)),
            *_declared_field_evidence("root", row, ("InstantDamage",)),
            *_declared_field_evidence("projectile", projectile, ("Damage",)),
            *_declared_field_evidence("area_effect", area_effect, ("Damage",)),
        ],
        "hit_speed_ms": _declared_field_evidence("unit", unit, ("HitSpeed",)),
        "range_tiles": _declared_field_evidence("unit", unit, ("Range",)),
        "move_speed": _declared_field_evidence("unit", unit, ("Speed",)),
        "deploy_time_ms": [
            *_declared_field_evidence("root", row, ("CustomDeployTime",)),
            *_declared_field_evidence("unit", unit, ("DeployTime", "DeployDelay")),
        ],
        "count": _declared_field_evidence("root", row, ("SummonNumber", "SummonCharacterSecondCount")),
        "radius_tiles": [
            *_declared_field_evidence("root", row, ("Radius",)),
            *_declared_field_evidence("projectile", projectile, ("Radius",)),
            *_declared_field_evidence("unit", unit, ("AreaDamageRadius",)),
            *_declared_field_evidence("area_effect", area_effect, ("Radius",)),
        ],
        "projectile_speed": _declared_field_evidence("projectile", projectile, ("Speed",)),
        "knockback": [
            *_declared_field_evidence("root", row, ("Pushback",)),
            *_declared_field_evidence("projectile", projectile, ("Pushback",)),
            *_declared_field_evidence("unit", unit, ("AttackPushBack",)),
        ],
        "duration_ms": [
            *_declared_field_evidence("root", row, ("DurationSeconds",)),
            *_declared_field_evidence("unit", unit, ("LifeTime", "UpTimeMs")),
        ],
        "shield_hitpoints": _declared_field_evidence("unit", unit, ("ShieldHitpoints",)),
    }
    for normalized_name, source_name, applicable in (
        ("hitpoints", "Hitpoints", entity_kind and unit_available),
        (
            "move_speed",
            "Speed",
            entity_kind and kind != CardKind.BUILDING and unit_available and not bool(unit.get("IsBuilding")),
        ),
    ):
        if (
            applicable
            and unit.get(source_name) is None
            and standard.get(normalized_name) == _NATIVE_CHARACTER_INT_DEFAULTS[source_name]
        ):
            declared_evidence[normalized_name].extend(_NATIVE_CHARACTER_DEFAULT_EVIDENCE[source_name])
    if standard.get("count") is not None:
        declared_evidence["count"].extend(
            f"root.{name_key}:implicit_count=1"
            for name_key, count_key in (
                ("SummonCharacter", "SummonNumber"),
                ("SummonCharacterSecond", "SummonCharacterSecondCount"),
            )
            if isinstance(row.get(name_key), str) and row.get(name_key) and row.get(count_key) is None
        )
        listed_forms = tuple(item for item in row.get("SummonCharactersList", ()) if isinstance(item, str) and item)
        if listed_forms:
            declared_evidence["count"].append(f"root.SummonCharactersList:explicit_count={len(listed_forms)}")
        declared_evidence["count"].extend(count_projection_evidence)

    missing_applicability: dict[str, tuple[bool, tuple[str, ...]]] = {
        "elixir_cost": (True, ("all_playable_cards_require_cost",)),
        "hitpoints": (
            entity_kind,
            (
                "entity_card_requires_base_unit_hitpoints",
                f"base_unit_definition:{'present' if unit_available else 'missing'}",
            ),
        ),
        "damage": (
            (entity_kind and not unit_available) or attack_declared,
            (
                f"base_unit_definition:{'present' if unit_available else 'missing'}",
                f"direct_attack_declared:{str(attack_declared).lower()}",
            ),
        ),
        "hit_speed_ms": (
            (entity_kind and not unit_available) or attack_declared,
            (
                f"base_unit_definition:{'present' if unit_available else 'missing'}",
                f"direct_attack_declared:{str(attack_declared).lower()}",
            ),
        ),
        "range_tiles": (
            (entity_kind and not unit_available) or attack_declared,
            (
                f"base_unit_definition:{'present' if unit_available else 'missing'}",
                f"direct_attack_declared:{str(attack_declared).lower()}",
            ),
        ),
        "move_speed": (
            entity_kind and kind != CardKind.BUILDING and not bool(unit.get("IsBuilding")),
            (
                f"card_kind:{kind.value}",
                f"base_unit_definition:{'present' if unit_available else 'missing'}",
                f"unit_is_building:{str(bool(unit.get('IsBuilding'))).lower()}",
            ),
        ),
        "deploy_time_ms": (
            entity_kind,
            (f"card_kind:{kind.value}", f"base_unit_definition:{'present' if unit_available else 'missing'}"),
        ),
        "count": (
            unit_expected,
            (f"entity_card:{str(entity_kind).lower()}", f"summoned_form_reference_count:{len(summoned_names)}"),
        ),
        "radius_tiles": (area_declared, (f"area_mechanic_declared:{str(area_declared).lower()}",)),
        "projectile_speed": (
            projectile_expected,
            (
                f"projectile_reference:{projectile_name or 'absent'}",
                f"projectile_definition:{'present' if projectile else 'missing'}",
            ),
        ),
        "knockback": (knockback_declared, (f"knockback_mechanic_declared:{str(knockback_declared).lower()}",)),
        "duration_ms": (duration_declared, (f"lifetime_mechanic_declared:{str(duration_declared).lower()}",)),
        "shield_hitpoints": (shield_declared, (f"shield_mechanic_declared:{str(shield_declared).lower()}",)),
    }

    # A multi-form deployment has per-form values, not one card-wide scalar.
    # Keep the legacy compact value for wire compatibility, but never label it
    # as a complete projection of the whole deployment.
    per_form_fields = {
        "hitpoints",
        "damage",
        "hit_speed_ms",
        "range_tiles",
        "move_speed",
        "deploy_time_ms",
        "radius_tiles",
        "projectile_speed",
        "knockback",
        "duration_ms",
        "shield_hitpoints",
    }
    raw_only_when_missing = {
        "damage": unit_available
        and any(
            unit.get(key) not in (None, False, "") for key in ("DamageSpecial", "AttackSequenceList", "Projectile")
        ),
        "radius_tiles": area_declared,
        "projectile_speed": projectile_expected and bool(projectile),
        "knockback": knockback_declared,
        "duration_ms": duration_declared,
        "shield_hitpoints": shield_declared,
    }

    result: dict[str, dict[str, Any]] = {}
    for field_name, value in standard.items():
        if field_name == "count" and count_projection_status is not None:
            result[field_name] = {
                "status": count_projection_status,
                "evidence": count_projection_evidence,
                "projection_blocker": "runtime_schedule_has_no_unique_static_count",
            }
            continue
        if multiple_summoned_forms and field_name in per_form_fields:
            result[field_name] = {
                "status": COMPOUND_DECLARED,
                "evidence": tuple(
                    (
                        *(f"root.deploy_form[{index}]={name}" for index, name in enumerate(summoned_names)),
                        "multiple_deploy_forms:no_single_card_scalar",
                    )
                ),
                "projection_blocker": "multiple_deploy_forms",
            }
            continue
        if value is not None:
            result[field_name] = {
                "status": SemanticEvidenceLevel.STATIC_DECLARED.value,
                "evidence": tuple(declared_evidence[field_name]),
            }
            continue
        applicable, evidence = missing_applicability[field_name]
        status = (
            RAW_STATIC_ONLY
            if applicable and raw_only_when_missing.get(field_name, False)
            else (SemanticEvidenceLevel.UNKNOWN.value if applicable else SemanticEvidenceLevel.NOT_APPLICABLE.value)
        )
        result[field_name] = {"status": status, "evidence": evidence}
        if status == RAW_STATIC_ONLY:
            result[field_name]["projection_blocker"] = "mechanic_declared_without_unique_compact_scalar"
    return result


def _target_schema(card_id: int, kind: CardKind, row: Mapping[str, Any]) -> TargetSchemaV1:
    deploy_enemy = bool(row.get("CanDeployOnEnemySide", False))
    deploys_troop = str(row.get("TypeOfSpell") or "") == "Troop"
    if card_id == _HEAL_SPIRIT_CARD_ID:
        if not bool(row.get("SpellAsDeploy", False)) or row.get("SummonCharacter") != "HealSpirit" or deploy_enemy:
            raise ContractError("Heal Spirit no longer matches its exact own-side deployment contract")
        deploys_troop = True
    mask = "full_arena" if deploy_enemy or (kind == CardKind.SPELL and not deploys_troop) else "own_deployment_zone"
    return TargetSchemaV1(allowed=(TargetKind.GRID,), placement_mask_key=mask, requires_visible_target=False)


def _mechanics(
    row: Mapping[str, Any], unit: Mapping[str, Any], projectile: Mapping[str, Any], area_effect: Mapping[str, Any] = {}
) -> tuple[MechanicOpV1, ...]:
    result: list[MechanicOpV1] = []
    summons: list[tuple[str, int]] = []
    for name_key, count_key in (
        ("SummonCharacter", "SummonNumber"),
        ("SummonCharacterSecond", "SummonCharacterSecondCount"),
    ):
        name = row.get(name_key)
        if isinstance(name, str):
            count = int(row.get(count_key, 1) or 1)
            summons.append((name, count))
    summons.extend((str(name), 1) for name in row.get("SummonCharactersList", ()) if isinstance(name, str) and name)
    if summons:
        result.append(
            MechanicOpV1(
                trigger="OnDeploy",
                effect="Spawn",
                target_selector="deployment_area",
                parameters={"forms": [{"name": name, "count": count} for name, count in summons]},
            )
        )
    if row.get("InstantDamage") not in (None, 0):
        result.append(
            MechanicOpV1(
                "OnDeploy", "Damage", target_selector="configured_target", parameters={"raw": row["InstantDamage"]}
            )
        )
    if row.get("InstantHeal") not in (None, 0):
        result.append(
            MechanicOpV1(
                "OnDeploy", "Heal", target_selector="configured_target", parameters={"raw": row["InstantHeal"]}
            )
        )
    if row.get("HealPerSecond") not in (None, 0):
        result.append(
            MechanicOpV1(
                "Periodic",
                "Heal",
                target_selector="configured_area",
                parameters={"raw_per_second": row["HealPerSecond"]},
            )
        )
    if row.get("Pushback") not in (None, 0):
        result.append(
            MechanicOpV1("OnHit", "Knockback", target_selector="configured_target", parameters={"raw": row["Pushback"]})
        )
    if row.get("AreaEffectObject"):
        result.append(MechanicOpV1("OnDeploy", "CreateArea", parameters={"object": row["AreaEffectObject"]}))
    if area_effect.get("Damage") not in (None, 0):
        result.append(
            MechanicOpV1(
                "OnDeploy",
                "Damage",
                target_selector="configured_area",
                parameters={
                    "raw": area_effect["Damage"],
                    "radius_raw": area_effect.get("Radius"),
                    "crown_tower_damage_percent": area_effect.get("CrownTowerDamagePercent"),
                    "source": "resolved_area_effect",
                },
            )
        )
    if area_effect.get("Buff"):
        result.append(
            MechanicOpV1(
                "OnHit",
                "CustomOp",
                target_selector="configured_area",
                parameters={
                    "buff": area_effect["Buff"],
                    "duration_raw": area_effect.get("BuffTime"),
                    "source": "resolved_area_effect",
                },
                custom_op="ApplyBuff",
            )
        )
    if projectile.get("Damage") not in (None, 0):
        result.append(
            MechanicOpV1(
                "OnHit",
                "Damage",
                target_selector="projectile_target",
                parameters={
                    "raw": projectile["Damage"],
                    "radius_raw": projectile.get("Radius"),
                    "crown_tower_damage_percent": projectile.get("CrownTowerDamagePercent"),
                },
            )
        )
    if projectile.get("Heal") not in (None, 0):
        result.append(
            MechanicOpV1("OnHit", "Heal", target_selector="projectile_target", parameters={"raw": projectile["Heal"]})
        )
    if unit.get("DeathSpawnCharacter"):
        result.append(
            MechanicOpV1(
                "OnDeath",
                "Spawn",
                parameters={"form": unit["DeathSpawnCharacter"], "count": unit.get("DeathSpawnCount", 1)},
            )
        )
    if unit.get("DeathDamage") not in (None, 0):
        result.append(
            MechanicOpV1(
                "OnDeath",
                "Damage",
                parameters={"raw": unit["DeathDamage"], "radius_raw": unit.get("DeathDamageRadius")},
            )
        )
    if unit.get("MorphCharacter"):
        result.append(
            MechanicOpV1(
                "OnMorphCondition",
                "Transform",
                target_selector="self",
                parameters={
                    "form": unit["MorphCharacter"],
                    "time_raw": unit.get("MorphTime"),
                    "after_hits": unit.get("MorphAfterHitsCount"),
                },
            )
        )
    for trigger, buff_key, time_key in (
        ("OnSpawn", "StartingBuff", "StartingBuffTime"),
        ("OnDamage", "BuffOnDamage", "BuffOnDamageTime"),
        ("OnKill", "BuffOnKill", "BuffOnKillTime"),
        ("OnHalfHitpoints", "BuffOn50HP", "BuffOn50HPTime"),
        ("OnHitCount", "BuffAfterHits", "BuffAfterHitsTime"),
    ):
        if unit.get(buff_key):
            result.append(
                MechanicOpV1(
                    trigger,
                    "CustomOp",
                    target_selector="self",
                    parameters={
                        "buff": unit[buff_key],
                        "duration_raw": unit.get(time_key),
                        "hit_count": unit.get("BuffAfterHitsCount") if buff_key == "BuffAfterHits" else None,
                    },
                    custom_op="ApplyBuff",
                )
            )
    if any(
        unit.get(key) not in (None, False, 0, "")
        for key in (
            "HidesWhenNotAttacking",
            "HideTimeMs",
            "HideBeforeFirstHit",
            "UntargetableWhenSpawned",
            "AllowAreaDmgWhenInvisible",
        )
    ):
        result.append(
            MechanicOpV1(
                "OnVisibilityTransition",
                "CustomOp",
                target_selector="self",
                parameters={
                    key: unit.get(key)
                    for key in (
                        "HidesWhenNotAttacking",
                        "HideTimeMs",
                        "HideBeforeFirstHit",
                        "UntargetableWhenSpawned",
                        "AllowAreaDmgWhenInvisible",
                        "UpTimeMs",
                        "SpecialAttackWhenHidden",
                    )
                    if unit.get(key) is not None
                },
                custom_op="VisibilityState",
            )
        )
    if unit.get("OnAttackAction"):
        result.append(
            MechanicOpV1(
                "OnAttack",
                "CustomOp",
                target_selector="configured_target",
                parameters={"action": unit["OnAttackAction"]},
                custom_op=str(unit["OnAttackAction"]),
            )
        )
    if row.get("Effect") and not summons:
        result.append(
            MechanicOpV1("OnDeploy", "CustomOp", custom_op=str(row["Effect"]), parameters={"source_field": "Effect"})
        )
    return tuple(result)


_EVOLUTION_GAMEPLAY_FIELDS = (
    "Hitpoints",
    "Damage",
    "HitSpeed",
    "Range",
    "MinimumRange",
    "Speed",
    "ShieldHitpoints",
    "AreaDamageRadius",
    "DeathDamage",
    "DeathDamageRadius",
    "AttackPushBack",
    "SpawnCharacter",
    "SpawnInterval",
    "SpawnNumber",
    "StartingBuff",
    "StartingBuffTime",
    "MorphCharacter",
    "MorphAfterHitsCount",
    "HidesWhenNotAttacking",
    "HideTimeMs",
    "UntargetableWhenSpawned",
)


def _evolution_effects(
    evolution_form_id: str,
    evolved_spell: Mapping[str, Any],
    base_unit: Mapping[str, Any],
    evolved_unit: Mapping[str, Any],
    evolved_projectile: Mapping[str, Any],
) -> tuple[MechanicOpV1, ...]:
    """Describe only explicit evolution records; never infer hidden behaviour."""

    overrides = {
        key: {"base": base_unit.get(key), "evolved": evolved_unit.get(key)}
        for key in _EVOLUTION_GAMEPLAY_FIELDS
        if evolved_unit.get(key) is not None and evolved_unit.get(key) != base_unit.get(key)
    }
    result: list[MechanicOpV1] = [
        MechanicOpV1(
            "OnEvolutionDeploy",
            "Transform",
            target_selector="source_card",
            parameters={
                "spell_form": evolution_form_id,
                "summoned_form": evolved_spell.get("SummonCharacter"),
                "explicit_unit_overrides": overrides,
            },
        )
    ]
    # Preserve explicit special mechanics in the evolved form.  The generic
    # Spawn entry is represented by the Transform above and is not duplicated.
    result.extend(
        item for item in _mechanics(evolved_spell, evolved_unit, evolved_projectile) if item.effect != "Spawn"
    )
    return tuple(result)


def _ability_effects(definition: Mapping[str, Any]) -> tuple[MechanicOpV1, ...]:
    effects: list[MechanicOpV1] = []
    if definition.get("ActivationSpawnCharacter"):
        effects.append(MechanicOpV1("OnAbility", "Spawn", parameters={"form": definition["ActivationSpawnCharacter"]}))
    if definition.get("MorphTarget"):
        effects.append(MechanicOpV1("OnAbility", "Transform", parameters={"form": definition["MorphTarget"]}))
    if definition.get("AreaEffectObject"):
        effects.append(MechanicOpV1("OnAbility", "CreateArea", parameters={"object": definition["AreaEffectObject"]}))
    if definition.get("Buff"):
        effects.append(
            MechanicOpV1(
                "OnAbility",
                "CustomOp",
                custom_op="ApplyBuff",
                parameters={"buff": definition["Buff"], "duration_raw": definition.get("BuffTime")},
            )
        )
    if definition.get("OnActivationAction"):
        effects.append(
            MechanicOpV1(
                "OnAbility",
                "CustomOp",
                custom_op=str(definition["OnActivationAction"]),
                parameters={"source_field": "OnActivationAction"},
            )
        )
    return tuple(effects)


def _ability_from_sources(
    card_id: int,
    ability: str | None,
    row: Mapping[str, Any],
    definitions: Mapping[str, Mapping[str, Any]],
    source_records: tuple[str, ...],
) -> AbilitySpecV1 | None:
    if not isinstance(ability, str) or not ability:
        return None
    definition = definitions.get(ability, {})
    cooldown = _milliseconds(definition.get("Cooldown", row.get("AbilityCooldown")))
    mana_cost = _number(definition, "ManaCost")
    cast_time = _milliseconds(_number(definition, "CastTime"))
    max_charges = _number(definition, "MaxCharges")
    effects = _ability_effects(definition)
    unknown: list[str] = []
    if mana_cost is None:
        unknown.append("elixir_cost")
    # A single-charge ability is complete without a cooldown: after the sole
    # activation there is no next use to schedule.  The August ruleset removes
    # Cooldown from those definitions while retaining MaxCharges=1.
    if cooldown is None and max_charges != 1:
        unknown.append("cooldown_ms")
    if cast_time is None:
        unknown.append("cast_time_ms")
    if not effects:
        unknown.append("effect_graph")
    # Native v15 Champion activation is the verified ct=2 command, whose wire
    # payload contains only owner and source entity.  DashRange is therefore
    # an engine-side target-search parameter (Golden Knight), not a policy
    # target supplied by the caller.
    target_kinds = (TargetKind.NONE,)
    return AbilitySpecV1(
        ability_id=ability,
        source_card_id=card_id,
        elixir_cost=float(mana_cost) if mana_cost is not None else None,
        cooldown_ms=cooldown,
        charges=int(max_charges) if max_charges is not None else None,
        cast_time_ms=cast_time,
        target_schema=TargetSchemaV1(target_kinds),
        effect_graph=effects,
        activation_condition=str(row.get("PassiveAbility")) if row.get("PassiveAbility") else None,
        attributes={
            "card": {
                key: value for key, value in row.items() if key in {"HeroAbility", "PassiveAbility", "AbilityCooldown"}
            },
            "definition": definition,
        },
        source_records=source_records,
        unknown_fields=tuple(unknown),
    )


def _resolved_unit_row(
    card_row: Mapping[str, Any], characters: Mapping[str, Mapping[str, Any]], buildings: Mapping[str, Mapping[str, Any]]
) -> Mapping[str, Any]:
    name = card_row.get("SummonCharacter")
    if not isinstance(name, str):
        return {}
    units = {**buildings, **characters}
    current = units.get(name, {})
    seen = {name}
    while isinstance(current, Mapping):
        target = current.get("SpawnPathfindMorph")
        if not isinstance(target, str) or not target or target in seen:
            break
        next_record = units.get(target)
        if not isinstance(next_record, Mapping) or not next_record:
            break
        seen.add(target)
        current = next_record
    return current


def _resolved_projectile_row(
    card_row: Mapping[str, Any], unit: Mapping[str, Any], projectiles: Mapping[str, Mapping[str, Any]]
) -> Mapping[str, Any]:
    names = tuple(
        dict.fromkeys(
            str(name)
            for name in (card_row.get("Projectile"), unit.get("CustomFirstProjectile"), unit.get("Projectile"))
            if isinstance(name, str) and name
        )
    )
    candidates = tuple(projectiles[name] for name in names if name in projectiles)
    if not candidates:
        return {}
    damaging = tuple(candidate for candidate in candidates if _number(candidate, "Damage", "Heal") is not None)
    # A decorative/default projectile can coexist with one explicit damaging
    # first projectile (Princess is the canonical decoded example). Select the
    # unique gameplay-bearing definition; multiple gameplay projectiles remain
    # a compound graph concern rather than a name-based choice.
    return damaging[0] if len(damaging) == 1 else candidates[0]


@dataclass(frozen=True, slots=True)
class _DeployFormResolution:
    forms: tuple[str, ...] = ()
    count: int | None = None
    count_projection_status: str | None = None
    evidence: tuple[str, ...] = ()
    definition_keys: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class _HeroFormResolution:
    form_id: str
    ability_id: str
    form: Mapping[str, Any]
    carrier_name: str
    carrier: Mapping[str, Any]
    definition_keys: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class _DirectHeroResolution:
    ability_id: str
    carrier_name: str
    carrier: Mapping[str, Any]
    ability_definition: Mapping[str, Any]
    definition_keys: tuple[tuple[str, str], ...]


def _declared_card_form_names(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Normalize the scalar-or-array EvolvedSpells field without retyping it."""

    value = row.get("EvolvedSpells")
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, (list, tuple)):
        return tuple(dict.fromkeys(item for item in value if isinstance(item, str) and item))
    return ()


def _resolve_normal_mode_hero_form(
    card_id: int,
    card_name: str,
    definitions: Mapping[str, Mapping[str, Mapping[str, Any]]],
    units: Mapping[str, Mapping[str, Any]],
) -> _HeroFormResolution | None:
    binding = NORMAL_MODE_HERO_FORM_BINDINGS_BY_CARD_ID.get(card_id)
    if binding is None:
        return None
    if binding.source_card_name != card_name:
        # Synthetic/older source tables can reuse physical row IDs.  Exact
        # identity is required for promotion, but unrelated fixtures must still
        # build as their own catalog rather than inherit a current form.
        return None
    form = definitions["SPELL_HERO"].get(binding.hero_form_id, {})
    if not form or form.get("CardForm") != "HeroForm":
        raise ContractError(f"Hero form {binding.hero_form_id} lacks an exact SPELL_HERO record")
    ability_definition = definitions["ABILITY"].get(binding.ability_id, {})
    if not ability_definition:
        raise ContractError(f"Hero form {binding.hero_form_id} lacks ability {binding.ability_id}")

    candidate_names = tuple(
        dict.fromkeys(
            (
                *_declared_summoned_forms(form),
                *(
                    (str(form["LinkedChampionCharacter"]),)
                    if isinstance(form.get("LinkedChampionCharacter"), str) and form.get("LinkedChampionCharacter")
                    else ()
                ),
            )
        )
    )
    carriers = tuple(
        (name, units.get(name, {}))
        for name in candidate_names
        if units.get(name, {}).get("Ability") == binding.ability_id
    )
    if len(carriers) != 1:
        raise ContractError(
            f"Hero form {binding.hero_form_id} expected exactly one "
            f"{binding.ability_id} carrier, got {[name for name, _ in carriers]}"
        )
    carrier_name, carrier = carriers[0]
    definition_keys: list[tuple[str, str]] = [("SPELL_HERO", binding.hero_form_id), ("ABILITY", binding.ability_id)]
    for namespace in ("CHARACTER", "BUILDING", "EXT"):
        if carrier_name in definitions[namespace]:
            definition_keys.append((namespace, carrier_name))
    return _HeroFormResolution(
        form_id=binding.hero_form_id,
        ability_id=binding.ability_id,
        form=form,
        carrier_name=carrier_name,
        carrier=carrier,
        definition_keys=tuple(dict.fromkeys(definition_keys)),
    )


def _resolve_normal_mode_direct_hero(
    card_id: int,
    card_name: str,
    resolved_summoned_forms: tuple[tuple[str, Mapping[str, Any]], ...],
    definitions: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> _DirectHeroResolution | None:
    """Resolve one verified player-triggered Champion ability carrier."""

    binding = NORMAL_MODE_DIRECT_HERO_BINDINGS_BY_CARD_ID.get(card_id)
    if binding is None:
        return None
    if binding.source_card_name != card_name:
        # Preserve synthetic/older fixtures that reuse a physical row ID.
        return None
    carriers = tuple(
        (name, definition)
        for name, definition in resolved_summoned_forms
        if name == binding.carrier_name and definition.get("Ability") == binding.ability_id
    )
    if len(carriers) != 1:
        raise ContractError(
            f"direct Hero {card_id} expected exactly one "
            f"{binding.ability_id} carrier, got {[name for name, _ in carriers]}"
        )
    ability_definition = definitions["ABILITY"].get(binding.ability_id, {})
    if not ability_definition:
        raise ContractError(f"direct Hero {card_id} lacks ability {binding.ability_id}")
    if ability_definition.get("IsChampion") is False:
        raise ContractError(f"direct Hero {card_id} ability is explicitly not a Champion action")
    carrier_name, carrier = carriers[0]
    definition_keys: list[tuple[str, str]] = [("ABILITY", binding.ability_id)]
    for namespace in ("CHARACTER", "BUILDING", "EXT"):
        if carrier_name in definitions[namespace]:
            definition_keys.append((namespace, carrier_name))
    return _DirectHeroResolution(
        ability_id=binding.ability_id,
        carrier_name=carrier_name,
        carrier=carrier,
        ability_definition=ability_definition,
        definition_keys=tuple(dict.fromkeys(definition_keys)),
    )


def _declared_summoned_forms(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(row[key])
        for key in ("SummonCharacter", "SummonCharacterSecond")
        if isinstance(row.get(key), str) and row.get(key)
    ) + tuple(str(item) for item in row.get("SummonCharactersList", ()) if isinstance(item, str) and item)


def _iter_action_references(value: Any) -> tuple[Any, ...]:
    if isinstance(value, (str, Mapping)):
        return (value,)
    if isinstance(value, (tuple, list)):
        return tuple(item for item in value if isinstance(item, (str, Mapping)))
    return ()


def _resolve_deploy_forms_from_actions(
    row: Mapping[str, Any], definitions: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> _DeployFormResolution:
    """Resolve only a card's explicit deploy-AEO/action chain.

    This is deliberately a fallback for entity cards whose root row has no
    direct SummonCharacter field.  It follows typed native references and
    never chooses a definition by card name.  Periodic/conditional spawn
    schedules retain their raw graph and fail closed on a single count.
    """

    if _declared_summoned_forms(row):
        return _DeployFormResolution()
    root_aeo = row.get("AreaEffectObject")
    if not isinstance(root_aeo, str) or not root_aeo:
        return _DeployFormResolution()

    forms: list[str] = []
    evidence: list[str] = [f"root.AreaEffectObject={root_aeo}"]
    definition_keys: list[tuple[str, str]] = []
    exact_count = True
    active_actions: set[str] = set()
    active_aeos: set[str] = set()

    def add_form(name: Any, source: str) -> None:
        if isinstance(name, str) and name:
            forms.append(name)
            evidence.append(f"{source}={name}")

    def visit_action(value: Any, source: str, *, conditional: bool = False) -> None:
        nonlocal exact_count
        if conditional:
            exact_count = False
        if isinstance(value, str):
            if not value or value in active_actions:
                return
            record = definitions.get("ACTION", {}).get(value)
            if not isinstance(record, Mapping) or not record:
                exact_count = False
                evidence.append(f"{source}:unresolved_action={value}")
                return
            definition_keys.append(("ACTION", value))
            active_actions.add(value)
            visit_action(record, f"ACTION.{value}", conditional=conditional)
            active_actions.remove(value)
            return
        if not isinstance(value, Mapping):
            return

        spawn_data = value.get("SpawnData")
        spawn_type = value.get("SpawnType")
        if isinstance(spawn_data, str) and isinstance(spawn_type, str):
            if spawn_type in {"CharacterType", "BuildingType"}:
                add_form(spawn_data, f"{source}.SpawnData")
            elif spawn_type == "AreaEffectType":
                visit_aeo(spawn_data, f"{source}.SpawnData")

        for field_name in _ACTION_REFERENCE_FIELDS:
            if field_name not in value:
                continue
            branch = field_name in {"OnTrueAction", "OnFalseAction"}
            for item in _iter_action_references(value[field_name]):
                visit_action(item, f"{source}.{field_name}", conditional=conditional or branch)

    def visit_aeo(name: str, source: str) -> None:
        nonlocal exact_count
        if not name or name in active_aeos:
            return
        record = definitions.get("AEO", {}).get(name)
        if not isinstance(record, Mapping) or not record:
            exact_count = False
            evidence.append(f"{source}:unresolved_aeo={name}")
            return
        definition_keys.append(("AEO", name))
        active_aeos.add(name)
        spawn_character = record.get("SpawnCharacter")
        if isinstance(spawn_character, str) and spawn_character:
            add_form(spawn_character, f"AEO.{name}.SpawnCharacter")
            if record.get("SpawnInterval") is not None:
                exact_count = False
                evidence.append(f"AEO.{name}.SpawnInterval={record['SpawnInterval']}:periodic")
        for item in _iter_action_references(record.get("OnStartingAction")):
            visit_action(item, f"AEO.{name}.OnStartingAction")
        active_aeos.remove(name)

    visit_aeo(root_aeo, "root.AreaEffectObject")
    if not forms:
        return _DeployFormResolution(evidence=tuple(evidence), definition_keys=tuple(dict.fromkeys(definition_keys)))
    return _DeployFormResolution(
        forms=tuple(forms),
        count=len(forms) if exact_count else None,
        count_projection_status=None if exact_count else RAW_STATIC_ONLY,
        evidence=tuple(evidence),
        definition_keys=tuple(dict.fromkeys(definition_keys)),
    )


def _build_spec(
    card_id: int,
    kind: CardKind,
    row: Mapping[str, Any],
    unit: Mapping[str, Any],
    projectile: Mapping[str, Any],
    evolved_form: Mapping[str, Any],
    evolved_unit: Mapping[str, Any],
    evolved_projectile: Mapping[str, Any],
    area_effect: Mapping[str, Any],
    ability_definitions: Mapping[str, Mapping[str, Any]],
    source_record: str,
    definition_records: tuple[str, ...],
    *,
    use_native_character_int_defaults: bool,
    policy_ability_id: str | None = None,
    evolution_form_id: str | None = None,
    hero_form: _HeroFormResolution | None = None,
    direct_hero: _DirectHeroResolution | None = None,
    resolved_summoned_forms: tuple[tuple[str, Mapping[str, Any]], ...] = (),
    deploy_form_resolution: _DeployFormResolution = _DeployFormResolution(),
) -> tuple[CardSpecV1, AbilitySpecV1 | None]:
    elixir_cost = _number(row, "ManaCost", "DarkElixirCost")
    hitpoints = _number(unit, "Hitpoints")
    if (
        hitpoints is None
        and use_native_character_int_defaults
        and unit
        and kind in {CardKind.TROOP, CardKind.BUILDING, CardKind.HERO}
    ):
        hitpoints = _NATIVE_CHARACTER_INT_DEFAULTS["Hitpoints"]
    damage = _number(unit, "Damage")
    if damage is None:
        damage = _number(row, "InstantDamage")
    if damage is None:
        damage = _number(projectile, "Damage")
    if damage is None:
        damage = _number(area_effect, "Damage")
    direct_attack_declared = any(
        unit.get(key) not in (None, False, "")
        for key in ("Damage", "DamageSpecial", "Projectile", "CustomFirstProjectile", "AttackSequenceList")
    )
    hit_speed = _milliseconds(_number(unit, "HitSpeed")) if direct_attack_declared else None
    range_tiles = _game_units(_number(unit, "Range")) if direct_attack_declared else None
    move_speed = _number(unit, "Speed")
    if (
        move_speed is None
        and use_native_character_int_defaults
        and unit
        and kind in {CardKind.TROOP, CardKind.HERO}
        and not bool(unit.get("IsBuilding"))
    ):
        move_speed = _NATIVE_CHARACTER_INT_DEFAULTS["Speed"]
    deploy_time = _milliseconds(_number(row, "CustomDeployTime"))
    if deploy_time is None:
        deploy_time = _milliseconds(_number(unit, "DeployTime", "DeployDelay"))
    direct_summons = tuple(
        (name_key, count_key)
        for name_key, count_key in (
            ("SummonCharacter", "SummonNumber"),
            ("SummonCharacterSecond", "SummonCharacterSecondCount"),
        )
        if isinstance(row.get(name_key), str) and bool(row.get(name_key))
    )
    listed_summons = tuple(str(item) for item in row.get("SummonCharactersList", ()) if isinstance(item, str) and item)
    # Each explicit deploy-form reference has an implicit count of one.  A
    # two-form deployment is still exactly aggregatable even though its combat
    # stats are compound (Goblin Gang, Rascals, Goblinstein).
    count = (
        sum(int(_number(row, count_key) or 1) for _, count_key in direct_summons)
        if direct_summons
        else (len(listed_summons) if listed_summons else deploy_form_resolution.count)
    )
    radius_tiles = _game_units(_number(row, "Radius"))
    if radius_tiles is None:
        radius_tiles = _game_units(_number(projectile, "Radius"))
    if radius_tiles is None:
        radius_tiles = _game_units(_number(unit, "AreaDamageRadius"))
    if radius_tiles is None:
        radius_tiles = _game_units(_number(area_effect, "Radius"))
    duration = _milliseconds(_number(row, "DurationSeconds"), seconds=True)
    if duration is None:
        duration = _milliseconds(_number(unit, "LifeTime", "UpTimeMs"))
    shield = _number(unit, "ShieldHitpoints")
    projectile_speed = _number(projectile, "Speed")
    knockback = _number(row, "Pushback")
    if knockback is None:
        knockback = _number(projectile, "Pushback")
    if knockback is None:
        knockback = _number(unit, "AttackPushBack")
    summoned = _declared_summoned_forms(row) or deploy_form_resolution.forms
    evolution = None
    if evolution_form_id is not None:
        cycle_value = _number(evolved_form, "DarkElixirCost")
        evolution = EvolutionSpecV1(
            base_form_id=str(row["Name"]),
            evolution_form_id=evolution_form_id,
            cycle_required=int(cycle_value) if cycle_value is not None else None,
            effects=_evolution_effects(evolution_form_id, evolved_form, unit, evolved_unit, evolved_projectile),
        )
    ability = _ability_from_sources(card_id, policy_ability_id, row, ability_definitions, definition_records)
    standard = {
        "elixir_cost": elixir_cost,
        "hitpoints": hitpoints,
        "damage": damage,
        "hit_speed_ms": hit_speed,
        "range_tiles": range_tiles,
        "move_speed": move_speed,
        "deploy_time_ms": deploy_time,
        "count": count,
        "radius_tiles": radius_tiles,
        "projectile_speed": projectile_speed,
        "knockback": knockback,
        "duration_ms": duration,
        "shield_hitpoints": shield,
    }
    field_applicability = _field_applicability(
        kind=kind,
        row=row,
        unit=unit,
        projectile=projectile,
        area_effect=area_effect,
        standard=standard,
        summoned_names=summoned,
        count_projection_status=deploy_form_resolution.count_projection_status,
        count_projection_evidence=deploy_form_resolution.evidence,
    )
    unknown = [
        name
        for name, applicability in field_applicability.items()
        if applicability["status"] == SemanticEvidenceLevel.UNKNOWN.value
    ]
    resolved_form_names = {name for name, definition in resolved_summoned_forms if definition}
    if kind in {CardKind.TROOP, CardKind.BUILDING, CardKind.HERO} and (
        not summoned or any(name not in resolved_form_names for name in summoned)
    ):
        unknown.append("summoned_form_definition")
    categorical = {
        "rarity": row.get("Rarity"),
        "tribe": row.get("Tribe"),
        "unlock_arena": row.get("UnlockArena"),
        "attacks_ground": (unit.get("AttacksGround") if direct_attack_declared else None),
        "attacks_air": unit.get("AttacksAir") if direct_attack_declared else None,
        "target_only_buildings": (unit.get("TargetOnlyBuildings") if direct_attack_declared else None),
        "field_applicability_version": FIELD_APPLICABILITY_VERSION,
        "field_applicability": field_applicability,
        "not_applicable": tuple(
            sorted(
                name
                for name, applicability in field_applicability.items()
                if applicability["status"] == SemanticEvidenceLevel.NOT_APPLICABLE.value
            )
        ),
        "compound_declared": tuple(
            sorted(
                name
                for name, applicability in field_applicability.items()
                if applicability["status"] == COMPOUND_DECLARED
            )
        ),
        "raw_static_only": tuple(
            sorted(
                name
                for name, applicability in field_applicability.items()
                if applicability["status"] == RAW_STATIC_ONLY
            )
        ),
    }
    attributes = dict(row)
    if unit:
        attributes["resolved_summoned_form"] = dict(unit)
    if resolved_summoned_forms:
        attributes["resolved_summoned_forms"] = tuple(
            {"name": name, "definition": dict(definition)} for name, definition in resolved_summoned_forms
        )
    if deploy_form_resolution.evidence:
        attributes["deploy_form_resolution"] = {
            "method": "typed_deploy_aeo_action_chain",
            "forms": deploy_form_resolution.forms,
            "count_projection_status": (
                deploy_form_resolution.count_projection_status or SemanticEvidenceLevel.STATIC_DECLARED.value
            ),
            "evidence": deploy_form_resolution.evidence,
        }
    if projectile:
        attributes["resolved_projectile"] = dict(projectile)
    if area_effect:
        attributes["resolved_area_effect"] = dict(area_effect)
    if evolved_form:
        attributes["resolved_evolution_form"] = dict(evolved_form)
    if evolved_unit:
        attributes["resolved_evolution_unit"] = dict(evolved_unit)
    if hero_form is not None:
        attributes["resolved_hero_form"] = {
            "form_id": hero_form.form_id,
            "definition": dict(hero_form.form),
            "ability_id": hero_form.ability_id,
            "ability_carrier": hero_form.carrier_name,
            "ability_carrier_definition": dict(hero_form.carrier),
        }
    if direct_hero is not None:
        attributes["resolved_direct_hero"] = {
            "ability_id": direct_hero.ability_id,
            "ability_carrier": direct_hero.carrier_name,
            "ability_carrier_definition": dict(direct_hero.carrier),
            "ability_definition": dict(direct_hero.ability_definition),
        }
    mechanics = list(_mechanics(row, unit, projectile, area_effect))
    if deploy_form_resolution.forms:
        form_parameters: list[dict[str, Any]] = []
        for name in deploy_form_resolution.forms:
            item: dict[str, Any] = {"name": name}
            if deploy_form_resolution.count is not None:
                item["count"] = 1
            form_parameters.append(item)
        mechanics.insert(
            0,
            MechanicOpV1(
                trigger="OnDeploy",
                effect="Spawn",
                target_selector="deployment_area",
                parameters={
                    "forms": form_parameters,
                    "resolution": "typed_deploy_aeo_action_chain",
                    "count_projection_status": (
                        deploy_form_resolution.count_projection_status or SemanticEvidenceLevel.STATIC_DECLARED.value
                    ),
                },
            ),
        )
    spec = CardSpecV1(
        card_id=card_id,
        name=str(row["Name"]),
        kind=kind,
        elixir_cost=float(elixir_cost) if elixir_cost is not None else None,
        hitpoints=float(hitpoints) if hitpoints is not None else None,
        damage=float(damage) if damage is not None else None,
        hit_speed_ms=hit_speed,
        range_tiles=range_tiles,
        move_speed=float(move_speed) if move_speed is not None else None,
        deploy_time_ms=deploy_time,
        count=count,
        radius_tiles=radius_tiles,
        projectile_speed=projectile_speed,
        formation=str(row.get("TypeOfSpell")) if row.get("TypeOfSpell") else None,
        knockback=float(knockback) if knockback is not None else None,
        duration_ms=duration,
        shield_hitpoints=float(shield) if shield is not None else None,
        target_schema=_target_schema(card_id, kind, row),
        mechanics=tuple(mechanics),
        summoned_forms=summoned,
        evolution=evolution,
        ability_ids=(ability.ability_id,) if ability is not None else (),
        numeric_features=standard,
        categorical_features=categorical,
        attributes=attributes,
        source_records=(source_record, *definition_records),
        unknown_fields=tuple(unknown),
    )
    return spec, ability


@lru_cache(maxsize=16)
def _cached_specs_hash(specs: tuple[CardSpecV1, ...]) -> str:
    return content_hash([item.to_dict() for item in specs])


@lru_cache(maxsize=16)
def _cached_ability_specs_hash(abilities: tuple[AbilitySpecV1, ...]) -> str:
    return content_hash([item.to_dict() for item in abilities])


@dataclass(frozen=True, slots=True)
class CardSpecCatalog(ContractMixin):
    specs: tuple[CardSpecV1, ...]
    abilities: tuple[AbilitySpecV1, ...] = ()
    source_release: str = "unknown"
    source_files: Mapping[str, str] = field(default_factory=FrozenMapping)
    warnings: tuple[str, ...] = ()
    version: str = field(default=CATALOG_VERSION, init=False)
    VERSION = CATALOG_VERSION

    def __post_init__(self) -> None:
        normalized_specs = tuple(
            item if isinstance(item, CardSpecV1) else CardSpecV1.from_mapping(item) for item in self.specs
        )
        normalized_specs = tuple(sorted(normalized_specs, key=lambda item: item.card_id))
        if len({item.card_id for item in normalized_specs}) != len(normalized_specs):
            raise ContractError("card catalog contains duplicate card IDs")
        object.__setattr__(self, "specs", normalized_specs)
        object.__setattr__(
            self,
            "abilities",
            tuple(
                sorted(
                    (
                        item if isinstance(item, AbilitySpecV1) else AbilitySpecV1.from_mapping(item)
                        for item in self.abilities
                    ),
                    key=lambda item: item.ability_id,
                )
            ),
        )
        object.__setattr__(self, "source_files", frozen_mapping(self.source_files))
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))

    @property
    def by_id(self) -> dict[int, CardSpecV1]:
        return {item.card_id: item for item in self.specs}

    @property
    def specs_hash(self) -> str:
        return _cached_specs_hash(self.specs)

    @property
    def ability_specs_hash(self) -> str:
        return _cached_ability_specs_hash(self.abilities)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CardSpecCatalog":
        if value.get("version") != CATALOG_VERSION:
            raise ContractError(f"unsupported card catalog version: {value.get('version')}")
        return cls(
            specs=tuple(CardSpecV1.from_mapping(item) for item in value.get("specs", ())),
            abilities=tuple(AbilitySpecV1.from_mapping(item) for item in value.get("abilities", ())),
            source_release=str(value.get("source_release", "unknown")),
            source_files=value.get("source_files", {}),
            warnings=tuple(value.get("warnings", ())),
        )

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_json(pretty=True) + "\n"
        if destination.exists():
            existing = destination.read_text(encoding="utf-8")
            if existing != payload:
                raise FileExistsError(f"immutable card catalog path already contains different bytes: {destination}")
            return destination
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "CardSpecCatalog":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_mapping(value)


def build_card_catalog(
    workspace_root: str | Path | None = None, *, source_roots: Iterable[str | Path] = ()
) -> CardSpecCatalog:
    workspace = Path(workspace_root or WORKSPACE_ROOT).resolve()
    source = discover_card_source(workspace, source_roots)
    runtime_logic = workspace / "runtime-update" / "csv_logic"
    definitions, provenance = _load_definition_fragments(source, replacement_root=runtime_logic)
    if runtime_logic.is_dir():
        runtime_definitions, runtime_provenance = _load_definition_fragments(runtime_logic)
        for group, records in runtime_definitions.items():
            for name, record in records.items():
                definitions[group][name] = _deep_merge(definitions[group].get(name, {}), record)
        for key, paths in runtime_provenance.items():
            provenance[key] = (*provenance.get(key, ()), *paths)

    runtime_ability_table = runtime_logic / "character_abilities.csv"
    ability_table = (
        runtime_ability_table if _is_plaintext(runtime_ability_table) else source / "character_abilities.csv"
    )
    ability_rows = _load_named_table(ability_table)
    for name, row in ability_rows.items():
        definitions["ABILITY"][name] = _deep_merge(row, definitions["ABILITY"].get(name, {}))
        provenance[("ABILITY", name)] = (*provenance.get(("ABILITY", name), ()), ability_table)

    # Area-effect rows are part of the same native AEO registry as TOML
    # fragments.  In particular, deployment spawners such as TriWizardSpawn
    # keep their periodic SpawnCharacter fields in this CSV while their action
    # graph lives in TOML.
    runtime_aeo_table = runtime_logic / "area_effect_objects.csv"
    aeo_table = runtime_aeo_table if _is_plaintext(runtime_aeo_table) else source / "area_effect_objects.csv"
    aeo_rows = _load_named_table(aeo_table)
    aeo_sources = (aeo_table,) if aeo_table.is_file() else ()
    for name, row in aeo_rows.items():
        definitions["AEO"][name] = _deep_merge(row, definitions["AEO"].get(name, {}))
        provenance[("AEO", name)] = (
            *provenance.get(("AEO", name), ()),
            *(path for path in aeo_sources if path.is_file()),
        )
    resolved_definitions = _resolve_definition_inheritance(definitions, provenance)

    runtime_evolved_table = runtime_logic / "spells_evolved.csv"
    evolved_table = runtime_evolved_table if _is_plaintext(runtime_evolved_table) else source / "spells_evolved.csv"
    evolved_rows = _load_named_table(evolved_table)
    evolved_sources: tuple[Path, ...] = (evolved_table,) if evolved_rows else ()

    auxiliary_tables = {
        filename: (runtime_logic / filename if _is_plaintext(runtime_logic / filename) else source / filename)
        for filename in ("characters.csv", "buildings.csv", "projectiles.csv")
    }
    character_rows = _load_named_table(auxiliary_tables["characters.csv"])
    building_rows = _load_named_table(auxiliary_tables["buildings.csv"])
    projectile_rows = _load_named_table(auxiliary_tables["projectiles.csv"])
    units: dict[str, dict[str, Any]] = {}
    for name in (
        set(character_rows)
        | set(building_rows)
        | set(resolved_definitions["CHARACTER"])
        | set(resolved_definitions["BUILDING"])
    ):
        units[name] = _deep_merge(
            _deep_merge(
                _deep_merge(character_rows.get(name, {}), building_rows.get(name, {})),
                resolved_definitions["CHARACTER"].get(name, {}),
            ),
            resolved_definitions["BUILDING"].get(name, {}),
        )
    for name, record in resolved_definitions["EXT"].items():
        raw_base = definitions["EXT"].get(name, {}).get("Base")
        base_namespace = raw_base.split(".", 1)[0] if isinstance(raw_base, str) and "." in raw_base else ""
        semantic_overlay = (
            base_namespace in definitions
            and name in definitions[base_namespace]
            and "Base" not in definitions[base_namespace][name]
            and "Name" not in definitions[base_namespace][name]
            and _same_definition_sources(provenance, ("EXT", name), (base_namespace, name))
        )
        if base_namespace in {"CHARACTER", "BUILDING"} and (name not in units or semantic_overlay):
            units[name] = dict(record)
    projectiles: dict[str, dict[str, Any]] = {}
    for name in set(projectile_rows) | set(resolved_definitions["PROJECTILE"]):
        projectiles[name] = _deep_merge(projectile_rows.get(name, {}), resolved_definitions["PROJECTILE"].get(name, {}))
    for name, record in resolved_definitions["EXT"].items():
        raw_base = definitions["EXT"].get(name, {}).get("Base")
        if isinstance(raw_base, str) and raw_base.startswith("PROJECTILE.") and name not in projectiles:
            projectiles[name] = dict(record)
    specs: list[CardSpecV1] = []
    abilities: list[AbilitySpecV1] = []
    source_files: dict[str, str] = {}
    warnings: list[str] = []

    try:
        source_release = next(part for part in reversed(source.parts) if part.startswith("nr_"))
    except StopIteration:
        source_release = source.parent.name
    use_native_character_int_defaults = _matches_native_character_default_build(workspace, source, source_release)
    if source_release == _NATIVE_CHARACTER_DEFAULT_RELEASE and not use_native_character_int_defaults:
        warnings.append(
            "build-specific LogicCharacterData integer defaults disabled: "
            "the bound ARM64 libg.so identity is unavailable or mismatched"
        )

    for filename, base_id, kind in CARD_TABLES:
        runtime_path = runtime_logic / filename
        path = runtime_path if _is_plaintext(runtime_path) else source / filename
        if not path.exists():
            continue
        relative = path.relative_to(workspace).as_posix() if path.is_relative_to(workspace) else path.as_posix()
        source_files[relative] = sha256_file(path)
        if not _is_plaintext(path):
            warnings.append(f"skipped compressed table: {relative}")
            continue
        _, rows = _read_sc_csv(path)
        for row_index, row in enumerate(rows):
            card_id = base_id + row_index
            root_namespace = _CARD_DEFINITION_NAMESPACE[filename]
            card_name = str(row["Name"])
            root_definition = resolved_definitions[root_namespace].get(card_name, {})
            # Runtime named rows remain authoritative.  TOML definitions fill
            # fields omitted by the compact CSV without overriding a live
            # non-empty table value.
            row = _deep_merge(root_definition, row)
            declared_form_names = _declared_card_form_names(row)
            evolution_names = tuple(name for name in declared_form_names if name in evolved_rows)
            if len(evolution_names) > 1:
                raise ContractError(f"card {card_id} declares multiple evolution forms: {evolution_names}")
            evolution_name = evolution_names[0] if evolution_names else None
            deploy_form_resolution = (
                _resolve_deploy_forms_from_actions(row, resolved_definitions)
                if kind in {CardKind.TROOP, CardKind.BUILDING, CardKind.HERO}
                else _DeployFormResolution()
            )
            summoned_form_names = _declared_summoned_forms(row) or deploy_form_resolution.forms
            resolved_summoned_forms = tuple((name, units.get(name, {})) for name in summoned_form_names)
            direct_unit = next((definition for _, definition in resolved_summoned_forms if definition), {})
            unit = _resolved_unit_row(row, units, {}) if len(summoned_form_names) == 1 else direct_unit
            projectile = _resolved_projectile_row(row, unit, projectiles)
            area_effect_name = row.get("AreaEffectObject")
            area_effect = (
                resolved_definitions["AEO"].get(area_effect_name, {}) if isinstance(area_effect_name, str) else {}
            )
            evolved_form = evolved_rows.get(evolution_name, {}) if isinstance(evolution_name, str) else {}
            evolved_unit = _resolved_unit_row(evolved_form, units, {})
            evolved_projectile = _resolved_projectile_row(evolved_form, evolved_unit, projectiles)
            referenced: list[tuple[str, str]] = []
            if root_definition:
                referenced.append((root_namespace, card_name))
            referenced.extend(deploy_form_resolution.definition_keys)
            for unit_name in summoned_form_names:
                if unit_name in definitions["CHARACTER"]:
                    referenced.append(("CHARACTER", unit_name))
                if unit_name in definitions["BUILDING"]:
                    referenced.append(("BUILDING", unit_name))
                if unit_name in definitions["EXT"]:
                    referenced.append(("EXT", unit_name))
            projectile_name = row.get("Projectile") or unit.get("Projectile")
            if isinstance(projectile_name, str):
                referenced.append(("PROJECTILE", projectile_name))
            if isinstance(area_effect_name, str) and area_effect:
                referenced.append(("AEO", area_effect_name))
            explicit_ability_name = row.get("HeroAbility") or row.get("Ability")
            direct_hero = _resolve_normal_mode_direct_hero(
                card_id, card_name, resolved_summoned_forms, resolved_definitions
            )
            hero_form = _resolve_normal_mode_hero_form(card_id, card_name, resolved_definitions, units)
            if hero_form is not None and direct_hero is not None:
                raise ContractError(f"card {card_id} resolves both Hero-form and direct abilities")
            if (
                direct_hero is not None
                and isinstance(explicit_ability_name, str)
                and explicit_ability_name != direct_hero.ability_id
            ):
                raise ContractError(f"direct Hero {card_id} ability identity mismatch")
            if hero_form is not None and isinstance(explicit_ability_name, str):
                raise ContractError(f"base card {card_id} has both direct and Hero-form abilities")
            ability_name = (
                hero_form.ability_id
                if hero_form is not None
                else (direct_hero.ability_id if direct_hero is not None else explicit_ability_name)
            )
            if hero_form is not None:
                row = _deep_merge(row, {"HeroAbility": hero_form.ability_id})
                referenced.extend(hero_form.definition_keys)
            if direct_hero is not None:
                row = _deep_merge(row, {"HeroAbility": direct_hero.ability_id})
                referenced.extend(direct_hero.definition_keys)
            if isinstance(ability_name, str):
                referenced.append(("ABILITY", ability_name))
            effective_kind = (
                CardKind.HERO
                if kind == CardKind.TROOP and (direct_hero is not None or isinstance(explicit_ability_name, str))
                else kind
            )
            definition_records: list[str] = []
            for key in referenced:
                for definition_path in provenance.get(key, ()):
                    definition_relative = (
                        definition_path.relative_to(workspace).as_posix()
                        if definition_path.is_relative_to(workspace)
                        else definition_path.as_posix()
                    )
                    definition_records.append(f"{definition_relative}#{key[0]}.{key[1]}")
            if isinstance(evolution_name, str):
                for evolution_path in evolved_sources:
                    evolution_relative = (
                        evolution_path.relative_to(workspace).as_posix()
                        if evolution_path.is_relative_to(workspace)
                        else evolution_path.as_posix()
                    )
                    definition_records.append(f"{evolution_relative}#EVOLUTION.{evolution_name}")
            spec, ability = _build_spec(
                card_id,
                effective_kind,
                row,
                unit,
                projectile,
                evolved_form,
                evolved_unit,
                evolved_projectile,
                area_effect,
                resolved_definitions["ABILITY"],
                f"{relative}#named-row={row_index}",
                tuple(dict.fromkeys(definition_records)),
                use_native_character_int_defaults=use_native_character_int_defaults,
                policy_ability_id=(ability_name if isinstance(ability_name, str) else None),
                evolution_form_id=evolution_name,
                hero_form=hero_form,
                direct_hero=direct_hero,
                resolved_summoned_forms=resolved_summoned_forms,
                deploy_form_resolution=deploy_form_resolution,
            )
            specs.append(spec)
            if ability is not None:
                abilities.append(ability)

    for auxiliary in (
        "characters.csv",
        "buildings.csv",
        "projectiles.csv",
        "character_abilities.csv",
        "spells_evolved.csv",
    ):
        runtime_path = runtime_logic / auxiliary
        path = runtime_path if _is_plaintext(runtime_path) else source / auxiliary
        if path.exists():
            relative = path.relative_to(workspace).as_posix() if path.is_relative_to(workspace) else path.as_posix()
            source_files[relative] = sha256_file(path)
            if not _is_plaintext(path):
                warnings.append(f"compressed auxiliary table left explicit unknowns: {relative}")

    for paths in provenance.values():
        for path in paths:
            relative = path.relative_to(workspace).as_posix() if path.is_relative_to(workspace) else path.as_posix()
            source_files[relative] = sha256_file(path)

    if not specs:
        raise ContractError(f"no named card rows loaded from {source}")
    if not definitions["CHARACTER"]:
        warnings.append(
            "current source has no decoded CHARACTER definitions; "
            "HP/damage/timing fields are intentionally null and listed in unknown_fields"
        )
    unresolved_core = sum(
        1
        for spec in specs
        if spec.kind in {CardKind.TROOP, CardKind.BUILDING, CardKind.HERO}
        and (spec.hitpoints is None or spec.damage is None)
    )
    if unresolved_core:
        warnings.append(
            f"{unresolved_core} troop/building/hero specs still lack explicit HP or damage in this asset pack"
        )
    return CardSpecCatalog(
        specs=tuple(specs),
        abilities=tuple(abilities),
        source_release=source_release,
        source_files=source_files,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def build_card_specs(
    workspace_root: str | Path | None = None, *, source_roots: Iterable[str | Path] = ()
) -> tuple[CardSpecV1, ...]:
    return build_card_catalog(workspace_root, source_roots=source_roots).specs


def build_ability_specs(
    workspace_root: str | Path | None = None, *, source_roots: Iterable[str | Path] = ()
) -> tuple[AbilitySpecV1, ...]:
    return build_card_catalog(workspace_root, source_roots=source_roots).abilities
