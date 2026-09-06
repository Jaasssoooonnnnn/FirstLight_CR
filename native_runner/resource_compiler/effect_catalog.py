"""Content-addressed static semantics for native runtime effect instances.

The native type-3 component exposes an effect asset name, unsigned 32-bit
global ID, remaining duration, and (when it can be validated) a source entity.
This module joins that identity to the lossless ``BUFF`` nodes in
:mod:`native_runner.card_logic`.

The exact v15.535.13 loader does *not* assign every BUFF by CSV order.  It first
uses explicit ``global_ids_oldformat.csv`` entries and otherwise computes
FNV-1a over the UTF-8 bytes of ``"Buff" + Name``.  Keeping that distinction is
important: 44 current hashed BUFF IDs set bit 31 and are valid uint32 values,
not invalid negative IDs.

Every source field is retained in ``raw_record``.  Normalized tags carry
field-level evidence and applicator records retain the ACTION/AEO/PROJECTILE
fields that own duration and reset-target parameters.  Names alone are never
used to distinguish slow, rage, freeze, stun, or reset/interrupt behaviour.
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

# At 0xe09eb8..0xe09ed4, the exact libg projectile loader passes ``w2=1``
# while reading AllowResetTarget and stores the result at projectile data
# +0x2a2.  Missing static fields therefore mean true, not unknown/false.
_ALLOW_RESET_TARGET_DEFAULT_EVIDENCE = (
    "libg.arm64-v15.535.13@0xe09eb8:field=AllowResetTarget",
    "libg.arm64-v15.535.13@0xe09ec4:boolean_default=1",
    "libg.arm64-v15.535.13@0xe09ed0:store=projectile_data+0x2a2",
)

_BUFF_APPLICATION_FIELDS: Mapping[str, str] = {
    "Buff": "area_or_ability_buff",
    "TargetBuff": "projectile_target_buff",
    "ValidTargetBuff": "target_validation_buff",
    "BuffOnDamage": "damage_triggered_buff",
    "ReflectedAttackBuff": "reflected_attack_buff",
    "SpawnData": "action_spawn_buff",
    "BuffDuringCapture": "capture_lifecycle_buff",
    "BuffWhenNotAttacking": "idle_lifecycle_buff",
    "BuffOnHit": "hit_triggered_buff",
    "BuffAfterHits": "hit_count_buff",
    "PendingBuff": "pending_ability_buff",
    "CrownTowerBuff": "crown_tower_buff",
    "BuffOverride": "conditional_buff_override",
    "StartingBuff": "spawn_lifecycle_buff",
    "ChainPhaseBuff": "chain_phase_buff",
}

_APPLICATION_PARAMETER_FIELDS = frozenset(
    {
        "AllowResetTarget",
        "BuffNumber",
        "BuffAfterHitsCount",
        "BuffAfterHitsTime",
        "BuffOnDamageTime",
        "BuffTime",
        "BuffTimeIncreasePerLevel",
        "CapBuffTimeToAreaEffectTime",
        "CaptureDragTime",
        "ClassType",
        "Duration",
        "HitFrequency",
        "HitSpeed",
        "LifeDuration",
        "ReflectedAttackBuffDuration",
        "SpawnTime",
        "SpawnType",
        "StatsTags",
        "ValidDuration",
    }
)


def _application_parameters(owner_record: Mapping[str, Any], field_name: str, field_path: str) -> dict[str, Any]:
    parameters = {key: owner_record[key] for key in sorted(_APPLICATION_PARAMETER_FIELDS.intersection(owner_record))}
    duration_field = {
        "BuffOnDamage": "BuffOnDamageTime",
        "ReflectedAttackBuff": "ReflectedAttackBuffDuration",
        "BuffDuringCapture": "CaptureDragTime",
    }.get(field_name)
    if duration_field is not None:
        duration = owner_record.get(duration_field)
        if isinstance(duration, (int, float)) and not isinstance(duration, bool):
            parameters["BuffTime"] = duration
    if field_name != "BuffAfterHits":
        return parameters
    leaf = _FIELD_PATH_PART.fullmatch(field_path.rsplit(".", 1)[-1])
    index = int(leaf.group(2)) if leaf is not None and leaf.group(2) else None
    for source_name, target_name in (("BuffAfterHitsCount", "Count"), ("BuffAfterHitsTime", "BuffTime")):
        value = owner_record.get(source_name)
        if index is not None and isinstance(value, (tuple, list)) and 0 <= index < len(value):
            value = value[index]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            parameters[target_name] = value
    return parameters


# These are semantic engine fields.  Presentation-only fields are still kept
# verbatim in raw_record, but are not duplicated into the compact modifier map.
_MODIFIER_FIELDS = frozenset(
    {
        "AllowedOverHealPerc",
        "AttractMaxAngle",
        "AttractMinAngle",
        "AttractPercentage",
        "AudioPitchModifier",
        "BuildingDamagePercent",
        "CrownTowerDamagePerHit",
        "CrownTowerDamagePercent",
        "DamageMultiplier",
        "DamagePerSecond",
        "DamageReduction",
        "EffectScale",
        "HealPerSecond",
        "HitFrequency",
        "HitSpeedMultiplier",
        "HitpointMultiplier",
        "LevelIncrease",
        "PushSpeedFactor",
        "Scale",
        "ShadowAlpha",
        "Shield",
        "SpawnSpeedMultiplier",
        "SpeedMultiplier",
    }
)

_LIFECYCLE_FIELDS = frozenset(
    {
        "AddAsIndividualBuff",
        "AliveIfTrue",
        "AttachedInheritAs",
        "Base",
        "ControlledByParent",
        "EnableStacking",
        "HitTickFromSource",
        "NotCloned",
        "OnDamageReductionAction",
        "OnRemoveAction",
        "OnStartAction",
        "OtherBuffDeathSpawnAllowed",
        "PlayerSpecificBuff",
        "RemoveOnAttack",
        "RemoveOnHit",
        "SpawnerAliveRequired",
    }
)

_TARGETING_FIELDS = frozenset(
    {"GameTagsToSet", "IgnoreBuildings", "IgnorePushBack", "LockTarget", "NoEffectToCrownTowers", "SwitchTeam"}
)

_SPAWN_FIELDS = frozenset(
    {
        "DeathSpawn",
        "DeathSpawnCount",
        "DeathSpawnDeployDelay",
        "DeathSpawnIsEnemy",
        "DeathSpawnSameLocation",
        "SpawnInterval",
        "SpawnLimit",
        "SpawnNumber",
        "SpawnObject",
    }
)

_OVERRIDE_FIELDS = frozenset({"Clone", "DeathEffectOverride", "OverrideChargeRange", "OverrideProjectile"})


def _numeric(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _fnv1a32(value: str) -> int:
    result = _FNV1A_OFFSET_BASIS
    for byte in value.encode("utf-8"):
        result = ((result ^ byte) * _FNV1A_PRIME) & _UINT32_MAX
    return result


def _effect_global_id_key(effect_name: str) -> str:
    return f"{_BUFF_GLOBAL_ID_PREFIX}{effect_name}"


def _sequence_matches(value: Any, effect_name: str) -> tuple[int | None, ...]:
    if value == effect_name:
        return (None,)
    if isinstance(value, (tuple, list)):
        return tuple(index for index, item in enumerate(value) if item == effect_name)
    return ()


_FIELD_PATH_PART = re.compile(r"([^.[\]]+)(?:\[(\d+)\])?")


def _field_parent(record: Mapping[str, Any], field_path: str) -> tuple[Mapping[str, Any], str] | None:
    """Return the mapping that owns the leaf field in a logic reference."""

    current: Any = record
    parts = field_path.split(".")
    for raw_part in parts[:-1]:
        match = _FIELD_PATH_PART.fullmatch(raw_part)
        if match is None or not isinstance(current, Mapping):
            return None
        field_name, index_text = match.groups()
        current = current.get(field_name)
        if index_text is not None:
            if not isinstance(current, (tuple, list)):
                return None
            index = int(index_text)
            if not 0 <= index < len(current):
                return None
            current = current[index]

    leaf = _FIELD_PATH_PART.fullmatch(parts[-1])
    if leaf is None or not isinstance(current, Mapping):
        return None
    return current, leaf.group(1)


def _node_source_records(node: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        sorted(
            str(fragment["source_record"])
            for fragment in node.get("fragments", ())
            if isinstance(fragment, Mapping) and fragment.get("source_record")
        )
    )


def _effect_target_node_ids(effect_name: str, card_logic: StaticCardLogicCatalogV1) -> frozenset[str]:
    """Resolve EXT aliases that are proved to inherit from one BUFF node."""

    targets = {f"BUFF.{effect_name}"}
    changed = True
    while changed:
        changed = False
        for reference in card_logic.references:
            if (
                reference.get("reference_class") != "inheritance"
                or reference.get("field_path") != "Base"
                or not str(reference.get("source_node_id", "")).startswith("EXT.")
                or not targets.intersection(reference.get("target_node_ids", ()))
            ):
                continue
            source_node_id = str(reference["source_node_id"])
            if source_node_id not in targets:
                targets.add(source_node_id)
                changed = True
    return frozenset(targets)


def _trigger_sources(target_node_id: str, card_logic: StaticCardLogicCatalogV1) -> tuple[Mapping[str, Any], ...]:
    """Retain immediate gameplay edges that invoke an applicator action."""

    triggers: list[dict[str, Any]] = []
    for reference in card_logic.references:
        if (
            reference.get("reference_class") != "gameplay"
            or target_node_id not in reference.get("target_node_ids", ())
            or reference.get("source_node_id") == target_node_id
        ):
            continue
        source_node_id = str(reference["source_node_id"])
        source_node = card_logic.nodes.get(source_node_id, {})
        triggers.append(
            {
                "source_node_id": source_node_id,
                "field_path": str(reference["field_path"]),
                "source_records": _node_source_records(source_node),
            }
        )
    return tuple(sorted(triggers, key=lambda item: (str(item["source_node_id"]), str(item["field_path"]))))


def _effect_application_sources(
    effect_name: str, card_logic: StaticCardLogicCatalogV1
) -> tuple[Mapping[str, Any], ...]:
    """Retain exact applicator fields that own duration/reset parameters."""

    applications: dict[tuple[str, str], dict[str, Any]] = {}
    effect_targets = _effect_target_node_ids(effect_name, card_logic)
    for reference in card_logic.references:
        target_node_ids = tuple(reference.get("target_node_ids", ()))
        matched_targets = sorted(effect_targets.intersection(target_node_ids))
        if not matched_targets:
            continue
        node_id = str(reference.get("source_node_id", ""))
        node = card_logic.nodes.get(node_id, {})
        if str(node.get("namespace")) == "BUFF":
            continue
        record = node.get("effective_record", node.get("merged_record", {}))
        field_path = str(reference.get("field_path", ""))
        if not isinstance(record, Mapping) or not field_path:
            continue
        owner = _field_parent(record, field_path)
        if owner is None:
            continue
        owner_record, field_name = owner
        application_kind = _BUFF_APPLICATION_FIELDS.get(field_name)
        if application_kind is None or (field_name == "SpawnData" and owner_record.get("SpawnType") != "BuffType"):
            continue
        parameters = _application_parameters(owner_record, field_name, field_path)
        parameter_provenance: dict[str, tuple[str, ...]] = {}
        if field_name == "TargetBuff":
            if "AllowResetTarget" in owner_record:
                parameter_provenance["AllowResetTarget"] = (f"{node_id}.AllowResetTarget=static_record",)
            else:
                parameters["AllowResetTarget"] = True
                parameter_provenance["AllowResetTarget"] = (
                    *_ALLOW_RESET_TARGET_DEFAULT_EVIDENCE,
                    f"{node_id}.AllowResetTarget=missing_uses_native_default",
                )
        applications[(node_id, field_path)] = {
            "source_node_id": node_id,
            "field_path": field_path,
            "application_kind": application_kind,
            "parameters": parameters,
            "parameter_provenance": parameter_provenance,
            "source_records": _node_source_records(node),
            "target_node_id": matched_targets[0],
            "trigger_sources": _trigger_sources(node_id, card_logic),
        }

    # Some legacy application fields are intentionally retained in the
    # lossless node record even when the card-logic reference schema does not
    # yet classify their target namespace.  Preserve those direct, top-level
    # applications while the reference-driven path above handles nested
    # actions and proved EXT aliases.
    for node_id, node in sorted(card_logic.nodes.items()):
        if str(node.get("namespace")) == "BUFF":
            continue
        record = node.get("effective_record", node.get("merged_record", {}))
        if not isinstance(record, Mapping):
            continue
        for field_name, application_kind in _BUFF_APPLICATION_FIELDS.items():
            if field_name not in record or (field_name == "SpawnData" and record.get("SpawnType") != "BuffType"):
                continue
            for index in _sequence_matches(record[field_name], effect_name):
                field_path = field_name if index is None else f"{field_name}[{index}]"
                if (node_id, field_path) in applications:
                    continue
                parameters = _application_parameters(record, field_name, field_path)
                parameter_provenance: dict[str, tuple[str, ...]] = {}
                if field_name == "TargetBuff":
                    if "AllowResetTarget" in record:
                        parameter_provenance["AllowResetTarget"] = (f"{node_id}.AllowResetTarget=static_record",)
                    else:
                        parameters["AllowResetTarget"] = True
                        parameter_provenance["AllowResetTarget"] = (
                            *_ALLOW_RESET_TARGET_DEFAULT_EVIDENCE,
                            f"{node_id}.AllowResetTarget=missing_uses_native_default",
                        )
                applications[(node_id, field_path)] = {
                    "source_node_id": node_id,
                    "field_path": field_path,
                    "application_kind": application_kind,
                    "parameters": parameters,
                    "parameter_provenance": parameter_provenance,
                    "source_records": _node_source_records(node),
                    "target_node_id": f"BUFF.{effect_name}",
                    "trigger_sources": _trigger_sources(node_id, card_logic),
                }
    return tuple(sorted(applications.values(), key=lambda item: (str(item["source_node_id"]), str(item["field_path"]))))


def _walk_inline_buff_spawns(
    value: Any, prefix: str = ""
) -> tuple[tuple[str, Mapping[str, Any], Mapping[str, Any]], ...]:
    """Return exact inline ``ActionSpawn(BuffType)`` definitions.

    Most Buff assets live in the top-level ``BUFF`` namespace. A small
    number are defined directly inside an action's ``SpawnData`` mapping and
    are materialized by the native engine with the same ``Buff`` global-ID
    hash. String-reference traversal cannot discover those definitions.
    """

    found: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = []
    if isinstance(value, Mapping):
        spawn_data = value.get("SpawnData")
        if value.get("SpawnType") == "BuffType" and isinstance(spawn_data, Mapping):
            path = f"{prefix}.SpawnData" if prefix else "SpawnData"
            found.append((path, value, spawn_data))
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            found.extend(_walk_inline_buff_spawns(item, path))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            found.extend(_walk_inline_buff_spawns(item, f"{prefix}[{index}]"))
    return tuple(found)


def _inline_buff_definitions(card_logic: StaticCardLogicCatalogV1) -> Mapping[str, Mapping[str, Any]]:
    """Collect content-addressed inline Buff definitions and applicators."""

    definitions: dict[str, dict[str, Any]] = {}
    for node_id, node in sorted(card_logic.nodes.items()):
        record = node.get("effective_record", node.get("merged_record", {}))
        if not isinstance(record, Mapping):
            continue
        for field_path, owner, definition in _walk_inline_buff_spawns(record):
            name = definition.get("Name")
            if not isinstance(name, str) or not name:
                raise ContractError(f"inline Buff definition at {node_id}.{field_path} lacks a valid Name")
            raw_record = dict(definition)
            source_records = _node_source_records(node)
            application = {
                "source_node_id": str(node_id),
                "field_path": field_path,
                "application_kind": "action_spawn_buff",
                "parameters": _application_parameters(owner, "SpawnData", field_path),
                "parameter_provenance": {},
                "source_records": source_records,
                "target_node_id": f"BUFF.{name}",
                "trigger_sources": _trigger_sources(str(node_id), card_logic),
            }
            existing = definitions.get(name)
            if existing is None:
                definitions[name] = {
                    "raw_record": raw_record,
                    "record_hash": content_hash(raw_record),
                    "application_sources": [application],
                    "source_records": set(source_records),
                }
                continue
            if existing["record_hash"] != content_hash(raw_record):
                raise ContractError(f"inline Buff definition {name!r} has conflicting records")
            existing["application_sources"].append(application)
            existing["source_records"].update(source_records)
    return definitions


def _mechanic_semantics(
    record: Mapping[str, Any], applications: tuple[Mapping[str, Any], ...]
) -> tuple[tuple[str, ...], Mapping[str, tuple[str, ...]]]:
    evidence: dict[str, set[str]] = {}

    def add(tag: str, *items: str) -> None:
        evidence.setdefault(tag, set()).update(item for item in items if item)

    speed_fields = {
        key: value
        for key in ("SpeedMultiplier", "HitSpeedMultiplier", "SpawnSpeedMultiplier")
        if (value := _numeric(record.get(key))) is not None
    }
    for field_name, tag in (
        ("SpeedMultiplier", "movement_lock"),
        ("HitSpeedMultiplier", "attack_lock"),
        ("SpawnSpeedMultiplier", "spawn_lock"),
    ):
        value = speed_fields.get(field_name)
        if value is not None and value <= -100:
            add(tag, f"BUFF.resolved_record.{field_name}={value}")
    negative = {key: value for key, value in speed_fields.items() if value < 0}
    if negative and any(-100 < value < 0 for value in negative.values()):
        add("slow", *(f"BUFF.resolved_record.{key}={value}" for key, value in negative.items()))
    if set(speed_fields) == {"SpeedMultiplier", "HitSpeedMultiplier", "SpawnSpeedMultiplier"} and all(
        value <= -100 for value in speed_fields.values()
    ):
        add("incapacitate", *(f"BUFF.resolved_record.{key}={value}" for key, value in speed_fields.items()))
    accelerated = {key: value for key, value in speed_fields.items() if value > 100}
    if accelerated:
        add("haste", *(f"BUFF.resolved_record.{key}={value}" for key, value in accelerated.items()))

    tid = record.get("TID")
    filter_name = record.get("FilterExportName")
    if "haste" in evidence and (tid == "TID_SPELL_RAGE" or filter_name == "filter_rage"):
        add(
            "rage",
            *(
                item
                for item in (
                    f"BUFF.resolved_record.TID={tid}" if tid else "",
                    (f"BUFF.resolved_record.FilterExportName={filter_name}" if filter_name else ""),
                )
            ),
            *evidence["haste"],
        )
    if "incapacitate" in evidence and tid == "TID_SPELL_FREEZE":
        add("freeze", "BUFF.resolved_record.TID=TID_SPELL_FREEZE", *evidence["incapacitate"])

    stun_application_evidence: set[str] = set()
    reset_application_evidence: set[str] = set()
    for application in applications:
        source_node_id = str(application["source_node_id"])
        parameters = application.get("parameters", {})
        if not isinstance(parameters, Mapping):
            continue
        stats_tags = parameters.get("StatsTags")
        if isinstance(stats_tags, Mapping) and any(
            value in {"stun_duration", "stun_time", "stun_buff_tid"} for value in stats_tags.values()
        ):
            stun_application_evidence.add(f"{source_node_id}.StatsTags={dict(stats_tags)!r}")
        if (
            application.get("application_kind") == "projectile_target_buff"
            and parameters.get("AllowResetTarget") is True
        ):
            reset_application_evidence.add(f"{source_node_id}.AllowResetTarget=True")
    if "incapacitate" in evidence and (tid == "TID_SPELL_ZAP_FREEZE" or stun_application_evidence):
        add(
            "stun",
            *(("BUFF.resolved_record.TID=TID_SPELL_ZAP_FREEZE",) if tid == "TID_SPELL_ZAP_FREEZE" else ()),
            *sorted(stun_application_evidence),
            *evidence["incapacitate"],
        )
    if reset_application_evidence:
        add("target_reset_allowed", *sorted(reset_application_evidence))
    if "stun" in evidence and reset_application_evidence:
        add("attack_interrupt", *evidence["stun"], *sorted(reset_application_evidence))

    if record.get("Invisible") is True:
        add("invisibility", "BUFF.resolved_record.Invisible=True")
    if record.get("Clone") is True:
        add("clone", "BUFF.resolved_record.Clone=True")
    if (value := _numeric(record.get("DamagePerSecond"))) not in (None, 0):
        add("periodic_damage", f"BUFF.resolved_record.DamagePerSecond={value}")
    if (value := _numeric(record.get("HealPerSecond"))) not in (None, 0):
        add("periodic_heal", f"BUFF.resolved_record.HealPerSecond={value}")
    if (value := _numeric(record.get("DamageReduction"))) not in (None, 0):
        add("damage_reduction", f"BUFF.resolved_record.DamageReduction={value}")
        if value >= 100:
            add("immunity", f"BUFF.resolved_record.DamageReduction={value}")
    if (value := _numeric(record.get("DamageMultiplier"))) not in (None, 0):
        add("damage_multiplier", f"BUFF.resolved_record.DamageMultiplier={value}")
    if (value := _numeric(record.get("HitpointMultiplier"))) not in (None, 0):
        add("hitpoint_multiplier", f"BUFF.resolved_record.HitpointMultiplier={value}")
    if record.get("Shield"):
        add("shield", f"BUFF.resolved_record.Shield={record['Shield']!r}")
    if record.get("DeathSpawn"):
        add("death_spawn", f"BUFF.resolved_record.DeathSpawn={record['DeathSpawn']!r}")
        if (
            isinstance(tid, str)
            and "CURSE" in tid
            or isinstance(filter_name, str)
            and "curse" in filter_name.casefold()
        ):
            add(
                "curse",
                *(
                    item
                    for item in (
                        f"BUFF.resolved_record.TID={tid}" if tid else "",
                        (f"BUFF.resolved_record.FilterExportName={filter_name}" if filter_name else ""),
                    )
                ),
            )
    if record.get("SpawnObject"):
        add("periodic_spawn", f"BUFF.resolved_record.SpawnObject={record['SpawnObject']!r}")
    if record.get("OverrideProjectile"):
        add("projectile_override", f"BUFF.resolved_record.OverrideProjectile={record['OverrideProjectile']!r}")
    if record.get("SwitchTeam") is True:
        add("team_switch", "BUFF.resolved_record.SwitchTeam=True")
    if "LockTarget" in record:
        add("target_lock_control", f"BUFF.resolved_record.LockTarget={record['LockTarget']!r}")
        add("targeting_modifier", f"BUFF.resolved_record.LockTarget={record['LockTarget']!r}")
    if record.get("IgnorePushBack") is True:
        add("pushback_immunity", "BUFF.resolved_record.IgnorePushBack=True")
    if record.get("RemoveOnAttack") is True:
        add("remove_on_attack", "BUFF.resolved_record.RemoveOnAttack=True")
    if record.get("RemoveOnHit") is True:
        add("remove_on_hit", "BUFF.resolved_record.RemoveOnHit=True")
    if record.get("LevelIncrease") is not None:
        add("level_change", f"BUFF.resolved_record.LevelIncrease={record['LevelIncrease']!r}")
    if record.get("AttractPercentage") is not None:
        add("attraction", f"BUFF.resolved_record.AttractPercentage={record['AttractPercentage']!r}")
    if record.get("GameTagsToSet"):
        item = f"BUFF.resolved_record.GameTagsToSet={record['GameTagsToSet']!r}"
        add("game_tag_mutation", item)
        add("targeting_modifier", item)
    if record.get("OverrideChargeRange") is not None:
        add("charge", f"BUFF.resolved_record.OverrideChargeRange={record['OverrideChargeRange']!r}")
    if record.get("OnStartAction"):
        add("on_start_action", f"BUFF.resolved_record.OnStartAction={record['OnStartAction']!r}")
    if record.get("OnRemoveAction"):
        add("on_remove_action", f"BUFF.resolved_record.OnRemoveAction={record['OnRemoveAction']!r}")
    if not evidence:
        add("opaque_native_buff", "BUFF.resolved_record:no_proved_normalized_mechanic")
    normalized = {tag: tuple(sorted(items)) for tag, items in sorted(evidence.items())}
    return tuple(normalized), normalized


def _select(record: Mapping[str, Any], names: frozenset[str]) -> Mapping[str, Any]:
    return {name: record[name] for name in sorted(names.intersection(record))}


def _relative_source(path: Path, workspace: Path) -> str:
    resolved = path.resolve()
    return resolved.relative_to(workspace).as_posix() if resolved.is_relative_to(workspace) else resolved.as_posix()


def _load_buff_global_id_registry(card_logic: StaticCardLogicCatalogV1, workspace: Path) -> Mapping[str, Any]:
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
        raise ContractError(f"exact Buff global-ID registry is missing or undecodable: {path}")
    with io.StringIO(plain.decode("utf-8-sig"), newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or rows[0].get("ID") != "int":
        raise ContractError("global_ids_oldformat.csv is missing its SC type row")

    entries: dict[str, int] = {}
    entry_sources: dict[str, str] = {}
    ids: dict[int, str] = {}
    relative = _relative_source(path, workspace)
    for row_index, row in enumerate(rows[1:]):
        if row.get("Type") != _BUFF_GLOBAL_ID_TYPE:
            continue
        name = str(row.get("Name", "")).strip()
        try:
            global_id = int(str(row.get("ID", "")))
        except ValueError as error:
            raise ContractError(f"invalid Buff global ID at {relative}#named-row={row_index}") from error
        if not name or not 1 <= global_id <= _UINT32_MAX:
            raise ContractError(f"invalid Buff registry entry at {relative}#named-row={row_index}")
        if name in entries:
            raise ContractError(f"duplicate Buff registry name: {name}")
        if global_id in ids:
            raise ContractError(f"duplicate Buff registry ID {global_id}: {ids[global_id]}, {name}")
        entries[name] = global_id
        ids[global_id] = name
        entry_sources[name] = f"{relative}#named-row={row_index}"

    return {
        "registry_kind": "global_ids_oldformat.csv",
        "global_id_type": _BUFF_GLOBAL_ID_TYPE,
        "source_record": relative,
        "raw_sha256": sha256_file(path),
        "decoded_sha256": hashlib.sha256(plain).hexdigest(),
        "explicit_buff_count": len(entries),
        "buff_entries": entries,
        "entry_source_records": entry_sources,
        "fallback_algorithm": "fnv1a32_utf8_type_plus_name",
        "fallback_prefix": _BUFF_GLOBAL_ID_PREFIX,
        "fallback_offset_basis": _FNV1A_OFFSET_BASIS,
        "fallback_prime": _FNV1A_PRIME,
        "native_evidence": (
            "libg.arm64-v15.535.13@0xd7401c:name_hash_registry_path",
            "libg.arm64-v15.535.13@0xd740a4:offset_basis=0x811c9dc5",
            "libg.arm64-v15.535.13@0xd740c4:prime=0x01000193",
            "libg.arm64-v15.535.13@0xd961f4:global_id_type=Buff",
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
    if base not in records:
        return record, (name,), (f"missing_base:{base}",)
    inherited, inherited_chain, unresolved = _resolve_inherited_record(base, records, chain=chain + (name,))
    inherited.update(record)
    return inherited, inherited_chain + (name,), unresolved


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
    workspace = Path(workspace_root or WORKSPACE_ROOT).resolve()
    global_id_registry = _load_buff_global_id_registry(card_logic, workspace)
    explicit_ids = global_id_registry["buff_entries"]
    if not isinstance(explicit_ids, Mapping):
        raise ContractError("Buff global-ID registry entries are invalid")
    nodes = {str(node["name"]): node for node in card_logic.nodes.values() if node.get("namespace") == "BUFF"}
    inline_definitions = _inline_buff_definitions(card_logic)
    duplicate_definitions = sorted(set(nodes).intersection(inline_definitions))
    if duplicate_definitions:
        raise ContractError(
            "inline Buff definitions duplicate top-level BUFF nodes: " + ", ".join(duplicate_definitions)
        )
    records = {name: dict(node.get("merged_record", {})) for name, node in nodes.items()}
    effects: list[NativeEffectDefinitionV1] = []
    for name, node in sorted(nodes.items()):
        raw_record = records[name]
        resolved, inheritance, unresolved = _resolve_inherited_record(name, records)
        source_records = tuple(
            str(fragment["source_record"])
            for fragment in node.get("fragments", ())
            if isinstance(fragment, Mapping) and fragment.get("source_record")
        )
        global_id_key = _effect_global_id_key(name)
        explicit_global_id = explicit_ids.get(name)
        if explicit_global_id is None:
            buff_global_id = _fnv1a32(global_id_key)
            global_id_strategy = "fnv1a_type_name"
        else:
            buff_global_id = int(explicit_global_id)
            global_id_strategy = "explicit_oldformat_registry"
        applications = _effect_application_sources(name, card_logic)
        mechanic_tags, mechanic_evidence = _mechanic_semantics(resolved, applications)
        effects.append(
            NativeEffectDefinitionV1(
                effect_name=name,
                node_id=str(node["node_id"]),
                buff_global_id=buff_global_id,
                global_id_strategy=global_id_strategy,
                global_id_key=global_id_key,
                mechanic_tags=mechanic_tags,
                mechanic_evidence=mechanic_evidence,
                application_sources=applications,
                modifiers=_select(resolved, _MODIFIER_FIELDS),
                lifecycle=_select(resolved, _LIFECYCLE_FIELDS),
                targeting=_select(resolved, _TARGETING_FIELDS),
                spawn=_select(resolved, _SPAWN_FIELDS),
                overrides=_select(resolved, _OVERRIDE_FIELDS),
                raw_record=raw_record,
                resolved_record=resolved,
                inheritance_chain=inheritance,
                unresolved_semantics=unresolved,
                source_records=source_records,
            )
        )
    for name, definition in sorted(inline_definitions.items()):
        raw_record = dict(definition["raw_record"])
        applications = tuple(definition["application_sources"])
        global_id_key = _effect_global_id_key(name)
        explicit_global_id = explicit_ids.get(name)
        if explicit_global_id is None:
            buff_global_id = _fnv1a32(global_id_key)
            global_id_strategy = "fnv1a_type_name"
        else:
            buff_global_id = int(explicit_global_id)
            global_id_strategy = "explicit_oldformat_registry"
        mechanic_tags, mechanic_evidence = _mechanic_semantics(raw_record, applications)
        effects.append(
            NativeEffectDefinitionV1(
                effect_name=name,
                node_id=f"BUFF.{name}",
                buff_global_id=buff_global_id,
                global_id_strategy=global_id_strategy,
                global_id_key=global_id_key,
                mechanic_tags=mechanic_tags,
                mechanic_evidence=mechanic_evidence,
                application_sources=applications,
                modifiers=_select(raw_record, _MODIFIER_FIELDS),
                lifecycle=_select(raw_record, _LIFECYCLE_FIELDS),
                targeting=_select(raw_record, _TARGETING_FIELDS),
                spawn=_select(raw_record, _SPAWN_FIELDS),
                overrides=_select(raw_record, _OVERRIDE_FIELDS),
                raw_record=raw_record,
                resolved_record=raw_record,
                inheritance_chain=(name,),
                unresolved_semantics=(),
                source_records=tuple(definition["source_records"]),
            )
        )
    return NativeEffectCatalogV1(
        card_logic_catalog_id=card_logic.catalog_id,
        criteria_version=EFFECT_CATALOG_CRITERIA_VERSION,
        global_id_registry=global_id_registry,
        effects=tuple(effects),
    )
