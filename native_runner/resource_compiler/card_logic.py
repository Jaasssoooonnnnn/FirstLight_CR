"""Build a lossless, content-addressed static card-logic graph.

``card_specs.py`` intentionally exposes a compact model-facing projection.  It
is not a suitable place to flatten every native data-table extension, action,
buff, area object, projectile, or inline action.  This module keeps that richer
configuration as an immutable graph:

* every non-empty source field is retained in a provenance-bearing fragment;
* gameplay references are resolved conservatively and closure is computed per
  card;
* ambiguous and missing gameplay references are explicit field paths;
* visual/audio/localisation strings remain in raw records but are never
  promoted to gameplay edges merely because their names look familiar; and
* rarity level multipliers are preserved as exact projection inputs; and
* exact numeric carrier fields expose an explicit level-11 projection using
  the native integral-stat truncation rule, without treating relative percent
  overrides as standalone base values.

The graph is static evidence.  It does not claim that the current native probe
can observe these states at runtime.
"""

from __future__ import annotations

from native_runner.paths import WORKSPACE_ROOT

from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Mapping
import csv
from dataclasses import dataclass, field
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sqlite3
import tomllib
from typing import Any, ClassVar

from native_runner.resource_compiler.card_specs import (
    CardSpecCatalog,
    _NATIVE_CHARACTER_DEFAULT_BINARY_SHA256,
    _NATIVE_CHARACTER_DEFAULT_RELEASE,
    _asset_bytes,
    _deep_merge,
    _matches_native_character_default_build,
    _scalar,
    build_card_catalog,
    discover_card_source,
    sha256_file,
)
from native_runner.contracts import ContractError, ContractMixin, content_hash, frozen_mapping
from native_runner.resource_compiler.static_logic_resolution import merge_effective_record


STATIC_CARD_LOGIC_VERSION = "static-card-logic-catalog.v1"
STATIC_CARD_LOGIC_CRITERIA_VERSION = "static-card-logic-criteria.2026-08-18.v8"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_QUALIFIED_REFERENCE = re.compile(r"^([A-Z][A-Z0-9_]*)\.(.+)$")
_EXPRESSION_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_EMPTY_TOML_STUB_SHA256 = "0573b69cc2c4df3d5e5a3f29e7d2929ae2655bfe1e03e2edba4a020c5ba56da8"

# Namespaces that can participate in ordinary card behaviour.  Other decoded
# TOML documents are deliberately ignored by this graph (economy, cosmetics,
# tasks, game-mode rewards, and similar non-card configuration).
LOGIC_NAMESPACES = frozenset(
    {
        "ABILITY",
        "ACTION",
        "AEO",
        "BUFF",
        "BUILDING",
        "CHARACTER",
        "DAMAGE_TYPE",
        "EXT",
        "FILTER",
        "GLOBAL",
        "PROJECTILE",
        "RARITY",
        "SHAPE",
        "SPELL_BUILDING",
        "SPELL_CHARACTER",
        "SPELL_EVOLVED",
        "SPELL_HERO",
        "SPELL_OTHER",
        "STATS",
        "TARGET_RESOLVER",
        "VARIABLE",
    }
)

_CSV_NODE_TABLES: Mapping[str, str] = {
    "area_effect_objects.csv": "AEO",
    "buildings.csv": "BUILDING",
    "character_abilities.csv": "ABILITY",
    "character_buffs.csv": "BUFF",
    "characters.csv": "CHARACTER",
    "globals.csv": "GLOBAL",
    "projectiles.csv": "PROJECTILE",
    "rarities.csv": "RARITY",
    "spells_buildings.csv": "SPELL_BUILDING",
    "spells_characters.csv": "SPELL_CHARACTER",
    "spells_evolved.csv": "SPELL_EVOLVED",
    "spells_hero_form.csv": "SPELL_HERO",
    "spells_other.csv": "SPELL_OTHER",
}

# Several aggregate TOMLs use ``[Name]`` rather than ``[NAMESPACE.Name]``.
# The filename is the only explicit namespace declaration for those files.
_TOML_DEFAULT_NAMESPACE: Mapping[str, str] = {
    "actions.toml": "ACTION",
    "area_effect_objects.toml": "AEO",
    "area_effect_objects_evo.toml": "AEO",
    "buildings.toml": "BUILDING",
    "buildings_evo.toml": "BUILDING",
    "character_buffs.toml": "BUFF",
    "character_buffs_evo.toml": "BUFF",
    "characters.toml": "CHARACTER",
    "characters_evo.toml": "CHARACTER",
    "damage_types.toml": "DAMAGE_TYPE",
    "game_object_filters.toml": "FILTER",
    "projectiles.toml": "PROJECTILE",
    "projectiles_evo.toml": "PROJECTILE",
    "shapes.toml": "SHAPE",
    "spells_buildings.toml": "SPELL_BUILDING",
    "spells_characters.toml": "SPELL_CHARACTER",
    "spells_evolved.toml": "SPELL_EVOLVED",
    "spells_heroes.toml": "SPELL_HERO",
    "spells_other.toml": "SPELL_OTHER",
    "targetresolvers.toml": "TARGET_RESOLVER",
    "variables.toml": "VARIABLE",
}

_CARD_ROOT_NAMESPACE: Mapping[int, str] = {
    26: "SPELL_CHARACTER",
    27: "SPELL_BUILDING",
    28: "SPELL_OTHER",
    29: "SPELL_HERO",
}

_SPELL_NAMESPACES = ("SPELL_CHARACTER", "SPELL_BUILDING", "SPELL_OTHER", "SPELL_EVOLVED", "SPELL_HERO")
_UNIT_NAMESPACES = ("CHARACTER", "BUILDING", "EXT")

_EXTERNAL_FIELD_MARKERS = (
    "Animation",
    "AnimExport",
    "Audio",
    "Clip",
    "ContinuousEffect",
    "DamageEffect",
    "DeathEffect",
    "DeployEffect",
    "EffectFlags",
    "ExportName",
    "FileName",
    "FilterExport",
    "FilterFile",
    "FlameEffect",
    "HealthBar",
    "HitEffect",
    "Icon",
    "LoopingEffect",
    "MarkEffect",
    "Mesh",
    "MoveEffect",
    "OneShotEffect",
    "ProjectileEffect",
    "ReadyEffect",
    "ScaledEffect",
    "Shader",
    "Sound",
    "SpawnEffect",
    "TargetedEffect",
    "TargettedEffect",
    "TID",
    "TrailEffect",
    "Visual",
    "VFX",
)

# ``DashFilter`` is a render filter export, not a LogicGameObjectFilter name.
# The active release values are exported by ``assets/sc/*.sc``; additionally,
# ``filter_bandit_charge`` is explicitly declared as a
# ``RenderableComponents[Type=Filter].ExportName`` in ``vfx/effects.json``.
# Native corroboration: the LogicCharacterData getter at VA 0xd98c50 is used
# by the view-side effect/filter attachment path at 0x83e1d4..0x83e280.  The
# values do not name any decoded FILTER/TARGET_RESOLVER node.  Keep them in the
# lossless raw record, but never manufacture a gameplay graph edge for them.
_EXTERNAL_FIELD_NAMES = frozenset({"DashFilter"})

_ACTION_FIELDS = frozenset(
    {
        "Action",
        "ActionIfNoMatch",
        "ActionOnCapturedObject",
        "ActionOnCooldownReady",
        "ActionOnDeflector",
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
        "Actions",
        "DoAttackAction",
        "FailureAction",
        "FailureActionOnInstigator",
        "FirstAppearAction",
        "NextAction",
        "HasTargetOnDeployAction",
        "HideAction",
        "HideActions",
        "InstigatorAction",
        "NoTargetOnDeployAction",
        "OnActivationAction",
        "OnActivateAction",
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
        "TetherHitAction",
        "OnTetherActivationActionOnConnectedUnit",
        "WarpAction",
        "VisualActionForEnemyTarget",
    }
)

# These fields point at static data used while evaluating an action or choosing
# a runtime variant.  They are not execution edges and must not make their
# targets look like spawned runtime objects merely because the target happens
# to live in the catch-all EXT namespace.
_DATA_DEPENDENCY_FIELDS = frozenset(
    {
        "BuffToConsider",
        "ChampionCharacterData",
        "Filter",
        "GameObjectFilter",
        "IgnoreBuff",
        "IgnoreTargetsWithBuff",
        "LinkedChampionCharacter",
        "ObjectFilter",
        "Resolver",
        "SnipeTargetFilter",
        "TargetFilter",
        "TargetResolver",
        "TetherDamageTargets",
        "TroopFilter",
        "ValidTargetBuff",
    }
)

_VISUAL_ACTION_CLASS_TYPES = frozenset(
    {"ActionAttachEffect", "ActionPlayEffect", "ActionPlaySound", "ActionSetAnimation", "ActionSetShader", "ActionTint"}
)

# Each entry is grounded by a field signature shared by every decoded instance
# of the native class. Class names alone are never sufficient classification
# evidence. ``mechanic_tags`` intentionally use the same stable vocabulary as
# the ordinary field classifier below.
NATIVE_ACTION_SEMANTIC_RULES: Mapping[str, Mapping[str, Any]] = {
    "ActionAttackChain": {
        "operation": "target_seeking_attack_chain",
        "required_fields": ("ChainCompleteIfTrue", "ChainCount", "OnChainBegan", "OnFinishedAction", "TargetResolver"),
        "mechanic_tags": ("attack_sequence", "target_seeking"),
    },
    "ActionBarbBarrelHeroReRoll": {
        "operation": "projectile_reroll_state",
        "required_fields": (
            "DeployDuration",
            "GameTagsToSetWhileOnReRolling",
            "OnReRollEndAction",
            "OnReRollStartAction",
            "ReRollProjectile",
        ),
        "mechanic_tags": ("projectile", "transform", "visibility_transition"),
    },
    "ActionBlowdartGoblinEvoController": {
        "operation": "stacked_area_damage_controller",
        "required_fields": ("AeoList", "CrownTowerDuration", "MaxStacks", "StackAmountChecks"),
        "mechanic_tags": ("area_effect", "damage_over_time", "variable_damage_stage"),
    },
    "ActionBlowdartGoblinEvoDartSelect": {
        "operation": "stack_dependent_projectile_select",
        "required_fields": ("ActionToTakeDataFrom", "SpecialProjectile"),
        "mechanic_tags": ("projectile_override", "variable_damage_stage"),
    },
    "ActionBossBanditAbility": {
        "operation": "ability_warp_lock",
        "required_fields": ("LockDelay", "ReleaseLockDelay", "WarpAction", "WarpDelay"),
        "mechanic_tags": ("active_ability", "dash_teleport_jump"),
    },
    "ActionCaptureCharacter": {
        "operation": "capture_and_drag",
        "required_fields": ("CaptureDragTime", "CaptureRadius", "NumberOfUnitsToCapture", "TargetFilter"),
        "mechanic_tags": ("capture", "pull", "target_lock"),
    },
    "ActionCannonBarrage": {
        "operation": "scheduled_area_barrage",
        "required_fields": ("BombAreaEffectObjects", "BombHorizontalOffsets", "BombVerticalOffsets"),
        "mechanic_tags": ("area_effect", "multi_projectile", "scheduled_spawn"),
    },
    "ActionCreateParallelProjectiles": {
        "operation": "parallel_projectile_attack",
        "required_fields": ("ProjectileCount", "ProjectileDistance", "ProjectileType"),
        "mechanic_tags": ("multi_projectile", "projectile"),
    },
    "ActionExecutionerEvoProjectile": {
        "operation": "distance_staged_returning_projectile",
        "required_fields": (
            "Damage",
            "FirstStrongHitPushback",
            "HitAction",
            "StrongDamage",
            "StrongDamageRange",
            "StrongHitAction",
        ),
        "mechanic_tags": ("bounce_return", "knockback", "projectile", "variable_damage_stage"),
    },
    "ActionGhostEvoAction": {
        "operation": "symmetric_area_summon",
        "required_fields": ("DamageAEO", "LeftSummonAreaType", "RightSummonAreaType", "SummonActionData"),
        "mechanic_tags": ("area_effect", "damage", "periodic_spawn"),
    },
    "ActionGoblinDrillEvoRelocate": {
        "operation": "health_threshold_relocate",
        "required_fields": ("FirstAppearAction", "HideActions", "HideHpThresholds", "HideTime", "SpawnCharaterRadius"),
        "mechanic_tags": ("phase_change", "spawn", "visibility_transition"),
    },
    "ActionGiantBufferBuff": {
        "operation": "periodic_attack_damage_modifier",
        "required_fields": (
            "AddedDamage",
            "AttackAmount",
            "DamageMultiplierPerUnitNames",
            "DamageMultiplierPerUnitValues",
            "FinishIfInstigatorDies",
            "InstigatorDepth",
        ),
        "mechanic_tags": ("damage_modifier",),
    },
    "ActionGiantBufferCollectFriends": {
        "operation": "nearest_ally_collect_and_buff",
        "required_fields": ("ActionWhenUnitBuffed", "MaxFriendlyTroops", "Projectile", "TargetFilter"),
        "mechanic_tags": ("damage_modifier", "projectile", "target_lock"),
    },
    "ActionGoblinHutLifeState": {
        "operation": "lifecycle_spawn_controller",
        "required_fields": ("SpawnData", "SpawnInterval", "SpawnNumber"),
        "mechanic_tags": ("lifetime", "periodic_spawn"),
    },
    "ActionGoblinsteinAbility": {
        "operation": "ability_damage_tether",
        "required_fields": ("TetherDamage", "TetherDuration", "TetherHitInterval", "TetherWidth"),
        "mechanic_tags": ("active_ability", "attachment", "damage_over_time"),
    },
    "ActionHunterNetAttack": {
        "operation": "cooldown_projectile_attack",
        "required_fields": ("Cooldown", "Projectile", "Range", "TargetFilter"),
        "mechanic_tags": ("projectile", "target_lock"),
    },
    "ActionLaserBall": {
        "operation": "target_count_damage_controller",
        "required_fields": (
            "DetectionRadius",
            "FirstHitDelay",
            "HitFrequency",
            "MaxUnitPerActionList",
            "OnDetectedUnitActionList",
        ),
        "mechanic_tags": ("periodic_damage", "target_lock", "variable_damage_stage"),
    },
    "ActionHide": {
        "operation": "hide_and_disable_interactions",
        "required_fields": (),
        # Capture variants carry HealthBarYOffset while long-lived hide
        # variants carry Duration/GameTagsToSet.  Requiring at least one
        # decoded parameter keeps this a field-signature classification rather
        # than trusting the native class name alone.
        "required_any_fields": ("Duration", "GameTagsToSet", "HealthBarYOffset"),
        "mechanic_tags": ("visibility_transition", "target_lock", "movement_modifier"),
    },
    "ActionLumberjackGhostWaitUntilLooseBuff": {
        "operation": "buff_loss_lifecycle_gate",
        "required_fields": ("ActionToExecute", "BuffToConsider", "PortalTimer"),
        "mechanic_tags": ("death_effect", "lifetime"),
    },
    "ActionMegaMinionHeroAbility": {
        "operation": "targeted_ability_teleport",
        "required_fields": ("ActionToExecute", "ActionToGetTargetFrom", "ForceStopIfTrue"),
        "mechanic_tags": ("active_ability", "dash_teleport_jump", "target_lock"),
    },
    "ActionMiniPekkaHeroQuest": {
        "operation": "timed_ability_progression",
        "required_fields": ("Intervals", "MaxResets", "OnIntervalReachedAction"),
        "mechanic_tags": ("ability_progress", "active_ability"),
    },
    "ActionMirroredExtraSpell": {
        "operation": "mirrored_extra_projectile",
        "required_fields": ("Projectile",),
        "mechanic_tags": ("multi_projectile", "projectile"),
    },
    "ActionMusketeerSnipe": {
        "operation": "ammo_limited_snipe_targeting",
        "required_fields": ("AmmoCount", "SnipeMaxRange", "SnipeTargetFilter"),
        "mechanic_tags": ("ammo", "projectile", "target_lock"),
    },
    "ActionSetInstantHit": {
        "operation": "conditional_instant_hit",
        "required_fields": ("ExecuteIfTrue",),
        "mechanic_tags": ("projectile", "target_lock"),
    },
    "ActionSkeletonBarrelPopBalloon": {
        "operation": "health_threshold_form_transition",
        "required_fields": ("BalloonFlyStartFrameList", "DropBalloonAtHpList", "TransitionTime"),
        "mechanic_tags": ("health_phase", "transform"),
    },
}

_MECHANIC_FIELD_FAMILIES: Mapping[str, str] = {
    "Ability": "active_ability",
    "AreaEffectObject": "area_effect",
    "AreaEffectOnDash": "dash_area_effect",
    "AreaEffectOnHit": "hit_area_effect",
    "AttackSequence": "attack_sequence",
    "Buff": "applies_buff",
    "BuffAfterHits": "hit_count_buff",
    "BuffOnDamage": "damage_triggered_buff",
    "BuffOnKill": "kill_triggered_buff",
    "BuffWhenNotAttacking": "idle_buff",
    "ChainedBuff": "chained_buff",
    "ChainedHitCount": "chained_hit",
    "ConvertOnKill": "convert_on_kill",
    "DamagePerSecond": "damage_over_time",
    "DamageReduction": "damage_reduction",
    "DashDamage": "dash_damage",
    "DeathAreaEffect": "death_area_effect",
    "DeathSpawnCharacter": "death_spawn",
    "DeflectProjectilesEnabled": "projectile_deflection",
    "HealPerSecond": "heal_over_time",
    "HidesWhenNotAttacking": "visibility_transition",
    "HitSpeedMultiplier": "hit_speed_modifier",
    "Invisible": "invisibility",
    "LockTarget": "target_lock",
    "MorphCharacter": "transform",
    "MorphTarget": "transform",
    "OverrideProjectile": "projectile_override",
    "ReflectedAttackDamage": "damage_reflection",
    "RemoveOnAttack": "remove_status_on_attack",
    "RemoveOnHit": "remove_status_on_hit",
    "Shield": "shield_status",
    "ShieldHitpoints": "shield_hitpoints",
    "SpawnAreaEffectObject": "spawns_area_effect",
    "SpawnCharacter": "periodic_spawn",
    "SpawnObject": "status_spawn",
    "SpawnProjectile": "spawns_projectile",
    "SpawnSpeedMultiplier": "spawn_speed_modifier",
    "SpeedMultiplier": "movement_speed_modifier",
    "SummonCharactersList": "periodic_spawn",
    "SwitchTeam": "team_switch",
    "TargetBuff": "projectile_applies_buff",
    "UntargetableWhenSpawned": "spawn_untargetable",
    "VariableDamage2": "variable_damage_stage",
    "VariableDamage3": "variable_damage_stage",
}


def classify_native_action(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return a signature-validated semantic operation for one action record."""

    class_type = str(record.get("ClassType") or "")
    rule = NATIVE_ACTION_SEMANTIC_RULES.get(class_type)
    if rule is None:
        return None
    required = tuple(str(item) for item in rule["required_fields"])
    missing = tuple(field_name for field_name in required if field_name not in record)
    if missing:
        return None
    required_any = tuple(str(item) for item in rule.get("required_any_fields", ()))
    matched_any = tuple(field_name for field_name in required_any if field_name in record)
    if required_any and not matched_any:
        return None
    return {
        "class_type": class_type,
        "operation": str(rule["operation"]),
        "mechanic_tags": tuple(str(item) for item in rule["mechanic_tags"]),
        "signature_fields": (*required, *matched_any),
    }


def _typed_native_actions(closure: Iterable[str], nodes: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    for node_id in sorted(closure):
        classification = classify_native_action(_node_record(nodes[node_id]))
        if classification is not None:
            result.append({"node_id": node_id, **classification})
    return tuple(result)


def _mechanic_field_is_active(field_name: str, value: Any) -> bool:
    """Reject serialized defaults before promoting a raw field to a mechanic.

    SC tables commonly serialize defaults such as ``False``, ``0`` and empty
    lists.  Presence alone therefore is not evidence that a mechanic is
    enabled (notably projectile deflection/clone-related fields).
    """

    del field_name  # Reserved for future field-specific default sentinels.
    if value is None or value is False:
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (tuple, list, Mapping)):
        return bool(value)
    return True


def _relative(path: Path, workspace: Path) -> str:
    resolved = path.resolve()
    return resolved.relative_to(workspace).as_posix() if resolved.is_relative_to(workspace) else resolved.as_posix()


def _decoded_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_sc_csv_groups(path: Path) -> tuple[dict[str, str], list[tuple[str, dict[str, Any], int]]]:
    """Read named SC rows while preserving their unnamed continuation rows."""

    plain = _asset_bytes(path)
    if plain is None:
        raise ContractError(f"asset table is not decoded plaintext: {path}")
    with io.StringIO(plain.decode("utf-8-sig"), newline="") as stream:
        raw_rows = list(csv.DictReader(stream))
    if not raw_rows:
        return {}, []
    type_row = raw_rows[0]
    result: list[tuple[str, dict[str, Any], int]] = []
    current: dict[str, Any] | None = None
    current_name = ""
    current_row = 0
    continuations: list[dict[str, Any]] = []

    def finish() -> None:
        nonlocal current
        if current is None:
            return
        record = dict(current)
        if continuations:
            record["__continuation_rows__"] = tuple(continuations)
        result.append((current_name, record, current_row))

    for raw_index, raw in enumerate(raw_rows[1:]):
        row = {
            str(key): parsed for key, value in raw.items() if key is not None and (parsed := _scalar(value)) is not None
        }
        name = str(row.get("Name", "")).strip()
        if name:
            finish()
            current = row
            current_name = name
            current_row = raw_index
            continuations = []
        elif current is not None and row:
            continuations.append(row)
    finish()
    schema = {str(key): str(value) for key, value in type_row.items() if key is not None and value}
    return schema, result


def _flatten_leaves(value: Any, prefix: str = "") -> dict[str, Any]:
    leaves: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            leaves.update(_flatten_leaves(item, child))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            leaves.update(_flatten_leaves(item, f"{prefix}[{index}]"))
    else:
        leaves[prefix] = value
    return leaves


def _fragment_conflicts(fragments: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    observed: dict[str, set[str]] = defaultdict(set)
    for fragment in fragments:
        for path, value in _flatten_leaves(fragment.get("record", {})).items():
            observed[path].add(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return tuple(sorted(path for path, values in observed.items() if len(values) > 1))


def _record_fragment(
    registry: dict[str, list[dict[str, Any]]],
    *,
    namespace: str,
    name: str,
    record: Mapping[str, Any],
    source_record: str,
    layer: str,
) -> None:
    if namespace not in LOGIC_NAMESPACES or not name:
        return
    node_id = f"{namespace}.{name}"
    registry[node_id].append({"source_record": source_record, "layer": layer, "record": dict(record)})


def _merge_semantic_ext_overlays(nodes: dict[str, dict[str, Any]]) -> None:
    """Fold proved same-file partial tables into their derived EXT object.

    Per-card TOMLs can declare a derived object as ``EXT.Name`` and append a
    nested list through ``SEMANTIC_NAMESPACE.Name`` in the same file.  Such a
    partial table is not a second game object.  We require all of the
    following before folding: the EXT Base names that semantic namespace, the
    same-name table has no independent Base/Name identity, and both records
    have the exact same source-file set.  The contributing fragments and
    original partial node ID remain recorded on the EXT node.
    """

    remove: list[str] = []
    for node_id, extension in tuple(nodes.items()):
        if extension["namespace"] != "EXT":
            continue
        extension_record = extension["merged_record"]
        base = extension_record.get("Base")
        match = _QUALIFIED_REFERENCE.fullmatch(base) if isinstance(base, str) else None
        if match is None:
            continue
        overlay_id = f"{match.group(1)}.{extension['name']}"
        overlay = nodes.get(overlay_id)
        if overlay is None:
            continue
        overlay_record = overlay["merged_record"]
        if "Base" in overlay_record or "Name" in overlay_record:
            continue
        extension_sources = {str(fragment["source_record"]).split("#", 1)[0] for fragment in extension["fragments"]}
        overlay_sources = {str(fragment["source_record"]).split("#", 1)[0] for fragment in overlay["fragments"]}
        if not extension_sources or extension_sources != overlay_sources:
            continue
        fragments = tuple((*extension["fragments"], *overlay["fragments"]))
        extension["merged_record"] = _deep_merge(extension_record, overlay_record)
        extension["fragments"] = fragments
        extension["conflict_fields"] = _fragment_conflicts(fragments)
        extension["semantic_overlay_source_node_ids"] = (overlay_id,)
        remove.append(overlay_id)
    for node_id in remove:
        del nodes[node_id]


def _load_logic_registry(
    workspace: Path, source: Path, runtime_logic: Path
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, str]],
    dict[str, dict[str, Any]],
    tuple[dict[str, str], ...],
    tuple[dict[str, Any], ...],
]:
    fragments: dict[str, list[dict[str, Any]]] = defaultdict(list)
    schemas: dict[str, dict[str, str]] = {}
    source_files: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []
    placeholders: list[dict[str, Any]] = []
    runtime_replacement_paths = (
        {
            path.relative_to(runtime_logic).as_posix().casefold()
            for path in runtime_logic.rglob("*")
            if path.suffix.lower() in {".csv", ".toml"} and _asset_bytes(path) is not None
        }
        if runtime_logic.is_dir()
        else set()
    )

    def bind_file(path: Path, plain: bytes) -> str:
        relative = _relative(path, workspace)
        source_files[relative] = {
            "raw_sha256": sha256_file(path),
            "decoded_sha256": _decoded_sha256(plain),
            "raw_size": path.stat().st_size,
            "decoded_size": len(plain),
        }
        return relative

    for layer, logic_root in (("release", source), ("runtime", runtime_logic)):
        if not logic_root.is_dir():
            continue
        for filename, namespace in _CSV_NODE_TABLES.items():
            path = logic_root / filename
            if not path.is_file():
                continue
            if layer == "release" and Path(filename).as_posix().casefold() in runtime_replacement_paths:
                continue
            plain = _asset_bytes(path)
            if plain is None:
                failures.append({"source": _relative(path, workspace), "reason": "decoded_plaintext_unavailable"})
                continue
            relative = bind_file(path, plain)
            schema, rows = _read_sc_csv_groups(path)
            schemas[f"{relative}#{namespace}"] = schema
            for name, record, row_index in rows:
                _record_fragment(
                    fragments,
                    namespace=namespace,
                    name=name,
                    record=record,
                    source_record=f"{relative}#named-row={row_index}",
                    layer=layer,
                )

        for path in sorted(logic_root.rglob("*.toml")):
            if layer == "release" and path.relative_to(logic_root).as_posix().casefold() in runtime_replacement_paths:
                continue
            plain = _asset_bytes(path)
            default_namespace = _TOML_DEFAULT_NAMESPACE.get(path.name.lower())
            if plain is None:
                if default_namespace is not None:
                    relative = _relative(path, workspace)
                    raw_sha256 = sha256_file(path)
                    if path.stat().st_size == 24 and raw_sha256 == _EMPTY_TOML_STUB_SHA256:
                        source_files[relative] = {
                            "raw_sha256": raw_sha256,
                            "decoded_sha256": None,
                            "raw_size": path.stat().st_size,
                            "decoded_size": 0,
                            "classification": "empty_redirect_placeholder",
                        }
                        placeholders.append(
                            {
                                "source": relative,
                                "raw_sha256": raw_sha256,
                                "raw_size": path.stat().st_size,
                                "classification": "empty_redirect_placeholder",
                            }
                        )
                    else:
                        failures.append({"source": relative, "reason": "decoded_plaintext_unavailable"})
                continue
            try:
                document = tomllib.loads(plain.decode("utf-8-sig").rstrip("\x00\r\n\t "))
            except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
                if default_namespace is not None:
                    failures.append(
                        {"source": _relative(path, workspace), "reason": f"invalid_toml:{type(error).__name__}"}
                    )
                continue

            explicit = any(key in LOGIC_NAMESPACES for key in document)
            if not explicit and default_namespace is None:
                continue
            relative = bind_file(path, plain)
            for namespace in sorted(LOGIC_NAMESPACES.intersection(document)):
                records = document.get(namespace)
                if not isinstance(records, Mapping):
                    continue
                for name, record in records.items():
                    if isinstance(record, Mapping):
                        _record_fragment(
                            fragments,
                            namespace=namespace,
                            name=str(name),
                            record=record,
                            source_record=f"{relative}#{namespace}.{name}",
                            layer=layer,
                        )
            if default_namespace is not None:
                # A namespaced file can also contain aggregate unnamespaced
                # records.  Namespace keys themselves are skipped above.
                for name, record in document.items():
                    if name in LOGIC_NAMESPACES or not isinstance(record, Mapping):
                        continue
                    _record_fragment(
                        fragments,
                        namespace=default_namespace,
                        name=str(name),
                        record=record,
                        source_record=f"{relative}#{default_namespace}.{name}",
                        layer=layer,
                    )

    nodes: dict[str, dict[str, Any]] = {}
    for node_id, node_fragments in sorted(fragments.items()):
        namespace, name = node_id.split(".", 1)
        merged: dict[str, Any] = {}
        for fragment in node_fragments:
            merged = _deep_merge(merged, fragment["record"])
        nodes[node_id] = {
            "node_id": node_id,
            "namespace": namespace,
            "name": name,
            "merged_record": merged,
            "fragments": tuple(node_fragments),
            "conflict_fields": _fragment_conflicts(node_fragments),
        }
    _merge_semantic_ext_overlays(nodes)
    return nodes, schemas, source_files, tuple(failures), tuple(placeholders)


def _field_name(path: str) -> str:
    value = path.rsplit(".", 1)[-1]
    return value.split("[", 1)[0]


def _node_record(node: Mapping[str, Any]) -> Mapping[str, Any]:
    record = node.get("effective_record", node.get("merged_record"))
    if not isinstance(record, Mapping):
        raise ContractError("static logic node lacks a materialized record")
    return record


def _is_external_field(path: str) -> bool:
    if ".StatsTags." in f".{path}." or path.startswith("StatsTags."):
        return True
    name = _field_name(path)
    if name in _EXTERNAL_FIELD_NAMES:
        return True
    return any(marker.lower() in name.lower() for marker in _EXTERNAL_FIELD_MARKERS)


def _expected_namespaces(source_namespace: str, path: str, parent: Mapping[str, Any]) -> tuple[str, ...]:
    name = _field_name(path)
    if _is_external_field(path):
        return ()
    if name == "Base":
        return (
            (source_namespace,)
            if source_namespace != "EXT"
            else ("CHARACTER", "BUILDING", "PROJECTILE", "AEO", "BUFF", "ABILITY")
        )
    if name in _ACTION_FIELDS or name.endswith("Action") or name.endswith("Actions"):
        return ("ACTION", "EXT")
    if name == "SpawnData":
        spawn_type = str(parent.get("SpawnType", ""))
        if spawn_type in {"CharacterType", "BuildingType"}:
            return _UNIT_NAMESPACES
        if spawn_type in {"AreaEffectType", "AreaEffectObjectType"}:
            return ("AEO", "EXT")
        if spawn_type == "BuffType":
            return ("BUFF", "EXT")
        if spawn_type == "ProjectileType":
            return ("PROJECTILE", "EXT")
        if spawn_type == "AbilityType":
            return ("ABILITY", "EXT")
        if spawn_type == "ActionType":
            return ("ACTION",)
        return ("CHARACTER", "BUILDING", "PROJECTILE", "AEO", "BUFF", "ACTION", "EXT")
    if name in {
        "AttachedCharacter",
        "ActivationSpawnCharacter",
        "ChampionCharacterData",
        "ClonedVersion",
        "DeathSpawnCharacter",
        "DeathSpawnCharacter2",
        "DeathSpawnCharacter3",
        "DeflectedCharacterSpawn",
        "DeathSpawn",
        "LinkedChampionCharacter",
        "LeftSummonType",
        "MorphCharacter",
        "MorphTarget",
        "NewCharacterData",
        "SpawnCharacter",
        "SpawnCharacter2",
        "SpawnCharacter3",
        "SpawnCharacterWithDeploy",
        "SpawnObject",
        "SpawnPathfindMorph",
        "SummonCharacter",
        "SummonCharacterSecond",
        "SummonCharactersList",
        "RightSummonType",
    }:
        return _UNIT_NAMESPACES
    if name in {
        "CustomFirstProjectile",
        "BombProjectile",
        "DeathSpawnProjectile",
        "NewProjectileData",
        "OverrideProjectile",
        "Projectile",
        "Projectile2",
        "Projectile3",
        "ProjectileSpecial",
        "ProjectileType",
        "Projectiles",
        "ReRollProjectile",
        "SpecialProjectile",
        "SpawnProjectile",
    }:
        return ("PROJECTILE", "EXT")
    if name in {
        "AppearAreaObject",
        "Aeo",
        "AeoList",
        "AreaEffectObject",
        "AreaEffectOnDash",
        "AreaEffectOnHit",
        "AreaEffectOnMorph",
        "BombAreaEffectObjects",
        "ContainerAeoList",
        "DamageAEO",
        "DeathAreaEffectData",
        "DeathAreaEffect",
        "LeftSummonAreaType",
        "RightSummonAreaType",
        "SpawnAreaEffectObject",
        "SpawnAreaObject",
        "SpawnsAEO",
        "TargetAoE",
    }:
        return ("AEO", "EXT")
    if name in {
        "Buff",
        "AttachedInheritAs",
        "BuffDuringCapture",
        "BuffOverride",
        "BuffToConsider",
        "BuffAfterHits",
        "BuffOnHit",
        "BuffOn50HP",
        "BuffOnDamage",
        "BuffOnKill",
        "BuffOnXHP",
        "BuffWhenNotAttacking",
        "ChainPhaseBuff",
        "ChainedBuff",
        "CrownTowerBuff",
        "IgnoreTargetsWithBuff",
        "IgnoreBuff",
        "PendingBuff",
        "ReflectedAttackBuff",
        "StartingBuff",
        "TargetBuff",
        "ValidTargetBuff",
    }:
        return ("BUFF", "EXT")
    if name in {"Ability", "HeroAbility"}:
        return ("ABILITY", "EXT")
    if name == "EvolvedSpells":
        # Hero-form fragments append hero spells to the same explicit list as
        # ordinary evolutions.  The summoned evolved/hero unit is reached from
        # the selected spell node; EXT is not itself a spell record.
        return ("SPELL_EVOLVED", "SPELL_HERO")
    if name in {"DeathSpell", "SpellAsDeploy", "SpellData"}:
        return _SPELL_NAMESPACES
    if name in {"Stats", "InBattleStats"}:
        return ("STATS",)
    if name in {"BaseDamageType", "DamageType"} or name.endswith("DamageType"):
        return ("DAMAGE_TYPE",)
    if name in {
        "Filter",
        "GameObjectFilter",
        "ObjectFilter",
        "SnipeTargetFilter",
        "TargetFilter",
        "TetherDamageTargets",
        "TroopFilter",
    }:
        return ("FILTER", "TARGET_RESOLVER")
    if name in {"TargetResolver", "Resolver"}:
        return ("TARGET_RESOLVER", "FILTER")
    if name == "Shape":
        return ("SHAPE",)
    if name == "Variable":
        return ("VARIABLE",)
    return ()


def _walk_string_fields(
    value: Any, prefix: str = "", parent: Mapping[str, Any] | None = None
) -> Iterable[tuple[str, str, Mapping[str, Any]]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _walk_string_fields(item, child, value)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _walk_string_fields(item, f"{prefix}[{index}]", parent)
    elif isinstance(value, str) and value:
        yield prefix, value, parent or {}


def _resolve_reference(
    *,
    source_node_id: str,
    field_path: str,
    raw_value: str,
    expected: tuple[str, ...],
    nodes: Mapping[str, Mapping[str, Any]],
    names: Mapping[str, Mapping[str, tuple[str, ...]]],
    token: str | None = None,
    rule: str,
) -> dict[str, Any]:
    lookup = token or raw_value
    qualified = _QUALIFIED_REFERENCE.fullmatch(lookup)
    targets: tuple[str, ...]
    if qualified:
        candidate = f"{qualified.group(1)}.{qualified.group(2)}"
        if candidate in nodes:
            targets = (candidate,)
        else:
            # Per-card files encode typed derived records as ``EXT.Name`` but
            # their Base links continue to use the semantic namespace, e.g.
            # ``PROJECTILE.OtherDerivedProjectile``.  Recover that explicit
            # alias only when the EXT record's own Base declares the same
            # namespace; a mere name match is not sufficient evidence.
            alias = f"EXT.{qualified.group(2)}"
            alias_record = nodes.get(alias, {}).get("merged_record", {})
            alias_base = alias_record.get("Base") if isinstance(alias_record, Mapping) else None
            alias_base_match = _QUALIFIED_REFERENCE.fullmatch(alias_base) if isinstance(alias_base, str) else None
            targets = (alias,) if alias_base_match and alias_base_match.group(1) == qualified.group(1) else ()
            if targets:
                rule = "qualified_ext_semantic_alias"
            elif qualified.group(1) == "CHARACTER":
                # ``CHARACTER`` is the TOML semantic namespace for native
                # LogicCharacterData, whose physical source sections include
                # both CHARACTER and BUILDING.  Across the active corpus all
                # nine otherwise-missing CHARACTER bases have one exact
                # BUILDING namesake and explicitly set IsBuilding=true.
                building = f"BUILDING.{qualified.group(2)}"
                building_record = nodes.get(building, {}).get("merged_record", {})
                targets = (
                    (building,)
                    if isinstance(building_record, Mapping) and building_record.get("IsBuilding") is True
                    else ()
                )
                if targets:
                    rule = "qualified_character_data_building_alias"
    else:
        found: set[str] = set()
        for namespace in expected:
            found.update(names.get(namespace, {}).get(lookup, ()))
        targets = tuple(sorted(found))
    status = "resolved" if len(targets) == 1 else ("ambiguous" if targets else "unresolved")
    field_name = _field_name(field_path)
    target_action_classes = {
        str(_node_record(nodes[target]).get("ClassType", ""))
        for target in targets
        if target in nodes and str(nodes[target].get("inheritance_root_node_id", target)).split(".", 1)[0] == "ACTION"
    }
    if field_name == "Base":
        reference_class = "inheritance"
    elif expected == ("STATS",):
        reference_class = "stats_metadata"
    elif targets and target_action_classes and target_action_classes.issubset(_VISUAL_ACTION_CLASS_TYPES):
        reference_class = "visual_native_action"
    else:
        reference_class = "gameplay"
    if reference_class == "inheritance":
        edge_role = "inheritance"
    elif reference_class == "stats_metadata":
        edge_role = "data_dependency"
    elif "ACTION" in expected or target_action_classes:
        edge_role = (
            "data_dependency"
            if field_name in {"ActionToGetDataFrom", "ActionToGetTargetFrom", "ActionToTakeDataFrom"}
            else "control"
        )
    elif reference_class == "visual_native_action":
        edge_role = "visual"
    elif field_name in _DATA_DEPENDENCY_FIELDS or set(expected).intersection({"FILTER", "TARGET_RESOLVER", "VARIABLE"}):
        edge_role = "data_dependency"
    else:
        edge_role = "resource"
    payload: dict[str, Any] = {
        "source_node_id": source_node_id,
        "field_path": field_path,
        "raw_value": raw_value,
        "expected_namespaces": expected,
        "target_node_ids": targets,
        "status": status,
        "resolution_rule": rule,
        "reference_class": reference_class,
        "edge_role": edge_role,
        "target_visual": reference_class == "visual_native_action",
        "gameplay_reference": reference_class == "gameplay",
    }
    if token is not None:
        payload["expression_token"] = token
    return {"reference_id": content_hash(payload), **payload}


def _build_references(
    nodes: Mapping[str, Mapping[str, Any]], *, record_field: str = "merged_record"
) -> tuple[dict[str, Any], ...]:
    names: dict[str, dict[str, tuple[str, ...]]] = {}
    grouped: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for node_id, node in nodes.items():
        grouped[str(node["namespace"])][str(node["name"])].append(node_id)
    for namespace, values in grouped.items():
        names[namespace] = {name: tuple(sorted(node_ids)) for name, node_ids in values.items()}

    references: dict[str, dict[str, Any]] = {}
    for node_id, node in sorted(nodes.items()):
        namespace = str(node["namespace"])
        record = node[record_field]
        for path, raw, parent in _walk_string_fields(record):
            expected = _expected_namespaces(namespace, path, parent)
            qualified = _QUALIFIED_REFERENCE.fullmatch(raw)
            if qualified and qualified.group(1) in LOGIC_NAMESPACES:
                expected = (qualified.group(1),)
                rule = "qualified_node_id"
            elif expected:
                rule = "field_schema"
            else:
                continue
            reference = _resolve_reference(
                source_node_id=node_id,
                field_path=path,
                raw_value=raw,
                expected=expected,
                nodes=nodes,
                names=names,
                rule=rule,
            )
            references[reference["reference_id"]] = reference

            # Conditions and value expressions can depend on explicitly
            # declared VARIABLE nodes.  Record only tokens that actually name
            # one; function names and enum literals are ignored.
        if namespace == "ACTION":
            for path, raw, _parent in _walk_string_fields(record):
                if _field_name(path) not in {
                    "AliveIfTrue",
                    "ChainCompleteIfTrue",
                    "Condition",
                    "ExecuteIfFalse",
                    "ExecuteIfTrue",
                    "ForceStopIfTrue",
                    "Value",
                }:
                    continue
                for token in sorted(set(_EXPRESSION_TOKEN.findall(raw))):
                    if token not in names.get("VARIABLE", {}):
                        continue
                    reference = _resolve_reference(
                        source_node_id=node_id,
                        field_path=path,
                        raw_value=raw,
                        expected=("VARIABLE",),
                        nodes=nodes,
                        names=names,
                        token=token,
                        rule="expression_variable_token",
                    )
                    references[reference["reference_id"]] = reference
    return tuple(
        sorted(
            references.values(),
            key=lambda item: (
                item["source_node_id"],
                item["field_path"],
                item.get("expression_token", ""),
                item["reference_id"],
            ),
        )
    )


def _materialize_effective_nodes(
    nodes: Mapping[str, Mapping[str, Any]], references: Iterable[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Return nodes with Base inheritance applied exactly once.

    Base remains in the record as provenance, but downstream control walkers
    classify it as inheritance and must not execute it as a second behaviour
    branch.
    """

    inheritance: dict[str, str] = {}
    for reference in references:
        if reference.get("reference_class") != "inheritance" or reference.get("status") != "resolved":
            continue
        targets = tuple(str(value) for value in reference.get("target_node_ids", ()))
        if len(targets) != 1:
            continue
        source = str(reference["source_node_id"])
        previous = inheritance.get(source)
        if previous is not None and previous != targets[0]:
            raise ContractError(f"static logic node {source} has conflicting Base targets")
        inheritance[source] = targets[0]

    resolved: dict[str, dict[str, Any]] = {}
    active: list[str] = []

    def resolve(node_id: str) -> dict[str, Any]:
        cached = resolved.get(node_id)
        if cached is not None:
            return cached
        if node_id in active:
            cycle = " -> ".join((*active[active.index(node_id) :], node_id))
            raise ContractError(f"static logic inheritance cycle: {cycle}")
        active.append(node_id)
        raw_node = nodes[node_id]
        raw_record = raw_node.get("merged_record")
        if not isinstance(raw_record, Mapping):
            raise ContractError(f"static logic node {node_id} lacks merged_record")
        base_id = inheritance.get(node_id)
        if base_id is None:
            effective = dict(raw_record)
            path = (node_id,)
            root = node_id
        else:
            base_node = resolve(base_id)
            effective = merge_effective_record(base_node["effective_record"], raw_record)
            path = (*base_node["inheritance_path"], node_id)
            root = str(base_node["inheritance_root_node_id"])
        active.pop()
        value = dict(raw_node)
        value["effective_record"] = effective
        value["inheritance_path"] = tuple(path)
        value["inheritance_root_node_id"] = root
        resolved[node_id] = value
        return value

    for node_id in sorted(nodes):
        resolve(node_id)
    return resolved


def _resolve_mode_scoped_supplemental_references(
    references: tuple[dict[str, Any], ...], assets_scdb: Path, *, source_name: str | None = None
) -> tuple[tuple[dict[str, Any], ...], dict[str, dict[str, Any]]]:
    """Resolve otherwise-missing gameplay names from unique SCDB JSON records.

    SCDB logic assets are mode-scoped and therefore never become ordinary
    global graph nodes.  This pass only annotates an already-unresolved
    gameplay edge after an exact namespace/name lookup has one JSON-parseable
    record across the database.  Duplicate definitions remain unresolved.
    """

    unresolved = {
        (namespace, str(reference["raw_value"]))
        for reference in references
        if reference["status"] == "unresolved" and bool(reference["gameplay_reference"])
        for namespace in reference["expected_namespaces"]
    }
    if not unresolved or not assets_scdb.is_file():
        return references, {}

    candidates: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    invalid_candidate_assets: set[tuple[str, str]] = set()
    try:
        connection = sqlite3.connect(f"{assets_scdb.resolve().as_uri()}?mode=ro", uri=True)
        try:
            rows = connection.execute(
                "SELECT a.path, a.data, m.assetType "
                "FROM asset AS a JOIN meta AS m USING (path) "
                "WHERE a.data IS NOT NULL ORDER BY a.path"
            )
            for asset_path, raw_data, asset_type in rows:
                raw_bytes = bytes(raw_data)
                try:
                    document = json.loads(raw_bytes.decode("utf-8-sig"))
                except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
                    # A malformed blob that contains a requested name prevents
                    # proof of uniqueness.  It may hold another definition we
                    # cannot safely inspect, so leave that reference unresolved.
                    for namespace, name in unresolved:
                        if name.encode("utf-8") in raw_bytes:
                            invalid_candidate_assets.add((namespace, name))
                    continue
                data = document.get("data") if isinstance(document, Mapping) else None
                if not isinstance(data, Mapping):
                    continue
                for namespace, name in unresolved.intersection(
                    (str(group), str(item_name))
                    for group, records in data.items()
                    if isinstance(records, Mapping)
                    for item_name in records
                ):
                    record = data[namespace][name]
                    if not isinstance(record, Mapping):
                        continue
                    candidates[(namespace, name)].append(
                        {
                            "namespace": namespace,
                            "name": name,
                            "record": dict(record),
                            "asset_path": str(asset_path),
                            "asset_type": str(asset_type),
                        }
                    )
        finally:
            connection.close()
    except sqlite3.Error:
        return references, {}

    scdb_sha256 = sha256_file(assets_scdb)
    source = source_name or assets_scdb.as_posix()
    supplemental: dict[str, dict[str, Any]] = {}
    result: list[dict[str, Any]] = []
    for reference in references:
        matches = [
            candidate
            for namespace in reference["expected_namespaces"]
            for candidate in candidates.get((str(namespace), str(reference["raw_value"])), ())
        ]
        if (
            reference["status"] != "unresolved"
            or not bool(reference["gameplay_reference"])
            or len(matches) != 1
            or any(
                (str(namespace), str(reference["raw_value"])) in invalid_candidate_assets
                for namespace in reference["expected_namespaces"]
            )
        ):
            result.append(reference)
            continue
        match = matches[0]
        supplemental_id = f"{match['namespace']}.{match['name']}"
        provenance = {
            **match,
            "node_id": supplemental_id,
            "scope": "mode_scoped",
            "source": source,
            "source_sha256": scdb_sha256,
            "source_record": (f"{source}#asset.path={match['asset_path']}#data.{match['namespace']}.{match['name']}"),
        }
        supplemental[supplemental_id] = provenance
        payload = {key: value for key, value in reference.items() if key != "reference_id"}
        payload.update(
            {
                "target_node_ids": (),
                "status": "resolved_supplemental_mode_scoped",
                "resolution_rule": "unique_scdb_json_mode_scoped",
                "supplemental_target_ids": (supplemental_id,),
                "supplemental_source_records": (provenance["source_record"],),
                "normal_mode_runtime_available": None,
                "requires_native_runtime_evidence": True,
            }
        )
        result.append({"reference_id": content_hash(payload), **payload})
    return tuple(result), supplemental


def _rarity_curves(nodes: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for node_id, node in sorted(nodes.items()):
        if node["namespace"] != "RARITY":
            continue
        record = node["merged_record"]
        level_count = record.get("LevelCount")
        relative_level = record.get("RelativeLevel")
        tournament_index = record.get("TournamentLevelIndex")
        if not all(isinstance(value, int) for value in (level_count, relative_level, tournament_index)):
            continue
        continuation = record.get("__continuation_rows__", ())
        configured = []
        if isinstance(record.get("PowerLevelMultiplier"), int):
            configured.append(record["PowerLevelMultiplier"])
        if isinstance(continuation, (list, tuple)):
            configured.extend(
                row["PowerLevelMultiplier"]
                for row in continuation
                if isinstance(row, Mapping) and isinstance(row.get("PowerLevelMultiplier"), int)
            )
        active = (100, *configured[: max(0, level_count - 1)])
        card_levels = tuple(relative_level + index + 1 for index in range(level_count))
        unknown: list[str] = []
        if len(active) != level_count:
            unknown.append("active_power_level_multiplier_rows")
        result[str(node["name"])] = {
            "node_id": node_id,
            "level_count": level_count,
            "relative_level": relative_level,
            "tournament_level_index": tournament_index,
            "tournament_card_level": relative_level + tournament_index + 1,
            "card_levels": card_levels,
            "active_power_level_multipliers_percent": active,
            "configured_power_level_multipliers_percent": tuple(configured),
            "projection_formula": "base_value * multiplier / 100",
            "rounding_rule": None,
            "unknown_fields": tuple(unknown) + ("engine_rounding_rule",),
            "source_records": tuple(fragment["source_record"] for fragment in node["fragments"]),
        }
    return result


def _numeric_exact_leaves(value: Any, prefix: str = "") -> Iterable[tuple[str, int | float]]:
    """Yield numeric leaves while keeping percentage overrides non-scalar.

    The decoded overlay representation uses ``("%", value)`` for relative
    overrides.  Its numeric member is not an independently level-scalable base
    value, so descending into that pair would silently turn (for example) a
    125-percent override into raw damage 125.
    """

    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _numeric_exact_leaves(item, child)
        return
    if isinstance(value, (list, tuple)):
        if (
            len(value) == 2
            and value[0] == "%"
            and isinstance(value[1], (int, float))
            and not isinstance(value[1], bool)
        ):
            return
        for index, item in enumerate(value):
            yield from _numeric_exact_leaves(item, f"{prefix}[{index}]")
        return
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and (not isinstance(value, float) or math.isfinite(value))
    ):
        yield prefix, value


def _resolved_inheritance_targets(references: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """Return the unique, already schema-resolved Base edge per source node."""

    result: dict[str, str] = {}
    for reference in references:
        if (
            reference.get("reference_class") != "inheritance"
            or reference.get("field_path") != "Base"
            or reference.get("status") != "resolved"
        ):
            continue
        targets = tuple(str(item) for item in reference.get("target_node_ids", ()))
        if len(targets) != 1:
            continue
        source = str(reference.get("source_node_id") or "")
        previous = result.get(source)
        if previous is not None and previous != targets[0]:
            raise ContractError(f"static node {source} has conflicting resolved Base targets: {previous}, {targets[0]}")
        result[source] = targets[0]
    return result


def _carrier_rarity_evidence(
    node_id: str,
    nodes: Mapping[str, Mapping[str, Any]],
    rarity_curves: Mapping[str, Mapping[str, Any]],
    inheritance_targets: Mapping[str, str],
) -> dict[str, Any]:
    """Resolve one numeric carrier's rarity through exact Base edges."""

    current = node_id
    inheritance_path: list[str] = []
    seen: set[str] = set()
    while True:
        if current in seen:
            raise ContractError(
                f"static carrier rarity inheritance cycle from {node_id}: {' -> '.join((*inheritance_path, current))}"
            )
        seen.add(current)
        node = nodes.get(current)
        record = node.get("merged_record") if isinstance(node, Mapping) else None
        if not isinstance(record, Mapping):
            return {
                "carrier_projection_status": "unresolved",
                "carrier_rarity": None,
                "carrier_rarity_source_node_id": None,
                "carrier_rarity_resolution": None,
                "carrier_inheritance_path": tuple(inheritance_path),
                "projection_blocker": "carrier_node_missing",
            }
        raw_rarity = record.get("Rarity")
        if raw_rarity is not None:
            if not isinstance(raw_rarity, str) or raw_rarity not in rarity_curves:
                raise ContractError(f"static numeric carrier {current} declares unsupported rarity {raw_rarity!r}")
            return {
                "carrier_projection_status": "resolved",
                "carrier_rarity": raw_rarity,
                "carrier_rarity_source_node_id": current,
                "carrier_rarity_resolution": ("direct" if current == node_id else "base_inheritance"),
                "carrier_inheritance_path": tuple((*inheritance_path, current)),
            }
        target = inheritance_targets.get(current)
        if target is None:
            return {
                "carrier_projection_status": "unresolved",
                "carrier_rarity": None,
                "carrier_rarity_source_node_id": None,
                "carrier_rarity_resolution": None,
                "carrier_inheritance_path": tuple((*inheritance_path, current)),
                "projection_blocker": "carrier_rarity_not_declared_or_inherited",
            }
        inheritance_path.append(current)
        current = target


def _carrier_level_11_projection(
    base_value: int | float, rarity: str, rarity_curves: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Bind exact level-11 curve arithmetic to one resolved carrier."""

    curve = rarity_curves.get(rarity)
    if not isinstance(curve, Mapping):
        raise ContractError(f"static numeric carrier has no rarity curve for {rarity}")
    levels = tuple(curve.get("card_levels", ()))
    multipliers = tuple(curve.get("active_power_level_multipliers_percent", ()))
    if len(levels) != len(multipliers) or 11 not in levels:
        raise ContractError(f"static numeric carrier rarity {rarity} has no complete level-11 curve")
    multiplier = multipliers[levels.index(11)]
    if isinstance(multiplier, bool) or not isinstance(multiplier, int):
        raise ContractError(f"static numeric carrier rarity {rarity} has invalid level-11 multiplier")
    # v15's integral combat-stat accessor truncates toward zero after applying
    # the percent curve.  Keep the source base and multiplier alongside the
    # normalized value so consumers never need the card root's rarity.
    level_11_value = math.trunc(base_value * multiplier / 100)
    return {
        "carrier_curve": {
            "rarity": rarity,
            "node_id": curve["node_id"],
            "card_levels": levels,
            "power_level_multipliers_percent": multipliers,
        },
        "level_11_multiplier_percent": multiplier,
        "level_11_value": level_11_value,
        "level_11_rounding_rule": "integer_truncation_toward_zero",
    }


def _field_level_carrier_projections(
    card_id: int, numeric_features: Mapping[str, Any], candidates: Iterable[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Select unique carrier projections for compact scalable fields.

    A compact field/value can appear in multiple graph nodes.  Equal carrier
    rarity is redundant provenance; different rarities are semantically
    ambiguous and must stop publication rather than fall back to card rarity.
    """

    source_fields = {"hitpoints": "Hitpoints", "damage": "Damage", "shield_hitpoints": "ShieldHitpoints"}
    candidate_tuple = tuple(candidates)
    result: dict[str, dict[str, Any]] = {}
    for compact_field, source_field in source_fields.items():
        base_value = numeric_features.get(compact_field)
        if (
            isinstance(base_value, bool)
            or not isinstance(base_value, (int, float))
            or (isinstance(base_value, float) and not math.isfinite(base_value))
        ):
            continue
        matches = tuple(
            candidate
            for candidate in candidate_tuple
            if _field_name(str(candidate.get("field_path") or "")).casefold() == source_field.casefold()
            and candidate.get("base_value") == base_value
        )
        resolved = tuple(candidate for candidate in matches if candidate.get("carrier_projection_status") == "resolved")
        rarities = {str(candidate["carrier_rarity"]) for candidate in resolved}
        if len(rarities) > 1:
            evidence = ", ".join(
                f"{candidate['node_id']}.{candidate['field_path']}={candidate['carrier_rarity']}"
                for candidate in resolved
            )
            raise ContractError(
                f"card {card_id} field {compact_field} value {base_value} has conflicting carrier rarities: {evidence}"
            )
        if not resolved:
            result[compact_field] = {
                "status": "unresolved",
                "base_value": base_value,
                "source_field": source_field,
                "source_candidates": tuple(
                    {
                        "node_id": candidate.get("node_id"),
                        "field_path": candidate.get("field_path"),
                        "projection_blocker": candidate.get("projection_blocker"),
                    }
                    for candidate in matches
                ),
                "projection_blocker": (
                    "matching_carrier_rarity_unresolved" if matches else "no_matching_level_scaling_candidate"
                ),
            }
            continue
        selected = resolved[0]
        result[compact_field] = {
            "status": "resolved",
            "base_value": base_value,
            "source_field": source_field,
            "carrier_rarity": selected["carrier_rarity"],
            "carrier_curve": selected["carrier_curve"],
            "level_11_multiplier_percent": selected["level_11_multiplier_percent"],
            "level_11_value": selected["level_11_value"],
            "level_11_rounding_rule": selected["level_11_rounding_rule"],
            "source_candidates": tuple(
                {
                    "node_id": candidate["node_id"],
                    "field_path": candidate["field_path"],
                    "carrier_rarity_source_node_id": candidate["carrier_rarity_source_node_id"],
                }
                for candidate in resolved
            ),
        }
    return result


def _numeric_scaling_candidates(
    closure: Iterable[str],
    nodes: Mapping[str, Mapping[str, Any]],
    rarity_curves: Mapping[str, Mapping[str, Any]],
    inheritance_targets: Mapping[str, str],
) -> tuple[dict[str, Any], ...]:
    candidates: list[dict[str, Any]] = []
    for node_id in sorted(closure):
        record = _node_record(nodes[node_id])
        carrier = _carrier_rarity_evidence(node_id, nodes, rarity_curves, inheritance_targets)
        for path, value in _numeric_exact_leaves(record):
            name = _field_name(path).lower()
            family: str | None = None
            evidence = ""
            if ("hitpoint" in name or name in {"shield", "health"}) and not any(
                marker in name for marker in ("multiplier", "percent", "offset", "bar")
            ):
                family = "hitpoint"
                evidence = "global_hitpoint_scaling_family_and_field_name"
            elif "damage" in name and not any(
                marker in name for marker in ("percent", "multiplier", "reduction", "radius", "time", "type", "effect")
            ):
                family = "damage"
                evidence = "global_damage_scaling_family_and_field_name"
            elif "heal" in name and not any(marker in name for marker in ("percent", "effect", "time")):
                family = "heal_unbound"
                evidence = "field_name_only_no_decoded_global_scaling_rule"
            if family is None:
                continue
            projection = (
                _carrier_level_11_projection(value, str(carrier["carrier_rarity"]), rarity_curves)
                if carrier["carrier_projection_status"] == "resolved"
                else {
                    "carrier_curve": None,
                    "level_11_multiplier_percent": None,
                    "level_11_value": None,
                    "level_11_rounding_rule": None,
                }
            )
            candidates.append(
                {
                    "node_id": node_id,
                    "field_path": path,
                    "base_value": value,
                    "base_value_kind": "exact_numeric",
                    "scaling_family": family,
                    "evidence": evidence,
                    **carrier,
                    **projection,
                }
            )
    return tuple(sorted(candidates, key=lambda item: (item["node_id"], item["field_path"])))


def _mechanic_families(closure: Iterable[str], nodes: Mapping[str, Mapping[str, Any]]) -> dict[str, tuple[str, ...]]:
    """Classify only explicit namespaces, fields, and native action classes."""

    gameplay: set[str] = set()
    visual_actions: set[str] = set()
    for node_id in closure:
        node = nodes[node_id]
        namespace = str(node["namespace"])
        record = _node_record(node)
        if namespace == "ABILITY":
            gameplay.add("active_ability_definition")
        elif namespace == "AEO":
            gameplay.add("area_effect_definition")
        elif namespace == "BUFF":
            gameplay.add("buff_or_debuff_definition")
        elif namespace == "PROJECTILE":
            gameplay.add("projectile_definition")
        elif namespace == "SPELL_EVOLVED":
            gameplay.add("evolution_definition")
        class_type = str(record.get("ClassType", ""))
        if namespace in {"ACTION", "EXT"} and class_type.startswith("Action"):
            if class_type in _VISUAL_ACTION_CLASS_TYPES:
                visual_actions.add(class_type)
            else:
                gameplay.add(f"native_action:{class_type}")
                classification = classify_native_action(record)
                if classification is not None:
                    gameplay.add(f"typed_native_action:{classification['operation']}")
                    gameplay.update(f"field:{tag}" for tag in classification["mechanic_tags"])
        for path, value in _flatten_leaves(record).items():
            family = _MECHANIC_FIELD_FAMILIES.get(_field_name(path))
            if family is not None and _mechanic_field_is_active(_field_name(path), value):
                gameplay.add(f"field:{family}")
    return {"gameplay": tuple(sorted(gameplay)), "visual_native_actions": tuple(sorted(visual_actions))}


def _global_scaling_inputs(nodes: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    wanted = (
        "DAMAGE_INCREASE_PERCENT_PER_SPELL_LEVEL",
        "DAMAGE_INCREASE_PERCENT_PER_SPELL_LEVEL_AFTER_TOURNAMENTCAP",
        "HITPOINT_INCREASE_PERCENT_PER_SPELL_LEVEL",
        "HITPOINT_INCREASE_PERCENT_PER_SPELL_LEVEL_AFTER_TOURNAMENTCAP",
    )
    result: dict[str, Any] = {}
    for name in wanted:
        node = nodes.get(f"GLOBAL.{name}")
        if node is not None:
            result[name] = node["merged_record"].get("NumberValue")
    return result


def _native_loader_defaults(workspace: Path, source: Path, source_release: str) -> dict[str, Any]:
    """Return only defaults proved for the exact bound ARM64 binary."""

    if not _matches_native_character_default_build(workspace, source, source_release):
        return {}
    return {
        "LogicAreaEffectObjectData.Radius": {
            "value": 0,
            "source_release": _NATIVE_CHARACTER_DEFAULT_RELEASE,
            "binary_sha256": _NATIVE_CHARACTER_DEFAULT_BINARY_SHA256,
            "evidence": (
                "native_loader_default:LogicAreaEffectObjectData.Radius=0",
                "libg.arm64-v15.535.13@0xd7b6f0:Radius",
                "libg.arm64-v15.535.13@0xd7b6fc:integer_default=0",
                "libg.arm64-v15.535.13@0xd7b704:store=data+0x17c",
            ),
        },
        "LogicProjectileData.SpawnCharacterCount": {
            "value": 1,
            "raw_default": 0,
            "condition": "SpawnCharacter!=null",
            "source_release": _NATIVE_CHARACTER_DEFAULT_RELEASE,
            "binary_sha256": _NATIVE_CHARACTER_DEFAULT_BINARY_SHA256,
            "evidence": (
                "native_effective_default:LogicProjectileData.SpawnCharacterCount=1",
                "libg.arm64-v15.535.13@0xe0889c:SpawnCharacterCount",
                "libg.arm64-v15.535.13@0xe088a8:integer_default=0",
                "libg.arm64-v15.535.13@0xe088b0:store=raw_data+0xcc",
                "libg.arm64-v15.535.13@0xe088dc..0xe088f4:SpawnCharacter_present_clamp_minimum=1",
            ),
        },
    }


def _card_records(
    catalog: CardSpecCatalog,
    nodes: Mapping[str, Mapping[str, Any]],
    references: tuple[Mapping[str, Any], ...],
    rarity_curves: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for reference in references:
        by_source[str(reference["source_node_id"])].append(reference)
    inheritance_targets = _resolved_inheritance_targets(references)
    result: list[dict[str, Any]] = []
    for spec in catalog.specs:
        namespace = _CARD_ROOT_NAMESPACE.get(spec.card_id // 1_000_000)
        root = f"{namespace}.{spec.name}" if namespace else ""
        closure: set[str] = set()
        if root in nodes:
            queue = deque([root])
            while queue:
                node_id = queue.popleft()
                if node_id in closure:
                    continue
                closure.add(node_id)
                for reference in by_source.get(node_id, ()):
                    if reference.get("edge_role") == "inheritance":
                        continue
                    # An ambiguous edge is still part of the lossless static
                    # closure: traverse every explicit candidate and retain
                    # the ambiguity marker for consumers that require a
                    # single native-exact interpretation.
                    if reference["status"] in {"resolved", "resolved_supplemental_mode_scoped", "ambiguous"}:
                        for raw_target in reference["target_node_ids"]:
                            target = str(raw_target)
                            if target not in closure:
                                queue.append(target)
        closure_references = tuple(reference for node_id in closure for reference in by_source.get(node_id, ()))
        unresolved = tuple(
            sorted(
                str(reference["reference_id"])
                for reference in closure_references
                if reference["status"] == "unresolved"
            )
        )
        unresolved_gameplay = tuple(
            sorted(
                str(reference["reference_id"])
                for reference in closure_references
                if reference["status"] == "unresolved" and bool(reference["gameplay_reference"])
            )
        )
        unresolved_non_gameplay = tuple(
            sorted(
                str(reference["reference_id"])
                for reference in closure_references
                if reference["status"] == "unresolved" and not bool(reference["gameplay_reference"])
            )
        )
        ambiguous = tuple(
            sorted(
                str(reference["reference_id"]) for reference in closure_references if reference["status"] == "ambiguous"
            )
        )
        mode_scoped_supplemental = tuple(
            sorted(
                str(reference["reference_id"])
                for reference in closure_references
                if reference["status"] == "resolved_supplemental_mode_scoped"
            )
        )
        unknown = list(spec.unknown_fields)
        if root not in nodes:
            unknown.append("root_logic_node")
        root_node = nodes.get(root, {})
        root_record = _node_record(root_node) if root_node else {}
        if spec.kind.value in {"troop", "building", "hero"} and not any(
            key in root_record
            for key in ("SummonCharacter", "SummonCharactersList", "AreaEffectObject", "PostSpawnAction")
        ):
            unknown.append("deploy_form_reference_absent")
        if unresolved_gameplay:
            unknown.append("unresolved_gameplay_references")
        if unresolved_non_gameplay:
            unknown.append("unresolved_non_gameplay_references")
        if ambiguous:
            unknown.append("ambiguous_gameplay_references")
        rarity = str(spec.categorical_features.get("rarity") or "")
        curve = rarity_curves.get(rarity)
        level_projection = (
            {
                # This is the deployable card's own progression metadata.  It
                # must never be reused to project stats owned by CHARACTER,
                # PROJECTILE, AEO, or other carrier nodes in the card closure.
                "scope": "card_progression_only",
                "stat_value_projection_allowed": False,
                "stat_value_projection_source": "level_scaling_candidates",
                "rarity_curve": rarity,
                "card_levels": curve["card_levels"],
                "power_level_multipliers_percent": curve["active_power_level_multipliers_percent"],
                "projection_formula": curve["projection_formula"],
                "rounding_rule": curve["rounding_rule"],
            }
            if curve is not None
            else None
        )
        numeric_candidates = _numeric_scaling_candidates(closure, nodes, rarity_curves, inheritance_targets)
        field_level_projections = _field_level_carrier_projections(
            spec.card_id, spec.numeric_features, numeric_candidates
        )
        result.append(
            {
                "card_id": spec.card_id,
                "name": spec.name,
                "kind": spec.kind.value,
                "rarity": rarity or None,
                # A syntactically plausible namespace/name is only a candidate.
                # Custom/test catalogs can legitimately use names absent from
                # the decoded native registry; do not publish a dangling graph
                # root in that case.  The explicit root_logic_node unknown
                # marker above preserves the fail-closed reason.
                "root_node_id": root if root in nodes else None,
                "card_spec_sha256": content_hash(spec.to_dict()),
                "closure_node_ids": tuple(sorted(closure)),
                "closure_namespace_counts": dict(
                    sorted(Counter(node_id.split(".", 1)[0] for node_id in closure).items())
                ),
                "resolved_reference_count": sum(
                    str(reference["status"]).startswith("resolved") for reference in closure_references
                ),
                "gameplay_reference_count": sum(
                    bool(reference["gameplay_reference"]) for reference in closure_references
                ),
                "ambiguous_reference_ids": ambiguous,
                "unresolved_reference_ids": unresolved,
                "unresolved_gameplay_reference_ids": unresolved_gameplay,
                "unresolved_non_gameplay_reference_ids": unresolved_non_gameplay,
                "mode_scoped_supplemental_reference_ids": (mode_scoped_supplemental),
                "mechanic_families": _mechanic_families(closure, nodes),
                "typed_native_actions": _typed_native_actions(closure, nodes),
                "normalized_base_numeric_features": dict(spec.numeric_features),
                "level_projection": level_projection,
                "field_level_projections": field_level_projections,
                "level_scaling_candidates": numeric_candidates,
                "unknown_fields": tuple(sorted(set(unknown))),
            }
        )
    return tuple(sorted(result, key=lambda item: item["card_id"]))


@dataclass(frozen=True, slots=True)
class StaticCardLogicCatalogV1(ContractMixin):
    """Immutable, lossless static logic graph bound to a CardSpec catalog."""

    source_release: str
    card_catalog_content_hash: str
    card_specs_hash: str
    ability_specs_hash: str
    criteria_version: str
    cards: tuple[Mapping[str, Any], ...]
    nodes: Mapping[str, Mapping[str, Any]]
    references: tuple[Mapping[str, Any], ...]
    rarity_curves: Mapping[str, Mapping[str, Any]]
    global_scaling_inputs: Mapping[str, Any]
    source_files: Mapping[str, Mapping[str, Any]]
    table_schemas: Mapping[str, Mapping[str, str]]
    supplemental_mode_scoped_nodes: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    source_placeholders: tuple[Mapping[str, Any], ...] = ()
    source_failures: tuple[Mapping[str, str], ...] = ()
    warnings: tuple[str, ...] = ()
    version: str = field(default=STATIC_CARD_LOGIC_VERSION, init=False)
    VERSION: ClassVar[str] = STATIC_CARD_LOGIC_VERSION

    def __post_init__(self) -> None:
        for label, value in (
            ("card_catalog_content_hash", self.card_catalog_content_hash),
            ("card_specs_hash", self.card_specs_hash),
            ("ability_specs_hash", self.ability_specs_hash),
        ):
            if not _SHA256.fullmatch(value):
                raise ContractError(f"{label} must be a lowercase SHA-256 digest")
        if self.criteria_version != STATIC_CARD_LOGIC_CRITERIA_VERSION:
            raise ContractError(f"unsupported static card logic criteria: {self.criteria_version}")
        cards = tuple(sorted((frozen_mapping(item) for item in self.cards), key=lambda item: int(item["card_id"])))
        if len({int(item["card_id"]) for item in cards}) != len(cards):
            raise ContractError("static card logic catalog has duplicate card IDs")
        nodes = frozen_mapping(self.nodes)
        for card in cards:
            root = card.get("root_node_id")
            if root is not None and root not in nodes:
                raise ContractError(f"card root missing from logic nodes: {root}")
        references = tuple(
            sorted((frozen_mapping(item) for item in self.references), key=lambda item: str(item["reference_id"]))
        )
        if len({str(item["reference_id"]) for item in references}) != len(references):
            raise ContractError("static card logic catalog has duplicate references")
        object.__setattr__(self, "cards", cards)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "references", references)
        object.__setattr__(self, "rarity_curves", frozen_mapping(self.rarity_curves))
        object.__setattr__(self, "global_scaling_inputs", frozen_mapping(self.global_scaling_inputs))
        object.__setattr__(self, "source_files", frozen_mapping(self.source_files))
        object.__setattr__(self, "table_schemas", frozen_mapping(self.table_schemas))
        object.__setattr__(self, "supplemental_mode_scoped_nodes", frozen_mapping(self.supplemental_mode_scoped_nodes))
        object.__setattr__(
            self, "source_placeholders", tuple(frozen_mapping(item) for item in self.source_placeholders)
        )
        object.__setattr__(self, "source_failures", tuple(frozen_mapping(item) for item in self.source_failures))
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))

    @property
    def catalog_id(self) -> str:
        return content_hash(self)

    @property
    def cards_by_name(self) -> dict[str, Mapping[str, Any]]:
        return {str(item["name"]): item for item in self.cards}

    @property
    def summary(self) -> dict[str, Any]:
        namespace_counts = Counter(str(node["namespace"]) for node in self.nodes.values())
        reference_counts = Counter(str(reference["status"]) for reference in self.references)
        reference_class_counts = Counter(str(reference["reference_class"]) for reference in self.references)
        unresolved_class_counts = Counter(
            str(reference["reference_class"]) for reference in self.references if reference["status"] == "unresolved"
        )
        return {
            "catalog_id": self.catalog_id,
            "card_count": len(self.cards),
            "rooted_card_count": sum(bool(card["closure_node_ids"]) for card in self.cards),
            "node_count": len(self.nodes),
            "supplemental_mode_scoped_node_count": len(self.supplemental_mode_scoped_nodes),
            "node_counts_by_namespace": dict(sorted(namespace_counts.items())),
            "reference_count": len(self.references),
            "reference_counts_by_status": dict(sorted(reference_counts.items())),
            "reference_counts_by_class": dict(sorted(reference_class_counts.items())),
            "unresolved_reference_counts_by_class": dict(sorted(unresolved_class_counts.items())),
            "cards_with_unresolved_references": sum(bool(card["unresolved_reference_ids"]) for card in self.cards),
            "cards_with_unresolved_gameplay_references": sum(
                bool(card["unresolved_gameplay_reference_ids"]) for card in self.cards
            ),
            "cards_with_unresolved_non_gameplay_references": sum(
                bool(card["unresolved_non_gameplay_reference_ids"]) for card in self.cards
            ),
            "cards_with_ambiguous_references": sum(bool(card["ambiguous_reference_ids"]) for card in self.cards),
            "cards_with_mode_scoped_supplemental_references": sum(
                bool(card.get("mode_scoped_supplemental_reference_ids")) for card in self.cards
            ),
            "cards_with_level_projection": sum(card["level_projection"] is not None for card in self.cards),
            "cards_with_unknown_fields": sum(bool(card["unknown_fields"]) for card in self.cards),
            "card_unknown_field_count": sum(len(card["unknown_fields"]) for card in self.cards),
            "typed_native_action_count": sum(len(card.get("typed_native_actions", ())) for card in self.cards),
            "cards_with_typed_native_actions": sum(bool(card.get("typed_native_actions")) for card in self.cards),
            "level_scaling_candidate_count": sum(len(card["level_scaling_candidates"]) for card in self.cards),
            "rarity_curve_count": len(self.rarity_curves),
            "source_file_count": len(self.source_files),
            "source_placeholder_count": len(self.source_placeholders),
            "source_failure_count": len(self.source_failures),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StaticCardLogicCatalogV1":
        if value.get("version") != cls.VERSION:
            raise ContractError(f"unsupported static card logic version: {value.get('version')}")
        return cls(
            source_release=str(value["source_release"]),
            card_catalog_content_hash=str(value["card_catalog_content_hash"]),
            card_specs_hash=str(value["card_specs_hash"]),
            ability_specs_hash=str(value["ability_specs_hash"]),
            criteria_version=str(value["criteria_version"]),
            cards=tuple(value.get("cards", ())),
            nodes=value.get("nodes", {}),
            references=tuple(value.get("references", ())),
            rarity_curves=value.get("rarity_curves", {}),
            global_scaling_inputs=value.get("global_scaling_inputs", {}),
            source_files=value.get("source_files", {}),
            table_schemas=value.get("table_schemas", {}),
            supplemental_mode_scoped_nodes=value.get("supplemental_mode_scoped_nodes", {}),
            source_placeholders=tuple(value.get("source_placeholders", ())),
            source_failures=tuple(value.get("source_failures", ())),
            warnings=tuple(value.get("warnings", ())),
        )

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_json(pretty=True) + "\n"
        if destination.exists():
            if destination.read_text(encoding="utf-8") != payload:
                raise FileExistsError(f"immutable static card logic path contains different bytes: {destination}")
            return destination
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(destination)
        return destination

    def save_content_addressed(self, directory: str | Path) -> Path:
        return self.save(Path(directory) / f"card-logic-{self.catalog_id}.json")

    @classmethod
    def load(cls, path: str | Path) -> "StaticCardLogicCatalogV1":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))


def build_static_card_logic_catalog(
    workspace_root: str | Path | None = None,
    *,
    source_roots: Iterable[str | Path] = (),
    card_catalog: CardSpecCatalog | None = None,
) -> StaticCardLogicCatalogV1:
    workspace = Path(workspace_root or WORKSPACE_ROOT).resolve()
    source = discover_card_source(workspace, source_roots)
    try:
        source_release = next(part for part in reversed(source.parts) if part.startswith("nr_"))
    except StopIteration:
        source_release = source.parent.name
    runtime_logic = workspace / "runtime-update" / "csv_logic"
    catalog = card_catalog or build_card_catalog(workspace, source_roots=source_roots)
    nodes, schemas, source_files, failures, placeholders = _load_logic_registry(workspace, source, runtime_logic)
    inheritance_references = _build_references(nodes)
    nodes = _materialize_effective_nodes(nodes, inheritance_references)
    references = _build_references(nodes, record_field="effective_record")
    assets_scdb = workspace / "runtime-update" / "assets.scdb"
    references, supplemental_mode_scoped_nodes = _resolve_mode_scoped_supplemental_references(
        references, assets_scdb, source_name=_relative(assets_scdb, workspace)
    )
    if supplemental_mode_scoped_nodes:
        source_files[_relative(assets_scdb, workspace)] = {
            "raw_sha256": sha256_file(assets_scdb),
            "raw_size": assets_scdb.stat().st_size,
            "classification": "scdb_mode_scoped_supplemental",
        }
    rarity_curves = _rarity_curves(nodes)
    cards = _card_records(catalog, nodes, references, rarity_curves)
    warnings: list[str] = []
    if failures:
        warnings.append(
            f"{len(failures)} relevant source files could not be decoded; each is listed in source_failures"
        )
    missing_roots = sum(not card["closure_node_ids"] for card in cards)
    if missing_roots:
        warnings.append(f"{missing_roots} cards have no resolved root logic node")
    global_scaling_inputs = _global_scaling_inputs(nodes)
    native_defaults = _native_loader_defaults(workspace, source, source_release)
    if native_defaults:
        global_scaling_inputs["native_loader_defaults"] = native_defaults
    return StaticCardLogicCatalogV1(
        # The decoded graph and Buff registry come from ``source`` even when a
        # caller supplies a synthetic/custom CardSpec catalog.  Bind the logic
        # artifact to that real native release; the independent
        # card_catalog_content_hash below still binds the supplied catalog.
        source_release=source_release,
        card_catalog_content_hash=catalog.content_hash(),
        card_specs_hash=catalog.specs_hash,
        ability_specs_hash=catalog.ability_specs_hash,
        criteria_version=STATIC_CARD_LOGIC_CRITERIA_VERSION,
        cards=cards,
        nodes=nodes,
        references=references,
        rarity_curves=rarity_curves,
        global_scaling_inputs=global_scaling_inputs,
        source_files=source_files,
        table_schemas=schemas,
        supplemental_mode_scoped_nodes=supplemental_mode_scoped_nodes,
        source_placeholders=placeholders,
        source_failures=failures,
        warnings=tuple(warnings),
    )
