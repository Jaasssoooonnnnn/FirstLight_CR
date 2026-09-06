"""Reproduce the pinned runtime datasets from locally supplied resources."""

import argparse
from dataclasses import fields, is_dataclass
from collections.abc import Mapping
from enum import Enum
import hashlib
import json
import os
from pathlib import Path


def normalize(value):
    if is_dataclass(value):
        return {f.name: normalize(getattr(value, f.name)) for f in fields(value) if f.init}
    if isinstance(value, Mapping):
        return {str(k): normalize(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [normalize(v) for v in value]
    if isinstance(value, Enum):
        return value.value
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["CR_WORKSPACE_ROOT"] = str(args.workspace.resolve())
    os.environ["CR_COMPETITIVE_DATA_ROOT"] = str(args.output.resolve())

    from native_runner.competitive_data import validate_competitive_sources
    from native_runner.contracts import content_hash
    from .card_specs import build_card_catalog
    from .card_logic import build_static_card_logic_catalog
    from .effect_catalog import build_effect_catalog
    from .projectile_catalog import build_projectile_catalog

    manifest = validate_competitive_sources(args.workspace)
    args.output.mkdir(parents=True, exist_ok=True)

    def write(name, value, *, contract=False):
        # Contract JSON uses sorted keys; model snapshots retain schema field order.
        raw = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=contract) + "\n").encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        if digest != manifest["files"][name]["sha256"]:
            raise ValueError(f"Generated {name} differs from the supported release: {digest}")
        destination = args.output / f"{name}.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_bytes(raw)
        temporary.replace(destination)
        print(f"Verified {name}: {digest}", flush=True)

    cards = build_card_catalog(args.workspace)
    logic = build_static_card_logic_catalog(args.workspace, card_catalog=cards)
    effects = build_effect_catalog(logic, args.workspace)
    projectiles = build_projectile_catalog(logic, args.workspace)
    for name, value in (
        ("card_specs", cards),
        ("card_logic", logic),
        ("effects", effects),
        ("projectiles", projectiles),
    ):
        write(name, value.to_dict(), contract=True)

    # Runtime form compatibility initialization can now read the four base files.
    from .catalog import CardCatalogV1, AbilityCatalogV1, EntityArchetypeCatalogV1, normal_mode_card_specs
    from .mechanics import EffectSemanticCatalogV1, MechanicProfileCatalogV1
    from .card_features import build_card_feature_rows

    specs = normal_mode_card_specs(cards.by_id, require_frozen_baseline=True)
    card_catalog = CardCatalogV1.from_card_spec_catalog(cards, static_logic=logic)
    abilities = AbilityCatalogV1.from_card_spec_catalog(cards)
    entities = EntityArchetypeCatalogV1.from_card_specs(specs, static_logic=logic, projectile_catalog=projectiles)
    semantic_effects = EffectSemanticCatalogV1.from_native_catalog(
        specs, static_logic=logic, native_effect_catalog=effects
    )
    mechanics = MechanicProfileCatalogV1.from_sources(
        specs,
        card_catalog=card_catalog,
        ability_catalog=abilities,
        entity_archetype_catalog=entities,
        effect_catalog=semantic_effects,
        static_logic=logic,
        native_effect_catalog=effects,
        projectile_catalog=projectiles,
    )
    catalogs = dict(
        card_catalog=card_catalog,
        ability_catalog=abilities,
        entity_archetype_catalog=entities,
        effect_catalog=semantic_effects,
        mechanic_profile_catalog=mechanics,
    )
    model = {
        "schema": "competitive-model-catalogs.v1",
        "source_release": cards.source_release,
        "card_spec_hashes": {str(k): content_hash(v.to_dict()) for k, v in specs.items()},
        "ability_spec_hashes": {
            v.ability_id: content_hash(v.to_dict()) for v in cards.abilities if v.ability_id in abilities.ability_ids
        },
        "card_feature_rows": normalize(
            build_card_feature_rows(specs, static_logic=logic, ability_specs=cards.abilities)
        ),
        "catalogs": {k: normalize(v) for k, v in catalogs.items()},
        "catalog_ids": {k: v.catalog_id for k, v in catalogs.items()},
        "input_catalog_ids": {
            k: manifest["files"][k]["content_hash"] for k in ("card_logic", "effects", "projectiles")
        },
        "description": "Exact level-11 feature rows, ordered vocabulary facts, and descriptive mechanic operations for the frozen competitive V4 checkpoint contract. Conditions and paths are categorical feature labels; the loader never executes them.",
    }
    write("model_catalogs", model)


if __name__ == "__main__":
    main()
