from __future__ import annotations

from native_runner.paths import WORKSPACE_ROOT
from native_runner.tests.asset_helpers import user_apk_bytes

import json
from dataclasses import replace

import pytest

from native_runner.card_specs import build_card_catalog, sha256_file
from native_runner.tests.asset_helpers import read_sc_csv as _read_sc_csv
from native_runner.contracts import (
    EntityStateV1,
    ObservationTier,
    ObservationV1,
    PlayerStateV1,
)
from native_runner.normal_form_evidence import (
    NATIVE_CHAMPION_ABILITY_BINDINGS,
    NATIVE_CHAMPION_ABILITY_BINDINGS_SHA256,
    NATIVE_HERO_FORM_ABILITY_BINDINGS,
    NATIVE_HERO_FORM_ABILITY_BINDINGS_SHA256,
    NORMAL_MODE_DIRECT_HERO_BINDINGS_BY_CARD_ID,
    NORMAL_MODE_DIRECT_HERO_COUNT,
    NORMAL_MODE_EVOLUTION_FORM_COUNT,
    NORMAL_MODE_HERO_FORM_BINDINGS_BY_CARD_ID,
    NORMAL_MODE_HERO_FORM_COUNT,
    NORMAL_MODE_HERO_FORM_TABLE_RELATIVE_PATH,
    NORMAL_MODE_HERO_FORM_TABLE_SHA256,
    NORMAL_MODE_HERO_FORM_TO_BASE_CARD,
    NORMAL_MODE_POLICY_FORM_CARD_COUNT,
    NORMAL_MODE_FORM_POLICY_CRITERIA_VERSION,
    build_normal_mode_policy_form_evidence,
)
from native_runner.royaleapi_replay import (
    NATIVE_CHAMPION_ABILITY_BINDINGS as REPLAY_CHAMPION_BINDINGS,
    NATIVE_HERO_FORM_ABILITY_BINDINGS as REPLAY_HERO_BINDINGS,
)
import native_runner.royaleapi_replay as royaleapi_replay
import native_runner.semantic_subset as semantic_subset
from native_runner.training.v4.native_actions import ability_source_card_id
from native_runner.training.card_features import HERO_FORM_TO_BASE_CARD


WORKSPACE = WORKSPACE_ROOT

EXPECTED_REPLAY_HERO_BINDINGS = {
    "balloon": ("BalloonHero_Ability", "balloon"),
    "barbarian-barrel": ("BarbLogHeroAbility", "barblog"),
    "berserker": ("BerserkerHeroAbility", "berserker"),
    "bowler": ("BowlerHeroAbility", "bowler"),
    "dark-prince": ("DarkPrinceHero_Ability", "darkprince"),
    "magic-archer": ("EliteArcherHero_Ability", "elitearcher"),
    "giant": ("GiantHero_Ability", "giant"),
    "goblins": ("GoblinHero_Ability", "goblin"),
    "ice-golem": ("IceGolemiteHero_Ability", "icegolemite"),
    "knight": ("Knight_hero_Ability", "knight"),
    "mega-minion": ("MegaMinion_Teleport_Ability", "megaminion"),
    "mini-pekka": ("MiniPekkHeroAbility", "minipekk"),
    "musketeer": ("Musketeer_hero_Ability", "musketeer"),
    "tombstone": ("Tombstone_hero_Ability", "tombstone"),
    "valkyrie": ("ValkyrieHero_Ability", "valkyrie"),
    "wizard": ("WizardHeroAbility", "wizard"),
}

EXPECTED_HERO_ABILITY_FIELDS = {
    26_000_006: (2.0, None, 1, 0, "BalloonHero_Ability_Target_Seeker"),
    28_000_015: (1.0, None, 1, 950, "BarbLogHero_spawn_reroll"),
    26_000_102: (3.0, None, 1, 1_450, "BerserkerHero_ability_group"),
    26_000_034: (2.0, None, 1, 2_500, "BowlerHero_ability_activation_group"),
    26_000_027: (
        3.0,
        None,
        1,
        2_000,
        "DarkPrinceHero_Ability_Activation_Group",
    ),
    26_000_062: (
        2.0,
        None,
        1,
        950,
        "EliteArcherHero_Ability_Activation_Group",
    ),
    26_000_003: (2.0, None, 1, 0, "GiantHero_OnAbilityActivationGroup"),
    26_000_002: (1.0, None, 1, 500, "GoblinHero_Ability_Activated_Group"),
    26_000_038: (
        2.0,
        None,
        1,
        0,
        "IceGolemiteHero_OnAbilityActivationGroup",
    ),
    26_000_000: (2.0, None, 1, 1_200, "Knight_hero_OnAbilityActivationGroup"),
    26_000_039: (2.0, 1_000, 1, 250, "MegaMinion_Lock_ability_action"),
    26_000_018: (1.0, None, 1, 950, "MiniPekka_hero_ability_activation"),
    26_000_014: (
        3.0,
        None,
        1,
        950,
        "Musketeer_hero_Turret_Ability_Group",
    ),
    27_000_009: (5.0, None, 1, 0, "Tombstone_hero_OnAbilityActivationGroup"),
    26_000_011: (3.0, None, 1, 0, "ValkyrieHero_OnAbilityActivationGroup"),
    26_000_017: (1.0, None, 1, 950, "WizardHero_ability_activation"),
}

EXPECTED_CHAMPION_BINDINGS = {
    26_000_065: ("MightyMinerLaneSwitch", "mightyminer"),
    26_000_069: ("SkeletonKing", "skeletonking"),
    26_000_072: ("ArcherQueenRapid", "archerqueen"),
    26_000_074: ("GoldenKnightChain", "goldenknight"),
    26_000_077: ("Deflect", "deflect"),
    26_000_093: ("ChampGuardianAbility", "champguardian"),
    26_000_099: ("goblinstein_ability", "goblinstein"),
    26_000_103: ("BossBandit_ability", "bossbandit"),
}

EXPECTED_CHAMPION_ABILITY_FIELDS = {
    26_000_065: (1.0, None, 1, 933, "Spawn"),
    26_000_069: (2.0, None, 1, 933, "CreateArea"),
    26_000_072: (1.0, None, 1, 933, "ApplyBuff"),
    26_000_074: (
        1.0,
        None,
        1,
        0,
        "GoldenKnight_OnAbilityActivationGroup",
    ),
    26_000_077: (1.0, None, 1, 933, "CreateArea"),
    26_000_093: (3.0, None, 1, 944, "LittlePrinceWaitGuard"),
    26_000_099: (2.0, None, 1, 933, "goblinstein_enable_doctor_aura"),
    26_000_103: (1.0, 3_000, 2, 933, "BossBandit_ability_warp_group"),
}


@pytest.fixture(scope="module")
def catalog():
    return build_card_catalog(WORKSPACE)


def test_replay_hero_binding_api_and_serialization_are_unchanged() -> None:
    assert REPLAY_HERO_BINDINGS is NATIVE_HERO_FORM_ABILITY_BINDINGS
    assert NATIVE_HERO_FORM_ABILITY_BINDINGS == EXPECTED_REPLAY_HERO_BINDINGS
    assert list(NATIVE_HERO_FORM_ABILITY_BINDINGS) == list(
        EXPECTED_REPLAY_HERO_BINDINGS
    )
    assert json.dumps(NATIVE_HERO_FORM_ABILITY_BINDINGS) == json.dumps(
        EXPECTED_REPLAY_HERO_BINDINGS
    )
    assert len(NATIVE_HERO_FORM_ABILITY_BINDINGS_SHA256) == 64
    assert REPLAY_CHAMPION_BINDINGS is NATIVE_CHAMPION_ABILITY_BINDINGS
    assert NATIVE_CHAMPION_ABILITY_BINDINGS == EXPECTED_CHAMPION_BINDINGS
    assert json.dumps(NATIVE_CHAMPION_ABILITY_BINDINGS) == json.dumps(
        EXPECTED_CHAMPION_BINDINGS
    )
    assert len(NATIVE_CHAMPION_ABILITY_BINDINGS_SHA256) == 64


def test_catalog_resolves_all_explicit_normal_mode_forms(catalog) -> None:
    evolutions = {
        spec.card_id: spec.evolution
        for spec in catalog.specs
        if spec.evolution is not None
    }
    assert len(evolutions) == NORMAL_MODE_EVOLUTION_FORM_COUNT == 42
    assert len(NORMAL_MODE_HERO_FORM_BINDINGS_BY_CARD_ID) == (
        NORMAL_MODE_HERO_FORM_COUNT
    ) == 16
    assert len(NORMAL_MODE_DIRECT_HERO_BINDINGS_BY_CARD_ID) == (
        NORMAL_MODE_DIRECT_HERO_COUNT
    ) == 8
    assert evolutions[26_000_000].evolution_form_id == "Knight_EV1"

    evolution_ids = set(evolutions)
    hero_ids = set(NORMAL_MODE_HERO_FORM_BINDINGS_BY_CARD_ID)
    direct_hero_ids = set(NORMAL_MODE_DIRECT_HERO_BINDINGS_BY_CARD_ID)
    assert len(evolution_ids | hero_ids | direct_hero_ids) == (
        NORMAL_MODE_POLICY_FORM_CARD_COUNT
    ) == 62
    assert evolution_ids & hero_ids == {
        26_000_000,
        26_000_011,
        26_000_014,
        26_000_017,
    }
    assert not direct_hero_ids & (evolution_ids | hero_ids)
    assert all(
        not (
            catalog.by_id[card_id].attributes.get("NotVisible")
            or catalog.by_id[card_id].attributes.get("NotInUse")
        )
        for card_id in evolution_ids | hero_ids | direct_hero_ids
    )

    abilities = {ability.ability_id: ability for ability in catalog.abilities}
    for card_id, expected in EXPECTED_HERO_ABILITY_FIELDS.items():
        binding = NORMAL_MODE_HERO_FORM_BINDINGS_BY_CARD_ID[card_id]
        ability = abilities[binding.ability_id]
        assert (
            ability.elixir_cost,
            ability.cooldown_ms,
            ability.charges,
            ability.cast_time_ms,
            ability.effect_graph[0].custom_op,
        ) == expected

    for card_id, expected in EXPECTED_CHAMPION_ABILITY_FIELDS.items():
        binding = NORMAL_MODE_DIRECT_HERO_BINDINGS_BY_CARD_ID[card_id]
        spec = catalog.by_id[card_id]
        ability = abilities[binding.ability_id]
        first_effect = ability.effect_graph[0]
        assert spec.kind.value == "hero"
        assert spec.ability_ids == (binding.ability_id,)
        assert ability.unknown_fields == ()
        assert (
            ability.elixir_cost,
            ability.cooldown_ms,
            ability.charges,
            ability.cast_time_ms,
            first_effect.custom_op or first_effect.effect,
        ) == expected

    rune_giant = catalog.by_id[26_000_101]
    assert rune_giant.name == "GiantBuffer"
    assert rune_giant.kind.value == "troop"
    assert rune_giant.ability_ids == ()


@pytest.mark.parametrize(("ability_id", "mutation"), [
    ("ArcherQueenRapid", {"charges": 2}),
    ("ArcherQueenRapid", {"cooldown_ms": 17_000}),
    ("MegaMinion_Teleport_Ability", {"cooldown_ms": None}),
    ("BossBandit_ability", {"charges": 1}),
    ("BossBandit_ability", {"cooldown_ms": None}),
])
def test_form_evidence_rejects_ability_lifecycle_drift(catalog, ability_id, mutation):
    changed = replace(catalog, abilities=tuple(
        replace(ability, **mutation) if ability.ability_id == ability_id else ability
        for ability in catalog.abilities
    ))
    with pytest.raises(semantic_subset.SemanticSubsetError, match="catalog identity"):
        build_normal_mode_policy_form_evidence(changed)


def test_form_evidence_rejects_source_provenance_drift(catalog):
    changed = replace(catalog, source_files={**catalog.source_files, "extra": "0" * 64})
    with pytest.raises(semantic_subset.SemanticSubsetError, match="source-file identity"):
        build_normal_mode_policy_form_evidence(changed)


def test_all_hero_runtime_form_ids_are_bound_to_the_decoded_source_table(tmp_path) -> None:
    source = tmp_path / "card_forms.csv"
    member = "assets/" + NORMAL_MODE_HERO_FORM_TABLE_RELATIVE_PATH.split("/assets/", 1)[1]
    from sc_compression import decompress
    source.write_bytes(decompress(user_apk_bytes(member))[0])
    assert sha256_file(source) == NORMAL_MODE_HERO_FORM_TABLE_SHA256
    _, rows = _read_sc_csv(source)
    names_by_global_id = {
        203_000_000 + index: str(row["Name"])
        for index, row in enumerate(rows)
    }

    assert HERO_FORM_TO_BASE_CARD is NORMAL_MODE_HERO_FORM_TO_BASE_CARD
    assert len(HERO_FORM_TO_BASE_CARD) == NORMAL_MODE_HERO_FORM_COUNT == 16
    for binding in NORMAL_MODE_HERO_FORM_BINDINGS_BY_CARD_ID.values():
        assert names_by_global_id[binding.runtime_form_card_id] == (
            binding.hero_form_id
        )
        assert HERO_FORM_TO_BASE_CARD[binding.runtime_form_card_id] == (
            binding.source_card_id
        )


@pytest.mark.parametrize(
    ("runtime_form_card_id", "base_card_id"),
    tuple(NORMAL_MODE_HERO_FORM_TO_BASE_CARD.items()),
)
def test_every_hero_runtime_entity_joins_its_base_deck_card(
    runtime_form_card_id: int,
    base_card_id: int,
) -> None:
    filler = tuple(
        card_id
        for card_id in range(26_900_000, 26_900_008)
        if card_id != base_card_id
    )
    deck = (base_card_id, *filler[:7])
    observation = ObservationV1(
        tier=ObservationTier.FAIR,
        tick=1,
        owner=0,
        players=(PlayerStateV1(owner=0), PlayerStateV1(owner=1)),
        entities=(
            EntityStateV1(
                entity_id=123,
                owner=0,
                card_id=runtime_form_card_id,
                entity_kind="hero",
                position=(9_000.0, 9_000.0),
            ),
        ),
    )

    assert ability_source_card_id(
        observation,
        source_entity=123,
        deck=deck,
    ) == base_card_id


def test_direct_champion_keys_require_bit2_but_rune_giant_does_not() -> None:
    public_keys = {
        26_000_065: "mighty-miner",
        26_000_069: "skeleton-king",
        26_000_072: "archer-queen",
        26_000_074: "golden-knight",
        26_000_077: "monk",
        26_000_093: "little-prince",
        26_000_099: "goblinstein",
        26_000_103: "boss-bandit",
    }
    for card_id, key in public_keys.items():
        info = royaleapi_replay.resolve_native_card(key)
        assert info.card_id == card_id
        assert royaleapi_replay._deck_form_mask(key, info) == 2

    rune_giant = royaleapi_replay.resolve_native_card("rune-giant")
    assert rune_giant.card_id == 26_000_101
    # Its decoded internal zero-cost ability says IsChampion=false.
    assert royaleapi_replay._deck_form_mask("rune-giant", rune_giant) == 0


def test_compiled_form_evidence_is_exact_and_content_addressed(catalog) -> None:
    evidence = build_normal_mode_policy_form_evidence(catalog)
    masks = dict(evidence.form_masks)

    assert len(evidence.cards) == NORMAL_MODE_POLICY_FORM_CARD_COUNT
    assert evidence.criteria_version == (
        NORMAL_MODE_FORM_POLICY_CRITERIA_VERSION
    )
    assert sum(mask == 1 for mask in masks.values()) == 38
    assert sum(mask == 2 for mask in masks.values()) == 20
    assert sum(mask == 3 for mask in masks.values()) == 4
    assert masks[26_000_000] == 3
    assert masks[26_000_014] == 3
    assert masks[26_000_017] == 3
    assert all(masks[card_id] == 2 for card_id in EXPECTED_CHAMPION_BINDINGS)
    assert 26_000_101 not in masks
    assert all(
        any("mechanics-readiness" in ref for ref in row.evidence_refs)
        and any("source" in ref for ref in row.evidence_refs)
        for row in evidence.cards
    )
    assert all(
        any("native-binding" in ref for ref in row.evidence_refs)
        for row in evidence.cards
        if row.card_id in NORMAL_MODE_HERO_FORM_BINDINGS_BY_CARD_ID
    )
    assert all(
        any("native-champion-binding" in ref for ref in row.evidence_refs)
        for row in evidence.cards
        if row.card_id in NORMAL_MODE_DIRECT_HERO_BINDINGS_BY_CARD_ID
    )

    restored = semantic_subset.NormalModePolicyFormEvidenceV1.from_mapping(
        json.loads(evidence.to_json())
    )
    assert restored == evidence
    assert restored.evidence_id == evidence.evidence_id
