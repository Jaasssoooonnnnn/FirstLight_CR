"""Content-addressed evolution and player-ability policy evidence.

The base-card mechanics ledger and form/Champion admission answer different
questions.  This module keeps the exact normal-mode form identities, native
action names, and policy evidence binding in one place so CardSpec, RoyaleAPI
replay adaptation, and the supported-subset contract cannot drift.

Importing this module is intentionally lightweight.  The catalog and semantic
subset modules are imported only inside the evidence builder, which keeps the
shared binding constants safe to import from :mod:`card_specs` and
:mod:`royaleapi_replay` without a cycle.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from .contracts import content_hash

if TYPE_CHECKING:
    from .card_specs import CardSpecCatalog
    from .semantic_subset import NormalModePolicyFormEvidenceV1, NormalModePolicyReadinessV1


NORMAL_MODE_FORM_POLICY_CRITERIA_VERSION = "normal-mode-policy-form-evidence.2026-08-12.2"
NORMAL_MODE_EVOLUTION_FORM_COUNT = 42
NORMAL_MODE_HERO_FORM_COUNT = 16
NORMAL_MODE_DIRECT_HERO_COUNT = 8
NORMAL_MODE_POLICY_FORM_CARD_COUNT = 62
NORMAL_MODE_POLICY_FORM_EVIDENCE_SHA256 = "d2e0048f5d060df930a6a1dd34758fe2dce6800f83543300c845a25ec9ff7be5"
NORMAL_MODE_HERO_FORM_TABLE_RELATIVE_PATH = (
    "captures/decompressed_assets/nr_15.535.13_release_ff1c6c29/assets/csv_logic/spells_hero_form.csv"
)
NORMAL_MODE_HERO_FORM_TABLE_SHA256 = "ed964474a08bb85a80c3e39810c6464078d327b2004b2f71d79ab3d24489c87b"


@dataclass(frozen=True, slots=True)
class NormalModeHeroFormBindingV1:
    """Exact public/base/form/native-action identity for one Hero form."""

    public_key: str
    source_card_id: int
    source_card_name: str
    hero_form_id: str
    runtime_form_card_id: int
    ability_id: str
    runtime_hint: str

    def __post_init__(self) -> None:
        if not self.public_key or self.public_key.endswith("-hero"):
            raise ValueError("Hero binding public_key must be the base card key")
        if self.source_card_id <= 0 or not self.source_card_name:
            raise ValueError("Hero binding requires a base card identity")
        if not self.hero_form_id.endswith("_hero"):
            raise ValueError("Hero binding form ID must end in _hero")
        if self.runtime_form_card_id // 1_000_000 != 203:
            raise ValueError("Hero runtime form must be an exact class-203 ID")
        if not self.ability_id or not self.runtime_hint:
            raise ValueError("Hero binding requires native ability and hint names")
        if self.runtime_hint.casefold() not in self.ability_id.casefold():
            raise ValueError(f"Hero runtime hint {self.runtime_hint!r} is not in {self.ability_id!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "public_key": self.public_key,
            "source_card_id": self.source_card_id,
            "source_card_name": self.source_card_name,
            "hero_form_id": self.hero_form_id,
            "runtime_form_card_id": self.runtime_form_card_id,
            "ability_id": self.ability_id,
            "runtime_hint": self.runtime_hint,
        }


@dataclass(frozen=True, slots=True)
class NormalModeDirectHeroBindingV1:
    """Exact base-card/carrier/native-action identity for one Champion."""

    source_card_id: int
    source_card_name: str
    carrier_name: str
    ability_id: str
    runtime_hint: str

    def __post_init__(self) -> None:
        if self.source_card_id <= 0 or not self.source_card_name:
            raise ValueError("direct Hero binding requires a card identity")
        if not self.carrier_name or not self.ability_id or not self.runtime_hint:
            raise ValueError("direct Hero binding requires carrier and ability names")
        if self.runtime_hint.casefold() not in self.ability_id.casefold():
            raise ValueError(f"Champion runtime hint {self.runtime_hint!r} is not in {self.ability_id!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_card_id": self.source_card_id,
            "source_card_name": self.source_card_name,
            "carrier_name": self.carrier_name,
            "ability_id": self.ability_id,
            "runtime_hint": self.runtime_hint,
        }


_NORMAL_MODE_HERO_FORM_BINDINGS = (
    NormalModeHeroFormBindingV1(
        "balloon", 26_000_006, "Balloon", "Balloon_hero", 203_000_006, "BalloonHero_Ability", "balloon"
    ),
    NormalModeHeroFormBindingV1(
        "barbarian-barrel", 28_000_015, "BarbLog", "BarbLog_hero", 203_000_107, "BarbLogHeroAbility", "barblog"
    ),
    NormalModeHeroFormBindingV1(
        "berserker", 26_000_102, "Berserker", "Berserker_hero", 203_000_076, "BerserkerHeroAbility", "berserker"
    ),
    NormalModeHeroFormBindingV1(
        "bowler", 26_000_034, "Bowler", "Bowler_hero", 203_000_034, "BowlerHeroAbility", "bowler"
    ),
    NormalModeHeroFormBindingV1(
        "dark-prince", 26_000_027, "DarkPrince", "DarkPrince_hero", 203_000_027, "DarkPrinceHero_Ability", "darkprince"
    ),
    NormalModeHeroFormBindingV1(
        "magic-archer",
        26_000_062,
        "EliteArcher",
        "EliteArcher_hero",
        203_000_062,
        "EliteArcherHero_Ability",
        "elitearcher",
    ),
    NormalModeHeroFormBindingV1("giant", 26_000_003, "Giant", "Giant_hero", 203_000_003, "GiantHero_Ability", "giant"),
    NormalModeHeroFormBindingV1(
        "goblins", 26_000_002, "Goblins", "Goblins_hero", 203_000_002, "GoblinHero_Ability", "goblin"
    ),
    NormalModeHeroFormBindingV1(
        "ice-golem",
        26_000_038,
        "IceGolemite",
        "IceGolemite_hero",
        203_000_038,
        "IceGolemiteHero_Ability",
        "icegolemite",
    ),
    NormalModeHeroFormBindingV1(
        "knight", 26_000_000, "Knight", "Knight_hero", 203_000_000, "Knight_hero_Ability", "knight"
    ),
    NormalModeHeroFormBindingV1(
        "mega-minion",
        26_000_039,
        "MegaMinion",
        "MegaMinion_hero",
        203_000_039,
        "MegaMinion_Teleport_Ability",
        "megaminion",
    ),
    NormalModeHeroFormBindingV1(
        "mini-pekka", 26_000_018, "MiniPekka", "MiniPekka_hero", 203_000_018, "MiniPekkHeroAbility", "minipekk"
    ),
    NormalModeHeroFormBindingV1(
        "musketeer", 26_000_014, "Musketeer", "Musketeer_hero", 203_000_014, "Musketeer_hero_Ability", "musketeer"
    ),
    NormalModeHeroFormBindingV1(
        "tombstone", 27_000_009, "Tombstone", "Tombstone_hero", 203_000_088, "Tombstone_hero_Ability", "tombstone"
    ),
    NormalModeHeroFormBindingV1(
        "valkyrie", 26_000_011, "Valkyrie", "Valkyrie_hero", 203_000_011, "ValkyrieHero_Ability", "valkyrie"
    ),
    NormalModeHeroFormBindingV1(
        "wizard", 26_000_017, "Wizard", "Wizard_hero", 203_000_017, "WizardHeroAbility", "wizard"
    ),
)


_NORMAL_MODE_DIRECT_HERO_BINDINGS = (
    NormalModeDirectHeroBindingV1(26_000_065, "MightyMiner", "MightyMiner", "MightyMinerLaneSwitch", "mightyminer"),
    NormalModeDirectHeroBindingV1(26_000_069, "SkeletonKing", "SkeletonKing", "SkeletonKing", "skeletonking"),
    NormalModeDirectHeroBindingV1(26_000_072, "ArcherQueen", "ArcherQueen", "ArcherQueenRapid", "archerqueen"),
    NormalModeDirectHeroBindingV1(26_000_074, "GoldenKnight", "GoldenKnight", "GoldenKnightChain", "goldenknight"),
    NormalModeDirectHeroBindingV1(26_000_077, "Monk", "Monk", "Deflect", "deflect"),
    NormalModeDirectHeroBindingV1(26_000_093, "LittlePrince", "LittlePrince", "ChampGuardianAbility", "champguardian"),
    NormalModeDirectHeroBindingV1(
        26_000_099, "Goblinstein", "goblinstein_doctor", "goblinstein_ability", "goblinstein"
    ),
    NormalModeDirectHeroBindingV1(26_000_103, "BossBandit", "BossBandit", "BossBandit_ability", "bossbandit"),
)


NORMAL_MODE_HERO_FORM_BINDINGS_BY_CARD_ID: Mapping[int, NormalModeHeroFormBindingV1] = MappingProxyType(
    {item.source_card_id: item for item in _NORMAL_MODE_HERO_FORM_BINDINGS}
)
NORMAL_MODE_HERO_FORM_BINDINGS_BY_FORM_ID: Mapping[str, NormalModeHeroFormBindingV1] = MappingProxyType(
    {item.hero_form_id: item for item in _NORMAL_MODE_HERO_FORM_BINDINGS}
)
NORMAL_MODE_HERO_FORM_TO_BASE_CARD: Mapping[int, int] = MappingProxyType(
    {item.runtime_form_card_id: item.source_card_id for item in _NORMAL_MODE_HERO_FORM_BINDINGS}
)
NORMAL_MODE_DIRECT_HERO_BINDINGS_BY_CARD_ID: Mapping[int, NormalModeDirectHeroBindingV1] = MappingProxyType(
    {item.source_card_id: item for item in _NORMAL_MODE_DIRECT_HERO_BINDINGS}
)

# Public replay API retained verbatim.  Insertion order and values intentionally
# match the previous royaleapi_replay-local mapping.
NATIVE_HERO_FORM_ABILITY_BINDINGS: Mapping[str, tuple[str, str]] = {
    item.public_key: (item.ability_id, item.runtime_hint) for item in _NORMAL_MODE_HERO_FORM_BINDINGS
}
NATIVE_HERO_FORM_ABILITY_BINDINGS_SHA256 = content_hash(NATIVE_HERO_FORM_ABILITY_BINDINGS)

# Public replay API retained verbatim.  These eight entries are active
# player-triggered Champion abilities; internal/passive unit Ability fields
# such as Rune Giant's ``IsChampion=false`` action are intentionally absent.
NATIVE_CHAMPION_ABILITY_BINDINGS: Mapping[int, tuple[str, str]] = {
    item.source_card_id: (item.ability_id, item.runtime_hint) for item in _NORMAL_MODE_DIRECT_HERO_BINDINGS
}
NATIVE_CHAMPION_ABILITY_BINDINGS_SHA256 = content_hash(NATIVE_CHAMPION_ABILITY_BINDINGS)


def build_normal_mode_policy_form_evidence(
    catalog: "CardSpecCatalog", *, readiness: "NormalModePolicyReadinessV1 | None" = None
) -> "NormalModePolicyFormEvidenceV1":
    """Bind the compiled 62-form evidence to the exact active catalog."""
    # Lazy imports keep the shared native bindings safe during initialization.
    from .competitive_data import load_competitive_data
    from .semantic_subset import (
        NORMAL_MODE_POLICY_ABILITY_SPECS_HASH,
        NORMAL_MODE_POLICY_CARD_SPECS_HASH,
        NORMAL_MODE_POLICY_SOURCE_RELEASE,
        NormalModePolicyFormEvidenceV1,
        SemanticSubsetError,
        load_normal_mode_policy_readiness,
    )

    if catalog.source_release != NORMAL_MODE_POLICY_SOURCE_RELEASE:
        raise SemanticSubsetError("normal-mode form evidence source-release identity mismatch")
    if (
        catalog.specs_hash != NORMAL_MODE_POLICY_CARD_SPECS_HASH
        or catalog.ability_specs_hash != NORMAL_MODE_POLICY_ABILITY_SPECS_HASH
    ):
        raise SemanticSubsetError("normal-mode form evidence catalog identity mismatch")
    canonical_readiness = load_normal_mode_policy_readiness()
    if readiness is not None and readiness != canonical_readiness:
        raise SemanticSubsetError("normal-mode form evidence readiness partition mismatch")
    compiled = load_competitive_data("form_evidence")
    if content_hash(dict(catalog.source_files)) != compiled["catalog_source_files_sha256"]:
        raise SemanticSubsetError("normal-mode form evidence source-file identity mismatch")
    evidence = NormalModePolicyFormEvidenceV1.from_mapping(compiled["evidence"])
    if evidence.evidence_id != NORMAL_MODE_POLICY_FORM_EVIDENCE_SHA256:
        raise SemanticSubsetError("canonical normal-mode form evidence identity drifted")
    return evidence
