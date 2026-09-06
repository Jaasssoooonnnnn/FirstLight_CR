"""Content-addressed card pool for the semantic-headless supported subset.

The compact, checksum-pinned card support manifest defines the supported
partition and preserves the compatibility identifiers used by checkpoints.
Experiment definitions, logs and reports are not runtime dependencies.
Hidden, event and disabled catalog entries remain excluded.

Low-level compatibility work may still use :class:`NativeClashEnv` directly.
Canonical :class:`BattleEnvV1` episodes are gated by this contract before a
native match is created.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, ClassVar

from .card_specs import CATALOG_VERSION, CardSpecCatalog
from .competitive_data import competitive_data_initialization, load_competitive_data
from .runtime_scope import NORMAL_MODE_CASE_SCOPE_SHA256
from .contracts import CardKind, CardSpecV1, ContractError, ContractMixin, content_hash


SEMANTIC_SUBSET_VERSION = "semantic-supported-subset.v4"
SEMANTIC_SUBSET_CARD_VERSION = "semantic-supported-card.v3"
SEMANTIC_SUBSET_CRITERIA_VERSION = "semantic-supported-subset-criteria.2026-08-12.2"
NATIVE_SEMANTIC_CAPABILITY_PROFILE = "native-normal-mode-base-cards-position-hp-elixir-form-gated.v7"
MINIMUM_BATTLE_CARDS = 8

NORMAL_MODE_POLICY_SCOPE_VERSION = "normal-mode-card-scope.v1"
NORMAL_MODE_POLICY_MECHANICS_READINESS_REPORT_ID = "19058712db46281ee3e3ae41dcfe13d40af695259c299f26245e44bc6ee38ca6"
NORMAL_MODE_POLICY_STATIC_CATALOG_ID = "6b9a35d0647e944697579421d2a766d65ea96c56b3363352ca5dd73a69907d81"
NORMAL_MODE_POLICY_CASE_SCOPE_ID = NORMAL_MODE_CASE_SCOPE_SHA256
NORMAL_MODE_POLICY_CARD_SPECS_HASH = "28b415b67a85244e3db6106efc458d73b74f2f5d1aad7f3baf7ce10c04f6954a"
NORMAL_MODE_POLICY_ABILITY_SPECS_HASH = "077bc3241eed2a1efa6c892e4c2229d7dd2e873dd707815e6231cc426b745ffc"
NORMAL_MODE_POLICY_SOURCE_RELEASE = "nr_15.535.13_release_ff1c6c29"

# The reviewed partition is also bound by the compact manifest's checksum.
NORMAL_MODE_POLICY_CARD_IDS: frozenset[int] = frozenset(
    {
        26_000_000,
        26_000_001,
        26_000_002,
        26_000_003,
        26_000_004,
        26_000_005,
        26_000_006,
        26_000_007,
        26_000_008,
        26_000_009,
        26_000_010,
        26_000_011,
        26_000_012,
        26_000_013,
        26_000_014,
        26_000_015,
        26_000_016,
        26_000_017,
        26_000_018,
        26_000_019,
        26_000_020,
        26_000_021,
        26_000_022,
        26_000_023,
        26_000_024,
        26_000_025,
        26_000_026,
        26_000_027,
        26_000_028,
        26_000_029,
        26_000_030,
        26_000_031,
        26_000_032,
        26_000_033,
        26_000_034,
        26_000_035,
        26_000_036,
        26_000_037,
        26_000_038,
        26_000_039,
        26_000_040,
        26_000_041,
        26_000_042,
        26_000_043,
        26_000_044,
        26_000_045,
        26_000_046,
        26_000_047,
        26_000_048,
        26_000_049,
        26_000_050,
        26_000_051,
        26_000_052,
        26_000_053,
        26_000_054,
        26_000_055,
        26_000_056,
        26_000_057,
        26_000_058,
        26_000_059,
        26_000_060,
        26_000_061,
        26_000_062,
        26_000_063,
        26_000_064,
        26_000_065,
        26_000_067,
        26_000_068,
        26_000_069,
        26_000_072,
        26_000_074,
        26_000_077,
        26_000_080,
        26_000_083,
        26_000_084,
        26_000_085,
        26_000_087,
        26_000_093,
        26_000_095,
        26_000_096,
        26_000_097,
        26_000_099,
        26_000_101,
        26_000_102,
        26_000_103,
        26_000_106,
        27_000_000,
        27_000_001,
        27_000_002,
        27_000_003,
        27_000_004,
        27_000_005,
        27_000_006,
        27_000_007,
        27_000_008,
        27_000_009,
        27_000_010,
        27_000_012,
        27_000_013,
        28_000_000,
        28_000_001,
        28_000_002,
        28_000_003,
        28_000_004,
        28_000_005,
        28_000_006,
        28_000_007,
        28_000_008,
        28_000_009,
        28_000_010,
        28_000_011,
        28_000_012,
        28_000_013,
        28_000_014,
        28_000_015,
        28_000_016,
        28_000_017,
        28_000_018,
        28_000_023,
        28_000_024,
        28_000_025,
        28_000_026,
    }
)
NORMAL_MODE_POLICY_EXCLUDED_CARD_IDS: frozenset[int] = frozenset(
    {
        26_000_066,
        26_000_070,
        26_000_071,
        26_000_073,
        26_000_075,
        26_000_076,
        26_000_078,
        26_000_079,
        26_000_081,
        26_000_082,
        26_000_086,
        26_000_088,
        26_000_089,
        26_000_090,
        26_000_091,
        26_000_092,
        26_000_094,
        26_000_098,
        26_000_100,
        26_000_104,
        26_000_105,
        27_000_011,
        27_000_014,
        27_000_015,
        27_000_016,
        27_000_017,
        28_000_019,
        28_000_020,
        28_000_021,
        28_000_022,
    }
)

SEMANTIC_SPEED_XBOW_DECK: tuple[int, ...] = (
    27_000_008,  # X-Bow
    26_000_000,  # Hero Knight
    26_000_001,  # Archers
    27_000_006,  # Tesla
    26_000_010,  # Skeletons
    26_000_084,  # Electro Spirit
    28_000_000,  # Fireball
    28_000_011,  # The Log
)

# Bit zero enables the evolution form; bit one enables the Hero form.  The
# contract intentionally permits only the two selected evolutions and the
# Knight Hero skin/ability used by the training match.  Skeletons remains
# base-only even though the catalog also contains its evolution.
SEMANTIC_SPEED_XBOW_FORM_AVAILABILITY: tuple[int, ...] = (0, 2, 1, 1, 0, 0, 0, 0)

SEMANTIC_PEKKA_BRIDGE_SPAM_DECK: tuple[int, ...] = (
    26_000_036,  # Evolved Battle Ram
    26_000_050,  # Evolved Royal Ghost
    26_000_062,  # Hero Magic Archer
    26_000_004,  # P.E.K.K.A
    26_000_046,  # Bandit
    26_000_042,  # Electro Wizard
    28_000_000,  # Fireball
    28_000_008,  # Zap
)

# Evolution bit 0 is enabled only for Battle Ram and Royal Ghost.  Hero bit 1
# is enabled only for Magic Archer.  P.E.K.K.A and Zap deliberately remain
# base-only even though this exact content build also contains their evolutions.
SEMANTIC_PEKKA_BRIDGE_SPAM_FORM_AVAILABILITY: tuple[int, ...] = (1, 1, 2, 0, 0, 0, 0, 0)

SEMANTIC_BASELINE_DECK: tuple[int, ...] = (
    26_000_000,
    26_000_002,
    26_000_003,
    26_000_005,
    26_000_018,
    26_000_019,
    28_000_000,
    28_000_003,
)

REASON_CATALOG_UNAVAILABLE = "catalog_marks_card_unavailable"
REASON_MISSING_MECHANICS_READINESS = "missing_definition_exact_mechanics_readiness"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
NORMAL_MODE_POLICY_FORM_EVIDENCE_VERSION = "normal-mode-policy-form-evidence.2026-08-12.2"
NORMAL_MODE_POLICY_FORM_CONTRACT_VERSION = "normal-mode-policy-form-evidence.v1"


class SemanticSubsetError(ContractError):
    """Raised when a catalog or deck crosses the supported-subset boundary."""


def _require_sha256(value: str, label: str) -> None:
    if not _SHA256.fullmatch(value):
        raise SemanticSubsetError(f"{label} must be a lowercase SHA-256 digest")


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise SemanticSubsetError(f"unknown {label} fields: {', '.join(unknown)}")


@dataclass(frozen=True, slots=True)
class NormalModePolicyFormCardEvidenceV1(ContractMixin):
    """Evidence references for the non-base form bits of one card."""

    card_id: int
    allowed_form_availability: int
    evidence_refs: tuple[str, ...]
    version: str = field(default="normal-mode-policy-form-card-evidence.v1", init=False)
    VERSION: ClassVar[str] = "normal-mode-policy-form-card-evidence.v1"

    def __post_init__(self) -> None:
        if self.card_id not in NORMAL_MODE_POLICY_CARD_IDS:
            raise SemanticSubsetError(f"form evidence card {self.card_id} is not normal-mode ready")
        if isinstance(self.allowed_form_availability, bool) or self.allowed_form_availability not in {1, 2, 3}:
            raise SemanticSubsetError("form evidence mask must enable evolution, Hero, or both")
        refs = tuple(sorted(set(str(item) for item in self.evidence_refs)))
        if not refs or any(not ref or re.search(r"[0-9a-f]{64}", ref) is None for ref in refs):
            raise SemanticSubsetError("form evidence refs must be non-empty and content-addressed")
        if self.allowed_form_availability & 1 and not any(ref.startswith("evolution:") for ref in refs):
            raise SemanticSubsetError(f"card {self.card_id} evolution bit lacks evolution evidence")
        if self.allowed_form_availability & 2 and not any(ref.startswith("hero:") for ref in refs):
            raise SemanticSubsetError(f"card {self.card_id} Hero bit lacks Hero evidence")
        object.__setattr__(self, "evidence_refs", refs)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "NormalModePolicyFormCardEvidenceV1":
        _reject_unknown(
            value,
            {"version", "card_id", "allowed_form_availability", "evidence_refs"},
            "normal-mode policy form card evidence",
        )
        if value.get("version") != cls.VERSION:
            raise SemanticSubsetError(f"unsupported form card evidence version: {value.get('version')}")
        return cls(
            card_id=int(value["card_id"]),
            allowed_form_availability=int(value["allowed_form_availability"]),
            evidence_refs=tuple(str(item) for item in value.get("evidence_refs", ())),
        )


@dataclass(frozen=True, slots=True)
class NormalModePolicyFormEvidenceV1(ContractMixin):
    """Injectable, content-addressed non-base-form admission contract."""

    criteria_version: str
    mechanics_readiness_report_id: str
    card_specs_hash: str
    ability_specs_hash: str
    cards: tuple[NormalModePolicyFormCardEvidenceV1, ...]
    version: str = field(default=NORMAL_MODE_POLICY_FORM_CONTRACT_VERSION, init=False)
    VERSION: ClassVar[str] = NORMAL_MODE_POLICY_FORM_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not self.criteria_version:
            raise SemanticSubsetError("form evidence requires a criteria version")
        if self.mechanics_readiness_report_id != NORMAL_MODE_POLICY_MECHANICS_READINESS_REPORT_ID:
            raise SemanticSubsetError("form evidence mechanics readiness identity mismatch")
        if self.card_specs_hash != NORMAL_MODE_POLICY_CARD_SPECS_HASH:
            raise SemanticSubsetError("form evidence card-spec identity mismatch")
        if self.ability_specs_hash != NORMAL_MODE_POLICY_ABILITY_SPECS_HASH:
            raise SemanticSubsetError("form evidence ability-spec identity mismatch")
        cards = tuple(
            sorted(
                (
                    item
                    if isinstance(item, NormalModePolicyFormCardEvidenceV1)
                    else NormalModePolicyFormCardEvidenceV1.from_mapping(item)
                    for item in self.cards
                ),
                key=lambda item: item.card_id,
            )
        )
        if len({item.card_id for item in cards}) != len(cards):
            raise SemanticSubsetError("form evidence contains duplicate card IDs")
        object.__setattr__(self, "cards", cards)

    @property
    def evidence_id(self) -> str:
        return content_hash(self)

    @property
    def form_masks(self) -> Mapping[int, int]:
        return MappingProxyType({item.card_id: item.allowed_form_availability for item in self.cards})

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "NormalModePolicyFormEvidenceV1":
        _reject_unknown(
            value,
            {
                "version",
                "criteria_version",
                "mechanics_readiness_report_id",
                "card_specs_hash",
                "ability_specs_hash",
                "cards",
            },
            "normal-mode policy form evidence",
        )
        if value.get("version") != cls.VERSION:
            raise SemanticSubsetError(f"unsupported form evidence version: {value.get('version')}")
        return cls(
            criteria_version=str(value["criteria_version"]),
            mechanics_readiness_report_id=str(value["mechanics_readiness_report_id"]),
            card_specs_hash=str(value["card_specs_hash"]),
            ability_specs_hash=str(value["ability_specs_hash"]),
            cards=tuple(NormalModePolicyFormCardEvidenceV1.from_mapping(item) for item in value.get("cards", ())),
        )


@dataclass(frozen=True, slots=True)
class NormalModePolicyReadinessV1:
    """Supported cards and stable checkpoint compatibility identifiers."""

    report_id: str
    static_catalog_id: str
    ready_card_ids: frozenset[int]
    excluded_card_ids: frozenset[int]
    card_names_by_id: Mapping[int, str]
    excluded_card_names_by_id: Mapping[int, str]
    card_evidence_sha256_by_id: Mapping[int, str]
    case_scope_id: str


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SemanticSubsetError(f"{label} must be an object")
    return value


@lru_cache(maxsize=1)
def load_normal_mode_policy_readiness() -> NormalModePolicyReadinessV1:
    """Load the compact supported-card manifest, verified by competitive_data."""
    manifest = load_competitive_data("card_support")
    compatibility = manifest["compatibility"]
    names = {row["card_id"]: row["name"] for row in manifest["supported_cards"]}
    excluded_names = {row["card_id"]: row["name"] for row in manifest["excluded_cards"]}
    if (
        manifest["version"] != "competitive-card-support.v1"
        or compatibility["mechanics_readiness_report_id"] != NORMAL_MODE_POLICY_MECHANICS_READINESS_REPORT_ID
        or compatibility["static_catalog_id"] != NORMAL_MODE_POLICY_STATIC_CATALOG_ID
        or compatibility["case_scope_id"] != NORMAL_MODE_POLICY_CASE_SCOPE_ID
        or frozenset(names) != NORMAL_MODE_POLICY_CARD_IDS
        or frozenset(excluded_names) != NORMAL_MODE_POLICY_EXCLUDED_CARD_IDS
    ):
        raise SemanticSubsetError("card support compatibility identity mismatch")
    return NormalModePolicyReadinessV1(
        report_id=compatibility["mechanics_readiness_report_id"],
        static_catalog_id=compatibility["static_catalog_id"],
        ready_card_ids=frozenset(names),
        excluded_card_ids=frozenset(excluded_names),
        card_names_by_id=MappingProxyType(names),
        excluded_card_names_by_id=MappingProxyType(excluded_names),
        card_evidence_sha256_by_id=MappingProxyType({
            row["card_id"]: row["compatibility_sha256"] for row in manifest["supported_cards"]
        }),
        case_scope_id=compatibility["case_scope_id"],
    )


def policy_ready_form_mask(card_id: int, *, form_evidence: NormalModePolicyFormEvidenceV1 | None = None) -> int:
    """Return only the form bits backed by the versioned form evidence."""

    evidence = form_evidence or DEFAULT_NORMAL_MODE_POLICY_FORM_EVIDENCE
    return evidence.form_masks.get(int(card_id), 0)


def exclusion_reasons(spec: CardSpecV1) -> tuple[str, ...]:
    """Classify a card against the validated supported-card manifest."""

    if spec.attributes.get("NotVisible") or spec.attributes.get("NotInUse"):
        return (REASON_CATALOG_UNAVAILABLE,)
    readiness = load_normal_mode_policy_readiness()
    if spec.card_id not in readiness.ready_card_ids:
        return (REASON_MISSING_MECHANICS_READINESS,)
    return ()


@dataclass(frozen=True, slots=True)
class SemanticSupportedCardV1(ContractMixin):
    card_id: int
    name: str
    kind: str
    card_spec_sha256: str
    mechanics_readiness_sha256: str | None = None
    allowed_form_availability: int = 0
    exclusion_reasons: tuple[str, ...] = ()
    version: str = field(default=SEMANTIC_SUBSET_CARD_VERSION, init=False)
    VERSION: ClassVar[str] = SEMANTIC_SUBSET_CARD_VERSION

    def __post_init__(self) -> None:
        if self.card_id <= 0 or not self.name:
            raise SemanticSubsetError("supported-subset card requires positive ID and name")
        if self.kind not in {item.value for item in CardKind}:
            raise SemanticSubsetError(f"unknown supported-subset card kind: {self.kind}")
        _require_sha256(self.card_spec_sha256, "card_spec_sha256")
        if self.mechanics_readiness_sha256 is not None:
            _require_sha256(self.mechanics_readiness_sha256, "mechanics_readiness_sha256")
        if isinstance(self.allowed_form_availability, bool) or self.allowed_form_availability not in range(4):
            raise SemanticSubsetError("allowed_form_availability must be an integer bit mask in 0..3")
        reasons = tuple(sorted(set(str(item) for item in self.exclusion_reasons)))
        if any(not item for item in reasons):
            raise SemanticSubsetError("exclusion reasons must be non-empty strings")
        if reasons and self.allowed_form_availability:
            raise SemanticSubsetError("excluded cards cannot enable an evolution or Hero form")
        object.__setattr__(self, "exclusion_reasons", reasons)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SemanticSupportedCardV1":
        _reject_unknown(
            value,
            {
                "version",
                "card_id",
                "name",
                "kind",
                "card_spec_sha256",
                "mechanics_readiness_sha256",
                "allowed_form_availability",
                "exclusion_reasons",
            },
            "semantic supported card",
        )
        if value.get("version") != cls.VERSION:
            raise SemanticSubsetError(f"unsupported semantic supported card version: {value.get('version')}")
        return cls(
            card_id=int(value["card_id"]),
            name=str(value["name"]),
            kind=str(value["kind"]),
            card_spec_sha256=str(value["card_spec_sha256"]),
            mechanics_readiness_sha256=(
                str(value["mechanics_readiness_sha256"])
                if value.get("mechanics_readiness_sha256") is not None
                else None
            ),
            allowed_form_availability=int(value.get("allowed_form_availability", 0)),
            exclusion_reasons=tuple(str(item) for item in value.get("exclusion_reasons", ())),
        )


@dataclass(frozen=True, slots=True)
class SemanticSupportedSubsetV1(ContractMixin):
    catalog_version: str
    card_specs_hash: str
    ability_specs_hash: str
    source_release: str
    criteria_version: str
    native_capability_profile: str
    mechanics_readiness_report_id: str
    normal_mode_scope_version: str
    form_evidence_version: str
    form_evidence_sha256: str
    allowed_cards: tuple[SemanticSupportedCardV1, ...]
    excluded_cards: tuple[SemanticSupportedCardV1, ...]
    baseline_deck: tuple[int, ...]
    speed_xbow_deck: tuple[int, ...]
    speed_xbow_form_availability: tuple[int, ...]
    pekka_bridge_spam_deck: tuple[int, ...]
    pekka_bridge_spam_form_availability: tuple[int, ...]
    minimum_battle_cards: int = MINIMUM_BATTLE_CARDS
    version: str = field(default=SEMANTIC_SUBSET_VERSION, init=False)
    VERSION: ClassVar[str] = SEMANTIC_SUBSET_VERSION

    def __post_init__(self) -> None:
        if self.catalog_version != CATALOG_VERSION:
            raise SemanticSubsetError(f"semantic subset requires catalog {CATALOG_VERSION}, got {self.catalog_version}")
        _require_sha256(self.card_specs_hash, "card_specs_hash")
        _require_sha256(self.ability_specs_hash, "ability_specs_hash")
        if self.criteria_version != SEMANTIC_SUBSET_CRITERIA_VERSION:
            raise SemanticSubsetError(f"unsupported semantic subset criteria: {self.criteria_version}")
        if self.native_capability_profile != NATIVE_SEMANTIC_CAPABILITY_PROFILE:
            raise SemanticSubsetError(f"unsupported native semantic profile: {self.native_capability_profile}")
        if self.mechanics_readiness_report_id != NORMAL_MODE_POLICY_MECHANICS_READINESS_REPORT_ID:
            raise SemanticSubsetError("unsupported normal-mode mechanics readiness report")
        if self.normal_mode_scope_version != NORMAL_MODE_POLICY_SCOPE_VERSION:
            raise SemanticSubsetError("unsupported normal-mode policy scope")
        if not self.form_evidence_version:
            raise SemanticSubsetError("normal-mode form evidence version is required")
        _require_sha256(self.form_evidence_sha256, "form_evidence_sha256")
        allowed = tuple(
            sorted(
                (
                    item if isinstance(item, SemanticSupportedCardV1) else SemanticSupportedCardV1.from_mapping(item)
                    for item in self.allowed_cards
                ),
                key=lambda item: item.card_id,
            )
        )
        excluded = tuple(
            sorted(
                (
                    item if isinstance(item, SemanticSupportedCardV1) else SemanticSupportedCardV1.from_mapping(item)
                    for item in self.excluded_cards
                ),
                key=lambda item: item.card_id,
            )
        )
        if any(item.exclusion_reasons for item in allowed):
            raise SemanticSubsetError("allowed cards cannot carry exclusion reasons")
        if any(item.mechanics_readiness_sha256 is None for item in allowed):
            raise SemanticSubsetError("allowed cards require card-scoped mechanics readiness evidence")
        if any(not item.exclusion_reasons for item in excluded):
            raise SemanticSubsetError("excluded cards require at least one explicit reason")
        if any(item.mechanics_readiness_sha256 is not None for item in excluded):
            raise SemanticSubsetError("excluded cards cannot carry admission readiness evidence")
        all_ids = [item.card_id for item in (*allowed, *excluded)]
        if len(set(all_ids)) != len(all_ids):
            raise SemanticSubsetError("allowed/excluded card decisions overlap or duplicate IDs")
        if self.minimum_battle_cards != MINIMUM_BATTLE_CARDS:
            raise SemanticSubsetError("V1 semantic subset requires an eight-card minimum")
        if len(allowed) < self.minimum_battle_cards:
            raise SemanticSubsetError(
                f"semantic subset has {len(allowed)} cards; at least {self.minimum_battle_cards} are required"
            )
        baseline = tuple(int(card_id) for card_id in self.baseline_deck)
        if len(baseline) != 8 or len(set(baseline)) != 8:
            raise SemanticSubsetError("semantic baseline deck must contain eight distinct cards")
        allowed_ids = {item.card_id for item in allowed}
        if not set(baseline).issubset(allowed_ids):
            raise SemanticSubsetError("semantic baseline deck contains excluded cards")
        speed_xbow = tuple(int(card_id) for card_id in self.speed_xbow_deck)
        speed_forms = tuple(int(mask) for mask in self.speed_xbow_form_availability)
        if speed_xbow != SEMANTIC_SPEED_XBOW_DECK:
            raise SemanticSubsetError("semantic subset must bind the exact speed-X-Bow deck order")
        if speed_forms != SEMANTIC_SPEED_XBOW_FORM_AVAILABILITY:
            raise SemanticSubsetError("semantic subset must bind the exact speed-X-Bow form masks")
        if not set(speed_xbow).issubset(allowed_ids):
            raise SemanticSubsetError("semantic speed-X-Bow deck contains excluded cards")
        allowed_by_id = {item.card_id: item for item in allowed}
        for card_id, form_mask in zip(speed_xbow, speed_forms, strict=True):
            allowed_mask = allowed_by_id[card_id].allowed_form_availability
            if form_mask & ~allowed_mask:
                raise SemanticSubsetError(
                    f"speed-X-Bow card {card_id} requests unsupported form bits {form_mask & ~allowed_mask}"
                )
        pekka_bridge_spam = tuple(int(card_id) for card_id in self.pekka_bridge_spam_deck)
        pekka_forms = tuple(int(mask) for mask in self.pekka_bridge_spam_form_availability)
        if pekka_bridge_spam != SEMANTIC_PEKKA_BRIDGE_SPAM_DECK:
            raise SemanticSubsetError("semantic subset must bind the exact P.E.K.K.A bridge-spam deck order")
        if pekka_forms != SEMANTIC_PEKKA_BRIDGE_SPAM_FORM_AVAILABILITY:
            raise SemanticSubsetError("semantic subset must bind the exact P.E.K.K.A bridge-spam form masks")
        if not set(pekka_bridge_spam).issubset(allowed_ids):
            raise SemanticSubsetError("semantic P.E.K.K.A bridge-spam deck contains excluded cards")
        for card_id, form_mask in zip(pekka_bridge_spam, pekka_forms, strict=True):
            allowed_mask = allowed_by_id[card_id].allowed_form_availability
            if form_mask & ~allowed_mask:
                raise SemanticSubsetError(
                    f"P.E.K.K.A bridge-spam card {card_id} requests unsupported form bits {form_mask & ~allowed_mask}"
                )
        object.__setattr__(self, "allowed_cards", allowed)
        object.__setattr__(self, "excluded_cards", excluded)
        object.__setattr__(self, "baseline_deck", baseline)
        object.__setattr__(self, "speed_xbow_deck", speed_xbow)
        object.__setattr__(self, "speed_xbow_form_availability", speed_forms)
        object.__setattr__(self, "pekka_bridge_spam_deck", pekka_bridge_spam)
        object.__setattr__(self, "pekka_bridge_spam_form_availability", pekka_forms)

    @property
    def subset_id(self) -> str:
        return content_hash(self)

    @property
    def allowed_ids(self) -> frozenset[int]:
        return frozenset(item.card_id for item in self.allowed_cards)

    @property
    def excluded_by_id(self) -> dict[int, SemanticSupportedCardV1]:
        return {item.card_id: item for item in self.excluded_cards}

    @property
    def exclusion_reason_counts(self) -> dict[str, int]:
        counts = Counter(reason for item in self.excluded_cards for reason in item.exclusion_reasons)
        return dict(sorted(counts.items()))

    def binding(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "criteria_version": self.criteria_version,
            "native_capability_profile": self.native_capability_profile,
            "mechanics_readiness_report_id": (self.mechanics_readiness_report_id),
            "normal_mode_scope_version": self.normal_mode_scope_version,
            "form_evidence_version": self.form_evidence_version,
            "form_evidence_sha256": self.form_evidence_sha256,
            "subset_id": self.subset_id,
            "card_specs_hash": self.card_specs_hash,
            "ability_specs_hash": self.ability_specs_hash,
            "allowed_card_count": len(self.allowed_cards),
            "excluded_card_count": len(self.excluded_cards),
            "minimum_battle_cards": self.minimum_battle_cards,
            "speed_xbow_deck": self.speed_xbow_deck,
            "speed_xbow_form_availability": (self.speed_xbow_form_availability),
            "pekka_bridge_spam_deck": self.pekka_bridge_spam_deck,
            "pekka_bridge_spam_form_availability": (self.pekka_bridge_spam_form_availability),
            "fail_closed_before_native_mutation": True,
        }

    def assert_deck(
        self, deck: Iterable[int], *, form_availability: Iterable[int] | None = None, label: str = "deck"
    ) -> tuple[int, ...]:
        cards = tuple(int(card_id) for card_id in deck)
        if len(cards) != 8 or len(set(cards)) != 8 or any(card_id <= 0 for card_id in cards):
            raise SemanticSubsetError(f"{label} must contain eight distinct positive card IDs")
        rejected: list[str] = []
        excluded = self.excluded_by_id
        for card_id in sorted(set(cards).difference(self.allowed_ids)):
            decision = excluded.get(card_id)
            if decision is None:
                rejected.append(f"{card_id}:not_in_bound_catalog")
            else:
                rejected.append(f"{card_id}:{decision.name}[{','.join(decision.exclusion_reasons)}]")
        if rejected:
            raise SemanticSubsetError(
                f"{label} crosses semantic supported-subset {self.subset_id}: " + "; ".join(rejected)
            )
        forms = (0,) * len(cards) if form_availability is None else tuple(form_availability)
        if len(forms) != len(cards) or any(
            isinstance(mask, bool) or not isinstance(mask, int) or mask not in range(4) for mask in forms
        ):
            raise SemanticSubsetError(f"{label} form availability must contain eight integer masks in 0..3")
        allowed_by_id = {item.card_id: item for item in self.allowed_cards}
        rejected_forms = []
        for slot, (card_id, requested) in enumerate(zip(cards, forms, strict=True)):
            supported = allowed_by_id[card_id].allowed_form_availability
            unsupported = requested & ~supported
            if unsupported:
                rejected_forms.append(f"slot{slot}:{card_id}:requested={requested}:supported={supported}")
        if rejected_forms:
            raise SemanticSubsetError(f"{label} enables unsupported semantic forms: " + "; ".join(rejected_forms))
        return cards

    @lru_cache(maxsize=16)
    def verify_catalog(
        self, catalog: CardSpecCatalog, *, form_evidence: NormalModePolicyFormEvidenceV1 | None = None
    ) -> None:
        if (
            catalog.version != self.catalog_version
            or catalog.specs_hash != self.card_specs_hash
            or catalog.ability_specs_hash != self.ability_specs_hash
            or catalog.source_release != self.source_release
        ):
            raise SemanticSubsetError("semantic subset catalog identity mismatch")
        evidence = form_evidence
        if evidence is None:
            if (
                self.form_evidence_version != DEFAULT_NORMAL_MODE_POLICY_FORM_EVIDENCE.criteria_version
                or self.form_evidence_sha256 != DEFAULT_NORMAL_MODE_POLICY_FORM_EVIDENCE.evidence_id
            ):
                raise SemanticSubsetError("non-default form evidence is required to verify this subset")
            evidence = DEFAULT_NORMAL_MODE_POLICY_FORM_EVIDENCE
        if self.form_evidence_version != evidence.criteria_version or self.form_evidence_sha256 != evidence.evidence_id:
            raise SemanticSubsetError("semantic subset form evidence mismatch")
        expected = build_semantic_supported_subset(catalog, form_evidence=evidence)
        if expected.to_dict() != self.to_dict():
            raise SemanticSubsetError("semantic subset decisions do not match the bound catalog and criteria")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SemanticSupportedSubsetV1":
        _reject_unknown(
            value,
            {
                "version",
                "catalog_version",
                "card_specs_hash",
                "ability_specs_hash",
                "source_release",
                "criteria_version",
                "native_capability_profile",
                "mechanics_readiness_report_id",
                "normal_mode_scope_version",
                "form_evidence_version",
                "form_evidence_sha256",
                "allowed_cards",
                "excluded_cards",
                "baseline_deck",
                "speed_xbow_deck",
                "speed_xbow_form_availability",
                "pekka_bridge_spam_deck",
                "pekka_bridge_spam_form_availability",
                "minimum_battle_cards",
            },
            "semantic supported subset",
        )
        if value.get("version") != cls.VERSION:
            raise SemanticSubsetError(f"unsupported semantic supported subset version: {value.get('version')}")
        return cls(
            catalog_version=str(value["catalog_version"]),
            card_specs_hash=str(value["card_specs_hash"]),
            ability_specs_hash=str(value["ability_specs_hash"]),
            source_release=str(value["source_release"]),
            criteria_version=str(value["criteria_version"]),
            native_capability_profile=str(value["native_capability_profile"]),
            mechanics_readiness_report_id=str(value["mechanics_readiness_report_id"]),
            normal_mode_scope_version=str(value["normal_mode_scope_version"]),
            form_evidence_version=str(value["form_evidence_version"]),
            form_evidence_sha256=str(value["form_evidence_sha256"]),
            allowed_cards=tuple(SemanticSupportedCardV1.from_mapping(item) for item in value.get("allowed_cards", ())),
            excluded_cards=tuple(
                SemanticSupportedCardV1.from_mapping(item) for item in value.get("excluded_cards", ())
            ),
            baseline_deck=tuple(int(item) for item in value.get("baseline_deck", ())),
            speed_xbow_deck=tuple(int(item) for item in value.get("speed_xbow_deck", ())),
            speed_xbow_form_availability=tuple(int(item) for item in value.get("speed_xbow_form_availability", ())),
            pekka_bridge_spam_deck=tuple(int(item) for item in value.get("pekka_bridge_spam_deck", ())),
            pekka_bridge_spam_form_availability=tuple(
                int(item) for item in value.get("pekka_bridge_spam_form_availability", ())
            ),
            minimum_battle_cards=int(value.get("minimum_battle_cards", 0)),
        )

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_json(pretty=True) + "\n"
        if destination.exists():
            if destination.read_text(encoding="utf-8") != payload:
                raise FileExistsError(f"immutable semantic subset path contains different bytes: {destination}")
            return destination
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "SemanticSupportedSubsetV1":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))


def _validate_policy_catalog(catalog: CardSpecCatalog, readiness: NormalModePolicyReadinessV1) -> None:
    if (
        catalog.version != CATALOG_VERSION
        or catalog.specs_hash != NORMAL_MODE_POLICY_CARD_SPECS_HASH
        or catalog.ability_specs_hash != NORMAL_MODE_POLICY_ABILITY_SPECS_HASH
        or catalog.source_release != NORMAL_MODE_POLICY_SOURCE_RELEASE
    ):
        raise SemanticSubsetError("card catalog is not the definition set bound to policy readiness")
    catalog_ids = frozenset(spec.card_id for spec in catalog.specs)
    if catalog_ids != (NORMAL_MODE_POLICY_CARD_IDS | NORMAL_MODE_POLICY_EXCLUDED_CARD_IDS):
        raise SemanticSubsetError("policy readiness catalog partition drifted")
    for spec in catalog.specs:
        unavailable = bool(spec.attributes.get("NotVisible") or spec.attributes.get("NotInUse"))
        if spec.card_id in readiness.ready_card_ids:
            if unavailable or readiness.card_names_by_id[spec.card_id] != spec.name:
                raise SemanticSubsetError(f"ready card {spec.card_id} identity or availability drifted")
        elif (
            spec.card_id not in readiness.excluded_card_ids
            or not unavailable
            or readiness.excluded_card_names_by_id[spec.card_id] != spec.name
        ):
            raise SemanticSubsetError(f"excluded card {spec.card_id} identity or availability drifted")


def validate_normal_mode_policy_form_evidence(
    catalog: CardSpecCatalog,
    form_evidence: NormalModePolicyFormEvidenceV1,
    *,
    readiness: NormalModePolicyReadinessV1 | None = None,
) -> None:
    """Validate injected form evidence against catalog and base readiness."""

    if not isinstance(form_evidence, NormalModePolicyFormEvidenceV1):
        raise SemanticSubsetError("form_evidence must be NormalModePolicyFormEvidenceV1")
    closed = readiness or load_normal_mode_policy_readiness()
    if (
        form_evidence.mechanics_readiness_report_id != closed.report_id
        or form_evidence.card_specs_hash != catalog.specs_hash
        or form_evidence.ability_specs_hash != catalog.ability_specs_hash
    ):
        raise SemanticSubsetError("form evidence is not bound to the active policy catalog")
    by_id = catalog.by_id
    for row in form_evidence.cards:
        spec = by_id.get(row.card_id)
        if spec is None or row.card_id not in closed.ready_card_ids:
            raise SemanticSubsetError(f"form evidence card {row.card_id} is not base-policy ready")
        if row.allowed_form_availability & 1 and spec.evolution is None:
            raise SemanticSubsetError(f"form evidence enables missing evolution for card {row.card_id}")
        if row.allowed_form_availability & 2 and not spec.ability_ids:
            raise SemanticSubsetError(f"form evidence enables missing Hero form for card {row.card_id}")


def _decision(
    spec: CardSpecV1,
    reasons: tuple[str, ...],
    *,
    mechanics_readiness_sha256: str | None,
    form_evidence: NormalModePolicyFormEvidenceV1,
) -> SemanticSupportedCardV1:
    return SemanticSupportedCardV1(
        card_id=spec.card_id,
        name=spec.name,
        kind=spec.kind.value,
        card_spec_sha256=content_hash(spec.to_dict()),
        mechanics_readiness_sha256=mechanics_readiness_sha256,
        allowed_form_availability=(
            policy_ready_form_mask(spec.card_id, form_evidence=form_evidence) if not reasons else 0
        ),
        exclusion_reasons=reasons,
    )


def build_semantic_supported_subset(
    catalog: CardSpecCatalog, *, form_evidence: NormalModePolicyFormEvidenceV1 | None = None
) -> SemanticSupportedSubsetV1:
    """Build the exact current supported-card partition for ``catalog``."""

    readiness = load_normal_mode_policy_readiness()
    _validate_policy_catalog(catalog, readiness)
    forms = form_evidence or DEFAULT_NORMAL_MODE_POLICY_FORM_EVIDENCE
    validate_normal_mode_policy_form_evidence(catalog, forms, readiness=readiness)
    allowed: list[SemanticSupportedCardV1] = []
    excluded: list[SemanticSupportedCardV1] = []
    for spec in catalog.specs:
        reasons = exclusion_reasons(spec)
        evidence_sha256 = readiness.card_evidence_sha256_by_id[spec.card_id] if not reasons else None
        (excluded if reasons else allowed).append(
            _decision(spec, reasons, mechanics_readiness_sha256=evidence_sha256, form_evidence=forms)
        )
    return SemanticSupportedSubsetV1(
        catalog_version=catalog.version,
        card_specs_hash=catalog.specs_hash,
        ability_specs_hash=catalog.ability_specs_hash,
        source_release=catalog.source_release,
        criteria_version=SEMANTIC_SUBSET_CRITERIA_VERSION,
        native_capability_profile=NATIVE_SEMANTIC_CAPABILITY_PROFILE,
        mechanics_readiness_report_id=readiness.report_id,
        normal_mode_scope_version=NORMAL_MODE_POLICY_SCOPE_VERSION,
        form_evidence_version=forms.criteria_version,
        form_evidence_sha256=forms.evidence_id,
        allowed_cards=tuple(allowed),
        excluded_cards=tuple(excluded),
        baseline_deck=SEMANTIC_BASELINE_DECK,
        speed_xbow_deck=SEMANTIC_SPEED_XBOW_DECK,
        speed_xbow_form_availability=(SEMANTIC_SPEED_XBOW_FORM_AVAILABILITY),
        pekka_bridge_spam_deck=SEMANTIC_PEKKA_BRIDGE_SPAM_DECK,
        pekka_bridge_spam_form_availability=(SEMANTIC_PEKKA_BRIDGE_SPAM_FORM_AVAILABILITY),
    )


from .card_specs import build_card_catalog
from .normal_form_evidence import build_normal_mode_policy_form_evidence, NORMAL_MODE_POLICY_FORM_EVIDENCE_SHA256


# Build the canonical 62-card form contract only after all semantic contract
# types and readiness validators exist, then pin the resulting content hash.
with competitive_data_initialization():
    DEFAULT_NORMAL_MODE_POLICY_FORM_EVIDENCE = build_normal_mode_policy_form_evidence(build_card_catalog())
if DEFAULT_NORMAL_MODE_POLICY_FORM_EVIDENCE.evidence_id != NORMAL_MODE_POLICY_FORM_EVIDENCE_SHA256:
    raise SemanticSubsetError(
        f"canonical normal-mode form evidence identity drifted: {DEFAULT_NORMAL_MODE_POLICY_FORM_EVIDENCE.evidence_id}"
    )
NORMAL_MODE_POLICY_FORM_MASKS: Mapping[int, int] = MappingProxyType(
    dict(DEFAULT_NORMAL_MODE_POLICY_FORM_EVIDENCE.form_masks)
)
