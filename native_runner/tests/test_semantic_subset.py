from __future__ import annotations

from native_runner.paths import WORKSPACE_ROOT

from dataclasses import replace
import json

import pytest

from native_runner.card_specs import build_card_catalog
from native_runner.semantic_subset import (
    DEFAULT_NORMAL_MODE_POLICY_FORM_EVIDENCE,
    NORMAL_MODE_POLICY_CARD_IDS,
    NORMAL_MODE_POLICY_CASE_SCOPE_ID,
    NORMAL_MODE_POLICY_EXCLUDED_CARD_IDS,
    NORMAL_MODE_POLICY_FORM_EVIDENCE_SHA256,
    NORMAL_MODE_POLICY_FORM_EVIDENCE_VERSION,
    NORMAL_MODE_POLICY_FORM_MASKS,
    NORMAL_MODE_POLICY_MECHANICS_READINESS_REPORT_ID,
    NORMAL_MODE_POLICY_SCOPE_VERSION,
    NATIVE_SEMANTIC_CAPABILITY_PROFILE,
    REASON_CATALOG_UNAVAILABLE,
    SEMANTIC_BASELINE_DECK,
    SEMANTIC_PEKKA_BRIDGE_SPAM_DECK,
    SEMANTIC_PEKKA_BRIDGE_SPAM_FORM_AVAILABILITY,
    SEMANTIC_SPEED_XBOW_DECK,
    SEMANTIC_SPEED_XBOW_FORM_AVAILABILITY,
    SEMANTIC_SUBSET_CRITERIA_VERSION,
    SemanticSubsetError,
    SemanticSupportedSubsetV1,
    NormalModePolicyFormCardEvidenceV1,
    build_semantic_supported_subset,
    load_normal_mode_policy_readiness,
    policy_ready_form_mask,
)
import native_runner.semantic_subset as semantic_subset
from native_runner.ruleset import build_ruleset_manifest


WORKSPACE = WORKSPACE_ROOT


@pytest.fixture(scope="module")
def catalog():
    return build_card_catalog(WORKSPACE)


@pytest.fixture(scope="module")
def subset(catalog):
    return build_semantic_supported_subset(catalog)


def test_current_subset_is_explicit_complete_partition_and_has_a_deck(catalog, subset) -> None:
    subset.verify_catalog(catalog)

    assert len(subset.allowed_cards) == 122
    assert len(subset.excluded_cards) == 30
    assert len(subset.allowed_cards) + len(subset.excluded_cards) == len(catalog.specs)
    assert subset.allowed_ids == NORMAL_MODE_POLICY_CARD_IDS
    assert frozenset(subset.excluded_by_id) == NORMAL_MODE_POLICY_EXCLUDED_CARD_IDS
    assert all(
        item.mechanics_readiness_sha256 is not None
        for item in subset.allowed_cards
    )
    assert all(
        item.mechanics_readiness_sha256 is None
        for item in subset.excluded_cards
    )
    assert subset.assert_deck(SEMANTIC_BASELINE_DECK) == SEMANTIC_BASELINE_DECK
    assert subset.assert_deck(
        SEMANTIC_SPEED_XBOW_DECK,
        form_availability=SEMANTIC_SPEED_XBOW_FORM_AVAILABILITY,
    ) == SEMANTIC_SPEED_XBOW_DECK
    assert subset.assert_deck(
        SEMANTIC_PEKKA_BRIDGE_SPAM_DECK,
        form_availability=SEMANTIC_PEKKA_BRIDGE_SPAM_FORM_AVAILABILITY,
    ) == SEMANTIC_PEKKA_BRIDGE_SPAM_DECK
    assert len(subset.subset_id) == 64


def test_ruleset_manifest_binds_exact_subset_identity(catalog, subset) -> None:
    manifest = build_ruleset_manifest(WORKSPACE, catalog=catalog)

    assert dict(manifest.config["semantic_supported_subset"]) == subset.binding()
    assert (
        subset.mechanics_readiness_report_id
        == NORMAL_MODE_POLICY_MECHANICS_READINESS_REPORT_ID
    )
    assert subset.normal_mode_scope_version == NORMAL_MODE_POLICY_SCOPE_VERSION
    assert subset.form_evidence_version == NORMAL_MODE_POLICY_FORM_EVIDENCE_VERSION
    assert subset.form_evidence_sha256 == NORMAL_MODE_POLICY_FORM_EVIDENCE_SHA256


def test_normal_mode_base_cards_are_admitted_and_unavailable_cards_are_not(subset) -> None:
    excluded = subset.excluded_by_id
    allowed = {item.card_id: item for item in subset.allowed_cards}

    # Representatives of the old conservative blockers are now admitted by
    # card-scoped definition-exact evidence, including their proven forms.
    assert 26_000_072 in allowed  # Archer Queen / active ability
    assert 27_000_003 in allowed  # Inferno Tower / native placement
    assert allowed[26_000_004].allowed_form_availability == 1
    assert all(
        reason == REASON_CATALOG_UNAVAILABLE
        for item in excluded.values()
        for reason in item.exclusion_reasons
    )
    assert excluded[26_000_081].name == "SuperHogRiderTerry"

    # A Hero bit still fails closed on a card without an exact Hero binding.
    invalid_pekka_hero = list(
        SEMANTIC_PEKKA_BRIDGE_SPAM_FORM_AVAILABILITY
    )
    invalid_pekka_hero[3] = 2
    with pytest.raises(SemanticSubsetError, match="slot3:26000004"):
        subset.assert_deck(
            SEMANTIC_PEKKA_BRIDGE_SPAM_DECK,
            form_availability=invalid_pekka_hero,
            label="deck0",
        )


def test_fixed_readiness_report_is_closed_and_card_scoped() -> None:
    readiness = load_normal_mode_policy_readiness()

    assert readiness.report_id == NORMAL_MODE_POLICY_MECHANICS_READINESS_REPORT_ID
    assert readiness.ready_card_ids == NORMAL_MODE_POLICY_CARD_IDS
    assert readiness.excluded_card_ids == NORMAL_MODE_POLICY_EXCLUDED_CARD_IDS
    assert readiness.case_scope_id == NORMAL_MODE_POLICY_CASE_SCOPE_ID
    assert set(readiness.card_evidence_sha256_by_id) == NORMAL_MODE_POLICY_CARD_IDS
    assert all(
        len(value) == 64
        for value in readiness.card_evidence_sha256_by_id.values()
    )


def test_speed_xbow_gate_is_form_specific_and_fails_closed(subset) -> None:
    assert len(NORMAL_MODE_POLICY_FORM_MASKS) == 62
    assert policy_ready_form_mask(26_000_000) == 3
    assert policy_ready_form_mask(26_000_010) == 1
    assert policy_ready_form_mask(26_000_084) == 0

    # Musketeer has neither an evolution nor a Hero form in this contract.
    invalid_musketeer_evolution = list(SEMANTIC_SPEED_XBOW_FORM_AVAILABILITY)
    invalid_musketeer_evolution[5] = 1
    with pytest.raises(SemanticSubsetError, match="slot5:26000084"):
        subset.assert_deck(
            SEMANTIC_SPEED_XBOW_DECK,
            form_availability=invalid_musketeer_evolution,
            label="deck0",
        )

    # The selected Hero/evolution bits are also tied to their exact cards.
    invalid_hero_bit = list(SEMANTIC_SPEED_XBOW_FORM_AVAILABILITY)
    invalid_hero_bit[5] = 2
    with pytest.raises(SemanticSubsetError, match="slot5:26000084"):
        subset.assert_deck(
            SEMANTIC_SPEED_XBOW_DECK,
            form_availability=invalid_hero_bit,
            label="deck0",
        )


def test_injected_content_addressed_form_contract_can_restrict_masks(catalog) -> None:
    evidence = replace(
        DEFAULT_NORMAL_MODE_POLICY_FORM_EVIDENCE,
        criteria_version="normal-mode-policy-form-evidence.test-restriction",
        cards=tuple(
            item
            for item in DEFAULT_NORMAL_MODE_POLICY_FORM_EVIDENCE.cards
            if item.card_id != 26_000_010
        ),
    )

    extended = build_semantic_supported_subset(catalog, form_evidence=evidence)
    allowed = {item.card_id: item for item in extended.allowed_cards}
    assert allowed[26_000_010].allowed_form_availability == 0
    extended.verify_catalog(catalog, form_evidence=evidence)
    with pytest.raises(SemanticSubsetError, match="non-default form evidence"):
        extended.verify_catalog(catalog)


def test_injected_form_contract_rejects_catalog_capability_mismatch(catalog) -> None:
    unsupported_hog_evolution = NormalModePolicyFormCardEvidenceV1(
        card_id=26_000_021,
        allowed_form_availability=1,
        evidence_refs=(f"evolution:test.{'b' * 64}",),
    )
    evidence = replace(
        DEFAULT_NORMAL_MODE_POLICY_FORM_EVIDENCE,
        criteria_version="normal-mode-policy-form-evidence.invalid-test",
        cards=(
            *DEFAULT_NORMAL_MODE_POLICY_FORM_EVIDENCE.cards,
            unsupported_hog_evolution,
        ),
    )

    with pytest.raises(SemanticSubsetError, match="missing evolution"):
        build_semantic_supported_subset(catalog, form_evidence=evidence)


def test_contract_round_trip_is_strict_and_content_addressed(tmp_path, catalog, subset) -> None:
    path = tmp_path / f"{subset.subset_id}.json"
    subset.save(path)
    restored = SemanticSupportedSubsetV1.load(path)

    assert restored == subset
    assert restored.subset_id == subset.subset_id
    restored.verify_catalog(catalog)

    with pytest.raises(SemanticSubsetError, match="criteria"):
        SemanticSupportedSubsetV1.from_mapping({
            **subset.to_dict(),
            "criteria_version": "semantic-supported-subset-criteria.2026-08-12.1",
        })
    with pytest.raises(SemanticSubsetError, match="native semantic profile"):
        SemanticSupportedSubsetV1.from_mapping({
            **subset.to_dict(),
            "native_capability_profile": (
                "native-normal-mode-base-cards-position-hp-elixir-"
                "form-gated.v6"
            ),
        })
    assert subset.criteria_version == SEMANTIC_SUBSET_CRITERIA_VERSION
    assert subset.native_capability_profile == NATIVE_SEMANTIC_CAPABILITY_PROFILE


@pytest.mark.parametrize("field", ["card_id", "name", "compatibility_sha256", "form_availability"])
def test_support_manifest_rejects_modified_card(tmp_path, monkeypatch, field):
    from native_runner import competitive_data
    from native_runner.contracts import ContractError
    data = competitive_data.DATA_ROOT
    report = json.loads((data / "card_support.json").read_text(encoding="utf-8"))
    report["supported_cards"][0][field] = "changed"
    (tmp_path / "card_support.json").write_text(json.dumps(report), encoding="utf-8")
    (tmp_path / "manifest.json").write_bytes((data / "manifest.json").read_bytes())
    monkeypatch.setattr(competitive_data, "DATA_ROOT", tmp_path)
    semantic_subset.load_normal_mode_policy_readiness.cache_clear()
    try:
        with pytest.raises(ContractError, match="SHA-256 mismatch"):
            semantic_subset.load_normal_mode_policy_readiness()
    finally:
        semantic_subset.load_normal_mode_policy_readiness.cache_clear()


def test_deck_gate_reports_unavailable_card_reason(subset) -> None:
    bad_deck = (26_000_081, *SEMANTIC_BASELINE_DECK[:7])

    with pytest.raises(SemanticSubsetError) as caught:
        subset.assert_deck(bad_deck, label="deck0")

    message = str(caught.value)
    assert "26000081:SuperHogRiderTerry" in message
    assert REASON_CATALOG_UNAVAILABLE in message


def test_catalog_identity_mismatch_fails_closed(catalog, subset) -> None:
    tampered_catalog = replace(catalog, specs=catalog.specs[:-1])

    with pytest.raises(SemanticSubsetError, match="catalog identity mismatch"):
        subset.verify_catalog(tampered_catalog)


def test_build_rejects_catalog_drift_before_admission(catalog) -> None:
    tampered_catalog = replace(
        catalog,
        specs=(replace(catalog.specs[0], name="TamperedKnight"), *catalog.specs[1:]),
    )

    with pytest.raises(SemanticSubsetError, match="definition set"):
        build_semantic_supported_subset(tampered_catalog)


def test_contract_rejects_allowed_card_without_scoped_evidence(subset) -> None:
    first = replace(subset.allowed_cards[0], mechanics_readiness_sha256=None)

    with pytest.raises(SemanticSubsetError, match="card-scoped"):
        replace(subset, allowed_cards=(first, *subset.allowed_cards[1:]))
