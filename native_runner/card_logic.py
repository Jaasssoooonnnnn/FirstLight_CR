"""Load the pinned static gameplay graph; no activity or alternate-release parsing.

The graph is evidence data, with the original identity preserved for checkpoints.
Current gameplay is constrained by normal-mode policy contracts.
"""

from __future__ import annotations
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, ClassVar
from .card_specs import CardSpecCatalog, build_card_catalog, discover_card_source
from .competitive_data import load_competitive_data
from .contracts import ContractError, ContractMixin, content_hash, frozen_mapping
from .paths import WORKSPACE_ROOT

STATIC_CARD_LOGIC_VERSION = "static-card-logic-catalog.v1"


STATIC_CARD_LOGIC_CRITERIA_VERSION = "static-card-logic-criteria.2026-08-18.v8"


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


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
    discover_card_source(workspace, source_roots)
    data = load_competitive_data("card_logic", workspace)
    cards = card_catalog or build_card_catalog(workspace)
    if cards.content_hash() != data["card_catalog_content_hash"]:
        raise ContractError("card catalog differs from the pinned competitive release")
    return StaticCardLogicCatalogV1.from_mapping(data)
