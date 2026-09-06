"""Immutable card contracts for the supported competitive game release.

The source-derived records retain historical catalog slots for checkpoint identity.
Policy and deck validation admit only the 122 cards in normal-mode readiness.
"""

from __future__ import annotations
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any
from .contracts import (
    AbilitySpecV1,
    CardSpecV1,
    ContractError,
    ContractMixin,
    FrozenMapping,
    content_hash,
    frozen_mapping,
)
from .competitive_data import load_competitive_data, _manifest
from .paths import WORKSPACE_ROOT

CATALOG_VERSION = "card-spec-catalog.v1"


FIELD_APPLICABILITY_VERSION = "card-spec-field-applicability.v3"


COMPOUND_DECLARED = "compound_declared"


RAW_STATIC_ONLY = "raw_static_only"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def discover_card_source(workspace_root: str | Path, source_roots: Iterable[str | Path] = ()) -> Path:
    workspace = Path(workspace_root).resolve()
    manifest = _manifest()
    source = workspace / manifest["source_root"]
    for candidate in source_roots:
        root = Path(candidate).resolve()
        if source not in (root, root / "assets" / "csv_logic", root / "csv_logic"):
            raise ContractError("only the pinned competitive resource release is supported")
    return source


def build_card_catalog(
    workspace_root: str | Path | None = None, *, source_roots: Iterable[str | Path] = ()
) -> CardSpecCatalog:
    workspace = Path(workspace_root or WORKSPACE_ROOT).resolve()
    discover_card_source(workspace, source_roots)
    return CardSpecCatalog.from_mapping(load_competitive_data("card_specs", workspace))


def build_card_specs(
    workspace_root: str | Path | None = None, *, source_roots: Iterable[str | Path] = ()
) -> tuple[CardSpecV1, ...]:
    return build_card_catalog(workspace_root, source_roots=source_roots).specs


def build_ability_specs(
    workspace_root: str | Path | None = None, *, source_roots: Iterable[str | Path] = ()
) -> tuple[AbilitySpecV1, ...]:
    return build_card_catalog(workspace_root, source_roots=source_roots).abilities
