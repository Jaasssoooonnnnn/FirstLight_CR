"""Load reviewed static facts bound to the exact competitive resource release.

No executable formulas or parser rules are stored in these files. The JSON is
the old compiler's output, including the historical catalog identities needed
to load existing checkpoints. Changed resources require an explicit rebuild.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
import hashlib
import gzip
import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import ContractError, frozen_mapping
from .paths import PACKAGE_ROOT, WORKSPACE_ROOT
from .local_config import setting

DATA_ROOT = PACKAGE_ROOT / "data" / "competitive"
GENERATED_DATA_ROOT = Path(setting("CR_COMPETITIVE_DATA_ROOT", str(DATA_ROOT)))
MANIFEST_SHA256 = "699c0c77cc5d2ec9a475ec0d10875f9e38710af9fc5472cc3431276384c89b78"
_verified_files: ContextVar[set[tuple[Path, str]] | None] = ContextVar("competitive_verified_files", default=None)


def _data_path(name: str) -> Path:
    root = DATA_ROOT if name in {"card_support", "form_evidence", "runtime"} else GENERATED_DATA_ROOT
    path = root / f"{name}.json"
    return path if path.is_file() else path.with_suffix(".json.gz")


@contextmanager
def competitive_data_initialization():
    """Verify each file once while building one immutable production snapshot."""
    token = _verified_files.set(set())
    try:
        yield
    finally:
        _verified_files.reset(token)


def _verify_file(path: Path, expected: str) -> None:
    verified = _verified_files.get()
    key = (path, expected)
    if verified is not None and key in verified:
        return
    try:
        with (gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")) as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError as error:
        raise ContractError(
            f"competitive resource unavailable: {path}; restore the release files or check the configured data directory"
        ) from error
    if digest != expected:
        raise ContractError(f"competitive resource SHA-256 mismatch: {path}")
    if verified is not None:
        verified.add(key)


@lru_cache(maxsize=1)
def _manifest() -> Mapping[str, Any]:
    raw = (DATA_ROOT / "manifest.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ContractError("competitive data manifest SHA-256 mismatch")
    return frozen_mapping(json.loads(raw))


def validate_competitive_sources(workspace_root: str | Path | None = None) -> Mapping[str, Any]:
    """Compiler-only audit of original inputs; runtime uses verified compiled facts."""
    _verify_file(DATA_ROOT / "manifest.json", MANIFEST_SHA256)
    manifest = _manifest()
    workspace = Path(workspace_root or WORKSPACE_ROOT).resolve()
    for relative, expected in manifest["source_files"].items():
        _verify_file(workspace / relative, expected)
    return manifest


@lru_cache(maxsize=8)
def _read_data(name: str, digest: str) -> Mapping[str, Any]:
    path = _data_path(name)
    raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ContractError(f"competitive data SHA-256 mismatch: {name}")
    return frozen_mapping(json.loads(raw))


def load_competitive_data(name: str, workspace_root: str | Path | None = None) -> Mapping[str, Any]:
    manifest = _manifest()
    if name not in manifest["files"]:
        raise ContractError(f"unknown competitive dataset: {name}")
    digest = manifest["files"][name]["sha256"]
    _verify_file(_data_path(name), digest)
    return _read_data(name, digest)
