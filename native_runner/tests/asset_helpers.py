"""Independent resource readers for source-to-compiled-data regression tests."""
import csv
from functools import lru_cache
import io
from pathlib import Path
import re
import zipfile
from sc_compression import decompress as _sc_decompress
from native_runner.contracts import ContractError
from typing import Any
from native_runner.local_config import setting


@lru_cache(maxsize=8)
def user_apk_bytes(name: str) -> bytes:
    """Optional original-input checks never require a private decoded directory."""
    import pytest
    apk = setting("CR_INPUT_APK")
    if not apk:
        pytest.skip("Original-input check requires the user-provided CR_INPUT_APK")
    with zipfile.ZipFile(apk) as archive:
        return archive.read(name)

_INT = re.compile(r"^[+-]?\d+$")


_FLOAT = re.compile(r"^[+-]?(?:\d+\.\d*|\d*\.\d+)(?:[eE][+-]?\d+)?$")


def _looks_like_text_asset(value: bytes) -> bool:
    prefix = value[:256]
    if b"\x00" in prefix:
        return False
    stripped = prefix.lstrip(b"\xef\xbb\xbf\r\n\t ")
    return stripped.startswith((b'"Name"', b"Name,", b"["))


@lru_cache(maxsize=512)
def _asset_bytes_cached(name: str, size: int, modified_ns: int) -> bytes | None:
    raw = Path(name).read_bytes()
    if _looks_like_text_asset(raw):
        return raw
    if _sc_decompress is None:
        return None
    try:
        plain, _signature, _version = _sc_decompress(raw)
    except Exception:
        return None
    return plain if _looks_like_text_asset(plain) else None


def asset_bytes(path: Path) -> bytes | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    stat = path.stat()
    return _asset_bytes_cached(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def _scalar(value: str | None) -> Any:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if _INT.fullmatch(text):
        return int(text)
    if _FLOAT.fullmatch(text):
        return float(text)
    return text


def read_sc_csv(path: Path) -> tuple[dict[str, str], list[dict[str, Any]]]:
    plain = asset_bytes(path)
    if plain is None:
        raise ContractError(f"asset table is not decompressed plaintext: {path}")
    with io.StringIO(plain.decode("utf-8-sig"), newline="") as stream:
        reader = csv.DictReader(stream)
        raw_rows = list(reader)
    if not raw_rows:
        return {}, []
    type_row = raw_rows[0]
    # SC tables use an object-name continuation convention.  Only named rows
    # allocate data-table IDs; this is why RoyalGiant is 26000024 rather than
    # 26000025 in the current capture.
    rows = [
        {key: _scalar(value) for key, value in row.items() if key is not None and _scalar(value) is not None}
        for row in raw_rows[1:]
        if row.get("Name", "").strip()
    ]
    return {key: str(value) for key, value in type_row.items() if key is not None and value}, rows
