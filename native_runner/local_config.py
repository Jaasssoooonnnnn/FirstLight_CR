"""Repository-local deployment settings; values are literal, never shell code."""

import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def load_env(path: Path = ENV_FILE) -> None:
    if not path.is_file():
        return
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not separator or not key.isascii() or not key.isidentifier():
            raise ValueError(f"{path.name}:{number}: expected KEY=value")
        if value[:1] in ("'", '"'):
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError(f"{path.name}:{number}: unmatched quote")
            value = value[1:-1]
        if value:
            os.environ.setdefault(key, value)


def setting(key: str, default: str = "") -> str:
    return os.environ.get(key) or default


def required_setting(key: str) -> str:
    value = setting(key)
    if not value:
        raise ValueError(f"Set {key} in {ENV_FILE.name} (see .env.example)")
    return value
