"""Check the current source-release file set, excluding Git-ignored local inputs."""
from pathlib import Path
import subprocess
import json
import hashlib
import gzip

ROOT = Path(__file__).resolve().parents[1]


def main():
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
    ).decode("utf-8").split("\0")
    failures = []
    model_manifest = json.loads((ROOT / "checkpoints/manifest.json").read_text(encoding="utf-8"))
    models = {"checkpoints/" + item["path"]: item for item in model_manifest["models"]}
    count = size = 0
    for name in sorted(set(names)):
        path = ROOT / name
        if not name or not path.is_file():
            continue
        count += 1
        size += path.stat().st_size
        if path.suffix.lower() in {".apk", ".so", ".scdb", ".keystore", ".jks", ".parquet"} or (path.suffix.lower() == ".pt" and name not in models):
            failures.append(f"Local binary/data in release: {name}")
        if path.name == ".env" or path.name.startswith(".env.") and path.name != ".env.example":
            failures.append(f"Local configuration in release: {name}")
        if name.startswith(("build/", "builds/", "user-provided/", ".runtime/", ".audit/", "runs/")):
            failures.append(f"Local artifact directory in release: {name}")
    template = ROOT / ".env.example"
    for name, entry in models.items():
        path = ROOT / name
        if name not in names or not path.is_file():
            failures.append(f"Public model missing or ignored: {name}")
        elif path.stat().st_size >= 100_000_000 or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            failures.append(f"Public model hash/size mismatch: {name}")
    data = ROOT / "native_runner/data/competitive"
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    for name, entry in manifest["files"].items():
        path = data / f"{name}.json"
        if not path.is_file():
            path = path.with_suffix(".json.gz")
        if not path.is_file() or path.relative_to(ROOT).as_posix() not in names:
            failures.append(f"Model companion data missing or ignored: {name}")
        else:
            raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                failures.append(f"Model companion data hash mismatch: {name}")
    for line in template.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#") and line.partition("=")[2].strip():
            failures.append(".env.example must contain only blank configuration values")
    for required in ("LICENSE", "NOTICE", "build_offline_engine.ps1", "native_runner/supported_engine.json"):
        if not (ROOT / required).is_file():
            failures.append(f"Missing release file: {required}")
    for failure in failures:
        print(failure)
    print(f"Source release: {count} files, {size} bytes, {len(failures)} errors")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
