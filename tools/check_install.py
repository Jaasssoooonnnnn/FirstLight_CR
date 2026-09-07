"""Verify shipped model/data compatibility without a VM or original game assets."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from native_runner.competitive_data import _manifest, load_competitive_data
    from native_runner.training.v4.policy_session import load_policy_v4

    torch.set_num_threads(1)
    for name in _manifest()["files"]:
        load_competitive_data(name)
    print("Companion data: all pinned hashes verified", flush=True)
    release = json.loads((ROOT / "checkpoints/manifest.json").read_text(encoding="utf-8"))
    for entry in release["models"]:
        loaded = load_policy_v4(ROOT / "checkpoints" / entry["path"], device="cpu")
        if loaded.checkpoint_sha256 != entry["sha256"]:
            raise ValueError(f"Released checkpoint changed: {entry['path']}")
        print(f"Model compatible: {entry['path']}", flush=True)
        del loaded
    print(f"Ready: {len(release['models'])} models; PyTorch {torch.__version__}; CUDA {torch.cuda.is_available()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
