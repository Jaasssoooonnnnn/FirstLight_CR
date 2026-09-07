"""Install a locally built offline APK on a user-prepared, rooted Android AVD."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from native_runner.local_config import setting
    from native_runner import offline_install
    from native_runner.training.v4 import configure_emulator_engine_guests as guests

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adb", default=setting("CR_TRAINING_ADB", "adb"))
    parser.add_argument("--serial", required=True)
    parser.add_argument("--apk", type=Path, required=True)
    args = parser.parse_args()
    guests.ADB = Path(args.adb)
    guests.wait_root(args.serial)
    guests.adb(args.serial, "shell", "am", "force-stop", guests.PACKAGE)
    guests.configure_firewall(args.serial)
    return offline_install.main(["--adb", args.adb, "--serial", args.serial, "--apk", str(args.apk)])


if __name__ == "__main__":
    raise SystemExit(main())
