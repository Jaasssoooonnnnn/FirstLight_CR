"""Code lives in this repository; game assets may live in a shared workspace."""

from .local_config import setting
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parent
WORKSPACE_ROOT = Path(setting("CR_WORKSPACE_ROOT", str(REPOSITORY_ROOT))).resolve()
