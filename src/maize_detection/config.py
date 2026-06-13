"""Load configs/baseline.yaml as a plain dict.

A thin wrapper so every module reads the same config the same way. Validation is
deliberately light in V1 — the canonical-label contract is enforced in labels.py,
not here.
"""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "baseline.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
