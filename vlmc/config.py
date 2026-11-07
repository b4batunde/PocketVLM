import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | None = None) -> dict:
    path = path or os.environ.get("VLMC_CONFIG", ROOT / "configs" / "qwen25vl3b.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = str(path)
    return cfg
