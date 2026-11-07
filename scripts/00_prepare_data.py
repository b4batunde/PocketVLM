"""Materialize train / calib / eval splits (image-disjoint) under <work_dir>/data."""
import argparse
import json
from pathlib import Path

from vlmc.config import load_config
from vlmc.data import prepare_synthetic, prepare_vqav2

ap = argparse.ArgumentParser()
ap.add_argument("--config")
ap.add_argument("--synthetic", action="store_true", help="tiny shapes dataset for smoke tests")
ap.add_argument("--synthetic-n-train", type=int, default=64)
a = ap.parse_args()
cfg = load_config(a.config)
out = Path(cfg["work_dir"]) / "data"
out.mkdir(parents=True, exist_ok=True)
stats = prepare_synthetic(out, n_train=a.synthetic_n_train) if a.synthetic else prepare_vqav2(cfg, out)
print(json.dumps(stats, indent=2))
