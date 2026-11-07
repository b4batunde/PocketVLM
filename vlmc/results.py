"""Shared helpers for reading measured results (report, demo, Mac runner)."""

import json
from pathlib import Path


def load_dir(d: Path) -> dict:
    d = Path(d)
    if not d.exists():
        return {}
    return {p.stem: json.load(open(p)) for p in sorted(d.glob("*.json")) if not p.name.endswith(".raw.json")}


def recommended(cfg: dict, baseline: str = "ft-bf16") -> dict | None:
    """Smallest llama.cpp variant, evaluated on the full eval set, keeping >= retain_threshold of baseline.

    Only pod-side sweep results count (names starting with 'mac-' are cross-device re-runs)."""
    R = Path(cfg["work_dir"]) / "results"
    hf, lc = load_dir(R / "hf"), load_dir(R / "llamacpp")
    if baseline not in hf:
        return None
    b = hf[baseline]
    ok = [r for n, r in lc.items() if not n.startswith("mac-") and r["n"] == b["n"]
          and r["vqa_acc"] >= cfg["retain_threshold"] * b["vqa_acc"]]
    return min(ok, key=lambda r: r["disk_bytes"]) if ok else None
