"""Export a merged HF checkpoint to GGUF (LLM + mmproj), build an imatrix, and quantize.

  python scripts/06_export_gguf.py --hf runs/.../models/ft-merged --name ft
  python scripts/06_export_gguf.py --lora runs/.../adapters/lora --base Qwen/Qwen2.5-VL-3B-Instruct --name lora

Outputs in <work_dir>/gguf/:  <name>-F16.gguf, <name>-<TYPE>.gguf, mmproj-<name>-<TYPE>.gguf, imatrix-<name>.gguf
Env: LLAMA_CPP=path/to/llama.cpp (source dir, has convert_*.py and build/bin/)
     CONVERT_PREFIX="python tests/smoke/convert_shim.py"  (smoke tests only)
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from vlmc.config import load_config
from vlmc.data import load_items

ap = argparse.ArgumentParser()
ap.add_argument("--config")
ap.add_argument("--name", required=True)
ap.add_argument("--hf", help="merged HF checkpoint to convert")
ap.add_argument("--lora", help="PEFT adapter dir to convert to a GGUF LoRA (needs --base)")
ap.add_argument("--base", help="HF base model for --lora")
ap.add_argument("--types", nargs="*", help="override cfg.gguf.llm_types")
ap.add_argument("--mmproj-types", nargs="*", help="override cfg.gguf.mmproj_types")
ap.add_argument("--imatrix-ctx", type=int, default=512)
ap.add_argument("--no-imatrix", action="store_true")
a = ap.parse_args()
cfg = load_config(a.config)
G = cfg["gguf"]
wd = Path(cfg["work_dir"])
out = wd / "gguf"
out.mkdir(parents=True, exist_ok=True)
LCPP = Path(os.environ.get("LLAMA_CPP", "llama.cpp"))
BIN = Path(os.environ.get("LLAMA_BIN", LCPP / "build" / "bin"))
CONVERT = shlex.split(os.environ.get("CONVERT_PREFIX", sys.executable))


def sh(cmd):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def local(path_or_id):
    """convert_*.py need a local directory; resolve a Hub id through the HF cache."""
    if path_or_id and not Path(path_or_id).exists():
        from huggingface_hub import snapshot_download
        return snapshot_download(path_or_id)
    return path_or_id


a.hf, a.base = local(a.hf), local(a.base)
manifest = {"name": a.name, "files": {}}

if a.lora:
    assert a.base, "--lora needs --base"
    f = out / f"lora-{a.name}-F16.gguf"
    sh(CONVERT + [LCPP / "convert_lora_to_gguf.py", a.lora, "--base", a.base, "--outtype", "f16", "--outfile", f])
    manifest["files"]["lora"] = f.name
else:
    f16 = out / f"{a.name}-F16.gguf"
    if not f16.exists():
        sh(CONVERT + [LCPP / "convert_hf_to_gguf.py", a.hf, "--outtype", "f16", "--outfile", f16])
    manifest["files"]["F16"] = f16.name
    for t in a.mmproj_types or G["mmproj_types"]:
        mp = out / f"mmproj-{a.name}-{t}.gguf"
        if not mp.exists():
            sh(CONVERT + [LCPP / "convert_hf_to_gguf.py", a.hf, "--mmproj", "--outtype", t.lower(), "--outfile", mp])
        manifest["files"][f"mmproj-{t}"] = mp.name

    imat = None
    if G["use_imatrix"] and not a.no_imatrix:
        # Text-side calibration from the same calib split GPTQ/AWQ use (chat-formatted Q + A).
        calib_txt = out / "calib.txt"
        with open(calib_txt, "w") as fh:
            for it in load_items(wd / "data" / "calib.jsonl"):
                fh.write(f"<|im_start|>user\n{it['question']}\n{cfg['prompt_suffix']}<|im_end|>\n"
                         f"<|im_start|>assistant\n{it['answer']}<|im_end|>\n")
        imat = out / f"imatrix-{a.name}.gguf"
        if not imat.exists():
            sh([BIN / "llama-imatrix", "-m", f16, "-f", calib_txt, "-o", imat, "-c", a.imatrix_ctx, "-ngl", "99"])
        manifest["imatrix"] = imat.name

    for t in a.types or G["llm_types"]:
        q = out / f"{a.name}-{t}.gguf"
        if not q.exists():
            cmd = [BIN / "llama-quantize"] + (["--imatrix", imat] if imat else []) + [f16, q, t]
            sh(cmd)
        manifest["files"][t] = q.name

for k, v in manifest["files"].items():
    print(f"{k:12s} {(out / v).stat().st_size / 1e9:8.3f} GB  {v}")
json.dump(manifest, open(out / f"manifest-{a.name}.json", "w"), indent=2)
