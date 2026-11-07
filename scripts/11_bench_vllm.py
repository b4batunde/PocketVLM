"""Real-kernel GPU numbers for the HF-format variants with vLLM (bf16 vs GPTQ-W4 vs AWQ-W4).

The HF harness dequantizes GPTQ/AWQ at load, so its speed/memory say nothing about 4-bit. vLLM runs
the packed weights with Marlin kernels, giving honest weight memory, latency and throughput.
2- and 3-bit have no vLLM kernels; those rows stay accuracy-only (stated in the report).

Runs in its OWN venv (vLLM pins its own torch/transformers):
  python -m venv /workspace/vllm-env && /workspace/vllm-env/bin/pip install -r requirements-vllm.txt
  /workspace/vllm-env/bin/python scripts/11_bench_vllm.py --name vllm-ft-gptq-w4 --model runs/.../ft-gptq-w4
"""
import argparse
import base64
import json
import logging
import os
import re
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # engine in-process so we can read its log
# FlashInfer's sampler JIT-compiles kernels with ninja/nvcc on first use; greedy decoding doesn't need it.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
# The venv's bin (ninja etc.) isn't on PATH when the venv python is called directly.
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")

from vllm import LLM, SamplingParams  # noqa: E402

from vlmc.config import load_config  # noqa: E402
from vlmc.data import load_items  # noqa: E402
from vlmc.vqa_metric import clean_prediction, vqa_accuracy  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config")
ap.add_argument("--name", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--limit", type=int, help="eval questions for accuracy/throughput (default: all)")
ap.add_argument("--latency-n", type=int, default=200)
ap.add_argument("--gpu-mem-util", type=float, default=0.85)
ap.add_argument("--meta", default="{}")
a = ap.parse_args()
cfg = load_config(a.config)
wd = Path(cfg["work_dir"])
items = load_items(wd / "data" / "eval.jsonl", a.limit)

model_gib = []


class Grab(logging.Handler):
    def emit(self, rec):
        m = re.search(r"Model loading took ([\d.]+) GiB", rec.getMessage())
        if m:
            model_gib.append(float(m.group(1)))


logging.getLogger("vllm").addHandler(Grab())
t0 = time.time()
common = dict(model=a.model, max_model_len=4096, limit_mm_per_prompt={"image": 1},
              gpu_memory_utilization=a.gpu_mem_util, seed=0,
              # no prefix / image caching: latency runs re-ask eval items, which would otherwise hit the cache
              enable_prefix_caching=False, mm_processor_cache_gb=0)
try:  # same visual-token cap as HF / llama.cpp
    llm = LLM(**common, mm_processor_kwargs={"min_pixels": cfg["image"]["min_pixels"],
                                             "max_pixels": cfg["image"]["max_pixels"]})
    image_cap = True
except (TypeError, ValueError) as e:
    print(f"WARN: mm_processor_kwargs rejected ({e}); running with the model's default image size")
    llm, image_cap = LLM(**common), False
load_s = time.time() - t0
if not model_gib:  # log capture is best-effort; ask the worker directly
    try:
        mem = llm.collective_rpc(lambda self: getattr(self.model_runner, "model_memory_usage", None))[0]
        model_gib.append(mem / 2**30 if mem else None)
    except Exception as e:
        print(f"WARN: could not read model memory from worker ({e})")


def conv(it):
    b64 = base64.b64encode(open(it["image_path"], "rb").read()).decode()
    return [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        {"type": "text", "text": f"{it['question']}\n{cfg['prompt_suffix']}"}]}]


greedy = SamplingParams(temperature=0, max_tokens=cfg["eval"]["max_new_tokens"])
llm.chat([conv(items[0])], greedy, use_tqdm=False)  # warm-up (CUDA graphs, kernels)

# 1) accuracy + offline throughput (whole eval set in one batch, vLLM schedules it)
t = time.time()
outs = llm.chat([conv(it) for it in items], greedy, use_tqdm=True)
thr_s = time.time() - t
preds = []
for it, o in zip(items, outs):
    p = clean_prediction(o.outputs[0].text)
    preds.append({"qid": it["qid"], "pred": p, "acc": vqa_accuracy(p, it["answers"]), "answer_type": it["answer_type"]})

# 2) single-stream end-to-end latency (one question at a time)
lat = []
for it in items[: a.latency_n]:
    t = time.time()
    llm.chat([conv(it)], greedy, use_tqdm=False)
    lat.append(time.time() - t)
lat.sort()

# 3) single-stream decode speed: time(128 tokens) - time(1 token) over the same image prompt
probe = conv(items[0])
probe[0]["content"][1]["text"] = items[0]["question"]
def timed(n):  # noqa: E302
    sp = SamplingParams(temperature=0, max_tokens=n, min_tokens=n, ignore_eos=True)
    ts = []
    for _ in range(3):
        t = time.time()
        llm.chat([probe], sp, use_tqdm=False)
        ts.append(time.time() - t)
    return statistics.median(ts)
t1, t128 = timed(1), timed(128)

by_type = {}
for p in preds:
    by_type.setdefault(p["answer_type"], []).append(p["acc"])
import torch  # noqa: E402
res = {
    "name": a.name, "backend": "vllm", "model": a.model, "n": len(preds),
    "vqa_acc": sum(p["acc"] for p in preds) / len(preds),
    "vqa_acc_by_type": {k: sum(v) / len(v) for k, v in by_type.items()},
    "weight_memory_gib": model_gib[0] if model_gib else None,
    "throughput_questions_per_s": len(preds) / thr_s,
    "latency_s": {"median": statistics.median(lat), "p90": lat[int(0.9 * (len(lat) - 1))], "n": len(lat)},
    "decode_tokens_per_s_single_stream": 127 / (t128 - t1),
    "image_cap_applied": image_cap, "gpu": torch.cuda.get_device_name(0), "load_s": load_s, **json.loads(a.meta),
}
out = wd / "results" / "vllm"
out.mkdir(parents=True, exist_ok=True)
json.dump(res, open(out / f"{a.name}.json", "w"), indent=2)
print(json.dumps(res, indent=2))
