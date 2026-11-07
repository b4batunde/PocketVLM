"""Evaluate a GGUF variant with llama.cpp's llama-server (same binary on the RunPod box and the Mac).

Accuracy uses the same eval split, prompt and VQA metric as the HF harness. Also records end-to-end
latency per question (image encode + prefill + decode), server timings, and peak server RSS.

  python scripts/07_eval_llamacpp.py --name ft-Q4_K_M --model ft-Q4_K_M.gguf --mmproj mmproj-ft-F16.gguf
  (paths relative to <work_dir>/gguf unless absolute)
"""
import argparse
import base64
import json
import os
import socket
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psutil
import requests

from vlmc.config import load_config
from vlmc.data import load_items
from vlmc.vqa_metric import clean_prediction, vqa_accuracy

ap = argparse.ArgumentParser()
ap.add_argument("--config")
ap.add_argument("--name", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--mmproj", required=True)
ap.add_argument("--lora", help="GGUF LoRA adapter applied at runtime (quantize-then-adapt)")
ap.add_argument("--split", default="eval")
ap.add_argument("--limit", type=int)
ap.add_argument("--parallel", type=int, help="server slots + client threads (use 1 for latency runs)")
ap.add_argument("--device-label", default=os.environ.get("VLMC_DEVICE", "unknown"),
                help="e.g. 'M3 MacBook Air 16GB' or 'RunPod L40S'; stored with the timings")
ap.add_argument("--ngl", default="99")
ap.add_argument("--threads", type=int)
ap.add_argument("--meta", default="{}")
a = ap.parse_args()
cfg = load_config(a.config)
wd = Path(cfg["work_dir"])
gdir = wd / "gguf"
res_path = lambda p: Path(p) if Path(p).is_absolute() or Path(p).exists() else gdir / p  # noqa: E731
model, mmproj = res_path(a.model), res_path(a.mmproj)
lora = res_path(a.lora) if a.lora else None
LCPP = Path(os.environ.get("LLAMA_CPP", "llama.cpp"))
BIN = Path(os.environ.get("LLAMA_BIN", LCPP / "build" / "bin"))
par = a.parallel or cfg["eval"]["llamacpp_parallel"]
items = load_items(wd / "data" / f"{a.split}.jsonl", a.limit)
max_img_tokens = cfg["image"]["max_pixels"] // (28 * 28)
min_img_tokens = cfg["image"]["min_pixels"] // (28 * 28)

with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
cmd = [str(BIN / "llama-server"), "-m", str(model), "--mmproj", str(mmproj), "--port", str(port),
       "-np", str(par), "-c", str(2048 * par), "-ngl", a.ngl, "--temp", "0", "--no-webui",
       "--image-max-tokens", str(max_img_tokens), "--image-min-tokens", str(min_img_tokens)]
if lora:
    cmd += ["--lora", str(lora)]
if a.threads:
    cmd += ["-t", str(a.threads)]
log = open(wd / f"llama-server-{a.name}.log", "w")
print("+", " ".join(cmd), flush=True)
srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
proc = psutil.Process(srv.pid)
peak_rss, stop = [0], threading.Event()


def watch():
    while not stop.is_set():
        try:
            peak_rss[0] = max(peak_rss[0], proc.memory_info().rss)
        except psutil.Error:
            pass
        time.sleep(0.2)


threading.Thread(target=watch, daemon=True).start()
base = f"http://127.0.0.1:{port}"
try:
    t0 = time.time()
    while True:
        if srv.poll() is not None:
            raise RuntimeError(f"llama-server exited; see {log.name}")
        try:
            if requests.get(base + "/health", timeout=1).status_code == 200:
                break
        except requests.RequestException:
            pass
        if time.time() - t0 > 600:
            raise TimeoutError("llama-server did not become healthy")
        time.sleep(0.5)
    load_s = time.time() - t0

    def ask(it):
        b64 = base64.b64encode(open(it["image_path"], "rb").read()).decode()
        body = {"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": f"{it['question']}\n{cfg['prompt_suffix']}"}]}],
            "max_tokens": cfg["eval"]["max_new_tokens"], "temperature": 0, "cache_prompt": False}
        t = time.time()
        r = requests.post(base + "/v1/chat/completions", json=body, timeout=600)
        r.raise_for_status()
        j = r.json()
        raw = j["choices"][0]["message"]["content"] or ""
        p = clean_prediction(raw)
        return {"qid": it["qid"], "pred": p, "raw": raw, "acc": vqa_accuracy(p, it["answers"]),
                "answer_type": it["answer_type"], "latency_s": time.time() - t, "timings": j.get("timings", {})}

    # warm-up (first request pays shader/graph compilation on Metal/CUDA)
    ask(items[0])
    t_run = time.time()
    with ThreadPoolExecutor(par) as ex:
        preds = []
        for i, p in enumerate(ex.map(ask, items)):
            preds.append(p)
            if i % 200 == 0:
                print(f"[{a.name}] {i + 1}/{len(items)} running acc "
                      f"{sum(x['acc'] for x in preds) / len(preds):.4f}", flush=True)
    wall = time.time() - t_run
finally:
    stop.set()
    srv.terminate()
    try:
        srv.wait(10)
    except subprocess.TimeoutExpired:
        srv.kill()

by_type = {}
for p in preds:
    by_type.setdefault(p["answer_type"], []).append(p["acc"])
lat = sorted(p["latency_s"] for p in preds)
tm = [p["timings"] for p in preds if p["timings"]]


def med(key):
    v = [t[key] for t in tm if key in t]
    return statistics.median(v) if v else None


files = {"llm": model, "mmproj": mmproj} | ({"lora": lora} if lora else {})
res = {
    "name": a.name, "backend": "llama.cpp", "n": len(preds),
    "vqa_acc": sum(p["acc"] for p in preds) / len(preds),
    "vqa_acc_by_type": {k: sum(v) / len(v) for k, v in by_type.items()},
    "files": {k: v.name for k, v in files.items()},
    "disk_bytes": sum(v.stat().st_size for v in files.values()),
    "disk_bytes_by_part": {k: v.stat().st_size for k, v in files.items()},
    "device": a.device_label, "parallel": par,
    "latency_s": {"median": statistics.median(lat), "p90": lat[int(0.9 * (len(lat) - 1))]} if par == 1 else None,
    "throughput_questions_per_s": len(preds) / wall,
    "server_timings_median": {k: med(k) for k in ("prompt_n", "prompt_ms", "predicted_n", "predicted_ms",
                                                    "predicted_per_second", "prompt_per_second")},
    "peak_server_rss_bytes": peak_rss[0], "server_load_s": load_s, "server_cmd": cmd, **json.loads(a.meta),
}
out = wd / "results" / "llamacpp"
out.mkdir(parents=True, exist_ok=True)
json.dump(res, open(out / f"{a.name}.json", "w"), indent=2)
with open(out / f"{a.name}.preds.jsonl", "w") as f:
    for p in preds:
        f.write(json.dumps(p) + "\n")
print(json.dumps({k: res[k] for k in ("name", "n", "vqa_acc", "disk_bytes", "throughput_questions_per_s")}, indent=2))
