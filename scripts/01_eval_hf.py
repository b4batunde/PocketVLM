"""Evaluate a Hugging Face variant (bf16 / GPTQ / AWQ / NF4+LoRA) on the eval split.

Writes <work_dir>/results/hf/<name>.json (+ predictions jsonl). Accuracy is device-independent;
latency/memory here are GPU numbers and are labeled as such (on-device numbers come from llama.cpp).
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

from vlmc.config import load_config
from vlmc.data import load_items
from vlmc.hf import dir_size_bytes, env_info, generate, load_model, load_processor
from vlmc.vqa_metric import clean_prediction, vqa_accuracy

ap = argparse.ArgumentParser()
ap.add_argument("--config")
ap.add_argument("--name", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--adapter")
ap.add_argument("--load-in-4bit", action="store_true")
ap.add_argument("--processor", help="defaults to --model")
ap.add_argument("--split", default="eval")
ap.add_argument("--limit", type=int)
ap.add_argument("--batch-size", type=int)
ap.add_argument("--meta", default="{}", help="extra JSON merged into the result (method, bits, ...)")
a = ap.parse_args()
cfg = load_config(a.config)
wd = Path(cfg["work_dir"])
items = load_items(wd / "data" / f"{a.split}.jsonl", a.limit)
bs = a.batch_size or cfg["eval"]["hf_batch_size"]

proc = load_processor(a.processor or a.model, cfg)
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
t0 = time.time()
model = load_model(a.model, adapter=a.adapter, load_in_4bit=a.load_in_4bit)
load_s = time.time() - t0
weight_bytes = sum(t.numel() * t.element_size() for t in list(model.parameters()) + list(model.buffers()))

preds, by_type, gen_s, gen_tokens = [], defaultdict(list), 0.0, 0
for i in range(0, len(items), bs):
    chunk = items[i:i + bs]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t = time.time()
    outs, ntok = generate(model, proc, chunk, cfg["prompt_suffix"], cfg["eval"]["max_new_tokens"])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gen_s += time.time() - t
    gen_tokens += ntok
    for it, raw in zip(chunk, outs):
        p = clean_prediction(raw)
        acc = vqa_accuracy(p, it["answers"])
        by_type[it["answer_type"]].append(acc)
        preds.append({"qid": it["qid"], "pred": p, "raw": raw, "acc": acc})
    if (i // bs) % 20 == 0:
        run = sum(x["acc"] for x in preds) / len(preds)
        print(f"[{a.name}] {len(preds)}/{len(items)} running acc {run:.4f}", flush=True)

acc = sum(x["acc"] for x in preds) / len(preds)


def local_dir(p):
    if p and not Path(p).exists():  # Hub id -> its snapshot in the HF cache
        from huggingface_hub import snapshot_download
        return snapshot_download(p)
    return p


disk = dir_size_bytes(local_dir(a.model)) + (dir_size_bytes(a.adapter) if a.adapter else 0)
# NF4 (QLoRA) runs quantize the bf16 checkpoint at load, so its deployable size is what sits in memory.
deploy = weight_bytes if a.load_in_4bit else disk  # weight_bytes already includes the adapter
res = {
    "name": a.name, "backend": "hf", "model": a.model, "adapter": a.adapter, "load_in_4bit": a.load_in_4bit,
    "n": len(preds), "vqa_acc": acc,
    "vqa_acc_by_type": {k: sum(v) / len(v) for k, v in by_type.items()},
    "disk_bytes": deploy, "checkpoint_disk_bytes": disk,
    "weight_bytes_in_memory": weight_bytes,
    "peak_gpu_mem_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
    "gpu_questions_per_s": len(preds) / gen_s, "gpu_gen_tokens_per_s": gen_tokens / gen_s,
    "batch_size": bs, "load_s": load_s, "env": env_info(), **json.loads(a.meta),
}
out = wd / "results" / "hf"
out.mkdir(parents=True, exist_ok=True)
json.dump(res, open(out / f"{a.name}.json", "w"), indent=2)
with open(out / f"{a.name}.preds.jsonl", "w") as f:
    for p in preds:
        f.write(json.dumps(p) + "\n")
print(json.dumps({k: res[k] for k in ("name", "n", "vqa_acc", "disk_bytes")}, indent=2))
