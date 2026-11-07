"""LoRA fine-tuning of the base VLM on VQAv2 (LLM decoder only, vision tower frozen).

--qlora trains the same adapter on an NF4-quantized base (quantize-then-adapt).
"""
import argparse
from pathlib import Path

import torch

from vlmc import train
from vlmc.config import load_config
from vlmc.data import load_items
from vlmc.hf import build_train_batch, load_model, load_processor

ap = argparse.ArgumentParser()
ap.add_argument("--config")
ap.add_argument("--qlora", action="store_true")
ap.add_argument("--out")
ap.add_argument("--limit", type=int)
ap.add_argument("--max-steps", type=int)
a = ap.parse_args()
cfg = load_config(a.config)
wd = Path(cfg["work_dir"])
out = Path(a.out or wd / ("adapters/qlora" if a.qlora else "adapters/lora"))
items = load_items(wd / "data" / "train.jsonl", a.limit)

proc = load_processor(cfg["model_id"], cfg)
proc.tokenizer.padding_side = "right"
model = load_model(cfg["model_id"], load_in_4bit=a.qlora)
model.config.use_cache = False
L, T = cfg["lora"], cfg["train"]
model = train.attach_lora(model, L["r"], L["alpha"], L["dropout"], L["target_regex"],
                          grad_ckpt=T["grad_checkpointing"], kbit=a.qlora)


def make_loss(mb):
    batch = build_train_batch(proc, mb, cfg["prompt_suffix"]).to(model.device)
    with torch.autocast(model.device.type, dtype=torch.bfloat16, enabled=model.device.type == "cuda"):
        return model(**batch).loss


train.run(model, items, make_loss, out, epochs=T["epochs"], lr=T["lr"], batch_size=T["batch_size"],
          grad_accum=T["grad_accum"], warmup_ratio=T["warmup_ratio"], log_every=T["log_every"],
          max_steps=a.max_steps)
proc.save_pretrained(out)
print(f"saved adapter -> {out}")
