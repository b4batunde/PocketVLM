"""Merge a LoRA adapter into the bf16 base (merge-then-quantize starts here)."""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
model = AutoModelForImageTextToText.from_pretrained(a.base, dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
model.save_pretrained(a.out)
AutoProcessor.from_pretrained(a.base).save_pretrained(a.out)
print(f"merged -> {a.out}")
