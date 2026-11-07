"""GPTQ / AWQ post-training quantization of the fine-tuned (merged) model with llm-compressor.

Only the LLM decoder's Linear layers are quantized; the vision tower and lm_head stay bf16
(see README "fairness" note: GGUF's mmproj is also kept at F16/Q8_0 separately).
Output: compressed-tensors checkpoint that transformers loads directly.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from datasets import Dataset
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from transformers import AutoModelForImageTextToText

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier
from vlmc.config import load_config
from vlmc.data import load_items
from vlmc.hf import build_train_batch, load_processor

ap = argparse.ArgumentParser()
ap.add_argument("--config")
ap.add_argument("--model", required=True, help="merged fine-tuned checkpoint")
ap.add_argument("--method", choices=["gptq", "awq"], required=True)
ap.add_argument("--bits", type=int, choices=[2, 3, 4, 8], required=True)
ap.add_argument("--group-size", type=int)
ap.add_argument("--out", required=True)
a = ap.parse_args()
if not torch.cuda.is_available():  # CPU smoke tests only: llm-compressor's AWQ cache pins memory (CUDA-only)
    torch.Tensor.pin_memory = lambda self, *args, **kw: self
cfg = load_config(a.config)
Q = cfg["quant"]
wd = Path(cfg["work_dir"])
gs = a.group_size or Q["group_size"][a.bits]
sym = Q["gptq_symmetric"] if a.method == "gptq" else Q["awq_symmetric"]

model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16)
proc = load_processor(a.model, cfg)
decoder_cls = type(model.model.language_model.layers[0]).__name__

# Calibration = real (image, question, answer) conversations from the train images.
calib = load_items(wd / "data" / "calib.jsonl", Q["calib_samples"])
samples = []
for it in calib:
    b = build_train_batch(proc, [it], cfg["prompt_suffix"])
    b.pop("labels")
    samples.append({k: v for k, v in b.items()})


DTYPES = {k: v.dtype for k, v in samples[0].items()}
calib_ds = Dataset.from_list([{k: v.tolist() for k, v in s.items()} for s in samples])


def collate(batch):
    assert len(batch) == 1
    return {k: torch.tensor(v, dtype=DTYPES[k]) for k, v in batch[0].items()}


scheme = QuantizationScheme(targets=["Linear"], weights=QuantizationArgs(
    num_bits=a.bits, type="int", symmetric=sym, strategy="group", group_size=gs))
if a.method == "gptq":
    recipe = [GPTQModifier(config_groups={"group_0": scheme}, ignore=Q["ignore"])]
else:
    recipe = [AWQModifier(), QuantizationModifier(config_groups={"group_0": scheme}, ignore=Q["ignore"])]

t0 = time.time()
oneshot(model=model, dataset=calib_ds, recipe=recipe, num_calibration_samples=len(samples),
        max_seq_length=Q["max_seq_len"], data_collator=collate, sequential_targets=[decoder_cls])
secs = time.time() - t0

model.save_pretrained(a.out, save_compressed=True)
proc.save_pretrained(a.out)
# llm-compressor expands `ignore` into concrete module names using the checkpoint's legacy layout
# ("visual.blocks.0...") but transformers>=5 loads the module as "model.visual.blocks.0...", so the
# vision tower would be wrongly treated as quantized at load time. Store the original patterns instead.
cfg_path = Path(a.out) / "config.json"
mc = json.load(open(cfg_path))
mc["quantization_config"]["ignore"] = Q["ignore"]
json.dump(mc, open(cfg_path, "w"), indent=2)
meta = {"method": a.method, "bits": a.bits, "group_size": gs, "symmetric": sym, "calib_samples": len(samples),
        "quantize_seconds": secs, "decoder_layer": decoder_cls}
json.dump(meta, open(Path(a.out) / "vlmc_quant.json", "w"), indent=2)
print(json.dumps(meta, indent=2))
