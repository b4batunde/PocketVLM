"""Hugging Face side: loading every variant, building inputs, batched generation, training batches."""

import os
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_processor(path: str, cfg: dict):
    proc = AutoProcessor.from_pretrained(path)
    ip = proc.image_processor
    lo, hi = cfg["image"]["min_pixels"], cfg["image"]["max_pixels"]
    for k, v in (("min_pixels", lo), ("max_pixels", hi)):
        if hasattr(ip, k):
            setattr(ip, k, v)
    ip.size = {"shortest_edge": lo, "longest_edge": hi}
    proc.tokenizer.padding_side = "left"
    return proc


def load_model(path: str, adapter: str | None = None, load_in_4bit: bool = False, dtype=torch.bfloat16):
    """bf16 checkpoints, llm-compressor (compressed-tensors) checkpoints, or NF4 + LoRA adapter."""
    kw = dict(dtype=dtype)
    if device() == "cuda":
        kw["device_map"] = "cuda"
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype, llm_int8_skip_modules=["visual", "lm_head"])
    model = AutoModelForImageTextToText.from_pretrained(path, **kw)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    if "device_map" not in kw:
        model = model.to(device())
    model.eval()
    return model


def user_messages(question: str, image_path: str, suffix: str):
    return [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text", "text": f"{question}\n{suffix}"},
    ]}]


def _open(p):
    return Image.open(p).convert("RGB")


def build_gen_batch(proc, items, suffix):
    texts = [proc.apply_chat_template(user_messages(it["question"], it["image_path"], suffix),
                                      tokenize=False, add_generation_prompt=True) for it in items]
    images = [_open(it["image_path"]) for it in items]
    return proc(text=texts, images=images, padding=True, return_tensors="pt")


@torch.inference_mode()
def generate(model, proc, items, suffix, max_new_tokens=10):
    batch = build_gen_batch(proc, items, suffix).to(model.device)
    out = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=proc.tokenizer.pad_token_id)
    gen = out[:, batch["input_ids"].shape[1]:]
    return proc.batch_decode(gen, skip_special_tokens=True), int((gen != proc.tokenizer.pad_token_id).sum())


def build_train_batch(proc, items, suffix, answer_key="answer"):
    """Full conversation; labels only on the assistant answer + <|im_end|>."""
    tok = proc.tokenizer
    texts = []
    for it in items:
        msgs = user_messages(it["question"], it["image_path"], suffix)
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": it[answer_key]}]})
        texts.append(proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False).rstrip("\n"))
    images = [_open(it["image_path"]) for it in items]
    batch = proc(text=texts, images=images, padding=True, return_tensors="pt")
    ids = batch["input_ids"]
    labels = torch.full_like(ids, -100)
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    header = tok("assistant\n", add_special_tokens=False)["input_ids"]
    for b in range(ids.shape[0]):
        starts = (ids[b] == im_start).nonzero().flatten()
        s = int(starts[-1]) + 1 + len(header)       # first answer token
        e = int(batch["attention_mask"][b].nonzero().flatten()[-1]) + 1   # through <|im_end|>
        labels[b, s:e] = ids[b, s:e]
    batch["labels"] = labels
    return batch


def dir_size_bytes(path: str | Path, patterns=("*.safetensors", "*.bin", "*.gguf")) -> int:
    p = Path(path)
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for pat in patterns for f in p.rglob(pat))


def lora_target_modules(model, regex: str) -> list[str]:
    """Resolve the LoRA regex to concrete module names (and prove the vision tower is excluded)."""
    import re
    names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and re.fullmatch(regex, n)]
    assert names, "LoRA regex matched nothing"
    assert not any("visual" in n for n in names)
    return names


def env_info() -> dict:
    info = {"torch": torch.__version__, "device": device()}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    import transformers
    info["transformers"] = transformers.__version__
    info["host"] = os.uname().nodename
    return info
