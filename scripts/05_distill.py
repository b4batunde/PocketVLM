"""Knowledge distillation: fine-tuned teacher (merged Qwen2.5-VL-3B) -> smaller student (Qwen2-VL-2B) + LoRA.

Each model gets its own processor-built inputs for the same (image, question, answer). We align on
the answer tokens (identical because both use the Qwen2 tokenizer; asserted per batch) and train
    loss = alpha * T^2 * KL(softmax(t/T) || softmax(s/T)) + (1 - alpha) * CE(gold)
over answer positions. The question this answers: at equal on-device size, is a distilled 2B at
4-bit better than the 3B teacher squeezed to 2-bit?
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from vlmc import train
from vlmc.config import load_config
from vlmc.data import load_items
from vlmc.hf import build_train_batch, load_model, load_processor

ap = argparse.ArgumentParser()
ap.add_argument("--config")
ap.add_argument("--teacher", required=True, help="merged fine-tuned teacher")
ap.add_argument("--student", help="defaults to cfg.student_id")
ap.add_argument("--out")
ap.add_argument("--limit", type=int)
ap.add_argument("--max-steps", type=int)
a = ap.parse_args()
cfg = load_config(a.config)
D, L = cfg["distill"], cfg["lora"]
wd = Path(cfg["work_dir"])
student_id = a.student or cfg["student_id"]
out = Path(a.out or wd / "adapters/student-kd")
items = load_items(wd / "data" / "train.jsonl", a.limit)

t_proc, s_proc = load_processor(a.teacher, cfg), load_processor(student_id, cfg)
for p in (t_proc, s_proc):
    p.tokenizer.padding_side = "right"
probe = "yes no 2 red <|im_end|>"
assert t_proc.tokenizer(probe)["input_ids"] == s_proc.tokenizer(probe)["input_ids"], "teacher/student tokenizers differ"

teacher = load_model(a.teacher)
for p in teacher.parameters():
    p.requires_grad_(False)
student = load_model(student_id)
student.config.use_cache = False
student = train.attach_lora(student, D["student_lora_r"], D["student_lora_alpha"], L["dropout"], L["target_regex"],
                            grad_ckpt=cfg["train"]["grad_checkpointing"])
T, alpha = D["temperature"], D["alpha_kd"]


def answer_logits(model, batch):
    logits = model(**{k: v for k, v in batch.items() if k != "labels"}).logits[:, :-1]
    tgt = batch["labels"][:, 1:]
    mask = tgt != -100
    return logits[mask].float(), tgt[mask]


def make_loss(mb):
    tb = build_train_batch(t_proc, mb, cfg["prompt_suffix"]).to(teacher.device)
    sb = build_train_batch(s_proc, mb, cfg["prompt_suffix"]).to(student.device)
    ac = dict(device_type=student.device.type, dtype=torch.bfloat16, enabled=student.device.type == "cuda")
    with torch.no_grad(), torch.autocast(**ac):
        t_logits, t_tgt = answer_logits(teacher, tb)
    with torch.autocast(**ac):
        s_logits, s_tgt = answer_logits(student, sb)
    assert torch.equal(t_tgt.cpu(), s_tgt.cpu()), "answer tokens misaligned between teacher and student"
    v = min(t_logits.shape[-1], s_logits.shape[-1])     # both pad vocab; real tokens are shared
    t_logits, s_logits = t_logits[:, :v], s_logits[:, :v]
    kd = F.kl_div(F.log_softmax(s_logits / T, -1), F.log_softmax(t_logits / T, -1), log_target=True,
                  reduction="batchmean") * T * T
    ce = F.cross_entropy(s_logits, s_tgt)
    return alpha * kd + (1 - alpha) * ce


train.run(student, items, make_loss, out, epochs=D["epochs"], lr=D["lr"], batch_size=D["batch_size"],
          grad_accum=D["grad_accum"], log_every=cfg["train"]["log_every"], max_steps=a.max_steps)
s_proc.save_pretrained(out)
print(f"saved student adapter -> {out}")
