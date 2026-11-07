"""Minimal, transparent training loop shared by LoRA, QLoRA and distillation."""

import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model

from .hf import lora_target_modules


def attach_lora(model, r, alpha, dropout, regex, grad_ckpt=True, kbit=False):
    if kbit:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=grad_ckpt)
    elif grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    targets = lora_target_modules(model, regex)
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=targets, task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


def run(model, items, make_loss, out_dir, *, epochs, lr, batch_size, grad_accum, warmup_ratio=0.03,
        log_every=10, seed=0, max_steps=None):
    """make_loss(batch_items) -> scalar loss tensor (with grad). Saves adapter + train_log.jsonl."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(items) / (batch_size * grad_accum))
    total = steps_per_epoch * epochs if max_steps is None else max_steps
    warm = max(1, int(total * warmup_ratio))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total)))))
    rng = random.Random(seed)
    log = open(out_dir / "train_log.jsonl", "w")
    model.train()
    step, t0 = 0, time.time()
    for ep in range(epochs):
        order = items[:]
        rng.shuffle(order)
        micro = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
        for mi, mb in enumerate(micro):
            loss = make_loss(mb) / grad_accum
            loss.backward()
            if (mi + 1) % grad_accum == 0 or mi == len(micro) - 1:
                gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % log_every == 0 or step == 1:
                    rec = {"step": step, "epoch": ep, "loss": loss.item() * grad_accum, "grad_norm": float(gn),
                           "lr": sched.get_last_lr()[0], "elapsed_s": time.time() - t0}
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                    print(rec, flush=True)
                if max_steps and step >= max_steps:
                    break
        if max_steps and step >= max_steps:
            break
    model.save_pretrained(out_dir)
    json.dump({"steps": step, "train_seconds": time.time() - t0, "n_items": len(items)},
              open(out_dir / "train_summary.json", "w"), indent=2)
    return model
