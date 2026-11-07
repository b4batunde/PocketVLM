"""Aggregate every measured result into results/REPORT.md, summary.csv, latency.csv and figures.

Only reads result JSONs written by the eval/bench scripts. Anything not yet measured is shown as
"pending", never estimated. The headline sentence is filled in from data or not at all.
"""
import argparse
import csv
import json
from pathlib import Path

from vlmc.config import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--config")
ap.add_argument("--baseline", default="ft-bf16", help="HF result name of the full-precision fine-tuned model")
ap.add_argument("--base-model", default="base-bf16", help="HF result name of the un-fine-tuned model")
a = ap.parse_args()
cfg = load_config(a.config)
R = Path(cfg["work_dir"]) / "results"


def load(sub):
    d = R / sub
    return {p.stem: json.load(open(p)) for p in sorted(d.glob("*.json")) if not p.name.endswith(".raw.json")} \
        if d.exists() else {}


hf, lc, bench, vl = load("hf"), load("llamacpp"), load("bench"), load("vllm")
if a.baseline not in hf:
    raise SystemExit(f"baseline result '{a.baseline}' not found in {R/'hf'}; run 01_eval_hf.py first")
B = hf[a.baseline]
b_acc, b_size, b_n = B["vqa_acc"], B["disk_bytes"], B["n"]
thr = cfg["retain_threshold"]
GB = 1e9


def family(name, r):
    if r["backend"] == "hf":
        return r.get("method", "").upper() or ("QLoRA (NF4)" if r.get("load_in_4bit") else "bf16")
    if "lora" in r.get("files", {}):
        return "GGUF + runtime LoRA"
    if name.startswith("mac-"):
        return "GGUF (Mac re-run)"
    if name.startswith("student"):
        return "GGUF (distilled student)"
    return "GGUF"


rows = []
for name, r in list(hf.items()) + list(lc.items()):
    if r.get("parallel") == 1 and r["backend"] == "llama.cpp" and r["n"] != b_n:
        continue  # latency-only runs live in their own table
    llm_file = r.get("files", {}).get("llm", "")
    bn = bench.get(Path(llm_file).stem, {}) if llm_file else {}
    rows.append({
        "variant": name, "backend": r["backend"], "family": family(name, r), "n": r["n"],
        "size_gb": r["disk_bytes"] / GB, "compression_x": b_size / r["disk_bytes"],
        "vqa_acc": r["vqa_acc"], "retained_pct": 100 * r["vqa_acc"] / b_acc,
        "acc_number": r["vqa_acc_by_type"].get("number"), "acc_yesno": r["vqa_acc_by_type"].get("yes/no"),
        "acc_other": r["vqa_acc_by_type"].get("other"),
        "tg_tok_s": next((v for k, v in bn.items() if k.startswith("tg") and k.endswith("tokens_per_s")), None),
        "pp_tok_s": next((v for k, v in bn.items() if k.startswith("pp") and k.endswith("tokens_per_s")), None),
        "bench_device": bn.get("device"), "comparable_n": r["n"] == b_n,
    })
rows.sort(key=lambda x: (x["backend"], -x["size_gb"]))

with open(R / "summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# Recommendation: smallest on-device (GGUF) variant over the full eval set that keeps >= thr of baseline.
cands = [r for r in rows if r["backend"] == "llama.cpp" and r["comparable_n"] and r["retained_pct"] >= 100 * thr
         and not r["variant"].startswith("mac-")]   # mac-* rows are cross-device re-runs, not candidates
rec = min(cands, key=lambda r: r["size_gb"]) if cands else None

latency = [r for r in lc.values() if r.get("parallel") == 1 and r.get("latency_s")]


def fmt(v, spec, pending="pending"):
    return pending if v is None else format(v, spec)


L = [f"# VLM compression results: {cfg['model_id']}", "",
     f"Baseline (full-precision fine-tuned, `{a.baseline}`): VQA acc **{b_acc:.4f}** on {b_n} held-out questions, "
     f"{b_size / GB:.2f} GB on disk.", ""]
if a.base_model in hf:
    bm = hf[a.base_model]
    L.append(f"Un-fine-tuned base (`{a.base_model}`): VQA acc {bm['vqa_acc']:.4f} "
             f"(LoRA delta {100 * (b_acc - bm['vqa_acc']):+.2f} pts).")
    L.append("")
L += ["## Headline", ""]
if rec:
    tok = fmt(rec["tg_tok_s"], ".1f", None)
    speed = f", running at {tok} tokens/s on {rec['bench_device']}" if tok else \
        " (tokens/s pending: run scripts/08_bench_llamacpp.sh on the laptop)"
    L.append(f"> Shrank the model **{rec['compression_x']:.1f}x** ({b_size / GB:.2f} GB -> {rec['size_gb']:.2f} GB, "
             f"LLM + vision projector) retaining **{rec['retained_pct']:.1f}%** of VQA accuracy{speed}. "
             f"Variant: `{rec['variant']}`.")
else:
    L.append(f"> No GGUF variant measured yet keeps >= {100 * thr:.0f}% of baseline accuracy on the full eval set.")
L += ["", "## All variants", "",
      "| Variant | Backend | Family | Size (GB) | Compression | VQA acc | % of baseline | yes/no | number | other | decode tok/s | prefill tok/s |",
      "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
for r in rows:
    flag = "" if r["comparable_n"] else f" (n={r['n']})"
    L.append(f"| {r['variant']}{flag} | {r['backend']} | {r['family']} | {r['size_gb']:.3f} | {r['compression_x']:.1f}x | "
             f"{r['vqa_acc']:.4f} | {r['retained_pct']:.1f}% | {fmt(r['acc_yesno'], '.3f', '-')} | "
             f"{fmt(r['acc_number'], '.3f', '-')} | {fmt(r['acc_other'], '.3f', '-')} | "
             f"{fmt(r['tg_tok_s'], '.1f', '-')} | {fmt(r['pp_tok_s'], '.1f', '-')} |")
if latency:
    L += ["", "## End-to-end latency (single stream, image + question -> answer)", "",
          "| Variant | Device | n | Size (GB) | median s | p90 s | median prefill tokens |", "|---|---|---:|---:|---:|---:|---:|"]
    lat_rows = []
    for r in sorted(latency, key=lambda r: r["latency_s"]["median"]):
        lat_rows.append({"variant": r["name"], "device": r["device"], "n": r["n"], "size_gb": r["disk_bytes"] / GB,
                         "median_s": r["latency_s"]["median"], "p90_s": r["latency_s"]["p90"],
                         "prefill_tokens": r["server_timings_median"].get("prompt_n")})
        L.append(f"| {r['name']} | {r['device']} | {r['n']} | {r['disk_bytes'] / GB:.2f} | {r['latency_s']['median']:.3f} | "
                 f"{r['latency_s']['p90']:.3f} | {fmt(r['server_timings_median'].get('prompt_n'), '.0f', '-')} |")
    with open(R / "latency.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lat_rows[0].keys()))
        w.writeheader()
        w.writerows(lat_rows)
    L += ["", "Peak RSS is not reported: on Apple Silicon, Metal buffers holding the weights are not counted in "
          "process RSS, so it understates memory. File size is the memory figure to use."]
if vl:
    L += ["", "## GPU serving with real 4-bit kernels (vLLM)", "",
          "| Variant | GPU | n | VQA acc | weight memory (GiB) | single-stream median s | p90 s | decode tok/s | throughput (q/s) |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in vl.values():
        L.append(f"| {r['name']} | {r['gpu']} | {r['n']} | {r['vqa_acc']:.4f} | {fmt(r['weight_memory_gib'], '.2f', '-')} | "
                 f"{r['latency_s']['median']:.3f} | {r['latency_s']['p90']:.3f} | "
                 f"{r['decode_tokens_per_s_single_stream']:.1f} | {r['throughput_questions_per_s']:.1f} |")
    L.append("")
    L.append("2- and 3-bit GPTQ/AWQ have no vLLM (or Metal) kernels, so those rows are accuracy-only.")

# ---------------- Key comparisons: the numbers the write-up interprets ----------------
acc = {n: r["vqa_acc"] for n, r in list(hf.items()) + list(lc.items())}
size = {n: r["disk_bytes"] / GB for n, r in list(hf.items()) + list(lc.items())}
bytype = {n: r["vqa_acc_by_type"] for n, r in list(hf.items()) + list(lc.items())}


def pct(n):
    return f"{acc[n]:.4f} ({100 * acc[n] / b_acc:.1f}%)" if n in acc else "pending"


def delta(x, y):
    return f"{100 * (acc[x] - acc[y]):+.2f} pts" if x in acc and y in acc else "pending"


C = ["", "## Key comparisons", "",
     "### 1. Method vs bit-width (accuracy, % of baseline)", "",
     "| Bits | GPTQ | AWQ | GGUF (closest type) |", "|---|---|---|---|"]
for b, g in [(4, "ft-Q4_K_M"), (3, "ft-Q3_K_M"), (2, "ft-Q2_K"), (2, "ft-IQ2_XS")]:
    C.append(f"| {b} | {pct(f'ft-gptq-w{b}')} | {pct(f'ft-awq-w{b}')} | {g}: {pct(g)} |")
C += ["", "GGUF K-quants mix precisions per tensor, so their effective bits/weight sit above the nominal number; "
      "compare by the Size column in the main table, not by the name.", "",
      "### 2. Merge-then-quantize vs quantize-then-adapt (4-bit)", "",
      "| Recipe | Variant | Accuracy |", "|---|---|---|",
      f"| merge LoRA in bf16, then quantize | ft-Q4_K_M | {pct('ft-Q4_K_M')} |",
      f"| quantize base, apply bf16-trained LoRA at runtime | base-Q4_K_M+lora | {pct('base-Q4_K_M+lora')} |",
      f"| quantize base, apply QLoRA-trained LoRA at runtime | base-Q4_K_M+qlora | {pct('base-Q4_K_M+qlora')} |",
      f"| QLoRA as trained (NF4 base + adapter, HF) | qlora-nf4 | {pct('qlora-nf4')} |",
      f"| quantized base, no adapter (reference) | base-Q4_K_M | {pct('base-Q4_K_M')} |", "",
      "### 3. Does the vision encoder tolerate 8-bit?", "",
      "| LLM quant | mmproj F16 | mmproj Q8_0 | delta | size saved (GB) |", "|---|---|---|---|---|"]
for t in ["Q8_0", "Q6_K", "Q4_K_M", "Q3_K_M", "Q2_K", "IQ2_XS"]:
    x, y = f"ft-{t}-mmprojQ8", f"ft-{t}"
    if x not in acc:   # the sweep runs this ablation on a subset of levels only
        C.append(f"| {t} | {pct(y)} | not run | - | - |")
        continue
    saved = f"{size[y] - size[x]:.2f}" if x in size and y in size else "pending"
    C.append(f"| {t} | {pct(y)} | {pct(x)} | {delta(x, y)} | {saved} |")
C += ["", "### 4. Distill smaller or crush bigger? (similar bytes)", ""]
dist = [n for n in ["student-kd-Q4_K_M-mmprojQ8", "student-kd-Q3_K_M-mmprojQ8", "ft-Q2_K-mmprojQ8", "ft-IQ2_XS-mmprojQ8",
                    "ft-Q3_K_M-mmprojQ8"] if n in acc]
if dist:
    C += ["| Variant | Size (GB) | Accuracy |", "|---|---:|---|"]
    for n in sorted(dist, key=lambda n: size[n]):
        C.append(f"| {n} | {size[n]:.2f} | {pct(n)} |")
else:
    C.append("pending")
C += ["", f"KD gain on the student (bf16): {delta('student-kd-bf16', 'student-base-bf16')}; "
      f"student vs teacher (bf16): {delta('student-kd-bf16', a.baseline)}.", "",
      "### 5. Where the accuracy cliff is (GGUF ladder, mmproj F16)", ""]
ladder = [n for n in ["ft-F16", "ft-Q8_0", "ft-Q6_K", "ft-Q4_K_M", "ft-Q3_K_M", "ft-Q2_K", "ft-IQ2_XS"] if n in acc]
if len(ladder) > 1:
    steps = [(ladder[i], ladder[i + 1], acc[ladder[i]] - acc[ladder[i + 1]]) for i in range(len(ladder) - 1)]
    C.append(" -> ".join(f"{n.split('-', 1)[1]} {acc[n]:.4f}" for n in ladder))
    big = max(steps, key=lambda s: s[2])
    C.append(f"\nLargest single drop: {big[0]} -> {big[1]} ({100 * big[2]:.2f} pts).")
else:
    C.append("pending")
C += ["", "### 6. Which questions break first at 2-bit (per answer type, change vs baseline)", "",
      "| Variant | yes/no | number | other |", "|---|---|---|---|"]
for n in ["ft-gptq-w2", "ft-awq-w2", "ft-Q2_K", "ft-IQ2_XS"]:
    if n in bytype:
        cells = [f"{100 * (bytype[n].get(k, 0) - B['vqa_acc_by_type'].get(k, 0)):+.1f} pts" if k in bytype[n] else "-"
                 for k in ("yes/no", "number", "other")]
        C.append(f"| {n} | " + " | ".join(cells) + " |")
macs = [n for n in lc if n.startswith("mac-") and lc[n]["n"] == b_n]
if macs:
    C += ["", "### 7. Cross-device check (same GGUF, pod GPU vs laptop)", ""]
    for n in macs:
        pod = n[len("mac-"):-len("-full")] if n.endswith("-full") else None
        C.append(f"- {n}: {acc[n]:.4f} on {lc[n]['device']}"
                 + (f" vs {acc[pod]:.4f} on the pod ({delta(n, pod)})" if pod in acc else ""))
L += C
L += ["", "## Notes", "",
      "- vLLM rows use real packed 4-bit kernels on the pod GPU; they are GPU serving numbers, not on-device.",
      "- Size = bytes on disk of everything needed to run (HF: all safetensors; GGUF: LLM + mmproj [+ LoRA]).",
      "- HF GPTQ/AWQ accuracy is measured with weights dequantized at load (compressed-tensors); their size is the packed checkpoint.",
      "- decode/prefill tok/s come from llama-bench (text-only, pp512/tg128) on the device named in results/bench. "
      "A fanless laptop throttles as it heats up, so treat small tok/s differences between variants as noise.",
      "- Eval split: image-disjoint partition of VQAv2 val (see data/splits.json); not comparable to the official test-dev leaderboard."]
(R / "REPORT.md").write_text("\n".join(L) + "\n")

from vlmc.figures import make_figures  # noqa: E402

make_figures(R, b_acc, b_size / GB, thr, rec["variant"] if rec else None)
print((R / "REPORT.md").read_text())
