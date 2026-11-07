#!/usr/bin/env bash
# Step 1 on the pod: exercise every stage on the REAL model and data with tiny limits (~20-30 min).
# Writes to runs/dryrun (separate from the real run). Never stops early: prints a PASS/FAIL table at the end,
# plus measured per-step timings so you can budget the full run before paying for it.
#   bash scripts/dry_run_pod.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD
BASE_CFG=${VLMC_CONFIG:-configs/qwen25vl3b.yaml}
export LLAMA_CPP=${LLAMA_CPP:-$PWD/llama.cpp}
export HF_HOME=${HF_HOME:-/workspace/hf}
VLLM_PY=${VLLM_PY:-/workspace/vllm-env/bin/python}
DRY=runs/dryrun; mkdir -p $DRY/logs

# ---- environment checks ----
echo "== environment"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "WARN: no nvidia-smi (no GPU?)"
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
[ -n "$VRAM" ] && [ "$VRAM" -lt 40000 ] && echo "WARN: ${VRAM} MiB VRAM; 48 GB recommended (LoRA/distill may OOM)"
FREE=$(df -BG --output=avail "${HF_HOME%/*}" 2>/dev/null | tail -1 | tr -dc 0-9)
[ -n "$FREE" ] && [ "$FREE" -lt 120 ] && echo "WARN: only ${FREE} GB free on the volume; full run needs ~150 GB"
[ -x "$LLAMA_CPP/build/bin/llama-server" ] || echo "WARN: llama.cpp not built; run scripts/setup_llamacpp.sh"

# ---- dry-run config: same as the real one, tiny sizes, separate work_dir ----
python3 - "$BASE_CFG" "$DRY/config.yaml" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
c["work_dir"] = "runs/dryrun"
c["data"].update(train_questions=160, calib_questions=32, eval_questions=64)
c["quant"]["calib_samples"] = 16
yaml.safe_dump(c, open(sys.argv[2], "w"), sort_keys=False)
PY
export VLMC_CONFIG=$DRY/config.yaml
BASE=$(python3 -c "from vlmc.config import load_config;print(load_config()['model_id'])")
STUDENT=$(python3 -c "from vlmc.config import load_config;print(load_config()['student_id'])")
M=$DRY/models; A=$DRY/adapters

declare -a NAMES STATUS SECS
stage() {  # stage <name> <cmd...>
  local name=$1; shift; local t0=$(date +%s)
  echo; echo "== [$name] $*"
  if "$@" > "$DRY/logs/$name.log" 2>&1; then STATUS+=("PASS"); else STATUS+=("FAIL"); tail -15 "$DRY/logs/$name.log"; fi
  NAMES+=("$name"); SECS+=($(( $(date +%s) - t0 )))
  echo "   -> ${STATUS[-1]} in ${SECS[-1]}s (log: $DRY/logs/$name.log)"
}

stage data          python3 scripts/00_prepare_data.py ${DRY_SYNTHETIC:+--synthetic --synthetic-n-train 160}
stage eval-base     python3 scripts/01_eval_hf.py --name base-bf16 --model $BASE --limit 32
stage lora          python3 scripts/02_train_lora.py --max-steps 5
stage merge         python3 scripts/03_merge_lora.py --base $BASE --adapter $A/lora --out $M/ft-merged
stage eval-ft       python3 scripts/01_eval_hf.py --name ft-bf16 --model $M/ft-merged --limit 32
stage qlora         python3 scripts/02_train_lora.py --qlora --max-steps 3 --limit 96
stage eval-qlora    python3 scripts/01_eval_hf.py --name qlora-nf4 --model $BASE --adapter $A/qlora --load-in-4bit --limit 16
stage gptq-w4       python3 scripts/04_quantize_llmc.py --model $M/ft-merged --method gptq --bits 4 --out $M/ft-gptq-w4
stage eval-gptq-w4  python3 scripts/01_eval_hf.py --name ft-gptq-w4 --model $M/ft-gptq-w4 --limit 16
stage awq-w2        python3 scripts/04_quantize_llmc.py --model $M/ft-merged --method awq --bits 2 --out $M/ft-awq-w2
stage eval-awq-w2   python3 scripts/01_eval_hf.py --name ft-awq-w2 --model $M/ft-awq-w2 --limit 16
if [ ! -x "$VLLM_PY" ]; then
  stage vllm-install bash -c "python3 -m venv /workspace/vllm-env && /workspace/vllm-env/bin/pip install -q -r requirements-vllm.txt"
fi
stage vllm-gptq-w4  $VLLM_PY scripts/11_bench_vllm.py --name vllm-ft-gptq-w4 --model $M/ft-gptq-w4 --limit 20 --latency-n 5
stage distill       python3 scripts/05_distill.py --teacher $M/ft-merged --max-steps 3 --limit 96
stage merge-student python3 scripts/03_merge_lora.py --base $STUDENT --adapter $A/student-kd --out $M/student-kd-merged
stage eval-student  python3 scripts/01_eval_hf.py --name student-kd-bf16 --model $M/student-kd-merged --limit 16
stage gguf-ft       python3 scripts/06_export_gguf.py --hf $M/ft-merged --name ft --types Q4_K_M IQ2_XS --mmproj-types F16 --imatrix-ctx 128
stage gguf-lora     python3 scripts/06_export_gguf.py --lora $A/lora --base $BASE --name lora
stage gguf-student  python3 scripts/06_export_gguf.py --hf $M/student-kd-merged --name student-kd --types Q4_K_M --mmproj-types F16 --no-imatrix
stage lc-ft-F16     python3 scripts/07_eval_llamacpp.py --name ft-F16 --model ft-F16.gguf --mmproj mmproj-ft-F16.gguf --limit 32
stage lc-ft-Q4      python3 scripts/07_eval_llamacpp.py --name ft-Q4_K_M --model ft-Q4_K_M.gguf --mmproj mmproj-ft-F16.gguf --limit 16
stage lc-ft-IQ2     python3 scripts/07_eval_llamacpp.py --name ft-IQ2_XS --model ft-IQ2_XS.gguf --mmproj mmproj-ft-F16.gguf --limit 16
stage lc-student    python3 scripts/07_eval_llamacpp.py --name student-kd-Q4_K_M --model student-kd-Q4_K_M.gguf --mmproj mmproj-student-kd-F16.gguf --limit 16
stage report        python3 scripts/09_report.py

# ---- summary ----
echo; echo "================ DRY RUN SUMMARY ================"
fails=0
for i in "${!NAMES[@]}"; do
  printf "  %-14s %-4s %6ss\n" "${NAMES[$i]}" "${STATUS[$i]}" "${SECS[$i]}"
  [ "${STATUS[$i]}" = FAIL ] && fails=$((fails+1))
done
echo
BASE_CFG=$BASE_CFG python3 - <<'PY'
# Consistency checks + measured timings to budget the full run (projections are labeled as such).
import json, os, yaml
from pathlib import Path
R = Path("runs/dryrun/results"); real = yaml.safe_load(open(os.environ["BASE_CFG"]))
def j(p):
    try: return json.load(open(p))
    except Exception: return None
ft, f16 = j(R/"hf/ft-bf16.json"), j(R/"llamacpp/ft-F16.json")
if ft and f16:
    d = 100 * (f16["vqa_acc"] - ft["vqa_acc"])
    print(f"HF bf16 vs GGUF F16 on the same 32 questions: {ft['vqa_acc']:.3f} vs {f16['vqa_acc']:.3f} ({d:+.1f} pts)"
          + ("   OK" if abs(d) <= 7 else "   CHECK: backends disagree; compare preds.jsonl files"))
log = [json.loads(l) for l in open("runs/dryrun/adapters/lora/train_log.jsonl")] if Path("runs/dryrun/adapters/lora/train_log.jsonl").exists() else []
summ = j("runs/dryrun/adapters/lora/train_summary.json")
if summ and summ["steps"]:
    per = summ["train_seconds"] / summ["steps"]
    T = real["train"]; steps = -(-real["data"]["train_questions"] // (T["batch_size"] * T["grad_accum"])) * T["epochs"]
    print(f"LoRA: {per:.1f} s/step measured (includes warm-up) -> projected full run ~{per * steps / 3600:.1f} h for {steps} steps")
if ft:
    q = ft["gpu_questions_per_s"]
    print(f"HF eval: {q:.1f} questions/s measured -> projected ~{real['data']['eval_questions'] / q / 60:.0f} min per HF variant")
sample = [json.loads(l) for l in open(R/"hf/ft-bf16.preds.jsonl")][:5] if (R/"hf/ft-bf16.preds.jsonl").exists() else []
for s in sample:
    print(f"  sample pred: {s['pred']!r:20} acc={s['acc']:.2f}")
PY
echo
if [ $fails -eq 0 ]; then echo "ALL STAGES PASSED. Next: bash run_pod.sh"; else echo "$fails stage(s) failed. Send me the failing log(s)."; fi
