#!/usr/bin/env bash
# Every GPU stage, in order, on one RunPod box (48 GB card recommended: L40S / A6000; A100 also fine).
# Idempotent: each step is skipped if its output exists, so you can re-run after a pod restart.
#   bash run_pod.sh            # everything
#   STAGES="data lora" bash run_pod.sh
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=$PWD
export VLMC_CONFIG=${VLMC_CONFIG:-configs/qwen25vl3b.yaml}
export LLAMA_CPP=${LLAMA_CPP:-$PWD/llama.cpp}
export HF_HOME=${HF_HOME:-/workspace/hf}          # keep the HF cache on the persistent volume
export VLMC_DEVICE=${VLMC_DEVICE:-"RunPod $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"}
W=$(python3 -c "from vlmc.config import load_config;print(load_config()['work_dir'])")
BASE=$(python3 -c "from vlmc.config import load_config;print(load_config()['model_id'])")
STUDENT=$(python3 -c "from vlmc.config import load_config;print(load_config()['student_id'])")
M=$W/models; A=$W/adapters; RH=$W/results/hf; RL=$W/results/llamacpp
STAGES=${STAGES:-"data base lora qlora gptq awq vllm distill gguf gguf_eval pack"}
VLLM_PY=${VLLM_PY:-/workspace/vllm-env/bin/python}
has() { [[ " $STAGES " == *" $1 "* ]]; }
step() { local out=$1; shift; if [ -e "$out" ]; then echo "skip ($out exists)"; else "$@"; fi; }
hf_eval() { local name=$1; shift; step $RH/$name.json python3 scripts/01_eval_hf.py --name "$name" "$@"; }
lc_eval() { local name=$1; shift; step $RL/$name.json python3 scripts/07_eval_llamacpp.py --name "$name" "$@"; }
mkdir -p $W/logs; exec > >(tee -a $W/logs/run_pod.log) 2>&1

has data    && step $W/data/eval.jsonl python3 scripts/00_prepare_data.py
has base    && hf_eval base-bf16 --model $BASE
if has lora; then
  step $A/lora/adapter_model.safetensors python3 scripts/02_train_lora.py
  step $M/ft-merged/config.json python3 scripts/03_merge_lora.py --base $BASE --adapter $A/lora --out $M/ft-merged
  hf_eval ft-bf16 --model $M/ft-merged --meta '{"role":"baseline"}'
fi
if has qlora; then   # quantize-then-adapt
  step $A/qlora/adapter_model.safetensors python3 scripts/02_train_lora.py --qlora
  hf_eval qlora-nf4 --model $BASE --adapter $A/qlora --load-in-4bit --meta '{"method":"qlora","bits":4}'
fi
for meth in gptq awq; do has $meth || continue
  for b in 4 3 2; do
    step $M/ft-$meth-w$b/config.json python3 scripts/04_quantize_llmc.py --model $M/ft-merged --method $meth --bits $b --out $M/ft-$meth-w$b
    hf_eval ft-$meth-w$b --model $M/ft-$meth-w$b --meta "{\"method\":\"$meth\",\"bits\":$b}"
  done
done
if has vllm; then   # real 4-bit kernels: weight memory, latency, throughput (bf16 vs GPTQ-W4 vs AWQ-W4)
  if [ ! -x "$VLLM_PY" ]; then
    python3 -m venv /workspace/vllm-env && /workspace/vllm-env/bin/pip install -q -r requirements-vllm.txt
  fi
  for v in ft-merged:vllm-ft-bf16 ft-gptq-w4:vllm-ft-gptq-w4 ft-awq-w4:vllm-ft-awq-w4; do
    step $W/results/vllm/${v#*:}.json $VLLM_PY scripts/11_bench_vllm.py --name ${v#*:} --model $M/${v%:*} \
      || echo "WARN: vLLM bench failed for ${v%:*}; continuing (see log). Not fatal: accuracy comes from the HF harness."
  done
fi
if has distill; then
  hf_eval student-base-bf16 --model $STUDENT
  step $A/student-kd/adapter_model.safetensors python3 scripts/05_distill.py --teacher $M/ft-merged
  step $M/student-kd-merged/config.json python3 scripts/03_merge_lora.py --base $STUDENT --adapter $A/student-kd --out $M/student-kd-merged
  hf_eval student-kd-bf16 --model $M/student-kd-merged
fi
if has gguf; then
  python3 scripts/06_export_gguf.py --hf $M/ft-merged --name ft
  python3 scripts/06_export_gguf.py --hf $BASE --name base --types Q4_K_M --mmproj-types F16
  python3 scripts/06_export_gguf.py --lora $A/lora --base $BASE --name lora
  [ -d $A/qlora ] && python3 scripts/06_export_gguf.py --lora $A/qlora --base $BASE --name qlora
  python3 scripts/06_export_gguf.py --hf $M/student-kd-merged --name student-kd --types Q8_0 Q4_K_M Q3_K_M --mmproj-types F16 Q8_0
fi
if has gguf_eval; then   # accuracy sweep (GPU, parallel slots). On-device speed is measured on the Mac.
  lc_eval ft-F16 --model ft-F16.gguf --mmproj mmproj-ft-F16.gguf
  for t in Q8_0 Q6_K Q4_K_M Q3_K_M Q2_K IQ2_XS; do
    lc_eval ft-$t --model ft-$t.gguf --mmproj mmproj-ft-F16.gguf
    case $t in Q4_K_M|Q2_K) lc_eval ft-$t-mmprojQ8 --model ft-$t.gguf --mmproj mmproj-ft-Q8_0.gguf ;; esac   # vision-encoder 8-bit ablation on two levels
  done
  lc_eval base-Q4_K_M --model base-Q4_K_M.gguf --mmproj mmproj-base-F16.gguf
  lc_eval base-Q4_K_M+lora --model base-Q4_K_M.gguf --mmproj mmproj-base-F16.gguf --lora lora-lora-F16.gguf
  [ -f $W/gguf/lora-qlora-F16.gguf ] && lc_eval base-Q4_K_M+qlora --model base-Q4_K_M.gguf --mmproj mmproj-base-F16.gguf --lora lora-qlora-F16.gguf
  for t in Q8_0 Q4_K_M Q3_K_M; do
    lc_eval student-kd-$t-mmprojQ8 --model student-kd-$t.gguf --mmproj mmproj-student-kd-Q8_0.gguf
  done
fi
python3 scripts/09_report.py || true
if has pack; then   # everything the Mac needs: GGUFs, eval split, results so far
  python3 -c "import json;[print('data/'+json.loads(l)['image']) for l in open('$W/data/eval.jsonl')]" | sort -u > $W/eval_images.txt
  (cd $W && ls gguf/*.gguf data/eval.jsonl data/splits.json && echo results && cat eval_images.txt) > $W/pack_list.txt
  tar -C $W -cf $W/for_mac.tar -T $W/pack_list.txt
  echo "Download $W/for_mac.tar to the Mac (runpodctl send / scp), then run run_mac.sh"
fi
