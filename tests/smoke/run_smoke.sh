#!/usr/bin/env bash
# End-to-end CPU smoke test of every stage on tiny random-weight models + synthetic shapes data.
# Proves the code paths run; the numbers it prints mean nothing about Qwen2.5-VL.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD VLMC_CONFIG=configs/smoke.yaml
export LLAMA_CPP=${LLAMA_CPP:-../llama.cpp}
export CONVERT_PREFIX="python3 tests/smoke/convert_shim.py"
W=smoke_artifacts/run; M=$W/models; T=smoke_artifacts/models
rm -rf $W
python3 scripts/00_prepare_data.py --synthetic --synthetic-n-train 1200
python3 tests/smoke/make_tiny_model.py --out $T --pretrain-data $W/data/train.jsonl
# baseline + LoRA + merge
python3 scripts/01_eval_hf.py --name base-bf16 --model $T/tiny-qwen25vl
python3 scripts/02_train_lora.py --max-steps 300
python3 scripts/03_merge_lora.py --base $T/tiny-qwen25vl --adapter $W/adapters/lora --out $M/ft-merged
python3 scripts/01_eval_hf.py --name ft-bf16 --model $M/ft-merged --meta '{"role":"baseline"}'
# quantize-then-adapt (QLoRA) -- needs bitsandbytes; skipped on CPU if unavailable
if python3 -c "import bitsandbytes" 2>/dev/null; then
  python3 scripts/02_train_lora.py --qlora --max-steps 100 && \
  python3 scripts/01_eval_hf.py --name qlora-nf4 --model $T/tiny-qwen25vl --adapter $W/adapters/qlora --load-in-4bit || echo "QLoRA skipped on this host"
fi
# GPTQ / AWQ
for v in gptq:4 gptq:3 gptq:2 awq:4 awq:3 awq:2; do m=${v%:*}; b=${v#*:}
  python3 scripts/04_quantize_llmc.py --model $M/ft-merged --method $m --bits $b --out $M/ft-$m-w$b
  python3 scripts/01_eval_hf.py --name ft-$m-w$b --model $M/ft-$m-w$b --meta "{\"method\":\"$m\",\"bits\":$b}"
done
# distillation
python3 scripts/01_eval_hf.py --name student-base-bf16 --model $T/tiny-qwen2vl-student
python3 scripts/05_distill.py --teacher $M/ft-merged --max-steps 300
python3 scripts/03_merge_lora.py --base $T/tiny-qwen2vl-student --adapter $W/adapters/student-kd --out $M/student-kd-merged
python3 scripts/01_eval_hf.py --name student-kd-bf16 --model $M/student-kd-merged
# GGUF
python3 scripts/06_export_gguf.py --hf $M/ft-merged --name ft --imatrix-ctx 64
python3 scripts/06_export_gguf.py --hf $T/tiny-qwen25vl --name base --types Q4_K_M --mmproj-types F16 --imatrix-ctx 64
python3 scripts/06_export_gguf.py --lora $W/adapters/lora --base $T/tiny-qwen25vl --name lora
[ -d $W/adapters/qlora ] && python3 scripts/06_export_gguf.py --lora $W/adapters/qlora --base $T/tiny-qwen25vl --name qlora
python3 scripts/06_export_gguf.py --hf $M/student-kd-merged --name student-kd --types Q8_0 Q4_K_M Q3_K_M --mmproj-types F16 Q8_0 --imatrix-ctx 64
python3 scripts/07_eval_llamacpp.py --name ft-F16 --model ft-F16.gguf --mmproj mmproj-ft-F16.gguf --parallel 2
for t in Q8_0 Q6_K Q4_K_M Q3_K_M Q2_K IQ2_XS; do
  python3 scripts/07_eval_llamacpp.py --name ft-$t --model ft-$t.gguf --mmproj mmproj-ft-F16.gguf --parallel 2
  python3 scripts/07_eval_llamacpp.py --name ft-$t-mmprojQ8 --model ft-$t.gguf --mmproj mmproj-ft-Q8_0.gguf --parallel 2
done
python3 scripts/07_eval_llamacpp.py --name base-Q4_K_M --model base-Q4_K_M.gguf --mmproj mmproj-base-F16.gguf --parallel 2
python3 scripts/07_eval_llamacpp.py --name base-Q4_K_M+lora --model base-Q4_K_M.gguf --mmproj mmproj-base-F16.gguf --lora lora-lora-F16.gguf --parallel 2
[ -f $W/gguf/lora-qlora-F16.gguf ] && python3 scripts/07_eval_llamacpp.py --name base-Q4_K_M+qlora --model base-Q4_K_M.gguf --mmproj mmproj-base-F16.gguf --lora lora-qlora-F16.gguf --parallel 2
for t in Q8_0 Q4_K_M Q3_K_M; do
  python3 scripts/07_eval_llamacpp.py --name student-kd-$t-mmprojQ8 --model student-kd-$t.gguf --mmproj mmproj-student-kd-Q8_0.gguf --parallel 2
done
python3 scripts/07_eval_llamacpp.py --name ft-Q4_K_M-latency --model ft-Q4_K_M.gguf --mmproj mmproj-ft-F16.gguf --parallel 1 --limit 8 --device-label "smoke-cpu"
bash scripts/08_bench_llamacpp.sh "$W/gguf" "smoke-cpu" "ft-F16 ft-Q4_K_M ft-Q2_K" 16 8
python3 scripts/09_report.py
# the demo, one-shot mode, on the recommended config
python3 scripts/10_demo.py --image $W/data/images/$(ls $W/data/images | grep eval | head -1) --question "What color are the squares?"
python3 -m pytest -q tests/test_vqa_metric.py
echo "SMOKE TEST PASSED"
