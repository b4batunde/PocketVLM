#!/usr/bin/env bash
# On-device half: run on the Apple Silicon laptop after copying for_mac.tar from the pod.
#   tar -xf for_mac.tar -C runs/qwen25vl3b && VLMC_DEVICE="M2 MacBook Air 16GB" bash run_mac.sh
# Close other apps and plug in power: these are the numbers that go on the resume.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=$PWD VLMC_CONFIG=${VLMC_CONFIG:-configs/qwen25vl3b.yaml}
export LLAMA_CPP=${LLAMA_CPP:-$PWD/llama.cpp}
: "${VLMC_DEVICE:?set VLMC_DEVICE, e.g. 'M3 Pro MacBook Pro 18GB'}"
W=$(python3 -c "from vlmc.config import load_config;print(load_config()['work_dir'])")
LAT_N=${LAT_N:-300}

# 1) raw throughput (llama-bench pp512 / tg128) for every LLM file
NAMES=$(cd $W/gguf && ls *.gguf | grep -vE '^(mmproj|lora|imatrix)' | sed 's/\.gguf$//' | tr '\n' ' ')
bash scripts/08_bench_llamacpp.sh $W/gguf "$VLMC_DEVICE" "$NAMES"

# 2) end-to-end single-stream latency + peak memory with images (first LAT_N eval questions)
for t in F16 Q8_0 Q4_K_M Q3_K_M Q2_K IQ2_XS; do
  [ -f $W/gguf/ft-$t.gguf ] || continue            # only the variants you copied over
  for mp in F16 Q8_0; do
    [ -f $W/gguf/mmproj-ft-$mp.gguf ] || continue
    python3 scripts/07_eval_llamacpp.py --name "mac-ft-$t-mmproj$mp" --model ft-$t.gguf --mmproj mmproj-ft-$mp.gguf \
      --parallel 1 --limit $LAT_N --device-label "$VLMC_DEVICE"
  done
done
for t in Q4_K_M Q3_K_M; do
  [ -f $W/gguf/student-kd-$t.gguf ] || continue
  python3 scripts/07_eval_llamacpp.py --name "mac-student-kd-$t" --model student-kd-$t.gguf \
    --mmproj mmproj-student-kd-Q8_0.gguf --parallel 1 --limit $LAT_N --device-label "$VLMC_DEVICE"
done

# 3) cross-device check: full eval of the recommended config on the Mac (should match the pod within noise)
REC=$(python3 -c "
from vlmc.config import load_config; from vlmc.results import recommended
r = recommended(load_config()); print(' '.join([r['name'], r['files']['llm'], r['files']['mmproj'], r['files'].get('lora', '-')]) if r else '')")
if [ -n "$REC" ]; then
  set -- $REC
  LORA_ARG=(); [ "$4" != "-" ] && LORA_ARG=(--lora "$4")
  python3 scripts/07_eval_llamacpp.py --name "mac-$1-full" --model "$2" --mmproj "$3" ${LORA_ARG[@]+"${LORA_ARG[@]}"} \
    --parallel 4 --device-label "$VLMC_DEVICE"
fi
python3 scripts/09_report.py
echo "Report: $W/results/REPORT.md"
echo "Demo:   python3 scripts/10_demo.py   (opens a local page; add --image x.jpg --question ... for one-shot)"
