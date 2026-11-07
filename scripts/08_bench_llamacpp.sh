#!/usr/bin/env bash
# Raw on-device throughput with llama-bench (text-only prefill/decode; the standard tokens/s number).
# usage: scripts/08_bench_llamacpp.sh <gguf_dir> "<device label>" "<gguf stems>" [n_prompt=512] [n_gen=128]
set -euo pipefail
GDIR=$1; DEV=$2; NAMES=$3; PP=${4:-512}; TG=${5:-128}
LLAMA_BIN=${LLAMA_BIN:-${LLAMA_CPP:-llama.cpp}/build/bin}
OUT="$(dirname "$GDIR")/results/bench"; mkdir -p "$OUT"
for n in $NAMES; do
  echo "== $n on $DEV"
  "$LLAMA_BIN/llama-bench" -m "$GDIR/$n.gguf" -p "$PP" -n "$TG" -r 3 -ngl 99 -o json > "$OUT/$n.raw.json"
  python3 - "$OUT/$n.raw.json" "$OUT/$n.json" "$n" "$DEV" <<'PY'
import json, sys
raw, out, name, dev = sys.argv[1:]
rows = json.load(open(raw))
res = {"name": name, "device": dev, "backend": rows[0].get("backends") if rows else None,
       "build": rows[0].get("build_commit") if rows else None, "cpu": rows[0].get("cpu_info"),
       "gpu": rows[0].get("gpu_info"), "model_size_bytes": rows[0].get("model_size")}
for r in rows:
    key = "pp" if r["n_prompt"] and not r["n_gen"] else "tg"
    res[f"{key}{r['n_prompt'] or r['n_gen']}_tokens_per_s"] = r["avg_ts"]
    res[f"{key}{r['n_prompt'] or r['n_gen']}_stddev"] = r["stddev_ts"]
json.dump(res, open(out, "w"), indent=2)
print(json.dumps(res))
PY
done
