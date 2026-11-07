#!/usr/bin/env bash
# Build llama.cpp at the commit this project was tested against. CUDA on Linux, Metal on macOS.
set -euo pipefail
LLAMA_COMMIT=${LLAMA_COMMIT:-52c9bad}
DIR=${LLAMA_CPP:-llama.cpp}
[ -d "$DIR" ] || git clone https://github.com/ggml-org/llama.cpp "$DIR"
cd "$DIR" && git fetch --depth 1 origin "$LLAMA_COMMIT" 2>/dev/null || git fetch origin
git checkout -q "$LLAMA_COMMIT"
if [[ "$(uname)" == "Darwin" ]]; then FLAGS="-DGGML_METAL=ON"
elif command -v nvcc >/dev/null; then FLAGS="-DGGML_CUDA=ON"
else FLAGS=""; fi
cmake -B build $FLAGS -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-quantize llama-server llama-mtmd-cli llama-imatrix llama-bench
# convert_hf_to_gguf.py uses the in-repo gguf-py; sentencepiece comes from requirements.txt.
# (Not installing llama.cpp's own requirements: they pin a CPU torch that would replace the CUDA one.)
echo "llama.cpp ready at $DIR ($(git rev-parse --short HEAD))"
