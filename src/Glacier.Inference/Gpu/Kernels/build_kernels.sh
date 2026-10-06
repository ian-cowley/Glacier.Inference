#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "==================================================="
echo "Glacier.Inference CUDA Kernel Fatbinary Builder"
echo "==================================================="

if ! command -v nvcc &> /dev/null; then
    echo "[ERROR] nvcc not found in PATH."
    exit 1
fi

echo "[INFO] Compiling kernels.cu to kernels.cubin..."
nvcc -fatbin -O3 \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_90,code=sm_90 \
  -gencode arch=compute_75,code=compute_75 \
  -o kernels.cubin kernels.cu

echo "[SUCCESS] Successfully built kernels.cubin fatbinary ($(wc -c < kernels.cubin) bytes)."
