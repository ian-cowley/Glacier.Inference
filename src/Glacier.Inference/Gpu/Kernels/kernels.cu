// Glacier.Inference High-Performance CUDA Kernels
// Target Architectures:
//   sm_75 (Turing: RTX 2060, GTX 1660, T4)
//   sm_80 (Ampere Data Center: A100)
//   sm_86 (Ampere Consumer: RTX 3060, RTX 3070, RTX 3080, RTX 3090)
//   sm_89 (Ada Lovelace: RTX 4060, RTX 4070, RTX 4080, RTX 4090, L40)
//   sm_90 (Hopper: H100)
//   compute_75 (Forward-compatible PTX JIT for Blackwell & future architectures)
//
// Zero C++ DLL dependencies: Compiled directly into universal fatbinary CUBIN,
// embedded as an unmanaged resource, and executed via pure C# nvcuda.dll Driver API.

#include "common.cuh"
#include "gemv.cuh"
#include "ops.cuh"
#include "attention.cuh"
#include "gemm_batch.cuh"
