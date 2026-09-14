// Glacier.Inference High-Performance CUDA Kernels
// Elementwise, Normalization, RoPE, KV-Cache Store, and Reduction/Penalty Kernels
#pragma once

#include "common.cuh"

extern "C" {

// =========================================================================
// 3. RMSNorm Kernel (in-place or out-of-place in VRAM)
// =========================================================================
__global__ void rms_norm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ dst,
    int size,
    float eps
) {
    __shared__ float s_sum;
    int tid = threadIdx.x;

    if (tid == 0) {
        s_sum = 0.0f;
    }
    __syncthreads();

    float local_sum = 0.0f;

    for (int i = tid; i < size; i += blockDim.x) {
        float val = x[i];
        local_sum += val * val;
    }

    local_sum = warp_reduce_sum(local_sum);

    if ((tid % WARP_SIZE) == 0) {
        atomicAdd(&s_sum, local_sum);
    }
    __syncthreads();

    float rms = rsqrtf((s_sum / (float)size) + eps);

    for (int i = tid; i < size; i += blockDim.x) {
        dst[i] = x[i] * rms * weight[i];
    }
}

// =========================================================================
// 4. SwiGLU Activation Kernel: dst[i] = SiLU(gate[i]) * up[i]
// =========================================================================
__global__ void swiglu_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ dst,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = gate[idx];
        float silu = g / (1.0f + __expf(-g));
        dst[idx] = silu * up[idx];
    }
}

// =========================================================================
// 5. Add Bias Vector: y[i] += bias[i]
// =========================================================================
__global__ void add_bias_kernel(
    float* __restrict__ y,
    const float* __restrict__ bias,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        y[idx] += bias[idx];
    }
}

// =========================================================================
// 6. Vector Add: a[i] += b[i]
// =========================================================================
__global__ void vec_add_kernel(
    float* __restrict__ a,
    const float* __restrict__ b,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        a[idx] += b[idx];
    }
}

// =========================================================================
// 7. RoPE Kernel (Rotary Position Embedding, NeOX style)
// =========================================================================
__global__ void rope_kernel(
    float* __restrict__ q,
    float* __restrict__ k,
    int n_heads_q,
    int n_heads_kv,
    int head_dim,
    int pos,
    float freq_base,
    float freq_scale
) {
    int half_dim = head_dim / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int q_half = n_heads_q * half_dim;
    int total_half = (n_heads_q + n_heads_kv) * half_dim;

    if (idx >= total_half) return;

    bool is_k = (idx >= q_half);
    int head_idx = is_k ? (idx - q_half) / half_dim : idx / half_dim;
    int i = is_k ? (idx - q_half) % half_dim : idx % half_dim;

    float freq = 1.0f / powf(freq_base, (float)(2 * i) / (float)head_dim);
    float theta = (float)pos * freq * freq_scale;
    float cos_theta = cosf(theta);
    float sin_theta = sinf(theta);

    float* vec = is_k ? (k + head_idx * head_dim) : (q + head_idx * head_dim);
    float v0 = vec[i];
    float v1 = vec[i + half_dim];

    vec[i] = v0 * cos_theta - v1 * sin_theta;
    vec[i + half_dim] = v0 * sin_theta + v1 * cos_theta;
}

// =========================================================================
// 8. KV Cache Store Kernel: writes K and V vectors for current pos (FP32)
// Layout in VRAM per layer: [n_heads_kv, max_seq_len, head_dim]
// =========================================================================
__global__ void kv_cache_store_kernel(
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    const float* __restrict__ k,
    const float* __restrict__ v,
    int n_heads_kv,
    int head_dim,
    int max_seq_len,
    int pos
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_heads_kv * head_dim;
    if (idx >= total) return;

    int h = idx / head_dim;
    int d = idx % head_dim;

    size_t offset = ((size_t)h * max_seq_len + pos) * head_dim + d;
    k_cache[offset] = k[idx];
    v_cache[offset] = v[idx];
}

// =========================================================================
// 8b. FP16 KV Cache Store Kernel
// =========================================================================
__global__ void kv_cache_store_f16(
    half* __restrict__ k_cache,
    half* __restrict__ v_cache,
    const float* __restrict__ k,
    const float* __restrict__ v,
    int n_heads_kv,
    int head_dim,
    int max_seq_len,
    int pos
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_heads_kv * head_dim;
    if (idx >= total) return;

    int h = idx / head_dim;
    int d = idx % head_dim;

    size_t offset = ((size_t)h * max_seq_len + pos) * head_dim + d;
    k_cache[offset] = __float2half(k[idx]);
    v_cache[offset] = __float2half(v[idx]);
}

// =========================================================================
// 8c. FP8 (e4m3) KV Cache Store Kernel
// =========================================================================
__global__ void kv_cache_store_fp8(
    __nv_fp8_e4m3* __restrict__ k_cache,
    __nv_fp8_e4m3* __restrict__ v_cache,
    const float* __restrict__ k,
    const float* __restrict__ v,
    int n_heads_kv,
    int head_dim,
    int max_seq_len,
    int pos
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_heads_kv * head_dim;
    if (idx >= total) return;

    int h = idx / head_dim;
    int d = idx % head_dim;

    size_t offset = ((size_t)h * max_seq_len + pos) * head_dim + d;
    k_cache[offset] = __nv_fp8_e4m3(k[idx]);
    v_cache[offset] = __nv_fp8_e4m3(v[idx]);
}

// =========================================================================
// 13. Batched RMSNorm Kernel
// =========================================================================
__global__ void rms_norm_batch(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ dst,
    int size,
    float eps
) {
    int b = blockIdx.x; // token index in batch
    const float* x_b = x + (size_t)b * size;
    float* dst_b = dst + (size_t)b * size;

    __shared__ float s_sum;
    int tid = threadIdx.x;

    if (tid == 0) s_sum = 0.0f;
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
        float val = x_b[i];
        local_sum += val * val;
    }
    local_sum = warp_reduce_sum(local_sum);

    if ((tid % WARP_SIZE) == 0) {
        atomicAdd(&s_sum, local_sum);
    }
    __syncthreads();

    float rms = rsqrtf((s_sum / (float)size) + eps);

    for (int i = tid; i < size; i += blockDim.x) {
        dst_b[i] = x_b[i] * rms * weight[i];
    }
}

// =========================================================================
// 14. Batched Add Bias Vector: y[t, i] += bias[i]
// =========================================================================
__global__ void add_bias_batch(
    float* __restrict__ y,
    const float* __restrict__ bias,
    int size,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        y[idx] += bias[idx % size];
    }
}

// =========================================================================
// 15. Batched Vector Add: a[i] += b[i]
// =========================================================================
__global__ void vec_add_batch(
    float* __restrict__ a,
    const float* __restrict__ b,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        a[idx] += b[idx];
    }
}

// =========================================================================
// 16. Batched RoPE Kernel
// =========================================================================
__global__ void rope_batch(
    float* __restrict__ q,
    float* __restrict__ k,
    int n_heads_q,
    int n_heads_kv,
    int head_dim,
    int start_pos,
    int batch_size,
    float freq_base,
    float freq_scale
) {
    int half_dim = head_dim / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_half = (n_heads_q + n_heads_kv) * half_dim;
    int total_all = total_half * batch_size;

    if (idx >= total_all) return;

    int t = idx / total_half;
    int pos = start_pos + t;
    int sub_idx = idx % total_half;

    int q_half = n_heads_q * half_dim;
    bool is_k = (sub_idx >= q_half);
    int head_idx = is_k ? (sub_idx - q_half) / half_dim : sub_idx / half_dim;
    int i = is_k ? (sub_idx - q_half) % half_dim : sub_idx % half_dim;

    float freq = 1.0f / powf(freq_base, (float)(2 * i) / (float)head_dim);
    float theta = (float)pos * freq * freq_scale;
    float cos_theta = cosf(theta);
    float sin_theta = sinf(theta);

    int q_token_dim = n_heads_q * head_dim;
    int k_token_dim = n_heads_kv * head_dim;
    float* vec = is_k ? (k + (size_t)t * k_token_dim + head_idx * head_dim) 
                      : (q + (size_t)t * q_token_dim + head_idx * head_dim);

    float v0 = vec[i];
    float v1 = vec[i + half_dim];

    vec[i] = v0 * cos_theta - v1 * sin_theta;
    vec[i + half_dim] = v0 * sin_theta + v1 * cos_theta;
}

// =========================================================================
// 17. Batched KV Cache Store Kernel
// =========================================================================
__global__ void kv_cache_store_batch(
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    const float* __restrict__ k,
    const float* __restrict__ v,
    int n_heads_kv,
    int head_dim,
    int max_seq_len,
    int start_pos,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int kv_dim = n_heads_kv * head_dim;
    int total = kv_dim * batch_size;
    if (idx >= total) return;

    int t = idx / kv_dim;
    int sub_idx = idx % kv_dim;
    int h = sub_idx / head_dim;
    int d = sub_idx % head_dim;
    int pos = start_pos + t;

    size_t offset = ((size_t)h * max_seq_len + pos) * head_dim + d;
    k_cache[offset] = k[idx];
    v_cache[offset] = v[idx];
}

// =========================================================================
// 17b. Batched FP16 KV Cache Store Kernel
// =========================================================================
__global__ void kv_cache_store_batch_f16(
    half* __restrict__ k_cache,
    half* __restrict__ v_cache,
    const float* __restrict__ k,
    const float* __restrict__ v,
    int n_heads_kv,
    int head_dim,
    int max_seq_len,
    int start_pos,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int kv_dim = n_heads_kv * head_dim;
    int total = kv_dim * batch_size;
    if (idx >= total) return;

    int t = idx / kv_dim;
    int sub_idx = idx % kv_dim;
    int h = sub_idx / head_dim;
    int d = sub_idx % head_dim;
    int pos = start_pos + t;

    size_t offset = ((size_t)h * max_seq_len + pos) * head_dim + d;
    k_cache[offset] = __float2half(k[idx]);
    v_cache[offset] = __float2half(v[idx]);
}

// =========================================================================
// 17c. Batched FP8 (e4m3) KV Cache Store Kernel
// =========================================================================
__global__ void kv_cache_store_batch_fp8(
    __nv_fp8_e4m3* __restrict__ k_cache,
    __nv_fp8_e4m3* __restrict__ v_cache,
    const float* __restrict__ k,
    const float* __restrict__ v,
    int n_heads_kv,
    int head_dim,
    int max_seq_len,
    int start_pos,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int kv_dim = n_heads_kv * head_dim;
    int total = kv_dim * batch_size;
    if (idx >= total) return;

    int t = idx / kv_dim;
    int sub_idx = idx % kv_dim;
    int h = sub_idx / head_dim;
    int d = sub_idx % head_dim;
    int pos = start_pos + t;

    size_t offset = ((size_t)h * max_seq_len + pos) * head_dim + d;
    k_cache[offset] = __nv_fp8_e4m3(k[idx]);
    v_cache[offset] = __nv_fp8_e4m3(v[idx]);
}

// =========================================================================
// 18. GPU Argmax Reduction Kernel
// Finds the token ID with the maximum logit across the entire vocabulary.
// Runs in ~3 microseconds, completely eliminating 608 KB DtoH PCIe transfers!
// =========================================================================
__global__ void argmax_kernel(
    const float* __restrict__ logits,
    int vocab_size,
    int* __restrict__ out_best_token,
    float* __restrict__ out_best_logit
) {
    int tid = threadIdx.x; // 0..511 (512 threads)
    float best_val = -1e30f;
    int best_idx = -1;

    // Grid-stride loop over vocab_size
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float v = logits[i];
        if (v > best_val) {
            best_val = v;
            best_idx = i;
        }
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, best_val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, best_idx, offset);
        if (other_val > best_val) {
            best_val = other_val;
            best_idx = other_idx;
        }
    }

    __shared__ float s_max_val[16]; // 512 / 32 = 16 warps
    __shared__ int s_max_idx[16];

    int warp_id = tid / 32;
    int lane_id = tid % 32;

    if (lane_id == 0) {
        s_max_val[warp_id] = best_val;
        s_max_idx[warp_id] = best_idx;
    }
    __syncthreads();

    // First warp reduces the 16 warp winners
    if (warp_id == 0) {
        float val = (lane_id < 16) ? s_max_val[lane_id] : -1e30f;
        int idx = (lane_id < 16) ? s_max_idx[lane_id] : -1;

        #pragma unroll
        for (int offset = 8; offset > 0; offset /= 2) {
            float other_val = __shfl_down_sync(0xffffffff, val, offset);
            int other_idx = __shfl_down_sync(0xffffffff, idx, offset);
            if (other_val > val) {
                val = other_val;
                idx = other_idx;
            }
        }

        if (lane_id == 0) {
            if (out_best_token != nullptr) *out_best_token = idx;
            if (out_best_logit != nullptr) *out_best_logit = val;
        }
    }
}

// =========================================================================
// 19. Apply Repetition Penalty Directly on GPU Logits
// =========================================================================
__global__ void apply_repetition_penalty_kernel(
    float* __restrict__ logits,
    const int* __restrict__ recent_tokens,
    int count,
    float penalty
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    int tid = recent_tokens[idx];
    if (tid >= 0) {
        float val = logits[tid];
        logits[tid] = (val > 0.0f) ? (val / penalty) : (val * penalty);
    }
}

// =========================================================================
// 20. Per-Head RMSNorm Kernel (Qwen3 / Qwen3.8 QK-Norm)
// =========================================================================
__global__ void rms_norm_heads_kernel(
    float* __restrict__ x,               // [n_heads, head_dim] in-place
    const float* __restrict__ weight,   // [head_dim]
    int n_heads,
    int head_dim,
    float eps
) {
    int h = blockIdx.x;
    if (h >= n_heads) return;

    float* head_x = x + h * head_dim;
    int tid = threadIdx.x;

    float local_sum = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float val = head_x[i];
        local_sum += val * val;
    }

    local_sum = warp_reduce_sum(local_sum);
    __shared__ float s_sum;
    if (tid == 0) s_sum = 0.0f;
    __syncthreads();

    if ((tid % WARP_SIZE) == 0) {
        atomicAdd(&s_sum, local_sum);
    }
    __syncthreads();

    float rms = rsqrtf((s_sum / (float)head_dim) + eps);
    for (int i = tid; i < head_dim; i += blockDim.x) {
        head_x[i] = head_x[i] * rms * weight[i];
    }
}

// =========================================================================
// 21. Batched Per-Head RMSNorm Kernel (Qwen3 / Qwen3.8 Prompt Prefill QK-Norm)
// =========================================================================
__global__ void rms_norm_batch_heads_kernel(
    float* __restrict__ x,               // [batch_size, n_heads, head_dim] in-place
    const float* __restrict__ weight,   // [head_dim]
    int n_heads,
    int head_dim,
    int batch_size,
    float eps
) {
    int h = blockIdx.x;
    int t = blockIdx.y;
    if (h >= n_heads || t >= batch_size) return;

    float* head_x = x + (size_t)t * ((size_t)n_heads * head_dim) + h * head_dim;
    int tid = threadIdx.x;

    float local_sum = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float val = head_x[i];
        local_sum += val * val;
    }

    local_sum = warp_reduce_sum(local_sum);
    __shared__ float s_sum;
    if (tid == 0) s_sum = 0.0f;
    __syncthreads();

    if ((tid % WARP_SIZE) == 0) {
        atomicAdd(&s_sum, local_sum);
    }
    __syncthreads();

    float rms = rsqrtf((s_sum / (float)head_dim) + eps);
    for (int i = tid; i < head_dim; i += blockDim.x) {
        head_x[i] = head_x[i] * rms * weight[i];
    }
}

} // extern "C"
