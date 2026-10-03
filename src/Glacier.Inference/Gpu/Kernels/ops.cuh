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

__global__ void swiglu_bwd_kernel(
    const float* __restrict__ d_hidden,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ d_gate,
    float* __restrict__ d_up,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float dh = d_hidden[idx];
        float g = gate[idx];
        float u = up[idx];
        float sig = 1.0f / (1.0f + __expf(-g));
        float siluG = g * sig;
        d_up[idx] = dh * siluG;
        float dSilu = sig * (1.0f + g * (1.0f - sig));
        d_gate[idx] = dh * u * dSilu;
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

// =========================================================================
// 17. Fused QK-Norm (RMSNorm) + 3D-RoPE for FLUX.1
// Grid: blockIdx.x = h (0..n_heads - 1), blockIdx.y = b (0..num_tokens - 1)
// Block: 128 threads (threadIdx.x = d in 0..127)
// =========================================================================
__global__ void flux_qk_norm_rope(
    float* __restrict__ q,
    float* __restrict__ k,
    const float* __restrict__ query_scale,
    const float* __restrict__ key_scale,
    const float* __restrict__ rope_cos,
    const float* __restrict__ rope_sin,
    int n_heads,
    int head_dim,
    int num_tokens
) {
    int h = blockIdx.x;
    int b = blockIdx.y;
    if (h >= n_heads || b >= num_tokens) return;

    int tid = threadIdx.x; // 0..127
    size_t offset = (size_t)b * n_heads * head_dim + h * head_dim + tid;

    float q_val = q[offset];
    float k_val = k[offset];

    float q_sq = q_val * q_val;
    float k_sq = k_val * k_val;

    q_sq = warp_reduce_sum(q_sq);
    k_sq = warp_reduce_sum(k_sq);

    __shared__ float s_q_sum[4];
    __shared__ float s_k_sum[4];

    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    if (lane_id == 0) {
        s_q_sum[warp_id] = q_sq;
        s_k_sum[warp_id] = k_sq;
    }
    __syncthreads();

    float total_q_sq = s_q_sum[0] + s_q_sum[1] + s_q_sum[2] + s_q_sum[3];
    float total_k_sq = s_k_sum[0] + s_k_sum[1] + s_k_sum[2] + s_k_sum[3];

    float inv_rms_q = rsqrtf((total_q_sq / 128.0f) + 1e-6f);
    float inv_rms_k = rsqrtf((total_k_sq / 128.0f) + 1e-6f);

    float q_norm = q_val * inv_rms_q * query_scale[tid];
    float k_norm = k_val * inv_rms_k * key_scale[tid];

    // 3D RoPE rotation: pairs (2*i, 2*i + 1)
    int pair_idx = tid / 2;
    float cos_val = rope_cos[b * 64 + pair_idx];
    float sin_val = rope_sin[b * 64 + pair_idx];

    // Neighbor exchange across pair
    float q_other = __shfl_xor_sync(0xFFFFFFFF, q_norm, 1);
    float k_other = __shfl_xor_sync(0xFFFFFFFF, k_norm, 1);

    float q_rot, k_rot;
    if ((tid & 1) == 0) {
        q_rot = q_norm * cos_val - q_other * sin_val;
        k_rot = k_norm * cos_val - k_other * sin_val;
    } else {
        q_rot = q_other * sin_val + q_norm * cos_val;
        k_rot = k_other * sin_val + k_norm * cos_val;
    }

    q[offset] = q_rot;
    k[offset] = k_rot;
}

// =========================================================================
// 18. Fused GELU + Concat for FLUX.1 SingleStreamBlocks:
// Concatenates attn_out [num_tokens, 3072] and GELU(qkv_mlp[:, 9216..21503])
// into dst [num_tokens, 15360]
// =========================================================================
__global__ void flux_fused_gelu_concat(
    const float* __restrict__ attn_out,
    const float* __restrict__ qkv_mlp,
    float* __restrict__ dst,
    int num_tokens
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_tokens * 15360;
    if (idx >= total_elements) return;

    int tok = idx / 15360;
    int col = idx % 15360;

    if (col < 3072) {
        dst[idx] = attn_out[tok * 3072 + col];
    } else {
        int mlp_col = col - 3072;
        float x = qkv_mlp[tok * 21504 + 9216 + mlp_col];
        // GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        float cdf = 0.5f * (1.0f + erff(x * 0.7071067811865475f));
        dst[idx] = x * cdf;
    }
}

// =========================================================================
// 18b. Pure In-Place GeLU Activation: x[i] = 0.5 * x * (1 + erf(x / sqrt(2)))
// =========================================================================
__global__ void flux_gelu_kernel(
    float* __restrict__ x,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        float val = x[idx];
        float cdf = 0.5f * (1.0f + erff(val * 0.7071067811865475f));
        x[idx] = val * cdf;
    }
}

// =========================================================================
// 19. AdaLN Modulation + LayerNorm: dst = ((x - mean) / std) * (1 + scale) + shift
// 1 block of 128 threads per token
// =========================================================================
__global__ void flux_adaln_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const float* __restrict__ shift,
    const float* __restrict__ scale,
    int num_tokens,
    int dim
) {
    int tok = blockIdx.x;
    if (tok >= num_tokens) return;

    int tid = threadIdx.x;
    const float* s = src + tok * dim;
    float* d = dst + tok * dim;

    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (int i = tid; i < dim; i += blockDim.x) {
        float v = s[i];
        sum += v;
        sum_sq += v * v;
    }

    sum = warp_reduce_sum(sum);
    sum_sq = warp_reduce_sum(sum_sq);

    __shared__ float s_sum[4];
    __shared__ float s_sum_sq[4];

    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    if (lane_id == 0) {
        s_sum[warp_id] = sum;
        s_sum_sq[warp_id] = sum_sq;
    }
    __syncthreads();

    float total_sum = s_sum[0] + s_sum[1] + s_sum[2] + s_sum[3];
    float total_sum_sq = s_sum_sq[0] + s_sum_sq[1] + s_sum_sq[2] + s_sum_sq[3];

    float mean = total_sum / (float)dim;
    float var = fmaxf(0.0f, (total_sum_sq / (float)dim) - (mean * mean));
    float inv_std = rsqrtf(var + 1e-6f);

    for (int i = tid; i < dim; i += blockDim.x) {
        d[i] = ((s[i] - mean) * inv_std) * (1.0f + scale[i]) + shift[i];
    }
}

// =========================================================================
// 20. Residual Gated Add: target[i] += gate[i % dim] * update[i]
// =========================================================================
__global__ void flux_residual_gated(
    float* __restrict__ target,
    const float* __restrict__ update,
    const float* __restrict__ gate,
    int total_elements,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int col = idx % dim;
    target[idx] += gate[col] * update[idx];
}

// =========================================================================
// 21. Fused QKV Unpack + RMSNorm + 3D-RoPE for FLUX.1
// Unpacks Q, K, V from strided source (e.g. 21504 or 9216), normalizes Q & K,
// applies 3D-RoPE rotation, and stores contiguous [num_tokens, 3072] in q_dst, k_dst, v_dst
// =========================================================================
__global__ void flux_qkv_prep(
    const float* __restrict__ qkv_src,
    float* __restrict__ q_dst,
    float* __restrict__ k_dst,
    float* __restrict__ v_dst,
    const float* __restrict__ query_scale,
    const float* __restrict__ key_scale,
    const float* __restrict__ rope_cos,
    const float* __restrict__ rope_sin,
    int src_stride,
    int n_heads,
    int head_dim,
    int num_tokens,
    int token_offset
) {
    int h = blockIdx.x;
    int b = blockIdx.y;
    if (h >= n_heads || b >= num_tokens) return;

    int tid = threadIdx.x; // 0..127
    size_t src_offset = (size_t)b * src_stride + h * head_dim + tid;
    float q_val = qkv_src[src_offset];
    float k_val = qkv_src[src_offset + 3072];
    float v_val = qkv_src[src_offset + 6144];

    int global_t = token_offset + b;
    size_t dst_offset = (size_t)global_t * (n_heads * head_dim) + h * head_dim + tid;
    v_dst[dst_offset] = v_val;

    float q_sq = q_val * q_val;
    float k_sq = k_val * k_val;

    q_sq = warp_reduce_sum(q_sq);
    k_sq = warp_reduce_sum(k_sq);

    __shared__ float s_q_sum[4];
    __shared__ float s_k_sum[4];

    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    if (lane_id == 0) {
        s_q_sum[warp_id] = q_sq;
        s_k_sum[warp_id] = k_sq;
    }
    __syncthreads();

    float total_q_sq = s_q_sum[0] + s_q_sum[1] + s_q_sum[2] + s_q_sum[3];
    float total_k_sq = s_k_sum[0] + s_k_sum[1] + s_k_sum[2] + s_k_sum[3];

    float inv_rms_q = rsqrtf((total_q_sq / 128.0f) + 1e-6f);
    float inv_rms_k = rsqrtf((total_k_sq / 128.0f) + 1e-6f);

    float q_norm = q_val * inv_rms_q * query_scale[tid];
    float k_norm = k_val * inv_rms_k * key_scale[tid];

    int pair_idx = tid / 2;
    float cos_val = rope_cos[global_t * 64 + pair_idx];
    float sin_val = rope_sin[global_t * 64 + pair_idx];

    float q_other = __shfl_xor_sync(0xFFFFFFFF, q_norm, 1);
    float k_other = __shfl_xor_sync(0xFFFFFFFF, k_norm, 1);

    float q_rot, k_rot;
    if ((tid & 1) == 0) {
        q_rot = q_norm * cos_val - q_other * sin_val;
        k_rot = k_norm * cos_val - k_other * sin_val;
    } else {
        q_rot = q_other * sin_val + q_norm * cos_val;
        k_rot = k_other * sin_val + k_norm * cos_val;
    }

    q_dst[dst_offset] = q_rot;
    k_dst[dst_offset] = k_rot;
}

// =========================================================================
// 15. Neural VAE Preprocess Latents: dst[i] = (src[i] / 0.3611f) + 0.1159f
// =========================================================================
__global__ void vae_scale_latents(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = (src[idx] / 0.3611f) + 0.1159f;
    }
}

// =========================================================================
// 16. Tiled High-Throughput VAE 2D Convolution 3x3 (stride=1, pad=1)
// Grid: blockIdx.x = (W + 15)/16, blockIdx.y = (H + 15)/16, blockIdx.z = oc
// Block: 16 x 16 (256 threads)
// =========================================================================
__global__ void conv2d_3x3(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int inC,
    int outC,
    int H,
    int W
) {
    int oc = blockIdx.z;
    if (oc >= outC) return;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * 16 + tx;

    int xBase = blockIdx.x * 16;
    int yBase = blockIdx.y * 16;
    int x = xBase + tx;
    int y = yBase + ty;

    int spatial = H * W;
    float sum = bias ? bias[oc] : 0.0f;
    const float* wOc = weight + (size_t)oc * inC * 9;

    __shared__ float s_tile[18][18];

    for (int ic = 0; ic < inC; ic++) {
        const float* pSrcIc = src + (size_t)ic * spatial;
        const float* wIc = wOc + ic * 9;

        #pragma unroll
        for (int idx = tid; idx < 324; idx += 256) {
            int sy = idx / 18;
            int sx = idx % 18;
            int gx = xBase - 1 + sx;
            int gy = yBase - 1 + sy;

            float val = 0.0f;
            if (gx >= 0 && gx < W && gy >= 0 && gy < H) {
                val = pSrcIc[gy * W + gx];
            }
            s_tile[sy][sx] = val;
        }
        __syncthreads();

        if (x < W && y < H) {
            sum += s_tile[ty + 0][tx + 0] * wIc[0];
            sum += s_tile[ty + 0][tx + 1] * wIc[1];
            sum += s_tile[ty + 0][tx + 2] * wIc[2];

            sum += s_tile[ty + 1][tx + 0] * wIc[3];
            sum += s_tile[ty + 1][tx + 1] * wIc[4];
            sum += s_tile[ty + 1][tx + 2] * wIc[5];

            sum += s_tile[ty + 2][tx + 0] * wIc[6];
            sum += s_tile[ty + 2][tx + 1] * wIc[7];
            sum += s_tile[ty + 2][tx + 2] * wIc[8];
        }
        __syncthreads();
    }

    if (x < W && y < H) {
        dst[(size_t)oc * spatial + y * W + x] = sum;
    }
}

// =========================================================================
// 17. VAE 2D Convolution 1x1 (Pointwise Channel Projection)
// Grid: blockIdx.x = (spatial + 15)/16, blockIdx.y = (outC + 15)/16
// Block: 16 x 16 (256 threads)
// =========================================================================
__global__ void conv2d_1x1(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int inC,
    int outC,
    int spatial
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int oc = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= spatial || oc >= outC) return;

    float sum = bias ? bias[oc] : 0.0f;
    const float* wRow = weight + (size_t)oc * inC;

    for (int ic = 0; ic < inC; ic++) {
        sum += wRow[ic] * src[(size_t)ic * spatial + i];
    }

    dst[(size_t)oc * spatial + i] = sum;
}

// =========================================================================
// 18. Fused GroupNorm (32 Groups) + Optional SiLU Activation
// Grid: blockIdx.x = g (0 .. groups - 1)
// Block: 256 threads
// =========================================================================
__global__ void group_norm_silu(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int channels,
    int spatial,
    int groups,
    int applySilu
) {
    int g = blockIdx.x;
    if (g >= groups) return;

    int channelsPerGroup = channels / groups;
    int groupSpatial = channelsPerGroup * spatial;

    float threadSum = 0.0f;
    float threadSumSq = 0.0f;

    for (int idx = threadIdx.x; idx < groupSpatial; idx += blockDim.x) {
        int cLocal = idx / spatial;
        int s = idx % spatial;
        int fullC = g * channelsPerGroup + cLocal;
        float val = src[(size_t)fullC * spatial + s];
        threadSum += val;
        threadSumSq += val * val;
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        threadSum += __shfl_down_sync(0xFFFFFFFF, threadSum, offset);
        threadSumSq += __shfl_down_sync(0xFFFFFFFF, threadSumSq, offset);
    }

    __shared__ float sSum[8];
    __shared__ float sSumSq[8];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    if (lane == 0) {
        sSum[wid] = threadSum;
        sSumSq[wid] = threadSumSq;
    }
    __syncthreads();

    if (wid == 0) {
        float valSum = (lane < blockDim.x / 32) ? sSum[lane] : 0.0f;
        float valSumSq = (lane < blockDim.x / 32) ? sSumSq[lane] : 0.0f;
        for (int offset = 4; offset > 0; offset /= 2) {
            valSum += __shfl_down_sync(0xFFFFFFFF, valSum, offset);
            valSumSq += __shfl_down_sync(0xFFFFFFFF, valSumSq, offset);
        }
        if (lane == 0) {
            sSum[0] = valSum;
            sSumSq[0] = valSumSq;
        }
    }
    __syncthreads();

    float mean = sSum[0] / (float)groupSpatial;
    float var = fmaxf(0.0f, (sSumSq[0] / (float)groupSpatial) - (mean * mean));
    float invStd = rsqrtf(var + 1e-6f);

    for (int idx = threadIdx.x; idx < groupSpatial; idx += blockDim.x) {
        int cLocal = idx / spatial;
        int s = idx % spatial;
        int fullC = g * channelsPerGroup + cLocal;
        float gamma = weight[fullC];
        float beta = bias[fullC];
        float val = src[(size_t)fullC * spatial + s];
        float norm = ((val - mean) * invStd) * gamma + beta;
        if (applySilu) {
            norm = norm / (1.0f + __expf(-norm));
        }
        dst[(size_t)fullC * spatial + s] = norm;
    }
}

// =========================================================================
// 19. 2x Nearest-Neighbor Spatial Upsampling
// Grid: blockIdx.x = (outW + 15)/16, blockIdx.y = (outH + 15)/16, blockIdx.z = c
// Block: 16 x 16 (256 threads)
// =========================================================================
__global__ void upsample2x_nearest(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int channels,
    int inH,
    int inW
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;

    int outW = inW * 2;
    int outH = inH * 2;
    if (x >= outW || y >= outH || c >= channels) return;

    int inX = x / 2;
    int inY = y / 2;

    dst[(size_t)c * outH * outW + y * outW + x] = src[(size_t)c * inH * inW + inY * inW + inX];
}

// =========================================================================
// 20. Tensor Add: a[i] += b[i]
// =========================================================================
__global__ void tensor_add(
    float* __restrict__ a,
    const float* __restrict__ b,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        a[idx] += b[idx];
    }
}

// =========================================================================
// 21. VAE Spatial Self-Attention (FlashAttention with Online Softmax)
// Grid: blockIdx.x = i (0 .. spatial - 1)
// Block: 128 threads
// =========================================================================
__global__ void vae_spatial_attention(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ attn_out,
    int spatial,
    int channels,
    float scale
) {
    int i = blockIdx.x;
    if (i >= spatial) return;

    int tid = threadIdx.x;

    __shared__ float s_q[512];
    for (int c = tid; c < channels; c += blockDim.x) {
        s_q[c] = q[(size_t)c * spatial + i];
    }
    __syncthreads();

    float m = -1e30f;
    float l = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    __shared__ float s_warp_sum[4];

    for (int j = 0; j < spatial; j++) {
        float dot = 0.0f;
        for (int c = tid; c < channels; c += blockDim.x) {
            dot += s_q[c] * k[(size_t)c * spatial + j];
        }
        dot = warp_reduce_sum(dot);
        int warp_id = tid / WARP_SIZE;
        int lane_id = tid % WARP_SIZE;
        if (lane_id == 0) s_warp_sum[warp_id] = dot;
        __syncthreads();

        float total_dot = s_warp_sum[0] + s_warp_sum[1] + s_warp_sum[2] + s_warp_sum[3];
        float s_j = total_dot * scale;

        float m_new = fmaxf(m, s_j);
        float alpha = expf(m - m_new);
        float w_j = expf(s_j - m_new);

        l = l * alpha + w_j;
        m = m_new;

        acc0 = acc0 * alpha + w_j * v[(size_t)(tid + 0) * spatial + j];
        acc1 = acc1 * alpha + w_j * v[(size_t)(tid + 128) * spatial + j];
        acc2 = acc2 * alpha + w_j * v[(size_t)(tid + 256) * spatial + j];
        acc3 = acc3 * alpha + w_j * v[(size_t)(tid + 384) * spatial + j];

        __syncthreads();
    }

    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    attn_out[(size_t)(tid + 0) * spatial + i] = acc0 * inv_l;
    attn_out[(size_t)(tid + 128) * spatial + i] = acc1 * inv_l;
    attn_out[(size_t)(tid + 256) * spatial + i] = acc2 * inv_l;
    attn_out[(size_t)(tid + 384) * spatial + i] = acc3 * inv_l;
}

// =========================================================================
// 22. VAE RGB Planar Float -> Standard 24-bit Packed sRGB
// =========================================================================
__global__ void vae_clamp_rgb(
    const float* __restrict__ srcPlanar,
    unsigned char* __restrict__ dstRgb,
    int H,
    int W
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int spatial = H * W;
    int srcIdx = y * W + x;
    int dstIdx = (y * W + x) * 3;

    float r = srcPlanar[0 * spatial + srcIdx];
    float g = srcPlanar[1 * spatial + srcIdx];
    float b = srcPlanar[2 * spatial + srcIdx];

    int ir = (int)((r * 0.5f + 0.5f) * 255.0f);
    int ig = (int)((g * 0.5f + 0.5f) * 255.0f);
    int ib = (int)((b * 0.5f + 0.5f) * 255.0f);

    dstRgb[dstIdx]     = (unsigned char)(ir < 0 ? 0 : (ir > 255 ? 255 : ir));
    dstRgb[dstIdx + 1] = (unsigned char)(ig < 0 ? 0 : (ig > 255 ? 255 : ig));
    dstRgb[dstIdx + 2] = (unsigned char)(ib < 0 ? 0 : (ib > 255 ? 255 : ib));
}

} // extern "C"


