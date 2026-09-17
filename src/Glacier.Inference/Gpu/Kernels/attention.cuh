// Glacier.Inference High-Performance CUDA Kernels
// Multi-head & Grouped-Query FlashAttention-2 Kernels (Single-Token & Batched)
// Features online register-tracked softmax, fused QK dot-product + scaling + softmax + V reduction
#pragma once

#include "common.cuh"

extern "C" {

// =========================================================================
// 9. GQA Attention Kernel (FP32)
// 1 block of 128 threads per Q head (blockIdx.x = h, 0..n_heads_q - 1)
// threadIdx.x = d (0..127)
// =========================================================================
__global__ void attention_gqa_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ attn_out,
    float* __restrict__ scores_buf,
    int n_heads_q,
    int n_heads_kv,
    int head_dim,
    int max_seq_len,
    int pos,
    float attn_scale
) {
    int h = blockIdx.x;
    if (h >= n_heads_q) return;

    int tid = threadIdx.x; // 0..127
    int group_size = n_heads_q / n_heads_kv;
    int h_kv = h / group_size;

    float q_d = q[h * head_dim + tid];

    __shared__ float s_warp_sum[4];

    // Online softmax tracking in registers (FlashAttention-2)
    float m = -1e30f;
    float l = 0.0f;
    float acc = 0.0f;

    // Single fused pass over past tokens t = 0..pos
    for (int t = 0; t <= pos; t++) {
        size_t kv_offset = ((size_t)h_kv * max_seq_len + t) * head_dim + tid;
        float k_d = k_cache[kv_offset];
        float v_d = v_cache[kv_offset];

        float prod = q_d * k_d;
        prod = warp_reduce_sum(prod);
        int warp_id = tid / WARP_SIZE;
        int lane_id = tid % WARP_SIZE;

        if (lane_id == 0) {
            s_warp_sum[warp_id] = prod;
        }
        __syncthreads();

        float total_dot = s_warp_sum[0] + s_warp_sum[1] + s_warp_sum[2] + s_warp_sum[3];
        float s_t = total_dot * attn_scale;

        // Numerically stable online softmax update
        float m_new = fmaxf(m, s_t);
        float alpha = expf(m - m_new);
        float w_t = expf(s_t - m_new);

        acc = acc * alpha + w_t * v_d;
        l = l * alpha + w_t;
        m = m_new;

        __syncthreads();
    }

    attn_out[h * head_dim + tid] = (l > 0.0f) ? (acc / l) : 0.0f;
}

// =========================================================================
// 9e. FP16 GQA Attention Kernel (2x VRAM compression, lossless)
// Preserved exact implementation from RTX 4060 & RTX 3060
// =========================================================================
__global__ void attention_gqa_f16(
    const float* __restrict__ q,
    const half* __restrict__ k_cache,
    const half* __restrict__ v_cache,
    float* __restrict__ attn_out,
    float* __restrict__ scores_buf,
    int n_heads_q,
    int n_heads_kv,
    int head_dim,
    int max_seq_len,
    int pos,
    float attn_scale
) {
    int h = blockIdx.x;
    if (h >= n_heads_q) return;

    int tid = threadIdx.x; // 0..127
    int group_size = n_heads_q / n_heads_kv;
    int h_kv = h / group_size;

    float q_d = q[h * head_dim + tid];

    __shared__ float s_warp_sum[4];

    // Online softmax tracking in registers (FlashAttention-2)
    float m = -1e30f;
    float l = 0.0f;
    float acc = 0.0f;

    for (int t = 0; t <= pos; t++) {
        size_t kv_offset = ((size_t)h_kv * max_seq_len + t) * head_dim + tid;
        float k_d = __half2float(k_cache[kv_offset]);
        float v_d = __half2float(v_cache[kv_offset]);

        float prod = q_d * k_d;
        prod = warp_reduce_sum(prod);
        int warp_id = tid / WARP_SIZE;
        int lane_id = tid % WARP_SIZE;

        if (lane_id == 0) {
            s_warp_sum[warp_id] = prod;
        }
        __syncthreads();

        float total_dot = s_warp_sum[0] + s_warp_sum[1] + s_warp_sum[2] + s_warp_sum[3];
        float s_t = total_dot * attn_scale;

        float m_new = fmaxf(m, s_t);
        float alpha = expf(m - m_new);
        float w_t = expf(s_t - m_new);

        acc = acc * alpha + w_t * v_d;
        l = l * alpha + w_t;
        m = m_new;

        __syncthreads();
    }

    attn_out[h * head_dim + tid] = (l > 0.0f) ? (acc / l) : 0.0f;
}

// =========================================================================
// 9f. Native FP8 (e4m3) GQA Attention Kernel (4x VRAM compression, Ada sm_89 silicon)
// =========================================================================
__global__ void attention_gqa_fp8(
    const float* __restrict__ q,
    const __nv_fp8_e4m3* __restrict__ k_cache,
    const __nv_fp8_e4m3* __restrict__ v_cache,
    float* __restrict__ attn_out,
    float* __restrict__ scores_buf,
    int n_heads_q,
    int n_heads_kv,
    int head_dim,
    int max_seq_len,
    int pos,
    float attn_scale
) {
    int h = blockIdx.x;
    if (h >= n_heads_q) return;

    int tid = threadIdx.x; // 0..127
    int group_size = n_heads_q / n_heads_kv;
    int h_kv = h / group_size;

    float q_d = q[h * head_dim + tid];

    __shared__ float s_warp_sum[4];

    // Online softmax tracking in registers (FlashAttention-2)
    float m = -1e30f;
    float l = 0.0f;
    float acc = 0.0f;

    for (int t = 0; t <= pos; t++) {
        size_t kv_offset = ((size_t)h_kv * max_seq_len + t) * head_dim + tid;
        float k_d = float(k_cache[kv_offset]);
        float v_d = float(v_cache[kv_offset]);

        float prod = q_d * k_d;
        prod = warp_reduce_sum(prod);
        int warp_id = tid / WARP_SIZE;
        int lane_id = tid % WARP_SIZE;

        if (lane_id == 0) {
            s_warp_sum[warp_id] = prod;
        }
        __syncthreads();

        float total_dot = s_warp_sum[0] + s_warp_sum[1] + s_warp_sum[2] + s_warp_sum[3];
        float s_t = total_dot * attn_scale;

        float m_new = fmaxf(m, s_t);
        float alpha = expf(m - m_new);
        float w_t = expf(s_t - m_new);

        acc = acc * alpha + w_t * v_d;
        l = l * alpha + w_t;
        m = m_new;

        __syncthreads();
    }

    attn_out[h * head_dim + tid] = (l > 0.0f) ? (acc / l) : 0.0f;
}

// =========================================================================
// 9b. Batched GQA Attention Kernel
// 2D grid: blockIdx.x = h (0..n_heads_q - 1), blockIdx.y = b (0..batch_size - 1)
// 1 block of 128 threads per Q head per token
// threadIdx.x = d (0..127)
// Computes attention for all tokens in the batch chunk in a single kernel launch!
// =========================================================================
__global__ void attention_gqa_batch(
    const float* __restrict__ q_batch,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ attn_out_batch,
    float* __restrict__ scores_buf,
    int n_heads_q,
    int n_heads_kv,
    int head_dim,
    int max_seq_len,
    int chunk_start_pos,
    int batch_size,
    float attn_scale
) {
    int h = blockIdx.x;
    int b = blockIdx.y;
    if (h >= n_heads_q || b >= batch_size) return;

    int pos = chunk_start_pos + b;
    int tid = threadIdx.x; // 0..127
    int group_size = n_heads_q / n_heads_kv;
    int h_kv = h / group_size;

    size_t q_offset = (size_t)b * n_heads_q * head_dim + h * head_dim + tid;
    float q_d = q_batch[q_offset];

    __shared__ float s_warp_sum[4];

    // Online softmax tracking in registers (FlashAttention-2)
    float m = -1e30f;
    float l = 0.0f;
    float acc = 0.0f;

    for (int t = 0; t <= pos; t++) {
        size_t kv_offset = ((size_t)h_kv * max_seq_len + t) * head_dim + tid;
        float k_d = k_cache[kv_offset];
        float v_d = v_cache[kv_offset];

        float prod = q_d * k_d;
        prod = warp_reduce_sum(prod);
        int warp_id = tid / WARP_SIZE;
        int lane_id = tid % WARP_SIZE;

        if (lane_id == 0) {
            s_warp_sum[warp_id] = prod;
        }
        __syncthreads();

        float total_dot = s_warp_sum[0] + s_warp_sum[1] + s_warp_sum[2] + s_warp_sum[3];
        float s_t = total_dot * attn_scale;

        float m_new = fmaxf(m, s_t);
        float alpha = expf(m - m_new);
        float w_t = expf(s_t - m_new);

        acc = acc * alpha + w_t * v_d;
        l = l * alpha + w_t;
        m = m_new;

        __syncthreads();
    }

    size_t out_offset = (size_t)b * n_heads_q * head_dim + h * head_dim + tid;
    attn_out_batch[out_offset] = (l > 0.0f) ? (acc / l) : 0.0f;
}

// =========================================================================
// 9g. Batched FP16 GQA Attention Kernel
// =========================================================================
__global__ void attention_gqa_batch_f16(
    const float* __restrict__ q_batch,
    const half* __restrict__ k_cache,
    const half* __restrict__ v_cache,
    float* __restrict__ attn_out_batch,
    float* __restrict__ scores_buf,
    int n_heads_q,
    int n_heads_kv,
    int head_dim,
    int max_seq_len,
    int chunk_start_pos,
    int batch_size,
    float attn_scale
) {
    int h = blockIdx.x;
    int b = blockIdx.y;
    if (h >= n_heads_q || b >= batch_size) return;

    int pos = chunk_start_pos + b;
    int tid = threadIdx.x; // 0..127
    int group_size = n_heads_q / n_heads_kv;
    int h_kv = h / group_size;

    size_t q_offset = (size_t)b * n_heads_q * head_dim + h * head_dim + tid;
    float q_d = q_batch[q_offset];

    __shared__ float s_warp_sum[4];

    // Online softmax tracking in registers (FlashAttention-2)
    float m = -1e30f;
    float l = 0.0f;
    float acc = 0.0f;

    for (int t = 0; t <= pos; t++) {
        size_t kv_offset = ((size_t)h_kv * max_seq_len + t) * head_dim + tid;
        float k_d = __half2float(k_cache[kv_offset]);
        float v_d = __half2float(v_cache[kv_offset]);

        float prod = q_d * k_d;
        prod = warp_reduce_sum(prod);
        int warp_id = tid / WARP_SIZE;
        int lane_id = tid % WARP_SIZE;

        if (lane_id == 0) {
            s_warp_sum[warp_id] = prod;
        }
        __syncthreads();

        float total_dot = s_warp_sum[0] + s_warp_sum[1] + s_warp_sum[2] + s_warp_sum[3];
        float s_t = total_dot * attn_scale;

        float m_new = fmaxf(m, s_t);
        float alpha = expf(m - m_new);
        float w_t = expf(s_t - m_new);

        acc = acc * alpha + w_t * v_d;
        l = l * alpha + w_t;
        m = m_new;

        __syncthreads();
    }

    size_t out_offset = (size_t)b * n_heads_q * head_dim + h * head_dim + tid;
    attn_out_batch[out_offset] = (l > 0.0f) ? (acc / l) : 0.0f;
}

// =========================================================================
// 9h. Batched Native FP8 (e4m3) GQA Attention Kernel
// =========================================================================
__global__ void attention_gqa_batch_fp8(
    const float* __restrict__ q_batch,
    const __nv_fp8_e4m3* __restrict__ k_cache,
    const __nv_fp8_e4m3* __restrict__ v_cache,
    float* __restrict__ attn_out_batch,
    float* __restrict__ scores_buf,
    int n_heads_q,
    int n_heads_kv,
    int head_dim,
    int max_seq_len,
    int chunk_start_pos,
    int batch_size,
    float attn_scale
) {
    int h = blockIdx.x;
    int b = blockIdx.y;
    if (h >= n_heads_q || b >= batch_size) return;

    int pos = chunk_start_pos + b;
    int tid = threadIdx.x; // 0..127
    int group_size = n_heads_q / n_heads_kv;
    int h_kv = h / group_size;

    size_t q_offset = (size_t)b * n_heads_q * head_dim + h * head_dim + tid;
    float q_d = q_batch[q_offset];

    __shared__ float s_warp_sum[4];

    // Online softmax tracking in registers (FlashAttention-2)
    float m = -1e30f;
    float l = 0.0f;
    float acc = 0.0f;

    for (int t = 0; t <= pos; t++) {
        size_t kv_offset = ((size_t)h_kv * max_seq_len + t) * head_dim + tid;
        float k_d = float(k_cache[kv_offset]);
        float v_d = float(v_cache[kv_offset]);

        float prod = q_d * k_d;
        prod = warp_reduce_sum(prod);
        int warp_id = tid / WARP_SIZE;
        int lane_id = tid % WARP_SIZE;

        if (lane_id == 0) {
            s_warp_sum[warp_id] = prod;
        }
        __syncthreads();

        float total_dot = s_warp_sum[0] + s_warp_sum[1] + s_warp_sum[2] + s_warp_sum[3];
        float s_t = total_dot * attn_scale;

        float m_new = fmaxf(m, s_t);
        float alpha = expf(m - m_new);
        float w_t = expf(s_t - m_new);

        acc = acc * alpha + w_t * v_d;
        l = l * alpha + w_t;
        m = m_new;

        __syncthreads();
    }

    size_t out_offset = (size_t)b * n_heads_q * head_dim + h * head_dim + tid;
    attn_out_batch[out_offset] = (l > 0.0f) ? (acc / l) : 0.0f;
}

// =========================================================================
// 9i. Training Causal GQA Attention Forward Kernel
// Token-major layout:
//   q: [seq_len, n_heads_q, head_dim]
//   k: [seq_len, n_heads_kv, head_dim]
//   v: [seq_len, n_heads_kv, head_dim]
//   attn_out: [seq_len, n_heads_q, head_dim]
//   probs: optional [n_heads_q, seq_len, seq_len]
// Grid: blockIdx.x = h (0..n_heads_q - 1), blockIdx.y = i (0..seq_len - 1)
// Block: 128 threads (threadIdx.x = d, 0..head_dim - 1)
// =========================================================================
__global__ void attention_causal_gqa_train_fwd(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ attn_out,
    float* __restrict__ probs,
    int seq_len,
    int n_heads_q,
    int n_heads_kv,
    int head_dim,
    float attn_scale
) {
    int h = blockIdx.x;
    int i = blockIdx.y;
    if (h >= n_heads_q || i >= seq_len) return;

    int tid = threadIdx.x; // 0..head_dim - 1
    int group_size = n_heads_q / n_heads_kv;
    int h_kv = h / group_size;

    size_t q_offset = ((size_t)i * n_heads_q + h) * head_dim + tid;
    float q_d = q[q_offset];

    extern __shared__ float s_mem[];
    float* s_warp_sum = s_mem; // 4 floats
    float* s_scores = s_mem + 4; // seq_len floats if probs != nullptr

    // Online softmax tracking in registers (FlashAttention-2)
    float m = -1e30f;
    float l = 0.0f;
    float acc = 0.0f;

    for (int j = 0; j <= i; j++) {
        size_t kv_offset = ((size_t)j * n_heads_kv + h_kv) * head_dim + tid;
        float k_d = k[kv_offset];
        float v_d = v[kv_offset];

        float prod = q_d * k_d;
        prod = warp_reduce_sum(prod);
        int warp_id = tid / WARP_SIZE;
        int lane_id = tid % WARP_SIZE;

        if (lane_id == 0) {
            s_warp_sum[warp_id] = prod;
        }
        __syncthreads();

        float total_dot = s_warp_sum[0] + s_warp_sum[1] + s_warp_sum[2] + s_warp_sum[3];
        float s_t = total_dot * attn_scale;

        if (probs != nullptr && tid == 0) {
            s_scores[j] = s_t;
        }

        float m_new = fmaxf(m, s_t);
        float alpha = expf(m - m_new);
        float w_t = expf(s_t - m_new);

        acc = acc * alpha + w_t * v_d;
        l = l * alpha + w_t;
        m = m_new;

        __syncthreads();
    }

    size_t out_offset = ((size_t)i * n_heads_q + h) * head_dim + tid;
    attn_out[out_offset] = (l > 0.0f) ? (acc / l) : 0.0f;

    if (probs != nullptr) {
        float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
        for (int j = tid; j <= i; j += blockDim.x) {
            size_t p_offset = ((size_t)h * seq_len + i) * seq_len + j;
            probs[p_offset] = expf(s_scores[j] - m) * inv_l;
        }
        for (int j = i + 1 + tid; j < seq_len; j += blockDim.x) {
            size_t p_offset = ((size_t)h * seq_len + i) * seq_len + j;
            probs[p_offset] = 0.0f;
        }
    }
}

// =========================================================================
// 9j. Training Causal GQA Attention Backward Kernel: dQ and dS
// Grid: blockIdx.x = h (0..n_heads_q - 1), blockIdx.y = i (0..seq_len - 1)
// Block: 128 threads (threadIdx.x = tid, 0..127)
// =========================================================================
__global__ void attention_causal_gqa_train_bwd_dq_ds(
    const float* __restrict__ d_out,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ probs,
    float* __restrict__ dq,
    float* __restrict__ ds_out,
    int seq_len,
    int n_heads_q,
    int n_heads_kv,
    int head_dim,
    float attn_scale
) {
    int h = blockIdx.x;
    int i = blockIdx.y;
    if (h >= n_heads_q || i >= seq_len) return;

    int tid = threadIdx.x;
    int group_size = n_heads_q / n_heads_kv;
    int h_kv = h / group_size;

    size_t out_offset = ((size_t)i * n_heads_q + h) * head_dim + tid;
    float dout_d = d_out[out_offset];

    extern __shared__ float s_bwd_mem[];
    float* s_warp_sum = s_bwd_mem; // 4 floats
    float* s_dp = s_bwd_mem + 4;   // seq_len floats
    __shared__ float s_dotPdP;

    // 1. Compute dP_{ij} = sum_d dOut_{id} * V_{jd} for j <= i
    for (int j = 0; j <= i; j++) {
        size_t kv_offset = ((size_t)j * n_heads_kv + h_kv) * head_dim + tid;
        float v_d = v[kv_offset];

        float prod = dout_d * v_d;
        prod = warp_reduce_sum(prod);
        int warp_id = tid / WARP_SIZE;
        int lane_id = tid % WARP_SIZE;

        if (lane_id == 0) s_warp_sum[warp_id] = prod;
        __syncthreads();

        if (tid == 0) {
            s_dp[j] = s_warp_sum[0] + s_warp_sum[1] + s_warp_sum[2] + s_warp_sum[3];
        }
        __syncthreads();
    }

    // 2. Compute dotPdP = sum_{j<=i} P_{ij} * dP_{ij}
    float my_dot = 0.0f;
    for (int j = tid; j <= i; j += blockDim.x) {
        size_t p_offset = ((size_t)h * seq_len + i) * seq_len + j;
        my_dot += probs[p_offset] * s_dp[j];
    }
    my_dot = warp_reduce_sum(my_dot);
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    if (lane_id == 0) s_warp_sum[warp_id] = my_dot;
    __syncthreads();

    if (tid == 0) {
        s_dotPdP = s_warp_sum[0] + s_warp_sum[1] + s_warp_sum[2] + s_warp_sum[3];
    }
    __syncthreads();

    float dotPdP = s_dotPdP;

    // 3. Compute dS_{ij} and accumulate into dQ_{id} = sum_{j<=i} dS_{ij} * K_{jd}
    float acc_dq = 0.0f;
    for (int j = 0; j <= i; j++) {
        size_t p_offset = ((size_t)h * seq_len + i) * seq_len + j;
        float p_val = probs[p_offset];
        float ds_val = p_val * (s_dp[j] - dotPdP) * attn_scale;

        if (tid == 0) {
            ds_out[p_offset] = ds_val;
        }

        size_t kv_offset = ((size_t)j * n_heads_kv + h_kv) * head_dim + tid;
        float k_d = k[kv_offset];
        acc_dq += ds_val * k_d;
    }

    dq[out_offset] = acc_dq;
}

// =========================================================================
// 9k. Training Causal GQA Attention Backward Kernel: dK and dV
// Grid: blockIdx.x = kvH (0..n_heads_kv - 1), blockIdx.y = j (0..seq_len - 1)
// Block: 128 threads (threadIdx.x = tid, 0..127)
// =========================================================================
__global__ void attention_causal_gqa_train_bwd_dk_dv(
    const float* __restrict__ q,
    const float* __restrict__ d_out,
    const float* __restrict__ probs,
    const float* __restrict__ ds_buf,
    float* __restrict__ dk,
    float* __restrict__ dv,
    int seq_len,
    int n_heads_q,
    int n_heads_kv,
    int head_dim
) {
    int kv_h = blockIdx.x;
    int j = blockIdx.y;
    if (kv_h >= n_heads_kv || j >= seq_len) return;

    int tid = threadIdx.x;
    int group_size = n_heads_q / n_heads_kv;
    int start_h = kv_h * group_size;
    int end_h = start_h + group_size;

    float acc_dk = 0.0f;
    float acc_dv = 0.0f;

    for (int h = start_h; h < end_h; h++) {
        for (int i = j; i < seq_len; i++) {
            size_t p_offset = ((size_t)h * seq_len + i) * seq_len + j;
            float ds_val = ds_buf[p_offset];
            float p_val = probs[p_offset];

            size_t q_offset = ((size_t)i * n_heads_q + h) * head_dim + tid;
            acc_dk += ds_val * q[q_offset];
            acc_dv += p_val * d_out[q_offset];
        }
    }

    size_t kv_offset = ((size_t)j * n_heads_kv + kv_h) * head_dim + tid;
    dk[kv_offset] = acc_dk;
    dv[kv_offset] = acc_dv;
}

} // extern "C"

