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

} // extern "C"
