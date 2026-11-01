// Glacier.Inference High-Performance CUDA Kernels
// Multi-head & Grouped-Query FlashAttention-2 Kernels (Single-Token, Batched, & Paged)
// Features online register-tracked softmax, fused QK dot-product + scaling + softmax + V reduction
#pragma once

#include "common.cuh"

#define MAX_ATTN_CHUNKS 16

extern "C" {

// =========================================================================
// 9. GQA Attention Kernel (FP32)
// 1 block per Q head (blockIdx.x = h, 0..n_heads_q - 1)
// Supports arbitrary head_dim (e.g. 64, 80, 96, 128, 256, 512) and warp-aligned blockDim.x
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
    if (h >= n_heads_q || pos < 0 || pos >= max_seq_len) return;

    int tid = threadIdx.x;
    int group_size = n_heads_q / n_heads_kv;
    int h_kv = h / group_size;

    float q_reg[MAX_ATTN_CHUNKS];
    float acc[MAX_ATTN_CHUNKS];
    #pragma unroll
    for (int c = 0; c < MAX_ATTN_CHUNKS; c++) {
        int d = tid + c * blockDim.x;
        q_reg[c] = (d < head_dim) ? q[(size_t)h * head_dim + d] : 0.0f;
        acc[c] = 0.0f;
    }

    float m = -1e30f;
    float l = 0.0f;

    for (int t = 0; t <= pos; t++) {
        size_t kv_base = ((size_t)h_kv * max_seq_len + t) * head_dim;
        float dot_part = 0.0f;
        for (int d = tid, c = 0; d < head_dim && c < MAX_ATTN_CHUNKS; d += blockDim.x, c++) {
            dot_part += q_reg[c] * k_cache[kv_base + d];
        }

        float s_t = block_reduce_sum(dot_part) * attn_scale;

        float m_new = fmaxf(m, s_t);
        float alpha = expf(m - m_new);
        float w_t = expf(s_t - m_new);

        for (int d = tid, c = 0; d < head_dim && c < MAX_ATTN_CHUNKS; d += blockDim.x, c++) {
            acc[c] = acc[c] * alpha + w_t * v_cache[kv_base + d];
        }
        l = l * alpha + w_t;
        m = m_new;
    }

    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (int d = tid, c = 0; d < head_dim && c < MAX_ATTN_CHUNKS; d += blockDim.x, c++) {
        attn_out[(size_t)h * head_dim + d] = acc[c] * inv_l;
    }
}

// =========================================================================
// 9e. FP16 GQA Attention Kernel (2x VRAM compression, lossless)
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
    if (h >= n_heads_q || pos < 0 || pos >= max_seq_len) return;

    int tid = threadIdx.x;
    int group_size = n_heads_q / n_heads_kv;
    int h_kv = h / group_size;

    float q_reg[MAX_ATTN_CHUNKS];
    float acc[MAX_ATTN_CHUNKS];
    #pragma unroll
    for (int c = 0; c < MAX_ATTN_CHUNKS; c++) {
        int d = tid + c * blockDim.x;
        q_reg[c] = (d < head_dim) ? q[(size_t)h * head_dim + d] : 0.0f;
        acc[c] = 0.0f;
    }

    float m = -1e30f;
    float l = 0.0f;

    for (int t = 0; t <= pos; t++) {
        size_t kv_base = ((size_t)h_kv * max_seq_len + t) * head_dim;
        float dot_part = 0.0f;
        for (int d = tid, c = 0; d < head_dim && c < MAX_ATTN_CHUNKS; d += blockDim.x, c++) {
            dot_part += q_reg[c] * __half2float(k_cache[kv_base + d]);
        }

        float s_t = block_reduce_sum(dot_part) * attn_scale;

        float m_new = fmaxf(m, s_t);
        float alpha = expf(m - m_new);
        float w_t = expf(s_t - m_new);

        for (int d = tid, c = 0; d < head_dim && c < MAX_ATTN_CHUNKS; d += blockDim.x, c++) {
            acc[c] = acc[c] * alpha + w_t * __half2float(v_cache[kv_base + d]);
        }
        l = l * alpha + w_t;
        m = m_new;
    }

    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (int d = tid, c = 0; d < head_dim && c < MAX_ATTN_CHUNKS; d += blockDim.x, c++) {
        attn_out[(size_t)h * head_dim + d] = acc[c] * inv_l;
    }
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
    if (h >= n_heads_q || pos < 0 || pos >= max_seq_len) return;

    int tid = threadIdx.x;
    int group_size = n_heads_q / n_heads_kv;
    int h_kv = h / group_size;

    float q_reg[MAX_ATTN_CHUNKS];
    float acc[MAX_ATTN_CHUNKS];
    #pragma unroll
    for (int c = 0; c < MAX_ATTN_CHUNKS; c++) {
        int d = tid + c * blockDim.x;
        q_reg[c] = (d < head_dim) ? q[(size_t)h * head_dim + d] : 0.0f;
        acc[c] = 0.0f;
    }

    float m = -1e30f;
    float l = 0.0f;

    for (int t = 0; t <= pos; t++) {
        size_t kv_base = ((size_t)h_kv * max_seq_len + t) * head_dim;
        float dot_part = 0.0f;
        for (int d = tid, c = 0; d < head_dim && c < MAX_ATTN_CHUNKS; d += blockDim.x, c++) {
            dot_part += q_reg[c] * float(k_cache[kv_base + d]);
        }

        float s_t = block_reduce_sum(dot_part) * attn_scale;

        float m_new = fmaxf(m, s_t);
        float alpha = expf(m - m_new);
        float w_t = expf(s_t - m_new);

        for (int d = tid, c = 0; d < head_dim && c < MAX_ATTN_CHUNKS; d += blockDim.x, c++) {
            acc[c] = acc[c] * alpha + w_t * float(v_cache[kv_base + d]);
        }
        l = l * alpha + w_t;
        m = m_new;
    }

    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (int d = tid, c = 0; d < head_dim && c < MAX_ATTN_CHUNKS; d += blockDim.x, c++) {
        attn_out[(size_t)h * head_dim + d] = acc[c] * inv_l;
    }
}

// =========================================================================
// 9b. Batched GQA Attention Kernel
// 2D grid: blockIdx.x = h (0..n_heads_q - 1), blockIdx.y = b (0..batch_size - 1)
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
    if (pos < 0 || pos >= max_seq_len) return;

    int tid = threadIdx.x;
    int group_size = n_heads_q / n_heads_kv;
    int h_kv = h / group_size;

    size_t q_base = ((size_t)b * n_heads_q + h) * head_dim;
    float q_reg[MAX_ATTN_CHUNKS];
    float acc[MAX_ATTN_CHUNKS];
    #pragma unroll
    for (int c = 0; c < MAX_ATTN_CHUNKS; c++) {
        int d = tid + c * blockDim.x;
        q_reg[c] = (d < head_dim) ? q_batch[q_base + d] : 0.0f;
        acc[c] = 0.0f;
    }

    float m = -1e30f;
    float l = 0.0f;

    for (int t = 0; t <= pos; t++) {
        size_t kv_base = ((size_t)h_kv * max_seq_len + t) * head_dim;
        float dot_part = 0.0f;
        for (int d = tid, c = 0; d < head_dim && c < MAX_ATTN_CHUNKS; d += blockDim.x, c++) {
            dot_part += q_reg[c] * k_cache[kv_base + d];
        }

        float s_t = block_reduce_sum(dot_part) * attn_scale;

        float m_new = fmaxf(m, s_t);
        float alpha = expf(m - m_new);
        float w_t = expf(s_t - m_new);

        for (int d = tid, c = 0; d < head_dim && c < MAX_ATTN_CHUNKS; d += blockDim.x, c++) {
            acc[c] = acc[c] * alpha + w_t * v_cache[kv_base + d];
        }
        l = l * alpha + w_t;
        m = m_new;
    }

    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (int d = tid, c = 0; d < head_dim && c < MAX_ATTN_CHUNKS; d += blockDim.x, c++) {
        attn_out_batch[q_base + d] = acc[c] * inv_l;
    }
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
    if (pos < 0 || pos >= max_seq_len) return;

    int tid = threadIdx.x;
    int group_size = n_heads_q / n_heads_kv;
    int h_kv = h / group_size;

    size_t q_base = ((size_t)b * n_heads_q + h) * head_dim;
    float q_reg[MAX_ATTN_CHUNKS];
    float acc[MAX_ATTN_CHUNKS];
    #pragma unroll
    for (int c = 0; c < MAX_ATTN_CHUNKS; c++) {
        int d = tid + c * blockDim.x;
        q_reg[c] = (d < head_dim) ? q_batch[q_base + d] : 0.0f;
        acc[c] = 0.0f;
    }

    float m = -1e30f;
    float l = 0.0f;

    for (int t = 0; t <= pos; t++) {
        size_t kv_base = ((size_t)h_kv * max_seq_len + t) * head_dim;
        float dot_part = 0.0f;
        for (int d = tid, c = 0; d < head_dim && c < MAX_ATTN_CHUNKS; d += blockDim.x, c++) {
            dot_part += q_reg[c] * __half2float(k_cache[kv_base + d]);
        }

        float s_t = block_reduce_sum(dot_part) * attn_scale;

        float m_new = fmaxf(m, s_t);
        float alpha = expf(m - m_new);
        float w_t = expf(s_t - m_new);

        for (int d = tid, c = 0; d < head_dim && c < MAX_ATTN_CHUNKS; d += blockDim.x, c++) {
            acc[c] = acc[c] * alpha + w_t * __half2float(v_cache[kv_base + d]);
        }
        l = l * alpha + w_t;
        m = m_new;
    }

    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (int d = tid, c = 0; d < head_dim && c < MAX_ATTN_CHUNKS; d += blockDim.x, c++) {
        attn_out_batch[q_base + d] = acc[c] * inv_l;
    }
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
    if (pos < 0 || pos >= max_seq_len) return;

    int tid = threadIdx.x;
    int group_size = n_heads_q / n_heads_kv;
    int h_kv = h / group_size;

    size_t q_base = ((size_t)b * n_heads_q + h) * head_dim;
    float q_reg[MAX_ATTN_CHUNKS];
    float acc[MAX_ATTN_CHUNKS];
    #pragma unroll
    for (int c = 0; c < MAX_ATTN_CHUNKS; c++) {
        int d = tid + c * blockDim.x;
        q_reg[c] = (d < head_dim) ? q_batch[q_base + d] : 0.0f;
        acc[c] = 0.0f;
    }

    float m = -1e30f;
    float l = 0.0f;

    for (int t = 0; t <= pos; t++) {
        size_t kv_base = ((size_t)h_kv * max_seq_len + t) * head_dim;
        float dot_part = 0.0f;
        for (int d = tid, c = 0; d < head_dim && c < MAX_ATTN_CHUNKS; d += blockDim.x, c++) {
            dot_part += q_reg[c] * float(k_cache[kv_base + d]);
        }

        float s_t = block_reduce_sum(dot_part) * attn_scale;

        float m_new = fmaxf(m, s_t);
        float alpha = expf(m - m_new);
        float w_t = expf(s_t - m_new);

        for (int d = tid, c = 0; d < head_dim && c < MAX_ATTN_CHUNKS; d += blockDim.x, c++) {
            acc[c] = acc[c] * alpha + w_t * float(v_cache[kv_base + d]);
        }
        l = l * alpha + w_t;
        m = m_new;
    }

    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (int d = tid, c = 0; d < head_dim && c < MAX_ATTN_CHUNKS; d += blockDim.x, c++) {
        attn_out_batch[q_base + d] = acc[c] * inv_l;
    }
}

// =========================================================================
// 9p. PagedAttention FP16 GQA Single-Token Kernel
// Pool layout: [num_blocks, num_layers, n_heads_kv, block_size, dim]
// =========================================================================
__global__ void paged_attention_gqa_f16_kernel(
    const float* __restrict__ q,
    const half* __restrict__ k_pool,
    const half* __restrict__ v_pool,
    float* __restrict__ attn_out,
    const int* __restrict__ block_table,
    int pos,
    int layer,
    int num_layers,
    int n_heads_q,
    int n_heads_kv,
    int head_dim,
    int v_head_dim,
    int block_size,
    float attn_scale
) {
    int h = blockIdx.x;
    if (h >= n_heads_q || pos < 0 || block_size <= 0) return;

    int tid = threadIdx.x;
    int group_size = n_heads_q / n_heads_kv;
    int h_kv = h / group_size;

    float q_reg[MAX_ATTN_CHUNKS];
    float acc[MAX_ATTN_CHUNKS];
    #pragma unroll
    for (int c = 0; c < MAX_ATTN_CHUNKS; c++) {
        int d = tid + c * blockDim.x;
        q_reg[c] = (d < head_dim) ? q[(size_t)h * head_dim + d] : 0.0f;
        acc[c] = 0.0f;
    }

    float m = -1e30f;
    float l = 0.0f;

    for (int t = 0; t <= pos; t++) {
        int logical_block = t / block_size;
        int offset_in_block = t % block_size;
        int phys_block = block_table[logical_block];
        if (phys_block < 0) continue;

        size_t head_slot = (((size_t)phys_block * num_layers + layer) * n_heads_kv + h_kv) * block_size + offset_in_block;
        size_t k_base = head_slot * head_dim;
        size_t v_base = head_slot * v_head_dim;

        float dot_part = 0.0f;
        for (int d = tid, c = 0; d < head_dim && c < MAX_ATTN_CHUNKS; d += blockDim.x, c++) {
            dot_part += q_reg[c] * __half2float(k_pool[k_base + d]);
        }

        float s_t = block_reduce_sum(dot_part) * attn_scale;

        float m_new = fmaxf(m, s_t);
        float alpha = expf(m - m_new);
        float w_t = expf(s_t - m_new);

        for (int d = tid, c = 0; d < v_head_dim && c < MAX_ATTN_CHUNKS; d += blockDim.x, c++) {
            acc[c] = acc[c] * alpha + w_t * __half2float(v_pool[v_base + d]);
        }
        l = l * alpha + w_t;
        m = m_new;
    }

    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (int d = tid, c = 0; d < v_head_dim && c < MAX_ATTN_CHUNKS; d += blockDim.x, c++) {
        attn_out[(size_t)h * v_head_dim + d] = acc[c] * inv_l;
    }
}

// =========================================================================
// 9q. PagedAttention FP16 GQA Batched Prefill Kernel
// 2D grid: blockIdx.x = h (0..n_heads_q - 1), blockIdx.y = b (0..batch_size - 1)
// =========================================================================
__global__ void paged_attention_gqa_batch_f16_kernel(
    const float* __restrict__ q_batch,
    const half* __restrict__ k_pool,
    const half* __restrict__ v_pool,
    float* __restrict__ attn_out_batch,
    const int* __restrict__ block_table,
    int start_pos,
    int batch_size,
    int layer,
    int num_layers,
    int n_heads_q,
    int n_heads_kv,
    int head_dim,
    int v_head_dim,
    int block_size,
    float attn_scale
) {
    int h = blockIdx.x;
    int b = blockIdx.y;
    if (h >= n_heads_q || b >= batch_size || block_size <= 0) return;

    int pos = start_pos + b;
    if (pos < 0) return;

    int tid = threadIdx.x;
    int group_size = n_heads_q / n_heads_kv;
    int h_kv = h / group_size;

    size_t q_base = ((size_t)b * n_heads_q + h) * head_dim;
    size_t out_base = ((size_t)b * n_heads_q + h) * v_head_dim;

    float q_reg[MAX_ATTN_CHUNKS];
    float acc[MAX_ATTN_CHUNKS];
    #pragma unroll
    for (int c = 0; c < MAX_ATTN_CHUNKS; c++) {
        int d = tid + c * blockDim.x;
        q_reg[c] = (d < head_dim) ? q_batch[q_base + d] : 0.0f;
        acc[c] = 0.0f;
    }

    float m = -1e30f;
    float l = 0.0f;

    for (int t = 0; t <= pos; t++) {
        int logical_block = t / block_size;
        int offset_in_block = t % block_size;
        int phys_block = block_table[logical_block];
        if (phys_block < 0) continue;

        size_t head_slot = (((size_t)phys_block * num_layers + layer) * n_heads_kv + h_kv) * block_size + offset_in_block;
        size_t k_base = head_slot * head_dim;
        size_t v_base = head_slot * v_head_dim;

        float dot_part = 0.0f;
        for (int d = tid, c = 0; d < head_dim && c < MAX_ATTN_CHUNKS; d += blockDim.x, c++) {
            dot_part += q_reg[c] * __half2float(k_pool[k_base + d]);
        }

        float s_t = block_reduce_sum(dot_part) * attn_scale;

        float m_new = fmaxf(m, s_t);
        float alpha = expf(m - m_new);
        float w_t = expf(s_t - m_new);

        for (int d = tid, c = 0; d < v_head_dim && c < MAX_ATTN_CHUNKS; d += blockDim.x, c++) {
            acc[c] = acc[c] * alpha + w_t * __half2float(v_pool[v_base + d]);
        }
        l = l * alpha + w_t;
        m = m_new;
    }

    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (int d = tid, c = 0; d < v_head_dim && c < MAX_ATTN_CHUNKS; d += blockDim.x, c++) {
        attn_out_batch[out_base + d] = acc[c] * inv_l;
    }
}

// =========================================================================
// 9i. Training Causal GQA Attention Forward Kernel
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

    int tid = threadIdx.x;
    int group_size = n_heads_q / n_heads_kv;
    int h_kv = h / group_size;

    size_t q_base = ((size_t)i * n_heads_q + h) * head_dim;
    float q_reg[MAX_ATTN_CHUNKS];
    float acc[MAX_ATTN_CHUNKS];
    #pragma unroll
    for (int c = 0; c < MAX_ATTN_CHUNKS; c++) {
        int d = tid + c * blockDim.x;
        q_reg[c] = (d < head_dim) ? q[q_base + d] : 0.0f;
        acc[c] = 0.0f;
    }

    extern __shared__ float s_mem[];
    float* s_scores = s_mem + 4; // Preserves caller dynamic shared-memory offset (4 + seq_len floats)

    float m = -1e30f;
    float l = 0.0f;

    for (int j = 0; j <= i; j++) {
        size_t kv_base = ((size_t)j * n_heads_kv + h_kv) * head_dim;
        float dot_part = 0.0f;
        for (int d = tid, c = 0; d < head_dim && c < MAX_ATTN_CHUNKS; d += blockDim.x, c++) {
            dot_part += q_reg[c] * k[kv_base + d];
        }

        float s_t = block_reduce_sum(dot_part) * attn_scale;

        if (probs != nullptr && tid == 0) {
            s_scores[j] = s_t;
        }

        float m_new = fmaxf(m, s_t);
        float alpha = expf(m - m_new);
        float w_t = expf(s_t - m_new);

        for (int d = tid, c = 0; d < head_dim && c < MAX_ATTN_CHUNKS; d += blockDim.x, c++) {
            acc[c] = acc[c] * alpha + w_t * v[kv_base + d];
        }
        l = l * alpha + w_t;
        m = m_new;
    }

    __syncthreads();

    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (int d = tid, c = 0; d < head_dim && c < MAX_ATTN_CHUNKS; d += blockDim.x, c++) {
        attn_out[q_base + d] = acc[c] * inv_l;
    }

    if (probs != nullptr) {
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

    size_t out_base = ((size_t)i * n_heads_q + h) * head_dim;
    float dout_reg[MAX_ATTN_CHUNKS];
    float acc_dq[MAX_ATTN_CHUNKS];
    #pragma unroll
    for (int c = 0; c < MAX_ATTN_CHUNKS; c++) {
        int d = tid + c * blockDim.x;
        dout_reg[c] = (d < head_dim) ? d_out[out_base + d] : 0.0f;
        acc_dq[c] = 0.0f;
    }

    extern __shared__ float s_bwd_mem[];
    float* s_dp = s_bwd_mem + 4; // Preserves caller dynamic shared-memory offset

    // 1. Compute dP_{ij} = sum_d dOut_{id} * V_{jd} for j <= i
    for (int j = 0; j <= i; j++) {
        size_t kv_base = ((size_t)j * n_heads_kv + h_kv) * head_dim;
        float prod = 0.0f;
        for (int d = tid, c = 0; d < head_dim && c < MAX_ATTN_CHUNKS; d += blockDim.x, c++) {
            prod += dout_reg[c] * v[kv_base + d];
        }
        float dp_j = block_reduce_sum(prod);
        if (tid == 0) {
            s_dp[j] = dp_j;
        }
    }
    __syncthreads();

    // 2. Compute dotPdP = sum_{j<=i} P_{ij} * dP_{ij}
    float my_dot = 0.0f;
    for (int j = tid; j <= i; j += blockDim.x) {
        size_t p_offset = ((size_t)h * seq_len + i) * seq_len + j;
        my_dot += probs[p_offset] * s_dp[j];
    }
    float dotPdP = block_reduce_sum(my_dot);

    // 3. Compute dS_{ij} and accumulate into dQ_{id} = sum_{j<=i} dS_{ij} * K_{jd}
    for (int j = 0; j <= i; j++) {
        size_t p_offset = ((size_t)h * seq_len + i) * seq_len + j;
        float p_val = probs[p_offset];
        float ds_val = p_val * (s_dp[j] - dotPdP) * attn_scale;

        if (tid == 0) {
            ds_out[p_offset] = ds_val;
        }

        size_t kv_base = ((size_t)j * n_heads_kv + h_kv) * head_dim;
        for (int d = tid, c = 0; d < head_dim && c < MAX_ATTN_CHUNKS; d += blockDim.x, c++) {
            acc_dq[c] += ds_val * k[kv_base + d];
        }
    }

    for (int d = tid, c = 0; d < head_dim && c < MAX_ATTN_CHUNKS; d += blockDim.x, c++) {
        dq[out_base + d] = acc_dq[c];
    }
}

// =========================================================================
// 9k. Training Causal GQA Attention Backward Kernel: dK and dV
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

    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc_dk = 0.0f;
        float acc_dv = 0.0f;

        for (int h = start_h; h < end_h; h++) {
            for (int i = j; i < seq_len; i++) {
                size_t p_offset = ((size_t)h * seq_len + i) * seq_len + j;
                float ds_val = ds_buf[p_offset];
                float p_val = probs[p_offset];

                size_t q_offset = ((size_t)i * n_heads_q + h) * head_dim + d;
                acc_dk += ds_val * q[q_offset];
                acc_dv += p_val * d_out[q_offset];
            }
        }

        size_t kv_offset = ((size_t)j * n_heads_kv + kv_h) * head_dim + d;
        dk[kv_offset] = acc_dk;
        dv[kv_offset] = acc_dv;
    }
}

// =========================================================================
// 12. Bidirectional Full Multi-Head Self-Attention for Diffusion Transformers (FLUX.1 / SD3)
// =========================================================================
__global__ void attention_bidirectional_batch(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ attn_out,
    int n_heads,
    int head_dim,
    int num_tokens,
    float attn_scale
) {
    int h = blockIdx.x;
    int b = blockIdx.y;
    if (h >= n_heads || b >= num_tokens) return;

    int tid = threadIdx.x;
    size_t q_base = ((size_t)b * n_heads + h) * head_dim;

    float q_reg[MAX_ATTN_CHUNKS];
    float acc[MAX_ATTN_CHUNKS];
    #pragma unroll
    for (int c = 0; c < MAX_ATTN_CHUNKS; c++) {
        int d = tid + c * blockDim.x;
        q_reg[c] = (d < head_dim) ? q[q_base + d] : 0.0f;
        acc[c] = 0.0f;
    }

    float m = -1e30f;
    float l = 0.0f;

    for (int t = 0; t < num_tokens; t++) {
        size_t kv_base = ((size_t)t * n_heads + h) * head_dim;
        float dot_part = 0.0f;
        for (int d = tid, c = 0; d < head_dim && c < MAX_ATTN_CHUNKS; d += blockDim.x, c++) {
            dot_part += q_reg[c] * k[kv_base + d];
        }

        float s_t = block_reduce_sum(dot_part) * attn_scale;

        float m_new = fmaxf(m, s_t);
        float alpha = expf(m - m_new);
        float w_t = expf(s_t - m_new);

        for (int d = tid, c = 0; d < head_dim && c < MAX_ATTN_CHUNKS; d += blockDim.x, c++) {
            acc[c] = acc[c] * alpha + w_t * v[kv_base + d];
        }
        l = l * alpha + w_t;
        m = m_new;
    }

    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (int d = tid, c = 0; d < head_dim && c < MAX_ATTN_CHUNKS; d += blockDim.x, c++) {
        attn_out[q_base + d] = acc[c] * inv_l;
    }
}

// =========================================================================
// 13. Cross Multi-Head Attention for Wan 2.1 Video DiT
// =========================================================================
__global__ void attention_cross_batch(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ attn_out,
    int n_heads,
    int head_dim,
    int num_q_tokens,
    int num_kv_tokens,
    float attn_scale
) {
    int h = blockIdx.x;
    int b = blockIdx.y;
    if (h >= n_heads || b >= num_q_tokens) return;

    int tid = threadIdx.x;
    size_t q_base = ((size_t)b * n_heads + h) * head_dim;

    float q_reg[MAX_ATTN_CHUNKS];
    float acc[MAX_ATTN_CHUNKS];
    #pragma unroll
    for (int c = 0; c < MAX_ATTN_CHUNKS; c++) {
        int d = tid + c * blockDim.x;
        q_reg[c] = (d < head_dim) ? q[q_base + d] : 0.0f;
        acc[c] = 0.0f;
    }

    float m = -1e30f;
    float l = 0.0f;

    for (int t = 0; t < num_kv_tokens; t++) {
        size_t kv_base = ((size_t)t * n_heads + h) * head_dim;
        float dot_part = 0.0f;
        for (int d = tid, c = 0; d < head_dim && c < MAX_ATTN_CHUNKS; d += blockDim.x, c++) {
            dot_part += q_reg[c] * k[kv_base + d];
        }

        float s_t = block_reduce_sum(dot_part) * attn_scale;

        float m_new = fmaxf(m, s_t);
        float alpha = expf(m - m_new);
        float w_t = expf(s_t - m_new);

        for (int d = tid, c = 0; d < head_dim && c < MAX_ATTN_CHUNKS; d += blockDim.x, c++) {
            acc[c] = acc[c] * alpha + w_t * v[kv_base + d];
        }
        l = l * alpha + w_t;
        m = m_new;
    }

    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (int d = tid, c = 0; d < head_dim && c < MAX_ATTN_CHUNKS; d += blockDim.x, c++) {
        attn_out[q_base + d] = acc[c] * inv_l;
    }
}

// =========================================================================
// 14. 3D Rotary Position Embedding (3D-RoPE) In-Place in GPU VRAM
// =========================================================================
__global__ void rope_3d_in_vram(
    float* __restrict__ qk,
    const float* __restrict__ rope_cos,
    const float* __restrict__ rope_sin,
    int total_tokens,
    int n_heads,
    int head_dim
) {
    int t = blockIdx.x;
    int h = blockIdx.y;
    int p = threadIdx.x; // 0..63

    if (t >= total_tokens || h >= n_heads || p >= 64) return;

    float c = rope_cos[(size_t)t * 64 + p];
    float s = rope_sin[(size_t)t * 64 + p];

    float2* qk2 = (float2*)qk;
    size_t pair_idx = ((size_t)t * n_heads + h) * 64 + p;
    float2 v = qk2[pair_idx];

    float2 res;
    res.x = v.x * c - v.y * s;
    res.y = v.x * s + v.y * c;

    qk2[pair_idx] = res;
}

} // extern "C"
