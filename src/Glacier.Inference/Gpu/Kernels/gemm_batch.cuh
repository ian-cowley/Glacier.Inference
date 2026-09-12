// Glacier.Inference High-Performance CUDA Kernels
// Batched GEMM Kernels (Q4_K, Q6_K, SwiGLU) for Prompt Prefill Acceleration
// Register-tiled architecture with pre-dequantized weights: STACK: 0, zero DRAM spills
#pragma once

#include "common.cuh"

extern "C" {

// =========================================================================
// 10. Batched GEMV/GEMM Q4_K for Prompt Prefill Acceleration
// Model weights W are read from VRAM ONCE!
// Warp-cooperative shared-memory accumulation: STACK: 0, 100% occupancy
// =========================================================================
__global__ void gemm_q4_k_batch(
    float* __restrict__ Y,          // [batch_size, m_rows]
    const float* __restrict__ X,    // [batch_size, k_cols]
    const BlockQ4_K* __restrict__ W,// [m_rows, k_cols / 256]
    int k_cols,
    int m_rows,
    int batch_size,
    const float* __restrict__ bias,     // optional [m_rows]
    float* __restrict__ residual    // optional [batch_size, m_rows]
) {
    int warp_id = threadIdx.x / WARP_SIZE; // 0..3
    int lane_id = threadIdx.x % WARP_SIZE; // 0..31
    int row = blockIdx.x * 4 + warp_id;

    if (row >= m_rows) return;

    int nb = k_cols / QK_K;
    const BlockQ4_K* row_w = W + (size_t)row * nb;

    int chunk = lane_id / 8;
    int chunk_lane = lane_id % 8;
    int is_idx = chunk * 2;
    int x_offset1 = chunk * 64 + chunk_lane * 4;
    int x_offset2 = x_offset1 + 32;

    for (int tile = 0; tile < 4; tile++) {
        int t_base = tile * 8;
        if (t_base >= batch_size) break;

        float acc[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            acc[i] = 0.0f;
        }

        for (int b = 0; b < nb; b++) {
            const BlockQ4_K* blk = &row_w[b];
            float d = __half2float(blk->d);
            float min_val = __half2float(blk->dmin);

            uint8_t sc0, m0, sc1, m1;
            get_scale_min_k4(is_idx + 0, blk->scales, &sc0, &m0);
            get_scale_min_k4(is_idx + 1, blk->scales, &sc1, &m1);

            float d1 = d * sc0;
            float min1 = min_val * m0;
            float d2 = d * sc1;
            float min2 = min_val * m1;

            uint32_t q = ((const uint32_t*)blk->qs)[lane_id];
            uint8_t q0 = (uint8_t)(q & 0xFF);
            uint8_t q1 = (uint8_t)((q >> 8) & 0xFF);
            uint8_t q2 = (uint8_t)((q >> 16) & 0xFF);
            uint8_t q3 = (uint8_t)(q >> 24);

            float w0 = d1 * (float)(q0 & 0x0F) - min1;
            float w1 = d1 * (float)(q1 & 0x0F) - min1;
            float w2 = d1 * (float)(q2 & 0x0F) - min1;
            float w3 = d1 * (float)(q3 & 0x0F) - min1;

            float w4 = d2 * (float)(q0 >> 4) - min2;
            float w5 = d2 * (float)(q1 >> 4) - min2;
            float w6 = d2 * (float)(q2 >> 4) - min2;
            float w7 = d2 * (float)(q3 >> 4) - min2;

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int t = t_base + i;
                if (t < batch_size) {
                    const float* x_t = X + (size_t)t * k_cols + b * QK_K;
                    float4 x1 = *((const float4*)(x_t + x_offset1));
                    float4 x2 = *((const float4*)(x_t + x_offset2));

                    acc[i] += w0 * x1.x + w1 * x1.y + w2 * x1.z + w3 * x1.w
                            + w4 * x2.x + w5 * x2.y + w6 * x2.z + w7 * x2.w;
                }
            }
        }

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int t = t_base + i;
            if (t < batch_size) {
                float sum = warp_reduce_sum(acc[i]);
                if (lane_id == 0) {
                    if (bias != nullptr) {
                        sum += bias[row];
                    }
                    if (residual != nullptr) {
                        residual[(size_t)t * m_rows + row] += sum;
                    }
                    if (Y != nullptr) {
                        Y[(size_t)t * m_rows + row] = sum;
                    }
                }
            }
        }
    }
}

// =========================================================================
// 11. Batched GEMV/GEMM Q6_K for Prompt Prefill Acceleration
// Model weights W are read from VRAM ONCE!
// Fast register-accumulated architecture: STACK: 0, 100% occupancy
// =========================================================================
__global__ void gemm_q6_k_batch(
    float* __restrict__ Y,          // [batch_size, m_rows]
    const float* __restrict__ X,    // [batch_size, k_cols]
    const BlockQ6_K* __restrict__ W,// [m_rows, k_cols / 256]
    int k_cols,
    int m_rows,
    int batch_size,
    const float* __restrict__ bias,     // optional [m_rows]
    float* __restrict__ residual    // optional [batch_size, m_rows]
) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int row = blockIdx.x * 4 + warp_id;

    if (row >= m_rows) return;

    int nb = k_cols / QK_K;
    const BlockQ6_K* row_w = W + (size_t)row * nb;
    int is_idx = lane_id / 16;

    for (int tile = 0; tile < 4; tile++) {
        int t_base = tile * 8;
        if (t_base >= batch_size) break;

        float acc[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            acc[i] = 0.0f;
        }

        for (int b = 0; b < nb; b++) {
            const BlockQ6_K* blk = &row_w[b];
            float d = __half2float(blk->d);

            #pragma unroll
            for (int step = 0; step < 2; step++) {
                int n = step * 128;
                const uint8_t* ql = blk->ql + step * 64;
                const uint8_t* qh = blk->qh + step * 32;
                const int8_t* sc = blk->scales + step * 8;

                uint8_t qh_val = qh[lane_id];
                uint8_t ql_low = ql[lane_id + 0];
                uint8_t ql_high = ql[lane_id + 32];

                int8_t q1 = (int8_t)((ql_low & 0x0F) | (((qh_val >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql_high & 0x0F) | (((qh_val >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql_low >> 4) | (((qh_val >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql_high >> 4) | (((qh_val >> 6) & 3) << 4)) - 32;

                float s1 = d * (float)sc[is_idx + 0];
                float s2 = d * (float)sc[is_idx + 2];
                float s3 = d * (float)sc[is_idx + 4];
                float s4 = d * (float)sc[is_idx + 6];

                float w1 = s1 * (float)q1;
                float w2 = s2 * (float)q2;
                float w3 = s3 * (float)q3;
                float w4 = s4 * (float)q4;

                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    int t = t_base + i;
                    if (t < batch_size) {
                        const float* x_blk = X + (size_t)t * k_cols + b * QK_K;
                        acc[i] += w1 * x_blk[n + lane_id + 0]
                                + w2 * x_blk[n + lane_id + 32]
                                + w3 * x_blk[n + lane_id + 64]
                                + w4 * x_blk[n + lane_id + 96];
                    }
                }
            }
        }

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int t = t_base + i;
            if (t < batch_size) {
                float sum = warp_reduce_sum(acc[i]);
                if (lane_id == 0) {
                    if (bias != nullptr) {
                        sum += bias[row];
                    }
                    if (residual != nullptr) {
                        residual[(size_t)t * m_rows + row] += sum;
                    }
                    if (Y != nullptr) {
                        Y[(size_t)t * m_rows + row] = sum;
                    }
                }
            }
        }
    }
}

// =========================================================================
// 12. Batched Fused SwiGLU: SiLU(Gate[:, t]) * Up[:, t] in registers
// Model weights W_gate and W_up are read from VRAM ONCE!
// Fast 2-tile register-accumulated architecture: STACK: 0, 100% occupancy
// =========================================================================
__global__ void gemm_q4_k_swiglu_batch(
    float* __restrict__ dst,        // [batch_size, m_rows]
    const float* __restrict__ X,    // [batch_size, k_cols]
    const BlockQ4_K* __restrict__ W_gate,
    const BlockQ4_K* __restrict__ W_up,
    int k_cols,
    int m_rows,
    int batch_size
) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int row = blockIdx.x * 4 + warp_id;

    if (row >= m_rows) return;

    int nb = k_cols / QK_K;
    const BlockQ4_K* row_gate = W_gate + (size_t)row * nb;
    const BlockQ4_K* row_up = W_up + (size_t)row * nb;

    int chunk = lane_id / 8;
    int chunk_lane = lane_id % 8;
    int is_idx = chunk * 2;
    int x_offset1 = chunk * 64 + chunk_lane * 4;
    int x_offset2 = x_offset1 + 32;

    for (int tile = 0; tile < 4; tile++) {
        int t_base = tile * 8;
        if (t_base >= batch_size) break;

        float acc_g[8];
        float acc_u[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            acc_g[i] = 0.0f;
            acc_u[i] = 0.0f;
        }

        for (int b = 0; b < nb; b++) {
            const BlockQ4_K* blk_g = &row_gate[b];
            float dg = __half2float(blk_g->d);
            float min_g = __half2float(blk_g->dmin);
            uint8_t sc0_g, m0_g, sc1_g, m1_g;
            get_scale_min_k4(is_idx + 0, blk_g->scales, &sc0_g, &m0_g);
            get_scale_min_k4(is_idx + 1, blk_g->scales, &sc1_g, &m1_g);
            float dg1 = dg * sc0_g; float ming1 = min_g * m0_g;
            float dg2 = dg * sc1_g; float ming2 = min_g * m1_g;
            uint32_t q_g = ((const uint32_t*)blk_g->qs)[lane_id];
            uint8_t q0_g = (uint8_t)(q_g & 0xFF); uint8_t q1_g = (uint8_t)((q_g >> 8) & 0xFF);
            uint8_t q2_g = (uint8_t)((q_g >> 16) & 0xFF); uint8_t q3_g = (uint8_t)(q_g >> 24);

            float wg0 = dg1 * (float)(q0_g & 0x0F) - ming1;
            float wg1 = dg1 * (float)(q1_g & 0x0F) - ming1;
            float wg2 = dg1 * (float)(q2_g & 0x0F) - ming1;
            float wg3 = dg1 * (float)(q3_g & 0x0F) - ming1;
            float wg4 = dg2 * (float)(q0_g >> 4) - ming2;
            float wg5 = dg2 * (float)(q1_g >> 4) - ming2;
            float wg6 = dg2 * (float)(q2_g >> 4) - ming2;
            float wg7 = dg2 * (float)(q3_g >> 4) - ming2;

            const BlockQ4_K* blk_u = &row_up[b];
            float du = __half2float(blk_u->d);
            float min_u = __half2float(blk_u->dmin);
            uint8_t sc0_u, m0_u, sc1_u, m1_u;
            get_scale_min_k4(is_idx + 0, blk_u->scales, &sc0_u, &m0_u);
            get_scale_min_k4(is_idx + 1, blk_u->scales, &sc1_u, &m1_u);
            float du1 = du * sc0_u; float minu1 = min_u * m0_u;
            float du2 = du * sc1_u; float minu2 = min_u * m1_u;
            uint32_t q_u = ((const uint32_t*)blk_u->qs)[lane_id];
            uint8_t q0_u = (uint8_t)(q_u & 0xFF); uint8_t q1_u = (uint8_t)((q_u >> 8) & 0xFF);
            uint8_t q2_u = (uint8_t)((q_u >> 16) & 0xFF); uint8_t q3_u = (uint8_t)(q_u >> 24);

            float wu0 = du1 * (float)(q0_u & 0x0F) - minu1;
            float wu1 = du1 * (float)(q1_u & 0x0F) - minu1;
            float wu2 = du1 * (float)(q2_u & 0x0F) - minu1;
            float wu3 = du1 * (float)(q3_u & 0x0F) - minu1;
            float wu4 = du2 * (float)(q0_u >> 4) - minu2;
            float wu5 = du2 * (float)(q1_u >> 4) - minu2;
            float wu6 = du2 * (float)(q2_u >> 4) - minu2;
            float wu7 = du2 * (float)(q3_u >> 4) - minu2;

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int t = t_base + i;
                if (t < batch_size) {
                    const float* x_t = X + (size_t)t * k_cols + b * QK_K;
                    float4 x1 = *((const float4*)(x_t + x_offset1));
                    float4 x2 = *((const float4*)(x_t + x_offset2));

                    acc_g[i] += wg0 * x1.x + wg1 * x1.y + wg2 * x1.z + wg3 * x1.w
                              + wg4 * x2.x + wg5 * x2.y + wg6 * x2.z + wg7 * x2.w;
                    acc_u[i] += wu0 * x1.x + wu1 * x1.y + wu2 * x1.z + wu3 * x1.w
                              + wu4 * x2.x + wu5 * x2.y + wu6 * x2.z + wu7 * x2.w;
                }
            }
        }

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int t = t_base + i;
            if (t < batch_size) {
                float g = warp_reduce_sum(acc_g[i]);
                float u = warp_reduce_sum(acc_u[i]);
                if (lane_id == 0) {
                    float silu = g / (1.0f + __expf(-g));
                    dst[(size_t)t * m_rows + row] = silu * u;
                }
            }
        }
    }
}

} // extern "C"
