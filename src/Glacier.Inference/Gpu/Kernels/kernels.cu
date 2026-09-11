// Glacier.Inference High-Performance CUDA Kernels
// Target: NVIDIA Ada Lovelace sm_89 (RTX 4060) & modern CUDA architectures
// Zero-dependency direct CUBIN compilation via nvcc

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define QK_K 256
#define WARP_SIZE 32

// Block layouts matching C# structs
struct __align__(4) BlockQ4_K {
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

struct __align__(2) BlockQ6_K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    half d;
};

// Unpack scales & mins for Q4_K
__device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// Warp reduction helper using hardware shuffles
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

extern "C" {

// =========================================================================
// 1. GEMV Q4_K: Matrix-Vector Multiplication (y = W * x)
// Each warp of 32 threads computes 1 row of output y
// =========================================================================
__global__ void gemv_q4_k(
    float* __restrict__ y,
    const float* __restrict__ x,
    const BlockQ4_K* __restrict__ W,
    int k_cols,
    int m_rows
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE; // 0..31

    if (warp_id >= m_rows) return;

    int nb = k_cols / QK_K;
    const BlockQ4_K* row_w = W + (size_t)warp_id * nb;
    float row_sum = 0.0f;

    for (int b = 0; b < nb; b++) {
        const BlockQ4_K* blk = &row_w[b];
        float d = __half2float(blk->d);
        float min_val = __half2float(blk->dmin);
        const float* x_blk = x + b * QK_K;

        int is_idx = 0;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc0, m0, sc1, m1;
            get_scale_min_k4(is_idx + 0, blk->scales, &sc0, &m0);
            get_scale_min_k4(is_idx + 1, blk->scales, &sc1, &m1);

            float d1 = d * sc0;
            float min1 = min_val * m0;
            float d2 = d * sc1;
            float min2 = min_val * m1;

            // Lane_id (0..31) loads exactly 1 byte from qs (containing two 4-bit nibbles)
            uint8_t q_byte = blk->qs[(j / 2) + lane_id];
            float val_x1 = x_blk[j + lane_id];
            float val_x2 = x_blk[j + 32 + lane_id];

            float w1 = d1 * (q_byte & 0x0F) - min1;
            float w2 = d2 * (q_byte >> 4) - min2;

            row_sum += (w1 * val_x1) + (w2 * val_x2);

            is_idx += 2;
        }
    }

    // Warp-level reduction
    row_sum = warp_reduce_sum(row_sum);

    if (lane_id == 0) {
        y[warp_id] = row_sum;
    }
}

// =========================================================================
// 1b. GEMV Q4_K Fast: Vectorized 128-byte coalesced loads + float4 + factored scales
// Each warp of 32 threads computes 1 row of output y
// Processes all 256 weights of BlockQ4_K in a single step with zero chunk loops
// =========================================================================
__global__ void gemv_q4_k_fast(
    float* __restrict__ y,
    const float* __restrict__ x,
    const BlockQ4_K* __restrict__ W,
    int k_cols,
    int m_rows,
    const float* __restrict__ bias,
    float* __restrict__ residual
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE; // 0..31

    if (warp_id >= m_rows) return;

    int nb = k_cols / QK_K;
    const BlockQ4_K* row_w = W + (size_t)warp_id * nb;
    float row_sum = 0.0f;

    int chunk = lane_id / 8;        // 0..3
    int chunk_lane = lane_id % 8;   // 0..7
    int is_idx = chunk * 2;         // 0, 2, 4, 6
    int x_offset1 = chunk * 64 + chunk_lane * 4;
    int x_offset2 = x_offset1 + 32;

    for (int b = 0; b < nb; b++) {
        const BlockQ4_K* blk = &row_w[b];
        float d = __half2float(blk->d);
        float min_val = __half2float(blk->dmin);
        const float* x_blk = x + b * QK_K;

        // Unpack scales for this lane's chunk
        uint8_t sc0, m0, sc1, m1;
        get_scale_min_k4(is_idx + 0, blk->scales, &sc0, &m0);
        get_scale_min_k4(is_idx + 1, blk->scales, &sc1, &m1);

        float d1 = d * sc0;
        float min1 = min_val * m0;
        float d2 = d * sc1;
        float min2 = min_val * m1;

        // Single 32-bit load per thread: 32 threads load exactly 128 bytes (all 256 nibbles)
        uint32_t q = ((const uint32_t*)blk->qs)[lane_id];

        // 16-byte aligned float4 loads for low nibbles and high nibbles
        float4 x1 = *((const float4*)(x_blk + x_offset1));
        float4 x2 = *((const float4*)(x_blk + x_offset2));

        uint8_t q0 = (uint8_t)(q & 0xFF);
        uint8_t q1 = (uint8_t)((q >> 8) & 0xFF);
        uint8_t q2 = (uint8_t)((q >> 16) & 0xFF);
        uint8_t q3 = (uint8_t)(q >> 24);

        float dot1 = (float)(q0 & 0x0F) * x1.x + (float)(q1 & 0x0F) * x1.y + (float)(q2 & 0x0F) * x1.z + (float)(q3 & 0x0F) * x1.w;
        float sum_x1 = x1.x + x1.y + x1.z + x1.w;

        float dot2 = (float)(q0 >> 4) * x2.x + (float)(q1 >> 4) * x2.y + (float)(q2 >> 4) * x2.z + (float)(q3 >> 4) * x2.w;
        float sum_x2 = x2.x + x2.y + x2.z + x2.w;

        row_sum += (d1 * dot1 - min1 * sum_x1) + (d2 * dot2 - min2 * sum_x2);
    }

    row_sum = warp_reduce_sum(row_sum);

    if (lane_id == 0) {
        if (bias != nullptr) {
            row_sum += bias[warp_id];
        }
        if (residual != nullptr) {
            residual[warp_id] += row_sum;
        }
        if (y != nullptr) {
            y[warp_id] = row_sum;
        }
    }
}

// =========================================================================
// 1c. GEMV Q4_K Fused SwiGLU: Computes Gate and Up GEMV simultaneously in registers
// Writes directly: dst[r] = SiLU(Gate[r]) * Up[r]
// Eliminates 2 intermediate VRAM buffers and 1 extra kernel launch per layer!
// =========================================================================
__global__ void gemv_q4_k_swiglu_fused(
    float* __restrict__ dst,
    const float* __restrict__ x,
    const BlockQ4_K* __restrict__ W_gate,
    const BlockQ4_K* __restrict__ W_up,
    int k_cols,
    int m_rows
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE; // 0..31

    if (warp_id >= m_rows) return;

    int nb = k_cols / QK_K;
    const BlockQ4_K* row_gate = W_gate + (size_t)warp_id * nb;
    const BlockQ4_K* row_up = W_up + (size_t)warp_id * nb;
    float sum_gate = 0.0f;
    float sum_up = 0.0f;

    int chunk = lane_id / 8;
    int chunk_lane = lane_id % 8;
    int is_idx = chunk * 2;
    int x_offset1 = chunk * 64 + chunk_lane * 4;
    int x_offset2 = x_offset1 + 32;

    for (int b = 0; b < nb; b++) {
        const float* x_blk = x + b * QK_K;
        float4 x1 = *((const float4*)(x_blk + x_offset1));
        float4 x2 = *((const float4*)(x_blk + x_offset2));
        float sum_x1 = x1.x + x1.y + x1.z + x1.w;
        float sum_x2 = x2.x + x2.y + x2.z + x2.w;

        const BlockQ4_K* blk_g = &row_gate[b];
        const BlockQ4_K* blk_u = &row_up[b];

        // Dual-issue both loads concurrently to hide memory latency
        uint32_t q_g = ((const uint32_t*)blk_g->qs)[lane_id];
        uint32_t q_u = ((const uint32_t*)blk_u->qs)[lane_id];

        // 1. Gate row computation
        {
            float d = __half2float(blk_g->d);
            float min_val = __half2float(blk_g->dmin);

            uint8_t sc0, m0, sc1, m1;
            get_scale_min_k4(is_idx + 0, blk_g->scales, &sc0, &m0);
            get_scale_min_k4(is_idx + 1, blk_g->scales, &sc1, &m1);

            float d1 = d * sc0;
            float min1 = min_val * m0;
            float d2 = d * sc1;
            float min2 = min_val * m1;

            uint8_t q0 = (uint8_t)(q_g & 0xFF);
            uint8_t q1 = (uint8_t)((q_g >> 8) & 0xFF);
            uint8_t q2 = (uint8_t)((q_g >> 16) & 0xFF);
            uint8_t q3 = (uint8_t)(q_g >> 24);

            float dot1 = (float)(q0 & 0x0F) * x1.x + (float)(q1 & 0x0F) * x1.y + (float)(q2 & 0x0F) * x1.z + (float)(q3 & 0x0F) * x1.w;
            float dot2 = (float)(q0 >> 4) * x2.x + (float)(q1 >> 4) * x2.y + (float)(q2 >> 4) * x2.z + (float)(q3 >> 4) * x2.w;

            sum_gate += (d1 * dot1 - min1 * sum_x1) + (d2 * dot2 - min2 * sum_x2);
        }

        // 2. Up row computation
        {
            float d = __half2float(blk_u->d);
            float min_val = __half2float(blk_u->dmin);

            uint8_t sc0, m0, sc1, m1;
            get_scale_min_k4(is_idx + 0, blk_u->scales, &sc0, &m0);
            get_scale_min_k4(is_idx + 1, blk_u->scales, &sc1, &m1);

            float d1 = d * sc0;
            float min1 = min_val * m0;
            float d2 = d * sc1;
            float min2 = min_val * m1;

            uint8_t q0 = (uint8_t)(q_u & 0xFF);
            uint8_t q1 = (uint8_t)((q_u >> 8) & 0xFF);
            uint8_t q2 = (uint8_t)((q_u >> 16) & 0xFF);
            uint8_t q3 = (uint8_t)(q_u >> 24);

            float dot1 = (float)(q0 & 0x0F) * x1.x + (float)(q1 & 0x0F) * x1.y + (float)(q2 & 0x0F) * x1.z + (float)(q3 & 0x0F) * x1.w;
            float dot2 = (float)(q0 >> 4) * x2.x + (float)(q1 >> 4) * x2.y + (float)(q2 >> 4) * x2.z + (float)(q3 >> 4) * x2.w;

            sum_up += (d1 * dot1 - min1 * sum_x1) + (d2 * dot2 - min2 * sum_x2);
        }
    }

    sum_gate = warp_reduce_sum(sum_gate);
    sum_up = warp_reduce_sum(sum_up);

    if (lane_id == 0) {
        float silu = sum_gate / (1.0f + __expf(-sum_gate));
        dst[warp_id] = silu * sum_up;
    }
}

// =========================================================================
// 1b. GEMV Q4_K Aligned: 128-byte hardware cache coalesced layout
// W_qs is 128-byte aligned contiguous array of nibbles
// W_scales is pre-unpacked FP16 scales (d * sc0, dmin * m0, etc.) as half2
// =========================================================================
__global__ void gemv_q4_k_aligned(
    float* __restrict__ y,
    const float* __restrict__ x,
    const uint8_t* __restrict__ W_qs,
    const half2* __restrict__ W_scales,
    int k_cols,
    int m_rows
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id >= m_rows) return;

    int nb = k_cols / QK_K;
    const uint8_t* row_qs = W_qs + (size_t)warp_id * nb * 128;
    const half2* row_scales = W_scales + (size_t)warp_id * nb * 8;
    float row_sum = 0.0f;

    for (int b = 0; b < nb; b++) {
        const uint8_t* blk_qs = row_qs + b * 128;
        const half2* blk_scales = row_scales + b * 8;
        const float* x_blk = x + b * QK_K;

        #pragma unroll
        for (int s = 0; s < 4; s++) {
            half2 sc0 = blk_scales[s * 2 + 0];
            half2 sc1 = blk_scales[s * 2 + 1];

            float d1 = __low2float(sc0);
            float min1 = __high2float(sc0);
            float d2 = __low2float(sc1);
            float min2 = __high2float(sc1);

            int j = s * 64;
            uint8_t q_byte = blk_qs[s * 32 + lane_id];
            float val_x1 = x_blk[j + lane_id];
            float val_x2 = x_blk[j + 32 + lane_id];

            float w1 = d1 * (float)(q_byte & 0x0F) - min1;
            float w2 = d2 * (float)(q_byte >> 4) - min2;

            row_sum += (w1 * val_x1) + (w2 * val_x2);
        }
    }

    row_sum = warp_reduce_sum(row_sum);

    if (lane_id == 0) {
        y[warp_id] = row_sum;
    }
}

// =========================================================================
// 2. GEMV Q6_K: Matrix-Vector Multiplication (y = W * x)
// Each warp of 32 threads computes 1 row of output y
// =========================================================================
__global__ void gemv_q6_k(
    float* __restrict__ y,
    const float* __restrict__ x,
    const BlockQ6_K* __restrict__ W,
    int k_cols,
    int m_rows
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE; // 0..31

    if (warp_id >= m_rows) return;

    int nb = k_cols / QK_K;
    const BlockQ6_K* row_w = W + (size_t)warp_id * nb;
    float row_sum = 0.0f;

    for (int b = 0; b < nb; b++) {
        const BlockQ6_K* blk = &row_w[b];
        float d = __half2float(blk->d);
        const float* x_blk = x + b * QK_K;

        const uint8_t* ql = blk->ql;
        const uint8_t* qh = blk->qh;
        const int8_t* sc = blk->scales;

        for (int n = 0; n < QK_K; n += 128) {
            int is_idx = lane_id / 16;
            int8_t q1 = (int8_t)((ql[lane_id + 0] & 0x0F) | (((qh[lane_id] >> 0) & 3) << 4)) - 32;
            int8_t q2 = (int8_t)((ql[lane_id + 32] & 0x0F) | (((qh[lane_id] >> 2) & 3) << 4)) - 32;
            int8_t q3 = (int8_t)((ql[lane_id + 0] >> 4) | (((qh[lane_id] >> 4) & 3) << 4)) - 32;
            int8_t q4 = (int8_t)((ql[lane_id + 32] >> 4) | (((qh[lane_id] >> 6) & 3) << 4)) - 32;

            float w1 = d * sc[is_idx + 0] * q1;
            float w2 = d * sc[is_idx + 2] * q2;
            float w3 = d * sc[is_idx + 4] * q3;
            float w4 = d * sc[is_idx + 6] * q4;

            row_sum += w1 * x_blk[n + lane_id + 0];
            row_sum += w2 * x_blk[n + lane_id + 32];
            row_sum += w3 * x_blk[n + lane_id + 64];
            row_sum += w4 * x_blk[n + lane_id + 96];

            ql += 64;
            qh += 32;
            sc += 8;
        }
    }

    row_sum = warp_reduce_sum(row_sum);

    if (lane_id == 0) {
        y[warp_id] = row_sum;
    }
}

// =========================================================================
// 2b. GEMV Q6_K Fast: Unrolled with factored scales & vectorized lane access
// =========================================================================
__global__ void gemv_q6_k_fast(
    float* __restrict__ y,
    const float* __restrict__ x,
    const BlockQ6_K* __restrict__ W,
    int k_cols,
    int m_rows,
    const float* __restrict__ bias,
    float* __restrict__ residual
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE; // 0..31

    if (warp_id >= m_rows) return;

    int nb = k_cols / QK_K;
    const BlockQ6_K* row_w = W + (size_t)warp_id * nb;
    float row_sum = 0.0f;
    int is_idx = lane_id / 16;

    for (int b = 0; b < nb; b++) {
        const BlockQ6_K* blk = &row_w[b];
        float d = __half2float(blk->d);
        const float* x_blk = x + b * QK_K;
        float blk_sum = 0.0f;

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

            float s1 = (float)sc[is_idx + 0];
            float s2 = (float)sc[is_idx + 2];
            float s3 = (float)sc[is_idx + 4];
            float s4 = (float)sc[is_idx + 6];

            blk_sum += (s1 * (float)q1) * x_blk[n + lane_id + 0]
                     + (s2 * (float)q2) * x_blk[n + lane_id + 32]
                     + (s3 * (float)q3) * x_blk[n + lane_id + 64]
                     + (s4 * (float)q4) * x_blk[n + lane_id + 96];
        }

        row_sum += d * blk_sum;
    }

    row_sum = warp_reduce_sum(row_sum);

    if (lane_id == 0) {
        if (bias != nullptr) {
            row_sum += bias[warp_id];
        }
        if (residual != nullptr) {
            residual[warp_id] += row_sum;
        }
        if (y != nullptr) {
            y[warp_id] = row_sum;
        }
    }
}

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
// 8. KV Cache Store Kernel: writes K and V vectors for current pos
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
// 9. GQA Attention Kernel
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
    float* head_scores = scores_buf + (size_t)h * max_seq_len;

    __shared__ float s_warp_sum[4];

    // 1. Compute dot product scores with all past tokens t = 0..pos
    for (int t = 0; t <= pos; t++) {
        size_t kv_offset = ((size_t)h_kv * max_seq_len + t) * head_dim + tid;
        float k_d = k_cache[kv_offset];
        float prod = q_d * k_d;

        prod = warp_reduce_sum(prod);
        int warp_id = tid / WARP_SIZE;
        int lane_id = tid % WARP_SIZE;

        if (lane_id == 0) {
            s_warp_sum[warp_id] = prod;
        }
        __syncthreads();

        if (tid == 0) {
            float total_dot = s_warp_sum[0] + s_warp_sum[1] + s_warp_sum[2] + s_warp_sum[3];
            head_scores[t] = total_dot * attn_scale;
        }
        __syncthreads();
    }

    // 2. Softmax over t = 0..pos
    if (tid == 0) {
        float max_s = head_scores[0];
        for (int t = 1; t <= pos; t++) {
            if (head_scores[t] > max_s) max_s = head_scores[t];
        }

        float sum_exp = 0.0f;
        for (int t = 0; t <= pos; t++) {
            float exp_val = expf(head_scores[t] - max_s);
            head_scores[t] = exp_val;
            sum_exp += exp_val;
        }

        float inv_sum = 1.0f / sum_exp;
        for (int t = 0; t <= pos; t++) {
            head_scores[t] *= inv_sum;
        }
    }
    __syncthreads();

    // 3. Aggregate values: out[d] = sum_{t=0..pos} (score[t] * V[t, d])
    float out_d = 0.0f;
    for (int t = 0; t <= pos; t++) {
        float w = head_scores[t];
        size_t kv_offset = ((size_t)h_kv * max_seq_len + t) * head_dim + tid;
        out_d += w * v_cache[kv_offset];
    }

    attn_out[h * head_dim + tid] = out_d;
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
    float* head_scores = scores_buf + ((size_t)b * n_heads_q + h) * max_seq_len;

    __shared__ float s_warp_sum[4];

    // 1. Compute dot product scores with all past tokens t = 0..pos
    for (int t = 0; t <= pos; t++) {
        size_t kv_offset = ((size_t)h_kv * max_seq_len + t) * head_dim + tid;
        float k_d = k_cache[kv_offset];
        float prod = q_d * k_d;

        prod = warp_reduce_sum(prod);
        int warp_id = tid / WARP_SIZE;
        int lane_id = tid % WARP_SIZE;

        if (lane_id == 0) {
            s_warp_sum[warp_id] = prod;
        }
        __syncthreads();

        if (tid == 0) {
            float total_dot = s_warp_sum[0] + s_warp_sum[1] + s_warp_sum[2] + s_warp_sum[3];
            head_scores[t] = total_dot * attn_scale;
        }
        __syncthreads();
    }

    // 2. Softmax over t = 0..pos
    if (tid == 0) {
        float max_s = head_scores[0];
        for (int t = 1; t <= pos; t++) {
            if (head_scores[t] > max_s) max_s = head_scores[t];
        }

        float sum_exp = 0.0f;
        for (int t = 0; t <= pos; t++) {
            float exp_val = expf(head_scores[t] - max_s);
            head_scores[t] = exp_val;
            sum_exp += exp_val;
        }

        float inv_sum = 1.0f / sum_exp;
        for (int t = 0; t <= pos; t++) {
            head_scores[t] *= inv_sum;
        }
    }
    __syncthreads();

    // 3. Aggregate values: out[d] = sum_{t=0..pos} (score[t] * V[t, d])
    float out_d = 0.0f;
    for (int t = 0; t <= pos; t++) {
        float w = head_scores[t];
        size_t kv_offset = ((size_t)h_kv * max_seq_len + t) * head_dim + tid;
        out_d += w * v_cache[kv_offset];
    }

    size_t out_offset = (size_t)b * n_heads_q * head_dim + h * head_dim + tid;
    attn_out_batch[out_offset] = out_d;
}

// =========================================================================
// 10. Batched GEMV/GEMM Q4_K for Prompt Prefill Acceleration
// Computes Y[:, t] = W * X[:, t] for t = 0..batch_size - 1 (batch_size <= 32)
// Model weights W are read from VRAM ONCE and multiplied across all batch tokens!
// Fused bias and fused residual accumulation support
// =========================================================================
__global__ void gemm_q4_k_batch(
    float* __restrict__ Y,          // [batch_size, m_rows]
    const float* __restrict__ X,    // [batch_size, k_cols]
    const BlockQ4_K* __restrict__ W,// [m_rows, k_cols / 256]
    int k_cols,
    int m_rows,
    int batch_size,                 // 1..32
    const float* __restrict__ bias,     // optional [m_rows]
    float* __restrict__ residual    // optional [batch_size, m_rows]
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id >= m_rows) return;

    int nb = k_cols / QK_K;
    const BlockQ4_K* row_w = W + (size_t)warp_id * nb;

    float row_sum[32];
    #pragma unroll
    for (int t = 0; t < 32; t++) {
        row_sum[t] = 0.0f;
    }

    int chunk = lane_id / 8;
    int chunk_lane = lane_id % 8;
    int is_idx = chunk * 2;
    int x_offset1 = chunk * 64 + chunk_lane * 4;
    int x_offset2 = x_offset1 + 32;

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

        int q0_low = q0 & 0x0F; int q1_low = q1 & 0x0F; int q2_low = q2 & 0x0F; int q3_low = q3 & 0x0F;
        int q0_high = q0 >> 4;   int q1_high = q1 >> 4;   int q2_high = q2 >> 4;   int q3_high = q3 >> 4;

        float ql0 = (float)q0_low;  float ql1 = (float)q1_low;  float ql2 = (float)q2_low;  float ql3 = (float)q3_low;
        float qh0 = (float)q0_high; float qh1 = (float)q1_high; float qh2 = (float)q2_high; float qh3 = (float)q3_high;

        for (int t = 0; t < batch_size; t++) {
            const float* x_t = X + (size_t)t * k_cols + b * QK_K;
            float4 x1 = *((const float4*)(x_t + x_offset1));
            float4 x2 = *((const float4*)(x_t + x_offset2));

            float dot1 = ql0 * x1.x + ql1 * x1.y + ql2 * x1.z + ql3 * x1.w;
            float sum_x1 = x1.x + x1.y + x1.z + x1.w;

            float dot2 = qh0 * x2.x + qh1 * x2.y + qh2 * x2.z + qh3 * x2.w;
            float sum_x2 = x2.x + x2.y + x2.z + x2.w;

            row_sum[t] += (d1 * dot1 - min1 * sum_x1) + (d2 * dot2 - min2 * sum_x2);
        }
    }

    for (int t = 0; t < batch_size; t++) {
        float sum = warp_reduce_sum(row_sum[t]);
        if (lane_id == 0) {
            if (bias != nullptr) {
                sum += bias[warp_id];
            }
            if (residual != nullptr) {
                residual[(size_t)t * m_rows + warp_id] += sum;
            }
            if (Y != nullptr) {
                Y[(size_t)t * m_rows + warp_id] = sum;
            }
        }
    }
}

// =========================================================================
// 11. Batched GEMV/GEMM Q6_K for Prompt Prefill Acceleration
// Computes Y[:, t] = W * X[:, t] for t = 0..batch_size - 1 (batch_size <= 32)
// Fused bias and fused residual accumulation support
// =========================================================================
__global__ void gemm_q6_k_batch(
    float* __restrict__ Y,          // [batch_size, m_rows]
    const float* __restrict__ X,    // [batch_size, k_cols]
    const BlockQ6_K* __restrict__ W,// [m_rows, k_cols / 256]
    int k_cols,
    int m_rows,
    int batch_size,                 // 1..32
    const float* __restrict__ bias,     // optional [m_rows]
    float* __restrict__ residual    // optional [batch_size, m_rows]
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id >= m_rows) return;

    int nb = k_cols / QK_K;
    const BlockQ6_K* row_w = W + (size_t)warp_id * nb;
    int is_idx = lane_id / 16;

    float row_sum[32];
    #pragma unroll
    for (int t = 0; t < 32; t++) {
        row_sum[t] = 0.0f;
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

            for (int t = 0; t < batch_size; t++) {
                const float* x_blk = X + (size_t)t * k_cols + b * QK_K;
                row_sum[t] += w1 * x_blk[n + lane_id + 0]
                            + w2 * x_blk[n + lane_id + 32]
                            + w3 * x_blk[n + lane_id + 64]
                            + w4 * x_blk[n + lane_id + 96];
            }
        }
    }

    for (int t = 0; t < batch_size; t++) {
        float sum = warp_reduce_sum(row_sum[t]);
        if (lane_id == 0) {
            if (bias != nullptr) {
                sum += bias[warp_id];
            }
            if (residual != nullptr) {
                residual[(size_t)t * m_rows + warp_id] += sum;
            }
            if (Y != nullptr) {
                Y[(size_t)t * m_rows + warp_id] = sum;
            }
        }
    }
}

// =========================================================================
// 12. Batched Fused SwiGLU: Computes SiLU(Gate[:, t]) * Up[:, t] in registers
// Writes directly: dst[t, r] = SiLU(Gate[t, r]) * Up[t, r]
// =========================================================================
__global__ void gemm_q4_k_swiglu_batch(
    float* __restrict__ dst,        // [batch_size, m_rows]
    const float* __restrict__ X,    // [batch_size, k_cols]
    const BlockQ4_K* __restrict__ W_gate,
    const BlockQ4_K* __restrict__ W_up,
    int k_cols,
    int m_rows,
    int batch_size                  // 1..32
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id >= m_rows) return;

    int nb = k_cols / QK_K;
    const BlockQ4_K* row_gate = W_gate + (size_t)warp_id * nb;
    const BlockQ4_K* row_up = W_up + (size_t)warp_id * nb;

    float sum_gate[32];
    float sum_up[32];
    #pragma unroll
    for (int t = 0; t < 32; t++) {
        sum_gate[t] = 0.0f;
        sum_up[t] = 0.0f;
    }

    int chunk = lane_id / 8;
    int chunk_lane = lane_id % 8;
    int is_idx = chunk * 2;
    int x_offset1 = chunk * 64 + chunk_lane * 4;
    int x_offset2 = x_offset1 + 32;

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

        float qg0_low = (float)(q0_g & 0x0F); float qg1_low = (float)(q1_g & 0x0F);
        float qg2_low = (float)(q2_g & 0x0F); float qg3_low = (float)(q3_g & 0x0F);
        float qg0_high = (float)(q0_g >> 4);  float qg1_high = (float)(q1_g >> 4);
        float qg2_high = (float)(q2_g >> 4);  float qg3_high = (float)(q3_g >> 4);

        float qu0_low = (float)(q0_u & 0x0F); float qu1_low = (float)(q1_u & 0x0F);
        float qu2_low = (float)(q2_u & 0x0F); float qu3_low = (float)(q3_u & 0x0F);
        float qu0_high = (float)(q0_u >> 4);  float qu1_high = (float)(q1_u >> 4);
        float qu2_high = (float)(q2_u >> 4);  float qu3_high = (float)(q3_u >> 4);

        for (int t = 0; t < batch_size; t++) {
            const float* x_t = X + (size_t)t * k_cols + b * QK_K;
            float4 x1 = *((const float4*)(x_t + x_offset1));
            float4 x2 = *((const float4*)(x_t + x_offset2));
            float sum_x1 = x1.x + x1.y + x1.z + x1.w;
            float sum_x2 = x2.x + x2.y + x2.z + x2.w;

            float dot1_g = qg0_low * x1.x + qg1_low * x1.y + qg2_low * x1.z + qg3_low * x1.w;
            float dot2_g = qg0_high * x2.x + qg1_high * x2.y + qg2_high * x2.z + qg3_high * x2.w;
            sum_gate[t] += (dg1 * dot1_g - ming1 * sum_x1) + (dg2 * dot2_g - ming2 * sum_x2);

            float dot1_u = qu0_low * x1.x + qu1_low * x1.y + qu2_low * x1.z + qu3_low * x1.w;
            float dot2_u = qu0_high * x2.x + qu1_high * x2.y + qu2_high * x2.z + qu3_high * x2.w;
            sum_up[t] += (du1 * dot1_u - minu1 * sum_x1) + (du2 * dot2_u - minu2 * sum_x2);
        }
    }

    for (int t = 0; t < batch_size; t++) {
        float g = warp_reduce_sum(sum_gate[t]);
        float u = warp_reduce_sum(sum_up[t]);
        if (lane_id == 0) {
            float silu = g / (1.0f + __expf(-g));
            dst[(size_t)t * m_rows + warp_id] = silu * u;
        }
    }
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

} // extern "C"
