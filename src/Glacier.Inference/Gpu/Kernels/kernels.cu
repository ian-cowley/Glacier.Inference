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

} // extern "C"
