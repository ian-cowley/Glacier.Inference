// Glacier.Inference Qwen 3.5 / 3.6 / 3.8 Hybrid Gated DeltaNet + Full-Attention CUDA Kernels
// Bit-exact CUDA port of Qwen35VulkanModel compute shaders for native full-GPU execution on NVIDIA.
#pragma once

#include "common.cuh"

extern "C" {

__device__ __forceinline__ float q35_warp_sum(float v) {
    v += __shfl_xor_sync(0xFFFFFFFFu, v, 16);
    v += __shfl_xor_sync(0xFFFFFFFFu, v, 8);
    v += __shfl_xor_sync(0xFFFFFFFFu, v, 4);
    v += __shfl_xor_sync(0xFFFFFFFFu, v, 2);
    v += __shfl_xor_sync(0xFFFFFFFFu, v, 1);
    return v;
}

__device__ __forceinline__ float q35_warp_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, 16));
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, 8));
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, 4));
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, 2));
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, 1));
    return v;
}

// 1. RMSNorm: Grid = (rows), Block = 256
__global__ void q35_rmsnorm(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ y,
    int size,
    int rows,
    float eps
) {
    int r = blockIdx.x;
    if (r >= rows) return;
    int tid = threadIdx.x;
    const float* xr = x + (size_t)r * size;
    float* yr = y + (size_t)r * size;

    float sum = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
        float v = xr[i];
        sum = fmaf(v, v, sum);
    }
    sum = q35_warp_sum(sum);
    __shared__ float s_w[8];
    if ((tid & 31) == 0) s_w[tid >> 5] = sum;
    __syncthreads();

    float total = (tid < 8) ? s_w[tid] : 0.0f;
    if (tid < 32) total = q35_warp_sum(total);
    __shared__ float s_scale;
    if (tid == 0) s_scale = rsqrtf(total / (float)size + eps);
    __syncthreads();

    float sc = s_scale;
    for (int i = tid; i < size; i += blockDim.x) {
        yr[i] = xr[i] * sc * w[i];
    }
}

// 1b. Fused Residual Add + RMSNorm: x[i] += residual[i]; y[i] = rmsnorm(x)[i] * w[i]
//     Eliminates 128 kernel launches + 128 global read/writes of _dX per token!
//     Grid = (1), Block = 256
__global__ void q35_add_rmsnorm(
    float* __restrict__ x,
    const float* __restrict__ residual,
    const float* __restrict__ w,
    float* __restrict__ y,
    int size,
    float eps
) {
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
        float v = x[i] + residual[i];
        x[i] = v;
        sum = fmaf(v, v, sum);
    }
    sum = q35_warp_sum(sum);
    __shared__ float s_w[8];
    if ((tid & 31) == 0) s_w[tid >> 5] = sum;
    __syncthreads();

    float total = (tid < 8) ? s_w[tid] : 0.0f;
    if (tid < 32) total = q35_warp_sum(total);
    __shared__ float s_scale;
    if (tid == 0) s_scale = rsqrtf(total / (float)size + eps);
    __syncthreads();

    float sc = s_scale;
    for (int i = tid; i < size; i += blockDim.x) {
        y[i] = x[i] * sc * w[i];
    }
}

// 2. SwiGLU: out[i] = silu(gate[i]) * up[i]
__global__ void q35_swiglu(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        out[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

// 3. In-place Add: a[i] += b[i]
__global__ void q35_add_inplace(
    float* __restrict__ a,
    const float* __restrict__ b,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] += b[i];
    }
}

// 4. Small F32 MatVec (ssm_alpha / ssm_beta): Grid = ((N + 3) / 4), Block = 128 (4 warps = 4 rows)
__global__ void q35_f32_matvec(
    const float* __restrict__ W,
    const float* __restrict__ x,
    float* __restrict__ y,
    int K,
    int N
) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 4 + warp;
    if (row >= N) return;

    const float* wr = W + (size_t)row * K;
    float sum = 0.0f;
    if ((K & 3) == 0) {
        int k4 = K >> 2;
        const float4* wr4 = reinterpret_cast<const float4*>(wr);
        const float4* x4 = reinterpret_cast<const float4*>(x);
        for (int i = lane; i < k4; i += 32) {
            float4 wv = __ldg(&wr4[i]);
            float4 xv = __ldg(&x4[i]);
            sum = fmaf(wv.x, xv.x, sum);
            sum = fmaf(wv.y, xv.y, sum);
            sum = fmaf(wv.z, xv.z, sum);
            sum = fmaf(wv.w, xv.w, sum);
        }
    } else {
        for (int i = lane; i < K; i += 32) {
            sum = fmaf(__ldg(&wr[i]), __ldg(&x[i]), sum);
        }
    }
    sum = q35_warp_sum(sum);
    if (lane == 0) y[row] = sum;
}

// 5. GDN Depthwise Causal Conv1D (kernel=4) + SiLU + rolling history state [channels][3]
__global__ void q35_gdn_conv(
    const float* __restrict__ qkv,
    const float* __restrict__ cw,
    float* __restrict__ cs,
    float* __restrict__ out,
    int channels
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    int s3 = c * 3;
    int w4 = c * 4;
    float h0 = cs[s3], h1 = cs[s3 + 1], h2 = cs[s3 + 2];
    float x = qkv[c];
    float acc = h0 * cw[w4] + h1 * cw[w4 + 1] + h2 * cw[w4 + 2] + x * cw[w4 + 3];
    cs[s3] = h1; cs[s3 + 1] = h2; cs[s3 + 2] = x;
    out[c] = acc / (1.0f + expf(-acc));
}

// 6. GDN Per-Group L2 Normalization of Q and K (stateDim = 128): Grid = (nGroups), Block = 128
__global__ void q35_gdn_l2(
    float* __restrict__ buf,
    int nGroups
) {
    int g = blockIdx.x;
    if (g >= nGroups) return;
    int t = threadIdx.x; // 0..127
    int qi = g * 128 + t;
    int ki = nGroups * 128 + g * 128 + t;
    float q = buf[qi];
    float k = buf[ki];

    float sq = q35_warp_sum(q * q);
    float sk = q35_warp_sum(k * k);
    __shared__ float rq[4], rk[4];
    if ((t & 31) == 0) {
        rq[t >> 5] = sq;
        rk[t >> 5] = sk;
    }
    __syncthreads();

    float sumQ = rq[0] + rq[1] + rq[2] + rq[3];
    float sumK = rk[0] + rk[1] + rk[2] + rk[3];
    float invQ = (sumQ > 1e-12f) ? (rsqrtf(sumQ + 1e-6f) * 0.08838834764831845f) : 0.0f; // 1/sqrt(128)
    float invK = (sumK > 1e-12f) ? rsqrtf(sumK + 1e-6f) : 0.0f;
    buf[qi] = q * invQ;
    buf[ki] = k * invK;
}

// 7. GDN Recurrent State Update: Grid = (nV), Block = 128 (stateDim = 128, transposed state St[h][j*128 + i])
__global__ void q35_gdn_rec(
    const float* __restrict__ qkv,
    const float* __restrict__ alpha,
    const float* __restrict__ betaIn,
    const float* __restrict__ dtBias,
    const float* __restrict__ aW,
    float* __restrict__ st,
    float* __restrict__ y,
    int nK,
    int nV
) {
    int h = blockIdx.x;
    if (h >= nV) return;
    int i = threadIdx.x; // 0..127
    int g = h % nK;

    __shared__ float sq[128];
    __shared__ float sk[128];
    sq[i] = qkv[g * 128 + i];
    sk[i] = qkv[nK * 128 + g * 128 + i];
    __syncthreads();

    float a = alpha[h] + dtBias[h];
    float dt = (a > 20.0f) ? a : log1pf(expf(a));
    float decay = expf(aW[h] * dt);
    float beta = 1.0f / (1.0f + expf(-betaIn[h]));

    size_t base = (size_t)h * (128 * 128);
    float kv = 0.0f;
    #pragma unroll 4
    for (int j = 0; j < 128; j++) {
        kv = fmaf(st[base + j * 128 + i], sk[j], kv);
    }
    kv *= decay;

    float v = qkv[2 * nK * 128 + h * 128 + i];
    float delta = (v - kv) * beta;
    float dot = 0.0f;
    #pragma unroll 4
    for (int j = 0; j < 128; j++) {
        float s = fmaf(delta, sk[j], st[base + j * 128 + i] * decay);
        st[base + j * 128 + i] = s;
        dot = fmaf(s, sq[j], dot);
    }
    y[h * 128 + i] = dot;
}

// 8. Full-Attention Prep (headDim = 256): Grid = (nQ + 2*nKv), Block = 256
//    Reads `pos` from device pointer `d_pos` so the entire 64-layer pass is 100% CUDA-Graph capturable!
__global__ void q35_attn_prep(
    const float* __restrict__ qFull,
    const float* __restrict__ kRaw,
    const float* __restrict__ vRaw,
    const float* __restrict__ qNormW,
    const float* __restrict__ kNormW,
    float* __restrict__ qOut,
    float* __restrict__ gateOut,
    float* __restrict__ kCache,
    float* __restrict__ vCache,
    const int* __restrict__ d_pos,
    int nQ,
    int nKv,
    int maxSeq,
    int ropeDim,
    float eps,
    float freqBase
) {
    int pos = *d_pos;
    int w = blockIdx.x;
    int d = threadIdx.x; // 0..255
    __shared__ float sh[256];
    __shared__ float s_w[8];

    auto rms_scale = [&](float v) -> float {
        float ws = q35_warp_sum(v * v);
        if ((d & 31) == 0) s_w[d >> 5] = ws;
        __syncthreads();
        float tot = (d < 8) ? s_w[d] : 0.0f;
        if (d < 32) tot = q35_warp_sum(tot);
        __shared__ float s_r;
        if (d == 0) s_r = rsqrtf(tot * (1.0f / 256.0f) + eps);
        __syncthreads();
        return s_r;
    };

    auto rope_apply = [&]() -> float {
        int half_d = ropeDim >> 1;
        if (d >= ropeDim) return sh[d];
        int i = (d < half_d) ? d : (d - half_d);
        float freq = 1.0f / powf(freqBase, (float)(2 * i) / (float)ropeDim);
        float theta = (float)pos * freq;
        float s, c;
        __sincosf(theta, &s, &c);
        float a = sh[i], b = sh[i + half_d];
        return (d < half_d) ? (a * c - b * s) : (a * s + b * c);
    };

    if (w < nQ) {
        int qRawOff = w * 512;
        int qOutOff = w * 256;
        float q = qFull[qRawOff + d];
        gateOut[qOutOff + d] = qFull[qRawOff + 256 + d];
        float r = rms_scale(q);
        sh[d] = q * r * qNormW[d];
        __syncthreads();
        qOut[qOutOff + d] = rope_apply();
    } else if (w < nQ + nKv) {
        int kh = w - nQ;
        float k = kRaw[kh * 256 + d];
        float r = rms_scale(k);
        sh[d] = k * r * kNormW[d];
        __syncthreads();
        kCache[((size_t)kh * maxSeq + pos) * 256 + d] = rope_apply();
    } else {
        int kh = w - nQ - nKv;
        vCache[((size_t)kh * maxSeq + pos) * 256 + d] = vRaw[kh * 256 + d];
    }
}

// 9. Full-Attention Decode (headDim = 256): Grid = (nHeads), Block = 256
//    Reads `kvLen = *d_pos + 1` from device pointer `d_pos` for 100% static CUDA Graph replay!
__global__ void q35_attn_decode(
    const float* __restrict__ q,
    const float* __restrict__ gate,
    const float* __restrict__ kCache,
    const float* __restrict__ vCache,
    float* __restrict__ scores,
    float* __restrict__ out,
    const int* __restrict__ d_pos,
    int nHeads,
    int nKv,
    int maxSeq,
    float scale
) {
    int kvLen = (*d_pos) + 1;
    int h = blockIdx.x;
    if (h >= nHeads) return;
    int t = threadIdx.x; // 0..255
    int kvh = h / (nHeads / nKv);
    int qBase = h * 256;

    __shared__ float sq[256];
    __shared__ float s_w[8];
    sq[t] = q[qBase + t];
    __syncthreads();

    size_t sb = (size_t)h * maxSeq;
    size_t kb = (size_t)kvh * maxSeq * 256;

    float mx = -3.4e38f;
    for (int p = t; p < kvLen; p += 256) {
        const float* kp = kCache + kb + (size_t)p * 256;
        float dot = 0.0f;
        #pragma unroll 4
        for (int d = 0; d < 256; d++) dot = fmaf(sq[d], kp[d], dot);
        dot *= scale;
        scores[sb + p] = dot;
        mx = fmaxf(mx, dot);
    }

    float wm = q35_warp_max(mx);
    if ((t & 31) == 0) s_w[t >> 5] = wm;
    __syncthreads();
    float m = (t < 8) ? s_w[t] : -3.4e38f;
    if (t < 32) m = q35_warp_max(m);
    __shared__ float s_m;
    if (t == 0) s_m = m;
    __syncthreads();
    m = s_m;

    float sum = 0.0f;
    for (int p = t; p < kvLen; p += 256) {
        float e = expf(scores[sb + p] - m);
        scores[sb + p] = e;
        sum += e;
    }

    float ws = q35_warp_sum(sum);
    if ((t & 31) == 0) s_w[t >> 5] = ws;
    __syncthreads();
    float tot = (t < 8) ? s_w[t] : 0.0f;
    if (t < 32) tot = q35_warp_sum(tot);
    __shared__ float s_inv;
    if (t == 0) s_inv = 1.0f / tot;
    __syncthreads();
    float inv = s_inv;

    float acc = 0.0f;
    for (int p = 0; p < kvLen; p++) {
        acc = fmaf(scores[sb + p], vCache[kb + (size_t)p * 256 + t], acc);
    }
    float g = gate[qBase + t];
    out[qBase + t] = acc * inv * (1.0f / (1.0f + expf(-g)));
}

} // extern "C"
