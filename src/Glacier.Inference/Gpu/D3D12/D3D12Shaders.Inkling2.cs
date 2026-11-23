#if !ANDROID
namespace Glacier.Inference.Gpu.D3D12;

/// <summary>
/// Dedicated, zero-aliasing HLSL Compute Shaders (cs_5_0: NVIDIA + AMD RDNA 3.5)
/// for 100% GPU-resident Inkling 2 Mini execution (1 fence sync per token on UMA/16GB+ GPUs).
/// All buffers use raw ByteAddressBuffer / RWByteAddressBuffer for strict D3D12 root-descriptor compliance.
/// </summary>
public static partial class D3D12Shaders
{
    // 1. Vectorized Load4 LDS-cached Q8_0 GEMV (4-byte-aligned 36-byte blocks: [f16 d | u16 pad | 32 x i8 qs])
    //    [numthreads(16, 16, 1)]: 16 rows/group (2x s_x reuse), 2 lanes/block (8 blocks/iter), zero warp-tail waste!
    public const string Inkling2GemvQ8_0 = @"
cbuffer Params : register(b0)
{
    uint k_cols;
    uint m_rows;
    uint x_offset;
    uint y_offset;
};

ByteAddressBuffer W : register(t0);
RWByteAddressBuffer x : register(u0);
RWByteAddressBuffer y : register(u1);

groupshared float s_x[3072];
groupshared float s_mem[256];

[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint row_in_grp = gtid.y;
    uint warp_id = gid.x * 16u + row_in_grp;
    uint lane_id = gtid.x;
    uint s_idx = row_in_grp * 16u + lane_id;

    for (uint ci = s_idx; ci < k_cols; ci += 256u)
    {
        s_x[ci] = asfloat(x.Load((x_offset + ci) * 4u));
    }
    GroupMemoryBarrierWithGroupSync();

    if (warp_id < m_rows)
    {
        uint nb = k_cols >> 5;
        uint row_byte_offset = warp_id * nb * 36u;
        float row_sum = 0.0f;

        uint b_lane = lane_id >> 1;
        uint half_id = lane_id & 1u;
        uint q_off = 4u + (half_id << 4);

        for (uint b = b_lane; b < nb; b += 8u)
        {
            uint blk_addr = row_byte_offset + b * 36u;
            float d = f16tof32(W.Load(blk_addr) & 0xFFFFu);
            uint4 q4 = W.Load4(blk_addr + q_off);
            uint xb = (b << 5) + (half_id << 4);

            float4 v0 = float4((int)(q4.x << 24) >> 24, (int)(q4.x << 16) >> 24, (int)(q4.x << 8) >> 24, (int)q4.x >> 24);
            float4 v1 = float4((int)(q4.y << 24) >> 24, (int)(q4.y << 16) >> 24, (int)(q4.y << 8) >> 24, (int)q4.y >> 24);
            float4 v2 = float4((int)(q4.z << 24) >> 24, (int)(q4.z << 16) >> 24, (int)(q4.z << 8) >> 24, (int)q4.z >> 24);
            float4 v3 = float4((int)(q4.w << 24) >> 24, (int)(q4.w << 16) >> 24, (int)(q4.w << 8) >> 24, (int)q4.w >> 24);

            float acc = dot(v0, float4(s_x[xb + 0u], s_x[xb + 1u], s_x[xb + 2u], s_x[xb + 3u]))
                      + dot(v1, float4(s_x[xb + 4u], s_x[xb + 5u], s_x[xb + 6u], s_x[xb + 7u]))
                      + dot(v2, float4(s_x[xb + 8u], s_x[xb + 9u], s_x[xb + 10u], s_x[xb + 11u]))
                      + dot(v3, float4(s_x[xb + 12u], s_x[xb + 13u], s_x[xb + 14u], s_x[xb + 15u]));
            row_sum += d * acc;
        }

        s_mem[s_idx] = row_sum;
    }
    else
    {
        s_mem[s_idx] = 0.0f;
    }

    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 8u) s_mem[s_idx] += s_mem[s_idx + 8u];
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 4u) s_mem[s_idx] += s_mem[s_idx + 4u];
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 2u) s_mem[s_idx] += s_mem[s_idx + 2u];
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 1u) s_mem[s_idx] += s_mem[s_idx + 1u];

    if (lane_id == 0u && warp_id < m_rows)
    {
        y.Store((y_offset + warp_id) * 4u, asuint(s_mem[row_in_grp * 16u]));
    }
}
";

    // 1b. LDS-cached Q5_K GEMV (176 bytes per 256-element super-block, natively 4-byte aligned)
    public const string Inkling2GemvQ5K = @"
cbuffer Params : register(b0)
{
    uint k_cols;
    uint m_rows;
    uint x_offset;
    uint y_offset;
};

ByteAddressBuffer W : register(t0);
RWByteAddressBuffer x : register(u0);
RWByteAddressBuffer y : register(u1);

groupshared float s_x[3072];
groupshared float s_mem[256];

[numthreads(32, 8, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint row_in_grp = gtid.y;
    uint warp_id = gid.x * 8u + row_in_grp;
    uint lane_id = gtid.x;
    uint s_idx = row_in_grp * 32u + lane_id;

    for (uint ci = s_idx; ci < k_cols; ci += 256u)
    {
        s_x[ci] = asfloat(x.Load((x_offset + ci) * 4u));
    }
    GroupMemoryBarrierWithGroupSync();

    if (warp_id < m_rows)
    {
        uint nb = k_cols >> 8;
        uint row_byte_offset = warp_id * nb * 176u;
        float row_sum = 0.0f;

        uint lane_dword = lane_id & ~3u;
        uint lane_byte = (lane_id & 3u) * 8u;

        for (uint b = 0u; b < nb; b++)
        {
            uint blk_addr = row_byte_offset + b * 176u;
            uint dd = W.Load(blk_addr);
            float d = f16tof32(dd & 0xFFFFu);
            float dmin = f16tof32(dd >> 16u);

            uint sc0 = W.Load(blk_addr + 4u);
            uint sc1 = W.Load(blk_addr + 8u);
            uint sc2 = W.Load(blk_addr + 12u);

            uint qh_byte = (W.Load(blk_addr + 16u + lane_dword) >> lane_byte) & 0xFFu;
            uint x_blk = b * 256u;

            [unroll]
            for (uint j = 0u; j < 4u; j++)
            {
                uint is0 = j * 2u;
                uint is1 = j * 2u + 1u;

                uint sc_lo, mn_lo, sc_hi, mn_hi;
                if (is0 < 4u)
                {
                    sc_lo = (sc0 >> (is0 * 8u)) & 63u;
                    mn_lo = (sc1 >> (is0 * 8u)) & 63u;
                    sc_hi = (sc0 >> (is1 * 8u)) & 63u;
                    mn_hi = (sc1 >> (is1 * 8u)) & 63u;
                }
                else
                {
                    uint sh0 = (is0 - 4u) * 8u;
                    uint sh1 = (is1 - 4u) * 8u;
                    sc_lo = ((sc2 >> sh0) & 0xFu) | ((((sc0 >> (sh0 + 6u)) & 3u)) << 4u);
                    mn_lo = ((sc2 >> (sh0 + 4u)) & 0xFu) | ((((sc1 >> (sh0 + 6u)) & 3u)) << 4u);
                    sc_hi = ((sc2 >> sh1) & 0xFu) | ((((sc0 >> (sh1 + 6u)) & 3u)) << 4u);
                    mn_hi = ((sc2 >> (sh1 + 4u)) & 0xFu) | ((((sc1 >> (sh1 + 6u)) & 3u)) << 4u);
                }

                float d1 = d * (float)sc_lo;
                float m1 = dmin * (float)mn_lo;
                float d2 = d * (float)sc_hi;
                float m2 = dmin * (float)mn_hi;

                uint qs_byte = (W.Load(blk_addr + 48u + j * 32u + lane_dword) >> lane_byte) & 0xFFu;
                uint q1 = (qs_byte & 0xFu) + (((qh_byte >> (j * 2u)) & 1u) << 4u);
                uint q2 = (qs_byte >> 4u) + (((qh_byte >> (j * 2u + 1u)) & 1u) << 4u);

                float x1 = s_x[x_blk + j * 64u + lane_id];
                float x2 = s_x[x_blk + j * 64u + 32u + lane_id];
                row_sum += (d1 * (float)q1 - m1) * x1 + (d2 * (float)q2 - m2) * x2;
            }
        }

        s_mem[s_idx] = row_sum;
    }
    else
    {
        s_mem[s_idx] = 0.0f;
    }

    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 16u) s_mem[s_idx] += s_mem[s_idx + 16u];
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 8u)  s_mem[s_idx] += s_mem[s_idx + 8u];
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 4u)  s_mem[s_idx] += s_mem[s_idx + 4u];
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 2u)  s_mem[s_idx] += s_mem[s_idx + 2u];
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 1u)  s_mem[s_idx] += s_mem[s_idx + 1u];

    if (lane_id == 0u && warp_id < m_rows)
    {
        y.Store((y_offset + warp_id) * 4u, asuint(s_mem[row_in_grp * 32u]));
    }
}
";

    // 2. Vectorized Load4 F32 GEMV for small projections (Router 128x2304, Sinks 24x2304, DynConv 32x2304, LoopCtrl 64x4608 / 2304x64)
    //    [numthreads(16, 16, 1)]: 16 rows/group, 128-bit W.Load4 + x.Load4 (64 floats/iter = 1 exact iter for GateRank=64!)
    public const string Inkling2GemvF32 = @"
cbuffer Params : register(b0)
{
    uint k_cols;
    uint m_rows;
    uint x_offset;
    uint y_offset;
};

ByteAddressBuffer W : register(t0);
RWByteAddressBuffer x : register(u0);
RWByteAddressBuffer y : register(u1);

groupshared float s_mem[256];

[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint row_in_grp = gtid.y;
    uint warp_id = gid.x * 16u + row_in_grp;
    uint lane_id = gtid.x;
    uint s_idx = row_in_grp * 16u + lane_id;

    if (warp_id < m_rows)
    {
        uint row_base = warp_id * k_cols;
        float sum = 0.0f;
        uint c4_end = k_cols & ~3u;
        for (uint c = lane_id * 4u; c < c4_end; c += 64u)
        {
            float4 w4 = asfloat(W.Load4((row_base + c) * 4u));
            float4 x4 = asfloat(x.Load4((x_offset + c) * 4u));
            sum += dot(w4, x4);
        }
        if (c4_end < k_cols)
        {
            for (uint c = c4_end + lane_id; c < k_cols; c += 16u)
            {
                sum += asfloat(W.Load((row_base + c) * 4u)) * asfloat(x.Load((x_offset + c) * 4u));
            }
        }
        s_mem[s_idx] = sum;
    }
    else
    {
        s_mem[s_idx] = 0.0f;
    }

    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 8u) s_mem[s_idx] += s_mem[s_idx + 8u];
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 4u) s_mem[s_idx] += s_mem[s_idx + 4u];
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 2u) s_mem[s_idx] += s_mem[s_idx + 2u];
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 1u) s_mem[s_idx] += s_mem[s_idx + 1u];

    if (lane_id == 0u && warp_id < m_rows)
    {
        y.Store((y_offset + warp_id) * 4u, asuint(s_mem[row_in_grp * 16u]));
    }
}
";

    // 3. Indirect GPU-indexed 2D-Batched NVFP4 MoE GEMV (gid.y = k_slot 0..7 -> ALL 8 experts in 1 dispatch!)
    //    [numthreads(16, 16, 1)] = 16 rows/group (2x s_x reuse), branchless bit-math decode_fp4, zero warp-tail waste!
    public const string Inkling2GemvNVFP4Indirect = @"
cbuffer Params : register(b0)
{
    uint k_cols;
    uint m_rows;
    uint x_stride;
    uint y_stride;
    uint exp_stride_bytes;
    uint pad0;
};

ByteAddressBuffer W : register(t0);
RWByteAddressBuffer x : register(u0);
RWByteAddressBuffer y : register(u1);
RWByteAddressBuffer topk_ids : register(u2);

groupshared float s_x[2304];
groupshared float s_mem[256];

float decode_ue4m3(uint b)
{
    b &= 0x7Fu;
    if (b == 0u || b == 127u) return 0.0f;
    uint exp_bits = b >> 3;
    uint man_bits = b & 7u;
    if (exp_bits == 0u) return (float)man_bits * 0.001953125f;
    return asfloat(((exp_bits + 120u) << 23) | (man_bits << 20));
}

float decode_fp4(uint nib)
{
    uint mag = nib & 7u;
    uint sign = (nib & 8u) << 28u;
    uint bits = (mag == 0u) ? 0u : ((mag == 1u) ? 0x3F000000u : ((mag + 252u) << 22u));
    return asfloat(bits | sign);
}

[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint k_slot = gid.y;
    uint row_in_grp = gtid.y;
    uint warp_id = gid.x * 16u + row_in_grp;
    uint lane_id = gtid.x;
    uint s_idx = row_in_grp * 16u + lane_id;

    uint x_off = k_slot * x_stride;
    uint y_off = k_slot * y_stride;

    for (uint ci = s_idx; ci < k_cols; ci += 256u)
    {
        s_x[ci] = asfloat(x.Load((x_off + ci) * 4u));
    }
    GroupMemoryBarrierWithGroupSync();

    if (warp_id < m_rows)
    {
        uint exp_id = topk_ids.Load(k_slot * 4u);
        uint nb = k_cols >> 6;
        uint row_byte_offset = exp_id * exp_stride_bytes + warp_id * nb * 36u;
        float row_sum = 0.0f;

        // 4 lanes per 64-element block (4 blocks/iter in parallel, 64-bit coalesced W.Load2!)
        // Exact 9 iterations for k=2304, 5 iterations (90% util) for k=1152!
        uint b_lane = lane_id >> 2;
        uint sub = lane_id & 3u;
        uint q_off = 4u + (sub << 3);
        uint sc_shift = sub << 3;

        for (uint b = b_lane; b < nb; b += 4u)
        {
            uint blk_addr = row_byte_offset + b * 36u;
            float d = decode_ue4m3((W.Load(blk_addr) >> sc_shift) & 0xFFu);
            uint2 q2 = W.Load2(blk_addr + q_off);
            uint col0 = (b << 6) + (sub << 4);

            float acc =
                decode_fp4((q2.x >>  0u) & 0xFu) * s_x[col0 +  0u] +
                decode_fp4((q2.x >>  4u) & 0xFu) * s_x[col0 +  8u] +
                decode_fp4((q2.x >>  8u) & 0xFu) * s_x[col0 +  1u] +
                decode_fp4((q2.x >> 12u) & 0xFu) * s_x[col0 +  9u] +
                decode_fp4((q2.x >> 16u) & 0xFu) * s_x[col0 +  2u] +
                decode_fp4((q2.x >> 20u) & 0xFu) * s_x[col0 + 10u] +
                decode_fp4((q2.x >> 24u) & 0xFu) * s_x[col0 +  3u] +
                decode_fp4((q2.x >> 28u) & 0xFu) * s_x[col0 + 11u] +
                decode_fp4((q2.y >>  0u) & 0xFu) * s_x[col0 +  4u] +
                decode_fp4((q2.y >>  4u) & 0xFu) * s_x[col0 + 12u] +
                decode_fp4((q2.y >>  8u) & 0xFu) * s_x[col0 +  5u] +
                decode_fp4((q2.y >> 12u) & 0xFu) * s_x[col0 + 13u] +
                decode_fp4((q2.y >> 16u) & 0xFu) * s_x[col0 +  6u] +
                decode_fp4((q2.y >> 20u) & 0xFu) * s_x[col0 + 14u] +
                decode_fp4((q2.y >> 24u) & 0xFu) * s_x[col0 +  7u] +
                decode_fp4((q2.y >> 28u) & 0xFu) * s_x[col0 + 15u];
            row_sum += d * acc;
        }

        s_mem[s_idx] = row_sum;
    }
    else
    {
        s_mem[s_idx] = 0.0f;
    }

    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 8u) s_mem[s_idx] += s_mem[s_idx + 8u];
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 4u) s_mem[s_idx] += s_mem[s_idx + 4u];
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 2u) s_mem[s_idx] += s_mem[s_idx + 2u];
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 1u) s_mem[s_idx] += s_mem[s_idx + 1u];

    if (lane_id == 0u && warp_id < m_rows)
    {
        y.Store((y_off + warp_id) * 4u, asuint(s_mem[row_in_grp * 16u]));
    }
}
";

    // 4. RMSNorm (with or without weight vector) — 1 threadgroup of 256 threads
    public const string Inkling2RmsNorm = @"
cbuffer Params : register(b0)
{
    uint dim;
    float eps;
    uint in_off;
    uint out_off;
    uint has_weight;
    uint pad0;
};

ByteAddressBuffer weight : register(t0);
RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);

groupshared float s_sum[256];

[numthreads(256, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID)
{
    uint tid = gtid.x;
    float sumSq = 0.0f;
    for (uint i = tid; i < dim; i += 256u)
    {
        float v = asfloat(src.Load((in_off + i) * 4u));
        sumSq += v * v;
    }
    s_sum[tid] = sumSq;
    GroupMemoryBarrierWithGroupSync();

    if (tid < 128u) s_sum[tid] += s_sum[tid + 128u]; GroupMemoryBarrierWithGroupSync();
    if (tid < 64u)  s_sum[tid] += s_sum[tid + 64u];  GroupMemoryBarrierWithGroupSync();
    if (tid < 32u)  s_sum[tid] += s_sum[tid + 32u];  GroupMemoryBarrierWithGroupSync();
    if (tid < 16u)  s_sum[tid] += s_sum[tid + 16u];  GroupMemoryBarrierWithGroupSync();
    if (tid < 8u)   s_sum[tid] += s_sum[tid + 8u];   GroupMemoryBarrierWithGroupSync();
    if (tid < 4u)   s_sum[tid] += s_sum[tid + 4u];   GroupMemoryBarrierWithGroupSync();
    if (tid < 2u)   s_sum[tid] += s_sum[tid + 2u];   GroupMemoryBarrierWithGroupSync();
    if (tid < 1u)   s_sum[tid] += s_sum[tid + 1u];   GroupMemoryBarrierWithGroupSync();

    float inv = rsqrt(s_sum[0] / (float)dim + eps);
    for (uint i = tid; i < dim; i += 256u)
    {
        float v = asfloat(src.Load((in_off + i) * 4u)) * inv;
        if (has_weight != 0u) v *= asfloat(weight.Load(i * 4u));
        dst.Store((out_off + i) * 4u, asuint(v));
    }
}
";

    // 5. Block-Attention Residual Mix (nBlocks entries in blocksBuf + curBuf -> RMSNorm(1e-6) -> softmax(q*s/sqrt(d)) -> outBuf)
    //    Dispatch: (nQueries, 1, 1) with 256 threads per query. Max nStates = 16.
    public const string Inkling2Mix = @"
cbuffer Params : register(b0)
{
    uint dim;
    uint n_blocks;
    uint query_base_floats;
    uint out_base_floats;
    uint blocks_start_idx;
    uint pad0;
};

ByteAddressBuffer queries : register(t0);
RWByteAddressBuffer blocks_buf : register(u0);
RWByteAddressBuffer cur_buf : register(u1);
RWByteAddressBuffer out_buf : register(u2);

groupshared float s_red_sq[256];
groupshared float s_red_dot[256];
groupshared float s_inv_rms[16];
groupshared float s_weights[16];

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint q = gid.x;
    uint tid = gtid.x;
    uint n_states = n_blocks + 1u;
    float inv_sqrt_d = rsqrt((float)dim);

    // Step 1: fused sumSq + dot(state, query) reduction per state s in [0..n_states)
    for (uint s = 0u; s < n_states; s++)
    {
        float sumSq = 0.0f;
        float dotVal = 0.0f;
        uint q_off = query_base_floats + q * dim;
        uint blk_off = (blocks_start_idx + s) * dim;

        for (uint d = tid; d < dim; d += 256u)
        {
            float v = (s < n_blocks)
                ? asfloat(blocks_buf.Load((blk_off + d) * 4u))
                : asfloat(cur_buf.Load(d * 4u));
            sumSq += v * v;
            dotVal += v * asfloat(queries.Load((q_off + d) * 4u));
        }

        s_red_sq[tid] = sumSq;
        s_red_dot[tid] = dotVal;
        GroupMemoryBarrierWithGroupSync();
        if (tid < 128u) { s_red_sq[tid] += s_red_sq[tid + 128u]; s_red_dot[tid] += s_red_dot[tid + 128u]; } GroupMemoryBarrierWithGroupSync();
        if (tid < 64u)  { s_red_sq[tid] += s_red_sq[tid + 64u];  s_red_dot[tid] += s_red_dot[tid + 64u];  } GroupMemoryBarrierWithGroupSync();
        if (tid < 32u)  { s_red_sq[tid] += s_red_sq[tid + 32u];  s_red_dot[tid] += s_red_dot[tid + 32u];  } GroupMemoryBarrierWithGroupSync();
        if (tid < 16u)  { s_red_sq[tid] += s_red_sq[tid + 16u];  s_red_dot[tid] += s_red_dot[tid + 16u];  } GroupMemoryBarrierWithGroupSync();
        if (tid < 8u)   { s_red_sq[tid] += s_red_sq[tid + 8u];   s_red_dot[tid] += s_red_dot[tid + 8u];   } GroupMemoryBarrierWithGroupSync();
        if (tid < 4u)   { s_red_sq[tid] += s_red_sq[tid + 4u];   s_red_dot[tid] += s_red_dot[tid + 4u];   } GroupMemoryBarrierWithGroupSync();
        if (tid < 2u)   { s_red_sq[tid] += s_red_sq[tid + 2u];   s_red_dot[tid] += s_red_dot[tid + 2u];   } GroupMemoryBarrierWithGroupSync();
        if (tid < 1u)   { s_red_sq[tid] += s_red_sq[tid + 1u];   s_red_dot[tid] += s_red_dot[tid + 1u];   } GroupMemoryBarrierWithGroupSync();

        if (tid == 0u)
        {
            float inv_rms = rsqrt(s_red_sq[0] / (float)dim + 1e-6f);
            s_inv_rms[s] = inv_rms;
            s_weights[s] = s_red_dot[0] * inv_rms * inv_sqrt_d;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // Step 2: Softmax over s_weights[0..n_states) on lane 0
    if (tid == 0u)
    {
        float maxS = s_weights[0];
        for (uint s = 1u; s < n_states; s++) maxS = max(maxS, s_weights[s]);
        float sumExp = 0.0f;
        for (uint s = 0u; s < n_states; s++)
        {
            float e = exp(s_weights[s] - maxS);
            s_weights[s] = e;
            sumExp += e;
        }
        float invSum = 1.0f / sumExp;
        for (uint s = 0u; s < n_states; s++) s_weights[s] *= invSum * s_inv_rms[s];
    }
    GroupMemoryBarrierWithGroupSync();

    // Step 3: Weighted sum of normalized states into out_buf
    uint dst_base = out_base_floats + q * dim;
    for (uint d = tid; d < dim; d += 256u)
    {
        float acc = 0.0f;
        for (uint s = 0u; s < n_states; s++)
        {
            float v = (s < n_blocks)
                ? asfloat(blocks_buf.Load(((blocks_start_idx + s) * dim + d) * 4u))
                : asfloat(cur_buf.Load(d * 4u));
            acc += s_weights[s] * v;
        }
        out_buf.Store((dst_base + d) * 4u, asuint(acc));
    }
}
";

    // 6. Cross-pass Loop Control Blend: cur[d] = (1 - tanh(forget[d])) * cur[d] + tanh(gain[d]) * msg[msg_off + d]
    public const string Inkling2LoopCtrlBlend = @"
cbuffer Params : register(b0)
{
    uint dim;
    uint msg_off;
    uint pad0;
    uint pad1;
};

RWByteAddressBuffer cur_buf : register(u0);
RWByteAddressBuffer msg_buf : register(u1);
RWByteAddressBuffer gain_buf : register(u2);
RWByteAddressBuffer forget_buf : register(u3);

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    uint d = dtid.x;
    if (d >= dim) return;
    float c = asfloat(cur_buf.Load(d * 4u));
    float m = asfloat(msg_buf.Load((msg_off + d) * 4u));
    float g = tanh(asfloat(gain_buf.Load(d * 4u)));
    float f = tanh(asfloat(forget_buf.Load(d * 4u)));
    cur_buf.Store(d * 4u, asuint((1.0f - f) * c + g * m));
}
";

    // 7. Causal Short-Conv (l_cache = 4, updates 3-step history ring buffer in-place, optional dynamic taps)
    public const string Inkling2ShortConv = @"
cbuffer Params : register(b0)
{
    uint channels;
    uint hist_base_floats;
    uint has_dyn;
    uint head_dim;
};

ByteAddressBuffer conv_w : register(t0);
RWByteAddressBuffer cur_buf : register(u0);
RWByteAddressBuffer hist_buf : register(u1);
RWByteAddressBuffer dyn_buf : register(u2);

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    uint c = dtid.x;
    if (c >= channels) return;

    uint h_off = hist_base_floats + c * 3u;
    float x0 = asfloat(hist_buf.Load((h_off + 0u) * 4u));
    float x1 = asfloat(hist_buf.Load((h_off + 1u) * 4u));
    float x2 = asfloat(hist_buf.Load((h_off + 2u) * 4u));
    float x3 = asfloat(cur_buf.Load(c * 4u));

    uint w_off = c * 4u;
    float conv = asfloat(conv_w.Load((w_off + 0u) * 4u)) * x0
               + asfloat(conv_w.Load((w_off + 1u) * 4u)) * x1
               + asfloat(conv_w.Load((w_off + 2u) * 4u)) * x2
               + asfloat(conv_w.Load((w_off + 3u) * 4u)) * x3;

    float scale = (has_dyn != 0u) ? rsqrt(9.0f) : rsqrt(5.0f);
    if (has_dyn != 0u)
    {
        uint h = c / head_dim;
        uint d_off = h * 4u;
        conv += asfloat(dyn_buf.Load((d_off + 0u) * 4u)) * x3
              + asfloat(dyn_buf.Load((d_off + 1u) * 4u)) * x2
              + asfloat(dyn_buf.Load((d_off + 2u) * 4u)) * x1
              + asfloat(dyn_buf.Load((d_off + 3u) * 4u)) * x0;
    }

    hist_buf.Store((h_off + 0u) * 4u, asuint(x1));
    hist_buf.Store((h_off + 1u) * 4u, asuint(x2));
    hist_buf.Store((h_off + 2u) * 4u, asuint(x3));
    cur_buf.Store(c * 4u, asuint((x3 + conv) * scale));
}
";

    // 8. Per-Head L2 Clip (optional scale + optional tau) OR Per-Head RMSNorm + optional FP16 KV Cache Write
    //    mode: 0 = ClipPerHead (optional scale + tau), 1 = RmsNormPerHead
    //    Dispatch: (nHeads, 1, 1) with 128 threads per head (supports headDim <= 128).
    public const string Inkling2HeadNormClipKv = @"
cbuffer Params : register(b0)
{
    uint dim;
    uint mode;
    uint has_scale;
    float thresh_or_eps;
    float tau;
    uint write_kv;
    uint kv_slot_offset_u32;
    uint pad0;
};

ByteAddressBuffer scale_buf : register(t0);
RWByteAddressBuffer buf : register(u0);
RWByteAddressBuffer kv_fp16 : register(u1);

groupshared float s_sq[128];

[numthreads(128, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint h = gid.x;
    uint tid = gtid.x;
    uint off = h * dim;

    float v = 0.0f;
    if (tid < dim)
    {
        v = asfloat(buf.Load((off + tid) * 4u));
        if (mode == 0u && has_scale != 0u) v *= asfloat(scale_buf.Load((off + tid) * 4u));
    }
    s_sq[tid] = v * v;
    GroupMemoryBarrierWithGroupSync();

    if (tid < 64u) s_sq[tid] += s_sq[tid + 64u]; GroupMemoryBarrierWithGroupSync();
    if (tid < 32u) s_sq[tid] += s_sq[tid + 32u]; GroupMemoryBarrierWithGroupSync();
    if (tid < 16u) s_sq[tid] += s_sq[tid + 16u]; GroupMemoryBarrierWithGroupSync();
    if (tid < 8u)  s_sq[tid] += s_sq[tid + 8u];  GroupMemoryBarrierWithGroupSync();
    if (tid < 4u)  s_sq[tid] += s_sq[tid + 4u];  GroupMemoryBarrierWithGroupSync();
    if (tid < 2u)  s_sq[tid] += s_sq[tid + 2u];  GroupMemoryBarrierWithGroupSync();
    if (tid < 1u)  s_sq[tid] += s_sq[tid + 1u];  GroupMemoryBarrierWithGroupSync();

    float mult = 1.0f;
    if (mode == 0u)
    {
        float norm = sqrt(s_sq[0]) + 1e-6f;
        if (norm > thresh_or_eps) mult = thresh_or_eps / norm;
        mult *= tau;
    }
    else
    {
        mult = rsqrt(s_sq[0] / (float)dim + thresh_or_eps);
    }

    float outV = v * mult;
    if (tid < dim)
    {
        buf.Store((off + tid) * 4u, asuint(outV));
    }

    // Barrier before reusing s_sq[] so all warps finish reading s_sq[0]
    GroupMemoryBarrierWithGroupSync();

    // Pack pairs of floats into FP16 (2 x uint16 per uint32) in GPU KV cache
    if (write_kv != 0u)
    {
        s_sq[tid] = outV;
        GroupMemoryBarrierWithGroupSync();
        if (tid < (dim >> 1))
        {
            uint lo = f32tof16(s_sq[tid * 2u]);
            uint hi = f32tof16(s_sq[tid * 2u + 1u]);
            uint packed = (lo & 0xFFFFu) | (hi << 16u);
            kv_fp16.Store((kv_slot_offset_u32 + (h * (dim >> 1)) + tid) * 4u, packed);
        }
    }
}
";

    // 9. GPU Relative Positional Attention with Attention Sinks & FP16 KV Cache + Inline Sigmoid Output Gate
    //    Dispatch: (nHead, 1, 1) with 128 threads per head (each thread handles 1 dimension of headDim=128!).
    //    Online FlashAttention-style streaming softmax over [startT..pos] — ZERO score array allocation, supports any seqLen!
    public const string Inkling2AttnDecode = @"
cbuffer Params : register(b0)
{
    uint pos;
    uint start_t;
    uint kv_cap;
    uint kv_dim_half;
    uint group_size;
    uint rel_extent;
    uint n_rel;
    uint kv_layer_base_u32;
};

ByteAddressBuffer rel_bias_buf : register(t0);
RWByteAddressBuffer q_buf : register(u0);
RWByteAddressBuffer r_buf : register(u1);
RWByteAddressBuffer sinks_buf : register(u2);
RWByteAddressBuffer gate_buf : register(u3);
RWByteAddressBuffer k_cache_fp16 : register(u4);
RWByteAddressBuffer v_cache_fp16 : register(u5);
RWByteAddressBuffer attn_out_buf : register(u6);

groupshared float s_q[128];
groupshared float s_r[16];
groupshared float s_dot[128];

[numthreads(128, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint h = gid.x;
    uint d = gtid.x; // 0..127 = exact headDim!
    uint kv_head = h / group_size;
    uint q_off = h * 128u;
    uint r_off = h * n_rel;

    s_q[d] = asfloat(q_buf.Load((q_off + d) * 4u));
    if (d < n_rel) s_r[d] = asfloat(r_buf.Load((r_off + d) * 4u));
    float sink_val = asfloat(sinks_buf.Load(h * 4u));
    GroupMemoryBarrierWithGroupSync();

    // Online FlashAttention state initialized with virtual zero-logit attention sink (score = 0.0f, V = 0)
    float m_max = 0.0f;
    float l_sum = 1.0f; // exp(0 - 0) = 1.0f for the virtual sink!
    float acc_v = 0.0f;
    float inv_sqrt_d = rsqrt(128.0f);

    uint pair_idx = d >> 1;
    uint is_hi = d & 1u;

    for (uint t = start_t; t <= pos; t++)
    {
        uint slot = t % kv_cap;
        uint kv_u32_addr = kv_layer_base_u32 + slot * kv_dim_half + kv_head * 64u + pair_idx;

        uint k_pack = k_cache_fp16.Load(kv_u32_addr * 4u);
        float k_val = f16tof32((is_hi != 0u) ? (k_pack >> 16u) : (k_pack & 0xFFFFu));

        s_dot[d] = s_q[d] * k_val;
        GroupMemoryBarrierWithGroupSync();
        if (d < 64u) s_dot[d] += s_dot[d + 64u]; GroupMemoryBarrierWithGroupSync();
        if (d < 32u) s_dot[d] += s_dot[d + 32u]; GroupMemoryBarrierWithGroupSync();
        if (d < 16u) s_dot[d] += s_dot[d + 16u]; GroupMemoryBarrierWithGroupSync();
        if (d < 8u)  s_dot[d] += s_dot[d + 8u];  GroupMemoryBarrierWithGroupSync();
        if (d < 4u)  s_dot[d] += s_dot[d + 4u];  GroupMemoryBarrierWithGroupSync();
        if (d < 2u)  s_dot[d] += s_dot[d + 2u];  GroupMemoryBarrierWithGroupSync();
        if (d < 1u)  s_dot[d] += s_dot[d + 1u];  GroupMemoryBarrierWithGroupSync();

        float score = s_dot[0] * inv_sqrt_d - sink_val;
        // Barrier before next iteration overwrites s_dot[] so all warps finish reading s_dot[0]
        GroupMemoryBarrierWithGroupSync();
        uint dist = pos - t;
        if (dist < rel_extent)
        {
            uint rb_base = dist * n_rel;
            float rb = 0.0f;
            for (uint ri = 0u; ri < n_rel; ri++)
                rb += s_r[ri] * asfloat(rel_bias_buf.Load((rb_base + ri) * 4u));
            score += rb;
        }

        uint v_pack = v_cache_fp16.Load(kv_u32_addr * 4u);
        float v_val = f16tof32((is_hi != 0u) ? (v_pack >> 16u) : (v_pack & 0xFFFFu));

        float new_max = max(m_max, score);
        float alpha = exp(m_max - new_max);
        float p = exp(score - new_max);
        l_sum = l_sum * alpha + p;
        acc_v = acc_v * alpha + p * v_val;
        m_max = new_max;
    }

    float g = asfloat(gate_buf.Load((q_off + d) * 4u));
    float sig = 1.0f / (1.0f + exp(-g));
    attn_out_buf.Store((q_off + d) * 4u, asuint((acc_v / l_sum) * sig));
}
";

    // 10. GPU Router Top-K Selection (128 experts -> Top-8 -> writes topkIds[0..7], meta[0..7]=guScale, meta[8..15]=weight*dnScale)
    public const string Inkling2RouterTopK = @"
cbuffer Params : register(b0)
{
    uint n_exp;
    uint top_k;
    float weight_scale;
    uint has_bias;
    uint has_gu_scale;
    uint has_dn_scale;
};

ByteAddressBuffer exp_bias : register(t0);
ByteAddressBuffer gu_scales : register(t1);
ByteAddressBuffer dn_scales : register(t2);
RWByteAddressBuffer logits_buf : register(u0);
RWByteAddressBuffer topk_ids : register(u1);
RWByteAddressBuffer meta_buf : register(u2);

groupshared float s_probs[128];
groupshared float s_scores[128];

[numthreads(128, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID)
{
    uint e = gtid.x;
    if (e < n_exp)
    {
        float p = 1.0f / (1.0f + exp(-asfloat(logits_buf.Load(e * 4u))));
        float b = (has_bias != 0u) ? asfloat(exp_bias.Load(e * 4u)) : 0.0f;
        s_probs[e] = p;
        s_scores[e] = p + b;
    }
    GroupMemoryBarrierWithGroupSync();

    if (e == 0u)
    {
        uint ids[8];
        float probs[8];
        float prob_sum = 1e-20f;

        [unroll]
        for (uint k = 0u; k < 8u; k++)
        {
            uint best_i = 0u;
            float best_sc = -1e30f;
            for (uint i = 0u; i < n_exp; i++)
            {
                float sc = s_scores[i];
                if (sc > best_sc)
                {
                    best_sc = sc;
                    best_i = i;
                }
            }
            ids[k] = best_i;
            probs[k] = s_probs[best_i];
            prob_sum += s_probs[best_i];
            s_scores[best_i] = -1e30f;
        }

        float w_mul = weight_scale / prob_sum;

        [unroll]
        for (uint k2 = 0u; k2 < 8u; k2++)
        {
            uint id = ids[k2];
            float gs = (has_gu_scale != 0u) ? asfloat(gu_scales.Load(id * 4u)) : 1.0f;
            float ds = (has_dn_scale != 0u) ? asfloat(dn_scales.Load(id * 4u)) : 1.0f;
            topk_ids.Store(k2 * 4u, id);
            meta_buf.Store(k2 * 4u, asuint(gs));
            meta_buf.Store((8u + k2) * 4u, asuint(probs[k2] * w_mul * ds));
        }
    }
}
";

    // 11. Vector Add: dst[i] = a[i] + b[i]
    public const string Inkling2VecAdd = @"
cbuffer Params : register(b0)
{
    uint count;
    uint pad0;
    uint pad1;
    uint pad2;
};

RWByteAddressBuffer a_buf : register(u0);
RWByteAddressBuffer b_buf : register(u1);
RWByteAddressBuffer dst_buf : register(u2);

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    uint i = dtid.x;
    if (i >= count) return;
    float v = asfloat(a_buf.Load(i * 4u)) + asfloat(b_buf.Load(i * 4u));
    dst_buf.Store(i * 4u, asuint(v));
}
";

    // 12. Softcap logits in-place: logits[v] = cap * tanh(logits[v] / cap)
    public const string Inkling2Softcap = @"
cbuffer Params : register(b0)
{
    uint count;
    float cap;
    float inv_cap;
    uint pad0;
};

RWByteAddressBuffer logits_buf : register(u0);

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    uint i = dtid.x;
    if (i >= count) return;
    float v = asfloat(logits_buf.Load(i * 4u));
    logits_buf.Store(i * 4u, asuint(cap * tanh(v * inv_cap)));
}
";
    // 13. Zero-Transition UAV-to-UAV Copy (eliminates 4,320 CopySource/CopyDest pipeline flushes per token!)
    public const string Inkling2UavCopy = @"
cbuffer Params : register(b0)
{
    uint count;
    uint src_off;
    uint dst_off;
    uint pad0;
};

RWByteAddressBuffer src_buf : register(u0);
RWByteAddressBuffer dst_buf : register(u1);
RWByteAddressBuffer unused_buf : register(u2);

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    uint i = dtid.x;
    if (i >= count) return;
    dst_buf.Store((dst_off + i) * 4u, src_buf.Load((src_off + i) * 4u));
}
";
}
#endif
