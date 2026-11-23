#if !ANDROID
namespace Glacier.Inference.Gpu.D3D12;

/// <summary>
/// True 16-Token GPU Batched Prefill HLSL Compute Shaders (cs_5_0: NVIDIA + AMD RDNA 3.5)
/// for Inkling 2 Mini. Reads every Q8_0 / Q6_K / Q5_K / F32 weight matrix ONCE across up to 16
/// tokens in registers, runs channel-parallel register-resident causal Short-Conv, and executes
/// causal FlashAttention + Router + 2D NVFP4 MoE across all B tokens in ONE command list!
/// </summary>
public static partial class D3D12Shaders
{
    // 1. Batched Load4 Q8_0 GEMM: reads each 36-byte Q8_0 block ONCE for all B <= 16 tokens!
    //    [numthreads(16, 8, 1)]: 8 rows/group, 2 lanes/block (8 blocks/iter), zero warp-tail waste! Dispatch: ((m_rows + 7) / 8, 1, 1).
    public const string Inkling2GemmQ8_0Batch = @"
cbuffer Params : register(b0)
{
    uint k_cols;
    uint m_rows;
    uint batch_size;
    uint pad0;
};

ByteAddressBuffer W : register(t0);
RWByteAddressBuffer x : register(u0);
RWByteAddressBuffer y : register(u1);

groupshared float s_mem[16 * 128];

[numthreads(16, 8, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint row_in_grp = gtid.y;
    uint warp_id = gid.x * 8u + row_in_grp;
    uint lane_id = gtid.x;
    uint s_idx = row_in_grp * 16u + lane_id;

    float acc[16];
    [unroll] for (uint t = 0u; t < 16u; t++) acc[t] = 0.0f;

    if (warp_id < m_rows)
    {
        uint nb = k_cols >> 5;
        uint row_byte_offset = warp_id * nb * 36u;
        uint b_lane = lane_id >> 1;
        uint half_id = lane_id & 1u;
        uint q_off = 4u + (half_id << 4);

        for (uint b = b_lane; b < nb; b += 8u)
        {
            uint blk_addr = row_byte_offset + b * 36u;
            float d = f16tof32(W.Load(blk_addr) & 0xFFFFu);
            uint4 q4 = W.Load4(blk_addr + q_off);
            uint xb = (b << 5) + (half_id << 4);

            float4 v0 = d * float4((int)(q4.x << 24) >> 24, (int)(q4.x << 16) >> 24, (int)(q4.x << 8) >> 24, (int)q4.x >> 24);
            float4 v1 = d * float4((int)(q4.y << 24) >> 24, (int)(q4.y << 16) >> 24, (int)(q4.y << 8) >> 24, (int)q4.y >> 24);
            float4 v2 = d * float4((int)(q4.z << 24) >> 24, (int)(q4.z << 16) >> 24, (int)(q4.z << 8) >> 24, (int)q4.z >> 24);
            float4 v3 = d * float4((int)(q4.w << 24) >> 24, (int)(q4.w << 16) >> 24, (int)(q4.w << 8) >> 24, (int)q4.w >> 24);

            for (uint t = 0u; t < batch_size; t++)
            {
                uint x_addr = (t * k_cols + xb) * 4u;
                float4 x0 = asfloat(x.Load4(x_addr +  0u));
                float4 x1 = asfloat(x.Load4(x_addr + 16u));
                float4 x2 = asfloat(x.Load4(x_addr + 32u));
                float4 x3 = asfloat(x.Load4(x_addr + 48u));
                acc[t] += dot(v0, x0) + dot(v1, x1) + dot(v2, x2) + dot(v3, x3);
            }
        }
    }

    for (uint t = 0u; t < batch_size; t++) s_mem[t * 128u + s_idx] = acc[t];
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 8u) { for (uint t = 0u; t < batch_size; t++) s_mem[t * 128u + s_idx] += s_mem[t * 128u + s_idx + 8u]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 4u) { for (uint t = 0u; t < batch_size; t++) s_mem[t * 128u + s_idx] += s_mem[t * 128u + s_idx + 4u]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 2u) { for (uint t = 0u; t < batch_size; t++) s_mem[t * 128u + s_idx] += s_mem[t * 128u + s_idx + 2u]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 1u) { for (uint t = 0u; t < batch_size; t++) s_mem[t * 128u + s_idx] += s_mem[t * 128u + s_idx + 1u]; }

    if (lane_id == 0u && warp_id < m_rows)
    {
        for (uint t = 0u; t < batch_size; t++)
        {
            y.Store((t * m_rows + warp_id) * 4u, asuint(s_mem[t * 128u + row_in_grp * 16u]));
        }
    }
}
";

    // 2. Batched Q6_K GEMM: reads each 212-byte Q6_K block ONCE for all B <= 16 tokens!
    public const string Inkling2GemmQ6KBatch = @"
cbuffer Params : register(b0)
{
    uint k_cols;
    uint m_rows;
    uint batch_size;
    uint pad0;
};

ByteAddressBuffer W : register(t0);
RWByteAddressBuffer x : register(u0);
RWByteAddressBuffer y : register(u1);

groupshared float s_mem[16 * 128];

int decode_s8(uint b) { return (int)(b << 24) >> 24; }

[numthreads(32, 4, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint row_in_grp = gtid.y;
    uint warp_id = gid.x * 4u + row_in_grp;
    uint lane_id = gtid.x;
    uint s_idx = row_in_grp * 32u + lane_id;

    float acc[16];
    [unroll] for (uint t = 0u; t < 16u; t++) acc[t] = 0.0f;

    if (warp_id < m_rows)
    {
        uint nb = k_cols >> 8;
        uint row_byte_offset = warp_id * nb * 212u;
        uint is_idx = lane_id >> 4;
        uint lane_byte = (lane_id & 3u) * 8u;
        uint lane_dword = lane_id & ~3u;

        for (uint b = 0u; b < nb; b++)
        {
            uint blk_addr = row_byte_offset + b * 212u;
            float d = f16tof32(W.Load(blk_addr + 208u) & 0xFFFFu);
            uint x_blk = b * 256u;

            uint sc0 = W.Load(blk_addr + 192u);
            uint sc1 = W.Load(blk_addr + 196u);
            uint sc2 = W.Load(blk_addr + 200u);
            uint sc3 = W.Load(blk_addr + 204u);

            [unroll]
            for (uint step = 0u; step < 2u; step++)
            {
                uint n = step * 128u;
                uint ql_addr = blk_addr + step * 64u;
                uint qh_addr = blk_addr + 128u + step * 32u;

                uint qh_val = (W.Load(qh_addr + lane_dword) >> lane_byte) & 0xFFu;
                uint ql_low = (W.Load(ql_addr + lane_dword) >> lane_byte) & 0xFFu;
                uint ql_high = (W.Load(ql_addr + 32u + lane_dword) >> lane_byte) & 0xFFu;

                int q1 = (int)((ql_low & 0x0Fu) | (((qh_val >> 0u) & 3u) << 4u)) - 32;
                int q2 = (int)((ql_high & 0x0Fu) | (((qh_val >> 2u) & 3u) << 4u)) - 32;
                int q3 = (int)((ql_low >> 4u) | (((qh_val >> 4u) & 3u) << 4u)) - 32;
                int q4 = (int)((ql_high >> 4u) | (((qh_val >> 6u) & 3u) << 4u)) - 32;

                uint2 sc_words = (step == 0u) ? uint2(sc0, sc1) : uint2(sc2, sc3);
                float w1 = d * (float)(decode_s8((sc_words.x >> (is_idx * 8u)) & 0xFFu) * q1);
                float w2 = d * (float)(decode_s8((sc_words.x >> ((is_idx + 2u) * 8u)) & 0xFFu) * q2);
                float w3 = d * (float)(decode_s8((sc_words.y >> (is_idx * 8u)) & 0xFFu) * q3);
                float w4 = d * (float)(decode_s8((sc_words.y >> ((is_idx + 2u) * 8u)) & 0xFFu) * q4);

                uint xi = x_blk + n + lane_id;
                for (uint t = 0u; t < batch_size; t++)
                {
                    uint xa = (t * k_cols + xi) * 4u;
                    acc[t] += w1 * asfloat(x.Load(xa + 0u))
                            + w2 * asfloat(x.Load(xa + 128u))
                            + w3 * asfloat(x.Load(xa + 256u))
                            + w4 * asfloat(x.Load(xa + 384u));
                }
            }
        }
    }

    for (uint t = 0u; t < batch_size; t++) s_mem[t * 128u + s_idx] = acc[t];
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 16u) { for (uint t = 0u; t < batch_size; t++) s_mem[t * 128u + s_idx] += s_mem[t * 128u + s_idx + 16u]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 8u)  { for (uint t = 0u; t < batch_size; t++) s_mem[t * 128u + s_idx] += s_mem[t * 128u + s_idx + 8u]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 4u)  { for (uint t = 0u; t < batch_size; t++) s_mem[t * 128u + s_idx] += s_mem[t * 128u + s_idx + 4u]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 2u)  { for (uint t = 0u; t < batch_size; t++) s_mem[t * 128u + s_idx] += s_mem[t * 128u + s_idx + 2u]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 1u)  { for (uint t = 0u; t < batch_size; t++) s_mem[t * 128u + s_idx] += s_mem[t * 128u + s_idx + 1u]; }

    if (lane_id == 0u && warp_id < m_rows)
    {
        for (uint t = 0u; t < batch_size; t++)
        {
            y.Store((t * m_rows + warp_id) * 4u, asuint(s_mem[t * 128u + row_in_grp * 32u]));
        }
    }
}
";

    // 3. Batched Q5_K GEMM: reads each 176-byte Q5_K block ONCE for all B <= 16 tokens!
    public const string Inkling2GemmQ5KBatch = @"
cbuffer Params : register(b0)
{
    uint k_cols;
    uint m_rows;
    uint batch_size;
    uint pad0;
};

ByteAddressBuffer W : register(t0);
RWByteAddressBuffer x : register(u0);
RWByteAddressBuffer y : register(u1);

groupshared float s_mem[16 * 128];

[numthreads(32, 4, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint row_in_grp = gtid.y;
    uint warp_id = gid.x * 4u + row_in_grp;
    uint lane_id = gtid.x;
    uint s_idx = row_in_grp * 32u + lane_id;

    float acc[16];
    [unroll] for (uint t = 0u; t < 16u; t++) acc[t] = 0.0f;

    if (warp_id < m_rows)
    {
        uint nb = k_cols >> 8;
        uint row_byte_offset = warp_id * nb * 176u;
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

                uint qs_byte = (W.Load(blk_addr + 48u + j * 32u + lane_dword) >> lane_byte) & 0xFFu;
                uint q1 = (qs_byte & 0xFu) + (((qh_byte >> (j * 2u)) & 1u) << 4u);
                uint q2 = (qs_byte >> 4u) + (((qh_byte >> (j * 2u + 1u)) & 1u) << 4u);

                float w1 = d * (float)sc_lo * (float)q1 - dmin * (float)mn_lo;
                float w2 = d * (float)sc_hi * (float)q2 - dmin * (float)mn_hi;

                uint xi = x_blk + j * 64u + lane_id;
                for (uint t = 0u; t < batch_size; t++)
                {
                    uint xa = (t * k_cols + xi) * 4u;
                    acc[t] += w1 * asfloat(x.Load(xa)) + w2 * asfloat(x.Load(xa + 128u));
                }
            }
        }
    }

    for (uint t = 0u; t < batch_size; t++) s_mem[t * 128u + s_idx] = acc[t];
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 16u) { for (uint t = 0u; t < batch_size; t++) s_mem[t * 128u + s_idx] += s_mem[t * 128u + s_idx + 16u]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 8u)  { for (uint t = 0u; t < batch_size; t++) s_mem[t * 128u + s_idx] += s_mem[t * 128u + s_idx + 8u]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 4u)  { for (uint t = 0u; t < batch_size; t++) s_mem[t * 128u + s_idx] += s_mem[t * 128u + s_idx + 4u]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 2u)  { for (uint t = 0u; t < batch_size; t++) s_mem[t * 128u + s_idx] += s_mem[t * 128u + s_idx + 2u]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 1u)  { for (uint t = 0u; t < batch_size; t++) s_mem[t * 128u + s_idx] += s_mem[t * 128u + s_idx + 1u]; }

    if (lane_id == 0u && warp_id < m_rows)
    {
        for (uint t = 0u; t < batch_size; t++)
        {
            y.Store((t * m_rows + warp_id) * 4u, asuint(s_mem[t * 128u + row_in_grp * 32u]));
        }
    }
}
";

    // 4. Batched Load4 F32 GEMM: reads each F32 weight row ONCE for all B <= 16 tokens with 128-bit Load4!
    public const string Inkling2GemmF32Batch = @"
cbuffer Params : register(b0)
{
    uint k_cols;
    uint m_rows;
    uint batch_size;
    uint pad0;
};

ByteAddressBuffer W : register(t0);
RWByteAddressBuffer x : register(u0);
RWByteAddressBuffer y : register(u1);

groupshared float s_mem[16 * 128];

[numthreads(16, 8, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint row_in_grp = gtid.y;
    uint warp_id = gid.x * 8u + row_in_grp;
    uint lane_id = gtid.x;
    uint s_idx = row_in_grp * 16u + lane_id;

    float acc[16];
    [unroll] for (uint t = 0u; t < 16u; t++) acc[t] = 0.0f;

    if (warp_id < m_rows)
    {
        uint row_base = warp_id * k_cols;
        uint c4_end = k_cols & ~3u;
        for (uint c = lane_id * 4u; c < c4_end; c += 64u)
        {
            float4 w4 = asfloat(W.Load4((row_base + c) * 4u));
            for (uint t = 0u; t < batch_size; t++)
            {
                float4 x4 = asfloat(x.Load4((t * k_cols + c) * 4u));
                acc[t] += dot(w4, x4);
            }
        }
        if (c4_end < k_cols)
        {
            for (uint c = c4_end + lane_id; c < k_cols; c += 16u)
            {
                float w = asfloat(W.Load((row_base + c) * 4u));
                for (uint t = 0u; t < batch_size; t++)
                {
                    acc[t] += w * asfloat(x.Load((t * k_cols + c) * 4u));
                }
            }
        }
    }

    for (uint t = 0u; t < batch_size; t++) s_mem[t * 128u + s_idx] = acc[t];
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 8u) { for (uint t = 0u; t < batch_size; t++) s_mem[t * 128u + s_idx] += s_mem[t * 128u + s_idx + 8u]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 4u) { for (uint t = 0u; t < batch_size; t++) s_mem[t * 128u + s_idx] += s_mem[t * 128u + s_idx + 4u]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 2u) { for (uint t = 0u; t < batch_size; t++) s_mem[t * 128u + s_idx] += s_mem[t * 128u + s_idx + 2u]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 1u) { for (uint t = 0u; t < batch_size; t++) s_mem[t * 128u + s_idx] += s_mem[t * 128u + s_idx + 1u]; }

    if (lane_id == 0u && warp_id < m_rows)
    {
        for (uint t = 0u; t < batch_size; t++)
        {
            y.Store((t * m_rows + warp_id) * 4u, asuint(s_mem[t * 128u + row_in_grp * 16u]));
        }
    }
}
";

    // 5. Batched 2D-Indexed Indirect NVFP4 MoE GEMV: gid.y = pair_idx (0 .. B*8 - 1)!
    //    [numthreads(16, 16, 1)]: 16 rows/group (2x s_x reuse), branchless decode_fp4, zero warp-tail waste!
    public const string Inkling2GemvNVFP4IndirectBatch = @"
cbuffer Params : register(b0)
{
    uint k_cols;
    uint m_rows;
    uint x_stride;
    uint y_stride;
    uint exp_stride_bytes;
    uint x_pair_div;
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
    uint pair_idx = gid.y;
    uint row_in_grp = gtid.y;
    uint warp_id = gid.x * 16u + row_in_grp;
    uint lane_id = gtid.x;
    uint s_idx = row_in_grp * 16u + lane_id;

    uint x_off = (pair_idx / x_pair_div) * x_stride;
    uint y_off = pair_idx * y_stride;

    for (uint ci = s_idx; ci < k_cols; ci += 256u)
    {
        s_x[ci] = asfloat(x.Load((x_off + ci) * 4u));
    }
    GroupMemoryBarrierWithGroupSync();

    if (warp_id < m_rows)
    {
        uint exp_id = topk_ids.Load(pair_idx * 4u);
        uint nb = k_cols >> 6;
        uint row_byte_offset = exp_id * exp_stride_bytes + warp_id * nb * 36u;
        float row_sum = 0.0f;

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

    // 6. Batched RMSNorm: gid.y = tokenIdx (0..B-1). Dispatch: (1, B, 1).
    public const string Inkling2RmsNormBatch = @"
cbuffer Params : register(b0)
{
    uint dim;
    float eps;
    uint in_stride;
    uint in_sub_off;
    uint out_stride;
    uint out_sub_off;
    uint has_weight;
    uint pad0;
};

ByteAddressBuffer weight : register(t0);
RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);

groupshared float s_sum[256];

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint t = gid.y;
    uint tid = gtid.x;
    uint i_base = t * in_stride + in_sub_off;
    uint o_base = t * out_stride + out_sub_off;

    float sumSq = 0.0f;
    for (uint i = tid; i < dim; i += 256u)
    {
        float v = asfloat(src.Load((i_base + i) * 4u));
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
        float v = asfloat(src.Load((i_base + i) * 4u)) * inv;
        if (has_weight != 0u) v *= asfloat(weight.Load(i * 4u));
        dst.Store((o_base + i) * 4u, asuint(v));
    }
}
";

    // 7. Batched Block-Attention Residual Mix: gid.x = queryIdx, gid.y = tokenIdx (0..B-1).
    //    blocks_buf layout: [blockIdx * batch_stride_floats + tokenIdx * dim + d]
    //    out_buf layout:    [tokenIdx * out_stride_floats + out_sub_off + queryIdx * dim + d]
    public const string Inkling2MixBatch = @"
cbuffer Params : register(b0)
{
    uint dim;
    uint n_blocks;
    uint query_base_floats;
    uint out_stride_floats;
    uint out_sub_off;
    uint block_stride_floats;
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
    uint t = gid.y;
    uint tid = gtid.x;
    uint n_states = n_blocks + 1u;
    float inv_sqrt_d = rsqrt((float)dim);
    uint q_off = query_base_floats + q * dim;
    uint cur_off = t * dim;

    for (uint s = 0u; s < n_states; s++)
    {
        float sumSq = 0.0f;
        float dotVal = 0.0f;
        uint blk_off = s * block_stride_floats + t * dim;

        for (uint d = tid; d < dim; d += 256u)
        {
            float v = (s < n_blocks)
                ? asfloat(blocks_buf.Load((blk_off + d) * 4u))
                : asfloat(cur_buf.Load((cur_off + d) * 4u));
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

    uint dst_base = t * out_stride_floats + out_sub_off + q * dim;
    for (uint d = tid; d < dim; d += 256u)
    {
        float acc = 0.0f;
        for (uint s = 0u; s < n_states; s++)
        {
            float v = (s < n_blocks)
                ? asfloat(blocks_buf.Load((s * block_stride_floats + t * dim + d) * 4u))
                : asfloat(cur_buf.Load((cur_off + d) * 4u));
            acc += s_weights[s] * v;
        }
        out_buf.Store((dst_base + d) * 4u, asuint(acc));
    }
}
";

    // 8. Batched Cross-Pass Loop Control Blend: gid.y = tokenIdx (0..B-1).
    public const string Inkling2LoopCtrlBlendBatch = @"
cbuffer Params : register(b0)
{
    uint dim;
    uint msg_sub_off;
    uint msg_stride;
    uint pad0;
};

RWByteAddressBuffer cur_buf : register(u0);
RWByteAddressBuffer msg_buf : register(u1);
RWByteAddressBuffer gain_buf : register(u2);
RWByteAddressBuffer forget_buf : register(u3);

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID, uint3 gid : SV_GroupID)
{
    uint d = dtid.x;
    uint t = gid.y;
    if (d >= dim) return;
    uint off = t * dim + d;
    float c = asfloat(cur_buf.Load(off * 4u));
    float m = asfloat(msg_buf.Load((t * msg_stride + msg_sub_off + d) * 4u));
    float g = tanh(asfloat(gain_buf.Load(off * 4u)));
    float f = tanh(asfloat(forget_buf.Load(off * 4u)));
    cur_buf.Store(off * 4u, asuint((1.0f - f) * c + g * m));
}
";

    // 9. Register-Resident Causal Short-Conv Batch: each channel thread loops t=0..B-1 in registers!
    public const string Inkling2ShortConvBatch = @"
cbuffer Params : register(b0)
{
    uint channels;
    uint hist_base_floats;
    uint has_dyn;
    uint head_dim;
    uint batch_size;
    uint dyn_stride;
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

    uint w_off = c * 4u;
    float w0 = asfloat(conv_w.Load((w_off + 0u) * 4u));
    float w1 = asfloat(conv_w.Load((w_off + 1u) * 4u));
    float w2 = asfloat(conv_w.Load((w_off + 2u) * 4u));
    float w3 = asfloat(conv_w.Load((w_off + 3u) * 4u));

    float scale = (has_dyn != 0u) ? rsqrt(9.0f) : rsqrt(5.0f);
    uint h = (has_dyn != 0u) ? (c / head_dim) : 0u;

    for (uint t = 0u; t < batch_size; t++)
    {
        uint cur_idx = t * channels + c;
        float x3 = asfloat(cur_buf.Load(cur_idx * 4u));
        float conv = w0 * x0 + w1 * x1 + w2 * x2 + w3 * x3;

        if (has_dyn != 0u)
        {
            uint d_off = t * dyn_stride + h * 4u;
            conv += asfloat(dyn_buf.Load((d_off + 0u) * 4u)) * x3
                  + asfloat(dyn_buf.Load((d_off + 1u) * 4u)) * x2
                  + asfloat(dyn_buf.Load((d_off + 2u) * 4u)) * x1
                  + asfloat(dyn_buf.Load((d_off + 3u) * 4u)) * x0;
        }

        cur_buf.Store(cur_idx * 4u, asuint((x3 + conv) * scale));
        x0 = x1;
        x1 = x2;
        x2 = x3;
    }

    hist_buf.Store((h_off + 0u) * 4u, asuint(x0));
    hist_buf.Store((h_off + 1u) * 4u, asuint(x1));
    hist_buf.Store((h_off + 2u) * 4u, asuint(x2));
}
";

    // 10. Batched Per-Head L2 Clip / RMSNorm + tau(pos+t) + FP16 KV Cache Write!
    //     Dispatch: (nHeads, B, 1) with 128 threads/head.
    public const string Inkling2HeadNormClipKvBatch = @"
cbuffer Params : register(b0)
{
    uint dim;
    uint mode;
    uint has_scale;
    float thresh_or_eps;
    uint start_pos;
    uint kv_cap;
    uint kv_layer_base_u32;
    uint write_kv;
    uint apply_tau;
    float log_scale_alpha;
    float log_scale_floor;
    uint n_heads;
};

ByteAddressBuffer scale_buf : register(t0);
RWByteAddressBuffer buf : register(u0);
RWByteAddressBuffer kv_fp16 : register(u1);

groupshared float s_sq[128];

[numthreads(128, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint h = gid.x;
    uint t = gid.y;
    uint tid = gtid.x;
    uint head_stride = n_heads * dim;
    uint off = t * head_stride + h * dim;
    uint sc_off = h * dim;

    float v = 0.0f;
    if (tid < dim)
    {
        v = asfloat(buf.Load((off + tid) * 4u));
        if (mode == 0u && has_scale != 0u) v *= asfloat(scale_buf.Load((sc_off + tid) * 4u));
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
        if (apply_tau != 0u)
        {
            float pos_f = (float)(start_pos + t) + 1.0f;
            float tau = 1.0f + log_scale_alpha * log(max(1.0f, pos_f / log_scale_floor));
            mult *= tau;
        }
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

    GroupMemoryBarrierWithGroupSync();

    if (write_kv != 0u)
    {
        s_sq[tid] = outV;
        GroupMemoryBarrierWithGroupSync();
        if (tid < (dim >> 1))
        {
            uint pos = start_pos + t;
            uint slot = pos % kv_cap;
            uint kv_dim_half = head_stride >> 1;
            uint kv_slot_u32 = kv_layer_base_u32 + slot * kv_dim_half;
            uint lo = f32tof16(s_sq[tid * 2u]);
            uint hi = f32tof16(s_sq[tid * 2u + 1u]);
            uint packed = (lo & 0xFFFFu) | (hi << 16u);
            kv_fp16.Store((kv_slot_u32 + (h * (dim >> 1)) + tid) * 4u, packed);
        }
    }
}
";

    // 11. Batched Causal FlashAttention: Dispatch (nHead, B, 1) with 128 threads/head!
    public const string Inkling2AttnBatch = @"
cbuffer Params : register(b0)
{
    uint start_pos;
    uint swa_window;
    uint is_swa;
    uint kv_cap;
    uint kv_dim_half;
    uint group_size;
    uint rel_extent;
    uint n_rel;
    uint kv_layer_base_u32;
    uint n_head;
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
    uint tok = gid.y;
    uint d = gtid.x;
    uint pos = start_pos + tok;
    uint start_t = (is_swa != 0u && pos + 1u > swa_window) ? (pos + 1u - swa_window) : 0u;

    uint kv_head = h / group_size;
    uint q_off = (tok * n_head + h) * 128u;
    uint r_off = (tok * n_head + h) * n_rel;

    s_q[d] = asfloat(q_buf.Load((q_off + d) * 4u));
    if (d < n_rel) s_r[d] = asfloat(r_buf.Load((r_off + d) * 4u));
    float sink_val = asfloat(sinks_buf.Load((tok * n_head + h) * 4u));
    GroupMemoryBarrierWithGroupSync();

    float m_max = 0.0f;
    float l_sum = 1.0f;
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

    // 12. Batched Router Top-K: Dispatch (1, B, 1) with 128 threads/token!
    //     Writes topk_ids[t*8 + k], meta_buf[t*8 + k] = guScale, meta_buf[B*8 + t*8 + k] = weight * dnScale.
    public const string Inkling2RouterTopKBatch = @"
cbuffer Params : register(b0)
{
    uint n_exp;
    uint top_k;
    float weight_scale;
    uint has_bias;
    uint has_gu_scale;
    uint has_dn_scale;
    uint batch_size;
    uint pad0;
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
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint t = gid.y;
    uint e = gtid.x;
    if (e < n_exp)
    {
        float p = 1.0f / (1.0f + exp(-asfloat(logits_buf.Load((t * n_exp + e) * 4u))));
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
        uint base_pair = t * 8u;
        uint total_pairs = batch_size * 8u;

        [unroll]
        for (uint k2 = 0u; k2 < 8u; k2++)
        {
            uint id = ids[k2];
            float gs = (has_gu_scale != 0u) ? asfloat(gu_scales.Load(id * 4u)) : 1.0f;
            float ds = (has_dn_scale != 0u) ? asfloat(dn_scales.Load(id * 4u)) : 1.0f;
            topk_ids.Store((base_pair + k2) * 4u, id);
            meta_buf.Store((base_pair + k2) * 4u, asuint(gs));
            meta_buf.Store((total_pairs + base_pair + k2) * 4u, asuint(probs[k2] * w_mul * ds));
        }
    }
}
";
}
#endif
