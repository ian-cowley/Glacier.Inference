namespace Glacier.Inference.Gpu.D3D12;

/// <summary>
/// Direct3D 12 HLSL Compute Shaders (cs_5_0) for additional GGUF quantizations:
/// - IQ4_XS (256-element non-linear 4-bit super-blocks, 136 bytes/block)
/// - MXFP4 (32-element E2M1 FP4 + E8M0 blocks, 20 bytes aligned)
/// - Q4_0 (32-element signed 4-bit blocks, 20 bytes aligned)
/// - Q5_0 (32-element signed 5-bit blocks, 24 bytes aligned)
/// - BF16 (compact 2-byte brain-float rows, 50% VRAM savings vs F32)
/// </summary>
public static partial class D3D12Shaders
{
    public const string GemvIQ4_XS = @"
cbuffer Params : register(b0)
{
    uint k_cols;
    uint m_rows;
    uint has_bias;
    uint has_residual;
    uint has_y;
    uint row_offset;
};

ByteAddressBuffer W : register(t0);
ByteAddressBuffer bias : register(t1);
RWByteAddressBuffer x : register(u0);
RWByteAddressBuffer residual : register(u1);
RWByteAddressBuffer y : register(u2);

groupshared float s_mem[128];
" + IQ4_NL_Table + @"
[numthreads(32, 4, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint row_in_grp = gtid.y;
    uint warp_id = row_offset + gid.x * 4 + row_in_grp;
    uint lane_id = gtid.x;
    uint s_idx = row_in_grp * 32 + lane_id;

    if (warp_id < m_rows)
    {
        uint nb = k_cols / 256u;
        uint row_byte_offset = warp_id * nb * 136u;
        float row_sum = 0.0f;

        uint ib = lane_id >> 2;
        uint quad = lane_id & 3u;
        uint x_base_lane = ib * 32u + quad * 4u;

        for (uint b = 0u; b < nb; b++)
        {
            uint blk_addr = row_byte_offset + b * 136u;
            uint w0 = W.Load(blk_addr);
            float d = f16tof32(w0 & 0xFFFFu);
            uint scales_h = w0 >> 16;
            uint scales_l = W.Load(blk_addr + 4u);
            uint ls = ((scales_l >> (ib * 4u)) & 0xFu) | (((scales_h >> (ib * 2u)) & 3u) << 4);
            float dl = d * (float)((int)ls - 32);

            uint q4 = W.Load(blk_addr + 8u + ib * 16u + quad * 4u);
            uint col_lo = b * 256u + x_base_lane;
            float4 x_lo = asfloat(x.Load4(col_lo * 4u));
            float4 x_hi = asfloat(x.Load4((col_lo + 16u) * 4u));

            float dot_lo =
                kvalues_iq4nl[q4 & 0xFu] * x_lo.x +
                kvalues_iq4nl[(q4 >> 8) & 0xFu] * x_lo.y +
                kvalues_iq4nl[(q4 >> 16) & 0xFu] * x_lo.z +
                kvalues_iq4nl[(q4 >> 24) & 0xFu] * x_lo.w;

            float dot_hi =
                kvalues_iq4nl[(q4 >> 4) & 0xFu] * x_hi.x +
                kvalues_iq4nl[(q4 >> 12) & 0xFu] * x_hi.y +
                kvalues_iq4nl[(q4 >> 20) & 0xFu] * x_hi.z +
                kvalues_iq4nl[(q4 >> 28) & 0xFu] * x_hi.w;

            row_sum += dl * (dot_lo + dot_hi);
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
        float final_sum = s_mem[row_in_grp * 32u];
        if (has_bias != 0u) final_sum += asfloat(bias.Load(warp_id * 4u));
        if (has_residual != 0u)
        {
            float prev = asfloat(residual.Load(warp_id * 4u));
            residual.Store(warp_id * 4u, asuint(prev + final_sum));
        }
        if (has_y != 0u) y.Store(warp_id * 4u, asuint(final_sum));
    }
}
";

    public const string GemmIQ4_XSBatch = @"
cbuffer Params : register(b0)
{
    uint k_cols;
    uint m_rows;
    uint batch_size;
    uint has_bias;
    uint has_residual;
    uint has_y;
};

ByteAddressBuffer W : register(t0);
ByteAddressBuffer bias : register(t1);
RWByteAddressBuffer x : register(u2);
RWByteAddressBuffer residual : register(u0);
RWByteAddressBuffer y : register(u1);

groupshared float s_mem[4096];
" + IQ4_NL_Table + @"
[numthreads(32, 4, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint row_in_grp = gtid.y;
    uint warp_id = gid.x * 4u + row_in_grp;
    uint lane_id = gtid.x;
    uint s_idx = row_in_grp * 32u + lane_id;
    uint t_base = gid.y * 32u;

    float acc[32];
    [unroll]
    for (uint i = 0u; i < 32u; i++) acc[i] = 0.0f;

    if (warp_id < m_rows)
    {
        uint nb = k_cols / 256u;
        uint row_byte_offset = warp_id * nb * 136u;
        uint ib = lane_id >> 2;
        uint quad = lane_id & 3u;
        uint x_base_lane = ib * 32u + quad * 4u;

        for (uint b = 0u; b < nb; b++)
        {
            uint blk_addr = row_byte_offset + b * 136u;
            uint w0 = W.Load(blk_addr);
            float d = f16tof32(w0 & 0xFFFFu);
            uint scales_h = w0 >> 16;
            uint scales_l = W.Load(blk_addr + 4u);
            uint ls = ((scales_l >> (ib * 4u)) & 0xFu) | (((scales_h >> (ib * 2u)) & 3u) << 4);
            float dl = d * (float)((int)ls - 32);

            uint q4 = W.Load(blk_addr + 8u + ib * 16u + quad * 4u);
            float4 w_lo = dl * float4(
                kvalues_iq4nl[q4 & 0xFu],
                kvalues_iq4nl[(q4 >> 8) & 0xFu],
                kvalues_iq4nl[(q4 >> 16) & 0xFu],
                kvalues_iq4nl[(q4 >> 24) & 0xFu]);
            float4 w_hi = dl * float4(
                kvalues_iq4nl[(q4 >> 4) & 0xFu],
                kvalues_iq4nl[(q4 >> 12) & 0xFu],
                kvalues_iq4nl[(q4 >> 20) & 0xFu],
                kvalues_iq4nl[(q4 >> 28) & 0xFu]);

            uint col_lo = b * 256u + x_base_lane;

            [unroll]
            for (uint i = 0u; i < 32u; i++)
            {
                uint t = t_base + i;
                uint safe_t = (t < batch_size) ? t : 0u;
                uint row_base = safe_t * k_cols + col_lo;
                float4 xv0 = asfloat(x.Load4(row_base * 4u));
                float4 xv1 = asfloat(x.Load4((row_base + 16u) * 4u));
                if (t < batch_size)
                {
                    acc[i] += dot(w_lo, xv0) + dot(w_hi, xv1);
                }
            }
        }
    }

    [unroll]
    for (uint i = 0u; i < 32u; i++)
    {
        uint t = t_base + i;
        s_mem[i * 128u + s_idx] = (t < batch_size && warp_id < m_rows) ? acc[i] : 0.0f;
    }

    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 16u) { [unroll] for (uint i = 0u; i < 32u; i++) s_mem[i * 128u + s_idx] += s_mem[i * 128u + s_idx + 16u]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 8u)  { [unroll] for (uint i = 0u; i < 32u; i++) s_mem[i * 128u + s_idx] += s_mem[i * 128u + s_idx + 8u]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 4u)  { [unroll] for (uint i = 0u; i < 32u; i++) s_mem[i * 128u + s_idx] += s_mem[i * 128u + s_idx + 4u]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 2u)  { [unroll] for (uint i = 0u; i < 32u; i++) s_mem[i * 128u + s_idx] += s_mem[i * 128u + s_idx + 2u]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 1u)  { [unroll] for (uint i = 0u; i < 32u; i++) s_mem[i * 128u + s_idx] += s_mem[i * 128u + s_idx + 1u]; }

    if (lane_id == 0u && warp_id < m_rows)
    {
        float b_val = 0.0f;
        if (has_bias != 0u) b_val = asfloat(bias.Load(warp_id * 4u));
        [unroll]
        for (uint i = 0u; i < 32u; i++)
        {
            uint t = t_base + i;
            if (t < batch_size)
            {
                float final_sum = s_mem[i * 128u + row_in_grp * 32u] + b_val;
                uint out_byte_addr = (t * m_rows + warp_id) * 4u;
                if (has_residual != 0u)
                {
                    float prev = asfloat(residual.Load(out_byte_addr));
                    residual.Store(out_byte_addr, asuint(prev + final_sum));
                }
                if (has_y != 0u)
                {
                    y.Store(out_byte_addr, asuint(final_sum));
                }
            }
        }
    }
}
";

    public const string GemvMXFP4 = @"
cbuffer Params : register(b0)
{
    uint k_cols;
    uint m_rows;
    uint has_bias;
    uint has_residual;
    uint has_y;
    uint row_offset;
};

ByteAddressBuffer W : register(t0);
ByteAddressBuffer bias : register(t1);
RWByteAddressBuffer x : register(u0);
RWByteAddressBuffer residual : register(u1);
RWByteAddressBuffer y : register(u2);

static const float kvalues_mxfp4[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};
groupshared float s_mem[128];

[numthreads(32, 4, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint row_in_grp = gtid.y;
    uint warp_id = row_offset + gid.x * 4u + row_in_grp;
    uint lane_id = gtid.x;
    uint s_idx = row_in_grp * 32u + lane_id;

    if (warp_id < m_rows)
    {
        uint nb = k_cols / 32u;
        uint row_byte_offset = warp_id * nb * 20u;
        float row_sum = 0.0f;

        uint l = lane_id & 15u;
        uint is_high = lane_id >> 4;
        uint q_word_off = 4u + (l & ~3u);
        uint q_byte_shift = (l & 3u) * 8u;

        for (uint b = 0u; b < nb; b++)
        {
            uint blk_addr = row_byte_offset + b * 20u;
            uint e = W.Load(blk_addr) & 0xFFu;
            float d = (e == 0u) ? asfloat(0x00400000u) : ((e == 255u) ? 0.0f : asfloat(e << 23));

            uint q_word = W.Load(blk_addr + q_word_off);
            uint q_byte = (q_word >> q_byte_shift) & 0xFFu;
            uint nib = (is_high != 0u) ? (q_byte >> 4) : (q_byte & 0x0Fu);

            float w = d * kvalues_mxfp4[nib];
            row_sum += w * asfloat(x.Load((b * 32u + lane_id) * 4u));
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
        float final_sum = s_mem[row_in_grp * 32u];
        if (has_bias != 0u) final_sum += asfloat(bias.Load(warp_id * 4u));
        if (has_residual != 0u)
        {
            float prev = asfloat(residual.Load(warp_id * 4u));
            residual.Store(warp_id * 4u, asuint(prev + final_sum));
        }
        if (has_y != 0u) y.Store(warp_id * 4u, asuint(final_sum));
    }
}
";

    public const string GemvNVFP4 = @"
cbuffer Params : register(b0)
{
    uint k_cols;
    uint m_rows;
    uint has_bias;
    uint has_residual;
    uint has_y;
    uint row_offset;
};

ByteAddressBuffer W : register(t0);
ByteAddressBuffer bias : register(t1);
RWByteAddressBuffer x : register(u0);
RWByteAddressBuffer residual : register(u1);
RWByteAddressBuffer y : register(u2);

static const float kvalues_nvfp4[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};
groupshared float s_mem[128];

float decode_ue4m3(uint b)
{
    b &= 0x7Fu;
    if (b == 0u || b == 127u) return 0.0f;
    uint exp_bits = b >> 3;
    uint man_bits = b & 7u;
    if (exp_bits == 0u) return (float)man_bits * 0.001953125f;
    return asfloat(((exp_bits + 120u) << 23) | (man_bits << 20));
}

[numthreads(32, 4, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint row_in_grp = gtid.y;
    uint warp_id = row_offset + gid.x * 4u + row_in_grp;
    uint lane_id = gtid.x;
    uint s_idx = row_in_grp * 32u + lane_id;

    if (warp_id < m_rows)
    {
        uint nb = k_cols / 64u;
        uint row_byte_offset = warp_id * nb * 36u;
        float row_sum = 0.0f;

        uint sub = lane_id >> 3;
        uint j = lane_id & 7u;
        uint q_word_off = 4u + (lane_id & ~3u);
        uint q_byte_shift = (lane_id & 3u) * 8u;

        for (uint b = 0u; b < nb; b++)
        {
            uint blk_addr = row_byte_offset + b * 36u;
            uint scales_u32 = W.Load(blk_addr);
            float d = decode_ue4m3((scales_u32 >> (sub * 8u)) & 0xFFu);

            uint q_word = W.Load(blk_addr + q_word_off);
            uint q_byte = (q_word >> q_byte_shift) & 0xFFu;
            float w0 = d * kvalues_nvfp4[q_byte & 0xFu];
            float w1 = d * kvalues_nvfp4[q_byte >> 4];

            uint col0 = b * 64u + sub * 16u + j;
            float x0 = asfloat(x.Load(col0 * 4u));
            float x1 = asfloat(x.Load((col0 + 8u) * 4u));
            row_sum += w0 * x0 + w1 * x1;
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
        float final_sum = s_mem[row_in_grp * 32u];
        if (has_bias != 0u) final_sum += asfloat(bias.Load(warp_id * 4u));
        if (has_residual != 0u)
        {
            float prev = asfloat(residual.Load(warp_id * 4u));
            residual.Store(warp_id * 4u, asuint(prev + final_sum));
        }
        if (has_y != 0u) y.Store(warp_id * 4u, asuint(final_sum));
    }
}
";

    public const string Inkling2GemvNVFP4 = @"
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

static const float kvalues_nvfp4[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};
groupshared float s_x[2304];
groupshared float s_mem[128];

float decode_ue4m3(uint b)
{
    b &= 0x7Fu;
    if (b == 0u || b == 127u) return 0.0f;
    uint exp_bits = b >> 3;
    uint man_bits = b & 7u;
    if (exp_bits == 0u) return (float)man_bits * 0.001953125f;
    return asfloat(((exp_bits + 120u) << 23) | (man_bits << 20));
}

[numthreads(32, 4, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint row_in_grp = gtid.y;
    uint warp_id = gid.x * 4u + row_in_grp;
    uint lane_id = gtid.x;
    uint s_idx = row_in_grp * 32u + lane_id;

    // Cooperative coalesced load of x[0..k_cols) into on-chip LDS/Shared Memory
    for (uint ci = s_idx; ci < k_cols; ci += 128u)
    {
        s_x[ci] = asfloat(x.Load((x_offset + ci) * 4u));
    }
    GroupMemoryBarrierWithGroupSync();

    if (warp_id < m_rows)
    {
        uint nb = k_cols / 64u;
        uint row_byte_offset = warp_id * nb * 36u;
        float row_sum = 0.0f;

        uint sub = lane_id >> 3;
        uint j = lane_id & 7u;
        uint q_word_off = 4u + (lane_id & ~3u);
        uint q_byte_shift = (lane_id & 3u) * 8u;

        for (uint b = 0u; b < nb; b++)
        {
            uint blk_addr = row_byte_offset + b * 36u;
            uint scales_u32 = W.Load(blk_addr);
            float d = decode_ue4m3((scales_u32 >> (sub * 8u)) & 0xFFu);

            uint q_word = W.Load(blk_addr + q_word_off);
            uint q_byte = (q_word >> q_byte_shift) & 0xFFu;
            float w0 = d * kvalues_nvfp4[q_byte & 0xFu];
            float w1 = d * kvalues_nvfp4[q_byte >> 4];

            uint col0 = b * 64u + sub * 16u + j;
            row_sum += w0 * s_x[col0] + w1 * s_x[col0 + 8u];
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

    public const string Inkling2GemvQ6K = @"
cbuffer Params : register(b0)
{
    uint k_cols;
    uint m_rows;
    uint x_offset;
    uint row_offset;
};

ByteAddressBuffer W : register(t0);
RWByteAddressBuffer x : register(u0);
RWByteAddressBuffer y : register(u1);

groupshared float s_x[3072];
groupshared float s_mem[256];

int decode_s8(uint b)
{
    return (int)(b << 24) >> 24;
}

[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint row_in_grp = gtid.y;
    uint warp_id = row_offset + gid.x * 16u + row_in_grp;
    uint lane_id = gtid.x;
    uint s_idx = row_in_grp * 16u + lane_id;

    for (uint ci = s_idx; ci < k_cols; ci += 256u)
    {
        s_x[ci] = asfloat(x.Load((x_offset + ci) * 4u));
    }
    GroupMemoryBarrierWithGroupSync();

    if (warp_id < m_rows)
    {
        uint nb = k_cols / 256u;
        uint row_byte_offset = warp_id * nb * 212u;
        float row_sum = 0.0f;

        uint b_lane = lane_id >> 3;
        uint sub_lane = lane_id & 7u;
        uint is_idx = sub_lane >> 2;
        uint byte_off = sub_lane * 4u;

        for (uint b = b_lane; b < nb; b += 2u)
        {
            uint blk_addr = row_byte_offset + b * 212u;
            float d = f16tof32(W.Load(blk_addr + 208u) & 0xFFFFu);
            uint x_blk = b * 256u;

            uint4 sc = W.Load4(blk_addr + 192u);
            float blk_sum = 0.0f;

            [unroll]
            for (uint step = 0u; step < 2u; step++)
            {
                uint ql_addr = blk_addr + step * 64u;
                uint qh_addr = blk_addr + 128u + step * 32u;

                uint qh4 = W.Load(qh_addr + byte_off);
                uint qlLo4 = W.Load(ql_addr + byte_off);
                uint qlHi4 = W.Load(ql_addr + 32u + byte_off);

                uint2 sc_words = (step == 0u) ? sc.xy : sc.zw;
                float fs1 = (float)decode_s8((sc_words.x >> (is_idx * 8u)) & 0xFFu);
                float fs2 = (float)decode_s8((sc_words.x >> ((is_idx + 2u) * 8u)) & 0xFFu);
                float fs3 = (float)decode_s8((sc_words.y >> (is_idx * 8u)) & 0xFFu);
                float fs4 = (float)decode_s8((sc_words.y >> ((is_idx + 2u) * 8u)) & 0xFFu);

                float sum1 = 0.0f;
                float sum2 = 0.0f;
                float sum3 = 0.0f;
                float sum4 = 0.0f;
                uint base_x = x_blk + step * 128u + byte_off;

                [unroll]
                for (uint c = 0u; c < 4u; c++)
                {
                    uint sh = c * 8u;
                    uint qh_val = (qh4 >> sh) & 0xFFu;
                    uint ql_low = (qlLo4 >> sh) & 0xFFu;
                    uint ql_high = (qlHi4 >> sh) & 0xFFu;

                    int q1 = (int)((ql_low & 0x0Fu) | (((qh_val >> 0u) & 3u) << 4u)) - 32;
                    int q2 = (int)((ql_high & 0x0Fu) | (((qh_val >> 2u) & 3u) << 4u)) - 32;
                    int q3 = (int)((ql_low >> 4u) | (((qh_val >> 4u) & 3u) << 4u)) - 32;
                    int q4 = (int)((ql_high >> 4u) | (((qh_val >> 6u) & 3u) << 4u)) - 32;

                    uint xi = base_x + c;
                    sum1 += (float)q1 * s_x[xi + 0u];
                    sum2 += (float)q2 * s_x[xi + 32u];
                    sum3 += (float)q3 * s_x[xi + 64u];
                    sum4 += (float)q4 * s_x[xi + 96u];
                }

                blk_sum += fs1 * sum1 + fs2 * sum2 + fs3 * sum3 + fs4 * sum4;
            }

            row_sum += d * blk_sum;
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
        y.Store(warp_id * 4u, asuint(s_mem[row_in_grp * 16u]));
    }
}
";

    public const string Inkling2SituBatch = @"
cbuffer Params : register(b0)
{
    uint n_ff;
    uint k_count;
    float beta;
    float lb;
};

RWByteAddressBuffer gate_up : register(u0);
RWByteAddressBuffer hidden_out : register(u1);
RWByteAddressBuffer meta : register(u2);

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    uint idx = dtid.x;
    uint total = n_ff * k_count;
    if (idx >= total) return;

    uint k = idx / n_ff;
    uint j = idx - k * n_ff;
    float gu_scale = asfloat(meta.Load(k * 4u));

    uint gu_base = k * (2u * n_ff);
    float g = asfloat(gate_up.Load((gu_base + j) * 4u)) * gu_scale;
    float u = asfloat(gate_up.Load((gu_base + n_ff + j) * 4u)) * gu_scale;
    float act = beta * tanh(g / beta) / (1.0f + exp(-g));
    if (lb > 0.0f) u = lb * tanh(u / lb);
    hidden_out.Store(idx * 4u, asuint(act * u));
}
";

    public const string Inkling2WeightedSum8 = @"
cbuffer Params : register(b0)
{
    uint n_embd;
    uint k_count;
    uint weights_offset;
    uint batch_size;
};

RWByteAddressBuffer dn_all : register(u0);
RWByteAddressBuffer dst : register(u1);
RWByteAddressBuffer meta : register(u2);

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID, uint3 gid : SV_GroupID)
{
    uint d = dtid.x;
    uint t = gid.y;
    if (d >= n_embd || t >= batch_size) return;

    uint base_k = t * k_count;
    float sum = 0.0f;
    for (uint k = 0u; k < k_count; k++)
    {
        float w = asfloat(meta.Load((weights_offset + base_k + k) * 4u));
        sum += asfloat(dn_all.Load(((base_k + k) * n_embd + d) * 4u)) * w;
    }
    dst.Store((t * n_embd + d) * 4u, asuint(sum));
}
";

    public const string GemvQ4_0 = @"
cbuffer Params : register(b0)
{
    uint k_cols;
    uint m_rows;
    uint has_bias;
    uint has_residual;
    uint has_y;
    uint row_offset;
};

ByteAddressBuffer W : register(t0);
ByteAddressBuffer bias : register(t1);
RWByteAddressBuffer x : register(u0);
RWByteAddressBuffer residual : register(u1);
RWByteAddressBuffer y : register(u2);

groupshared float s_mem[128];

[numthreads(32, 4, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint row_in_grp = gtid.y;
    uint warp_id = row_offset + gid.x * 4u + row_in_grp;
    uint lane_id = gtid.x;
    uint s_idx = row_in_grp * 32u + lane_id;

    if (warp_id < m_rows)
    {
        uint nb = k_cols / 32u;
        uint row_byte_offset = warp_id * nb * 20u;
        float row_sum = 0.0f;

        uint l = lane_id & 15u;
        uint is_high = lane_id >> 4;
        uint q_word_off = 4u + (l & ~3u);
        uint q_byte_shift = (l & 3u) * 8u;

        for (uint b = 0u; b < nb; b++)
        {
            uint blk_addr = row_byte_offset + b * 20u;
            float d = f16tof32(W.Load(blk_addr) & 0xFFFFu);

            uint q_word = W.Load(blk_addr + q_word_off);
            uint q_byte = (q_word >> q_byte_shift) & 0xFFu;
            uint nib = (is_high != 0u) ? (q_byte >> 4) : (q_byte & 0x0Fu);

            float w = d * (float)((int)nib - 8);
            row_sum += w * asfloat(x.Load((b * 32u + lane_id) * 4u));
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
        float final_sum = s_mem[row_in_grp * 32u];
        if (has_bias != 0u) final_sum += asfloat(bias.Load(warp_id * 4u));
        if (has_residual != 0u)
        {
            float prev = asfloat(residual.Load(warp_id * 4u));
            residual.Store(warp_id * 4u, asuint(prev + final_sum));
        }
        if (has_y != 0u) y.Store(warp_id * 4u, asuint(final_sum));
    }
}
";

    public const string GemmQ4_0Batch = @"
cbuffer Params : register(b0)
{
    uint k_cols;
    uint m_rows;
    uint batch_size;
    uint has_bias;
    uint has_residual;
    uint has_y;
};

ByteAddressBuffer W : register(t0);
ByteAddressBuffer bias : register(t1);
RWByteAddressBuffer x : register(u2);
RWByteAddressBuffer residual : register(u0);
RWByteAddressBuffer y : register(u1);

groupshared float s_mem[4096];

[numthreads(32, 4, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint row_in_grp = gtid.y;
    uint warp_id = gid.x * 4u + row_in_grp;
    uint lane_id = gtid.x;
    uint s_idx = row_in_grp * 32u + lane_id;
    uint t_base = gid.y * 32u;

    float acc[32];
    [unroll]
    for (uint i = 0u; i < 32u; i++) acc[i] = 0.0f;

    if (warp_id < m_rows)
    {
        uint nb = k_cols / 32u;
        uint row_byte_offset = warp_id * nb * 20u;

        uint l = lane_id & 15u;
        uint is_high = lane_id >> 4;
        uint q_word_off = 4u + (l & ~3u);
        uint q_byte_shift = (l & 3u) * 8u;

        for (uint b = 0u; b < nb; b++)
        {
            uint blk_addr = row_byte_offset + b * 20u;
            float d = f16tof32(W.Load(blk_addr) & 0xFFFFu);

            uint q_word = W.Load(blk_addr + q_word_off);
            uint q_byte = (q_word >> q_byte_shift) & 0xFFu;
            uint nib = (is_high != 0u) ? (q_byte >> 4) : (q_byte & 0x0Fu);
            float w = d * (float)((int)nib - 8);
            uint col_idx = b * 32u + lane_id;

            [unroll]
            for (uint i = 0u; i < 32u; i++)
            {
                uint t = t_base + i;
                uint safe_t = (t < batch_size) ? t : 0u;
                float xv = asfloat(x.Load((safe_t * k_cols + col_idx) * 4u));
                if (t < batch_size)
                {
                    acc[i] += w * xv;
                }
            }
        }
    }

    [unroll]
    for (uint i = 0u; i < 32u; i++)
    {
        uint t = t_base + i;
        s_mem[i * 128u + s_idx] = (t < batch_size && warp_id < m_rows) ? acc[i] : 0.0f;
    }

    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 16u) { [unroll] for (uint i = 0u; i < 32u; i++) s_mem[i * 128u + s_idx] += s_mem[i * 128u + s_idx + 16u]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 8u)  { [unroll] for (uint i = 0u; i < 32u; i++) s_mem[i * 128u + s_idx] += s_mem[i * 128u + s_idx + 8u]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 4u)  { [unroll] for (uint i = 0u; i < 32u; i++) s_mem[i * 128u + s_idx] += s_mem[i * 128u + s_idx + 4u]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 2u)  { [unroll] for (uint i = 0u; i < 32u; i++) s_mem[i * 128u + s_idx] += s_mem[i * 128u + s_idx + 2u]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 1u)  { [unroll] for (uint i = 0u; i < 32u; i++) s_mem[i * 128u + s_idx] += s_mem[i * 128u + s_idx + 1u]; }

    if (lane_id == 0u && warp_id < m_rows)
    {
        float b_val = 0.0f;
        if (has_bias != 0u) b_val = asfloat(bias.Load(warp_id * 4u));
        [unroll]
        for (uint i = 0u; i < 32u; i++)
        {
            uint t = t_base + i;
            if (t < batch_size)
            {
                float final_sum = s_mem[i * 128u + row_in_grp * 32u] + b_val;
                uint out_byte_addr = (t * m_rows + warp_id) * 4u;
                if (has_residual != 0u)
                {
                    float prev = asfloat(residual.Load(out_byte_addr));
                    residual.Store(out_byte_addr, asuint(prev + final_sum));
                }
                if (has_y != 0u)
                {
                    y.Store(out_byte_addr, asuint(final_sum));
                }
            }
        }
    }
}
";

    public const string GemvQ5_0 = @"
cbuffer Params : register(b0)
{
    uint k_cols;
    uint m_rows;
    uint has_bias;
    uint has_residual;
    uint has_y;
    uint row_offset;
};

ByteAddressBuffer W : register(t0);
ByteAddressBuffer bias : register(t1);
RWByteAddressBuffer x : register(u0);
RWByteAddressBuffer residual : register(u1);
RWByteAddressBuffer y : register(u2);

groupshared float s_mem[128];

[numthreads(32, 4, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint row_in_grp = gtid.y;
    uint warp_id = row_offset + gid.x * 4u + row_in_grp;
    uint lane_id = gtid.x;
    uint s_idx = row_in_grp * 32u + lane_id;

    if (warp_id < m_rows)
    {
        uint nb = k_cols / 32u;
        uint row_byte_offset = warp_id * nb * 24u;
        float row_sum = 0.0f;

        uint l = lane_id & 15u;
        uint is_high = lane_id >> 4;
        uint q_word_off = 8u + (l & ~3u);
        uint q_byte_shift = (l & 3u) * 8u;

        for (uint b = 0u; b < nb; b++)
        {
            uint blk_addr = row_byte_offset + b * 24u;
            float d = f16tof32(W.Load(blk_addr) & 0xFFFFu);
            uint qh = W.Load(blk_addr + 4u);

            uint q_word = W.Load(blk_addr + q_word_off);
            uint q_byte = (q_word >> q_byte_shift) & 0xFFu;
            uint lo4 = (is_high != 0u) ? (q_byte >> 4) : (q_byte & 0x0Fu);
            uint hbit = ((qh >> lane_id) & 1u) << 4;

            float w = d * (float)((int)(lo4 | hbit) - 16);
            row_sum += w * asfloat(x.Load((b * 32u + lane_id) * 4u));
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
        float final_sum = s_mem[row_in_grp * 32u];
        if (has_bias != 0u) final_sum += asfloat(bias.Load(warp_id * 4u));
        if (has_residual != 0u)
        {
            float prev = asfloat(residual.Load(warp_id * 4u));
            residual.Store(warp_id * 4u, asuint(prev + final_sum));
        }
        if (has_y != 0u) y.Store(warp_id * 4u, asuint(final_sum));
    }
}
";

    public const string GemmQ5_0Batch = @"
cbuffer Params : register(b0)
{
    uint k_cols;
    uint m_rows;
    uint batch_size;
    uint has_bias;
    uint has_residual;
    uint has_y;
};

ByteAddressBuffer W : register(t0);
ByteAddressBuffer bias : register(t1);
RWByteAddressBuffer x : register(u2);
RWByteAddressBuffer residual : register(u0);
RWByteAddressBuffer y : register(u1);

groupshared float s_mem[4096];

[numthreads(32, 4, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint row_in_grp = gtid.y;
    uint warp_id = gid.x * 4u + row_in_grp;
    uint lane_id = gtid.x;
    uint s_idx = row_in_grp * 32u + lane_id;
    uint t_base = gid.y * 32u;

    float acc[32];
    [unroll]
    for (uint i = 0u; i < 32u; i++) acc[i] = 0.0f;

    if (warp_id < m_rows)
    {
        uint nb = k_cols / 32u;
        uint row_byte_offset = warp_id * nb * 24u;

        uint l = lane_id & 15u;
        uint is_high = lane_id >> 4;
        uint q_word_off = 8u + (l & ~3u);
        uint q_byte_shift = (l & 3u) * 8u;

        for (uint b = 0u; b < nb; b++)
        {
            uint blk_addr = row_byte_offset + b * 24u;
            float d = f16tof32(W.Load(blk_addr) & 0xFFFFu);
            uint qh = W.Load(blk_addr + 4u);

            uint q_word = W.Load(blk_addr + q_word_off);
            uint q_byte = (q_word >> q_byte_shift) & 0xFFu;
            uint lo4 = (is_high != 0u) ? (q_byte >> 4) : (q_byte & 0x0Fu);
            uint hbit = ((qh >> lane_id) & 1u) << 4;
            float w = d * (float)((int)(lo4 | hbit) - 16);
            uint col_idx = b * 32u + lane_id;

            [unroll]
            for (uint i = 0u; i < 32u; i++)
            {
                uint t = t_base + i;
                uint safe_t = (t < batch_size) ? t : 0u;
                float xv = asfloat(x.Load((safe_t * k_cols + col_idx) * 4u));
                if (t < batch_size)
                {
                    acc[i] += w * xv;
                }
            }
        }
    }

    [unroll]
    for (uint i = 0u; i < 32u; i++)
    {
        uint t = t_base + i;
        s_mem[i * 128u + s_idx] = (t < batch_size && warp_id < m_rows) ? acc[i] : 0.0f;
    }

    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 16u) { [unroll] for (uint i = 0u; i < 32u; i++) s_mem[i * 128u + s_idx] += s_mem[i * 128u + s_idx + 16u]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 8u)  { [unroll] for (uint i = 0u; i < 32u; i++) s_mem[i * 128u + s_idx] += s_mem[i * 128u + s_idx + 8u]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 4u)  { [unroll] for (uint i = 0u; i < 32u; i++) s_mem[i * 128u + s_idx] += s_mem[i * 128u + s_idx + 4u]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 2u)  { [unroll] for (uint i = 0u; i < 32u; i++) s_mem[i * 128u + s_idx] += s_mem[i * 128u + s_idx + 2u]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 1u)  { [unroll] for (uint i = 0u; i < 32u; i++) s_mem[i * 128u + s_idx] += s_mem[i * 128u + s_idx + 1u]; }

    if (lane_id == 0u && warp_id < m_rows)
    {
        float b_val = 0.0f;
        if (has_bias != 0u) b_val = asfloat(bias.Load(warp_id * 4u));
        [unroll]
        for (uint i = 0u; i < 32u; i++)
        {
            uint t = t_base + i;
            if (t < batch_size)
            {
                float final_sum = s_mem[i * 128u + row_in_grp * 32u] + b_val;
                uint out_byte_addr = (t * m_rows + warp_id) * 4u;
                if (has_residual != 0u)
                {
                    float prev = asfloat(residual.Load(out_byte_addr));
                    residual.Store(out_byte_addr, asuint(prev + final_sum));
                }
                if (has_y != 0u)
                {
                    y.Store(out_byte_addr, asuint(final_sum));
                }
            }
        }
    }
}
";

    public const string GemvBf16 = @"
cbuffer Params : register(b0)
{
    uint k_cols;
    uint m_rows;
    uint has_bias;
    uint has_residual;
    uint has_y;
    uint row_offset;
};

ByteAddressBuffer W : register(t0);
ByteAddressBuffer bias : register(t1);
RWByteAddressBuffer x : register(u0);
RWByteAddressBuffer residual : register(u1);
RWByteAddressBuffer y : register(u2);

groupshared float s_mem[128];

[numthreads(32, 4, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint row_in_grp = gtid.y;
    uint warp_id = row_offset + gid.x * 4u + row_in_grp;
    uint lane_id = gtid.x;
    uint s_idx = row_in_grp * 32u + lane_id;

    if (warp_id < m_rows)
    {
        uint n_pairs = k_cols / 2u;
        uint row_byte_offset = warp_id * (k_cols * 2u);
        float row_sum = 0.0f;

        for (uint p = lane_id; p < n_pairs; p += 32u)
        {
            uint pair = W.Load(row_byte_offset + p * 4u);
            float w0 = asfloat((pair & 0xFFFFu) << 16);
            float w1 = asfloat(pair & 0xFFFF0000u);
            float2 xv = asfloat(x.Load2(p * 8u));
            row_sum += w0 * xv.x + w1 * xv.y;
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
        float final_sum = s_mem[row_in_grp * 32u];
        if (has_bias != 0u) final_sum += asfloat(bias.Load(warp_id * 4u));
        if (has_residual != 0u)
        {
            float prev = asfloat(residual.Load(warp_id * 4u));
            residual.Store(warp_id * 4u, asuint(prev + final_sum));
        }
        if (has_y != 0u) y.Store(warp_id * 4u, asuint(final_sum));
    }
}
";

    public const string GemmBf16Batch = @"
cbuffer Params : register(b0)
{
    uint k_cols;
    uint m_rows;
    uint batch_size;
    uint has_bias;
    uint has_residual;
    uint has_y;
};

ByteAddressBuffer W : register(t0);
ByteAddressBuffer bias : register(t1);
RWByteAddressBuffer x : register(u2);
RWByteAddressBuffer residual : register(u0);
RWByteAddressBuffer y : register(u1);

groupshared float s_mem[4096];

[numthreads(32, 4, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint row_in_grp = gtid.y;
    uint warp_id = gid.x * 4u + row_in_grp;
    uint lane_id = gtid.x;
    uint s_idx = row_in_grp * 32u + lane_id;
    uint t_base = gid.y * 32u;

    float acc[32];
    [unroll]
    for (uint i = 0u; i < 32u; i++) acc[i] = 0.0f;

    if (warp_id < m_rows)
    {
        uint n_pairs = k_cols / 2u;
        uint row_byte_offset = warp_id * (k_cols * 2u);

        for (uint p = lane_id; p < n_pairs; p += 32u)
        {
            uint pair = W.Load(row_byte_offset + p * 4u);
            float w0 = asfloat((pair & 0xFFFFu) << 16);
            float w1 = asfloat(pair & 0xFFFF0000u);
            uint col = p * 2u;

            [unroll]
            for (uint i = 0u; i < 32u; i++)
            {
                uint t = t_base + i;
                uint safe_t = (t < batch_size) ? t : 0u;
                float2 xv = asfloat(x.Load2((safe_t * k_cols + col) * 4u));
                if (t < batch_size)
                {
                    acc[i] += w0 * xv.x + w1 * xv.y;
                }
            }
        }
    }

    [unroll]
    for (uint i = 0u; i < 32u; i++)
    {
        uint t = t_base + i;
        s_mem[i * 128u + s_idx] = (t < batch_size && warp_id < m_rows) ? acc[i] : 0.0f;
    }

    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 16u) { [unroll] for (uint i = 0u; i < 32u; i++) s_mem[i * 128u + s_idx] += s_mem[i * 128u + s_idx + 16u]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 8u)  { [unroll] for (uint i = 0u; i < 32u; i++) s_mem[i * 128u + s_idx] += s_mem[i * 128u + s_idx + 8u]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 4u)  { [unroll] for (uint i = 0u; i < 32u; i++) s_mem[i * 128u + s_idx] += s_mem[i * 128u + s_idx + 4u]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 2u)  { [unroll] for (uint i = 0u; i < 32u; i++) s_mem[i * 128u + s_idx] += s_mem[i * 128u + s_idx + 2u]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 1u)  { [unroll] for (uint i = 0u; i < 32u; i++) s_mem[i * 128u + s_idx] += s_mem[i * 128u + s_idx + 1u]; }

    if (lane_id == 0u && warp_id < m_rows)
    {
        float b_val = 0.0f;
        if (has_bias != 0u) b_val = asfloat(bias.Load(warp_id * 4u));
        [unroll]
        for (uint i = 0u; i < 32u; i++)
        {
            uint t = t_base + i;
            if (t < batch_size)
            {
                float final_sum = s_mem[i * 128u + row_in_grp * 32u] + b_val;
                uint out_byte_addr = (t * m_rows + warp_id) * 4u;
                if (has_residual != 0u)
                {
                    float prev = asfloat(residual.Load(out_byte_addr));
                    residual.Store(out_byte_addr, asuint(prev + final_sum));
                }
                if (has_y != 0u)
                {
                    y.Store(out_byte_addr, asuint(final_sum));
                }
            }
        }
    }
}
";
}
