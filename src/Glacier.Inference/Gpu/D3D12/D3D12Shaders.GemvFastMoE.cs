namespace Glacier.Inference.Gpu.D3D12;

/// <summary>
/// Direct3D 12 HLSL Compute Shaders (cs_5_0) for Fast MoE Quantizations:
/// - Q2_0 (64-element lookup-free 2-bit blocks, 20 bytes/block, 3.4x prefill throughput)
/// - IQ4_NL (32-element non-linear 4-bit blocks, 20 bytes aligned, for 640-col ffn_down_exps)
/// - IQ3_S (256-element non-linear 3-bit super-blocks, 112 bytes aligned, for GSQ-RCO experts)
/// </summary>
public static partial class D3D12Shaders
{
    // =========================================================================
    // 1. Q2_0 GEMV & Batched GEMM (64 elements in 20 bytes, 4-byte aligned)
    // =========================================================================

    public const string GemvQ2_0 = @"
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
StructuredBuffer<float> bias : register(t1);
RWStructuredBuffer<float> x : register(u0);
RWStructuredBuffer<float> residual : register(u1);
RWStructuredBuffer<float> y : register(u2);

groupshared float s_mem[128];

[numthreads(32, 4, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint row_in_grp = gtid.y;
    uint warp_id = row_offset + gid.x * 4 + row_in_grp;
    uint lane_id = gtid.x;
    uint s_idx = row_in_grp * 32 + lane_id;

    if (warp_id < m_rows)
    {
        uint nb = k_cols / 64;
        uint row_byte_offset = warp_id * nb * 20;
        float row_sum = 0.0f;

        uint l = lane_id & 15;
        uint half_blk = lane_id >> 4;
        uint q_word_off = 4 + (l & ~3);
        uint q_byte_shift = (l & 3) * 8;
        uint shift0 = half_blk * 4;
        uint shift1 = shift0 + 2;
        uint x_sub_off = half_blk * 32 + l;

        for (uint b = 0; b < nb; b++)
        {
            uint blk_addr = row_byte_offset + b * 20;
            uint dm_word = W.Load(blk_addr);
            float d = f16tof32(dm_word & 0xFFFF);
            float m = f16tof32((dm_word >> 16) & 0xFFFF);

            uint q_word = W.Load(blk_addr + q_word_off);
            uint q_byte = (q_word >> q_byte_shift) & 0xFF;

            float w0 = d * (float)((q_byte >> shift0) & 3) + m;
            float w1 = d * (float)((q_byte >> shift1) & 3) + m;

            uint x_base = b * 64 + x_sub_off;
            row_sum += w0 * x[x_base] + w1 * x[x_base + 16];
        }

        s_mem[s_idx] = row_sum;
    }
    else
    {
        s_mem[s_idx] = 0.0f;
    }

    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 16) s_mem[s_idx] += s_mem[s_idx + 16];
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 8)  s_mem[s_idx] += s_mem[s_idx + 8];
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 4)  s_mem[s_idx] += s_mem[s_idx + 4];
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 2)  s_mem[s_idx] += s_mem[s_idx + 2];
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 1)  s_mem[s_idx] += s_mem[s_idx + 1];

    if (lane_id == 0 && warp_id < m_rows)
    {
        float final_sum = s_mem[row_in_grp * 32];
        if (has_bias != 0) final_sum += bias[warp_id];
        if (has_residual != 0) residual[warp_id] += final_sum;
        if (has_y != 0) y[warp_id] = final_sum;
    }
}
";

    public const string GemmQ2_0Batch = @"
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
    uint warp_id = gid.x * 4 + row_in_grp;
    uint lane_id = gtid.x;
    uint s_idx = row_in_grp * 32 + lane_id;
    uint t_base = gid.y * 32;

    float acc[32];
    [unroll]
    for (uint i = 0; i < 32; i++) acc[i] = 0.0f;

    if (warp_id < m_rows)
    {
        uint nb = k_cols / 64;
        uint row_byte_offset = warp_id * nb * 20;

        uint l = lane_id & 15;
        uint half_blk = lane_id >> 4;
        uint q_word_off = 4 + (l & ~3);
        uint q_byte_shift = (l & 3) * 8;
        uint shift0 = half_blk * 4;
        uint shift1 = shift0 + 2;
        uint x_sub_off = half_blk * 32 + l;

        for (uint b = 0; b < nb; b++)
        {
            uint blk_addr = row_byte_offset + b * 20;
            uint dm_word = W.Load(blk_addr);
            float d = f16tof32(dm_word & 0xFFFF);
            float m = f16tof32((dm_word >> 16) & 0xFFFF);

            uint q_word = W.Load(blk_addr + q_word_off);
            uint q_byte = (q_word >> q_byte_shift) & 0xFF;

            float w0 = d * (float)((q_byte >> shift0) & 3) + m;
            float w1 = d * (float)((q_byte >> shift1) & 3) + m;

            uint col0 = b * 64 + x_sub_off;
            uint col1 = col0 + 16;

            [unroll]
            for (uint i = 0; i < 32; i++)
            {
                uint t = t_base + i;
                if (t < batch_size)
                {
                    uint row_base = t * k_cols;
                    acc[i] += w0 * asfloat(x.Load((row_base + col0) * 4))
                            + w1 * asfloat(x.Load((row_base + col1) * 4));
                }
            }
        }
    }

    [unroll]
    for (uint i = 0; i < 32; i++)
    {
        uint t = t_base + i;
        s_mem[i * 128 + s_idx] = (t < batch_size && warp_id < m_rows) ? acc[i] : 0.0f;
    }

    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 16) { [unroll] for (uint i = 0; i < 32; i++) s_mem[i * 128 + s_idx] += s_mem[i * 128 + s_idx + 16]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 8)  { [unroll] for (uint i = 0; i < 32; i++) s_mem[i * 128 + s_idx] += s_mem[i * 128 + s_idx + 8]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 4)  { [unroll] for (uint i = 0; i < 32; i++) s_mem[i * 128 + s_idx] += s_mem[i * 128 + s_idx + 4]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 2)  { [unroll] for (uint i = 0; i < 32; i++) s_mem[i * 128 + s_idx] += s_mem[i * 128 + s_idx + 2]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 1)  { [unroll] for (uint i = 0; i < 32; i++) s_mem[i * 128 + s_idx] += s_mem[i * 128 + s_idx + 1]; }

    if (lane_id == 0 && warp_id < m_rows)
    {
        float b_val = 0.0f;
        if (has_bias != 0) b_val = asfloat(bias.Load(warp_id * 4));
        [unroll]
        for (uint i = 0; i < 32; i++)
        {
            uint t = t_base + i;
            if (t < batch_size)
            {
                float final_sum = s_mem[i * 128 + row_in_grp * 32] + b_val;
                uint out_byte_addr = (t * m_rows + warp_id) * 4;
                if (has_residual != 0)
                {
                    float prev = asfloat(residual.Load(out_byte_addr));
                    residual.Store(out_byte_addr, asuint(prev + final_sum));
                }
                if (has_y != 0)
                {
                    y.Store(out_byte_addr, asuint(final_sum));
                }
            }
        }
    }
}
";

    // =========================================================================
    // 2. IQ4_NL GEMV & Batched GEMM (32 elements in 20 aligned bytes)
    // =========================================================================

    private const string IQ4_NL_Table = @"
static const float kvalues_iq4nl[16] = {
    -127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
       1.0f,   13.0f,  25.0f,  38.0f,  53.0f,  69.0f,  89.0f, 113.0f
};
";

    public const string GemvIQ4_NL = @"
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
StructuredBuffer<float> bias : register(t1);
RWStructuredBuffer<float> x : register(u0);
RWStructuredBuffer<float> residual : register(u1);
RWStructuredBuffer<float> y : register(u2);

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
        uint nb = k_cols / 32;
        uint row_byte_offset = warp_id * nb * 20;
        float row_sum = 0.0f;

        uint l = lane_id & 15;
        uint is_high = lane_id >> 4;
        uint q_word_off = 4 + (l & ~3);
        uint q_byte_shift = (l & 3) * 8;

        for (uint b = 0; b < nb; b++)
        {
            uint blk_addr = row_byte_offset + b * 20;
            float d = f16tof32(W.Load(blk_addr) & 0xFFFF);

            uint q_word = W.Load(blk_addr + q_word_off);
            uint q_byte = (q_word >> q_byte_shift) & 0xFF;
            uint nib = (is_high != 0) ? (q_byte >> 4) : (q_byte & 0x0F);

            float w = d * kvalues_iq4nl[nib];
            row_sum += w * x[b * 32 + lane_id];
        }

        s_mem[s_idx] = row_sum;
    }
    else
    {
        s_mem[s_idx] = 0.0f;
    }

    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 16) s_mem[s_idx] += s_mem[s_idx + 16];
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 8)  s_mem[s_idx] += s_mem[s_idx + 8];
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 4)  s_mem[s_idx] += s_mem[s_idx + 4];
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 2)  s_mem[s_idx] += s_mem[s_idx + 2];
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 1)  s_mem[s_idx] += s_mem[s_idx + 1];

    if (lane_id == 0 && warp_id < m_rows)
    {
        float final_sum = s_mem[row_in_grp * 32];
        if (has_bias != 0) final_sum += bias[warp_id];
        if (has_residual != 0) residual[warp_id] += final_sum;
        if (has_y != 0) y[warp_id] = final_sum;
    }
}
";

    public const string GemmIQ4_NLBatch = @"
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
    uint warp_id = gid.x * 4 + row_in_grp;
    uint lane_id = gtid.x;
    uint s_idx = row_in_grp * 32 + lane_id;
    uint t_base = gid.y * 32;

    float acc[32];
    [unroll]
    for (uint i = 0; i < 32; i++) acc[i] = 0.0f;

    if (warp_id < m_rows)
    {
        uint nb = k_cols / 32;
        uint row_byte_offset = warp_id * nb * 20;

        uint l = lane_id & 15;
        uint is_high = lane_id >> 4;
        uint q_word_off = 4 + (l & ~3);
        uint q_byte_shift = (l & 3) * 8;

        for (uint b = 0; b < nb; b++)
        {
            uint blk_addr = row_byte_offset + b * 20;
            float d = f16tof32(W.Load(blk_addr) & 0xFFFF);

            uint q_word = W.Load(blk_addr + q_word_off);
            uint q_byte = (q_word >> q_byte_shift) & 0xFF;
            uint nib = (is_high != 0) ? (q_byte >> 4) : (q_byte & 0x0F);

            float w = d * kvalues_iq4nl[nib];
            uint col_idx = b * 32 + lane_id;

            [unroll]
            for (uint i = 0; i < 32; i++)
            {
                uint t = t_base + i;
                if (t < batch_size)
                {
                    acc[i] += w * asfloat(x.Load((t * k_cols + col_idx) * 4));
                }
            }
        }
    }

    [unroll]
    for (uint i = 0; i < 32; i++)
    {
        uint t = t_base + i;
        s_mem[i * 128 + s_idx] = (t < batch_size && warp_id < m_rows) ? acc[i] : 0.0f;
    }

    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 16) { [unroll] for (uint i = 0; i < 32; i++) s_mem[i * 128 + s_idx] += s_mem[i * 128 + s_idx + 16]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 8)  { [unroll] for (uint i = 0; i < 32; i++) s_mem[i * 128 + s_idx] += s_mem[i * 128 + s_idx + 8]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 4)  { [unroll] for (uint i = 0; i < 32; i++) s_mem[i * 128 + s_idx] += s_mem[i * 128 + s_idx + 4]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 2)  { [unroll] for (uint i = 0; i < 32; i++) s_mem[i * 128 + s_idx] += s_mem[i * 128 + s_idx + 2]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 1)  { [unroll] for (uint i = 0; i < 32; i++) s_mem[i * 128 + s_idx] += s_mem[i * 128 + s_idx + 1]; }

    if (lane_id == 0 && warp_id < m_rows)
    {
        float b_val = 0.0f;
        if (has_bias != 0) b_val = asfloat(bias.Load(warp_id * 4));
        [unroll]
        for (uint i = 0; i < 32; i++)
        {
            uint t = t_base + i;
            if (t < batch_size)
            {
                float final_sum = s_mem[i * 128 + row_in_grp * 32] + b_val;
                uint out_byte_addr = (t * m_rows + warp_id) * 4;
                if (has_residual != 0)
                {
                    float prev = asfloat(residual.Load(out_byte_addr));
                    residual.Store(out_byte_addr, asuint(prev + final_sum));
                }
                if (has_y != 0)
                {
                    y.Store(out_byte_addr, asuint(final_sum));
                }
            }
        }
    }
}
";

    // =========================================================================
    // 3. IQ3_S GEMV & Batched GEMM (256 elements in 112 aligned bytes)
    // =========================================================================

    private const string IQ3_S_Helper = @"
uint decode_iq3s_grid(uint idx)
{
    uint b0 = 1u + 2u * ((idx >> 0) & 3u);
    uint b1 = 1u + 2u * ((idx >> 2) & 3u);
    uint b2 = 1u + 2u * ((idx >> 4) & 3u);
    uint b3 = 1u + 2u * ((idx >> 6) & 3u);
    if ((idx & 256u) != 0u)
    {
        b0 += 8u;
        b1 += ((idx >> 1) & 1u) * 8u;
        b2 += ((idx >> 3) & 1u) * 8u;
        b3 += ((idx >> 5) & 1u) * 8u;
    }
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}
";

    public const string GemvIQ3_S = @"
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
StructuredBuffer<float> bias : register(t1);
RWByteAddressBuffer x : register(u0);
RWStructuredBuffer<float> residual : register(u1);
RWStructuredBuffer<float> y : register(u2);

groupshared float s_mem[128];
" + IQ3_S_Helper + @"
[numthreads(32, 4, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint row_in_grp = gtid.y;
    uint warp_id = row_offset + gid.x * 4 + row_in_grp;
    uint lane_id = gtid.x;
    uint s_idx = row_in_grp * 32 + lane_id;

    if (warp_id < m_rows)
    {
        uint nb = k_cols / 256;
        uint row_byte_offset = warp_id * nb * 112;
        float row_sum = 0.0f;

        uint ib = lane_id >> 2;
        uint gp = lane_id & 3;
        uint g0 = gp * 2;
        uint g1 = g0 + 1;
        uint x_offset = lane_id * 8;

        for (uint b = 0; b < nb; b++)
        {
            uint blk_addr = row_byte_offset + b * 112;
            float d = f16tof32(W.Load(blk_addr) & 0xFFFF);

            uint scales_word = W.Load(blk_addr + 108);
            uint sc_byte = (scales_word >> ((ib / 2) * 8)) & 0xFF;
            uint ls = ((ib & 1) == 0) ? (sc_byte & 0x0F) : (sc_byte >> 4);
            float db = d * (float)(1 + 2 * ls);

            uint qh_word = W.Load(blk_addr + 68 + (ib & ~3));
            uint qh_byte = (qh_word >> ((ib & 3) * 8)) & 0xFF;

            uint qs_word = W.Load(blk_addr + 4 + ib * 8 + (g0 & ~3));
            uint qs_shift = (g0 & 2) * 8;
            uint qs0 = (qs_word >> qs_shift) & 0xFF;
            uint qs1 = (qs_word >> (qs_shift + 8)) & 0xFF;

            uint idx0 = qs0 | (((qh_byte >> g0) & 1u) << 8);
            uint idx1 = qs1 | (((qh_byte >> g1) & 1u) << 8);

            uint grid0 = decode_iq3s_grid(idx0);
            uint grid1 = decode_iq3s_grid(idx1);

            uint sign_word = W.Load(blk_addr + 76 + ib * 4);
            uint sign_byte = (sign_word >> (gp * 8)) & 0xFF;
            uint sn0 = sign_byte & 0x0F;
            uint sn1 = sign_byte >> 4;

            uint x_addr = (b * 256 + x_offset) * 4;
            float4 x0 = asfloat(x.Load4(x_addr));
            float4 x1 = asfloat(x.Load4(x_addr + 16));

            float dot = 0.0f;
            dot += x0.x * ((float)(grid0 & 0xFF) * ((sn0 & 1u) ? -1.0f : 1.0f));
            dot += x0.y * ((float)((grid0 >> 8) & 0xFF) * ((sn0 & 2u) ? -1.0f : 1.0f));
            dot += x0.z * ((float)((grid0 >> 16) & 0xFF) * ((sn0 & 4u) ? -1.0f : 1.0f));
            dot += x0.w * ((float)(grid0 >> 24) * ((sn0 & 8u) ? -1.0f : 1.0f));

            dot += x1.x * ((float)(grid1 & 0xFF) * ((sn1 & 1u) ? -1.0f : 1.0f));
            dot += x1.y * ((float)((grid1 >> 8) & 0xFF) * ((sn1 & 2u) ? -1.0f : 1.0f));
            dot += x1.z * ((float)((grid1 >> 16) & 0xFF) * ((sn1 & 4u) ? -1.0f : 1.0f));
            dot += x1.w * ((float)(grid1 >> 24) * ((sn1 & 8u) ? -1.0f : 1.0f));

            row_sum += db * dot;
        }

        s_mem[s_idx] = row_sum;
    }
    else
    {
        s_mem[s_idx] = 0.0f;
    }

    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 16) s_mem[s_idx] += s_mem[s_idx + 16];
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 8)  s_mem[s_idx] += s_mem[s_idx + 8];
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 4)  s_mem[s_idx] += s_mem[s_idx + 4];
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 2)  s_mem[s_idx] += s_mem[s_idx + 2];
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 1)  s_mem[s_idx] += s_mem[s_idx + 1];

    if (lane_id == 0 && warp_id < m_rows)
    {
        float final_sum = s_mem[row_in_grp * 32];
        if (has_bias != 0) final_sum += bias[warp_id];
        if (has_residual != 0) residual[warp_id] += final_sum;
        if (has_y != 0) y[warp_id] = final_sum;
    }
}
";

    public const string GemmIQ3_SBatch = @"
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
" + IQ3_S_Helper + @"
[numthreads(32, 4, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint row_in_grp = gtid.y;
    uint warp_id = gid.x * 4 + row_in_grp;
    uint lane_id = gtid.x;
    uint s_idx = row_in_grp * 32 + lane_id;
    uint t_base = gid.y * 32;

    float acc[32];
    [unroll]
    for (uint i = 0; i < 32; i++) acc[i] = 0.0f;

    if (warp_id < m_rows)
    {
        uint nb = k_cols / 256;
        uint row_byte_offset = warp_id * nb * 112;

        uint ib = lane_id >> 2;
        uint gp = lane_id & 3;
        uint g0 = gp * 2;
        uint g1 = g0 + 1;
        uint x_offset = lane_id * 8;

        for (uint b = 0; b < nb; b++)
        {
            uint blk_addr = row_byte_offset + b * 112;
            float d = f16tof32(W.Load(blk_addr) & 0xFFFF);

            uint scales_word = W.Load(blk_addr + 108);
            uint sc_byte = (scales_word >> ((ib / 2) * 8)) & 0xFF;
            uint ls = ((ib & 1) == 0) ? (sc_byte & 0x0F) : (sc_byte >> 4);
            float db = d * (float)(1 + 2 * ls);

            uint qh_word = W.Load(blk_addr + 68 + (ib & ~3));
            uint qh_byte = (qh_word >> ((ib & 3) * 8)) & 0xFF;

            uint qs_word = W.Load(blk_addr + 4 + ib * 8 + (g0 & ~3));
            uint qs_shift = (g0 & 2) * 8;
            uint qs0 = (qs_word >> qs_shift) & 0xFF;
            uint qs1 = (qs_word >> (qs_shift + 8)) & 0xFF;

            uint idx0 = qs0 | (((qh_byte >> g0) & 1u) << 8);
            uint idx1 = qs1 | (((qh_byte >> g1) & 1u) << 8);

            uint grid0 = decode_iq3s_grid(idx0);
            uint grid1 = decode_iq3s_grid(idx1);

            uint sign_word = W.Load(blk_addr + 76 + ib * 4);
            uint sign_byte = (sign_word >> (gp * 8)) & 0xFF;
            uint sn0 = sign_byte & 0x0F;
            uint sn1 = sign_byte >> 4;

            float w0 = db * ((float)(grid0 & 0xFF) * ((sn0 & 1u) ? -1.0f : 1.0f));
            float w1 = db * ((float)((grid0 >> 8) & 0xFF) * ((sn0 & 2u) ? -1.0f : 1.0f));
            float w2 = db * ((float)((grid0 >> 16) & 0xFF) * ((sn0 & 4u) ? -1.0f : 1.0f));
            float w3 = db * ((float)(grid0 >> 24) * ((sn0 & 8u) ? -1.0f : 1.0f));
            float w4 = db * ((float)(grid1 & 0xFF) * ((sn1 & 1u) ? -1.0f : 1.0f));
            float w5 = db * ((float)((grid1 >> 8) & 0xFF) * ((sn1 & 2u) ? -1.0f : 1.0f));
            float w6 = db * ((float)((grid1 >> 16) & 0xFF) * ((sn1 & 4u) ? -1.0f : 1.0f));
            float w7 = db * ((float)(grid1 >> 24) * ((sn1 & 8u) ? -1.0f : 1.0f));

            uint col_base = b * 256 + x_offset;

            [unroll]
            for (uint i = 0; i < 32; i++)
            {
                uint t = t_base + i;
                if (t < batch_size)
                {
                    uint t_x_byte = (t * k_cols + col_base) * 4;
                    acc[i] += w0 * asfloat(x.Load(t_x_byte + 0))
                            + w1 * asfloat(x.Load(t_x_byte + 4))
                            + w2 * asfloat(x.Load(t_x_byte + 8))
                            + w3 * asfloat(x.Load(t_x_byte + 12))
                            + w4 * asfloat(x.Load(t_x_byte + 16))
                            + w5 * asfloat(x.Load(t_x_byte + 20))
                            + w6 * asfloat(x.Load(t_x_byte + 24))
                            + w7 * asfloat(x.Load(t_x_byte + 28));
                }
            }
        }
    }

    [unroll]
    for (uint i = 0; i < 32; i++)
    {
        uint t = t_base + i;
        s_mem[i * 128 + s_idx] = (t < batch_size && warp_id < m_rows) ? acc[i] : 0.0f;
    }

    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 16) { [unroll] for (uint i = 0; i < 32; i++) s_mem[i * 128 + s_idx] += s_mem[i * 128 + s_idx + 16]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 8)  { [unroll] for (uint i = 0; i < 32; i++) s_mem[i * 128 + s_idx] += s_mem[i * 128 + s_idx + 8]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 4)  { [unroll] for (uint i = 0; i < 32; i++) s_mem[i * 128 + s_idx] += s_mem[i * 128 + s_idx + 4]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 2)  { [unroll] for (uint i = 0; i < 32; i++) s_mem[i * 128 + s_idx] += s_mem[i * 128 + s_idx + 2]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 1)  { [unroll] for (uint i = 0; i < 32; i++) s_mem[i * 128 + s_idx] += s_mem[i * 128 + s_idx + 1]; }

    if (lane_id == 0 && warp_id < m_rows)
    {
        float b_val = 0.0f;
        if (has_bias != 0) b_val = asfloat(bias.Load(warp_id * 4));
        [unroll]
        for (uint i = 0; i < 32; i++)
        {
            uint t = t_base + i;
            if (t < batch_size)
            {
                float final_sum = s_mem[i * 128 + row_in_grp * 32] + b_val;
                uint out_byte_addr = (t * m_rows + warp_id) * 4;
                if (has_residual != 0)
                {
                    float prev = asfloat(residual.Load(out_byte_addr));
                    residual.Store(out_byte_addr, asuint(prev + final_sum));
                }
                if (has_y != 0)
                {
                    y.Store(out_byte_addr, asuint(final_sum));
                }
            }
        }
    }
}
";
}
