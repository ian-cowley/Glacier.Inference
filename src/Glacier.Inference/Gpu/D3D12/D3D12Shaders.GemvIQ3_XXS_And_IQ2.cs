namespace Glacier.Inference.Gpu.D3D12;

using System.Globalization;
using System.Text;
using Glacier.Inference.Quant;

public static partial class D3D12Shaders
{
    private static string FormatUint2Grid(string name, ulong[] grid)
    {
        var sb = new StringBuilder(grid.Length * 28);
        sb.Append("static const uint2 ").Append(name).Append('[').Append(grid.Length).AppendLine("] = {");
        for (int i = 0; i < grid.Length; i++)
        {
            ulong val = grid[i];
            uint lo = (uint)(val & 0xFFFF_FFFFUL);
            uint hi = (uint)(val >> 32);
            if ((i & 3) == 0) sb.Append("    ");
            sb.AppendFormat(CultureInfo.InvariantCulture, "uint2(0x{0:x8}, 0x{1:x8}), ", lo, hi);
            if ((i & 3) == 3 || i == grid.Length - 1) sb.AppendLine();
        }
        sb.AppendLine("};");
        return sb.ToString();
    }

    private const string IQ2_KSIGNS_Table = @"
static const uint ksigns_iq2_tbl[32] = {
    0x03828100, 0x87060584, 0x8b0a0988, 0x0f8e8d0c,
    0x93121190, 0x17969514, 0x1b9a9918, 0x9f1e1d9c,
    0xa32221a0, 0x27a6a524, 0x2baaa928, 0xaf2e2dac,
    0x33b2b130, 0xb73635b4, 0xbb3a39b8, 0x3fbebd3c,
    0xc34241c0, 0x47c6c544, 0x4bcac948, 0xcf4e4dcc,
    0x53d2d150, 0xd75655d4, 0xdb5a59d8, 0x5fdedd5c,
    0x63e2e160, 0xe76665e4, 0xeb6a69e8, 0x6feeed6c,
    0xf37271f0, 0x77f6f574, 0x7bfaf978, 0xff7e7dfc,
};
";

    private static readonly string IQ2_XS_Tables = FormatUint2Grid("iq2xs_grid", QuantKernels.Iq2xsGrid) + IQ2_KSIGNS_Table;
    private static readonly string IQ2_S_Tables = FormatUint2Grid("iq2s_grid", QuantKernels.Iq2sGrid);

    // =========================================================================
    // 1. IQ2_XS GEMV & Batched GEMM (256 elements in 76 aligned bytes)
    // =========================================================================

    public static readonly string GemvIQ2_XS = @"
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
" + IQ2_XS_Tables + @"
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
        uint row_byte_offset = warp_id * nb * 76;
        float row_sum = 0.0f;

        uint ib32 = lane_id >> 2;
        uint l = lane_id & 3;
        uint q_word_off = 4 + ib32 * 8 + ((l >= 2) ? 4 : 0);
        uint q_shift = (l & 1) * 16;
        uint sc_word_off = 68 + (ib32 & ~3);
        uint sc_shift = (ib32 & 3) * 8;
        uint x_offset = lane_id * 8;

        for (uint b = 0; b < nb; b++)
        {
            uint blk_addr = row_byte_offset + b * 76;
            float d = f16tof32(W.Load(blk_addr) & 0xFFFF);

            uint q_word = W.Load(blk_addr + q_word_off);
            uint q = (q_word >> q_shift) & 0xFFFF;

            uint sc_word = W.Load(blk_addr + sc_word_off);
            uint sc_byte = (sc_word >> sc_shift) & 0xFF;
            uint ls = (l < 2) ? (sc_byte & 0x0F) : (sc_byte >> 4);
            float db = d * (0.5f + (float)ls) * 0.25f;

            uint grid_idx = q & 511u;
            uint signs_idx = q >> 9;
            uint s_word = ksigns_iq2_tbl[signs_idx >> 2];
            uint signs = (s_word >> ((signs_idx & 3u) * 8)) & 0xFF;

            uint2 g = iq2xs_grid[grid_idx];
            float g0 = (float)(g.x & 0xFF);
            float g1 = (float)((g.x >> 8) & 0xFF);
            float g2 = (float)((g.x >> 16) & 0xFF);
            float g3 = (float)(g.x >> 24);
            float g4 = (float)(g.y & 0xFF);
            float g5 = (float)((g.y >> 8) & 0xFF);
            float g6 = (float)((g.y >> 16) & 0xFF);
            float g7 = (float)(g.y >> 24);

            uint x_addr = (b * 256 + x_offset) * 4;
            float4 x0 = asfloat(x.Load4(x_addr));
            float4 x1 = asfloat(x.Load4(x_addr + 16));

            float dot = 0.0f;
            dot += x0.x * (g0 * ((signs & 1) ? -1.0f : 1.0f));
            dot += x0.y * (g1 * ((signs & 2) ? -1.0f : 1.0f));
            dot += x0.z * (g2 * ((signs & 4) ? -1.0f : 1.0f));
            dot += x0.w * (g3 * ((signs & 8) ? -1.0f : 1.0f));
            dot += x1.x * (g4 * ((signs & 16) ? -1.0f : 1.0f));
            dot += x1.y * (g5 * ((signs & 32) ? -1.0f : 1.0f));
            dot += x1.z * (g6 * ((signs & 64) ? -1.0f : 1.0f));
            dot += x1.w * (g7 * ((signs & 128) ? -1.0f : 1.0f));

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

    public static readonly string GemmIQ2_XSBatch = @"
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
" + IQ2_XS_Tables + @"
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
        uint row_byte_offset = warp_id * nb * 76;

        uint ib32 = lane_id >> 2;
        uint l = lane_id & 3;
        uint q_word_off = 4 + ib32 * 8 + ((l >= 2) ? 4 : 0);
        uint q_shift = (l & 1) * 16;
        uint sc_word_off = 68 + (ib32 & ~3);
        uint sc_shift = (ib32 & 3) * 8;
        uint x_offset = lane_id * 8;

        for (uint b = 0; b < nb; b++)
        {
            uint blk_addr = row_byte_offset + b * 76;
            float d = f16tof32(W.Load(blk_addr) & 0xFFFF);

            uint q_word = W.Load(blk_addr + q_word_off);
            uint q = (q_word >> q_shift) & 0xFFFF;

            uint sc_word = W.Load(blk_addr + sc_word_off);
            uint sc_byte = (sc_word >> sc_shift) & 0xFF;
            uint ls = (l < 2) ? (sc_byte & 0x0F) : (sc_byte >> 4);
            float db = d * (0.5f + (float)ls) * 0.25f;

            uint grid_idx = q & 511u;
            uint signs_idx = q >> 9;
            uint s_word = ksigns_iq2_tbl[signs_idx >> 2];
            uint signs = (s_word >> ((signs_idx & 3u) * 8)) & 0xFF;

            uint2 g = iq2xs_grid[grid_idx];
            float w0 = db * ((float)(g.x & 0xFF) * ((signs & 1) ? -1.0f : 1.0f));
            float w1 = db * ((float)((g.x >> 8) & 0xFF) * ((signs & 2) ? -1.0f : 1.0f));
            float w2 = db * ((float)((g.x >> 16) & 0xFF) * ((signs & 4) ? -1.0f : 1.0f));
            float w3 = db * ((float)(g.x >> 24) * ((signs & 8) ? -1.0f : 1.0f));
            float w4 = db * ((float)(g.y & 0xFF) * ((signs & 16) ? -1.0f : 1.0f));
            float w5 = db * ((float)((g.y >> 8) & 0xFF) * ((signs & 32) ? -1.0f : 1.0f));
            float w6 = db * ((float)((g.y >> 16) & 0xFF) * ((signs & 64) ? -1.0f : 1.0f));
            float w7 = db * ((float)(g.y >> 24) * ((signs & 128) ? -1.0f : 1.0f));

            uint col_base = b * 256 + x_offset;

            [unroll]
            for (uint i = 0; i < 32; i++)
            {
                uint t = t_base + i;
                uint safe_t = (t < batch_size) ? t : 0;
                uint t_x_byte = (safe_t * k_cols + col_base) * 4;
                float4 xv0 = asfloat(x.Load4(t_x_byte));
                float4 xv1 = asfloat(x.Load4(t_x_byte + 16));
                if (t < batch_size)
                {
                    acc[i] += w0 * xv0.x + w1 * xv0.y + w2 * xv0.z + w3 * xv0.w
                            + w4 * xv1.x + w5 * xv1.y + w6 * xv1.z + w7 * xv1.w;
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
    // 2. IQ2_S GEMV & Batched GEMM (256 elements in 84 aligned bytes)
    // =========================================================================

    public static readonly string GemvIQ2_S = @"
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
" + IQ2_S_Tables + @"
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
        uint row_byte_offset = warp_id * nb * 84;
        float row_sum = 0.0f;

        uint ib32 = lane_id >> 2;
        uint l = lane_id & 3;
        uint q_word_off = 4 + ib32 * 4;
        uint signs_word_off = 36 + ib32 * 4;
        uint qh_word_off = 68 + (ib32 & ~3);
        uint sc_word_off = 76 + (ib32 & ~3);
        uint sub_shift = (ib32 & 3) * 8;
        uint l_shift = l * 8;
        uint x_offset = lane_id * 8;

        for (uint b = 0; b < nb; b++)
        {
            uint blk_addr = row_byte_offset + b * 84;
            float d = f16tof32(W.Load(blk_addr) & 0xFFFF);

            uint q_lo = (W.Load(blk_addr + q_word_off) >> l_shift) & 0xFF;
            uint signs = (W.Load(blk_addr + signs_word_off) >> l_shift) & 0xFF;
            uint qh_byte = (W.Load(blk_addr + qh_word_off) >> sub_shift) & 0xFF;
            uint sc_byte = (W.Load(blk_addr + sc_word_off) >> sub_shift) & 0xFF;

            uint grid_idx = q_lo | (((qh_byte >> (2 * l)) & 3u) << 8);
            uint ls = (l < 2) ? (sc_byte & 0x0F) : (sc_byte >> 4);
            float db = d * (0.5f + (float)ls) * 0.25f;

            uint2 g = iq2s_grid[grid_idx];
            float g0 = (float)(g.x & 0xFF);
            float g1 = (float)((g.x >> 8) & 0xFF);
            float g2 = (float)((g.x >> 16) & 0xFF);
            float g3 = (float)(g.x >> 24);
            float g4 = (float)(g.y & 0xFF);
            float g5 = (float)((g.y >> 8) & 0xFF);
            float g6 = (float)((g.y >> 16) & 0xFF);
            float g7 = (float)(g.y >> 24);

            uint x_addr = (b * 256 + x_offset) * 4;
            float4 x0 = asfloat(x.Load4(x_addr));
            float4 x1 = asfloat(x.Load4(x_addr + 16));

            float dot = 0.0f;
            dot += x0.x * (g0 * ((signs & 1) ? -1.0f : 1.0f));
            dot += x0.y * (g1 * ((signs & 2) ? -1.0f : 1.0f));
            dot += x0.z * (g2 * ((signs & 4) ? -1.0f : 1.0f));
            dot += x0.w * (g3 * ((signs & 8) ? -1.0f : 1.0f));
            dot += x1.x * (g4 * ((signs & 16) ? -1.0f : 1.0f));
            dot += x1.y * (g5 * ((signs & 32) ? -1.0f : 1.0f));
            dot += x1.z * (g6 * ((signs & 64) ? -1.0f : 1.0f));
            dot += x1.w * (g7 * ((signs & 128) ? -1.0f : 1.0f));

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

    public static readonly string GemmIQ2_SBatch = @"
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
" + IQ2_S_Tables + @"
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
        uint row_byte_offset = warp_id * nb * 84;

        uint ib32 = lane_id >> 2;
        uint l = lane_id & 3;
        uint q_word_off = 4 + ib32 * 4;
        uint signs_word_off = 36 + ib32 * 4;
        uint qh_word_off = 68 + (ib32 & ~3);
        uint sc_word_off = 76 + (ib32 & ~3);
        uint sub_shift = (ib32 & 3) * 8;
        uint l_shift = l * 8;
        uint x_offset = lane_id * 8;

        for (uint b = 0; b < nb; b++)
        {
            uint blk_addr = row_byte_offset + b * 84;
            float d = f16tof32(W.Load(blk_addr) & 0xFFFF);

            uint q_lo = (W.Load(blk_addr + q_word_off) >> l_shift) & 0xFF;
            uint signs = (W.Load(blk_addr + signs_word_off) >> l_shift) & 0xFF;
            uint qh_byte = (W.Load(blk_addr + qh_word_off) >> sub_shift) & 0xFF;
            uint sc_byte = (W.Load(blk_addr + sc_word_off) >> sub_shift) & 0xFF;

            uint grid_idx = q_lo | (((qh_byte >> (2 * l)) & 3u) << 8);
            uint ls = (l < 2) ? (sc_byte & 0x0F) : (sc_byte >> 4);
            float db = d * (0.5f + (float)ls) * 0.25f;

            uint2 g = iq2s_grid[grid_idx];
            float w0 = db * ((float)(g.x & 0xFF) * ((signs & 1) ? -1.0f : 1.0f));
            float w1 = db * ((float)((g.x >> 8) & 0xFF) * ((signs & 2) ? -1.0f : 1.0f));
            float w2 = db * ((float)((g.x >> 16) & 0xFF) * ((signs & 4) ? -1.0f : 1.0f));
            float w3 = db * ((float)(g.x >> 24) * ((signs & 8) ? -1.0f : 1.0f));
            float w4 = db * ((float)(g.y & 0xFF) * ((signs & 16) ? -1.0f : 1.0f));
            float w5 = db * ((float)((g.y >> 8) & 0xFF) * ((signs & 32) ? -1.0f : 1.0f));
            float w6 = db * ((float)((g.y >> 16) & 0xFF) * ((signs & 64) ? -1.0f : 1.0f));
            float w7 = db * ((float)(g.y >> 24) * ((signs & 128) ? -1.0f : 1.0f));

            uint col_base = b * 256 + x_offset;

            [unroll]
            for (uint i = 0; i < 32; i++)
            {
                uint t = t_base + i;
                uint safe_t = (t < batch_size) ? t : 0;
                uint t_x_byte = (safe_t * k_cols + col_base) * 4;
                float4 xv0 = asfloat(x.Load4(t_x_byte));
                float4 xv1 = asfloat(x.Load4(t_x_byte + 16));
                if (t < batch_size)
                {
                    acc[i] += w0 * xv0.x + w1 * xv0.y + w2 * xv0.z + w3 * xv0.w
                            + w4 * xv1.x + w5 * xv1.y + w6 * xv1.z + w7 * xv1.w;
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
    // 3. IQ3_XXS GEMV & Batched GEMM (256 elements in 100 aligned bytes)
    // =========================================================================

    public const string GemvIQ3_XXS = @"
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
" + IQ3_S_Helper + IQ2_KSIGNS_Table + @"
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
        uint row_byte_offset = warp_id * nb * 100;
        float row_sum = 0.0f;

        uint ib32 = lane_id >> 2;
        uint l = lane_id & 3;
        uint q_word_off = 4 + ib32 * 8 + ((l >= 2) ? 4 : 0);
        uint q_shift = (l & 1) * 16;
        uint aux_off = 68 + ib32 * 4;
        uint x_offset = lane_id * 8;

        for (uint b = 0; b < nb; b++)
        {
            uint blk_addr = row_byte_offset + b * 100;
            float d = f16tof32(W.Load(blk_addr) & 0xFFFF);

            uint aux32 = W.Load(blk_addr + aux_off);
            float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;

            uint signs_idx = (aux32 >> (7 * l)) & 127u;
            uint s_word = ksigns_iq2_tbl[signs_idx >> 2];
            uint signs = (s_word >> ((signs_idx & 3u) * 8)) & 0xFF;
            uint sn0 = signs & 0x0F;
            uint sn1 = signs >> 4;

            uint q_word = W.Load(blk_addr + q_word_off);
            uint qs0 = (q_word >> q_shift) & 0xFF;
            uint qs1 = (q_word >> (q_shift + 8)) & 0xFF;

            uint grid0 = decode_iq3s_grid(qs0);
            uint grid1 = decode_iq3s_grid(qs1);

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

    public const string GemmIQ3_XXSBatch = @"
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
" + IQ3_S_Helper + IQ2_KSIGNS_Table + @"
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
        uint row_byte_offset = warp_id * nb * 100;

        uint ib32 = lane_id >> 2;
        uint l = lane_id & 3;
        uint q_word_off = 4 + ib32 * 8 + ((l >= 2) ? 4 : 0);
        uint q_shift = (l & 1) * 16;
        uint aux_off = 68 + ib32 * 4;
        uint x_offset = lane_id * 8;

        for (uint b = 0; b < nb; b++)
        {
            uint blk_addr = row_byte_offset + b * 100;
            float d = f16tof32(W.Load(blk_addr) & 0xFFFF);

            uint aux32 = W.Load(blk_addr + aux_off);
            float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;

            uint signs_idx = (aux32 >> (7 * l)) & 127u;
            uint s_word = ksigns_iq2_tbl[signs_idx >> 2];
            uint signs = (s_word >> ((signs_idx & 3u) * 8)) & 0xFF;
            uint sn0 = signs & 0x0F;
            uint sn1 = signs >> 4;

            uint q_word = W.Load(blk_addr + q_word_off);
            uint qs0 = (q_word >> q_shift) & 0xFF;
            uint qs1 = (q_word >> (q_shift + 8)) & 0xFF;

            uint grid0 = decode_iq3s_grid(qs0);
            uint grid1 = decode_iq3s_grid(qs1);

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
                uint safe_t = (t < batch_size) ? t : 0;
                uint t_x_byte = (safe_t * k_cols + col_base) * 4;
                float4 xv0 = asfloat(x.Load4(t_x_byte));
                float4 xv1 = asfloat(x.Load4(t_x_byte + 16));
                if (t < batch_size)
                {
                    acc[i] += w0 * xv0.x + w1 * xv0.y + w2 * xv0.z + w3 * xv0.w
                            + w4 * xv1.x + w5 * xv1.y + w6 * xv1.z + w7 * xv1.w;
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
