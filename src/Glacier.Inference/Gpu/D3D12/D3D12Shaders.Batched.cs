namespace Glacier.Inference.Gpu.D3D12;

public static partial class D3D12Shaders
{
    public const string GemmQ4KBatch = @"
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
StructuredBuffer<float> bias : register(t1);
ByteAddressBuffer x : register(t2);
RWStructuredBuffer<float> residual : register(u0);
RWStructuredBuffer<float> y : register(u1);

groupshared float s_mem[4096];

void get_scale_min(uint j, uint s0, uint s1, uint s2, out float d, out float m, float d_val, float min_val)
{
    uint sc, m_code;
    if (j < 4)
    {
        sc = (s0 >> (j * 8)) & 63;
        m_code = (s1 >> (j * 8)) & 63;
    }
    else
    {
        sc = ((s2 >> ((j - 4) * 8)) & 0xF) | (((s0 >> ((j - 4) * 8 + 6)) & 3) << 4);
        m_code = ((s2 >> ((j - 4) * 8 + 4)) & 0xF) | (((s1 >> ((j - 4) * 8 + 6)) & 3) << 4);
    }
    d = d_val * (float)sc;
    m = min_val * (float)m_code;
}

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
    for (uint i = 0; i < 32; i++)
    {
        acc[i] = 0.0f;
    }

    if (warp_id < m_rows)
    {
        uint nb = k_cols / 256;
        uint row_byte_offset = warp_id * nb * 144;

        uint chunk = lane_id / 8;
        uint chunk_lane = lane_id % 8;
        uint is_idx = chunk * 2;
        uint x_offset1 = (chunk * 64 + chunk_lane * 4) * 4;
        uint stride_bytes = k_cols * 4;
        uint t_base_bytes = t_base * stride_bytes;

        for (uint b = 0; b < nb; b++)
        {
            uint blk_addr = row_byte_offset + b * 144;
            uint4 header = W.Load4(blk_addr);
            uint d_dmin = header.x;
            float d = f16tof32(d_dmin & 0xFFFF);
            float min_val = f16tof32(d_dmin >> 16);

            uint s0 = header.y;
            uint s1 = header.z;
            uint s2 = header.w;

            float d1, min1, d2, min2;
            get_scale_min(is_idx + 0, s0, s1, s2, d1, min1, d, min_val);
            get_scale_min(is_idx + 1, s0, s1, s2, d2, min2, d, min_val);

            uint q = W.Load(blk_addr + 16 + lane_id * 4);
            uint q0 = q & 0xFF;
            uint q1 = (q >> 8) & 0xFF;
            uint q2 = (q >> 16) & 0xFF;
            uint q3 = q >> 24;

            float4 w_vec1 = d1 * float4((float)(q0 & 0x0F), (float)(q1 & 0x0F), (float)(q2 & 0x0F), (float)(q3 & 0x0F)) - min1;
            float4 w_vec2 = d2 * float4((float)(q0 >> 4), (float)(q1 >> 4), (float)(q2 >> 4), (float)(q3 >> 4)) - min2;

            uint x_addr_base = t_base_bytes + (b * 256 * 4) + x_offset1;

            [unroll]
            for (uint i = 0; i < 32; i++)
            {
                uint t = t_base + i;
                if (t < batch_size)
                {
                    uint cur_addr = x_addr_base + i * stride_bytes;
                    float4 x1 = asfloat(x.Load4(cur_addr));
                    float4 x2 = asfloat(x.Load4(cur_addr + 128));

                    acc[i] += dot(w_vec1, x1) + dot(w_vec2, x2);
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
    if (lane_id < 16)
    {
        [unroll]
        for (uint i = 0; i < 32; i++) s_mem[i * 128 + s_idx] += s_mem[i * 128 + s_idx + 16];
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 8)
    {
        [unroll]
        for (uint i = 0; i < 32; i++) s_mem[i * 128 + s_idx] += s_mem[i * 128 + s_idx + 8];
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 4)
    {
        [unroll]
        for (uint i = 0; i < 32; i++) s_mem[i * 128 + s_idx] += s_mem[i * 128 + s_idx + 4];
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 2)
    {
        [unroll]
        for (uint i = 0; i < 32; i++) s_mem[i * 128 + s_idx] += s_mem[i * 128 + s_idx + 2];
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 1)
    {
        [unroll]
        for (uint i = 0; i < 32; i++) s_mem[i * 128 + s_idx] += s_mem[i * 128 + s_idx + 1];
    }

    if (lane_id == 0 && warp_id < m_rows)
    {
        float b_val = (has_bias != 0) ? bias[warp_id] : 0.0f;
        [unroll]
        for (uint i = 0; i < 32; i++)
        {
            uint t = t_base + i;
            if (t < batch_size)
            {
                float final_sum = s_mem[i * 128 + row_in_grp * 32] + b_val;
                uint out_idx = t * m_rows + warp_id;
                if (has_residual != 0) residual[out_idx] += final_sum;
                if (has_y != 0) y[out_idx] = final_sum;
            }
        }
    }
}
";

    public const string GemmQ6KBatch = @"
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
StructuredBuffer<float> bias : register(t1);
StructuredBuffer<float> x : register(t2);
RWStructuredBuffer<float> residual : register(u0);
RWStructuredBuffer<float> y : register(u1);

groupshared float s_mem[4096];

int decode_signed_byte_batch(uint b)
{
    return (b >= 128) ? (int)b - 256 : (int)b;
}

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
    for (uint i = 0; i < 32; i++)
    {
        acc[i] = 0.0f;
    }

    if (warp_id < m_rows)
    {
        uint nb = k_cols / 256;
        uint row_byte_offset = warp_id * nb * 212;
        uint is_idx = lane_id / 16;

        for (uint b = 0; b < nb; b++)
        {
            uint blk_addr = row_byte_offset + b * 212;
            uint d_raw = W.Load(blk_addr + 208) & 0xFFFF;
            float d = f16tof32(d_raw);
            uint x_blk = b * 256;

            [unroll]
            for (uint step = 0; step < 2; step++)
            {
                uint n = step * 128;
                uint ql_addr = blk_addr + step * 64;
                uint qh_addr = blk_addr + 128 + step * 32;
                uint sc_addr = blk_addr + 192 + step * 8;

                uint qh_word = W.Load(qh_addr + (lane_id & ~3));
                uint qh_val = (qh_word >> ((lane_id & 3) * 8)) & 0xFF;

                uint ql_low_word = W.Load(ql_addr + (lane_id & ~3));
                uint ql_low = (ql_low_word >> ((lane_id & 3) * 8)) & 0xFF;

                uint ql_high_word = W.Load(ql_addr + 32 + (lane_id & ~3));
                uint ql_high = (ql_high_word >> ((lane_id & 3) * 8)) & 0xFF;

                int q1 = (int)((ql_low & 0x0F) | (((qh_val >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql_high & 0x0F) | (((qh_val >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql_low >> 4) | (((qh_val >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql_high >> 4) | (((qh_val >> 6) & 3) << 4)) - 32;

                uint2 sc_words = W.Load2(sc_addr);
                int s1 = decode_signed_byte_batch((sc_words.x >> (is_idx * 8)) & 0xFF);
                int s2 = decode_signed_byte_batch((sc_words.x >> ((is_idx + 2) * 8)) & 0xFF);
                int s3 = decode_signed_byte_batch((sc_words.y >> (is_idx * 8)) & 0xFF);
                int s4 = decode_signed_byte_batch((sc_words.y >> ((is_idx + 2) * 8)) & 0xFF);

                float w1 = d * (float)s1 * (float)q1;
                float w2 = d * (float)s2 * (float)q2;
                float w3 = d * (float)s3 * (float)q3;
                float w4 = d * (float)s4 * (float)q4;

                [unroll]
                for (uint i = 0; i < 32; i++)
                {
                    uint t = t_base + i;
                    if (t < batch_size)
                    {
                        uint t_x_base = t * k_cols + x_blk + n;
                        acc[i] += w1 * x[t_x_base + lane_id + 0]
                                + w2 * x[t_x_base + lane_id + 32]
                                + w3 * x[t_x_base + lane_id + 64]
                                + w4 * x[t_x_base + lane_id + 96];
                    }
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
    if (lane_id < 16)
    {
        [unroll]
        for (uint i = 0; i < 32; i++) s_mem[i * 128 + s_idx] += s_mem[i * 128 + s_idx + 16];
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 8)
    {
        [unroll]
        for (uint i = 0; i < 32; i++) s_mem[i * 128 + s_idx] += s_mem[i * 128 + s_idx + 8];
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 4)
    {
        [unroll]
        for (uint i = 0; i < 32; i++) s_mem[i * 128 + s_idx] += s_mem[i * 128 + s_idx + 4];
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 2)
    {
        [unroll]
        for (uint i = 0; i < 32; i++) s_mem[i * 128 + s_idx] += s_mem[i * 128 + s_idx + 2];
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 1)
    {
        [unroll]
        for (uint i = 0; i < 32; i++) s_mem[i * 128 + s_idx] += s_mem[i * 128 + s_idx + 1];
    }

    if (lane_id == 0 && warp_id < m_rows)
    {
        float b_val = (has_bias != 0) ? bias[warp_id] : 0.0f;
        [unroll]
        for (uint i = 0; i < 32; i++)
        {
            uint t = t_base + i;
            if (t < batch_size)
            {
                float final_sum = s_mem[i * 128 + row_in_grp * 32] + b_val;
                uint out_idx = t * m_rows + warp_id;
                if (has_residual != 0) residual[out_idx] += final_sum;
                if (has_y != 0) y[out_idx] = final_sum;
            }
        }
    }
}
";

    public const string GemmQ8_0Batch = @"
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
StructuredBuffer<float> bias : register(t1);
StructuredBuffer<float> x : register(t2);
RWStructuredBuffer<float> residual : register(u0);
RWStructuredBuffer<float> y : register(u1);

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
    for (uint i = 0; i < 32; i++)
    {
        acc[i] = 0.0f;
    }

    if (warp_id < m_rows)
    {
        uint nb = k_cols / 32;
        uint row_byte_offset = warp_id * nb * 36;
        uint q_word_offset = (lane_id / 4) * 4;
        uint q_shift = (lane_id % 4) * 8;

        for (uint b = 0; b < nb; b++)
        {
            uint blk_addr = row_byte_offset + b * 36;
            uint d_raw = W.Load(blk_addr) & 0xFFFF;
            float d = f16tof32(d_raw);

            uint q_word = W.Load(blk_addr + 4 + q_word_offset);
            int q = (int)(q_word << (24 - q_shift)) >> 24;
            float w_val = d * (float)q;

            uint col_idx = b * 32 + lane_id;

            [unroll]
            for (uint i = 0; i < 32; i++)
            {
                uint t = t_base + i;
                if (t < batch_size)
                {
                    acc[i] += w_val * x[t * k_cols + col_idx];
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
    if (lane_id < 16)
    {
        [unroll]
        for (uint i = 0; i < 32; i++) s_mem[i * 128 + s_idx] += s_mem[i * 128 + s_idx + 16];
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 8)
    {
        [unroll]
        for (uint i = 0; i < 32; i++) s_mem[i * 128 + s_idx] += s_mem[i * 128 + s_idx + 8];
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 4)
    {
        [unroll]
        for (uint i = 0; i < 32; i++) s_mem[i * 128 + s_idx] += s_mem[i * 128 + s_idx + 4];
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 2)
    {
        [unroll]
        for (uint i = 0; i < 32; i++) s_mem[i * 128 + s_idx] += s_mem[i * 128 + s_idx + 2];
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 1)
    {
        [unroll]
        for (uint i = 0; i < 32; i++) s_mem[i * 128 + s_idx] += s_mem[i * 128 + s_idx + 1];
    }

    if (lane_id == 0 && warp_id < m_rows)
    {
        float b_val = (has_bias != 0) ? bias[warp_id] : 0.0f;
        [unroll]
        for (uint i = 0; i < 32; i++)
        {
            uint t = t_base + i;
            if (t < batch_size)
            {
                float final_sum = s_mem[i * 128 + row_in_grp * 32] + b_val;
                uint out_idx = t * m_rows + warp_id;
                if (has_residual != 0) residual[out_idx] += final_sum;
                if (has_y != 0) y[out_idx] = final_sum;
            }
        }
    }
}
";

    public const string GemmFp32Batch = @"
cbuffer Params : register(b0)
{
    uint k_cols;
    uint m_rows;
    uint batch_size;
    uint has_bias;
    uint has_residual;
    uint has_y;
};

StructuredBuffer<float> W : register(t0);
StructuredBuffer<float> bias : register(t1);
StructuredBuffer<float> x : register(t2);
RWStructuredBuffer<float> residual : register(u0);
RWStructuredBuffer<float> y : register(u1);

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
    for (uint i = 0; i < 32; i++)
    {
        acc[i] = 0.0f;
    }

    if (warp_id < m_rows)
    {
        uint row_offset = warp_id * k_cols;

        for (uint k = lane_id; k < k_cols; k += 32)
        {
            float w = W[row_offset + k];
            [unroll]
            for (uint i = 0; i < 32; i++)
            {
                uint t = t_base + i;
                if (t < batch_size)
                {
                    acc[i] += w * x[t * k_cols + k];
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
    if (lane_id < 16)
    {
        [unroll]
        for (uint i = 0; i < 32; i++) s_mem[i * 128 + s_idx] += s_mem[i * 128 + s_idx + 16];
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 8)
    {
        [unroll]
        for (uint i = 0; i < 32; i++) s_mem[i * 128 + s_idx] += s_mem[i * 128 + s_idx + 8];
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 4)
    {
        [unroll]
        for (uint i = 0; i < 32; i++) s_mem[i * 128 + s_idx] += s_mem[i * 128 + s_idx + 4];
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 2)
    {
        [unroll]
        for (uint i = 0; i < 32; i++) s_mem[i * 128 + s_idx] += s_mem[i * 128 + s_idx + 2];
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 1)
    {
        [unroll]
        for (uint i = 0; i < 32; i++) s_mem[i * 128 + s_idx] += s_mem[i * 128 + s_idx + 1];
    }

    if (lane_id == 0 && warp_id < m_rows)
    {
        float b_val = (has_bias != 0) ? bias[warp_id] : 0.0f;
        [unroll]
        for (uint i = 0; i < 32; i++)
        {
            uint t = t_base + i;
            if (t < batch_size)
            {
                float final_sum = s_mem[i * 128 + row_in_grp * 32] + b_val;
                uint out_idx = t * m_rows + warp_id;
                if (has_residual != 0) residual[out_idx] += final_sum;
                if (has_y != 0) y[out_idx] = final_sum;
            }
        }
    }
}
";
}
