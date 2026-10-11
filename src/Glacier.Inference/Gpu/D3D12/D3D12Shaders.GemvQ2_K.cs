namespace Glacier.Inference.Gpu.D3D12;

public static partial class D3D12Shaders
{
    public const string GemvQ2_K = @"
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

uint get_scale_q2(uint idx, uint sc0, uint sc1, uint sc2, uint sc3)
{
    if (idx < 4) return (sc0 >> (idx * 8)) & 0xFF;
    else if (idx < 8) return (sc1 >> ((idx - 4) * 8)) & 0xFF;
    else if (idx < 12) return (sc2 >> ((idx - 8) * 8)) & 0xFF;
    else return (sc3 >> ((idx - 12) * 8)) & 0xFF;
}

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

        for (uint b = 0; b < nb; b++)
        {
            uint blk_addr = row_byte_offset + b * 84;

            uint sc0 = W.Load(blk_addr + 0);
            uint sc1 = W.Load(blk_addr + 4);
            uint sc2 = W.Load(blk_addr + 8);
            uint sc3 = W.Load(blk_addr + 12);

            uint d_word = W.Load(blk_addr + 80);
            float d = f16tof32(d_word & 0xFFFF);
            float min_val = f16tof32((d_word >> 16) & 0xFFFF);

            uint x_blk = b * 256;

            [unroll]
            for (uint step = 0; step < 2; step++)
            {
                uint n = step * 128;
                uint q_base = blk_addr + 16 + step * 32;
                uint q_word = W.Load(q_base + (lane_id & ~3));
                uint q_byte = (q_word >> ((lane_id & 3) * 8)) & 0xFF;

                [unroll]
                for (uint j = 0; j < 4; j++)
                {
                    uint shift = j * 2;
                    uint is_idx = step * 8 + j * 2 + ((lane_id < 16) ? 0 : 1);
                    uint sc = get_scale_q2(is_idx, sc0, sc1, sc2, sc3);
                    float dl = d * (float)(sc & 0x0F);
                    float ml = min_val * (float)(sc >> 4);

                    uint q_val = (q_byte >> shift) & 3;
                    float val_x = x[x_blk + n + j * 32 + lane_id];

                    row_sum += (dl * (float)q_val - ml) * val_x;
                }
            }
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

    public const string GemmQ2_KBatch = @"
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

uint get_scale_q2_batch(uint idx, uint sc0, uint sc1, uint sc2, uint sc3)
{
    if (idx < 4) return (sc0 >> (idx * 8)) & 0xFF;
    else if (idx < 8) return (sc1 >> ((idx - 4) * 8)) & 0xFF;
    else if (idx < 12) return (sc2 >> ((idx - 8) * 8)) & 0xFF;
    else return (sc3 >> ((idx - 12) * 8)) & 0xFF;
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
        uint row_byte_offset = warp_id * nb * 84;

        for (uint b = 0; b < nb; b++)
        {
            uint blk_addr = row_byte_offset + b * 84;

            uint sc0 = W.Load(blk_addr + 0);
            uint sc1 = W.Load(blk_addr + 4);
            uint sc2 = W.Load(blk_addr + 8);
            uint sc3 = W.Load(blk_addr + 12);

            uint d_word = W.Load(blk_addr + 80);
            float d = f16tof32(d_word & 0xFFFF);
            float min_val = f16tof32((d_word >> 16) & 0xFFFF);

            uint x_blk = b * 256;

            [unroll]
            for (uint step = 0; step < 2; step++)
            {
                uint n = step * 128;
                uint q_base = blk_addr + 16 + step * 32;
                uint q_word = W.Load(q_base + (lane_id & ~3));
                uint q_byte = (q_word >> ((lane_id & 3) * 8)) & 0xFF;

                [unroll]
                for (uint j = 0; j < 4; j++)
                {
                    uint shift = j * 2;
                    uint is_idx = step * 8 + j * 2 + ((lane_id < 16) ? 0 : 1);
                    uint sc = get_scale_q2_batch(is_idx, sc0, sc1, sc2, sc3);
                    float dl = d * (float)(sc & 0x0F);
                    float ml = min_val * (float)(sc >> 4);

                    uint q_val = (q_byte >> shift) & 3;
                    float w_val = dl * (float)q_val - ml;
                    uint col_idx = x_blk + n + j * 32 + lane_id;

                    [unroll]
                    for (uint i = 0; i < 32; i++)
                    {
                        uint t = t_base + i;
                        if (t < batch_size)
                        {
                            acc[i] += w_val * asfloat(x.Load((t * k_cols + col_idx) * 4));
                        }
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

