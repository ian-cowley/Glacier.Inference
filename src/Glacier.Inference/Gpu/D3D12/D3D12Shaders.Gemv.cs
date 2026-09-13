namespace Glacier.Inference.Gpu.D3D12;

public static partial class D3D12Shaders
{
    public const string GemvQ4K = @"
cbuffer Params : register(b0)
{
    uint k_cols;
    uint m_rows;
    uint has_bias;
    uint has_residual;
    uint has_y;
};

ByteAddressBuffer W : register(t0);
StructuredBuffer<float> bias : register(t1);
RWByteAddressBuffer x : register(u0);
RWStructuredBuffer<float> residual : register(u1);
RWStructuredBuffer<float> y : register(u2);

groupshared float s_mem[128];

uint get_scale_byte(uint idx, uint s0, uint s1, uint s2)
{
    if (idx < 4) return (s0 >> (idx * 8)) & 0xFF;
    else if (idx < 8) return (s1 >> ((idx - 4) * 8)) & 0xFF;
    else return (s2 >> ((idx - 8) * 8)) & 0xFF;
}

void get_scale_min(uint j, uint s0, uint s1, uint s2, out float d_out, out float min_out, float d, float min_val)
{
    uint sc, m;
    if (j < 4) {
        sc = get_scale_byte(j, s0, s1, s2) & 63;
        m  = get_scale_byte(j + 4, s0, s1, s2) & 63;
    } else {
        sc = (get_scale_byte(j + 4, s0, s1, s2) & 0x0F) | ((get_scale_byte(j - 4, s0, s1, s2) >> 6) << 4);
        m  = (get_scale_byte(j + 4, s0, s1, s2) >> 4)   | ((get_scale_byte(j, s0, s1, s2) >> 6) << 4);
    }
    d_out = d * (float)sc;
    min_out = min_val * (float)m;
}

[numthreads(32, 4, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint row_in_grp = gtid.y;
    uint warp_id = gid.x * 4 + row_in_grp;
    uint lane_id = gtid.x;
    uint s_idx = row_in_grp * 32 + lane_id;

    if (warp_id < m_rows)
    {
        uint nb = k_cols / 256;
        uint row_byte_offset = warp_id * nb * 144;
        float row_sum = 0.0f;

        uint chunk = lane_id / 8;
        uint chunk_lane = lane_id % 8;
        uint is_idx = chunk * 2;
        uint x_offset1 = chunk * 64 + chunk_lane * 4;
        uint x_offset2 = x_offset1 + 32;

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

            uint x_blk = b * 256;
            float4 x1 = asfloat(x.Load4((x_blk + x_offset1) * 4));
            float4 x2 = asfloat(x.Load4((x_blk + x_offset2) * 4));

            uint q = W.Load(blk_addr + 16 + lane_id * 4);
            uint q0 = q & 0xFF;
            uint q1 = (q >> 8) & 0xFF;
            uint q2 = (q >> 16) & 0xFF;
            uint q3 = q >> 24;

            float dot1 = (float)(q0 & 0x0F) * x1.x + (float)(q1 & 0x0F) * x1.y + (float)(q2 & 0x0F) * x1.z + (float)(q3 & 0x0F) * x1.w;
            float sum_x1 = x1.x + x1.y + x1.z + x1.w;

            float dot2 = (float)(q0 >> 4) * x2.x + (float)(q1 >> 4) * x2.y + (float)(q2 >> 4) * x2.z + (float)(q3 >> 4) * x2.w;
            float sum_x2 = x2.x + x2.y + x2.z + x2.w;

            row_sum += (d1 * dot1 - min1 * sum_x1) + (d2 * dot2 - min2 * sum_x2);
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

    public const string GemvQ5K = @"
cbuffer Params : register(b0)
{
    uint k_cols;
    uint m_rows;
    uint has_bias;
    uint has_residual;
    uint has_y;
};

ByteAddressBuffer W : register(t0);
StructuredBuffer<float> bias : register(t1);
RWStructuredBuffer<float> x : register(u0);
RWStructuredBuffer<float> residual : register(u1);
RWStructuredBuffer<float> y : register(u2);

groupshared float s_mem[128];

uint get_scale_byte_q5(uint idx, uint s0, uint s1, uint s2)
{
    if (idx < 4) return (s0 >> (idx * 8)) & 0xFF;
    else if (idx < 8) return (s1 >> ((idx - 4) * 8)) & 0xFF;
    else return (s2 >> ((idx - 8) * 8)) & 0xFF;
}

void get_scale_min_q5(uint j, uint s0, uint s1, uint s2, out float d_out, out float min_out, float d, float min_val)
{
    uint sc, m;
    if (j < 4) {
        sc = get_scale_byte_q5(j, s0, s1, s2) & 63;
        m  = get_scale_byte_q5(j + 4, s0, s1, s2) & 63;
    } else {
        sc = (get_scale_byte_q5(j + 4, s0, s1, s2) & 0x0F) | ((get_scale_byte_q5(j - 4, s0, s1, s2) >> 6) << 4);
        m  = (get_scale_byte_q5(j + 4, s0, s1, s2) >> 4)   | ((get_scale_byte_q5(j, s0, s1, s2) >> 6) << 4);
    }
    d_out = d * (float)sc;
    min_out = min_val * (float)m;
}

[numthreads(32, 4, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint row_in_grp = gtid.y;
    uint warp_id = gid.x * 4 + row_in_grp;
    uint lane_id = gtid.x;
    uint s_idx = row_in_grp * 32 + lane_id;

    if (warp_id < m_rows)
    {
        uint nb = k_cols / 256;
        uint row_byte_offset = warp_id * nb * 176;
        float row_sum = 0.0f;

        for (uint b = 0; b < nb; b++)
        {
            uint blk_addr = row_byte_offset + b * 176;
            uint2 d_hdr = W.Load2(blk_addr);
            float d = f16tof32(d_hdr.x & 0xFFFF);
            float min_val = f16tof32(d_hdr.x >> 16);

            uint s0 = d_hdr.y;
            uint s1 = W.Load(blk_addr + 8);
            uint s2 = W.Load(blk_addr + 12);

            uint qh_word = W.Load(blk_addr + 16 + (lane_id & ~3));
            uint qh_val = (qh_word >> ((lane_id & 3) * 8)) & 0xFF;

            uint x_blk = b * 256;

            [unroll]
            for (uint c = 0; c < 4; c++)
            {
                uint is_idx = c * 2;
                float d1, min1, d2, min2;
                get_scale_min_q5(is_idx + 0, s0, s1, s2, d1, min1, d, min_val);
                get_scale_min_q5(is_idx + 1, s0, s1, s2, d2, min2, d, min_val);

                uint ql_word = W.Load(blk_addr + 48 + c * 32 + (lane_id & ~3));
                uint ql_val = (ql_word >> ((lane_id & 3) * 8)) & 0xFF;

                uint bit0 = (qh_val >> (c * 2 + 0)) & 1;
                uint bit1 = (qh_val >> (c * 2 + 1)) & 1;

                int q0 = (int)((ql_val & 0x0F) | (bit0 << 4));
                int q1 = (int)((ql_val >> 4)   | (bit1 << 4));

                float x0 = x[x_blk + c * 64 + lane_id];
                float x1 = x[x_blk + c * 64 + 32 + lane_id];

                row_sum += (d1 * (float)q0 - min1) * x0 + (d2 * (float)q1 - min2) * x1;
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

    public const string GemvQ3K = @"
cbuffer Params : register(b0)
{
    uint k_cols;
    uint m_rows;
    uint has_bias;
    uint has_residual;
    uint has_y;
};

ByteAddressBuffer W : register(t0);
StructuredBuffer<float> bias : register(t1);
RWStructuredBuffer<float> x : register(u0);
RWStructuredBuffer<float> residual : register(u1);
RWStructuredBuffer<float> y : register(u2);

groupshared float s_mem[128];

int get_scale_q3(uint idx, uint sc0, uint sc1, uint sc2, uint sc3)
{
    uint b;
    if (idx < 4) b = (sc0 >> (idx * 8)) & 0xFF;
    else if (idx < 8) b = (sc1 >> ((idx - 4) * 8)) & 0xFF;
    else if (idx < 12) b = (sc2 >> ((idx - 8) * 8)) & 0xFF;
    else b = (sc3 >> ((idx - 12) * 8)) & 0xFF;
    return (int)b - 32;
}

[numthreads(32, 4, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint row_in_grp = gtid.y;
    uint warp_id = gid.x * 4 + row_in_grp;
    uint lane_id = gtid.x;
    uint s_idx = row_in_grp * 32 + lane_id;

    if (warp_id < m_rows)
    {
        uint nb = k_cols / 256;
        uint row_byte_offset = warp_id * nb * 112;
        float row_sum = 0.0f;

        for (uint b = 0; b < nb; b++)
        {
            uint blk_addr = row_byte_offset + b * 112;
            
            uint raw_aux0 = W.Load(blk_addr + 96);
            uint raw_aux1 = W.Load(blk_addr + 100);
            uint raw_aux2 = W.Load(blk_addr + 104);
            
            const uint kmask1 = 0x03030303;
            const uint kmask2 = 0x0F0F0F0F;
            uint sc0 = (raw_aux0 & kmask2) | (((raw_aux2 >> 0) & kmask1) << 4);
            uint sc1 = (raw_aux1 & kmask2) | (((raw_aux2 >> 2) & kmask1) << 4);
            uint sc2 = ((raw_aux0 >> 4) & kmask2) | (((raw_aux2 >> 4) & kmask1) << 4);
            uint sc3 = ((raw_aux1 >> 4) & kmask2) | (((raw_aux2 >> 6) & kmask1) << 4);

            float d_all = f16tof32(W.Load(blk_addr + 108) & 0xFFFF);

            uint hm_word = W.Load(blk_addr + (lane_id & ~3));
            uint hm_byte = (hm_word >> ((lane_id & 3) * 8)) & 0xFF;

            uint x_blk = b * 256;

            [unroll]
            for (uint step = 0; step < 2; step++)
            {
                uint n = step * 128;
                uint q_base = blk_addr + 32 + step * 32;
                uint q_word = W.Load(q_base + (lane_id & ~3));
                uint q_byte = (q_word >> ((lane_id & 3) * 8)) & 0xFF;

                [unroll]
                for (uint j = 0; j < 4; j++)
                {
                    uint shift = j * 2;
                    uint m = 1 << (step * 4 + j);

                    uint is_idx = step * 8 + j * 2 + ((lane_id < 16) ? 0 : 1);
                    int s = get_scale_q3(is_idx, sc0, sc1, sc2, sc3);
                    float dl = d_all * (float)s;

                    int q = (int)((q_byte >> shift) & 3) - (((hm_byte & m) != 0) ? 0 : 4);
                    float val_x = x[x_blk + n + j * 32 + lane_id];

                    row_sum += (dl * (float)q) * val_x;
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

    public const string GemvQ6K = @"
cbuffer Params : register(b0)
{
    uint k_cols;
    uint m_rows;
    uint has_bias;
    uint has_residual;
    uint has_y;
};

ByteAddressBuffer W : register(t0);
StructuredBuffer<float> bias : register(t1);
RWStructuredBuffer<float> x : register(u0);
RWStructuredBuffer<float> residual : register(u1);
RWStructuredBuffer<float> y : register(u2);

groupshared float s_mem[128];

int decode_signed_byte(uint b)
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

    if (warp_id < m_rows)
    {
        uint nb = k_cols / 256;
        // Block size aligned to 212 bytes (4-byte aligned)
        uint row_byte_offset = warp_id * nb * 212;
        float row_sum = 0.0f;
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
                int s1 = decode_signed_byte((sc_words.x >> (is_idx * 8)) & 0xFF);
                int s2 = decode_signed_byte((sc_words.x >> ((is_idx + 2) * 8)) & 0xFF);
                int s3 = decode_signed_byte((sc_words.y >> (is_idx * 8)) & 0xFF);
                int s4 = decode_signed_byte((sc_words.y >> ((is_idx + 2) * 8)) & 0xFF);

                float w1 = d * (float)s1 * (float)q1;
                float w2 = d * (float)s2 * (float)q2;
                float w3 = d * (float)s3 * (float)q3;
                float w4 = d * (float)s4 * (float)q4;

                row_sum += w1 * x[x_blk + n + lane_id + 0]
                         + w2 * x[x_blk + n + lane_id + 32]
                         + w3 * x[x_blk + n + lane_id + 64]
                         + w4 * x[x_blk + n + lane_id + 96];
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

    public const string GemvQ8_0 = @"
cbuffer Params : register(b0)
{
    uint k_cols;
    uint m_rows;
    uint has_bias;
    uint has_residual;
    uint has_y;
};

ByteAddressBuffer W : register(t0);
StructuredBuffer<float> bias : register(t1);
RWByteAddressBuffer x : register(u0);
RWStructuredBuffer<float> residual : register(u1);
RWStructuredBuffer<float> y : register(u2);

groupshared float s_mem[128];

[numthreads(32, 4, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint row_in_grp = gtid.y;
    uint warp_id = gid.x * 4 + row_in_grp;
    uint lane_id = gtid.x;
    uint s_idx = row_in_grp * 32 + lane_id;

    if (warp_id < m_rows)
    {
        uint nb = k_cols / 32;
        uint row_byte_offset = warp_id * nb * 36;
        float row_sum = 0.0f;

        uint q_word_offset = (lane_id / 4) * 4;
        uint q_shift = (lane_id % 4) * 8;

        for (uint b = 0; b < nb; b++)
        {
            uint blk_addr = row_byte_offset + b * 36;
            uint d_raw = W.Load(blk_addr) & 0xFFFF;
            float d = f16tof32(d_raw);

            uint q_word = W.Load(blk_addr + 4 + q_word_offset);
            int q = (int)(q_word << (24 - q_shift)) >> 24;

            float xv = asfloat(x.Load((b * 32 + lane_id) * 4));
            row_sum += d * ((float)q * xv);
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

    public const string GemvFp32 = @"
cbuffer Params : register(b0)
{
    uint k_cols;
    uint m_rows;
    uint has_bias;
    uint has_residual;
    uint has_y;
};

StructuredBuffer<float> W : register(t0);
StructuredBuffer<float> bias : register(t1);
RWStructuredBuffer<float> x : register(u0);
RWStructuredBuffer<float> residual : register(u1);
RWStructuredBuffer<float> y : register(u2);

groupshared float s_mem[128];

[numthreads(32, 4, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint row_in_grp = gtid.y;
    uint warp_id = gid.x * 4 + row_in_grp;
    uint lane_id = gtid.x;
    uint s_idx = row_in_grp * 32 + lane_id;

    if (warp_id < m_rows)
    {
        uint row_offset = warp_id * k_cols;
        float row_sum = 0.0f;

        for (uint i = lane_id; i < k_cols; i += 32)
        {
            row_sum += W[row_offset + i] * x[i];
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
}
