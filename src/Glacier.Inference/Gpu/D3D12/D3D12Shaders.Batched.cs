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
ByteAddressBuffer bias : register(t1);
RWByteAddressBuffer x : register(u2);
RWByteAddressBuffer residual : register(u0);
RWByteAddressBuffer y : register(u1);

groupshared float4 s_mem0[128];
groupshared float4 s_mem1[128];

uint get_scale_byte_q4_batch(uint idx, uint s0, uint s1, uint s2)
{
    if (idx < 4) return (s0 >> (idx * 8)) & 0xFF;
    else if (idx < 8) return (s1 >> ((idx - 4) * 8)) & 0xFF;
    else return (s2 >> ((idx - 8) * 8)) & 0xFF;
}

void get_scale_min(uint j, uint s0, uint s1, uint s2, out float d, out float m, float d_val, float min_val)
{
    uint sc, m_code;
    if (j < 4)
    {
        sc = get_scale_byte_q4_batch(j, s0, s1, s2) & 63;
        m_code = get_scale_byte_q4_batch(j + 4, s0, s1, s2) & 63;
    }
    else
    {
        sc = (get_scale_byte_q4_batch(j + 4, s0, s1, s2) & 0x0F) | ((get_scale_byte_q4_batch(j - 4, s0, s1, s2) >> 6) << 4);
        m_code = (get_scale_byte_q4_batch(j + 4, s0, s1, s2) >> 4) | ((get_scale_byte_q4_batch(j, s0, s1, s2) >> 6) << 4);
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
    uint t_base = gid.y * 8;
    uint valid_t = (t_base < batch_size) ? min(8u, batch_size - t_base) : 0u;

    float4 sum0 = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 sum1 = float4(0.0f, 0.0f, 0.0f, 0.0f);

    if (warp_id < m_rows && valid_t > 0)
    {
        uint nb = k_cols / 256;
        uint row_byte_offset = warp_id * nb * 144;

        uint chunk = lane_id / 8;
        uint chunk_lane = lane_id % 8;
        uint is_idx = chunk * 2;
        uint x_offset1 = chunk * 64 + chunk_lane * 4;
        uint x_offset2 = x_offset1 + 32;

        if (valid_t >= 8)
        {
            uint base0 = (t_base + 0) * k_cols;
            uint base1 = (t_base + 1) * k_cols;
            uint base2 = (t_base + 2) * k_cols;
            uint base3 = (t_base + 3) * k_cols;
            uint base4 = (t_base + 4) * k_cols;
            uint base5 = (t_base + 5) * k_cols;
            uint base6 = (t_base + 6) * k_cols;
            uint base7 = (t_base + 7) * k_cols;

            for (uint b = 0; b < nb; b++)
            {
                uint blk_addr = row_byte_offset + b * 144;
                uint4 header = W.Load4(blk_addr);
                float d = f16tof32(header.x & 0xFFFF);
                float min_val = f16tof32(header.x >> 16);

                float d1, min1, d2, min2;
                get_scale_min(is_idx + 0, header.y, header.z, header.w, d1, min1, d, min_val);
                get_scale_min(is_idx + 1, header.y, header.z, header.w, d2, min2, d, min_val);

                uint q = W.Load(blk_addr + 16 + lane_id * 4);
                uint q0 = q & 0xFF;
                uint q1 = (q >> 8) & 0xFF;
                uint q2 = (q >> 16) & 0xFF;
                uint q3 = q >> 24;

                float4 w1 = d1 * float4((float)(q0 & 0x0F), (float)(q1 & 0x0F), (float)(q2 & 0x0F), (float)(q3 & 0x0F)) - min1;
                float4 w2 = d2 * float4((float)(q0 >> 4), (float)(q1 >> 4), (float)(q2 >> 4), (float)(q3 >> 4)) - min2;

                uint k1 = b * 256 + x_offset1;
                uint k2 = b * 256 + x_offset2;

                sum0.x += dot(w1, asfloat(x.Load4((base0 + k1) * 4))) + dot(w2, asfloat(x.Load4((base0 + k2) * 4)));
                sum0.y += dot(w1, asfloat(x.Load4((base1 + k1) * 4))) + dot(w2, asfloat(x.Load4((base1 + k2) * 4)));
                sum0.z += dot(w1, asfloat(x.Load4((base2 + k1) * 4))) + dot(w2, asfloat(x.Load4((base2 + k2) * 4)));
                sum0.w += dot(w1, asfloat(x.Load4((base3 + k1) * 4))) + dot(w2, asfloat(x.Load4((base3 + k2) * 4)));
                sum1.x += dot(w1, asfloat(x.Load4((base4 + k1) * 4))) + dot(w2, asfloat(x.Load4((base4 + k2) * 4)));
                sum1.y += dot(w1, asfloat(x.Load4((base5 + k1) * 4))) + dot(w2, asfloat(x.Load4((base5 + k2) * 4)));
                sum1.z += dot(w1, asfloat(x.Load4((base6 + k1) * 4))) + dot(w2, asfloat(x.Load4((base6 + k2) * 4)));
                sum1.w += dot(w1, asfloat(x.Load4((base7 + k1) * 4))) + dot(w2, asfloat(x.Load4((base7 + k2) * 4)));
            }
        }
        else
        {
            for (uint b = 0; b < nb; b++)
            {
                uint blk_addr = row_byte_offset + b * 144;
                uint4 header = W.Load4(blk_addr);
                float d = f16tof32(header.x & 0xFFFF);
                float min_val = f16tof32(header.x >> 16);

                float d1, min1, d2, min2;
                get_scale_min(is_idx + 0, header.y, header.z, header.w, d1, min1, d, min_val);
                get_scale_min(is_idx + 1, header.y, header.z, header.w, d2, min2, d, min_val);

                uint q = W.Load(blk_addr + 16 + lane_id * 4);
                uint q0 = q & 0xFF;
                uint q1 = (q >> 8) & 0xFF;
                uint q2 = (q >> 16) & 0xFF;
                uint q3 = q >> 24;

                float4 w1 = d1 * float4((float)(q0 & 0x0F), (float)(q1 & 0x0F), (float)(q2 & 0x0F), (float)(q3 & 0x0F)) - min1;
                float4 w2 = d2 * float4((float)(q0 >> 4), (float)(q1 >> 4), (float)(q2 >> 4), (float)(q3 >> 4)) - min2;

                uint k1 = b * 256 + x_offset1;
                uint k2 = b * 256 + x_offset2;

                [unroll]
                for (uint tt = 0; tt < 4; tt++)
                {
                    if (tt < valid_t)
                    {
                        uint base_t = (t_base + tt) * k_cols;
                        float4 x1 = asfloat(x.Load4((base_t + k1) * 4));
                        float4 x2 = asfloat(x.Load4((base_t + k2) * 4));
                        sum0[tt] += dot(w1, x1) + dot(w2, x2);
                    }
                    if ((tt + 4) < valid_t)
                    {
                        uint base_t = (t_base + tt + 4) * k_cols;
                        float4 x1 = asfloat(x.Load4((base_t + k1) * 4));
                        float4 x2 = asfloat(x.Load4((base_t + k2) * 4));
                        sum1[tt] += dot(w1, x1) + dot(w2, x2);
                    }
                }
            }
        }
    }

    s_mem0[s_idx] = sum0;
    s_mem1[s_idx] = sum1;
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 16) { s_mem0[s_idx] += s_mem0[s_idx + 16]; s_mem1[s_idx] += s_mem1[s_idx + 16]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 8)  { s_mem0[s_idx] += s_mem0[s_idx + 8];  s_mem1[s_idx] += s_mem1[s_idx + 8]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 4)  { s_mem0[s_idx] += s_mem0[s_idx + 4];  s_mem1[s_idx] += s_mem1[s_idx + 4]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 2)  { s_mem0[s_idx] += s_mem0[s_idx + 2];  s_mem1[s_idx] += s_mem1[s_idx + 2]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 1)  { s_mem0[s_idx] += s_mem0[s_idx + 1];  s_mem1[s_idx] += s_mem1[s_idx + 1]; }
    GroupMemoryBarrierWithGroupSync();

    if (lane_id < valid_t && warp_id < m_rows)
    {
        uint base_s = row_in_grp * 32;
        float final_sum = (lane_id < 4) ? s_mem0[base_s][lane_id] : s_mem1[base_s][lane_id - 4];
        if (has_bias != 0) final_sum += asfloat(bias.Load(warp_id * 4));
        uint out_byte_addr = ((t_base + lane_id) * m_rows + warp_id) * 4;

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
ByteAddressBuffer bias : register(t1);
RWByteAddressBuffer x : register(u2);
RWByteAddressBuffer residual : register(u0);
RWByteAddressBuffer y : register(u1);

groupshared float4 s_mem0[128];
groupshared float4 s_mem1[128];

int decode_signed_byte_batch(uint b)
{
    return (int)(b << 24) >> 24;
}

[numthreads(32, 4, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint row_in_grp = gtid.y;
    uint warp_id = gid.x * 4 + row_in_grp;
    uint lane_id = gtid.x;
    uint s_idx = row_in_grp * 32 + lane_id;
    uint t_base = gid.y * 8;
    uint valid_t = (t_base < batch_size) ? min(8u, batch_size - t_base) : 0u;

    float4 sum0 = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 sum1 = float4(0.0f, 0.0f, 0.0f, 0.0f);

    if (warp_id < m_rows && valid_t > 0)
    {
        uint nb = k_cols / 256;
        uint row_byte_offset = warp_id * nb * 212;
        uint is_idx = lane_id / 16;
        uint lane_byte = (lane_id & 3) * 8;
        uint lane_dword = lane_id & ~3;

        if (valid_t >= 8)
        {
            uint base0 = (t_base + 0) * k_cols;
            uint base1 = (t_base + 1) * k_cols;
            uint base2 = (t_base + 2) * k_cols;
            uint base3 = (t_base + 3) * k_cols;
            uint base4 = (t_base + 4) * k_cols;
            uint base5 = (t_base + 5) * k_cols;
            uint base6 = (t_base + 6) * k_cols;
            uint base7 = (t_base + 7) * k_cols;

            for (uint b = 0; b < nb; b++)
            {
                uint blk_addr = row_byte_offset + b * 212;
                float d = f16tof32(W.Load(blk_addr + 208) & 0xFFFF);
                uint x_blk = b * 256;

                uint sc0 = W.Load(blk_addr + 192);
                uint sc1 = W.Load(blk_addr + 196);
                uint sc2 = W.Load(blk_addr + 200);
                uint sc3 = W.Load(blk_addr + 204);

                [unroll]
                for (uint step = 0; step < 2; step++)
                {
                    uint n = step * 128;
                    uint ql_addr = blk_addr + step * 64;
                    uint qh_addr = blk_addr + 128 + step * 32;

                    uint qh_val = (W.Load(qh_addr + lane_dword) >> lane_byte) & 0xFF;
                    uint ql_low = (W.Load(ql_addr + lane_dword) >> lane_byte) & 0xFF;
                    uint ql_high = (W.Load(ql_addr + 32 + lane_dword) >> lane_byte) & 0xFF;

                    int q1 = (int)((ql_low & 0x0F) | (((qh_val >> 0) & 3) << 4)) - 32;
                    int q2 = (int)((ql_high & 0x0F) | (((qh_val >> 2) & 3) << 4)) - 32;
                    int q3 = (int)((ql_low >> 4) | (((qh_val >> 4) & 3) << 4)) - 32;
                    int q4 = (int)((ql_high >> 4) | (((qh_val >> 6) & 3) << 4)) - 32;

                    uint2 sc_words = (step == 0) ? uint2(sc0, sc1) : uint2(sc2, sc3);
                    int s1 = decode_signed_byte_batch((sc_words.x >> (is_idx * 8)) & 0xFF);
                    int s2 = decode_signed_byte_batch((sc_words.x >> ((is_idx + 2) * 8)) & 0xFF);
                    int s3 = decode_signed_byte_batch((sc_words.y >> (is_idx * 8)) & 0xFF);
                    int s4 = decode_signed_byte_batch((sc_words.y >> ((is_idx + 2) * 8)) & 0xFF);

                    float4 w = d * float4(
                        (float)s1 * (float)q1,
                        (float)s2 * (float)q2,
                        (float)s3 * (float)q3,
                        (float)s4 * (float)q4);

                    uint k_col = x_blk + n + lane_id;

                    uint a0 = (base0 + k_col) * 4;
                    sum0.x += dot(w, float4(asfloat(x.Load(a0)), asfloat(x.Load(a0 + 128)), asfloat(x.Load(a0 + 256)), asfloat(x.Load(a0 + 384))));
                    uint a1 = (base1 + k_col) * 4;
                    sum0.y += dot(w, float4(asfloat(x.Load(a1)), asfloat(x.Load(a1 + 128)), asfloat(x.Load(a1 + 256)), asfloat(x.Load(a1 + 384))));
                    uint a2 = (base2 + k_col) * 4;
                    sum0.z += dot(w, float4(asfloat(x.Load(a2)), asfloat(x.Load(a2 + 128)), asfloat(x.Load(a2 + 256)), asfloat(x.Load(a2 + 384))));
                    uint a3 = (base3 + k_col) * 4;
                    sum0.w += dot(w, float4(asfloat(x.Load(a3)), asfloat(x.Load(a3 + 128)), asfloat(x.Load(a3 + 256)), asfloat(x.Load(a3 + 384))));
                    uint a4 = (base4 + k_col) * 4;
                    sum1.x += dot(w, float4(asfloat(x.Load(a4)), asfloat(x.Load(a4 + 128)), asfloat(x.Load(a4 + 256)), asfloat(x.Load(a4 + 384))));
                    uint a5 = (base5 + k_col) * 4;
                    sum1.y += dot(w, float4(asfloat(x.Load(a5)), asfloat(x.Load(a5 + 128)), asfloat(x.Load(a5 + 256)), asfloat(x.Load(a5 + 384))));
                    uint a6 = (base6 + k_col) * 4;
                    sum1.z += dot(w, float4(asfloat(x.Load(a6)), asfloat(x.Load(a6 + 128)), asfloat(x.Load(a6 + 256)), asfloat(x.Load(a6 + 384))));
                    uint a7 = (base7 + k_col) * 4;
                    sum1.w += dot(w, float4(asfloat(x.Load(a7)), asfloat(x.Load(a7 + 128)), asfloat(x.Load(a7 + 256)), asfloat(x.Load(a7 + 384))));
                }
            }
        }
        else
        {
            for (uint b = 0; b < nb; b++)
            {
                uint blk_addr = row_byte_offset + b * 212;
                float d = f16tof32(W.Load(blk_addr + 208) & 0xFFFF);
                uint x_blk = b * 256;

                uint sc0 = W.Load(blk_addr + 192);
                uint sc1 = W.Load(blk_addr + 196);
                uint sc2 = W.Load(blk_addr + 200);
                uint sc3 = W.Load(blk_addr + 204);

                [unroll]
                for (uint step = 0; step < 2; step++)
                {
                    uint n = step * 128;
                    uint ql_addr = blk_addr + step * 64;
                    uint qh_addr = blk_addr + 128 + step * 32;

                    uint qh_val = (W.Load(qh_addr + lane_dword) >> lane_byte) & 0xFF;
                    uint ql_low = (W.Load(ql_addr + lane_dword) >> lane_byte) & 0xFF;
                    uint ql_high = (W.Load(ql_addr + 32 + lane_dword) >> lane_byte) & 0xFF;

                    int q1 = (int)((ql_low & 0x0F) | (((qh_val >> 0) & 3) << 4)) - 32;
                    int q2 = (int)((ql_high & 0x0F) | (((qh_val >> 2) & 3) << 4)) - 32;
                    int q3 = (int)((ql_low >> 4) | (((qh_val >> 4) & 3) << 4)) - 32;
                    int q4 = (int)((ql_high >> 4) | (((qh_val >> 6) & 3) << 4)) - 32;

                    uint2 sc_words = (step == 0) ? uint2(sc0, sc1) : uint2(sc2, sc3);
                    int s1 = decode_signed_byte_batch((sc_words.x >> (is_idx * 8)) & 0xFF);
                    int s2 = decode_signed_byte_batch((sc_words.x >> ((is_idx + 2) * 8)) & 0xFF);
                    int s3 = decode_signed_byte_batch((sc_words.y >> (is_idx * 8)) & 0xFF);
                    int s4 = decode_signed_byte_batch((sc_words.y >> ((is_idx + 2) * 8)) & 0xFF);

                    float4 w = d * float4(
                        (float)s1 * (float)q1,
                        (float)s2 * (float)q2,
                        (float)s3 * (float)q3,
                        (float)s4 * (float)q4);

                    uint k_col = x_blk + n + lane_id;

                    [unroll]
                    for (uint tt = 0; tt < 4; tt++)
                    {
                        if (tt < valid_t)
                        {
                            uint x_addr = ((t_base + tt) * k_cols + k_col) * 4;
                            float4 xv = float4(
                                asfloat(x.Load(x_addr + 0)),
                                asfloat(x.Load(x_addr + 128)),
                                asfloat(x.Load(x_addr + 256)),
                                asfloat(x.Load(x_addr + 384)));
                            sum0[tt] += dot(w, xv);
                        }
                        if ((tt + 4) < valid_t)
                        {
                            uint x_addr = ((t_base + tt + 4) * k_cols + k_col) * 4;
                            float4 xv = float4(
                                asfloat(x.Load(x_addr + 0)),
                                asfloat(x.Load(x_addr + 128)),
                                asfloat(x.Load(x_addr + 256)),
                                asfloat(x.Load(x_addr + 384)));
                            sum1[tt] += dot(w, xv);
                        }
                    }
                }
            }
        }
    }

    s_mem0[s_idx] = sum0;
    s_mem1[s_idx] = sum1;
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 16) { s_mem0[s_idx] += s_mem0[s_idx + 16]; s_mem1[s_idx] += s_mem1[s_idx + 16]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 8)  { s_mem0[s_idx] += s_mem0[s_idx + 8];  s_mem1[s_idx] += s_mem1[s_idx + 8]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 4)  { s_mem0[s_idx] += s_mem0[s_idx + 4];  s_mem1[s_idx] += s_mem1[s_idx + 4]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 2)  { s_mem0[s_idx] += s_mem0[s_idx + 2];  s_mem1[s_idx] += s_mem1[s_idx + 2]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 1)  { s_mem0[s_idx] += s_mem0[s_idx + 1];  s_mem1[s_idx] += s_mem1[s_idx + 1]; }
    GroupMemoryBarrierWithGroupSync();

    if (lane_id < valid_t && warp_id < m_rows)
    {
        uint base_s = row_in_grp * 32;
        float final_sum = (lane_id < 4) ? s_mem0[base_s][lane_id] : s_mem1[base_s][lane_id - 4];
        if (has_bias != 0) final_sum += asfloat(bias.Load(warp_id * 4));
        uint out_byte_addr = ((t_base + lane_id) * m_rows + warp_id) * 4;

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
ByteAddressBuffer bias : register(t1);
RWByteAddressBuffer x : register(u2);
RWByteAddressBuffer residual : register(u0);
RWByteAddressBuffer y : register(u1);

groupshared float4 s_mem0[128];
groupshared float4 s_mem1[128];

[numthreads(32, 4, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint row_in_grp = gtid.y;
    uint warp_id = gid.x * 4 + row_in_grp;
    uint lane_id = gtid.x;
    uint s_idx = row_in_grp * 32 + lane_id;
    uint t_base = gid.y * 8;
    uint valid_t = (t_base < batch_size) ? min(8u, batch_size - t_base) : 0u;

    float4 sum0 = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 sum1 = float4(0.0f, 0.0f, 0.0f, 0.0f);

    if (warp_id < m_rows && valid_t > 0)
    {
        uint nb = k_cols / 32;
        uint row_byte_offset = warp_id * nb * 36;
        uint q_word_offset = (lane_id / 4) * 4;
        uint q_shift = (lane_id % 4) * 8;

        for (uint b = 0; b < nb; b++)
        {
            uint blk_addr = row_byte_offset + b * 36;
            float d = f16tof32(W.Load(blk_addr) & 0xFFFF);
            uint q_word = W.Load(blk_addr + 4 + q_word_offset);
            int q = (int)(q_word << (24 - q_shift)) >> 24;
            float w = d * (float)q;
            uint k_col = b * 32 + lane_id;

            [unroll]
            for (uint tt = 0; tt < 4; tt++)
            {
                if (tt < valid_t)
                    sum0[tt] += w * asfloat(x.Load(((t_base + tt) * k_cols + k_col) * 4));
                if ((tt + 4) < valid_t)
                    sum1[tt] += w * asfloat(x.Load(((t_base + tt + 4) * k_cols + k_col) * 4));
            }
        }
    }

    s_mem0[s_idx] = sum0;
    s_mem1[s_idx] = sum1;
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 16) { s_mem0[s_idx] += s_mem0[s_idx + 16]; s_mem1[s_idx] += s_mem1[s_idx + 16]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 8)  { s_mem0[s_idx] += s_mem0[s_idx + 8];  s_mem1[s_idx] += s_mem1[s_idx + 8]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 4)  { s_mem0[s_idx] += s_mem0[s_idx + 4];  s_mem1[s_idx] += s_mem1[s_idx + 4]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 2)  { s_mem0[s_idx] += s_mem0[s_idx + 2];  s_mem1[s_idx] += s_mem1[s_idx + 2]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 1)  { s_mem0[s_idx] += s_mem0[s_idx + 1];  s_mem1[s_idx] += s_mem1[s_idx + 1]; }
    GroupMemoryBarrierWithGroupSync();

    if (lane_id < valid_t && warp_id < m_rows)
    {
        uint base_s = row_in_grp * 32;
        float final_sum = (lane_id < 4) ? s_mem0[base_s][lane_id] : s_mem1[base_s][lane_id - 4];
        if (has_bias != 0) final_sum += asfloat(bias.Load(warp_id * 4));
        uint out_byte_addr = ((t_base + lane_id) * m_rows + warp_id) * 4;

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

ByteAddressBuffer W : register(t0);
ByteAddressBuffer bias : register(t1);
RWByteAddressBuffer x : register(u2);
RWByteAddressBuffer residual : register(u0);
RWByteAddressBuffer y : register(u1);

groupshared float4 s_mem0[128];
groupshared float4 s_mem1[128];

[numthreads(32, 4, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint row_in_grp = gtid.y;
    uint warp_id = gid.x * 4 + row_in_grp;
    uint lane_id = gtid.x;
    uint s_idx = row_in_grp * 32 + lane_id;
    uint t_base = gid.y * 8;
    uint valid_t = (t_base < batch_size) ? min(8u, batch_size - t_base) : 0u;

    float4 sum0 = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 sum1 = float4(0.0f, 0.0f, 0.0f, 0.0f);

    if (warp_id < m_rows && valid_t > 0)
    {
        uint row_offset = warp_id * k_cols;

        for (uint i = lane_id; i < k_cols; i += 32)
        {
            float w = asfloat(W.Load((row_offset + i) * 4));
            [unroll]
            for (uint tt = 0; tt < 4; tt++)
            {
                if (tt < valid_t)
                    sum0[tt] += w * asfloat(x.Load(((t_base + tt) * k_cols + i) * 4));
                if ((tt + 4) < valid_t)
                    sum1[tt] += w * asfloat(x.Load(((t_base + tt + 4) * k_cols + i) * 4));
            }
        }
    }

    s_mem0[s_idx] = sum0;
    s_mem1[s_idx] = sum1;
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 16) { s_mem0[s_idx] += s_mem0[s_idx + 16]; s_mem1[s_idx] += s_mem1[s_idx + 16]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 8)  { s_mem0[s_idx] += s_mem0[s_idx + 8];  s_mem1[s_idx] += s_mem1[s_idx + 8]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 4)  { s_mem0[s_idx] += s_mem0[s_idx + 4];  s_mem1[s_idx] += s_mem1[s_idx + 4]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 2)  { s_mem0[s_idx] += s_mem0[s_idx + 2];  s_mem1[s_idx] += s_mem1[s_idx + 2]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 1)  { s_mem0[s_idx] += s_mem0[s_idx + 1];  s_mem1[s_idx] += s_mem1[s_idx + 1]; }
    GroupMemoryBarrierWithGroupSync();

    if (lane_id < valid_t && warp_id < m_rows)
    {
        uint base_s = row_in_grp * 32;
        float final_sum = (lane_id < 4) ? s_mem0[base_s][lane_id] : s_mem1[base_s][lane_id - 4];
        if (has_bias != 0) final_sum += asfloat(bias.Load(warp_id * 4));
        uint out_byte_addr = ((t_base + lane_id) * m_rows + warp_id) * 4;

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
";

    public const string GemmQ3KBatch = @"
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

groupshared float4 s_mem0[128];
groupshared float4 s_mem1[128];

int get_scale_q3_batch(uint idx, uint sc0, uint sc1, uint sc2, uint sc3)
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
    uint t_base = gid.y * 8;
    uint valid_t = (t_base < batch_size) ? min(8u, batch_size - t_base) : 0u;

    float4 sum0 = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 sum1 = float4(0.0f, 0.0f, 0.0f, 0.0f);

    if (warp_id < m_rows && valid_t > 0)
    {
        uint nb = k_cols / 256;
        uint row_byte_offset = warp_id * nb * 112;
        uint lane_dword = lane_id & ~3u;
        uint lane_shift = (lane_id & 3u) * 8;

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
            uint hm_byte = (W.Load(blk_addr + lane_dword) >> lane_shift) & 0xFF;
            uint x_blk = b * 256;

            [unroll]
            for (uint step = 0; step < 2; step++)
            {
                uint n = step * 128;
                uint q_base = blk_addr + 32 + step * 32;
                uint q_byte = (W.Load(q_base + lane_dword) >> lane_shift) & 0xFF;

                float4 w;
                [unroll]
                for (uint j = 0; j < 4; j++)
                {
                    uint m = 1u << (step * 4 + j);
                    uint is_idx = step * 8 + j * 2 + ((lane_id < 16) ? 0u : 1u);
                    float dl = d_all * (float)get_scale_q3_batch(is_idx, sc0, sc1, sc2, sc3);
                    int q = (int)((q_byte >> (j * 2)) & 3) - (((hm_byte & m) != 0) ? 0 : 4);
                    w[j] = dl * (float)q;
                }

                uint k_col = x_blk + n + lane_id;

                [unroll]
                for (uint tt = 0; tt < 4; tt++)
                {
                    if (tt < valid_t)
                    {
                        uint x_addr = ((t_base + tt) * k_cols + k_col) * 4;
                        float4 xv = float4(
                            asfloat(x.Load(x_addr + 0)),
                            asfloat(x.Load(x_addr + 128)),
                            asfloat(x.Load(x_addr + 256)),
                            asfloat(x.Load(x_addr + 384)));
                        sum0[tt] += dot(w, xv);
                    }
                    if ((tt + 4) < valid_t)
                    {
                        uint x_addr = ((t_base + tt + 4) * k_cols + k_col) * 4;
                        float4 xv = float4(
                            asfloat(x.Load(x_addr + 0)),
                            asfloat(x.Load(x_addr + 128)),
                            asfloat(x.Load(x_addr + 256)),
                            asfloat(x.Load(x_addr + 384)));
                        sum1[tt] += dot(w, xv);
                    }
                }
            }
        }
    }

    s_mem0[s_idx] = sum0;
    s_mem1[s_idx] = sum1;
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 16) { s_mem0[s_idx] += s_mem0[s_idx + 16]; s_mem1[s_idx] += s_mem1[s_idx + 16]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 8)  { s_mem0[s_idx] += s_mem0[s_idx + 8];  s_mem1[s_idx] += s_mem1[s_idx + 8]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 4)  { s_mem0[s_idx] += s_mem0[s_idx + 4];  s_mem1[s_idx] += s_mem1[s_idx + 4]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 2)  { s_mem0[s_idx] += s_mem0[s_idx + 2];  s_mem1[s_idx] += s_mem1[s_idx + 2]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 1)  { s_mem0[s_idx] += s_mem0[s_idx + 1];  s_mem1[s_idx] += s_mem1[s_idx + 1]; }
    GroupMemoryBarrierWithGroupSync();

    if (lane_id < valid_t && warp_id < m_rows)
    {
        uint base_s = row_in_grp * 32;
        float final_sum = (lane_id < 4) ? s_mem0[base_s][lane_id] : s_mem1[base_s][lane_id - 4];
        if (has_bias != 0) final_sum += asfloat(bias.Load(warp_id * 4));
        uint out_byte_addr = ((t_base + lane_id) * m_rows + warp_id) * 4;

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
";

    public const string GemmQ5KBatch = @"
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

groupshared float4 s_mem0[128];
groupshared float4 s_mem1[128];

uint get_scale_byte_q5_batch(uint idx, uint s0, uint s1, uint s2)
{
    if (idx < 4) return (s0 >> (idx * 8)) & 0xFF;
    else if (idx < 8) return (s1 >> ((idx - 4) * 8)) & 0xFF;
    else return (s2 >> ((idx - 8) * 8)) & 0xFF;
}

void get_scale_min_q5_batch(uint j, uint s0, uint s1, uint s2, out float d_out, out float min_out, float d, float min_val)
{
    uint sc, m;
    if (j < 4) {
        sc = get_scale_byte_q5_batch(j, s0, s1, s2) & 63;
        m  = get_scale_byte_q5_batch(j + 4, s0, s1, s2) & 63;
    } else {
        sc = (get_scale_byte_q5_batch(j + 4, s0, s1, s2) & 0x0F) | ((get_scale_byte_q5_batch(j - 4, s0, s1, s2) >> 6) << 4);
        m  = (get_scale_byte_q5_batch(j + 4, s0, s1, s2) >> 4)   | ((get_scale_byte_q5_batch(j, s0, s1, s2) >> 6) << 4);
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
    uint t_base = gid.y * 8;
    uint valid_t = (t_base < batch_size) ? min(8u, batch_size - t_base) : 0u;

    float4 sum0 = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 sum1 = float4(0.0f, 0.0f, 0.0f, 0.0f);

    if (warp_id < m_rows && valid_t > 0)
    {
        uint nb = k_cols / 256;
        uint row_byte_offset = warp_id * nb * 176;
        uint lane_dword = lane_id & ~3u;
        uint lane_shift = (lane_id & 3u) * 8;

        for (uint b = 0; b < nb; b++)
        {
            uint blk_addr = row_byte_offset + b * 176;
            uint2 d_hdr = W.Load2(blk_addr);
            float d = f16tof32(d_hdr.x & 0xFFFF);
            float min_val = f16tof32(d_hdr.x >> 16);
            uint s0 = d_hdr.y;
            uint s1 = W.Load(blk_addr + 8);
            uint s2 = W.Load(blk_addr + 12);

            uint qh_val = (W.Load(blk_addr + 16 + lane_dword) >> lane_shift) & 0xFF;
            uint x_blk = b * 256;

            [unroll]
            for (uint c = 0; c < 4; c++)
            {
                uint is_idx = c * 2;
                float d1, min1, d2, min2;
                get_scale_min_q5_batch(is_idx + 0, s0, s1, s2, d1, min1, d, min_val);
                get_scale_min_q5_batch(is_idx + 1, s0, s1, s2, d2, min2, d, min_val);

                uint ql_val = (W.Load(blk_addr + 48 + c * 32 + lane_dword) >> lane_shift) & 0xFF;
                uint bit0 = (qh_val >> (c * 2 + 0)) & 1;
                uint bit1 = (qh_val >> (c * 2 + 1)) & 1;

                float w0 = d1 * (float)((ql_val & 0x0F) | (bit0 << 4)) - min1;
                float w1 = d2 * (float)((ql_val >> 4)   | (bit1 << 4)) - min2;

                uint k0 = x_blk + c * 64 + lane_id;
                uint k1 = k0 + 32;

                [unroll]
                for (uint tt = 0; tt < 4; tt++)
                {
                    if (tt < valid_t)
                    {
                        uint base_t = (t_base + tt) * k_cols;
                        sum0[tt] += w0 * asfloat(x.Load((base_t + k0) * 4)) + w1 * asfloat(x.Load((base_t + k1) * 4));
                    }
                    if ((tt + 4) < valid_t)
                    {
                        uint base_t = (t_base + tt + 4) * k_cols;
                        sum1[tt] += w0 * asfloat(x.Load((base_t + k0) * 4)) + w1 * asfloat(x.Load((base_t + k1) * 4));
                    }
                }
            }
        }
    }

    s_mem0[s_idx] = sum0;
    s_mem1[s_idx] = sum1;
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 16) { s_mem0[s_idx] += s_mem0[s_idx + 16]; s_mem1[s_idx] += s_mem1[s_idx + 16]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 8)  { s_mem0[s_idx] += s_mem0[s_idx + 8];  s_mem1[s_idx] += s_mem1[s_idx + 8]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 4)  { s_mem0[s_idx] += s_mem0[s_idx + 4];  s_mem1[s_idx] += s_mem1[s_idx + 4]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 2)  { s_mem0[s_idx] += s_mem0[s_idx + 2];  s_mem1[s_idx] += s_mem1[s_idx + 2]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 1)  { s_mem0[s_idx] += s_mem0[s_idx + 1];  s_mem1[s_idx] += s_mem1[s_idx + 1]; }
    GroupMemoryBarrierWithGroupSync();

    if (lane_id < valid_t && warp_id < m_rows)
    {
        uint base_s = row_in_grp * 32;
        float final_sum = (lane_id < 4) ? s_mem0[base_s][lane_id] : s_mem1[base_s][lane_id - 4];
        if (has_bias != 0) final_sum += asfloat(bias.Load(warp_id * 4));
        uint out_byte_addr = ((t_base + lane_id) * m_rows + warp_id) * 4;

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
";
}


