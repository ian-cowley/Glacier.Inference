namespace Glacier.Inference.Gpu.D3D12;

/// <summary>
/// High-performance HLSL compute shaders for transformer LLM inference on Direct3D 12.
/// Optimized for AMD Radeon Wave32 architecture and DirectX 12 hardware.
/// </summary>
public static class D3D12Shaders
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

    public const string RmsNorm = @"
cbuffer Params : register(b0)
{
    uint size;
    float eps;
};

StructuredBuffer<float> weight : register(t0);
RWStructuredBuffer<float> x : register(u0);
RWStructuredBuffer<float> dst : register(u1);

groupshared float s_sum[256];

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_GroupThreadID)
{
    float local_sum = 0.0f;
    for (uint i = tid.x; i < size; i += 256)
    {
        float val = x[i];
        local_sum += val * val;
    }
    s_sum[tid.x] = local_sum;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint s = 128; s > 0; s >>= 1)
    {
        if (tid.x < s)
        {
            s_sum[tid.x] += s_sum[tid.x + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    float rms = rsqrt((s_sum[0] / (float)size) + eps);

    for (uint j = tid.x; j < size; j += 256)
    {
        dst[j] = x[j] * rms * weight[j];
    }
}
";

    public const string SwiGLU = @"
cbuffer Params : register(b0)
{
    uint size;
};

RWStructuredBuffer<float> gate : register(u0);
RWStructuredBuffer<float> up : register(u1);
RWStructuredBuffer<float> dst : register(u2);

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x < size)
    {
        float g = gate[id.x];
        float silu = g / (1.0f + exp(-g));
        dst[id.x] = silu * up[id.x];
    }
}
";

    public const string VecAdd = @"
cbuffer Params : register(b0)
{
    uint size;
};

RWStructuredBuffer<float> b : register(u0);
RWStructuredBuffer<float> a : register(u1);

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x < size)
    {
        a[id.x] += b[id.x];
    }
}
";

    public const string RoPE = @"
cbuffer Params : register(b0)
{
    uint n_heads_q;
    uint n_heads_kv;
    uint head_dim;
    uint pos;
    float freq_base;
    float freq_scale;
};

RWStructuredBuffer<float> q : register(u0);
RWStructuredBuffer<float> k : register(u1);

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint half_dim = head_dim / 2;
    uint q_half = n_heads_q * half_dim;
    uint total_half = (n_heads_q + n_heads_kv) * half_dim;

    if (id.x >= total_half) return;

    bool is_k = (id.x >= q_half);
    uint head_idx = is_k ? (id.x - q_half) / half_dim : id.x / half_dim;
    uint i = is_k ? (id.x - q_half) % half_dim : id.x % half_dim;

    float freq = 1.0f / pow(freq_base, (float)(2 * i) / (float)head_dim);
    float theta = (float)pos * freq * freq_scale;
    float cos_theta = cos(theta);
    float sin_theta = sin(theta);

    uint offset = head_idx * head_dim;
    if (is_k)
    {
        float v0 = k[offset + i];
        float v1 = k[offset + i + half_dim];
        k[offset + i] = v0 * cos_theta - v1 * sin_theta;
        k[offset + i + half_dim] = v0 * sin_theta + v1 * cos_theta;
    }
    else
    {
        float v0 = q[offset + i];
        float v1 = q[offset + i + half_dim];
        q[offset + i] = v0 * cos_theta - v1 * sin_theta;
        q[offset + i + half_dim] = v0 * sin_theta + v1 * cos_theta;
    }
}
";

    public const string KvCacheStore = @"
cbuffer Params : register(b0)
{
    uint n_heads_kv;
    uint head_dim;
    uint max_seq_len;
    uint pos;
};

RWStructuredBuffer<float> k : register(u0);
RWStructuredBuffer<float> v : register(u1);
RWStructuredBuffer<float> k_cache : register(u2);
RWStructuredBuffer<float> v_cache : register(u3);

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint total = n_heads_kv * head_dim;
    if (id.x >= total) return;

    uint h = id.x / head_dim;
    uint d = id.x % head_dim;
    uint cache_idx = (h * max_seq_len + pos) * head_dim + d;

    k_cache[cache_idx] = k[id.x];
    v_cache[cache_idx] = v[id.x];
}
";

    public const string AttentionGqa = @"
cbuffer Params : register(b0)
{
    uint n_heads_q;
    uint n_heads_kv;
    uint head_dim;
    uint max_seq_len;
    uint pos;
    float attn_scale;
};

RWStructuredBuffer<float> q : register(u0);
RWStructuredBuffer<float> k_cache : register(u1);
RWStructuredBuffer<float> v_cache : register(u2);
RWStructuredBuffer<float> attn_out : register(u3);

groupshared float s_red[128];

[numthreads(128, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint h = gid.x;
    if (h >= n_heads_q) return;

    uint tid = gtid.x;
    uint group_size = n_heads_q / n_heads_kv;
    uint h_kv = h / group_size;

    float q_d = q[h * head_dim + tid];

    float m = -1e30f;
    float l = 0.0f;
    float acc = 0.0f;

    for (uint t = 0; t <= pos; t++)
    {
        uint kv_offset = (h_kv * max_seq_len + t) * head_dim + tid;
        float k_d = k_cache[kv_offset];
        float v_d = v_cache[kv_offset];

        s_red[tid] = q_d * k_d;
        GroupMemoryBarrierWithGroupSync();

        if (tid < 64) s_red[tid] += s_red[tid + 64];
        GroupMemoryBarrierWithGroupSync();
        if (tid < 32) s_red[tid] += s_red[tid + 32];
        GroupMemoryBarrierWithGroupSync();
        if (tid < 16) s_red[tid] += s_red[tid + 16];
        GroupMemoryBarrierWithGroupSync();
        if (tid < 8)  s_red[tid] += s_red[tid + 8];
        GroupMemoryBarrierWithGroupSync();
        if (tid < 4)  s_red[tid] += s_red[tid + 4];
        GroupMemoryBarrierWithGroupSync();
        if (tid < 2)  s_red[tid] += s_red[tid + 2];
        GroupMemoryBarrierWithGroupSync();
        if (tid < 1)  s_red[tid] += s_red[tid + 1];
        GroupMemoryBarrierWithGroupSync();

        float total_dot = s_red[0];
        float s_t = total_dot * attn_scale;

        float m_new = max(m, s_t);
        float alpha = exp(m - m_new);
        float w_t = exp(s_t - m_new);

        acc = acc * alpha + w_t * v_d;
        l = l * alpha + w_t;
        m = m_new;

        GroupMemoryBarrierWithGroupSync();
    }

    attn_out[h * head_dim + tid] = (l > 0.0f) ? (acc / l) : 0.0f;
}
";

    public const string Argmax = @"
cbuffer Params : register(b0)
{
    uint size;
};

RWStructuredBuffer<float> logits : register(u0);
RWStructuredBuffer<int> best_token : register(u1);
RWStructuredBuffer<float> best_logit : register(u2);

groupshared float s_val[256];
groupshared int s_idx[256];

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_GroupThreadID)
{
    float max_val = -1e30f;
    int max_idx = 0;

    for (uint i = tid.x; i < size; i += 256)
    {
        float val = logits[i];
        if (val > max_val)
        {
            max_val = val;
            max_idx = (int)i;
        }
    }

    s_val[tid.x] = max_val;
    s_idx[tid.x] = max_idx;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint s = 128; s > 0; s >>= 1)
    {
        if (tid.x < s)
        {
            if (s_val[tid.x + s] > s_val[tid.x])
            {
                s_val[tid.x] = s_val[tid.x + s];
                s_idx[tid.x] = s_idx[tid.x + s];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid.x == 0)
    {
        best_token[0] = s_idx[0];
        best_logit[0] = s_val[0];
    }
}
";

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

    public const string RmsNormBatch = @"
cbuffer Params : register(b0)
{
    uint size;
    float eps;
    uint batch_size;
};

StructuredBuffer<float> weight : register(t0);
RWStructuredBuffer<float> x : register(u0);
RWStructuredBuffer<float> dst : register(u1);

groupshared float s_sum[256];

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint t = gid.x;
    if (t >= batch_size) return;
    uint base = t * size;

    float local_sum = 0.0f;
    for (uint i = gtid.x; i < size; i += 256)
    {
        float val = x[base + i];
        local_sum += val * val;
    }
    s_sum[gtid.x] = local_sum;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint s = 128; s > 0; s >>= 1)
    {
        if (gtid.x < s)
        {
            s_sum[gtid.x] += s_sum[gtid.x + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    float rms = rsqrt((s_sum[0] / (float)size) + eps);

    for (uint j = gtid.x; j < size; j += 256)
    {
        dst[base + j] = x[base + j] * rms * weight[j];
    }
}
";

    public const string RoPEBatch = @"
cbuffer Params : register(b0)
{
    uint n_heads_q;
    uint n_heads_kv;
    uint head_dim;
    uint start_pos;
    float freq_base;
    float freq_scale;
    uint batch_size;
};

RWStructuredBuffer<float> q : register(u0);
RWStructuredBuffer<float> k : register(u1);

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint half_dim = head_dim / 2;
    uint q_half = n_heads_q * half_dim;
    uint total_half_per_token = (n_heads_q + n_heads_kv) * half_dim;
    uint total_half = batch_size * total_half_per_token;

    if (id.x >= total_half) return;

    uint t = id.x / total_half_per_token;
    uint local_id = id.x % total_half_per_token;
    uint pos = start_pos + t;

    bool is_k = (local_id >= q_half);
    uint head_idx = is_k ? (local_id - q_half) / half_dim : local_id / half_dim;
    uint i = is_k ? (local_id - q_half) % half_dim : local_id % half_dim;

    float freq = 1.0f / pow(freq_base, (float)(2 * i) / (float)head_dim);
    float theta = (float)pos * freq * freq_scale;
    float cos_theta = cos(theta);
    float sin_theta = sin(theta);

    if (is_k)
    {
        uint offset = t * (n_heads_kv * head_dim) + head_idx * head_dim;
        float v0 = k[offset + i];
        float v1 = k[offset + i + half_dim];
        k[offset + i] = v0 * cos_theta - v1 * sin_theta;
        k[offset + i + half_dim] = v0 * sin_theta + v1 * cos_theta;
    }
    else
    {
        uint offset = t * (n_heads_q * head_dim) + head_idx * head_dim;
        float v0 = q[offset + i];
        float v1 = q[offset + i + half_dim];
        q[offset + i] = v0 * cos_theta - v1 * sin_theta;
        q[offset + i + half_dim] = v0 * sin_theta + v1 * cos_theta;
    }
}
";

    public const string KvCacheStoreBatch = @"
cbuffer Params : register(b0)
{
    uint n_heads_kv;
    uint head_dim;
    uint max_seq_len;
    uint start_pos;
    uint batch_size;
};

RWStructuredBuffer<float> k : register(u0);
RWStructuredBuffer<float> v : register(u1);
RWStructuredBuffer<float> k_cache : register(u2);
RWStructuredBuffer<float> v_cache : register(u3);

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint total_per_token = n_heads_kv * head_dim;
    uint total = batch_size * total_per_token;
    if (id.x >= total) return;

    uint t = id.x / total_per_token;
    uint local_id = id.x % total_per_token;
    uint pos = start_pos + t;

    uint h = local_id / head_dim;
    uint d = local_id % head_dim;
    uint cache_idx = (h * max_seq_len + pos) * head_dim + d;

    k_cache[cache_idx] = k[id.x];
    v_cache[cache_idx] = v[id.x];
}
";

    public const string AttentionBatch = @"
cbuffer Params : register(b0)
{
    uint n_heads_q;
    uint n_heads_kv;
    uint head_dim;
    uint max_seq_len;
    uint start_pos;
    float attn_scale;
    uint batch_size;
};

RWStructuredBuffer<float> q : register(u0);
RWStructuredBuffer<float> k_cache : register(u1);
RWStructuredBuffer<float> v_cache : register(u2);
RWStructuredBuffer<float> attn_out : register(u3);

groupshared float s_red[128];

[numthreads(128, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint h = gid.x;
    uint t = gid.y;
    if (h >= n_heads_q || t >= batch_size) return;

    uint tid = gtid.x;
    uint group_size = n_heads_q / n_heads_kv;
    uint h_kv = h / group_size;
    uint pos_t = start_pos + t;

    uint q_offset = (t * n_heads_q + h) * head_dim + tid;
    float q_d = q[q_offset];

    float m = -1e30f;
    float l = 0.0f;
    float acc = 0.0f;

    for (uint p = 0; p <= pos_t; p++)
    {
        uint kv_offset = (h_kv * max_seq_len + p) * head_dim + tid;
        float k_d = k_cache[kv_offset];
        float v_d = v_cache[kv_offset];

        s_red[tid] = q_d * k_d;
        GroupMemoryBarrierWithGroupSync();

        if (tid < 64) s_red[tid] += s_red[tid + 64];
        GroupMemoryBarrierWithGroupSync();
        if (tid < 32) s_red[tid] += s_red[tid + 32];
        GroupMemoryBarrierWithGroupSync();
        if (tid < 16) s_red[tid] += s_red[tid + 16];
        GroupMemoryBarrierWithGroupSync();
        if (tid < 8)  s_red[tid] += s_red[tid + 8];
        GroupMemoryBarrierWithGroupSync();
        if (tid < 4)  s_red[tid] += s_red[tid + 4];
        GroupMemoryBarrierWithGroupSync();
        if (tid < 2)  s_red[tid] += s_red[tid + 2];
        GroupMemoryBarrierWithGroupSync();
        if (tid < 1)  s_red[tid] += s_red[tid + 1];
        GroupMemoryBarrierWithGroupSync();

        float total_dot = s_red[0];
        float s_p = total_dot * attn_scale;

        float m_new = max(m, s_p);
        float alpha = exp(m - m_new);
        float w_p = exp(s_p - m_new);

        acc = acc * alpha + w_p * v_d;
        l = l * alpha + w_p;
        m = m_new;

        GroupMemoryBarrierWithGroupSync();
    }

    uint out_offset = (t * n_heads_q + h) * head_dim + tid;
    attn_out[out_offset] = (l > 0.0f) ? (acc / l) : 0.0f;
}
";
}

