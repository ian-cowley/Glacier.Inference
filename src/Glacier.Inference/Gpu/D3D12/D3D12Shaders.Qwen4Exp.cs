namespace Glacier.Inference.Gpu.D3D12;

public static partial class D3D12Shaders
{
    public const string Qwen4HcBroadcastInit = @"
cbuffer Params : register(b0)
{
    uint batch_size;
    uint hc_count;
    uint dim;
};

RWStructuredBuffer<float> x : register(u0);
RWStructuredBuffer<float> r : register(u1);

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint hc_dim = hc_count * dim;
    uint total = batch_size * hc_dim;
    if (id.x >= total) return;

    uint t = id.x / hc_dim;
    uint rem = id.x % hc_dim;
    uint d = rem % dim;
    r[id.x] = x[t * dim + d];
}
";

    public const string Qwen4HcStreamRmsNorm = @"
cbuffer Params : register(b0)
{
    uint dim;
    uint hc_count;
    float eps;
    uint has_weight;
};

StructuredBuffer<float> weight : register(t0);
RWStructuredBuffer<float> r : register(u0);
RWStructuredBuffer<float> xn : register(u1);

groupshared float s_sum[256];

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint stream_idx = gid.x;
    uint c = stream_idx % hc_count;
    uint tid = gtid.x;
    uint base_off = stream_idx * dim;

    float local_sum = 0.0f;
    for (uint i = tid; i < dim; i += 256u)
    {
        float v = r[base_off + i];
        local_sum += v * v;
    }
    s_sum[tid] = local_sum;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint s = 128u; s > 0u; s >>= 1)
    {
        if (tid < s)
        {
            s_sum[tid] += s_sum[tid + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    float rrms = rsqrt(s_sum[0] / (float)dim + eps);
    uint w_base = c * dim;
    for (uint j = tid; j < dim; j += 256u)
    {
        float w = has_weight != 0u ? weight[w_base + j] : 1.0f;
        xn[base_off + j] = r[base_off + j] * rrms * w;
    }
}
";

    public const string Qwen4HcScaleSilu = @"
cbuffer Params : register(b0)
{
    uint total_len;
    float inv_hc;
};

RWStructuredBuffer<float> lo : register(u0);

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= total_len) return;
    float z = lo[id.x] * inv_hc;
    lo[id.x] = z / (1.0f + exp(-z));
}
";

    public const string Qwen4HcGateMean = @"
cbuffer Params : register(b0)
{
    uint batch_size;
    uint hc_count;
    uint dim;
    float inv_hc;
};

RWStructuredBuffer<float> xn : register(u0);
RWStructuredBuffer<float> up : register(u1);
RWStructuredBuffer<float> mixed_out : register(u2);

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint total = batch_size * dim;
    if (id.x >= total) return;

    uint t = id.x / dim;
    uint d = id.x % dim;
    float acc = 0.0f;

    for (uint c = 0u; c < hc_count; c++)
    {
        uint off = (t * hc_count + c) * dim + d;
        float g = 1.0f / (1.0f + exp(-up[off]));
        acc += xn[off] * g;
    }

    mixed_out[id.x] = acc * inv_hc;
}
";

    public const string Qwen4HcWrite = @"
cbuffer Params : register(b0)
{
    uint batch_size;
    uint hc_count;
    uint dim;
    float inv_hc;
};

RWStructuredBuffer<float> r : register(u0);
RWStructuredBuffer<float> sub_out : register(u1);
RWStructuredBuffer<float> inject : register(u2);

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint total = batch_size * dim;
    if (id.x >= total) return;

    uint t = id.x / dim;
    uint d = id.x % dim;
    float sv = sub_out[id.x];

    for (uint c = 0u; c < hc_count; c++)
    {
        float inj = inject[t * hc_count + c];
        float w = 2.0f / (1.0f + exp(-(inj * inv_hc)));
        r[(t * hc_count + c) * dim + d] += w * sv;
    }
}
";

    public const string Qwen3SparseAttention = @"
cbuffer Params : register(b0)
{
    uint n_heads_q;
    uint n_heads_kv;
    uint head_dim;
    uint max_seq_len;
    uint selected_count;
    float attn_scale;
};

StructuredBuffer<uint> selected_ids : register(t0);
RWStructuredBuffer<float> q : register(u0);
RWStructuredBuffer<float> k_cache : register(u1);
RWStructuredBuffer<float> v_cache : register(u2);
RWStructuredBuffer<float> attn_out : register(u3);

groupshared float s_red[256];

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint h = gid.x;
    if (h >= n_heads_q) return;

    uint tid = gtid.x;
    uint group_size = n_heads_q / n_heads_kv;
    uint h_kv = h / group_size;

    float q_d = (tid < head_dim) ? q[h * head_dim + tid] : 0.0f;

    float m = -1e30f;
    float l = 0.0f;
    float acc = 0.0f;

    for (uint s_i = 0u; s_i < selected_count; s_i++)
    {
        uint t = selected_ids[s_i];
        uint kv_offset = (h_kv * max_seq_len + t) * head_dim + tid;
        float k_d = (tid < head_dim) ? k_cache[kv_offset] : 0.0f;
        float v_d = (tid < head_dim) ? v_cache[kv_offset] : 0.0f;

        s_red[tid] = q_d * k_d;
        GroupMemoryBarrierWithGroupSync();

        if (tid < 128u) s_red[tid] += s_red[tid + 128u];
        GroupMemoryBarrierWithGroupSync();
        if (tid < 64u) s_red[tid] += s_red[tid + 64u];
        GroupMemoryBarrierWithGroupSync();
        if (tid < 32u) s_red[tid] += s_red[tid + 32u];
        GroupMemoryBarrierWithGroupSync();
        if (tid < 16u) s_red[tid] += s_red[tid + 16u];
        GroupMemoryBarrierWithGroupSync();
        if (tid < 8u)  s_red[tid] += s_red[tid + 8u];
        GroupMemoryBarrierWithGroupSync();
        if (tid < 4u)  s_red[tid] += s_red[tid + 4u];
        GroupMemoryBarrierWithGroupSync();
        if (tid < 2u)  s_red[tid] += s_red[tid + 2u];
        GroupMemoryBarrierWithGroupSync();
        if (tid < 1u)  s_red[tid] += s_red[tid + 1u];
        GroupMemoryBarrierWithGroupSync();

        float s_t = s_red[0] * attn_scale;

        float m_new = max(m, s_t);
        float alpha = exp(m - m_new);
        float w_t = exp(s_t - m_new);

        acc = acc * alpha + w_t * v_d;
        l = l * alpha + w_t;
        m = m_new;

        GroupMemoryBarrierWithGroupSync();
    }

    if (tid < head_dim)
    {
        attn_out[h * head_dim + tid] = (l > 0.0f) ? (acc / l) : 0.0f;
    }
}
";

    public const string Qwen3SparseAttentionF16 = @"
cbuffer Params : register(b0)
{
    uint n_heads_q;
    uint n_heads_kv;
    uint head_dim;
    uint max_seq_len;
    uint selected_count;
    float attn_scale;
};

StructuredBuffer<uint> selected_ids : register(t0);
RWStructuredBuffer<float> q : register(u0);
RWStructuredBuffer<uint> k_cache : register(u1);
RWStructuredBuffer<uint> v_cache : register(u2);
RWStructuredBuffer<float> attn_out : register(u3);

groupshared float s_red[256];

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint h = gid.x;
    if (h >= n_heads_q) return;

    uint tid = gtid.x;
    uint group_size = n_heads_q / n_heads_kv;
    uint h_kv = h / group_size;
    uint half_dim = (head_dim + 1u) / 2u;

    float q_d = (tid < head_dim) ? q[h * head_dim + tid] : 0.0f;

    float m = -1e30f;
    float l = 0.0f;
    float acc = 0.0f;

    for (uint s_i = 0u; s_i < selected_count; s_i++)
    {
        uint t = selected_ids[s_i];
        uint kv_base = (h_kv * max_seq_len + t) * half_dim;
        float k_d = 0.0f;
        float v_d = 0.0f;
        if (tid < head_dim)
        {
            uint pair_idx = tid >> 1;
            uint kp = k_cache[kv_base + pair_idx];
            uint vp = v_cache[kv_base + pair_idx];
            k_d = ((tid & 1u) == 0u) ? f16tof32(kp & 0xFFFFu) : f16tof32(kp >> 16);
            v_d = ((tid & 1u) == 0u) ? f16tof32(vp & 0xFFFFu) : f16tof32(vp >> 16);
        }

        s_red[tid] = q_d * k_d;
        GroupMemoryBarrierWithGroupSync();

        if (tid < 128u) s_red[tid] += s_red[tid + 128u];
        GroupMemoryBarrierWithGroupSync();
        if (tid < 64u) s_red[tid] += s_red[tid + 64u];
        GroupMemoryBarrierWithGroupSync();
        if (tid < 32u) s_red[tid] += s_red[tid + 32u];
        GroupMemoryBarrierWithGroupSync();
        if (tid < 16u) s_red[tid] += s_red[tid + 16u];
        GroupMemoryBarrierWithGroupSync();
        if (tid < 8u)  s_red[tid] += s_red[tid + 8u];
        GroupMemoryBarrierWithGroupSync();
        if (tid < 4u)  s_red[tid] += s_red[tid + 4u];
        GroupMemoryBarrierWithGroupSync();
        if (tid < 2u)  s_red[tid] += s_red[tid + 2u];
        GroupMemoryBarrierWithGroupSync();
        if (tid < 1u)  s_red[tid] += s_red[tid + 1u];
        GroupMemoryBarrierWithGroupSync();

        float s_t = s_red[0] * attn_scale;

        float m_new = max(m, s_t);
        float alpha = exp(m - m_new);
        float w_t = exp(s_t - m_new);

        acc = acc * alpha + w_t * v_d;
        l = l * alpha + w_t;
        m = m_new;

        GroupMemoryBarrierWithGroupSync();
    }

    if (tid < head_dim)
    {
        attn_out[h * head_dim + tid] = (l > 0.0f) ? (acc / l) : 0.0f;
    }
}
";

    public const string Qwen3SparseAttentionQ4_0 = @"
cbuffer Params : register(b0)
{
    uint n_heads_q;
    uint n_heads_kv;
    uint head_dim;
    uint max_seq_len;
    uint selected_count;
    float attn_scale;
};

StructuredBuffer<uint> selected_ids : register(t0);
RWStructuredBuffer<float> q : register(u0);
RWStructuredBuffer<uint> k_cache : register(u1);
RWStructuredBuffer<uint> v_cache : register(u2);
RWStructuredBuffer<float> attn_out : register(u3);

groupshared float s_red[256];

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint h = gid.x;
    if (h >= n_heads_q) return;

    uint tid = gtid.x;
    uint group_size = n_heads_q / n_heads_kv;
    uint h_kv = h / group_size;
    uint nb = (head_dim + 31u) >> 5;
    uint row_uints = nb * 4u + ((nb + 1u) >> 1);

    float q_d = (tid < head_dim) ? q[h * head_dim + tid] : 0.0f;

    float m = -1e30f;
    float l = 0.0f;
    float acc = 0.0f;

    for (uint s_i = 0u; s_i < selected_count; s_i++)
    {
        uint t = selected_ids[s_i];
        uint row_base = (h_kv * max_seq_len + t) * row_uints;
        float k_d = 0.0f;
        float v_d = 0.0f;
        if (tid < head_dim)
        {
            uint blk = tid >> 5;
            uint elem = tid & 31u;
            uint l_idx = elem & 15u;
            uint shift = ((l_idx & 3u) * 8u) + ((elem >= 16u) ? 4u : 0u);

            uint kw = k_cache[row_base + blk * 4u + (l_idx >> 2)];
            uint ksw = k_cache[row_base + nb * 4u + (blk >> 1)];
            float ks = ((blk & 1u) == 0u) ? f16tof32(ksw & 0xFFFFu) : f16tof32(ksw >> 16);
            k_d = ks * (float)((int)((kw >> shift) & 0xFu) - 8);

            uint vw = v_cache[row_base + blk * 4u + (l_idx >> 2)];
            uint vsw = v_cache[row_base + nb * 4u + (blk >> 1)];
            float vs = ((blk & 1u) == 0u) ? f16tof32(vsw & 0xFFFFu) : f16tof32(vsw >> 16);
            v_d = vs * (float)((int)((vw >> shift) & 0xFu) - 8);
        }

        s_red[tid] = q_d * k_d;
        GroupMemoryBarrierWithGroupSync();

        if (tid < 128u) s_red[tid] += s_red[tid + 128u];
        GroupMemoryBarrierWithGroupSync();
        if (tid < 64u) s_red[tid] += s_red[tid + 64u];
        GroupMemoryBarrierWithGroupSync();
        if (tid < 32u) s_red[tid] += s_red[tid + 32u];
        GroupMemoryBarrierWithGroupSync();
        if (tid < 16u) s_red[tid] += s_red[tid + 16u];
        GroupMemoryBarrierWithGroupSync();
        if (tid < 8u)  s_red[tid] += s_red[tid + 8u];
        GroupMemoryBarrierWithGroupSync();
        if (tid < 4u)  s_red[tid] += s_red[tid + 4u];
        GroupMemoryBarrierWithGroupSync();
        if (tid < 2u)  s_red[tid] += s_red[tid + 2u];
        GroupMemoryBarrierWithGroupSync();
        if (tid < 1u)  s_red[tid] += s_red[tid + 1u];
        GroupMemoryBarrierWithGroupSync();

        float s_t = s_red[0] * attn_scale;

        float m_new = max(m, s_t);
        float alpha = exp(m - m_new);
        float w_t = exp(s_t - m_new);

        acc = acc * alpha + w_t * v_d;
        l = l * alpha + w_t;
        m = m_new;

        GroupMemoryBarrierWithGroupSync();
    }

    if (tid < head_dim)
    {
        attn_out[h * head_dim + tid] = (l > 0.0f) ? (acc / l) : 0.0f;
    }
}
";
}
