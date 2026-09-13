namespace Glacier.Inference.Gpu.D3D12;

public static partial class D3D12Shaders
{
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

    public const string VecAddWeighted = @"
cbuffer Params : register(b0)
{
    uint size;
    float weight;
    uint accumulate;
};

RWStructuredBuffer<float> b : register(u0);
RWStructuredBuffer<float> a : register(u1);

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x < size)
    {
        if (accumulate != 0)
        {
            a[id.x] += weight * b[id.x];
        }
        else
        {
            a[id.x] = weight * b[id.x];
        }
    }
}
";

    public const string RmsNormHeads = @"
cbuffer Params : register(b0)
{
    uint head_dim;
    uint n_heads;
    float eps;
};

ByteAddressBuffer weight : register(t0);
RWStructuredBuffer<float> x : register(u0);

groupshared float s_sum[128];

[numthreads(128, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint head_idx = gid.x;
    uint tid = gtid.x;
    if (head_idx >= n_heads) return;

    uint head_offset = head_idx * head_dim;
    float val = (tid < head_dim) ? x[head_offset + tid] : 0.0f;
    s_sum[tid] = val * val;

    GroupMemoryBarrierWithGroupSync();
    if (tid < 64) s_sum[tid] += s_sum[tid + 64];
    GroupMemoryBarrierWithGroupSync();
    if (tid < 32) s_sum[tid] += s_sum[tid + 32];
    GroupMemoryBarrierWithGroupSync();
    if (tid < 16) s_sum[tid] += s_sum[tid + 16];
    GroupMemoryBarrierWithGroupSync();
    if (tid < 8)  s_sum[tid] += s_sum[tid + 8];
    GroupMemoryBarrierWithGroupSync();
    if (tid < 4)  s_sum[tid] += s_sum[tid + 4];
    GroupMemoryBarrierWithGroupSync();
    if (tid < 2)  s_sum[tid] += s_sum[tid + 2];
    GroupMemoryBarrierWithGroupSync();
    if (tid < 1)  s_sum[tid] += s_sum[tid + 1];

    float rrms = rsqrt(s_sum[0] / (float)head_dim + eps);

    if (tid < head_dim)
    {
        float w = asfloat(weight.Load(tid * 4));
        x[head_offset + tid] = val * rrms * w;
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
