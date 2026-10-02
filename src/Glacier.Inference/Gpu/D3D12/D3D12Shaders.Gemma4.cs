namespace Glacier.Inference.Gpu.D3D12;

public static partial class D3D12Shaders
{
    /// <summary>
    /// Gemma 4 GeLU-GLU activation shader: dst[i] = GeLU(gate[i]) * up[i].
    /// Uses native GPU hardware tanh instruction.
    /// </summary>
    public const string Gemma4GeluGLU = @"
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
        float gelu = 0.5f * g * (1.0f + tanh(0.7978845608f * g * (1.0f + 0.044715f * g * g)));
        dst[id.x] = gelu * up[id.x];
    }
}
";

    /// <summary>
    /// Gemma 4 Tanh Logit Softcapping: logits[i] = cap * tanh(logits[i] / cap).
    /// </summary>
    public const string Gemma4SoftcapLogits = @"
cbuffer Params : register(b0)
{
    uint size;
    float cap;
};

RWStructuredBuffer<float> logits : register(u0);

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x < size)
    {
        float l = logits[id.x];
        logits[id.x] = cap * tanh(l / cap);
    }
}
";

    /// <summary>
    /// Scales vector x by a scalar and elementwise multiplies by an optional scale vector: dst[i] = x[i] * scalar * scaleVec[i].
    /// </summary>
    public const string Gemma4ScaleAndMul = @"
cbuffer Params : register(b0)
{
    uint size;
    float scalar;
    uint has_scale_vec;
};

RWStructuredBuffer<float> x : register(u0);
StructuredBuffer<float> scale_vec : register(t0);
RWStructuredBuffer<float> dst : register(u1);

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x < size)
    {
        float s = (has_scale_vec != 0) ? scale_vec[id.x] : 1.0f;
        dst[id.x] = x[id.x] * scalar * s;
    }
}
";

    /// <summary>
    /// Interleaved Sliding Window Attention (ISWA) shader for Gemma 4.
    /// Supports dynamic head dimensions (256 for SWA, 512 for Dense) and sliding window masking.
    /// Evaluates online softmax reduction on GPU registers.
    /// </summary>
    public const string Gemma4SwaAttention = @"
cbuffer Params : register(b0)
{
    uint n_heads_q;
    uint n_heads_kv;
    uint head_dim;
    uint max_seq_len;
    uint pos;
    float attn_scale;
    uint sliding_window;
};

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

    float q0 = (tid < head_dim) ? q[h * head_dim + tid] : 0.0f;
    float q1 = (tid + 256 < head_dim) ? q[h * head_dim + tid + 256] : 0.0f;

    float m = -1e30f;
    float l = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    uint window_start = (sliding_window > 0 && pos >= sliding_window) ? (pos - sliding_window + 1) : 0;

    for (uint t = window_start; t <= pos; t++)
    {
        uint kv_offset0 = (h_kv * max_seq_len + t) * head_dim + tid;
        uint kv_offset1 = kv_offset0 + 256;

        float k0 = (tid < head_dim) ? k_cache[kv_offset0] : 0.0f;
        float v0 = (tid < head_dim) ? v_cache[kv_offset0] : 0.0f;

        float dot_part = q0 * k0;
        if (tid + 256 < head_dim)
        {
            float k1 = k_cache[kv_offset1];
            dot_part += q1 * k1;
        }

        s_red[tid] = dot_part;
        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (uint s = 128; s > 0; s >>= 1)
        {
            if (tid < s) s_red[tid] += s_red[tid + s];
            GroupMemoryBarrierWithGroupSync();
        }

        float total_dot = s_red[0];
        float s_t = total_dot * attn_scale;

        float m_new = max(m, s_t);
        float alpha = exp(m - m_new);
        float w_t = exp(s_t - m_new);

        acc0 = acc0 * alpha + w_t * v0;
        if (tid + 256 < head_dim)
        {
            float v1 = v_cache[kv_offset1];
            acc1 = acc1 * alpha + w_t * v1;
        }

        l = l * alpha + w_t;
        m = m_new;

        GroupMemoryBarrierWithGroupSync();
    }

    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    if (tid < head_dim)
    {
        attn_out[h * head_dim + tid] = acc0 * inv_l;
    }
    if (tid + 256 < head_dim)
    {
        attn_out[h * head_dim + tid + 256] = acc1 * inv_l;
    }
}
";

    /// <summary>
    /// Gemma 4 KV Cache Store shader with variable head dimension per layer.
    /// </summary>
    public const string Gemma4KvCacheStore = @"
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

    /// <summary>
    /// Gemma 4 Layer Output Scaling: x[i] = x[i] * scale.
    /// </summary>
    public const string Gemma4LayerOutputScale = @"
cbuffer Params : register(b0)
{
    uint size;
    float scale;
};

RWStructuredBuffer<float> x : register(u0);

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x < size)
    {
        x[id.x] *= scale;
    }
}
";
}
