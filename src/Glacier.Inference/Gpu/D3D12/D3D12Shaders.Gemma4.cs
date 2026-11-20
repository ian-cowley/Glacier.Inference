namespace Glacier.Inference.Gpu.D3D12;

public static partial class D3D12Shaders
{
    /// <summary>
    /// Gemma 4 GeLU-GLU activation shader: dst[i] = GeLU(gate[i]) * up[up_offset + i].
    /// Saturates large activations cleanly without HLSL tanh Inf/Inf NaN overflow.
    /// </summary>
    public const string Gemma4GeluGLU = @"
cbuffer Params : register(b0)
{
    uint size;
    uint up_offset;
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
        float gelu;
        if (g >= 10.0f)
        {
            gelu = g;
        }
        else if (g <= -10.0f)
        {
            gelu = 0.0f;
        }
        else
        {
            float t = clamp(0.7978845608f * g * (1.0f + 0.044715f * g * g), -20.0f, 20.0f);
            gelu = 0.5f * g * (1.0f + tanh(t));
        }
        dst[id.x] = gelu * up[up_offset + id.x];
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
        logits[id.x] = cap * tanh(clamp(l / cap, -20.0f, 20.0f));
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
    float attn_logit_softcapping;
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
        if (attn_logit_softcapping > 0.0f)
        {
            s_t = attn_logit_softcapping * tanh(clamp(s_t / attn_logit_softcapping, -20.0f, 20.0f));
        }

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

    public const string Gemma4SwaAttentionF16 = @"
cbuffer Params : register(b0)
{
    uint n_heads_q;
    uint n_heads_kv;
    uint head_dim;
    uint max_seq_len;
    uint pos;
    float attn_scale;
    uint sliding_window;
    float attn_logit_softcapping;
};

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

    float q0 = (tid < head_dim) ? q[h * head_dim + tid] : 0.0f;
    float q1 = (tid + 256 < head_dim) ? q[h * head_dim + tid + 256] : 0.0f;

    float m = -1e30f;
    float l = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    uint window_start = (sliding_window > 0 && pos >= sliding_window) ? (pos - sliding_window + 1) : 0;

    for (uint t = window_start; t <= pos; t++)
    {
        uint kv_base = (h_kv * max_seq_len + t) * half_dim;

        float k0 = 0.0f;
        float v0 = 0.0f;
        if (tid < head_dim)
        {
            uint kp0 = k_cache[kv_base + (tid >> 1)];
            uint vp0 = v_cache[kv_base + (tid >> 1)];
            k0 = ((tid & 1u) == 0u) ? f16tof32(kp0 & 0xFFFFu) : f16tof32(kp0 >> 16);
            v0 = ((tid & 1u) == 0u) ? f16tof32(vp0 & 0xFFFFu) : f16tof32(vp0 >> 16);
        }

        float dot_part = q0 * k0;
        float v1 = 0.0f;
        uint d1 = tid + 256u;
        if (d1 < head_dim)
        {
            uint kp1 = k_cache[kv_base + (d1 >> 1)];
            uint vp1 = v_cache[kv_base + (d1 >> 1)];
            float k1 = ((d1 & 1u) == 0u) ? f16tof32(kp1 & 0xFFFFu) : f16tof32(kp1 >> 16);
            v1 = ((d1 & 1u) == 0u) ? f16tof32(vp1 & 0xFFFFu) : f16tof32(vp1 >> 16);
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

        float s_t = s_red[0] * attn_scale;
        if (attn_logit_softcapping > 0.0f)
        {
            s_t = attn_logit_softcapping * tanh(clamp(s_t / attn_logit_softcapping, -20.0f, 20.0f));
        }

        float m_new = max(m, s_t);
        float alpha = exp(m - m_new);
        float w_t = exp(s_t - m_new);

        acc0 = acc0 * alpha + w_t * v0;
        if (d1 < head_dim)
        {
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

    public const string Gemma4SwaAttentionQ4_0 = @"
cbuffer Params : register(b0)
{
    uint n_heads_q;
    uint n_heads_kv;
    uint head_dim;
    uint max_seq_len;
    uint pos;
    float attn_scale;
    uint sliding_window;
    float attn_logit_softcapping;
};

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

    float q0 = (tid < head_dim) ? q[h * head_dim + tid] : 0.0f;
    float q1 = (tid + 256 < head_dim) ? q[h * head_dim + tid + 256] : 0.0f;

    float m = -1e30f;
    float l = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    uint window_start = (sliding_window > 0 && pos >= sliding_window) ? (pos - sliding_window + 1) : 0;

    for (uint t = window_start; t <= pos; t++)
    {
        uint row_base = (h_kv * max_seq_len + t) * row_uints;

        float k0 = 0.0f;
        float v0 = 0.0f;
        if (tid < head_dim)
        {
            uint blk0 = tid >> 5;
            uint elem0 = tid & 31u;
            uint l0 = elem0 & 15u;
            uint sh0 = ((l0 & 3u) * 8u) + ((elem0 >= 16u) ? 4u : 0u);

            uint kw0 = k_cache[row_base + blk0 * 4u + (l0 >> 2)];
            uint ksw0 = k_cache[row_base + nb * 4u + (blk0 >> 1)];
            float ks0 = ((blk0 & 1u) == 0u) ? f16tof32(ksw0 & 0xFFFFu) : f16tof32(ksw0 >> 16);
            k0 = ks0 * (float)((int)((kw0 >> sh0) & 0xFu) - 8);

            uint vw0 = v_cache[row_base + blk0 * 4u + (l0 >> 2)];
            uint vsw0 = v_cache[row_base + nb * 4u + (blk0 >> 1)];
            float vs0 = ((blk0 & 1u) == 0u) ? f16tof32(vsw0 & 0xFFFFu) : f16tof32(vsw0 >> 16);
            v0 = vs0 * (float)((int)((vw0 >> sh0) & 0xFu) - 8);
        }

        float dot_part = q0 * k0;
        float v1 = 0.0f;
        uint d1 = tid + 256u;
        if (d1 < head_dim)
        {
            uint blk1 = d1 >> 5;
            uint elem1 = d1 & 31u;
            uint l1 = elem1 & 15u;
            uint sh1 = ((l1 & 3u) * 8u) + ((elem1 >= 16u) ? 4u : 0u);

            uint kw1 = k_cache[row_base + blk1 * 4u + (l1 >> 2)];
            uint ksw1 = k_cache[row_base + nb * 4u + (blk1 >> 1)];
            float ks1 = ((blk1 & 1u) == 0u) ? f16tof32(ksw1 & 0xFFFFu) : f16tof32(ksw1 >> 16);
            float k1 = ks1 * (float)((int)((kw1 >> sh1) & 0xFu) - 8);
            dot_part += q1 * k1;

            uint vw1 = v_cache[row_base + blk1 * 4u + (l1 >> 2)];
            uint vsw1 = v_cache[row_base + nb * 4u + (blk1 >> 1)];
            float vs1 = ((blk1 & 1u) == 0u) ? f16tof32(vsw1 & 0xFFFFu) : f16tof32(vsw1 >> 16);
            v1 = vs1 * (float)((int)((vw1 >> sh1) & 0xFu) - 8);
        }

        s_red[tid] = dot_part;
        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (uint s = 128; s > 0; s >>= 1)
        {
            if (tid < s) s_red[tid] += s_red[tid + s];
            GroupMemoryBarrierWithGroupSync();
        }

        float s_t = s_red[0] * attn_scale;
        if (attn_logit_softcapping > 0.0f)
        {
            s_t = attn_logit_softcapping * tanh(clamp(s_t / attn_logit_softcapping, -20.0f, 20.0f));
        }

        float m_new = max(m, s_t);
        float alpha = exp(m - m_new);
        float w_t = exp(s_t - m_new);

        acc0 = acc0 * alpha + w_t * v0;
        if (d1 < head_dim)
        {
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

    /// <summary>
    /// Gemma 4 RoPE shader supporting optional per-frequency rope_freqs factors on dense attention layers.
    /// </summary>
    public const string Gemma4RoPE = @"
cbuffer Params : register(b0)
{
    uint n_heads_q;
    uint n_heads_kv;
    uint head_dim;
    uint pos;
    float freq_base;
    float freq_scale;
    uint has_rope_freqs;
};

StructuredBuffer<float> rope_freqs : register(t0);
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

    float base_freq = 1.0f / pow(freq_base, (float)(2 * i) / (float)head_dim);
    float freq = (has_rope_freqs != 0) ? (base_freq / rope_freqs[i]) : base_freq;
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
}
