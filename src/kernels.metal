// Metal kernels, embedded into backend_metal.m at build time and compiled at startup.
// Each kernel mirrors a function in ops_ref.c.
#include <metal_stdlib>
using namespace metal;

// ---- helpers ----------------------------------------------------------------

// reductions over a whole threadgroup (<= 1024 threads); every thread gets the result
inline float tg_sum(float v, threadgroup float *tmp, uint tid, uint ntg) {
    v = simd_sum(v);
    if (tid % 32 == 0) tmp[tid / 32] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = 0.0f;
    for (uint i = 0; i < (ntg + 31) / 32; i++) r += tmp[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

inline float tg_max(float v, threadgroup float *tmp, uint tid, uint ntg) {
    v = simd_max(v);
    if (tid % 32 == 0) tmp[tid / 32] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = -INFINITY;
    for (uint i = 0; i < (ntg + 31) / 32; i++) r = max(r, tmp[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

inline float bf16(ushort h) { return as_type<float>(uint(h) << 16); }
inline float silu(float x) { return x / (1.0f + exp(-x)); }
inline float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

// ---- elementwise and small ops ----------------------------------------------

// weight formats, must match WeightType in backend.h
constant int WT_BF16 = 0, WT_Q8 = 1, WT_Q4 = 2;
constant int QUANT_GROUP = 32;

kernel void embed(device const uchar *table  [[buffer(0)]],
                  device float        *out    [[buffer(1)]],
                  constant int        &token  [[buffer(2)]],
                  constant int        &n      [[buffer(3)]],
                  constant int        &type   [[buffer(4)]],
                  device const half   *scales [[buffer(5)]],
                  uint i [[thread_position_in_grid]]) {
    if ((int)i >= n) return;
    ulong j = (ulong)token * n + i;
    if (type == WT_BF16) out[i] = bf16(((device const ushort *)table)[j]);
    else if (type == WT_Q8) out[i] = ((device const char *)table)[j] * float(scales[j / QUANT_GROUP]);
    else out[i] = (int((table[j / 2] >> (j % 2 * 4)) & 15) - 8) * float(scales[j / QUANT_GROUP]);
}

// The matvec kernels: one simdgroup (32 threads) per output row, 8 rows per
// threadgroup. Lanes read consecutive chunks of a row so the simdgroup's loads
// are contiguous.

kernel void matvec_bf16(device const ushort *w   [[buffer(0)]],
                        device const float  *x   [[buffer(1)]],
                        device float        *out [[buffer(2)]],
                        constant int        &n   [[buffer(3)]],
                        constant int        &d   [[buffer(4)]],
                        uint tg   [[threadgroup_position_in_grid]],
                        uint sg   [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    int row = tg * 8 + sg;
    if (row >= d) return;
    device const ushort4 *w4 = (device const ushort4 *)(w + (ulong)row * n);
    device const float4  *x4 = (device const float4 *)x;
    float acc = 0.0f;
    for (int i = lane; i < n / 4; i += 32) {
        uint4 u = uint4(w4[i]) << 16;  // bf16 -> f32 is a 16-bit shift
        acc += dot(as_type<float4>(u), x4[i]);
    }
    acc = simd_sum(acc);
    if (lane == 0) out[row] = acc;
}

// each lane takes 16 weights (16 bytes) per step; a chunk never spans two scale groups
kernel void matvec_q8(device const char4  *w      [[buffer(0)]],
                      device const float4 *x      [[buffer(1)]],
                      device float        *out    [[buffer(2)]],
                      constant int        &n      [[buffer(3)]],
                      constant int        &d      [[buffer(4)]],
                      device const half   *scales [[buffer(5)]],
                      uint tg   [[threadgroup_position_in_grid]],
                      uint sg   [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    int row = tg * 8 + sg;
    if (row >= d) return;
    device const char4 *wr = w + (ulong)row * n / 4;
    device const half  *sr = scales + (ulong)row * n / QUANT_GROUP;
    float acc = 0.0f;
    for (int i = lane * 16; i < n; i += 32 * 16) {
        int c = i / 4;
        float sum = dot(float4(wr[c]), x[c]) + dot(float4(wr[c + 1]), x[c + 1])
                  + dot(float4(wr[c + 2]), x[c + 2]) + dot(float4(wr[c + 3]), x[c + 3]);
        acc += sum * float(sr[i / QUANT_GROUP]);
    }
    acc = simd_sum(acc);
    if (lane == 0) out[row] = acc;
}

// 8 int4 weights packed in a uint (low nibble first, stored +8), dotted with 8 inputs
inline float dot_q4(uint u, float4 a, float4 b) {
    float4 lo = float4(u & 15, (u >> 4) & 15, (u >> 8) & 15, (u >> 12) & 15) - 8.0f;
    float4 hi = float4((u >> 16) & 15, (u >> 20) & 15, (u >> 24) & 15, (u >> 28) & 15) - 8.0f;
    return dot(lo, a) + dot(hi, b);
}

// each lane takes 16 weights (8 bytes) per step; a chunk never spans two scale groups
kernel void matvec_q4(device const uint2  *w      [[buffer(0)]],
                      device const float4 *x      [[buffer(1)]],
                      device float        *out    [[buffer(2)]],
                      constant int        &n      [[buffer(3)]],
                      constant int        &d      [[buffer(4)]],
                      device const half   *scales [[buffer(5)]],
                      uint tg   [[threadgroup_position_in_grid]],
                      uint sg   [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    int row = tg * 8 + sg;
    if (row >= d) return;
    device const uint2 *wr = w + (ulong)row * n / 16;
    device const half  *sr = scales + (ulong)row * n / QUANT_GROUP;
    float acc = 0.0f;
    for (int i = lane * 16; i < n; i += 32 * 16) {
        uint2 u = wr[i / 16];
        int c = i / 4;
        float sum = dot_q4(u.x, x[c], x[c + 1]) + dot_q4(u.y, x[c + 2], x[c + 3]);
        acc += sum * float(sr[i / QUANT_GROUP]);
    }
    acc = simd_sum(acc);
    if (lane == 0) out[row] = acc;
}

// a single threadgroup
kernel void rmsnorm(device const float *x   [[buffer(0)]],
                    device const float *w   [[buffer(1)]],
                    device float       *out [[buffer(2)]],
                    constant int       &n   [[buffer(3)]],
                    constant float     &eps [[buffer(4)]],
                    uint tid [[thread_position_in_threadgroup]],
                    uint ntg [[threads_per_threadgroup]]) {
    threadgroup float tmp[32];
    float ss = 0.0f;
    for (int i = tid; i < n; i += ntg) ss += x[i] * x[i];
    float r = rsqrt(tg_sum(ss, tmp, tid, ntg) / n + eps);
    for (int i = tid; i < n; i += ntg) out[i] = x[i] * r * w[i];
}

kernel void add(device float       *x [[buffer(0)]],
                device const float *y [[buffer(1)]],
                constant int       &n [[buffer(2)]],
                uint i [[thread_position_in_grid]]) {
    if ((int)i < n) x[i] += y[i];
}

kernel void swiglu(device float       *out  [[buffer(0)]],
                   device const float *gate [[buffer(1)]],
                   device const float *up   [[buffer(2)]],
                   constant int       &n    [[buffer(3)]],
                   uint i [[thread_position_in_grid]]) {
    if ((int)i < n) out[i] = silu(gate[i]) * up[i];
}

// ---- full attention ---------------------------------------------------------
// both kernels share one binding layout, see bind_attention() in backend_metal.m:
//   0 q  1 k  2 v  3 k_cache  4 v_cache  5 att  6 q_norm  7 k_norm  8 out  9 params

struct AttnParams { int n_heads, n_kv_heads, head_dim, rot_dim, pos, seq_len; float eps, rope_theta; };

// RMSNorm + RoPE of every query head (in place) and key head (into the f16
// cache), plus the value copy. threadgroup g < n_heads is query head g, the rest are
// kv heads. one thread per dimension.
kernel void attn_prep(device float          *q       [[buffer(0)]],
                      device const float    *k       [[buffer(1)]],
                      device const float    *v       [[buffer(2)]],
                      device half           *k_cache [[buffer(3)]],
                      device half           *v_cache [[buffer(4)]],
                      device const float    *q_norm  [[buffer(6)]],
                      device const float    *k_norm  [[buffer(7)]],
                      constant AttnParams   &p       [[buffer(9)]],
                      uint g [[threadgroup_position_in_grid]],
                      uint i [[thread_position_in_threadgroup]],
                      uint ntg [[threads_per_threadgroup]]) {
    threadgroup float tmp[32], xs[1024];
    int hd = p.head_dim, kv_dim = p.n_kv_heads * hd;
    device const float *src, *w;
    device float *q_dst = nullptr;
    device half *k_dst = nullptr;
    if ((int)g < p.n_heads) {
        src = q_dst = q + g * 2 * hd;
        w = q_norm;
    } else {
        int h = g - p.n_heads;
        src = k + h * hd;
        k_dst = k_cache + (ulong)p.pos * kv_dim + h * hd;
        w = k_norm;
        v_cache[(ulong)p.pos * kv_dim + h * hd + i] = half(v[h * hd + i]);
    }
    float x = src[i];
    x *= rsqrt(tg_sum(x * x, tmp, i, ntg) / hd + p.eps) * w[i];
    xs[i] = x;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // rotate-half RoPE on the first rot_dim dims ("half" is a type in MSL, hence hr)
    int hr = p.rot_dim / 2;
    if ((int)i < p.rot_dim) {
        int j = (int)i < hr ? i : i - hr;
        float f = p.pos * precise::pow(p.rope_theta, -2.0f * j / p.rot_dim);
        float c = precise::cos(f), s = precise::sin(f);
        x = (int)i < hr ? xs[i] * c - xs[i + hr] * s : xs[i] * c + xs[j] * s;
    }
    if (q_dst) q_dst[i] = x;
    else k_dst[i] = half(x);
}

// one threadgroup per query head: scores, softmax, weighted sum of values, output gate
kernel void attn_decode(device float        *q       [[buffer(0)]],
                        device const half   *k_cache [[buffer(3)]],
                        device const half   *v_cache [[buffer(4)]],
                        device float        *att     [[buffer(5)]],
                        device float        *out     [[buffer(8)]],
                        constant AttnParams &p       [[buffer(9)]],
                        uint h [[threadgroup_position_in_grid]],
                        uint tid [[thread_position_in_threadgroup]],
                        uint ntg [[threads_per_threadgroup]]) {
    threadgroup float tmp[32], qs[1024];
    int hd = p.head_dim, kv_dim = p.n_kv_heads * hd, n = p.pos + 1;
    int kv_off = (h / (p.n_heads / p.n_kv_heads)) * hd;
    device const float *qh = q + h * 2 * hd, *gate = qh + hd;
    device float *a = att + (ulong)h * p.seq_len;
    for (int i = tid; i < hd; i += ntg) qs[i] = qh[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = rsqrt((float)hd), mx = -INFINITY;
    for (int t = tid; t < n; t += ntg) {
        device const half *kt = k_cache + (ulong)t * kv_dim + kv_off;
        float dot = 0.0f;
        for (int i = 0; i < hd; i++) dot += qs[i] * float(kt[i]);
        a[t] = dot * scale;
        mx = max(mx, a[t]);
    }
    mx = tg_max(mx, tmp, tid, ntg);
    float sum = 0.0f;
    for (int t = tid; t < n; t += ntg) {
        a[t] = exp(a[t] - mx);
        sum += a[t];
    }
    sum = tg_sum(sum, tmp, tid, ntg);
    threadgroup_barrier(mem_flags::mem_device);  // every a[t] is visible to every thread

    for (int i = tid; i < hd; i += ntg) {
        float acc = 0.0f;
        for (int t = 0; t < n; t++) acc += a[t] * float(v_cache[(ulong)t * kv_dim + kv_off + i]);
        out[h * hd + i] = acc / sum * sigmoid(gate[i]);
    }
}

// ---- gated deltanet ---------------------------------------------------------
// both kernels share one binding layout, see bind_deltanet() in backend_metal.m:
//   0 qkv  1 z  2 a  3 b  4 conv_state  5 ssm_state  6 conv_w  7 A_log  8 dt_bias
//   9 norm  10 out  11 params

struct DeltaParams { int nk, nv, dk, dv, conv_k; float eps; };

// causal depthwise conv1d + SiLU, one thread per channel
kernel void delta_conv(device float         *qkv        [[buffer(0)]],
                       device float         *conv_state [[buffer(4)]],
                       device const float   *conv_w     [[buffer(6)]],
                       constant DeltaParams &p          [[buffer(11)]],
                       uint ch [[thread_position_in_grid]]) {
    int K = p.conv_k, conv_dim = 2 * p.nk * p.dk + p.nv * p.dv;
    if ((int)ch >= conv_dim) return;
    device const float *w = conv_w + ch * K;
    float in = qkv[ch], acc = w[K - 1] * in;
    for (int k = 0; k < K - 1; k++) acc += w[k] * conv_state[k * conv_dim + ch];
    for (int k = 0; k < K - 2; k++) conv_state[k * conv_dim + ch] = conv_state[(k + 1) * conv_dim + ch];
    conv_state[(K - 2) * conv_dim + ch] = in;
    qkv[ch] = silu(acc);
}

// gated delta rule step + gated RMSNorm. one threadgroup per value head and
// one thread per state column: S is [dk, dv] and thread j owns S[:, j].
kernel void delta_step(device const float   *qkv       [[buffer(0)]],
                       device const float   *z         [[buffer(1)]],
                       device const float   *a         [[buffer(2)]],
                       device const float   *b         [[buffer(3)]],
                       device float         *ssm_state [[buffer(5)]],
                       device const float   *A_log     [[buffer(7)]],
                       device const float   *dt_bias   [[buffer(8)]],
                       device const float   *norm      [[buffer(9)]],
                       device float         *out       [[buffer(10)]],
                       constant DeltaParams &p         [[buffer(11)]],
                       uint h [[threadgroup_position_in_grid]],
                       uint j [[thread_position_in_threadgroup]],
                       uint ntg [[threads_per_threadgroup]]) {
    threadgroup float tmp[32], qs[1024], ks[1024];
    int dk = p.dk, dv = p.dv, key_dim = p.nk * dk, kh = h / (p.nv / p.nk);
    float qss = 0.0f, kss = 0.0f;
    for (int i = j; i < dk; i += ntg) {
        qs[i] = qkv[kh * dk + i];
        ks[i] = qkv[key_dim + kh * dk + i];
        qss += qs[i] * qs[i];
        kss += ks[i] * ks[i];
    }
    // l2norm(q) / sqrt(dk) and l2norm(k), applied as scalars
    float rq = rsqrt(tg_sum(qss, tmp, j, ntg) + 1e-6f) * rsqrt((float)dk);
    float rk = rsqrt(tg_sum(kss, tmp, j, ntg) + 1e-6f);

    float beta = sigmoid(b[h]);
    float x = a[h] + dt_bias[h], softplus = x > 20.0f ? x : log(1.0f + exp(x));
    float decay = exp(-exp(A_log[h]) * softplus);

    // S = S * decay;  delta = beta * (v - S^T k);  S = S + k delta^T;  o = S^T q
    device float *S = ssm_state + (ulong)h * dk * dv;
    float kv = 0.0f;
    for (int i = 0; i < dk; i++) kv += S[i * dv + j] * decay * ks[i] * rk;
    float delta = beta * (qkv[2 * key_dim + h * dv + j] - kv);
    float o = 0.0f;
    for (int i = 0; i < dk; i++) {
        float s = S[i * dv + j] * decay + ks[i] * rk * delta;
        S[i * dv + j] = s;
        o += s * qs[i] * rq;
    }

    // gated RMSNorm: norm(o) * w * silu(z)
    float r = rsqrt(tg_sum(o * o, tmp, j, ntg) / dv + p.eps);
    out[h * dv + j] = o * r * norm[j] * silu(z[h * dv + j]);
}
