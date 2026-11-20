// Reference CPU implementations of every backend op. Plain C, f32.
// These are the readable spec of the model math; GPU kernels must match them.
#include <math.h>
#include <string.h>
#include "backend.h"
#include "util.h"

static float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }
static float siluf(float x) { return x / (1.0f + expf(-x)); }
static float softplusf(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

static void l2norm(float *x, int n) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss + 1e-6f);
    for (int i = 0; i < n; i++) x[i] *= ss;
}

static void softmax(float *x, int n) {
    float max = x[0], sum = 0.0f;
    for (int i = 1; i < n; i++) if (x[i] > max) max = x[i];
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

// rotate-half RoPE on the first rot_dim dims of a head
static void rope(float *x, int pos, int rot_dim, float theta) {
    int half = rot_dim / 2;
    for (int i = 0; i < half; i++) {
        float f = (float)pos * powf(theta, -2.0f * i / rot_dim), c = cosf(f), s = sinf(f);
        float x1 = x[i], x2 = x[i + half];
        x[i] = x1 * c - x2 * s;
        x[i + half] = x2 * c + x1 * s;
    }
}

// element i of a matrix (row-major index), as a float
static float mat_get(Mat m, size_t i) {
    switch (m.type) {
    case WT_BF16: return bf16_to_f32(((const uint16_t *)m.data)[i]);
    case WT_Q8:   return ((const int8_t *)m.data)[i] * f16_to_f32(mat_scales(m)[i / QUANT_GROUP]);
    case WT_Q4: {
        uint8_t byte = ((const uint8_t *)m.data)[i / 2];
        int q = (i % 2 ? byte >> 4 : byte & 15) - 8;
        return q * f16_to_f32(mat_scales(m)[i / QUANT_GROUP]);
    }
    }
    return 0.0f;
}

void ref_embed(float *out, Mat table, int token) {
    for (int i = 0; i < table.cols; i++) out[i] = mat_get(table, (size_t)token * table.cols + i);
}

void ref_matvec(float *out, Mat w, const float *x) {
    for (int i = 0; i < w.rows; i++) {
        float acc = 0.0f;
        for (int j = 0; j < w.cols; j++) acc += mat_get(w, (size_t)i * w.cols + j) * x[j];
        out[i] = acc;
    }
}

void ref_rmsnorm(float *out, const float *x, const float *w, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * ss * w[i];
}

void ref_add(float *x, const float *y, int n) {
    for (int i = 0; i < n; i++) x[i] += y[i];
}

void ref_swiglu(float *out, const float *gate, const float *up, int n) {
    for (int i = 0; i < n; i++) out[i] = siluf(gate[i]) * up[i];
}

void ref_attention(const AttnArgs *a) {
    int hd = a->head_dim, kv_dim = a->n_kv_heads * hd;
    _Float16 *kc = a->k_cache + (size_t)a->pos * kv_dim, *vc = a->v_cache + (size_t)a->pos * kv_dim;
    float kh[hd];
    for (int h = 0; h < a->n_kv_heads; h++) {
        ref_rmsnorm(kh, a->k + h * hd, a->k_norm, hd, a->eps);
        rope(kh, a->pos, a->rot_dim, a->rope_theta);
        for (int i = 0; i < hd; i++) kc[h * hd + i] = kh[i];
    }
    for (int i = 0; i < kv_dim; i++) vc[i] = a->v[i];

    float scale = 1.0f / sqrtf((float)hd);
    for (int h = 0; h < a->n_heads; h++) {
        float *q = a->q + h * 2 * hd, *gate = q + hd;
        ref_rmsnorm(q, q, a->q_norm, hd, a->eps);
        rope(q, a->pos, a->rot_dim, a->rope_theta);
        int kvh = h / (a->n_heads / a->n_kv_heads);
        float *att = a->att + (size_t)h * a->seq_len;
        for (int t = 0; t <= a->pos; t++) {
            const _Float16 *k = a->k_cache + (size_t)t * kv_dim + kvh * hd;
            float dot = 0.0f;
            for (int i = 0; i < hd; i++) dot += q[i] * k[i];
            att[t] = dot * scale;
        }
        softmax(att, a->pos + 1);
        float *o = a->out + h * hd;
        memset(o, 0, hd * sizeof(float));
        for (int t = 0; t <= a->pos; t++) {
            const _Float16 *v = a->v_cache + (size_t)t * kv_dim + kvh * hd;
            for (int i = 0; i < hd; i++) o[i] += att[t] * v[i];
        }
        for (int i = 0; i < hd; i++) o[i] *= sigmoidf(gate[i]);
    }
}

void ref_deltanet(const DeltaArgs *a) {
    int nk = a->nk, nv = a->nv, dk = a->dk, dv = a->dv, K = a->conv_k;
    int key_dim = nk * dk, conv_dim = 2 * key_dim + nv * dv;

    // causal depthwise conv1d over the last K inputs, then SiLU
    float *cs = a->conv_state;
    for (int ch = 0; ch < conv_dim; ch++) {
        const float *w = a->conv_w + ch * K;
        float in = a->qkv[ch], acc = w[K - 1] * in;
        for (int k = 0; k < K - 1; k++) acc += w[k] * cs[k * conv_dim + ch];
        for (int k = 0; k < K - 2; k++) cs[k * conv_dim + ch] = cs[(k + 1) * conv_dim + ch];
        cs[(K - 2) * conv_dim + ch] = in;
        a->qkv[ch] = siluf(acc);
    }

    float *q = a->qkv, *k = a->qkv + key_dim, *v = a->qkv + 2 * key_dim;
    for (int h = 0; h < nk; h++) {
        l2norm(q + h * dk, dk);
        l2norm(k + h * dk, dk);
        for (int i = 0; i < dk; i++) q[h * dk + i] /= sqrtf((float)dk);
    }

    // gated delta rule, one step. S is [dk, dv]:
    //   S = S * exp(g);  delta = beta * (v - S^T k);  S = S + k delta^T;  o = S^T q
    for (int h = 0; h < nv; h++) {
        int kh = h / (nv / nk);
        float *qh = q + kh * dk, *kh_ = k + kh * dk, *vh = v + h * dv;
        float beta = sigmoidf(a->b[h]);
        float decay = expf(-expf(a->A_log[h]) * softplusf(a->a[h] + a->dt_bias[h]));
        float *S = a->ssm_state + (size_t)h * dk * dv;

        float kv[dv], delta[dv];
        memset(kv, 0, sizeof(kv));
        for (int i = 0; i < dk; i++)
            for (int j = 0; j < dv; j++) {
                S[i * dv + j] *= decay;
                kv[j] += S[i * dv + j] * kh_[i];
            }
        for (int j = 0; j < dv; j++) delta[j] = beta * (vh[j] - kv[j]);

        float *o = a->out + h * dv;
        memset(o, 0, dv * sizeof(float));
        for (int i = 0; i < dk; i++)
            for (int j = 0; j < dv; j++) {
                S[i * dv + j] += kh_[i] * delta[j];
                o[j] += S[i * dv + j] * qh[i];
            }

        // gated RMSNorm: norm(o) * w * silu(z)
        const float *z = a->z + h * dv;
        ref_rmsnorm(o, o, a->norm, dv, a->eps);
        for (int j = 0; j < dv; j++) o[j] *= siluf(z[j]);
    }
}
