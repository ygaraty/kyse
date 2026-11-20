// Compute backend interface.
//
// model.c only wires ops together; every op of the forward pass goes through
// this vtable, so a GPU backend can run a whole token without the CPU waiting.
//
// Contract:
//   - Weights (matrices, f32 vectors) live inside the blob passed to init():
//     the mmapped model file. Matrices come in one of the WeightType formats.
//   - Activations and state (f32) live inside the memory returned by alloc(),
//     which is called once. The CPU only touches it after sync().
//   - Ops may run asynchronously, but always in the order they are issued.
//
// To add a backend, implement matvec (the hot op) and reuse the reference
// ops from ops_ref.c for the rest (see backend_accelerate.c). Move ops over
// one by one when you want to get rid of CPU round trips.
#ifndef BACKEND_H
#define BACKEND_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

// Matrix storage. A [rows, cols] matrix is row-major. The quantized formats
// keep one f16 scale per group of QUANT_GROUP consecutive weights of a row,
// stored after the data:
//   WT_BF16  rows*cols bf16
//   WT_Q8    rows*cols int8                                   + scales
//   WT_Q4    rows*cols/2 bytes, two weights per byte, low nibble first,
//            each stored as q + 8 (q in -8..7)                  + scales
// weight = q * scale
typedef enum { WT_BF16, WT_Q8, WT_Q4 } WeightType;
#define QUANT_GROUP 32

typedef struct {
    const void *data;
    WeightType type;
    int rows, cols;
} Mat;

static inline size_t mat_data_bytes(WeightType t, size_t rows, size_t cols) {
    return t == WT_BF16 ? rows * cols * 2 : t == WT_Q8 ? rows * cols : rows * cols / 2;
}
static inline size_t mat_bytes(WeightType t, size_t rows, size_t cols) {
    return mat_data_bytes(t, rows, cols) + (t == WT_BF16 ? 0 : rows * cols / QUANT_GROUP * 2);
}
static inline const uint16_t *mat_scales(Mat m) {
    return (const uint16_t *)((const char *)m.data + mat_data_bytes(m.type, m.rows, m.cols));
}

// one decode step of gated full attention for one layer
typedef struct {
    float *q;                      // [n_heads, 2*head_dim]: query | gate per head (q is clobbered)
    const float *k, *v;            // new key / value [n_kv_heads*head_dim], before norm and rope
    float *k_cache, *v_cache;      // this layer's cache [seq_len, n_kv_heads*head_dim]
    float *att;                    // scratch [n_heads, seq_len]
    const float *q_norm, *k_norm;  // [head_dim]
    float *out;                    // [n_heads*head_dim]
    int n_heads, n_kv_heads, head_dim, rot_dim, pos, seq_len;
    float eps, rope_theta;
} AttnArgs;

// one decode step of a gated deltanet layer
typedef struct {
    float *qkv;                    // [2*nk*dk + nv*dv] projections, replaced by conv + silu output
    const float *z, *a, *b;        // [nv*dv], [nv], [nv]
    float *conv_state;             // [conv_k-1, conv_dim], oldest first
    float *ssm_state;              // [nv, dk, dv]
    const float *conv_w, *A_log, *dt_bias, *norm;  // [conv_dim, conv_k], [nv], [nv], [dv]
    float *out;                    // [nv*dv]
    int nk, nv, dk, dv, conv_k;
    float eps;
} DeltaArgs;

typedef struct {
    const char *name;
    int    (*init)(void *weights, size_t size);  // 0 on success
    float *(*alloc)(size_t n_floats);            // zeroed
    void   (*embed)(float *out, Mat table, int token);  // out = row `token` of the table
    void   (*matvec)(float *out, Mat w, const float *x);  // out[rows] = W[rows, cols] x[cols]
    void   (*rmsnorm)(float *out, const float *x, const float *w, int n, float eps);
    void   (*add)(float *x, const float *y, int n);                                  // x += y
    void   (*swiglu)(float *out, const float *gate, const float *up, int n);         // silu(gate) * up
    void   (*attention)(const AttnArgs *a);
    void   (*deltanet)(const DeltaArgs *a);
    void   (*sync)(void);  // wait until everything issued so far is done
    void   (*shutdown)(void);
} Backend;

extern Backend backend_cpu;
#ifdef __APPLE__
extern Backend backend_accelerate;
extern Backend backend_metal;
#endif

// every compiled backend; the first one is the default
static Backend *const backends[] = {
#ifdef __APPLE__
    &backend_metal,
    &backend_accelerate,
#endif
    &backend_cpu,
};
#define N_BACKENDS (int)(sizeof(backends) / sizeof(backends[0]))

static inline Backend *backend_find(const char *name) {  // NULL if unknown
    for (int i = 0; i < N_BACKENDS; i++)
        if (strcmp(backends[i]->name, name) == 0) return backends[i];
    return NULL;
}

// reference CPU implementations (ops_ref.c)
void ref_embed(float *out, Mat table, int token);
void ref_matvec(float *out, Mat w, const float *x);
void ref_rmsnorm(float *out, const float *x, const float *w, int n, float eps);
void ref_add(float *x, const float *y, int n);
void ref_swiglu(float *out, const float *gate, const float *up, int n);
void ref_attention(const AttnArgs *a);
void ref_deltanet(const DeltaArgs *a);

#endif
