// Apple Accelerate backend (CPU, uses the AMX matrix units through BLAS).
//
// BLAS has no bf16 type, so each bf16 matrix is expanded to f32 the first time
// it is used (2x memory). Quantized matrices fall back to the reference op.
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include "backend.h"
#include "util.h"

#define MAX_MATS 1024

static struct { const void *data; float *f32; } cache[MAX_MATS];
static int n_cached;
static float *arena;

static int acc_init(void *weights, size_t size) { (void)weights; (void)size; return 0; }

static float *acc_alloc(size_t n) { return arena = calloc(n, sizeof(float)); }

static const float *as_f32(Mat w) {
    for (int i = 0; i < n_cached; i++)
        if (cache[i].data == w.data) return cache[i].f32;
    if (n_cached == MAX_MATS) die("accelerate: too many matrices");
    size_t n = (size_t)w.rows * w.cols;
    float *f = malloc(n * sizeof(float));
    if (!f) die("out of memory");
    for (size_t i = 0; i < n; i++) f[i] = bf16_to_f32(((const uint16_t *)w.data)[i]);
    cache[n_cached++] = (typeof(cache[0])){ w.data, f };
    return f;
}

static void acc_matvec(float *out, Mat w, const float *x) {
    if (w.type != WT_BF16) return ref_matvec(out, w, x);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, w.rows, w.cols, 1.0f, as_f32(w), w.cols, x, 1, 0.0f, out, 1);
}

static void acc_sync(void) {}

static void acc_shutdown(void) {
    for (int i = 0; i < n_cached; i++) free(cache[i].f32);
    n_cached = 0;
    free(arena);
}

Backend backend_accelerate = {
    "accelerate", acc_init, acc_alloc, ref_embed, acc_matvec, ref_rmsnorm, ref_add, ref_swiglu,
    ref_attention, ref_deltanet, acc_sync, acc_shutdown,
};
