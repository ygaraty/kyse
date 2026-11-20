// Apple Metal backend.
//
// Unified memory means no copies: the mmapped model file is wrapped as one
// MTLBuffer and all activations/state live in another (the arena), so the
// pointers model.c passes around just become (buffer, offset) pairs.
//
// Every op appends a dispatch to the current command buffer. The CPU only
// waits in sync(), i.e. once per generated token.
#import <Metal/Metal.h>
#include "backend.h"
#include "util.h"

static const char kernel_source[] = {
#embed "kernels.metal"
    , 0
};

#define MATVEC_ROWS_PER_GROUP 8      // must match the matvec kernels in kernels.metal
#define MAX_DISPATCHES_PER_CMD 1024  // keeps command buffers bounded during long prefills

static id<MTLDevice> device;
static id<MTLCommandQueue> queue;
static id<MTLBuffer> weights_buf, arena_buf;
static id<MTLCommandBuffer> cmd, last_cmd;
static id<MTLComputeCommandEncoder> enc;
static int n_dispatches;

static struct {
    id<MTLComputePipelineState> embed, matvec[3], rmsnorm, add, swiglu;  // matvec by WeightType
    id<MTLComputePipelineState> attn_prep, attn_decode, delta_conv, delta_step;
} k;

// ---------------------------------------------------------------------------
// setup

static id<MTLComputePipelineState> make_pipeline(id<MTLLibrary> lib, const char *name) {
    NSError *err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:@(name)];
    id<MTLComputePipelineState> p = fn ? [device newComputePipelineStateWithFunction:fn error:&err] : nil;
    if (!p) die("metal: cannot build kernel %s: %s", name, err.localizedDescription.UTF8String);
    return p;
}

static int metal_init(void *weights, size_t size) {
    device = MTLCreateSystemDefaultDevice();
    if (!device) return 1;
    queue = [device newCommandQueue];

    NSError *err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:@(kernel_source) options:nil error:&err];
    if (!lib) die("metal: kernel compile failed: %s", err.localizedDescription.UTF8String);
    k.embed = make_pipeline(lib, "embed");
    k.matvec[WT_BF16] = make_pipeline(lib, "matvec_bf16");
    k.matvec[WT_Q8] = make_pipeline(lib, "matvec_q8");
    k.matvec[WT_Q4] = make_pipeline(lib, "matvec_q4");
    k.rmsnorm = make_pipeline(lib, "rmsnorm");
    k.add = make_pipeline(lib, "add");
    k.swiglu = make_pipeline(lib, "swiglu");
    k.attn_prep = make_pipeline(lib, "attn_prep");
    k.attn_decode = make_pipeline(lib, "attn_decode");
    k.delta_conv = make_pipeline(lib, "delta_conv");
    k.delta_step = make_pipeline(lib, "delta_step");

    weights_buf = [device newBufferWithBytesNoCopy:weights length:size
                                           options:MTLResourceStorageModeShared deallocator:nil];
    if (!weights_buf) die("metal: cannot wrap the %zu byte weight file", size);
    fprintf(stderr, "metal: %s\n", device.name.UTF8String);
    return 0;
}

static float *metal_alloc(size_t n) {
    arena_buf = [device newBufferWithLength:n * sizeof(float) options:MTLResourceStorageModeShared];
    if (!arena_buf) return NULL;
    memset(arena_buf.contents, 0, n * sizeof(float));
    return arena_buf.contents;
}

// ---------------------------------------------------------------------------
// encoding helpers

// commits the current command buffer without waiting; the queue keeps order
static void flush(void) {
    if (!cmd) return;
    [enc endEncoding];
    [cmd commit];
    last_cmd = cmd;
    cmd = nil;
    enc = nil;
    n_dispatches = 0;
}

static void begin(id<MTLComputePipelineState> pipeline) {
    if (n_dispatches >= MAX_DISPATCHES_PER_CMD) flush();
    if (!cmd) {
        cmd = [queue commandBufferWithUnretainedReferences];
        enc = [cmd computeCommandEncoder];
    }
    n_dispatches++;
    [enc setComputePipelineState:pipeline];
}

// binds a pointer that lives in the weights or in the arena
static void bind(const void *ptr, int index) {
    const char *p = ptr, *w = weights_buf.contents, *a = arena_buf.contents;
    if (p >= w && p < w + weights_buf.length) [enc setBuffer:weights_buf offset:p - w atIndex:index];
    else if (p >= a && p < a + arena_buf.length) [enc setBuffer:arena_buf offset:p - a atIndex:index];
    else die("metal: pointer %p is not in a backend buffer", ptr);
}

static void bind_bytes(const void *data, size_t size, int index) {
    [enc setBytes:data length:size atIndex:index];
}

// n threads total, for elementwise kernels
static void run_threads(id<MTLComputePipelineState> p, int n) {
    NSUInteger group = MIN(p.maxTotalThreadsPerThreadgroup, 256);
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
}

static void run_groups(int n_groups, int threads_per_group) {
    [enc dispatchThreadgroups:MTLSizeMake(n_groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
}

// ---------------------------------------------------------------------------
// ops

// quantized matrices bind their scales at index 5 (bf16 ones bind the data as a dummy)
static void bind_scales(Mat m, int index) {
    bind(m.type == WT_BF16 ? m.data : mat_scales(m), index);
}

static void metal_embed(float *out, Mat table, int token) {
    begin(k.embed);
    bind(table.data, 0);
    bind(out, 1);
    bind_bytes(&token, sizeof(int), 2);
    bind_bytes(&table.cols, sizeof(int), 3);
    bind_bytes(&table.type, sizeof(int), 4);
    bind_scales(table, 5);
    run_threads(k.embed, table.cols);
}

static void metal_matvec(float *out, Mat w, const float *x) {
    if (w.cols % 4 || (w.type != WT_BF16 && w.cols % 16)) die("metal matvec: unsupported column count %d", w.cols);
    begin(k.matvec[w.type]);
    bind(w.data, 0);
    bind(x, 1);
    bind(out, 2);
    bind_bytes(&w.cols, sizeof(int), 3);
    bind_bytes(&w.rows, sizeof(int), 4);
    bind_scales(w, 5);
    run_groups((w.rows + MATVEC_ROWS_PER_GROUP - 1) / MATVEC_ROWS_PER_GROUP, MATVEC_ROWS_PER_GROUP * 32);
}

static void metal_rmsnorm(float *out, const float *x, const float *w, int n, float eps) {
    begin(k.rmsnorm);
    bind(x, 0);
    bind(w, 1);
    bind(out, 2);
    bind_bytes(&n, sizeof(int), 3);
    bind_bytes(&eps, sizeof(float), 4);
    run_groups(1, (int)k.rmsnorm.maxTotalThreadsPerThreadgroup);
}

static void metal_add(float *x, const float *y, int n) {
    begin(k.add);
    bind(x, 0);
    bind(y, 1);
    bind_bytes(&n, sizeof(int), 2);
    run_threads(k.add, n);
}

static void metal_swiglu(float *out, const float *gate, const float *up, int n) {
    begin(k.swiglu);
    bind(out, 0);
    bind(gate, 1);
    bind(up, 2);
    bind_bytes(&n, sizeof(int), 3);
    run_threads(k.swiglu, n);
}

// binding layout shared by attn_prep and attn_decode (see kernels.metal)
static void bind_attention(const AttnArgs *a) {
    struct { int n_heads, n_kv_heads, head_dim, rot_dim, pos, seq_len; float eps, rope_theta; } params = {
        a->n_heads, a->n_kv_heads, a->head_dim, a->rot_dim, a->pos, a->seq_len, a->eps, a->rope_theta,
    };
    bind(a->q, 0);
    bind(a->k, 1);
    bind(a->v, 2);
    bind(a->k_cache, 3);
    bind(a->v_cache, 4);
    bind(a->att, 5);
    bind(a->q_norm, 6);
    bind(a->k_norm, 7);
    bind(a->out, 8);
    bind_bytes(&params, sizeof(params), 9);
}

static void metal_attention(const AttnArgs *a) {
    if (a->head_dim > 1024) die("metal attention: head_dim > 1024");
    begin(k.attn_prep);
    bind_attention(a);
    run_groups(a->n_heads + a->n_kv_heads, a->head_dim);

    begin(k.attn_decode);
    bind_attention(a);
    run_groups(a->n_heads, 256);
}

// binding layout shared by delta_conv and delta_step (see kernels.metal)
static void bind_deltanet(const DeltaArgs *a) {
    struct { int nk, nv, dk, dv, conv_k; float eps; } params = { a->nk, a->nv, a->dk, a->dv, a->conv_k, a->eps };
    bind(a->qkv, 0);
    bind(a->z, 1);
    bind(a->a, 2);
    bind(a->b, 3);
    bind(a->conv_state, 4);
    bind(a->ssm_state, 5);
    bind(a->conv_w, 6);
    bind(a->A_log, 7);
    bind(a->dt_bias, 8);
    bind(a->norm, 9);
    bind(a->out, 10);
    bind_bytes(&params, sizeof(params), 11);
}

static void metal_deltanet(const DeltaArgs *a) {
    if (a->dk > 1024 || a->dv > 1024) die("metal deltanet: head dim > 1024");
    begin(k.delta_conv);
    bind_deltanet(a);
    run_threads(k.delta_conv, 2 * a->nk * a->dk + a->nv * a->dv);

    begin(k.delta_step);
    bind_deltanet(a);
    run_groups(a->nv, a->dv);
}

static void metal_sync(void) {
    flush();
    [last_cmd waitUntilCompleted];
    last_cmd = nil;
}

static void metal_shutdown(void) {
    metal_sync();
    arena_buf = nil;
    weights_buf = nil;
    queue = nil;
    device = nil;
}

Backend backend_metal = {
    "metal", metal_init, metal_alloc, metal_embed, metal_matvec, metal_rmsnorm, metal_add, metal_swiglu,
    metal_attention, metal_deltanet, metal_sync, metal_shutdown,
};
