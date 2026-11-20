#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "model.h"
#include "util.h"

#define FILE_MAGIC   "QW35"
#define FILE_VERSION 2  // 1: all bf16; 2: adds the matrix and embedding formats
#define TENSOR_ALIGN 64  // bytes, must match export.py

static int max_i(int a, int b) { return a > b ? a : b; }
static int kv_dim(const Config *c)   { return c->n_kv_heads * c->head_dim; }
static int key_dim(const Config *c)  { return c->lin_k_heads * c->lin_k_dim; }
static int val_dim(const Config *c)  { return c->lin_v_heads * c->lin_v_dim; }
static int conv_dim(const Config *c) { return 2 * key_dim(c) + val_dim(c); }

static int count_full_layers(const Model *m) {
    int n = 0;
    for (int l = 0; l < m->c.n_layers; l++) n += m->full_attn[l];
    return n;
}

// ---------------------------------------------------------------------------
// loading

// walks the tensors of model.bin in the order export.py wrote them
typedef struct { char *base; size_t off; WeightType type; } Reader;

static void *next(Reader *r, size_t bytes) {
    r->off = (r->off + TENSOR_ALIGN - 1) & ~(size_t)(TENSOR_ALIGN - 1);
    void *p = r->base + r->off;
    r->off += bytes;
    return p;
}
static Mat mat(Reader *r, int rows, int cols) {
    if (r->type != WT_BF16 && cols % QUANT_GROUP) die("quantized matrix with %d columns", cols);
    return (Mat){ next(r, mat_bytes(r->type, rows, cols)), r->type, rows, cols };
}
static const float *vec(Reader *r, int n) { return next(r, (size_t)n * sizeof(float)); }

static void map_weights(Model *m, const char *start, WeightType embed_type) {
    const Config *c = &m->c;
    int D = c->dim, F = c->hidden_dim, hd = c->head_dim, nv = c->lin_v_heads;
    Reader r = { m->file, start - (char *)m->file, embed_type };

    m->embed = mat(&r, c->vocab_size, D);
    r.type = m->mat_type;
    m->layers = calloc(c->n_layers, sizeof(Layer));
    for (int l = 0; l < c->n_layers; l++) {
        Layer *L = &m->layers[l];
        L->ln1 = vec(&r, D);
        L->ln2 = vec(&r, D);
        if (m->full_attn[l]) {
            L->wq = mat(&r, 2 * c->n_heads * hd, D);
            L->wk = mat(&r, kv_dim(c), D);
            L->wv = mat(&r, kv_dim(c), D);
            L->wo = mat(&r, D, c->n_heads * hd);
            L->q_norm = vec(&r, hd);
            L->k_norm = vec(&r, hd);
        } else {
            L->w_qkv = mat(&r, conv_dim(c), D);
            L->w_z = mat(&r, val_dim(c), D);
            L->w_a = mat(&r, nv, D);
            L->w_b = mat(&r, nv, D);
            L->w_out = mat(&r, D, val_dim(c));
            L->conv = vec(&r, conv_dim(c) * c->conv_k);
            L->A_log = vec(&r, nv);
            L->dt_bias = vec(&r, nv);
            L->norm = vec(&r, c->lin_v_dim);
        }
        L->w_gate = mat(&r, F, D);
        L->w_up = mat(&r, F, D);
        L->w_down = mat(&r, D, F);
    }
    m->final_norm = vec(&r, D);
    if (r.off > m->file_size) die("model file is truncated");
}

// Lays out every activation and state buffer in one arena. With arena == NULL
// it only measures, so the caller can allocate the exact size first.
static size_t layout_state(Model *m, float *arena) {
    const Config *c = &m->c;
    State *s = &m->s;
    int n_full = count_full_layers(m), n_lin = c->n_layers - n_full;
    size_t off = 0;
#define CARVE(buf, n) do { if (arena) s->buf = arena + off; off += ((size_t)(n) + 15) & ~(size_t)15; } while (0)
    CARVE(x, c->dim);
    CARVE(xb, c->dim);
    CARVE(xb2, c->dim);
    CARVE(proj, max_i(2 * c->n_heads * c->head_dim, conv_dim(c)));
    CARVE(k, kv_dim(c));
    CARVE(v, kv_dim(c));
    CARVE(z, val_dim(c));
    CARVE(a, c->lin_v_heads);
    CARVE(b, c->lin_v_heads);
    CARVE(mix, max_i(c->n_heads * c->head_dim, val_dim(c)));
    CARVE(hb, c->hidden_dim);
    CARVE(hb2, c->hidden_dim);
    CARVE(logits, c->vocab_size);
    CARVE(att, (size_t)c->n_heads * m->seq_len);
    CARVE(k_cache, (size_t)n_full * m->seq_len * kv_dim(c));
    CARVE(v_cache, (size_t)n_full * m->seq_len * kv_dim(c));
    CARVE(conv_state, (size_t)n_lin * (c->conv_k - 1) * conv_dim(c));
    CARVE(ssm_state, (size_t)n_lin * c->lin_v_heads * c->lin_k_dim * c->lin_v_dim);
#undef CARVE
    return off;  // in floats; every buffer starts 64-byte aligned
}

void model_load(Model *m, const char *path, Backend *be, int seq_len) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) die("cannot open %s", path);
    struct stat st;
    fstat(fd, &st);
    size_t page = (size_t)getpagesize();
    m->file_size = ((size_t)st.st_size + page - 1) / page * page;  // page-rounded so a GPU can wrap it
    m->file = mmap(NULL, m->file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m->file == MAP_FAILED) die("cannot mmap %s", path);

    // header: magic, version, Config, one int32 per layer, then (version 2)
    // the int32 formats of the layer matrices and of the embedding table
    const char *p = m->file;
    int version = *(const int32_t *)(p + 4);
    if (memcmp(p, FILE_MAGIC, 4) != 0 || version < 1 || version > FILE_VERSION)
        die("%s is not a model.bin from export.py", path);
    memcpy(&m->c, p + 8, sizeof(Config));
    m->full_attn = (const int32_t *)(p + 8 + sizeof(Config));
    const int32_t *types = m->full_attn + m->c.n_layers;
    m->mat_type = version >= 2 ? types[0] : WT_BF16;
    WeightType embed_type = version >= 2 ? types[1] : WT_BF16;
    m->seq_len = seq_len;
    map_weights(m, (const char *)(version >= 2 ? types + 2 : types), embed_type);

    m->be = be;
    if (be->init(m->file, m->file_size) != 0) die("backend %s failed to initialize", be->name);
    float *arena = be->alloc(layout_state(m, NULL));
    if (!arena) die("out of memory");
    layout_state(m, arena);
}

// the recurrent state is conv_state followed by ssm_state (see layout_state)
size_t model_snapshot_size(const Model *m) {
    const Config *c = &m->c;
    int n_lin = c->n_layers - count_full_layers(m);
    return ((char *)m->s.ssm_state - (char *)m->s.conv_state)
         + (size_t)n_lin * c->lin_v_heads * c->lin_k_dim * c->lin_v_dim * sizeof(float);
}

void model_snapshot(Model *m, void *buf) {
    m->be->sync();
    memcpy(buf, m->s.conv_state, model_snapshot_size(m));
}

void model_restore(Model *m, const void *buf) {
    m->be->sync();
    memcpy(m->s.conv_state, buf, model_snapshot_size(m));
}

void model_reset(Model *m) {
    // the KV cache is simply overwritten from position 0, but the deltanet
    // state is accumulated, so it has to start from zero again
    m->be->sync();
    memset(m->s.conv_state, 0, model_snapshot_size(m));
}

void model_free(Model *m) {
    m->be->shutdown();
    free(m->layers);
    munmap(m->file, m->file_size);
}

// ---------------------------------------------------------------------------
// forward pass: pure wiring, every op runs on the backend

static void full_attention(Model *m, const Layer *L, int idx, int pos) {
    const Config *c = &m->c;
    State *s = &m->s;
    Backend *be = m->be;
    size_t cache = (size_t)idx * m->seq_len * kv_dim(c);

    be->matvec(s->proj, L->wq, s->xb);
    be->matvec(s->k, L->wk, s->xb);
    be->matvec(s->v, L->wv, s->xb);
    be->attention(&(AttnArgs){
        .q = s->proj, .k = s->k, .v = s->v,
        .k_cache = s->k_cache + cache, .v_cache = s->v_cache + cache, .att = s->att,
        .q_norm = L->q_norm, .k_norm = L->k_norm, .out = s->mix,
        .n_heads = c->n_heads, .n_kv_heads = c->n_kv_heads, .head_dim = c->head_dim,
        .rot_dim = c->rot_dim, .pos = pos, .seq_len = m->seq_len,
        .eps = c->eps, .rope_theta = c->rope_theta,
    });
    be->matvec(s->xb2, L->wo, s->mix);
}

static void gated_deltanet(Model *m, const Layer *L, int idx) {
    const Config *c = &m->c;
    State *s = &m->s;
    Backend *be = m->be;
    int nv = c->lin_v_heads;

    be->matvec(s->proj, L->w_qkv, s->xb);
    be->matvec(s->z, L->w_z, s->xb);
    be->matvec(s->a, L->w_a, s->xb);
    be->matvec(s->b, L->w_b, s->xb);
    be->deltanet(&(DeltaArgs){
        .qkv = s->proj, .z = s->z, .a = s->a, .b = s->b,
        .conv_state = s->conv_state + (size_t)idx * (c->conv_k - 1) * conv_dim(c),
        .ssm_state = s->ssm_state + (size_t)idx * nv * c->lin_k_dim * c->lin_v_dim,
        .conv_w = L->conv, .A_log = L->A_log, .dt_bias = L->dt_bias, .norm = L->norm, .out = s->mix,
        .nk = c->lin_k_heads, .nv = nv, .dk = c->lin_k_dim, .dv = c->lin_v_dim, .conv_k = c->conv_k,
        .eps = c->eps,
    });
    be->matvec(s->xb2, L->w_out, s->mix);
}

static void mlp(Model *m, const Layer *L) {
    const Config *c = &m->c;
    State *s = &m->s;
    Backend *be = m->be;
    be->matvec(s->hb, L->w_gate, s->xb);
    be->matvec(s->hb2, L->w_up, s->xb);
    be->swiglu(s->hb, s->hb, s->hb2, c->hidden_dim);
    be->matvec(s->xb2, L->w_down, s->hb);
}

float *model_forward(Model *m, int token, int pos, bool want_logits) {
    const Config *c = &m->c;
    State *s = &m->s;
    Backend *be = m->be;
    int D = c->dim, n_full = 0, n_lin = 0;

    be->embed(s->x, m->embed, token);
    for (int l = 0; l < c->n_layers; l++) {
        const Layer *L = &m->layers[l];
        be->rmsnorm(s->xb, s->x, L->ln1, D, c->eps);
        if (m->full_attn[l]) full_attention(m, L, n_full++, pos);
        else                 gated_deltanet(m, L, n_lin++);
        be->add(s->x, s->xb2, D);

        be->rmsnorm(s->xb, s->x, L->ln2, D, c->eps);
        mlp(m, L);
        be->add(s->x, s->xb2, D);
    }
    if (!want_logits) return NULL;

    be->rmsnorm(s->xb, s->x, m->final_norm, D, c->eps);
    be->matvec(s->logits, m->embed, s->xb);
    be->sync();
    return s->logits;
}
