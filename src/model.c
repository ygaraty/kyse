#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "model.h"
#include "util.h"

#define FILE_MAGIC   "QW35"
#define FILE_VERSION 4  // 2: matrix formats; 3: optional untied output head; 4: format per matrix
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

// walks the tensors of model.bin in the order export.py wrote them; a matrix
// takes its format from the per-matrix list (version 4) or else from `type`
typedef struct { char *base; size_t off, size; WeightType type; const int32_t *types; int max_rank; } Reader;

static void *next(Reader *r, size_t bytes) {
    r->off = (r->off + TENSOR_ALIGN - 1) & ~(size_t)(TENSOR_ALIGN - 1);
    if (r->off > r->size || bytes > r->size - r->off) die("model file is truncated");
    void *p = r->base + r->off;
    r->off += bytes;
    return p;
}
// a format list entry is the WeightType plus 256 * the rank of the low-rank
// correction, which follows the matrix data
static Mat mat(Reader *r, int rows, int cols) {
    int32_t entry = r->types ? *r->types++ : (int32_t)r->type;
    WeightType t = entry & 255;
    int rank = entry >> 8;
    if (t != WT_BF16 && cols % QUANT_GROUP) die("quantized matrix with %d columns", cols);
    if (rank % 4 || rank > rows || rank > cols) die("invalid low-rank correction (rank %d)", rank);
    Mat w = { next(r, mat_bytes(t, rows, cols)), t, rows, cols, rank, NULL };
    if (rank) w.lr = next(r, (size_t)rank * (rows + cols) * 2);
    r->max_rank = max_i(r->max_rank, rank);
    return w;
}
static const float *vec(Reader *r, int n) { return next(r, (size_t)n * sizeof(float)); }

// matrices in model.bin: the embedding, 4 + 3 (mlp) per attention layer,
// 5 + 3 per deltanet layer, and the output head when it is untied
static int count_mats(const Model *m, bool untied) {
    int n_full = count_full_layers(m);
    return 1 + 7 * n_full + 8 * (m->c.n_layers - n_full) + untied;
}

static void map_weights(Model *m, const char *start, WeightType embed_type, bool untied, const int32_t *types) {
    const Config *c = &m->c;
    int D = c->dim, F = c->hidden_dim, hd = c->head_dim, nv = c->lin_v_heads;
    Reader r = { m->file, start - (char *)m->file, m->data_size, embed_type, types, 0 };

    m->embed = mat(&r, c->vocab_size, D);
    r.type = m->mat_type;
    m->layers = calloc(c->n_layers, sizeof(Layer));
    if (!m->layers) die("out of memory");
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
    r.type = embed_type;
    m->head = untied ? mat(&r, c->vocab_size, D) : m->embed;
    if (m->embed.rank || m->head.rank) die("the embedding and output head cannot have a low-rank correction");
    m->max_rank = r.max_rank;
    if (r.off > m->file_size) die("model file is truncated");
}

// Lays out every activation and state buffer in one arena. With arena == NULL
// it only measures, so the caller can allocate the exact size first.
static size_t layout_state(Model *m, float *arena) {
    const Config *c = &m->c;
    State *s = &m->s;
    int n_full = count_full_layers(m), n_lin = c->n_layers - n_full;
    size_t off = 0;
#define CARVE(buf, n) do { \
        if (arena) s->buf = (typeof(s->buf))(arena + off); \
        off += ((size_t)(n) * sizeof(*s->buf) / sizeof(float) + 15) & ~(size_t)15; \
    } while (0)
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
    CARVE(lr_mid, m->max_rank);
    CARVE(lr_out, m->max_rank ? max_i(max_i(2 * c->n_heads * c->head_dim, conv_dim(c)), max_i(c->hidden_dim, c->dim)) : 0);
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
    if (fstat(fd, &st) != 0 || st.st_size < 68) {
        close(fd);
        die("invalid model file: %s", path);
    }
    m->data_size = (size_t)st.st_size;
    size_t page = (size_t)getpagesize();
    m->file_size = ((size_t)st.st_size + page - 1) / page * page;  // page-rounded so a GPU can wrap it
    m->file = mmap(NULL, m->file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m->file == MAP_FAILED) die("cannot mmap %s", path);

    // header: magic, version, Config, one int32 per layer, then (version 2)
    // the int32 formats of the layer matrices and of the embedding table,
    // (version 3) the untied-head flag, (version 4) the number of matrices
    // and the format of each one
    const char *p = m->file;
    int version = *(const int32_t *)(p + 4);
    if (memcmp(p, FILE_MAGIC, 4) != 0 || version < 1 || version > FILE_VERSION)
        die("%s is not a model.bin from export.py", path);
    memcpy(&m->c, p + 8, sizeof(Config));
    const Config *c = &m->c;
    if (seq_len <= 0 || c->dim <= 0 || c->hidden_dim <= 0 || c->vocab_size <= 0 ||
        c->n_layers <= 0 || c->n_heads <= 0 || c->n_kv_heads <= 0 || c->head_dim <= 0 ||
        c->lin_k_heads <= 0 || c->lin_v_heads <= 0 || c->lin_k_dim <= 0 ||
        c->lin_v_dim <= 0 || c->conv_k < 2 || c->n_heads % c->n_kv_heads ||
        c->lin_v_heads % c->lin_k_heads || c->rot_dim < 0 ||
        c->rot_dim > c->head_dim || c->rot_dim % 2)
        die("invalid model configuration");
    size_t header_size = 68 + (size_t)c->n_layers * 4 + (version >= 3 ? 12 : version >= 2 ? 8 : 0);
    if (header_size > m->data_size) die("model header is truncated");
    m->full_attn = (const int32_t *)(p + 8 + sizeof(Config));
    for (int l = 0; l < c->n_layers; l++)
        if (m->full_attn[l] != 0 && m->full_attn[l] != 1) die("invalid layer type");
    const int32_t *types = m->full_attn + m->c.n_layers;
    m->mat_type = version >= 2 ? types[0] : WT_BF16;
    WeightType embed_type = version >= 2 ? types[1] : WT_BF16;
    if (m->mat_type < WT_BF16 || m->mat_type > WT_Q4 ||
        embed_type < WT_BF16 || embed_type > WT_Q4 ||
        (version >= 3 && types[2] != 0 && types[2] != 1))
        die("invalid model weight format");
    m->seq_len = seq_len;
    bool untied = version >= 3 && types[2];
    const int32_t *mat_types = NULL, *start = types + (version >= 3 ? 3 : version >= 2 ? 2 : 0);
    if (version >= 4) {
        int n = count_mats(m, untied);
        if (header_size + 4 + 4 * (size_t)n > m->data_size || start[0] != n) die("invalid matrix format list");
        mat_types = start + 1;
        for (int i = 0; i < n; i++)
            if ((mat_types[i] & 255) > WT_Q4 || mat_types[i] < 0) die("invalid model weight format");
        start = mat_types + n;
    }
    map_weights(m, (const char *)start, embed_type, untied, mat_types);

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

// out = W x, plus the low-rank correction if W has one
static void linear(Model *m, float *out, Mat w, const float *x) {
    m->be->matvec(out, w, x);
    if (!w.rank) return;
    Mat down = { w.lr, WT_BF16, w.rank, w.cols, 0, NULL };
    Mat up = { w.lr + (size_t)w.rank * w.cols, WT_BF16, w.rows, w.rank, 0, NULL };
    m->be->matvec(m->s.lr_mid, down, x);
    m->be->matvec(m->s.lr_out, up, m->s.lr_mid);
    m->be->add(out, m->s.lr_out, w.rows);
}

static void full_attention(Model *m, const Layer *L, int idx, int pos) {
    const Config *c = &m->c;
    State *s = &m->s;
    Backend *be = m->be;
    size_t cache = (size_t)idx * m->seq_len * kv_dim(c);

    linear(m, s->proj, L->wq, s->xb);
    linear(m, s->k, L->wk, s->xb);
    linear(m, s->v, L->wv, s->xb);
    be->attention(&(AttnArgs){
        .q = s->proj, .k = s->k, .v = s->v,
        .k_cache = s->k_cache + cache, .v_cache = s->v_cache + cache, .att = s->att,
        .q_norm = L->q_norm, .k_norm = L->k_norm, .out = s->mix,
        .n_heads = c->n_heads, .n_kv_heads = c->n_kv_heads, .head_dim = c->head_dim,
        .rot_dim = c->rot_dim, .pos = pos, .seq_len = m->seq_len,
        .eps = c->eps, .rope_theta = c->rope_theta,
    });
    linear(m, s->xb2, L->wo, s->mix);
}

static void gated_deltanet(Model *m, const Layer *L, int idx) {
    const Config *c = &m->c;
    State *s = &m->s;
    Backend *be = m->be;
    int nv = c->lin_v_heads;

    linear(m, s->proj, L->w_qkv, s->xb);
    linear(m, s->z, L->w_z, s->xb);
    linear(m, s->a, L->w_a, s->xb);
    linear(m, s->b, L->w_b, s->xb);
    be->deltanet(&(DeltaArgs){
        .qkv = s->proj, .z = s->z, .a = s->a, .b = s->b,
        .conv_state = s->conv_state + (size_t)idx * (c->conv_k - 1) * conv_dim(c),
        .ssm_state = s->ssm_state + (size_t)idx * nv * c->lin_k_dim * c->lin_v_dim,
        .conv_w = L->conv, .A_log = L->A_log, .dt_bias = L->dt_bias, .norm = L->norm, .out = s->mix,
        .nk = c->lin_k_heads, .nv = nv, .dk = c->lin_k_dim, .dv = c->lin_v_dim, .conv_k = c->conv_k,
        .eps = c->eps,
    });
    linear(m, s->xb2, L->w_out, s->mix);
}

static void mlp(Model *m, const Layer *L) {
    const Config *c = &m->c;
    State *s = &m->s;
    Backend *be = m->be;
    linear(m, s->hb, L->w_gate, s->xb);
    linear(m, s->hb2, L->w_up, s->xb);
    be->swiglu(s->hb, s->hb, s->hb2, c->hidden_dim);
    linear(m, s->xb2, L->w_down, s->hb);
}

float *model_forward(Model *m, int token, int pos, bool want_logits) {
    const Config *c = &m->c;
    State *s = &m->s;
    Backend *be = m->be;
    int D = c->dim, n_full = 0, n_lin = 0;

    if (token < 0 || token >= c->vocab_size || pos < 0 || pos >= m->seq_len)
        die("token or context position out of range");
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
    linear(m, s->logits, m->head, s->xb);
    be->sync();
    return s->logits;
}
