// Small evaluation tools.
//
//   bench bpb <model.bin> <text> [-k chunks] [-l bytes] [-b backend] [-z tokenizer]
//       Language-modeling quality: bits per byte on a text file. The text is cut
//       into chunks of about -l bytes (at line ends); each chunk starts from a
//       fresh state with <|endoftext|> as context and every token is scored.
//       Bits per byte, unlike perplexity, does not depend on the vocabulary, so
//       models with different tokenizers can be compared.
//
//   bench mc <model.bin> <items.txt> [-k items] [-b backend] [-z tokenizer]
//       Multiple choice (ARC, HellaSwag, ...): each ending is scored by its
//       log-likelihood after the context, the highest one is the answer.
//       Reports acc (total log-likelihood) and acc_norm (per byte of the
//       ending), as lm-evaluation-harness does. Item format: "label n" on a
//       line, then the context and the n endings, each as "len\n<bytes>\n".
//
//   bench calib <model.bin> <text> <stats.bin> [-k chunks] [-l bytes] [-b backend] [-z tokenizer]
//       Activation statistics for `shrink.py --awq`: the mean |x| of every
//       input channel of every layer matrix, over -k chunks of about -l bytes
//       spread evenly through the text. Writes, for each layer matrix in
//       model.bin order: int32 cols, float mean_abs[cols].
//
//   bench count <tokenizer.bin> <text>...
//       How often each token id occurs when the files are tokenized.
//       Prints "id count" lines (non-zero only); shrink.py reads them.
#include <math.h>
#include <string.h>
#include "model.h"
#include "tokenizer.h"
#include "util.h"

static char *read_file(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open %s", path);
    fseek(f, 0, SEEK_END);
    *size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc(*size + 1);
    if (fread(buf, 1, *size, f) != *size) die("cannot read %s", path);
    buf[*size] = '\0';
    fclose(f);
    return buf;
}

// length of the chunk starting at text: at least `target` bytes, ending after a newline
static size_t chunk_len(const char *text, size_t target) {
    size_t n = strlen(text);
    if (n <= target) return n;
    const char *nl = strchr(text + target, '\n');
    return nl ? (size_t)(nl - text + 1) : n;
}

static Backend *find_backend(const char *name) {
    Backend *be = backend_find(name);
    if (!be) die("unknown backend '%s'", name);
    return be;
}

// log p(target) from raw logits; *argmax gets the most likely token
static double log_prob(const float *logits, int n, int target, int *argmax) {
    float max = logits[0];
    int best = 0;
    for (int j = 1; j < n; j++)
        if (logits[j] > max) max = logits[best = j];
    double sum = 0.0;
    for (int j = 0; j < n; j++) sum += exp(logits[j] - max);
    *argmax = best;
    return logits[target] - max - log(sum);
}

// options shared by bpb and mc: -k <limit> -l <bytes> -b <backend> -z <tokenizer>
typedef struct { int limit; size_t chunk_bytes; Backend *be; const char *tokenizer; } Options;

static Options parse_options(int argc, char **argv, int limit) {
    Options o = { limit, 4096, backends[0], NULL };
    for (int i = 0; i + 1 < argc; i += 2) {
        if (strcmp(argv[i], "-k") == 0) o.limit = atoi(argv[i + 1]);
        else if (strcmp(argv[i], "-l") == 0) o.chunk_bytes = atol(argv[i + 1]);
        else if (strcmp(argv[i], "-b") == 0) o.be = find_backend(argv[i + 1]);
        else if (strcmp(argv[i], "-z") == 0) o.tokenizer = argv[i + 1];
        else die("unknown option %s", argv[i]);
    }
    return o;
}

// loads the model and its tokenizer; returns the <|endoftext|> id, used as the start token
static int load(Model *m, Tokenizer *tok, const char *model_path, const Options *o, int seq_len) {
    char *default_tokenizer = sibling_path(model_path, "tokenizer.bin");
    tokenizer_load(tok, o->tokenizer ? o->tokenizer : default_tokenizer);
    free(default_tokenizer);
    int start = tokenizer_special(tok, "<|endoftext|>");
    if (start < 0) die("tokenizer has no <|endoftext|>");
    model_load(m, model_path, o->be, seq_len);
    if (tok->vocab_size != m->c.vocab_size) die("tokenizer and model have different vocab sizes");
    return start;
}

static int bpb(int argc, char **argv) {
    if (argc < 2) die("usage: bench bpb <model.bin> <text> [-k chunks] [-l bytes] [-b backend] [-z tokenizer]");
    const char *model_path = argv[0], *text_path = argv[1];
    Options o = parse_options(argc - 2, argv + 2, 8);
    size_t size;
    char *text = read_file(text_path, &size);
    Model m;
    Tokenizer tok;
    int start = load(&m, &tok, model_path, &o, 2 * (int)o.chunk_bytes + 16);  // a token is at least one byte
    int V = m.c.vocab_size;

    double nll = 0.0, t_start = now();  // nats
    long bytes = 0, tokens = 0, correct = 0;
    int chunks = 0;
    int *ids = malloc((size + 2) * sizeof(int));
    for (size_t off = 0; off < size && chunks < o.limit; chunks++) {
        size_t len = chunk_len(text + off, o.chunk_bytes);
        char saved = text[off + len];
        text[off + len] = '\0';
        ids[0] = start;
        int n = 1 + tokenizer_encode(&tok, text + off, ids + 1);
        text[off + len] = saved;
        if (n > m.seq_len) die("chunk does not fit the context");

        model_reset(&m);
        for (int i = 0; i + 1 < n; i++) {
            int best;
            nll -= log_prob(model_forward(&m, ids[i], i, true), V, ids[i + 1], &best);
            correct += best == ids[i + 1];
        }
        tokens += n - 1;
        bytes += len;
        off += len;
        fprintf(stderr, "\rchunk %d/%d: %.4f bits/byte so far", chunks + 1, o.limit, nll / M_LN2 / bytes);
    }
    double secs = now() - t_start;
    fprintf(stderr, "\n");
    printf("%s: %.4f bits/byte | %.3f bits/token | top-1 %.1f%% | %ld tokens for %ld bytes (%.2f bytes/token) | %.1f ms/token [%s]\n",
           model_path, nll / M_LN2 / bytes, nll / M_LN2 / tokens, 100.0 * correct / tokens, tokens, bytes,
           (double)bytes / tokens, 1000.0 * secs / tokens, o.be->name);

    free(ids);
    free(text);
    tokenizer_free(&tok);
    model_free(&m);
    return 0;
}

// reads one length-prefixed field of an mc item file; NULL at the end of the file
static char *read_field(FILE *f) {
    int len;
    if (fscanf(f, "%d", &len) != 1) return NULL;
    fgetc(f);  // the newline after the length
    char *s = malloc(len + 1);
    if (fread(s, 1, len, f) != (size_t)len) die("truncated item file");
    s[len] = '\0';
    fgetc(f);
    return s;
}

static int mc(int argc, char **argv) {
    if (argc < 2) die("usage: bench mc <model.bin> <items.txt> [-k items] [-b backend] [-z tokenizer]");
    const char *model_path = argv[0];
    Options o = parse_options(argc - 2, argv + 2, 1 << 30);
    FILE *f = fopen(argv[1], "rb");
    if (!f) die("cannot open %s", argv[1]);
    Model m;
    Tokenizer tok;
    int start = load(&m, &tok, model_path, &o, 2048);
    int V = m.c.vocab_size;
    void *snapshot = malloc(model_snapshot_size(&m));

    int items = 0, correct = 0, correct_norm = 0;
    double t_start = now();
    int label, n_endings;
    while (items < o.limit && fscanf(f, "%d %d", &label, &n_endings) == 2) {
        char *ctx = read_field(f);
        int *ids = malloc((strlen(ctx) + 2) * sizeof(int));
        ids[0] = start;
        int n_ctx = 1 + tokenizer_encode(&tok, ctx, ids + 1);
        // run the context up to its last token once, then branch for each ending
        model_reset(&m);
        for (int i = 0; i < n_ctx - 1; i++) model_forward(&m, ids[i], i, false);
        model_snapshot(&m, snapshot);

        double best = -INFINITY, best_norm = -INFINITY;
        int pick = -1, pick_norm = -1;
        for (int e = 0; e < n_endings; e++) {
            char *ending = read_field(f);
            int *end_ids = malloc((strlen(ending) + 1) * sizeof(int));
            int n_end = tokenizer_encode(&tok, ending, end_ids);
            if (n_ctx + n_end > m.seq_len) die("item does not fit the context");
            model_restore(&m, snapshot);
            double lp = 0.0;
            int prev = ids[n_ctx - 1], argmax;
            for (int i = 0; i < n_end; i++) {
                lp += log_prob(model_forward(&m, prev, n_ctx - 1 + i, true), V, end_ids[i], &argmax);
                prev = end_ids[i];
            }
            double norm = lp / strlen(ending);
            if (lp > best) { best = lp; pick = e; }
            if (norm > best_norm) { best_norm = norm; pick_norm = e; }
            free(end_ids);
            free(ending);
        }
        correct += pick == label;
        correct_norm += pick_norm == label;
        items++;
        fprintf(stderr, "\ritem %d: acc %.1f%% acc_norm %.1f%%", items, 100.0 * correct / items, 100.0 * correct_norm / items);
        free(ids);
        free(ctx);
    }
    fprintf(stderr, "\n");
    // binomial standard error of an accuracy p over n items
    double p = (double)correct / items, pn = (double)correct_norm / items;
    printf("%s on %s: acc %.1f%% (±%.1f) | acc_norm %.1f%% (±%.1f) | %d items | %.1fs [%s]\n",
           model_path, argv[1], 100 * p, 100 * sqrt(p * (1 - p) / items), 100 * pn, 100 * sqrt(pn * (1 - pn) / items),
           items, now() - t_start, o.be->name);
    fclose(f);
    free(snapshot);
    tokenizer_free(&tok);
    model_free(&m);
    return 0;
}

// calib wraps the chosen backend: its matvec records the input before running it
static Backend *calib_inner;
static struct { const void *data; int cols; long n; double *abs; } stats[1024];
static int n_stats;

static void calib_matvec(float *out, Mat w, const float *x) {
    for (int i = 0; i < n_stats; i++)
        if (stats[i].data == w.data) {
            calib_inner->sync();  // x is complete
            for (int j = 0; j < w.cols; j++) stats[i].abs[j] += fabsf(x[j]);
            stats[i].n++;
            break;
        }
    calib_inner->matvec(out, w, x);
}

static void track(Mat w) {
    if (n_stats == 1024) die("too many matrices");
    stats[n_stats++] = (typeof(stats[0])){ w.data, w.cols, 0, calloc(w.cols, sizeof(double)) };
}

static int calib(int argc, char **argv) {
    if (argc < 3) die("usage: bench calib <model.bin> <text> <stats.bin> [-k chunks] [-l bytes] [-b backend] [-z tokenizer]");
    const char *model_path = argv[0];
    Options o = parse_options(argc - 3, argv + 3, 32);
    calib_inner = o.be;
    Backend wrapper = *o.be;
    wrapper.matvec = calib_matvec;
    o.be = &wrapper;
    size_t size;
    char *text = read_file(argv[1], &size);
    Model m;
    Tokenizer tok;
    int start = load(&m, &tok, model_path, &o, 2 * (int)o.chunk_bytes + 16);

    // the layer matrices in the order map_weights() reads them
    for (int l = 0; l < m.c.n_layers; l++) {
        const Layer *L = &m.layers[l];
        if (m.full_attn[l]) {
            track(L->wq), track(L->wk), track(L->wv), track(L->wo);
        } else {
            track(L->w_qkv), track(L->w_z), track(L->w_a), track(L->w_b), track(L->w_out);
        }
        track(L->w_gate), track(L->w_up), track(L->w_down);
    }

    int *ids = malloc((size + 2) * sizeof(int));
    long tokens = 0;
    for (int c = 0; c < o.limit; c++) {
        // chunk c starts at the first line after c/limit of the text
        const char *p = text + size / o.limit * c, *nl = c ? strchr(p, '\n') : NULL;
        size_t off = c ? (nl ? (size_t)(nl + 1 - text) : size) : 0;
        size_t len = chunk_len(text + off, o.chunk_bytes);
        char saved = text[off + len];
        text[off + len] = '\0';
        ids[0] = start;
        int n = 1 + tokenizer_encode(&tok, text + off, ids + 1);
        text[off + len] = saved;
        if (n > m.seq_len) die("chunk does not fit the context");
        model_reset(&m);
        for (int i = 0; i < n; i++) model_forward(&m, ids[i], i, false);
        tokens += n;
        fprintf(stderr, "\rchunk %d/%d, %ld tokens", c + 1, o.limit, tokens);
    }
    m.be->sync();
    fprintf(stderr, "\n");

    FILE *f = fopen(argv[2], "wb");
    if (!f) die("cannot write %s", argv[2]);
    for (int i = 0; i < n_stats; i++) {
        float *row = malloc(stats[i].cols * sizeof(float));
        fwrite(&stats[i].cols, sizeof(int), 1, f);
        for (int j = 0; j < stats[i].cols; j++) row[j] = stats[i].abs[j] / stats[i].n;
        fwrite(row, sizeof(float), stats[i].cols, f);
        free(row);
        free(stats[i].abs);
    }
    fclose(f);
    printf("wrote %s: %d matrices, %ld tokens\n", argv[2], n_stats, tokens);
    free(ids);
    free(text);
    tokenizer_free(&tok);
    model_free(&m);
    return 0;
}

static int count(int argc, char **argv) {
    if (argc < 2) die("usage: bench count <tokenizer.bin> <text>...");
    Tokenizer tok;
    tokenizer_load(&tok, argv[0]);
    long *counts = calloc(tok.vocab_size, sizeof(long)), total = 0;
    for (int f = 1; f < argc; f++) {
        size_t size;
        char *text = read_file(argv[f], &size);
        int *ids = malloc((size + 1) * sizeof(int));
        // encode in ~64 KB pieces cut at line ends (the pre-tokenizer splits there anyway)
        for (size_t off = 0; off < size;) {
            size_t len = chunk_len(text + off, 1 << 16);
            char saved = text[off + len];
            text[off + len] = '\0';
            int n = tokenizer_encode(&tok, text + off, ids);
            text[off + len] = saved;
            for (int i = 0; i < n; i++) counts[ids[i]]++;
            total += n;
            off += len;
        }
        free(ids);
        free(text);
    }
    int distinct = 0;
    for (int i = 0; i < tok.vocab_size; i++)
        if (counts[i]) {
            printf("%d %ld\n", i, counts[i]);
            distinct++;
        }
    fprintf(stderr, "%ld tokens, %d distinct\n", total, distinct);
    free(counts);
    tokenizer_free(&tok);
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "bpb") == 0) return bpb(argc - 2, argv + 2);
    if (argc >= 2 && strcmp(argv[1], "mc") == 0) return mc(argc - 2, argv + 2);
    if (argc >= 2 && strcmp(argv[1], "calib") == 0) return calib(argc - 2, argv + 2);
    if (argc >= 2 && strcmp(argv[1], "count") == 0) return count(argc - 2, argv + 2);
    die("usage: bench bpb|mc <model.bin> <file> [options] | bench calib <model.bin> <text> <stats.bin> [options]"
        " | bench count <tokenizer.bin> <text>...");
}
