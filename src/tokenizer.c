#include <stdint.h>
#include <sys/stat.h>
#include <string.h>
#include "tokenizer.h"
#include "util.h"

// ---------------------------------------------------------------------------
// loading

static void read_exact(FILE *f, void *dst, size_t size) {
    if (fread(dst, 1, size, f) != size) die("tokenizer file is truncated");
}

static int cmp_bytes(const char *a, int la, const char *b, int lb) {
    int c = memcmp(a, b, la < lb ? la : lb);
    return c ? c : la - lb;
}

static const Tokenizer *sort_ctx;  // qsort has no user pointer

static int cmp_ids(const void *a, const void *b) {
    int i = *(const int *)a, j = *(const int *)b;
    return cmp_bytes(sort_ctx->tokens[i], sort_ctx->lens[i], sort_ctx->tokens[j], sort_ctx->lens[j]);
}

static bool is_special(const Tokenizer *t, int id) {
    for (int i = 0; i < t->n_special; i++)
        if (t->special[i] == id) return true;
    return false;
}

// file: vocab_size, n_special, max_len, special ids, then per token: rank, len, bytes
void tokenizer_load(Tokenizer *t, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open %s", path);
    t->file = f;
    int32_t header[3];
    read_exact(f, header, sizeof(header));
    struct stat st;
    if (fstat(fileno(f), &st) != 0 || header[0] <= 0 || header[1] < 0 ||
        header[1] > header[0] || header[2] < 0 ||
        (uint64_t)st.st_size < 12 + (uint64_t)header[1] * 4 + (uint64_t)header[0] * 8)
        die("invalid tokenizer header");
    t->vocab_size = header[0];
    t->n_special = header[1];
    t->special = malloc(t->n_special * sizeof(int));
    t->tokens = calloc(t->vocab_size, sizeof(char *));
    t->lens = malloc(t->vocab_size * sizeof(int));
    t->ranks = malloc(t->vocab_size * sizeof(int));
    t->sorted = malloc(t->vocab_size * sizeof(int));
    if ((!t->special && t->n_special) || !t->tokens || !t->lens || !t->ranks || !t->sorted) die("out of memory");
    read_exact(f, t->special, t->n_special * sizeof(int));
    for (int i = 0; i < t->n_special; i++)
        if (t->special[i] < 0 || t->special[i] >= t->vocab_size) die("invalid special token id");
    for (int i = 0; i < t->vocab_size; i++) {
        int32_t rank_len[2];
        read_exact(f, rank_len, sizeof(rank_len));
        if (rank_len[1] < 0 || rank_len[1] > header[2]) die("invalid token length");
        t->ranks[i] = rank_len[0];
        t->lens[i] = rank_len[1];
        t->tokens[i] = malloc(t->lens[i] + 1);
        if (!t->tokens[i]) die("out of memory");
        read_exact(f, t->tokens[i], t->lens[i]);
        t->tokens[i][t->lens[i]] = '\0';
    }
    fclose(f);
    t->file = NULL;

    t->n_sorted = 0;
    for (int i = 0; i < t->vocab_size; i++)
        if (t->lens[i] > 0 && !is_special(t, i)) t->sorted[t->n_sorted++] = i;
    sort_ctx = t;
    qsort(t->sorted, t->n_sorted, sizeof(int), cmp_ids);
}

void tokenizer_free(Tokenizer *t) {
    if (t->file) fclose(t->file);
    if (t->tokens) for (int i = 0; i < t->vocab_size; i++) free(t->tokens[i]);
    free(t->tokens);
    free(t->lens);
    free(t->ranks);
    free(t->special);
    free(t->sorted);
}

const char *tokenizer_decode(const Tokenizer *t, int id, int *len) {
    *len = t->lens[id];
    return t->tokens[id];
}

// token id for exactly these bytes, or -1
static int lookup(const Tokenizer *t, const char *s, int len) {
    int lo = 0, hi = t->n_sorted - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2, id = t->sorted[mid];
        int c = cmp_bytes(t->tokens[id], t->lens[id], s, len);
        if (c == 0) return id;
        if (c < 0) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

// id of the special token that text starts with, or -1
static int match_special(const Tokenizer *t, const char *text) {
    for (int i = 0; i < t->n_special; i++) {
        int id = t->special[i];
        if (strncmp(text, t->tokens[id], t->lens[id]) == 0) return id;
    }
    return -1;
}

int tokenizer_special(const Tokenizer *t, const char *text) {
    for (int i = 0; i < t->n_special; i++) {
        int id = t->special[i];
        if (t->lens[id] == (int)strlen(text) && memcmp(t->tokens[id], text, t->lens[id]) == 0) return id;
    }
    return -1;
}

// ---------------------------------------------------------------------------
// Pre-tokenizer: a hand-written version of Qwen's split regex
//   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+ | \p{N}
//   | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
// Simplification: every non-ASCII byte counts as a letter.

static bool is_letter(unsigned char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80; }
static bool is_digit(unsigned char c) { return c >= '0' && c <= '9'; }
static bool is_space(unsigned char c) { return c == ' ' || (c >= '\t' && c <= '\r'); }
static bool is_newline(unsigned char c) { return c == '\n' || c == '\r'; }
static bool is_punct(unsigned char c) { return c && !is_letter(c) && !is_digit(c) && !is_space(c); }

// length of the pre-token at the start of s (s is non-empty)
static int next_pretoken(const unsigned char *s) {
    if (s[0] == '\'') {  // contractions
        static const char *suffixes[] = { "re", "ve", "ll", "s", "t", "m", "d" };
        for (int k = 0; k < 7; k++) {
            int len = (int)strlen(suffixes[k]);
            bool ok = true;
            for (int j = 0; j < len && ok; j++) ok = (s[1 + j] | 0x20) == suffixes[k][j];
            if (ok) return 1 + len;
        }
    }
    if (is_letter(s[0]) || (!is_newline(s[0]) && !is_digit(s[0]) && is_letter(s[1]))) {  // optional prefix + word
        int i = 1;
        while (is_letter(s[i])) i++;
        return i;
    }
    if (is_digit(s[0])) return 1;  // digits are split one by one

    int i = s[0] == ' ';
    if (is_punct(s[i])) {  // optional space + punctuation + trailing newlines
        while (is_punct(s[i])) i++;
        while (is_newline(s[i])) i++;
        return i;
    }

    int end = 0, last_newline = -1;  // whitespace run
    while (is_space(s[end])) {
        if (is_newline(s[end])) last_newline = end;
        end++;
    }
    if (last_newline >= 0) return last_newline + 1;  // \s*[\r\n]+
    if (s[end] == '\0' || end == 1) return end;      // \s+ at the end, or a single space
    return end - 1;                                  // \s+(?!\S): leave the last space to the next word
}

// ---------------------------------------------------------------------------
// BPE on one pre-token: start from single bytes and keep merging the adjacent
// pair whose merged token has the lowest merge rank.

typedef struct { int id, offset, len; } Piece;

static int bpe(const Tokenizer *t, const char *s, int n, int *out) {
    Piece *p = malloc(n * sizeof(Piece));
    for (int i = 0; i < n; i++) p[i] = (Piece){ lookup(t, s + i, 1), i, 1 };

    int count = n;
    for (;;) {
        int best = -1, best_rank = INT32_MAX, best_id = -1;
        for (int i = 0; i + 1 < count; i++) {
            int id = lookup(t, s + p[i].offset, p[i].len + p[i + 1].len);
            if (id >= 0 && t->ranks[id] >= 0 && t->ranks[id] < best_rank) {
                best = i;
                best_rank = t->ranks[id];
                best_id = id;
            }
        }
        if (best < 0) break;
        p[best] = (Piece){ best_id, p[best].offset, p[best].len + p[best + 1].len };
        memmove(p + best + 1, p + best + 2, (count - best - 2) * sizeof(Piece));
        count--;
    }
    for (int i = 0; i < count; i++) out[i] = p[i].id;
    free(p);
    return count;
}

int tokenizer_encode(const Tokenizer *t, const char *text, int *out) {
    int n = 0;
    while (*text) {
        int special = match_special(t, text);
        if (special >= 0) {
            out[n++] = special;
            text += t->lens[special];
            continue;
        }
        // plain text up to the next special token, pre-tokenized on its own like HF does
        const char *end = text;
        while (*end && (*end != '<' || match_special(t, end) < 0)) end++;
        char *segment = strndup(text, end - text);
        for (char *s = segment; *s;) {
            int len = next_pretoken((const unsigned char *)s);
            n += bpe(t, s, len, out + n);
            s += len;
        }
        free(segment);
        text = end;
    }
    return n;
}
