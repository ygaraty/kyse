// Qwen3.5 inference in plain C: command line, one-shot generation and chat.
#include <string.h>
#include "model.h"
#include "sampler.h"
#include "tokenizer.h"
#include "util.h"

// chat turns as rendered by the model's chat_template.jinja, thinking disabled
#define SYSTEM_TURN "<|im_start|>system\n%s<|im_end|>\n"
#define USER_TURN   "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

static Backend *find_backend(const char *name) {
    Backend *be = backend_find(name);
    if (!be) die("unknown backend '%s'", name);
    return be;
}

// ---------------------------------------------------------------------------
// generation

typedef struct {
    Model model;
    Tokenizer tok;
    Sampler sampler;
    int end_of_text, im_end;  // token ids that end a reply
    int pos;      // next position in the context
    int max_new;  // per reply
    bool verbose;
} Session;

// Feeds text at the current position, then samples and prints tokens until an
// end token, max_new or the end of the context. Returns false if the text
// did not fit into the context.
static bool generate(Session *s, const char *text) {
    int *tokens = malloc((strlen(text) + 1) * sizeof(int));
    int n = tokenizer_encode(&s->tok, text, tokens);
    if (n < 1 || s->pos + n >= s->model.seq_len) {
        free(tokens);
        return false;
    }
    if (s->verbose) {
        fprintf(stderr, "tokens (%d):", n);
        for (int i = 0; i < n; i++) fprintf(stderr, " %d", tokens[i]);
        fprintf(stderr, "\n");
    }

    // prefill: all but the last token are only queued, never waited on
    double t_start = now();
    for (int i = 0; i < n - 1; i++) model_forward(&s->model, tokens[i], s->pos++, false);

    // decode: one sync per token
    int token = tokens[n - 1], generated = 0;
    double t_first = 0;
    while (generated < s->max_new && s->pos < s->model.seq_len) {
        float *logits = model_forward(&s->model, token, s->pos++, true);
        if (!generated) t_first = now();
        token = sampler_sample(&s->sampler, logits);
        if (token == s->im_end || token == s->end_of_text) break;
        int len;
        const char *piece = tokenizer_decode(&s->tok, token, &len);
        fwrite(piece, 1, len, stdout);
        fflush(stdout);
        generated++;
    }
    double t_end = now();
    printf("\n");
    fflush(stdout);
    if (s->verbose)
        fprintf(stderr, "[%s] prompt: %d tok in %.2fs (%.1f tok/s) | generation: %d tok in %.2fs (%.1f tok/s)\n",
                s->model.be->name, n, t_first - t_start, n / (t_first - t_start),
                generated, t_end - t_first, generated / (t_end - t_first));
    free(tokens);
    return true;
}

// interactive multi-turn chat. The context keeps growing, so the model
// remembers earlier turns until /reset or the context is full.
static void chat(Session *s, const char *system) {
    fprintf(stderr, "chat with Qwen3.5 (/reset to forget, ctrl-d to quit)\n");
    char line[16384], *text = malloc(sizeof(line) + 1024);
    bool first = true;
    for (;;) {
        printf("\n> ");
        fflush(stdout);
        if (!fgets(line, sizeof(line), stdin)) break;
        line[strcspn(line, "\n")] = '\0';
        if (!line[0]) continue;
        if (strcmp(line, "/reset") == 0) {
            model_reset(&s->model);
            s->pos = 0;
            first = true;
            fprintf(stderr, "(conversation cleared)\n");
            continue;
        }

        // close the previous reply (its <|im_end|> was sampled but never fed), then the new turn
        int off = 0;
        if (first && system) off = sprintf(text, SYSTEM_TURN, system);
        if (!first) off = sprintf(text, "<|im_end|>\n");
        sprintf(text + off, USER_TURN, line);

        printf("\n");
        if (!generate(s, text)) {
            fprintf(stderr, "(context full, use /reset or a larger -c)\n");
            continue;
        }
        first = false;
    }
    printf("\n");
    free(text);
}

// ---------------------------------------------------------------------------

[[noreturn]] static void usage(void) {
    fprintf(stderr,
        "usage: qwen <model.bin> [options]\n"
        "  without -i you get an interactive chat\n"
        "  -i <text>   one-shot prompt\n"
        "  -y <text>   system prompt (chat / one-shot)\n"
        "  -r          raw completion of -i, no chat template\n"
        "  -n <int>    max tokens per reply (default 1024)\n"
        "  -t <float>  temperature, 0 = greedy (default 0.7)\n"
        "  -p <float>  top-p (default 0.8)\n"
        "  -s <int>    random seed (default: time)\n"
        "  -c <int>    context length (default 4096)\n"
        "  -z <path>   tokenizer (default: tokenizer.bin next to the model)\n"
        "  -b <name>   backend:");
    for (int i = 0; i < N_BACKENDS; i++)
        fprintf(stderr, " %s%s", backends[i]->name, i == 0 ? " (default)" : "");
    fprintf(stderr, "\n"
        "  -v          print token ids and speed\n");
    exit(1);
}

int main(int argc, char **argv) {
    if (argc < 2 || argv[1][0] == '-') usage();
    const char *model_path = argv[1], *prompt = NULL, *system = NULL, *tokenizer_path = NULL;
    int max_new = 1024, seq_len = 4096;
    float temperature = 0.7f, topp = 0.8f;
    uint64_t seed = (uint64_t)time(NULL);
    bool raw = false, verbose = false;
    Backend *be = backends[0];

    for (int i = 2; i < argc; i++) {
        if (argv[i][0] != '-' || strlen(argv[i]) != 2) usage();
        char flag = argv[i][1];
        if (flag == 'r') { raw = true; continue; }
        if (flag == 'v') { verbose = true; continue; }
        if (i + 1 >= argc) usage();
        const char *arg = argv[++i];
        switch (flag) {
            case 'i': prompt = arg; break;
            case 'y': system = arg; break;
            case 'z': tokenizer_path = arg; break;
            case 'n': max_new = atoi(arg); break;
            case 't': temperature = strtof(arg, NULL); break;
            case 'p': topp = strtof(arg, NULL); break;
            case 's': seed = strtoull(arg, NULL, 10); break;
            case 'c': seq_len = atoi(arg); break;
            case 'b': be = find_backend(arg); break;
            default: usage();
        }
    }
    if (raw && !prompt) die("-r needs a prompt (-i)");

    Session s = { .max_new = max_new, .verbose = verbose };
    model_load(&s.model, model_path, be, seq_len);
    char *default_tokenizer = sibling_path(model_path, "tokenizer.bin");
    tokenizer_load(&s.tok, tokenizer_path ? tokenizer_path : default_tokenizer);
    if (s.tok.vocab_size != s.model.c.vocab_size) die("tokenizer and model have different vocab sizes");
    s.end_of_text = tokenizer_special(&s.tok, "<|endoftext|>");
    s.im_end = tokenizer_special(&s.tok, "<|im_end|>");
    sampler_init(&s.sampler, s.model.c.vocab_size, temperature, topp, seed);

    if (!prompt) {
        chat(&s, system);
    } else if (raw) {
        printf("%s", prompt);
        if (!generate(&s, prompt)) die("prompt does not fit the context");
    } else {
        size_t size = strlen(prompt) + (system ? strlen(system) : 0) + sizeof(SYSTEM_TURN) + sizeof(USER_TURN);
        char *text = malloc(size);
        int off = system ? snprintf(text, size, SYSTEM_TURN, system) : 0;
        snprintf(text + off, size - off, USER_TURN, prompt);
        if (!generate(&s, text)) die("prompt does not fit the context");
        free(text);
    }

    free(default_tokenizer);
    sampler_free(&s.sampler);
    tokenizer_free(&s.tok);
    model_free(&s.model);
    return 0;
}
