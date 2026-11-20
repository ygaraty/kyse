// Persistent inference worker for the companion Piara server. Stdout is a framed protocol:
//   request: "generate <max_tokens> <temperature> <top_p> <seed> <bytes>\n" + prompt
//   response: "token <id> <bytes>\n" + token bytes; caller replies "next\n" or "stop\n"
//   terminal: "done <prompt_tokens> <output_tokens> <length:0|1>\n"
//   request error: "error <bytes>\n" + message (no done frame)
// "ready\n" follows startup. Diagnostics go to stderr. No prompt is echoed.
#include <inttypes.h>
#include <math.h>
#include "model.h"
#include "sampler.h"
#include "tokenizer.h"
#include "util.h"

#define MAX_PROMPT_BYTES (16 * 1024 * 1024)

static void request_error(const char *message) {
    printf("error %zu\n%s", strlen(message), message);
    fflush(stdout);
}

int main(int argc, char **argv) {
    if (argc < 2) die("usage: qwen-worker <model.bin> [-b backend] [-c context] [-z tokenizer]");
    Backend *be = backends[0];
    int context = 4096;
    const char *tokenizer_path = NULL;
    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc) die("worker option needs a value");
        if (!strcmp(argv[i], "-b")) {
            be = backend_find(argv[i + 1]);
            if (!be) die("unknown backend %s", argv[i + 1]);
        } else if (!strcmp(argv[i], "-c")) context = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "-z")) tokenizer_path = argv[i + 1];
        else die("unknown worker option %s", argv[i]);
    }
    Model model = {0};
    Tokenizer tok;
    Sampler sampler;
    model_load(&model, argv[1], be, context);
    char *sibling = sibling_path(argv[1], "tokenizer.bin");
    tokenizer_load(&tok, tokenizer_path ? tokenizer_path : sibling);
    free(sibling);
    if (tok.vocab_size != model.c.vocab_size) die("tokenizer and model vocab sizes differ");
    int eos = tokenizer_special(&tok, "<|endoftext|>");
    int im_end = tokenizer_special(&tok, "<|im_end|>");
    sampler_init(&sampler, tok.vocab_size, 0.7f, 1.0f, 1);
    puts("ready");
    fflush(stdout);

    char header[256];
    while (fgets(header, sizeof(header), stdin)) {
        int max_tokens;
        float temperature, top_p;
        uint64_t seed;
        size_t bytes;
        if (sscanf(header, "generate %d %f %f %" SCNu64 " %zu", &max_tokens,
                   &temperature, &top_p, &seed, &bytes) != 5 || bytes > MAX_PROMPT_BYTES ||
            max_tokens <= 0 || !isfinite(temperature) || temperature < 0 ||
            !isfinite(top_p) || top_p < 0 || top_p > 1)
            die("invalid worker request header");
        char *prompt = malloc(bytes + 1);
        int *ids = malloc((bytes + 1) * sizeof(int));
        if (!prompt || !ids) die("out of memory");
        if (fread(prompt, 1, bytes, stdin) != bytes) die("truncated worker request");
        prompt[bytes] = '\0';
        if (memchr(prompt, '\0', bytes)) {
            free(prompt);
            free(ids);
            request_error("prompt must not contain NUL bytes");
            continue;
        }
        int n = tokenizer_encode(&tok, prompt, ids);
        free(prompt);
        if (n < 1 || n >= context) {
            free(ids);
            request_error("prompt is empty or exceeds the context window");
            continue;
        }
        model_reset(&model);
        sampler.temperature = temperature;
        sampler.topp = top_p;
        sampler.rng = seed ? seed : 1;
        int pos = 0, output = 0, length = 1;
        for (int i = 0; i < n - 1; i++) model_forward(&model, ids[i], pos++, false);
        int token = ids[n - 1];
        free(ids);
        while (output < max_tokens && pos < context) {
            token = sampler_sample(&sampler, model_forward(&model, token, pos++, true));
            if (token == eos || token == im_end) { length = 0; break; }
            int size;
            const char *piece = tokenizer_decode(&tok, token, &size);
            printf("token %d %d\n", token, size);
            fwrite(piece, 1, size, stdout);
            fflush(stdout);
            output++;
            // Per-token acknowledgments provide cancellation and backpressure.
            if (!fgets(header, sizeof(header), stdin)) goto shutdown;
            if (!strcmp(header, "stop\n")) { length = 0; break; }
            if (strcmp(header, "next\n")) die("invalid worker acknowledgment");
        }
        printf("done %d %d %d\n", n, output, length);
        fflush(stdout);
    }
shutdown:
    sampler_free(&sampler);
    tokenizer_free(&tok);
    model_free(&model);
    return 0;
}
