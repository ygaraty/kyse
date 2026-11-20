#include <math.h>
#include <pthread.h>
#include <setjmp.h>
#include <sys/mman.h>
#include "kyse.h"
#include "model.h"
#include "sampler.h"
#include "tokenizer.h"
#include "util.h"

struct Kyse {
    Model model;
    Tokenizer tokenizer;
    Sampler sampler;
    KyseUsage usage;
    int previous, position, max_tokens, eos, im_end;
    bool running, failed;
    int *prompt_ids;              // owned even if a native call fails partway through
    char *tokenizer_path;
};

static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
static Kyse *active;
static _Thread_local jmp_buf *error_target;
static _Thread_local char error_message[512];

int kyse_api_version(void) { return 1; }
const char *kyse_error(void) { return error_message; }

static int invalid(const char *message) {
    snprintf(error_message, sizeof(error_message), "%s", message);
    return KYSE_INVALID;
}

// Shared-library builds redirect util.h's fatal CLI errors here. The jump stays
// inside guarded(), so the outer Objective-C autorelease pool unwinds normally.
[[noreturn]] void kyse_fail(const char *format, ...) {
    va_list args;
    va_start(args, format);
    vsnprintf(error_message, sizeof(error_message), format, args);
    va_end(args);
    if (error_target) longjmp(*error_target, 1);
    abort(); // internal programming error: no public API operation is unguarded
}

typedef struct { int (*operation)(void *); void *argument; } Call;
static int guarded(void *opaque) {
    Call *call = opaque;
    jmp_buf target;
    jmp_buf *previous = error_target;
    error_target = &target;
    if (setjmp(target)) {
        error_target = previous;
        if (active) active->failed = true;
        return KYSE_FAILED;
    }
    int result = call->operation(call->argument);
    error_target = previous;
    return result;
}
#ifdef __APPLE__
int kyse_with_pool(int (*operation)(void *), void *argument);
#else
static int kyse_with_pool(int (*operation)(void *), void *argument) { return operation(argument); }
#endif
static int run(int (*operation)(void *), void *argument) {
    Call call = { operation, argument };
    return kyse_with_pool(guarded, &call);
}

static bool usable(Kyse *engine) {
    if (!engine || engine != active) { invalid("model is closed or invalid"); return false; }
    if (engine->failed) { invalid("native model failed; close it before reopening"); return false; }
    return true;
}

static int destroy_backend(void *opaque) {
    Kyse *engine = opaque;
    if (engine->model.be) engine->model.be->shutdown();
    return 0;
}
// Called under mutex, after all pending operations have returned.
static int dispose(Kyse *engine) {
    int result = run(destroy_backend, engine);
    free(engine->model.layers);
    if (engine->model.file && engine->model.file != MAP_FAILED)
        munmap(engine->model.file, engine->model.file_size);
    tokenizer_free(&engine->tokenizer);
    sampler_free(&engine->sampler);
    free(engine->prompt_ids);
    free(engine->tokenizer_path);
    free(engine);
    active = NULL;
    return result;
}

typedef struct { Kyse *engine; const char *path; Backend *backend; int context; } Load;
static int load(void *opaque) {
    Load *args = opaque;
    Kyse *e = args->engine;
    model_load(&e->model, args->path, args->backend, args->context);
    tokenizer_load(&e->tokenizer, e->tokenizer_path);
    if (e->tokenizer.vocab_size != e->model.c.vocab_size) die("tokenizer and model vocab sizes differ");
    e->eos = tokenizer_special(&e->tokenizer, "<|endoftext|>");
    e->im_end = tokenizer_special(&e->tokenizer, "<|im_end|>");
    sampler_init(&e->sampler, e->tokenizer.vocab_size, 0.7f, 1.0f, 1);
    if (!e->sampler.buf) die("out of memory");
    return 0;
}

Kyse *kyse_open(const char *path, const char *tokenizer, const char *backend, int context) {
    pthread_mutex_lock(&mutex);
    error_message[0] = '\0';
    Kyse *result = NULL;
    Backend *be = backend ? backend_find(backend) : backends[0];
    if (active) invalid("only one model may be open per process");
    else if (!path || !be || context < 2) invalid("invalid model path, backend or context");
    else {
        active = calloc(1, sizeof(Kyse));
        if (!active) invalid("out of memory");
        else {
            active->tokenizer_path = tokenizer ? strdup(tokenizer) : sibling_path(path, "tokenizer.bin");
            Load args = { active, path, be, context };
            if (!active->tokenizer_path) invalid("out of memory");
            else if (run(load, &args) == 0) result = active;
            if (!result) {
                char saved[sizeof(error_message)];
                memcpy(saved, error_message, sizeof(saved));
                dispose(active);
                memcpy(error_message, saved, sizeof(saved));
            }
        }
    }
    pthread_mutex_unlock(&mutex);
    return result;
}

int kyse_close(Kyse *engine) {
    pthread_mutex_lock(&mutex);
    int result = engine && engine == active ? dispose(engine) : invalid("model is closed or invalid");
    pthread_mutex_unlock(&mutex);
    return result;
}

// Bound command-buffer residency and drain Metal temporaries during long prefill.
// Each chunk has its own error boundary and (on macOS) autorelease pool.
static int prefill_chunk(void *opaque) {
    Kyse *e = opaque;
    int end = e->position + 128;
    if (end > e->usage.prompt_tokens - 1) end = e->usage.prompt_tokens - 1;
    while (e->position < end) {
        int token = e->prompt_ids[e->position];
        model_forward(&e->model, token, e->position++, false);
    }
    e->model.be->sync();
    return 0;
}

typedef struct { Kyse *engine; const char *prompt; size_t bytes; int max_tokens; float temperature, top_p; uint64_t seed; } Begin;
static int begin(void *opaque) {
    Begin *args = opaque;
    Kyse *e = args->engine;
    if (!args->prompt || args->bytes > 16 * 1024 * 1024 || !args->bytes ||
        memchr(args->prompt, '\0', args->bytes) || args->max_tokens <= 0 ||
        !isfinite(args->temperature) || args->temperature < 0 ||
        !isfinite(args->top_p) || args->top_p < 0 || args->top_p > 1)
        return invalid("invalid prompt or generation parameters");
    // The public API accepts a byte span; tokenizer_encode needs a C string.
    // Reuse an owned allocation for both strings and IDs to unwind errors safely.
    size_t size = (args->bytes + 1) * sizeof(int) + args->bytes + 1;
    free(e->prompt_ids);
    e->prompt_ids = malloc(size);
    if (!e->prompt_ids) die("out of memory");
    char *text = (char *)(e->prompt_ids + args->bytes + 1);
    memcpy(text, args->prompt, args->bytes);
    text[args->bytes] = '\0';
    int n = tokenizer_encode(&e->tokenizer, text, e->prompt_ids);
    if (n < 1 || n >= e->model.seq_len) return invalid("prompt is empty or exceeds the context window");
    model_reset(&e->model);
    e->sampler.temperature = args->temperature;
    e->sampler.topp = args->top_p;
    e->sampler.rng = args->seed ? args->seed : 1;
    e->usage = (KyseUsage){ .prompt_tokens = n, .length = 1 };
    e->position = 0;
    e->max_tokens = args->max_tokens;
    e->running = true;
    bool progress = getenv("KYSE_PREFILL_PROGRESS") != NULL;
    double started = now();
    while (e->position < n - 1) {
        if (run(prefill_chunk, e) < 0) return KYSE_FAILED;
        if (progress && (e->position % 1024 == 0 || e->position == n - 1))
            fprintf(stderr, "prefill: %d/%d tokens (%.1fs)\n", e->position, n - 1, now() - started);
    }
    e->previous = e->prompt_ids[n - 1];
    free(e->prompt_ids);
    e->prompt_ids = NULL;
    return 0;
}
int kyse_begin(Kyse *e, const char *prompt, size_t bytes, int max_tokens, float temperature, float top_p, uint64_t seed) {
    pthread_mutex_lock(&mutex);
    Begin args = { e, prompt, bytes, max_tokens, temperature, top_p, seed };
    int result = usable(e) ? run(begin, &args) : KYSE_FAILED;
    pthread_mutex_unlock(&mutex);
    return result;
}

typedef struct { Kyse *engine; KyseToken *token; } Next;
static int next_token(void *opaque) {
    Next *args = opaque;
    Kyse *e = args->engine;
    if (!args->token) return invalid("token output is NULL");
    if (!e->running) return KYSE_DONE;
    if (e->usage.completion_tokens >= e->max_tokens || e->position >= e->model.seq_len) {
        e->running = false;
        return KYSE_DONE;
    }
    int token = sampler_sample(&e->sampler, model_forward(&e->model, e->previous, e->position++, true));
    if (token == e->eos || token == e->im_end) {
        e->running = false;
        e->usage.length = 0;
        return KYSE_DONE;
    }
    int bytes;
    args->token->data = tokenizer_decode(&e->tokenizer, token, &bytes);
    args->token->size = (size_t)bytes;
    args->token->id = token;
    e->previous = token;
    e->usage.completion_tokens++;
    return KYSE_TOKEN;
}
int kyse_next(Kyse *e, KyseToken *token) {
    pthread_mutex_lock(&mutex);
    Next args = { e, token };
    int result = usable(e) ? run(next_token, &args) : KYSE_FAILED;
    pthread_mutex_unlock(&mutex);
    return result;
}
static int stop(void *opaque) {
    Kyse *e = opaque;
    e->model.be->sync();
    e->running = false;
    e->usage.length = 0;
    return 0;
}
int kyse_stop(Kyse *e) {
    pthread_mutex_lock(&mutex);
    int result = usable(e) ? run(stop, e) : KYSE_FAILED;
    pthread_mutex_unlock(&mutex);
    return result;
}
int kyse_usage(Kyse *e, KyseUsage *usage) {
    pthread_mutex_lock(&mutex);
    int result = KYSE_FAILED;
    if (usable(e)) {
        if (!usage) result = invalid("usage output is NULL");
        else { *usage = e->usage; result = 0; }
    }
    pthread_mutex_unlock(&mutex);
    return result;
}
