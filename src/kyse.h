// Stable, opaque shared-library API. One loaded model per process (backend globals).
// All calls are serialized. Token bytes remain valid until kyse_close().
#ifndef KYSE_H
#define KYSE_H
#include <stddef.h>
#include <stdint.h>
#define KYSE_API __attribute__((visibility("default")))

typedef struct Kyse Kyse;
typedef struct { const char *data; size_t size; int id; } KyseToken;
typedef struct { int prompt_tokens, completion_tokens, length; } KyseUsage;

enum { KYSE_FAILED = -2, KYSE_INVALID = -1, KYSE_DONE = 0, KYSE_TOKEN = 1 };
KYSE_API int kyse_api_version(void);
// Errors are thread-local; copy the message before the next call on that thread.
KYSE_API const char *kyse_error(void);
KYSE_API Kyse *kyse_open(const char *model, const char *tokenizer, const char *backend, int context);
KYSE_API int kyse_close(Kyse *model);
KYSE_API int kyse_begin(Kyse *model, const char *prompt, size_t bytes, int max_tokens,
                       float temperature, float top_p, uint64_t seed);
KYSE_API int kyse_next(Kyse *model, KyseToken *token); // TOKEN, DONE or FAILED
KYSE_API int kyse_stop(Kyse *model); // stop at a native-call boundary; keep weights loaded
KYSE_API int kyse_usage(Kyse *model, KyseUsage *usage);
#endif
