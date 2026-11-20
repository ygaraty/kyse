// Small helpers shared by every file.
#ifndef UTIL_H
#define UTIL_H

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef KYSE_LIBRARY
[[noreturn]] void kyse_fail(const char *fmt, ...);
#define die kyse_fail
#else
[[noreturn]] static inline void die(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    exit(1);
}

#endif

// bf16 is the top half of an f32
static inline float bf16_to_f32(uint16_t h) {
    union { uint32_t u; float f; } v = { (uint32_t)h << 16 };
    return v.f;
}

static inline double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// "dir/model.bin" + "tokenizer.bin" -> "dir/tokenizer.bin" (malloc'ed)
static inline char *sibling_path(const char *path, const char *name) {
    const char *slash = strrchr(path, '/');
    int dir_len = slash ? (int)(slash - path + 1) : 0;
    char *out = malloc(dir_len + strlen(name) + 1);
    sprintf(out, "%.*s%s", dir_len, path, name);
    return out;
}

static inline float f16_to_f32(uint16_t h) {
    union { uint16_t u; _Float16 f; } v = { h };
    return (float)v.f;
}

#endif
