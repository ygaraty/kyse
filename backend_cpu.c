// Reference CPU backend: plain C, single threaded, synchronous.
#include <stdlib.h>
#include "backend.h"

static float *arena;

static int cpu_init(void *weights, size_t size) { (void)weights; (void)size; return 0; }

static float *cpu_alloc(size_t n) { return arena = calloc(n, sizeof(float)); }

static void cpu_sync(void) {}

static void cpu_shutdown(void) { free(arena); }

Backend backend_cpu = {
    "cpu", cpu_init, cpu_alloc, ref_embed, ref_matvec, ref_rmsnorm, ref_add, ref_swiglu,
    ref_attention, ref_deltanet, cpu_sync, cpu_shutdown,
};
