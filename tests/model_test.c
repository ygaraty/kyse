#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "model.h"

int main(int argc, char **argv) {
    assert(argc == 3);
    Model m = {0};
    float baseline[3][32];
    model_load(&m, argv[1], &backend_cpu, 8);
    for (int pos = 0; pos < 3; pos++)
        memcpy(baseline[pos], model_forward(&m, pos, pos, true), m.c.vocab_size * sizeof(float));
    model_free(&m);
    Backend *be = &backend_cpu;
#ifdef KYSE_TEST_METAL
    be = backend_find("metal");
#endif
    assert(be);
    model_load(&m, argv[1], be, 8);
    for (int pos = 0; pos < 3; pos++) {
        float *logits = model_forward(&m, pos, pos, true);
        for (int i = 0; i < m.c.vocab_size; i++)
            assert(fabsf(logits[i] - baseline[pos][i]) < 1e-4f + fabsf(baseline[pos][i]) * 1e-3f);
    }
    model_reset(&m);
    assert((m.embed.data == m.head.data) == atoi(argv[2]));
    float *expected = malloc(m.c.vocab_size * sizeof(float));
    float *first = model_forward(&m, 0, 0, true);
    memcpy(expected, first, m.c.vocab_size * sizeof(float));
    for (int i = 0; i < m.c.vocab_size; i++) assert(isfinite(first[i]));
    model_reset(&m);
    first = model_forward(&m, 0, 0, true);
    assert(memcmp(expected, first, m.c.vocab_size * sizeof(float)) == 0);
    // The final logits must use the selected head, including after vocabulary pruning.
    ref_matvec(expected, m.head, m.s.xb);
    for (int i = 0; i < m.c.vocab_size; i++)
        assert(fabsf(expected[i] - first[i]) < 1e-4f + fabsf(expected[i]) * 1e-3f);
    void *snapshot = malloc(model_snapshot_size(&m));
    model_snapshot(&m, snapshot);
    first = model_forward(&m, 1, 1, true);
    memcpy(expected, first, m.c.vocab_size * sizeof(float));
    model_restore(&m, snapshot);
    first = model_forward(&m, 1, 1, true);
    assert(memcmp(expected, first, m.c.vocab_size * sizeof(float)) == 0);
    free(snapshot);
    free(expected);
    model_free(&m);
}
