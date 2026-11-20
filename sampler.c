#include <math.h>
#include <stdlib.h>
#include "sampler.h"

void sampler_init(Sampler *s, int vocab_size, float temperature, float topp, uint64_t seed) {
    *s = (Sampler){ vocab_size, temperature, topp, seed ? seed : 1, malloc(vocab_size * sizeof(ProbIndex)) };
}

void sampler_free(Sampler *s) { free(s->buf); }

static float random_f32(uint64_t *state) {  // xorshift*, uniform in [0, 1)
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (float)((*state * 0x2545F4914F6CDD1Dull) >> 40) / 16777216.0f;
}

static int argmax(const float *x, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (x[i] > x[best]) best = i;
    return best;
}

static void softmax(float *x, int n) {
    float max = x[argmax(x, n)], sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

static int by_prob_desc(const void *a, const void *b) {
    float pa = ((const ProbIndex *)a)->p, pb = ((const ProbIndex *)b)->p;
    return (pa < pb) - (pa > pb);
}

// sample from the smallest set of tokens whose probabilities add up to topp
static int sample_topp(Sampler *s, const float *probs) {
    int n = s->vocab_size, count = 0;
    // a token below this can never be in the nucleus, so skip it before sorting
    float cutoff = (1.0f - s->topp) / (n - 1);
    for (int i = 0; i < n; i++)
        if (probs[i] >= cutoff) s->buf[count++] = (ProbIndex){ probs[i], i };
    qsort(s->buf, count, sizeof(ProbIndex), by_prob_desc);

    float total = 0.0f;
    int last = count - 1;
    for (int i = 0; i < count; i++) {
        total += s->buf[i].p;
        if (total > s->topp) { last = i; break; }
    }
    float r = random_f32(&s->rng) * total, acc = 0.0f;
    for (int i = 0; i <= last; i++) {
        acc += s->buf[i].p;
        if (r < acc) return s->buf[i].id;
    }
    return s->buf[last].id;
}

int sampler_sample(Sampler *s, float *logits) {
    if (s->temperature == 0.0f) return argmax(logits, s->vocab_size);
    for (int i = 0; i < s->vocab_size; i++) logits[i] /= s->temperature;
    softmax(logits, s->vocab_size);
    return sample_topp(s, logits);
}
