// Token sampling: greedy (temperature 0) or temperature + top-p.
#ifndef SAMPLER_H
#define SAMPLER_H

#include <stdint.h>

typedef struct { float p; int id; } ProbIndex;

typedef struct {
    int vocab_size;
    float temperature, topp;
    uint64_t rng;
    ProbIndex *buf;  // [vocab_size] scratch for top-p
} Sampler;

void sampler_init(Sampler *s, int vocab_size, float temperature, float topp, uint64_t seed);
void sampler_free(Sampler *s);
int  sampler_sample(Sampler *s, float *logits);  // clobbers logits

#endif
