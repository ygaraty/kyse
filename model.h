// Qwen3.5 text model: weights, state and the forward pass.
//
// Architecture: configurable decoder layers in a 3:1 pattern of
//   - gated deltanet: linear attention with a recurrent [dk, dv] state per head
//   - gated full attention: GQA, q/k RMSNorm, partial RoPE, sigmoid output gate
// each followed by a SwiGLU MLP. The LM head may be tied or separate.
#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>
#include "backend.h"

// hyperparameters, stored verbatim in the model.bin header (see export.py)
typedef struct {
    int32_t dim, hidden_dim, n_layers, vocab_size;
    int32_t n_heads, n_kv_heads, head_dim, rot_dim;                  // full attention
    int32_t lin_k_heads, lin_v_heads, lin_k_dim, lin_v_dim, conv_k;  // gated deltanet
    float eps, rope_theta;
} Config;

typedef struct {
    const float *ln1, *ln2;             // RMSNorm weights, "+1" baked in by export.py
    // full attention
    Mat wq, wk, wv, wo;                 // wq outputs [query | gate] per head
    const float *q_norm, *k_norm;
    // gated deltanet
    Mat w_qkv, w_z, w_a, w_b, w_out;
    const float *conv, *A_log, *dt_bias, *norm;
    // mlp
    Mat w_gate, w_up, w_down;
} Layer;

// every buffer lives in one backend arena
typedef struct {
    float *x, *xb, *xb2;         // residual stream, normed input, block output  [dim]
    float *proj;                 // query|gate (full attention) or q|k|v (deltanet)
    float *k, *v;                // full attention key, value                    [n_kv_heads*head_dim]
    float *z, *a, *b;            // deltanet output gate, decay and beta inputs
    float *mix;                  // attention output, input to the out projection
    float *hb, *hb2;             // mlp                                          [hidden_dim]
    float *logits;               // [vocab_size]
    float *att;                  // attention scores [n_heads, seq_len]
    float *k_cache, *v_cache;    // [n_full_layers, seq_len, n_kv_heads*head_dim]
    float *conv_state;           // [n_deltanet_layers, conv_k-1, conv_dim]
    float *ssm_state;            // [n_deltanet_layers, lin_v_heads, lin_k_dim, lin_v_dim]
} State;

typedef struct {
    Config c;
    const int32_t *full_attn;    // per layer: 1 = full attention, 0 = gated deltanet
    int seq_len;                 // context length the state is allocated for
    WeightType mat_type;         // format of the layer matrices
    Mat embed, head;             // [vocab_size, dim]; head aliases embed when tied
    const float *final_norm;
    Layer *layers;
    State s;
    Backend *be;
    void *file;
    size_t file_size, data_size; // page-rounded mapping and actual file length
} Model;

void model_load(Model *m, const char *path, Backend *be, int seq_len);
void model_free(Model *m);
void model_reset(Model *m);  // forget the conversation (clears the recurrent state)

// Branching: besides the KV cache, which later positions simply overwrite,
// the only memory of past tokens is the recurrent state (conv + deltanet).
// Save it at some position, and restore it to continue from there again.
size_t model_snapshot_size(const Model *m);  // bytes
void model_snapshot(Model *m, void *buf);
void model_restore(Model *m, const void *buf);

// runs one token at position pos. With want_logits the backend is synced and
// the logits are returned; otherwise the work is only queued and NULL returned.
float *model_forward(Model *m, int token, int pos, bool want_logits);

#endif
