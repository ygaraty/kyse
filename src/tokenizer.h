// Byte-level BPE tokenizer (GPT-2 / Qwen style), loaded from tokenizer.bin.
#ifndef TOKENIZER_H
#define TOKENIZER_H
#include <stdio.h>

typedef struct {
    FILE *file;    // owned during loading, so a library error can close it
    int vocab_size;
    char **tokens;  // raw bytes of each token (may contain NULs, use lens)
    int *lens;
    int *ranks;     // rank of the merge that creates the token, -1 for single bytes and specials
    int n_special;
    int *special;   // ids of special tokens such as <|im_start|>
    int n_sorted;
    int *sorted;    // non-special token ids sorted by bytes, for binary search
} Tokenizer;

void tokenizer_load(Tokenizer *t, const char *path);
void tokenizer_free(Tokenizer *t);

// Encodes text; special tokens written literally (e.g. "<|im_start|>") are
// recognized. out must hold strlen(text) ints. Returns the number of tokens.
int tokenizer_encode(const Tokenizer *t, const char *text, int *out);

const char *tokenizer_decode(const Tokenizer *t, int id, int *len);

// id of the special token with exactly this text (e.g. "<|im_end|>"), or -1
int tokenizer_special(const Tokenizer *t, const char *text);

#endif
