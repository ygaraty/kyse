# qwen35

**Stupid simple C inference for Qwen3.5-0.8B**, the hybrid linear-attention model
(text only). About 2,500 lines of C, Metal and Python, with no dependencies
beyond the system frameworks.

- **Three backends** behind one small interface: plain C (the readable reference),
  Apple Accelerate, and Metal. On Metal every op is a GPU kernel and the CPU waits
  once per token.
- **Shrink it for English.** Drop the vocabulary rows English never uses and
  quantize to int8: **2.6× smaller, 2.3× faster, same benchmark scores.**
- **Measure it.** `bench` reports bits per byte, ARC-Easy and HellaSwag, so every
  change can be checked.

```
                     size      speed (M1)     bits/byte   ARC-Easy   HellaSwag
original (bf16)    1,506 MB    33 ms/token      1.061       57.6%      45.0%
english int8         588 MB    14 ms/token      1.077       61.2%      46.2%
english int4         339 MB    12 ms/token      1.139       57.2%      37.5%
```

## Quick start

```
hf download Qwen/Qwen3.5-0.8B --local-dir ../models/qwen3.5-0.8b
python3 export.py ../models/qwen3.5-0.8b .   # -> model.bin, tokenizer.bin (stdlib only)
make
./qwen model.bin                              # interactive chat (/reset, ctrl-d to quit)
./qwen model.bin -y "You are a pirate."      # chat with a system prompt
./qwen model.bin -i "Why is the sky blue?"   # one-shot
./qwen model.bin -b cpu -t 0 -i "..."        # reference CPU backend, greedy
```

`./qwen` with no arguments lists every option (`-v` prints token ids and speed).

## Make it smaller: English only + int8

Most of the 248,320-token vocabulary is other scripts (CJK alone is 22%). The
embedding table is also the LM head, so every unused row costs memory *and* time
on every token: the LM head is a third of the original's per-token work.

```
pip install pyarrow                                    # only for data.py
python3 data.py data                                   # corpus + benchmark sets (~52 MB)
./bench count tokenizer.bin data/english.txt > data/counts.txt
python3 shrink.py model.bin tokenizer.bin data/counts.txt en-q8 --quant q8   # needs numpy
./qwen en-q8/model.bin
```

What `shrink.py` does:

- **Vocabulary: 248,320 → 52,352 tokens.** It keeps the tokens that appear at
  least twice in 45 MB of English, plus:
  - all 256 single bytes, so any text still encodes (unseen words just take more
    tokens);
  - the special tokens and the chat-template words;
  - every intermediate BPE token of the kept ones, so text made of kept tokens
    tokenizes *exactly* as before (verified: 2,504,609 tokens on WikiText train,
    same as the original).

  This drops 203M parameters. On unseen English the new vocabulary needs only
  0.5% more tokens.
- **Quantization.** Every matrix becomes int8 (`q8`) or int4 (`q4`) with one f16
  scale per 32 weights. For int4, each group's scale is picked from a few
  candidates to minimize the squared error, and the embedding stays int8 by
  default (`--embed-quant`).

  The Metal and CPU backends run the quantized formats directly; Accelerate falls
  back to the reference op for them. Token ids change, so a shrunk `model.bin`
  comes with its own `tokenizer.bin`.

## Benchmarks

Apple M1 (16 GB), Metal backend. To reproduce, run `data.py` (above), then:

```
./bench bpb model.bin data/wiki.test.txt -k 8        # bits per byte, 8 chunks of ~4 KB
./bench mc  model.bin data/arc_easy.txt -k 250       # multiple choice
./bench mc  model.bin data/hellaswag.txt -k 80
```

| | original | english bf16 | **english int8** | english int4 |
|---|---|---|---|---|
| file size | 1,506 MB | 1,104 MB | **588 MB** | 339 MB |
| memory in use (4K context) | ~1.65 GB | ~1.24 GB | **~0.72 GB** | ~0.47 GB |
| generation speed | 30 tok/s | 42 tok/s | **70 tok/s** | 87 tok/s |
| WikiText-2 bits/byte ↓ | 1.0611 | 1.0766 | **1.0771** | 1.1393 |
| WikiText-2 top-1 next token | 40.6% | 40.3% | **40.3%** | 38.6% |
| ARC-Easy acc (250 questions, ±3.1) | 57.6% | 61.2% | **61.2%** | 57.2% |
| ARC-Easy acc_norm | 50.4% | 54.0% | **53.6%** | 52.8% |
| HellaSwag acc (80 items, ±5.5) | 37.5% | 36.2% | **36.2%** | 33.8% |
| HellaSwag acc_norm | 45.0% | 45.0% | **46.2%** | 37.5% |

How to read this:

- **int8 costs nothing measurable**, on top of the vocabulary change.
- **The English vocabulary costs 1.5% bits/byte.** Rare words now split into
  pieces the model never saw in training. Its ARC-Easy gain is not the model
  getting smarter: with fewer competing tokens, the answer choices get scored a
  little differently.
- **Plain int4 rounding hurts.** It loses 5.8% bits/byte and 7.5 points of
  HellaSwag acc_norm. Keeping the sensitive matrices at int8, or rotating /
  activation-aware scaling (AWQ), are the usual fixes.
- **Bits per byte is the sensitive measure.** Unlike perplexity, it doesn't depend
  on the vocabulary. The multiple-choice samples are small and only catch large
  changes.

`bench mc` scores every choice by its log-likelihood after the context, as
lm-evaluation-harness does. `acc` uses the total log-likelihood; `acc_norm`
divides it by the choice's length in bytes. The context runs once and each
choice branches from a snapshot of the recurrent state (`model_snapshot`).

## The model in one screen

```
token ─► embed (248,320 × 1024, tied with the LM head)
          │
          ├─ 24 × ┬─ RMSNorm ─► Gated DeltaNet      (layers 0-2, 4-6, ...: 18 layers)
          │       │             or gated attention   (every 4th layer: 6 layers)
          │       └─ RMSNorm ─► SwiGLU MLP 1024 → 3584 → 1024
          │
          └─ RMSNorm ─► LM head ─► logits ─► sampler
```

- **Gated DeltaNet** is linear attention with memory. Each of its 16 heads keeps a
  128×128 matrix S that it reads and updates every token:
  `S ← decay·S;  δ = β(v − Sᵀk);  S += k δᵀ;  out = Sᵀq`.
  A short causal conv (k = 4) runs before it and a gated RMSNorm after it. Its
  state is a fixed 18 MB, however long the context.
- **Gated attention** uses 8 query heads × 256 dims and 2 KV heads, q/k RMSNorm,
  RoPE on a quarter of each head (θ = 10⁷), and a sigmoid output gate. Its KV
  cache costs 24 KB per token.
- **Parameters: 752M.** Embedding 254M (34%), MLP 264M (35%), DeltaNet 190M (25%),
  attention 44M (6%).
- **Every token is memory-bound.** It reads all the weights once, so file size ≈
  speed.

## Files

| file | what |
|---|---|
| `main.c` | command line, chat template, prefill + decode loop |
| `model.c` | loads `model.bin`, lays out the state, wires the forward pass |
| `ops_ref.c` | reference C implementation of every op (the readable spec of the math) |
| `backend.h` | compute backend interface, matrix formats |
| `backend_cpu.c` | plain C backend (just the reference ops) |
| `backend_accelerate.c` | Accelerate BLAS matvec + reference ops |
| `backend_metal.m` + `kernels.metal` | Metal: every op is a kernel, one CPU wait per token |
| `tokenizer.c` | byte-level BPE + hand-written version of Qwen's pre-tokenizer regex |
| `sampler.c` | greedy / temperature + top-p |
| `export.py` | HF safetensors → flat `model.bin` / `tokenizer.bin` (stdlib only) |
| `shrink.py` | `model.bin` → English-only vocabulary + int8/int4 matrices (numpy) |
| `bench.c` | `bpb`: bits per byte · `mc`: multiple choice · `count`: token counts for `shrink.py` |
| `data.py` | downloads the corpus and the benchmark sets (pyarrow) |

## Interactive explainers

Built while writing this, with real numbers from this code:

- [Attention in Slow Motion](https://claude.ai/artifact/9iFSq1HpFzv5nEct7uYCJm): step-through animations of plain and gated attention
- [Qwen3.5-0.8B Architecture](https://claude.ai/artifact/JKaPA5myjQzfvEeVXzUt1y): the stack, both mixer types, the MLP, parameters and memory
- [The LM Head](https://claude.ai/artifact/9q95W4iSQxvkYFwhCSXLy7): from the last hidden state to logits to the sampled token, with real logits
- [Inside a DeltaNet Layer](https://claude.ai/artifact/5Yk8hLPQr6rkeqy5yfoKPw): every matrix multiplication of layer 0, drawn to scale with real weights

## Adding a backend

`model.c` only wires ops together; every op goes through the `Backend` vtable in
`backend.h` (embed, matvec, rmsnorm, add, swiglu, attention, deltanet, sync).
Ops may be async; the CPU only reads memory (the logits) after `sync()`.
Matrices arrive as a `Mat` (data, format, shape); a backend that only handles
bf16 can fall back to `ref_matvec` for the quantized formats.

The quickest start is to implement `matvec` only and reuse the `ref_*` ops for
the rest (see `backend_accelerate.c`). Move the other ops over when you want to
remove CPU round trips. Add an `extern Backend backend_xxx;` and list it in
`backends[]` in `backend.h`.

## Notes / simplifications

- One token at a time, prompt included (no batched prefill).
- Weights are bf16 (or int8/int4 after `shrink.py`); activations and state are f32.
- The pre-tokenizer treats every non-ASCII byte as a letter. It's exact for ASCII
  and very close for everything else. There's no NFC normalization.
- The chat template is hard-coded to non-thinking mode; `-r` gives raw completion.
- Chat keeps the whole conversation in the context (KV cache + DeltaNet state).
  When it fills up, use `/reset` or start with a larger `-c`.
- The vision tower and the MTP head are ignored.
- Developed and measured on macOS (Apple M1). The CPU backend is plain C23 and
  should build elsewhere, but only macOS has been tested.

## License

The code is MIT (see `LICENSE`). The Qwen3.5 weights are Apache-2.0, from
[Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B). The datasets
`data.py` downloads keep their own licenses: WikiText (CC BY-SA), Dolly-15k
(CC BY-SA 3.0), TinyStories (CDLA-Sharing), ARC (CC BY-SA 4.0) and HellaSwag (MIT).
