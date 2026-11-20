# Kyse

**Compressing Qwen3.5 for English, with every step measured.** Kyse takes the
Qwen3.5-0.8B and 9B language models (text only) and makes them smaller and
faster by pruning the vocabulary and quantizing the weights. It checks each
step with held-out benchmarks.

It also comes with `qwen`, a small command-line tool to chat with the original
or compressed models. It's plain C with three backends: a readable CPU
reference, Apple Accelerate, and Metal.

## Results

### Qwen3.5-0.8B (Apple M1, Metal)

| model | file size | smaller | speed | bits/byte ↓ | ARC-Easy | HellaSwag |
|---|---:|---:|---:|---:|---:|---:|
| original (bf16) | 1,506 MB | | 30 tok/s | 1.061 | 57.6% | 45.0% |
| English vocabulary (bf16) | 1,104 MB | 27% | 42 tok/s | 1.077 | 61.2% | 45.0% |
| **English int8** | **588 MB** | **61%** | **70 tok/s** | **1.077** | **61.2%** | **46.2%** |
| **English int4 mixed** | **408 MB** | **73%** | **~80 tok/s¹** | **1.077** | **59.2%** | **41.2%** |
| English int4 (plain) | 339 MB | 77% | 87 tok/s | 1.139 | 57.2% | 37.5% |

- **English int8:** the safe choice, with no measurable loss on top of the
  vocabulary change.
- **English int4 mixed:** the smallest model with int8's quality.
- **Plain int4:** smaller still, but it visibly loses quality.

How to read the columns:

- **bits/byte** is the main quality measure: how well the model predicts
  held-out English (WikiText-2 test, 36 KB). Lower is better. It doesn't depend
  on the tokenizer, so models with different vocabularies compare fairly. For
  scale: the English vocabulary costs 1.5%, and plain int4 costs 5.8%.
- **ARC-Easy** (250 science questions, ±3.1 points) is accuracy.
- **HellaSwag** (80 sentence completions, ±5.5) is length-normalized accuracy.
  These multiple-choice samples are small and only catch large changes.
- **speed** is the decode speed of the chat tool. Every token reads all the
  weights once, so speed roughly follows file size.

¹ Estimated: measured on a busy machine at 72 tok/s, when int8 measured 63
(70 on a quiet machine).

Memory in use for a 4K-token conversation is the file plus 71 MB of state:
about 1.60 GB for the original, 0.67 GB for int8 and 0.49 GB for int4 mixed.

### Qwen3.5-9B (Apple M5, Metal, smoke test)

| model | file size | smaller | speed | bits/byte ↓ |
|---|---:|---:|---:|---:|
| original (bf16) | 17.91 GB | | 6.8 tok/s | 0.8260 |
| **English int8** | **7.81 GB** | **56%** | **15.8 tok/s** | **0.8297** |
| English int4 (int8 embedding and head) | 4.35 GB | 76% | 26.2 tok/s | 0.8312 |

This sample is small: 3.4 KB of WikiText, plus 10 ARC and 7 HellaSwag items
(see [the 9B report](BENCHMARKS_9B.md)). AWQ and mixed precision haven't been
run on 9B yet.

## The compression methods

Every method below is an option of `scripts/shrink.py`, except the f16 KV
cache, which is built into the runtime.

| method | what it shrinks | goal | 0.8B result |
|---|---|---|---|
| [English vocabulary](#1-english-vocabulary) | embedding + output head | drop rows English never uses | −402 MB, +1.5% bits/byte |
| [int8](#2-int8-quantization) | every matrix | half the bytes per weight | −516 MB, no measurable loss |
| [int4](#3-int4-quantization) | layer matrices | a quarter of the bytes per weight | −249 MB more, +5.8% bits/byte |
| [AWQ](#4-awq-activation-aware-scaling) | int4 error | protect weights that see large inputs | recovers 35% of the int4 loss, free |
| [mixed precision](#5-mixed-precision) | int4 error | spend bytes where int4 hurts most | recovers ~100% for +69 MB |
| [low-rank correction](#6-low-rank-correction) | int4 error | move outlier directions to bf16 | recovers 50% for +2 MB; not worth it |
| [f16 KV cache](#7-f16-kv-cache) | memory in use | halve the attention cache | −50 MB at 4K context, no loss |

### 1. English vocabulary

**Goal:** remove the vocabulary entries English text never uses. Qwen's
248,320 tokens cover many scripts (CJK alone is 22%). Each token is a row of
the embedding table and of the output head, and the output head is computed
for every row on every token. In the original 0.8B model that is a third of
each token's work.

**How:** `bench count` tokenizes 45 MB of English (WikiText-2 train,
Dolly-15k, TinyStories). `shrink.py` keeps the tokens seen at least twice
(`--min-count`), plus:

- all 256 single bytes, so any text still encodes (rare words just take more
  tokens);
- the special tokens and the chat-template words;
- every intermediate token the BPE merges pass through to build a kept token.
  Text made of kept tokens therefore tokenizes exactly as before (verified:
  2,504,609 tokens on WikiText train, the same as the original).

This leaves 52,352 tokens. Token ids change, so every compressed model comes
with its own `tokenizer.bin`.

**Cost:** unseen English needs 0.5% more tokens, and bits/byte rises 1.5%
because rare words split into pieces the model saw less in training. Other
languages still encode, byte by byte, but weren't tested. The ARC-Easy gain
isn't the model getting smarter: with fewer competing tokens, the answer
choices are scored slightly differently.

### 2. int8 quantization

**Goal:** halve the bytes per weight. Decoding one token at a time is limited
by memory traffic, not arithmetic, so fewer bytes means proportionally faster
generation.

**How:** each group of 32 consecutive weights in a row gets one f16 scale
(absmax / 127) and 8-bit integers (`--quant q8`). The Metal and CPU backends
compute directly on this format, with no conversion back to bf16.

**Cost:** nothing measurable (1.0771 vs 1.0766 bits/byte).

### 3. int4 quantization

**Goal:** a quarter of the bytes per weight.

**How:** the same groups of 32, with 4-bit integers packed two per byte
(`--quant q4`). Each group's scale is chosen from 13 candidates between 0.70
and 1.0 × absmax / 7, to minimize the squared error. Clipping the largest
weight a little often helps the other 31. The embedding and output head stay
int8 by default (`--embed-quant`).

**Cost:** +5.8% bits/byte and 7.5 points of HellaSwag. The next three methods
try to win that back.

### 4. AWQ (activation-aware scaling)

**Goal:** reduce the int4 error where it matters, for free. A weight's
rounding error is multiplied by its input, so errors on the input channels
that carry large activations hurt most.

**How:** `bench calib` runs the model on 9K tokens of English and records the
mean |x| of every input channel of every matrix. `shrink.py --awq` then scales
those columns up by s = mean|x|^α before quantizing, so they get relatively
finer steps. It folds 1/s into whatever produced the input, so the model
computes exactly the same function: the RMSNorm weights, or the rows of `w_up`
for `w_down`. α is picked per group of matrices by an estimate of the output
error. The file format and kernels don't change.

**Result:** 35% of the int4 loss recovered (1.139 → 1.118 bits/byte), at the
same size and speed. That's less than the paper reports, because most of this
model's activations have no big outliers. Weighting the error estimate by mean
|x|² instead of mean x² matters: the few huge channels dominate x², and
weighting by x² recovered only 14%.

### 5. Mixed precision

**Goal:** spend extra bytes only where int4 hurts most.

**How:** `--q8 PATTERNS` keeps chosen matrices at int8, by layer and name,
e.g. `'*.w_down'` (every layer) or `'0.*'` (all of layer 0). The weak spot is
the out projections, `wo` and `w_out`. Their inputs come through a gate, so
AWQ can't fold a scale into them, and a few input channels of `w_out` are 50
to 100 times the median size. Their int4 error is twice that of every other
matrix.

**Result:** keeping `wo` and `w_out` at int8 (50M weights, +25 MB) recovers
81% of the int4 loss. Adding `w_down` (+44 MB) recovers all of it. That is
the **int4 mixed** model: int8's bits/byte at 408 MB instead of 588.

### 6. Low-rank correction

**Goal:** absorb a matrix's outlier directions into a small bf16 part, so the
rest quantizes better (SVDQuant).

**How:** `--lowrank PATTERNS --rank 16` splits each chosen matrix into its 16
strongest directions, found by an SVD weighted by input size and kept in bf16,
plus a residual that gets quantized. The runtime adds the correction with two
small extra matrix-vector products.

**Result:** not worth it here. On `wo`/`w_out` it recovers a third as much as
int8 does, for a tenth of the bytes (+2 MB). On every matrix it costs 20 MB,
does worse than on those two alone, and slows decoding by about 30%. It stays
available as an option.

### 7. f16 KV cache

**Goal:** less memory in use during long conversations. This doesn't change
the file.

**How:** the attention layers' key/value cache is stored as f16 instead of
f32, in all three backends.

**Result:** 12 KB per token instead of 24 (−50 MB at 4K context), with the
same bits/byte at 4K and 16K-byte contexts.

### Ideas that didn't make it

- **Replacing a matrix with a low-rank product.** At half the size of `w_qkv`
  (rank 439) the weight error is 46%, while int8 gives about 1%.
- **One shared basis for all layers' inputs:** 59% error at rank 512.
- **Lossless entropy coding of the bf16 weights**, as in DFloat11. The exponent
  bits carry only 2.7 bits of information, so about 67% of the size is
  possible losslessly. It isn't implemented, and it competes with int8 rather
  than adding to it.

All the int4 variants side by side, with every metric:
[Detailed results](#detailed-results).

## How to compress a model

You need Python 3 with numpy (compression) and pyarrow (data download), and
a C compiler. On macOS the Metal backend builds automatically.

```sh
make                                             # builds ./qwen and ./bench

# 1. get the model and convert it to model.bin + tokenizer.bin (stdlib only)
hf download Qwen/Qwen3.5-0.8B --local-dir ../models/qwen3.5-0.8b
python3 scripts/export.py ../models/qwen3.5-0.8b .

# 2. English data: the vocabulary corpus and the benchmark sets (~52 MB)
pip install numpy pyarrow
python3 scripts/data.py data

# 3. which tokens English uses (for the vocabulary)
./bench count tokenizer.bin data/english.txt > data/counts.txt

# 4. activation statistics (for AWQ; ~15 min on an M1)
./bench calib model.bin data/english.txt data/calib.bin -k 32 -l 1024
```

Then pick a recipe. Each writes `<out_dir>/model.bin` and `<out_dir>/tokenizer.bin`:

```sh
# English int8: the safe choice (steps 1-3 only)
python3 scripts/shrink.py model.bin tokenizer.bin data/counts.txt en-q8 --quant q8

# English int4 mixed: int8 quality, 30% smaller than int8
python3 scripts/shrink.py model.bin tokenizer.bin data/counts.txt en-q4-mixed \
    --quant q4 --awq data/calib.bin --q8 '*.wo,*.w_out,*.w_down'

# English int4: smallest, with a visible quality loss
python3 scripts/shrink.py model.bin tokenizer.bin data/counts.txt en-q4 --quant q4
```

Check the result, and compare it with the original:

```sh
./bench bpb en-q4-mixed/model.bin data/wiki.test.txt -k 8    # bits per byte
./bench mc  en-q4-mixed/model.bin data/arc_easy.txt -k 250   # multiple choice
./bench mc  en-q4-mixed/model.bin data/hellaswag.txt -k 80
./qwen en-q4-mixed/model.bin                                 # chat with it
```

### `shrink.py` options

| option | default | effect |
|---|---|---|
| `--quant q8\|q4\|bf16` | `q8` | format of the layer matrices |
| `--embed-quant q8\|q4\|bf16` | `q8` for q4, else `--quant` | format of the embedding and output head |
| `--min-count N` | 2 | keep tokens seen at least N times in the corpus |
| `--awq STATS` | off | AWQ scaling from `bench calib` statistics |
| `--q8 PATTERNS` | none | matrices kept at int8, as comma-separated `<layer>.<name>` patterns |
| `--lowrank PATTERNS` | none | matrices that get a bf16 low-rank correction |
| `--rank R` | 16 | rank of those corrections (a multiple of 4) |

Matrix names, as in `src/model.h`:

- attention layers: `wq`, `wk`, `wv`, `wo`;
- DeltaNet layers: `w_qkv`, `w_z`, `w_a`, `w_b`, `w_out`;
- MLP: `w_gate`, `w_up`, `w_down`.

### Qwen3.5-9B

The same steps work for 9B. Its output head is a separate table from its
embedding; both get the same vocabulary pruning, and `--embed-quant` sets the
format of both.

```sh
hf download Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --local-dir models/qwen3.5-9b
python3 scripts/export.py models/qwen3.5-9b models/9b
./bench count models/9b/tokenizer.bin data/english.txt > data/counts-9b.txt
python3 scripts/shrink.py models/9b/model.bin models/9b/tokenizer.bin data/counts-9b.txt en-9b-q8 --quant q8
python3 scripts/benchmark.py models/9b en-9b-q8 --hellaswag-items 7    # the 9B smoke tests
```

The exporter streams the safetensors shards, and `shrink.py` memory-maps its
input, but individual matrices are still converted in memory. Allow about
18 GB for the bf16 export.

## Chatting with the models: `qwen`

```sh
./qwen en-q8/model.bin                          # interactive chat
./qwen en-q8/model.bin -y "You are a pirate."   # with a system prompt
./qwen en-q8/model.bin -i "Why is the sky blue?"  # one question, one answer
./qwen en-q8/model.bin -r -i "Once upon a time"   # raw text completion, no chat template
./qwen en-q8/model.bin -b cpu -t 0 -i "..."       # reference CPU backend, greedy
```

Chat remembers the conversation. Type `/reset` to start over and press
Ctrl-D to quit. The tokenizer is read from next to the model file.

| option | default | |
|---|---|---|
| `-i <text>` | | one-shot prompt instead of a chat |
| `-y <text>` | | system prompt |
| `-r` | | raw completion of `-i`, no chat template |
| `-n <int>` | 1024 | max tokens per reply |
| `-t <float>` | 0.7 | temperature, 0 = greedy |
| `-p <float>` | 0.8 | top-p |
| `-s <int>` | time | random seed |
| `-c <int>` | 4096 | context length |
| `-z <path>` | next to the model | tokenizer |
| `-b <name>` | `metal` | backend: `metal`, `accelerate`, `cpu` |
| `-v` | | print token ids and speed |

Backends:

- **Metal** runs every op as a GPU kernel, and the CPU waits once per token.
- **Accelerate** uses Apple's BLAS for bf16 matrices. It expands them to f32,
  so it needs much more memory, and it uses the reference code for int8/int4.
- **cpu** is the plain C reference: slow, but the readable specification of
  every op.

## Measuring: `bench`

| command | what it measures |
|---|---|
| `bench bpb <model> <text> [-k chunks] [-l bytes]` | bits per byte: cuts the text into chunks of ~4 KB, each scored from a fresh state |
| `bench mc <model> <items> [-k items]` | multiple choice: picks the ending with the highest log-likelihood |
| `bench calib <model> <text> <out> [-k chunks] [-l bytes]` | activation statistics for `shrink.py --awq` |
| `bench count <tokenizer> <text>...` | token counts for `shrink.py` |

`bench bpb`, `mc` and `calib` take `-b <backend>` and `-z <tokenizer>` too.

`bench mc` scores like lm-evaluation-harness:

- `acc` uses each ending's total log-likelihood;
- `acc_norm` divides it by the ending's length in bytes.

The context runs once, and each ending branches from a snapshot of the
recurrent state (`model_snapshot`). `scripts/data.py` writes the item files.

## Detailed results

### Qwen3.5-0.8B

Apple M1 (16 GB), Metal backend:

| | original | English bf16 | English int8 | English int4 |
|---|---:|---:|---:|---:|
| file size | 1,506 MB | 1,104 MB | 588 MB | 339 MB |
| memory in use (4K context) | ~1.60 GB | ~1.19 GB | ~0.67 GB | ~0.42 GB |
| generation speed | 30 tok/s | 42 tok/s | 70 tok/s | 87 tok/s |
| WikiText-2 bits/byte ↓ | 1.0611 | 1.0766 | 1.0771 | 1.1393 |
| WikiText-2 top-1 next token | 40.6% | 40.3% | 40.3% | 38.6% |
| ARC-Easy acc (250 questions, ±3.1) | 57.6% | 61.2% | 61.2% | 57.2% |
| ARC-Easy acc_norm | 50.4% | 54.0% | 54.0% | 52.8% |
| HellaSwag acc (80 items, ±5.5) | 37.5% | 36.2% | 36.2% | 33.8% |
| HellaSwag acc_norm | 45.0% | 45.0% | 46.2% | 37.5% |

The quality rows were measured with an f32 KV cache. The int8 column was
re-measured with f16: identical, apart from one ARC-Easy question (acc_norm
53.6% → 54.0%).

### int4 variants (0.8B)

All with the English vocabulary, an int8 embedding and the f16 KV cache. The
"loss recovered" column is the share of plain int4's bits/byte loss against
int8 that the variant wins back. Speed is the best of 7 runs on a busy
machine, so compare it within the column only.

| int4 variant | size | bits/byte ↓ | loss recovered | top-1 | ARC acc / norm | HellaSwag acc / norm | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| English int8 (reference) | 588 MB | 1.0771 | | 40.3% | 61.2 / 54.0 | 36.2 / 46.2 | 63 |
| plain | 339 MB | 1.1393 | 0% | 38.6% | 57.2 / 52.8 | 33.8 / 37.5 | 79 |
| AWQ | 339 MB | 1.1176 | 35% | 38.6% | 58.4 / 53.2 | 35.0 / 41.2 | 76 |
| AWQ + rank 16 on `wo`, `w_out` | 341 MB | 1.1081 | 50% | 39.4% | | | 75 |
| AWQ + int8 `wo`, `w_out` | 364 MB | 1.0892 | 81% | 40.0% | 58.0 / 55.2 | 35.0 / 40.0 | 76 |
| AWQ + int8 `wo`, `w_out`, `w_down` | 408 MB | 1.0773 | ~100% | 40.6% | 59.2 / 55.2 | 35.0 / 41.2 | 72 |
| AWQ + rank 16 on every large matrix | 359 MB | 1.1118 | 44% | 39.7% | | | |
| AWQ + int8 `wo`, `w_out` + rank 16 on the rest | 382 MB | 1.0993 | 64% | 40.1% | | | |

The multiple-choice samples can't rank the int4 variants: all are within each
other's error bars. The HellaSwag gap between int4 mixed and int8 (41 vs 46)
is less than one standard error. Raw output is in
[results/0.8b/int4-experiments.txt](results/0.8b/int4-experiments.txt).

### Qwen3.5-9B

Apple M5 (24 GB), Metal backend:

| | original | English int8 | English int4 |
|---|---:|---:|---:|
| file size | 17.91 GB | 7.81 GB | 4.35 GB |
| generation speed | 6.8 tok/s | 15.8 tok/s | 26.2 tok/s |
| WikiText-2 bits/byte ↓ (3.4 KB) | 0.8260 | 0.8297 | 0.8312 |
| ARC-Easy acc / norm (10 questions) | 70.0 / 40.0 | 70.0 / 40.0 | 70.0 / 50.0 |
| HellaSwag acc / norm (7 items) | 71.4 / 71.4 | 71.4 / 71.4 | 42.9 / 85.7 |

The vocabulary pruning removes 1.61 billion weights from 9B's two tables. The
held-out WikiText test file needs 0.52% more tokens. No pruned bf16 control
was run, so the vocabulary's and quantization's effects aren't separated.
Methods, limits and raw logs:

- [BENCHMARKS_9B.md](BENCHMARKS_9B.md)
- [results/9b/measurements.json](results/9b/measurements.json)
- [results/calibration.json](results/calibration.json)

## Inside Qwen3.5-0.8B

```
token ─► embed (248,320 × 1024, tied with the LM head)
          │
          ├─ 24 × ┬─ RMSNorm ─► Gated DeltaNet      (layers 0-2, 4-6, ...: 18 layers)
          │       │             or gated attention   (every 4th layer: 6 layers)
          │       └─ RMSNorm ─► SwiGLU MLP 1024 → 3584 → 1024
          │
          └─ RMSNorm ─► LM head ─► logits ─► sampler
```

- **Gated DeltaNet** is linear attention with memory. Each of its 16 heads keeps
  a 128×128 matrix S that it reads and updates every token:
  `S ← decay·S;  δ = β(v − Sᵀk);  S += k δᵀ;  out = Sᵀq`.
  A short causal conv (k = 4) runs before it and a gated RMSNorm after it. Its
  state is a fixed 18 MB, however long the context.
- **Gated attention** has 8 query heads × 256 dims and 2 KV heads, q/k RMSNorm,
  RoPE on a quarter of each head (θ = 10⁷), and a sigmoid output gate. Its f16
  KV cache costs 12 KB per token.
- **Parameters: 752M.** Embedding 254M (34%), MLP 264M (35%), DeltaNet 190M
  (25%), attention 44M (6%).

9B has the same design with 32 layers, width 4096, 16 query and 4 KV heads,
and 16 key and 32 value DeltaNet heads.

Interactive explainers, built with real numbers from this code:

- [Attention in Slow Motion](https://claude.ai/artifact/9iFSq1HpFzv5nEct7uYCJm): step-through animations of plain and gated attention
- [Qwen3.5-0.8B Architecture](https://claude.ai/artifact/JKaPA5myjQzfvEeVXzUt1y): the stack, both mixer types, the MLP, parameters and memory
- [The LM Head](https://claude.ai/artifact/9q95W4iSQxvkYFwhCSXLy7): from the last hidden state to logits to the sampled token
- [Inside a DeltaNet Layer](https://claude.ai/artifact/5Yk8hLPQr6rkeqy5yfoKPw): every matrix multiplication of layer 0, drawn to scale

## Code

| file | what |
|---|---|
| `scripts/export.py` | Hugging Face safetensors → `model.bin` + `tokenizer.bin` (stdlib only) |
| `scripts/shrink.py` | compression: vocabulary, int8/int4, AWQ, mixed precision, low-rank (numpy) |
| `scripts/data.py` | downloads the English corpus and the benchmark sets (pyarrow) |
| `scripts/benchmark.py` | runs the 9B smoke tests and saves the measurements |
| `src/main.c` | `qwen`: command line, chat template, prompt + decode loop |
| `src/bench.c` | `bench`: bpb, mc, calib, count |
| `src/model.c` | loads `model.bin`, lays out the state, wires the forward pass |
| `src/ops_ref.c` | reference C implementation of every op (the readable spec of the math) |
| `src/backend.h` | compute backend interface and matrix formats |
| `src/backend_cpu.c` | plain C backend (the reference ops) |
| `src/backend_accelerate.c` | Accelerate BLAS matvec + reference ops |
| `src/backend_metal.m` + `src/kernels.metal` | Metal: every op is a GPU kernel |
| `src/tokenizer.c` | byte-level BPE + a hand-written version of Qwen's pre-tokenizer regex |
| `src/sampler.c` | greedy / temperature + top-p |
| `tests/` | synthetic checkpoints through export, compression and inference |

**Tests:** `make test` runs synthetic tied and untied checkpoints through
export, vocabulary pruning, bf16/int8/int4, and mixed precision with AWQ and
low-rank corrections. It then checks C inference, reset, snapshot/restore, and
loading of older files. It also checks that AWQ's rescaling leaves a layer's
output unchanged. `KYSE_TEST_METAL=1 make test` also compares Metal with the
CPU reference.

**File format:** `model.bin` is a small header followed by 64-byte-aligned
tensors, memory-mapped at load time. Each version adds to the header:

- v2: the matrix formats;
- v3: an untied output head;
- v4: a format per matrix and optional low-rank corrections.

`export.py` writes v3, `shrink.py` writes v4, and the loader reads v1–v4.

**Adding a backend:** `src/model.c` only wires ops together. Every op goes
through the `Backend` vtable in `src/backend.h`: embed, matvec, rmsnorm, add,
swiglu, attention, deltanet and sync.

- Ops may run asynchronously; the CPU reads memory (the logits) only after
  `sync()`.
- The quickest start is to implement `matvec` only and reuse the `ref_*` ops
  for the rest, as `src/backend_accelerate.c` does.
- To register it, add an `extern Backend backend_xxx;` and list it in
  `backends[]`.

## Notes

- Tokens are processed one at a time, prompt included (no batched prefill).
- Activations and the recurrent state are f32; the KV cache is f16.
- The pre-tokenizer treats every non-ASCII byte as a letter. That's exact for
  ASCII and very close for everything else. There's no NFC normalization.
- The chat template is hard-coded to non-thinking mode.
- Chat keeps the whole conversation in the context. When it fills up, use
  `/reset`, or start with a larger `-c`.
- The vision tower and the MTP head are ignored.
- It's developed and measured on macOS (Apple M1 and M5). The CPU backend is
  plain C23 and should build elsewhere, but only macOS has been tested.

## License

The code is MIT (see `LICENSE`). The Qwen3.5 weights are Apache-2.0, from
[Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) and
[Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B). The datasets that
`scripts/data.py` downloads keep their own licenses: WikiText (CC BY-SA),
Dolly-15k (CC BY-SA 3.0), TinyStories (CDLA-Sharing), ARC (CC BY-SA 4.0) and
HellaSwag (MIT).
