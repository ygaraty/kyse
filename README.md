# Kyse

**Simple C inference for Qwen3.5-0.8B and Qwen3.5-9B**, the hybrid linear-attention models
(text only). A small C, Metal and Python implementation, with no runtime
dependencies beyond the system frameworks.

- **Three backends** behind one small interface: plain C (the readable reference),
  Apple Accelerate, and Metal. On Metal every op is a GPU kernel and the CPU waits
  once per token.
- **Shrink it for English.** Prune rarely used vocabulary rows and quantize the
  matrices. On 9B, English Q8 is **56% smaller and 2.3× faster** in the M5 smoke
  tests below.
- **Measure it.** `bench` reports bits per byte, ARC-Easy and HellaSwag, so every
  change can be checked.

Qwen3.5-0.8B on Apple M1 (the 9B measurements use an M5):

```
                     size      speed (M1)     bits/byte   ARC-Easy   HellaSwag
original (bf16)    1,506 MB    33 ms/token      1.061       57.6%      45.0%
english int8         588 MB    14 ms/token      1.077       61.2%      46.2%
english int4         339 MB    12 ms/token      1.139       57.2%      37.5%
english int4 mixed   408 MB   ~13 ms/token      1.077       59.2%      41.2%
```

The last row is int4 with AWQ and three matrix types kept at int8 (see
[Better int4](#better-int4-awq-mixed-precision-low-rank)): int8's bits per byte
at 69% of its size.

## Quick start

```
hf download Qwen/Qwen3.5-0.8B --local-dir ../models/qwen3.5-0.8b
python3 scripts/export.py ../models/qwen3.5-0.8b .   # -> model.bin, tokenizer.bin (stdlib only)
make
./qwen model.bin                              # interactive chat (/reset, ctrl-d to quit)
./qwen model.bin -y "You are a pirate."      # chat with a system prompt
./qwen model.bin -i "Why is the sky blue?"   # one-shot
./qwen model.bin -b cpu -t 0 -i "..."        # reference CPU backend, greedy
```

`./qwen` with no arguments lists every option (`-v` prints token ids and speed).

## Qwen3.5-9B (text only)

The same C executable handles 9B from its exported configuration. The official
[9B config](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json)
has 32 layers, width 4096, 16 query / 4 KV heads, and 16 key / 32 value
DeltaNet heads. Unlike 0.8B, its output head is separate from its embedding.
The exporter reads all shards named in `model.safetensors.index.json` and streams
BF16 matrices without loading the whole checkpoint into Python memory.

```sh
hf download Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --local-dir models/qwen3.5-9b
python3 scripts/export.py models/qwen3.5-9b models/9b
make
./qwen models/9b/model.bin -b metal             # interactive chat on macOS
./qwen models/9b/model.bin -b cpu -t 0 -i "Why is the sky blue?"
```

Apply the same English vocabulary pruning and quantization workflow:

```sh
pip install numpy pyarrow                    # shrink.py and data.py
python3 scripts/data.py data
./bench count models/9b/tokenizer.bin data/english.txt > data/counts-9b.txt
python3 scripts/shrink.py models/9b/model.bin models/9b/tokenizer.bin data/counts-9b.txt en-9b-q8 --quant q8
python3 scripts/shrink.py models/9b/model.bin models/9b/tokenizer.bin data/counts-9b.txt en-9b-q4 --quant q4
./qwen en-9b-q8/model.bin -b metal              # recommended compact chat model
./qwen en-9b-q4/model.bin -b metal              # smaller, more aggressive option
```

Chat remembers previous turns. Type `/reset` to clear the conversation and
press Ctrl+D to exit. Use `-y "Your system prompt"` to set the assistant
instructions, or `-i "Your question"` for a one-shot reply.

Both embedding and output-head rows are pruned with exactly the same token map.
`--embed-quant` controls both tables. The shrinker memory-maps its source;
conversion and quantization still need working memory for individual matrices.
Allow roughly 18 GB for the exported BF16 text weights plus runtime state;
pruning and quantization reduce that. The f16 KV cache costs 32 KiB per context
token and DeltaNet recurrent state about 48 MiB. Accelerate expands BF16 weights
to FP32 and therefore needs substantially more memory. Metal uses page-aligned
views of the weight file to avoid a single checkpoint-sized GPU buffer; each
individual matrix must fit the device's buffer limit.

New exports use format v3, adding an untied-head flag and, when needed, an
output matrix after the final norm. `shrink.py` writes v4, which adds a format
per matrix (for mixed precision) and optional low-rank corrections. The C loader
continues to accept v1–v3 models. Earlier executables cannot read newer files.

`make test` exercises synthetic sharded tied/untied checkpoints through export,
vocabulary pruning, BF16/Q8/Q4, mixed precision with AWQ and low-rank
corrections, C CPU inference, reset, snapshot/restore, and v2 loading (Python +
numpy + a C23 compiler). It also checks that AWQ's rescaling leaves a layer's
output unchanged. On macOS, run
`KYSE_TEST_METAL=1 make test` to compare Metal with the CPU reference as well.
Vision and MTP remain excluded.

### 9B vocabulary-pruning experiments

Calibration used 44.9 MB of English from WikiText-2 train, Dolly-15k and
TinyStories validation. Keeping tokens seen at least twice, all single bytes,
special tokens, chat-template words and intermediate BPE tokens reduced the
vocabulary from **248,320 to 52,352**. On 9B, applying this map to both the
embedding and separate output head removes about **1.61 billion weights**.
The held-out WikiText test file needs 298,584 tokens instead of 297,053
(**+0.52%**); it was not used to select the vocabulary.

Apple M5, 24 GB, Metal. File sizes are decimal GB; generation is the median of
three greedy 64-token trials. Q4 retains Q8 embedding and output tables.

| 9B variant | Size | Generation | Wiki bits/byte ↓ | ARC acc / norm | HellaSwag acc / norm |
|---|---:|---:|---:|---:|---:|
| Original BF16 | 17.91 GB | 6.8 tok/s | 0.8260 | 70.0% / 40.0% | 71.4% / 71.4% |
| English Q8 | 7.81 GB | 15.8 tok/s | 0.8297 | 70.0% / 40.0% | 71.4% / 71.4% |
| English Q4, Q8 tables | 4.35 GB | 26.2 tok/s | 0.8312 | 70.0% / 50.0% | 42.9% / 85.7% |

**Q8 is the conservative default:** 56.4% smaller, 2.32× faster, and a 0.45%
increase in bits per byte on this sample. Q4 is 75.7% smaller and 3.85× faster,
but changed answer rankings more. Its higher normalized HellaSwag score and
lower raw score come from different length preferences, and seven items are
too few to establish a general quality change.

These are smoke tests: 3,423 bytes of held-out WikiText, 10 ARC-Easy questions,
and seven HellaSwag items. The original HellaSwag run was stopped after seven
complete items; compact models used that same prefix. These 9B experiments
combine vocabulary pruning and quantization; no pruned BF16 control was run,
so the individual effects are not isolated. Multilingual quality was not tested.
The larger 0.8B experiments below include a pruned BF16 control.

See [the complete 9B report](BENCHMARKS_9B.md) for methods and limitations,
[measurements and raw logs](results/9b/measurements.json), and
[calibration metadata and dataset hashes](results/calibration.json).
Reproduce the 9B checks with:

```sh
python3 scripts/benchmark.py models/9b en-9b-q8 en-9b-q4 --hellaswag-items 7
```

## Make it smaller: English only + int8

Most of the 248,320-token vocabulary is other scripts (CJK alone is 22%).
Each unused row costs embedding memory and LM-head work on every token.
The 0.8B model ties these two tables; 9B has separate tables that are pruned
together. The measurements in this section describe 0.8B.

```
pip install pyarrow                                    # only for data.py
python3 scripts/data.py data                                   # corpus + benchmark sets (~52 MB)
./bench count tokenizer.bin data/english.txt > data/counts.txt
python3 scripts/shrink.py model.bin tokenizer.bin data/counts.txt en-q8 --quant q8   # needs numpy
./qwen en-q8/model.bin
```

What `scripts/shrink.py` does:

- **Vocabulary: 248,320 → 52,352 tokens.** It keeps the tokens that appear at
  least twice in 45 MB of English, plus:
  - all 256 single bytes, so any text still encodes (unseen words just take more
    tokens);
  - the special tokens and the chat-template words;
  - every intermediate BPE token of the kept ones, so text made of kept tokens
    tokenizes *exactly* as before (verified: 2,504,609 tokens on WikiText train,
    same as the original).

  On 0.8B this drops 203M parameters. On unseen English the new vocabulary needs only
  0.5% more tokens.
- **Quantization.** Every matrix becomes int8 (`q8`) or int4 (`q4`) with one f16
  scale per 32 weights. For int4, each group's scale is picked from a few
  candidates to minimize the squared error, and the embedding stays int8 by
  default (`--embed-quant`).

  The Metal and CPU backends run the quantized formats directly; Accelerate falls
  back to the reference op for them. Token ids change, so a shrunk `model.bin`
  comes with its own `tokenizer.bin`.

## Benchmarks: Qwen3.5-0.8B

Apple M1 (16 GB), Metal backend. To reproduce, run `scripts/data.py` (above), then:

```
./bench bpb model.bin data/wiki.test.txt -k 8        # bits per byte, 8 chunks of ~4 KB
./bench mc  model.bin data/arc_easy.txt -k 250       # multiple choice
./bench mc  model.bin data/hellaswag.txt -k 80
```

| | original | english bf16 | **english int8** | english int4 |
|---|---|---|---|---|
| file size | 1,506 MB | 1,104 MB | **588 MB** | 339 MB |
| memory in use (4K context) | ~1.60 GB | ~1.19 GB | **~0.67 GB** | ~0.42 GB |
| generation speed | 30 tok/s | 42 tok/s | **70 tok/s** | 87 tok/s |
| WikiText-2 bits/byte ↓ | 1.0611 | 1.0766 | **1.0771** | 1.1393 |
| WikiText-2 top-1 next token | 40.6% | 40.3% | **40.3%** | 38.6% |
| ARC-Easy acc (250 questions, ±3.1) | 57.6% | 61.2% | **61.2%** | 57.2% |
| ARC-Easy acc_norm | 50.4% | 54.0% | **54.0%** | 52.8% |
| HellaSwag acc (80 items, ±5.5) | 37.5% | 36.2% | **36.2%** | 33.8% |
| HellaSwag acc_norm | 45.0% | 45.0% | **46.2%** | 37.5% |

Memory is the weights plus the state for a 4K context (71 MB, of which the
f16 KV cache is 50 MB). The quality rows were measured with an f32 KV cache,
except int8's, which was re-measured after the switch to f16: identical, apart
from one ARC-Easy question (acc_norm 53.6% → 54.0%).

How to read this:

- **int8 costs nothing measurable**, on top of the vocabulary change.
- **The English vocabulary costs 1.5% bits/byte.** Rare words now split into
  pieces the model never saw in training. Its ARC-Easy gain is not the model
  getting smarter: with fewer competing tokens, the answer choices get scored a
  little differently.
- **Plain int4 rounding hurts.** It loses 5.8% bits/byte and 7.5 points of
  HellaSwag acc_norm. The next section fixes most of that.
- **Bits per byte is the sensitive measure.** Unlike perplexity, it doesn't depend
  on the vocabulary. The multiple-choice samples are small and only catch large
  changes.

`bench mc` scores every choice by its log-likelihood after the context, as
lm-evaluation-harness does. `acc` uses the total log-likelihood; `acc_norm`
divides it by the choice's length in bytes. The context runs once and each
choice branches from a snapshot of the recurrent state (`model_snapshot`).

## Better int4: AWQ, mixed precision, low-rank

Ideas from the LLM-deployment lectures of MIT's EfficientML course, tried on
0.8B. Every variant below has the English vocabulary and an int8 embedding.

```
./bench calib model.bin data/english.txt data/calib.bin -k 32 -l 1024   # activation statistics
python3 scripts/shrink.py model.bin tokenizer.bin data/counts.txt en-q4-mixed \
    --quant q4 --awq data/calib.bin --q8 '*.wo,*.w_out,*.w_down'
```

- **AWQ** (`--awq`): `bench calib` records the mean |x| of every input channel of
  every matrix over 9K tokens of English. `shrink.py` scales each matrix's input
  channels by s = mean|x|^α before quantizing and folds 1/s into whatever
  produced the input: the RMSNorm weights, or the rows of `w_up` for `w_down`.
  α is picked per matrix group by a quick estimate of the output error. The
  file format and kernels don't change.
- **Mixed precision** (`--q8 PATTERNS`): chosen matrices stay int8, e.g.
  `'*.w_down'` (every layer) or `'0.*'` (all of layer 0).
- **Low-rank correction** (`--lowrank PATTERNS --rank 16`, SVDQuant): the 16
  largest directions of a matrix are kept in bf16 and only the rest is
  quantized. `model.c` applies the correction with two small extra matvecs.

Apple M1, Metal. Speed is the best of 7 runs on a busy machine (int8 ran at
70 tok/s on a quiet one), so compare within the column only.

| int4 variant | size | bits/byte ↓ | int4 gap closed | top-1 | ARC acc / norm | HellaSwag acc / norm | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| english int8 (reference) | 588 MB | 1.0771 | | 40.3% | 61.2 / 54.0 | 36.2 / 46.2 | 63 |
| plain | 339 MB | 1.1393 | 0% | 38.6% | 57.2 / 52.8 | 33.8 / 37.5 | 79 |
| AWQ | 339 MB | 1.1176 | 35% | 38.6% | 58.4 / 53.2 | 35.0 / 41.2 | 76 |
| AWQ + rank 16 on `wo`, `w_out` | 341 MB | 1.1081 | 50% | 39.4% | | | 75 |
| AWQ + int8 `wo`, `w_out` | 364 MB | 1.0892 | 81% | 40.0% | 58.0 / 55.2 | 35.0 / 40.0 | 76 |
| **AWQ + int8 `wo`, `w_out`, `w_down`** | **408 MB** | **1.0773** | **~100%** | **40.6%** | **59.2 / 55.2** | **35.0 / 41.2** | **72** |

What we learned:

- **The out projections are the weak spot.** `wo` and `w_out` read gated inputs,
  so AWQ can't fold a scale into them. Their int4 error (activation-weighted) is
  twice that of every other matrix: a few input channels of `w_out` have 50 to
  100 times the median magnitude. Keeping just these 50M weights at
  int8 (+25 MB) closes 81% of the gap; adding `w_down` closes all of it.
- **AWQ helps, but less than in the paper.** Most of this model's activations
  have no large outliers, so rescaling cuts each matrix's error by about 10%.
  How the error is weighted matters: weighting by mean x², which those few
  huge channels dominate, closed only 14% of the gap; mean |x|² closed 35%.
- **Low-rank corrections don't pay off here.** On `wo`/`w_out` they recover a
  third as much as int8 for a tenth of the bytes (+2 MB). On every matrix they
  cost 20 MB and do *worse* than on those two alone (1.1118), and the extra
  matvecs slow decoding (19.6 vs 15.1 ms/token in the bits-per-byte runs).
- **The multiple-choice samples are too small to rank these variants.** Every
  int4 variant is within the error bars of the others. The HellaSwag gap to
  int8 (41 vs 46) is under one standard error.
- **The KV cache is now f16** (all backends). This halves it to 24 KB → 12 KB
  per token, with the same bits per byte at 4K and 16K-byte contexts.

Raw output: [results/0.8b/int4-experiments.txt](results/0.8b/int4-experiments.txt).

## Qwen3.5-0.8B in one screen

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
  RoPE on a quarter of each head (θ = 10⁷), and a sigmoid output gate. Its f16
  KV cache costs 12 KB per token.
- **Parameters: 752M.** Embedding 254M (34%), MLP 264M (35%), DeltaNet 190M (25%),
  attention 44M (6%).
- **Every token is memory-bound.** It reads all the weights once, so file size ≈
  speed.

## Files

C and Metal sources and headers live in `src/`; Python utilities live in
`scripts/`. Build outputs remain `./qwen` and `./bench`, so chat and benchmark
commands stay the same.

| file | what |
|---|---|
| `src/main.c` | command line, chat template, prefill + decode loop |
| `src/model.c` | loads `model.bin`, lays out the state, wires the forward pass |
| `src/ops_ref.c` | reference C implementation of every op (the readable spec of the math) |
| `src/backend.h` | compute backend interface, matrix formats |
| `src/backend_cpu.c` | plain C backend (just the reference ops) |
| `src/backend_accelerate.c` | Accelerate BLAS matvec + reference ops |
| `src/backend_metal.m` + `src/kernels.metal` | Metal: every op is a kernel, one CPU wait per token |
| `src/tokenizer.c` | byte-level BPE + hand-written version of Qwen's pre-tokenizer regex |
| `src/sampler.c` | greedy / temperature + top-p |
| `scripts/export.py` | HF safetensors → flat `model.bin` / `tokenizer.bin` (stdlib only) |
| `scripts/shrink.py` | `model.bin` → English-only vocabulary + int8/int4 matrices, AWQ, mixed precision, low-rank (numpy) |
| `src/bench.c` | `bpb`: bits per byte · `mc`: multiple choice · `calib`: activation statistics for AWQ · `count`: token counts for `scripts/shrink.py` |
| `scripts/data.py` | downloads the corpus and the benchmark sets (pyarrow) |
| `scripts/benchmark.py` | runs generation and quality smoke tests and saves measurements |
| `tests/` | synthetic checkpoint tests and CPU/Metal parity checks |

## Interactive explainers

Built while writing this, with real numbers from this code:

- [Attention in Slow Motion](https://claude.ai/artifact/9iFSq1HpFzv5nEct7uYCJm): step-through animations of plain and gated attention
- [Qwen3.5-0.8B Architecture](https://claude.ai/artifact/JKaPA5myjQzfvEeVXzUt1y): the stack, both mixer types, the MLP, parameters and memory
- [The LM Head](https://claude.ai/artifact/9q95W4iSQxvkYFwhCSXLy7): from the last hidden state to logits to the sampled token, with real logits
- [Inside a DeltaNet Layer](https://claude.ai/artifact/5Yk8hLPQr6rkeqy5yfoKPw): every matrix multiplication of layer 0, drawn to scale with real weights

## Adding a backend

`src/model.c` only wires ops together; every op goes through the `Backend` vtable in
`src/backend.h` (embed, matvec, rmsnorm, add, swiglu, attention, deltanet, sync).
Ops may be async; the CPU only reads memory (the logits) after `sync()`.
Matrices arrive as a `Mat` (data, format, shape); a backend that only handles
bf16 can fall back to `ref_matvec` for the quantized formats.

The quickest start is to implement `matvec` only and reuse the `ref_*` ops for
the rest (see `src/backend_accelerate.c`). Move the other ops over when you want to
remove CPU round trips. Add an `extern Backend backend_xxx;` and list it in
`backends[]` in `src/backend.h`.

## Notes / simplifications

- One token at a time, prompt included (no batched prefill).
- Weights are bf16 (or int8/int4 after `scripts/shrink.py`); activations and the
  recurrent state are f32, the KV cache f16.
- The pre-tokenizer treats every non-ASCII byte as a letter. It's exact for ASCII
  and very close for everything else. There's no NFC normalization.
- The chat template is hard-coded to non-thinking mode; `-r` gives raw completion.
- Chat keeps the whole conversation in the context (KV cache + DeltaNet state).
  When it fills up, use `/reset` or start with a larger `-c`.
- The vision tower and the MTP head are ignored.
- Developed and measured on macOS (Apple M1 and M5). The CPU backend is plain C23 and
  should build elsewhere, but only macOS has been tested.

## License

The code is MIT (see `LICENSE`). The Qwen3.5 weights are Apache-2.0, from
[Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) and
[Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B). The datasets
`scripts/data.py` downloads keep their own licenses: WikiText (CC BY-SA), Dolly-15k
(CC BY-SA 3.0), TinyStories (CDLA-Sharing), ARC (CC BY-SA 4.0) and HellaSwag (MIT).
