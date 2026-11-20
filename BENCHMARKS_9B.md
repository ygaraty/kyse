# Qwen3.5-9B compaction smoke benchmarks

Measured on 2026-11-20, Apple M5 (10-core GPU), 24 GB unified memory, Metal backend. Text only; no vision or MTP.

Source: [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.

## Results

Sizes are decimal GB. Generation is the median of three greedy 64-token CLI trials using the same 27-token chat prompt and a 512-token context. The CLI reports decode timing separately from prompt processing; these are its reported rates, rounded to 0.1 tok/s. Runs were sequential.

| Model | GB | Smaller | Decode tok/s | Speedup | Wiki bits/byte ↓ | ARC acc / norm | HellaSwag acc / norm |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original BF16 | 17.91 | 0.0% | 6.8 | 1.00× | 0.8260 | 70.0% / 40.0% | 71.4% / 71.4% |
| English Q8 | 7.81 | 56.4% | 15.8 | 2.32× | 0.8297 | 70.0% / 40.0% | 71.4% / 71.4% |
| English Q4 (Q8 tables) | 4.35 | 75.7% | 26.2 | 3.85× | 0.8312 | 70.0% / 50.0% | 42.9% / 85.7% |

Q8 is the conservative default from this smoke test: 56.4% smaller, 2.32× faster generation, and only a 0.45% increase in bits per byte. Its multiple-choice predictions matched the original on these samples.

Q4 is 75.7% smaller and 3.85× faster, with a 0.63% increase in bits per byte here. Its answer rankings changed more: HellaSwag raw accuracy lost two answers while length-normalized accuracy gained one. These seven items cannot establish an overall quality change or a quality improvement. Use the Q8 version when choosing the less aggressive compression.

## Samples and limits

- WikiText-2 test: two chunks, requested 1,024 bytes each and extended to line boundaries, totaling 3,423 bytes. Original: 832 tokens, 53.1% top-1, 181.1 ms/token. Q8: 834 tokens, 53.4%, 68.6 ms/token. Q4: 834 tokens, 52.6%, 41.3 ms/token.
- ARC-Easy: first 10 test items. Raw accuracy uses total answer log-likelihood; norm divides by answer length in bytes.
- HellaSwag: first seven items of the deterministically shuffled validation file (seed 0). The original run requested eight items but was stopped after roughly 13 minutes, after seven complete items had printed. Its completed seven-item prefix was retained; both compact models were then run on exactly that prefix. The original log is intentionally incomplete and marked `partial_run` in the JSON. No eight-item score is claimed.
- These are smoke tests. The samples are too small to establish general quality parity; the multiple-choice percentages have large sampling uncertainty. Generation outputs differ, so decode trials are a practical generation comparison rather than identical token replay. Runtime also depends on system memory pressure and caching; peak memory was not measured.

## Compaction

A 44,858,696-byte English training/calibration corpus selected 52,352 of 248,320 vocabulary rows. The embedding and separate output head use the same token mapping. Q8 quantizes every matrix to grouped int8; Q4 uses grouped int4 for layer matrices and int8 for both vocabulary tables. Norms and recurrent state stay FP32.

On the entire held-out WikiText test file, the original tokenizer uses 297,053 tokens and the compact tokenizer uses 298,584 (+0.52%). Evaluation text was not used to select the vocabulary.

Artifacts:

- `models/9b/model.bin` and `models/9b/tokenizer.bin`: original exported text model.
- `en-9b-q8/model.bin` and `en-9b-q8/tokenizer.bin`: recommended compact version.
- `en-9b-q4/model.bin` and `en-9b-q4/tokenizer.bin`: smaller, more aggressive version.
- [Machine-readable measurements and raw logs](results/9b/measurements.json). Raw logs are alongside the JSON.
- [Calibration metadata and dataset SHA-256 fingerprints](results/calibration.json).

## Reproduce

From the repository directory, after downloading the checkpoint and running `scripts/data.py`:

```sh
make
python3 scripts/export.py models/qwen3.5-9b models/9b
./bench count models/9b/tokenizer.bin data/english.txt > data/counts-9b.txt
python3 scripts/shrink.py models/9b/model.bin models/9b/tokenizer.bin data/counts-9b.txt en-9b-q8 --quant q8
python3 scripts/shrink.py models/9b/model.bin models/9b/tokenizer.bin data/counts-9b.txt en-9b-q4 --quant q4
python3 scripts/benchmark.py models/9b en-9b-q8 en-9b-q4 --hellaswag-items 7
./qwen en-9b-q8/model.bin -b metal -i "Explain how a rainbow forms."
```

The runner writes live raw logs, measurements after each completed model, and bounds each command to 900 seconds by default. `--reuse-logs` explicitly reuses logs from identical model/settings; an interrupted multiple-choice log is accepted only when its last fully scored item matches the requested sample size.

The Metal backend now maps only each bound tensor, rather than large overlapping suffixes of the checkpoint, and surfaces command-buffer errors. CPU/Metal parity tests cover synthetic sharded tied/untied models, BF16/Q8/Q4, state reset, snapshots, and old-format loading:

```sh
make test
KYSE_TEST_METAL=1 make test
```
