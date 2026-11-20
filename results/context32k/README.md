# Qwen3.5-9B Q8: 32k context smoke test

Measured on 2026-11-20, Apple M5, 24 GB unified memory, Metal backend,
through the in-process CFFI engine. Context capacity: **32,768 tokens**.

This recorded run predates the FP16 KV-cache merge and used **FP32**. The
current runtime stores KV in FP16 (1 GiB at 32k instead of 2 GiB). The results
below remain the original FP32 measurements; a full-window FP16 run has not
been repeated as part of the merge. The Python server and binding have since
moved to the sibling Piara project.

## Result

**Passed:** the model returned `strawberry` exactly after a
**32,736-token prompt**. The verification word occurs once,
near the start of the prompt; the question is at the end. The complete prompt
includes the non-thinking chat template and WikiText-2 test text.

| Check | Input tokens | Output tokens | TTFT | Observed decode rate |
|---|---:|---:|---:|---:|
| Short control at 32k allocation | 42 | 3 | 6.40 s | 15.2 tok/s |
| Near-full context retrieval | 32,736 | 3 | 3272.98 s (54 min 33 s) | 7.2 tok/s |
| Short request after over-window rejection | 42 | 3 | 2.83 s | 15.1 tok/s |

All three answers were exactly `strawberry` and ended with EOS.
The long request used 32,739 total input/output tokens;
it exercised a near-full window, not every possible boundary position.
The native prompt-token count matched the independently prepared count.
An over-window prompt was rejected before inference, and the same model handle
remained usable afterward.

The decode rates above cover only the two intervals between three ASCII output
tokens; they are smoke measurements, not a stable throughput benchmark. The
first control also includes initial GPU weight-page warmup. Long-prompt total
time was 3273.40 seconds, with no other inference run in parallel.

## Memory and prefill

The FP32 KV cache is **2 GiB at 32k**:
`8 full-attention layers × 32768 positions × 4 KV heads × 256 dimensions × 2 (K/V) × 4 bytes`.
This is in addition to the 7.81 GB Q8 weight file, about 48 MiB of recurrent
state, activations and Python/Metal overhead. The OS reported peak process RSS
of 2.13 GiB; process RSS does not measure total GPU/shared
weight memory and must not be read as the full model's RAM requirement.

Prefill remains one token at a time. The first 1,024 tokens took 67.4 seconds;
the final 991-token block took 134.8 seconds. The CFFI API now synchronizes
prefill every 128 tokens and gives each chunk its own Metal autorelease pool,
so pending command buffers and temporary objects do not grow with the entire
prompt. This bounds queued work; it does not implement batched prefill.
`KYSE_PREFILL_PROGRESS=1` enables progress messages every 1,024 tokens.

The practical result is that **32k capacity works on this machine, but near-32k
inputs have very high TTFT with the current serial prefill path**. Batched
prefill is the main follow-up for reducing that latency.

## Scope

This is one synthetic retrieval probe embedded in real text, not a comprehensive
long-context accuracy evaluation. A short control was required to pass before
the long run. An initial uppercase/numeric-code probe failed even at 4k and was
abandoned before completing its long run; those results are not used here.
The tested model is the English-pruned Q8 variant; Q4 and unpruned BF16 were
not tested at 32k in this run.

## Reproduce

The benchmark lives in the sibling Piara project. From the Kyse root, with the
compact model and `data/wiki.test.txt` available:

```sh
make library bench
uv run --project ../piara piara-benchmark-context \
  --model en-9b-q8/model.bin --backend metal \
  --context 32768 --prompt-tokens 32736 --max-tokens 32
```

The current script writes new runs to `results/context32k-fp16/` by default,
keeping this historical FP32 run intact. Re-running the workload on current
code measures FP16, not the earlier results above.

Run the API with a long enough timeout:

```sh
uv run --project ../piara piara-server \
  --model en-9b-q8/model.bin --backend metal --context 32768 --timeout 7200
```

A client doing near-full-window jobs needs a correspondingly long timeout,
for example `AsyncOpenAI(..., timeout=7200)`. CFFI timeout/cancellation remains
cooperative at native-call boundaries.

[Measurements](measurements.json) and [raw progress log](run.log).
The generated prompt is retained locally in `prompt.txt` and can be regenerated
with the script; its SHA-256 is `a0386cf2f84df600a3be6b445b2c81961657fc701e4b078dfeeb4ba5efb9938a`.
WikiText provenance and dataset hashes are in [calibration metadata](../calibration.json).
