# Local completions server

Async Python HTTP server for local data jobs. It supports the text subset of
OpenAI's Completions and Chat Completions interfaces, including streaming.
By default, CFFI loads `libkyse` directly into Python. The same C model,
tokenizer, sampler and CPU/Accelerate/Metal backends perform inference. An
opaque C handle owns the weights and state; Python does not mirror internal
model structs. One dedicated inference thread keeps native work off the HTTP
event loop, and requests queue through it. Every request resets model state;
chat history comes from the supplied `messages`.

## Start

From the repository root (Python 3.11+):

```sh
make library
python3 -m venv server/.venv
server/.venv/bin/python -m pip install -r server/requirements.txt
server/.venv/bin/python server/server.py \
  --model en-9b-q8/model.bin --backend metal --model-name kyse --port 8000
```

The server listens on `127.0.0.1:8000`. Use `--backend cpu` on non-Apple systems,
or choose the Q4 model with `--model en-9b-q4/model.bin`. The matching
`tokenizer.bin` must be beside the model. `--context` defaults to 4096;
`--timeout` defaults to 300 seconds per generation, excluding queue wait.
Ctrl+C stops the server and closes its native model. Custom stops and client
cancellation keep weights loaded; the next request resets generation state.
CFFI cancellation and timeouts are cooperative: an in-flight native call must
return before its state can be reset or freed. HTTP remains asynchronous while
waiting, but a slow first-token/prefill call can delay timeout completion.

`--library /path/to/libkyse.dylib` (macOS) or `.so` (Linux) selects another build.
The default is `build/libkyse.dylib` or `build/libkyse.so` in this repository.
Only one model may be loaded per process because backend arenas are global.
Native load/generation errors return Python exceptions through the C API;
CLI binaries retain their existing fatal-error behavior.

For process isolation and hard timeout/cancellation, the earlier worker engine
is still available:

```sh
make qwen-worker
server/.venv/bin/python server/server.py \
  --engine worker --model en-9b-q8/model.bin --backend metal
```

Worker-mode cancellation discards the subprocess; the next request reloads it.

This is a local service with no authentication or TLS. The Python client can
use any placeholder API key. Keep the default loopback binding for local jobs.

## Larger context windows

For a 32,768-token window on the Q8 model:

```sh
server/.venv/bin/python server/server.py \
  --model en-9b-q8/model.bin --backend metal --context 32768 --timeout 7200
```

9B's FP32 KV cache uses 64 KiB per context token: about **2 GiB at 32k**,
in addition to the 7.81 GB Q8 weight file, recurrent state and runtime overhead.
Context capacity does not imply fast prompt ingestion: prefill still runs one
token at a time. A near-full window can require tens of minutes before the
first output token, so both server and client timeouts need to allow for that.
CFFI prefill synchronizes in 128-token chunks to bound pending GPU commands
and Metal temporary-object residency. Set `KYSE_PREFILL_PROGRESS=1` to log
prefill progress to stderr every 1,024 tokens.

Run the reproducible near-window retrieval test with:

```sh
server/.venv/bin/python scripts/benchmark_context.py \
  --model en-9b-q8/model.bin --context 32768 --prompt-tokens 32736
```

It counts the complete chat-template prompt with the C tokenizer, checks a
short retrieval control, fills the long prompt with WikiText, then checks
retrieval and recovery after an over-window rejection. Outputs go to
`results/context32k/`. The default source text is `data/wiki.test.txt` from
`scripts/data.py`. One retrieval probe is a functionality smoke test, not a
comprehensive long-context quality evaluation. The [recorded Q8/Metal run](../results/context32k/README.md)
retrieved the word correctly from 32,736 prompt tokens; TTFT was 54 min 33 s.

## Completions

```sh
curl http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"kyse","prompt":"The capital of France is","max_tokens":16,"temperature":0}'
```

The completions endpoint performs raw text continuation without a chat template
or prompt echo. For instruction-following jobs, use chat completions instead:

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"kyse","messages":[{"role":"system","content":"Answer concisely."},{"role":"user","content":"What is 2+2?"}],"max_tokens":32,"temperature":0}'
```

Responses include OpenAI-shaped `id`, `object`, `created`, `model`, `choices`,
`finish_reason` and `usage`. Token counts come from the native C tokenizer and generation engine.
Completion usage counts sampled non-EOS tokens, including tokens consumed while
matching a custom stop string. Finish reason is `stop` for EOS/custom stops or
`length` for a token/context limit.

## Async Python data jobs

Install `openai` in your job's environment, then point its client at the local API:

```python
import asyncio
from openai import AsyncOpenAI

async def main():
    async with AsyncOpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local") as client:
        # Bound outstanding jobs to avoid building a large server queue.
        slots = asyncio.Semaphore(4)

        async def classify(text):
            async with slots:
                result = await client.chat.completions.create(
                    model="kyse",
                    messages=[
                        {"role": "system", "content": "Classify sentiment as positive, negative or neutral. Return only the label."},
                        {"role": "user", "content": text},
                    ],
                    temperature=0,
                    max_tokens=16,
                )
                return result.choices[0].message.content

        labels = await asyncio.gather(*(classify(text) for text in ["Works perfectly", "It broke", "Arrived today"]))
        print(labels)

asyncio.run(main())
```

Concurrent HTTP requests are queued, not batched on the GPU. The C backends use
global state, so this server deliberately runs one inference request at a time.
The model is reused across normal requests and custom stop conditions in both engines.

## Streaming

Set `stream=true` to receive SSE chunks followed by `data: [DONE]`. For usage,
set `stream_options={"include_usage":true}`; the final usage chunk has empty
`choices`. UTF-8 and stop strings spanning token boundaries are handled without
leaking partial stop strings.

```python
stream = await client.chat.completions.create(
    model="kyse",
    messages=[{"role": "user", "content": "Explain a rainbow briefly."}],
    max_tokens=128,
    stream=True,
    stream_options={"include_usage": True},
)
async for chunk in stream:
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
    elif chunk.usage:
        print("\n", chunk.usage)
```

The same streaming option works for `/v1/completions`, with text in
`choices[0].text`. Generation errors after streaming headers have been sent
produce an SSE `error` object and end the stream, without a successful `[DONE]`.

## Supported surface

- `GET /health`: model and busy state.
- `GET /v1/models`: the configured model alias.
- `POST /v1/completions`: one nonempty string `prompt`.
- `POST /v1/chat/completions`: text `messages` with system, developer, user and
  assistant roles; developer messages map to system instructions.
- `model`, `max_tokens`, `temperature` (0–2), `top_p` (0–1), unsigned 64-bit
  `seed`, `stop` (one string or up to four), `stream`, `stream_options.include_usage`,
  `n=1` and optional `user` metadata (accepted, not stored). Chat also accepts
  `max_completion_tokens` as an alternative to `max_tokens`.

Default generation limit is 512 tokens or the configured context size,
whichever is smaller. Prompts exceeding the context window return HTTP 400;
the worker remains available. JSON errors use an OpenAI-shaped `error` envelope.
Timeouts return 504 and worker failures return 503 before streaming starts.
Request bodies are limited to 1 MiB.

Unsupported parameters are rejected explicitly: tools/function calls, JSON
schema/`response_format`, token-ID inputs, prompt batches, `n>1`, logprobs,
penalties, suffix/echo, images/audio and the Responses API. This is API
compatibility for local text jobs, not an implementation of every OpenAI feature.

## Tests

```sh
make test-server SERVER_PYTHON=server/.venv/bin/python
```

Tests run real C inference on tiny synthetic checkpoints through both CFFI and
the worker engine. They cover native ownership, cleanup after invalid model or
tokenizer loads, cancellation while native work is active, model reuse, UTF-8,
custom stops, queued requests, token usage, context errors and response/streaming
formats. No model download or GPU
is required for the tests.

## Use the CFFI binding directly

The same binding can be used in synchronous Python jobs without HTTP. Build
with `make library` and install `server/requirements.txt` first:

```python
import codecs
from server.native import Model

with Model("en-9b-q8/model.bin", backend="metal", context=4096) as model:
    model.begin("The capital of France is", max_tokens=32, temperature=0)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    while (token := model.next_token()) is not None:
        print(decoder.decode(token), end="", flush=True)
    print(decoder.decode(b"", final=True))
    print(model.usage())
```

`begin()` starts an independent raw completion; `next_token()` returns owned
bytes (possibly a partial UTF-8 character) or `None` at EOS/limit. `stop()` ends
a generation without unloading the model; `close()` is idempotent. A context
manager provides deterministic cleanup, with a finalizer as a fallback.
For asynchronous jobs, use the HTTP client or the server's `CFFIEngine`, which
runs calls on a dedicated thread and drains native work before cancellation.

The public C API is in `src/kyse.h`; the Python declarations are in
`server/native.py`. `make library` builds the platform shared library; only the
opaque API is exported. C API calls are serialized by a mutex. Error messages
are thread-local and copied by Python on the thread that made the native call.
The macOS library gives each call an autorelease pool for Metal objects.

Worker protocol reference: comments in `src/worker.c`. HTTP response formats follow
[Completions](https://developers.openai.com/api/reference/resources/completions/methods/create)
and [Chat Completions](https://developers.openai.com/api/reference/resources/chat).
HTTP transport uses [aiohttp](https://docs.aiohttp.org/en/stable/web_quickstart.html)
and bindings use [CFFI](https://cffi.readthedocs.io/en/latest/using.html).
