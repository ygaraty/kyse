"""Local OpenAI-compatible HTTP API backed by the Kyse C library (CFFI)."""
import argparse
import asyncio
import codecs
import contextlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
import json
import logging
import math
from pathlib import Path
import secrets
import time
import uuid

from aiohttp import web

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger('kyse.server')


class APIError(Exception):
    def __init__(self, message, param=None, status=400):
        self.message, self.param, self.status = message, param, status

    def body(self):
        return {'error': {'message': self.message, 'type': 'invalid_request_error' if self.status < 500
                          else 'server_error', 'param': self.param, 'code': None}}


class StopFilter:
    """Hold back possible stop prefixes so a stop split across tokens never leaks."""
    def __init__(self, stops):
        self.stops, self.pending, self.stopped = stops, '', False

    def feed(self, text, final=False):
        self.pending += text
        matches = [self.pending.find(s) for s in self.stops if s in self.pending]
        if matches:
            text = self.pending[:min(matches)]
            self.pending, self.stopped = '', True
            return text
        hold = 0 if final else max((n for s in self.stops for n in range(1, len(s))
                                   if self.pending.endswith(s[:n])), default=0)
        end = len(self.pending) - hold
        text, self.pending = self.pending[:end], self.pending[end:]
        return text


@dataclass
class Generation:
    prompt: str
    max_tokens: int
    temperature: float
    top_p: float
    seed: int
    stops: list[str]
    stream: bool
    include_usage: bool


def parse_request(body, chat, model_name, context):
    if not isinstance(body, dict):
        raise APIError('Expected a JSON object.')
    allowed = {'model', 'max_tokens', 'temperature', 'top_p', 'seed', 'stop', 'stream',
               'stream_options', 'n', 'user'} | ({'messages', 'max_completion_tokens'} if chat else {'prompt'})
    unknown = set(body) - allowed
    if unknown:
        param = sorted(unknown)[0]
        raise APIError(f'Unsupported parameter: {param}', param)
    if body.get('model') != model_name:
        raise APIError(f'Use model={model_name!r}; see /v1/models.', 'model')
    count = body.get('n') if body.get('n') is not None else 1
    if type(count) is not int or count != 1:
        raise APIError('Only n=1 is supported; submit jobs concurrently instead.', 'n')
    if chat:
        messages = body.get('messages')
        if not isinstance(messages, list) or not messages:
            raise APIError('messages must be a nonempty list of text messages.', 'messages')
        turns = []
        for message in messages:
            if not isinstance(message, dict) or set(message) - {'role', 'content'}:
                raise APIError('Each message must have only role and text content.', 'messages')
            role, content = message.get('role'), message.get('content')
            if role not in ('system', 'developer', 'user', 'assistant') or not isinstance(content, str):
                raise APIError('Supported roles: system, developer, user, assistant; content must be text.', 'messages')
            role = 'system' if role == 'developer' else role
            turns.append(f'<|im_start|>{role}\n{content}<|im_end|>\n')
        prompt = ''.join(turns) + '<|im_start|>assistant\n<think>\n\n</think>\n\n'
    else:
        prompt = body.get('prompt')
        if not isinstance(prompt, str) or not prompt:
            raise APIError('prompt must be a nonempty string; token arrays and prompt batches are unsupported.', 'prompt')
    if '\0' in prompt:
        raise APIError('Text must not contain NUL bytes.', 'messages' if chat else 'prompt')
    try:
        prompt.encode('utf-8')
    except UnicodeEncodeError:
        raise APIError('Text must be valid UTF-8.', 'messages' if chat else 'prompt') from None
    if body.get('max_tokens') is not None and body.get('max_completion_tokens') is not None:
        raise APIError('Specify only one token limit.', 'max_tokens')
    limit = body.get('max_completion_tokens')
    if limit is None:
        limit = body.get('max_tokens')
    if limit is None:
        limit = min(512, context)
    if type(limit) is not int or not 1 <= limit <= context:
        raise APIError(f'max_tokens must be an integer between 1 and {context}.', 'max_tokens')
    values = []
    for param, default, low, high in [('temperature', .7, 0, 2), ('top_p', 1., 0, 1)]:
        value = body.get(param)
        if value is None:
            value = default
        if type(value) not in (int, float) or not low <= value <= high or not math.isfinite(value):
            raise APIError(f'{param} must be finite and between {low} and {high}.', param)
        values.append(value)
    seed = body.get('seed')
    if seed is None:
        seed = secrets.randbits(64)
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise APIError('seed must be an unsigned 64-bit integer.', 'seed')
    stops = body.get('stop')
    if stops is None:
        stops = []
    if isinstance(stops, str):
        stops = [stops]
    if not isinstance(stops, list) or len(stops) > 4 or any(not isinstance(s, str) or not s for s in stops):
        raise APIError('stop must be a nonempty string or up to four nonempty strings.', 'stop')
    stream = body.get('stream')
    if stream is None:
        stream = False
    if type(stream) is not bool:
        raise APIError('stream must be a boolean.', 'stream')
    options = body.get('stream_options')
    if options is None:
        options = {}
    if not isinstance(options, dict) or set(options) - {'include_usage'} or type(options.get('include_usage', False)) is not bool:
        raise APIError('stream_options supports only boolean include_usage.', 'stream_options')
    if options and not stream:
        raise APIError('stream_options requires stream=true.', 'stream_options')
    return Generation(prompt, limit, *values, seed, stops, stream, options.get('include_usage', False))


class Worker:
    def __init__(self, command, timeout=300):
        self.command, self.timeout = command, timeout
        self.process, self.stderr_task = None, None
        self.lock = asyncio.Lock()

    @property
    def healthy(self):
        return self.process is not None and self.process.returncode is None

    async def start(self):
        if self.process and self.process.returncode is None:
            return
        await self.close()
        self.process = await asyncio.create_subprocess_exec(*self.command, stdin=asyncio.subprocess.PIPE,
                                                           stdout=asyncio.subprocess.PIPE,
                                                           stderr=asyncio.subprocess.PIPE)
        async def diagnostics(process):
            async for line in process.stderr:
                LOG.info('worker: %s', line.decode('utf-8', 'replace').rstrip())
        self.stderr_task = asyncio.create_task(diagnostics(self.process))
        if await self.process.stdout.readline() != b'ready\n':
            raise RuntimeError('Worker failed to start; see server logs.')

    async def close(self):
        process, self.process = self.process, None
        if process and process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            await process.wait()
        if self.stderr_task:
            await self.stderr_task
            self.stderr_task = None

    async def generate(self, request):
        async with self.lock:
            completed = False
            try:
                async with asyncio.timeout(self.timeout):
                    await self.start()
                    process = self.process
                    prompt = request.prompt.encode('utf-8')
                    header = f'generate {request.max_tokens} {request.temperature} {request.top_p} {request.seed} {len(prompt)}\n'
                    process.stdin.write(header.encode('ascii') + prompt)
                    await process.stdin.drain()
                    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
                    stop = StopFilter(request.stops)
                    while True:
                        line = await process.stdout.readline()
                        fields = line.split()
                        if fields and fields[0] == b'token' and len(fields) == 3:
                            size = int(fields[2])
                            if not 0 <= size <= 1024 * 1024:
                                raise RuntimeError('Invalid worker token frame.')
                            raw = await process.stdout.readexactly(size)
                            text = stop.feed(decoder.decode(raw))
                            process.stdin.write(b'stop\n' if stop.stopped else b'next\n')
                            await process.stdin.drain()
                            if text:
                                yield {'text': text}
                        elif fields and fields[0] == b'done' and len(fields) == 4:
                            if not stop.stopped:
                                tail = stop.feed(decoder.decode(b'', final=True), final=True)
                                if tail:
                                    yield {'text': tail}
                            prompt_tokens, completion_tokens, length = map(int, fields[1:])
                            completed = True
                            yield {'finish_reason': 'length' if length and not stop.stopped else 'stop', 'usage': {
                                'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens,
                                'total_tokens': prompt_tokens + completion_tokens}}
                            return
                        elif fields and fields[0] == b'error' and len(fields) == 2:
                            message = await process.stdout.readexactly(int(fields[1]))
                            raise APIError(message.decode('utf-8', 'replace'), 'prompt')
                        else:
                            raise RuntimeError('Worker exited or sent an invalid frame.')
            except APIError:
                raise  # Request errors leave the worker synchronized and reusable.
            except BaseException:
                if not completed:
                    await self.close()  # Cancellation must not leave a partial request in the protocol.
                raise


class CFFIEngine:
    """Serialize native calls on one thread; the HTTP loop never performs inference."""
    def __init__(self, path, backend='cpu', context=4096, timeout=300, library=None, model_factory=None):
        if model_factory is None:
            if __package__:
                from .native import Model, DEFAULT_LIBRARY
            else:
                from native import Model, DEFAULT_LIBRARY
            model_factory = partial(Model, library=library or DEFAULT_LIBRARY)
        self.factory = partial(model_factory, path, backend=backend, context=context)
        self.timeout = timeout
        self.model = None
        self.lock = asyncio.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='kyse-inference')

    @property
    def healthy(self):
        return self.model is not None and not self.model.closed

    async def _call(self, operation, *args):
        future = asyncio.get_running_loop().run_in_executor(self.executor, partial(operation, *args))
        # C cannot be preempted by cancelling a Python future. Retain/drain the call
        # before reset/close, so a cancelled request never frees live native state.
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                await future
            raise

    def _load(self):
        # Keep ownership even if the awaiting HTTP task is cancelled during load.
        self.model = self.factory()

    async def start(self):
        if not self.healthy:
            await self._call(self._load)

    async def close(self):
        if self.model is not None:
            model, self.model = self.model, None
            await self._call(model.close)

    async def shutdown(self):
        try:
            await self.close()
        finally:
            self.executor.shutdown(wait=True, cancel_futures=True)

    async def generate(self, request):
        async with self.lock:
            completed = False
            try:
                async with asyncio.timeout(self.timeout):
                    await self.start()
                    await self._call(self.model.begin, request.prompt, request.max_tokens,
                                     request.temperature, request.top_p, request.seed)
                    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
                    stop = StopFilter(request.stops)
                    while True:
                        raw = await self._call(self.model.next_token)
                        if raw is None:
                            tail = stop.feed(decoder.decode(b'', final=True), final=True)
                            if tail:
                                yield {'text': tail}
                            if stop.stopped:
                                await self._call(self.model.stop)
                            usage = await self._call(self.model.usage)
                            finish = usage.pop('finish_reason')
                            completed = True
                            yield {'finish_reason': finish, 'usage': usage}
                            return
                        text = stop.feed(decoder.decode(raw))
                        if stop.stopped:
                            await self._call(self.model.stop)
                        if text:
                            yield {'text': text}
                        if stop.stopped:
                            usage = await self._call(self.model.usage)
                            usage.pop('finish_reason')
                            completed = True
                            yield {'finish_reason': 'stop', 'usage': usage}
                            return
            except ValueError as error:
                raise APIError(str(error), 'prompt') from error
            except RuntimeError:
                await self.close()
                raise
            finally:
                if not completed and self.healthy:
                    await self._call(self.model.stop)


WORKER = web.AppKey('worker', object)
MODEL = web.AppKey('model', str)
CONTEXT = web.AppKey('context', int)


@web.middleware
async def errors(request, handler):
    try:
        return await handler(request)
    except APIError as error:
        return web.json_response(error.body(), status=error.status)
    except web.HTTPException as error:
        return web.json_response(APIError(error.reason, status=error.status).body(), status=error.status)
    except TimeoutError:
        return web.json_response(APIError('Generation timed out.', status=504).body(), status=504)
    except (ConnectionError, RuntimeError, OSError, asyncio.IncompleteReadError):
        LOG.exception('Inference failed')
        return web.json_response(APIError('Inference worker failed; see server logs.', status=503).body(), status=503)


async def complete(request):
    try:
        body = await request.json()
    except (ValueError, UnicodeDecodeError):
        raise APIError('Invalid JSON request body.') from None
    chat = request.path == '/v1/chat/completions'
    generation = parse_request(body, chat, request.app[MODEL], request.app[CONTEXT])
    identifier = ('chatcmpl-' if chat else 'cmpl-') + uuid.uuid4().hex
    base = {'id': identifier, 'object': 'chat.completion' if chat else 'text_completion',
            'created': int(time.time()), 'model': request.app[MODEL]}
    events = request.app[WORKER].generate(generation)
    response = None

    def choice(text='', finish=None, role=False):
        if chat:
            delta = {'role': 'assistant'} if role else ({'content': text} if text else {})
            return {'index': 0, 'delta': delta, 'finish_reason': finish}
        return {'index': 0, 'text': text, 'logprobs': None, 'finish_reason': finish}

    async def send(value):
        await response.write(('data: ' + json.dumps(value, ensure_ascii=False) + '\n\n').encode('utf-8'))

    try:
        first = await anext(events)  # Validate context before sending streaming headers.
        if generation.stream:
            response = web.StreamResponse(headers={'Content-Type': 'text/event-stream',
                                                  'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
            await response.prepare(request)
            if chat:
                base['object'] = 'chat.completion.chunk'
                await send(base | {'choices': [choice(role=True)]} | ({'usage': None} if generation.include_usage else {}))
        text = []
        event = first
        while True:
            if 'text' in event:
                if generation.stream:
                    await send(base | {'choices': [choice(event['text'])]} | ({'usage': None} if generation.include_usage else {}))
                else:
                    text.append(event['text'])
            else:
                usage, finish = event['usage'], event['finish_reason']
                if generation.stream:
                    await send(base | {'choices': [choice(finish=finish)]} | ({'usage': None} if generation.include_usage else {}))
                    if generation.include_usage:
                        await send(base | {'choices': [], 'usage': usage})
                    await response.write(b'data: [DONE]\n\n')
                    await response.write_eof()
                    return response
                result = {'message': {'role': 'assistant', 'content': ''.join(text)}} if chat else {'text': ''.join(text), 'logprobs': None}
                return web.json_response(base | {'choices': [result | {'index': 0, 'finish_reason': finish}], 'usage': usage})
            event = await anext(events)
    except (APIError, TimeoutError, ConnectionError, RuntimeError, OSError, asyncio.IncompleteReadError) as error:
        if response is None:
            raise
        LOG.warning('Stream interrupted: %s', error)
        if not isinstance(error, ConnectionError):
            with contextlib.suppress(ConnectionError):
                await send(APIError('Generation stream interrupted.', status=503).body())
                await response.write_eof()
        return response
    finally:
        await events.aclose()


def create_app(command=None, model_name='kyse', context=4096, timeout=300, *, engine=None):
    app = web.Application(middlewares=[errors], client_max_size=1024 * 1024)
    if engine is None and command is None:
        raise ValueError('Provide an engine or a worker command.')
    app[WORKER], app[MODEL], app[CONTEXT] = engine if engine is not None else Worker(command, timeout), model_name, context

    async def lifecycle(app):
        try:
            async with asyncio.timeout(120):
                await app[WORKER].start()
            yield
        finally:
            if hasattr(app[WORKER], 'shutdown'):
                await app[WORKER].shutdown()
            else:
                await app[WORKER].close()

    async def models(request):
        return web.json_response({'object': 'list', 'data': [
            {'id': model_name, 'object': 'model', 'created': 0, 'owned_by': 'local'}]})

    async def health(request):
        worker = app[WORKER]
        return web.json_response({'status': 'ok' if worker.healthy else 'restarting',
                                  'model': model_name, 'busy': worker.lock.locked()})

    app.cleanup_ctx.append(lifecycle)
    app.router.add_get('/health', health)
    app.router.add_get('/v1/models', models)
    app.router.add_post('/v1/completions', complete)
    app.router.add_post('/v1/chat/completions', complete)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True, help='model.bin (tokenizer.bin must be alongside it)')
    parser.add_argument('--model-name', default='kyse')
    parser.add_argument('--engine', choices=['cffi', 'worker'], default='cffi')
    parser.add_argument('--library', type=Path, help='path to libkyse (default: build/libkyse.dylib or .so)')
    parser.add_argument('--worker', type=Path, default=ROOT/'qwen-worker')
    parser.add_argument('--backend', choices=['cpu', 'metal', 'accelerate'], default='cpu')
    parser.add_argument('--context', type=int, default=4096)
    parser.add_argument('--timeout', type=float, default=300, help='seconds per generation, excluding queue wait')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    if args.context < 2 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('context must be at least 2 and timeout must be positive and finite')
    command = [str(args.worker.resolve()), str(args.model.resolve()), '-b', args.backend, '-c', str(args.context)]
    logging.basicConfig(level=logging.INFO)
    engine = (CFFIEngine(args.model, args.backend, args.context, args.timeout, args.library)
              if args.engine == 'cffi' else Worker(command, args.timeout))
    web.run_app(create_app(model_name=args.model_name, context=args.context, timeout=args.timeout, engine=engine),
                host=args.host, port=args.port, handler_cancellation=True)


if __name__ == '__main__':
    main()
