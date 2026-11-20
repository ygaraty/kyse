import asyncio
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from aiohttp.test_utils import TestClient, TestServer
from server.server import APIError, CFFIEngine, Generation, StopFilter, WORKER, Worker, create_app, parse_request
from server.native import Model

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('model_fixtures', ROOT/'tests/test_model.py')
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)

FAKE = r'''
import sys,time
r,w=sys.stdin.buffer,sys.stdout.buffer
w.write(b'ready\n');w.flush()
while line:=r.readline():
 _,limit,temp,top,seed,size=line.split();prompt=r.read(int(size))
 if b'hang' in prompt:time.sleep(60)
 if b'crash' in prompt:sys.exit(1)
 parts=[b'\xf0',b'\x9f\x8c',b'\x8dE',b'N',b'Dtail'] if b'unicode' in prompt else [b'a',b'b',b'c']
 length=1;count=0
 for raw in parts[:int(limit)]:
  w.write(f'token 0 {len(raw)}\n'.encode()+raw);w.flush();count+=1
  if r.readline()==b'stop\n':length=0;break
  if b'slow' in prompt:time.sleep(60)
 w.write(f'done 1 {count} {length}\n'.encode());w.flush()
'''


def request(prompt='hello', **kwargs):
    return replace(Generation(prompt, 8, 0, 1, 1, [], False, False), **kwargs)


class ValidationTests(unittest.TestCase):
    def test_stop_prefixes(self):
        stop = StopFilter(['END', 'STOP'])
        self.assertEqual(stop.feed('answer E'), 'answer ')
        self.assertEqual(stop.feed('N'), '')
        self.assertEqual(stop.feed('Dhidden'), '')
        self.assertTrue(stop.stopped)
        stop = StopFilter(['STOP'])
        self.assertEqual(stop.feed('ST'), '')
        self.assertEqual(stop.feed('', final=True), 'ST')

    def test_nullable_optional_parameters(self):
        generation = parse_request({'model': 'kyse', 'prompt': 'a', 'temperature': None,
                                    'top_p': None, 'max_tokens': None, 'seed': None,
                                    'n': None, 'stream': None}, False, 'kyse', 64)
        self.assertEqual(generation.max_tokens, 64)
        self.assertFalse(generation.stream)

    def test_reject_bad_parameters(self):
        for extra in [{'max_tokens': True}, {'max_tokens': 0}, {'stream': 'false'},
                      {'temperature': float('nan')}, {'top_p': 10**1000}, {'seed': -1},
                      {'stop': False}, {'stop': ''}, {'prompt': ['a', 'b']}, {'tools': []}]:
            with self.subTest(extra=extra), self.assertRaises(APIError):
                parse_request({'model': 'kyse', 'prompt': 'a'} | extra, False, 'kyse', 64)


class WorkerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.worker = Worker([sys.executable, '-u', '-c', FAKE], timeout=1)

    async def asyncTearDown(self):
        await self.worker.close()

    async def test_utf8_stop_and_reuse(self):
        events = [e async for e in self.worker.generate(request('unicode', stops=['END']))]
        self.assertEqual(''.join(e.get('text', '') for e in events), '🌍')
        self.assertEqual(events[-1]['finish_reason'], 'stop')
        pid = self.worker.process.pid
        output = [e async for e in self.worker.generate(request())]
        self.assertEqual(''.join(e.get('text', '') for e in output), 'abc')
        self.assertEqual(self.worker.process.pid, pid)

    async def test_cancel_and_restart(self):
        events = self.worker.generate(request('slow'))
        self.assertEqual(await anext(events), {'text': 'a'})
        await events.aclose()
        self.assertIsNone(self.worker.process)
        output = [e async for e in self.worker.generate(request())]
        self.assertIn('usage', output[-1])

    async def test_timeout_and_crash(self):
        self.worker.timeout = .05
        with self.assertRaises(TimeoutError):
            _ = [e async for e in self.worker.generate(request('hang'))]
        self.assertIsNone(self.worker.process)
        self.worker.timeout = 2
        with self.assertRaises(RuntimeError):
            _ = [e async for e in self.worker.generate(request('crash'))]
        self.assertIsNone(self.worker.process)
        self.assertTrue([e async for e in self.worker.generate(request())])


def make_checkpoint(path):
    fixtures.checkpoint(path, False, vocab_size=261, zero_head=True)
    fixtures.export.export_model(path, path/'model.bin')
    tokens = [bytes([i]) for i in range(256)]
    tokens[0], tokens[97] = tokens[97], tokens[0]
    tokens += [b'<|im_start|>', b'<|im_end|>', b'<think>', b'</think>', b'<|endoftext|>']
    import struct
    with (path/'tokenizer.bin').open('wb') as out:
        out.write(struct.pack('<iii5i', 261, 5, 14, *range(256, 261)))
        for token in tokens:
            out.write(struct.pack('<ii', -1, len(token)) + token)


class HTTPTests(unittest.IsolatedAsyncioTestCase):
    engine_type = 'worker'

    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        path = Path(self.tmp.name)
        make_checkpoint(path)
        command = [str(ROOT/'qwen-worker'), str(path/'model.bin'), '-b', 'cpu', '-c', '64']
        engine = CFFIEngine(path/'model.bin', context=64) if self.engine_type == 'cffi' else None
        self.app = create_app(command, context=64, engine=engine)
        self.client = TestClient(TestServer(self.app))
        await self.client.start_server()

    def identity(self):
        engine = self.app[WORKER]
        return id(engine.model) if self.engine_type == 'cffi' else engine.process.pid

    async def asyncTearDown(self):
        await self.client.close()
        self.tmp.cleanup()

    async def post(self, path='/v1/completions', **kwargs):
        return await self.client.post(path, json={'model': 'kyse', 'prompt': 'a',
                                                 'max_tokens': 3, 'temperature': 0} | kwargs)

    async def test_completion_usage_queue_and_persistence(self):
        pid = self.identity()
        responses = await asyncio.gather(*(self.post() for _ in range(3)))
        for response in responses:
            self.assertEqual(response.status, 200)
            body = await response.json()
            self.assertEqual(body['choices'][0]['text'], 'aaa')
            self.assertEqual(body['choices'][0]['finish_reason'], 'length')
            self.assertEqual(body['usage'], dict(prompt_tokens=1, completion_tokens=3, total_tokens=4))
        self.assertEqual(self.identity(), pid)
        response = await self.post(stop='aa')
        self.assertEqual((await response.json())['choices'][0]['text'], '')
        self.assertEqual(self.identity(), pid)

    async def test_chat_and_streaming(self):
        response = await self.client.post('/v1/chat/completions', json={
            'model': 'kyse', 'messages': [{'role': 'user', 'content': 'a'}],
            'max_tokens': 3, 'temperature': 0, 'stream': True, 'stream_options': {'include_usage': True}})
        self.assertEqual(response.status, 200)
        frames = [line[6:] for line in (await response.text()).splitlines() if line.startswith('data: ')]
        self.assertEqual(frames[-1], '[DONE]')
        chunks = [json.loads(frame) for frame in frames[:-1]]
        self.assertEqual(chunks[0]['choices'][0]['delta']['role'], 'assistant')
        self.assertEqual(''.join(c['choices'][0]['delta'].get('content', '') for c in chunks if c['choices']), 'aaa')
        self.assertEqual(chunks[-2]['choices'][0]['finish_reason'], 'length')
        self.assertEqual(chunks[-1]['usage']['completion_tokens'], 3)
        self.assertEqual(chunks[-1]['choices'], [])
        response = await self.post(stream=True)
        self.assertIn('"object": "text_completion"', await response.text())
        response = await self.client.post('/v1/chat/completions', json={
            'model': 'kyse', 'messages': [{'role': 'user', 'content': 'a'}], 'max_tokens': 1, 'temperature': 0})
        self.assertEqual((await response.json())['choices'][0]['message']['content'], 'a')

    async def test_context_error_keeps_worker_reusable(self):
        response = await self.post(prompt='a'*70)
        self.assertEqual(response.status, 400)
        self.assertIn('context', (await response.json())['error']['message'])
        self.assertEqual((await self.post()).status, 200)
        response = await self.post(prompt='a'*70, stream=True)
        self.assertEqual(response.status, 400)
        self.assertEqual(response.content_type, 'application/json')

    async def test_models_and_errors(self):
        self.assertEqual((await (await self.client.get('/v1/models')).json())['data'][0]['id'], 'kyse')
        self.assertEqual((await self.client.get('/health')).status, 200)
        for kwargs in [{'model': 'other'}, {'n': 2}, {'logprobs': True}]:
            response = await self.post(**kwargs)
            self.assertEqual(response.status, 400)
            self.assertIn('error', await response.json())
        self.assertEqual((await self.client.post('/v1/completions', data='{')).status, 400)


class CFFIHTTPTests(HTTPTests):
    engine_type = 'cffi'


class NativeModelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        make_checkpoint(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_tokens_usage_and_ownership(self):
        with Model(self.path/'model.bin', context=64) as model:
            model.begin('a', max_tokens=3, temperature=0)
            tokens = []
            while (token := model.next_token()) is not None:
                tokens.append(token)
            self.assertEqual(tokens, [b'a']*3)
            self.assertEqual(model.usage(), dict(prompt_tokens=1, completion_tokens=3,
                                                total_tokens=4, finish_reason='length'))
            with self.assertRaisesRegex(RuntimeError, 'one model'):
                Model(self.path/'model.bin')
            with self.assertRaises(ValueError):
                model.begin('a'*70)
            model.begin('a', max_tokens=3, temperature=.7, top_p=0)
            self.assertEqual(model.next_token(), b'a')
            model.stop()
            self.assertIsNone(model.next_token())
            self.assertEqual(model.usage()['finish_reason'], 'stop')
        self.assertEqual(b''.join(tokens), b'aaa')  # returned bytes own their memory
        model.close()  # idempotent
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            model.next_token()

    def test_chunked_prefill(self):
        with Model(self.path/'model.bin', context=512) as model:
            model.begin('a'*300, max_tokens=3, temperature=0)
            self.assertEqual([model.next_token() for _ in range(3)], [b'a']*3)
            self.assertIsNone(model.next_token())
            self.assertEqual(model.usage()['prompt_tokens'], 300)

    def test_unclosed_python_handle_is_finalized(self):
        import gc
        model = Model(self.path/'model.bin', context=64)
        del model
        gc.collect()
        with Model(self.path/'model.bin', context=64) as model:
            model.begin('a', max_tokens=1, temperature=0)
            self.assertEqual(model.next_token(), b'a')

    def test_load_errors_do_not_exit_or_poison_python(self):
        import struct
        with self.assertRaises(RuntimeError):
            Model(self.path/'missing.bin')
        invalid = self.path/'invalid.bin'
        invalid.write_bytes(b'bad')
        with self.assertRaises(RuntimeError):
            Model(invalid)
        corrupt = bytearray((self.path/'tokenizer.bin').read_bytes())
        struct.pack_into('<i', corrupt, 36, 1000000)  # first token length exceeds the header
        tokenizer = self.path/'corrupt-tokenizer.bin'
        tokenizer.write_bytes(corrupt)
        with self.assertRaisesRegex(RuntimeError, 'token length'):
            Model(self.path/'model.bin', tokenizer=tokenizer)
        with Model(self.path/'model.bin', context=64) as model:
            model.begin('a', max_tokens=1, temperature=0)
            self.assertEqual(model.next_token(), b'a')


class NativeCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancellation_drains_native_call_before_reusing_state(self):
        import threading
        entered, release = threading.Event(), threading.Event()

        class SlowModel:
            closed = False
            def __init__(self, *args, **kwargs):
                self.calls = 0
                self.stops = 0
            def begin(self, *args):
                self.calls = 0
            def next_token(self):
                self.calls += 1
                if self.calls == 1:
                    entered.set()
                    if not release.wait(2):
                        raise RuntimeError('test native call was not released')
                    return b'a'
                return None
            def stop(self):
                self.stops += 1
            def usage(self):
                return dict(prompt_tokens=1, completion_tokens=1, total_tokens=2, finish_reason='stop')
            def close(self):
                self.closed = True

        engine = CFFIEngine('unused', model_factory=SlowModel)
        try:
            await engine.start()
            generator = engine.generate(request())
            task = asyncio.create_task(anext(generator))
            while not entered.is_set():
                await asyncio.sleep(.001)  # the event loop stays responsive during native work
            task.cancel()
            await asyncio.sleep(.01)
            self.assertFalse(task.done())
            self.assertTrue(engine.lock.locked())
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(engine.model.stops, 1)
            self.assertTrue([e async for e in engine.generate(request())])
        finally:
            release.set()
            await engine.shutdown()


if __name__ == '__main__':
    unittest.main()
