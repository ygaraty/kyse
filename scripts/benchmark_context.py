"""Exercise a near-full context through the CFFI engine, with exact C token counts."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--piara-dir', type=Path, default=ROOT.parent/'piara', help='sibling Piara project')
    parser.add_argument('--model', type=Path, default=ROOT/'en-9b-q8/model.bin')
    parser.add_argument('--context', type=int, default=32768)
    parser.add_argument('--prompt-tokens', type=int, default=32736)
    parser.add_argument('--max-tokens', type=int, default=32)
    parser.add_argument('--backend', default='metal')
    parser.add_argument('--text', type=Path, default=ROOT/'data/wiki.test.txt')
    parser.add_argument('--output', type=Path, default=ROOT/'results/context32k-fp16')
    args = parser.parse_args()
    sys.path.insert(0, str(args.piara_dir.resolve().parent))
    try:
        from piara.server import CFFIEngine, parse_request
    except ModuleNotFoundError as error:
        parser.error(f'Piara and its dependencies are required: {error}')
    if not 32 < args.prompt_tokens < args.context or args.max_tokens <= 0:
        parser.error('prompt-tokens must be between 33 and context-1; max-tokens must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    prompt_path = args.output/'prompt.txt'
    tokenizer = args.model.resolve().with_name('tokenizer.bin')
    key = 'strawberry'
    corpus = args.text.read_text()

    def render(document, padding=''):
        content = (f'Remember this secret word: {key}.\n\n'
                   f'Document:\n{document}\n{padding}\n\n'
                   'What is the secret word? Reply with only the word.')
        return parse_request({'model': 'kyse', 'messages': [{'role': 'user', 'content': content}],
                              'max_tokens': args.max_tokens, 'temperature': 0}, True, 'kyse', args.context)

    def count(generation):
        prompt_path.write_text(generation.prompt)
        result = subprocess.run([str(ROOT/'bench'), 'count', str(tokenizer), str(prompt_path)],
                                capture_output=True, text=True, check=True)
        return int(re.search(r'(\d+) tokens,', result.stderr)[1])

    lo, hi = 0, len(corpus)
    while lo < hi:
        mid = (lo + hi + 1)//2
        if count(render(corpus[:mid])) <= args.prompt_tokens - 8:
            lo = mid
        else:
            hi = mid - 1
    document = corpus[:lo]
    for padding_tokens in range(64):
        padding = ' x' * padding_tokens
        generation = render(document, padding)
        n = count(generation)
        if n == args.prompt_tokens:
            break
    else:
        raise RuntimeError(f'Could not construct the requested token count: {n}')
    result = dict(context=args.context, requested_prompt_tokens=n, model=str(args.model.resolve()),
                  backend=args.backend, kv_cache_dtype='float16', expected_answer=key, prompt_sha256=hashlib.sha256(prompt_path.read_bytes()).hexdigest())

    def save():
        (args.output/'measurements.json').write_text(json.dumps(result, indent=2)+'\n')

    async def run():
        engine = CFFIEngine(args.model, args.backend, args.context, timeout=14400)
        start = time.monotonic()
        await engine.start()
        result['load_seconds'] = time.monotonic() - start
        print(f'Model loaded with {args.context} context in {result["load_seconds"]:.2f}s', flush=True)
        save()
        running = True

        async def heartbeat():
            while running:
                await asyncio.sleep(45)
                if running:
                    print(f'Long-prompt inference still running: {time.monotonic()-start:.0f}s elapsed', flush=True)

        async def generate(request):
            started = time.monotonic()
            output, first, last, chunks = [], None, None, 0
            events = engine.generate(request)
            try:
                async for event in events:
                    if 'text' in event:
                        last = time.monotonic()
                        if first is None:
                            first = last
                        chunks += 1
                        output.append(event['text'])
                    else:
                        return {'text': ''.join(output), 'usage': event['usage'],
                                'finish_reason': event['finish_reason'], 'ttft_seconds': None if first is None else first-started,
                                'total_seconds': time.monotonic()-started,
                                'stream_chunks': chunks,
                                'decode_chunks_per_second': (chunks-1)/(last-first) if chunks>1 and last>first else None}
            finally:
                await events.aclose()

        task = None
        try:
            short = render('A short document.')
            result['short_prompt_at_32k'] = await generate(short)
            print('Short-prompt smoke:', json.dumps(result['short_prompt_at_32k']), flush=True)
            result['short_control_passed'] = result['short_prompt_at_32k']['text'].strip() == key
            save()
            if not result['short_control_passed']:
                raise RuntimeError('Retrieval prompt failed the short control; do not run the long test.')
            start = time.monotonic()
            task = asyncio.create_task(heartbeat())
            result['long_prompt'] = await generate(generation)
            result['recall_passed'] = result['long_prompt']['text'].strip() == key
            result['actual_prompt_tokens_match'] = result['long_prompt']['usage']['prompt_tokens'] == n
            result['peak_process_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
            save()
            print('Long-prompt result:', json.dumps(result['long_prompt']), flush=True)
            print('Recall passed:', result['recall_passed'], flush=True)
            # Validation should reject an over-window prompt before doing inference.
            try:
                oversized = render(document, padding + ' x' * args.context)
                await engine._call(engine.model.begin, oversized.prompt, 1, 0., 1., 1)
            except ValueError as error:
                result['over_window_rejected'] = str(error)
            else:
                raise AssertionError('Expected an over-window prompt error')
            result['after_rejection'] = await generate(short)
            save()
        finally:
            running = False
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            await engine.shutdown()

    os.environ['KYSE_PREFILL_PROGRESS'] = '1'
    asyncio.run(run())


if __name__ == '__main__':
    main()
