"""Run the same small text-only evaluation for each exported model.

Example: python3 scripts/benchmark.py models/9b en-9b-q8 en-9b-q4
Raw command output and machine-readable measurements go into results/9b/.
"""
import argparse
import json
from pathlib import Path
import re
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('models', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'results/9b')
    parser.add_argument('--backend', default='metal')
    parser.add_argument('--chunks', type=int, default=2)
    parser.add_argument('--chunk-bytes', type=int, default=1024)
    parser.add_argument('--arc-items', type=int, default=10)
    parser.add_argument('--hellaswag-items', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--reuse-logs', action='store_true', help='reuse existing logs from the same model and settings')
    parser.add_argument('--timeout', type=int, default=900, help='seconds per command')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []

    def run(command, log):
        if args.reuse_logs and log.exists():
            print('REUSE', log, flush=True)
            return log.read_text(), None
        print('RUN', ' '.join(map(str, command)), flush=True)
        start = time.monotonic()
        with log.open('w') as output:
            result = subprocess.run(list(map(str, command)), cwd=ROOT, text=True,
                                    stdout=output, stderr=subprocess.STDOUT, timeout=args.timeout)
        elapsed = time.monotonic() - start
        if result.returncode:
            raise RuntimeError(f'command failed ({result.returncode}), see {log}')
        return log.read_text(), elapsed

    for directory in args.models:
        directory = directory.resolve()
        name = directory.name
        model = directory/'model.bin'
        item = dict(name=name, model=str(model), bytes=model.stat().st_size,
                    vocab_size=int.from_bytes(model.open('rb').read(24)[20:24], 'little'),
                    backend=args.backend, generation=[])
        for repeat in range(args.repeats):
            output, elapsed = run([ROOT/'qwen', model, '-b', args.backend, '-c', 512,
                                   '-t', 0, '-n', 64, '-v', '-i',
                                   'Explain how a rainbow forms, step by step, in plain English.'],
                                  args.output/f'{name}-generation-{repeat+1}.log')
            match = re.search(r'output: (\d+) tok .*?TTFT: ([\d.]+) ms .*?decode: ([\d.]+) tok/s \([\d.]+ ms/tok, ([\d.]+)s\)', output)
            if match:
                tokens, ttft, speed, seconds = match.groups()
                trial = dict(tokens=int(tokens), seconds=float(seconds), tokens_per_second=float(speed),
                             ttft_ms=float(ttft), timing='after_first_token')
            else:
                # Historical logs used the old CLI timing (including its token-count bias).
                match = re.search(r'generation: (\d+) tok in ([\d.]+)s \(([\d.]+) tok/s\)', output)
                if not match:
                    raise ValueError('missing generation timing')
                trial = dict(tokens=int(match[1]), seconds=float(match[2]),
                             tokens_per_second=float(match[3]), timing='legacy')
            trial['wall_seconds'] = elapsed
            item['generation'].append(trial)
        item['generation_median_tokens_per_second'] = statistics.median(
            r['tokens_per_second'] for r in item['generation'])
        output, _ = run([ROOT/'bench', 'bpb', model, ROOT/'data/wiki.test.txt', '-b', args.backend,
                         '-k', args.chunks, '-l', args.chunk_bytes], args.output/f'{name}-wiki.log')
        match = re.search(r': ([\d.]+) bits/byte \| ([\d.]+) bits/token \| top-1 ([\d.]+)% \| (\d+) tokens for (\d+) bytes.*?([\d.]+) ms/token', output)
        if not match:
            raise ValueError('missing text benchmark results')
        item['wiki'] = dict(bits_per_byte=float(match[1]), bits_per_token=float(match[2]),
                            top1_percent=float(match[3]), tokens=int(match[4]), bytes=int(match[5]),
                            ms_per_token=float(match[6]))
        for dataset, count in [('arc_easy', args.arc_items), ('hellaswag', args.hellaswag_items)]:
            output, _ = run([ROOT/'bench', 'mc', model, ROOT/f'data/{dataset}.txt', '-b', args.backend,
                             '-k', count], args.output/f'{name}-{dataset}.log')
            match = re.search(r': acc ([\d.]+)%.*?acc_norm ([\d.]+)%.*? (\d+) items', output)
            partial = match is None
            if partial:
                progress = re.findall(r'item (\d+): acc ([\d.]+)% acc_norm ([\d.]+)%', output)
                if not progress or int(progress[-1][0]) != count:
                    raise ValueError('incomplete multiple-choice sample')
                n, acc, norm = progress[-1]
            else:
                acc, norm, n = match.groups()
            if int(n) != count:
                raise ValueError('multiple-choice sample size differs from requested count')
            item[dataset] = dict(items=int(n), acc_percent=float(acc), acc_norm_percent=float(norm),
                                 partial_run=partial, output=output.splitlines()[-1])
        results.append(item)
        (args.output/'measurements.json').write_text(json.dumps(dict(
            parameters=vars(args) | {'models': list(map(str,args.models)), 'output': str(args.output)},
            results=results), indent=2))
        print(json.dumps(item, indent=2), flush=True)


if __name__ == '__main__':
    main()
