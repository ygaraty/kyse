"""Small deterministic checkpoints exercise export -> shrink -> C inference."""
import importlib.util
import json
import os
import re
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('export', ROOT / 'scripts/export.py')
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def checkpoint(path, tied, vocab_size=32, zero_head=False):
    d, f, hd, nk, nv, dk, dv = 32, 64, 8, 2, 4, 8, 8
    cfg = dict(hidden_size=d, intermediate_size=f, num_hidden_layers=2, vocab_size=vocab_size,
               num_attention_heads=4, num_key_value_heads=2, head_dim=hd,
               linear_num_key_heads=nk, linear_num_value_heads=nv,
               linear_key_head_dim=dk, linear_value_head_dim=dv, linear_conv_kernel_dim=4,
               rms_norm_eps=1e-6, rope_parameters=dict(partial_rotary_factor=.5, rope_theta=1e7),
               layer_types=['linear_attention', 'full_attention'])
    (path / 'config.json').write_text(json.dumps(dict(text_config=cfg, tie_word_embeddings=tied)))
    tensors = {}

    def tensor(name, rows, cols=None, norm=False):
        n = rows * (cols or 1)
        values = [0.0 if norm or (zero_head and name == "lm_head.weight") else ((i * 17 + len(name)) % 29 - 14) / 128 for i in range(n)]
        dtype = 'BF16' if cols else 'F32'
        raw = b''.join(struct.pack('<I', struct.unpack('<I', struct.pack('<f', v))[0])[2:]
                       for v in values) if cols else struct.pack('<%df' % n, *values)
        tensors[name] = (dtype, [rows, cols] if cols else [rows], raw)

    p = 'model.language_model.'
    tensor(p + 'embed_tokens.weight', vocab_size, d)
    for l in range(2):
        q = p + f'layers.{l}.'
        for name in ('input_layernorm.weight', 'post_attention_layernorm.weight'):
            tensor(q + name, d, norm=True)
        if l:
            a = q + 'self_attn.'
            for name, rows, cols in [('q_proj', 8*hd, d), ('k_proj', 2*hd, d),
                                     ('v_proj', 2*hd, d), ('o_proj', d, 4*hd)]:
                tensor(a + name + '.weight', rows, cols)
            for name in ('q_norm.weight', 'k_norm.weight'):
                tensor(a + name, hd, norm=True)
        else:
            a = q + 'linear_attn.'
            for name, rows, cols in [('in_proj_qkv', 2*nk*dk+nv*dv, d),
                                     ('in_proj_z', nv*dv, d), ('in_proj_a', nv, d),
                                     ('in_proj_b', nv, d), ('out_proj', d, nv*dv)]:
                tensor(a + name + '.weight', rows, cols)
            for name, size in [('conv1d.weight', (2*nk*dk+nv*dv)*4), ('A_log', nv),
                               ('dt_bias', nv), ('norm.weight', dv)]:
                tensor(a + name, size)
        for name, rows, cols in [('gate_proj', f, d), ('up_proj', f, d), ('down_proj', d, f)]:
            tensor(q + 'mlp.' + name + '.weight', rows, cols)
    tensor(p + 'norm.weight', d, norm=True)
    if not tied:
        tensor('lm_head.weight', vocab_size, d)
    index = {}
    for shard in range(2):
        name = f'model-{shard}.safetensors'
        header, payload = {}, bytearray()
        for key in list(tensors)[shard::2]:
            dtype, shape, raw = tensors[key]
            header[key] = dict(dtype=dtype, shape=shape, data_offsets=[len(payload), len(payload)+len(raw)])
            payload.extend(raw)
            index[key] = name
        h = json.dumps(header).encode()
        (path / name).write_bytes(struct.pack('<Q', len(h)) + h + payload)
    (path / 'model.safetensors.index.json').write_text(json.dumps(dict(weight_map=index)))


class ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.runner = Path(cls.temp.name) / 'runner'
        extra = []
        if os.environ.get('KYSE_TEST_METAL'):
            extra = ['-DKYSE_TEST_METAL', str(ROOT/'src/backend_accelerate.c'), str(ROOT/'src/backend_metal.m'), '-fobjc-arc',
                     '-framework', 'Metal', '-framework', 'Foundation', '-framework', 'Accelerate']
        subprocess.run([os.environ.get('CC', 'cc'), '-std=gnu23', '-Wall', '-Wextra',
                        '-I', str(ROOT/'src'), str(ROOT/'tests/model_test.c'), str(ROOT/'src/model.c'),
                        str(ROOT/'src/ops_ref.c'), str(ROOT/'src/backend_cpu.c'), '-lm', '-o', str(cls.runner)] + extra, check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_reply_stats(self):
        # A zero head deterministically emits token 0, including an EOS fixture.
        subprocess.run(['make', 'qwen'], cwd=ROOT, check=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            checkpoint(path, False, vocab_size=261, zero_head=True)
            export.export_model(path, path/'model.bin')

            def tokenizer(eos=False):
                tokens = [bytes([i]) for i in range(256)]
                tokens[0], tokens[97] = tokens[97], tokens[0]  # token 0 emits 'a'
                tokens += [b'<|im_start|>', b'<|im_end|>', b'<think>', b'</think>', b'<|endoftext|>']
                special = list(range(256, 261))
                if eos:
                    tokens[0], tokens[257] = tokens[257], b''
                    tokens[97] = b'a'
                    special = [0, 256, 258, 259, 260]
                with (path/'tokenizer.bin').open('wb') as out:
                    out.write(struct.pack('<iii', len(tokens), len(special), max(map(len, tokens))))
                    out.write(struct.pack('<%di' % len(special), *special))
                    for token in tokens:
                        out.write(struct.pack('<ii', -1, len(token)) + token)

            def run(*args, text=None):
                result = subprocess.run([str(ROOT/'qwen'), str(path/'model.bin'), '-b', 'cpu',
                                         '-t', '0', '-c', '256', '-n', '3', *args],
                                        input=text, capture_output=True, text=True, check=True)
                self.assertNotRegex(result.stderr, r'\b(?:nan|inf)\b')
                return result

            tokenizer()
            reply = run('-r', '-i', 'a', '-v')
            self.assertIn('output: 3 tok', reply.stderr)
            self.assertRegex(reply.stderr, r'TTFT: [\d.]+ ms.*decode: [\d.]+ tok/s')
            self.assertIn('context: 3/256 tok | stop: token limit', reply.stderr)
            self.assertEqual(reply.stdout, 'aaaa\n')  # echoed raw prompt + three output tokens
            single = run('-r', '-i', 'a', '-v', '-n', '1')
            self.assertIn('output: 1 tok', single.stderr)
            self.assertIn('decode: n/a', single.stderr)
            full = run('-r', '-i', 'a', '-v', '-n', '5', '-c', '3')
            self.assertIn('context: 3/3 tok | stop: context full', full.stderr)
            chat = run(text='a\nb\n/reset\na\n')
            self.assertEqual(chat.stderr.count(' | TTFT:'), 3)
            contexts = re.findall(r'context: (\d+)/256', chat.stderr)
            self.assertGreater(int(contexts[1]), int(contexts[0]))
            self.assertEqual(contexts[0], contexts[2])
            self.assertNotIn(' | TTFT:', run('-q', text='a\n').stderr)
            self.assertNotIn(' | TTFT:', run('-r', '-i', 'a').stderr)
            tokenizer(eos=True)
            empty = run('-r', '-i', 'a', '-v')
            self.assertIn('output: 0 tok | TTFT: n/a | decode: n/a', empty.stderr)
            self.assertIn('stop: eos', empty.stderr)

    def test_pipeline(self):
        for tied in (True, False):
            with self.subTest(tied=tied), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)
                checkpoint(path, tied)
                model = path / 'model.bin'
                export.export_model(path, model)
                subprocess.run([str(self.runner), str(model), str(int(tied))], check=True)
                try:
                    import shrink
                except ImportError:
                    self.skipTest('numpy is required for shrink validation')
                for quant in ('bf16', 'q8', 'q4'):
                    small = path / f'{quant}.bin'
                    shrink.shrink_model(model, small, [0, 2, 4, 6], quant, 'q8' if quant == 'q4' else quant)
                    subprocess.run([str(self.runner), str(small), str(int(tied))], check=True)
                # mixed precision + AWQ, with made-up activation statistics (2 layers: 8 + 7 matrices)
                import numpy as np
                rng = np.random.default_rng(0)
                stats = [rng.uniform(0.1, 3, c) for c in [32] * 4 + [32] + [32, 32, 64] + [32] * 3 + [32] + [32, 32, 64]]
                small = path / 'mixed.bin'
                shrink.shrink_model(model, small, [0, 2, 4, 6], 'q4', 'q8', ['*.w_down', '1.*'], stats, ['*.w_qkv', '*.wo', '*.w_up'], 8)
                subprocess.run([str(self.runner), str(small), str(int(tied))], check=True)
                if tied:
                    # v2's header ends four bytes earlier, with the same tensor alignment.
                    legacy = bytearray(model.read_bytes())
                    struct.pack_into('<i', legacy, 4, 2)
                    legacy[84:88] = b'\0' * 4
                    model.write_bytes(legacy)
                    subprocess.run([str(self.runner), str(model), '1'], check=True)


class AwqTest(unittest.TestCase):
    def test_scaling_keeps_the_layer_function(self):
        try:
            import numpy as np
            import shrink
        except ImportError:
            self.skipTest('numpy is required for AWQ validation')
        rng = np.random.default_rng(1)
        d, f = 64, 96
        t = dict(ln1=rng.uniform(.5, 2, d).astype(np.float32), ln2=rng.uniform(.5, 2, d).astype(np.float32),
                 **{n: rng.normal(size=(r, d)).astype(np.float32) for n, r in [('wq', 64), ('wk', 32), ('wv', 32)]},
                 w_gate=rng.normal(size=(f, d)).astype(np.float32), w_up=rng.normal(size=(f, d)).astype(np.float32),
                 w_down=rng.normal(size=(d, f)).astype(np.float32))
        x = rng.normal(size=d).astype(np.float32) * rng.uniform(.1, 10, d).astype(np.float32)

        def forward(t):
            silu = lambda v: v / (1 + np.exp(-v))
            n1, n2 = x * t['ln1'], x * t['ln2']
            h = silu(t['w_gate'] @ n2) * (t['w_up'] @ n2)
            return [t[n] @ n1 for n in ('wq', 'wk', 'wv')] + [t['w_down'] @ h]

        before, ln = forward(t), [t['ln1'].copy(), t['ln2'].copy(), t['w_up'].copy()]
        stats = {n: np.abs(x * t[ln]) + .1 for n, ln in (('wq', 'ln1'), ('w_gate', 'ln2'))}
        stats['w_down'] = rng.uniform(.01, 5, f)
        shrink.apply_awq(t, dict.fromkeys(t, 'q4'), stats)
        self.assertFalse(all(np.array_equal(a, t[n]) for a, n in zip(ln, ('ln1', 'ln2', 'w_up'))))
        for a, b in zip(before, forward(t)):
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-3)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(ROOT/'scripts'))
    unittest.main()
