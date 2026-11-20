"""Small deterministic checkpoints exercise export -> shrink -> C inference."""
import contextlib
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'python'))
from kyse import export  # noqa: E402


def checkpoint(path, tied):
    d, f, hd, nk, nv, dk, dv = 32, 64, 8, 2, 4, 8, 8
    cfg = dict(hidden_size=d, intermediate_size=f, num_hidden_layers=2, vocab_size=32,
               num_attention_heads=4, num_key_value_heads=2, head_dim=hd,
               linear_num_key_heads=nk, linear_num_value_heads=nv,
               linear_key_head_dim=dk, linear_value_head_dim=dv, linear_conv_kernel_dim=4,
               rms_norm_eps=1e-6, rope_parameters=dict(partial_rotary_factor=.5, rope_theta=1e7),
               layer_types=['linear_attention', 'full_attention'])
    (path / 'config.json').write_text(json.dumps(dict(text_config=cfg, tie_word_embeddings=tied)))
    tensors = {}

    def tensor(name, rows, cols=None, norm=False):
        n = rows * (cols or 1)
        values = [0.0 if norm else ((i * 17 + len(name)) % 29 - 14) / 128 for i in range(n)]
        dtype = 'BF16' if cols else 'F32'
        raw = b''.join(struct.pack('<I', struct.unpack('<I', struct.pack('<f', v))[0])[2:]
                       for v in values) if cols else struct.pack('<%df' % n, *values)
        tensors[name] = (dtype, [rows, cols] if cols else [rows], raw)

    p = 'model.language_model.'
    tensor(p + 'embed_tokens.weight', 32, d)
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
        tensor('lm_head.weight', 32, d)
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

    def test_pipeline(self):
        for tied in (True, False):
            with self.subTest(tied=tied), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)
                checkpoint(path, tied)
                model = path / 'model.bin'
                export.export_model(path, model)
                subprocess.run([str(self.runner), str(model), str(int(tied))], check=True)
                try:
                    from kyse import shrink
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
            from kyse import shrink
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


class CliTest(unittest.TestCase):
    """The `kyse` command on the small checkpoint, without the C bench tool (token counts given)."""

    @classmethod
    def setUpClass(cls):
        try:
            from kyse import shrink
        except ImportError:
            raise unittest.SkipTest('numpy is required for the kyse command')
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        hf, cls.src = cls.root / 'hf', cls.root / 'src'
        hf.mkdir()
        cls.src.mkdir()
        checkpoint(hf, True)
        export.export_model(hf, cls.src / 'model.bin')
        # 32 tokens: the chat template's words, two pairs and single bytes
        tokens = [b'system', b'user', b'assistant', b'\n', b'\n\n', b'ab', b'cd'] + [bytes([c]) for c in b'abcdefghijklmnopqrstuvwxy']
        shrink.write_tokenizer(cls.src / 'tokenizer.bin', tokens, [-1] * 32, [])
        (cls.root / 'counts.txt').write_text('5 10\n6 1\n')  # "ab" is common, "cd" is not

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def kyse(self, *args):
        from kyse.cli import main
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main([str(a) for a in args])
        return code, out.getvalue(), err.getvalue()

    def test_compress_and_inspect(self):
        out = self.root / 'int4'
        code, _, err = self.kyse('compress', self.src, out, '-p', 'int4-plain', '--int8', '*.wo',
                                 '--counts', self.root / 'counts.txt')
        self.assertEqual(code, 0, err)
        self.assertIn('vocabulary 32 -> 31 tokens', err)
        code, text, _ = self.kyse('inspect', out)
        self.assertEqual(code, 0)
        self.assertIn('vocabulary 31', text)
        self.assertRegex(text, r'wo\s+int8\n')
        self.assertRegex(text, r'w_gate\s+int4 ×2')
        self.assertRegex(text, r'embed\s+int8')
        # a second run must not overwrite without --force
        code, _, err = self.kyse('compress', self.src, out, '-p', 'int4-plain', '--counts', self.root / 'counts.txt')
        self.assertEqual(code, 1)
        self.assertIn('--force', err)

    def test_keep_vocab_copies_the_tokenizer(self):
        out = self.root / 'int8'
        code, _, err = self.kyse('compress', self.src, out, '--keep-vocab')
        self.assertEqual(code, 0, err)
        self.assertEqual((out / 'tokenizer.bin').read_bytes(), (self.src / 'tokenizer.bin').read_bytes())

    def test_mistakes_are_explained(self):
        code, _, err = self.kyse('compress', self.src, self.root / 'x', '--keep-vocab', '--int8', '*.w_dwn')
        self.assertEqual(code, 1)
        self.assertIn("'*.w_dwn' matches no layer matrix", err)
        code, _, err = self.kyse('compress', self.src, self.root / 'x', '-p', 'int4', '--keep-vocab', '--calib-text', self.root / 'missing.txt')
        self.assertEqual(code, 1)
        self.assertIn('calibration needs English text', err)
        code, _, err = self.kyse('compress', self.root / 'nowhere', self.root / 'x', '--keep-vocab')
        self.assertEqual(code, 1)
        self.assertIn('no model at', err)
        self.assertFalse((self.root / 'x').exists())

    def test_dry_run_writes_nothing(self):
        code, _, err = self.kyse('compress', self.src, self.root / 'dry', '-p', 'int4-mixed', '--counts',
                                 self.root / 'counts.txt', '--calib-text', self.root / 'counts.txt', '-n')
        self.assertEqual(code, 0, err)
        self.assertIn('int4 with AWQ; int8: *.wo, *.w_out, *.w_down', err)
        self.assertFalse((self.root / 'dry').exists())
        self.assertFalse((self.src / '.kyse').exists())


if __name__ == '__main__':
    unittest.main()
