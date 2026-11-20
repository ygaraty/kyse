"""Small deterministic checkpoints exercise export -> shrink -> C inference."""
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('export', ROOT / 'export.py')
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


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
            extra = ['-DKYSE_TEST_METAL', str(ROOT/'backend_accelerate.c'), str(ROOT/'backend_metal.m'), '-fobjc-arc',
                     '-framework', 'Metal', '-framework', 'Foundation', '-framework', 'Accelerate']
        subprocess.run([os.environ.get('CC', 'cc'), '-std=gnu23', '-Wall', '-Wextra',
                        '-I', str(ROOT), str(ROOT/'tests/model_test.c'), str(ROOT/'model.c'),
                        str(ROOT/'ops_ref.c'), str(ROOT/'backend_cpu.c'), '-lm', '-o', str(cls.runner)] + extra, check=True)

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
                    import shrink
                except ImportError:
                    self.skipTest('numpy is required for shrink validation')
                for quant in ('bf16', 'q8', 'q4'):
                    small = path / f'{quant}.bin'
                    shrink.shrink_model(model, small, [0, 2, 4, 6], quant, 'q8' if quant == 'q4' else quant)
                    subprocess.run([str(self.runner), str(small), str(int(tied))], check=True)
                if tied:
                    # v2's header ends four bytes earlier, with the same tensor alignment.
                    legacy = bytearray(model.read_bytes())
                    struct.pack_into('<i', legacy, 4, 2)
                    legacy[84:88] = b'\0' * 4
                    model.write_bytes(legacy)
                    subprocess.run([str(self.runner), str(model), '1'], check=True)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(ROOT))
    unittest.main()
