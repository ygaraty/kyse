"""
Convert a Hugging Face Qwen3.5 checkpoint into two flat files the C code can mmap:

  model.bin      header + tensors (matrices as bf16, everything else as f32)
  tokenizer.bin  vocab (token bytes) + merge ranks

Only uses the python standard library.

usage: python export.py <hf_model_dir> <out_dir>
"""
import json, os, struct, sys

ALIGN = 64  # every tensor starts on a 64-byte boundary (nice for GPU loads)


def load_safetensors(path):
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return f, 8 + n, header


def read_f32(f, base, info):
    a, b = info["data_offsets"]
    f.seek(base + a)
    raw = f.read(b - a)
    if info["dtype"] == "F32":
        return list(struct.unpack("<%df" % (len(raw) // 4), raw))
    assert info["dtype"] == "BF16", info["dtype"]
    u16 = struct.unpack("<%dH" % (len(raw) // 2), raw)
    return list(struct.unpack("<%df" % len(u16), struct.pack("<%dI" % len(u16), *[u << 16 for u in u16])))


def read_bf16(f, base, info):
    assert info["dtype"] == "BF16", info["dtype"]
    a, b = info["data_offsets"]
    f.seek(base + a)
    return f.read(b - a)


def export_model(hf_dir, out_path):
    cfg = json.load(open(os.path.join(hf_dir, "config.json")))["text_config"]
    st = [p for p in os.listdir(hf_dir) if p.endswith(".safetensors")]
    assert len(st) == 1, "expected a single safetensors file"
    f, base, h = load_safetensors(os.path.join(hf_dir, st[0]))

    L = cfg["num_hidden_layers"]
    layer_types = [1 if t == "full_attention" else 0 for t in cfg["layer_types"]]
    rot_dim = int(cfg["head_dim"] * cfg["rope_parameters"]["partial_rotary_factor"])

    out = open(out_path, "wb")

    def pad():
        out.write(b"\0" * ((-out.tell()) % ALIGN))

    # ---- header: magic, version, int32 config, float32 config, layer types,
    #      then the formats of the layer matrices and of the embedding (0 = bf16)
    ints = [
        cfg["hidden_size"], cfg["intermediate_size"], L, cfg["vocab_size"],
        cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"], rot_dim,
        cfg["linear_num_key_heads"], cfg["linear_num_value_heads"],
        cfg["linear_key_head_dim"], cfg["linear_value_head_dim"], cfg["linear_conv_kernel_dim"],
    ]
    floats = [cfg["rms_norm_eps"], cfg["rope_parameters"]["rope_theta"]]
    out.write(b"QW35")
    out.write(struct.pack("<i", 2))
    out.write(struct.pack("<%di" % len(ints), *ints))
    out.write(struct.pack("<%df" % len(floats), *floats))
    out.write(struct.pack("<%di" % L, *layer_types))
    out.write(struct.pack("<ii", 0, 0))
    pad()

    p = "model.language_model."

    def mat(name):  # bf16 matrix for the matvec backend
        pad()
        out.write(read_bf16(f, base, h[p + name]))

    def vec(name, plus_one=False):  # small f32 tensor used on the CPU
        pad()
        x = read_f32(f, base, h[p + name])
        if plus_one:  # Qwen3.5 RMSNorm computes x * (1 + w); bake the +1 in here
            x = [v + 1.0 for v in x]
        out.write(struct.pack("<%df" % len(x), *x))

    # order must match the C loader exactly
    mat("embed_tokens.weight")
    for i in range(L):
        q = "layers.%d." % i
        vec(q + "input_layernorm.weight", plus_one=True)
        vec(q + "post_attention_layernorm.weight", plus_one=True)
        if layer_types[i]:
            a = q + "self_attn."
            mat(a + "q_proj.weight")
            mat(a + "k_proj.weight")
            mat(a + "v_proj.weight")
            mat(a + "o_proj.weight")
            vec(a + "q_norm.weight", plus_one=True)
            vec(a + "k_norm.weight", plus_one=True)
        else:
            a = q + "linear_attn."
            mat(a + "in_proj_qkv.weight")
            mat(a + "in_proj_z.weight")
            mat(a + "in_proj_a.weight")
            mat(a + "in_proj_b.weight")
            mat(a + "out_proj.weight")
            vec(a + "conv1d.weight")
            vec(a + "A_log")
            vec(a + "dt_bias")
            vec(a + "norm.weight")  # gated norm: plain x * w, no +1
        mat(q + "mlp.gate_proj.weight")
        mat(q + "mlp.up_proj.weight")
        mat(q + "mlp.down_proj.weight")
    vec("norm.weight", plus_one=True)
    pad()
    print("wrote %s (%.1f MB)" % (out_path, out.tell() / 1e6))


def bytes_to_unicode():
    # GPT-2 byte-level mapping: every byte gets a printable unicode char
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def export_tokenizer(hf_dir, out_path, vocab_size):
    t = json.load(open(os.path.join(hf_dir, "tokenizer.json"), encoding="utf-8"))
    byte_dec = {c: b for b, c in bytes_to_unicode().items()}

    tokens = [b""] * vocab_size
    ranks = [-1] * vocab_size  # rank of the merge that creates this token (lower = merged first)
    for s, i in t["model"]["vocab"].items():
        tokens[i] = bytes(byte_dec[c] for c in s)
    vocab = t["model"]["vocab"]
    for r, m in enumerate(t["model"]["merges"]):
        a, b = m.split(" ") if isinstance(m, str) else m
        i = vocab[a + b]
        if ranks[i] < 0:
            ranks[i] = r
    special = []
    for a in t["added_tokens"]:
        tokens[a["id"]] = a["content"].encode("utf-8")
        special.append(a["id"])

    with open(out_path, "wb") as out:
        out.write(struct.pack("<iii", vocab_size, len(special), max(len(x) for x in tokens)))
        out.write(struct.pack("<%di" % len(special), *special))
        for tok, r in zip(tokens, ranks):
            out.write(struct.pack("<ii", r, len(tok)))
            out.write(tok)
    print("wrote %s" % out_path)


if __name__ == "__main__":
    hf_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    cfg = json.load(open(os.path.join(hf_dir, "config.json")))["text_config"]
    export_model(hf_dir, os.path.join(out_dir, "model.bin"))
    export_tokenizer(hf_dir, os.path.join(out_dir, "tokenizer.bin"), cfg["vocab_size"])
