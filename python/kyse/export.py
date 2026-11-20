"""
Convert a Hugging Face Qwen3.5 checkpoint into two flat files the C code can mmap:

  model.bin      header + tensors (matrices as bf16, everything else as f32)
  tokenizer.bin  vocab (token bytes) + merge ranks

Only uses the python standard library.
"""
import json, os, struct
from contextlib import ExitStack

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
    """Writes model.bin; returns its size."""
    with ExitStack() as stack:
        return _export_model(hf_dir, out_path, stack)


def _export_model(hf_dir, out_path, stack):
    with open(os.path.join(hf_dir, "config.json")) as config_file:
        root = json.load(config_file)
    cfg = root["text_config"]
    # Keep shard handles open, but stream matrix payloads instead of loading them.
    index_path = os.path.join(hf_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as index:
            shards = sorted(set(json.load(index)["weight_map"].values()))
    else:
        shards = sorted(p for p in os.listdir(hf_dir) if p.endswith(".safetensors"))
    if not shards:
        raise ValueError("no safetensors checkpoint found")
    tensors = {}
    for shard in shards:
        f, base, header = load_safetensors(os.path.join(hf_dir, shard))
        stack.enter_context(f)
        for name, info in header.items():
            if name in tensors:
                raise ValueError("duplicate tensor: " + name)
            tensors[name] = (f, base, info)
    tied = cfg.get("tie_word_embeddings",
                   root.get("tie_word_embeddings", False))

    L = cfg["num_hidden_layers"]
    layer_types = [1 if t == "full_attention" else 0 for t in cfg["layer_types"]]
    rot_dim = int(cfg["head_dim"] * cfg["rope_parameters"]["partial_rotary_factor"])

    out = stack.enter_context(open(out_path, "wb"))

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
    out.write(struct.pack("<i", 3))
    out.write(struct.pack("<%di" % len(ints), *ints))
    out.write(struct.pack("<%df" % len(floats), *floats))
    out.write(struct.pack("<%di" % L, *layer_types))
    out.write(struct.pack("<iii", 0, 0, int(not tied)))
    pad()

    p = "model.language_model."

    def mat(name):  # stream bf16 matrices in bounded chunks, including the 9B tables
        pad()
        f, base, info = tensors[name]
        if info["dtype"] != "BF16":
            raise ValueError("expected BF16 matrix: " + name)
        a, b = info["data_offsets"]
        f.seek(base + a)
        remaining = b - a
        while remaining:
            chunk = f.read(min(remaining, 8 * 1024 * 1024))
            if not chunk:
                raise ValueError("truncated tensor: " + name)
            out.write(chunk)
            remaining -= len(chunk)

    def vec(name, plus_one=False):  # small f32 tensor used on the CPU
        pad()
        f, base, info = tensors[p + name]
        x = read_f32(f, base, info)
        if plus_one:  # Qwen3.5 RMSNorm computes x * (1 + w); bake the +1 in here
            x = [v + 1.0 for v in x]
        out.write(struct.pack("<%df" % len(x), *x))

    # order must match the C loader exactly
    mat(p + "embed_tokens.weight")
    for i in range(L):
        q = "layers.%d." % i
        vec(q + "input_layernorm.weight", plus_one=True)
        vec(q + "post_attention_layernorm.weight", plus_one=True)
        if layer_types[i]:
            a = q + "self_attn."
            mat(p + a + "q_proj.weight")
            mat(p + a + "k_proj.weight")
            mat(p + a + "v_proj.weight")
            mat(p + a + "o_proj.weight")
            vec(a + "q_norm.weight", plus_one=True)
            vec(a + "k_norm.weight", plus_one=True)
        else:
            a = q + "linear_attn."
            mat(p + a + "in_proj_qkv.weight")
            mat(p + a + "in_proj_z.weight")
            mat(p + a + "in_proj_a.weight")
            mat(p + a + "in_proj_b.weight")
            mat(p + a + "out_proj.weight")
            vec(a + "conv1d.weight")
            vec(a + "A_log")
            vec(a + "dt_bias")
            vec(a + "norm.weight")  # gated norm: plain x * w, no +1
        mat(p + q + "mlp.gate_proj.weight")
        mat(p + q + "mlp.up_proj.weight")
        mat(p + q + "mlp.down_proj.weight")
    vec("norm.weight", plus_one=True)
    if not tied:
        mat("lm_head.weight")
    pad()
    return out.tell()


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


def export(hf_dir, out_dir):
    """Writes <out_dir>/model.bin and <out_dir>/tokenizer.bin; returns the model size."""
    config = os.path.join(hf_dir, "config.json")
    if not os.path.exists(config):
        raise ValueError("%s is not a Hugging Face checkpoint (no config.json)" % hf_dir)
    os.makedirs(out_dir, exist_ok=True)
    cfg = json.load(open(config))["text_config"]
    size = export_model(hf_dir, os.path.join(out_dir, "model.bin"))
    export_tokenizer(hf_dir, os.path.join(out_dir, "tokenizer.bin"), cfg["vocab_size"])
    return size
