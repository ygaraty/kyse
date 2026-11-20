"""
Shrink a model.bin: keep only the tokens a corpus uses, and quantize the matrices.

  1. Vocabulary. Tokens that occur at least --min-count times in the corpus are
     kept (counts come from `./bench count tokenizer.bin corpus.txt`), plus the
     256 single bytes (so any text can still be encoded), the special tokens,
     the words of the chat template in main.c, and every token the BPE builds on the way to a kept token (so text made of
     kept tokens is tokenized exactly as before). Both the embedding table and the LM head keep only those rows.
  2. Quantization. Every matrix is stored as int8 or int4 with one f16 scale per
     group of 32 weights (see WeightType in backend.h). For int4 the scale of
     each group is picked from a few candidates to minimize the squared error.

Writes <out_dir>/model.bin and <out_dir>/tokenizer.bin. Needs numpy.

usage: python shrink.py <model.bin> <tokenizer.bin> <counts.txt> <out_dir>
                        [--quant q8|q4|bf16] [--embed-quant q8|q4|bf16] [--min-count N]
"""
import argparse, mmap, os, struct
import numpy as np

ALIGN = 64  # must match export.py / model.c
GROUP = 32  # QUANT_GROUP in backend.h
TYPES = {"bf16": 0, "q8": 1, "q4": 2}  # WeightType in backend.h
CHAT_WORDS = [b"system", b"user", b"assistant", b"\n", b"\n\n"]  # USER_TURN / SYSTEM_TURN in main.c


# ---------------------------------------------------------------------------
# tokenizer.bin: vocab_size, n_special, max_len, special ids, then per token: rank, len, bytes

def read_tokenizer(path):
    data = open(path, "rb").read()
    vocab_size, n_special, _ = struct.unpack_from("<iii", data, 0)
    off = 12
    special = list(struct.unpack_from("<%di" % n_special, data, off))
    off += 4 * n_special
    tokens, ranks = [], []
    for _ in range(vocab_size):
        rank, n = struct.unpack_from("<ii", data, off)
        off += 8
        tokens.append(data[off:off + n])
        ranks.append(rank)
        off += n
    return tokens, ranks, special


def write_tokenizer(path, tokens, ranks, special):
    with open(path, "wb") as out:
        out.write(struct.pack("<iii", len(tokens), len(special), max(len(t) for t in tokens)))
        out.write(struct.pack("<%di" % len(special), *special))
        for tok, r in zip(tokens, ranks):
            out.write(struct.pack("<ii", r, len(tok)))
            out.write(tok)


def bpe_pieces(word, ids, ranks):
    """Every token the C tokenizer (tokenizer.c, bpe) creates while encoding `word`."""
    pieces, seen = [word[i:i + 1] for i in range(len(word))], set()
    while True:
        best, best_rank = -1, None
        for i in range(len(pieces) - 1):
            t = ids.get(pieces[i] + pieces[i + 1])
            if t is not None and ranks[t] >= 0 and (best_rank is None or ranks[t] < best_rank):
                best, best_rank = i, ranks[t]
        if best < 0:
            return seen
        pieces[best:best + 2] = [pieces[best] + pieces[best + 1]]
        seen.add(ids[pieces[best]])


def choose_vocab(tokens, ranks, special, counts, min_count):
    ids = {t: i for i, t in enumerate(tokens) if t and i not in special}
    keep = {i for i, c in counts.items() if c >= min_count}
    keep |= set(special)
    keep |= {i for i, t in enumerate(tokens) if len(t) == 1}
    keep |= {ids[w] for w in CHAT_WORDS}
    for i in list(keep):
        if i not in special and len(tokens[i]) > 1:
            keep |= bpe_pieces(tokens[i], ids, ranks)
    return sorted(keep)


# ---------------------------------------------------------------------------
# quantization: rows x cols float32 -> data bytes + f16 scales (row-major groups of 32)

def quantize(w, kind):
    if kind == "bf16":
        return (w.view(np.uint32) >> 16).astype(np.uint16).tobytes()  # input is exact bf16 already
    g = w.reshape(-1, GROUP)
    amax = np.abs(g).max(axis=1, keepdims=True)
    if kind == "q8":
        scale = (amax / 127).astype(np.float16)
        s = scale.astype(np.float32)
        q = np.clip(np.round(np.divide(g, s, out=np.zeros_like(g), where=s > 0)), -127, 127).astype(np.int8)
        return q.tobytes() + scale.tobytes()

    # q4: q in -8..7; try a few scales below absmax/7 (clipping the largest
    # weight a bit often lowers the error of all the others)
    best_err = np.full((g.shape[0], 1), np.inf, np.float32)
    best_q = np.zeros(g.shape, np.int8)
    best_scale = np.zeros((g.shape[0], 1), np.float16)
    for f in np.linspace(0.70, 1.0, 13):
        scale = (amax * f / 7).astype(np.float16)
        s = scale.astype(np.float32)
        q = np.clip(np.round(np.divide(g, s, out=np.zeros_like(g), where=s > 0)), -8, 7)
        err = ((q * s - g) ** 2).sum(axis=1, keepdims=True)
        better = err < best_err
        best_err = np.where(better, err, best_err)
        best_q = np.where(better, q, best_q).astype(np.int8)
        best_scale = np.where(better, scale, best_scale)
    u = (best_q.reshape(-1) + 8).astype(np.uint8)
    return (u[0::2] | (u[1::2] << 4)).tobytes() + best_scale.tobytes()


def quantize_rows(w, kind, block=4096):
    """quantize() in row blocks to bound memory; data of all blocks first, then all scales."""
    if kind == "bf16":
        return quantize(w, kind)
    data, scales = [], []
    for r in range(0, w.shape[0], block):
        b = quantize(np.ascontiguousarray(w[r:r + block]), kind)
        n = len(b) - w[r:r + block].size // GROUP * 2
        data.append(b[:n])
        scales.append(b[n:])
    return b"".join(data) + b"".join(scales)


# ---------------------------------------------------------------------------
# model.bin

class Reader:
    def __init__(self, data, off):
        self.data, self.off = data, off

    def take(self, nbytes):
        self.off += -self.off % ALIGN
        if self.off + nbytes > len(self.data):
            raise ValueError("model file is truncated")
        b = memoryview(self.data)[self.off:self.off + nbytes]
        self.off += nbytes
        return b

    def mat(self, rows, cols):
        u16 = np.frombuffer(self.take(rows * cols * 2), np.uint16).reshape(rows, cols)
        return (u16.astype(np.uint32) << 16).view(np.float32)

    def vec(self, n):
        return self.take(4 * n)


def shrink_model(src, dst, keep, quant, embed_quant):
    source = open(src, "rb")
    data = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)
    assert bytes(data[:4]) == b"QW35", "not a model.bin"
    version = struct.unpack_from("<i", data, 4)[0]
    if version not in (1, 2, 3):
        raise ValueError("unsupported model version")
    ints = list(struct.unpack_from("<13i", data, 8))
    floats = struct.unpack_from("<2f", data, 8 + 52)
    D, F, L, V, nh, nkv, hd, _, lk, lv, dk, dv, K = ints
    layer_types = struct.unpack_from("<%di" % L, data, 68)
    off = 68 + 4 * L
    if version >= 2:
        assert struct.unpack_from("<ii", data, off) == (0, 0), "input must be a bf16 model.bin"
        off += 8
    untied = version >= 3 and struct.unpack_from("<i", data, off)[0]
    if version >= 3:
        off += 4
    r = Reader(data, off)
    key_dim, val_dim = lk * dk, lv * dv
    conv_dim = 2 * key_dim + val_dim

    out = open(dst, "wb")

    def pad():
        out.write(b"\0" * (-out.tell() % ALIGN))

    def mat(rows, cols, kind=quant):
        pad()
        out.write(quantize_rows(r.mat(rows, cols), kind))

    def vec(n):
        pad()
        out.write(r.vec(n))

    ints[3] = len(keep)
    out.write(b"QW35" + struct.pack("<i", 3))
    out.write(struct.pack("<13i", *ints) + struct.pack("<2f", *floats))
    out.write(struct.pack("<%di" % L, *layer_types))
    out.write(struct.pack("<ii", TYPES[quant], TYPES[embed_quant]))

    out.write(struct.pack("<i", int(bool(untied))))

    # same order as export.py / map_weights() in model.c
    def table():
        # Select bf16 rows before expanding: the full 9B table is ~4 GB in f32.
        raw = np.frombuffer(r.take(V * D * 2), dtype="<u2").reshape(V, D)
        selected = raw[keep]
        weights = (selected.astype(np.uint32) << 16).view(np.float32)
        pad()
        out.write(quantize_rows(weights, embed_quant))

    table()
    for l in range(L):
        vec(D)
        vec(D)
        if layer_types[l]:
            mat(2 * nh * hd, D)
            mat(nkv * hd, D)
            mat(nkv * hd, D)
            mat(D, nh * hd)
            vec(hd)
            vec(hd)
        else:
            mat(conv_dim, D)
            mat(val_dim, D)
            mat(lv, D)
            mat(lv, D)
            mat(D, val_dim)
            vec(conv_dim * K)
            vec(lv)
            vec(lv)
            vec(dv)
        mat(F, D)
        mat(F, D)
        mat(D, F)
        print("\rlayer %d/%d" % (l + 1, L), end="", flush=True)
    vec(D)
    if untied:
        table()
    pad()
    size = out.tell()
    out.close()
    data.close()
    source.close()
    print()
    return size


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("tokenizer")
    ap.add_argument("counts", help="output of ./bench count")
    ap.add_argument("out_dir")
    ap.add_argument("--quant", choices=TYPES, default="q8", help="layer matrices (default q8)")
    ap.add_argument("--embed-quant", choices=TYPES, help="embedding / LM head (default: q8 for q4, else --quant)")
    ap.add_argument("--min-count", type=int, default=2, help="keep tokens seen at least this often (default 2)")
    a = ap.parse_args()
    embed_quant = a.embed_quant or ("q8" if a.quant == "q4" else a.quant)

    tokens, ranks, special = read_tokenizer(a.tokenizer)
    counts = {}
    for line in open(a.counts):
        i, c = line.split()
        counts[int(i)] = int(c)
    keep = choose_vocab(tokens, ranks, special, counts, a.min_count)
    new_id = {old: new for new, old in enumerate(keep)}
    print("vocab: %d -> %d tokens" % (len(tokens), len(keep)))

    os.makedirs(a.out_dir, exist_ok=True)
    write_tokenizer(os.path.join(a.out_dir, "tokenizer.bin"), [tokens[i] for i in keep],
                    [ranks[i] for i in keep], [new_id[i] for i in special])
    size = shrink_model(a.model, os.path.join(a.out_dir, "model.bin"), keep, a.quant, embed_quant)
    print("wrote %s (%.1f MB, layers %s, embedding %s)" % (a.out_dir, size / 1e6, a.quant, embed_quant))


if __name__ == "__main__":
    main()
