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
     --q8 keeps chosen layer matrices at int8 (mixed precision), --awq
     rescales input channels by their activation size first (see AWQ below),
     and --lowrank moves a matrix's largest directions into a small bf16
     correction before quantizing the rest (see SVDQuant below).

Writes <out_dir>/model.bin and <out_dir>/tokenizer.bin. Needs numpy.

usage: python3 scripts/shrink.py <model.bin> <tokenizer.bin> <counts.txt> <out_dir>
                        [--quant q8|q4|bf16] [--embed-quant q8|q4|bf16] [--min-count N]
                        [--q8 PATTERNS] [--awq STATS] [--lowrank PATTERNS] [--rank R]
"""
import argparse, fnmatch, mmap, os, struct
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

def quantize_groups(g, kind):
    """[n, GROUP] float32 -> integer codes q [n, GROUP] and f16 scales [n, 1], with g ~ q * scale."""
    amax = np.abs(g).max(axis=1, keepdims=True)
    if kind == "q8":
        scale = (amax / 127).astype(np.float16)
        s = scale.astype(np.float32)
        return np.clip(np.round(np.divide(g, s, out=np.zeros_like(g), where=s > 0)), -127, 127), scale

    # q4: q in -8..7; try a few scales below absmax/7 (clipping the largest
    # weight a bit often lowers the error of all the others)
    best_err = np.full((g.shape[0], 1), np.inf, np.float32)
    best_q = np.zeros(g.shape, np.float32)
    best_scale = np.zeros((g.shape[0], 1), np.float16)
    for f in np.linspace(0.70, 1.0, 13):
        scale = (amax * f / 7).astype(np.float16)
        s = scale.astype(np.float32)
        q = np.clip(np.round(np.divide(g, s, out=np.zeros_like(g), where=s > 0)), -8, 7)
        err = ((q * s - g) ** 2).sum(axis=1, keepdims=True)
        better = err < best_err
        best_err = np.where(better, err, best_err)
        best_q = np.where(better, q, best_q)
        best_scale = np.where(better, scale, best_scale)
    return best_q, best_scale


def quantize(w, kind):
    if kind == "bf16":  # round to nearest even (AWQ scaling leaves values that are not bf16)
        u = w.view(np.uint32)
        return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16).tobytes()
    q, scale = quantize_groups(w.reshape(-1, GROUP), kind)
    if kind == "q8":
        return q.astype(np.int8).tobytes() + scale.tobytes()
    u = (q.reshape(-1) + 8).astype(np.uint8)
    return (u[0::2] | (u[1::2] << 4)).tobytes() + scale.tobytes()


def dequantized(w, kind):
    """The weights the C code computes with after quantize()."""
    if kind == "bf16":
        return w
    q, scale = quantize_groups(w.reshape(-1, GROUP), kind)
    return (q * scale.astype(np.float32)).reshape(w.shape)


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
# AWQ (activation-aware weight quantization, Lin et al. 2024)
#
# W x = (W diag(s)) (x / s). Scaling up the input channels that see large
# activations makes their rounding error smaller relative to the weight, and
# those are the channels whose error matters most. The 1/s folds into whatever
# produced x: the RMSNorm weight for the matrices that read a normed input,
# and the rows of w_up for w_down (whose input is silu(gate) * up).
# The out projections (wo, w_out) read gated inputs and are left as they are.

def read_stats(path):
    """Output of `bench calib`: the mean |x| of each input channel, per layer matrix."""
    data, off, stats = open(path, "rb").read(), 0, []
    while off < len(data):
        (cols,) = struct.unpack_from("<i", data, off)
        stats.append(np.frombuffer(data, "<f4", cols, off + 4).astype(np.float32))
        off += 4 + 4 * cols
    return stats


def awq_scale(mats, mean_abs, grid=11):
    """Input-channel scales s = mean|x|^alpha for matrices that share one input,
    with alpha picked to minimize the estimated output error
    sum_j mean|x_j|^2 (Q(W s)_ij / s_j - W_ij)^2 over every fourth row.
    (Weighting by mean x^2 instead does worse: a few channels with rare, huge
    values then decide everything.)"""
    best, best_err = None, np.inf
    for alpha in np.linspace(0, 1, grid):
        s = np.maximum(mean_abs, 1e-6) ** alpha
        s = (s / np.sqrt(s.max() * s.min())).astype(np.float32)
        err = sum((((dequantized(w[::4] * s, kind) / s - w[::4]) ** 2) * mean_abs ** 2).sum() for w, kind in mats)
        if err < best_err:
            best, best_err = s, err
    return best


def apply_awq(t, kinds, stats):
    """Scales the matrices of one layer (tensors t, by name) in place; returns the
    scales, keyed by the matrix's name."""
    scales = {}
    groups = [("ln1", ["wq", "wk", "wv"] if "wq" in t else ["w_qkv", "w_z", "w_a", "w_b"]),
              ("ln2", ["w_gate", "w_up"]),
              ("w_up", ["w_down"])]
    for prev, names in groups:
        if all(kinds[n] == "bf16" for n in names):
            continue
        s = awq_scale([(t[n], kinds[n]) for n in names], stats[names[0]])
        for n in names:
            t[n] *= s
            scales[n] = s
        if prev == "w_up":
            t[prev] /= s[:, None]  # rows: one per channel of w_down's input
        else:
            t[prev] /= s
    return scales


# ---------------------------------------------------------------------------
# low-rank correction (SVDQuant, Li et al. 2025)
#
# W = up down + R, where up down is the best rank-r part of W (kept in bf16)
# and only the residual R is quantized. The few large singular directions,
# which carry the outlier weights, leave R, so its groups get finer scales.

def bf16(x):
    """x rounded to the nearest bf16 values, as float32"""
    u16 = np.frombuffer(quantize(np.ascontiguousarray(x, np.float32), "bf16"), np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32).reshape(x.shape)


def low_rank(w, rank, weight):
    """bf16 up [rows, rank] and down [rank, cols] with up @ down closest to w in
    the norm where column j counts weight_j^2 (its input's mean |x|): an SVD of w diag(weight)."""
    weight = np.maximum(weight, weight.max() * 1e-6).astype(np.float32)
    u, sv, vt = np.linalg.svd(w * weight, full_matrices=False)
    return bf16(u[:, :rank] * sv[:rank]), bf16(vt[:rank] / weight)


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
        return np.frombuffer(self.take(4 * n), "<f4").copy()


def shrink_model(src, dst, keep, quant, embed_quant, q8=(), stats=None, lowrank=(), rank=16):
    """q8: name patterns (fnmatch, e.g. "*.w_down" or "0.*") of layer matrices
    kept at q8 whatever `quant` is; stats: read_stats() output to apply AWQ;
    lowrank: patterns of layer matrices that get a rank-`rank` correction."""
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

    # (name, shape) of each layer's tensors in model.bin order; 2-d ones are matrices
    def layout(attn):
        if attn:
            mixer = [("wq", (2 * nh * hd, D)), ("wk", (nkv * hd, D)), ("wv", (nkv * hd, D)),
                     ("wo", (D, nh * hd)), ("q_norm", (hd,)), ("k_norm", (hd,))]
        else:
            mixer = [("w_qkv", (conv_dim, D)), ("w_z", (val_dim, D)), ("w_a", (lv, D)), ("w_b", (lv, D)),
                     ("w_out", (D, val_dim)), ("conv", (conv_dim * K,)), ("A_log", (lv,)),
                     ("dt_bias", (lv,)), ("norm", (dv,))]
        return [("ln1", (D,)), ("ln2", (D,))] + mixer + [("w_gate", (F, D)), ("w_up", (F, D)), ("w_down", (D, F))]

    kinds = [{n: "q8" if any(fnmatch.fnmatchcase("%d.%s" % (l, n), p) for p in q8) else quant
              for n, shape in layout(layer_types[l]) if len(shape) == 2} for l in range(L)]
    ranks = [{n: rank if any(fnmatch.fnmatchcase("%d.%s" % (l, n), p) for p in lowrank) else 0 for n in kinds[l]}
             for l in range(L)]
    for l in range(L):
        for n, shape in layout(layer_types[l]):
            if len(shape) == 2 and ranks[l][n] > min(shape):
                raise ValueError("rank %d is too large for %d.%s %s" % (rank, l, n, shape))
    # the format list entry of every matrix in file order (WeightType + 256 * rank),
    # and the input statistics of each layer's matrices
    mat_types = ([TYPES[embed_quant]] + [TYPES[kinds[l][n]] + 256 * ranks[l][n] for l in range(L) for n in kinds[l]]
                 + [TYPES[embed_quant]] * bool(untied))
    if stats is not None:
        n = sum(len(k) for k in kinds)
        if len(stats) != n:
            raise ValueError("statistics are for %d matrices, the model has %d" % (len(stats), n))
        it = iter(stats)
        stats = [{name: next(it) for name in k} for k in kinds]

    out = open(dst, "wb")

    def pad():
        out.write(b"\0" * (-out.tell() % ALIGN))

    ints[3] = len(keep)
    out.write(b"QW35" + struct.pack("<i", 4))
    out.write(struct.pack("<13i", *ints) + struct.pack("<2f", *floats))
    out.write(struct.pack("<%di" % L, *layer_types))
    out.write(struct.pack("<iii", TYPES[quant], TYPES[embed_quant], int(bool(untied))))
    out.write(struct.pack("<%di" % (len(mat_types) + 1), len(mat_types), *mat_types))

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
        t = {n: r.mat(*shape) if len(shape) == 2 else r.vec(*shape) for n, shape in layout(layer_types[l])}
        scales = apply_awq(t, kinds[l], stats[l]) if stats is not None else {}
        for n, shape in layout(layer_types[l]):
            pad()
            if len(shape) == 1:
                out.write(t[n].astype("<f4").tobytes())
            elif ranks[l][n]:
                weight = stats[l][n] / scales.get(n, 1) if stats is not None else np.ones(shape[1])
                up, down = low_rank(t[n], ranks[l][n], weight)
                out.write(quantize_rows(t[n] - up @ down, kinds[l][n]))
                pad()
                out.write(quantize(down, "bf16") + quantize(up, "bf16"))
            else:
                out.write(quantize_rows(t[n], kinds[l][n]))
        print("\rlayer %d/%d" % (l + 1, L), end="", flush=True)
    pad()
    out.write(r.vec(D).tobytes())
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
    ap.add_argument("--q8", default="", help="comma-separated layer matrices kept at q8, as <layer>.<name> "
                    "patterns, e.g. '*.w_down,0.*' (names as in model.h)")
    ap.add_argument("--awq", metavar="STATS", help="activation-aware scaling from `bench calib` statistics")
    ap.add_argument("--lowrank", default="", help="comma-separated layer matrices (patterns as for --q8) "
                    "that get a bf16 low-rank correction")
    ap.add_argument("--rank", type=int, default=16, help="rank of the --lowrank corrections (multiple of 4, default 16)")
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
    q8 = [p for p in a.q8.split(",") if p]
    lowrank = [p for p in a.lowrank.split(",") if p]
    if a.rank % 4 or a.rank <= 0:
        ap.error("--rank must be a positive multiple of 4")
    stats = read_stats(a.awq) if a.awq else None
    size = shrink_model(a.model, os.path.join(a.out_dir, "model.bin"), keep, a.quant, embed_quant, q8, stats,
                        lowrank, a.rank)
    print("wrote %s (%.1f MB, layers %s%s%s%s, embedding %s)" % (a.out_dir, size / 1e6, a.quant,
          ", q8 " + a.q8 if q8 else "", ", AWQ" if stats else "",
          ", rank %d on %s" % (a.rank, a.lowrank) if lowrank else "", embed_quant))


if __name__ == "__main__":
    main()
