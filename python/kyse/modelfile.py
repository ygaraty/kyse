"""
The model.bin header, and the order of the tensors after it (see map_weights in src/model.c).

  "QW35", int32 version
  13 int32 config, 2 float32 config, int32 layer type per layer (1 = attention)
  v2+: int32 format of the layer matrices, int32 format of the embedding
  v3+: int32 untied (the model has its own output head)
  v4:  int32 n, then the format of each matrix in file order (WeightType + 256 * low-rank rank)
"""
import struct
from dataclasses import dataclass

ALIGN = 64  # every tensor starts on a 64-byte boundary
TYPES = {"bf16": 0, "q8": 1, "q4": 2}  # WeightType in src/backend.h
TYPE_NAMES = {v: k for k, v in TYPES.items()}
GROUP = 32  # QUANT_GROUP in src/backend.h
BITS = {"bf16": 16, "q8": 8 + 16 / GROUP, "q4": 4 + 16 / GROUP}  # per weight, with the group scales
CONFIG = ("dim", "hidden_dim", "n_layers", "vocab_size", "n_heads", "n_kv_heads", "head_dim", "rot_dim",
          "lin_key_heads", "lin_value_heads", "lin_key_dim", "lin_value_dim", "conv_kernel")


@dataclass
class Header:
    version: int
    ints: list          # the 13 config ints, in CONFIG order
    floats: tuple       # rms_norm_eps, rope_theta
    layer_types: tuple  # 1 = gated attention, 0 = DeltaNet
    untied: bool
    mat_types: list     # per matrix in file order: (format name, low-rank rank)
    data_offset: int    # where the tensors start (before alignment)

    @property
    def config(self):
        return dict(zip(CONFIG, self.ints))

    def layout(self, layer):
        """(name, shape) of a layer's tensors in file order; 2-d ones are matrices."""
        D, F, L, V, nh, nkv, hd, _, lk, lv, dk, dv, K = self.ints
        conv_dim, val_dim = 2 * lk * dk + lv * dv, lv * dv
        if self.layer_types[layer]:
            mixer = [("wq", (2 * nh * hd, D)), ("wk", (nkv * hd, D)), ("wv", (nkv * hd, D)),
                     ("wo", (D, nh * hd)), ("q_norm", (hd,)), ("k_norm", (hd,))]
        else:
            mixer = [("w_qkv", (conv_dim, D)), ("w_z", (val_dim, D)), ("w_a", (lv, D)), ("w_b", (lv, D)),
                     ("w_out", (D, val_dim)), ("conv", (conv_dim * K,)), ("A_log", (lv,)),
                     ("dt_bias", (lv,)), ("norm", (dv,))]
        return [("ln1", (D,)), ("ln2", (D,))] + mixer + [("w_gate", (F, D)), ("w_up", (F, D)), ("w_down", (D, F))]

    def matrices(self):
        """(name, shape) of every matrix in file order: "embed", "<layer>.<name>", then "head" if untied."""
        D, V = self.ints[0], self.ints[3]
        mats = [("embed", (V, D))]
        for l in range(len(self.layer_types)):
            mats += [("%d.%s" % (l, n), s) for n, s in self.layout(l) if len(s) == 2]
        return mats + [("head", (V, D))] * self.untied


def read_header(data):
    """Parses the header at the start of `data` (bytes or mmap)."""
    if bytes(data[:4]) != b"QW35":
        raise ValueError("not a model.bin file (bad magic)")
    version = struct.unpack_from("<i", data, 4)[0]
    if version not in (1, 2, 3, 4):
        raise ValueError("unsupported model.bin version %d" % version)
    ints = list(struct.unpack_from("<13i", data, 8))
    floats = struct.unpack_from("<2f", data, 60)
    L = ints[2]
    layer_types = struct.unpack_from("<%di" % L, data, 68)
    off = 68 + 4 * L
    mat_type = embed_type = 0
    if version >= 2:
        mat_type, embed_type = struct.unpack_from("<ii", data, off)
        off += 8
    untied = False
    if version >= 3:
        untied = bool(struct.unpack_from("<i", data, off)[0])
        off += 4
    h = Header(version, ints, floats, layer_types, untied, [], off)
    mats = h.matrices()
    if version >= 4:
        (n,) = struct.unpack_from("<i", data, off)
        entries = struct.unpack_from("<%di" % n, data, off + 4)
        h.data_offset = off + 4 + 4 * n
        if n != len(mats):
            raise ValueError("model.bin lists %d matrix formats for %d matrices" % (n, len(mats)))
        h.mat_types = [(TYPE_NAMES[e & 255], e >> 8) for e in entries]
    else:
        h.mat_types = [(TYPE_NAMES[embed_type if name in ("embed", "head") else mat_type], 0) for name, _ in mats]
    return h


def read_header_file(path):
    with open(path, "rb") as f:
        head = f.read(68)
        if len(head) < 68 or head[:4] != b"QW35":
            raise ValueError("%s is not a model.bin file" % path)
        L = struct.unpack_from("<i", head, 16)[0]
        return read_header(head + f.read(4 * L + 16 + 4 * (4 * L * 8 + 8)))
