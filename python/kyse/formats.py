"""
Compare weight formats offline, before writing any kernel for them.

For every layer matrix of a bf16 model.bin, each format quantizes the weights
and we measure the activation-weighted error sum_j mean|x_j|^2 (Q(W) - W)_ij^2,
the same estimate AWQ minimizes (mean |x| from `bench calib`). The result is
relative to the format we ship (symmetric int4, groups of 32).

Formats:
  sym     q * scale,        q in -2^(b-1)..2^(b-1)-1, one f16 scale per group
          (scale picked from 13 candidates below absmax, as in shrink.py)
  affine  q * scale + bias, q in 0..2^b-1, f16 scale and f16 bias per group
          (MLX's format: bias = group min, scale = range / (2^b - 1))
  affine+ affine with the range shrunk from both ends by a searched factor
"""
import mmap
import numpy as np

from .modelfile import read_header
from .shrink import Reader

F16 = lambda x: x.astype(np.float16).astype(np.float32)


def sym(g, bits):
    amax = np.abs(g).max(axis=1, keepdims=True)
    lo, hi = -2 ** (bits - 1), 2 ** (bits - 1) - 1
    best, best_err = None, None
    for f in np.linspace(0.70, 1.0, 13):
        s = F16(amax * f / hi)
        q = np.clip(np.round(np.divide(g, s, out=np.zeros_like(g), where=s > 0)), lo, hi)
        d = q * s
        err = ((d - g) ** 2).sum(axis=1, keepdims=True)
        if best is None:
            best, best_err = d, err
        else:
            better = err < best_err
            best, best_err = np.where(better, d, best), np.where(better, err, best_err)
    return best


def affine(g, bits, shrink=(1.0,)):
    gmin, gmax = g.min(axis=1, keepdims=True), g.max(axis=1, keepdims=True)
    top = 2 ** bits - 1
    best, best_err = None, None
    for f in shrink:
        mid, half = (gmin + gmax) / 2, (gmax - gmin) / 2 * f
        bias = F16(mid - half)
        s = F16(2 * half / top)
        q = np.clip(np.round(np.divide(g - bias, s, out=np.zeros_like(g), where=s > 0)), 0, top)
        d = q * s + bias
        err = ((d - g) ** 2).sum(axis=1, keepdims=True)
        if best is None:
            best, best_err = d, err
        else:
            better = err < best_err
            best, best_err = np.where(better, d, best), np.where(better, err, best_err)
    return best


SHRINK = np.linspace(0.80, 1.0, 9)
# name: (bits per weight, rows x cols float32 -> dequantized)
FORMATS = {
    "sym8 g32":     (8 + 16 / 32, lambda g: sym(g.reshape(-1, 32), 8)),
    "sym4 g32":     (4 + 16 / 32, lambda g: sym(g.reshape(-1, 32), 4)),
    "affine4 g64":  (4 + 32 / 64, lambda g: affine(g.reshape(-1, 64), 4)),
    "affine4+ g64": (4 + 32 / 64, lambda g: affine(g.reshape(-1, 64), 4, SHRINK)),
    "affine4+ g32": (4 + 32 / 32, lambda g: affine(g.reshape(-1, 32), 4, SHRINK)),
    "sym4 g64":     (4 + 16 / 64, lambda g: sym(g.reshape(-1, 64), 4)),
    "sym3 g32":     (3 + 16 / 32, lambda g: sym(g.reshape(-1, 32), 3)),
    "affine3+ g64": (3 + 32 / 64, lambda g: affine(g.reshape(-1, 64), 3, SHRINK)),
    "affine3+ g32": (3 + 32 / 32, lambda g: affine(g.reshape(-1, 32), 3, SHRINK)),
}


def layer_matrices(path):
    """(layer, name, float32 matrix) for every layer matrix of a bf16 model.bin, in file order."""
    with open(path, "rb") as f:
        data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    h = read_header(data)
    if any(t != ("bf16", 0) for t in h.mat_types):
        raise ValueError("%s is compressed: compare formats on the bf16 model.bin" % path)
    D, V = h.ints[0], h.ints[3]
    r = Reader(data, h.data_offset)
    r.take(V * D * 2)  # embedding
    for l in range(len(h.layer_types)):
        for name, shape in h.layout(l):
            if len(shape) == 2:
                yield l, name, r.mat(*shape)
            else:
                r.vec(*shape)


def compare(model, stats, rows_step=4, progress=None):
    """Activation-weighted error of every format in FORMATS, summed over the
    model and per matrix name, both relative to "sym4 g32". stats: read_stats()
    output for this model. progress is called with (layer, name)."""
    err = {k: 0.0 for k in FORMATS}
    by_name = {}
    n = 0
    for (l, name, w), mean_abs in zip(layer_matrices(model), stats):
        if w.shape[1] != len(mean_abs):
            raise ValueError("the calibration statistics are for another model")
        n += 1
        w = np.ascontiguousarray(w[::rows_step])
        x2 = mean_abs.astype(np.float32) ** 2
        for k, (_, fn) in FORMATS.items():
            e = float((((fn(w).reshape(w.shape) - w) ** 2) * x2).sum())
            err[k] += e
            by_name.setdefault(name, dict.fromkeys(FORMATS, 0.0))[k] += e
        if progress:
            progress(l, name)
    if n != len(stats):
        raise ValueError("the calibration statistics are for another model")
    rel = lambda e: {k: e[k] / e["sym4 g32"] for k in FORMATS}
    return rel(err), {name: rel(e) for name, e in by_name.items()}
