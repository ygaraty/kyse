"""
The steps that run the model, done by the C `bench` tool: token counts for the
vocabulary, and activation statistics for AWQ. Their results are cached in a
.kyse directory next to the source model, keyed by everything they depend on,
so a second compression of the same model reuses them.
"""
import hashlib, os, subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # the checkout, when installed from it (editable)


class KyseError(Exception):
    """An error with a message for the user; the CLI prints it without a traceback."""


def find_bench(explicit=None):
    """The `bench` binary: --bench, $KYSE_BENCH, ./bench, or the checkout's."""
    for c in (explicit, os.environ.get("KYSE_BENCH"), "bench", REPO / "bench"):
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return str(Path(c).resolve())
    if explicit:
        raise KyseError("no executable bench at %s" % explicit)
    hint = " (run `make` in %s)" % REPO if (REPO / "Makefile").exists() else ""
    raise KyseError("this step needs the C `bench` tool%s; or point to it with --bench or $KYSE_BENCH" % hint)


def _fingerprint(*parts):
    """A short key from strings and files (path, size and modification time; not the contents)."""
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, Path):
            st = p.stat()
            p = "%s:%d:%d" % (p.resolve(), st.st_size, st.st_mtime_ns)
        h.update(str(p).encode() + b"\0")
    return h.hexdigest()[:12]


def _cache(model):
    return Path(model).parent / ".kyse"


def cached_counts(model, tokenizer, corpus):
    """Path of the token counts for this tokenizer and corpus, and whether they exist already."""
    key = _fingerprint("counts", Path(tokenizer), *map(Path, corpus))
    path = _cache(model) / ("counts-%s.txt" % key)
    return path, path.exists()


def cached_calib(model, text, chunks, chunk_bytes):
    key = _fingerprint("calib", Path(model), Path(text), chunks, chunk_bytes)
    path = _cache(model) / ("calib-%s.bin" % key)
    return path, path.exists()


def count_tokens(bench, tokenizer, corpus, out):
    """`bench count`: writes "id count" lines to out."""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(out) + ".partial")
    with open(tmp, "w") as f:
        try:
            subprocess.run([bench, "count", str(tokenizer)] + [str(c) for c in corpus], stdout=f, check=True)
        except subprocess.CalledProcessError as e:
            raise KyseError("`bench count` failed (exit %d)" % e.returncode) from None
    tmp.replace(out)


def calibrate(bench, model, tokenizer, text, out, chunks=32, chunk_bytes=1024):
    """`bench calib`: the mean |x| of every input channel of every layer matrix."""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(out) + ".partial")
    try:
        subprocess.run([bench, "calib", str(model), str(text), str(tmp), "-k", str(chunks), "-l", str(chunk_bytes),
                        "-z", str(tokenizer)], stdout=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        tmp.unlink(missing_ok=True)
        raise KyseError("`bench calib` failed (exit %d)" % e.returncode) from None
    tmp.replace(out)

