"""The `kyse` command: export, compress and inspect Qwen3.5 models."""
import argparse, shutil, sys, time
from collections import Counter
from pathlib import Path

from . import __version__
from .modelfile import read_header_file
from .native import KyseError, cached_calib, cached_counts, calibrate, count_tokens, find_bench

PRESETS = {
    "int8": dict(layers="q8", embed="q8", awq=False, int8=[], help="the safe choice: no measurable loss"),
    "int4-mixed": dict(layers="q4", embed="q8", awq=True, int8=["*.wo", "*.w_out", "*.w_down"],
                       help="int8's bits/byte, 30%% smaller than int8 (calibrates)"),
    "int4": dict(layers="q4", embed="q8", awq=True, int8=[], help="the smallest, with a visible quality loss (calibrates)"),
    "int4-plain": dict(layers="q4", embed="q8", awq=False, int8=[], help="int4 without calibration: quick, a bit worse"),
    "bf16": dict(layers="bf16", embed="bf16", awq=False, int8=[], help="only the vocabulary pruning"),
}
FORMAT_NAMES = {"bf16": "bf16", "q8": "int8", "q4": "int4"}
FORMATS_BY_NAME = {v: k for k, v in FORMAT_NAMES.items()}
CALIB_CHUNKS, CALIB_BYTES = 32, 1024
DEFAULT_CORPUS = Path("data/english.txt")

EXAMPLES = """\
examples:
  kyse export ../models/qwen3.5-0.8b qwen-0.8b     # Hugging Face checkpoint -> model.bin + tokenizer.bin
  kyse data                                        # English corpus and benchmark sets into data/
  kyse compress qwen-0.8b en-int8                  # int8, English vocabulary
  kyse compress qwen-0.8b en-int4 -p int4-mixed    # int4 with AWQ and int8 where int4 hurts
  kyse inspect en-int4                             # what is in a model file
"""


# ---------------------------------------------------------------------------
# output helpers

def say(*args):
    print(*args, file=sys.stderr, flush=True)


def mb(n):
    return "{:,.1f} MB".format(n / 1e6) if n < 1e10 else "{:,.2f} GB".format(n / 1e9)


def duration(s):
    return "%dm%02ds" % divmod(int(s), 60) if s >= 60 else "%.1fs" % s


def progress_line(text):
    if sys.stderr.isatty():
        print("\r\033[K" + text, end="", file=sys.stderr, flush=True)


def end_progress():
    if sys.stderr.isatty():
        print("\r\033[K", end="", file=sys.stderr, flush=True)


def model_paths(path):
    """(model.bin, tokenizer.bin) from a model directory or a model.bin path."""
    p = Path(path)
    if p.is_dir():
        if (p / "config.json").exists() and not (p / "model.bin").exists():
            raise KyseError("%s is a Hugging Face checkpoint: convert it first with `kyse export %s <out_dir>`" % (p, p))
        p = p / "model.bin"
    if not p.exists():
        raise KyseError("no model at %s" % path)
    tok = p.parent / "tokenizer.bin"
    if not tok.exists():
        raise KyseError("no tokenizer.bin next to %s" % p)
    return p, tok


def patterns(text):
    return [p.strip() for p in (text or "").split(",") if p.strip()]


# ---------------------------------------------------------------------------
# export, data

def cmd_export(a):
    from .export import export
    out = Path(a.out_dir)
    if (out / "model.bin").exists() and not a.force:
        raise KyseError("%s already holds a model.bin (use --force to replace it)" % out)
    t = time.time()
    say("exporting %s -> %s/" % (a.checkpoint, out))
    size = export(a.checkpoint, str(out))
    say("wrote %s/model.bin (%s) and tokenizer.bin in %s" % (out, mb(size), duration(time.time() - t)))
    say("next: kyse compress %s <out_dir>" % out)


def cmd_data(a):
    from .data import download
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        raise KyseError("downloading the data needs pyarrow: `uv sync --extra data`, "
                        "or `uv tool install --editable '.[data]'`") from None
    download(a.out_dir)


# ---------------------------------------------------------------------------
# compress

def compress_plan(a):
    """Resolves presets, defaults and cached steps; returns a dict that describes the run."""
    preset = PRESETS[a.preset]
    plan = dict(
        layers=FORMATS_BY_NAME[a.layers] if a.layers else preset["layers"],
        embed=FORMATS_BY_NAME[a.embed] if a.embed else preset["embed"],
        awq=preset["awq"] if a.awq is None else a.awq,
        int8=preset["int8"] if a.int8 is None else patterns(a.int8),
        lowrank=patterns(a.lowrank),
    )
    if plan["layers"] != "q4":
        plan["awq"] = False  # AWQ only helps int4: int8 already rounds finely
    model, tok = model_paths(a.source)
    plan.update(model=model, tokenizer=tok, out=Path(a.output))

    # vocabulary: counts given, cached, or to be computed from the corpus
    plan["counts"] = plan["corpus"] = None
    if not a.keep_vocab:
        if a.counts:
            plan["counts"] = Path(a.counts)
        else:
            corpus = [Path(c) for c in (a.corpus or ([DEFAULT_CORPUS] if DEFAULT_CORPUS.exists() else []))]
            if not corpus:
                raise KyseError("the vocabulary needs English text to count tokens on: pass --corpus FILE "
                                "(`kyse data` downloads one), or --keep-vocab to keep every token")
            for c in corpus:
                if not c.exists():
                    raise KyseError("no corpus file %s" % c)
            plan["corpus"] = corpus
            plan["counts"], plan["counts_cached"] = cached_counts(model, tok, corpus)

    # calibration for AWQ (and the weighting of low-rank corrections)
    plan["calib"] = None
    if plan["awq"] or (plan["lowrank"] and a.calib):
        if a.calib:
            plan["calib"], plan["calib_cached"] = Path(a.calib), True
        else:
            text = Path(a.calib_text) if a.calib_text else (plan["corpus"] or [DEFAULT_CORPUS])[0]
            if not text.exists():
                raise KyseError("calibration needs English text: pass --calib-text FILE (or --no-awq)")
            plan["calib_text"] = text
            plan["calib"], plan["calib_cached"] = cached_calib(model, text, CALIB_CHUNKS, CALIB_BYTES)
    return plan


def describe(plan, a):
    fmt = lambda k: FORMAT_NAMES[k]
    say("kyse compress: %s -> %s/  (preset %s)" % (plan["model"], plan["out"], a.preset))
    if plan["counts"] is None:
        say("  vocabulary   all tokens kept")
    elif plan["corpus"]:
        say("  vocabulary   tokens seen at least %d times in %s%s" % (
            a.min_count, ", ".join(map(str, plan["corpus"])), "" if plan["counts_cached"] else " (counting first)"))
    else:
        say("  vocabulary   tokens seen at least %d times, counts from %s" % (a.min_count, plan["counts"]))
    layers = fmt(plan["layers"]) + (" with AWQ" if plan["awq"] else "")
    if plan["int8"]:
        layers += "; int8: " + ", ".join(plan["int8"])
    if plan["lowrank"]:
        layers += "; rank-%d correction: %s" % (a.rank, ", ".join(plan["lowrank"]))
    say("  layers       " + layers)
    say("  embedding    " + fmt(plan["embed"]))
    if plan["calib"]:
        say("  calibration  " + ("cached, %s" % plan["calib"] if plan["calib_cached"] else
                                 "runs the model on %s first (~15 min for 0.8B on an M1)" % plan["calib_text"]))


def cmd_compress(a):
    from .shrink import choose_vocab, read_stats, read_tokenizer, shrink_model, write_tokenizer, unmatched
    plan = compress_plan(a)
    out = plan["out"]
    if out.resolve() == plan["model"].parent.resolve():
        raise KyseError("the output directory is the source's: pick another one")
    if (out / "model.bin").exists() and not a.force:
        raise KyseError("%s already holds a model.bin (use --force to replace it)" % out)
    if a.rank <= 0 or a.rank % 4:
        raise KyseError("--rank must be a positive multiple of 4")
    header = read_header_file(plan["model"])
    if any(t != ("bf16", 0) for t in header.mat_types):
        raise KyseError("%s is already compressed: start from the bf16 model that `kyse export` writes" % plan["model"])
    for p in unmatched(plan["int8"] + plan["lowrank"], header):
        raise KyseError("pattern '%s' matches no layer matrix; names look like 3.w_down or *.wo "
                        "(see `kyse inspect %s`)" % (p, a.source))
    describe(plan, a)
    if a.dry_run:
        return

    t0 = time.time()
    needs_bench = (plan["counts"] and plan["corpus"] and not plan["counts_cached"]) or \
                  (plan["calib"] and not plan["calib_cached"])
    bench = find_bench(a.bench) if needs_bench else None
    if plan["corpus"] and not plan["counts_cached"]:
        say("counting tokens in the corpus...")
        count_tokens(bench, plan["tokenizer"], plan["corpus"], plan["counts"])
    if plan["calib"] and not plan["calib_cached"]:
        say("calibrating: running the model on %s..." % plan["calib_text"])
        calibrate(bench, plan["model"], plan["tokenizer"], plan["calib_text"], plan["calib"], CALIB_CHUNKS, CALIB_BYTES)

    tokens, ranks, special = read_tokenizer(plan["tokenizer"])
    keep = None
    if plan["counts"]:
        counts = {}
        for line in open(plan["counts"]):
            i, c = line.split()
            counts[int(i)] = int(c)
        if counts and max(counts) >= len(tokens):
            raise KyseError("%s counts tokens of another tokenizer" % plan["counts"])
        keep = choose_vocab(tokens, ranks, special, counts, a.min_count)

    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "model.bin.partial"
    stats = read_stats(plan["calib"]) if plan["calib"] else None
    try:
        size = shrink_model(plan["model"], tmp, keep, plan["layers"], plan["embed"], plan["int8"], stats,
                            plan["lowrank"], a.rank,
                            progress=lambda i, n: progress_line("quantizing: layer %d/%d" % (i, n)))
    except BaseException:
        end_progress()
        tmp.unlink(missing_ok=True)
        raise
    end_progress()
    if keep is None:
        shutil.copyfile(plan["tokenizer"], out / "tokenizer.bin")
    else:
        new_id = {old: new for new, old in enumerate(keep)}
        write_tokenizer(out / "tokenizer.bin", [tokens[i] for i in keep], [ranks[i] for i in keep],
                        [new_id[i] for i in special])
    tmp.replace(out / "model.bin")

    src = plan["model"].stat().st_size
    say("wrote %s/ in %s: %s, %.0f%% smaller than the source (%s)%s" % (
        out, duration(time.time() - t0), mb(size), 100 * (1 - size / src), mb(src),
        "" if keep is None else "; vocabulary %s -> %s tokens" % ("{:,}".format(len(tokens)), "{:,}".format(len(keep)))))
    say("try it:   ./qwen %s/model.bin" % out)
    say("measure:  ./bench bpb %s/model.bin data/wiki.test.txt -k 8" % out)


# ---------------------------------------------------------------------------
# inspect, formats

def cmd_inspect(a):
    for i, path in enumerate(a.models):
        model, tok = model_paths(path)
        h = read_header_file(model)
        c = h.config
        mats = h.matrices()
        n_attn = sum(h.layer_types)
        size = model.stat().st_size
        if i:
            print()
        print("%s: %s, format v%d" % (model, mb(size), h.version))
        print("  Qwen3.5, %d layers (%d DeltaNet, %d attention), dim %d, MLP %d, vocabulary %s, %s head" % (
            c["n_layers"], c["n_layers"] - n_attn, n_attn, c["dim"], c["hidden_dim"],
            "{:,}".format(c["vocab_size"]), "own" if h.untied else "tied"))
        weights = sum(r * k for _, (r, k) in mats)
        print("  %.2fB matrix weights, %.2f bits per weight on average" % (weights / 1e9, 8 * size / weights))
        groups = {}
        for (name, (r, k)), (t, rank) in zip(mats, h.mat_types):
            short = name.split(".", 1)[-1]
            groups.setdefault(short, Counter())[FORMAT_NAMES[t] + (" + rank %d" % rank if rank else "")] += 1
        width = max(map(len, groups))
        for short, kinds in groups.items():
            print("  %-*s  %s" % (width, short, ", ".join(
                k if n == 1 else "%s ×%d" % (k, n) for k, n in kinds.most_common())))


def cmd_formats(a):
    from .formats import FORMATS, compare
    from .shrink import read_stats
    model, tok = model_paths(a.source)
    if a.calib:
        calib = Path(a.calib)
    else:
        text = Path(a.calib_text) if a.calib_text else DEFAULT_CORPUS
        if not text.exists():
            raise KyseError("the comparison weights errors by activation size: pass --calib STATS or --calib-text FILE")
        calib, cached = cached_calib(model, text, CALIB_CHUNKS, CALIB_BYTES)
        if not cached:
            say("calibrating: running the model on %s..." % text)
            calibrate(find_bench(a.bench), model, tok, text, calib, CALIB_CHUNKS, CALIB_BYTES)
    total, by_name = compare(model, read_stats(calib), a.rows_step,
                             progress=lambda l, n: progress_line("layer %d %s" % (l, n)))
    end_progress()
    print("error relative to int4 (sym4 g32), weighted by activation size; lower is better\n")
    print("%-14s %5s %7s" % ("format", "bits", "error"))
    for k, (bits, _) in FORMATS.items():
        print("%-14s %5.2f %7.3f" % (k, bits, total[k]))
    if a.per_matrix:
        keys = [k for k in FORMATS if k != "sym4 g32"]
        print("\n%-8s " % "" + " ".join("%12s" % k for k in keys))
        for name, e in by_name.items():
            print("%-8s " % name + " ".join("%12.3f" % e[k] for k in keys))


# ---------------------------------------------------------------------------

def parser():
    ap = argparse.ArgumentParser(prog="kyse", description="Compress Qwen3.5 models for English.",
                                 epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version="kyse " + __version__)
    sub = ap.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    p = sub.add_parser("export", help="convert a Hugging Face checkpoint to model.bin + tokenizer.bin",
                       description="Convert a Hugging Face Qwen3.5 checkpoint (safetensors) into the bf16 "
                                   "model.bin and tokenizer.bin that the C code and `kyse compress` read.")
    p.add_argument("checkpoint", help="Hugging Face checkpoint directory (config.json + *.safetensors)")
    p.add_argument("out_dir", help="directory to write model.bin and tokenizer.bin to")
    p.add_argument("-f", "--force", action="store_true", help="replace an existing model.bin")
    p.set_defaults(run=cmd_export)

    p = sub.add_parser("data", help="download the English corpus and the benchmark sets",
                       description="Download the English corpus (vocabulary and calibration) and the "
                                   "benchmark sets (~52 MB). Needs the `data` extra (pyarrow).")
    p.add_argument("out_dir", nargs="?", default="data", help="where to write them (default: data)")
    p.set_defaults(run=cmd_data)

    presets = "\n".join("  %-11s %s" % (k, v["help"].replace("%%", "%")) for k, v in PRESETS.items())
    p = sub.add_parser(
        "compress", help="prune the vocabulary and quantize a model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Prune the vocabulary to the tokens an English corpus uses, and quantize the weights.\n\n"
                    "presets (-p):\n" + presets + "\n\n"
                    "Token counts and calibration run the C `bench` tool; their results are cached in\n"
                    "<source>/.kyse, so later runs on the same model skip them.",
        epilog="examples:\n"
               "  kyse compress qwen-0.8b en-int8\n"
               "  kyse compress qwen-0.8b en-int4 -p int4-mixed\n"
               "  kyse compress qwen-0.8b en-int4 -p int4 --int8 '*.wo,*.w_out'   # a preset, adjusted\n"
               "  kyse compress qwen-0.8b all-int8 --keep-vocab                   # every token\n"
               "  kyse compress qwen-0.8b x -p int4-mixed -n                       # show the plan only")
    p.add_argument("source", help="bf16 model directory (model.bin + tokenizer.bin) or model.bin")
    p.add_argument("output", help="directory to write the compressed model.bin and tokenizer.bin to")
    p.add_argument("-p", "--preset", choices=PRESETS, default="int8", help="starting point (default: int8)")
    p.add_argument("-n", "--dry-run", action="store_true", help="print the plan and stop")
    p.add_argument("-f", "--force", action="store_true", help="replace an existing model in the output directory")
    g = p.add_argument_group("vocabulary")
    g.add_argument("--corpus", nargs="+", metavar="FILE", help="English text that decides the kept tokens "
                   "(default: %s)" % DEFAULT_CORPUS)
    g.add_argument("--min-count", type=int, default=2, metavar="N", help="keep tokens seen at least N times (default: 2)")
    g.add_argument("--counts", metavar="FILE", help="token counts from `bench count`, instead of --corpus")
    g.add_argument("--keep-vocab", action="store_true", help="keep every token")
    g = p.add_argument_group("weights (override the preset)")
    g.add_argument("--layers", choices=FORMATS_BY_NAME, help="format of the layer matrices")
    g.add_argument("--embed", choices=FORMATS_BY_NAME, help="format of the embedding and output head")
    g.add_argument("--int8", metavar="PATTERNS", help="layer matrices kept at int8, comma-separated "
                   "<layer>.<name> patterns: '*.w_down', '0.*', '*.wo,*.w_out' ('' for none)")
    g.add_argument("--awq", action=argparse.BooleanOptionalAction, default=None,
                   help="activation-aware scaling of int4 matrices (needs calibration)")
    g.add_argument("--lowrank", metavar="PATTERNS", help="layer matrices that get a bf16 low-rank correction")
    g.add_argument("--rank", type=int, default=16, help="rank of those corrections (multiple of 4, default: 16)")
    g = p.add_argument_group("calibration")
    g.add_argument("--calib-text", metavar="FILE", help="text to calibrate on (default: the first corpus file)")
    g.add_argument("--calib", metavar="STATS", help="statistics from `bench calib`, instead of calibrating")
    g.add_argument("--bench", metavar="PATH", help="the C bench tool (default: ./bench or the checkout's)")
    p.set_defaults(run=cmd_compress)

    p = sub.add_parser("inspect", help="show what a model file holds",
                       description="Show a model's architecture, size and the format of every matrix.")
    p.add_argument("models", nargs="+", metavar="model", help="model directory or model.bin")
    p.set_defaults(run=cmd_inspect)

    p = sub.add_parser("formats", help="compare candidate weight formats on a model (research)",
                       description="Quantize every layer matrix in each candidate format and compare the "
                                   "activation-weighted error, before writing any kernel for a format.")
    p.add_argument("source", help="bf16 model directory or model.bin")
    p.add_argument("--calib", metavar="STATS", help="statistics from `bench calib`")
    p.add_argument("--calib-text", metavar="FILE", help="text to calibrate on (default: %s)" % DEFAULT_CORPUS)
    p.add_argument("--rows-step", type=int, default=4, metavar="N", help="use every N-th row (default: 4)")
    p.add_argument("--per-matrix", action="store_true", help="also break the error down by matrix name")
    p.add_argument("--bench", metavar="PATH", help="the C bench tool")
    p.set_defaults(run=cmd_formats)
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    try:
        a.run(a)
    except KyseError as e:
        say("kyse: error: %s" % e)
        return 1
    except (ValueError, OSError) as e:
        end_progress()
        say("kyse: error: %s" % e)
        return 1
    except KeyboardInterrupt:
        end_progress()
        say("kyse: interrupted")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
