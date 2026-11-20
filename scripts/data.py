"""
Download the English data used by shrink.py and bench, into <out_dir>:

  english.txt     ~45 MB corpus that picks the kept vocabulary
                  (WikiText-2 train + Dolly-15k + TinyStories valid)
  wiki.test.txt   WikiText-2 test, for `bench bpb` (not part of the corpus)
  arc_easy.txt    ARC-Easy test, for `bench mc`
  hellaswag.txt   HellaSwag validation (shuffled, seed 0), for `bench mc`

Multiple-choice prompts follow lm-evaluation-harness. Needs pyarrow (parquet).

usage: python3 scripts/data.py <out_dir>
"""
import io, json, os, random, re, sys, urllib.request
import pyarrow.parquet as pq

HF = "https://huggingface.co/"
WIKITEXT = HF + "datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/%s-00000-of-00001.parquet"
DOLLY = HF + "datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"
TINYSTORIES = HF + "datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
ARC_EASY = HF + "api/datasets/allenai/ai2_arc/parquet/ARC-Easy/test/0.parquet"
HELLASWAG = HF + "api/datasets/Rowan/hellaswag/parquet/default/validation/0.parquet"


def fetch(url):
    print("downloading", url)
    with urllib.request.urlopen(url) as r:
        return r.read()


def parquet(url):
    return pq.read_table(io.BytesIO(fetch(url))).to_pylist()


def wikitext(split):
    return "".join(r["text"] for r in parquet(WIKITEXT % split))


# multiple-choice item: "label n", then the context and the n endings, each as "len\n<bytes>\n"
def write_mc(path, items):
    with open(path, "wb") as f:
        for ctx, endings, label in items:
            f.write(b"%d %d\n" % (label, len(endings)))
            for s in [ctx] + endings:
                b = s.encode("utf-8")
                f.write(b"%d\n" % len(b) + b + b"\n")
    print("wrote %s (%d items)" % (path, len(items)))


def arc_easy():
    items = []
    for r in parquet(ARC_EASY):
        labels = list(r["choices"]["label"])
        items.append(("Question: " + r["question"] + "\nAnswer:",
                      [" " + t for t in r["choices"]["text"]], labels.index(r["answerKey"])))
    return items


def hellaswag():
    def clean(t):
        t = t.strip().replace(" [title]", ". ")
        return re.sub("\\[.*?\\]", "", t).replace("  ", " ")

    rows = parquet(HELLASWAG)
    random.Random(0).shuffle(rows)  # the file is ordered by source; sample across it
    items = []
    for r in rows:
        ctx = r["activity_label"] + ": " + r["ctx_a"] + " " + r["ctx_b"].capitalize()
        items.append((clean(ctx), [" " + clean(e) for e in r["endings"]], int(r["label"])))
    return items


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)

    with open(os.path.join(out, "english.txt"), "w", encoding="utf-8") as f:
        f.write(wikitext("train"))
        for line in fetch(DOLLY).decode().splitlines():
            r = json.loads(line)
            f.write("\n".join(x for x in (r["instruction"], r["context"], r["response"]) if x) + "\n\n")
        f.write(fetch(TINYSTORIES).decode().replace("<|endoftext|>\n", ""))
    print("wrote english.txt")

    with open(os.path.join(out, "wiki.test.txt"), "w", encoding="utf-8") as f:
        f.write(wikitext("test"))
    print("wrote wiki.test.txt")

    write_mc(os.path.join(out, "arc_easy.txt"), arc_easy())
    write_mc(os.path.join(out, "hellaswag.txt"), hellaswag())


if __name__ == "__main__":
    main()
