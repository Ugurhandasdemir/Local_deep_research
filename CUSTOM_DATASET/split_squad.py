"""Split squad_dataset.json into train/dev/test by unique paragraph text (Y-01).

Paragraph text is the grouping unit: every copy of the same text, with all of its
questions, goes to exactly one split. In this dataset 547 texts appear in more than one
article, so splitting by article or by paragraph entry would leak text across splits.

Usage: python split_squad.py [--input FILE] [--out-dir DIR] [--seed 42]

Metni tek birim sayar: aynı paragraf metni ve soruları tek bölmeye girer. Veri setinde 547
metin birden çok makalede geçtiği için makale ya da girdi bazlı bölme sızıntı bırakır.
"""
import argparse
import json
import random
from pathlib import Path


def split_contexts(contexts, train_frac, dev_frac, seed):
    # Sort first so the result does not depend on file order, then shuffle with a fixed seed.
    # Önce sırala, sonra sabit tohumla karıştır; sonuç dosya sırasından etkilenmesin.
    ordered = sorted(contexts)
    random.Random(seed).shuffle(ordered)
    n_train = round(len(ordered) * train_frac)
    n_dev = round(len(ordered) * dev_frac)
    return {
        "train": set(ordered[:n_train]),
        "dev": set(ordered[n_train:n_train + n_dev]),
        "test": set(ordered[n_train + n_dev:]),
    }


def build_split(articles, keep):
    # Keep the SQuAD article structure and drop articles left without paragraphs.
    # SQuAD makale yapısını koru, paragrafı kalmayan makaleleri at.
    out = []
    for article in articles:
        paragraphs = [p for p in article["paragraphs"] if p["context"] in keep]
        if paragraphs:
            out.append({"title": article["title"], "paragraphs": paragraphs})
    return out


def count(articles):
    paragraphs = [p for a in articles for p in a["paragraphs"]]
    return {
        "articles": len(articles),
        "paragraph_entries": len(paragraphs),
        "unique_contexts": len({p["context"] for p in paragraphs}),
        "questions": sum(len(p["qas"]) for p in paragraphs),
    }


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", type=Path, default=here / "squad_dataset.json")
    parser.add_argument("--out-dir", type=Path, default=here / "splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--dev", type=float, default=0.1)
    args = parser.parse_args()
    if args.train <= 0 or args.dev < 0 or args.train + args.dev >= 1:
        parser.error("train and dev fractions must leave a non-empty test part")

    source = json.loads(args.input.read_text(encoding="utf-8"))
    articles = source["data"]
    contexts = {p["context"] for a in articles for p in a["paragraphs"]}
    parts = split_contexts(contexts, args.train, args.dev, args.seed)

    result = {name: build_split(articles, keep) for name, keep in parts.items()}

    # Safety checks: no text or question id may appear in two splits, nothing may be lost.
    # Güvenlik kontrolü: hiçbir metin ya da soru kimliği iki bölmede olmamalı, veri kaybolmamalı.
    texts = {n: {p["context"] for a in r for p in a["paragraphs"]} for n, r in result.items()}
    ids = {n: {q["id"] for a in r for p in a["paragraphs"] for q in p["qas"]} for n, r in result.items()}
    names = list(result)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            assert not texts[first] & texts[second], f"text overlap {first}/{second}"
            assert not ids[first] & ids[second], f"question id overlap {first}/{second}"
    total = count(articles)
    assert sum(count(r)["paragraph_entries"] for r in result.values()) == total["paragraph_entries"]
    assert sum(count(r)["questions"] for r in result.values()) == total["questions"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in result.items():
        payload = {"version": source.get("version"), "data": data}
        (args.out_dir / f"{name}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "source": args.input.name,
        "seed": args.seed,
        "unit": "unique paragraph text",
        "fractions": {"train": args.train, "dev": args.dev, "test": round(1 - args.train - args.dev, 6)},
        "total": total,
        "splits": {name: count(data) for name, data in result.items()},
    }
    (args.out_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
