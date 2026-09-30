#!/usr/bin/env python3
"""Build the MTP draft vocabulary this rig serves.

Credit: the technique is ported from MIA AI Lab's reduced-vocabulary MTP drafting ("mia's recipe",
https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark, AGPL-3.0-or-later,
Copyright (C) 2026 MiaAI Lab, https://x.com/MiaAI_lab) -- `files/build_draft_vocab.py`,
`files/build_draft_vocab_extend.py` and the vLLM wiring in `files/patch_mtp_draft_vocab.py`. This file is an
adaptation for this rig and the TensorFold `qwen4_exp` (Flash Next) CUDA engine, not a copy.

Why a reduced vocabulary at all: the engine scores MTP drafts over a *subset* of the vocabulary
(`families/qwen4_exp/cuda/weights.py::draft_token_ids`, `docs/recipes/qwen3.8-flash-next.md`). A token
outside the subset can never be drafted, and every draft is verified by the target, so a missing token
costs speed and never correctness. Choosing the subset well is therefore a speed decision alone.

The three rules this port keeps from MIA's extend script, which learned them from a failed rebuild:

  1. A base list is a FLOOR: `--base` enters whole, and is never lost to a ranking.
  2. The byte-fallback range is pinned unconditionally. Those ids are what BPE falls back to for anything
     the merges do not cover -- every multi-byte UTF-8 sequence, so accents, CJK and emoji. They are ~256
     rows, and frequency alone does NOT keep them: over a smaller or narrower corpus they are dropped
     silently and the drafter then proposes badly at exactly those boundaries.
  3. Only real text adds ids, by FREQUENCY. Never a dictionary: there every inflected form weighs as much
     as "the".

What this rig adds on top: the corpus is read as files or globs, or as `.jsonl` by its `text` field, each
source optionally repeated `:N` so a small in-distribution corpus (the model's own output) can be given the
same say as a large generic one; `--keep-below` carries the engine's own blunter prefix rule; `--sweep` and
`--holdout` report coverage, so the size is chosen on a measurement instead of a guess; and the file written
is exactly what `draft_token_ids()` reads -- one integer id per line, sorted, no header and no comments.

Usage:

  python3 scripts/build_draft_vocab.py TOKENIZER_JSON OUT.txt --size 98304 \\
      'engine/**/*.py' 'docs/**/*.md' 'outputs.jsonl:8' --holdout 'held-out/**/*.md'

  # rule 1: extend a list instead of rebuilding it, keeping what already works whole:
  python3 scripts/build_draft_vocab.py TOKENIZER_JSON OUT.txt --base existing.txt --size 98304 'more/**/*.py'
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

CHUNK = 1 << 20          # read/tokenize ~1 MiB at a time; the corpora are tens of MiB
MAX_BYTES = 2_000_000    # skip files larger than this (the engine's own builder does too)
BYTE_LIMIT = 512         # how far into the vocabulary the byte-fallback range is looked for


def load_tokenizer(path: str):
    """The tokenizers library's tokenizer for a ``tokenizer.json`` (no network, no transformers)."""

    from tokenizers import Tokenizer

    return Tokenizer.from_file(path)


def read_spec(source: str) -> tuple[str, int]:
    """Split a corpus argument into its path and its ``:N`` repeat weight (default 1)."""

    if ":" in source:
        head, _, tail = source.rpartition(":")
        if head and tail.isdigit():
            return head, int(tail)
    return source, 1


def iter_texts(tok, path: str):
    """Yield bounded chunks of one corpus file: ``.jsonl`` by its ``text`` field, anything else as text.

    The ``tok`` argument is unused here and kept so ``count_corpus`` can treat both kinds of file the same way.
    """

    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    text = json.loads(line).get("text", "")
                    if text:
                        yield text
        return
    with open(path, encoding="utf-8", errors="replace") as handle:
        while True:
            block = handle.read(CHUNK)
            if not block:
                break
            yield block


def count_corpus(tok, sources: list[str], *, max_bytes: int = MAX_BYTES) -> tuple[collections.Counter, dict]:
    """Count token ids over every source, honouring ``:N`` repeats and skipping unreadable/oversized files."""

    counts: collections.Counter = collections.Counter()
    files = used = skipped = tokens = 0
    for source in sources:
        path, repeat = read_spec(source)
        for name in sorted(glob.glob(path, recursive=True)):
            try:
                if not os.path.isfile(name) or os.path.getsize(name) > max_bytes:
                    skipped += 1
                    continue
                files += 1
                for _ in range(repeat):
                    for text in iter_texts(tok, name):
                        ids = tok.encode(text).ids
                        counts.update(ids)
                        tokens += len(ids)
                used += 1
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, IsADirectoryError):
                skipped += 1
    return counts, {"files": used, "skipped": skipped, "tokens": tokens, "distinct": len(counts)}


def byte_level_ids(tok, limit: int = BYTE_LIMIT) -> set[int]:
    """The byte-fallback ids at the bottom of the vocabulary: ``<0xNN>`` pieces, and one-character pieces.

    Rule 2: these are pinned whatever their frequency, and they are cheap -- a few hundred rows.
    """

    found: set[int] = set()
    size = tok.get_vocab_size()
    for tid in range(min(limit, size)):
        piece = tok.id_to_token(tid)
        if not isinstance(piece, str) or not piece:
            continue
        if (piece.startswith("<0x") and piece.endswith(">")) or len(piece) == 1:
            found.add(tid)
    return found


def added_ids(tokenizer_json) -> set[int]:
    """The tokenizer's added (special) ids, read from the JSON's ``added_tokens`` list."""

    try:
        with open(tokenizer_json, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return set()
    return {int(entry["id"]) for entry in data.get("added_tokens", []) if "id" in entry}


def read_id_file(path) -> set[int]:
    """A plain list of integer ids, one per line (blank lines and ``#`` comments are skipped)."""

    ids: set[int] = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                ids.add(int(line))
    return ids


def select(counts: collections.Counter, *, size: int, vocab_size: int, specials: set[int],
           byte_ids: set[int], base: set[int] | None = None, keep_below: int = 0,
           min_count: int = 1) -> tuple[list[int], dict]:
    """The ids to keep, at most ``size`` of them (rule 1 can exceed it: the floor always wins).

    The floor comes first, and nothing can remove it: every added/special id, every byte-fallback id
    (rule 2), every id below ``keep_below`` (the engine's own prefix rule), and the whole ``--base`` list
    (rule 1). Corpus ids then fill the rest of the budget by descending frequency (rule 3), skipping ids
    seen fewer than ``min_count`` times and ids outside the vocabulary.
    """

    pinned = {t for t in specials | byte_ids if 0 <= t < vocab_size}
    pinned |= set(range(min(keep_below, vocab_size)))
    floor = {t for t in (base or set()) if 0 <= t < vocab_size} | pinned
    budget = max(int(size), len(floor))
    keep = set(floor)
    for tid, count in counts.most_common():
        if len(keep) >= budget:
            break
        if count >= min_count and 0 <= tid < vocab_size:
            keep.add(tid)
    ids = sorted(keep)
    stats = {"pinned": len(pinned), "byte_fallback_in_list": len(byte_ids & set(ids)),
             "base": len(base or ()), "floor": len(floor), "kept": len(ids),
             "from_corpus": len(ids) - len(floor), "at_budget": len(ids) >= budget}
    return ids, stats


def coverage(ids, counts: collections.Counter) -> dict:
    """What fraction of the corpus's token occurrences the list can draft, and how many it misses."""

    kept = set(ids)
    total = sum(counts.values())
    hit = sum(count for tid, count in counts.items() if tid in kept)
    return {"occurrences": total, "covered": hit, "missed": total - hit,
            "coverage_pct": round(100.0 * hit / total, 4) if total else 0.0}


def sweep(counts: collections.Counter, *, vocab_size: int, specials: set[int], byte_ids: set[int],
          sizes: tuple[int, ...]) -> list[dict]:
    """Coverage at each candidate size, so the size is chosen on coverage and not guessed."""

    out = []
    for size in sizes:
        ids, _ = select(counts, size=size, vocab_size=vocab_size, specials=specials, byte_ids=byte_ids)
        out.append({"size": size, **coverage(ids, counts)})
    return out


def write_ids(path, ids: list[int]) -> None:
    """Write the file the engine reads: one id per line, sorted, nothing else -- no header, no comments.

    Sorted and de-duplicated here, not by the caller: the row order of the reduced head is the file order,
    and the MLX reader parses the file without sorting it.
    """

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("".join(f"{tid}\n" for tid in sorted(set(ids))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tokenizer", help="the checkpoint's tokenizer.json")
    ap.add_argument("out", help="where to write the id list (the engine reads one id per line)")
    ap.add_argument("corpus", nargs="*", help="corpus files or globs, each optionally suffixed :N to repeat it N times")
    ap.add_argument("--size", type=int, default=98304,
                    help="most ids in the output, the pinned floor included (default: 98304)")
    ap.add_argument("--base", default=None,
                    help="an existing list kept whole as the floor (rule 1: never lose what works)")
    ap.add_argument("--byte-fallback-max", type=int, default=BYTE_LIMIT,
                    help="ids searched for byte-fallback pieces (default: 512)")
    ap.add_argument("--keep-below", type=int, default=0,
                    help="also keep every id below this (the engine's blunter prefix rule; default: 0, off)")
    ap.add_argument("--min-count", type=int, default=1, help="fewer corpus occurrences than this are left out")
    ap.add_argument("--holdout", action="append", default=[], metavar="GLOB",
                    help="a glob measured only: reported against the built list, never ranked on (repeatable)")
    ap.add_argument("--sweep", default="32768,47172,65536,79591,98304", help="sizes to report coverage at")
    ap.add_argument("--max-bytes", type=int, default=MAX_BYTES)
    ap.add_argument("--report-only", action="store_true", help="print the report and do not write OUT")
    # intermixed: the corpus globs may sit before or after the options, which is how the README's rebuild
    # command and a hand-typed one are both written (plain parse_args rejects globs that follow an option)
    args = ap.parse_intermixed_args()

    tok = load_tokenizer(args.tokenizer)
    vocab_size = tok.get_vocab_size()
    specials = added_ids(args.tokenizer)
    byte_ids = byte_level_ids(tok, args.byte_fallback_max)
    base = read_id_file(args.base) if args.base else set()

    counts, corpus = count_corpus(tok, args.corpus, max_bytes=args.max_bytes)
    if not corpus["tokens"]:
        print("ERROR: the corpus produced no tokens", file=sys.stderr)
        return 1
    ids, stats = select(counts, size=args.size, vocab_size=vocab_size, specials=specials, byte_ids=byte_ids,
                        base=base, keep_below=args.keep_below, min_count=args.min_count)
    report = {"tokenizer": args.tokenizer, "vocab_size": vocab_size, "size": args.size,
              "base": args.base, "corpus": corpus, "specials": len(specials),
              "byte_fallback_ids": len(byte_ids), **stats, **coverage(ids, counts),
              "sweep": sweep(counts, vocab_size=vocab_size, specials=specials, byte_ids=byte_ids,
                             sizes=tuple(int(s) for s in args.sweep.split(",") if s.strip()))}
    for spec in args.holdout:
        held, meta = count_corpus(tok, [spec], max_bytes=args.max_bytes)
        if not held:
            continue
        against = {**coverage(ids, held), "tokens": meta["tokens"]}
        if base:
            against["base_coverage_pct"] = coverage(base, held)["coverage_pct"]
        report.setdefault("holdout", {})[spec] = against
    print(json.dumps(report, indent=2, default=str))
    if args.report_only:
        return 0
    write_ids(args.out, ids)
    print(f"wrote {len(ids)} ids -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
