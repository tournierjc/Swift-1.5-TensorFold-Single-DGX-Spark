"""The reduced MTP draft vocabulary this rig installs, and the builder that makes it.

Credit: the doctrine under test is ported from MIA AI Lab's reduced-vocabulary MTP drafting ("mia's recipe",
https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark, AGPL-3.0-or-later, Copyright (C) 2026
MiaAI Lab, https://x.com/MiaAI_lab): `files/build_draft_vocab.py` and `files/build_draft_vocab_extend.py`.
`scripts/build_draft_vocab.py` is this rig's adaptation of them for the TensorFold `qwen4_exp` CUDA engine.

Two things are checked here. First the builder's three rules, as pure functions -- a base list is a floor,
the byte-fallback range is pinned whatever its frequency, and only real text adds ids. Then the artifact the
image installs, `patch/draft_vocab.txt`: that it is a legal draft vocabulary for this checkpoint (one id per
line, sorted, unique, inside the vocabulary), that it still pins the byte-fallback range, and that it never
drops an id the engine's own shipped list carries -- the property that makes this port a monotone extension
rather than a bet. Both would fail on a tree without the port: there would be no builder to import and no
`patch/draft_vocab.txt` to read.
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

build_draft_vocab = pytest.importorskip("build_draft_vocab", reason="scripts/build_draft_vocab.py is the port")

# The checkpoint's tokenizer: 248,044 BPE pieces plus 33 added tokens (ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4,
# revision 3ff05202; its config.json text_config.vocab_size is the padded 248,320, which is not what the
# tokenizer can produce). The engine's own list stops at 248,076, one below this.
VOCAB_SIZE = 248_077
SHIPPED = ROOT / "patch" / "draft_vocab.txt"
ENGINE_BASELINE = [
    ROOT / "dev" / "repo" / "src" / "tensorfold" / "families" / "qwen4_exp" / "cuda" / "draft_vocab.txt",
    ROOT / "dev" / "port" / "src" / "tensorfold" / "families" / "qwen4_exp" / "cuda" / "draft_vocab.txt",
]


class FakeTokenizer:
    """Enough of the tokenizers API to count a corpus: text in, ids out."""

    def __init__(self, mapping: dict[str, list[int]], vocab_size: int = VOCAB_SIZE):
        self.mapping = mapping
        self.vocab_size = vocab_size

    def encode(self, text):
        return collections.namedtuple("Encoding", "ids")(self.mapping.get(text, []))

    def get_vocab_size(self):
        return self.vocab_size


def read_ids(path) -> list[int]:
    return [int(line) for line in Path(path).read_text().split()]


# -- the three rules, as pure functions ---------------------------------------------------------------------

def test_read_spec_splits_the_corpus_repeat_weight():
    assert build_draft_vocab.read_spec("corpus/a.txt") == ("corpus/a.txt", 1)
    assert build_draft_vocab.read_spec("corpus/a.txt:8") == ("corpus/a.txt", 8)
    # a path that only looks like it carries a weight is left alone
    assert build_draft_vocab.read_spec("corpus/0123.jsonl") == ("corpus/0123.jsonl", 1)


def test_count_corpus_honours_the_repeat_weight(tmp_path):
    file = tmp_path / "a.txt"
    file.write_text("hello")
    tok = FakeTokenizer({"hello": [7, 8, 9]})
    once, meta = build_draft_vocab.count_corpus(tok, [str(file)])
    assert once[7] == 1 and meta["tokens"] == 3 and meta["files"] == 1
    thrice, meta = build_draft_vocab.count_corpus(tok, [f"{file}:3"])          # the :N knob MIA's builder has
    assert thrice[7] == 3 and meta["tokens"] == 9


def test_count_corpus_reads_jsonl_by_its_text_field(tmp_path):
    file = tmp_path / "out.jsonl"
    file.write_text('{"text": "hello"}\n{"other": "ignored"}\n')
    counts, meta = build_draft_vocab.count_corpus(FakeTokenizer({"hello": [7, 8]}), [str(file)])
    assert counts[7] == 1 and meta["tokens"] == 2                             # the text field, not the raw line


def test_select_pins_the_byte_fallback_range_whatever_its_frequency():
    """Rule 2: the ids BPE falls back to are pinned, even when a ranking would leave them out."""

    counts = collections.Counter({900: 1000, 901: 900})                       # nothing down at the bytes
    ids, stats = build_draft_vocab.select(counts, size=8, vocab_size=4096, specials={4000, 4001},
                                          byte_ids={0, 1, 2, 3})
    assert set(ids) >= {0, 1, 2, 3, 4000, 4001, 900, 901}
    assert set(ids) & {0, 1, 2, 3} == {0, 1, 2, 3}
    assert stats["byte_fallback_in_list"] == 4
    assert ids == sorted(ids) and len(ids) == len(set(ids))


def test_select_keeps_a_base_list_whole_as_the_floor():
    """Rule 1: a base list is never lost to a ranking, and it wins over --size."""

    base = {50, 60, 70}
    counts = collections.Counter({11: 5000, 12: 4000})
    ids, stats = build_draft_vocab.select(counts, size=2, vocab_size=4096, specials=set(), byte_ids=set(),
                                          base=base)
    assert set(base) <= set(ids) and stats["kept"] >= 3                      # floor overrides the size
    ids, stats = build_draft_vocab.select(counts, size=5, vocab_size=4096, specials=set(), byte_ids=set(),
                                          base=base)
    assert set(base) <= set(ids) and set(ids) >= {11, 12}                    # and the budget still fills


def test_select_ranks_by_frequency_and_skips_rare_ids():
    """Rule 3: frequency, with --min-count cutting one-off ids out."""

    counts = collections.Counter({10: 5, 11: 4, 12: 3, 13: 1})
    ids, _ = build_draft_vocab.select(counts, size=2, vocab_size=4096, specials=set(), byte_ids=set())
    assert ids == [10, 11]
    ids, _ = build_draft_vocab.select(counts, size=4, vocab_size=4096, specials=set(), byte_ids=set(),
                                      min_count=3)
    assert ids == [10, 11, 12]


def test_select_keeps_ids_inside_the_vocabulary():
    counts = collections.Counter({4096: 9, 5: 8})                             # 4096 is one past a 4096 vocab
    ids, _ = build_draft_vocab.select(counts, size=4, vocab_size=4096, specials={9999}, byte_ids={0})
    assert max(ids) < 4096 and 9999 not in ids and 0 in ids and 5 in ids


def test_select_can_carry_the_engines_own_prefix_rule():
    ids, stats = build_draft_vocab.select(collections.Counter(), size=16, vocab_size=4096, specials=set(),
                                          byte_ids=set(), keep_below=8)
    assert ids == list(range(8)) and stats["pinned"] == 8


def test_coverage_counts_occurrences_not_distinct_ids():
    ids = [10, 11]
    counts = collections.Counter({10: 3, 11: 1, 12: 96})
    got = build_draft_vocab.coverage(ids, counts)
    assert (got["occurrences"], got["covered"], got["missed"]) == (100, 4, 96)
    assert got["coverage_pct"] == 4.0


def test_write_ids_is_the_format_the_engine_reads(tmp_path):
    """One id per line, sorted, and nothing else: `draft_token_ids` parses it with np.loadtxt, and the MLX
    reader splits it and int()s every token, so a header or a comment would break the engine."""

    out = tmp_path / "draft_vocab.txt"
    build_draft_vocab.write_ids(out, [3, 1, 2])
    text = out.read_text()
    assert text == "1\n2\n3\n"
    assert all(line.strip().isdigit() for line in text.splitlines())
    assert build_draft_vocab.read_id_file(out) == {1, 2, 3}


# -- the artifact the image installs ------------------------------------------------------------------------

def test_the_shipped_file_is_a_legal_draft_vocabulary():
    ids = read_ids(SHIPPED)
    assert ids, f"{SHIPPED} is empty or missing: the port is not installed"
    assert ids == sorted(ids), "the engine reads these in file order as draft-head row order"
    assert len(ids) == len(set(ids))
    assert 0 < len(ids) < VOCAB_SIZE
    assert max(ids) < VOCAB_SIZE and min(ids) >= 0


def test_the_shipped_file_pins_the_byte_fallback_range():
    """Rule 2 at the artifact level: a corpus ranking must not be allowed to drop the byte pieces."""

    ids = set(read_ids(SHIPPED))
    assert len(ids & set(range(256))) == 256, "the byte-fallback ids are missing from the shipped list"


def test_the_shipped_file_never_drops_the_engines_shipped_list():
    """Rule 1 at the artifact level, and the reason this port is safe: a superset can only add coverage.

    Skipped where the engine checkout is not on this host (the rig ships it under `dev/`, untracked).
    """

    baseline = next((p for p in ENGINE_BASELINE if p.exists()), None)
    if baseline is None:
        pytest.skip("no engine source checkout under dev/: cannot compare against the shipped list")
    missing = set(read_ids(baseline)) - set(read_ids(SHIPPED))
    assert not missing, f"{len(missing)} ids the engine ships are missing, e.g. {sorted(missing)[:8]}"


def test_the_image_overlays_the_file_where_the_engine_reads_it():
    """The wiring: `patch/` is copied over .../families/qwen4_exp/cuda/, beside weights.py, where
    `draft_token_ids("default")` looks for draft_vocab.txt. A file in the wrong place is a silent no-op."""

    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "localpatch" in dockerfile and "families/qwen4_exp/cuda/" in dockerfile
    assert SHIPPED.name == "draft_vocab.txt"


def test_the_engine_reads_the_file_into_exactly_these_rows():
    """Where torch and the engine are installed (inside the image), the engine's own reader must agree.

    `draft_ids` is the draft head's row order, and `mtp.py` maps a draft back with `id_map=w.draft_ids`, so
    the file is the id-to-row table: getting this list right is the whole change.
    """

    pytest.importorskip("torch")
    weights = pytest.importorskip("tensorfold.families.qwen4_exp.cuda.weights")

    import numpy as np

    got = weights.draft_token_ids(str(SHIPPED))
    assert got is not None
    assert got.tolist() == np.unique(np.asarray(read_ids(SHIPPED))).tolist()
