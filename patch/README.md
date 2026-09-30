# patch/

Files here are copied over the installed `tensorfold` package on **every** build -- the Dockerfile's
`COPY patch /tmp/localpatch` block, with no opt-in flag to leave unset. Where a file comes from a branch of the
engine fork rather than from this rig, the README's "Patch sources" section names the branch it is sourced
from. This is a **list of files**, never a tree: an overlay of a whole working tree replaces every file that
tree lacks at whatever revision it happens to carry, which is how an older `vision/config.py` once shipped and
refused the checkpoint at launch. Keep it to files that differ, each derived from the revision `TF_REF`
installs.

## `draft_vocab.txt`

The MTP draft head's reduced vocabulary for this rig, sourced from `feat/mtp-draft-vocab` in
`tournierjc/TensorFold` (the README's "Patch sources" has the branch's head). The Dockerfile's overlay copies
this directory over the installed package on every build, so this file lands on
`families/qwen4_exp/cuda/draft_vocab.txt` -- the name `draft_token_ids("default")` reads. 80,014 sorted,
unique ids: the engine's own shipped 79,591-id list
whole (rule 1: a base list is a floor), plus 423 ids ranked by frequency over the corpus under test
(rule 3), with the byte-fallback range pinned (rule 2). It is a strict superset of the 79,591-id list the
built image installs (`swift-tensorfold:local`, digest
`88d5b483a849ae9245b78b69f41f11cdfc8b5c024f0786c1c8196263857cc93e`), so it cannot lower coverage on any
text. Its own digest is `8facf56e11ad522ca8ba1d396755b6ce7cc98f2bf226498780fcc7806231c192`.

The construction is ported from MIA AI Lab's reduced-vocabulary MTP drafting, "mia's recipe"
(https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark, AGPL-3.0-or-later, Copyright (C) 2026
MiaAI Lab, https://x.com/MiaAI_lab): `files/build_draft_vocab.py`, `files/build_draft_vocab_extend.py` and
its vLLM wiring `files/patch_mtp_draft_vocab.py`. `scripts/build_draft_vocab.py` here is this rig's
adaptation for the TensorFold `qwen4_exp` CUDA engine. Rebuild with the command in the README's "MTP draft
vocabulary" section; `tests/test_draft_vocab.py` checks the file and the builder.
