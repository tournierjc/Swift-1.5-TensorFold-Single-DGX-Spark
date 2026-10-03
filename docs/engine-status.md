# Engine status and history

The long form behind the README: the engine versions, the changes carried on top of upstream, the full-checkpoint
load, and every benchmark table this rig has taken. The README states the short version and the numbers the rig
runs at today; where a figure here differs, the README's is the newer one.

## Status of the engine under test

Built and checked on the DGX Spark this rig targets:

- The first end-to-end run found a blocker and it is fixed on the branch at `ba6cb76`: the startup memory
  estimate could not size the checkpoint's `F8_E4M3` block scales (73,728 tensors), so `serve` died in 15 s
  with `CUDA startup memory geometry could not be established on every rank: 'F8_E4M3'`. After the fix the
  estimate runs over the real headers and the load proceeds.
- `scripts/build.sh` → image `swift-tensorfold:local`, 36.5 GB; `tensorfold --version` = 0.3.6.2 (from the
  pinned commit), and the build-time import of `families.qwen4_exp.cuda.nvfp4` — a module that exists only on
  this branch — passed, so a wrong ref fails at build time.
- Dependencies inside the image: hf-hub 1.24.0, tokenizers 0.23.1, safetensors 0.8.0, jinja2 3.1.6, numpy 2.1.0,
  torch 2.13.0a0 (NVIDIA's 26.07 container).
- `tensorfold info ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4` inside the image resolved the family from the
  published `config.json`: `qwen4_exp`, 48 layers, 512 experts, `quantization modelopt`, "runs on NVIDIA GPUs
  (CUDA)".
- Hugging Face answers the repository listing anonymously from the Spark (100 files, 31 `embedding-model-*`),
  so `scripts/pull.sh` needs no token today; keep `HF_TOKEN` in `.env` for the day that changes.

0.3.6.3's CUDA suite runs green in the same container this image is built from: **728 passed, 75 skipped,
0 failed** on a GB10 (the `nvfp4-flash-next` branch this rig pinned before it, 0.3.6.2: 460 passed, 70
skipped, 0 failed; with the reduce and `_fp4mm` changes below in, 728/75/0 again). Against the real published
weights, checked in-process:

- 16 routed experts bit-exact against the reference dequantization; FP4 matmul error 2.6e-3 / 2.3e-3; BF16
  faces 2.9e-3; row invariance from 1 to 64 rows; 512 MTP experts exact; the n-gram hashing constants equal
  value for value.
- The n-gram table on real bytes: shard 2 of `embedding-model-00006-of-00131.safetensors` gathers rows
  bit-identical to an independent read of the file, and 128 shards of that size make the 320,001,536-row
  table the digest derives.

**Measured on this rig on 0.3.6.3 (`1911880`), the complete 186 GB checkpoint, one DGX Spark:** the load runs
end to end — `97.39 GiB within 105.04 GiB`, a 262144-token window, 21 decode graphs captured, the 95.4 GiB of
mapped tables paged from disk, n-gram tables read in 291 s, loaded in 545 s. Greedy, TTFT from the first
streamed chunk (which on this checkpoint is `reasoning_content`, not `content`), decode as
`(completion - 1) / (total - TTFT)`, prefill as `prompt / TTFT`, two rounds each, identical across rounds:

| workload | prompt tok | decode | prefill | TTFT |
| --- | --- | --- | --- | --- |
| prose (400-word essay) | 37 | **27.2 tok/s** | — | 0.16 s |
| code (`merge_intervals` + pytest) | 59 | **54.6 tok/s** | — | 0.20 s |
| 2275 tokens of context, one-line question | 2275 | 32.9 tok/s | **1181 tok/s** | 1.93 s |

Against the `b4bf826` ref this rig had been serving, same checkpoint and profile: prose 17.9 → 27.2 tok/s,
code 27.6 → 54.6, prefill 798 → 1181 (the non-stream totals of `scripts/bench.sh` agree: 14.52 → 9.53 s,
9.62 → 4.86 s, 5.90 → 3.41 s). The targets this rig was pointed at are prose 30 / code 45 / prefill 1000:
**code and prefill clear them, prose reaches 91% of it.**

**The draft window this rig carries is the prose lever.** `--mtp-drafts 10` was tuned on the branch, where a
round cost less; on 0.3.6.3 the same window pays for a wider verification of drafts that prose does not
accept. Sweeping it on the warm server, same checkpoint and everything else fixed:

| `--mtp-drafts` | prose | code | prefill |
| --- | --- | --- | --- |
| 10 | 22.9 | 53.8 | 1175 |
| 5 | **26.4** | **54.5** | **1190** |
| 3 | 27.3 | 42.8 | 1117 |

Three drafts buy nothing over five on prose and cost code a fifth of its rate, so the rig now serves five.
Prose accepts roughly 2.5 drafts a round against code's 5.5, at the same round cost - which is why the
window that suits code over-pays on prose, and why the next gain here is the draft head's acceptance rather
than bandwidth.

`--mtp-confidence` is the other half of that trade-off, and it pays off without a cost: on five drafts, 0.30
against the 0.20 this rig carried gives prose 27.2 against 26.4 tok/s while code (54.6 against 54.5) and
prefill (1181 against 1190) do not move - a higher threshold stops a chain where the draft head is unsure,
which is what prose's drafts are, and leaves a confident chain alone. So the rig serves 5 drafts at 0.30.

Not verified yet, and worth reporting from a run here:

- The 95.4 GiB of n-gram tables do not fit beside the weights, so every lookup pages from disk. Measured on
  the served endpoint: decode reads **0.6-0.8 KiB per token from storage**, so the paging is not a throughput
  factor at this window. (`--ple-on-ssd` is refused for an NVFP4 checkpoint - `serve` exits 1 - and a smaller
  the window is not free: the engine bounds the attention launches by it, so the same 2275-token prompt
  measured the same 1509-1533 tok/s three times at **`--context 262144`** as at 8192, while decode did not
  move either. A window is not a rate: the allocator's own default - no `--context` at all - is the affordable
  native capacity, which on this box admitted the model's full 262144 when asked explicitly.)
- **Decode does not feel the context.** A 700-token reply measured **39.0 and 41.3 tok/s at 20,035 tokens of
  context**, against **39.6** for the same reply at 36 tokens (window 65536, 1.53 drafts accepted a round). The
  long conversation is paid once in prefill - 20,035 tokens in **11.9 s**, about 1680 tok/s - and not per token
  after. Quote a long reply: 128-token ones measure near 16 tok/s, because a round's drafting needs a few dozen
  rounds to reach its steady acceptance and a short reply is mostly that ramp. The bench's 39.8 and 75.8 are
  steady-state lines, not best cases.
- **Set `--parallel N` to serve more than one session at a time.** On CUDA the default `auto` means one request
  decoded at a time and the rest waiting their turn, which costs N times the wall time with no gain: two clients
  took 11.0 s and four a 5.5 s staircase, aggregate 38.1 then 22.2 tok/s against 35.0 for one. An explicit
  number batches instead, and since a round is weight-bound the lanes share those reads nearly for free -
  30.7 / 63.7 / 99.4 tok/s aggregate for 1 / 2 / 4 concurrent clients, at 6.8 / 6.6 / 8.4 s each (2.1x and 3.2x
  the throughput for 24% more latency). The lanes share the slot pool, so N lanes of `--context C` need
  N x C slots at about 4.8 KB each: four lanes of 65536 fit the 262150 this rig reports.
- **The three-session server this rig runs**: `--parallel 3 --context 196608`, granted as asked (98.83 GiB within
  105.03, 196614 slots). Measured there: **31.5 / 63.6 / 85.9 tok/s aggregate for 1 / 2 / 3 concurrent clients**
  at 6.6 / 6.6 / 7.3 s each, a fourth queuing at 13.3 s, and a 20,020-token prompt answering in 13.0 s. Asking
  for 262144 instead of 196608 is accepted and then clamped to 8192 - see the note below.
- **Always read the startup line's allocated window, never the flag you passed.** With more than one lane the
  budget is asked for N x C and the answer is not always a refusal: `--parallel 4 --context 262144` was refused
  with "estimated largest fitting prompt-plus-reply window: 198779 tokens", while `--parallel 3 --context
  262144` was *accepted* and then allocated **8192** - the safe fallback - with no error anywhere except that
  line. An 8192 window is what makes a normal conversation fail with HTTP 400 "no room for a reply", so a
  silent clamp looks exactly like a broken model. Check `allocated prompt/reply window` in the log after every
  serve that sets `--context` with `--parallel`.
- **A repeated conversation is re-prefilled every turn.** Five identical 20k requests all reported `cached=0`,
  with and without a client `user` field. The prefix cache is a CLI and server feature (`--prompt-cache-gib`,
  `--checkpoint-slots`, `--spill-gib`) that this CUDA family does not implement: the names appear in `cli.py`,
  `server/app.py` and the vendored MLX drafter, nowhere under `families/qwen4_exp/`. So at a long context every
  turn pays the prefill again, and the lever is the length of the conversation, not those flags.
- One of the 128 n-gram shards is proven against real bytes; the other 127 are read by the same code path and
  each shard's header is checked at load time, a mixed layout raising rather than loading wrongly.

**The dense BF16 faces are the round's other half, and an 8-bit copy of them pays.** This checkpoint quantizes
only the routed experts: a layer is 1200 MiB of NVFP4 experts and 150 MiB of e4m3 scales against **147.6 MiB
of BF16**, 110 MiB of it the DeltaNet and attention projections. A round verifies a handful of rows, so those
dense faces are re-read whole every round however few tokens come out of it - **6.9 GiB a round**, and 53% of
the round's device time in `_b16mm` (a full `block_n`/`bk`/`warps`/`stages` sweep at 7 rows buys 5%, so the
matmul is not mistuned: it is reading bytes). `TENSORFOLD_FACES_FP8=1` loads those projections with an 8-bit
copy a round reads instead of the rows (prompts keep the rows), and the same client then measures:

| `TENSORFOLD_FACES_FP8` | prose | code | prefill |
| --- | --- | --- | --- |
| off | 27.2 | 54.6 | 1181 |
| `1` | 32.2 | 63.6 | 1196 |
| `1`, with the 16-pair multi-row item below | **40.0** | **75.8** | **1486** |
| `4`, same | 38.5 | 73.5 | 1496 |

`4` copies the same projections affine group-of-32 (0.31x the stored bytes against 0.39x for e4m3) and is **not**
kept: fewer bytes, but the affine lane is slower a byte than the e4m3 one, and the round ends up 4% longer on
prose and 3% on code. The same reason `all` failed - on this chip the quantized lanes' efficiency does not scale
with the narrower format, so `1` is the end of that road and the bytes left are the ones that are already cheap.

The server's own round counters move with it: prose 76.4 -> 73.6 ms a round at 2.19 tokens, code 83.8 -> 76.7
ms at 4.57. `all` extends the copy to every BF16 face and was **not** kept - it cuts the round further, but
the router, hyper-connections and shared expert steer which experts run and how the streams mix, so the
drafts' head (calibrated to the BF16 body) accepts 48% where it accepted 64% and the end-to-end number does
not move. The lm_head's rows never get a copy for the same reason: the drafts' head is a quantized copy of
those very rows (0 of 63 drafts accepted once they were coarsened, replies garbled).

**The multi-row item holds 16 pairs, not 64, and that is worth 24%.** ``Plan`` gives each item 16 pairs when the
arithmetic is the decode form and 64 in the prefill form, and the kernel holds only those two. Measured at a
prompt's own row count (2275 x 10 pairs over 512 experts, `dev/moe_tile.py`): a 64-pair item reads 549 items of
the stack in **21.00 ms**, a 16-pair item reads 1788 items in **14.09 ms**. The fatter item buys 3x less traffic
that turns out to be L2 hits anyway (351 GB/s against the GB10's ~273 GB/s of DRAM), and pays for it with 3x
fewer items competing for the SMs. Since the verify window of a decoding round runs the same multi-row
arithmetic a prompt does, the change moves everything at once, served on the published checkpoint:

| item | prose | code | prefill |
| --- | --- | --- | --- |
| 64 pairs (upstream's default) | 32.2 | 63.6 | 1196 |
| **16 pairs** | **40.0** | **75.8** | **1486** |

`TF_REF=3ec1227 scripts/build.sh` puts the branch in the image and the image then serves the 16-pair row with
no mount at all: prose 40.1, code 75.9, prefill 1509 (the two rows above were taken through the mount and agree
within noise). **Verify the image by what it does, not by the build's exit code.** The first attempt here was a
silent no-op - `build.sh` sourced `.env` *after* the caller's variables, so `.env`'s pinned ref won and the
image kept serving 27.2/54.5/1180, byte-identical to the released package. The script now lets the caller win,
and `direct_url.json` in the image names the commit it was built from.

**The 4-bit checkpoint is the remaining reference, measured here.** Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP
(4-bit throughout, 29.8 GiB of n-gram tables), same rig, same client, upstream's defaults (`--no-thinking
--context 8192`, six drafts):

| checkpoint | prose | code | prefill | loaded | start-up |
| --- | --- | --- | --- | --- | --- |
| ukisai NVFP4, `TENSORFOLD_FACES_FP8=1`, 16-pair items | 40.0 | 75.8 | 1486 | 97.39 GiB | 548 s |
| Vontra MLX-4bit | **50.1** | **97.7** | **2024** | 84.26 GiB | 271 s |

Its round is 49.2 ms against 73.6 ms on prose - and it accepts *fewer* drafts (25% against 31%), so the whole
difference is bytes a round reads, which is the point: the NVFP4 checkpoint's dense linears stay BF16. Prefill
separated by more (+69%) only against the mistuned item: with the 16-pair item it is 1486 against 2024 (+36%),
and the gap is the dense linears again - 4-bit throughout against BF16 rows re-read whole (prompts keep the
rows by design, so an 8-bit copy does not help a prompt either). The tables are not it: decode reads 0.6-0.8
KiB a token from them, and a 2275-token prompt pulls about 1.6 MB, a millisecond.

`tensorfold serve Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP --name swift-1.5 --host 0.0.0.0 --port 8083
--no-update-check --no-thinking --context 8192` serves it (the container needs the HF cache mounted and
`HF_HOME` pointing at it). Switch back to the NVFP4 checkpoint when byte-exact agreement with serial decoding
on that checkpoint's own format is what matters.

**45 tok/s on prose is the next 11%**, and what is left of it is host, not bytes. The round's device work is
~7.3 GB of routed experts plus the dense faces a round re-reads (6.9 GiB of stored BF16, 2.76 GB with the
8-bit copies above), and the projected profile shows what is left after them: 37.6 ms of kernels in a 59.6 ms
round, so ~15 ms a round is the host between graph replays - and the CPU profile names it. Per round the
forward gathers 72 n-gram rows through numpy's mmap and issues 22.6 pageable copies, and the thread pool that
serves them is where the round blocks; everything under 0.5 ms a launch amounts to 0.3 ms a round, so it is
not launch overhead and not small kernels. Read-ahead exists in the tree but for *prompt* chunks
(`families/qwen4_exp/runtime.py`, `engine/family_prefill.py`) and as a start-up page warm-up
(`engine.py`); the decode round's gather at `forward.py:244` is synchronous. Dispatching it for the round's
own ids before the verify is the change that closes prose, and it is a hot-loop one. The same model converted
4-bit throughout still leads (table above) - that gap is the checkpoint's layout, not the engine.

## Checkpoint and engine detail (moved out of the README)

- Engine pinned by the `Dockerfile` as `ARG TF_REF`:
  `f5de97f917b51735dec57459daad8bd736642cb9`, `tournierjc/TensorFold@integration/0.6.3` — upstream
  `ashhart/TensorFold@9356df5` (0.6.3) plus the rig's own changes, one commit each: the 8-bit projection
  copies and the 80,014-id draft vocabulary. Upstream 0.6.3 merged the vision port whole, so video left the
  branch; the PLE row prefetch and the 12-bit decode faces are deliberately not carried (the 12-bit branch is
  kept for future work). The README's "Engine revision and the branches" lists every branch and its head.
  History: `integration/0.6.2` (`d26e09f`, upstream 0.6.2 `56e2e3e`) before it,
  `integration/0.6.1` (`808767fd`, upstream 0.6.1 `17c73e1`) before that, and before that
  the rig served `191188075bca56a7c71074a79375eb4c1cb22e1c`, `ashhart/TensorFold@main` (0.3.6.3,
  upstream PR [ashhart/TensorFold#67](https://github.com/ashhart/TensorFold/pull/67) merged as 0.3.6.3) and
  before that the 0.3.6.2 `nvfp4-flash-next` branch, then 0.5.0 with three PRs on top, then 0.6.0
  (`c3fa14f4`, `integration/0.6.0`) — the speed tables below were taken on 0.5.0, not on 0.6.x.
- **The 0.6.2 rebase.** 12 commits and 50 files (+1,487/−367) over 0.6.1's tip. The five branch commits replayed
  with **no conflict at all**, and their own diffstat is identical to the one they had on 0.6.1 (23 files,
  +1,912/−91): this rebase changes no line of the branch's work, which is exactly why the deployed A/B, not the
  rebase log, is what says whether the new base costs anything. What upstream changed is elsewhere: Flash Next at
  64k-128k *on Macs*, the 27B's GDN tree kernel and DFlash2 drafter launch on CUDA, the GLM-5.3 checkpoint credit
  and its mixed-bit EXL3 refusal, and the CUDA server's own fixes (`cuda/http.py`, `cuda/server.py`,
  `engine/prefix_snapshots.py`, `server/cancellation.py`, `server/scheduler.py`) — the branch's files
  (`families/qwen4_exp/cuda/*`, `vision/*`, `server/messages.py`, `server/prompts.py`) are not
  among them. Checked before the rebase, by symbol rather than by subject: **none of the five is upstream in
  0.6.2** — `TENSORFOLD_FACES_FP8`, `TENSORFOLD_FACES_12BIT`, `TENSORFOLD_PLE_PREFETCH`, `vision/videos.py` and
  `families/qwen4_exp/cuda/draft_vocab.txt` are all still absent there. Two release entries do touch this rig's
  configuration: the config check now accepts an **FP8 n-gram table** in NVIDIA's MIXED_PRECISION Flash Next
  export ([#179](https://github.com/ashhart/TensorFold/pull/179)) — this checkpoint's n-gram table is BF16, so
  nothing changes for it, but a MIXED_PRECISION sibling of it was refused outright on 0.6.1 — and
  `--mtp-confidence` now defaults to **0.70** upstream where this rig pins **0.60** explicitly.
  - **Deployed on the Spark, and the whole tree checked against upstream's.** `scripts/build.sh` built the pinned
    image (`tensorfold 0.6.2`, the video and 8/12-bit symbols import, the draft vocabulary is 80,014 ids) and it
    was then *served* and measured, not only built — see the A/B below. Every one of the **336** test files of the
    pinned tree was run on its own (a fresh pytest per file, a 120 s timeout each, four at a time, `CUDA_VISIBLE_DEVICES` empty) against a clean `upstream/main` checkout of 0.6.2
    (**332** files; the four extra names are this branch's own): **the failure sets are identical** — the same 155
    files all-skipped (that container has no GPU and no `mlx`), the same failing files, the same two collection
    errors (`test_alternating_kv.py`, `test_dflash_tree_search.py`) and `test_cancellation.py` hitting the 120 s
    timeout on both trees. Exactly one file present on both sides differs: `tests/cuda/test_flashnext_nvfp4_kernels.py`,
    **14 skipped against 10**, which is the 8/12-bit helper coverage the branch adds.
    Of the branch's four own files three pass, and `tests/test_vision_video.py` reports 1 failed of 20 — its
    "video inputs require PyAV" assertion fails *because the image installs PyAV*, so the loader reaches its
    invalid-bytes path instead. That file and `vision/videos.py` are byte-identical on 0.6.1, where the same test
    fails the same way: it predates this rebase and is this branch's own defect, not the new base's. (The counts
    here are `test_*.py` files; the 0.6.1 sweep's "343" counted every `.py` under `tests/`, which is why the two
    runs do not read alike — `integration/0.6.1` carries 329 test files, 0.6.2's upstream 332, this branch 336.)
  - **No speed cost, measured rather than argued.** Both pins served fresh on the same box, from the same `.env`,
    back to back, one `scripts/bench-suite.sh` run per arm, with the id of the image *actually serving* taken from
    the container: `swift-tensorfold:061` (`integration/0.6.1`, `808767f`, image `fa5ff1616f41`)
    **34.9 / 57.2 / 72.9** tok/s at one, two and three clients against `swift-tensorfold:local`
    (`integration/0.6.2`, `d26e09f`, image `b3677c5c500b`) **35.0 / 57.2 / 73.2**, and the 0.6.3 rebase
    (`integration/0.6.3`, `f5de97f`, image `43be03b19d69`) **34.9 / 57.3 / 73.2** on its second pass
    (**32.9 / 55.4 / 70.6** on its first, this rig's usual run-to-run spread) — `speed.py` within 0.04 s a
    workload, the vision probes within 0.04 s a case, warm load 115.3 s on both. Two further passes of the 0.6.2
    arm read 35.0 / 57.2 / 73.2 and 35.2 / 57.5 / 73.6. The 0.6.2 arm's *first* start after the rebase took
    210.6 s: it recompiled the revision's kernel extensions (`prompt kernels warmed in 98.2s`); every start since
    is 115.3 s, the same as 0.6.1's. One point is published rather than dropped, a first pass whose two-client
    reading was **19.5 tok/s**: it ran while the host's `hermes-agent` container — whose local model provider is
    this same `:8083` endpoint — was taking a turn, and the serve log shows a **71,004-token `finish=tool_calls`
    request prefilling for 55.09 s** between the one- and the two-client runs, then three more turns on a ~72k
    context. 0.6.2 is what makes that visible, since it prints a line a request where 0.6.1 leaves the `/health`
    counters; a bench here has to check that nothing else is on the endpoint, and a point that disagrees gets
    repeated, not published or quietly rerun.
- **The 0.6.1 rebase.** 54 commits and 200 files (+11,886/−598) over 0.6.0's tip. In, out and what it costs:
  - **Upstream 0.6.1 serves images on `qwen4_exp`.** The image port this rig had carried since 0.6.0 is the
    same MiaAI-Lab patch (`EncodedVision`, `vision_config`, `image_positions` byte-identical; upstream's own
    `tests/cuda/test_flashnext_vision.py` asserts `pbuf.rope_rows` and `st.rope_delta`, the names the port
    introduced), so the port came *out*: what the branch carries now is video, on top of upstream's frontend,
    as `feat/vision-video`. Upstream's frontend also moved the per-stream rotary tables into `image_rows.py`,
    which is where the video path now reads them.
  - **Upstream reserves 4 GiB for the tower's workspace** (`VISION_WORKSPACE` in
    `families/qwen4_exp/cuda/engine.py`), independently of the tower's own estimate, and counts it against the
    startup admission. This rig sets `TENSORFOLD_VISION_WORKSPACE_MIB=1280` — its measured peaks are 0.76 GiB
    for a 4M-pixel image and 0.83 GiB for a video — so an unset value would plan ~2.7 GiB more than what was
    measured. The knob is new in 0.6.1 (0..16,384 MiB).
  - **The draft vocabulary is still not upstream**: 79,591 ids there, 80,014 on the branch (423 ids this
    rig's corpus needs), which is what the build assertion holds.
  - **The rest of the branch replayed clean**: the 8-bit copies, the prefetch and the 12-bit faces touch files
    0.6.1 barely moved (`bf16.py`, `weights.py`, `host_table.py`, `decode.py`, `forward.py`, `multi.py`). The
    5 commits rebased with one conflict hunk — the vision path — because the vision port is exactly what
    upstream had merged in the meantime. Verified after the rebase: `compileall` clean, the branch's import
    carries `tensorfold 0.6.1` with `vision FAMILIES = ('qwen3_5', 'qwen4_exp', 'glm5_next')`, the kernel lint
    over `src/` finds nothing, and 20 new CPU tests cover the video prompt path.
  - **Deployed on the Spark, and the whole tree checked against upstream's.** `scripts/build.sh` builds the
    pinned image (`tensorfold 0.6.1`, the video and 8/12-bit symbols import, the draft vocabulary is 80,014
    ids), and the served endpoint answered a 512x512 image, a 2048x2048 image (4M pixels, 4172 prompt tokens,
    7.32 s) and a 2 s video whose reasoning reads the frame timestamps back — the video path's first
    measurement here. Every one of the 343 test files of the pinned tree was run on its own (120 s each,
    eight at a time) against a clean `upstream/main` checkout: **the failure sets are identical**, and the only
    differences are this branch's four own test files (all passing) plus four more skips in
    `tests/cuda/test_flashnext_nvfp4_kernels.py`, which is the 8/12-bit helper coverage the branch adds. The
    box has no GPU in that container and no `mlx`, so 142 files report all-skipped and two fail to collect
    (`test_alternating_kv.py`, `test_dflash_tree_search.py`) — on both trees alike.
  - **No speed cost, measured rather than argued.** Both pins were served on the same box from the same `.env`,
    one after the other: `integration/0.6.0` (35.0 / 57.4 / 72.9 tok/s at one, two, three clients) against
    `integration/0.6.1` (34.8 / 57.1 / 73.0), 206.9 s against 214.3 s to load. The 0.5.0-era 111.6 tok/s at
    three lanes was a different bench and does not compare; the two arms are what settle it.
- **The 0.6.0 rebase.** Two of those three PRs are upstream now, with the same patch, so the rig no longer
  carries them: the multi-row item-16 pair path (#102, upstream as `9933492`) and the fp32 reduce / `_fp4mm`
  block change (#105, upstream as `443ad4a`). The third, the 8-bit projection copies (#104), upstream declined
  (*no precision traded for speed*) and it stays, gated by `TENSORFOLD_FACES_FP8`. The draft vocabulary is no
  longer overlaid from this repository: it is a commit on the pinned branch, byte-identical
  (`8facf56e11ad522ca8ba1d396755b6ce7cc98f2bf226498780fcc7806231c192`), and the build asserts the installed
  package's own file instead of copying one in. Another branch, `pr/host-table-rows-by-file` (#103, closed
  unmerged), kept the round's n-gram rows read by file over the pool; it was never in the pinned branch, so it
  was never in the served image. That branch is deleted now, its head on the tag
  `backup/removed/pr_host-table-rows-by-file` (`79aa2a0`), together with the redundant 12-bit spelling
  `cursor/lossless-12bit-faces-1ba6` (`410d2ff`, tag `backup/removed/cursor_lossless-12bit-faces-1ba6`).
- **The rebase's one defect, and how it surfaced.** 0.6.0 split the attention kernels into a wrapper and a
  per-head (per-block) worker — `_attn_prep`/`_prep_row` in `glue.py`, `_pool`/`_pool_block` in
  `attention.py` — and the vision port's rotary parameters (`ROPE`, `DELTA`, `MODE`, `S1`, `S2`) merged into
  the *workers' bodies* while their signatures kept upstream's shape, and into the wrappers' signatures while
  their bodies only forwarded. The file is valid Python: `compileall` and the whole CPU suite pass, because
  nothing on the CPU compiles a kernel — Triton fails on the GPU, at the first request, with a `CompilationError`
  pointing at the wrapper's call. The same merge left `attn_multi.py`, upstream's new lane kernel, calling those
  workers without the parameters. Fixed by moving the parameters down with the body (`_prep_row` gains a MODE 3:
  the row's own stream's offset, for a launch that serves several streams) and by giving `attn_multi.Step` a
  per-row `rdelta` and per-stream `deltas` table. A rebase that touches kernels needs a static check of kernel
  names and call arities — the suite cannot see this class of defect.
- Base image `nvcr.io/nvidia/pytorch:26.07-py3` (36.5 GB as pulled here) — CUDA, torch 2.13, triton, the
  extension compiler.
- Revision `3ff05202` of the checkpoint: 186.4 GB over 296,474 tensors.
- **Quality:** ModelOpt NVFP4 dequant is `W = E2M1 * fp32(e4m3) * weight_scale_2` (no extra `2**-7`). This
  rig's own pinned fix for that factor was `36a5bc4` on the fork; upstream fixed the same formula on top of
  the merge and 0.3.6.3 carries it, along with the grouped NVFP4 MoE that reads the routing plan on the GPU.
  Prior empty/`im_end` loops came from the erroneous `2**-7` scale; with either fix the replies read as text.
- On 0.3.6.3 the reply streams as `reasoning_content` and leaves `content` null until the thinking budget is
  spent, so a client that reads only `delta.content` sees an empty stream, counts `deltas=0` and has no TTFT.
  `--no-thinking` puts the text back in `content`; a benchmark that wants the first token's time should read
  either field.
- The checkpoint's own `ple_embedding.ngram_embedding.shard_N.weight` tensors are BF16 `[2500012, 160]` rows
  with no per-shard scales — 128 shards, 320,001,536 rows, 29.8 GiB, memory-mapped and gathered a lookup at a
  time.

## Serving levers and earlier profiles (moved out of the README)

The header-only preflight (`scripts/preflight.py`, revision `3ff05202`): **97.39 GiB within a 104.28 GiB
budget**, native window 262,144 tokens, weights 78.54 GiB, loading 18.85 GiB, cache workspace 18.14 GiB, and
the n-gram table at **95.37 GiB** — 102.4 GB over 128 shards in the BF16 layout this revision ships, three
times the 29.8 GiB the MLX 4-bit layout takes. It does not fit beside the weights and caches, so its pages are
read from disk during lookups.

Levers, as `EXTRA_ARGS` in `.env`:

- `--ssd-experts 90` — stream routed experts into a 90 GiB GPU pool for models past the memory budget.
- `--mtp-drafts 5 --mtp-confidence 0.20 --no-thinking --context 8192` — the profile this rig served while the
  numbers above were taken: MTP for decode, answers in chat `content`, and `CONTEXT=N` for the prompt-plus-reply
  window. Leave the window unset for a server people talk to: the CUDA default is the affordable native
  capacity, and a long conversation needs room. The 8k this rig pinned was for headroom beside hermes-agent,
  which measurement says is not needed — 65536 loaded at 97.39 GiB within 105.12 with hermes-agent running —
  and an 8k window is what broke an 18,758-token session with HTTP 400 (no room for a reply). The benches here
  pin `CONTEXT=8192` because their numbers were taken there and a window is not free; drafts `0` disables
  drafting. Five rather than ten because prose accepts about 2.5 drafts a round and code about 5.5 at the same
  round cost (table above); ten costs prose 3.5 tok/s and buys code nothing.
- `--ple-on-ssd` — refused for an NVFP4 checkpoint on 0.3.6.3 (`serve` exits 1 before the weights load; the
  tables stay memory-mapped here, see Troubleshooting). It applies to the MLX checkpoint's n-gram shards, where
  it is worth about 40 GiB at peak for a few percent of decode speed.
- `--parallel 2` — two requests decoded together, windows sharing each round's forward (Flash Next, one rank).
- `--context N` — prompt plus reply window; the CUDA default is the affordable native capacity.
- `--no-drafts` — the serial reference: same output, slower.

Port **8083** here (matches the hermes-agent / historical Spark OpenAI endpoint). The vLLM sibling also used
8083 — only one of the two fits in memory at a time.

## The speed sweep

Every lever of the serving engine on this checkpoint, one configuration per load, three lanes of 262144 with int8
KV, two passes each. The first pass after every load is cold and is discarded — it runs 1172-1503 tok/s prefill
against 1644-1674 warm, on *every* configuration including the untouched reference, so an undeclared first pass
masquerades as the effect of whatever was just added. Aggregate is three concurrent clients.

| configuration | prefill | decode 1024 | decode 20k | agg 1 / 2 / 3 | verdict |
|---|---|---|---|---|---|
| reference, `--mtp-confidence 0.30` | 1660 | 34.4 | 48.1 | 31.9 / 58.6 / **80.8** | baseline |
| `--mtp-confidence 0.45` | 1668 | 35.0 | 50.8 | 31.4 / 62.1 / 84.9 | +5.1% |
| `--mtp-confidence 0.60` | 1664 | 34.4 | 49.4 | 33.1 / 64.4 / **88.9** | **accepted, +10.1%** |
| `--mtp-confidence 0.75` | 1674 | 32.9 | 40.6 | 33.0 / 64.1 / 88.9 | rejected, decode breaks |
| `--checkpoint-slots 32 --spill-gib 8` | 1644 | 33.7 | 48.0 | 32.0 / 64.1 / 88.4 | no effect |
| `--checkpoint-slots 200` | — | — | — | — | no effect on the cache |
| `--lane-kernels on` | 1667 | 34.5 | 49.4 | 32.9 / 64.1 / 88.9 | no effect |
| `--kv-dtype int4` | 1664 | 33.8 | 41.0 | 32.5 / 63.1 / 87.5 | rejected, −17% decode |
| `TENSORFOLD_FACES_FP8=all` | 1666 | 43.9 | 51.5 | 42.1 / 82.0 / **111.6** | **accepted, +25.5%** |

The accepted pair compounds to **80.8 → 112.1 tok/s** aggregate at three lanes, reproduced on a second load, with
prefill and TTFT unmoved. `TENSORFOLD_FACES_FP8` was the expensive miss: it was set to `1`, which restricts the
8-bit faces to the vision tower, when `all` extends them to every layer.

Rejected earlier or refused by the engine: the 4-bit `lm_head` (implementation verified row-local and correct, but
**−9.5%** prefill, −6% aggregate — a byte-for-latency trade the owner declined), `--cpuset-cpus=5-9,15-19` (0.0%,
measured before and after on the same container), per-lane graph capture (+2.6% — the 3.15x of multi-lane comes
from `MultiDecoder`, not from CUDA graphs, and the multi-stream path is eager by construction), `--ple-on-ssd`
(refused: an NVFP4 checkpoint's tables stay memory-mapped, so there is nothing to stream from disk).

### The prefix cache, measured properly

A prompt that *strictly extends* a cached prefix returns `cached_tokens=19,862` and answers in **0.2 s** where the
cold call took **12.2 s**. Re-asking the *identical* prompt and two requests sharing a long *system* block both
return `cached_tokens=0` — 12.3 s and 17.8 s. A probe built on repeated identical prompts therefore concludes the
cache is unimplemented; that wrong verdict stood in this file for a session before the extension pattern was tried.

**It is per lane, and the lane count — not the slot count — is the limit.** With `--parallel 3` and a barrier
synchronising the sends, three conversations running *simultaneously* all hit on turns 2 and 3
(`cached_tokens=3476`, 0.42 s each), while their three cold first turns took 6.68 s apiece — exactly three times a
single cold prefill, so the three requests did occupy three lanes. Four and six conversations *rotating* over those
three lanes hit NEVER, at 32 checkpoint slots and again at 200. The "one conversation at a time" reading that stood
here was that artefact: up to `--parallel` concurrent conversations each keep their prefix, and beyond it they evict
one another. Size the lane count to the concurrency you expect. `cached-tokens` also never appears for a shared system block, which is
what "pinned system blocks bypass slot limits" in the code means in practice. The plumbing is real on CUDA:
`checkpoint_slots` reaches `CheckpointStore` in `server/app.py:133`, and the recipe already points `--snapshot-dir`
at the persistent `/state` bind, not the in-container default.

## The EXL3 4.05bpw arm (2026-10-03)

`turboderp/Qwen3.8-Flash-Next-exl3`, branch `4.05bpw_h6_ng6`, served as a directory (`MODELS_DIR/exl3-405`)
on the `integration/0.6.3` head that names the vision sidecar (`d31685e`). It replaces the 186 GB NVFP4 arm as
the arm this rig serves: 107.5 GB on disk (63.47 GiB resident weights, 36.36 GiB of n-gram rows in one mapped
`I16` tensor), and the memory that frees is what buys the fourth lane.

    startup estimate 75.17 GiB within 104.86 GiB; native 262144, allocated prompt/reply window 262144, cache slots 262151
    vision: image and video input, a 0.84 GiB tower with 1.25 GiB of workspace reserved
    Flash Next on CUDA: 1 to 6 MTP drafts a round, a chain stops before a later draft under 60%; up to 4 streams, each growing to 262144 prompt/reply tokens while memory lasts (34.9 GiB free for their caches, 4.47 GiB for one at the full window), eager; int8 KV cache (fp16 scale per 32 values); n-gram tables read alongside the weights (0.0s after them); 0 decode graphs captured; idle prompt pieces 2048 rows; prompt kernels warmed in 74.9s
    serving qwen3.8-flash-next at http://0.0.0.0:8083/v1 on CUDA (sampling: temperature 1.0, top_k 20, top_p 0.95; drafts: on; context: 262144; loaded in 139.0s)

**The staged draft chain measures even.** Branch `perf/mtp-device-chain` (`88206d9`) keeps a draft chain's tokens on the device: the draw runs there (`cuda.sampling.choose_rows_device`, the host hash bit for bit), each drawn token is written where the next step's MTP head embeds it, and `mtp_stage` takes that row - so the pinned rebuild, its H2D copy and the `b.staged` wait they are guarded by leave every drafted token. The counters say that port is exact, not close:

| `TENSORFOLD_MTP_STAGED` | 1 client | 2 | 3 | 4 | rounds | drafted | accepted |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `0` (control) | 43.7 | 68.9 | 85.1 | 106.2 | 4900 | 9351 | 5880 |
| `1` | 43.8 | 68.8 | 84.8 | 106.2 | 4900 | 9351 | 5880 |

Same tokens, drafts, acceptances and rounds at every lane count, and no speed for it: 0.3% either way inside noise, `decode_seconds_total` +0.15% at three lanes and -0.13% at four. The host round-trips were real and are gone, but they were worth about what the steps the cutoff now drafts and discards cost. The EXL3 recipe's own win came from a *blocking* 1.65 ms x ndt memcpy of per-step tables, which this engine's staging never had: the port transfers the removal, not the number. Not integrated, and the branch is dropped: its head is held on a local tag `backup/removed/mtp-device-chain` (`88206d9`) alone, the fork no longer carries it, and the rig serves `d31685e` as before.

The 139 s load is a first load with a fresh `STATE_DIR`: five CUDA extensions were compiled
(`tensorfold_exl3_linear_v3`, `qwen4_exp_gdn_io`, `gdn_v2`, `exl3_experts_v1`, `qmm_v5`). Later starts reuse
them. The arm's own directory and state dir are what keep it from being confused with the cached arm:
`tensorfold pull` takes a repo id and the cache resolver falls back to the newest config-bearing snapshot,
so the variants of this repo must be addressed as directories (`bench/arms.json`).

**The gate is the window, and it is granted.** `scripts/preflight-arm.sh exl3-405 --streams 4 --context
262144 --vision`, on the idle budget, reports 4 streams fitting at 77.92 GiB of 105.17 with `window 262144
(native 262144, asked 262144, explicit True)` - no clamp. The NVFP4 arm's third lane is its last: asking it for
four comes back with `estimated largest fitting prompt-plus-reply window: 246909 tokens`. The lanes share one
pool (34.9 GiB here, 4.47 GiB per lane at the full window), so the pool is worth about 7.8 full-window lanes
and four of 262144 coexist with room to spare: `--parallel 4` is a concurrency choice, not a context divided
by four.

**Speed, same instrument (`scripts/bench-suite.sh`, aggregate at 1024-token replies), same host, both arms on
0.6.3:**

| clients | NVFP4, 3 lanes | EXL3 4.05bpw, 4 lanes |
| --- | --- | --- |
| 1 | 32.9 / 34.9 tok/s | **43.4** |
| 2 | 55.4 / 57.3 | **64.6** |
| 3 | 70.6 / 73.2 | **83.5** |
| 4 | refused (window) | **109.9** |

The NVFP4 column is two runs of that arm (`bench/after-063.log`, `bench/after-063b.log`); the README's
111.6 tok/s at three lanes is the 0.5.0-era figure and is not this arm on this revision. The EXL3 arm is
faster at every count, on a first load with cold kernel caches, and it is the only arm measured here that
opens a fourth lane. The quantized MTP head works: 13055 drafted, 8306 accepted (64%).

**The e4m3 face copies do not pay on this pack.** The NVFP4 arm's own lever, built for the EXL3 loader on the fork (`exl3_mm.F16` takes the same `Mx8Linear.from_bf16` copy, with `prefill`/`__call__` split so a prompt keeps the stored rows), covers the two faces a round re-reads whole: the hyper-connections' `input_mix_weight_up` (625 MiB over 96 sites, 6.5 MiB each) and the PLE key/value projections (62 MiB). The *down* projections stay on the stored rows - their decode path sums fp32 K slices in order, and the copy's own matmul returns bf16 - and the DeltaNet `in_proj_a`/`in_proj_b` are 0.25 MiB each, nothing for a lane matmul to win. One image (`063-5f62ae7`), one `.env`, flag off against flag on:

| `TENSORFOLD_FACES_FP8` | 1 client | 2 | 3 | rounds | drafted | accepted |
| --- | --- | --- | --- | --- | --- | --- |
| `0` (control) | 43.5 | 68.9 | 85.1 | 4900 | 9351 | 5880 (62.9%) |
| `all` | 42.9 | 67.4 | 83.4 | 4871 | 9956 | 5922 (59.5%) |

The last three columns are over each arm's whole suite run. The copies cost 1.4 to 2.2% at every lane count, outside the 0.3% this instrument repeats at, and buy nothing: the eligible faces are ~687 MiB of a ~6.5 GiB dense round, and at those shapes (10240x320, 10240x2560) the e4m3 lane's dequantisation costs more than the bytes it saves. Accepted *per round* is flat (1.200 against 1.216); the accepted *ratio* falls only because the chain drafted further under the changed stream mixing (1.91 to 2.04 drafts a round), which is what a coarser copy of the stream-mixing face does to the draft head's confidence. Not kept: the rig serves the flag off, and the engine change was dropped from `integration/0.6.3` (local recovery tag `backup/removed/e4m3-exl3-faces`, `5f62ae7`).

**Quality against the NVFP4 arm** (`scripts/quality-suite.sh exl3-405 --compare
bench/quality/nvfp4-ukisai-20261003-005959.json`; 43 frozen items scored through `/v1/decisions`, nothing
generated):

- **0/28 multiple-choice decisions flipped**, mean Jensen-Shannon **0.0003** (max 0.0031)
- mean **|dlogp| 0.0347 nats** (max 0.376), mean |d label_mass| 0.0238
- accuracy 43/43 on both arms

The divergences sit where the model's belief is smallest - the yes/no items about a common misconception
(`true-lightning`: |dlogp| 0.376, label_mass 0.404 -> 0.589) - i.e. the label mass moves while the decisions
do not. That is the shape a 4.05-bit quant is expected to have: turboderp's own KL table puts it at 0.0067
against a 0.00249 noise floor, between NVFP4 W4A16 (0.0100) and W4A4 (0.0241).

**Vision** (`bench/vision_probe.py`, which generates what it asks about): red image -> `Rouge` (1.56 s), red
2 s clip -> `Rouge` (3.38 s), **blue image -> `Bleu`** (2.55 s). The tower is the converted sidecar
(`tensorfold.vision.exl3_convert`, 856 MiB, source hash and codec recorded in the artifact); re-running
`scripts/convert-vision.sh` returns the same artifact and the same md5.

**What this pass does not settle.** `bench/speed.py` reports `None` for TTFT and decode on both arms because
`--thinking` spends the reply budget on `reasoning_tokens` (the README's own caveat), and its prefill workload
is a prefix-cache hit in both rounds (`cached_tokens` 2314 of 2315), so its 1.22 s is not a prompt-speed
measurement. The prompt-speed question this arm raises - its n-gram rows were read alongside the weights and
the engine no longer prints the "do not fit beside the weights" line that the NVFP4 arm prints - needs a
`--no-thinking` profile or a client that reads `reasoning_content`, plus a cold prompt no lane has seen.
