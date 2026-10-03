"""The startup receipt for one arm, header-only, exactly as families/*/cuda/engine.py builds it.

scripts/preflight.py sizes the *single-stream* plan without the EXL3 wrapper and without vision: it
cannot answer "does this arm fit at N lanes with the vision tower and this KV dtype", which is the
question that decides whether an arm is servable. This tool mirrors the engine's own construction for
the serving profile it is given, so a refusal - or a silently clamped window - is found before a
multi-hundred-GB load rather than after it.

It reads tensor headers and file sizes only: no weight byte is loaded, and it runs in seconds as long as
the checkpoint's files are on disk (header stubs of the right size answer the same question, see the
rig's probe notes). On a unified-memory host the budget comes from /proc/meminfo's MemAvailable, so
nothing else may be serving: stop the server first, or pass --budget-gib for a projection.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

GIB = 1024 ** 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("model_dir")
    parser.add_argument("--context", type=int, default=262144)
    parser.add_argument("--streams", type=int, default=3, help="--parallel; 1 sizes the serial plan")
    parser.add_argument("--mtp-drafts", type=int, default=6, help="--mtp-drafts; 0 sizes a --no-drafts serve")
    parser.add_argument("--kv-dtype", default="int8", choices=("bf16", "int8", "int4"))
    parser.add_argument("--vision", action="store_true", help="include the tower and its workspace reserve")
    parser.add_argument("--budget-gib", type=float, default=None,
                        help="the idle budget to plan against instead of the live MemAvailable")
    args = parser.parse_args(argv)

    root = Path(args.model_dir)
    import torch
    from tensorfold.cuda import capacity
    from tensorfold.cuda.capacity import admit, available_bytes, estimate_weights, page_room, total_bytes
    from tensorfold.cuda.geometry import (PREFILL_ROWS, gdn_geometry, indexed_prefill_rows,
                                          indexed_stream_geometry, indexed_weights)
    from tensorfold.families import quant_method, read_config
    from tensorfold.families.qwen4_exp.cuda.kvcache import BITS_OF

    exl3 = quant_method(read_config(root)) == "exl3"
    if exl3:
        from tensorfold.families.qwen4_exp.cuda.exl3_pack import admission, extra_files
        from tensorfold.families.qwen4_exp.cuda.engine import KEEP, KEEP_SERIAL
    else:
        admission, extra_files, KEEP, KEEP_SERIAL = (lambda g: g), (lambda _: ()), None, None
    from tensorfold.families.qwen4_exp.cuda.engine import vision_workspace
    from tensorfold.vision.qwen_cuda import capacity_geometry
    from tensorfold.vision.qwen_cuda import weight_transform as vision_weights

    if args.budget_gib:
        budget = int(args.budget_gib * GIB)
        capacity.available_bytes = lambda _: budget
    else:
        budget = None

    chunks = None if exl3 else indexed_prefill_rows()
    rows = chunks or PREFILL_ROWS
    depth = args.mtp_drafts
    each, mtp, bits = depth + 1, depth > 0, BITS_OF[args.kv_dtype]
    extras = extra_files(root)
    transform = vision_weights(indexed_weights(1, mtp, mapped_tables=True), args.vision, 0)

    print(f"arm            : {root}")
    print(f"format         : {quant_method(read_config(root))}{' (EXL3 wrapper)' if exl3 else ''}")
    print(f"profile        : {args.streams} stream(s), context {args.context}, KV {args.kv_dtype}, "
          f"{'no MTP drafts' if not mtp else f'{depth} MTP drafts'}, vision {'on' if args.vision else 'off'}")
    print(f"vision override: {os.environ.get('TENSORFOLD_VISION_WEIGHTS') or 'unset'}")
    print(f"extra files    : {[f.name for f in extras] or 'none'}")
    live = page_room(torch)
    print(f"budget         : {(budget if budget is not None else available_bytes(torch)) / GIB:.2f} GiB"
          f"{'' if args.budget_gib else f' (live MemAvailable {live / GIB:.2f} GiB)'}"
          f", device total {total_bytes(torch) / GIB:.2f} GiB")
    if not args.budget_gib:
        print("                 (a server holding the device shrinks this: stop it, or pass --budget-gib)")

    weights = estimate_weights(root, transform, rank=0)
    more = estimate_weights(root, transform, files=list(extras)) if extras else None
    print(f"weights        : resident {weights.resident / GIB:.2f} GiB, staging {weights.staging / GIB:.2f} GiB, "
          f"mapped {weights.mapped / GIB:.2f} GiB"
          + (f" (+ extras resident {more.resident / GIB:.2f}, mapped {more.mapped / GIB:.2f})" if more else ""))
    print()

    failed = 0
    for streams in ([1, args.streams] if args.streams > 1 else [1]):
        if streams > 1:
            geometry = (lambda text: indexed_stream_geometry(text, streams, each, KEEP, mtp=mtp, kv_bits=bits,
                                                             prefill_rows=rows))
        else:
            geometry = (lambda text: gdn_geometry(text, 1, each, indexed=True, mtp=mtp, kv_bits=bits,
                                                  kept=(KEEP_SERIAL or 0) + 1, prefill_rows=rows))
        try:
            plan = admit(root, args.context, True, torch,
                         capacity_geometry(admission(geometry), root, args.vision, 0,
                                           vision_workspace() if args.vision else 0),
                         transform, rank=0, world=1, extra_files=extras)
        except Exception as exc:  # noqa: BLE001 - the refusal text is the answer
            print(f"streams={streams}: REFUSED -> {type(exc).__name__}: {exc}")
            failed += 1
            continue
        allocated = plan.get("allocated_window", plan.get("window"))
        print(f"streams={streams}: fits, estimate {plan['total_bytes_estimate'] / GIB:.2f} GiB, "
              f"allocated window {allocated}, cache slots {plan.get('cache_slots')}")
        if allocated is not None and allocated != args.context and args.context:
            print(f"   WARNING: asked for {args.context} and allocated {allocated} - a silent clamp")
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
