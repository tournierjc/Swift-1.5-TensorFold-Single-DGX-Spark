"""Header-only preflight: read the checkpoint's tensor headers and print the startup memory plan.

It runs the same estimate the CUDA engine runs before it allocates anything (`capacity.admit` with the Flash
Next geometry and weight transform), so a checkpoint that cannot be sized, or a window that cannot fit, is
reported here in seconds instead of after a 186 GB load.

    docker run --rm --gpus all -v <snapshot>:/snap:ro --entrypoint python3 <image> preflight.py /snap
"""

from __future__ import annotations

import argparse
import sys

GIB = 1024 ** 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Header-only CUDA startup estimate for a checkpoint directory.")
    parser.add_argument("model_dir")
    parser.add_argument("--drafts", type=int, default=6, help="MTP drafts a round (sets the rows the geometry sizes)")
    parser.add_argument("--context", type=int, default=None, help="explicit prompt-plus-reply window")
    parser.add_argument("--tp", type=int, default=1, choices=(1, 2))
    args = parser.parse_args(argv)

    import torch

    from tensorfold.cuda.capacity import admit
    from tensorfold.cuda.geometry import gdn_geometry, indexed_weights

    each = args.drafts + 1
    plan = admit(args.model_dir, args.context, args.context is not None, torch,
                 lambda text: gdn_geometry(text, args.tp, each, indexed=True, mtp=True),
                 indexed_weights(args.tp, True), rank=0, world=args.tp)

    print("\nreceipt:")
    for key, value in plan.items():
        if isinstance(value, int) and abs(value) > 1024:
            print(f"  {key}: {value:,} ({value / GIB:.2f} GiB)")
        else:
            print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
