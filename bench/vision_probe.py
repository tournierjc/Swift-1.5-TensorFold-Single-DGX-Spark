#!/usr/bin/env python3
"""Ask a running serve about media, and generate what it is asked about.

    # a solid red PNG, a black-banded 2048x2048 PNG, or a 2 s all-red clip, in one go
    docker run --rm --network host -v "$PWD/bench:/bench" --entrypoint python3 tensorfold-spark:local \\
      /bench/vision_probe.py --base http://127.0.0.1:8083 --model qwen3.8-flash-next --cases image video

Needs `--vision` on the server, and at least two lanes (the engine refuses visual input below that). The clip is
built with PyAV, which the image carries, so the host needs neither ffmpeg nor a Python environment. Vision is
also what makes a *small* client `max_tokens` look like a broken model: with `--thinking` the budget is spent
entirely on `reasoning_tokens` and `content` comes back empty, so the probes use a budget the reply fits in.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
import urllib.error
import urllib.request

QUESTIONS = {
    "image": "Quelle est la couleur dominante de cette image ? Un mot.",
    "video": "Quelle est la couleur dominante de cette video ? Un mot.",
}

# RGB triples. Asking for the same picture in two colours is what makes the probe discriminating: a model that
# answered "Rouge" whatever arrived would pass the red case and fail the blue one.
COLOURS = {"red": (220, 30, 30), "blue": (30, 30, 220)}


def solid_png(rgb: tuple[int, int, int], side: int = 512) -> bytes:
    from PIL import Image, ImageDraw

    buf = io.BytesIO()
    image = Image.new("RGB", (side, side), rgb)
    if side > 1024:  # a band, so a large image is not one flat colour the tower could shortcut
        ImageDraw.Draw(image).rectangle([0, side // 3, side, side // 3 + side // 8], fill=(10, 10, 10))
    image.save(buf, format="PNG")
    return buf.getvalue()


def solid_mp4(rgb: tuple[int, int, int], seconds: float = 2.0, side: int = 320, fps: int = 4) -> bytes:
    import av
    import numpy as np

    buf = io.BytesIO()
    with av.open(buf, mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=fps, options={"movflags": "faststart", "crf": "30"})
        stream.width, stream.height, stream.pix_fmt = side, side, "yuv420p"
        for _ in range(int(seconds * fps)):
            frame = av.VideoFrame.from_ndarray(np.full((side, side, 3), rgb, np.uint8), format="rgb24")
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))
    return buf.getvalue()


def ask(base: str, model: str, kind: str, payload: bytes, max_tokens: int) -> int:
    url = f"data:{'image/png' if kind == 'image' else 'video/mp4'};base64," + base64.b64encode(payload).decode()
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": QUESTIONS[kind]},
            {"type": kind + "_url", kind + "_url": {"url": url}},
        ]}],
        "max_tokens": max_tokens,
    }).encode()
    request = urllib.request.Request(f"{base}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
    start = time.time()
    try:
        with urllib.request.urlopen(request, timeout=1800) as reply:
            payload_json = json.load(reply)
    except urllib.error.HTTPError as error:
        print(f"{kind}: HTTP {error.code} in {time.time() - start:.1f} s :: {error.read()[:400].decode(errors='replace')}")
        return 1
    took = time.time() - start
    message = payload_json["choices"][0]["message"]
    usage = payload_json.get("usage", {})
    answers = (message.get("content") or "").strip() or (message.get("reasoning_content") or "").strip()
    print(f"{kind}: {took:.2f} s, {usage}")
    print(f"  answer: {answers[:200]!r}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe a running serve with a generated image and clip.")
    parser.add_argument("--base", default="http://127.0.0.1:8083")
    parser.add_argument("--model", default="swift-1.5")
    parser.add_argument("--cases", nargs="+", default=["image", "video"], choices=("image", "video"),
                        help="image: a solid square PNG; video: a 2 s solid clip at 4 fps")
    parser.add_argument("--side", type=int, default=512, help="the image's side, and the clip's")
    parser.add_argument("--colour", default="red", choices=tuple(COLOURS), help="the picture's colour")
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    failed = 0
    for kind in args.cases:
        rgb = COLOURS[args.colour]
        payload = solid_png(rgb, args.side) if kind == "image" else solid_mp4(rgb, side=args.side)
        failed += ask(args.base, args.model, kind, payload, args.max_tokens)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
