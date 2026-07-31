#!/usr/bin/env python3
"""
decode.py — captured/simulated video -> recovered file + channel report.

    python3 decode.py capture.mp4 recovered.bin [--grid 120x68]

Reports per-frame outcomes and overall goodput accounting. The `cell_margin`
statistic is the seed of the soft-decision work: today it is only reported;
the decoder still makes hard cell decisions. When margins are persistently
low but RS still succeeds, that is the regime where feeding per-cell
confidence into the fountain decoder (soft-decision LT) buys real rate —
the research hook, deliberately left visible in the stats output.
"""
import argparse
import hashlib
import struct
import sys
import time
import zlib
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))
from codec import fountain, grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--grid", default="120x68")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    gw, gh = (int(v) for v in args.grid.split("x"))
    layout = grid.Layout(gw, gh)

    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    dec = None
    n = located = header_ok = rs_ok = 0
    margins = []
    t0 = time.time()
    frames_at_done = None

    while True:
        ok, img = cap.read()
        if not ok:
            break
        n += 1
        if args.max_frames and n > args.max_frames:
            break
        header, payload, stats = grid.decode_frame(img, layout)
        located += stats["located"]
        header_ok += stats["header_ok"]
        if stats["cell_margin"]:
            margins.append(stats["cell_margin"])
        if header is None or payload is None:
            continue
        bs = header["block_size"]
        crc = struct.unpack("<I", payload[:4])[0]
        block = payload[4:4 + bs]
        if zlib.crc32(block) & 0xFFFFFFFF != crc:
            continue  # RS mis-correction or straddled frame — reject
        rs_ok += 1
        if dec is None:
            dec = fountain.Decoder(header["k"], bs, header["file_size"])
        dec.add(header["seq"], block)
        if dec.done and frames_at_done is None:
            frames_at_done = n
            break

    if dec is not None and not dec.done:
        if dec.gaussian_fallback():
            frames_at_done = n
    wall = time.time() - t0

    print(f"frames read        {n}")
    print(f"  located          {located}")
    print(f"  header ok        {header_ok}")
    print(f"  frame decoded    {rs_ok}")
    if margins:
        import numpy as np
        print(f"cell margin        median {np.median(margins):.2f} "
              f"(>0.35 comfortable, <0.15 soft-decision territory)")
    if dec is None or not dec.done:
        got = 0 if dec is None else len(dec.decoded)
        want = 0 if dec is None else dec.k
        sys.exit(f"FAILED: fountain incomplete ({got}/{want} blocks)")

    data = dec.result()
    Path(args.output).write_bytes(data)
    seconds_of_video = frames_at_done / fps
    goodput = len(data) / seconds_of_video / 1024
    print(f"recovered          {len(data):,} bytes  sha256 {hashlib.sha256(data).hexdigest()[:16]}")
    print(f"video time used    {seconds_of_video:.1f}s of capture ({frames_at_done} frames @ {fps:.0f}fps)")
    print(f"GOODPUT            {goodput:.1f} KB/s")
    print(f"decode wall time   {wall:.1f}s "
          f"({'faster' if wall < seconds_of_video else 'SLOWER'} than realtime)")


if __name__ == "__main__":
    main()
