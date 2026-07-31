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
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from codec import fountain, grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--grid", default="120x68")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--combine", action="store_true",
                    help="evidence-integrating receiver: average cell samples "
                         "across every capture of the same displayed frame "
                         "before deciding, instead of hard-deciding each "
                         "capture independently")
    ap.add_argument("--track", action="store_true",
                    help="tracking receiver: when per-frame detection fails, "
                         "locate on a sliding-window average of recent frames "
                         "(fiducials are static, noise integrates away, tremor "
                         "is smooth) and sample the current frame with that "
                         "homography")
    ap.add_argument("--repeat-hint", type=int, default=0,
                    help="captures per displayed frame (camera fps / display "
                         "fps). Lets the receiver treat the frame counter as "
                         "predictable state: after one successful header "
                         "anywhere, seq is propagated by the capture clock and "
                         "the per-block CRC guards mispredictions. 0 = off")
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
    evidence = {}   # seq -> [sum_of_samples, capture_count]
    added = set()   # seqs already fed to the fountain
    window = deque(maxlen=6)   # recent grayscale frames for tracked localization
    tracked = 0
    proto_header = None        # constants (k, block_size, mode) from any good header
    offsets = []               # capture-clock-to-seq sync measurements
    predicted = 0

    while True:
        ok, img = cap.read()
        if not ok:
            break
        n += 1
        if args.max_frames and n > args.max_frames:
            break
        H = None
        if args.track:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            window.append(gray)
            H = grid.locate(img, layout)
            if H is None and len(window) >= 3:
                mean = (sum(window) / len(window)).astype(np.uint8)
                H = grid.locate(mean, layout)
                if H is not None:
                    tracked += 1
            if H is None:
                continue
            H = grid.refine_H(img, layout, H)
        if args.combine or args.track:
            header, pay_samples, stats = grid.sample_frame(img, layout, H)
            located += stats["located"]
            header_ok += stats["header_ok"]
            if stats["cell_margin"]:
                margins.append(stats["cell_margin"])
            if header is not None and args.repeat_hint:
                # sync the capture clock to the sequence counter
                offsets.append(n - header["seq"] * args.repeat_hint)
                proto_header = header
            if header is None and args.repeat_hint and proto_header is not None \
                    and pay_samples is not None:
                # header unreadable but geometry held: predict seq from the
                # capture clock. A wrong prediction dies at the block CRC.
                off = sorted(offsets)[len(offsets) // 2]
                pred = (n - off) // args.repeat_hint
                if pred >= 0:
                    header = dict(proto_header, seq=pred)
                    predicted += 1
            if header is None or header["seq"] in added:
                continue
            if args.combine:
                acc = evidence.setdefault(header["seq"], [0.0, 0])
                acc[0] = acc[0] + pay_samples
                acc[1] += 1
                payload = grid.decide_payload(header, acc[0] / acc[1])
            else:
                payload = grid.decide_payload(header, pay_samples)
            if payload is None:
                continue
            added.add(header["seq"])
        else:
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
    print(f"  located          {located}" +
          (f" ({tracked} rescued by tracking)" if args.track else ""))
    print(f"  header ok        {header_ok}" +
          (f" (+{predicted} seq-predicted)" if predicted else ""))
    print(f"  frame decoded    {rs_ok}")
    if margins:
        print(f"cell margin        median {np.median(margins):.2f} "
              f"(>0.35 comfortable, <0.15 soft-decision territory)")
    if dec is None or not dec.done:
        got = 0 if dec is None else len(dec.decoded)
        want = 0 if dec is None else dec.k
        held = 0 if dec is None else len(dec.pending)
        sys.exit(f"FAILED: fountain incomplete ({got}/{want} blocks solved, "
                 f"{held} coded blocks held as unsolved equations)")

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
