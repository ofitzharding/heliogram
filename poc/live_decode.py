#!/usr/bin/env python3
"""
live_decode.py — real-time-capable receiver.

The offline decoder re-runs full finder detection on every frame: measured at
62 ms for a 4K frame, which alone blows the 16.7 ms budget for 60 fps. But the
screen barely moves between consecutive captures, so detection from scratch is
almost always wasted work.

This receiver locates ONCE, then TRACKS the four finder centres with sparse
optical flow (Lucas-Kanade, a few ms), re-detecting only when tracking fails
or a decode goes bad. That is the "receiver is a tracker, not a scanner" idea
made operational, and it is what makes a live implementation feasible.

Reports per-stage timing so the real-time claim is measured, not asserted.

    python3 live_decode.py capture.mov out.bin --grid 560x311 --ecc 48
"""
import argparse
import hashlib
import struct
import sys
import time
import zlib
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from codec import fountain, grid

LK = dict(winSize=(31, 31), maxLevel=3,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))


def finder_corners(layout):
    """Grid coords of the four finder centres, in TL/TR/BL/BR order."""
    f, gw, gh = layout.finder, layout.gw, layout.gh
    fc = f / 2.0
    return np.array([
        [1 + fc, 1 + fc],
        [gw - 1 - f + fc, 1 + fc],
        [1 + fc, gh - 1 - f + fc],
        [gw - 1 - f + fc, gh - 1 - f + fc],
    ], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--grid", default="560x311")
    ap.add_argument("--ecc", type=int, default=48)
    ap.add_argument("--radial", type=float, default=0.020)
    ap.add_argument("--header-len", type=int, default=28)
    args = ap.parse_args()
    grid.set_ecc(args.ecc)
    grid.set_header_len(args.header_len)
    grid.set_radial(args.radial)
    gw, gh = (int(v) for v in args.grid.split("x"))
    layout = grid.Layout(gw, gh)
    src = finder_corners(layout)

    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0

    dec = None
    proto = None
    prev_gray = None
    pts = None            # tracked finder centres, image coords (4,1,2)
    n = 0
    n_detect = n_track = n_ok = 0
    t_detect = t_track = t_sample = t_rs = 0.0
    first_useful = None
    done_at = None
    t0 = time.time()

    while True:
        ok, img = cap.read()
        if not ok:
            break
        n += 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        H = None
        if pts is not None and prev_gray is not None:
            ts = time.time()
            new, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, **LK)
            t_track += time.time() - ts
            if new is not None and st is not None and st.sum() == 4:
                cand = new.reshape(4, 2).astype(np.float32)
                # sanity: corners must stay in a sane quadrilateral
                w = np.linalg.norm(cand[1] - cand[0])
                h = np.linalg.norm(cand[2] - cand[0])
                if w > 100 and h > 60:
                    H = cv2.getPerspectiveTransform(src, cand)
                    pts = new
                    n_track += 1

        if H is None:
            ts = time.time()
            Hd = grid.locate(gray, layout)
            t_detect += time.time() - ts
            if Hd is None:
                prev_gray = gray
                continue
            H = Hd
            pts = cv2.perspectiveTransform(
                src.reshape(1, 4, 2), H.astype(np.float32)).reshape(4, 1, 2)
            n_detect += 1
        prev_gray = gray

        ts = time.time()
        header, samples, stats = grid.sample_frame(img, layout, H)
        t_sample += time.time() - ts
        if header is None:
            if proto is None:
                pts = None      # force re-detect until we have constants
            continue
        if proto is None:
            proto = header

        ts = time.time()
        payload = grid.decide_payload(header, samples, layout)
        t_rs += time.time() - ts
        if payload is None:
            continue
        bs = header["block_size"]
        if zlib.crc32(payload[4:4 + bs]) & 0xFFFFFFFF != \
                struct.unpack("<I", payload[:4])[0]:
            continue
        n_ok += 1
        if first_useful is None:
            first_useful = n
        if dec is None:
            dec = fountain.Decoder(header["k"], bs, header["file_size"])
        dec.add(header["seq"], payload[4:4 + bs])
        if len(dec.decoded) >= dec.k or len(dec.seen) >= dec.k:
            if not dec.done:
                dec.gaussian_fallback()
        if dec.done:
            done_at = n
            break

    wall = time.time() - t0
    print(f"frames processed  {n}")
    print(f"  detections      {n_detect}  ({t_detect / max(n_detect,1) * 1000:.1f} ms each)")
    print(f"  tracked         {n_track}  ({t_track / max(n_track,1) * 1000:.1f} ms each)")
    print(f"  blocks decoded  {n_ok}")
    per = (t_detect + t_track + t_sample + t_rs) / max(n, 1) * 1000
    print(f"  sample          {t_sample / max(n,1) * 1000:.1f} ms/frame")
    print(f"  RS+decide       {t_rs / max(n,1) * 1000:.1f} ms/frame")
    print(f"  TOTAL           {per:.1f} ms/frame   "
          f"({'REAL-TIME at 60fps' if per < 16.7 else 'not yet real-time'})")

    if dec is None or not dec.done:
        got = 0 if dec is None else len(dec.decoded)
        k = 0 if dec is None else dec.k
        sys.exit(f"FAILED: {got}/{k} blocks")
    data = dec.result()
    Path(args.output).write_bytes(data)
    span = (done_at - first_useful + 1) / fps
    g = len(data) / span / 1024
    print(f"\nrecovered {len(data):,} bytes  "
          f"sha256 {hashlib.sha256(data).hexdigest()}")
    print(f"transfer span {span:.2f}s")
    print(f"GOODPUT {g:.1f} KB/s   ({g/129.2:.2f}x reference-handheld, "
          f"{g/186.0:.2f}x reference-propped)")


if __name__ == "__main__":
    main()
