#!/usr/bin/env python3
"""
probe_sections.py — find which capture frames hold which probe density, then
report what each layout can actually do with them.

extract_cells.py at 350x194 returned the IDENTICAL located-frame set as
252x140 and zero headers, which says grid.locate() finds finder patterns and
fits a homography without checking that the density it was handed is the
density on screen. So "0 headers at 350x194" was never evidence about 350x194
- it is evidence that the frames being sampled were the wrong ones.

Density is identifiable from the frame itself: make_probe renders 252x140 at
12 px/cell (3024 wide, filling the canvas) but 350x194 at 8 (2800 wide) and
466x259 at 6 (2796 wide), each centred on a 3024x1964 black canvas. The width
of the bright content therefore names the density without any decoding.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from codec import grid

DENS = [("252x140", 12), ("350x194", 8), ("466x259", 6)]


def content_box(gray):
    """Columns/rows containing bright content, ignoring the black canvas."""
    m = gray > 60
    cols = np.flatnonzero(m.mean(axis=0) > 0.05)
    rows = np.flatnonzero(m.mean(axis=1) > 0.05)
    if len(cols) < 10 or len(rows) < 10:
        return None
    return cols[0], cols[-1], rows[0], rows[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--start", type=int, default=600)
    ap.add_argument("--n", type=int, default=900)
    ap.add_argument("--stride", type=int, default=6)
    args = ap.parse_args()
    grid.set_ecc(48); grid.set_header_len(28); grid.set_header_centered(True)

    cap = cv2.VideoCapture(args.capture)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    rows = []
    n = args.start
    end = args.start + args.n
    while n < end:
        ok, img = cap.read()
        if not ok:
            break
        fn = n
        n += 1
        if (fn - args.start) % args.stride:
            continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bb = content_box(g)
        if bb is None:
            rows.append((fn, "dark", 0, 0))
            continue
        c0, c1, r0, r1 = bb
        w = c1 - c0 + 1
        rows.append((fn, "code", w, r1 - r0 + 1))
    cap.release()

    ws = np.array([r[2] for r in rows if r[1] == "code"])
    print(f"content widths seen (camera px): "
          f"{np.percentile(ws,[1,25,50,75,99]).round(0) if len(ws) else 'none'}")
    # cluster widths: the three densities differ by ~8% of frame width
    print(f"\n{'frame':>7s} {'width':>7s} {'height':>7s}  guess")
    prev = None
    for fn, kind, w, h in rows:
        if kind == "dark":
            g = "field/stripe"
        else:
            g = "?"
        if g != prev:
            print(f"{fn:7d} {w:7d} {h:7d}  {g}")
        prev = g
    # width histogram is the real output
    if len(ws):
        hist = {}
        for w in ws:
            hist[int(round(w / 20.0) * 20)] = hist.get(int(round(w / 20.0) * 20), 0) + 1
        print("\nwidth histogram (rounded to 20px):")
        for k in sorted(hist):
            print(f"  {k:5d}px  {hist[k]:4d} frames  {'#'*min(60,hist[k])}")


if __name__ == "__main__":
    main()
