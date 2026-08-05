#!/usr/bin/env python3
"""
analyze_rolling.py — measure the rolling-shutter time axis inside one frame.

The sensor reads rows over ~8-10 ms while the display refreshes every
8.33 ms, so different ROWS of one camera frame sample different refreshes.
The strobe take makes this directly visible: lit refreshes carry consecutive
code frames, dark refreshes carry black, and each grid-row strip's
luminance decomposes over [X(s), X(s+1), const] into exactly one of:

    pure s      (strip exposed inside lit refresh k)
    pure s+1    (strip exposed inside lit refresh k+1)
    dark        (strip exposed inside the black refresh: intercept only)
    seam        (strip straddling a boundary: mixed shares)

Deliverables, measured not assumed:
  - per-frame seam positions and widths (in grid rows)
  - seam drift per frame (camera/display clock beat -> is the seam
    PREDICTABLE, which the rolling-harvest decoder requires)
  - clean-strip fraction: the ceiling multiplier a non-strobe 120 fps
    transmit would get through this same camera at this same exposure

Dress-rehearse on the clean transmit first: every strip must be pure s.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from codec import fountain, grid
from analyze_strobe import truth_cells

N_STRIPS = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--grid", default="252x163")
    ap.add_argument("--ecc", type=int, default=48)
    ap.add_argument("--payload", default=str(Path(__file__).parent.parent /
                                             "demo" / "kitten_big.png"))
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--lo", type=float, default=0.05)
    ap.add_argument("--hi", type=float, default=0.95)
    ap.add_argument("--radial", type=float, default=0.020)
    ap.add_argument("--consecutive", type=int, default=0,
                    help="also analyze N consecutive frames from the middle "
                         "(seam drift needs adjacent frames)")
    args = ap.parse_args()

    grid.set_ecc(args.ecc); grid.set_header_len(28)
    grid.set_header_centered(True); grid.set_radial(args.radial)
    gw, gh = (int(v) for v in args.grid.split("x"))
    L = grid.Layout(gw, gh)
    n_sub = grid.sub_count(L, grid.MODE_MONO)
    SUB = (255 - args.ecc) - 4
    allc = np.argwhere(np.ones((gh, gw), bool))
    data = Path(args.payload).read_bytes()
    enc = fountain.Encoder(data, SUB)

    cap = cv2.VideoCapture(args.capture)
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.consecutive:
        idxs = np.arange(tot // 2, tot // 2 + args.consecutive)
    else:
        idxs = np.linspace(tot * args.lo, tot * args.hi,
                           args.frames * 3).astype(int)

    scratch = {}
    strips = np.array_split(np.arange(gh), N_STRIPS)
    kinds_per_frame = []
    seam_rows = []
    n_probe = 0
    print(f"{'frame':>7s} {'seq':>6s}  strip map (s=this seq, n=next, "
          f"p=prev, .=dark, x=seam)   clean%")
    for fi in idxs:
        if not args.consecutive and n_probe >= args.frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, img = cap.read()
        if not ok:
            continue
        big = img.shape[1] >= 3000
        sm = cv2.resize(img, None, fx=0.5, fy=0.5) if big else img
        Hs = grid.locate(sm, L)
        H = (((np.diag([2., 2., 1.]) @ Hs) if big else Hs)
             if Hs is not None else grid.locate(img, L))
        if H is None:
            continue
        hd, _s, _t = grid.sample_frame(img, L, H)
        if hd is None or int(hd["k"]) != enc.k:
            continue
        s = int(hd["seq"])
        if s < 1:
            continue
        n_probe += 1
        y = grid.sample_cells(img, L, H, allc).mean(axis=1).reshape(
            gh, gw).astype(np.float32)

        xm = truth_cells(L, enc, n_sub, SUB, len(data), s - 1, cache=scratch)
        x0 = truth_cells(L, enc, n_sub, SUB, len(data), s, cache=scratch)
        xp = truth_cells(L, enc, n_sub, SUB, len(data), s + 1, cache=scratch)
        scratch.clear()

        kinds = []
        for st in strips:
            yy = y[st].ravel()
            A = np.stack([xm[st].ravel(), x0[st].ravel(), xp[st].ravel(),
                          np.ones(len(yy))], axis=1)
            coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
            a, b, c = (float(coef[0]), float(coef[1]), float(coef[2]))
            amps = np.array([abs(a), abs(b), abs(c)])
            tot_amp = amps.sum()
            if tot_amp < 25:
                kinds.append(".")          # dark strip: black refresh
                continue
            shares = amps / tot_amp
            top = int(np.argmax(shares))
            if shares[top] > 0.85:
                kinds.append("psn"[top])   # pure prev / this / next
            else:
                kinds.append("x")          # seam strip
        kinds_per_frame.append((fi, s, "".join(kinds)))
        clean = sum(k in "psn" for k in kinds) / len(kinds)
        xpos = [i for i, k in enumerate(kinds) if k == "x"]
        seam_rows.append((fi, xpos))
        print(f"{fi:7d} {s:6d}  {''.join(kinds)}   {100*clean:.0f}%")
    cap.release()

    if not kinds_per_frame:
        print("no usable frames")
        return
    all_kinds = "".join(k for _f, _s, k in kinds_per_frame)
    n = len(all_kinds)
    print(f"\n{len(kinds_per_frame)} frames x {N_STRIPS} strips:")
    for ch, name in ((".", "dark"), ("s", "pure this-seq"),
                     ("n", "pure next-seq"), ("p", "pure prev-seq"),
                     ("x", "seam/mixed")):
        print(f"  {name:>14s}: {100*all_kinds.count(ch)/n:5.1f}%")
    lit = sum(all_kinds.count(ch) for ch in "snp")
    mixed = all_kinds.count("x")
    if lit + mixed:
        print(f"\nCLEAN FRACTION of lit strips: {100*lit/(lit+mixed):.1f}%")
        print("(this is the ceiling multiplier a NON-strobe 120fps transmit")
        print(" gets through this camera at this exposure: rows are clean")
        print(" single-refresh samples wherever a strip is pure)")
    if args.consecutive:
        drifts = []
        prev = None
        for fi, xpos in seam_rows:
            mid = np.mean(xpos) if xpos else None
            if prev is not None and mid is not None and prev[1] is not None \
               and fi == prev[0] + 1:
                drifts.append(mid - prev[1])
            prev = (fi, mid)
        if drifts:
            print(f"\nseam drift per frame (strips): "
                  f"median {np.median(drifts):+.2f}, "
                  f"std {np.std(drifts):.2f}  over {len(drifts)} pairs")
            print("(low std = phase-locked = the seam is PREDICTABLE)")


if __name__ == "__main__":
    main()
