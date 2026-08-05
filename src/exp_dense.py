#!/usr/bin/env python3
"""
exp_dense.py — what does 350x194 actually yield, with the geometry it needs?

Every previous statement about the dense regime ("certifies ZERO codewords
cold") rests on decoding dense frames with a radial coefficient chosen for a
different grid. k1 is a property of the FRAMING - how far the phone was from
the screen and where the code sat in the lens field - not of the code. The
252 grid fills the panel and needed k1 ~ +0.020; the 350 grid is rendered at
8 px/cell and letterboxed to 2800 of 3024 px, so it occupies a smaller,
more central part of the lens field and needs k1 ~ +0.0025. That value was
in nobody's candidate list.

Measured on IMG_7867 frame 1170, structure-cell agreement (finders, ring and
separators - cells the receiver knows a priori, so no source file is needed):

    252x140   52.1%   at k1 +0.0425     <- chance; wrong density
    350x194   96.5%   at k1 +0.0025     <- correct density AND correct geometry
    466x259   58.9%   at k1 -0.0100     <- chance; wrong density

96.5% of known cells landing correctly is not a broken lattice. So the
question the dense regime was never actually asked is what it yields once
sampled on grid.

This identifies dense frames by that same structure agreement rather than by
a header, because the header is what fails first at this density and using it
as the gate is what hid the result.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from codec import grid
from softdec import FrameDecoder


def ref_structure(L):
    blank = np.zeros(L.payload_capacity_bytes(grid.MODE_MONO), np.uint8).tobytes()
    raw = grid.pack_header(0, 1, 10, 10, grid.MODE_MONO, 0, 0)
    img = grid.render_frame(L, raw, blank, grid.MODE_MONO, cell_px=1)
    return (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 127).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--grid", default="350x194")
    ap.add_argument("--start", type=int, default=1100)
    ap.add_argument("--n", type=int, default=260)
    ap.add_argument("--ecc", type=int, default=48)
    ap.add_argument("--gate", type=float, default=0.85,
                    help="structure agreement a frame must reach to count as "
                         "this density")
    ap.add_argument("--k1lo", type=float, default=-0.010)
    ap.add_argument("--k1hi", type=float, default=0.030)
    ap.add_argument("--k1step", type=float, default=0.0025)
    ap.add_argument("--sweeps", type=int, default=3)
    args = ap.parse_args()

    grid.set_ecc(args.ecc); grid.set_header_len(28)
    grid.set_header_centered(True)
    gw, gh = (int(v) for v in args.grid.split("x"))
    L = grid.Layout(gw, gh)
    n_sub = grid.sub_count(L, grid.MODE_MONO)
    SUB = (255 - args.ecc) - 4
    xt = ref_structure(L)
    st = L.is_finder | L.is_ring | L.is_sep
    scells = np.argwhere(st)
    swant = xt[scells[:, 0], scells[:, 1]]
    allc = np.argwhere(np.ones((gh, gw), bool))

    fd = FrameDecoder(L, args.ecc, n_sub, sweeps=args.sweeps,
                      erase=True, prml=True)
    # No header at this density yet, so pin only cells that are structurally
    # known. The header strip is excluded rather than guessed.
    fd.known = st

    cap = cv2.VideoCapture(args.capture)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    ks = np.arange(args.k1lo, args.k1hi + 1e-9, args.k1step)
    print(f"{args.grid}: n_sub={n_sub} SUB={SUB} ceiling "
          f"{n_sub*SUB*60/1024:.1f} KB/s at 60fps full yield")
    print(f"{'frame':>7s} {'k1':>8s} {'struct':>7s} {'px/cell':>8s} "
          f"{'cw':>7s} {'hdr seq':>8s}")
    n = args.start
    tot = ok_frames = 0
    hist = []
    while n < args.start + args.n:
        got, img = cap.read()
        if not got:
            break
        fn = n
        n += 1
        H = grid.locate(img, L)
        if H is None:
            continue
        best = (-1.0, 0.0)
        for k1 in ks:
            grid.set_radial(float(k1))
            v = grid.sample_cells(img, L, H, scells).mean(axis=1)
            t, _ = cv2.threshold(np.clip(v, 0, 255).astype(np.uint8), 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            a = float(((v > t).astype(np.float32) == swant).mean())
            if a > best[0]:
                best = (a, float(k1))
        acc, k1 = best
        if acc < args.gate:
            continue
        grid.set_radial(k1)
        y = grid.sample_cells(img, L, H, allc).mean(axis=1).reshape(gh, gw
                                                                   ).astype(np.float32)
        hd, _s, _t = grid.sample_frame(img, L, H)
        header = hd if hd is not None else dict(seq=0, k=1, block_size=SUB,
                                                file_size=10)
        blocks = fd.decode(y, header)
        c = np.array([[0, 0], [L.gw, 0]], np.float32).reshape(-1, 1, 2)
        p = cv2.perspectiveTransform(c, H).reshape(-1, 2)
        pxc = np.linalg.norm(p[1] - p[0]) / L.gw
        tot += len(blocks)
        ok_frames += 1
        hist.append(len(blocks))
        if ok_frames <= 40 or ok_frames % 10 == 0:
            print(f"{fn:7d} {k1:+8.4f} {100*acc:6.1f}% {pxc:8.2f} "
                  f"{len(blocks):3d}/{n_sub:<3d} "
                  f"{'-' if hd is None else hd['seq']:>8}")
    cap.release()
    if not ok_frames:
        print("\nno frame passed the structure gate at this density")
        return
    y_ = tot / (ok_frames * n_sub)
    rate = n_sub * SUB * 60 / 1024
    print(f"\n{ok_frames} frames passed the structure gate, {fd.donors} donors")
    print(f"codeword yield {100*y_:.1f}%  ->  {rate*y_:.1f} KB/s at 60 fps")
    h = np.array(hist)
    print(f"per-frame codewords: median {np.median(h):.0f}  "
          f"max {h.max()}  frames at full {int((h==n_sub).sum())}")
    print(f"yield needed for 200 KB/s at this density: "
          f"{100*200.0/rate:.1f}%")


if __name__ == "__main__":
    main()
