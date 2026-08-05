#!/usr/bin/env python3
"""
analyze_density.py — read the density ladder and locate the cliff.

One row per cell size. The 12px row is the control: this rig gives it ~90%
yield on a good take, so if it comes back far below that the take is bad and no
other row means anything. That check is what stopped IMG_7879 being read as a
gray4 refutation when it was actually a saturated sensor.

Figure of merit is KB/s, not yield. A smaller cell carries more codewords per
frame, so it can win at a much lower yield percentage.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from codec import grid
from softdec import FrameDecoder

PW, PH = 3024, 1964
LADDER = [12, 11, 10, 9]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--ecc", type=int, default=48)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--lo", type=float, default=0.15)
    ap.add_argument("--hi", type=float, default=0.97)
    ap.add_argument("--k1", default="0.010,0.015,0.020,0.025,0.005,0.000")
    args = ap.parse_args()
    grid.set_ecc(args.ecc); grid.set_header_len(28)
    grid.set_header_centered(True)
    SUB = (255 - args.ecc) - 4
    ks = [float(v) for v in args.k1.split(",")]

    plans = {}
    for cp in LADDER:
        gw, gh = PW // cp, PH // cp
        L = grid.Layout(gw, gh)
        ns = grid.sub_count(L, grid.MODE_MONO)
        plans[cp] = (gw, gh, L, ns,
                     np.argwhere(np.ones((gh, gw), bool)),
                     FrameDecoder(L, args.ecc, ns, erase=True, prml=False))

    cap = cv2.VideoCapture(args.capture)
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    idxs = np.linspace(tot * args.lo, tot * args.hi, args.frames).astype(int)

    tally = {}
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, img = cap.read()
        if not ok:
            continue
        best = None
        for cp, (gw, gh, L, ns, allc, fd) in plans.items():
            H = grid.locate(img, L)
            if H is None:
                continue
            for k1 in ks:
                grid.set_radial(k1)
                hd, _s, _t = grid.sample_frame(img, L, H)
                if hd is None:
                    continue
                if int(hd.get("zone_w", 0)) != cp:
                    continue          # a frame from a different rung
                y = grid.sample_cells(img, L, H, allc).mean(axis=1).reshape(
                    gh, gw).astype(np.float32)
                n = fd.quick_count(y)
                if best is None or n > best[0]:
                    best = (n, cp, ns)
                break
        if best is None:
            continue
        n, cp, ns = best
        t = tally.setdefault(cp, [0, 0, 0])
        t[0] += n; t[1] += ns; t[2] += 1
    cap.release()

    if not tally:
        print("no frame produced a header at any density")
        return
    print(f"{args.capture}, {fps:.0f} fps\n")
    print(f"{'cell px':>8s} {'grid':>10s} {'cam px/cell':>12s} {'frames':>7s} "
          f"{'yield':>7s} {'ceiling':>9s} {'KB/s':>8s}")
    for cp in LADDER:
        if cp not in tally:
            print(f"{cp:8d} {'-':>10s} {cp*13.4/12:12.1f} {0:7d} "
                  f"{'none decoded':>17s}")
            continue
        got, poss, nf = tally[cp]
        gw, gh, L, ns, _a, _f = plans[cp]
        ceil = ns * SUB * fps / 1024
        y_ = got / max(poss, 1)
        print(f"{cp:8d} {str(gw)+'x'+str(gh):>10s} {cp*13.4/12:12.1f} {nf:7d} "
              f"{100*y_:6.1f}% {ceil:8.1f}K {ceil*y_:7.1f}")

    if 12 in tally:
        got, poss, _n = tally[12]
        c = got / max(poss, 1)
        print(f"\nCONTROL: 12px yield {100*c:.1f}%.", end=" ")
        print("Take is sound." if c >= 0.60 else
              "TAKE IS BAD - this rig gives 12px ~90% when the exposure is\n"
              "right, so no row above is a verdict on density. Refilm.")
    print("\nRead KB/s, not yield: a smaller cell carries more codewords per\n"
          "frame, so it wins at a lower yield percentage.")


if __name__ == "__main__":
    main()
