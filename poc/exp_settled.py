#!/usr/bin/env python3
"""
exp_settled.py — the number the record take will actually get.

Everything measured so far was contaminated by something: the record take by
overexposure and 27% straddle, the earlier probe measurement by a window that
sat inside the camera's settling transient AND mixed hold=1/2/4 sections into
one average.

This measures the one regime a record take is actually in: ONE density, hold=1,
after the camera has settled, decoded by the full production path with a
per-frame radial estimate. The result answers the only question left - whether
252x163 (ceiling 226.0 KB/s) clears 200 KB/s, which needs 88.5% codeword yield.

Frames are taken in capture order and the kernel donor rolls forward exactly as
it would live, so nothing here is an upper bound obtained by cherry-picking.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from codec import grid
from softdec import FrameDecoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--grid", default="252x140")
    ap.add_argument("--ecc", type=int, default=48)
    ap.add_argument("--start", type=int, default=1500,
                    help="skip the AE/AF settling transient (~7s = 420 frames)")
    ap.add_argument("--n", type=int, default=2600)
    ap.add_argument("--hold", type=int, default=1,
                    help="only frames whose header advertises this hold")
    ap.add_argument("--k1", default="0.010,0.013,0.015,0.018,0.020,0.023",
                    help="per-frame candidates; the one certifying most wins")
    ap.add_argument("--refit", type=int, default=0)
    ap.add_argument("--target-grid", default="252x163",
                    help="report what this yield would give at the grid we "
                         "intend to film")
    args = ap.parse_args()

    grid.set_ecc(args.ecc); grid.set_header_len(28)
    grid.set_header_centered(True)
    gw, gh = (int(v) for v in args.grid.split("x"))
    L = grid.Layout(gw, gh)
    n_sub = grid.sub_count(L, grid.MODE_MONO)
    SUB = (255 - args.ecc) - 4
    allc = np.argwhere(np.ones((gh, gw), bool))
    ks = [float(v) for v in args.k1.split(",")]
    fd = FrameDecoder(L, args.ecc, n_sub, erase=True, prml=True,
                      refit=args.refit or None)

    cap = cv2.VideoCapture(args.capture)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    n = args.start
    tot = frames = 0
    hist = []
    print(f"{args.grid} hold={args.hold} from frame {args.start}, "
          f"n_sub={n_sub}, ceiling {n_sub*SUB*60/1024:.1f} KB/s")
    while n < args.start + args.n:
        ok, img = cap.read()
        if not ok:
            break
        fn = n
        n += 1
        H = grid.locate(img, L)
        if H is None:
            continue
        best = None
        for k1 in ks:
            grid.set_radial(k1)
            hd, _s, _t = grid.sample_frame(img, L, H)
            if hd is None:
                continue
            if int(hd.get("zone_w", 0) or 1) != args.hold:
                best = "wronghold"
                break
            y = grid.sample_cells(img, L, H, allc).mean(axis=1).reshape(
                gh, gw).astype(np.float32)
            blocks = fd.decode(y, hd)
            if best is None or not isinstance(best, tuple) or \
                    len(blocks) > len(best[1]):
                best = (k1, blocks, hd)
            if len(blocks) == n_sub:
                break
        if best is None or best == "wronghold":
            continue
        tot += len(best[1])
        frames += 1
        hist.append(len(best[1]))
        if frames <= 25 or frames % 25 == 0:
            print(f"  f{fn:5d} k1={best[0]:+.3f} seq={best[2]['seq']:4d} "
                  f"{len(best[1]):3d}/{n_sub}")
    cap.release()
    if not frames:
        print("no frames matched")
        return
    y_ = tot / (frames * n_sub)
    h = np.array(hist)
    print(f"\n{frames} frames, {fd.donors} kernel donors")
    print(f"codeword yield {100*y_:.1f}%   "
          f"(median {np.median(h):.0f}/{n_sub}, "
          f"{100*(h==n_sub).mean():.0f}% of frames at full)")
    print(f"at {args.grid}: {n_sub*SUB*60/1024*y_:.1f} KB/s")
    tw, th_ = (int(v) for v in args.target_grid.split("x"))
    L2 = grid.Layout(tw, th_)
    ns2 = grid.sub_count(L2, grid.MODE_MONO)
    rate2 = ns2 * SUB * 60 / 1024
    print(f"at {args.target_grid} ({ns2} codewords/frame, same 12px cells, "
          f"same camera px/cell):")
    print(f"    ceiling {rate2:.1f} KB/s, at this yield -> "
          f"{rate2*y_:.1f} KB/s")
    print(f"    200 KB/s needs {100*200.0/rate2:.1f}% yield  "
          f"[{'CLEARS' if rate2*y_ >= 200 else 'SHORT by %.1f KB/s' % (200-rate2*y_)}]")


if __name__ == "__main__":
    main()
