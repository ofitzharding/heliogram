#!/usr/bin/env python3
"""
exp_straddle_sic.py — a straddled frame carries TWO transmit frames. Decode both.

At hold=1 the display advances every refresh, so a camera frame whose exposure
is not phase-locked integrates the tail of transmit frame s and the head of
s+1. Measured on IMG_7872 by regressing each camera frame against both:

    beta share 0.00-0.01  ->  17-18/19 codewords
    beta share 0.12-0.13  ->  17-19/19
    beta share 0.15-0.19  ->  13-16/19
    beta share 0.22       ->  11/19

so straddle share predicts yield, and the two-frame model collapses the
residual whenever the share is nonzero (frame 956: R^2 0.793 -> 0.859). Every
receiver in this project treats that second component as noise and throws it
away.

It is not noise, it is a second transmission. The channel is a two-user
multiple-access channel and the standard answer is successive interference
cancellation: decode the stronger user, subtract it, decode the weaker one out
of the residual.

SIC normally fails on error propagation - you subtract your estimate, and any
error in it corrupts everything downstream. That failure mode is impossible
here, for the same structural reason certified-label channel learning works
(Findings section 3). The subtracted component is not an estimate. It is a set
of RS codewords that each passed their own CRC32, so re-encoding them
reproduces the transmitted cells EXACTLY. The canceller cannot drift because
there is nothing approximate in it.

Per frame:
  1. certify what you can normally           -> exact cells of X(s)
  2. fit y = a*X(s) + c on THOSE cells only  -> the mixing coefficient
  3. r = y - a*X(s) - c                      -> what is left is b*X(s+1)
  4. threshold r locally and certify against seq s+1

Step 4 needs no ground truth: a codeword either passes RS+CRC or it does not,
so every symbol this recovers is correct by construction. The measurement below
counts only symbols the ORDINARY path missed.
"""
import argparse
import struct
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from codec import grid, fountain
from softdec import FrameDecoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--grid", default="252x163")
    ap.add_argument("--ecc", type=int, default=48)
    ap.add_argument("--payload", default=str(Path(__file__).parent.parent /
                                             "demo" / "payload.png"))
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--lo", type=float, default=0.15)
    ap.add_argument("--hi", type=float, default=0.90)
    ap.add_argument("--radial", type=float, default=0.025)
    args = ap.parse_args()

    grid.set_ecc(args.ecc); grid.set_header_len(28)
    grid.set_header_centered(True); grid.set_radial(args.radial)
    gw, gh = (int(v) for v in args.grid.split("x"))
    L = grid.Layout(gw, gh)
    n_sub = grid.sub_count(L, grid.MODE_MONO)
    SUB = (255 - args.ecc) - 4
    pc = L.payload_cells
    allc = np.argwhere(np.ones((gh, gw), bool))
    data = Path(args.payload).read_bytes()
    enc = fountain.Encoder(data, SUB)
    fd = FrameDecoder(L, args.ecc, n_sub, erase=True, prml=False)

    cap = cv2.VideoCapture(args.capture)
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(tot * args.lo, tot * args.hi, args.frames * 3).astype(int)

    base_tot = extra_tot = frames = 0
    print(f"{'frame':>7s} {'seq':>6s} {'base':>7s} {'a':>6s} "
          f"{'SIC extra':>10s}  (extra = codewords of seq+1 the base path missed)")
    for fi in idxs:
        if frames >= args.frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, img = cap.read()
        if not ok:
            continue
        H = grid.locate(img, L)
        if H is None:
            continue
        hd, _s, _t = grid.sample_frame(img, L, H)
        if hd is None or int(hd["k"]) != enc.k:
            continue
        s = int(hd["seq"])
        y = grid.sample_cells(img, L, H, allc).mean(axis=1).reshape(
            gh, gw).astype(np.float32)
        lum = y[pc[:, 0], pc[:, 1]]

        # --- 1. ordinary path
        bits, conf = grid._mono_decide(lum, L, pc)
        nb = n_sub * 255
        bc = conf[: nb * 8].reshape(nb, 8).min(axis=1)
        blocks, cmask, cbits = fd.certify(bits, bc)
        base = len(blocks)
        if base == 0:
            continue
        frames += 1
        base_tot += base

        # --- 2. mixing coefficient, fitted ONLY on cells the code certified
        cells = fd.cells
        idx = np.flatnonzero(cmask)
        if len(idx) < 2000:
            print(f"{fi:7d} {s:6d} {base:3d}/{n_sub:<3d} {'-':>6s} "
                  f"{0:10d}   (too few certified cells to fit)")
            continue
        # map certified cell positions back into the payload-order vector
        cy = y[cells[idx, 0], cells[idx, 1]]
        cx = cbits[idx]
        A = np.stack([cx, np.ones_like(cx)], axis=1)
        coef, *_ = np.linalg.lstsq(A, cy, rcond=None)
        a, c = float(coef[0]), float(coef[1])

        # --- 3. cancel the certified component everywhere it is known
        known = np.zeros(len(pc), np.float32)
        kmask = np.zeros(len(pc), bool)
        pos = np.zeros(len(cells), np.int64)
        pos[:] = np.arange(len(cells))
        known[pos[idx]] = cbits[idx]
        kmask[pos[idx]] = True
        r = lum.copy()
        r[kmask] = lum[kmask] - a * known[kmask]

        # --- 4. demodulate the residual and certify against seq+1
        rb, rconf = grid._mono_decide(r, L, pc)
        rbc = rconf[: nb * 8].reshape(nb, 8).min(axis=1)
        blocks2, _m2, _b2 = fd.certify(rb, rbc)
        # only count codewords that decode as symbols of seq+1, i.e. whose
        # content matches the NEXT frame's fountain blocks
        extra = 0
        for j, blk in blocks2:
            want = enc.block((s + 1) * n_sub + j)
            want = want + b"\x00" * (SUB - len(want))
            if blk == want:
                extra += 1
        extra_tot += extra
        print(f"{fi:7d} {s:6d} {base:3d}/{n_sub:<3d} {a:6.1f} {extra:10d}")
    cap.release()

    if not frames:
        print("no usable frames")
        return
    print(f"\n{frames} frames")
    print(f"ordinary path       : {base_tot:5d} codewords "
          f"({base_tot/frames:.1f}/frame)")
    print(f"SIC on the residual : {extra_tot:5d} extra codewords of seq+1 "
          f"({extra_tot/frames:.1f}/frame)")
    if base_tot:
        print(f"gain                : {1 + extra_tot/base_tot:.3f}x symbols "
              f"from the SAME footage")


if __name__ == "__main__":
    main()
