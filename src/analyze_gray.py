#!/usr/bin/env python3
"""
analyze_gray.py — read the gray4 probe and report yield per configuration.

Frames are attributed by CONTENT, not by offset: each carries its own mode and
ecc in the header, and a capture starts at an arbitrary point in the loop and
may straddle a wrap. So the analyser tries each configuration on every frame
and keeps whichever produces a header, then certifies that frame's codewords
under that configuration.

Optics, framing, exposure and hold are pinned across the whole take, so a
difference between rows is the alphabet and the parity and nothing else.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from codec import grid
from softdec import FrameDecoder

CONFIGS = [("mono", grid.MODE_MONO, 48),
           ("gray4", grid.MODE_GRAY4, 48),
           ("gray4", grid.MODE_GRAY4, 64)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--grid", default="252x163")
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--lo", type=float, default=0.20)
    ap.add_argument("--hi", type=float, default=0.95)
    ap.add_argument("--k1", default="0.010,0.015,0.020,0.025,0.005")
    args = ap.parse_args()
    grid.set_header_len(28); grid.set_header_centered(True)
    gw, gh = (int(v) for v in args.grid.split("x"))
    allc = np.argwhere(np.ones((gh, gw), bool))
    ks = [float(v) for v in args.k1.split(",")]

    cap = cv2.VideoCapture(args.capture)
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    idxs = np.linspace(tot * args.lo, tot * args.hi, args.frames).astype(int)

    tally = {}          # (name, ecc) -> [certified, total_possible, frames]
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, img = cap.read()
        if not ok:
            continue
        grid.set_ecc(48)
        L = grid.Layout(gw, gh)
        H = grid.locate(img, L)
        if H is None:
            continue
        best = None
        for name, mode, ecc in CONFIGS:
            grid.set_ecc(ecc)
            Lc = grid.Layout(gw, gh)
            n_sub = grid.sub_count(Lc, mode)
            bpc = 1 if mode == grid.MODE_MONO else 2
            fd = FrameDecoder(Lc, ecc, n_sub, erase=True, prml=False,
                              bits_per_cell=bpc)
            for k1 in ks:
                grid.set_radial(k1)
                hd, _s, _t = grid.sample_frame(img, Lc, H)
                # Match mode AND ecc. Two configs share MODE_GRAY4 and differ
                # only in parity, so a mode-only test let whichever gray4 entry
                # was tried first claim every gray4 frame and decode half of
                # them at the wrong ecc - which reads as "gray4/64: 0 frames"
                # and drags gray4/48's yield to zero.
                if hd is None or int(hd["mode"]) != mode:
                    continue
                if int(hd.get("ecc", ecc)) != ecc:
                    continue
                y = grid.sample_cells(img, Lc, H, allc).mean(axis=1).reshape(
                    gh, gw).astype(np.float32)
                if mode == grid.MODE_MONO:
                    n = fd.quick_count(y)
                else:
                    # gray4 demodulation lives in grid.raw_bits_and_conf
                    samp = np.repeat(
                        y[Lc.payload_cells[:, 0],
                          Lc.payload_cells[:, 1]][:, None], 3, axis=1)
                    raw, bc = grid.raw_bits_and_conf(hd, samp, Lc)
                    bits = np.unpackbits(np.frombuffer(raw, np.uint8)
                                         ).astype(np.uint8)
                    n = len(fd.certify(bits, bc)[0])
                pv = y[Lc.payload_cells[:, 0], Lc.payload_cells[:, 1]]
                if best is None or n > best[0]:
                    best = (n, name, ecc, n_sub, pv)
                break
        if best is None:
            continue
        n, name, ecc, n_sub, pv = best
        key = (name, ecc)
        t = tally.setdefault(key, [0, 0, 0, []])
        t[0] += n; t[1] += n_sub; t[2] += 1
        # Photometry, so a null result can be told apart from a dark take.
        # gray4 needs FOUR separable levels; if the eye is crushed the middle
        # two collapse and the answer is falsely negative (this is how the
        # Pass-15 gray4 verdict died).
        t[3].append((np.percentile(pv, 5), np.percentile(pv, 50),
                     np.percentile(pv, 95)))

    cap.release()
    if not tally:
        print("no frame produced a header at any configuration")
        return
    print(f"{args.capture}, {args.grid}, {fps:.0f} fps\n")
    print(f"{'config':>12s} {'frames':>7s} {'yield':>7s} {'ceiling':>9s} "
          f"{'KB/s':>8s}   {'p5':>4s} {'p50':>4s} {'p95':>4s}")
    for (name, ecc), (got, poss, nf, phot) in sorted(tally.items()):
        grid.set_ecc(ecc)
        Lc = grid.Layout(gw, gh)
        mode = grid.MODE_MONO if name == "mono" else grid.MODE_GRAY4
        ns = grid.sub_count(Lc, mode); sub = (255 - ecc) - 4
        ceil = ns * sub * fps / 1024
        y_ = got / max(poss, 1)
        p = np.median(np.array(phot), axis=0) if phot else [0, 0, 0]
        print(f"{name+'/'+str(ecc):>12s} {nf:7d} {100*y_:6.1f}% {ceil:8.1f}K "
              f"{ceil*y_:7.1f}   {p[0]:4.0f} {p[1]:4.0f} {p[2]:4.0f}")
    print("\nOptics, framing, exposure and hold are identical across every row,")
    print("so a difference between rows is the alphabet and the parity alone.")


if __name__ == "__main__":
    main()
