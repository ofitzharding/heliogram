#!/usr/bin/env python3
"""
quickcheck.py — is this take worth decoding? Answer in under a minute.

A full decode of a 70-second 4K clip is ~10 minutes, and the last several takes
were each found to be unusable only at the end of one. This samples ~120 frames
and reports the three factors that multiply into goodput, so a bad take is
caught while the lamp is still where it was and the phone is still in hand.

    goodput = header yield x codeword yield x ceiling

Diagnoses the cause when it is short, because the two failure modes need
opposite responses: straddle is fought with the display, exposure with the
room. It never touches the exposure slider as a remedy - that cuts exposure
without raising ISO, so a short shutter necessarily underexposes, and one take
at 3 stops down read full-white at 25/255.
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
    ap.add_argument("--grid", default="252x163")
    ap.add_argument("--ecc", type=int, default=48)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--lo", type=float, default=0.25)
    ap.add_argument("--hi", type=float, default=0.95)
    ap.add_argument("--k1", default="0.010,0.015,0.020,0.025,0.005,0.000")
    args = ap.parse_args()

    grid.set_ecc(args.ecc); grid.set_header_len(28)
    grid.set_header_centered(True)
    gw, gh = (int(v) for v in args.grid.split("x"))
    L = grid.Layout(gw, gh)
    n_sub = grid.sub_count(L, grid.MODE_MONO)
    SUB = (255 - args.ecc) - 4
    ceiling = n_sub * SUB * 60 / 1024
    allc = np.argwhere(np.ones((gh, gw), bool))
    pc = L.payload_cells
    fd = FrameDecoder(L, args.ecc, n_sub, erase=True, prml=False)
    ks = [float(v) for v in args.k1.split(",")]

    cap = cv2.VideoCapture(args.capture)
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    idxs = np.linspace(tot * args.lo, tot * args.hi, args.frames).astype(int)
    loc = hdr = 0
    cw, mids, p50s, p5s, p95s, k1s = [], [], [], [], [], []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, img = cap.read()
        if not ok:
            continue
        H = grid.locate(img, L)
        if H is None:
            continue
        loc += 1
        best = None
        for k1 in ks:
            grid.set_radial(k1)
            hd, _s, _t = grid.sample_frame(img, L, H)
            if hd is None:
                continue
            y = grid.sample_cells(img, L, H, allc).mean(axis=1).reshape(
                gh, gw).astype(np.float32)
            n = fd.quick_count(y)
            if best is None or n > best[0]:
                best = (n, k1, y)
            if n == n_sub:
                break
        if best is None:
            continue
        hdr += 1
        n, k1, y = best
        cw.append(n)
        k1s.append(k1)
        pv = y[pc[:, 0], pc[:, 1]]
        a, b = np.percentile(pv, 3), np.percentile(pv, 97)
        mids.append(float(((pv > a + .3 * (b - a)) & (pv < a + .7 * (b - a))).mean()))
        p5s.append(np.percentile(pv, 5)); p50s.append(np.percentile(pv, 50))
        p95s.append(np.percentile(pv, 95))
    cap.release()

    ns = len(idxs)
    if not cw:
        print(f"located {loc}/{ns}, headers 0 — nothing decodable at {args.grid}")
        return
    hy, cy = hdr / ns, float(np.mean(cw)) / n_sub
    est = ceiling * hy * cy
    p5, p50, p95 = np.median(p5s), np.median(p50s), np.median(p95s)
    mid = float(np.median(mids))
    print(f"{args.grid}  ceiling {ceiling:.1f} KB/s at {fps:.0f}fps\n")
    print(f"  located          {100*loc/ns:5.1f}%")
    print(f"  header yield     {100*hy:5.1f}%")
    print(f"  codeword yield   {100*cy:5.1f}%   "
          f"(median {np.median(cw):.0f}/{n_sub}, "
          f"{100*np.mean(np.array(cw)==n_sub):.0f}% of frames full)")
    print(f"  k1 chosen        {np.median(k1s):+.4f}")
    print(f"  straddle         {100*mid:5.1f}%   (mid-band payload cells)")
    print(f"  photometry       p5 {p5:.0f}  p50 {p50:.0f}  p95 {p95:.0f}"
          f"   eye midpoint {(p5+p95)/2:.0f}")
    print(f"\n  ESTIMATE (hard RS, no PRML): {est:.1f} KB/s")
    print(f"  the soft path adds roughly 1.4x on top of this\n")

    # ---- verdict.
    #
    # Driven by the estimated RATE, with the photometric readings used only to
    # explain a shortfall. An earlier version gated on the readings themselves
    # and called IMG_7872 REFILM over a 24-count exposure skew while that take
    # was in fact certifying 87.9% of codewords - the best yet measured. A
    # proxy that the decoder already handles is not a reason to refilm.
    #
    # Low header yield in particular is usually not a header problem: measured
    # on IMG_7872, frames whose header failed carried a median of 2/19
    # codewords, so the header was an honest gate on frames that were bad
    # anyway. Recovering every one of them would have been worth 1.19x.
    bad = []
    if mid > 0.18:
        bad.append(f"STRADDLE {100*mid:.0f}%: the camera is integrating across "
                   "two displayed frames. Re-lock AE/AF and hold still.")
    if p50 > (p5 + p95) / 2 + 35:
        bad.append("OVEREXPOSED: the median cell sits well above the midpoint "
                   "of the eye, so black cells are being flooded. Evening "
                   "light, ONE lamp, and do NOT use the exposure slider.")
    if cy < 0.6:
        bad.append(f"CODEWORD YIELD {100*cy:.0f}%: even the frames that "
                   "decode are marginal, so this is the optics, not the gate.")
    if est * 1.4 >= 200:
        bad = []
    if not bad and est * 1.4 >= 200:
        print("  VERDICT: GOOD — decode it in full:")
        print(f"    python3 poc/fast_decode.py {args.capture} /tmp/out.bin \\")
        print(f"        --grid {args.grid} --ecc {args.ecc} --subblock --soft --scan")
    else:
        print("  VERDICT: REFILM" if bad else
              "  VERDICT: decodable but short of 200 KB/s")
        for b in bad:
            print(f"    - {b}")


if __name__ == "__main__":
    main()
