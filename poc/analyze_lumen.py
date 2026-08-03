#!/usr/bin/env python3
"""
analyze_lumen.py — which transmit peak the sensor can actually take.

Reports one row per (alphabet, parity, peak), attributing each frame by the
mode/ecc/peak in its own header rather than by frame offset.

Read it in this order:

  1. THE MONO ROW IS THE CONTROL. It is transmitted at full peak in every loop
     and this rig gives it ~90% yield on a good take. If mono is far below
     that, the take is bad and every gray4 row is meaningless - which is
     exactly what happened on IMG_7879 (mono 23.6%, gray4 0%, and gray4
     therefore untested rather than refuted).
  2. Then read gray4 down the peak column. The eye statistics say why a row
     failed: p5 well above 0 is black lifted by bloom, p50 far above the
     midpoint of p5..p95 is a saturating transfer curve, and either one merges
     gray4's two middle levels first.
  3. The figure of merit is KB/s, not yield. gray4 carries twice the bits, so
     it only has to be half as good as mono to win.
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
           ("gray4", grid.MODE_GRAY4, 64)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--grid", default="252x163")
    ap.add_argument("--frames", type=int, default=180)
    ap.add_argument("--lo", type=float, default=0.15)
    ap.add_argument("--hi", type=float, default=0.97)
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

    tally = {}
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, img = cap.read()
        if not ok:
            continue
        grid.set_ecc(48)
        H = grid.locate(img, grid.Layout(gw, gh))
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
                if hd is None or int(hd["mode"]) != mode:
                    continue
                if int(hd.get("ecc", ecc)) != ecc:
                    continue
                peak = int(hd.get("zone_w", 255)) or 255
                y = grid.sample_cells(img, Lc, H, allc).mean(axis=1).reshape(
                    gh, gw).astype(np.float32)
                if mode == grid.MODE_MONO:
                    n = fd.quick_count(y)
                else:
                    samp = np.repeat(
                        y[Lc.payload_cells[:, 0],
                          Lc.payload_cells[:, 1]][:, None], 3, axis=1)
                    raw, bc = grid.raw_bits_and_conf(hd, samp, Lc)
                    bits = np.unpackbits(
                        np.frombuffer(raw, np.uint8)).astype(np.uint8)
                    n = len(fd.certify(bits, bc)[0])
                pv = y[Lc.payload_cells[:, 0], Lc.payload_cells[:, 1]]
                if best is None or n > best[0]:
                    best = (n, name, ecc, n_sub, peak, pv)
                break
        if best is None:
            continue
        n, name, ecc, n_sub, peak, pv = best
        t = tally.setdefault((name, ecc, peak), [0, 0, 0, []])
        t[0] += n; t[1] += n_sub; t[2] += 1
        t[3].append((np.percentile(pv, 5), np.percentile(pv, 50),
                     np.percentile(pv, 95)))
    cap.release()
    if not tally:
        print("no frame produced a header at any configuration")
        return

    print(f"{args.capture}, {args.grid}, {fps:.0f} fps\n")
    print(f"{'config':>14s} {'peak':>5s} {'frames':>7s} {'yield':>7s} "
          f"{'ceiling':>9s} {'KB/s':>8s}   {'p5':>4s} {'p50':>4s} {'p95':>4s} "
          f"{'skew':>5s}")
    rows = sorted(tally.items(), key=lambda kv: (kv[0][0], -kv[0][2]))
    for (name, ecc, peak), (got, poss, nf, phot) in rows:
        grid.set_ecc(ecc)
        Lc = grid.Layout(gw, gh)
        mode = grid.MODE_MONO if name == "mono" else grid.MODE_GRAY4
        ns = grid.sub_count(Lc, mode); sub = (255 - ecc) - 4
        ceil = ns * sub * fps / 1024
        y_ = got / max(poss, 1)
        p = np.median(np.array(phot), axis=0)
        skew = p[1] - (p[0] + p[2]) / 2.0
        print(f"{name+'/'+str(ecc):>14s} {peak:5d} {nf:7d} {100*y_:6.1f}% "
              f"{ceil:8.1f}K {ceil*y_:7.1f}   {p[0]:4.0f} {p[1]:4.0f} "
              f"{p[2]:4.0f} {skew:+5.0f}")

    mono = [(v, k) for k, v in tally.items() if k[0] == "mono"]
    if mono:
        (got, poss, _nf, _ph), _k = mono[0]
        my = got / max(poss, 1)
        print(f"\nCONTROL: mono {100*my:.1f}% yield.", end=" ")
        if my < 0.60:
            print("The take is BAD - this rig gives mono ~90% when the\n"
                  "exposure is right, so no gray4 row below is a verdict on "
                  "gray4. Refilm darker/dimmer before concluding anything.")
        else:
            print("Take is sound, so the gray4 rows are real measurements.")
    print("\n'skew' is p50 minus the midpoint of p5..p95: positive means a\n"
          "saturating transfer curve, which merges gray4's middle levels first.")


if __name__ == "__main__":
    main()
