#!/usr/bin/env python3
"""
k1_objective.py — find a per-frame radial estimate that tracks CODEWORDS.

Three facts, all measured on IMG_7867:

  1. k1 is worth a lot. Frame 2800 certifies 0/16 codewords at k1=0.000 and
     16/16 at k1=+0.018.
  2. The optimum MOVES between frames of the same take (2700 peaks at +0.015,
     2800 at +0.018) because the phone is hand-held and the code's position in
     the lens field changes.
  3. Structure-cell agreement, the obvious cheap objective, peaks in the WRONG
     PLACE: 98.3% at +0.007 on frame 2800, where codewords are 7/16. Finders,
     ring and separators all sit at the grid border, so they measure edge
     geometry rather than the payload interior that actually carries the data.

So a cheap objective is needed that peaks where the codewords do, since running
Reed-Solomon over a k1 sweep for every frame of a 4000-frame clip is not
affordable. This scores candidates against the codeword count directly.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from codec import grid
from softdec import FrameDecoder


def objectives(y, L, pc):
    """Cheap per-frame sharpness/eye measures, all higher-is-better."""
    lum = y[pc[:, 0], pc[:, 1]]
    th, sd = grid.local_levels(lum, L, pc, 15)
    z = (lum - th) / np.maximum(sd, 1e-3)
    out = {}
    # 1. eye opening: how far the average cell sits from its local threshold
    out["eye"] = float(np.abs(z).mean())
    # 2. anti-straddle: fraction NOT in the mid-band of the local eye
    out["clear"] = float((np.abs(z) > 0.5).mean())
    # 3. global Otsu between-class variance (what estimate_radial uses)
    u8 = np.clip(lum, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    a, b = lum[lum <= t], lum[lum > t]
    out["otsu"] = (float(len(a) * len(b) / len(lum) ** 2 *
                         (a.mean() - b.mean()) ** 2) if len(a) > 9 and len(b) > 9
                   else 0.0)
    # 4. bimodality of the LOCALLY normalised cells - the same statistic as
    #    otsu but computed after the illumination field is removed, so a
    #    lighting gradient cannot masquerade as a closed eye
    out["locvar"] = float((z ** 2).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--grid", default="252x140")
    ap.add_argument("--frames", default="")
    ap.add_argument("--scan-lo", type=int, default=2000)
    ap.add_argument("--scan-hi", type=int, default=4200)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--ecc", type=int, default=48)
    args = ap.parse_args()
    grid.set_ecc(args.ecc); grid.set_header_len(28)
    grid.set_header_centered(True)
    gw, gh = (int(v) for v in args.grid.split("x"))
    L = grid.Layout(gw, gh)
    n_sub = grid.sub_count(L, grid.MODE_MONO)
    pc = L.payload_cells
    allc = np.argwhere(np.ones((gh, gw), bool))
    fd = FrameDecoder(L, args.ecc, n_sub, erase=True, prml=False)

    if args.frames:
        fis = [int(v) for v in args.frames.split(",")]
    else:
        fis = np.linspace(args.scan_lo, args.scan_hi, args.n).astype(int).tolist()
    ks = np.arange(0.0, 0.036, 0.0025)
    names = ["eye", "clear", "otsu", "locvar"]
    hits = {n: 0 for n in names}
    loss = {n: [] for n in names}
    used = 0
    cap = cv2.VideoCapture(args.capture)
    print(f"{'frame':>6s} {'best k1':>8s} {'best cw':>8s}   " +
          "  ".join(f"{n:>6s} k1/cw" for n in names))
    for fi in fis:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, img = cap.read()
        if not ok:
            continue
        H = grid.locate(img, L)
        if H is None:
            continue
        cw, obj = [], {n: [] for n in names}
        for k1 in ks:
            grid.set_radial(float(k1))
            y = grid.sample_cells(img, L, H, allc).mean(axis=1).reshape(
                gh, gw).astype(np.float32)
            hd, _s, _t = grid.sample_frame(img, L, H)
            hdr = hd if hd else dict(seq=0, k=1, block_size=203, file_size=10)
            cw.append(len(fd.decode(y, hdr)))
            o = objectives(y, L, pc)
            for n in names:
                obj[n].append(o[n])
        cw = np.array(cw)
        if cw.max() == 0:
            continue
        used += 1
        bi = int(cw.argmax())
        line = f"{fi:6d} {ks[bi]:+8.4f} {cw[bi]:5d}/{n_sub:<2d}  "
        for n in names:
            oi = int(np.argmax(obj[n]))
            hits[n] += int(cw[oi] == cw[bi])
            loss[n].append(cw[bi] - cw[oi])
            line += f"  {ks[oi]:+.4f}/{cw[oi]:<2d}"
        print(line)
    cap.release()
    print(f"\nover {used} frames that decoded anything:")
    for n in names:
        print(f"  {n:>7s}: picks the optimum {hits[n]}/{used} times, "
              f"mean codewords lost {np.mean(loss[n]):.2f}")


if __name__ == "__main__":
    main()
