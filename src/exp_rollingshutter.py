#!/usr/bin/env python3
"""
exp_rollingshutter.py — one homography cannot describe a rolling shutter.

A homography maps a plane to a plane under ONE camera pose. A CMOS rolling
shutter exposes the sensor row by row over most of a frame period, so a
hand-held camera has a DIFFERENT pose for every row of the image. The
projection is therefore not a homography at all; it is a stack of homographies
indexed by row, and fitting one to the whole frame leaves a residual that grows
with distance from whichever row the fit happened to favour.

Two measurements on IMG_7870 point straight at this:

  - per-codeword byte errors ramp monotonically down the frame (13.8 at the
    top, 30-37 through the middle), which is asymmetric and so cannot be a
    focus or illumination effect;
  - letting the radial-distortion CENTRE move - two extra free parameters that
    can only bend geometry, not brightness - lifts codewords-inside-budget from
    32.5% to 45.0%, and the optimum lands at cy=0.90, well outside the code.
    A "radial" correction centred outside the pattern is not modelling a lens.
    It is the cheapest available proxy for a row-dependent warp.

If that reading is right, then giving each row band its own sub-cell sampling
offset should collapse the residual, and the best offsets should vary
SYSTEMATICALLY down the frame rather than randomly. Random offsets would mean
this is just overfitting 16 extra parameters to noise; a monotone ramp means
the sensor was moving while it read the frame, and the amount it moved is
recoverable.

Offsets are in CELL units, applied before the homography, so they are exactly
"sample a fraction of a cell further left/up in this band".
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


def sample_offset(img, L, H, cells, k1, dx, dy):
    centers = np.stack([cells[:, 1] + 0.5 + dx, cells[:, 0] + 0.5 + dy],
                       axis=1).astype(np.float32)
    pts = cv2.perspectiveTransform(centers[None], H)[0]
    pts = grid._apply_radial(pts, img.shape)
    h, w = img.shape[:2]
    px = np.clip(pts[:, 0].round().astype(np.int32), 1, w - 2)
    py = np.clip(pts[:, 1].round().astype(np.int32), 1, h - 2)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    bl = cv2.boxFilter(g, cv2.CV_32F, (3, 3))
    return bl[py, px]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--payload", default=str(Path(__file__).parent.parent /
                                             "demo" / "payload.png"))
    ap.add_argument("--grid", default="252x140")
    ap.add_argument("--ecc", type=int, default=48)
    ap.add_argument("--start", type=int, default=1556)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--stride", type=int, default=17)
    ap.add_argument("--radial", type=float, default=0.020)
    ap.add_argument("--span", type=float, default=0.6)
    ap.add_argument("--step", type=float, default=0.1)
    args = ap.parse_args()

    grid.set_ecc(args.ecc); grid.set_header_len(28)
    grid.set_header_centered(True); grid.set_radial(args.radial)
    gw, gh = (int(v) for v in args.grid.split("x"))
    L = grid.Layout(gw, gh)
    n_sub = grid.sub_count(L, grid.MODE_MONO)
    SUB = (255 - args.ecc) - 4
    budget = args.ecc // 2
    pc = L.payload_cells
    data = Path(args.payload).read_bytes()
    enc = fountain.Encoder(data, SUB)
    nbits = n_sub * 255 * 8

    frames = []
    cap = cv2.VideoCapture(args.capture)
    n = args.start
    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    while len(frames) < args.n:
        ok, img = cap.read()
        if not ok:
            break
        fn = n
        n += 1
        if (fn - args.start) % args.stride:
            continue
        H = grid.locate(img, L)
        if H is None:
            continue
        hd, _s, _t = grid.sample_frame(img, L, H)
        if hd is None or int(hd["k"]) != enc.k:
            continue
        seq = int(hd["seq"])
        parts = []
        for j in range(n_sub):
            b = enc.block(seq * n_sub + j)
            b = b + b"\x00" * (SUB - len(b))
            parts.append(struct.pack("<I", zlib.crc32(b) & 0xFFFFFFFF) + b)
        hdr = grid.pack_header(seq, enc.k, SUB, len(data), grid.MODE_MONO, 0, 0)
        tr = grid.render_frame(L, hdr, b"".join(parts), grid.MODE_MONO, cell_px=1)
        xt = (cv2.cvtColor(tr, cv2.COLOR_BGR2GRAY) > 127).astype(np.uint8)
        frames.append((fn, img, H, xt[pc[:, 0], pc[:, 1]]))
    cap.release()
    print(f"{len(frames)} frames, {n_sub} bands, "
          f"offsets swept +-{args.span} cells in {args.step} steps\n")
    if not frames:
        return

    offs = np.round(np.arange(-args.span, args.span + 1e-9, args.step), 3)
    # errors[frame][dy][dx][band]
    band_err = np.zeros((len(frames), len(offs), len(offs), n_sub))
    for fi, (_fn, img, H, tb) in enumerate(frames):
        for iy, dy in enumerate(offs):
            for ix, dx in enumerate(offs):
                lum = sample_offset(img, L, H, pc, args.radial, dx, dy)
                bits, _c = grid._mono_decide(lum, L, pc)
                err = (bits[:nbits] != tb[:nbits]).reshape(n_sub * 255,
                                                           8).any(axis=1)
                band_err[fi, iy, ix] = err.reshape(n_sub, 255).sum(axis=1)

    zero = int(np.argmin(np.abs(offs)))
    base = band_err[:, zero, zero, :]
    base_ok = int((base <= budget).sum())
    tot = len(frames) * n_sub
    print(f"baseline (one offset for the whole frame, dx=dy=0): "
          f"{base_ok}/{tot} codewords inside budget")
    print("  band errors: " + " ".join(f"{v:.0f}" for v in base.mean(axis=0)))

    # ---- one GLOBAL best offset (a fairer baseline than dx=dy=0)
    g_tot = band_err.sum(axis=3).mean(axis=0)
    giy, gix = np.unravel_index(np.argmin(g_tot), g_tot.shape)
    gb = band_err[:, giy, gix, :]
    g_ok = int((gb <= budget).sum())
    print(f"\nbest GLOBAL offset dx={offs[gix]:+.2f} dy={offs[giy]:+.2f}: "
          f"{g_ok}/{tot} inside budget")
    print("  band errors: " + " ".join(f"{v:.0f}" for v in gb.mean(axis=0)))

    # ---- per-band best offset, chosen per band but SHARED across frames
    print(f"\n{'band':>5s} {'dx':>6s} {'dy':>6s} {'err@global':>11s} "
          f"{'err@band':>9s}")
    per_ok = 0
    best_dx, best_dy = [], []
    for b in range(n_sub):
        e = band_err[:, :, :, b].mean(axis=0)
        iy, ix = np.unravel_index(np.argmin(e), e.shape)
        best_dx.append(offs[ix]); best_dy.append(offs[iy])
        per_ok += int((band_err[:, iy, ix, b] <= budget).sum())
        print(f"{b:5d} {offs[ix]:+6.2f} {offs[iy]:+6.2f} "
              f"{gb[:, b].mean():11.1f} {e[iy, ix]:9.1f}")
    print(f"\nper-band offsets: {per_ok}/{tot} inside budget "
          f"({100*per_ok/tot:.1f}%)  vs global {100*g_ok/tot:.1f}%  "
          f"-> {per_ok/max(g_ok,1):.2f}x")

    dy = np.array(best_dy); dx = np.array(best_dx)
    r = np.arange(n_sub)
    print(f"\nIs the per-band offset SYSTEMATIC or noise?")
    for nm, v in (("dy", dy), ("dx", dx)):
        c = float(np.corrcoef(r, v)[0, 1]) if v.std() > 1e-9 else 0.0
        slope = float(np.polyfit(r, v, 1)[0]) if v.std() > 1e-9 else 0.0
        print(f"  {nm}: corr with band index {c:+.3f}, "
              f"slope {slope:+.4f} cells/band, total drift "
              f"{slope*(n_sub-1):+.3f} cells top-to-bottom")


if __name__ == "__main__":
    main()
