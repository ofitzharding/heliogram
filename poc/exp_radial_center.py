#!/usr/bin/env python3
"""
exp_radial_center.py — the radial model is centred on the wrong point.

grid._apply_radial expands sample points about the IMAGE centre (w/2, h/2).
That is only the right centre if the lens's optical axis passes through the
middle of the sensor AND the code is symmetric about it. Neither holds when a
hand-held phone is aimed at a laptop screen: the code sits wherever the framing
put it, and a single k1 about the wrong point produces a residual that grows
in ONE direction instead of symmetrically.

Which is exactly the signature measured on IMG_7870. Per-codeword byte errors,
top of the frame to the bottom:

    cw 0   13.8      <- comfortably inside the 24-byte RS budget
    cw 4   30.6
    cw 7   36.8
    cw 15 138.8      <- 54% of bytes wrong, i.e. essentially unsampled

A soft optical gradient is symmetric about the best-focused band. A 10x
monotone ramp with one end fine and the other end at chance is a GEOMETRY
error, and a geometry error is recoverable - the cells are still there on the
sensor, they are just not where the model says they are.

This sweeps the radial centre as two extra free parameters. If the asymmetry is
geometric, some offset flattens the band profile and the mid-frame codewords,
which currently sit just above budget at 30-37 errors, drop under it.
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


def sample_with_center(img, L, H, cells, k1, cx_f, cy_f):
    """grid.sample_cells, but with the radial expansion about an arbitrary
    centre expressed as a fraction of image width/height."""
    centers = np.stack([cells[:, 1] + 0.5, cells[:, 0] + 0.5],
                       axis=1).astype(np.float32)
    pts = cv2.perspectiveTransform(centers[None], H)[0]
    h, w = img.shape[:2]
    cx, cy = cx_f * w, cy_f * h
    dx = (pts[:, 0] - cx) / w
    dy = (pts[:, 1] - cy) / w
    f = 1.0 + k1 * (dx * dx + dy * dy)
    px = np.clip((cx + (pts[:, 0] - cx) * f).round().astype(np.int32), 1, w - 2)
    py = np.clip((cy + (pts[:, 1] - cy) * f).round().astype(np.int32), 1, h - 2)
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
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--stride", type=int, default=13)
    args = ap.parse_args()

    grid.set_ecc(args.ecc); grid.set_header_len(28)
    grid.set_header_centered(True)
    gw, gh = (int(v) for v in args.grid.split("x"))
    L = grid.Layout(gw, gh)
    n_sub = grid.sub_count(L, grid.MODE_MONO)
    SUB = (255 - args.ecc) - 4
    budget = args.ecc // 2
    pc = L.payload_cells
    data = Path(args.payload).read_bytes()
    enc = fountain.Encoder(data, SUB)

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
        grid.set_radial(0.020)
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
    print(f"{len(frames)} frames from {args.capture}\n")
    if not frames:
        return

    nbits = n_sub * 255 * 8

    def score(k1, cxf, cyf):
        tot_ok = 0
        bands = np.zeros(n_sub)
        for _fn, img, H, tb in frames:
            lum = sample_with_center(img, L, H, pc, k1, cxf, cyf)
            bits, _c = grid._mono_decide(lum, L, pc)
            err = (bits[:nbits] != tb[:nbits]).reshape(n_sub * 255, 8).any(axis=1)
            cnt = err.reshape(n_sub, 255).sum(axis=1)
            bands += cnt
            tot_ok += int((cnt <= budget).sum())
        return tot_ok, bands / len(frames)

    base_ok, base_bands = score(0.020, 0.5, 0.5)
    tot = len(frames) * n_sub
    print(f"baseline (k1=0.020, centre = image centre): "
          f"{base_ok}/{tot} codewords inside budget")
    print("  bands: " + " ".join(f"{v:.0f}" for v in base_bands))

    best = (base_ok, 0.020, 0.5, 0.5)
    print(f"\n{'k1':>7s} {'cx':>6s} {'cy':>6s} {'in budget':>10s}")
    for k1 in (0.0, 0.010, 0.020, 0.030, 0.040):
        for cxf in (0.30, 0.40, 0.50, 0.60, 0.70):
            for cyf in (0.10, 0.30, 0.50, 0.70, 0.90):
                ok_, _b = score(k1, cxf, cyf)
                if ok_ > best[0]:
                    best = (ok_, k1, cxf, cyf)
                    print(f"{k1:+7.3f} {cxf:6.2f} {cyf:6.2f} {ok_:7d}/{tot}  *")
    ok_, bands = score(best[1], best[2], best[3])
    print(f"\nBEST k1={best[1]:+.3f} centre=({best[2]:.2f}, {best[3]:.2f}): "
          f"{ok_}/{tot} = {100*ok_/tot:.1f}% of codewords inside budget")
    print("  bands: " + " ".join(f"{v:.0f}" for v in bands))
    print(f"\nvs baseline {base_ok}/{tot} = {100*base_ok/tot:.1f}%   "
          f"-> {ok_/max(base_ok,1):.2f}x")


if __name__ == "__main__":
    main()
