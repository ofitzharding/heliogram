#!/usr/bin/env python3
"""
exp_ramp.py — density-ramp preamble: carry the receiver past its own wall.

THE PROBLEM (Findings §14, stated precisely there and unsolved since)
---------------------------------------------------------------------
Tile-PRML decodes real captures at 8.2 camera-px/cell (2.3% BER) where plain
thresholding gives 11.8% and certifies nothing. That is roughly 2.9x the cell
count of the density anyone can read conventionally. But it needs certified
labels to fit its kernels, and at that density NOTHING certifies from a cold
threshold. No first rung, no ladder.

THE IDEA
--------
Put the first rung in the transmission. Open at a density every receiver
decodes, certify those frames through the ordinary FEC path, learn the
channel from them, then step the density up past what conventional decoding
can reach and keep decoding on the learned channel.

The one real technical wrinkle: kernels are fitted in CELL units, and a cell
changes size when the grid changes. The camera PSF does not — it is fixed in
CAMERA pixels. So the preamble measures sigma in camera pixels, and the dense
phase rescales it by the px/cell ratio. That conversion is the whole trick,
and it is what this file tests.

Validated in simulation first, because a filming session cannot distinguish
"the idea is wrong" from "the take was bad", and this project has already
lost six takes to that ambiguity.
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
import exp_tile_prml as T
import exp_turbo_frame as TB

R = 2


def render(layout, seq, enc, n_sub, sub_size, file_size, cell_px):
    parts = []
    for j in range(n_sub):
        b = enc.block(seq * n_sub + j)
        b = b + b"\x00" * (sub_size - len(b))
        parts.append(struct.pack("<I", zlib.crc32(b) & 0xFFFFFFFF) + b)
    hdr = grid.pack_header(seq, enc.k,
                           layout.payload_capacity_bytes(grid.MODE_MONO) - 4,
                           file_size, grid.MODE_MONO, 0, 0)
    return grid.render_frame(layout, hdr, b"".join(parts), grid.MODE_MONO,
                             cell_px=cell_px)


def camera(img, out_w, sigma_px, noise, gain=1.0, bias=0.0):
    """Crude but honest camera: resample to the sensor, blur by a PSF fixed in
    CAMERA pixels, add noise. The whole point of the experiment is that sigma
    is constant here while cells shrink."""
    h = int(round(img.shape[0] * out_w / img.shape[1]))
    s = cv2.resize(img, (out_w, h), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = int(sigma_px * 6) | 1
    g = cv2.GaussianBlur(g, (k, k), sigma_px)
    g = g * gain + bias
    if noise:
        g = g + np.random.RandomState(0).normal(0, noise, g.shape)
    return np.clip(g, 0, 255).astype(np.uint8)


def sample_grid(gray, layout):
    """Axis-aligned sampling: the simulated capture has no perspective, so the
    grid maps directly and geometry is removed as a confound."""
    H, W = gray.shape
    cw, ch = W / layout.gw, H / layout.gh
    xs = ((np.arange(layout.gw) + 0.5) * cw).astype(int)
    ys = ((np.arange(layout.gh) + 0.5) * ch).astype(int)
    return gray[np.ix_(ys, xs)].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparse", default="252x140")
    ap.add_argument("--dense", default="400x222")
    ap.add_argument("--sensor", type=int, default=3500,
                    help="camera px across the code (measured ~3500 real)")
    ap.add_argument("--sigma", type=float, default=3.6,
                    help="camera PSF sigma in CAMERA px (measured 3.6)")
    ap.add_argument("--noise", type=float, default=3.0)
    ap.add_argument("--ecc", type=int, default=48)
    args = ap.parse_args()

    grid.set_ecc(args.ecc); grid.set_header_len(28)
    grid.set_header_centered(True); grid.set_radial(0.0)
    data = (Path(__file__).parent.parent / "demo" / "payload.png").read_bytes()
    sub_size = 255 - args.ecc - 4

    def build(spec, cell_px):
        gw, gh = (int(v) for v in spec.split("x"))
        L = grid.Layout(gw, gh)
        n_sub = grid.sub_count(L, grid.MODE_MONO)
        enc = fountain.Encoder(data, sub_size)
        return L, n_sub, enc, gw, gh, cell_px

    Ls, ns, es, gws, ghs, cps = build(args.sparse, 12)
    Ld, nd, ed, gwd, ghd, cpd = build(args.dense, 8)
    print(f"sparse {args.sparse}: {args.sensor/gws:.1f} camera px/cell, "
          f"{ns} codewords/frame")
    print(f"dense  {args.dense}: {args.sensor/gwd:.1f} camera px/cell, "
          f"{nd} codewords/frame  "
          f"({nd/ns:.2f}x the data per frame)\n")

    subS = TB.SubBlock(Ls, args.ecc, Ls.payload_capacity_bytes(grid.MODE_MONO))
    subD = TB.SubBlock(Ld, args.ecc, Ld.payload_capacity_bytes(grid.MODE_MONO))

    # ---------- PHASE 1: preamble, ordinary decoding, harvest certified labels
    knownS = Ls.is_finder | Ls.is_sep | Ls.is_ring | Ls.is_header
    truthS, obsS = [], []
    for seq in range(3):
        img = render(Ls, seq, es, ns, sub_size, len(data), cps)
        cap = camera(img, args.sensor, args.sigma, args.noise)
        y = sample_grid(cap, Ls)
        t = (cv2.cvtColor(cv2.resize(img, (Ls.gw, Ls.gh),
             interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY) > 127).astype(np.float32)
        truthS.append(t); obsS.append(y)
    th = np.median(obsS[0])
    x0 = (obsS[0] > th).astype(np.float32)
    nS = subS.try_certify(x0[subS.cells[:, 0], subS.cells[:, 1]])[2]
    berS = 100 * (x0 != truthS[0]).mean()
    print(f"PHASE 1 preamble : BER {berS:.2f}%, certified {nS}/{ns} codewords"
          f"  {'-> labels available' if nS > 0 else '-> NO LABELS, ramp cannot start'}")

    # measure the PSF in CAMERA pixels from the certified preamble frame
    best = None
    for sg_cells in np.arange(0.10, 1.20, 0.02):
        k = T.gaussian_kernel(sg_cells) if hasattr(T, 'gaussian_kernel') else None
        ax = np.arange(-R, R + 1, dtype=np.float32)
        g1 = np.exp(-0.5 * (ax / max(sg_cells, 1e-3)) ** 2)
        K = np.outer(g1, g1); K /= K.sum()
        pred = cv2.filter2D(truthS[0], -1, K, borderType=cv2.BORDER_REPLICATE)
        A = np.stack([pred.ravel(), np.ones(pred.size)], 1)
        co, *_ = np.linalg.lstsq(A, obsS[0].ravel(), rcond=None)
        r = ((A @ co - obsS[0].ravel()) ** 2).mean()
        if best is None or r < best[0]:
            best = (r, sg_cells, co)
    _, sg_cells, co = best
    px_per_cell_S = args.sensor / gws
    sigma_camera_px = sg_cells * px_per_cell_S
    print(f"           learned PSF: {sg_cells:.2f} cells = "
          f"{sigma_camera_px:.2f} CAMERA px (true {args.sigma})")

    # ---------- PHASE 2: dense, conventional vs ramp-carried
    knownD = Ld.is_finder | Ld.is_sep | Ld.is_ring | Ld.is_header
    img = render(Ld, 0, ed, nd, sub_size, len(data), cpd)
    cap = camera(img, args.sensor, args.sigma, args.noise)
    yD = sample_grid(cap, Ld)
    tD = (cv2.cvtColor(cv2.resize(img, (Ld.gw, Ld.gh),
          interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY) > 127).astype(np.float32)
    thD = np.median(yD)
    x0D = (yD > thD).astype(np.float32)
    nD0 = subD.try_certify(x0D[subD.cells[:, 0], subD.cells[:, 1]])[2]
    print(f"\nPHASE 2 dense, conventional threshold: BER "
          f"{100*(x0D != tD).mean():.2f}%, certified {nD0}/{nd}")

    # rescale the PSF into the DENSE grid's cell units - the crux
    px_per_cell_D = args.sensor / gwd
    sg_dense = sigma_camera_px / px_per_cell_D
    ax = np.arange(-R, R + 1, dtype=np.float32)
    g1 = np.exp(-0.5 * (ax / max(sg_dense, 1e-3)) ** 2)
    K = np.outer(g1, g1); K /= K.sum()
    tap = np.zeros((Ld.gh, Ld.gw, 25), np.float32)
    tap[:, :] = (K.ravel() * co[0])[None, None, :]
    bias = np.full((Ld.gh, Ld.gw), co[1], np.float32)
    print(f"           PSF rescaled to dense grid: {sg_dense:.2f} cells")

    xD = T.prml_tiles(yD, tap, bias, knownD, tD * knownD, x0D, sweeps=3)
    nD1 = subD.try_certify(xD[subD.cells[:, 0], subD.cells[:, 1]])[2]
    print(f"PHASE 2 dense, ramp-carried PRML     : BER "
          f"{100*(xD != tD).mean():.2f}%, certified {nD1}/{nd}")
    print(f"\ndata per frame: sparse {ns*sub_size} B, dense {nd*sub_size} B")
    print(f"if the ramp holds, throughput ratio = {nd/ns:.2f}x")


if __name__ == "__main__":
    main()
