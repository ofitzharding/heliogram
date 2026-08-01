#!/usr/bin/env python3
"""
isi_cancel.py — 2D decision-feedback interference cancellation.

THE IDEA
--------
Camera blur mixes each cell with its neighbours. Every screen-camera and
barcode system in existence treats that as damage and avoids it by keeping
cells large — which is exactly what caps throughput. Measured on our own
footage: at 13.1 camera-px/cell each cell keeps ~86% of its own signal and
decodes at 0.4% BER; at 7.45 px/cell it keeps only ~49% and collapses to 18%.

Radio does not avoid this. Radio CANCELS it. Intersymbol interference from a
known channel is removable: decide the confident symbols, subtract their
contribution from their neighbours, and re-decide the rest. That is
decision-feedback equalization, standard since the 1970s in every modem, and
absent from 2D barcode decoding.

The 2D version:
    y = x (*) K + noise            observed = true cells convolved with
                                   the cell-domain blur kernel
    x_hat = threshold(y)           initial hard decisions
    repeat:
        interference_i = sum_{j != i} K_j * x_hat_{i+j}
        clean_i        = (y_i - interference_i) / K_0
        x_hat          = threshold(clean), keeping the most confident fixed

Why it should work here: the interference is DETERMINISTIC given neighbours,
not random noise. At 7.45 px/cell about half of each cell's reading is
neighbour leakage — an enormous, perfectly structured error signal that hard
thresholding simply throws away.
"""
import argparse
import struct
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from codec import grid


def cells_to_image(vals: np.ndarray, layout: grid.Layout, cells: np.ndarray):
    """Scatter per-cell values onto a (gh, gw) grid image."""
    G = np.zeros((layout.gh, layout.gw), np.float32)
    G[cells[:, 0], cells[:, 1]] = vals
    return G


def estimate_cell_kernel(y_img: np.ndarray, x_img: np.ndarray, radius: int = 2,
                         reg: float = 1e-3) -> np.ndarray:
    """Least-squares fit of the cell-domain blur kernel from a KNOWN frame.

    At decode time the fountain layer supplies known frames for free: any
    CRC-verified block is a perfectly known transmitted pattern. So this is
    trainable in-band, with no calibration step.
    """
    n = 2 * radius + 1
    H, W = x_img.shape
    cols = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            cols.append(np.roll(np.roll(x_img, dr, axis=0), dc, axis=1).ravel())
    A = np.stack(cols, axis=1)
    b = y_img.ravel()
    ATA = A.T @ A + reg * np.eye(A.shape[1]) * A.shape[0]
    ATb = A.T @ b
    k = np.linalg.solve(ATA, ATb)
    return k.reshape(n, n)


def cancel(y_img: np.ndarray, K: np.ndarray, mask: np.ndarray,
           iters: int = 8, keep_frac: float = 0.5):
    """Iterative 2D interference cancellation.

    `mask` marks payload cells (others are known structure and left alone).
    Each round: predict every cell's neighbour interference from the current
    decisions, subtract it, re-decide. Confident cells are re-decided too, but
    they rarely flip, so the fixed points act as anchors that pull the
    uncertain ones in.
    """
    r = K.shape[0] // 2
    K0 = K[r, r]
    Kn = K.copy()
    Kn[r, r] = 0.0                       # neighbours only

    lo, hi = np.percentile(y_img[mask], [5, 95])
    x = ((y_img - (lo + hi) / 2) > 0).astype(np.float32)   # 0/1 decisions
    x_lo, x_hi = 0.0, 1.0

    for _ in range(iters):
        # scale decisions into observation units, then convolve with neighbours
        drive = lo + x * (hi - lo)
        interference = cv2.filter2D(drive, -1, Kn, borderType=cv2.BORDER_REPLICATE)
        clean = (y_img - interference) / max(K0, 1e-3)
        t = (np.percentile(clean[mask], 5) + np.percentile(clean[mask], 95)) / 2
        x_new = (clean > t).astype(np.float32)
        if np.array_equal(x_new, x):
            break
        x = x_new
    return x, clean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--grid", default="466x259")
    ap.add_argument("--payload", default="../demo/payload_big.png")
    ap.add_argument("--ecc", type=int, default=48)
    ap.add_argument("--radial", type=float, default=0.020)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--radius", type=int, default=2)
    args = ap.parse_args()

    from codec import fountain
    from reedsolo import RSCodec
    grid.set_ecc(args.ecc)
    grid.set_header_len(28)
    grid.set_radial(args.radial)
    gw, gh = (int(v) for v in args.grid.split("x"))
    L = grid.Layout(gw, gh)
    cells = L.payload_cells

    data = Path(args.payload).read_bytes()
    bs = L.payload_capacity_bytes(grid.MODE_MONO) - 4
    enc = fountain.Encoder(data, bs)

    def truth_bits(seq):
        b = enc.block(seq)
        b = b + b"\x00" * (bs - len(b))
        p = struct.pack("<I", zlib.crc32(b) & 0xFFFFFFFF) + b
        return np.unpackbits(np.frombuffer(bytes(RSCodec(args.ecc).encode(p)),
                                           np.uint8))

    cap = cv2.VideoCapture(args.capture)
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{args.capture}: {N} frames, grid {gw}x{gh}, kernel radius {args.radius}")
    print(f"{'frame':>6s} {'plain BER':>10s} {'ISI-cancelled':>14s} {'K0':>6s}")

    befores, afters = [], []
    for fi in np.linspace(N * 0.25, N * 0.8, args.frames).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, img = cap.read()
        if not ok:
            continue
        H = grid.locate(img, L)
        if H is None:
            continue
        hdr, samples, _ = grid.sample_frame(img, L, H)
        if hdr is None or samples is None:
            continue
        tb = truth_bits(hdr["seq"])
        m = min(len(tb), len(samples), len(cells))
        lum = samples[:m].mean(axis=1)
        true = tb[:m]
        C = cells[:m]

        y_img = cells_to_image(lum, L, C)
        x_img = cells_to_image(true.astype(np.float32), L, C)
        mask = np.zeros((L.gh, L.gw), bool)
        mask[C[:, 0], C[:, 1]] = True

        # plain
        th, _ = cv2.threshold(np.clip(lum, 0, 255).astype(np.uint8), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        b0 = float(((lum > th).astype(np.uint8) != true).mean())

        # kernel fitted from this frame's known content (stands in for the
        # fountain-supplied known frames the real receiver has)
        K = estimate_cell_kernel(y_img, x_img, radius=args.radius)
        x_hat, _ = cancel(y_img, K, mask)
        got = x_hat[C[:, 0], C[:, 1]].astype(np.uint8)
        b1 = float((got != true).mean())

        r = K.shape[0] // 2
        befores.append(b0); afters.append(b1)
        print(f"{fi:6d} {b0*100:9.2f}% {b1*100:13.2f}% {K[r,r]:6.2f}")

    if befores:
        print(f"\nmedian: {np.median(befores)*100:.2f}%  ->  "
              f"{np.median(afters)*100:.2f}%   (RS limit 1.23%)")
        print(f"frames under RS limit: before {sum(b<0.0123 for b in befores)}"
              f"/{len(befores)}, after {sum(b<0.0123 for b in afters)}/{len(afters)}")


if __name__ == "__main__":
    main()
