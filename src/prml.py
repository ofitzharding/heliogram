#!/usr/bin/env python3
"""
prml.py — partial-response detection with row/column Viterbi, the magnetic-
recording move applied to the screen-camera grid.

WHY THIS EXISTS
---------------
isi_cancel.py (2D decision-feedback cancellation) collapsed 11% -> 48% BER.
Post-mortem found implementation faults, not physics:
  1. The kernel was fitted mapping x in {0,1} to observation units, but the
     cancellation loop re-scaled decisions into observation units AGAIN
     (drive = lo + x*(hi-lo)) before convolving — interference was applied
     at roughly double its true magnitude.
  2. The least-squares fit had no bias column, so the black-level offset
     (mean luminance) leaked into every kernel tap.
  3. cv2.filter2D computes correlation while the fit used np.roll(x, +dr)
     columns (a convolution convention) — only harmless if K is symmetric.

This file fixes all three and then goes further than per-cell DFE: hard
drives do not cancel ISI cell-by-cell, they run maximum-likelihood SEQUENCE
detection (PRML) against the known response. Here:

  y[r,c] = c0 + sum_{dr,dc} K[dr+R, dc+R] * x[r+dr, c+dc] + noise
           (correlation convention, matches cv2.filter2D directly)

  - fit K (+bias c0) by least squares from a KNOWN frame. In the real
    receiver the fountain layer supplies known frames for free (any
    CRC-certified block); here the frame's own truth stands in, same as
    isi_cancel.py did.
  - detect by iterated dimension-split Viterbi: subtract the current
    estimate's out-of-row interference, run an exact 16-state Viterbi along
    each row against the center-row taps, then the same down each column,
    and repeat. Structure cells (finders/ring/separators/header) are known
    a priori and pinned; only payload cells are free.

Usage:
  python3 prml.py FRAME.png [FRAME2.png ...] --grid 466x259
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

INF = 1e18


# ---------------------------------------------------------------- channel fit

def fit_kernel(y: np.ndarray, x: np.ndarray, radius: int = 2, reg: float = 1e-3):
    """Least-squares fit of (K, c0): y ~ c0 + corr(x, K), on interior cells.

    Correlation convention: column (dr,dc) is x[r+dr, c+dc], matching
    cv2.filter2D. A bias column absorbs the black level instead of letting
    it leak into the taps.
    """
    n = 2 * radius + 1
    gh, gw = x.shape
    cols = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            cols.append(np.roll(np.roll(x, -dr, axis=0), -dc, axis=1).ravel())
    cols.append(np.ones(x.size, np.float32))
    A = np.stack(cols, axis=1)

    interior = np.zeros((gh, gw), bool)
    interior[radius:gh - radius, radius:gw - radius] = True
    sel = interior.ravel()
    A, b = A[sel], y.ravel()[sel]
    ATA = A.T @ A + reg * np.eye(A.shape[1]) * A.shape[0]
    k = np.linalg.solve(ATA, A.T @ b)
    return k[:-1].reshape(n, n), float(k[-1])


# ---------------------------------------------------------------- detectors

def dfe(y: np.ndarray, K: np.ndarray, c0: float, known: np.ndarray,
        x_true: np.ndarray, x0: np.ndarray, iters: int = 8):
    """Corrected 2D decision feedback: decisions stay in {0,1} units because
    the kernel already maps {0,1} to observation units. Known cells pinned."""
    R = K.shape[0] // 2
    K0 = K[R, R]
    Kn = K.copy()
    Kn[R, R] = 0.0
    x = x0.copy()
    x[known] = x_true[known]
    for _ in range(iters):
        interference = cv2.filter2D(x, -1, Kn, borderType=cv2.BORDER_REPLICATE)
        clean = (y - c0 - interference) / max(K0, 1e-3)
        x_new = (clean > 0.5).astype(np.float32)
        x_new[known] = x_true[known]
        if np.array_equal(x_new, x):
            break
        x = x_new
    return x


def viterbi_lines(z: np.ndarray, h: np.ndarray, allowed: np.ndarray):
    """Exact ML sequence detection along axis 1, batched over axis 0.

    Model: z[i, c] = sum_{j=0..4} h[j] * u[i, c+j-2] + noise, u in {0,1}.
    allowed[i, c, v] — whether symbol v is permitted at (i, c); known cells
    have exactly one permitted value. Two pad cells on each side are pinned
    to the edge value of the initial estimate via `allowed` padding by the
    caller? No: pads here are pinned dark (0), the grid is surrounded by
    the black letterbox in every capture of this pipeline.
    """
    nr, W = z.shape
    taps = len(h)                      # 5
    mem = taps - 1                     # 4 -> 16 states
    Wp = W + 2 * (taps // 2)

    allowedp = np.ones((nr, Wp, 2), bool)
    allowedp[:, 2:-2] = allowed
    allowedp[:, :2, 1] = False         # pads pinned to 0
    allowedp[:, -2:, 1] = False

    # pred[s*2+b0] for state s=(b4,b3,b2,b1) oldest-first, new symbol b0
    idx = np.arange(32)
    bits = ((idx[:, None] >> np.arange(4, -1, -1)[None, :]) & 1).astype(np.float32)
    predvec = bits @ h.astype(np.float32)

    cost = np.zeros((nr, 16), np.float32)
    bt = np.zeros((Wp, nr, 16), np.uint8)
    # precompute per-new-state predecessor pairs
    preds = [(sp >> 1, (sp >> 1) | 8) for sp in range(16)]

    for cp in range(Wp):
        if cp >= mem:
            pm = (z[:, cp - mem][:, None] - predvec[None, :]) ** 2
            bm = pm.reshape(nr, 16, 2)
        else:
            bm = np.zeros((nr, 16, 2), np.float32)
        gate = np.where(allowedp[:, cp, :], 0.0, INF).astype(np.float32)
        tot = cost[:, :, None] + bm + gate[:, None, :]

        newcost = np.empty((nr, 16), np.float32)
        btc = np.empty((nr, 16), np.uint8)
        for sp in range(16):
            b0 = sp & 1
            pa, pb = preds[sp]
            ca, cb = tot[:, pa, b0], tot[:, pb, b0]
            pick = cb < ca
            newcost[:, sp] = np.where(pick, cb, ca)
            btc[:, sp] = np.where(pick, pb, pa)
        cost = newcost - newcost.min(axis=1, keepdims=True)
        bt[cp] = btc

    s = cost.argmin(axis=1)
    u = np.zeros((nr, Wp), np.uint8)
    rows = np.arange(nr)
    for cp in range(Wp - 1, -1, -1):
        u[:, cp] = s & 1
        s = bt[cp][rows, s]
    return u[:, 2:-2].astype(np.float32)


def prml(y: np.ndarray, K: np.ndarray, c0: float, known: np.ndarray,
         x_true: np.ndarray, x0: np.ndarray, sweeps: int = 3):
    """Iterated dimension-split PRML: row Viterbi, column Viterbi, repeat."""
    R = K.shape[0] // 2
    x = x0.copy()
    x[known] = x_true[known]

    def allowed_for(kn, xt):
        a = np.ones(kn.shape + (2,), bool)
        a[kn & (xt < 0.5), 1] = False
        a[kn & (xt > 0.5), 0] = False
        return a

    for _ in range(sweeps):
        # row pass: subtract everything except the center row's taps
        Kv = K.copy()
        Kv[R, :] = 0.0
        z = y - c0 - cv2.filter2D(x, -1, Kv, borderType=cv2.BORDER_REPLICATE)
        x = viterbi_lines(z, K[R, :], allowed_for(known, x_true))
        x[known] = x_true[known]

        # column pass
        Kh = K.copy()
        Kh[:, R] = 0.0
        z = y - c0 - cv2.filter2D(x, -1, Kh, borderType=cv2.BORDER_REPLICATE)
        xt = viterbi_lines(z.T, K[:, R],
                           allowed_for(known.T, x_true.T))
        x = xt.T
        x[known] = x_true[known]
    return x


# ---------------------------------------------------------------- experiment

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", nargs="+")
    ap.add_argument("--grid", default="466x259")
    ap.add_argument("--payload", default=str(Path(__file__).parent.parent /
                                            "demo" / "payload_big.png"))
    ap.add_argument("--ecc", type=int, default=48)
    ap.add_argument("--radial", type=float, default=0.020)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--sweeps", type=int, default=3)
    ap.add_argument("--header-top", action="store_true",
                    help="take466/take380 use the top-edge header layout")
    args = ap.parse_args()

    from codec import fountain
    grid.set_ecc(args.ecc)
    grid.set_header_len(28)
    grid.set_header_centered(not args.header_top)
    grid.set_radial(args.radial)
    gw, gh = (int(v) for v in args.grid.split("x"))
    L = grid.Layout(gw, gh)

    data = Path(args.payload).read_bytes()
    bs = L.payload_capacity_bytes(grid.MODE_MONO) - 4
    enc = fountain.Encoder(data, bs)

    known = L.is_finder | L.is_sep | L.is_ring | L.is_header
    pay_mask = ~known
    allc = np.argwhere(np.ones((gh, gw), bool))

    print(f"grid {gw}x{gh}, kernel radius {args.radius}, sweeps {args.sweeps}")
    print(f"{'frame':>28s} {'thresh':>8s} {'DFE-fix':>8s} {'PRML':>8s}")

    res = []
    for path in args.frames:
        img = cv2.imread(path)
        if img is None:
            print(f"{path}: unreadable")
            continue
        H = grid.locate(img, L)
        if H is None:
            print(f"{Path(path).name:>28s}  no locate")
            continue
        hdr, _pay, _st = grid.sample_frame(img, L, H)
        if hdr is None:
            print(f"{Path(path).name:>28s}  no header")
            continue

        block = enc.block(hdr["seq"])
        block = block + b"\x00" * (bs - len(block))
        p = struct.pack("<I", zlib.crc32(block) & 0xFFFFFFFF) + block
        hdr_raw = grid.pack_header(hdr["seq"], hdr["k"], hdr["block_size"],
                                   hdr["file_size"], hdr["mode"],
                                   hdr["zone_w"], hdr["zone_modes"])
        truth_img = grid.render_frame(L, hdr_raw, p, grid.MODE_MONO, cell_px=1)
        x_true = (cv2.cvtColor(truth_img, cv2.COLOR_BGR2GRAY) > 127
                  ).astype(np.float32)

        lum = grid.sample_cells(img, L, H, allc).mean(axis=1)
        y = lum.reshape(gh, gw).astype(np.float32)

        th, _ = cv2.threshold(np.clip(lum, 0, 255).astype(np.uint8), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        x0 = (y > th).astype(np.float32)

        def ber(x):
            return float((x[pay_mask] != x_true[pay_mask]).mean())

        K, c0 = fit_kernel(y, x_true, radius=args.radius)
        b_thr = ber(x0)
        b_dfe = ber(dfe(y, K, c0, known, x_true, x0))
        b_prm = ber(prml(y, K, c0, known, x_true, x0, sweeps=args.sweeps))
        res.append((b_thr, b_dfe, b_prm))
        print(f"{Path(path).name:>28s} {b_thr*100:7.2f}% {b_dfe*100:7.2f}% "
              f"{b_prm*100:7.2f}%")

    if res:
        a = np.array(res)
        print(f"\nmedian: thresh {np.median(a[:,0])*100:.2f}%  "
              f"DFE {np.median(a[:,1])*100:.2f}%  "
              f"PRML {np.median(a[:,2])*100:.2f}%   (RS limit 1.23%)")
        for name, col in (("thresh", 0), ("DFE", 1), ("PRML", 2)):
            print(f"frames under RS limit, {name}: "
                  f"{int((a[:,col] < 0.0123).sum())}/{len(a)}")


if __name__ == "__main__":
    main()
