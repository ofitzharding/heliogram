#!/usr/bin/env python3
"""
exp_tile_prml.py — PRML with PER-TILE fitted kernels.

Follow-on to prml.py after its first real-capture run. Diagnosis on take466
f35: errors concentrate in edge columns (13-23%) while the center runs
0.3-3%, and the per-frame radial sweep shows k1=0.020 is already optimal.
So the residual impairment is spatially-varying sub-cell sampling offset —
higher-order lens geometry that one global k1 cannot express.

A locally-fitted kernel absorbs a local constant sampling offset: sampling
0.3 cells to the left of center shows up as asymmetric tap weights, which
the equalizer then uses correctly. This is what adaptive equalization has
always done in modems; here the tiles make it spatial.

Model per tile t:  y[r,c] = c0_t + sum K_t[dr,dc] x[r+dr, c+dc] + n
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
R = 2
TAPS = 2 * R + 1


def fit_tile_kernels(y, x, tiles_r, tiles_c, reg=1e-3):
    """Per-tile LS fit -> tap maps (gh, gw, 25) and bias map (gh, gw)."""
    gh, gw = x.shape
    cols = []
    for dr in range(-R, R + 1):
        for dc in range(-R, R + 1):
            cols.append(np.roll(np.roll(x, -dr, axis=0), -dc, axis=1).ravel())
    cols.append(np.ones(x.size, np.float32))
    A = np.stack(cols, axis=1)

    tapmap = np.zeros((gh, gw, TAPS * TAPS), np.float32)
    biasmap = np.zeros((gh, gw), np.float32)
    interior = np.zeros((gh, gw), bool)
    interior[R:gh - R, R:gw - R] = True

    for i in range(tiles_r):
        for j in range(tiles_c):
            r0, r1 = i * gh // tiles_r, (i + 1) * gh // tiles_r
            c0, c1 = j * gw // tiles_c, (j + 1) * gw // tiles_c
            m = np.zeros((gh, gw), bool)
            m[r0:r1, c0:c1] = True
            sel = (m & interior).ravel()
            At, bt = A[sel], y.ravel()[sel]
            ATA = At.T @ At + reg * np.eye(At.shape[1]) * At.shape[0]
            k = np.linalg.solve(ATA, At.T @ bt)
            tapmap[r0:r1, c0:c1] = k[:-1]
            biasmap[r0:r1, c0:c1] = k[-1]
    return tapmap, biasmap


def conv_varying(x, tapmap, skip=None):
    """sum over (dr,dc) of tapmap[...,idx] * x[r+dr, c+dc], optionally
    skipping a set of tap indices (e.g. the center row for the row pass)."""
    gh, gw = x.shape
    xp = np.pad(x, R, mode="edge")
    out = np.zeros((gh, gw), np.float32)
    idx = 0
    for dr in range(-R, R + 1):
        for dc in range(-R, R + 1):
            if skip is None or idx not in skip:
                out += tapmap[:, :, idx] * xp[R + dr:R + dr + gh,
                                              R + dc:R + dc + gw]
            idx += 1
    return out


def viterbi_lines_varying(z, hmap, allowed):
    """Viterbi along axis 1, with per-position taps hmap (nr, W, 5)."""
    nr, W = z.shape
    mem = TAPS - 1
    Wp = W + 2 * R

    allowedp = np.ones((nr, Wp, 2), bool)
    allowedp[:, R:-R] = allowed
    allowedp[:, :R, 1] = False
    allowedp[:, -R:, 1] = False

    idx = np.arange(32)
    bits = ((idx[:, None] >> np.arange(4, -1, -1)[None, :]) & 1).astype(np.float32)
    # predvec[r, c, 32] for observation column c
    predvec = np.einsum("rct,bt->rcb", hmap.astype(np.float32), bits)

    cost = np.zeros((nr, 16), np.float32)
    bt_ = np.zeros((Wp, nr, 16), np.uint8)
    preds = [(sp >> 1, (sp >> 1) | 8) for sp in range(16)]
    rows = np.arange(nr)

    for cp in range(Wp):
        if cp >= mem:
            c = cp - mem
            pm = (z[:, c][:, None] - predvec[:, c, :]) ** 2
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
        bt_[cp] = btc

    s = cost.argmin(axis=1)
    u = np.zeros((nr, Wp), np.uint8)
    for cp in range(Wp - 1, -1, -1):
        u[:, cp] = s & 1
        s = bt_[cp][rows, s]
    return u[:, R:-R].astype(np.float32)


CENTER_ROW = [R * TAPS + j for j in range(TAPS)]          # taps with dr=0
CENTER_COL = [i * TAPS + R for i in range(TAPS)]          # taps with dc=0


def prml_tiles(y, tapmap, biasmap, known, x_true, x0, sweeps=3):
    x = x0.copy()
    x[known] = x_true[known]

    def allowed_for(kn, xt):
        a = np.ones(kn.shape + (2,), bool)
        a[kn & (xt < 0.5), 1] = False
        a[kn & (xt > 0.5), 0] = False
        return a

    hmap_row = tapmap[:, :, CENTER_ROW]                    # (gh, gw, 5)
    hmap_col = tapmap[:, :, CENTER_COL]

    for _ in range(sweeps):
        z = y - biasmap - conv_varying(x, tapmap, skip=set(CENTER_ROW))
        x = viterbi_lines_varying(z, hmap_row, allowed_for(known, x_true))
        x[known] = x_true[known]

        z = y - biasmap - conv_varying(x, tapmap, skip=set(CENTER_COL))
        zt = z.T
        hm = np.transpose(hmap_col, (1, 0, 2))
        x = viterbi_lines_varying(zt, hm, allowed_for(known.T, x_true.T)).T
        x[known] = x_true[known]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", nargs="+")
    ap.add_argument("--grid", default="466x259")
    ap.add_argument("--payload", default=str(Path(__file__).parent.parent /
                                            "demo" / "payload_big.png"))
    ap.add_argument("--ecc", type=int, default=48)
    ap.add_argument("--radial", type=float, default=0.020)
    ap.add_argument("--tiles", default="6x10", help="tile grid ROWSxCOLS")
    ap.add_argument("--sweeps", type=int, default=3)
    ap.add_argument("--header-top", action="store_true")
    args = ap.parse_args()

    from codec import fountain
    grid.set_ecc(args.ecc)
    grid.set_header_len(28)
    grid.set_header_centered(not args.header_top)
    grid.set_radial(args.radial)
    gw, gh = (int(v) for v in args.grid.split("x"))
    tr, tc = (int(v) for v in args.tiles.split("x"))
    L = grid.Layout(gw, gh)

    data = Path(args.payload).read_bytes()
    bs = L.payload_capacity_bytes(grid.MODE_MONO) - 4
    enc = fountain.Encoder(data, bs)

    known = L.is_finder | L.is_sep | L.is_ring | L.is_header
    pay_mask = ~known
    allc = np.argwhere(np.ones((gh, gw), bool))

    print(f"grid {gw}x{gh}, tiles {tr}x{tc}, sweeps {args.sweeps}")
    print(f"{'frame':>28s} {'thresh':>8s} {'globPRML':>9s} {'tilePRML':>9s}")

    import prml as prml_mod
    res = []
    for path in args.frames:
        img = cv2.imread(path)
        if img is None:
            continue
        H = grid.locate(img, L)
        if H is None:
            print(f"{Path(path).name:>28s}  no locate")
            continue
        hdr, _p, _s = grid.sample_frame(img, L, H)
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

        Kg, c0g = prml_mod.fit_kernel(y, x_true)
        xg = prml_mod.prml(y, Kg, c0g, known, x_true, x0, sweeps=args.sweeps)

        tapmap, biasmap = fit_tile_kernels(y, x_true, tr, tc)
        xt_ = prml_tiles(y, tapmap, biasmap, known, x_true, x0,
                         sweeps=args.sweeps)
        res.append((ber(x0), ber(xg), ber(xt_)))
        print(f"{Path(path).name:>28s} {res[-1][0]*100:7.2f}% "
              f"{res[-1][1]*100:8.2f}% {res[-1][2]*100:8.2f}%")

    if res:
        a = np.array(res)
        print(f"\nmedian: thresh {np.median(a[:,0])*100:.2f}%  "
              f"globPRML {np.median(a[:,1])*100:.2f}%  "
              f"tilePRML {np.median(a[:,2])*100:.2f}%   (RS limit 1.23%)")


if __name__ == "__main__":
    main()
