#!/usr/bin/env python3
"""
exp_local_demod.py — local-adaptive multilevel demodulation.

THE REFRAME
-----------
The failed gray4 take was diagnosed as "veiling glare collapsed the
constellation." That diagnosis was half right and led to a false choice
(fix the room, or fall back to mono). Measurement says otherwise:

  globally the 4 levels are still resolvable — modes at 106/158/211/241 —
  they are just UNEVENLY spaced (52, 53, 30) and the whole constellation
  RIDES ON A SPATIAL GRADIENT (black floor 57 on the left, 91 on the right).

Glare is not noise. It is a smooth, static, additive field; the data is
per-cell high-frequency. Those separate. What actually failed is that the
receiver estimates ONE global palette:

  1. GLOBAL k-means cannot track a black floor that moves 34 counts across
     the panel — comparable to the 52-count level spacing.
  2. 23% of payload cells carry no data (a subblock frame renders
     n_sub*255 = 4080 of 5320 available bytes; the rest stay black). That
     dead mass is a fifth population, so fitting 4 clusters to it drags the
     dark centroid down and skews every boundary.

Fix both: estimate the palette PER TILE, over DATA CELLS ONLY. The channel
is locally affine even when it is globally a mess.

Two estimators, because they fail differently:
  - quantile: symbols are ~uniform in RS-coded data, so the 25/50/75
    percentiles ARE the decision boundaries. Immune to any monotone
    distortion (offset, gamma, vignette). Degenerate on ties.
  - k-means: robust to ties, but assumes roughly balanced clusters.
"""
import argparse
import struct
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np
from reedsolo import RSCodec, ReedSolomonError

sys.path.insert(0, str(Path(__file__).parent))
from codec import grid


def data_cell_count(header, layout):
    """How many payload cells actually carry data.

    A subblock frame renders exactly n_sub codewords of 255 bytes; whatever
    the grid holds beyond that stays black and must be excluded from any
    palette estimate.
    """
    n_sub = max(1, (header["block_size"] + 4) // 255)
    bpc = 2 if header["mode"] in (grid.MODE_GRAY4, grid.MODE_COLOR4) else 1
    return (n_sub * 255 * 8) // bpc, n_sub


def demod_gray4_local(pay_lum, layout, n_cells, tiles=(6, 10), method="kmeans"):
    """Per-tile 4-level demodulation. Returns (bits, conf) over n_cells."""
    cells = layout.payload_cells[:n_cells]
    v = pay_lum[:n_cells].astype(np.float32)
    syms = np.zeros(n_cells, np.int64)
    conf = np.zeros(n_cells, np.float32)
    tr, tc = tiles

    for i in range(tr):
        for j in range(tc):
            m = ((cells[:, 0] >= i * layout.gh // tr) &
                 (cells[:, 0] < (i + 1) * layout.gh // tr) &
                 (cells[:, 1] >= j * layout.gw // tc) &
                 (cells[:, 1] < (j + 1) * layout.gw // tc))
            n = int(m.sum())
            if n < 40:
                continue
            vt = v[m]
            if method == "quantile":
                b = np.percentile(vt, [25, 50, 75])
                if len(set(np.round(b, 3))) < 3:       # degenerate, fall back
                    method_t = "kmeans"
                else:
                    centers = np.array([vt[vt <= b[0]].mean(),
                                        vt[(vt > b[0]) & (vt <= b[1])].mean(),
                                        vt[(vt > b[1]) & (vt <= b[2])].mean(),
                                        vt[vt > b[2]].mean()])
                    method_t = "quantile"
            else:
                method_t = "kmeans"
            if method_t == "kmeans":
                crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.2)
                _r, lab, cent = cv2.kmeans(vt.reshape(-1, 1), 4, None, crit, 6,
                                           cv2.KMEANS_PP_CENTERS)
                centers = np.sort(cent.ravel())
                b = (centers[:-1] + centers[1:]) / 2.0
            s = np.digitize(vt, b)
            syms[m] = s
            d = np.min(np.abs(vt[:, None] - b[None, :]), axis=1)
            conf[m] = d / max(1e-3, centers[-1] - centers[0])

    bits = np.array([grid.GRAY4_BITS[int(x)] for x in syms],
                    dtype=np.uint8).reshape(-1)
    bconf = np.repeat(conf, 2)
    raw = grid._bytes(bits)
    nb = min(len(raw), len(bconf) // 8)
    return raw, bconf[: nb * 8].reshape(nb, 8).min(axis=1)


def certify(raw, cbyte, n_sub, ecc=48):
    """Per-codeword RS with GMD erasure escalation; returns codewords passing
    their own CRC32 (the fountain layer's certification).

    raw: packed bytes. cbyte: per-byte confidence (min over its 8 bits)."""
    rs = RSCodec(ecc)
    sub_size = (255 - ecc) - 4
    good = []
    for j in range(min(n_sub, len(raw) // 255)):
        chunk = raw[j * 255:(j + 1) * 255]
        m = cbyte[j * 255:(j + 1) * 255]
        dec = None
        try:
            dec = bytes(rs.decode(chunk)[0])
        except ReedSolomonError:
            if len(m) == 255:
                order = np.argsort(m)
                for n_er in range(4, int(ecc * 0.7) + 1, 6):
                    try:
                        dec = bytes(rs.decode(chunk,
                                    erase_pos=[int(i) for i in order[:n_er]])[0])
                        break
                    except ReedSolomonError:
                        continue
        if dec is None or len(dec) < 4 + sub_size:
            continue
        blk = dec[4:4 + sub_size]
        if zlib.crc32(blk) & 0xFFFFFFFF == struct.unpack("<I", dec[:4])[0]:
            good.append(j)
    return good


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", nargs="+")
    ap.add_argument("--grid", default="203x112")
    ap.add_argument("--ecc", type=int, default=48)
    ap.add_argument("--radial", type=float, default=0.0)
    ap.add_argument("--tiles", default="6x10")
    args = ap.parse_args()

    grid.set_ecc(args.ecc)
    grid.set_header_len(28)
    grid.set_header_centered(True)
    grid.set_radial(args.radial)
    gw, gh = (int(v) for v in args.grid.split("x"))
    L = grid.Layout(gw, gh)
    tr, tc = (int(v) for v in args.tiles.split("x"))

    proto = dict(k=261, block_size=4312, file_size=1121502, mode=2,
                 zone_w=0, zone_modes=0)
    T = grid.header_templates(proto, 1000)

    tot = {"global": 0, "local-km": 0, "local-q": 0}
    nfr = 0
    for path in args.frames:
        img = cv2.imread(path)
        if img is None:
            continue
        H = grid.locate(img, L)
        if H is None:
            print(f"{Path(path).name}: no locate")
            continue
        header, samples, _ = grid.sample_frame(img, L, H)
        how = "hard"
        if header is None:
            hl = grid.sample_cells(img, L, H, L.header_cells).mean(axis=1)
            seq, margin = grid.ml_header_seq(hl, T)
            if margin < 3.0:
                print(f"{Path(path).name}: no header")
                continue
            header = dict(proto, seq=seq)
            how = f"ml{margin:.0f}"
            samples = grid.sample_cells(img, L, H, L.payload_cells)
        nfr += 1
        n_cells, n_sub = data_cell_count(header, L)
        pay_lum = samples.mean(axis=1)

        # baseline: the shipped global k-means over ALL payload cells
        raw_b, conf_b = grid.raw_bits_and_conf(header, samples, L)
        g = certify(raw_b, conf_b, n_sub, args.ecc)

        bits_k, conf_k = demod_gray4_local(pay_lum, L, n_cells, (tr, tc), "kmeans")
        lk = certify(bits_k, conf_k, n_sub, args.ecc)

        bits_q, conf_q = demod_gray4_local(pay_lum, L, n_cells, (tr, tc), "quantile")
        lq = certify(bits_q, conf_q, n_sub, args.ecc)

        tot["global"] += len(g); tot["local-km"] += len(lk); tot["local-q"] += len(lq)
        print(f"{Path(path).name:>10s} ({how}): codewords certified — "
              f"global {len(g):2d}/{n_sub}   local-kmeans {len(lk):2d}/{n_sub}   "
              f"local-quantile {len(lq):2d}/{n_sub}")

    if nfr:
        print(f"\nTOTAL over {nfr} frames: global {tot['global']}, "
              f"local-kmeans {tot['local-km']}, local-quantile {tot['local-q']}")


if __name__ == "__main__":
    main()
