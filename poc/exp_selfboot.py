#!/usr/bin/env python3
"""
exp_selfboot.py — can a single frame decode itself from cold?

The ladder: a grid frame carries ~5% known cells (finders, timing ring,
separators, and the header band once it decodes). Those are enough equations
for a rough global channel kernel. PRML with that kernel drops BER enough
for a few Reed-Solomon codewords to certify (with GMD erasure escalation on
soft margins). Certified codewords label whole bands of cells exactly.
Refit on the grown label set, re-detect, certify more. Repeat.

No pilot frames, no cross-frame kernel, no training content, no camera
profile: the code's own structure plus its FEC layer form the training
ladder. Tested on real captures at 466x259 (~8.2 camera px/cell), a density
where hard thresholding certifies ZERO codewords.

Ground truth is used for REPORTING BER only, never inside the loop.
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
import exp_tile_prml as T
import exp_turbo_frame as TB
import prml as P

R = 2


def fit_sel(y, x, sel, reg=1e-3, pure=False):
    """Global 5x5+bias LS fit restricted to `sel` cells.

    pure=True keeps only equations whose ENTIRE 5x5 neighborhood is inside
    `sel` — every regressor value is then a known label rather than a noisy
    decision. Essential for the cold-start fit from structure cells only.
    """
    gh, gw = x.shape
    cols = []
    for dr in range(-R, R + 1):
        for dc in range(-R, R + 1):
            cols.append(np.roll(np.roll(x, -dr, 0), -dc, 1).ravel())
    cols.append(np.ones(x.size, np.float32))
    A = np.stack(cols, 1)
    interior = np.zeros((gh, gw), bool)
    interior[R:gh - R, R:gw - R] = True
    eq = sel & interior
    if pure:
        eroded = cv2.erode(sel.astype(np.uint8), np.ones((5, 5), np.uint8))
        if (eroded.astype(bool) & interior).sum() >= 200:
            eq = eroded.astype(bool) & interior
    s = eq.ravel()
    At, bt = A[s], y.ravel()[s]
    ATA = At.T @ At + reg * np.eye(26) * At.shape[0]
    k = np.linalg.solve(ATA, At.T @ bt)
    return k[:-1].reshape(5, 5), float(k[-1])


def certify_gmd(bits, conf_bits, sub, max_erasures=32, min_agree=0.85):
    """Per-codeword RS decode with GMD-style erasure escalation.

    conf_bits: per-bit confidence; a byte's confidence is its weakest bit.
    Escalate erasures over the least-confident bytes until the codeword
    decodes.

    CAUTION, learned the hard way: at s = nsym erasures RS "decoding" always
    succeeds (any k received symbols determine a codeword), so unbounded
    escalation certifies garbage. Cap s well below nsym to keep detection
    margin, and additionally demand the re-encoded codeword agree with the
    received hard bits: a genuine correction of <=24 byte errors flips at
    most 9.4% of the bits, so require >= min_agree agreement.
    """
    n = sub.n_coded
    by = np.packbits(bits[: n * 8].astype(np.uint8))
    conf_byte = conf_bits[: n * 8].reshape(-1, 8).min(axis=1)
    cert_mask = np.zeros(len(sub.cells), bool)
    cert_bits = np.zeros(len(sub.cells), np.float32)
    n_ok = 0
    for (mlo, mhi, clo, chi) in sub.spans:
        chunk = bytes(by[clo:chi])
        rx_bits = np.unpackbits(np.frombuffer(chunk, np.uint8))
        conf = conf_byte[clo:chi]
        order = np.argsort(conf)                      # least confident first
        for s in (0, 8, 16, 24, 32):
            if s > max_erasures:
                break
            try:
                dec = bytes(sub.rs.decode(chunk,
                                          erase_pos=list(order[:s]))[0])
            except (ReedSolomonError, ValueError):
                continue
            coded = bytes(sub.rs.encode(dec))
            cb = np.unpackbits(np.frombuffer(coded, np.uint8))
            if (cb == rx_bits).mean() < min_agree:
                continue                              # garbage "success"
            cert_mask[clo * 8:chi * 8] = True
            cert_bits[clo * 8:chi * 8] = cb
            n_ok += 1
            break
    return cert_mask, cert_bits, n_ok


def gaussian_kernel(sigma):
    ax = np.arange(-R, R + 1, dtype=np.float32)
    g = np.exp(-0.5 * (ax / max(sigma, 1e-3)) ** 2)
    K = np.outer(g, g)
    return K / K.sum()


def _poly_terms(gh, gw):
    """Quadratic surface basis over the frame, (gh*gw, 6)."""
    r = np.linspace(-1, 1, gh, dtype=np.float32)[:, None] * np.ones((1, gw), np.float32)
    c = np.ones((gh, 1), np.float32) * np.linspace(-1, 1, gw, dtype=np.float32)[None, :]
    return np.stack([np.ones_like(r), r, c, r * r, c * c, r * c],
                    axis=2).reshape(-1, 6)


def fit_parametric(y, x, sel):
    """Cold-start channel: y ~ A(r,c) * (G_sigma conv x) + B(r,c).

    A free 25-tap fit from structure cells fails twice over: they are mostly
    uniform blocks (tap shape unobservable) at the frame's periphery (one
    global gain fitted there mismatches the center under vignetting; the
    first attempt with scalar gain DOUBLED the BER). So: isotropic blur with
    sigma by line search, and gain/offset as smooth quadratic surfaces (12
    coefficients, closed form per sigma). Returns per-cell tap and bias maps
    ready for the spatially-varying detector.
    """
    gh, gw = y.shape
    P6 = _poly_terms(gh, gw)
    eq = sel.ravel()
    yv = y.ravel()[eq]
    best = None
    for sigma in (0.25, 0.35, 0.45, 0.55, 0.65, 0.8):
        G = gaussian_kernel(sigma)
        z = cv2.filter2D(x, -1, G, borderType=cv2.BORDER_REPLICATE).ravel()
        D = np.concatenate([P6 * z[:, None], P6], axis=1)[eq]   # (n, 12)
        coef, *_ = np.linalg.lstsq(D, yv, rcond=None)
        r = float(((D @ coef - yv) ** 2).mean())
        if best is None or r < best[0]:
            best = (r, sigma, coef)
    _, sigma, coef = best
    G = gaussian_kernel(sigma)
    Afield = (P6 @ coef[:6]).reshape(gh, gw)
    Bfield = (P6 @ coef[6:]).reshape(gh, gw)
    tapmap = Afield[:, :, None] * G.ravel()[None, None, :]
    return tapmap.astype(np.float32), Bfield.astype(np.float32), sigma


def selfboot(y, layout, sub, struct_truth, known, tiles=(16, 28), rounds=8,
             x_true=None, verbose=True):
    """The ladder. Returns (x, certified_history)."""
    gh, gw = y.shape
    cells = sub.cells

    th, _ = cv2.threshold(np.clip(y.ravel(), 0, 255).astype(np.uint8), 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    x = (y > th).astype(np.float32)
    x[known] = struct_truth[known]

    lab = x.copy()
    lab[known] = struct_truth[known]
    sel = known.copy()
    hist = []

    for rd in range(rounds):
        n_lab = int(sel.sum())
        # kernel(s) from current labels: parametric cold-start, then global,
        # then per-tile as the label set grows
        if rd == 0:
            # cold start: free 25-tap kernel from the HEADER BAND's middle
            # columns. Known random bits, best geometry on the frame (the
            # error map fails at left/right edge columns; the top-middle is
            # 0.3-2%). Structure-cell fits failed twice here: uniform blocks
            # can't excite tap shape, and edge cells train an edge channel.
            hdr_sel = layout.is_header.copy()
            q = gw // 4
            hdr_sel[:, :q] = False
            hdr_sel[:, gw - q:] = False
            K, c0 = fit_sel(y, lab, hdr_sel)
            x = P.prml(y, K, c0, sel, lab, x, sweeps=2)
            use_tiles = False
        elif n_lab < 12000:
            K, c0 = fit_sel(y, lab, sel)
            x = P.prml(y, K, c0, sel, lab, x, sweeps=2)
            use_tiles = False
        else:
            tap, bias = TB.fit_tiles_sel(y, lab, sel, *tiles)
            x = T.prml_tiles(y, tap, bias, sel, lab, x, sweeps=2)
            use_tiles = True
        x[sel] = lab[sel]

        # soft margins for GMD: residual of the full model at each cell
        if use_tiles:
            pred = bias + T.conv_varying(x, tap)
            K0 = tap[:, :, 12].mean()
        else:
            pred = c0 + cv2.filter2D(x, -1, K, borderType=cv2.BORDER_REPLICATE)
            K0 = K[R, R]
        margin = 1.0 - np.clip(np.abs(y - pred) / max(abs(K0), 1e-3), 0, 1)

        bits = x[cells[:, 0], cells[:, 1]]
        conf = margin[cells[:, 0], cells[:, 1]]
        cert_mask, cert_bits, n_ok = certify_gmd(bits, conf, sub)
        hist.append(n_ok)

        ber = (None if x_true is None else
               float((x[~known] != x_true[~known]).mean()))
        if verbose:
            print(f"    round {rd}: labels {n_lab:6d} cells, "
                  f"certified {n_ok}/{len(sub.spans)} codewords"
                  + (f", BER {ber*100:.2f}%" if ber is not None else ""))

        cc = cells[cert_mask]
        new_sel = sel.copy()
        new_sel[cc[:, 0], cc[:, 1]] = True
        lab[cc[:, 0], cc[:, 1]] = cert_bits[cert_mask]
        if new_sel.sum() == sel.sum() and rd > 0:
            break
        sel = new_sel
    return x, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", nargs="+")
    ap.add_argument("--grid", default="466x259")
    ap.add_argument("--payload", default=str(Path(__file__).parent.parent /
                                            "demo" / "payload_big.png"))
    ap.add_argument("--ecc", type=int, default=48)
    ap.add_argument("--radial", type=float, default=0.020)
    ap.add_argument("--header-top", action="store_true")
    ap.add_argument("--max-seq", type=int, default=600)
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
    sub = TB.SubBlock(L, args.ecc, bs + 4)
    known = L.is_finder | L.is_sep | L.is_ring | L.is_header
    allc = np.argwhere(np.ones((gh, gw), bool))

    print(f"grid {gw}x{gh}, {len(sub.spans)} codewords/frame")

    for path in args.frames:
        img = cv2.imread(path)
        if img is None:
            continue
        H = grid.locate(img, L)
        if H is None:
            print(f"{Path(path).name}: no locate")
            continue
        hdr, _p, _s = grid.sample_frame(img, L, H)
        how = "hard"
        if hdr is None:
            proto = None  # need one hard header somewhere to know constants
            print(f"{Path(path).name}: no hard header (run a hard-header "
                  f"frame first to learn constants, or pass them)")
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
        struct_truth = x_true * 0
        struct_truth[known] = x_true[known]

        y = grid.sample_cells(img, L, H, allc).mean(axis=1)\
            .reshape(gh, gw).astype(np.float32)

        # baseline: threshold + GMD certify (no equalization at all)
        th, _ = cv2.threshold(np.clip(y.ravel(), 0, 255).astype(np.uint8),
                              0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        x0 = (y > th).astype(np.float32)
        bits0 = x0[sub.cells[:, 0], sub.cells[:, 1]]
        conf0 = np.abs(y - th)[sub.cells[:, 0], sub.cells[:, 1]]
        conf0 = conf0 / max(conf0.max(), 1e-6)
        _, _, n0 = certify_gmd(bits0, conf0, sub)
        b0 = float((x0[~known] != x_true[~known]).mean())
        print(f"\n{Path(path).name} (header via {how}): threshold BER "
              f"{b0*100:.2f}%, baseline certified {n0}/{len(sub.spans)}")

        x, hist = selfboot(y, L, sub, struct_truth, known, x_true=x_true)
        print(f"  -> self-boot: {hist[0]} -> {hist[-1]}/{len(sub.spans)} "
              f"codewords, trajectory {hist}")


if __name__ == "__main__":
    main()
