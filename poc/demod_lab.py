#!/usr/bin/env python3
"""
demod_lab.py — compare demodulators on ALREADY-EXTRACTED cell luminances.

Geometry is fixed by extract_cells.py, so this is pure numpy over a memmap and
an idea costs seconds instead of eight minutes of 4K decoding.

The metric is the only one that matters: codewords CERTIFIED per frame, where
certified means RS decoded AND the codeword's own CRC32 passed. No truth is
consulted, so the number is exactly what a receiver would harvest.

  yield = certified / (n_sub * frames)
  KB/s  = yield * n_sub * SUB * fps / 1024

    python3 demod_lab.py rec7870 --methods global,local31,local31+erase
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


def local_mean(y, k):
    """Illumination estimate: box mean over a k x k neighbourhood IN CELLS.

    The payload is RS+fountain output, i.e. pseudorandom, so over a window of
    hundreds of cells the mean converges on the midpoint between the black and
    white levels AT THAT POINT ON THE SCREEN. That makes the box mean a direct
    estimate of the local decision threshold, and it tracks vignetting, backlight
    non-uniformity, glare and off-axis roll-off that one global Otsu cannot.
    """
    return cv2.boxFilter(y.astype(np.float32), -1, (k, k),
                         borderType=cv2.BORDER_REFLECT)


def local_std(y, lm, k):
    m2 = cv2.boxFilter((y.astype(np.float32) ** 2), -1, (k, k),
                       borderType=cv2.BORDER_REFLECT)
    return np.sqrt(np.maximum(m2 - lm ** 2, 1e-6))


class Certifier:
    """RS-decode each codeword and check its CRC32. Receiver-honest."""

    def __init__(self, layout, ecc, n_sub):
        self.rs = RSCodec(ecc)
        self.ecc = ecc
        self.n_sub = n_sub
        self.sub_size = (255 - ecc) - 4
        self.cells = layout.payload_cells[: n_sub * 255 * 8]

    def run(self, bits, conf=None, erase=False):
        """bits over payload cells (payload order). Returns
        (n_certified, cert_mask over self.cells, cert_bits)."""
        by = np.packbits(bits[: self.n_sub * 255 * 8].astype(np.uint8))
        n = 0
        cmask = np.zeros(len(self.cells), bool)
        cbits = np.zeros(len(self.cells), np.float32)
        for j in range(self.n_sub):
            lo, hi = j * 255, (j + 1) * 255
            chunk = bytes(by[lo:hi])
            dec = None
            try:
                dec = bytes(self.rs.decode(chunk)[0])
            except ReedSolomonError:
                if erase and conf is not None:
                    m = conf[lo:hi]
                    order = np.argsort(m)
                    for n_er in range(4, int(self.ecc * 0.7) + 1, 6):
                        try:
                            dec = bytes(self.rs.decode(
                                chunk,
                                erase_pos=[int(i) for i in order[:n_er]])[0])
                            break
                        except ReedSolomonError:
                            continue
            if dec is None or len(dec) < 4 + self.sub_size:
                continue
            blk = dec[4:4 + self.sub_size]
            if zlib.crc32(blk) & 0xFFFFFFFF != struct.unpack("<I", dec[:4])[0]:
                continue
            n += 1
            coded = bytes(self.rs.encode(dec))
            cb = np.unpackbits(np.frombuffer(coded, np.uint8))
            cmask[lo * 8:hi * 8] = True
            cbits[lo * 8:hi * 8] = cb
        return n, cmask, cbits


def byte_conf(cellconf, n_bytes):
    c = cellconf[: n_bytes * 8].reshape(n_bytes, 8).min(axis=1)
    return c


def struct_truth(L, hdr):
    """The cells a receiver knows a priori: finders, ring, separators, and the
    header (whose content it has just parsed). Used to pin the equalizer."""
    pay = np.zeros(L.payload_capacity_bytes(grid.MODE_MONO), np.uint8)
    raw = grid.pack_header(int(hdr["seq"]), int(hdr["k"]),
                           int(hdr["block_size"]), int(hdr["file_size"]),
                           grid.MODE_MONO, 0, 0)
    img = grid.render_frame(L, raw, pay.tobytes(), grid.MODE_MONO, cell_px=1)
    return (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 127).astype(np.float32)


def demod(y, L, base):
    """Return (bits over full grid, per-cell confidence over full grid)."""
    if base == "global":
        pl = y[L.payload_cells[:, 0], L.payload_cells[:, 1]]
        th, _ = cv2.threshold(np.clip(pl, 0, 255).astype(np.uint8), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        spread = max(1e-3, np.percentile(pl, 90) - np.percentile(pl, 10))
        return (y > th).astype(np.float32), np.abs(y - th) / spread
    if base.startswith("local"):
        k = int(base[5:])
        lm = local_mean(y, k)
        sd = local_std(y, lm, k)
        return (y > lm).astype(np.float32), np.abs(y - lm) / np.maximum(sd, 1e-3)
    raise SystemExit(f"unknown demodulator {base}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("--methods", default="global,local15,local31,local61")
    ap.add_argument("--frames", type=int, default=120,
                    help="how many header-bearing frames to evaluate")
    ap.add_argument("--lo", type=float, default=0.15)
    ap.add_argument("--hi", type=float, default=0.97)
    ap.add_argument("--contiguous", action="store_true",
                    help="evaluate a CONSECUTIVE run of frames, which is what a "
                         "receiver actually sees. Kernel transfer decays in "
                         "about a second, so a spread sample understates the "
                         "rolling donor and a contiguous run does not.")
    ap.add_argument("--refit", type=int, default=0,
                    help="codewords a frame must certify to become the next "
                         "kernel donor (0 = n_sub - 2)")
    ap.add_argument("--sweeps", type=int, default=3)
    ap.add_argument("--tiles", default="8x14")
    ap.add_argument("--k", type=int, default=0,
                    help="keep only frames whose header advertises this k "
                         "(default: the majority transmission)")
    args = ap.parse_args()

    meta = np.load(args.stem + ".npz")
    gw, gh = int(meta["gw"]), int(meta["gh"])
    fps = float(meta["fps"])
    ecc = int(meta["ecc"])
    ys = np.lib.format.open_memmap(args.stem + ".dat.npy", mode="r")

    grid.set_ecc(ecc); grid.set_header_len(28); grid.set_header_centered(True)
    L = grid.Layout(gw, gh)
    n_sub = grid.sub_count(L, grid.MODE_MONO)
    SUB = (255 - ecc) - 4
    cert = Certifier(L, ecc, n_sub)
    pc = L.payload_cells
    known = L.is_finder | L.is_sep | L.is_ring | L.is_header

    # Keep ONE transmission. The countdown clip was rendered from a different
    # transmit and carries a perfectly valid header for another file, so
    # mixing it in measures the wrong channel as well as the wrong file.
    ok = meta["seq"] >= 0
    if ok.sum():
        from collections import Counter
        maj = Counter(zip(meta["k"][ok].tolist(),
                          meta["file_size"][ok].tolist())).most_common()
        want = args.k if args.k else maj[0][0]
        if args.k:
            want = int(args.k)
            ok &= meta["k"] == want
        else:
            ok &= (meta["k"] == maj[0][0][0]) & (meta["file_size"] == maj[0][0][1])
        if len(maj) > 1:
            print(f"transmissions present: {maj}  -> keeping k={want}")
    hdr_rows = np.flatnonzero(ok)
    lo, hi = int(len(hdr_rows) * args.lo), int(len(hdr_rows) * args.hi)
    hdr_rows = hdr_rows[lo:hi]
    if args.contiguous:
        hdr_rows = hdr_rows[:args.frames]
    elif len(hdr_rows) > args.frames:
        hdr_rows = hdr_rows[np.linspace(0, len(hdr_rows) - 1,
                                        args.frames).astype(int)]
    have = meta["frame"][hdr_rows]
    hdrs = [dict(seq=meta["seq"][i], k=meta["k"][i],
                 block_size=meta["block_size"][i],
                 file_size=meta["file_size"][i]) for i in hdr_rows]
    print(f"{args.stem}: {gw}x{gh} ecc={ecc} n_sub={n_sub} SUB={SUB} "
          f"fps={fps:.0f}")
    full = n_sub * SUB * fps / 1024
    print(f"ceiling at 100% yield: {full:.1f} KB/s")
    span = (have[-1] - have[0] + 1) / fps if len(have) > 1 else 0
    print(f"evaluating {len(have)} header-bearing frames over "
          f"{span:.1f}s (frames {have[0]}..{have[-1]})")

    # straddle: payload cells sitting in the middle of the eye. Independent of
    # any demodulator, so it separates "the frame is a blend of two displayed
    # frames" from "the demodulator picked the wrong threshold".
    mids = []
    for fn in have[: min(40, len(have))]:
        y = ys[fn].astype(np.float32)
        pv = y[pc[:, 0], pc[:, 1]]
        a, b = np.percentile(pv, 3), np.percentile(pv, 97)
        mids.append(float(((pv > a + 0.30 * (b - a)) &
                           (pv < a + 0.70 * (b - a))).mean()))
    print(f"straddle (mid-band fraction): median {100*np.median(mids):.1f}%\n")

    REFIT = args.refit or (n_sub - 2)
    tr, tc = (int(v) for v in args.tiles.split("x"))
    methods = args.methods.split(",")
    results = {}
    for name in methods:
        parts = name.split("+")
        base, erase, prml = parts[0], "erase" in parts, "prml" in parts
        tot = 0
        tap = bias = None
        n_donor = 0
        for fn, hd in zip(have, hdrs):
            y = ys[fn].astype(np.float32)
            x0, conf = demod(y, L, base)
            bits0 = x0[pc[:, 0], pc[:, 1]]
            bc = byte_conf(conf[pc[:, 0], pc[:, 1]], n_sub * 255) if erase else None
            n0, m0, b0 = cert.run(bits0, bc, erase=erase)
            best_n, best_x, best_m, best_b = n0, x0, m0, b0
            if prml and tap is not None:
                xt = struct_truth(L, hd)
                x1 = T.prml_tiles(y, tap, bias, known, xt, x0,
                                  sweeps=args.sweeps)
                bits1 = x1[pc[:, 0], pc[:, 1]]
                n1, m1, b1 = cert.run(bits1, bc, erase=erase)
                if n1 > best_n:
                    best_n, best_x, best_m, best_b = n1, x1, m1, b1
            tot += best_n
            if prml and best_n >= REFIT:
                # Fit ONLY on cells the receiver genuinely recovered: the
                # certified codewords plus the structure it knows a priori.
                xt = struct_truth(L, hd)
                lab = best_x.copy()
                lab[known] = xt[known]
                cc = cert.cells[best_m]
                lab[cc[:, 0], cc[:, 1]] = best_b[best_m]
                sel = known.copy()
                sel[cc[:, 0], cc[:, 1]] = True
                tap, bias = TB.fit_tiles_sel(y, lab, sel, tr, tc)
                n_donor += 1
        yld = tot / (len(have) * n_sub)
        results[name] = yld
        extra = f"  [{n_donor} donors]" if prml else ""
        print(f"{name:>18s}: {tot:5d}/{len(have)*n_sub:<6d} cw = "
              f"{100*yld:5.1f}% yield -> {full*yld:6.1f} KB/s{extra}")

    best = max(results, key=results.get)
    print(f"\nbest: {best} at {full*results[best]:.1f} KB/s")


if __name__ == "__main__":
    main()
