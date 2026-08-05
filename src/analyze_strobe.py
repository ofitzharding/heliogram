#!/usr/bin/env python3
"""
analyze_strobe.py — did the strobe take meet its exposure condition?

The strobe design (tx120.html?strobe=1) shows code on even 120Hz refreshes and
black on odd ones. If the camera's exposure is <= 1/120 s, a 4K60 frame
integrates AT MOST one code frame plus black, and black is not an interferer:
the channel is one-signal by construction. If the exposure is longer than
8.33 ms, the window can span the black gap and catch the NEXT lit refresh too,
and two-code mixing returns - the exact thing strobe exists to kill.

The take self-reports which case happened. Every capture frame's cell
luminance is regressed on the three candidate transmit frames it could
contain:

    y  ~  a*X(s-1) + b*X(s) + c*X(s+1) + const

where s is the header's own seq. The transmitted cell matrices are known
bit-exactly (same fountain encoder, same renderer as the transmitter), so the
fit needs no ground truth from the capture beyond the header. With the
exposure condition met, the side shares are ~0. With mixing, the side shares
are material and should predict per-frame yield, as measured on IMG_7872
(share 0.00-0.01 -> 17-18/19 codewords, 0.22 -> 11/19).

Dress-rehearse on the clean transmit before running on a take.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from codec import fountain, grid
from softdec import FrameDecoder

import struct
import zlib


def truth_cells(L, enc, n_sub, SUB, data_len, seq, cache={}):
    """Bit-exact cell matrix of transmit frame `seq`, {0,1} float32."""
    if seq in cache:
        return cache[seq]
    parts = []
    for j in range(n_sub):
        b = enc.block(seq * n_sub + j)
        b = b + b"\x00" * (SUB - len(b))
        parts.append(struct.pack("<I", zlib.crc32(b) & 0xFFFFFFFF) + b)
    hdr = grid.pack_header(seq, enc.k, SUB, data_len, grid.MODE_MONO, 0, 0)
    img = grid.render_frame(L, hdr, b"".join(parts), grid.MODE_MONO, cell_px=1)
    out = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 127).astype(np.float32)
    if len(cache) > 4096:
        cache.clear()
    cache[seq] = out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--grid", default="252x163")
    ap.add_argument("--ecc", type=int, default=48)
    ap.add_argument("--payload", default=str(Path(__file__).parent.parent /
                                             "demo" / "kitten_big.png"))
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--lo", type=float, default=0.05)
    ap.add_argument("--hi", type=float, default=0.95)
    ap.add_argument("--radial", type=float, default=0.020)
    args = ap.parse_args()

    grid.set_ecc(args.ecc); grid.set_header_len(28)
    grid.set_header_centered(True); grid.set_radial(args.radial)
    gw, gh = (int(v) for v in args.grid.split("x"))
    L = grid.Layout(gw, gh)
    n_sub = grid.sub_count(L, grid.MODE_MONO)
    SUB = (255 - args.ecc) - 4
    pc = L.payload_cells
    allc = np.argwhere(np.ones((gh, gw), bool))
    data = Path(args.payload).read_bytes()
    enc = fountain.Encoder(data, SUB)
    fd = FrameDecoder(L, args.ecc, n_sub, erase=True, prml=False)

    cap = cv2.VideoCapture(args.capture)
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(tot * args.lo, tot * args.hi,
                       args.frames * 3).astype(int)

    rows = []
    print(f"{'frame':>7s} {'seq':>6s} {'mean':>6s} "
          f"{'a(s-1)':>7s} {'b(s)':>7s} {'c(s+1)':>7s} "
          f"{'side':>6s} {'R2':>5s} {'yield':>7s}")
    for fi in idxs:
        if len(rows) >= args.frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, img = cap.read()
        if not ok:
            continue
        big = img.shape[1] >= 3000
        sm = cv2.resize(img, None, fx=0.5, fy=0.5) if big else img
        Hs = grid.locate(sm, L)
        H = (((np.diag([2., 2., 1.]) @ Hs) if big else Hs)
             if Hs is not None else grid.locate(img, L))
        if H is None:
            continue
        hd, _s, _t = grid.sample_frame(img, L, H)
        if hd is None or int(hd["k"]) != enc.k:
            continue
        s = int(hd["seq"])
        if s < 1:
            continue
        if not rows:
            # cam-px/cell decides the take before any decode: the density
            # cliff sits between 12.3 (works) and 11.2 (dead). H's linear
            # part maps cells to sensor px; sqrt|det| is the mean pitch.
            pitch = float(np.sqrt(abs(np.linalg.det(H[:2, :2]))))
            print(f"  framing: {pitch:.1f} cam-px/cell "
                  f"(cliff: 12.3 works, 11.2 dead)")
        y = grid.sample_cells(img, L, H, allc).mean(axis=1).reshape(
            gh, gw).astype(np.float32)
        lum = y[pc[:, 0], pc[:, 1]]

        xm = truth_cells(L, enc, n_sub, SUB, len(data), s - 1)
        x0 = truth_cells(L, enc, n_sub, SUB, len(data), s)
        xp = truth_cells(L, enc, n_sub, SUB, len(data), s + 1)
        A = np.stack([xm[pc[:, 0], pc[:, 1]], x0[pc[:, 0], pc[:, 1]],
                      xp[pc[:, 0], pc[:, 1]], np.ones(len(pc))], axis=1)
        coef, res, *_ = np.linalg.lstsq(A, lum, rcond=None)
        a, b, c = float(coef[0]), float(coef[1]), float(coef[2])
        ss_tot = float(((lum - lum.mean()) ** 2).sum())
        r2 = 1.0 - float(res[0]) / ss_tot if len(res) else float("nan")
        tot_amp = abs(a) + abs(b) + abs(c)
        side = (abs(a) + abs(c)) / tot_amp if tot_amp > 0 else 0.0

        bits, conf = grid._mono_decide(lum, L, pc)
        nb = n_sub * 255
        bc = conf[: nb * 8].reshape(nb, 8).min(axis=1)
        blocks, _m, _cb = fd.certify(bits, bc)

        rows.append((fi, s, lum.mean(), a, b, c, side, r2, len(blocks)))
        print(f"{fi:7d} {s:6d} {lum.mean():6.1f} "
              f"{a:7.1f} {b:7.1f} {c:7.1f} {side:6.3f} {r2:5.2f} "
              f"{len(blocks):3d}/{n_sub:<3d}")
    cap.release()

    if not rows:
        print("no usable frames")
        return
    r = np.array([(x[6], x[8]) for x in rows], np.float64)
    side, yld = r[:, 0], r[:, 1]
    print(f"\n{len(rows)} frames  "
          f"side-share median {np.median(side):.3f}  "
          f"p90 {np.percentile(side, 90):.3f}  max {side.max():.3f}")
    print(f"yield mean {yld.mean():.1f}/{n_sub}  "
          f"at side<=0.05: {yld[side <= 0.05].mean() if (side <= 0.05).any() else float('nan'):.1f}  "
          f"at side>0.15: {yld[side > 0.15].mean() if (side > 0.15).any() else float('nan'):.1f}")
    if side.std() > 1e-9 and yld.std() > 1e-9:
        cc = float(np.corrcoef(side, yld)[0, 1])
        print(f"corr(side share, yield) = {cc:+.3f}")
    good = np.median(side) <= 0.05
    print("VERDICT: exposure condition "
          + ("HELD - channel ran one-signal" if good else
         "FAILED - two-code mixing returned; side share is material"))


if __name__ == "__main__":
    main()
