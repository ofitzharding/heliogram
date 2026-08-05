#!/usr/bin/env python3
"""
exp_certcancel.py — cancel the KNOWN interferer, keep the strong signal.

NOT the refuted SIC (pass 23). That experiment subtracted the STRONG
component to decode the WEAK one, and the oracle showed the weak component
sits under the sensor noise floor (residual d' 0.04-1.20). Dead physics.

This is the reverse direction, which that oracle does not touch. On a strobe
frame whose exposure straddled, the interferer is the PREVIOUS code frame:

    y ~ a*X(s-1) + b*X(s) + const,   a ~ 0.2-0.45 * b   (measured, §27)

X(s-1) is not an estimate. One camera frame earlier the receiver certified
its codewords through RS+CRC32, and re-encoding certified blocks reproduces
the transmitted cells exactly (the same structural argument as the certified
-label donor and CAG). Subtracting a*X(s-1) is cancellation of a proven
quantity: no error propagation is possible, and the component being decoded
afterwards is the STRONG one, whose d' the subtraction can only raise.

ORACLE MODE (this script): X(s-1) comes from the encoder, standing in for
"the previous frame certified fully" - the demonstrated behaviour at side
share ~0 (19/19 on every clean probe). This measures the CEILING of the
mechanism, exactly as the SIC oracle measured its floor. A production
harvest would re-encode the previous camera frame's certified codewords and
subtract only cells belonging to them.

Dress rehearsal: on the clean transmit the fit returns a ~ 0, so
cancellation must be a no-op (no frame loses a codeword).
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from codec import fountain, grid
from softdec import FrameDecoder
from analyze_strobe import truth_cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--grid", default="252x163")
    ap.add_argument("--ecc", type=int, default=48)
    ap.add_argument("--payload", default=str(Path(__file__).parent.parent /
                                             "demo" / "kitten.png"))
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

    scratch = {}
    nb = n_sub * 255
    rows = []
    print(f"{'frame':>7s} {'seq':>6s} {'side':>6s} "
          f"{'base':>7s} {'cancel':>7s} {'gain':>5s}")
    for fi in idxs:
        if len(rows) >= args.frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, img = cap.read()
        if not ok:
            continue
        sm = (cv2.resize(img, None, fx=0.5, fy=0.5)
              if img.shape[1] >= 3000 else img)
        Hs = grid.locate(sm, L)
        H = ((np.diag([2., 2., 1.]) @ Hs) if Hs is not None
             else grid.locate(img, L))
        if H is None:
            continue
        hd, _s, _t = grid.sample_frame(img, L, H)
        if hd is None or int(hd["k"]) != enc.k:
            continue
        s = int(hd["seq"])
        if s < 1:
            continue
        y = grid.sample_cells(img, L, H, allc).mean(axis=1).reshape(
            gh, gw).astype(np.float32)
        lum = y[pc[:, 0], pc[:, 1]]

        xm = truth_cells(L, enc, n_sub, SUB, len(data), s - 1,
                         cache=scratch)[pc[:, 0], pc[:, 1]]
        x0 = truth_cells(L, enc, n_sub, SUB, len(data), s,
                         cache=scratch)[pc[:, 0], pc[:, 1]]
        xp = truth_cells(L, enc, n_sub, SUB, len(data), s + 1,
                         cache=scratch)[pc[:, 0], pc[:, 1]]
        scratch.clear()
        A = np.stack([xm, x0, xp, np.ones(len(pc))], axis=1)
        coef, *_ = np.linalg.lstsq(A, lum, rcond=None)
        a, b, c = float(coef[0]), float(coef[1]), float(coef[2])
        tot_amp = abs(a) + abs(b) + abs(c)
        side = (abs(a) + abs(c)) / tot_amp if tot_amp > 0 else 0.0

        # base path
        bits, conf = grid._mono_decide(lum, L, pc)
        bc = conf[: nb * 8].reshape(nb, 8).min(axis=1)
        base = len(fd.certify(bits, bc)[0])

        # cancel the certified interferer(s); the strong component stays
        lum_c = lum - a * xm - c * xp
        bits2, conf2 = grid._mono_decide(lum_c, L, pc)
        bc2 = conf2[: nb * 8].reshape(nb, 8).min(axis=1)
        canc = len(fd.certify(bits2, bc2)[0])

        rows.append((side, base, canc))
        print(f"{fi:7d} {s:6d} {side:6.3f} {base:3d}/{n_sub:<3d} "
              f"{canc:3d}/{n_sub:<3d} {canc-base:+4d}")
    cap.release()

    if not rows:
        print("no usable frames")
        return
    r = np.array(rows, np.float64)
    side, base, canc = r[:, 0], r[:, 1], r[:, 2]
    print(f"\n{len(rows)} frames   base {base.mean():.1f}/{n_sub}   "
          f"cancelled {canc.mean():.1f}/{n_sub}")
    m = side > 0.10
    if m.any():
        print(f"mixed frames (side>0.10, n={int(m.sum())}): "
              f"base {base[m].mean():.1f} -> cancelled {canc[m].mean():.1f}  "
              f"({(canc[m].mean()/max(base[m].mean(),1e-9)):.2f}x)")
    cl = ~m
    if cl.any():
        print(f"clean frames (side<=0.10, n={int(cl.sum())}): "
              f"base {base[cl].mean():.1f} -> cancelled {canc[cl].mean():.1f} "
              f"(must be a no-op)")
    lost = int((canc < base).sum())
    print(f"frames losing codewords after cancel: {lost} "
          f"(any loss is a defect: subtraction of a proven quantity)")


if __name__ == "__main__":
    main()
