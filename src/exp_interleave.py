#!/usr/bin/env python3
"""
exp_interleave.py — what the row-band codeword layout costs, exactly.

Measured on IMG_7870: the fraction of payload cells sitting in the middle of
the eye climbs monotonically from ~9% in the top rows to ~23% in the bottom
rows, and the profile is the SAME on every frame across 1.4 seconds. That is
not exposure straddle (which would be uniform down the frame) and not a
rolling-shutter tear (which would be clean at both ends with a narrow
transition). It is a fixed spatial quality gradient of the optical path.

The codec lays codeword j over a contiguous horizontal BAND of cells. So the
per-codeword error count is drawn from that band's error rate, and a stable
2.5x gradient means the bottom codewords carry ~2.5x the errors of the top
ones - while every codeword is given exactly the same 24-byte correction
budget.

Reed-Solomon failure is a THRESHOLD on error count, so the number of codewords
that survive is a concave function of how the errors are distributed. By
Jensen, spreading a fixed error budget evenly over codewords weakly dominates
concentrating it - strictly, whenever some codewords sit above threshold while
others sit below with slack. This measures both sides of that inequality on
real frames:

    observed : codewords whose ACTUAL error count is within budget
    equalised: codewords that would survive if the SAME total errors were
               spread uniformly, i.e. what a full-frame cell interleaver buys

Truth comes from the fountain given seq, which is legitimate for a diagnostic
(it measures the channel) and would not be for a decoder claim.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("--payload", default=str(Path(__file__).parent.parent /
                                             "demo" / "payload.png"))
    ap.add_argument("--grid", default="252x140")
    ap.add_argument("--ecc", type=int, default=48)
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--frames", type=int, default=80)
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

    meta = np.load(args.stem + ".npz")
    ys = np.lib.format.open_memmap(args.stem + ".dat.npy", mode="r")
    ok = meta["seq"] >= 0
    if args.k:
        ok &= meta["k"] == args.k
    rows = np.flatnonzero(ok)[: args.frames]
    print(f"{args.grid} ecc={args.ecc}: {n_sub} codewords/frame, "
          f"RS corrects {budget} byte errors of 255\n")

    per_cw = []
    obs = eq = 0
    tot_frames = 0
    for i in rows:
        fn, seq = int(meta["frame"][i]), int(meta["seq"][i])
        y = ys[fn].astype(np.float32)
        parts = []
        for j in range(n_sub):
            b = enc.block(seq * n_sub + j)
            b = b + b"\x00" * (SUB - len(b))
            parts.append(struct.pack("<I", zlib.crc32(b) & 0xFFFFFFFF) + b)
        hdr = grid.pack_header(seq, enc.k, SUB, len(data), grid.MODE_MONO, 0, 0)
        tr = grid.render_frame(L, hdr, b"".join(parts), grid.MODE_MONO, cell_px=1)
        xt = (cv2.cvtColor(tr, cv2.COLOR_BGR2GRAY) > 127).astype(np.uint8)
        tb = xt[pc[:, 0], pc[:, 1]]
        lum = y[pc[:, 0], pc[:, 1]]
        bits, _c = grid._mono_decide(lum, L, pc)
        n = n_sub * 255 * 8
        err = (bits[:n] != tb[:n]).reshape(n_sub * 255, 8).any(axis=1)
        cnt = err.reshape(n_sub, 255).sum(axis=1)
        per_cw.append(cnt)
        obs += int((cnt <= budget).sum())
        # same TOTAL byte errors, spread evenly over the frame's codewords
        share = cnt.sum() / n_sub
        eq += int(n_sub * (share <= budget))
        tot_frames += 1

    A = np.array(per_cw)
    print("mean byte errors per codeword, by position down the frame:")
    for j in range(n_sub):
        v = A[:, j].mean()
        bar = "#" * int(min(60, v))
        print(f"  cw {j:2d} (rows {j*gh//n_sub:3d}-{(j+1)*gh//n_sub:3d}) "
              f"{v:6.1f} {bar}")
    print(f"\ntotal codewords      {tot_frames*n_sub}")
    print(f"observed within budget  {obs:5d}  ({100*obs/(tot_frames*n_sub):.1f}%)")
    print(f"if errors were EQUALISED {eq:5d}  "
          f"({100*eq/(tot_frames*n_sub):.1f}%)")
    if obs:
        print(f"\ninterleaving gain: {eq/max(obs,1):.2f}x codeword yield")
    worst = A.mean(axis=0).max() / max(A.mean(axis=0).min(), 1e-9)
    print(f"worst/best band error ratio: {worst:.2f}x")
    print(f"frames where the frame TOTAL is within n_sub*budget "
          f"(i.e. interleaving could in principle save the whole frame): "
          f"{int((A.sum(axis=1) <= n_sub*budget).sum())}/{tot_frames}")


if __name__ == "__main__":
    main()
