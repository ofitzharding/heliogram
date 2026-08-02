#!/usr/bin/env python3
"""
hdr_lab.py — how many headers does each threshold rule read?

Header yield multiplies codeword yield: a frame whose header is unreadable
contributes nothing regardless of how good its payload is. On the record take
only 22% of transmit-region frames produced a hard header, so this factor was
costing more than the payload demodulator was.

The header strip spans the FULL width of the grid, which is the worst possible
shape for a global threshold: it eats the whole left-to-right illumination
gradient. It also has 40 parity bytes on 28, so it fails only when the raw
bits are badly wrong, not marginally.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from codec import grid
from demod_lab import local_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("--frames", type=int, default=400)
    args = ap.parse_args()
    meta = np.load(args.stem + ".npz")
    gw, gh = int(meta["gw"]), int(meta["gh"])
    ecc = int(meta["ecc"])
    ys = np.lib.format.open_memmap(args.stem + ".dat.npy", mode="r")
    grid.set_ecc(ecc); grid.set_header_len(28); grid.set_header_centered(True)
    L = grid.Layout(gw, gh)
    hc = L.header_cells
    n_hdr_bits = (grid.HEADER_LEN + grid.HEADER_ECC) * 8

    fr = meta["frame"]
    if len(fr) > args.frames:
        fr = fr[np.linspace(0, len(fr) - 1, args.frames).astype(int)]
    print(f"{len(fr)} located frames, header strip = {len(hc)} cells "
          f"({n_hdr_bits} bits used)")

    def parse(bits):
        return grid.unpack_header(grid._bytes(bits[:n_hdr_bits].astype(np.uint8)))

    rules = {}
    for name in ("global-all", "global-hdr", "local9", "local15", "local31",
                 "local61", "rowlocal"):
        got = {}
        for fn in fr:
            y = ys[fn].astype(np.float32)
            hl = y[hc[:, 0], hc[:, 1]]
            if name == "global-all":
                th, _ = cv2.threshold(np.clip(y.ravel(), 0, 255).astype(np.uint8),
                                      0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                b = (hl > th)
            elif name == "global-hdr":
                th, _ = cv2.threshold(np.clip(hl, 0, 255).astype(np.uint8), 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                b = (hl > th)
            elif name == "rowlocal":
                # 1-D running mean ALONG the strip: the header is a few rows
                # spanning the full width, so the gradient it sees is
                # essentially one-dimensional.
                k = 33
                pad = np.pad(hl, k // 2, mode="reflect")
                lm = np.convolve(pad, np.ones(k) / k, mode="valid")
                b = (hl > lm)
            else:
                k = int(name[5:])
                lm = local_mean(y, k)
                b = (hl > lm[hc[:, 0], hc[:, 1]])
            h = parse(b)
            if h is not None:
                got[int(fn)] = h["seq"]
        rules[name] = got
        print(f"{name:>12s}: {len(got):4d}/{len(fr)} = {100*len(got)/len(fr):5.1f}%")

    # union: try rules in order, first that parses wins (what a decoder does)
    order = ["global-hdr", "global-all", "local15", "local31", "rowlocal", "local9"]
    u = {}
    for r in order:
        for fn, s in rules[r].items():
            u.setdefault(fn, s)
    print(f"\n{'UNION of all rules':>12s}: {len(u):4d}/{len(fr)} = "
          f"{100*len(u)/len(fr):5.1f}%")
    # consistency: rules that disagree on seq would mean silent miscorrection
    dis = sum(1 for fn in u for r in order
              if fn in rules[r] and rules[r][fn] != u[fn])
    print(f"disagreements on seq between rules: {dis} (0 = every parse agrees)")


if __name__ == "__main__":
    main()
