#!/usr/bin/env python3
"""
probe.py — measure the display->camera channel from a real capture, then
report the constellation it can actually carry.

This is leg one of the thesis: the link measures itself instead of assuming
a modulation. Feed it a capture of a probe video (or any decoded grid whose
true symbols we can regenerate) and it estimates, per luminance level and
per screen region, how separable the levels are — i.e. how many bits per
cell this specific screen + camera + geometry supports right now.

Usage:
    python3 probe.py capture.mov --grid 180x100 --truth demo/payload.png \
            --mode gray4 --ecc 64

Output: measured level centers, their overlap, achievable bits/cell overall
and in the worst region, and a verdict on whether gray4 (2 bit) clears the
bar or the channel is a 1-bit (mono) channel today.
"""
import argparse
import struct
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from codec import fountain, grid
from reedsolo import RSCodec


def true_symbols_for(seq, layout, enc, block_size, mode):
    block = enc.block(seq)
    block = block + b"\x00" * (block_size - len(block))
    payload = struct.pack("<I", zlib.crc32(block) & 0xFFFFFFFF) + block
    coded = bytes(RSCodec(grid.PAYLOAD_ECC).encode(payload))
    pb = np.unpackbits(np.frombuffer(coded, dtype=np.uint8))
    if mode == grid.MODE_GRAY4:
        pb = np.concatenate([pb, np.zeros((-len(pb)) % 2, dtype=np.uint8)])
        pairs = pb.reshape(-1, 2)
        return np.array([grid.GRAY4_SYM[(int(a), int(b))] for a, b in pairs])
    return pb  # mono: symbol == bit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--grid", default="180x100")
    ap.add_argument("--truth", required=True, help="original payload file")
    ap.add_argument("--mode", default="gray4", choices=["gray4", "mono"])
    ap.add_argument("--ecc", type=int, default=64)
    ap.add_argument("--frames", type=int, default=40)
    args = ap.parse_args()
    grid.set_ecc(args.ecc)

    gw, gh = (int(v) for v in args.grid.split("x"))
    layout = grid.Layout(gw, gh)
    mode = grid.MODE_GRAY4 if args.mode == "gray4" else grid.MODE_MONO
    nlev = 4 if mode == grid.MODE_GRAY4 else 2

    data = Path(args.truth).read_bytes()
    block_size = layout.payload_capacity_bytes(mode) - 4
    enc = fountain.Encoder(data, block_size)

    cap = cv2.VideoCapture(args.capture)
    lums_by_level = [[] for _ in range(nlev)]
    lums_by_region = [[[] for _ in range(nlev)] for _ in range(6)]  # column-sixths
    n = used = 0
    cells = None
    proto = None
    templates = None
    while used < args.frames:
        ok, img = cap.read()
        if not ok:
            break
        n += 1
        if n % 3:
            continue
        if img.shape[1] >= 3000:
            small = cv2.resize(img, None, fx=0.5, fy=0.5)
            Hs = grid.locate(small, layout)
            H = np.diag([2.0, 2.0, 1.0]) @ Hs if Hs is not None else None
        else:
            H = grid.locate(img, layout)
        if H is None:
            continue
        header, samples, st = grid.sample_frame(img, layout, H)
        if header is not None and proto is None:
            proto = header
            templates = grid.header_templates(proto, min(4000, 12 * proto["k"]))
        if header is None and templates is not None:
            # align by maximum-likelihood seq detection when the header strip
            # is unreadable — the whole point of the estimator receiver
            hl = grid.sample_cells(img, layout, H, layout.header_cells).mean(axis=1)
            seq, margin = grid.ml_header_seq(hl, templates)
            if margin >= 6.0:
                header = dict(proto, seq=seq)
        if header is None:
            continue
        used += 1
        if cells is None:
            cells = layout.payload_cells
        lum = samples.mean(axis=1)
        true = true_symbols_for(header["seq"], layout, enc, block_size, mode)
        m = min(len(true), len(lum))
        col6 = cells[:m, 1] * 6 // gw
        for s in range(nlev):
            sel = true[:m] == s
            lums_by_level[s].extend(lum[:m][sel].tolist())
            for reg in range(6):
                lums_by_region[reg][s].extend(lum[:m][sel & (col6 == reg)].tolist())

    if used == 0:
        sys.exit("no frames with readable headers — can't align to ground truth")

    print(f"measured on {used} frames\n")
    centers = np.array([np.median(x) if x else np.nan for x in lums_by_level])
    stds = np.array([np.std(x) if x else np.nan for x in lums_by_level])
    print("level   center   std    count")
    for s in range(nlev):
        print(f"  {s}     {centers[s]:6.1f}  {stds[s]:5.1f}  {len(lums_by_level[s]):6d}")

    # separability: min gap between adjacent centers, in units of pooled std
    gaps = np.diff(centers)
    pooled = (stds[:-1] + stds[1:]) / 2
    seps = gaps / np.maximum(pooled, 1e-6)
    print(f"\nadjacent-level separation (sigmas): "
          f"{', '.join(f'{x:.1f}' for x in seps)}")
    worst = np.min(seps)
    print(f"worst adjacent separation: {worst:.1f} sigma")

    # per-region worst separation — where does the channel choke
    print("\nworst adjacent separation by column-sixth:")
    region_line = []
    for reg in range(6):
        rc = np.array([np.median(lums_by_region[reg][s])
                       if lums_by_region[reg][s] else np.nan for s in range(nlev)])
        rs = np.array([np.std(lums_by_region[reg][s])
                       if lums_by_region[reg][s] else np.nan for s in range(nlev)])
        rsep = np.diff(rc) / np.maximum((rs[:-1] + rs[1:]) / 2, 1e-6)
        region_line.append(np.nanmin(rsep))
    print("  " + "  ".join(f"{x:.1f}" for x in region_line))

    # Verdict. ~3 sigma between adjacent levels keeps per-symbol error low
    # enough that ECC is affordable; below ~2 the level collapses.
    print()
    if mode == grid.MODE_GRAY4:
        if worst >= 3.0:
            print("VERDICT: 4-level (2 bit/cell) is comfortable on this channel.")
        elif worst >= 2.0:
            print("VERDICT: 4-level marginal — usable with heavy ECC or a "
                  "spatially-adaptive drop to 1 bit in the worst region.")
        else:
            print("VERDICT: 4-level collapses; this is a 1-bit (mono) channel "
                  "at this geometry. The adaptive move is to widen cells or "
                  "shrink the grid, not add levels.")


if __name__ == "__main__":
    main()
