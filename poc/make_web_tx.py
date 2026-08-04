#!/usr/bin/env python3
"""
make_web_tx.py — dump the transmit as raw CELL BITS for a browser to present.

Why not a video. The panel is 120Hz, but ffplay fullscreen presents 58.4 fps
(measured: a 15.00s 120fps clip took 30.84s to play), so every video-based
transmit has been throwing away half the refreshes no matter what the file
says. Raising the file's frame rate does not help if the player cannot keep up.

A browser canvas can: requestAnimationFrame fires once per refresh, and this
project already measured the canvas path as the SHARPEST transmitter available
here - 0/1000 mid-band pixels against ffplay fullscreen's 19/1000 and a
1512-point ffplay window's 981/1000 (see tx.html). the reference tool presents in a
browser for the same reason.

But drawing from a <video> element re-imposes the codec's clock. So this dumps
the CELL MATRIX itself, one bit per cell, and the page blits it with
putImageData plus a nearest-neighbour integer upscale. One rAF tick = one code
frame, guaranteed, and the cells never pass through a video codec at all.

    252x163 = 41,076 cells = 5,135 bytes/frame packed.
    1800 frames = 9.2 MB, which a browser loads in one fetch.
"""
import argparse
import json
import struct
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from codec import fountain, grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", default=str(Path(__file__).parent.parent /
                                             "demo" / "kitten.png"))
    ap.add_argument("--out", default=str(Path(__file__).parent.parent /
                                        "demo" / "webtx"))
    ap.add_argument("--grid", default="252x163")
    ap.add_argument("--ecc", type=int, default=48)
    ap.add_argument("--frames", type=int, default=1800)
    ap.add_argument("--fps", type=int, default=120,
                    help="metadata only; the page presents one frame per "
                         "refresh, so this is what the DECODER should assume")
    args = ap.parse_args()

    grid.set_ecc(args.ecc); grid.set_header_len(28)
    grid.set_header_centered(True)
    gw, gh = (int(v) for v in args.grid.split("x"))
    L = grid.Layout(gw, gh)
    n_sub = grid.sub_count(L, grid.MODE_MONO)
    SUB = (255 - args.ecc) - 4
    data = Path(args.payload).read_bytes()
    enc = fountain.Encoder(data, SUB)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bits_path = out / "frames.bin"
    rate = n_sub * SUB * args.fps / 1024
    print(f"payload   {len(data):,} B ({Path(args.payload).name})")
    print(f"grid      {gw}x{gh}, {n_sub} codewords/frame of {SUB} B, ecc {args.ecc}")
    print(f"CEILING   {rate:.1f} KB/s at {args.fps} fps")
    print(f"          200 KB/s needs {100*200.0/rate:.1f}% codeword yield")
    print(f"frames    {args.frames} ({args.frames/args.fps:.1f}s per loop at "
          f"{args.fps} fps)")

    with open(bits_path, "wb") as f:
        for seq in range(args.frames):
            parts = []
            for j in range(n_sub):
                b = enc.block(seq * n_sub + j)
                b = b + b"\x00" * (SUB - len(b))
                parts.append(struct.pack("<I", zlib.crc32(b) & 0xFFFFFFFF) + b)
            hdr = grid.pack_header(seq, enc.k, SUB, len(data),
                                   grid.MODE_MONO, 0, 0)
            # cell_px=1 makes the rendered image the CELL MATRIX itself
            img = grid.render_frame(L, hdr, b"".join(parts), grid.MODE_MONO,
                                    cell_px=1)
            cells = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 127).astype(np.uint8)
            f.write(np.packbits(cells.ravel()).tobytes())
            if seq % 100 == 0:
                print(f"  {seq}/{args.frames}", end="\r")

    meta = dict(gw=gw, gh=gh, frames=args.frames, fps=args.fps,
                ecc=args.ecc, n_sub=n_sub, sub=SUB, k=enc.k,
                file_size=len(data), name=Path(args.payload).name,
                bytes_per_frame=int(np.ceil(gw * gh / 8)),
                ceiling_kbs=round(rate, 1))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    sz = bits_path.stat().st_size
    print(f"\nwrote {bits_path} ({sz/1e6:.1f} MB) and {out/'meta.json'}")
    print(f"\nserve and open:\n"
          f"  cd {Path(__file__).parent.parent} && python3 -m http.server 8000\n"
          f"  then open  http://localhost:8000/tx120.html  and press F for "
          f"fullscreen")


if __name__ == "__main__":
    main()
