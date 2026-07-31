#!/usr/bin/env python3
"""
encode.py — file -> animated grid-code video (and/or PNG frames).

    python3 encode.py input.bin out.mp4 [--fps 30] [--mode mono|color8]
                      [--cell-px 12] [--overhead 1.6] [--frames-dir dir]

`--overhead 1.6` renders 1.6x the minimum number of fountain blocks; the
receiver only needs ~1.05-1.15x of them to decode, so the loop can be
recorded from any starting point without a sync protocol. For live
transmission, loop the video fullscreen (QuickTime: cmd-L for loop, then
fullscreen; or ffplay -loop 0).
"""
import argparse
import struct
import sys
import zlib
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))
from codec import fountain, grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output", help=".mp4 path")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--mode", default="mono", choices=["mono", "color8"])
    ap.add_argument("--cell-px", type=int, default=12)
    ap.add_argument("--grid", default="120x68")
    ap.add_argument("--overhead", type=float, default=1.6)
    ap.add_argument("--frames-dir", help="also dump PNG frames here")
    args = ap.parse_args()

    mode = grid.MODE_MONO if args.mode == "mono" else grid.MODE_COLOR8
    gw, gh = (int(v) for v in args.grid.split("x"))
    layout = grid.Layout(gw, gh)
    # 4 bytes of the frame carry a CRC32 of the fountain block: RS(255,223)
    # can mis-correct a heavily damaged codeword into a wrong-but-valid one,
    # and one bad block silently poisons the whole fountain-decoded file.
    block_size = layout.payload_capacity_bytes(mode) - 4

    data = Path(args.input).read_bytes()
    enc = fountain.Encoder(data, block_size)
    n_frames = max(enc.k + 8, int(enc.k * args.overhead))

    print(f"file            {len(data):,} bytes")
    print(f"grid            {gw}x{gh}  mode={args.mode}  block={block_size} B")
    print(f"k               {enc.k} source blocks")
    print(f"frames          {n_frames} ({args.overhead:.1f}x overhead)")
    print(f"raw rate        {block_size * args.fps / 1024:.1f} KB/s at {args.fps} fps "
          f"(before capture losses)")

    size = (gw * args.cell_px, gh * args.cell_px)
    vw = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, size)
    frames_dir = None
    if args.frames_dir:
        frames_dir = Path(args.frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)

    for seq in range(n_frames):
        block = enc.block(seq)
        block = block + b"\x00" * (block_size - len(block))
        payload = struct.pack("<I", zlib.crc32(block) & 0xFFFFFFFF) + block
        header = grid.pack_header(seq, enc.k, block_size, len(data), mode)
        img = grid.render_frame(layout, header, payload, mode, args.cell_px)
        vw.write(img)
        if frames_dir:
            cv2.imwrite(str(frames_dir / f"f{seq:05d}.png"), img)
        if seq % 50 == 0:
            print(f"  rendered {seq}/{n_frames}", end="\r")
    vw.release()
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
