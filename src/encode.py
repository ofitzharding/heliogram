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
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from codec import fountain, grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output", help=".mp4 path")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--mode", default="mono", choices=["mono", "color8", "gray4", "color4", "adaptive"])
    ap.add_argument("--cell-px", type=int, default=12)
    ap.add_argument("--grid", default="120x68")
    ap.add_argument("--overhead", type=float, default=1.6)
    ap.add_argument("--frames-dir", help="also dump PNG frames here")
    ap.add_argument("--strobe", action="store_true",
                    help="interleave a black frame after every code frame and "
                         "double the container fps. The display becomes the "
                         "shutter: a handheld camera's long auto-exposure "
                         "integrates one short flash plus darkness, so hand "
                         "motion during the dark interval adds no blur. "
                         "Transmit-side motion-freezing on an unmodified "
                         "screen; the camera keeps all its light (the black "
                         "interval contributes nothing but stillness)")
    ap.add_argument("--zone-w", type=int, default=12,
                    help="adaptive mode: edge-band width in cells")
    ap.add_argument("--zones", default="mono,color4,color4",
                    help="adaptive mode: alphabets for edge,mid,center zones")
    ap.add_argument("--subblock", action="store_true",
                    help="make each RS codeword its own fountain symbol. A frame "
                         "then contributes every codeword that survived instead "
                         "of nothing when one blows its budget. Measured 1.30x "
                         "yield on real handheld footage.")
    ap.add_argument("--ecc", type=int, default=32,
                    help="RS parity bytes per 255-byte codeword; "
                         "corrects ecc/2 byte errors")
    args = ap.parse_args()
    grid.set_ecc(args.ecc)

    mode = {"mono": grid.MODE_MONO, "color8": grid.MODE_COLOR8,
            "gray4": grid.MODE_GRAY4, "color4": grid.MODE_COLOR4,
            "adaptive": grid.MODE_ADAPTIVE}[args.mode]
    zone_w, zone_modes = 0, 0
    if mode == grid.MODE_ADAPTIVE:
        zone_w = args.zone_w
        zmap = {"mono": 0, "color4": 1}
        for i, name in enumerate(args.zones.split(",")):
            zone_modes |= zmap[name.strip()] << (2 * i)
    gw, gh = (int(v) for v in args.grid.split("x"))
    layout = grid.Layout(gw, gh)
    # 4 bytes of the frame carry a CRC32 of the fountain block: RS(255,223)
    # can mis-correct a heavily damaged codeword into a wrong-but-valid one,
    # and one bad block silently poisons the whole fountain-decoded file.
    block_size = layout.payload_capacity_bytes(mode, zone_w, zone_modes) - 4

    data = Path(args.input).read_bytes()
    enc = fountain.Encoder(data, block_size)
    n_frames = max(enc.k + 8, int(enc.k * args.overhead))

    sub_enc = sub_size = n_sub = None
    if args.subblock:
        # one fountain symbol per RS codeword; 4 bytes of each go to its CRC
        n_sub = grid.sub_count(layout, mode, zone_w, zone_modes)
        sub_size = (255 - args.ecc) - 4
        sub_enc = fountain.Encoder(data, sub_size)
        n_frames = max(int(np.ceil(sub_enc.k / n_sub)) + 4,
                       int(np.ceil(sub_enc.k * args.overhead / n_sub)))
        print(f"subblock        {n_sub} symbols/frame of {sub_size} B "
              f"(k={sub_enc.k} symbols total)")

    print(f"file            {len(data):,} bytes")
    print(f"grid            {gw}x{gh}  mode={args.mode}  block={block_size} B")
    print(f"k               {enc.k} source blocks")
    print(f"frames          {n_frames} ({args.overhead:.1f}x overhead)")
    print(f"raw rate        {block_size * args.fps / 1024:.1f} KB/s at {args.fps} fps "
          f"(before capture losses)")

    size = (gw * args.cell_px, gh * args.cell_px)
    out_fps = args.fps * 2 if args.strobe else args.fps
    vw = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, size)
    black = None
    frames_dir = None
    if args.frames_dir:
        frames_dir = Path(args.frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)

    for seq in range(n_frames):
        if args.subblock:
            # Each RS codeword is its own fountain symbol. A frame's sub-block
            # j carries fountain symbol (seq * n_sub + j), so a frame damaged
            # in one region still contributes every codeword that survived.
            parts = []
            for j in range(n_sub):
                s = seq * n_sub + j
                b = sub_enc.block(s)
                b = b + b"\x00" * (sub_size - len(b))
                parts.append(struct.pack("<I", zlib.crc32(b) & 0xFFFFFFFF) + b)
            payload = b"".join(parts)
        else:
            block = enc.block(seq)
            block = block + b"\x00" * (block_size - len(block))
            payload = struct.pack("<I", zlib.crc32(block) & 0xFFFFFFFF) + block
        header = grid.pack_header(seq, enc.k, block_size, len(data), mode,
                                  zone_w, zone_modes)
        img = grid.render_frame(layout, header, payload, mode, args.cell_px,
                                zone_w, zone_modes)
        vw.write(img)
        if args.strobe:
            if black is None:
                import numpy as _np
                black = _np.zeros_like(img)
            vw.write(black)
        if frames_dir:
            cv2.imwrite(str(frames_dir / f"f{seq:05d}.png"), img)
        if seq % 50 == 0:
            print(f"  rendered {seq}/{n_frames}", end="\r")
    vw.release()
    print(f"\nwrote {args.output}")
    # The decoder cannot infer grid dimensions — it needs them to find the
    # header in the first place. Print the exact command rather than let a
    # mismatch surface as a silent "header ok 0".
    print(f"\ndecode a capture of this with:\n"
          f"  python3 decode.py <capture.mov> <out.bin> --grid {gw}x{gh}")


if __name__ == "__main__":
    main()
