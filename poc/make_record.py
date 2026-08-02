#!/usr/bin/env python3
"""
make_record.py — the record transmit, sized to the PANEL instead of to habit.

The previous record transmit was 252x140 cells at 12 px/cell, i.e. 3024x1680
on a 3024x1964 display. 284 rows of panel - 14% of the screen the camera was
already pointed at, already in focus, already correctly exposed - were black
margin. At ecc=48 that grid carries 190.3 KB/s at a yield of 100%, so 200 KB/s
was arithmetically out of reach of that video no matter what the receiver did.

252x163 at the same 12 px/cell is 3024x1956. Same cell size, so the camera
still sees ~13.4 px/cell - the sampling density that decodes - and the extra
rows are free:

    252x140  ecc 48   16 codewords/frame   190.3 KB/s   needs 105% yield
    252x163  ecc 48   19 codewords/frame   226.0 KB/s   needs  88.5% yield

Density was never the lever. 350x194 puts 9.0 camera-px/cell on the sensor and
yields 0/32 codewords with the geometry verified correct at 97% structure
agreement (exp_dense.py), so going denser trades a factor the receiver cannot
recover. Going TALLER costs nothing at all.

The countdown is built from THIS transmit's own frames. The old one was
rendered from a different file, and because a countdown frame carries a
perfectly valid header, "first header wins" learned k=5525/1.12MB from it and
tried to rebuild that from a 277 KB transmission.
"""
import argparse
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from codec import fountain, grid

PW, PH = 3024, 1964


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", default=str(Path(__file__).parent.parent /
                                             "demo" / "payload.png"))
    ap.add_argument("--out", default=str(Path(__file__).parent.parent /
                                         "demo" / "_tx_record163.mp4"))
    ap.add_argument("--grid", default="252x163")
    ap.add_argument("--cell-px", type=int, default=12)
    ap.add_argument("--ecc", type=int, default=48)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--frames", type=int, default=900,
                    help="distinct code frames; the script loops the file, so "
                         "this only has to exceed what one transfer consumes")
    ap.add_argument("--crf", type=int, default=10)
    ap.add_argument("--lead-seconds", type=float, default=22.0,
                    help="lock-in lead: this transmit's own frames "
                         "with a countdown drawn over them")
    args = ap.parse_args()
    build(Path(args.payload).read_bytes(), args.out, args.grid, args.cell_px,
          args.ecc, args.fps, args.frames, args.crf, args.lead_seconds,
          Path(args.payload).name)


def build(data, out, grid_spec="252x163", cell_px=12, ecc=48, fps=60,
          frames=0, crf=10, lead_seconds=22.0, label="payload"):
    """Render `data` to a transmit video plus its lock-in lead.

    frames=0 sizes the loop from the payload: enough distinct frames to carry
    the file several times over, so a receiver that starts anywhere in the loop
    still collects a full set without waiting for a wrap.
    """
    class A:
        pass
    args = A()
    args.out, args.grid, args.cell_px, args.ecc = out, grid_spec, cell_px, ecc
    args.fps, args.crf, args.lead_seconds = fps, crf, lead_seconds
    args.payload = label

    grid.set_ecc(args.ecc); grid.set_header_len(28)
    grid.set_header_centered(True)
    gw, gh = (int(v) for v in args.grid.split("x"))
    L = grid.Layout(gw, gh)
    n_sub = grid.sub_count(L, grid.MODE_MONO)
    SUB = (255 - args.ecc) - 4
    enc = fountain.Encoder(data, SUB)
    # Enough distinct frames to carry the file ~4x over, so a receiver that
    # starts anywhere in the loop still gets a full set before it wraps.
    args.frames = frames or max(600, int(np.ceil(4.0 * enc.k / n_sub)))
    W, H = gw * args.cell_px, gh * args.cell_px
    if W > PW or H > PH:
        sys.exit(f"{W}x{H} does not fit the {PW}x{PH} panel")

    rate = n_sub * SUB * args.fps / 1024
    need = 1.05 * enc.k / n_sub
    print(f"payload      {len(data):,} B  ({label})")
    print(f"grid         {gw}x{gh} @ {args.cell_px}px = {W}x{H} on {PW}x{PH} "
          f"({100*W*H/(PW*PH):.1f}% of the panel)")
    print(f"codewords    {n_sub}/frame of {SUB} B, ecc {args.ecc}")
    print(f"fountain     k={enc.k} symbols; a transfer consumes ~{need:.0f} "
          f"frames at full yield")
    print(f"CEILING      {rate:.1f} KB/s at {args.fps} fps and 100% yield")
    print(f"             200 KB/s needs {100*200.0/rate:.1f}% codeword yield")
    print(f"frames       {args.frames} distinct "
          f"({args.frames/args.fps:.1f}s per loop, "
          f"{args.frames*n_sub/enc.k:.1f}x fountain overhead)\n")

    x0, y0 = (PW - W) // 2, (PH - H) // 2
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{PW}x{PH}", "-r", str(args.fps), "-i", "-",
           "-c:v", "libx264", "-preset", "fast", "-crf", str(args.crf),
           "-pix_fmt", "yuv420p", args.out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    canvas = np.zeros((PH, PW, 3), np.uint8)

    # LOCK-IN LEAD. The camera needs a target to lock AE/AF onto and the
    # operator needs to see when to do it, but a SEPARATE countdown clip is
    # what caused the k=5525 disaster: rendered from another transmit, it
    # carried a perfectly valid header for a different file, and "first header
    # wins" then tried to rebuild a 1.12MB file out of a 277KB transmission.
    #
    # So the lead is this transmit's OWN frames with a counter drawn over
    # them. Luminance-matched by construction (it IS the code), it advertises
    # the correct file, and the overlay only costs the codewords it covers -
    # which are the settling-transient frames the fountain was going to get
    # nothing from anyway. Overlaid cells fail RS and are rejected by CRC, so
    # they cannot corrupt anything either.
    # Written as its OWN file so the operator sees the countdown once, at the
    # start, instead of every time the code loop wraps. Same transmit, same
    # header, same file - so unlike the old separate countdown it cannot teach
    # the receiver the wrong k.
    lead_out = args.out.replace(".mp4", "_lead.mp4")
    pl = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{PW}x{PH}", "-r", str(args.fps), "-i", "-",
         "-c:v", "libx264", "-preset", "fast", "-crf", str(args.crf),
         "-pix_fmt", "yuv420p", lead_out], stdin=subprocess.PIPE)
    p, p_code = pl, p
    lead = int(args.lead_seconds * args.fps)
    for i in range(lead):
        seq = i % args.frames
        parts = []
        for j in range(n_sub):
            b = enc.block(seq * n_sub + j)
            b = b + b"\x00" * (SUB - len(b))
            parts.append(struct.pack("<I", zlib.crc32(b) & 0xFFFFFFFF) + b)
        hdr = grid.pack_header(seq, enc.k, SUB, len(data), grid.MODE_MONO, 0, 0)
        img = grid.render_frame(L, hdr, b"".join(parts), grid.MODE_MONO,
                                cell_px=args.cell_px)
        canvas[:] = 0
        canvas[y0:y0 + H, x0:x0 + W] = img
        left = args.lead_seconds - i / args.fps
        txt = f"{left:0.0f}" if left > 3 else "HOLD STILL"
        sc = 22.0 if left > 3 else 6.0
        (tw, th_), _b = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, sc, 40)
        org = ((PW - tw) // 2, (PH + th_) // 2)
        cv2.putText(canvas, txt, org, cv2.FONT_HERSHEY_SIMPLEX, sc,
                    (0, 0, 0), 90, cv2.LINE_AA)
        cv2.putText(canvas, txt, org, cv2.FONT_HERSHEY_SIMPLEX, sc,
                    (255, 255, 255), 40, cv2.LINE_AA)
        msg = "TAP-HOLD TO LOCK AE/AF NOW - do NOT touch the exposure slider"
        (mw, mh), _b2 = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 2.2, 6)
        cv2.putText(canvas, msg, ((PW - mw) // 2, PH - 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 16, cv2.LINE_AA)
        cv2.putText(canvas, msg, ((PW - mw) // 2, PH - 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 6,
                    cv2.LINE_AA)
        p.stdin.write(canvas.tobytes())
        if i % 50 == 0:
            print(f"  lead-in {i}/{lead}", end="\r")
    pl.stdin.close()
    pl.wait()
    canvas[:] = 0
    p = p_code

    for seq in range(args.frames):
        parts = []
        for j in range(n_sub):
            b = enc.block(seq * n_sub + j)
            b = b + b"\x00" * (SUB - len(b))
            parts.append(struct.pack("<I", zlib.crc32(b) & 0xFFFFFFFF) + b)
        hdr = grid.pack_header(seq, enc.k, SUB, len(data), grid.MODE_MONO, 0, 0)
        img = grid.render_frame(L, hdr, b"".join(parts), grid.MODE_MONO,
                                cell_px=args.cell_px)
        canvas[y0:y0 + H, x0:x0 + W] = img
        p.stdin.write(canvas.tobytes())
        if seq % 50 == 0:
            print(f"  rendered {seq}/{args.frames}", end="\r")
    p.stdin.close()
    p.wait()
    sz = Path(args.out).stat().st_size
    lz = Path(args.out.replace(".mp4", "_lead.mp4")).stat().st_size
    print(f"\nwrote {args.out} ({sz/1e6:.0f} MB)")
    print(f"wrote {args.out.replace('.mp4', '_lead.mp4')} ({lz/1e6:.0f} MB, "
          f"{args.lead_seconds:.0f}s lock-in)")
    return dict(grid=f"{gw}x{gh}", ecc=args.ecc, k=enc.k, n_sub=n_sub,
                frames=args.frames, ceiling=rate, lead=lead_out)


if __name__ == "__main__":
    main()
