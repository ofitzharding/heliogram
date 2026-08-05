#!/usr/bin/env python3
"""
make_lumen_probe.py — sweep the TRANSMIT peak, so one take finds the operating
point instead of one guess per take.

IMG_7879 did not measure gray4. It measured a saturated sensor: with AE locked,
the panel at full white gave p5=32 (black lifted by bloom) and p50=189 against
an eye midpoint of 135, and MONO - the control, on the same take, same optics -
fell to 23.6% from the 93.4% this rig gives on a good take. gray4 read 0%, but
nothing about gray4 was tested; its two middle levels are simply the first
casualty of a compressed transfer curve.

The exposure is not adjustable from here: the phone's AE is locked by the
operator and the slider is known to make things worse. But the light level is
a TRANSMITTER property, and the transmitter is fully controllable. So sweep it.

Each section renders gray4 at a different peak white while holding grid, cell
size, framing and hold fixed, so the only variable is how much light the link
puts into the sensor. A mono/48 section at full peak is carried in every loop
as the control: if mono does not recover its ~90% on this take, the take is
bad and no gray4 row means anything - the same reasoning that makes IMG_7879
inconclusive rather than negative.

Costs nothing on the receiver. The gray4 demodulator learns its four levels by
k-means from the frame's own data, so it adapts to whatever peak arrives.
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
# (label, mode, ecc, peak). Mono control first so a wrap always lands near one.
CONFIGS = [("mono", grid.MODE_MONO, 48, 255),
           ("gray4", grid.MODE_GRAY4, 64, 255),
           ("gray4", grid.MODE_GRAY4, 64, 200),
           ("gray4", grid.MODE_GRAY4, 64, 160),
           ("gray4", grid.MODE_GRAY4, 64, 120)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", default=str(Path(__file__).parent.parent /
                                             "demo" / "kitten.png"))
    ap.add_argument("--out", default=str(Path(__file__).parent.parent /
                                        "demo" / "_tx_lumenprobe.mp4"))
    ap.add_argument("--grid", default="252x163")
    ap.add_argument("--cell-px", type=int, default=12)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--crf", type=int, default=10)
    ap.add_argument("--lead-seconds", type=float, default=22.0)
    args = ap.parse_args()

    grid.set_header_len(28); grid.set_header_centered(True)
    gw, gh = (int(v) for v in args.grid.split("x"))
    L = grid.Layout(gw, gh)
    data = Path(args.payload).read_bytes()
    W, H = gw * args.cell_px, gh * args.cell_px
    x0, y0 = (PW - W) // 2, (PH - H) // 2
    nseq = int(args.seconds * args.fps)

    print(f"{'section':>16s} {'n_sub':>6s} {'ceiling':>9s}")
    for name, mode, ecc, peak in CONFIGS:
        grid.set_ecc(ecc)
        ns = grid.sub_count(L, mode); sub = (255 - ecc) - 4
        print(f"{name+'/'+str(ecc)+' @'+str(peak):>16s} {ns:6d} "
              f"{ns*sub*args.fps/1024:8.1f}K")
    print()

    def open_pipe(path):
        return subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo",
             "-pix_fmt", "bgr24", "-s", f"{PW}x{PH}", "-r", str(args.fps),
             "-i", "-", "-c:v", "libx264", "-preset", "fast",
             "-crf", str(args.crf), "-pix_fmt", "yuv420p", path],
            stdin=subprocess.PIPE)

    canvas = np.zeros((PH, PW, 3), np.uint8)

    # lock-in lead: mono at full peak, countdown drawn over it
    lead_out = args.out.replace(".mp4", "_lead.mp4")
    pl = open_pipe(lead_out)
    grid.set_ecc(48); grid.set_gray4_peak(255)
    ns0 = grid.sub_count(L, grid.MODE_MONO)
    enc0 = fountain.Encoder(data, 203)
    for i in range(int(args.lead_seconds * args.fps)):
        parts = []
        for j in range(ns0):
            b = enc0.block(i * ns0 + j); b = b + b"\x00" * (203 - len(b))
            parts.append(struct.pack("<I", zlib.crc32(b) & 0xFFFFFFFF) + b)
        hdr = grid.pack_header(i, enc0.k, 203, len(data), grid.MODE_MONO, 0, 0)
        img = grid.render_frame(L, hdr, b"".join(parts), grid.MODE_MONO,
                                cell_px=args.cell_px)
        canvas[:] = 0
        canvas[y0:y0 + H, x0:x0 + W] = img
        left = args.lead_seconds - i / args.fps
        txt = f"{left:0.0f}" if left > 3 else "HOLD STILL"
        sc = 22.0 if left > 3 else 6.0
        (tw, th_), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, sc, 40)
        org = ((PW - tw) // 2, (PH + th_) // 2)
        for col, thk in (((0, 0, 0), 90), ((255, 255, 255), 40)):
            cv2.putText(canvas, txt, org, cv2.FONT_HERSHEY_SIMPLEX, sc, col,
                        thk, cv2.LINE_AA)
        msg = "TAP-HOLD TO LOCK AE/AF NOW - do NOT touch the exposure slider"
        (mw, _m), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 2.2, 6)
        for col, thk in (((0, 0, 0), 16), ((255, 255, 255), 6)):
            cv2.putText(canvas, msg, ((PW - mw) // 2, PH - 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.2, col, thk, cv2.LINE_AA)
        pl.stdin.write(canvas.tobytes())
        if i % 100 == 0:
            print(f"  lead {i}", end="\r")
    pl.stdin.close(); pl.wait()

    p = open_pipe(args.out)
    canvas[:] = 0
    for name, mode, ecc, peak in CONFIGS:
        grid.set_ecc(ecc); grid.set_gray4_peak(peak)
        n_sub = grid.sub_count(L, mode)
        SUB = (255 - ecc) - 4
        enc = fountain.Encoder(data, SUB)
        for seq in range(nseq):
            parts = []
            for j in range(n_sub):
                b = enc.block(seq * n_sub + j)
                b = b + b"\x00" * (SUB - len(b))
                parts.append(struct.pack("<I", zlib.crc32(b) & 0xFFFFFFFF) + b)
            # Peak travels in zone_w so the analyser can attribute a frame by
            # CONTENT, not by offset - a capture starts anywhere in the loop
            # and may straddle a wrap. zone_w is an adaptive-mode field and is
            # ignored for mono/gray4 by both bits_per_cell and render_frame,
            # and the density probe already reuses it the same way for `hold`.
            hdr = grid.pack_header(seq, enc.k, SUB, len(data), mode, peak, 0)
            img = grid.render_frame(L, hdr, b"".join(parts), mode,
                                    cell_px=args.cell_px)
            canvas[y0:y0 + H, x0:x0 + W] = img
            p.stdin.write(canvas.tobytes())
        print(f"  rendered {name}/{ecc} @peak {peak}: {nseq} frames, "
              f"{n_sub} cw")
    p.stdin.close(); p.wait()
    grid.set_gray4_peak(255)
    print(f"\nwrote {args.out} "
          f"({Path(args.out).stat().st_size/1e6:.0f} MB, "
          f"{len(CONFIGS)*args.seconds:.0f}s per loop)")
    print(f"wrote {lead_out}")
    print("\nfilm it, then:\n  python3 src/analyze_lumen.py <capture.MOV>")


if __name__ == "__main__":
    main()
