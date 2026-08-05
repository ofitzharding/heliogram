#!/usr/bin/env python3
"""
make_gray_probe.py — one take that decides whether gray4 doubles the link.

mono spends 1 bit on a cell whose black-to-white separation was measured at
~12 sigma. gray4 spends 2, on four levels instead of two, so the decision
margins are roughly a third as wide - but the ceiling doubles:

    252x163 mono  ecc 48   19 codewords/frame   226.0 KB/s   200 needs 88.5%
    252x163 gray4 ecc 48   38 codewords/frame   452.0 KB/s   200 needs 44.2%
    252x163 gray4 ecc 64   38 codewords/frame   416.4 KB/s   200 needs 48.0%

So gray4 does not have to be nearly as good as mono to win; it has to be half
as good. That is the whole question, and it is not answerable from any footage
that exists, because all of it is mono.

The probe interleaves the three configurations in one video at a FIXED grid and
cell size, so optics, framing, exposure and hold are pinned across the
comparison and the only thing varying is the alphabet and the parity. Each
frame states its own mode and ecc in its header, so the analyser attributes it
by content rather than by frame offset - a capture starts at an arbitrary point
in the loop and may straddle a wrap.

ecc 64 is carried because gray4's failure mode, if it has one, is more symbol
errors rather than worse geometry, and 32-byte correction against 24 is the
cheapest hedge available. If gray4/48 fails and gray4/64 holds, that says the
alphabet is fine and the parity was the binding constraint.
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
CONFIGS = [("mono", grid.MODE_MONO, 48),
           ("gray4", grid.MODE_GRAY4, 48),
           ("gray4", grid.MODE_GRAY4, 64)]


def render_section(data, mode, ecc, L, cell_px, nseq, canvas, x0, y0, pipe):
    grid.set_ecc(ecc)
    n_sub = grid.sub_count(L, mode)
    SUB = (255 - ecc) - 4
    enc = fountain.Encoder(data, SUB)
    for seq in range(nseq):
        parts = []
        for j in range(n_sub):
            b = enc.block(seq * n_sub + j)
            b = b + b"\x00" * (SUB - len(b))
            parts.append(struct.pack("<I", zlib.crc32(b) & 0xFFFFFFFF) + b)
        hdr = grid.pack_header(seq, enc.k, SUB, len(data), mode, 0, 0)
        img = grid.render_frame(L, hdr, b"".join(parts), mode, cell_px=cell_px)
        canvas[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
        pipe.write(canvas.tobytes())
    return n_sub, SUB, enc.k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", default=str(Path(__file__).parent.parent /
                                             "demo" / "kitten.png"))
    ap.add_argument("--out", default=str(Path(__file__).parent.parent /
                                        "demo" / "_tx_grayprobe.mp4"))
    ap.add_argument("--grid", default="252x163")
    ap.add_argument("--cell-px", type=int, default=12)
    ap.add_argument("--seconds", type=float, default=8.0,
                    help="per configuration, per loop")
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

    print(f"{'config':>12s} {'n_sub':>6s} {'SUB':>5s} {'ceiling':>9s} "
          f"{'200 needs':>10s}")
    for name, mode, ecc in CONFIGS:
        grid.set_ecc(ecc)
        ns = grid.sub_count(L, mode); sub = (255 - ecc) - 4
        kbs = ns * sub * args.fps / 1024
        print(f"{name+'/'+str(ecc):>12s} {ns:6d} {sub:5d} {kbs:8.1f}K "
              f"{100*200/kbs:9.1f}%")
    print()

    def open_pipe(path):
        return subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo",
             "-pix_fmt", "bgr24", "-s", f"{PW}x{PH}", "-r", str(args.fps),
             "-i", "-", "-c:v", "libx264", "-preset", "fast",
             "-crf", str(args.crf), "-pix_fmt", "yuv420p", path],
            stdin=subprocess.PIPE)

    # lock-in lead: mono/48 frames with a countdown over them, same payload
    lead_out = args.out.replace(".mp4", "_lead.mp4")
    pl = open_pipe(lead_out)
    canvas = np.zeros((PH, PW, 3), np.uint8)
    grid.set_ecc(48)
    n_sub0 = grid.sub_count(L, grid.MODE_MONO)
    enc0 = fountain.Encoder(data, 203)
    for i in range(int(args.lead_seconds * args.fps)):
        parts = []
        for j in range(n_sub0):
            b = enc0.block(i * n_sub0 + j); b = b + b"\x00" * (203 - len(b))
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
        (mw, _mh), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 2.2, 6)
        for col, thk in (((0, 0, 0), 16), ((255, 255, 255), 6)):
            cv2.putText(canvas, msg, ((PW - mw) // 2, PH - 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.2, col, thk, cv2.LINE_AA)
        pl.stdin.write(canvas.tobytes())
        if i % 100 == 0:
            print(f"  lead {i}", end="\r")
    pl.stdin.close(); pl.wait()

    p = open_pipe(args.out)
    canvas[:] = 0
    for name, mode, ecc in CONFIGS:
        ns, sub, k = render_section(data, mode, ecc, L, args.cell_px, nseq,
                                    canvas, x0, y0, p.stdin)
        print(f"  rendered {name}/{ecc}: {nseq} frames, {ns} cw, k={k}")
    p.stdin.close(); p.wait()
    print(f"\nwrote {args.out} "
          f"({Path(args.out).stat().st_size/1e6:.0f} MB, "
          f"{len(CONFIGS)*args.seconds:.0f}s per loop)")
    print(f"wrote {lead_out}")
    print(f"\nfilm it, then:\n  python3 src/analyze_gray.py <capture.MOV>")


if __name__ == "__main__":
    main()
