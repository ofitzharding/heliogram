#!/usr/bin/env python3
"""
make_density_probe.py — find where the density cliff actually is.

The cliff has only ever been bracketed by two widely separated points:

    13.4 camera-px/cell  ->  works, 92% codeword yield        (252x163, current)
     9.0 camera-px/cell  ->  0/32 codewords, geometry proven  (350x194, section 24)

Everything between is unmeasured, and it is a wide gap. The interesting part is
that the ceiling climbs fast across it while the yield requirement falls:

    cell px   grid       cam px/cell   ceiling@60   yield needed for 200
       12     252x163       13.4         226.0        88.5%   <- current
       11     274x178       12.3         261.7        76.4%
       10     302x196       11.2         321.2        62.3%
        9     336x218       10.1         404.4        49.5%   <- near the dead point

At 11.2 px/cell the ceiling is 321 KB/s and 200 needs 62% yield, against the
70% this rig currently averages. If the cliff is anywhere below 11.2 that is a
real gain with no change of alphabet, no change of refresh rate, and no cost in
decision margin per cell - only smaller cells.

Unlike gray4 this does not divide the eye. Each cell is still two-level and
still has mono's full black-to-white separation. What shrinks is the number of
camera pixels averaged per cell, so the cost is sampling noise and blur, not
constellation margin. That is a different failure curve and it has to be
measured rather than reasoned about, because section 24 already showed the
reasoning is unreliable here: 350x194 samples ON GRID at 97% structure
agreement and still certifies nothing.

One take covers all four densities with optics, framing, exposure and hold
pinned, so a difference between rows is cell size alone. Each frame states its
own grid in the header's zone_w field, so the analyser attributes by content
rather than by frame offset.
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
# (cell_px, label). 12 is the known-good control and MUST stay in every loop.
LADDER = [12, 11, 10, 9]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", default=str(Path(__file__).parent.parent /
                                             "demo" / "kitten.png"))
    ap.add_argument("--out", default=str(Path(__file__).parent.parent /
                                        "demo" / "_tx_density.mp4"))
    ap.add_argument("--ecc", type=int, default=48)
    ap.add_argument("--seconds", type=float, default=7.0)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--crf", type=int, default=10)
    ap.add_argument("--lead-seconds", type=float, default=22.0)
    args = ap.parse_args()

    grid.set_ecc(args.ecc); grid.set_header_len(28)
    grid.set_header_centered(True)
    data = Path(args.payload).read_bytes()
    SUB = (255 - args.ecc) - 4
    nseq = int(args.seconds * args.fps)

    print(f"{'cell px':>8s} {'grid':>10s} {'cam px/cell':>12s} {'n_sub':>6s} "
          f"{'ceiling':>9s}")
    plans = []
    for cp in LADDER:
        gw, gh = PW // cp, PH // cp
        L = grid.Layout(gw, gh)
        ns = grid.sub_count(L, grid.MODE_MONO)
        plans.append((cp, gw, gh, L, ns))
        print(f"{cp:8d} {str(gw)+'x'+str(gh):>10s} {cp*13.4/12:12.1f} "
              f"{ns:6d} {ns*SUB*args.fps/1024:8.1f}K")
    print()

    def pipe(path):
        return subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo",
             "-pix_fmt", "bgr24", "-s", f"{PW}x{PH}", "-r", str(args.fps),
             "-i", "-", "-c:v", "libx264", "-preset", "fast",
             "-crf", str(args.crf), "-pix_fmt", "yuv420p", path],
            stdin=subprocess.PIPE)

    canvas = np.zeros((PH, PW, 3), np.uint8)

    def render(cp, gw, gh, L, ns, seq, enc):
        parts = []
        for j in range(ns):
            b = enc.block(seq * ns + j)
            b = b + b"\x00" * (SUB - len(b))
            parts.append(struct.pack("<I", zlib.crc32(b) & 0xFFFFFFFF) + b)
        # cell_px travels in zone_w so the analyser can attribute by content
        hdr = grid.pack_header(seq, enc.k, SUB, len(data), grid.MODE_MONO,
                               cp, 0)
        img = grid.render_frame(L, hdr, b"".join(parts), grid.MODE_MONO,
                                cell_px=cp)
        canvas[:] = 0
        y0 = (PH - img.shape[0]) // 2
        x0 = (PW - img.shape[1]) // 2
        canvas[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
        return canvas

    # lock-in lead at the known-good density
    lead_out = args.out.replace(".mp4", "_lead.mp4")
    pl = pipe(lead_out)
    cp, gw, gh, L, ns = plans[0]
    enc0 = fountain.Encoder(data, SUB)
    for i in range(int(args.lead_seconds * args.fps)):
        cv = render(cp, gw, gh, L, ns, i, enc0)
        left = args.lead_seconds - i / args.fps
        txt = f"{left:0.0f}" if left > 3 else "HOLD STILL"
        sc = 22.0 if left > 3 else 6.0
        (tw, th_), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, sc, 40)
        org = ((PW - tw) // 2, (PH + th_) // 2)
        for col, thk in (((0, 0, 0), 90), ((255, 255, 255), 40)):
            cv2.putText(cv, txt, org, cv2.FONT_HERSHEY_SIMPLEX, sc, col, thk,
                        cv2.LINE_AA)
        m = "TAP-HOLD TO LOCK AE/AF NOW - do NOT touch the exposure slider"
        (mw, _), _ = cv2.getTextSize(m, cv2.FONT_HERSHEY_SIMPLEX, 2.2, 6)
        for col, thk in (((0, 0, 0), 16), ((255, 255, 255), 6)):
            cv2.putText(cv, m, ((PW - mw) // 2, PH - 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.2, col, thk, cv2.LINE_AA)
        pl.stdin.write(cv.tobytes())
        if i % 200 == 0:
            print(f"  lead {i}", end="\r")
    pl.stdin.close(); pl.wait()

    p = pipe(args.out)
    for cp, gw, gh, L, ns in plans:
        enc = fountain.Encoder(data, SUB)
        for seq in range(nseq):
            p.stdin.write(render(cp, gw, gh, L, ns, seq, enc).tobytes())
        print(f"  rendered {gw}x{gh} @{cp}px: {nseq} frames, {ns} cw")
    p.stdin.close(); p.wait()
    print(f"\nwrote {args.out} "
          f"({Path(args.out).stat().st_size/1e6:.0f} MB, "
          f"{len(LADDER)*args.seconds:.0f}s per loop)")
    print(f"wrote {lead_out}")
    print("\nfilm it, then:\n  python3 src/analyze_density.py <capture.MOV>")


if __name__ == "__main__":
    main()
