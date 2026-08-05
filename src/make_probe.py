#!/usr/bin/env python3
"""
make_probe.py — one capture that measures the whole operating envelope.

WHY
---
Every take so far measured ONE configuration and got confounded by something
else that changed at the same time: density confounded by straddling,
straddling confounded by the display path, the display path confounded by a
CPU-load artefact. Six takes, six single points, several of them wrong.

A capture is ~50 seconds of the user's time. Spend it on a designed
experiment instead of a data transfer. Everything below shares one geometry,
one focus lock, one exposure lock, so every comparison inside it is
controlled by construction — which no pair of separate takes can claim.

WHAT IT MEASURES, all from one film
-----------------------------------
  1. PSF / MTF directly      - stripe pitches from 24 down to 4 px; the pitch
                               where contrast dies IS the resolution limit,
                               measured rather than inferred from BER
  2. black level + veiling   - full black and full white fields
  3. gamma / transfer curve  - a 5-step grey ramp
  4. straddle rate           - alternating field pairs; any mid-grey read on a
                               2-level pattern is exposure straddle, and the
                               fraction gives the rate without needing decode
  5. BER vs density          - six real code blocks, 200 -> 466 cells wide,
                               the sweep that locates the true wall
  6. cross-density transfer  - same geometry throughout, so kernels fitted at
                               one density can be tested at another, which two
                               separate takes can never test honestly
  7. drift                   - the calibration block is repeated at the end;
                               if it disagrees with the start, the take moved
"""
import struct
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from codec import grid, fountain

PW, PH = 3024, 1964
SUB = 255 - 48 - 4
FPS = 60


def canvas(img=None, fill=0):
    c = np.full((PH, PW, 3), fill, np.uint8)
    if img is not None:
        y0 = (PH - img.shape[0]) // 2
        x0 = (PW - img.shape[1]) // 2
        c[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
    return c


def stripes(pitch, vertical=True):
    """Square wave at a given pitch in PANEL pixels. Contrast measured off
    this is the MTF at that spatial frequency, no decoding involved."""
    c = np.zeros((PH, PW, 3), np.uint8)
    n = np.arange(PW if vertical else PH) // pitch % 2
    band = (n * 255).astype(np.uint8)
    if vertical:
        c[:, :, :] = band[None, :, None]
    else:
        c[:, :, :] = band[:, None, None]
    return c



def tag_margin(c, density_idx, hold_n):
    """Stamp the frame's CONDITION into the letterbox margin as plain blocks.

    The analyser otherwise learns a frame's density and hold from its header
    - and on the last real take 0 of 30 headers decoded. Since the dense
    blocks failing to decode is precisely the hypothesis under test, a probe
    that can only be interpreted when decoding succeeds is worthless exactly
    when it matters. These blocks survive total decode failure: they are
    large, high-contrast, and outside the code area, so they need nothing but
    a threshold.

    left group  = density index (1..3), right group = hold (1, 2 or 4).
    """
    h = 56
    y0 = PH - 2 * h
    def blocks(x0, n):
        for i in range(n):
            x = x0 + i * (h + 22)
            c[y0:y0 + h, x:x + h] = 255
    blocks(90, density_idx)
    blocks(PW // 2 + 90, hold_n)
    return c


def main():
    out = Path(__file__).parent.parent / "demo" / "_tx_probe.mp4"
    data = (Path(__file__).parent.parent / "demo" / "payload_big.png").read_bytes()
    grid.set_ecc(48); grid.set_header_len(28)
    grid.set_header_centered(True); grid.set_radial(0.0)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (PW, PH))
    log = []

    def hold(frame, n):
        for _ in range(n):
            vw.write(frame)

    def calib(tag):
        start = sum(x[2] for x in log)
        hold(canvas(fill=0), 18)                       # black field
        hold(canvas(fill=255), 18)                     # white field
        for lv in (51, 102, 153, 204):                 # grey steps
            hold(canvas(fill=lv), 12)
        for pitch in (24, 20, 16, 12, 10, 8, 6, 5, 4): # MTF sweep
            hold(stripes(pitch, True), 12)
        # straddle probe: two fields that differ everywhere, alternating every
        # frame. Any cell read as mid-grey can only be exposure straddle.
        a = stripes(8, True); b = 255 - a
        for i in range(24):
            hold(a if i % 2 == 0 else b, 1)
        n = sum(x[2] for x in log) + 0
        log.append((tag, start, 18+18+48+108+24))

    calib("calib-start")

    # TWO-AXIS DECOUPLING.
    #
    # v1 of this probe varied density while holding the temporal regime at
    # 60fps, which measures the LUMPED ceiling - precisely the conflation the
    # experiment is supposed to expose. Optics are pinned by construction
    # here (one take, one lens, one focus, one distance), so the only way to
    # separate the axes is to vary the TEMPORAL regime at each density.
    #
    #   hold=1  : a new frame every camera frame. Exposure straddles refresh
    #             boundaries; this is the regime every take so far used.
    #   hold=4  : each frame held ~4 camera frames. Captures land wholly
    #             inside one displayed frame - the FROZEN-FRAME ceiling.
    #
    # Frozen-frame BER at a given density is the optical limit at that
    # density. The gap between the two holds, with optics identical, is the
    # exposure-sync contribution. Neither number alone can show it.
    #
    # hold=4 also separates LCD pixel response from exposure straddle, which
    # a single-hold experiment cannot: response-time ghosting decays within a
    # few refreshes, so it contaminates the FIRST capture after a transition
    # and not the later ones. The analyser can therefore drop first-after-
    # transition frames and still have clean samples. On an LCD panel that
    # distinction is essential; on OLED it would be moot.
    for d_idx, (spec, cell_px) in enumerate(
            (("252x140", 12), ("350x194", 8), ("466x259", 6)), start=1):
        gw, gh = (int(v) for v in spec.split("x"))
        L = grid.Layout(gw, gh)
        n_sub = grid.sub_count(L, grid.MODE_MONO)
        enc = fountain.Encoder(data, SUB)
        for hold_n in (1, 2, 4):
            start = sum(x[2] for x in log)
            NSEQ = {1: 36, 2: 20, 4: 12}[hold_n]
            for seq in range(NSEQ):
                parts = []
                for j in range(n_sub):
                    b = enc.block(seq * n_sub + j)
                    b = b + b"\x00" * (SUB - len(b))
                    parts.append(struct.pack("<I", zlib.crc32(b) & 0xFFFFFFFF) + b)
                hdr = grid.pack_header(seq, enc.k,
                                       L.payload_capacity_bytes(grid.MODE_MONO) - 4,
                                       len(data), grid.MODE_MONO, hold_n, 0)
                img = grid.render_frame(L, hdr, b"".join(parts), grid.MODE_MONO,
                                        cell_px=cell_px)
                hold(tag_margin(canvas(img), d_idx, hold_n), hold_n)
            log.append((f"code-{spec}-hold{hold_n}", start, NSEQ * hold_n))
            print(f"  {spec:9s} hold={hold_n}  {n_sub:2d} codewords/frame, "
                  f"{NSEQ} distinct frames")

    # ROLLING-SHUTTER PROBE: full-field flips every refresh. A rolling shutter
    # reads rows at different times, so a single capture shows a row-dependent
    # mixture. Row-dependence is the signature; its absence would mean the
    # straddle is global (exposure) rather than per-row (readout).
    start = sum(x[2] for x in log)
    for i in range(40):
        hold(canvas(fill=255 if i % 2 == 0 else 0), 1)
    log.append(("rolling-shutter", start, 40))

    calib("calib-end")
    vw.release()

    total = sum(x[2] for x in log)
    print(f"\nwrote {out}  ({total} frames, {total/FPS:.1f}s per loop)")
    print("\nsegment map (frame offsets within one loop):")
    off = 0
    for tag, start, n in log:
        print(f"   {tag:16s} frames {off:5d}-{off+n-1:5d}")
        off += n
    (out.with_suffix(".map.txt")).write_text(
        "\n".join(f"{t}\t{s}\t{n}" for t, s, n in log))


if __name__ == "__main__":
    main()
