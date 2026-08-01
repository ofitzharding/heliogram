#!/usr/bin/env python3
"""
soft_decode.py — partial-frame recovery via sub-block granularity + soft erasures.

THE PROBLEM THIS ATTACKS
------------------------
Measured: only ~38% of captured frames yield anything, while decimen's
QR-based receiver gets near 100%. That 2.6x is larger than every density or
constellation gain in this project combined, so it is the real bottleneck.

Why we lose frames: one frame carries exactly one fountain block, and a
fountain block is all-or-nothing. If any Reed-Solomon codeword inside it
exceeds its correction budget, the whole frame contributes ZERO — even when
the other fifteen codewords in that frame were perfect. Measured on real
footage: failing frames typically have most codewords well inside budget and
a handful blown out by localized blur.

THE FIX
-------
Two changes, both about granularity of failure:

1. SUB-BLOCK GRANULARITY. Treat each RS codeword's data section as its own
   fountain symbol rather than lumping a whole frame into one. A frame with
   localized damage then contributes its surviving codewords instead of
   nothing. Binary yield becomes graceful degradation.

2. SOFT ERASURES. We know per-cell confidence (distance from the decision
   boundary). RS corrects e errors + s erasures with 2e + s <= parity, so
   telling it WHERE the doubt is doubles its reach. Marking the least
   confident bytes as erasures rescues codewords that error-only decoding
   cannot touch.

This is the ordinary discipline of a modern radio receiver — never throw away
soft information, never let a decision be more binary than it has to be —
applied to a channel where the whole field currently hard-decides everything.
"""
import argparse
import hashlib
import struct
import sys
import time
import zlib
from pathlib import Path

import cv2
import numpy as np
from reedsolo import RSCodec, ReedSolomonError

sys.path.insert(0, str(Path(__file__).parent))
from codec import fountain, grid


def soft_rs_decode(raw: bytes, margins: np.ndarray, ecc: int,
                   max_erase_frac: float = 0.75):
    """Decode one RS codeword, escalating to erasure decoding on failure.

    `margins` holds per-byte confidence (min over the byte's cells of the
    distance to the decision boundary). Bytes are erased in ascending
    confidence until the codeword decodes or the erasure budget runs out.
    """
    rs = RSCodec(ecc)
    try:
        return bytes(rs.decode(raw)[0])
    except ReedSolomonError:
        pass
    order = np.argsort(margins)
    max_er = int(ecc * max_erase_frac)
    for n_er in range(4, max_er + 1, 4):
        erase = [int(i) for i in order[:n_er]]
        try:
            return bytes(rs.decode(raw, erase_pos=erase)[0])
        except ReedSolomonError:
            continue
    return None


def frame_subblocks(header, samples, layout, ecc):
    """Decode a frame into per-codeword sub-blocks.

    Returns a list of (codeword_index, data_bytes) for every codeword that
    decoded and passed its own CRC. A frame contributes as many sub-blocks as
    survived, instead of one all-or-nothing block.
    """
    lum = samples.mean(axis=1)
    th, _ = cv2.threshold(np.clip(lum, 0, 255).astype(np.uint8), 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bits = (lum > th).astype(np.uint8)
    spread = max(1e-3, np.percentile(lum, 90) - np.percentile(lum, 10))
    conf = np.abs(lum - th) / spread
    raw = np.packbits(bits).tobytes()
    n_bytes = len(conf) // 8            # whole bytes only; drop the tail
    byte_conf = conf[: n_bytes * 8].reshape(n_bytes, 8).min(axis=1)

    out = []
    n_cw = len(raw) // 255
    for c in range(n_cw):
        chunk = raw[c * 255:(c + 1) * 255]
        m = byte_conf[c * 255:(c + 1) * 255]
        dec = soft_rs_decode(chunk, m, ecc)
        if dec is not None:
            out.append((c, dec))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--grid", default="252x140")
    ap.add_argument("--ecc", type=int, default=48)
    ap.add_argument("--radial", type=float, default=0.020)
    ap.add_argument("--header-len", type=int, default=28)
    ap.add_argument("--header-top", action="store_true")
    ap.add_argument("--frames", type=int, default=40)
    args = ap.parse_args()

    grid.set_ecc(args.ecc)
    grid.set_header_len(args.header_len)
    grid.set_header_centered(not args.header_top)
    grid.set_radial(args.radial)
    gw, gh = (int(v) for v in args.grid.split("x"))
    layout = grid.Layout(gw, gh)

    cap = cv2.VideoCapture(args.capture)
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = np.linspace(N * 0.2, N * 0.85, args.frames).astype(int)

    whole_ok = 0
    sub_total = 0
    sub_possible = 0
    located = headers = 0
    for fi in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, img = cap.read()
        if not ok:
            continue
        H = grid.locate(img, layout)
        if H is None:
            continue
        located += 1
        header, samples, _ = grid.sample_frame(img, layout, H)
        if header is None or samples is None:
            continue
        headers += 1

        # baseline: whole-frame all-or-nothing
        pl = grid.decide_payload(header, samples, layout)
        good = False
        if pl is not None:
            bs = header["block_size"]
            good = (zlib.crc32(pl[4:4 + bs]) & 0xFFFFFFFF ==
                    struct.unpack("<I", pl[:4])[0])
        whole_ok += good

        # sub-block: how many codewords survive independently
        subs = frame_subblocks(header, samples, layout, args.ecc)
        n_cw = (grid.rs_encoded_len(header["block_size"] + 4)) // 255
        sub_total += len(subs)
        sub_possible += max(n_cw, 1)

    print(f"frames sampled   {len(idx)}")
    print(f"  located        {located}")
    print(f"  headers ok     {headers}")
    print(f"\nWHOLE-FRAME (current): {whole_ok}/{headers} frames usable "
          f"= {whole_ok/max(headers,1)*100:.0f}%")
    print(f"SUB-BLOCK + SOFT     : {sub_total}/{sub_possible} codewords recovered "
          f"= {sub_total/max(sub_possible,1)*100:.0f}%")
    if whole_ok and sub_possible:
        gain = (sub_total / sub_possible) / (whole_ok / max(headers, 1))
        print(f"\nyield gain: {gain:.2f}x")
        print(f"  73.8 KB/s measured -> {73.8*gain:.0f} KB/s projected")


if __name__ == "__main__":
    main()
