#!/usr/bin/env python3
"""
select_decode.py — sharpness-selective reception.

Measured on real handheld 4K footage: bit error rate averages 9.8% but the
BEST individual frames are at 4.4%. Handheld capture does not degrade
uniformly; it produces a mixture of sharp frames and motion-blurred garbage.

Every receiver in this field (and our own fusion path) treats captures as
interchangeable evidence and averages them. That is exactly wrong for a
mixture: averaging a sharp frame with a blurred one destroys the sharp one.
The correct operation on a mixture is SELECTION, not integration.

So this receiver:
  1. scores every capture's sharpness directly from the code region
     (gradient energy of the sampled cell field — no reference needed)
  2. decodes only the top-quality captures, best-first
  3. within a displayed frame, fuses ONLY captures within a small quality
     band of that frame's best capture (never mixes tiers)
  4. stops as soon as the fountain completes, so the reported time is the
     time actually needed rather than the length of the clip

Reported goodput is deliberately honest: file size divided by the *span* of
capture time from the first used frame to the last used frame, which is what
a real transfer would take.

    python3 select_decode.py capture.mov out.bin --grid 252x140 --ecc 48
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

sys.path.insert(0, str(Path(__file__).parent))
from codec import fountain, grid


def sharpness(lum: np.ndarray, layout: grid.Layout) -> float:
    """Per-capture sharpness from the sampled cell field.

    Lay the sampled cells back onto the grid and measure horizontal +
    vertical gradient energy. A sharp capture has cells that differ strongly
    from their neighbours (data is pseudorandom, so adjacent cells disagree
    ~50% of the time); blur pulls neighbours toward each other and collapses
    the gradient. Needs no ground truth, so it works at receive time.
    """
    cells = layout.payload_cells[: len(lum)]
    G = np.zeros((layout.gh, layout.gw), np.float32)
    G[cells[:, 0], cells[:, 1]] = lum[: len(cells)]
    gx = np.abs(np.diff(G, axis=1)).mean()
    gy = np.abs(np.diff(G, axis=0)).mean()
    spread = max(1e-3, np.percentile(lum, 95) - np.percentile(lum, 5))
    return float((gx + gy) / spread)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--grid", default="252x140")
    ap.add_argument("--ecc", type=int, default=48)
    ap.add_argument("--band", type=float, default=0.90,
                    help="fuse captures scoring >= band * (best score for that "
                         "displayed frame). 1.0 = never fuse, only best frame")
    ap.add_argument("--top", type=float, default=0.5,
                    help="fraction of captures (by sharpness) to consider")
    args = ap.parse_args()
    grid.set_ecc(args.ecc)
    gw, gh = (int(v) for v in args.grid.split("x"))

    # header format probe across the whole clip
    probe = cv2.VideoCapture(args.input)
    total = int(probe.get(cv2.CAP_PROP_FRAME_COUNT)) or 600
    best_hl, best_hits = 28, -1
    for hl in (28, 24):
        grid.set_header_len(hl)
        lay = grid.Layout(gw, gh)
        hits = 0
        for fi in np.linspace(0, total - 1, 30).astype(int):
            probe.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, im = probe.read()
            if not ok:
                continue
            sm = cv2.resize(im, None, fx=0.5, fy=0.5) if im.shape[1] >= 3000 else im
            Hs = grid.locate(sm, lay)
            if Hs is None:
                continue
            H = (np.diag([2.0, 2.0, 1.0]) @ Hs) if im.shape[1] >= 3000 else Hs
            hd, _s, _st = grid.sample_frame(im, lay, H)
            hits += hd is not None
        if hits > best_hits:
            best_hl, best_hits = hl, hits
    probe.release()
    grid.set_header_len(best_hl)
    layout = grid.Layout(gw, gh)
    print(f"header v{'2' if best_hl == 28 else '1'} ({best_hits} probe hits)")

    # ---- pass A: sample every capture once, score it, keep it in memory ----
    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    obs = []          # (score, n, seq, lum)
    proto = None
    n = 0
    t0 = time.time()
    while True:
        ok, img = cap.read()
        if not ok:
            break
        n += 1
        if n % 300 == 0:
            print(f"  pass A frame {n}", file=sys.stderr)
        sm = cv2.resize(img, None, fx=0.5, fy=0.5) if img.shape[1] >= 3000 else img
        Hs = grid.locate(sm, layout)
        if Hs is None:
            continue
        H = (np.diag([2.0, 2.0, 1.0]) @ Hs) if img.shape[1] >= 3000 else Hs
        header, samples, st = grid.sample_frame(img, layout, H)
        if samples is None:
            continue
        lum = samples.mean(axis=1).astype(np.float32)
        if header is not None and proto is None:
            proto = header
        obs.append([sharpness(lum, layout), n,
                    None if header is None else header["seq"], lum])
    cap.release()
    if proto is None:
        sys.exit("no readable header anywhere")
    print(f"pass A: {n} frames, {len(obs)} located, "
          f"{sum(o[2] is not None for o in obs)} with headers")

    # ML seq for captures whose header failed but geometry held
    templates = grid.header_templates(proto, min(4000, 12 * proto["k"]))
    # (header luminances weren't kept; use clock sync from headered captures)
    known = [(o[1], o[2]) for o in obs if o[2] is not None]
    if len(known) >= 2:
        seqs = np.array([s for _, s in known])
        ns = np.array([x for x, _ in known])
        period = None
        # loop period: how many captures between repeats of the same seq
        for s in np.unique(seqs):
            idx = ns[seqs == s]
            if len(idx) >= 2:
                d = np.diff(np.sort(idx))
                period = int(np.median(d)) if period is None else period
                break
        if period:
            for o in obs:
                if o[2] is None:
                    j = int(np.argmin(np.abs(ns - o[1])))
                    if abs(o[1] - ns[j]) <= 30:
                        o[2] = int((seqs[j] + (o[1] - ns[j])) % period)

    obs = [o for o in obs if o[2] is not None]
    scores = np.array([o[0] for o in obs])
    print(f"sharpness: median {np.median(scores):.3f}  "
          f"p90 {np.percentile(scores, 90):.3f}  max {scores.max():.3f}")

    # ---- pass B: best-first decoding, tiered fusion ----
    cut = np.percentile(scores, 100 * (1 - args.top))
    pool = sorted([o for o in obs if o[0] >= cut], key=lambda o: -o[0])
    print(f"considering top {len(pool)} captures (score >= {cut:.3f})")

    byseq = {}
    for o in pool:
        byseq.setdefault(o[2], []).append(o)

    dec = fountain.Decoder(proto["k"], proto["block_size"], proto["file_size"])
    used_ns = []
    got = 0
    # process displayed frames in order of their best capture's sharpness
    order = sorted(byseq.items(), key=lambda kv: -kv[1][0][0])
    for seq, group in order:
        best = group[0][0]
        tier = [g for g in group if g[0] >= args.band * best]
        # fuse within the tier only
        votes = np.zeros(len(tier[0][3]), np.float32)
        for g in tier:
            lum = g[3]
            th, _ = cv2.threshold(np.clip(lum, 0, 255).astype(np.uint8), 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            votes[: len(lum)] += (lum > th)
        votes /= len(tier)
        payload = grid.decide_payload(dict(proto, seq=seq),
                                      (votes * 255.0)[:, None].repeat(3, axis=1),
                                      layout)
        if payload is None:
            continue
        bs = proto["block_size"]
        crc = struct.unpack("<I", payload[:4])[0]
        block = payload[4:4 + bs]
        if zlib.crc32(block) & 0xFFFFFFFF != crc:
            continue
        got += 1
        used_ns.extend(g[1] for g in tier)
        dec.add(seq, block)
        if len(dec.decoded) >= dec.k or (got >= dec.k and not dec.done):
            dec.gaussian_fallback()
        if dec.done:
            break

    wall = time.time() - t0
    print(f"blocks recovered: {got} (need {proto['k']})")
    if not dec.done:
        dec.gaussian_fallback()
    if not dec.done:
        sys.exit(f"FAILED: {len(dec.decoded)}/{proto['k']} blocks")

    data = dec.result()
    Path(args.output).write_bytes(data)
    span = (max(used_ns) - min(used_ns) + 1) / fps
    print(f"recovered {len(data):,} bytes  sha256 {hashlib.sha256(data).hexdigest()[:16]}")
    print(f"capture span used: {span:.1f}s")
    print(f"GOODPUT {len(data)/span/1024:.1f} KB/s   (decode wall {wall:.0f}s)")


if __name__ == "__main__":
    main()
