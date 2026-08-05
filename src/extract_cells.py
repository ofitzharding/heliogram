#!/usr/bin/env python3
"""
extract_cells.py — separate GEOMETRY from DEMODULATION, once per capture.

Every demodulator experiment in this project re-read the 4K clip: locate() is
~100 ms/frame, so a single idea cost 8 minutes of wall clock and most ideas
were never tried. But geometry does not depend on the demodulator. Sample the
cells ONCE into a (frames, gh, gw) float array and every subsequent experiment
is a numpy operation over ~300 MB, i.e. seconds.

Writes:
  <out>.dat    float16 memmap, (n_kept, gh, gw) raw cell luminance
  <out>.npz    frame numbers, seq, header fields, per-frame photometry

    python3 extract_cells.py cap.MOV out --grid 252x140 --ecc 48
"""
import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from codec import grid

_CFG = {}


def _init(cfg):
    _CFG.update(cfg)
    grid.set_ecc(cfg["ecc"])
    grid.set_header_len(cfg["header_len"])
    grid.set_header_centered(cfg["centered"])
    grid.set_radial(cfg["radial"])


def _worker(rng):
    start, end = rng
    gw, gh = _CFG["gw"], _CFG["gh"]
    L = grid.Layout(gw, gh)
    allc = np.argwhere(np.ones((gh, gw), bool))
    cap = cv2.VideoCapture(_CFG["path"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    ys = np.lib.format.open_memmap(_CFG["dat"], mode="r+")
    rows = []
    n = start
    while n < end:
        ok, img = cap.read()
        if not ok:
            break
        fn = n
        n += 1
        H = None
        # k1 candidates: the radial coefficient is a property of the FRAMING,
        # not of the code, and it changed between takes. Trying the handful of
        # plausible values and keeping the one whose header decodes costs four
        # samplings of a 3x3 box filter that is already computed.
        best = None
        for k1 in _CFG["k1s"]:
            grid.set_radial(k1)
            Hc = grid.locate(img, L)
            if Hc is None:
                continue
            hd, _s, _t = grid.sample_frame(img, L, Hc)
            score = 1 if hd is not None else 0
            if best is None or score > best[0]:
                best = (score, k1, Hc, hd)
            if score:
                break
        if best is None:
            continue
        _sc, k1, H, hd = best
        grid.set_radial(k1)
        y = grid.sample_cells(img, L, H, allc).mean(axis=1).reshape(gh, gw)
        ys[fn] = y.astype(np.float16)
        rows.append((fn, k1,
                     -1 if hd is None else hd["seq"],
                     -1 if hd is None else hd["k"],
                     -1 if hd is None else hd["block_size"],
                     -1 if hd is None else hd["file_size"],
                     -1 if hd is None else hd["mode"],
                     float(np.percentile(y, 5)), float(np.percentile(y, 50)),
                     float(np.percentile(y, 95))))
    cap.release()
    del ys
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("out")
    ap.add_argument("--grid", default="252x140")
    ap.add_argument("--ecc", type=int, default=48)
    ap.add_argument("--radial", type=float, default=0.020)
    ap.add_argument("--header-len", type=int, default=28)
    ap.add_argument("--header-top", action="store_true")
    ap.add_argument("--k1s", default="0.020,0.015,0.025,0.010,0.000")
    ap.add_argument("--workers", type=int, default=max(2, os.cpu_count() - 2))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    gw, gh = (int(v) for v in args.grid.split("x"))

    cap = cv2.VideoCapture(args.input)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    cap.release()
    if args.limit:
        total = min(total, args.limit)
    dat = args.out + ".dat.npy"
    arr = np.lib.format.open_memmap(dat, mode="w+", dtype=np.float16,
                                    shape=(total, gh, gw))
    del arr
    cfg = dict(path=args.input, gw=gw, gh=gh, ecc=args.ecc,
               header_len=args.header_len, radial=args.radial,
               centered=not args.header_top, dat=dat,
               k1s=[float(v) for v in args.k1s.split(",")])
    chunk = max(20, total // (args.workers * 6))
    ranges = [(s, min(s + chunk, total)) for s in range(0, total, chunk)]
    print(f"{total} frames @ {fps:.0f}fps -> {dat} "
          f"({total*gh*gw*2/1e6:.0f} MB), {args.workers} workers")
    with Pool(args.workers, initializer=_init, initargs=(cfg,)) as pool:
        res = pool.map(_worker, ranges)
    rows = sorted([r for rr in res for r in rr])
    a = np.array(rows, dtype=np.float64)
    np.savez(args.out + ".npz", frame=a[:, 0].astype(np.int64),
             k1=a[:, 1], seq=a[:, 2].astype(np.int64),
             k=a[:, 3].astype(np.int64), block_size=a[:, 4].astype(np.int64),
             file_size=a[:, 5].astype(np.int64), mode=a[:, 6].astype(np.int64),
             p5=a[:, 7], p50=a[:, 8], p95=a[:, 9],
             gw=gw, gh=gh, fps=fps, total=total, ecc=args.ecc)
    hdr = int((a[:, 2] >= 0).sum())
    print(f"located {len(rows)}/{total} ({100*len(rows)/total:.1f}%), "
          f"hard header {hdr}/{total} ({100*hdr/total:.1f}%)")
    print(f"photometry over located frames: p5 {np.median(a[:,7]):.0f}  "
          f"p50 {np.median(a[:,8]):.0f}  p95 {np.median(a[:,9]):.0f}")


if __name__ == "__main__":
    main()
