#!/usr/bin/env python3
"""
fast_decode.py — parallel offline decoder.

Every frame independently yields at most one fountain block, so frame decoding
is embarrassingly parallel. The single-threaded decoder was taking minutes per
4K clip, which throttled every experiment in this project. This splits the clip
across worker processes.

Also reports honest time-to-file goodput: file size over the span from the
first block-yielding capture to the capture that completes the file.

    python3 fast_decode.py capture.mov out.bin --grid 560x311 --ecc 48
    python3 fast_decode.py capture.mov out.bin --grid 560x311 --scan   # sweep k1
"""
import argparse
import hashlib
import os
import struct
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from codec import fountain, grid

_CFG = {}


def _init(cfg):
    _CFG.update(cfg)
    grid.set_ecc(cfg["ecc"])
    grid.set_header_len(cfg["header_len"])
    grid.set_header_centered(cfg.get("centered", True))
    grid.set_radial(cfg["radial"])


def _worker(rng):
    """Decode a contiguous frame range; return (frame_no, seq, block) hits."""
    start, end = rng
    gw, gh = _CFG["gw"], _CFG["gh"]
    layout = grid.Layout(gw, gh)
    cap = cv2.VideoCapture(_CFG["path"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out = []
    proto = None
    n = start
    while n < end:
        ok, img = cap.read()
        if not ok:
            break
        n += 1
        small = cv2.resize(img, None, fx=0.5, fy=0.5) if img.shape[1] >= 3000 else img
        Hs = grid.locate(small, layout)
        if Hs is None:
            Hs = grid.locate(img, layout)     # dense grids need full res
            H = Hs
        else:
            H = (np.diag([2.0, 2.0, 1.0]) @ Hs) if img.shape[1] >= 3000 else Hs
        if H is None:
            continue
        header, samples, _st = grid.sample_frame(img, layout, H)
        if header is None or samples is None:
            continue
        if proto is None:
            proto = {k: header[k] for k in ("k", "block_size", "file_size", "mode")}
        payload = grid.decide_payload(header, samples, layout)
        if payload is None:
            continue
        bs = header["block_size"]
        blk = payload[4:4 + bs]
        if zlib.crc32(blk) & 0xFFFFFFFF != struct.unpack("<I", payload[:4])[0]:
            continue
        out.append((n, header["seq"], blk, proto))
    cap.release()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--grid", default="560x311")
    ap.add_argument("--ecc", type=int, default=48)
    ap.add_argument("--radial", type=float, default=0.020)
    ap.add_argument("--header-len", type=int, default=28)
    ap.add_argument("--header-top", action="store_true",
                    help="older captures put the header at the top edge")
    ap.add_argument("--workers", type=int, default=max(2, os.cpu_count() - 2))
    ap.add_argument("--scan", action="store_true",
                    help="sweep k1 on a frame sample first, pick the best")
    args = ap.parse_args()
    gw, gh = (int(v) for v in args.grid.split("x"))

    cap = cv2.VideoCapture(args.input)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    cap.release()
    print(f"{total} frames @ {fps:.0f}fps, {args.workers} workers")

    radial = args.radial
    if args.scan:
        grid.set_ecc(args.ecc); grid.set_header_len(args.header_len)
        grid.set_header_centered(not args.header_top)
        layout = grid.Layout(gw, gh)
        c = cv2.VideoCapture(args.input)
        probes = []
        for fi in np.linspace(total * 0.25, total * 0.8, 10).astype(int):
            c.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, im = c.read()
            if not ok:
                continue
            sm = cv2.resize(im, None, fx=0.5, fy=0.5) if im.shape[1] >= 3000 else im
            Hs = grid.locate(sm, layout)
            H = (np.diag([2.,2.,1.]) @ Hs) if (Hs is not None and im.shape[1] >= 3000) else Hs
            if H is None:
                H = grid.locate(im, layout)
            if H is not None:
                probes.append((im, H))
        c.release()
        best, best_hits = 0.0, -1
        for k1 in np.arange(0.0, 0.041, 0.005):
            grid.set_radial(float(k1))
            hits = 0
            for im, H in probes:
                hd, s, _ = grid.sample_frame(im, layout, H)
                if hd is None or s is None:
                    continue
                pl = grid.decide_payload(hd, s, layout)
                if pl is None:
                    continue
                b = hd["block_size"]
                if zlib.crc32(pl[4:4+b]) & 0xFFFFFFFF == struct.unpack("<I", pl[:4])[0]:
                    hits += 1
            if hits > best_hits:
                best, best_hits = float(k1), hits
        radial = best
        print(f"k1 scan: {radial:+.3f} ({best_hits}/{len(probes)} probe frames)")

    cfg = dict(path=args.input, gw=gw, gh=gh, ecc=args.ecc,
               header_len=args.header_len, radial=radial,
               centered=not args.header_top)
    chunk = max(30, total // (args.workers * 4))
    ranges = [(s, min(s + chunk, total)) for s in range(0, total, chunk)]
    t0 = time.time()
    with Pool(args.workers, initializer=_init, initargs=(cfg,)) as pool:
        results = pool.map(_worker, ranges)
    wall = time.time() - t0
    hits = sorted([h for r in results for h in r], key=lambda x: x[0])
    print(f"decoded {len(hits)} blocks in {wall:.1f}s wall "
          f"({total/max(wall,1e-6):.0f} frames/s)")
    if not hits:
        sys.exit("FAILED: no blocks")
    proto = hits[0][3]
    dec = fountain.Decoder(proto["k"], proto["block_size"], proto["file_size"])
    first = done = None
    for n, seq, blk, _p in hits:
        if seq in dec.seen:
            continue
        if first is None:
            first = n
        dec.add(seq, blk)
        if len(dec.seen) >= dec.k and not dec.done:
            dec.gaussian_fallback()
        if dec.done:
            done = n
            break
    if not dec.done:
        dec.gaussian_fallback()
    if not dec.done:
        sys.exit(f"FAILED: {len(dec.decoded)}/{dec.k} blocks "
                 f"({len(set(h[1] for h in hits))} distinct seqs seen)")
    data = dec.result()
    Path(args.output).write_bytes(data)
    span = (done - first + 1) / fps
    g = len(data) / span / 1024
    print(f"\nrecovered {len(data):,} bytes")
    print(f"sha256 {hashlib.sha256(data).hexdigest()}")
    print(f"transfer span {span:.2f}s (frame {first} -> {done})")
    print(f"GOODPUT {g:.1f} KB/s")
    print(f"  vs decimen handheld 128 KB/s : {g/128:.2f}x")
    print(f"  vs decimen propped  186 KB/s : {g/186:.2f}x")


if __name__ == "__main__":
    main()
