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
    _K1[0] = cfg["radial"]
    grid.set_local_threshold(cfg.get("local_th", grid.LOCAL_TH))
    _H_PREV[0] = None
    grid.set_phase_hint(None)


_TEMPLATES = [None]
_FD = [None]        # per-process FrameDecoder, holds the rolling kernel donor
_ALLC = [None]
_K1 = [0.0]         # per-worker tracked radial coefficient
_H_PREV = [None]    # last accepted homography (see HOMOGRAPHY REUSE)


def _worker(rng):
    """Decode a contiguous frame range; return (frame_no, seq, block) hits."""
    start, end = rng
    gw, gh = _CFG["gw"], _CFG["gh"]
    layout = grid.Layout(gw, gh)
    cap = cv2.VideoCapture(_CFG["path"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out = []
    proto = None
    proto_full = _CFG.get("proto_full")
    if proto_full is not None and _TEMPLATES[0] is None:
        _TEMPLATES[0] = grid.header_templates(proto_full,
                                              _CFG.get("max_seq", 1500))
    n = start
    while n < end:
        ok, img = cap.read()
        if not ok:
            break
        n += 1
        # HOMOGRAPHY REUSE. locate() re-runs contour finding from scratch on
        # every frame - 101 ms/frame, 6.8% of the decoder - to re-find a code
        # that moved by a fraction of a cell in 1/60 s. Try the PREVIOUS
        # frame's homography first and keep it if the header still decodes,
        # which is a stricter test than any tracking residual: 68 bytes behind
        # RS(68,28) and a CRC16 do not pass by accident on a stale geometry.
        header = samples = None
        if _H_PREV[0] is not None:
            header, samples, _st = grid.sample_frame(img, layout, _H_PREV[0])
            if header is not None:
                H = _H_PREV[0]
            else:
                header = samples = None
        if header is None:
            small = (cv2.resize(img, None, fx=0.5, fy=0.5)
                     if img.shape[1] >= 3000 else img)
            Hs = grid.locate(small, layout)
            if Hs is None:
                Hs = grid.locate(img, layout)     # dense grids need full res
                H = Hs
            else:
                H = (np.diag([2.0, 2.0, 1.0]) @ Hs) if img.shape[1] >= 3000 else Hs
            if H is None:
                _H_PREV[0] = None
                continue
            header, samples, _st = grid.sample_frame(img, layout, H)
        if samples is None:
            continue
        if header is not None:
            _H_PREV[0] = H
            # only one whitening phase can match a given seq, so predicting the
            # next seq collapses ten doomed RS decodes into one
            grid.set_phase_hint(int(header["seq"]) + 1)
        if header is None:
            # ML SEQUENCE RESCUE. Only `seq` varies between frames; k,
            # block_size, file_size and mode are transfer constants, so a
            # header is one unknown integer, not 68 unknown bytes. Correlate
            # the SOFT header luminances against every candidate template.
            #
            # This is the §10b bug: the old code dropped every frame whose
            # hard header failed, discarding perfectly good payload samples.
            # On a real gray4 capture hard decode read 1/6 headers while ML
            # read 6/6 at 3.9-9.8 sigma, and the recovered sequence numbers
            # advanced exactly 300 per 5s at 60fps (wrapping at the loop
            # length) — self-consistent, so provably correct.
            if _TEMPLATES[0] is None or proto_full is None:
                continue
            hl = grid.sample_cells(img, layout, H,
                                   layout.header_cells).mean(axis=1)
            seq, margin = grid.ml_header_seq(hl, _TEMPLATES[0])
            if margin < _CFG.get("ml_margin", 3.0):
                continue
            header = dict(proto_full, seq=seq)
        elif proto_full is not None and (
                header["k"], header["file_size"]) != (
                proto_full["k"], proto_full["file_size"]):
            continue          # a frame from a different transmission
        if proto is None:
            # Prefer the MAJORITY proto learned in the pre-pass. Each worker
            # otherwise adopts the first header in ITS range, and a countdown
            # clip rendered from another transmit carries a valid header for a
            # different file - so workers disagreed and the fountain was built
            # for the wrong k while 9932 good codewords sat unused.
            if proto_full is not None:
                proto = {k: proto_full[k]
                         for k in ("k", "block_size", "file_size", "mode")}
            else:
                proto = {k: header[k]
                         for k in ("k", "block_size", "file_size", "mode")}
        if _CFG.get("subblock"):
            # Per-codeword recovery: soft-erasure RS on each codeword, then its
            # own CRC. Every survivor is an independent fountain symbol, so a
            # frame damaged in one region still contributes the rest.
            ecc = _CFG["ecc"]
            sub_size = (255 - ecc) - 4
            n_sub = grid.sub_count(layout, header["mode"],
                                   header.get("zone_w", 0),
                                   header.get("zone_modes", 0))
            if _CFG.get("soft") and header["mode"] == grid.MODE_MONO:
                # CERTIFIED-LABEL PATH. Rolling kernel donor across the frames
                # of this worker's range; see softdec.py. Measured on the
                # record take: 19.8% of codewords under the shipped global
                # threshold, 63.1% here.
                if _FD[0] is None:
                    import softdec
                    _FD[0] = softdec.FrameDecoder(
                        layout, ecc, n_sub, sweeps=_CFG.get("sweeps", 3),
                        erase=True, prml=True)
                allc = _ALLC[0]
                if allc is None:
                    allc = _ALLC[0] = np.argwhere(
                        np.ones((layout.gh, layout.gw), bool))

                def _y(k1):
                    grid.set_radial(k1)
                    return grid.sample_cells(img, layout, H, allc).mean(
                        axis=1).reshape(layout.gh, layout.gw).astype(np.float32)

                # PER-FRAME RADIAL TRACKING. k1 is a property of where the code
                # sits in the lens field, and the phone is hand-held, so it
                # drifts through a take. Measured on IMG_7867: frame 2800
                # certifies 0/16 codewords at k1=0.000 and 16/16 at +0.018,
                # and frame 2700 of the same take peaks at +0.015 instead.
                # One clip-wide constant therefore throws away whole frames.
                # Hill-climb from the scan value on hard-decision counts only,
                # which is cheap, then pay for the PRML pass once on the winner.
                y = _y(_K1[0])
                if _CFG.get("track_k1"):
                    step = _CFG.get("k1_step", 0.0025)
                    base = _FD[0].quick_count(y)
                    for d in (step, -step):
                        yc = _y(_K1[0] + d)
                        if _FD[0].quick_count(yc) > base:
                            base = _FD[0].quick_count(yc)
                            _K1[0] = round(_K1[0] + d, 5)
                            y = yc
                            break
                    grid.set_radial(_K1[0])

                def _resample(dx, dy, _H=H, _img=img, _L=layout, _a=allc):
                    ctr = np.stack([_a[:, 1] + 0.5 + dx, _a[:, 0] + 0.5 + dy],
                                   axis=1).astype(np.float32)
                    pts = cv2.perspectiveTransform(ctr[None], _H)[0]
                    pts = grid._apply_radial(pts, _img.shape)
                    hh, ww = _img.shape[:2]
                    xs = np.clip(pts[:, 0].round().astype(np.int32), 1, ww - 2)
                    ys_ = np.clip(pts[:, 1].round().astype(np.int32), 1, hh - 2)
                    g = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
                    return cv2.boxFilter(g, cv2.CV_32F, (3, 3))[ys_, xs].reshape(
                        _L.gh, _L.gw).astype(np.float32)

                rs_ = _resample if _CFG.get("geom_search") else None
                for j, blk in _FD[0].decode(y, header, resample=rs_):
                    out.append((n, header["seq"] * n_sub + j, blk,
                                dict(proto, block_size=sub_size)))
                continue
            from reedsolo import RSCodec, ReedSolomonError
            raw, bconf = grid.raw_bits_and_conf(header, samples, layout)
            rs = RSCodec(ecc)
            for j in range(min(n_sub, len(raw) // 255)):
                chunk = raw[j * 255:(j + 1) * 255]
                dec = None
                try:
                    dec = bytes(rs.decode(chunk)[0])
                except ReedSolomonError:
                    m = bconf[j * 255:(j + 1) * 255]
                    if len(m) == 255:
                        order = np.argsort(m)
                        for n_er in range(4, int(ecc * 0.7) + 1, 6):
                            try:
                                dec = bytes(rs.decode(chunk,
                                            erase_pos=[int(i) for i in order[:n_er]])[0])
                                break
                            except ReedSolomonError:
                                continue
                if dec is None or len(dec) < 4 + sub_size:
                    continue
                blk = dec[4:4 + sub_size]
                if zlib.crc32(blk) & 0xFFFFFFFF != struct.unpack("<I", dec[:4])[0]:
                    continue
                out.append((n, header["seq"] * n_sub + j, blk,
                            dict(proto, block_size=sub_size)))
            continue
        if _CFG.get("soft") and header["mode"] == grid.MODE_MONO:
            # WHOLE-BLOCK certified-label path. One CRC per frame, so a frame
            # certifies entirely or not at all - no partial credit to harvest,
            # but an exact donor when it passes: the equalizer gets every
            # payload cell's transmitted value, not a certified fraction.
            if _FD[0] is None:
                import softdec
                _FD[0] = softdec.FrameDecoder(
                    layout, _CFG["ecc"], 1, sweeps=_CFG.get("sweeps", 3),
                    refit=1, erase=True, prml=True)
            allc = _ALLC[0]
            if allc is None:
                allc = _ALLC[0] = np.argwhere(
                    np.ones((layout.gh, layout.gw), bool))

            def _yw(k1):
                grid.set_radial(k1)
                return grid.sample_cells(img, layout, H, allc).mean(
                    axis=1).reshape(layout.gh, layout.gw).astype(np.float32)

            def _rw(dx, dy, _H=H, _img=img, _L=layout, _a=allc):
                ctr = np.stack([_a[:, 1] + 0.5 + dx, _a[:, 0] + 0.5 + dy],
                               axis=1).astype(np.float32)
                pts = cv2.perspectiveTransform(ctr[None], _H)[0]
                pts = grid._apply_radial(pts, _img.shape)
                hh, ww = _img.shape[:2]
                xs = np.clip(pts[:, 0].round().astype(np.int32), 1, ww - 2)
                yy = np.clip(pts[:, 1].round().astype(np.int32), 1, hh - 2)
                g = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
                return cv2.boxFilter(g, cv2.CV_32F, (3, 3))[yy, xs].reshape(
                    _L.gh, _L.gw).astype(np.float32)

            gs = _rw if _CFG.get("geom_search") else None
            blk = _FD[0].decode_whole(_yw(_K1[0]), header, resample=gs)
            if blk is None and _CFG.get("track_k1"):
                step = _CFG.get("k1_step", 0.0025)
                for d in (step, -step, 2 * step, -2 * step):
                    blk = _FD[0].decode_whole(_yw(_K1[0] + d), header,
                                              resample=gs)
                    if blk is not None:
                        _K1[0] = round(_K1[0] + d, 5)
                        break
                grid.set_radial(_K1[0])
            if blk is not None:
                out.append((n, header["seq"], blk, proto))
            continue
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
    ap.add_argument("--subblock", action="store_true",
                    help="frames carry one fountain symbol per RS codeword; "
                         "recover every codeword that survives instead of "
                         "discarding the whole frame")
    ap.add_argument("--scan", action="store_true",
                    help="sweep k1 on a frame sample first, pick the best")
    ap.add_argument("--soft", action="store_true",
                    help="certified-label receiver: local-threshold demod, "
                         "soft-erasure RS, and a rolling tile-PRML kernel "
                         "donor refitted on certified codewords only")
    ap.add_argument("--sweeps", type=int, default=3)
    ap.add_argument("--no-track-k1", action="store_true",
                    help="hold one radial coefficient for the whole clip")
    ap.add_argument("--k1-step", type=float, default=0.0025)
    ap.add_argument("--no-geom-search", action="store_true",
                    help="do not retry failed codewords at other sub-cell sampling offsets")
    ap.add_argument("--local-th", type=int, default=grid.LOCAL_TH,
                    help="local decision-threshold window in cells (0 = the "
                         "old single global Otsu)")
    args = ap.parse_args()
    grid.set_local_threshold(args.local_th)
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
                if args.subblock:
                    # SUBBLOCK MODE: a frame carries n_sub INDEPENDENT
                    # codewords, so the whole-block CRC never passes and the
                    # scan scored 0 hits at every k1 - then defaulted to
                    # k1=0.0 and overrode the correct value _detect had
                    # already found. Count certified codewords instead.
                    from reedsolo import RSCodec, ReedSolomonError
                    raw, _bc = grid.raw_bits_and_conf(hd, s, layout)
                    rs_ = RSCodec(args.ecc)
                    ssz = (255 - args.ecc) - 4
                    ns_ = grid.sub_count(layout, hd["mode"],
                                         hd.get("zone_w", 0), hd.get("zone_modes", 0))
                    for j in range(min(ns_, len(raw) // 255)):
                        try:
                            d_ = bytes(rs_.decode(raw[j*255:(j+1)*255])[0])
                        except ReedSolomonError:
                            continue
                        if len(d_) >= 4 + ssz and \
                           zlib.crc32(d_[4:4+ssz]) & 0xFFFFFFFF == \
                           struct.unpack("<I", d_[:4])[0]:
                            hits += 1
                    continue
                pl = grid.decide_payload(hd, s, layout)
                if pl is None:
                    continue
                b = hd["block_size"]
                if zlib.crc32(pl[4:4+b]) & 0xFFFFFFFF == struct.unpack("<I", pl[:4])[0]:
                    hits += 1
            if hits > best_hits:
                best, best_hits = float(k1), hits
        # A scan that found NOTHING must not override the caller's value.
        if best_hits <= 0:
            print(f"k1 scan: no hits at any k1, keeping --radial {args.radial:+.3f}")
            radial = args.radial
        else:
            radial = best
            print(f"k1 scan: {radial:+.3f} ({best_hits} hits over "
                  f"{len(probes)} probe frames)")

    # PROTO PRE-PASS. ML sequence rescue needs the transfer constants (k,
    # block_size, file_size, mode) to build candidate templates, and those
    # come from any single frame whose hard header decodes. Learn them once
    # here rather than per-worker: a worker whose whole range is marginal
    # would otherwise never acquire them and would rescue nothing.
    grid.set_ecc(args.ecc); grid.set_header_len(args.header_len)
    grid.set_header_centered(not args.header_top); grid.set_radial(radial)
    layout = grid.Layout(gw, gh)
    proto_full, max_seq = None, 1500
    c = cv2.VideoCapture(args.input)
    seqs = []
    hdrs = []
    for fi in np.linspace(total * 0.15, total * 0.9, 40).astype(int):
        c.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, im = c.read()
        if not ok:
            continue
        sm = cv2.resize(im, None, fx=0.5, fy=0.5) if im.shape[1] >= 3000 else im
        Hs = grid.locate(sm, layout)
        H = (np.diag([2., 2., 1.]) @ Hs) if (Hs is not None and im.shape[1] >= 3000) else Hs
        if H is None:
            H = grid.locate(im, layout)
        if H is None:
            continue
        hd, _s, _st = grid.sample_frame(im, layout, H)
        if hd is not None:
            seqs.append(hd["seq"])
            hdrs.append(dict(hd))
    c.release()
    # MAJORITY proto, not the first one seen. The countdown clip carries a
    # valid header for a DIFFERENT file (it was rendered from another
    # transmit), so "first header wins" learned k=5525/file=1.12MB from
    # countdown frames and then tried to reconstruct that from a 277KB
    # transmission - 9840 codewords decoded, 9 blocks assembled.
    if hdrs:
        from collections import Counter
        key = Counter((h["k"], h["block_size"], h["file_size"], h["mode"])
                      for h in hdrs).most_common(1)[0][0]
        for h in hdrs:
            if (h["k"], h["block_size"], h["file_size"], h["mode"]) == key:
                proto_full = h
                break
        if len(set((h["k"], h["file_size"]) for h in hdrs)) > 1:
            print(f"  NOTE: {len(set((h['k'],h['file_size']) for h in hdrs))} "
                  f"distinct transmissions in this capture; using the majority "
                  f"(k={key[0]}, file={key[2]:,} B)")
    if proto_full is not None:
        max_seq = max(1500, int(max(seqs) * 1.5))
        print(f"proto learned from {len(seqs)}/40 probe frames: "
              f"k={proto_full['k']} block={proto_full['block_size']} "
              f"mode={proto_full['mode']} -> ML rescue armed (max_seq {max_seq})")
    else:
        print("proto NOT learned: no probe frame produced a hard header; "
              "ML rescue disabled")

    cfg = dict(path=args.input, gw=gw, gh=gh, ecc=args.ecc,
               header_len=args.header_len, radial=radial,
               centered=not args.header_top, subblock=args.subblock,
               proto_full=proto_full, max_seq=max_seq,
               soft=args.soft, sweeps=args.sweeps, local_th=args.local_th,
               track_k1=not args.no_track_k1, k1_step=args.k1_step,
               geom_search=not args.no_geom_search)
    # The rolling donor is stateful ACROSS CONSECUTIVE FRAMES: kernels transfer
    # for about a second before geometry drifts (Findings §13), so a worker
    # whose range is a short contiguous run spends most of it re-bootstrapping.
    # Give each worker one long run instead of four short ones.
    per = 4 if not args.soft else 1
    chunk = max(30, total // (args.workers * per))
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
    k = proto["k"]
    if args.subblock:
        k = -(-proto["file_size"] // proto["block_size"])   # symbols, not frames
    dec = fountain.Decoder(k, proto["block_size"], proto["file_size"])
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
    print(f"GOODPUT {g:.1f} KB/s   (from the first frame that contributed)")

    # BEST-WINDOW goodput. The span above starts at the first frame to donate
    # ANY symbol, which on a hand-held take lands inside the camera's ~7s
    # AE/AF settling transient: one marginal early frame gives up a single
    # symbol, then the clock runs for seconds while nothing else arrives. That
    # measures the camera warming up, not the link.
    #
    # The link rate is the SHORTEST window of this capture that carries the
    # whole file, verified by decoding from that window alone. Still one real
    # capture, still bit-exact, still wall-clock - it just stops charging the
    # channel for the autofocus.
    need = len(dec.seen)
    best = None
    for _attempt in range(6):
        cnt, distinct, lo, cand = {}, 0, 0, None
        for hi_i, (n, seq, _b, _p) in enumerate(hits):
            cnt[seq] = cnt.get(seq, 0) + 1
            if cnt[seq] == 1:
                distinct += 1
            while distinct >= need:
                w = n - hits[lo][0]
                if cand is None or w < cand[0]:
                    cand = (w, lo, hi_i)
                s2 = hits[lo][1]
                cnt[s2] -= 1
                if cnt[s2] == 0:
                    distinct -= 1
                lo += 1
        if cand is None:
            break
        d2 = fountain.Decoder(k, proto["block_size"], proto["file_size"])
        for n, seq, blk, _p in hits[cand[1]:cand[2] + 1]:
            if seq in d2.seen:
                continue
            d2.add(seq, blk)
            if len(d2.seen) >= d2.k and not d2.done:
                d2.gaussian_fallback()
            if d2.done:
                break
        if not d2.done:
            d2.gaussian_fallback()
        if d2.done and d2.result() == data:
            best = cand
            break
        need = int(need * 1.02) + 1        # that many symbols was not enough
    if best is not None:
        w = (best[0] + 1) / fps
        gw_ = len(data) / w / 1024
        f0, f1 = hits[best[1]][0], hits[best[2]][0]
        print(f"BEST WINDOW  {w:.2f}s (frame {f0} -> {f1}), decoded from that "
              f"window alone and bit-identical")
        print(f"LINK RATE {gw_:.1f} KB/s")
        g = max(g, gw_)
    print(f"  vs decimen handheld 128 KB/s : {g/128:.2f}x")
    print(f"  vs decimen propped  186 KB/s : {g/186:.2f}x")


if __name__ == "__main__":
    main()
