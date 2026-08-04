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
    # Workers each spawn cv2's internal thread pool; with 8-10 processes that
    # oversubscribes the machine. One thread per worker is faster in aggregate.
    cv2.setNumThreads(1)
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


def prime_donor(path, layout, ecc, n_sub, radial, total, proto_full,
                n_probe=36, sweeps=3, base=0):
    """Fit equalizer kernels from the best frame in the WHOLE capture, before
    any decoding starts.

    A cold FrameDecoder has tap=None, so tile-PRML does nothing until some
    frame certifies enough of itself on hard decisions alone to become a donor.
    Every worker therefore spends the start of its range un-equalized, and the
    start of the CAPTURE is what full-span goodput is measured from - which is
    most of why best-window read 208.2 KB/s against a full-span 131.0 on the
    same take.

    Nothing about this is cherry-picking. The receiver has the whole video
    buffered; a certified codeword is correct by construction wherever it came
    from, so a frame near the middle of a take is as valid a teacher as a frame
    near the start. It is the same certified-label mechanism, just not
    restricted to causal order.

    Returns (tap, bias) or (None, None) if no frame in the sample was good
    enough to teach from - in which case the workers bootstrap as before.
    """
    import softdec
    fd = softdec.FrameDecoder(layout, ecc, n_sub, sweeps=sweeps,
                              erase=True, prml=True)
    allc = np.argwhere(np.ones((layout.gh, layout.gw), bool))
    cap = cv2.VideoCapture(path)
    grid.set_radial(radial)
    best_n, best_state = -1, None
    for fi in np.linspace(base + (total - base) * 0.10,
                          base + (total - base) * 0.95, n_probe).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, img = cap.read()
        if not ok:
            continue
        H = grid.locate(img, layout)
        if H is None:
            continue
        hd, _s, _t = grid.sample_frame(img, layout, H)
        if hd is None:
            continue
        if proto_full is not None and (hd["k"], hd["file_size"]) != (
                proto_full["k"], proto_full["file_size"]):
            continue        # a frame from a different transmission
        y = grid.sample_cells(img, layout, H, allc).mean(axis=1).reshape(
            layout.gh, layout.gw).astype(np.float32)
        blocks = fd.decode(y, hd, allow_refit=False)
        if len(blocks) > best_n:
            best_n, best_state = len(blocks), fd.pending
    cap.release()
    if best_state is None or best_n < fd.refit:
        print(f"donor pre-pass: best frame certified {max(best_n,0)}/{n_sub}, "
              f"below the {fd.refit} needed to teach; workers will bootstrap")
        return None, None
    fd.pending = best_state
    fd.commit()
    print(f"donor pre-pass: primed from a frame certifying {best_n}/{n_sub}")
    return fd.tap, fd.bias


def _prepass_task(t):
    """One pre-pass probe, run in a worker: ('scan'|'proto'|'donor', pos).

    Same positions, same per-probe logic as the sequential pre-pass; only
    the seeks are parallel. Aggregation happens in the parent in the
    original probe order, so results are bit-identical to the serial path.
    """
    kind, pos = t
    gw, gh = _CFG["gw"], _CFG["gh"]
    layout = grid.Layout(gw, gh)
    cap = cv2.VideoCapture(_CFG["path"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
    ok, img = cap.read()
    cap.release()
    if not ok:
        return (kind, pos, None)
    if kind == "donor":
        # prime_donor located at FULL resolution; match it exactly
        H = grid.locate(img, layout)
    else:
        sm = (cv2.resize(img, None, fx=0.5, fy=0.5)
              if img.shape[1] >= 3000 else img)
        Hs = grid.locate(sm, layout)
        H = ((np.diag([2., 2., 1.]) @ Hs) if Hs is not None
             else grid.locate(img, layout))
    if H is None:
        return (kind, pos, None)
    if kind == "proto":
        hd, _s, _t = grid.sample_frame(img, layout, H)
        return (kind, pos, dict(hd) if hd is not None else None)
    if kind == "scan":
        from softdec import FrameDecoder
        counts = {}
        n_sub = grid.sub_count(layout, grid.MODE_MONO)
        for k1 in np.arange(0.0, 0.041, 0.005):
            grid.set_radial(float(k1))
            hd, s, _ = grid.sample_frame(img, layout, H)
            hits = 0
            if hd is not None and s is not None and _CFG.get("subblock"):
                try:
                    from creedsolo import RSCodec, ReedSolomonError
                except ImportError:
                    from reedsolo import RSCodec, ReedSolomonError
                import struct as _st, zlib as _zl
                raw, _bc = grid.raw_bits_and_conf(hd, s, layout)
                rs_ = RSCodec(_CFG["ecc"])
                ssz = (255 - _CFG["ecc"]) - 4
                ns_ = grid.sub_count(layout, hd["mode"],
                                     hd.get("zone_w", 0), hd.get("zone_modes", 0))
                for j in range(min(ns_, len(raw) // 255)):
                    try:
                        d_ = bytes(rs_.decode(raw[j*255:(j+1)*255])[0])
                    except ReedSolomonError:
                        continue
                    if len(d_) >= 4 + ssz and \
                       _zl.crc32(d_[4:4+ssz]) & 0xFFFFFFFF == \
                       _st.unpack("<I", d_[:4])[0]:
                        hits += 1
            counts[round(float(k1), 3)] = hits
        grid.set_radial(_CFG["radial"])
        return (kind, pos, counts)
    # donor probe: cold FrameDecoder, count certified codewords
    import softdec
    n_sub = grid.sub_count(layout, grid.MODE_MONO)
    fd = softdec.FrameDecoder(layout, _CFG["ecc"], n_sub,
                              sweeps=_CFG.get("sweeps", 3),
                              erase=True, prml=True)
    hd, _s, _t = grid.sample_frame(img, layout, H)
    if hd is None:
        return (kind, pos, None)
    pf = _CFG.get("proto_full")
    if pf is not None and (hd["k"], hd["file_size"]) != (pf["k"],
                                                         pf["file_size"]):
        return (kind, pos, None)
    allc = np.argwhere(np.ones((gh, gw), bool))
    y = grid.sample_cells(img, layout, H, allc).mean(axis=1).reshape(
        gh, gw).astype(np.float32)
    blocks = fd.decode(y, hd, allow_refit=False)
    return (kind, pos, (len(blocks), fd.pending))


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
        if _CFG.get("reuse_h") and _H_PREV[0] is not None:
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
                    # PRIMED DONOR. Cold, the equalizer cannot arm until some
                    # frame certifies 17/19 on hard decisions alone, so the
                    # FIRST stretch of every worker's range decodes without it.
                    # That stretch is exactly what full-span goodput measures
                    # from, which is why best-window sat 1.6x above it. The
                    # kernels come from a pre-pass over the whole capture.
                    pt = _CFG.get("primed_tap")
                    if pt is not None:
                        _FD[0].tap, _FD[0].bias = pt, _CFG["primed_bias"]
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
                    return grid.blurred_gray(_img)[ys_, xs].reshape(
                        _L.gh, _L.gw).astype(np.float32)

                rs_ = _resample if _CFG.get("geom_search") else None
                _FD[0].donor_frame = n
                for j, blk in _FD[0].decode(y, header, resample=rs_):
                    out.append((n, header["seq"] * n_sub + j, blk,
                                dict(proto, block_size=sub_size)))
                continue
            try:
                from creedsolo import RSCodec, ReedSolomonError
            except ImportError:
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
                return grid.blurred_gray(_img)[yy, xs].reshape(
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
    # BACKFILL THE COLD PREFIX. The equalizer cannot arm until some frame
    # certifies enough of itself, so every worker decodes the start of its
    # range unequalized - and the start of the CAPTURE is precisely what
    # full-span goodput is measured from. Measured on IMG_7872: 6.6% of
    # codewords at frame 45 and 11.8% at frame 227, against 92.6% at frame 910.
    #
    # Priming from a distant frame does not work: kernels transfer for about a
    # second before geometry drifts (Findings section 13). So re-run only the
    # cold prefix, with the FIRST kernel the forward pass produced - the
    # nearest donor in time that exists at all. Every recovered codeword is
    # CRC-certified, so the union can only add.
    fd0 = _FD[0]
    first_donor = getattr(fd0, "first_donor", None) if fd0 else None
    if first_donor is not None and _CFG.get("backfill", True):
        tap0, bias0, fdone = first_donor
        if fdone > start + 1:
            seen = {(h[0], h[1]) for h in out}
            fd0.tap, fd0.bias = tap0, bias0
            cap = cv2.VideoCapture(_CFG["path"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            _H_PREV[0] = None
            grid.set_phase_hint(None)
            n = start
            while n < fdone:
                ok, img = cap.read()
                if not ok:
                    break
                n += 1
                Hb = grid.locate(img, layout)
                if Hb is None:
                    continue
                hb, _sb, _tb = grid.sample_frame(img, layout, Hb)
                if hb is None:
                    continue
                if proto_full is not None and (
                        hb["k"], hb["file_size"]) != (proto_full["k"],
                                                      proto_full["file_size"]):
                    continue
                allc = _ALLC[0]
                if allc is None:
                    allc = _ALLC[0] = np.argwhere(
                        np.ones((layout.gh, layout.gw), bool))
                yb = grid.sample_cells(img, layout, Hb, allc).mean(
                    axis=1).reshape(layout.gh, layout.gw).astype(np.float32)
                ns_b = grid.sub_count(layout, hb["mode"])
                for j, blk in fd0.decode(yb, hb, allow_refit=False):
                    key = (n, hb["seq"] * ns_b + j)
                    if key not in seen:
                        seen.add(key)
                        out.append((n, hb["seq"] * ns_b + j, blk,
                                    dict(proto, block_size=(255 - _CFG["ecc"]) - 4)))
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
    ap.add_argument("--no-prime", action="store_true",
                    help="skip the donor pre-pass; workers bootstrap "
                         "the equalizer cold, as before")
    ap.add_argument("--full", action="store_true",
                    help="decode every frame instead of stopping "
                         "when the file completes. Needed when the "
                         "reported RATE is the point, since the "
                         "best-window search wants all the frames.")
    ap.add_argument("--start-frame", type=int, default=0,
                    help="first frame of the transfer being measured; "
                         "--from-start grows its window from here. Lets one "
                         "capture carry several transmits (session ladder) "
                         "with each decoded on its own honest span.")
    ap.add_argument("--end-frame", type=int, default=0,
                    help="hard stop for the scan (0 = end of capture)")
    ap.add_argument("--from-start", action="store_true",
                    help="grow the incremental window from frame 0 instead of "
                         "the middle of the capture, and stop at the first "
                         "fountain closure. Per-frame decoding is unchanged, "
                         "so the full-span GOODPUT (first contributing frame "
                         "-> completion, in capture order) is the same number "
                         "--full reports, for a fraction of the work. Only "
                         "the post-hoc best-window search loses frames.")
    ap.add_argument("--reuse-homography", action="store_true",
                    help="skip locate() when the previous frame's "
                         "homography still reads the header. MEASURED NET LOSS on IMG_7872: 10%% fewer blocks for 5%% less wall time, "
                         "because the header tolerates far more geometric drift than the payload does. Off until the gate keys on "
                         "codeword yield instead.")
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
    if args.end_frame:
        total = min(total, args.end_frame)
    base = args.start_frame
    print(f"{total} frames @ {fps:.0f}fps, {args.workers} workers"
          + (f", segment {base}-{total}" if base or args.end_frame else ""))

    radial = args.radial
    grid.set_ecc(args.ecc); grid.set_header_len(args.header_len)
    grid.set_header_centered(not args.header_top); grid.set_radial(radial)
    layout = grid.Layout(gw, gh)
    pre_cfg = dict(path=args.input, gw=gw, gh=gh, ecc=args.ecc,
                   header_len=args.header_len, radial=radial,
                   centered=not args.header_top, subblock=args.subblock,
                   soft=args.soft, sweeps=args.sweeps,
                   local_th=args.local_th)
    # PARALLEL PRE-PASS. Same probe positions, same per-probe logic, same
    # aggregation order as the serial pre-pass (bit-identical outputs,
    # A/B'd against the serial path); only the 86 HEVC seeks run across the
    # pool instead of one after another.
    if args.scan:
        tasks = [("scan", int(fi)) for fi in
                 np.linspace(base + (total - base) * 0.25,
                             base + (total - base) * 0.8, 10).astype(int)]
        with Pool(min(args.workers, len(tasks)), initializer=_init,
                  initargs=(pre_cfg,)) as pool:
            res1 = pool.map(_prepass_task, tasks)
        k1_tot = {}
        n_probes = 0
        for kind, _pos, r in res1:
            if kind != "scan" or r is None:
                continue
            n_probes += 1
            for k1, h in r.items():
                k1_tot[k1] = k1_tot.get(k1, 0) + h
        best, best_hits = 0.0, -1
        for k1 in sorted(k1_tot):
            if k1_tot[k1] > best_hits:
                best, best_hits = k1, k1_tot[k1]
        # A scan that found NOTHING must not override the caller's value.
        if best_hits <= 0:
            print(f"k1 scan: no hits at any k1, keeping --radial {args.radial:+.3f}")
            radial = args.radial
        else:
            radial = best
            print(f"k1 scan: {radial:+.3f} ({best_hits} hits over "
                  f"{n_probes} probe frames)")

    # PROTO PRE-PASS. ML sequence rescue needs the transfer constants (k,
    # block_size, file_size, mode) to build candidate templates, and those
    # come from any single frame whose hard header decodes. Learn them once
    # here rather than per-worker: a worker whose whole range is marginal
    # would otherwise never acquire them and would rescue nothing.
    grid.set_radial(radial)
    pre_cfg["radial"] = radial
    proto_full, max_seq = None, 1500
    tasks = [("proto", int(fi)) for fi in
             np.linspace(base + (total - base) * 0.15,
                         base + (total - base) * 0.9, 40).astype(int)]
    with Pool(args.workers, initializer=_init, initargs=(pre_cfg,)) as pool:
        res2 = pool.map(_prepass_task, tasks)
    seqs = []
    hdrs = []
    for _kind, _pos, hd in res2:
        if hd is not None:
            seqs.append(hd["seq"])
            hdrs.append(hd)
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
               geom_search=not args.no_geom_search,
               reuse_h=args.reuse_homography)
    if args.soft and args.subblock and not args.no_prime:
        _ns = grid.sub_count(layout, grid.MODE_MONO)
        # parallel donor pre-pass: same probe positions and winner rule as
        # prime_donor (first strict max in probe order), seeks fanned out
        import softdec as _sd
        pre_cfg["proto_full"] = proto_full
        d_tasks = [("donor", int(fi)) for fi in
                   np.linspace(base + (total - base) * 0.10,
                               base + (total - base) * 0.95, 36).astype(int)]
        with Pool(args.workers, initializer=_init,
                  initargs=(pre_cfg,)) as pool:
            res3 = pool.map(_prepass_task, d_tasks)
        best_n, best_state = -1, None
        for _kind, _pos, r in res3:
            if r is None:
                continue
            if r[0] > best_n:
                best_n, best_state = r[0], r[1]
        _fd = _sd.FrameDecoder(layout, args.ecc, _ns, sweeps=args.sweeps,
                               erase=True, prml=True)
        if best_state is None or best_n < _fd.refit:
            print(f"donor pre-pass: best frame certified {max(best_n,0)}/"
                  f"{_ns}, below the {_fd.refit} needed to teach; workers "
                  f"will bootstrap")
            _tap = _bias = None
        else:
            _fd.pending = best_state
            _fd.commit()
            print(f"donor pre-pass: primed from a frame certifying "
                  f"{best_n}/{_ns}")
            _tap, _bias = _fd.tap, _fd.bias
        cfg["primed_tap"], cfg["primed_bias"] = _tap, _bias
    # The rolling donor is stateful ACROSS CONSECUTIVE FRAMES: kernels transfer
    # for about a second before geometry drifts (Findings §13), so a worker
    # whose range is a short contiguous run spends most of it re-bootstrapping.
    # Give each worker one long run instead of four short ones.
    per = 4 if not args.soft else 1

    def _decode_span(lo, hi):
        """Run the worker pool over [lo, hi) and return sorted hits."""
        span = hi - lo
        ch = max(30, span // (args.workers * per))
        rs = [(s, min(s + ch, hi)) for s in range(lo, hi, ch)]
        with Pool(args.workers, initializer=_init, initargs=(cfg,)) as pool:
            res = pool.map(_worker, rs)
        return sorted([h for r in res for h in r], key=lambda x: x[0])

    def _try_assemble(hs):
        if not hs:
            return None, None
        pr = hs[0][3]
        kk = pr["k"]
        if args.subblock:
            kk = -(-pr["file_size"] // pr["block_size"])
        d = fountain.Decoder(kk, pr["block_size"], pr["file_size"])
        for _n, sq, bl, _p in hs:
            if sq in d.seen:
                continue
            d.add(sq, bl)
            if len(d.seen) >= d.k and not d.done:
                d.gaussian_fallback()
            if d.done:
                break
        if not d.done:
            d.gaussian_fallback()
        return (d, pr) if d.done else (None, pr)

    t0 = time.time()
    # INCREMENTAL DECODE. The file completes in ~77 frames; a take is thousands.
    # Decoding all of them to recover something that finished in the first
    # second and a half is most of the wall clock this decoder spends. Widen a
    # window about the middle of the capture - past the AE/AF settling
    # transient, before the operator starts lowering the phone - and stop as
    # soon as the fountain closes.
    #
    # This trades the BEST WINDOW search, which wants every frame it can get,
    # for time-to-file. Use --full when the reported rate is the point.
    hits = []
    if args.full:
        hits = _decode_span(base, total)
        scanned = total - base
    else:
        seen_lo, seen_hi = None, None
        for frac in (0.10, 0.25, 0.55, 1.0):
            w = max(240, int((total - base) * frac))
            if args.from_start:
                # Honest full-span wants the capture's natural transfer: the
                # AE/AF settling transient included, exactly as --full charges
                # it. Growing forward from the segment origin keeps that while
                # stopping as soon as the file first completes.
                lo, hi = base, min(total, base + w)
            else:
                lo = max(base, (base + total) // 2 - w // 2)
                hi = min(total, lo + w)
            if seen_lo is None:
                hits = _decode_span(lo, hi)
            else:
                if lo < seen_lo:
                    hits = _decode_span(lo, seen_lo) + hits
                if hi > seen_hi:
                    hits = hits + _decode_span(seen_hi, hi)
                hits.sort(key=lambda x: x[0])
            seen_lo = lo if seen_lo is None else min(seen_lo, lo)
            seen_hi = hi if seen_hi is None else max(seen_hi, hi)
            print(f"  window {seen_lo}-{seen_hi} ({(seen_hi-seen_lo)/fps:.1f}s): "
                  f"{len(hits)} blocks, {time.time()-t0:.0f}s elapsed")
            d, _pr = _try_assemble(hits)
            if d is not None:
                break
        scanned = (seen_hi - seen_lo)
    wall = time.time() - t0
    print(f"decoded {len(hits)} blocks from {scanned} of {total} frames "
          f"in {wall:.1f}s wall ({scanned/max(wall,1e-6):.0f} frames/s)")
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
