#!/usr/bin/env python3
"""
decode_rolling.py — dual-seq decoder for 120fps-transmit captures at 4K60.

The rolling shutter makes each camera frame carry TWO consecutive code
frames, split at a seam that advances on the camera/display clock beat.
Proven in silico (exp_rolling_poc): dual-seq band assignment closes the
fountain bit-exact where single-seq assignment closes it CORRUPT.

Two passes over the capture:
  1. COLLECT: per frame - locate, hard header (seq s of the header's side),
     demod bits/conf once, per-row mid-fraction seam estimate + strength.
  2. ASSIGN: phase-fit (r0, dr) on the strong seam estimates, then per
     frame certify all codewords and file each by its side of the PREDICTED
     seam: header side -> s, far side -> s±1 (scan direction learned from
     the data: the assignment that yields no cross-loop content conflicts
     is the true one; the wrong direction collides massively).

Every block still passes RS+CRC, the cross-loop conflict filter drops any
index certified with two contents, and the sha256 is the final judge.

    python3 poc/decode_rolling.py CAPTURE.MOV out.bin --grid 274x178
"""
import argparse
import hashlib
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, str(Path(__file__).parent))
from codec import fountain, grid
from softdec import FrameDecoder

_W = {}


def _winit(cfg):
    cv2.setNumThreads(1)
    _W.update(cfg)
    grid.set_ecc(cfg["ecc"]); grid.set_header_len(28)
    grid.set_header_centered(True); grid.set_radial(cfg["radial"])


def _wrange(rng):
    """Certify one contiguous frame range; return per-frame results.

    Does ALL the expensive work once per frame (locate, sample, demod,
    certify) and returns (frame, header_seq, seam_row, seam_strength,
    [(j, block)]). Seam fitting and seq assignment happen in the parent,
    so neither depends on which worker saw which frame.
    """
    lo, hi = rng
    gw, gh = _W["gw"], _W["gh"]
    L = grid.Layout(gw, gh)
    n_sub = grid.sub_count(L, grid.MODE_MONO)
    pc = L.payload_cells
    allc = np.argwhere(np.ones((gh, gw), bool))
    fd = FrameDecoder(L, _W["ecc"], n_sub, erase=True, prml=False)
    nb = n_sub * 255
    cap = cv2.VideoCapture(_W["path"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    out = []
    n = lo
    while n < hi:
        ok, img = cap.read()
        if not ok:
            break
        n += 1
        big = img.shape[1] >= 3000
        sm = cv2.resize(img, None, fx=0.5, fy=0.5) if big else img
        Hs = grid.locate(sm, L)
        H = (((np.diag([2., 2., 1.]) @ Hs) if big else Hs)
             if Hs is not None else grid.locate(img, L))
        if H is None:
            continue
        hd, _s, _t = grid.sample_frame(img, L, H)
        if hd is None:
            continue
        y = grid.sample_cells(img, L, H, allc).mean(axis=1).reshape(
            gh, gw).astype(np.float32)
        lum = y[pc[:, 0], pc[:, 1]]
        bits, conf = grid._mono_decide(lum, L, pc)
        tau = 0.25 * float(np.median(np.abs(conf)))
        mid = (np.abs(conf) < tau).astype(np.float32)
        prof = np.zeros(gh, np.float32); cnt = np.zeros(gh, np.float32)
        np.add.at(prof, pc[:, 0], mid); np.add.at(cnt, pc[:, 0], 1)
        prof = prof / np.maximum(cnt, 1)
        smp = np.convolve(prof, np.ones(7, np.float32) / 7, "same")
        r = int(np.argmax(smp))
        bc = conf[: nb * 8].reshape(nb, 8).min(axis=1)
        blocks, _m, _cb = fd.certify(bits, bc)
        out.append((n, int(hd["seq"]), r, float(smp[r]),
                    [(j, bytes(b)) for j, b in blocks]))
    cap.release()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("output")
    ap.add_argument("--grid", default="274x178")
    ap.add_argument("--ecc", type=int, default=48)
    ap.add_argument("--radial", type=float, default=0.020)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--end-frame", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--mix-rows", type=int, default=14,
                    help="rows around the predicted seam treated as erasures")
    args = ap.parse_args()

    grid.set_ecc(args.ecc); grid.set_header_len(28)
    grid.set_header_centered(True); grid.set_radial(args.radial)
    gw, gh = (int(v) for v in args.grid.split("x"))
    L = grid.Layout(gw, gh)
    n_sub = grid.sub_count(L, grid.MODE_MONO)
    SUB = (255 - args.ecc) - 4
    pc = L.payload_cells
    allc = np.argwhere(np.ones((gh, gw), bool))
    fd = FrameDecoder(L, args.ecc, n_sub, erase=True, prml=False)
    cells = pc[: n_sub * 255 * 8]
    cw_rows = [(int(cells[j*255*8:(j+1)*255*8, 0].min()),
                int(cells[j*255*8:(j+1)*255*8, 0].max()))
               for j in range(n_sub)]
    hdr_hi = int(L.header_cells[:, 0].max())
    nb = n_sub * 255

    cap = cv2.VideoCapture(args.capture)
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    end = min(tot, args.end_frame) if args.end_frame else tot
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    cap.release()

    # ---- pass 1: certify every frame, in parallel. Seam fitting and seq
    # assignment happen in the parent afterwards, so no worker's result
    # depends on which range it was given.
    t0 = time.time()
    wcfg = dict(path=args.capture, gw=gw, gh=gh, ecc=args.ecc,
                radial=args.radial)
    span_n = end - args.start_frame
    chunk = max(60, span_n // (args.workers * 2))
    ranges = [(a, min(a + chunk, end))
              for a in range(args.start_frame, end, chunk)]
    with Pool(args.workers, initializer=_winit, initargs=(wcfg,)) as pool:
        res = pool.map(_wrange, ranges)
    frames = {}
    est = []
    for r_ in res:
        for (n_, seq_, seam_r, strength, blocks) in r_:
            frames[n_] = (seq_, blocks)
            est.append((n_, seam_r, strength))
    print(f"pass 1: {len(frames)} frames certified in {time.time()-t0:.0f}s "
          f"({len(frames)/max(time.time()-t0,1e-9):.1f} fps)")
    if not frames:
        sys.exit("FAILED: nothing located")

    # ---- phase fit on strong seam frames
    est.sort(key=lambda e: -e[2])
    strong = est[: max(30, len(est) // 3)]
    fs = np.array([e[0] for e in strong], np.float64)
    rs_ = np.array([e[1] for e in strong], np.float64)
    best_fit, best_err = (0.0, 0.0), 1e18
    for dr_c in np.arange(1.0, 80.0, 0.05):
        ph = (rs_ - dr_c * fs) % gh
        med = np.median(ph)
        err = np.median(np.abs((ph - med + gh / 2) % gh - gh / 2))
        if err < best_err:
            best_err, best_fit = err, (float(med), float(dr_c))
    r0, dr = best_fit
    print(f"seam fit: r0 {r0:.1f}  dr {dr:.2f} rows/frame  "
          f"residual {best_err:.1f} rows over {len(strong)} strong frames")

    # ---- pass 2: certify + assign, both scan directions, pick by conflicts
    def harvest(direction):
        pool = {}
        conflicts = 0
        for f, (s_hdr, blocks) in frames.items():
            seam = int(r0 + f * dr) % gh
            hdr_top = hdr_hi < seam
            far = s_hdr + direction if hdr_top else s_hdr - direction
            near = s_hdr
            for j, blk in blocks:
                lo_r, hi_r = cw_rows[j]
                if hi_r < seam - args.mix_rows // 2:
                    sq = near if hdr_top else far
                elif lo_r >= seam + args.mix_rows // 2:
                    sq = far if hdr_top else near
                else:
                    continue
                if sq < 0:
                    continue          # far side before the first transmitted
                                      # frame; nothing legal lives there
                idx = sq * n_sub + j
                b = bytes(blk)
                if idx in pool and pool[idx] != b:
                    conflicts += 1
                pool.setdefault(idx, b)
        return pool, conflicts

    p_plus, c_plus = harvest(+1)
    p_minus, c_minus = harvest(-1)
    print(f"direction +1: {len(p_plus)} symbols, {c_plus} conflicts   "
          f"direction -1: {len(p_minus)} symbols, {c_minus} conflicts")
    # The true scan direction produces (near-)zero cross-loop content
    # conflicts; the wrong one collides on every repeated index. That test
    # only has evidence when indices actually RECUR, i.e. the capture spans
    # more than one transmit loop. With too little footage both directions
    # read zero and a coin flip silently closes the fountain CORRUPT (seen
    # on a 514-frame synthetic). Refuse to guess instead.
    if c_plus == c_minus:
        sys.exit(f"FAILED: scan direction undecidable ({c_plus} conflicts "
                 f"both ways). The capture must span more than one transmit "
                 f"loop for indices to recur; give it more frames.")
    pool = p_plus if c_plus < c_minus else p_minus
    print(f"scan direction {'+1' if c_plus < c_minus else '-1'} chosen "
          f"({min(c_plus, c_minus)} vs {max(c_plus, c_minus)} conflicts)")
    if min(c_plus, c_minus) > 0:
        print(f"  NOTE: {min(c_plus, c_minus)} residual conflicts under the "
              f"chosen direction; those indices keep first-seen content")

    # transfer constants from any collected frame's validated header
    order = sorted(pool)
    cap = cv2.VideoCapture(args.capture)
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(frames) - 1)
    ok, img = cap.read()
    cap.release()
    hd = None
    if ok:
        big = img.shape[1] >= 3000
        sm = cv2.resize(img, None, fx=0.5, fy=0.5) if big else img
        Hs = grid.locate(sm, L)
        H = (((np.diag([2., 2., 1.]) @ Hs) if big else Hs)
             if Hs is not None else grid.locate(img, L))
        if H is not None:
            hd, _s, _t = grid.sample_frame(img, L, H)
    if hd is None:
        sys.exit("FAILED: no proto header")
    file_size = int(hd["file_size"])
    kk = -(-file_size // SUB)
    d = fountain.Decoder(kk, SUB, file_size)
    for idx in order:
        if idx in d.seen:
            continue
        d.add(idx, pool[idx])
        if len(d.seen) >= d.k and not d.done:
            d.gaussian_fallback()
        if d.done:
            break
    if not d.done:
        d.gaussian_fallback()
    if not d.done:
        sys.exit(f"FAILED: {len(d.seen)}/{kk} symbols")
    data = d.result()
    Path(args.output).write_bytes(data)
    span = (max(frames) - min(frames) + 1) / fps
    print(f"recovered {len(data):,} bytes")
    print(f"sha256 {hashlib.sha256(data).hexdigest()}")
    print(f"symbols {len(pool)} over {len(frames)} frames "
          f"({len(pool)/max(len(frames),1):.1f}/frame)")
    print(f"span {span:.2f}s  GOODPUT {len(data)/span/1024:.1f} KB/s")


if __name__ == "__main__":
    main()
