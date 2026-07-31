#!/usr/bin/env python3
"""
decode.py — captured/simulated video -> recovered file + channel report.

    python3 decode.py capture.mp4 recovered.bin [--grid 120x68]

Reports per-frame outcomes and overall goodput accounting. The `cell_margin`
statistic is the seed of the soft-decision work: today it is only reported;
the decoder still makes hard cell decisions. When margins are persistently
low but RS still succeeds, that is the regime where feeding per-cell
confidence into the fountain decoder (soft-decision LT) buys real rate —
the research hook, deliberately left visible in the stats output.
"""
import argparse
import hashlib
import struct
import sys
import time
import zlib
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from codec import fountain, grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--grid", default="120x68")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--combine", action="store_true",
                    help="evidence-integrating receiver: average cell samples "
                         "across every capture of the same displayed frame "
                         "before deciding, instead of hard-deciding each "
                         "capture independently")
    ap.add_argument("--track", action="store_true",
                    help="tracking receiver: when per-frame detection fails, "
                         "locate on a sliding-window average of recent frames "
                         "(fiducials are static, noise integrates away, tremor "
                         "is smooth) and sample the current frame with that "
                         "homography")
    ap.add_argument("--repeat-hint", type=int, default=0,
                    help="captures per displayed frame (camera fps / display "
                         "fps). Lets the receiver treat the frame counter as "
                         "predictable state: after one successful header "
                         "anywhere, seq is propagated by the capture clock and "
                         "the per-block CRC guards mispredictions. 0 = off")
    ap.add_argument("--ml-header", action="store_true",
                    help="when hard-decision header decoding fails, pick seq by "
                         "maximum-likelihood correlation against all candidate "
                         "headers using soft luminances")
    ap.add_argument("--ml-margin", type=float, default=6.0,
                    help="sigmas the winning candidate must beat the runner-up "
                         "by; a wrong seq poisons the fountain, so be strict")
    ap.add_argument("--ecc", type=int, default=32,
                    help="RS parity bytes per 255-byte codeword; "
                         "corrects ecc/2 byte errors")
    args = ap.parse_args()
    grid.set_ecc(args.ecc)

    gw, gh = (int(v) for v in args.grid.split("x"))
    layout = grid.Layout(gw, gh)

    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    dec = None
    n = located = header_ok = rs_ok = 0
    margins = []
    t0 = time.time()
    frames_at_done = None
    evidence = {}   # seq -> [sum_of_samples, capture_count]
    added = set()   # seqs already fed to the fountain
    window = deque(maxlen=6)   # recent grayscale frames for tracked localization
    tracked = 0
    proto_header = None        # constants (k, block_size, mode) from any good header
    offsets = []               # capture-clock-to-seq sync measurements
    predicted = 0
    templates = None
    ml_rescued = 0

    while True:
        ok, img = cap.read()
        if not ok:
            break
        n += 1
        if args.max_frames and n > args.max_frames:
            break
        if n % 100 == 0:
            print(f"  ...frame {n}, {rs_ok} decoded", file=sys.stderr)
        H = None
        if img.shape[1] >= 3000:
            # 4K: detect on a half-scale copy, sample at full resolution
            small = cv2.resize(img, None, fx=0.5, fy=0.5)
            Hs = grid.locate(small, layout)
            if Hs is not None:
                H = np.diag([2.0, 2.0, 1.0]) @ Hs
        if args.track:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            window.append(gray)
            if H is None:
                H = grid.locate(img, layout)
            if H is None and len(window) >= 3:
                mean = (sum(window) / len(window)).astype(np.uint8)
                H = grid.locate(mean, layout)
                if H is not None:
                    tracked += 1
            if H is None:
                continue
            # NOTE: refine_H deliberately NOT applied — measured twice tonight
            # (sim noise-30 fusion 14->4 blocks, real v2 capture 97->4): under
            # real margins its translation snap injects alignment jitter.
        if args.combine or args.track:
            header, pay_samples, stats = grid.sample_frame(img, layout, H)
            located += stats["located"]
            header_ok += stats["header_ok"]
            if stats["cell_margin"]:
                margins.append(stats["cell_margin"])
            if header is not None:
                proto_header = header   # transfer constants; seq is the only variable
            if header is None and args.ml_header and proto_header is not None \
                    and pay_samples is not None:
                if templates is None:
                    templates = grid.header_templates(
                        proto_header, min(4000, 12 * proto_header["k"]))
                Hh = H if H is not None else grid.locate(img, layout)
                if Hh is None:
                    continue
                hdr_lum = grid.sample_cells(img, layout, Hh,
                                            layout.header_cells).mean(axis=1)
                seq, margin = grid.ml_header_seq(hdr_lum, templates)
                if margin >= args.ml_margin:
                    header = dict(proto_header, seq=seq)
                    ml_rescued += 1
            if header is not None and args.repeat_hint:
                # sync the capture clock to the sequence counter
                offsets.append(n - header["seq"] * args.repeat_hint)
                proto_header = header
            if header is None and args.repeat_hint and proto_header is not None \
                    and pay_samples is not None:
                # header unreadable but geometry held: predict seq from the
                # capture clock. A wrong prediction dies at the block CRC.
                off = sorted(offsets)[len(offsets) // 2]
                pred = (n - off) // args.repeat_hint
                if pred >= 0:
                    header = dict(proto_header, seq=pred)
                    predicted += 1
            if header is None or header["seq"] in added:
                continue
            if args.combine:
                acc = evidence.setdefault(header["seq"], [0.0, 0])
                if header["mode"] == grid.MODE_MONO:
                    # majority vote per cell across captures of this frame:
                    # robust to a degraded zone that wanders between passes,
                    # and the vote fraction doubles as soft confidence
                    lum = pay_samples.mean(axis=1)
                    th, _ = cv2.threshold(np.clip(lum, 0, 255).astype(np.uint8),
                                          0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    acc[0] = acc[0] + (lum > th).astype(np.float32)
                    acc[1] += 1
                    pseudo = (acc[0] / acc[1] * 255.0)[:, None].repeat(3, axis=1)
                    payload = grid.decide_payload(header, pseudo)
                else:
                    acc[0] = acc[0] + pay_samples
                    acc[1] += 1
                    payload = grid.decide_payload(header, acc[0] / acc[1])
            else:
                payload = grid.decide_payload(header, pay_samples)
            if payload is None:
                continue
            added.add(header["seq"])
        else:
            header, pay_samples, stats = grid.sample_frame(img, layout, H)
            payload = None
            if header is not None:
                payload = grid.decide_payload(header, pay_samples)
                stats["rs_ok"] = payload is not None
            located += stats["located"]
            header_ok += stats["header_ok"]
            if stats["cell_margin"]:
                margins.append(stats["cell_margin"])
        if header is None or payload is None:
            continue
        bs = header["block_size"]
        crc = struct.unpack("<I", payload[:4])[0]
        block = payload[4:4 + bs]
        if zlib.crc32(block) & 0xFFFFFFFF != crc:
            continue  # RS mis-correction or straddled frame — reject
        rs_ok += 1
        if dec is None:
            dec = fountain.Decoder(header["k"], bs, header["file_size"])
        dec.add(header["seq"], block)
        if dec.done and frames_at_done is None:
            frames_at_done = n
            break

    if dec is not None and not dec.done:
        if dec.gaussian_fallback():
            frames_at_done = n
    wall = time.time() - t0

    print(f"frames read        {n}")
    print(f"  located          {located}" +
          (f" ({tracked} rescued by tracking)" if args.track else ""))
    print(f"  header ok        {header_ok}" +
          (f" (+{predicted} seq-predicted)" if predicted else ""))
    print(f"  frame decoded    {rs_ok}")
    if margins:
        print(f"cell margin        median {np.median(margins):.2f} "
              f"(>0.35 comfortable, <0.15 soft-decision territory)")
    if dec is None or not dec.done:
        got = 0 if dec is None else len(dec.decoded)
        want = 0 if dec is None else dec.k
        held = 0 if dec is None else len(dec.pending)
        sys.exit(f"FAILED: fountain incomplete ({got}/{want} blocks solved, "
                 f"{held} coded blocks held as unsolved equations)")

    data = dec.result()
    Path(args.output).write_bytes(data)
    seconds_of_video = frames_at_done / fps
    goodput = len(data) / seconds_of_video / 1024
    print(f"recovered          {len(data):,} bytes  sha256 {hashlib.sha256(data).hexdigest()[:16]}")
    print(f"video time used    {seconds_of_video:.1f}s of capture ({frames_at_done} frames @ {fps:.0f}fps)")
    print(f"GOODPUT            {goodput:.1f} KB/s")
    print(f"decode wall time   {wall:.1f}s "
          f"({'faster' if wall < seconds_of_video else 'SLOWER'} than realtime)")


if __name__ == "__main__":
    main()
