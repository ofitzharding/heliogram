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
    ap.add_argument("--min-margin", type=float, default=0.0,
                    help="quality gate for fusion: captures below this cell "
                         "margin do not vote. Handheld motion blur makes some "
                         "frames garbage; letting them vote poisons the fusion")
    ap.add_argument("--radial-refine", action="store_true",
                    help="on a frame that fails to decode, retry at neighbouring "
                         "k1 values. Free on frames that already work")
    ap.add_argument("--radial", type=float, default=None,
                    help="lens radial distortion k1. Omit to self-calibrate "
                         "from the footage (recommended); 0 disables")
    ap.add_argument("--ecc", type=int, default=32,
                    help="RS parity bytes per 255-byte codeword; "
                         "corrects ecc/2 byte errors")
    args = ap.parse_args()
    grid.set_ecc(args.ecc)

    gw, gh = (int(v) for v in args.grid.split("x"))
    # Header format auto-detect: v2 (28B, zone fields) vs v1 (24B). The layout
    # depends on it, so probe a few frames with each before committing.
    layout = grid.Layout(gw, gh)
    probe = cv2.VideoCapture(args.input)
    total = int(probe.get(cv2.CAP_PROP_FRAME_COUNT)) or 600
    # Sample ACROSS the whole clip: real captures start and end with the code
    # absent (playback hasn't begun / has finished), so probing only the head
    # detects nothing and silently picks the wrong format.
    probe_at = np.linspace(0, max(0, total - 1), 48).astype(int)
    best, best_hits = 28, -1
    for hl in (28, 24):
        grid.set_header_len(hl)
        lay = grid.Layout(gw, gh)
        hits = 0
        for fi in probe_at:
            probe.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, im = probe.read()
            if not ok:
                continue
            if im.shape[1] >= 3000:
                sm = cv2.resize(im, None, fx=0.5, fy=0.5)
                Hs = grid.locate(sm, lay)
                H = np.diag([2.0, 2.0, 1.0]) @ Hs if Hs is not None else None
            else:
                H = grid.locate(im, lay)
            if H is None:
                continue
            hd, _s, _st = grid.sample_frame(im, lay, H)
            hits += hd is not None
        if hits > best_hits:
            best, best_hits = hl, hits
    probe.release()
    grid.set_header_len(best)
    layout = grid.Layout(gw, gh)
    print(f"header format v{'2' if best == 28 else '1'} "
          f"({best}B, {best_hits} hits in probe)")

    # Radial self-calibration: one k1 for the whole clip (lens is fixed).
    if args.radial is not None:
        grid.set_radial(args.radial)
        print(f"radial k1 = {args.radial:+.3f} (given)")
    else:
        # Decode-directed radial calibration. The Otsu-variance proxy was
        # measured to land at k1=+0.035 where ground truth was +0.020, and BER
        # is steep in k1 (0.44% vs ~4%). So optimise the objective we actually
        # care about: how many probe frames produce a CRC-verified block.
        cal = cv2.VideoCapture(args.input)
        probes = []
        for fi in np.linspace(total * 0.2, total * 0.85, 14).astype(int):
            cal.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, im = cal.read()
            if not ok:
                continue
            sm = cv2.resize(im, None, fx=0.5, fy=0.5) if im.shape[1] >= 3000 else im
            Hs = grid.locate(sm, layout)
            if Hs is None:
                continue
            probes.append((im, (np.diag([2.0, 2.0, 1.0]) @ Hs)
                           if im.shape[1] >= 3000 else Hs))
        cal.release()
        best_k, best_hits = 0.0, -1
        for k1 in np.arange(-0.02, 0.081, 0.005):
            grid.set_radial(float(k1))
            hits = 0
            for im, Hf in probes:
                hd, sm2, _st = grid.sample_frame(im, layout, Hf)
                if hd is None or sm2 is None:
                    continue
                pl = grid.decide_payload(hd, sm2, layout)
                if pl is None:
                    continue
                bsz = hd["block_size"]
                if zlib.crc32(pl[4:4 + bsz]) & 0xFFFFFFFF == \
                        struct.unpack("<I", pl[:4])[0]:
                    hits += 1
            if hits > best_hits:
                best_k, best_hits = float(k1), hits
        grid.set_radial(best_k)
        print(f"radial k1 = {best_k:+.3f} "
              f"(decode-directed: {best_hits}/{len(probes)} probe frames decode)")

    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    dec = None
    n = located = header_ok = rs_ok = 0
    margins = []
    t0 = time.time()
    frames_at_done = None
    first_useful = None
    evidence = {}   # seq -> [sum_of_samples, capture_count]
    added = set()   # seqs already fed to the fountain
    window = deque(maxlen=6)   # recent grayscale frames for tracked localization
    tracked = 0
    proto_header = None        # constants (k, block_size, mode) from any good header
    offsets = []               # capture-clock-to-seq sync measurements
    predicted = 0
    templates = None
    ml_rescued = 0
    refined = 0

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
        # ONE path for every frame. This used to be split so that header
        # recovery only ran under --combine/--track, which silently disabled
        # --ml-header when used alone: a config sweep returned byte-identical
        # results for plain / ml-header / ml-header+repeat-hint. Since the
        # measured funnel puts the header stage as the dominant loss (65% of
        # located frames), that bug sat directly on the critical path.
        header, pay_samples, stats = grid.sample_frame(img, layout, H)
        located += stats["located"]
        header_ok += stats["header_ok"]
        if stats["cell_margin"]:
            margins.append(stats["cell_margin"])
        if header is not None:
            proto_header = header   # transfer constants; seq is the only variable
            if args.repeat_hint:
                offsets.append(n - header["seq"] * args.repeat_hint)

        # Header unreadable but geometry held: recover seq rather than drop the
        # frame. ML template correlation first (uses soft luminances), then the
        # capture clock. A wrong seq dies at the per-block CRC below.
        if header is None and pay_samples is not None and proto_header is not None:
            if args.ml_header and H is not None:
                if templates is None:
                    templates = grid.header_templates(
                        proto_header, min(4000, 12 * proto_header["k"]))
                hdr_lum = grid.sample_cells(img, layout, H,
                                            layout.header_cells).mean(axis=1)
                seq, margin = grid.ml_header_seq(hdr_lum, templates)
                if margin >= args.ml_margin:
                    header = dict(proto_header, seq=seq)
                    ml_rescued += 1
            if header is None and args.repeat_hint and offsets:
                off = sorted(offsets)[len(offsets) // 2]
                pred = (n - off) // args.repeat_hint
                if pred >= 0:
                    header = dict(proto_header, seq=pred)
                    predicted += 1

        if header is None or pay_samples is None:
            continue
        if header["seq"] in added:
            continue
        if args.combine:
            if stats["cell_margin"] < args.min_margin:
                continue   # blurred capture: not allowed to vote
            acc = evidence.setdefault(header["seq"], [0.0, 0])
            if header["mode"] == grid.MODE_MONO:
                # majority vote per cell across captures of this frame
                lum = pay_samples.mean(axis=1)
                th, _ = cv2.threshold(np.clip(lum, 0, 255).astype(np.uint8),
                                      0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                acc[0] = acc[0] + (lum > th).astype(np.float32)
                acc[1] += 1
                pseudo = (acc[0] / acc[1] * 255.0)[:, None].repeat(3, axis=1)
                payload = grid.decide_payload(header, pseudo, layout)
            else:
                acc[0] = acc[0] + pay_samples
                acc[1] += 1
                payload = grid.decide_payload(header, acc[0] / acc[1], layout)
        else:
            payload = grid.decide_payload(header, pay_samples, layout)
            if payload is None and args.radial_refine and H is not None:
                # Per-frame radial refinement. The global k1 is a median; the
                # per-frame optimum was measured to vary 0.015-0.020, and BER is
                # steep in k1. Retrying a failed frame at neighbouring k1 lifted
                # frames-under-RS-limit from 70% to 80% in ground-truth tests.
                # Costs nothing on frames that already decode.
                k0 = grid.RADIAL_K1
                for dk in (-0.008, +0.008, -0.004, +0.004, -0.012, +0.012):
                    grid.set_radial(k0 + dk)
                    s2 = grid.sample_cells(img, layout, H, layout.payload_cells)
                    p2 = grid.decide_payload(header, s2, layout)
                    if p2 is not None:
                        bs2 = header["block_size"]
                        if zlib.crc32(p2[4:4 + bs2]) & 0xFFFFFFFF == \
                                struct.unpack("<I", p2[:4])[0]:
                            payload = p2
                            refined += 1
                            break
                grid.set_radial(k0)
        if payload is None:
            continue
        added.add(header["seq"])
        bs = header["block_size"]
        crc = struct.unpack("<I", payload[:4])[0]
        block = payload[4:4 + bs]
        if zlib.crc32(block) & 0xFFFFFFFF != crc:
            continue  # RS mis-correction or straddled frame — reject
        rs_ok += 1
        if first_useful is None:
            first_useful = n
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
          (f" (+{ml_rescued} ML-rescued)" if ml_rescued else "") +
          (f" (+{predicted} clock-predicted)" if predicted else ""))
    if refined:
        print(f"  radial-refined   {refined} frames rescued")
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
    # Honest accounting: a real transfer runs from the first capture that
    # yields a block to the capture that completes the file. Dividing by the
    # whole clip length understates the rate; dividing by frames_at_done
    # includes dead lead-in before playback started.
    span = (frames_at_done - (first_useful or 1) + 1) / fps
    seconds_of_video = frames_at_done / fps
    goodput = len(data) / span / 1024
    print(f"recovered          {len(data):,} bytes  sha256 {hashlib.sha256(data).hexdigest()[:16]}")
    print(f"transfer span      {span:.2f}s (frame {first_useful} -> {frames_at_done} @ {fps:.0f}fps)")
    print(f"GOODPUT            {goodput:.1f} KB/s   ({goodput/129.2:.2f}x decimen)")
    print(f"decode wall time   {wall:.1f}s "
          f"({'faster' if wall < seconds_of_video else 'SLOWER'} than realtime)")


if __name__ == "__main__":
    main()
