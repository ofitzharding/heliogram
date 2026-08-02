#!/usr/bin/env python3
"""
analyze_probe.py — turn one probe capture into the whole operating envelope.

Written and debugged against the CLEAN transmit file BEFORE any real capture
exists, because every take this week died to a different bug and the cost of
finding them afterwards is another 50 seconds of someone's evening plus an
hour of forensics. Running this on the clean file is a dress rehearsal: it
must report a perfect channel there, and anything it cannot do on perfect
input it certainly cannot do on a capture.

Segment identification is the part most likely to break on real footage, so
it is done WITHOUT relying on frame offsets: the capture starts at an
arbitrary point in the loop, may drop frames, and may span loop boundaries.
Instead each frame is classified by what it contains.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from codec import grid
import exp_turbo_frame as TB

DENSITIES = ["252x140", "350x194", "466x259"]


def classify(gray):
    """What kind of probe frame is this?

    Identified by CONTENT, not by frame offset, because a capture starts at an
    arbitrary point in the loop, may drop frames, and may straddle a loop
    boundary. Offsets would be wrong in all three cases.

    The discriminator for stripes is vertical uniformity: a vertical-stripe
    field is identical on every row, and code content never is. A first
    version used FFT periodicity and happily labelled code frames as stripes
    at pitches like 1008px (= width/3), which poisoned the MTF curve and
    starved the density sweep of frames.
    """
    m, sd = float(gray.mean()), float(gray.std())
    if sd < 12:
        return ("field", m)
    g = gray.astype(np.float32)
    rows = g[::max(1, g.shape[0] // 32)]
    row_var = rows.std(axis=0).mean()            # variation DOWN the columns
    if row_var < 6.0 and sd > 60:
        col = g[g.shape[0] // 2]
        d = np.diff((col > col.mean()).astype(np.int8))
        edges = np.flatnonzero(d != 0)
        if len(edges) >= 4:
            pitch = 2.0 * float(np.median(np.diff(edges)))
            return ("stripe", pitch)
    return ("code", 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--max-frames", type=int, default=2000)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()
    grid.set_ecc(48); grid.set_header_len(28)
    grid.set_header_centered(True); grid.set_radial(0.0)

    layouts = {}
    for spec in DENSITIES:
        gw, gh = (int(v) for v in spec.split("x"))
        L = grid.Layout(gw, gh)
        # SubBlock must span EXACTLY the codewords the encoder paints.
        # Building it from payload_capacity_bytes gave 17 spans where the
        # encoder writes 16, so yields came out over 100% - a denominator
        # error that would have silently inflated every number from the take.
        n_sub = grid.sub_count(L, grid.MODE_MONO)
        layouts[spec] = (L, TB.SubBlock(L, 48, n_sub * (255 - 48)), n_sub)

    cap = cv2.VideoCapture(args.capture)
    total = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), args.max_frames)
    fields, stripe = [], {}
    code = {}          # (density, hold) -> rows
    straddle_reads = []

    n = 0
    while n < total:
        ok, img = cap.read()
        if not ok:
            break
        n += 1
        if (n - 1) % args.stride:
            continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kind, val = classify(g)
        if kind == "field":
            fields.append(val)
        elif kind == "stripe":
            row = g[g.shape[0] // 2].astype(np.float32)
            lo, hi = np.percentile(row, 5), np.percentile(row, 95)
            mich = (hi - lo) / max(hi + lo, 1e-6)
            stripe.setdefault(round(val), []).append(mich)
            mid = ((row > lo + 0.3*(hi-lo)) & (row < lo + 0.7*(hi-lo))).mean()
            straddle_reads.append(mid)
        else:
            # try every density; the one that locates AND reads a header wins
            for spec, (L, sub, n_sub) in layouts.items():
                H = grid.locate(img, L)
                if H is None:
                    continue
                hd, _s, _t = grid.sample_frame(img, L, H)
                if hd is None:
                    continue
                allc = np.argwhere(np.ones((L.gh, L.gw), bool))
                y = grid.sample_cells(img, L, H, allc).mean(axis=1
                     ).reshape(L.gh, L.gw)
                th, _ = cv2.threshold(np.clip(y.ravel(), 0, 255).astype(np.uint8),
                                      0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                x = (y > th).astype(np.uint8)
                nc = sub.try_certify(x[sub.cells[:, 0], sub.cells[:, 1]])[2]
                c = np.array([[0, 0], [L.gw, 0]], np.float32).reshape(-1, 1, 2)
                p = cv2.perspectiveTransform(c, H).reshape(-1, 2)
                pxc = np.linalg.norm(p[1] - p[0]) / L.gw
                # the frame states its own hold, so a dropped or duplicated
                # camera frame cannot misattribute it to another condition
                hold = int(hd.get("zone_w", 0)) or 1
                # straddle measured on THIS frame, independently of decoding
                pv = y[~(L.is_finder | L.is_sep | L.is_ring | L.is_header)]
                lo, hi = np.percentile(pv, 3), np.percentile(pv, 97)
                mid = float(((pv > lo + 0.30*(hi-lo)) &
                             (pv < lo + 0.70*(hi-lo))).mean())
                code.setdefault((spec, hold), []).append((nc, n_sub, pxc, mid))
                break
    cap.release()

    print(f"analysed {n} frames of {args.capture}\n")
    if fields:
        f = np.array(sorted(fields))
        print(f"PHOTOMETRY   black {f.min():6.1f}   white {f.max():6.1f}   "
              f"range {f.max()-f.min():6.1f} of 255")
        print(f"             levels seen: {np.round(f,0)}")
    if stripe:
        print("\nMTF  (stripe pitch in panel px -> Michelson contrast)")
        for p in sorted(stripe, reverse=True):
            v = np.mean(stripe[p])
            bar = "#" * int(40*v)
            print(f"   pitch {p:5.0f}px  {v:5.3f} {bar}")
        good = [p for p in sorted(stripe) if np.mean(stripe[p]) > 0.25]
        if good:
            print(f"   -> contrast holds down to ~{min(good):.0f}px pitch")
    if straddle_reads:
        sr = np.array(straddle_reads)
        print(f"\nSTRADDLE     mid-band fraction on 2-level fields: "
              f"median {100*np.median(sr):.1f}%  (0% = no straddle)")
    print("\nTWO-AXIS DECOUPLING   (optics identical across every row)")
    print(f"   {'grid':>9s} {'px/cell':>8s} {'hold':>5s} {'frames':>7s} "
          f"{'straddle':>9s} {'yield':>7s} {'KB/s':>8s}")
    SUB = 255 - 48 - 4
    for spec in DENSITIES:
        for hold in (1, 2, 4):
            rows = code.get((spec, hold))
            if not rows:
                print(f"   {spec:>9s} {'-':>8s} {hold:5d} {0:7d} "
                      f"{'-':>9s} {'none decoded':>16s}")
                continue
            nc = np.array([r[0] for r in rows]); ns = rows[0][1]
            pxc = np.mean([r[2] for r in rows]); mid = np.mean([r[3] for r in rows])
            y_ = nc.mean() / ns
            kbs = ns * SUB * (60.0 / hold) / 1000.0 * y_
            print(f"   {spec:>9s} {pxc:8.1f} {hold:5d} {len(rows):7d} "
                  f"{100*mid:8.1f}% {100*y_:6.1f}% {kbs:7.1f}K")
    print("\n   The gap between hold=1 and hold=4 at fixed density IS the")
    print("   exposure-sync contribution: optics are pinned across the whole")
    print("   take, so nothing else can explain a difference between rows.")
    print("   decimen reference: 128 KB/s handheld, 186 propped.")


if __name__ == "__main__":
    main()
