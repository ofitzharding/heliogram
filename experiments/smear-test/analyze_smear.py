#!/usr/bin/env python3
"""
analyze_smear.py — decide whether the display+camera channel carries 120Hz
content alternation cleanly.

Usage:
    python3 analyze_smear.py capture.mov [--pattern alternate|split] [--ref static.mov]

Input is an iPhone slo-mo capture (240fps recommended) of the smeartest app
running the `alternate` pattern (key 1). Film in a DIM room so the phone
drops exposure — long exposures smear everything and prove nothing.

What it measures, and why each one matters:

  modulation depth   Can the camera tell a white display-frame from a black
                     one at all? (P95-P5)/(P95+P5) of band luminance.
                     > 0.7 is a usable channel, < 0.3 is dead.

  transition width   In a captured frame that straddles two display frames,
                     how many rows does the black->white boundary occupy?
                     This is THE smear number: it bounds how much of every
                     captured frame is usable for band-wise decoding.
                     Reported as % of screen height. Includes camera exposure
                     and panel response together — which is correct, because
                     that combination IS the channel.

  straddle fraction  How many captured frames contain a boundary. With a
                     240fps camera on a 120Hz display this should be ~100%
                     (every capture straddles or brackets a display flip).

Verdict:
  CLEAN     median transition < 12% of height and modulation > 0.7
  MARGINAL  transition < 25% and modulation > 0.5
  SMEARED   otherwise — sub-frame banding is not exploitable on this panel;
            move the transmitter to the iPhone OLED or drop Phase 3.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def locate_screen(cap, n_probe=60):
    """Find the display region: union of bright pixels across probe frames."""
    acc = None
    count = 0
    while count < n_probe:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        acc = gray if acc is None else np.maximum(acc, gray)
        count += 1
    if acc is None:
        sys.exit("no frames readable")
    # The screen is the big bright thing: threshold the max-image.
    norm = (acc / acc.max() * 255).astype(np.uint8)
    _, th = cv2.threshold(norm, 128, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        sys.exit("could not locate a bright screen region — film closer / dimmer room")
    big = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(big)
    # shrink 6% inward to avoid bezel/edge bleed
    dx, dy = int(w * 0.06), int(h * 0.06)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return x + dx, y + dy, w - 2 * dx, h - 2 * dy


def transition_widths(profile, lo, hi):
    """Widths (in rows) of every 10-90% monotone transition in a row profile."""
    if hi - lo < 1e-3:
        return []
    norm = np.clip((profile - lo) / (hi - lo), 0, 1)
    # classify rows; hysteresis via double threshold
    state = norm > 0.5
    edges = np.flatnonzero(np.diff(state.astype(np.int8)))
    widths = []
    for e in edges:
        # walk outward to the 0.1 / 0.9 crossings
        a = e
        while a > 0 and 0.1 < norm[a] < 0.9:
            a -= 1
        b = e + 1
        while b < len(norm) - 1 and 0.1 < norm[b] < 0.9:
            b += 1
        widths.append(max(1, b - a))
    return widths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--pattern", default="alternate", choices=["alternate", "split"])
    ap.add_argument("--ref", help="optional static-reference capture for contrast baseline")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    x, y, w, h = locate_screen(cap)

    means = []            # per-frame mean luminance of screen region
    all_widths = []       # transition widths, % of height
    straddle = 0
    analyzed = 0
    static_var = []       # for split pattern: bottom-half variance over time
    example_saved = False
    out_dir = Path(args.video).with_suffix("")
    out_dir.mkdir(exist_ok=True)

    # first pass: collect global luminance range for normalization
    lo_g, hi_g = 255.0, 0.0
    profiles = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        roi = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY).astype(np.float32)
        profile = roi.mean(axis=1)          # row means
        profiles.append(profile)
        means.append(float(roi.mean()))
        if args.pattern == "split":
            static_var.append(float(roi[h // 2:, :].mean()))
        lo_g = min(lo_g, float(np.percentile(profile, 5)))
        hi_g = max(hi_g, float(np.percentile(profile, 95)))
        analyzed += 1
        if analyzed >= 2000:                # ~8s at 240fps is plenty
            break

    for i, profile in enumerate(profiles):
        widths = transition_widths(profile, lo_g, hi_g)
        if widths:
            straddle += 1
            all_widths.extend(100.0 * np.asarray(widths) / len(profile))
        if widths and not example_saved:
            example_saved = True

    means_a = np.asarray(means)
    mod_depth = float((np.percentile(means_a, 95) - np.percentile(means_a, 5)) /
                      max(1e-6, (np.percentile(means_a, 95) + np.percentile(means_a, 5))))

    # dominant flicker frequency from the mean-luminance series
    if len(means_a) > 32 and fps > 0:
        spec = np.abs(np.fft.rfft(means_a - means_a.mean()))
        freqs = np.fft.rfftfreq(len(means_a), 1.0 / fps)
        dom = float(freqs[int(np.argmax(spec[1:])) + 1])
    else:
        dom = 0.0

    med_w = float(np.median(all_widths)) if all_widths else 100.0
    frac_straddle = straddle / max(1, analyzed)

    if med_w < 12 and mod_depth > 0.7:
        verdict = "CLEAN"
    elif med_w < 25 and mod_depth > 0.5:
        verdict = "MARGINAL"
    else:
        verdict = "SMEARED"

    report = {
        "video": args.video,
        "capture_fps": fps,
        "frames_analyzed": analyzed,
        "screen_region": [x, y, w, h],
        "modulation_depth": round(mod_depth, 3),
        "dominant_flicker_hz": round(dom, 1),
        "median_transition_pct_height": round(med_w, 1),
        "straddle_fraction": round(frac_straddle, 3),
        "verdict": verdict,
    }
    if args.pattern == "split" and static_var:
        sv = np.asarray(static_var)
        report["static_region_ripple"] = round(float(sv.std() / max(1e-6, sv.mean())), 4)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for k, v in report.items():
            print(f"{k:32s} {v}")
        print()
        if verdict == "CLEAN":
            print("Sub-frame banding is exploitable on this channel. Phase 3 lives.")
        elif verdict == "MARGINAL":
            print("Partially usable. Band-wise decode still helps; expect reduced rate.")
        else:
            print("Channel smears 120Hz alternation. Use the iPhone OLED as the")
            print("transmitter for Phase 3, or cap the display at 60Hz symbols.")


if __name__ == "__main__":
    main()
