#!/usr/bin/env python3
"""
analyze_smear.py — decide whether the display+camera channel carries 120Hz
content alternation cleanly.

Two ways to use it:

    python3 analyze_smear.py session.mov --auto
        ONE continuous slo-mo clip of `smeartest --auto` (alternate 25s,
        split 25s, static 25s). The three phases are re-identified from
        their flicker signatures; each gets its own analysis.

    python3 analyze_smear.py capture.mov [--pattern alternate|split]
        Manual per-pattern clips, as in the original protocol.

Film in a DIM room so the phone drops exposure; long exposures smear
everything and prove nothing.

Metrics (why each matters):
  modulation depth   can the camera tell a white display-frame from a
                     black one at all? >0.7 usable, <0.3 dead.
  transition width   rows the black->white boundary occupies in a
                     straddling capture: THE smear number. Includes camera
                     exposure and panel response together, which is
                     correct — that combination IS the channel.
  straddle fraction  captures containing a boundary (~100% expected at
                     240fps on a 120Hz display).
  static ripple      flicker leaking into the static half during `split`:
                     local-dimming zone coupling.

Verdict:
  CLEAN     median transition < 12% of height and modulation > 0.7
  MARGINAL  transition < 25% and modulation > 0.5
  SMEARED   otherwise — sub-frame banding is not exploitable on this
            panel; move the transmitter to the iPhone OLED or cap the
            symbol rate at 60Hz.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROFILE_BINS = 256   # row profiles are resampled to this many bins


def locate_screen(cap, n_probe=90):
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
    norm = (acc / acc.max() * 255).astype(np.uint8)
    _, th = cv2.threshold(norm, 128, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        sys.exit("could not locate a bright screen region — film closer / dimmer room")
    big = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(big)
    dx, dy = int(w * 0.06), int(h * 0.06)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return x + dx, y + dy, w - 2 * dx, h - 2 * dy


def scan(cap, region, max_frames=30000):
    """One pass: per-frame row profile (resampled), top/bottom half means."""
    x, y, w, h = region
    profiles, tops, bots = [], [], []
    while len(profiles) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        roi = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY).astype(np.float32)
        prof = roi.mean(axis=1)
        profiles.append(cv2.resize(prof[None], (PROFILE_BINS, 1))[0])
        tops.append(float(prof[: len(prof) // 2].mean()))
        bots.append(float(prof[len(prof) // 2:].mean()))
    return np.array(profiles), np.array(tops), np.array(bots)


def transition_widths(profile, lo, hi):
    """Widths (in bins) of every 10-90% transition in a row profile."""
    if hi - lo < 1e-3:
        return []
    norm = np.clip((profile - lo) / (hi - lo), 0, 1)
    state = norm > 0.5
    edges = np.flatnonzero(np.diff(state.astype(np.int8)))
    widths = []
    for e in edges:
        a = e
        while a > 0 and 0.1 < norm[a] < 0.9:
            a -= 1
        b = e + 1
        while b < len(norm) - 1 and 0.1 < norm[b] < 0.9:
            b += 1
        widths.append(max(1, b - a))
    return widths


def rolling_std(v, w=31):
    pad = np.pad(v, w // 2, mode="edge")
    k = np.ones(w) / w
    m = np.convolve(pad, k, "valid")
    m2 = np.convolve(pad ** 2, k, "valid")
    return np.sqrt(np.maximum(m2 - m ** 2, 0))[: len(v)]


def segment_auto(tops, bots):
    """Label each frame alternate/split/static from flicker location, then
    return the longest contiguous run of each phase."""
    rng = np.percentile(np.concatenate([tops, bots]), 98) - \
          np.percentile(np.concatenate([tops, bots]), 2)
    thr = 0.12 * max(rng, 1e-3)
    ft = rolling_std(tops) > thr
    fb = rolling_std(bots) > thr
    labels = np.where(ft & fb, 0, np.where(ft & ~fb, 1, 2))  # 0=alt 1=split 2=static

    runs = {}
    i = 0
    while i < len(labels):
        j = i
        while j < len(labels) and labels[j] == labels[i]:
            j += 1
        name = ["alternate", "split", "static"][labels[i]]
        if j - i > runs.get(name, (0, 0, 0))[2]:
            runs[name] = (i, j, j - i)
        i = j
    # trim 10% off each end of every run to avoid phase-boundary frames
    out = {}
    for name, (s, e, n) in runs.items():
        pad = n // 10
        out[name] = (s + pad, e - pad)
    return out


def analyze_alternate(profiles, fps):
    lo = float(np.percentile(profiles, 5))
    hi = float(np.percentile(profiles, 95))
    means = profiles.mean(axis=1)
    straddle = 0
    widths_pct = []
    for prof in profiles:
        w = transition_widths(prof, lo, hi)
        if w:
            straddle += 1
            widths_pct.extend(100.0 * np.asarray(w) / PROFILE_BINS)
    mod = float((np.percentile(means, 95) - np.percentile(means, 5)) /
                max(1e-6, np.percentile(means, 95) + np.percentile(means, 5)))
    if len(means) > 32 and fps > 0:
        spec = np.abs(np.fft.rfft(means - means.mean()))
        freqs = np.fft.rfftfreq(len(means), 1.0 / fps)
        dom = float(freqs[int(np.argmax(spec[1:])) + 1])
    else:
        dom = 0.0
    med_w = float(np.median(widths_pct)) if widths_pct else 100.0
    if med_w < 12 and mod > 0.7:
        verdict = "CLEAN"
    elif med_w < 25 and mod > 0.5:
        verdict = "MARGINAL"
    else:
        verdict = "SMEARED"
    return {
        "frames": len(profiles),
        "modulation_depth": round(mod, 3),
        "dominant_flicker_hz": round(dom, 1),
        "median_transition_pct_height": round(med_w, 1),
        "straddle_fraction": round(straddle / max(1, len(profiles)), 3),
        "verdict": verdict,
    }


def analyze_split(bots):
    return {"static_region_ripple": round(float(bots.std() / max(1e-6, bots.mean())), 4)}


def analyze_static(profiles):
    lo = float(np.percentile(profiles, 5))
    hi = float(np.percentile(profiles, 95))
    return {"static_contrast_lo_hi": [round(lo, 1), round(hi, 1)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--auto", action="store_true",
                    help="one continuous clip of `smeartest --auto`; phases "
                         "are re-identified from flicker signatures")
    ap.add_argument("--pattern", default="alternate", choices=["alternate", "split"])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    region = locate_screen(cap)
    profiles, tops, bots = scan(cap, region)
    if len(profiles) == 0:
        sys.exit("no frames")

    report = {"video": args.video, "capture_fps": fps,
              "frames": len(profiles), "screen_region": list(region)}

    if args.auto:
        segs = segment_auto(tops, bots)
        report["segments"] = {k: [int(v[0]), int(v[1])] for k, v in segs.items()}
        if "alternate" in segs:
            s, e = segs["alternate"]
            report["alternate"] = analyze_alternate(profiles[s:e], fps)
        if "split" in segs:
            s, e = segs["split"]
            report["split"] = analyze_split(bots[s:e])
        if "static" in segs:
            s, e = segs["static"]
            report["static"] = analyze_static(profiles[s:e])
        verdict = report.get("alternate", {}).get("verdict", "NO ALTERNATE PHASE FOUND")
    elif args.pattern == "split":
        report["split"] = analyze_split(bots)
        verdict = "n/a (split ripple only)"
    else:
        report["alternate"] = analyze_alternate(profiles, fps)
        verdict = report["alternate"]["verdict"]

    if args.json:
        print(json.dumps(report, indent=2))
        return
    for k, v in report.items():
        print(f"{k:24s} {v}")
    print()
    if verdict == "CLEAN":
        print("Sub-frame banding is exploitable on this channel. Phase 3 lives.")
    elif verdict == "MARGINAL":
        print("Partially usable. Band-wise decode still helps; expect reduced rate.")
    elif verdict == "SMEARED":
        print("Channel smears 120Hz alternation. Use the iPhone OLED as the")
        print("transmitter for Phase 3, or cap the display at 60Hz symbols.")
    ripple = report.get("split", {}).get("static_region_ripple")
    if ripple is not None and ripple > 0.02:
        print(f"WARNING: static-region ripple {ripple} — local-dimming zones couple;")
        print("expect zone-level noise even in static codes.")


if __name__ == "__main__":
    main()
