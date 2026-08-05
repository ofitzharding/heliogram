#!/usr/bin/env python3
"""
decode_session.py — drive the whole one-sitting session unattended.

    python3 src/decode_session.py LADDER.MOV [ROLLING.MOV] [SPEED.MOV]

For each capture: identify which transmit grid is on screen at regular
probes (headers are self-labelling: each webtx dir has a distinct grid),
split the capture into segments, gate each segment with analyze_strobe,
then run fast_decode --from-start --start-frame on it for the honest
per-rung full-span. Prints one table at the end.

Grids probed: 252x163 (12px), 274x178 (11px), 302x196 (10px), 336x218 (9px).
Payload: demo/kitten_big.png (sha256 checked on every recovered file).
"""
import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from codec import grid

GRIDS = [(252, 163, "12px"), (274, 178, "11px"),
         (302, 196, "10px"), (336, 218, "9px")]
PAYLOAD = Path(__file__).parent.parent / "demo" / "kitten_big.png"


def probe_grid(img):
    """Which grid decodes a header on this frame, if any."""
    for gw, gh, name in GRIDS:
        L = grid.Layout(gw, gh)
        big = img.shape[1] >= 3000
        sm = cv2.resize(img, None, fx=0.5, fy=0.5) if big else img
        Hs = grid.locate(sm, L)
        H = (((np.diag([2., 2., 1.]) @ Hs) if big else Hs)
             if Hs is not None else grid.locate(img, L))
        if H is None:
            continue
        hd, _s, _t = grid.sample_frame(img, L, H)
        if hd is not None:
            return (gw, gh, name)
    return None


def _probe_at(args_):
    """Pool worker: probe one frame position for its grid."""
    path, f = args_
    grid.set_ecc(48); grid.set_header_len(28); grid.set_header_centered(True)
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, img = cap.read()
    cap.release()
    if not ok:
        return (f, None)
    for k1 in (0.0, 0.020):
        grid.set_radial(k1)
        g = probe_grid(img)
        if g is not None:
            return (f, g)
    return (f, None)


def segments(path, step_s=5, workers=10):
    """Probe every step_s seconds (parallel), bisect the boundaries;
    return [(gw, gh, name, f0, f1)]."""
    from multiprocessing import Pool
    cap = cv2.VideoCapture(path)
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    cap.release()
    pos = list(range(0, tot, int(step_s * fps)))
    with Pool(workers) as pool:
        marks = pool.map(_probe_at, [(path, f) for f in pos])
    segs = []
    cur = None
    for f, g in marks:
        if g is None:
            continue
        if cur is None or cur[2] != g[2]:
            cur = (g[0], g[1], g[2], f, f)
            segs.append(cur)
        else:
            cur = (cur[0], cur[1], cur[2], cur[3], f)
            segs[-1] = cur
    # bisect each boundary: the coarse probe leaves up to step_s of slop
    # that the decoder would otherwise chew through as locate() misses
    for i in range(1, len(segs)):
        lo = segs[i - 1][4]              # last frame KNOWN to be prev grid
        hi = segs[i][3]                  # first frame KNOWN to be this grid
        want = segs[i][2]
        while hi - lo > 30:
            mid = (lo + hi) // 2
            _f, g = _probe_at((path, mid))
            if g is not None and g[2] == want:
                hi = mid
            else:
                lo = mid
        segs[i] = (segs[i][0], segs[i][1], segs[i][2], hi, segs[i][4])
    # extend each segment to the next segment's start (or EOF)
    out = []
    for i, s in enumerate(segs):
        f1 = segs[i + 1][3] if i + 1 < len(segs) else tot
        out.append((s[0], s[1], s[2], s[3], f1))
    return out, fps, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("captures", nargs="+")
    ap.add_argument("--skip-gate", action="store_true")
    args = ap.parse_args()
    want = hashlib.sha256(PAYLOAD.read_bytes()).hexdigest()
    results = []
    for cap_path in args.captures:
        print(f"\n===== {cap_path} =====")
        segs, fps, tot = segments(cap_path)
        for gw, gh, name, f0, f1 in segs:
            print(f"\n--- segment {name} ({gw}x{gh}) frames {f0}-{f1} ---")
            if not args.skip_gate:
                subprocess.run(
                    ["python3", str(Path(__file__).parent /
                                    "analyze_strobe.py"), cap_path,
                     "--grid", f"{gw}x{gh}", "--payload", str(PAYLOAD),
                     "--frames", "12",
                     "--lo", f"{f0 / tot:.3f}",
                     "--hi", f"{min(0.99, (f1 - 1) / tot):.3f}"])
            out = f"/tmp/session_{name}.bin"
            r = subprocess.run(
                ["python3", str(Path(__file__).parent / "fast_decode.py"),
                 cap_path, out, "--grid", f"{gw}x{gh}", "--ecc", "48",
                 "--subblock", "--soft", "--scan", "--from-start",
                 "--start-frame", str(f0), "--end-frame", str(f1),
                 "--workers", "10"], capture_output=True, text=True)
            print(r.stdout[-1200:])
            got = None
            try:
                got = hashlib.sha256(Path(out).read_bytes()).hexdigest()
            except FileNotFoundError:
                pass
            ok = got == want
            gp = ""
            for line in r.stdout.splitlines():
                if line.startswith("GOODPUT"):
                    gp = line.split()[1]
            results.append((Path(cap_path).name, name, gp,
                            "BIT-EXACT" if ok else "FAILED"))
    print("\n===== SESSION SUMMARY =====")
    print(f"{'capture':>22s} {'rung':>6s} {'KB/s':>8s}  verdict")
    for cp, name, gp, v in results:
        print(f"{cp:>22s} {name:>6s} {gp:>8s}  {v}")


if __name__ == "__main__":
    main()
