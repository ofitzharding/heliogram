#!/usr/bin/env python3
"""
simulate.py — synthetic camera channel: turns a clean encoded video into
something that looks like a phone filmed it. Lets the whole pipeline be
tested end-to-end without touching a camera, and lets each degradation be
dialed in isolation to find what actually breaks decoding.

    python3 simulate.py clean.mp4 dirty.mp4 [--blur 1.5] [--noise 6]
            [--tilt 4] [--scale 0.55] [--straddle 0.5] [--gamma 1.15]

--scale     display occupies this fraction of the camera frame width
--tilt      max perspective jitter, in output pixels, per corner
--straddle  fraction of output frames that straddle two display frames
            (top band = next frame), boundary position uniform random —
            models an unsynced camera at ~display rate
"""
import argparse

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--blur", type=float, default=1.5)
    ap.add_argument("--noise", type=float, default=6.0)
    ap.add_argument("--tilt", type=float, default=4.0)
    ap.add_argument("--scale", type=float, default=0.55)
    ap.add_argument("--straddle", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=1.15)
    ap.add_argument("--cam", default="1920x1080")
    ap.add_argument("--repeat", type=int, default=1,
                    help="captures per displayed frame (camera fps / display fps); "
                         "each repeat gets independent jitter and noise")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cw, ch = (int(v) for v in args.cam.split("x"))
    vw = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (cw, ch))
    rng = np.random.default_rng(7)

    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    if not frames:
        raise SystemExit("no input frames")
    h, w = frames[0].shape[:2]

    tw = int(cw * args.scale)
    thh = int(tw * h / w)
    x0, y0 = (cw - tw) // 2, (ch - thh) // 2

    frames = [f for f in frames for _ in range(args.repeat)]
    jit_state = np.zeros((4, 2))
    for i, f in enumerate(frames):
        src_img = f
        if args.straddle > 0 and rng.random() < args.straddle and i + 1 < len(frames):
            boundary = rng.integers(int(h * 0.15), int(h * 0.85))
            src_img = f.copy()
            src_img[:boundary] = frames[i + 1][:boundary]

        # hand tremor is a smooth trajectory, not independent per-frame jumps:
        # AR(1) random walk per corner, stationary std ~= 0.7 * tilt
        jit_state = 0.9 * jit_state + rng.normal(0, args.tilt * 0.3, (4, 2))
        dst_pts = np.array([[x0, y0], [x0 + tw, y0], [x0, y0 + thh],
                            [x0 + tw, y0 + thh]], dtype=np.float32) + jit_state.astype(np.float32)
        src_pts = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)
        H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        out = cv2.warpPerspective(src_img, H, (cw, ch),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(18, 16, 14))
        if args.blur > 0:
            k = int(args.blur * 4) | 1
            out = cv2.GaussianBlur(out, (k, k), args.blur)
        outf = out.astype(np.float32)
        # mild vignette + gamma + sensor noise
        yy, xx = np.mgrid[0:ch, 0:cw].astype(np.float32)
        r2 = ((xx - cw / 2) / (cw / 2)) ** 2 + ((yy - ch / 2) / (ch / 2)) ** 2
        outf *= (1.0 - 0.18 * r2)[..., None]
        outf = 255.0 * (outf / 255.0) ** args.gamma
        outf += rng.normal(0, args.noise, outf.shape)
        vw.write(np.clip(outf, 0, 255).astype(np.uint8))
    vw.release()
    print(f"wrote {args.output}  ({len(frames)} frames, {cw}x{ch})")


if __name__ == "__main__":
    main()
